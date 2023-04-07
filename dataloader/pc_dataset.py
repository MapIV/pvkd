# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import yaml
import time
import glob
import pickle
import numpy as np
from copy import copy
from pypcd import pypcd
from torch.utils import data
from scipy.spatial import KDTree
from transforms3d import _gohlketransforms as transformations


REGISTERED_PC_DATASET_CLASSES = {}

def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class InferenceDataset(data.Dataset):
    def __init__(self, data_path,
                 imageset='',
                 return_ref='',
                 nusc='',
                 data_type='pcd',
                 label_mapping="semantic-kitti.yaml"):
        """
        supported data_type:
            bin : KITTI binary '.bin' file
            pcd : .pcd files
        TODO: numpy: already formatted Nx4 numpy array
        """
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.data_type = data_type
        self.learning_map = semkittiyaml['learning_map']
        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)

    def normalize_intensity(self, intensity, norm_factor = 5, max_intensity_value=255):
        return 2 * np.e ** (norm_factor * intensity / max_intensity_value) / \
            (1 + np.e ** (norm_factor * intensity / max_intensity_value)) - 1

    def normalize_height(self, points_z, target_height=-1.73):
        ground_height = np.percentile(points_z, 0.99)
        return points_z - (ground_height - target_height)

    def to_XYZI_array(self, points_struct):
        points = np.zeros((points_struct['x'].shape[0], 4), dtype=float)
        points[:, 0] = points_struct['x']
        points[:, 1] = points_struct['y']
        points[:, 2] = points_struct['z']
        # Load intensity
        if 'intensity' in points_struct.dtype.names:
            points[:, 3] = points_struct['intensity']
        elif 'i' in points_struct.dtype.names:
            points[:, 3] = points_struct['intensity']
        else:
            print("intensity not found, that's probably not good but feel free to supress this")
        return points

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.im_idx)

    def __getitem__(self, index):
        if self.data_type == 'bin':
            raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        elif self.data_type == 'pcd':
            points_struct = pypcd.PointCloud.from_path(self.im_idx[index])
            raw_data = self.to_XYZI_array(points_struct.pc_data)
            # raw_data[:, 3] = self.normalize_intensity(raw_data[:, 3])
            # raw_data[:, 2] = self.normalize_height(raw_data[:, 2])

        annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class MapInferenceDataset(data.Dataset):
    def __init__(self, data_path,
                 imageset='',
                 return_ref='',
                 nusc='',
                 data_type='pcd',
                 label_mapping="semantic-kitti.yaml"):
        """
        I have to adhere with the base class' init definition, so the variable names are misleading
        data_path: map .pcd file folder and .ndt files. All PCD maps inside the folder will be concatenated. There should
        be one .csv file in the folder too
        ## hardcoded stuff for now
        search_radius: default is sqrt(25^2 + 25^2) since detections area is 50m rectangle
        voxel_size: [x, y, z] in meters, 0.2 is good
        ground_height: z value of ground points in local frame, like 0 or -2. target is -1.73
        """
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.data_type = data_type
        self.learning_map = semkittiyaml['learning_map']
        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.poses = []
        self.pts_vx_idx, self.uni_vx_idx = [], []
        self.flag = False

        # Hardcoded
        self.search_radius = np.sqrt(25*25*2)   # closest to a 50m cube
        self.voxel_size = [0.75, 0.75, 0.75]
        self.ground_height = 0
        self.boundary = {"minX": -50, "maxX": 50, "minY": -50, "maxY": 50, "minZ": -3.73, "maxZ": 2.27}


        if imageset == 'train':
            print("You cannot use this dataloader for training~")
            self.poses = [0]
        else:
            # Load map and NDT
            self.pcd_map = self.load_map(data_path)
            self.labels_map = np.zeros((self.pcd_map.shape[0]))
            ndt_file = glob.glob(data_path + '/*.csv')[0]
            assert os.path.exists(ndt_file)
            all_ndt_poses = self.load_ndt_file(ndt_file)
            self.kd_tree = self.build_kdtree(self.pcd_map[:, :3])

            # Chop up poses
            latest_pose = np.zeros((1, 3))
            print("Parsing NDT Poses")
            for idx, ndt_pose in enumerate(all_ndt_poses):
                if idx % 10 == 0: print(f"Parsing NDT Poses {idx+1}/{len(all_ndt_poses)}"
                                        + '.'*int(idx/len(all_ndt_poses)*10))
                t, pose, pose_quat = self.get_best_ndt_pose(all_ndt_poses, ndt_pose[0])
                dist = np.linalg.norm(pose - latest_pose)
                if idx == 0 or dist > self.search_radius:
                    latest_pose = pose
                    pose_matrix = transformations.quaternion_matrix(pose_quat)
                    pose_matrix[:3, 3] = pose
                    map_points_idx = self.query_kdtree(self.kd_tree, pose, self.search_radius)
                    if len(map_points_idx) < 50000: continue
                    self.poses.append(pose_matrix)
            print("Parsing NDT Poses" + '..........done!')

    def __getitem__(self, index):
        # Get points in radius
        self.flag = True
        self.map_points_idx = np.array(self.query_kdtree(self.kd_tree, self.poses[index][:3, 3], self.search_radius), dtype=int)
        current_pcd = self.pcd_map[self.map_points_idx, :]
        # map_index = np.arange(0, self.pcd_map.shape[0], 1)
        # self.current_index = map_index[self.map_points_idx]

        # Transform
        intensity = np.copy(current_pcd[:, 3])
        current_pcd[:, 3] = 1
        current_pcd = np.matmul(np.linalg.inv(self.poses[index]), current_pcd.T)
        current_pcd = current_pcd.T
        current_pcd[:, 3] = intensity

        box_filter = self.filtering_boundary(current_pcd, self.boundary)
        self.map_points_idx = self.map_points_idx[box_filter]
        current_pcd = current_pcd[box_filter, :]

        # Adjust height
        # current_pcd[:, 2] = self.shift_height(current_pcd[:, 2], self.ground_height)

        # Voxelize
        voxelized_pcd, self.pts_vx_idx, self.uni_vx_idx = self.faster_voxelize(current_pcd)
        self.voxel_grid = np.empty((np.max(self.pts_vx_idx[:, 0])+1,
                                    np.max(self.pts_vx_idx[:, 1])+1,
                                    np.max(self.pts_vx_idx[:, 2])+1))

        annotated_data = np.expand_dims(np.zeros_like(voxelized_pcd[:, 0], dtype=int), axis=1)
        data_tuple = (voxelized_pcd[:, :3], annotated_data.astype(np.uint8))
        data_tuple += (voxelized_pcd[:, 3],)
        return data_tuple

    def get_voxel_item(self, index):
        self.map_points_idx = self.query_kdtree(self.kd_tree, self.poses[index][:3, 3], self.search_radius)
        current_pcd = self.pcd_map[self.map_points_idx, :]

        # Transform
        intensity = np.copy(current_pcd[:, 3])
        current_pcd[:, 3] = 1
        current_pcd = np.matmul(np.linalg.inv(self.poses[index]), current_pcd.T)
        current_pcd = current_pcd.T
        current_pcd[:, 3] = intensity

        # filter
        box_filter = self.filtering_boundary(current_pcd, self.boundary)
        current_pcd = current_pcd[box_filter, :]

        # Adjust height
        # current_pcd[:, 2] = self.shift_height(current_pcd[:, 2], self.ground_height)

        # Voxelize
        voxelized_pcd, _, _ = self.faster_voxelize(current_pcd)
        return voxelized_pcd

    def update_labels_map(self, labels):
        self.voxel_grid[self.uni_vx_idx[:, 0], self.uni_vx_idx[:, 1], self.uni_vx_idx[:, 2]] = labels.reshape(-1)
        self.labels_map[self.map_points_idx] = self.voxel_grid[self.pts_vx_idx[:, 0], self.pts_vx_idx[:, 1], self.pts_vx_idx[:, 2]]

    def save_labels_map_rgb(self, save_fp, colormap=None, ignore_classes=None, remove_classes=None):
        """
        color_map: see label_color_map()
        ignore_classes: list of class IDs that you won't want colored (will be black)
        remove_classes: list of class IDs for which points should be removes (i.e. dynamic objects)
        """
        pcd_map, labels_map = np.copy(self.pcd_map), np.copy(self.labels_map)
        print(f"Labels info: {np.min(labels_map)}, {np.max(labels_map)}")
        if colormap is None: colormap = self.label_color_map()

        if ignore_classes is not None:
            for ignore_class in ignore_classes:
                colormap[ignore_class, :] = np.array([255, 255, 255])

        if remove_classes is not None:
            for remove_class in remove_classes:
                idxs = labels_map.astype(int) == remove_class
                pcd_map = np.delete(pcd_map, idxs, axis=0)
                labels_map = np.delete(labels_map, idxs)

        rgb_array = colormap[labels_map.reshape(-1).astype(int), :]
        rgb_encoded = pypcd.encode_rgb_for_pcl(rgb_array)

        # final_pcd_map = np.zeros((pcd_map.shape[0], 6))
        final_pcd_map = np.zeros((pcd_map.shape[0], 4), dtype=float)
        final_pcd_map[:, :3] = pcd_map[:, :3]
        # final_pcd_map[:, 4] = rgb_encoded
        # final_pcd_map[:, 5] = labels_map
        final_pcd_map[:, 3] = rgb_encoded
        # save files
        N = 10000000        # max points per file
        start = 0
        end = N
        pcd_save_idx = 0
        while start < final_pcd_map.shape[0]:
            if end > final_pcd_map.shape[0]: end = final_pcd_map.shape[0]
            # pcd = pypcd.make_xyzirgb_label_point_cloud(final_pcd_map[start:end, :])
            pcd = pypcd.make_xyz_rgb_point_cloud(final_pcd_map[start:end, :])
            pcd.save_pcd(save_fp + f"{pcd_save_idx:02d}.pcd")
            start = copy(end)
            end = end + N
            pcd_save_idx += 1

    @staticmethod
    def filtering_boundary(points, boundary):
        # reshape points and labels
        x_filter = np.logical_and(points[:, 0] >= boundary["minX"], points[:, 0] <= boundary["maxX"])
        y_filter = np.logical_and(points[:, 1] >= boundary["minY"], points[:, 1] <= boundary["maxY"])
        z_filter = np.logical_and(points[:, 2] >= boundary["minZ"], points[:, 2] <= boundary["maxZ"])
        filter_mask = np.logical_and(np.logical_and(x_filter, y_filter), z_filter)
        # filtering points and labels
        return filter_mask

    @staticmethod
    def label_color_map():
        pvkd_to_mapillary_rgb = {
            0: (0, 0, 0),  # outlier
            1: (0, 0, 142),  # car
            2: (119, 11, 32),  # bicycle
            3: (220, 20, 60),  # person
            4: (0, 0, 70),  # truck
            5: (128, 64, 128),  # road
            6: (244, 35, 232),  # other-ground
            7: (128, 64, 64),  # other_vehicle
            8: (152, 251, 152),  # terrain
            9: (70, 70, 70),  # building
            10: (190, 153, 153),  # fence
            11: (107, 142, 35),  # vegetation
            12: (255, 255, 255),  # moving
            13: (220, 220, 0),  # traffic-sign
        }
        color_array = np.array([value for key, value in pvkd_to_mapillary_rgb.items()], dtype=np.uint8).reshape(-1, 3)
        return color_array

    def faster_voxelize(self, points):
        """
            :param points: (N*4, ) numpy array
            :return: voxelized_points, vozelized_labels
        """
        # reshape
        vs = np.array(self.voxel_size).reshape(1, 3)
        points = points.reshape(-1, 4)

        # pointcloud limits
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        z_min = np.min(points[:, 2])
        min_range = np.array([x_min, y_min, z_min]).reshape(1, 3)

        # to voxel idx grid
        voxelized_point = (points[:, :3] - min_range)
        voxelized_point = voxelized_point / vs
        voxelized_point = np.floor(voxelized_point)
        all_pts_idx = np.copy(voxelized_point)

        # unique voxels
        voxelized_point, unique_idx = np.unique(voxelized_point, axis=0, return_index=True)
        unique_pts_idx = all_pts_idx[unique_idx, :]

        # make a point in voxel center
        voxelized_point = voxelized_point * vs + vs / 2 + min_range
        remissions = points[unique_idx, 3].reshape(-1, 1)
        voxelized_point = np.hstack([voxelized_point, remissions])

        return voxelized_point.astype(np.float32), all_pts_idx.astype(np.uint8), unique_pts_idx.astype(np.uint8)

    @staticmethod
    def get_best_ndt_pose(ndt_poses, ts):
        ndt_time_diff = np.abs(ndt_poses[:, 0] - ts)

        pose_idx = np.argmin(ndt_time_diff)
        t = ndt_poses[pose_idx, 0]
        pose = ndt_poses[pose_idx, 1:4]
        euler = ndt_poses[pose_idx, 4:]
        pose_quat = transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        return t, pose, pose_quat

    @staticmethod
    def normalize_intensity(intensity, norm_factor = 5, max_intensity_value=255):
        return 2 * np.e ** (norm_factor * intensity / max_intensity_value) / \
            (1 + np.e ** (norm_factor * intensity / max_intensity_value)) - 1

    @staticmethod
    def shift_height(points_z, ground_height=None, target_height=-1.73):
        if ground_height is None:
            ground_height = np.percentile(points_z, 0.99)
        return points_z - (ground_height - target_height)

    @staticmethod
    def to_XYZI_array(points_struct):
        points = np.zeros((points_struct['x'].shape[0], 4), dtype=float)
        points[:, 0] = points_struct['x']
        points[:, 1] = points_struct['y']
        points[:, 2] = points_struct['z']
        # Load intensity
        if 'intensity' in points_struct.dtype.names:
            points[:, 3] = points_struct['intensity']
        elif 'i' in points_struct.dtype.names:
            points[:, 3] = points_struct['i']
        else:
            print("intensity not found, that's probably not good but feel free to supress this")
        return points

    @staticmethod
    def load_map(pcd_map_path):
        """Loads all maps in a directory and accumulates them"""
        pcd_map = np.empty((0, 4))
        start_time = time.time()
        pcd_list = glob.glob(pcd_map_path + '/*.pcd')
        for pcd_fp in pcd_list:
            map_part = pypcd.point_cloud_from_path(pcd_fp)
            pcd = np.ones((map_part.pc_data['x'].shape[0], 4))
            pcd[:, 0] = map_part.pc_data['x']
            pcd[:, 1] = map_part.pc_data['y']
            pcd[:, 2] = map_part.pc_data['z']
            pcd[:, 3] = map_part.pc_data['intensity']
            pcd_map = np.vstack((pcd_map, pcd))
        print(f"Map loaded with {pcd_map.shape[0]} points in {time.time() - start_time} seconds")
        return pcd_map

    @staticmethod
    def build_kdtree(data):
        start_time = time.time()
        tree = KDTree(data)
        print(f"Built KDTree in {time.time() - start_time} seconds")
        return tree

    @staticmethod
    def query_kdtree(tree, pose, radius):
        start_time = time.time()
        search_idx = tree.query_ball_point(pose.reshape(3), radius)
        # print(f"Map queried with {search_idx.shape[0]} points in {time.time() - start_time} seconds")
        return search_idx

    @staticmethod
    def load_ndt_file(ndt_filepath):
        # [timestamp, lidar_pose.x, lidar_pose.y, lidar_pose.z, lidar_pose.roll	lidar_pose.pitch, lidar_pose.yaw]
        ndt_msgs = np.loadtxt(ndt_filepath, delimiter=',', skiprows=1, usecols=[1, 9, 10, 11, 12, 13, 14])
        return ndt_msgs

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.poses)

@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class Waymo(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="waymo.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.im_idx = []
        if imageset =='train':
            with open('/nvme/yuenan/train-0-31.txt', 'r') as f:
                for line in f.readlines():
                    self.im_idx.append(line.strip())
        else:
            with open('/nvme/yuenan/val-0-7.txt', 'r') as f: #val-0-7-label.txt
                for line in f.readlines():
                    self.im_idx.append(line.strip())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):

        raw_data = np.load(self.im_idx[index])[:, 3:6].reshape((-1,3))
        len_first = raw_data.shape[0]
        use_extra = True #False
        use_sec_return = True #True

        if use_extra:
            intensity = np.load(self.im_idx[index])[:, 1].reshape((-1,1))
            intensity = np.tanh(intensity) 
            #elongation = np.load(self.im_idx[index])[:, 2].reshape((-1,1))
            extra_data = intensity 
        if 'no_label' in self.im_idx[index]:
            annotated_data = np.load(self.im_idx[index].replace('test/', 'test_nolabel/').replace('train/', 'train_nolabel/').replace('no_label_point_clouds/', 'train_nolabel/')).reshape((-1,1)) #
            #annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif 'valid' in self.im_idx[index] and self.imageset == 'train':
            #annotated_data = np.load(self.im_idx[index].replace('valid/', 'valid_nolabel/')).reshape((-1,1)) #
            base_path = '/home/sysuser/cylinder3d/val_submit'#+self.imageset
            frame_id = self.im_idx[index].split('/')[-1]
            annotated_data = np.load(base_path+'/first/'+frame_id).reshape((-1, 1))
        else:
            #annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
            #                             dtype=np.uint32).reshape((-1, 1))
            #annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.load(self.im_idx[index])[:, -1].reshape((-1,1)) #annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            #annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)

        if use_sec_return:
            if True: #self.imageset == 'train': #and 'test' not in self.im_idx[index]:
                sec_path = self.im_idx[index].replace('first', 'second') #'validation%2F', 'validation_').replace('training%2F', 'training_')
                #sec_path = sec_path.replace('labels_', 'labels')
                sec_data = np.load(sec_path)[:, 3:6].reshape((-1,3))
                #len_first = raw_data.shape[0]
                assert len_first != 3
                raw_data = np.concatenate((raw_data, sec_data), axis=0)
                sec_annotated_data = np.load(sec_path)[:, -1].reshape((-1,1))
                annotated_data = np.concatenate((annotated_data, sec_annotated_data), axis=0)
                sec_intensity = np.load(sec_path)[:, 1].reshape((-1,1))
                sec_intensity = np.tanh(sec_intensity)
                extra_data = np.concatenate((intensity, sec_intensity), axis=0)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        data_tuple += (extra_data,)
        if use_sec_return:
            data_tuple += (len_first,)
        #data_tuple += (self.im_idx[index],)
        #data_tuple += (self.im_idx[index].split('/')[-1],)
        #assert 'segment' in self.im_idx[index].split('/')[-1]
        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 10 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:

            for fuse_idx in range(self.multiscan):
                plus_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan


        return data_tuple


# load Semantic KITTI class info

def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
