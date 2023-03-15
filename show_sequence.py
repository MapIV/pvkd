import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as transforms
import glob
import os
from pypcd import pypcd
from labels_colors import color_map

def read_labels(label_file):
    label = np.fromfile(label_file, dtype=np.uint32)
    label = label.reshape((-1))
    upper_half = label >> 16      # get upper half for instances
    lower_half = label & 0xFFFF   # get lower half for semantics
    return lower_half

def to_XYZI_array(points_struct):
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

label_in = '/home/map4/pvkd/out_cyl/semantickitti_multiscan10_dyn'
sequence_pcd_in = '/media/map4/SSD_62/SemanticKitti/semantic_kitti_ms10_v005_dyn/sequences/08/velodyne'

label_list = glob.glob(label_in + '/*')

for f, label_fp in enumerate(label_list):
    print('{}/{}'.format(f, len(label_list)))
    plt.style.use('dark_background')
    idx = os.path.basename(label_fp).split('.')[0] + '.' + os.path.basename(label_fp).split('.')[1]
    # BIN KITTI FILES
    lidar_fp = os.path.join(sequence_pcd_in, idx[:-6] + '.bin')
    lidar = np.fromfile(lidar_fp, dtype=np.float32).reshape(-1, 4)

    # PCD FILES
    # lidar_fp = os.path.join(sequence_pcd_in, idx + '.pcd')
    # points_struct = pypcd.PointCloud.from_path(lidar_fp)
    # lidar = to_XYZI_array(points_struct.pc_data)

    label = read_labels(label_fp)

    if np.any(label == 252):
        print("moving!")

    fig, ax = plt.subplots(figsize=(16, 16))
    # plt.set
    # plt.figure(num=1, figsize=(16, 16), dpi=600, facecolor='w', edgecolor='k')
    colors = [[color_map[i][0]/255, color_map[i][1]/255, color_map[i][2]/255] for i in label]
    ax.scatter(lidar[:, 0],
               lidar[:, 1],
               s=0.5, c=colors, marker='o', facecolor=colors)
    ax.axis('square')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_axis_off()
    # plt.show()

    plt.savefig(os.path.join('/home/map4/pvkd/out_cyl/semantickitti_multiscan10_dyn_fig', idx+'.png'), dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
