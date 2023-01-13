import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as transforms
import glob
import os
from pypcd import pypcd
from labels_colors import color_map

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

label_in = '/home/pe/pvkd/out_cyl/testpcd/'
sequence_pcd_in = '/home/pe/iSSD2/deepen_dataset_raw/HavlOgYpwQNuWPnzpPti53w5/original_pcd/'

label_list = glob.glob(label_in + '/*')

for f, label_fp in enumerate(label_list):
    print('{}/{}'.format(f, len(label_list)))
    plt.style.use('dark_background')
    idx = os.path.basename(label_fp).split('.')[0] + '.' + os.path.basename(label_fp).split('.')[1]
    # BIN KITTI FILES
    # lidar_fp = os.path.join(sequence_pcd_in, idx + '.bin')
    # lidar = np.fromfile(lidar_fp, dtype=np.float32).reshape(-1, 4)

    # PCD FILES
    lidar_fp = os.path.join(sequence_pcd_in, idx + '.pcd')
    points_struct = pypcd.PointCloud.from_path(lidar_fp)
    lidar = to_XYZI_array(points_struct.pc_data)

    label = np.fromfile(label_fp, dtype=np.uint32).reshape(-1)

    fig, ax = plt.subplots(figsize=(16,16))
    # plt.set
    # plt.figure(num=1, figsize=(16, 16), dpi=600, facecolor='w', edgecolor='k')
    colors = [[color_map[i][0]/255, color_map[i][1]/255, color_map[i][2]/255] for i in label]
    ax.scatter(lidar[:, 0],
               lidar[:, 1],
               s=0.5, c=colors, marker='o', facecolor=colors)
    ax.axis('square')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_axis_off()
    # plt.show()

    plt.savefig(os.path.join('/home/pe/pvkd/pcd_demo', idx+'.png'), dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
