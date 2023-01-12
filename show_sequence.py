import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as transforms
import glob
import os
from labels_colors import color_map

label_in = '/home/pe/pvkd/out_cyl/test/sequences/15/predictions/'
sequence_pcd_in = '/home/pe/iSSD1/semantic_kitti/sequences/15/velodyne'

label_list = glob.glob(label_in + '/*')

for label_fp in label_list:
    plt.style.use('dark_background')
    idx = os.path.basename(label_fp).split('.')[0]
    lidar_fp = os.path.join(sequence_pcd_in, idx + '.bin')
    label = np.fromfile(label_fp, dtype=np.uint32).reshape(-1)
    lidar = np.fromfile(lidar_fp, dtype=np.float32).reshape(-1, 4)

    fig, ax = plt.subplots()
    plt.figure(num=1, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    colors = [[color_map[i][0]/255, color_map[i][1]/255, color_map[i][2]/255] for i in label]
    ax.scatter(lidar[:, 0],
               lidar[:, 1],
               s=0.1, c=colors, marker='.')
    ax.axis('equal')
    ax.set_axis_off()
    plt.savefig(os.path.join('/home/pe/pvkd/kitti_demo', idx+'.png'), dpi=200)
    print(label.shape)
    print(lidar.shape)

