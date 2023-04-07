# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import model_builder, loss_builder, inference_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

import warnings

warnings.filterwarnings("ignore")

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']

    model_load_path = train_hypers['model_load_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        # my_model = load_checkpoint(model_load_path, my_model)
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    test_dataset_loader, test_dataset, test_pt_dataset = inference_builder.build(dataset_config,
                                                                   val_dataloader_config,
                                                                   grid_size=grid_size)


    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    voting_num = 4


    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    inf_time = []
    pbar = tqdm(total=len(test_dataset_loader))
    time.sleep(1)
    # my_model.eval()
    with torch.no_grad():
        for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(test_dataset_loader):
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]
            cur_time = time.perf_counter()
            predict_labels = my_model(test_pt_fea_ten, test_grid_ten, val_batch_size, test_grid, voting_num, use_tta=True)
            inf_time.append(time.perf_counter() - cur_time)
            predict_labels = torch.argmax(predict_labels, dim=0).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()
            test_pred_label = np.expand_dims(predict_labels,axis=1)
            test_pred_label = test_pred_label.astype(np.uint32)
            if args.map_inference:
                voxelized_pcd = test_pt_dataset.get_voxel_item(i_iter_test)
                color_map = test_pt_dataset.label_color_map()
                demo_plot(voxelized_pcd, test_pred_label, color_map)
                test_pt_dataset.update_labels_map(test_pred_label)
            else:
                ex_idx = os.path.basename(test_pt_dataset.im_idx[test_index[0]])
                new_save_dir = os.path.join(output_path, ex_idx[:-3] + 'label')
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    os.makedirs(os.path.dirname(new_save_dir))
                test_pred_label.tofile(new_save_dir)
            pbar.update(1)
    # del test_grid, test_pt_fea, test_grid_ten, test_index
    pbar.close()
    if args.map_inference:
        test_pt_dataset.save_labels_map_rgb(os.path.join(output_path, 'all_classes_'))
    print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
    print('Remapping script can be found in semantic-kitti-api.')
    print('Average inference time:', np.mean(inf_time[1:]))

def demo_plot(lidar, label, color_map):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 16))
    colors = [[color_map[i, 0] / 255, color_map[i, 1] / 255, color_map[i, 2] / 255, 1] for i in label]
    # print(lidar.shape, label.shape, colors.shape)
    ax.scatter(lidar[:, 0],
               lidar[:, 1],
               s=0.5, c=colors, marker='o', facecolor=colors)
    ax.axis('square')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_simple.yaml')
    parser.add_argument('-o', '--output_path', default='out_cyl/output')
    parser.add_argument('-m', '--map_inference', default=False, action='store_true')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
