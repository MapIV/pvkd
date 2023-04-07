# -*- coding:utf-8 -*-
# author: Xinge
# @file: data_builder.py 

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_tta, collate_fn_BEV_ms, collate_fn_BEV_ms_tta
from dataloader.pc_dataset import get_pc_model_class

def build(dataset_config,
          val_dataloader_config,
          grid_size=[480, 360, 32]):

    data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]
    val_ref = val_dataloader_config["return_ref"]
    label_mapping = dataset_config["label_mapping"]
    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=None)
    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        return_test=True,
        use_tta=True,
    )

    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV_tta,
                                                     shuffle=False,
                                                     num_workers=val_dataloader_config["num_workers"])

    return val_dataset_loader, val_dataset, val_pt_dataset
