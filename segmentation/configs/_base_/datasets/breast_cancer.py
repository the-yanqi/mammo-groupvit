# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

_base_ = ['../custom_import.py']
# dataset settings
dataset_type = 'BreastCancerDataset'
img_norm_cfg = dict(to_rgb=True)
meta_keys = dict(meta_keys=['PatientID','StudyUID', 'View', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg'])

test_pipeline = [
    dict(type='LoadBreastImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2048, 512),
        img_scale=(2944, 1920),
        img_ratios=[0.25],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='StandardizerTensor', **img_norm_cfg),
            dict(type='Collect', keys=['img'], **meta_keys),
        ])
]

# for dataset
data = dict(
    test=dict(
        type=dataset_type,
        datalist_dir ='/gpfs/data/geraslab/Yanqi/image-to-text/datalist/mammo_jp_09082022_withreport_meta_datalist_09292023_clean.pkl',
        datalist_prefix= '/gpfs/data/geraslab/Yanqi/image-to-text/mammo_data',
        img_dir='/gpfs/data/geraslab/jp4989/data/2021.07.16.combined_ffdm_cropped',
        ann_dir='/gpfs/data/geraslab/jp4989/extract_dbt_segmentation/2022.06.12.cropped_segmentation',
        pipeline=test_pipeline))

test_cfg = dict(bg_thresh=.95, mode='whole')
#test_cfg = dict(bg_thresh=.95, mode='slide', stride=(224, 224), crop_size=(448, 448))

