# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from utils import build_dataset_class_tokens
from breast_datasets import load_mammogram_img, load_segmentation_mammogram

import numpy as np

from .group_vit_seg import GroupViTSegInference


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config.cfg)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset):

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False)
    return data_loader


def build_seg_inference(model, dataset, text_transform, config):
    cfg = mmcv.Config.fromfile(config.cfg)
    if len(config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    with_bg = dataset.CLASSES[0] == 'background'
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES
    text_tokens = build_dataset_class_tokens(text_transform, config.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg
    seg_model = GroupViTSegInference(model, text_embedding, **kwargs)

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class LoadBreastImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = load_mammogram_img(results['img_info'], results['img_prefix'])

        if self.to_float32:
            img =  np.asarray(img).astype(np.float32)

        results['PatientID'] = results['img_info']['PatientID']
        results['StudyUID'] = results['img_info']['StudyUID']
        results['View'] = results['img_info']['View']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

def build_seg_breast_pipeline():
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    meta_keys = dict(meta_keys=['PatientID','StudyUID', 'View', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg'])
    test_pipeline = Compose([
        LoadBreastImageFromFile(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='NormalizeTensor', **img_norm_cfg),
                dict(type='Collect', keys=['img'], **meta_keys),
            ])
    ])
    return test_pipeline

def build_seg_demo_pipeline():
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = Compose([
        LoadImage(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline