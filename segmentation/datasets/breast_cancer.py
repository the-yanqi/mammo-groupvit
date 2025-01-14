# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
import pickle, os
import numpy as np
from mmseg.datasets import DATASETS, CustomDataset
from mmcv.utils import print_log
from mmseg.datasets.builder import PIPELINES
from mmseg.utils import get_root_logger

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from breast_datasets import load_mammogram_img, load_segmentation_mammogram
from torchvision.transforms.functional import InterpolationMode


@DATASETS.register_module()
class BreastCancerDataset(CustomDataset):
    """COCO-Object dataset.

    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    CLASSES = ('background', 'mass', 'calcification')

    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 64, 96]]

    def __init__(self, datalist_dir, datalist_prefix, dataset_type='seg', **kwargs):
        self.datalist_dir = datalist_dir 
        self.datalist_prefix = datalist_prefix
        self.dataset_type = dataset_type 
        super(BreastCancerDataset, self).__init__(**kwargs)
    

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load data with masks.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        self.gt_seg_map_loader = LoadBreastAnnotations(reduce_zero_label=self.reduce_zero_label)

        with open(self.datalist_dir, "rb") as f:
            meta_datalist = pickle.load(f)

        img_info = meta_datalist[self.dataset_type]
        print_log(f'Loaded {len(img_info)} images', logger=get_root_logger())
        return img_info 

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        meta_data_pac = self.img_infos[idx]
        with open(os.path.join(self.datalist_prefix, meta_data_pac['pkl']), "rb") as f:
            data = pickle.load(f)
        img_info = data[meta_data_pac['idx']]

        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def get_gt_seg_map_by_idx(self, idx):
        """Get one ground truth segmentation map for evaluation."""
        meta_data_pac = self.img_infos[idx]
        with open(os.path.join(self.datalist_prefix, meta_data_pac['pkl']), "rb") as f:
            data = pickle.load(f)
        ann_info = data[meta_data_pac['idx']]

        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            meta_data_pac = self.img_infos[idx]
            with open(os.path.join(self.datalist_prefix, meta_data_pac['pkl']), "rb") as f:
                data = pickle.load(f)
            ann_info = data[meta_data_pac['idx']]

            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

@PIPELINES.register_module()
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

@PIPELINES.register_module()
class LoadBreastAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        benign_seg_np, malignant_seg_np = load_segmentation_mammogram(results['ann_info'], results['seg_prefix'])
        #malignant_seg_np = np.where(malignant_seg_np==1,2,0)
        gt_semantic_seg = (benign_seg_np + malignant_seg_np).astype(np.uint8)

        seg_label =  1 if sum(results['ann_info']['ab_label'][:3])>0 else 2
        gt_semantic_seg = np.where(gt_semantic_seg==1,seg_label,0)

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class StandardizerTensor(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb
        

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.to_rgb:       
            results['img'] = results['img'].repeat([3, 1, 1])
        results['img'] = (results['img'] - results['img'].mean()) / torch.max(results['img'].std(), torch.tensor(10 ** (-5)))   
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class CenterCrop:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.crop_func = transforms.CenterCrop(size)

    def __call__(self, sample):
        sample['img'] = self.crop_func(sample['img'])    
        return sample
