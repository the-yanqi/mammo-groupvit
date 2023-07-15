import pickle, random
from functools import partial
from mmcv.parallel import collate
import numpy as np


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.breast_data import datasets, transformations
from datasets.builder import build_text_transform, build_breast_transform

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_breast_dataset(datalist_dir,
                      img_dir,
                      segmentation_dir,
                      data_mode,
                      imaging_modality="mammo",
                      load_seg=False,
                      config=None):
    """
    Entry-level function
    Args:
        pos_to_neg_ratio: The ratio of positive cases to negative cases. None is to use all examples.
        mal_as_positives: Definition of positive examples
                          True: only malignant examples as positive
                          False: Either contain benign and malignant lesions as positives
        num_positives: the exact number of positive examples. None is to use all the positive cases.
        
    :return:
    """
    # step #1: load data list
    with open(datalist_dir, "rb") as f:
        train_dl, val_dl, ts_dl = pickle.load(f)


    training_transformations = build_breast_transform(is_train=True,config=config.img_aug)
    val_transformations = build_breast_transform(is_train=False,config=config.img_aug)
    ts_transformations = build_breast_transform(is_train=False,config=config.img_aug)

    training_text_transformations = build_text_transform(is_train=True, config=config.text_aug)
    val_text_transformations = build_text_transform(is_train=False, config=config.text_aug)
    ts_text_transformations = build_text_transform(is_train=False, config=config.text_aug)
    
    if data_mode == 'image':
        tr_ds = datasets.ImageTextDataset(data_list=train_dl,
                                          img_dir=img_dir,
                                          seg_dir=segmentation_dir,
                                          imaging_modality=imaging_modality,
                                          image_transformations=training_transformations,
                                          text_transformations=training_text_transformations, 
                                          is_train=True,
                                          load_seg=False)
        val_ds = datasets.ImageTextDataset(val_dl, img_dir, segmentation_dir, imaging_modality, val_transformations,  
                                        val_text_transformations,is_train=False,load_seg=load_seg)
        ts_ds = datasets.ImageTextDataset(ts_dl, img_dir, segmentation_dir, imaging_modality, ts_transformations, 
                                        val_text_transformations,is_train=False,load_seg=load_seg)
    elif data_mode == 'breast':
        train_dl = datasets.BreastDataset.group_dl_for_breast(train_dl)
        val_dl = datasets.BreastDataset.group_dl_for_breast(val_dl)
        ts_dl = datasets.BreastDataset.group_dl_for_breast(ts_dl)

        
        tr_ds = datasets.BreastDataset(train_dl, img_dir, segmentation_dir, training_transformations)
        val_ds = datasets.BreastDataset(val_dl, img_dir, segmentation_dir, val_transformations)
        ts_ds = datasets.BreastDataset(ts_dl, img_dir, segmentation_dir, test_transformations)
    
    return tr_ds, val_ds, ts_ds


def build_breast_dataloader(config, data_mode="image" ,imaging_modality="mammo", load_seg=False):
    
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dataset_train, dataset_val, dataset_test = get_breast_dataset(config.datalist_dir,
                      config.img_dir,
                      config.segmentation_dir,
                      data_mode,
                      imaging_modality,
                      load_seg=load_seg,
                      config=config)

    print(f'local rank {local_rank} / global rank {rank} \
        successfully build train,val,test dataset')
     
    sampler_train = DistributedSampler(dataset_train, world_size, rank)
    sampler_val = DistributedSampler(dataset_val, world_size, rank, shuffle=False)
    sampler_test = DistributedSampler(dataset_test, world_size, rank, shuffle=False)

    dc_collate = partial(collate, samples_per_gpu=config.batch_size)
   
    init_fn = partial(worker_init_fn, num_workers=config.num_workers, rank=rank, seed=config.seed)
    data_loader_train = DataLoader(
        dataset_train,
        collate_fn=dc_collate,
        batch_size=config.batch_size,
        sampler=sampler_train,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn)

    data_loader_val = DataLoader(
        dataset_val,
        collate_fn=dc_collate,
        batch_size=config.batch_size,
        sampler=sampler_val,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn)

    data_loader_test = DataLoader(
        dataset_test,
        collate_fn=dc_collate,
        batch_size=config.batch_size,
        sampler=sampler_test,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test
