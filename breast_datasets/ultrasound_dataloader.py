"""
File that contains all data loader for ultrasound
"""
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os, sys, logging, json
import numpy as np
from PIL import Image
from bisect import bisect_left
# TODO:this is ugly change it later
sys.path.append("../utilities")
import reading_images

def load_an_img(img_dir, format):
    """
    Function that loads an image into PIL image format
    Always return 3-channel RGB format PIL image
    :param img_dir:
    :param format:
    :return:
    """
    assert format in [".hdf5", ".npy", ".png", ".pt"], "Bad input format {0}".format(format)
    try:
        # if it's tensor format, directly return tensor
        if format == ".pt":
            loaded_tensor = torch.load(img_dir)
            # RGB Tensor
            if len(loaded_tensor.size()) == 3:
                loaded_tensor = loaded_tensor.permute(2, 0, 1)
            else:
                loaded_tensor = loaded_tensor.unsqueeze(0)
            pil_img = F.to_pil_image(loaded_tensor.float()).convert("RGB")
        # for other format first transform to PIL images
        elif format == ".hdf5":
            img = reading_images.read_image_mat(img_dir)
            pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        elif format == ".npy":
            img = np.load(img_dir)
            pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        elif format == ".png":
            pil_img = Image.open(img_dir).convert("RGB")
    except Exception as err:
        logging.error("Error in loading {0}".format(img_dir))
        logging.error(err)
        logging.error(err, exc_info=True)
        pil_img = Image.fromarray(np.zeros((3, 512, 512)).astype(np.uint8))
    return pil_img


def ultrasound_collate_function(batch):
    """
    Collate functions used to collapse a mini-batch.
    :param batch:
    :return:
    """
    batch_img = torch.cat([img for img,_,_,_,_,_ in batch], dim=0)
    batch_label = torch.cat([label for _,label,_,_,_,_ in batch], dim=0)
    batch_lateral = np.concatenate([lateral for _, _, lateral, _, _,_ in batch], axis=0)
    exam_id = [x for img, _, _, exam_id, _, _ in batch for x in [exam_id] * img.size()[0]]
    img_filenames = [single_name for _, _, _, _, file_names, _ in batch for single_name in file_names]
    img_time = [single_time for _, _, _, _, _, img_times in batch for single_time in img_times]

    return batch_img, batch_label, batch_lateral, exam_id, img_filenames, img_time


class BucketQueue:
    """
    Object that queues each exam according to number of images per exam
    """
    def __init__(self, data_list):
        # filter out exams that we don't have any laterals
        self.data_list = data_list
        self.num_laterals = [len(x["laterality"]) if "laterality" in x and len(x["laterality"]) > 0 else 0 for x in
                             data_list]
        # create thresholds
        self.bucket_thresholds = [np.percentile(self.num_laterals, 30), np.percentile(self.num_laterals, 60),
                                  np.percentile(self.num_laterals, 75), np.percentile(self.num_laterals, 90),
                                  np.max(self.num_laterals)]
        # create list of queues
        self.idx_queue_list = [[] for _ in range(len(self.bucket_thresholds))]
        for i in range(len(self.num_laterals)):
            # ignore index where we have no laterals
            if self.num_laterals[i] > 0:
                for j in range(len(self.bucket_thresholds)):
                    if self.num_laterals[i] <= self.bucket_thresholds[j]:
                        self.idx_queue_list[j].append((i, self.num_laterals[i]))
                        break
        self.deplete = False

    def update(self):
        """
        Method that removes deleted queues and thresholds
        :return:
        """
        self.bucket_thresholds = [self.bucket_thresholds[i] for i in range(len(self.bucket_thresholds)) if len(self.idx_queue_list[i]) > 0]
        self.idx_queue_list = [self.idx_queue_list[i] for i in range(len(self.idx_queue_list)) if len(self.idx_queue_list[i]) > 0]
        if len(self.bucket_thresholds) == 0:
            self.deplete = True

    def sample_a_batch(self, max_imgs_per_batch, random=False):
        """
        Method that samples a minibatch from the queue list
        :param max_imgs_per_batch:
        :param random:
        :return:
        """
        output = []
        current_limit = max_imgs_per_batch
        need_sample = True
        while need_sample:
            # select the bucket with largest number of elements
            largest_threshold_idx = bisect_left(self.bucket_thresholds, current_limit)
            selected_queue = self.idx_queue_list[largest_threshold_idx - 1]
            # select an element out of the bucket
            if random:
                bucket_idx = np.random.randint(low=0, high=len(selected_queue))
                current_data_idx, new_imgs_added = selected_queue[bucket_idx]
                del selected_queue[bucket_idx]
            else:
                current_data_idx, new_imgs_added = selected_queue.pop()
            # update status
            output.append(current_data_idx)
            current_limit -= new_imgs_added
            self.update()
            # check if we can take more images
            if self.deplete or current_limit <= self.bucket_thresholds[0]:
                need_sample = False
        return output

    def give_all_batches(self, max_imgs_per_batch, random=False):
        """
        Method that creates all minibatch index
        :param max_imgs_per_batch:
        :param random:
        :return:
        """
        all_batches = []
        while not self.deplete:
            all_batches.append(self.sample_a_batch(max_imgs_per_batch, random))
        return all_batches


class MaxImageNumberSampler(Sampler):
    """
    Object that creates minibatches which has strictly less number of images than the input
    """
    def __init__(self, data_list, max_imgs_per_batch=150, random=False):
        super().__init__(None)
        self.data_list = data_list
        self.max_imgs_per_batch = max_imgs_per_batch
        # create a queuelist object
        self.bucket_queue = BucketQueue(data_list)
        # calculate all batch index
        self.all_batches = self.bucket_queue.give_all_batches(max_imgs_per_batch, random)

    def __iter__(self):
        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)


class UltrasoundDataset(Dataset):
    """
    Dataset object for ultrasound images on exam level.
    Supports loading of hdf5, npy, and png format.
    """
    # TODO: this part need to be changed for validation and training
    # TODO: we need to formalize what to do for training/validation/test
    transform_aug = transforms.Compose([
        # old augmentation
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomResizedCrop(size=(512, 512)),
        # transforms.RandomRotation(degrees=(-45,45)),

        # new augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1)),
    ])


    transform_noaug = transforms.Compose([
        #transforms.CenterCrop((512, 512)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1)),
    ])

    def __init__(self, parameters, data_list, phase, augmentation=True, customized_transform=None, max_img_per_exam=None):
        # fetch parameters
        self.img_folder = parameters["image_folder"]
        self.img_format = parameters["image_format"]
        self.data_list = data_list
        self.augmentation = augmentation
        self.phase = phase
        self.customized_transform = customized_transform
        self.remove_annotation = parameters["remove_annotation"]
        self.max_img_per_exam = max_img_per_exam
        self.inpaint_dir = parameters["annotation_inpaint_folder"]

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        exam_id = data_pac["AccessionnNumber"]
        exam_path_original = os.path.join(self.img_folder, exam_id)
        exam_path_inpaint = os.path.join(self.inpaint_dir, exam_id)

        # load image paths
        img_filenames = list(data_pac["laterality"].keys())
        img_folders = []

        # filter out images with annotation that we have clean version
        if self.remove_annotation:
            assert "match_dict" in data_pac
            # NOTE THAT THERE ARE TWO NESTED MATCH_DICT
            match_dict = data_pac["match_dict"]["match_dict"]
            clean_files = []
            for filename in img_filenames:
                # case 1: clean image
                if filename not in match_dict:
                    clean_files.append(filename)
                    img_folders.append(exam_path_original)
                # case 2: no replacement
                elif not "matched" in match_dict[filename]:
                    clean_files.append(filename)
                    # case 2.1: there is annotation, so directly load the inpainted images
                    if os.path.exists(os.path.join(exam_path_inpaint, "{}{}".format(filename, self.img_format))):
                        img_folders.append(exam_path_inpaint)
                    # case 2.2: segmentation model thinks there is not annotation
                    else:
                        img_folders.append(exam_path_original)
            img_filenames = clean_files
        if len(img_folders) == 0:
            img_dirs = [os.path.join(exam_path_original, "{0}{1}".format(x, self.img_format)) for x in img_filenames]
        else:
            assert len(img_folders)==len(img_filenames)
            img_dirs = [os.path.join(img_folders[i], "{0}{1}".format(img_filenames[i], self.img_format)) for i in range(len(img_filenames))]

        # cap the number of images per exam as an augmentation measure
        if self.max_img_per_exam is not None:
            random_index = np.random.choice(list(range(len(img_dirs))), self.max_img_per_exam)
            img_dirs = [img_dirs[x] for x in random_index]
            img_filenames = [img_filenames[x] for x in random_index]
        # read the images and transform to PIL images
        pil_imgs = [load_an_img(x, self.img_format) for x in img_dirs]

        # transform images
        if self.customized_transform is not None:
            selected_transform = self.customized_transform
        elif self.augmentation:
            selected_transform = self.transform_aug
        else:
            selected_transform = self.transform_noaug
        transformed_imgs = [selected_transform(x).unsqueeze(0) for x in pil_imgs]
        if len(transformed_imgs) == 0:
            logging.error("len(transformed_imgs)={0}".format(len(transformed_imgs)))
            logging.error("len(pil_imgs)={0}".format(len(pil_imgs)))
            logging.error("len(img_dirs)={0}".format(len(img_dirs)))
            logging.error("exam_path={0}".format(exam_path_original))
            raise Exception("len(transformed_imgs)=0")

        # concat images
        cat_tensor = torch.cat(transformed_imgs, dim=0)

        # load lateriality
        lateral_vector = [data_pac["laterality"][x] for x in img_filenames]

        # load metadata
        if "exam_time" in data_pac:
            img_time = [data_pac["exam_time"][x] for x in img_filenames]
        else:
            img_time = [np.NaN for _ in img_filenames]

        # create label tensor
        cancer_label = data_pac["cancer_label"]
        if isinstance(cancer_label, str):
            cancer_label = json.loads(cancer_label.replace("\'", "\""))
        label_list = [np.expand_dims(np.array([cancer_label["{0}_benign".format(x)], cancer_label["{0}_malignant".format(x)]]), 0) for x in lateral_vector]
        label_tensor = torch.Tensor(np.concatenate(label_list, axis=0)).float()

        return cat_tensor, label_tensor, lateral_vector, exam_id, img_filenames, img_time

    def __len__(self):
        return len(self.data_list)