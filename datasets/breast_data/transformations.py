import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from . import duke
import random
import torch

class Standardizer(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample["img"]
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        sample["img"] = img
        return sample


class CopyChannel(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample["img"].size()[0] == 1:
            sample["img"] = sample["img"].repeat([3, 1, 1])
        return sample


class ToNumpy(object):
    """
    Use this class to shut up "UserWarning: The given NumPy array is not writeable ..."
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["img"] = np.array(sample["img"])
        sample["bseg"] = np.array(sample["bseg"])
        sample["mseg"] = np.array(sample["mseg"])
        return sample


class Resize:
    def __init__(self, size):
        self.resize_func = transforms.Resize(size)

    def __call__(self, sample):
        sample["img"] = self.resize_func(sample["img"])
        sample["bseg"] = self.resize_func(sample["bseg"])
        sample["mseg"] = self.resize_func(sample["mseg"])
        return sample


class ToTensor:
    def __init__(self):
        self.operator = transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.operator(sample["img"])
        sample["bseg"] = self.operator(sample["bseg"])
        sample["mseg"] = self.operator(sample["mseg"])
        return sample


class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, sample["img"].size)
        sample["img"] = F.affine(sample["img"], *ret)
        sample["bseg"] = F.affine(sample["bseg"], *ret)
        sample["mseg"] = F.affine(sample["mseg"], *ret)
        return sample

class RandomBrightness:
    def __init__(self, p = 0.5, brightness=0):
        self.p = p
        self.transform = transforms.ColorJitter(brightness)
    
    def __call__(self, sample):
        prob = np.random.rand()
        if prob < self.p:
            sample['img'] = self.transform(sample['img'])
        return sample



class RandomFlip:
    def __init__(self, hprob=0.5, vprob=0.5):
        self.hprob = hprob
        self.vprob = vprob

    def __call__(self, sample):
        hrandom = np.random.rand()
        vrandom = np.random.rand()
        if hrandom < self.hprob :
            sample["img"] = F.hflip(sample["img"])
            sample["mseg"] = F.hflip(sample["mseg"])
            sample["bseg"] = F.hflip(sample["bseg"])
        if vrandom < self.vprob :
            sample["img"] = F.vflip(sample["img"])
            sample["mseg"] = F.vflip(sample["mseg"])
            sample["bseg"] = F.vflip(sample["bseg"])
        return sample


class RandomGrayScale:
    def __init__(self, p=0.2, num_output_channels=3, generator=None):
        self.p = p
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator
        self.transform = transforms.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, sample):
        if self.generator.random() < self.p:
            sample["img"] = self.transform(sample["img"])
        return sample


class RandomGaussianBlur:
    def __init__(self, p=0.5, generator=None):
        self.p = p
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator
        self.transform = transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=(23,23))

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample["img"] = self.transform(sample["img"])
        return sample


class RandomErasing:
    def __init__(self, p, scale, ratio, generator=None):
        """
        https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html
        :param p:
        :param scale: range of proportion of erased area against input image.
        :param ratio: range of aspect ratio of erased area.
        :param generator:
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator

    def __call__(self, sample):
        if self.generator.random() < self.p:
            region = transforms.RandomErasing.get_params(sample["img"], scale=self.scale, ratio=self.ratio)
            i, j, h, w, v = region
            # mask out image[:,  i->i+h, j->j+w]
            F.erase(sample["img"], i, j, h, w, v, inplace=True)
            sample["bseg"][0, i:i+h, j:j+w] = 0
            sample["mseg"][0, i:i+h, j:j+w] = 0
        return sample


class RandomResizedCrop:
    def __init__(self, p, scale, ratio, generator=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator

    def __call__(self, sample):
        if self.generator.random() < self.p:
            i, j, h, w = transforms.RandomResizedCrop.get_params(sample["img"], self.scale, self.ratio)
            _, h_img, w_img = sample["img"].size()
            sample["img"] = F.crop(sample["img"], i, j, h, w)
            sample["img"] = F.resize(sample["img"], (h_img, w_img))
            sample["mseg"] = F.crop(sample["mseg"], i, j, h, w)
            sample["mseg"] = F.resize(sample["mseg"], (h_img, w_img))
            sample["bseg"] = F.crop(sample["bseg"], i, j, h, w)
            sample["bseg"] = F.resize(sample["bseg"], (h_img, w_img))
        return sample

class CopyPaste:
    def __init__(self, n = 3, p = 0.5, max_iou_threshold=0, method = 'box', from_other_images = False):
        self.n = n
        self.p = p
        self.scale_segmentation = True
        self.flip_segmentation = True
        self.rotate_segmentation = True
        self.max_iou_threshold = max_iou_threshold
        self.from_other_images = from_other_images
        self.method = method

    def __call__(self, sample):
        _, h, w = sample["img"].shape
        n_segs = np.random.binomial(self.n, self.p)
        mseg = sample['mseg']
        bseg = sample['bseg']
        img = sample['img']

        all_segs = []
        all_candidates = []
        all_annots = []
        for seg in [mseg, bseg]:
            if seg.sum() != 0:
                manno = duke.DDSMAugmenter.get_connected_components(seg.data.numpy()[0, :, :], 1, "malignant")
                # ignore tiny segmentations
                candidates_manno = [lesion for idx, lesion in enumerate(manno) if lesion['Width']* lesion['Height'] > 50]
                all_segs.append(seg)
                all_candidates.append(candidates_manno)
                all_annots.extend([[lesion['X'],lesion['Y'],lesion['Width'],lesion['Height']] for lesion in candidates_manno])

        if len(all_segs) > 0:
            idx = random.choice(range(len(all_segs)))
            seg = all_segs[idx]
            candidates_manno = all_candidates[idx]
            if len(candidates_manno) > 0:
                for i in range(n_segs):
                    one_manno = random.choice(candidates_manno)
                    orig_segmentation = torch.from_numpy(one_manno['mask']).unsqueeze(0) * img
                    trans_segmentation, bbox = self.transform_seg(orig_segmentation.clone(), img)
                    seg_x1, seg_y1, seg_w, seg_h = bbox
                    
                    seg_x1_new = random.randint(0, w - seg_w)
                    seg_y1_new = random.randint(0, h - seg_h)
                   # need to check if the copy lesion overlap with existing lesions
                    paste = True
                    for annot in all_annots:
                        seg_annot = [seg_x1_new, seg_y1_new, seg_w, seg_h]

                        if self.get_iou(seg_annot, annot) > self.max_iou_threshold:
                            paste = False

                    if paste:
                        if self.method == 'segmentation':
                            seg_indices = (trans_segmentation[0] != 0).nonzero()
                            seg_indices_x = seg_indices[:,1] + (seg_x1_new - seg_x1)
                            seg_indices_y = seg_indices[:,0] + (seg_y1_new - seg_y1)
 
                            img[0,seg_indices_y, seg_indices_x] = trans_segmentation[0,seg_indices[:,0],seg_indices[:,1]]
                            seg[0,seg_indices_y, seg_indices_x] = torch.unique(seg)[-1]
                        elif self.method == 'box':
                            img[0,seg_y1_new: seg_y1_new+seg_h, seg_x1_new: seg_x1_new+seg_w] = trans_segmentation[0,seg_y1: seg_y1+seg_h, seg_x1: seg_x1+seg_w]
                            seg[0,seg_y1_new: seg_y1_new+seg_h, seg_x1_new: seg_x1_new+seg_w] = torch.unique(seg)[-1]

                        all_annots.append(seg_annot)
        else:
            for i in range(n_segs):
                region = transforms.RandomErasing.get_params(img, scale=(5.e-03, 3.e-02), ratio=(0.7,1.3))
                y1, x1, r_h, r_w, v = region
                
                x1_new = random.randint(0, w - r_w)
                y1_new = random.randint(0, h - r_h)
                img[0, y1_new:y1_new+r_h, x1_new:x1_new+r_w] = img[0, y1:y1+r_h, x1:x1+r_w]

        sample['mseg'] = mseg
        sample['bseg'] = bseg
        sample['img'] = img
        return sample

    def get_iou(self, bb1, bb2):

        x1, y1, x2, y2 = bb1[0], bb1[1], bb1[0]+bb1[2], bb1[1]+bb1[3]
        a1, b1, a2, b2 = bb2[0], bb2[1], bb2[0]+bb2[2], bb2[1]+bb2[3]
        
        x_left = max(x1, a1)
        y_top = max(y1, b1)
        x_right = min(x2, a2)
        y_bot = min(y2, b2)

        if x_right < x_left or y_bot < y_top:
            return 0.0

        intersection_area = (x_right - x_left + 1) * (y_bot - y_top + 1)
        bb1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        bb2_area = (a2 - a1 + 1) * (b2 - b1 + 1)

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou
                    
    def get_segmentation_wh(self,segmentation):
        y_edge_top, y_edge_bottom = duke.get_edge_values(segmentation, "y")
        x_edge_left, x_edge_right = duke.get_edge_values(segmentation, "x")

        width = x_edge_right - x_edge_left
        height = y_edge_bottom - y_edge_top
        return x_edge_left, y_edge_top, width, height

    def transform_seg(self, segmentation, img):
        # center the lesion
        _, h, w = segmentation.shape
        x0, y0, ww, hh = self.get_segmentation_wh(segmentation[0].data.numpy())
        if self.method == 'segmentation':
            lesion = segmentation[:,y0:y0+hh,x0:x0+ww]
        elif self.method == 'box':
            lesion = img[:,y0:y0+hh,x0:x0+ww]

        lesion = lesion[0].data.numpy()
        if self.flip_segmentation:
            hrandom = np.random.rand()
            vrandom = np.random.rand()
            if hrandom < 0.5 :
                lesion = np.flip(lesion, axis=1) #horizontal
            if vrandom < 0.5 :
                lesion = np.flip(lesion, axis=0) #vertical

        if self.rotate_segmentation:
            rotate_version = random.choice([1, 2, 3, 4])
            if rotate_version == 1:
                lesion = np.rot90(lesion, k=1)  # Rotate 90 degrees
            if rotate_version == 2:
                lesion= np.rot90(lesion, k=2)  # Rotate 180 degrees
            if rotate_version == 3:
                lesion = np.rot90(lesion, k=3) # Rotate 270 degrees
            if rotate_version == 4:
                pass  # No rotation
        
        lesion = torch.from_numpy(lesion.copy()).unsqueeze(0)
        _, hh, ww = lesion.shape
        if self.scale_segmentation:
            hh = min(int(random.uniform(0.8, 2) * hh),h)
            ww = min(int(random.uniform(0.8, 2) * ww),w)
            lesion = F.resize(lesion, size = (hh, ww))

        c_x0 = int((w-ww)/2)
        c_y0 = int((h-hh)/2)
        c_segmentation = torch.zeros(1,h,w).to(device=segmentation.device)
        c_segmentation[0,c_y0:c_y0 + hh, c_x0:c_x0 + ww] = lesion
        return c_segmentation, [c_x0, c_y0, ww, hh]

    def load_seg_from_other_images(self):
        segmentation = None
        if self.from_other_images:
            while segmentation is None:
                seg_id = random.randint(0, self.ffdm_seg_len - 1)
                seg_path = self.ffdm_seg_list[seg_id]
                seg_class = self.lesion_seg_suffix_to_class_dict[seg_path.split('.')[-3]]
                try:
                    data = h5.File(seg_path, 'r')
                    segmentation = np.array(data['img_box'])
                    data.close()
                except:
                    print(f'Failed to load img_box or seg_box from {seg_path}. Trying to load another segmentation..')
        else:
            segmentation = sample['mseg']


omni_augmentation = [RandomFlip(0.5, 0),
                     #RandomGrayScale(p=0.2), # this blacks out the entire image
                     #RandomGaussianBlur(p=0.5) # this is very slow
                     ]
omni_tensor_transformation = [RandomResizedCrop(p=0.5, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                     RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3)),
                     RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6)),
                     RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8))
                              ]
                      
standard_augmentation = [
    RandomFlip(0.5, 0.5),
    RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
]

medium_augmentation = [
    RandomFlip(0.5, 0),
    RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.3), shear=(-10, 10)),
]

weak_augmentation = [
    RandomFlip(0.5, 0.5),
    RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
]

seg_augmentation = [
    RandomFlip(0.5, 0.5), 
    RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)), 
]
seg_tensor_augmentation =[
    RandomBrightness(p=0.5, brightness=(0.2,0.4))
]

test_time_augmentation = [
    RandomFlip(0.5, 0.5), RandomAffine(degrees=(-15, 15)),
    ]

to_tensor = [ToNumpy(), ToTensor(), Standardizer()]


def compose_transform(augmentation=None, resize=None, image_format="greyscale", copy_paste=False):
    basic_transform = []
    # add augmentation
    if augmentation is not None:
        if augmentation == "standard":
            basic_transform += standard_augmentation
        elif augmentation == "omni":
            basic_transform += omni_augmentation
        elif augmentation == "mid":
            basic_transform += medium_augmentation
        elif augmentation == "weak":
            basic_transform += weak_augmentation
        elif augmentation == "seg":
            basic_transform += seg_augmentation
        elif augmentation == "test_time":
            basic_transform += test_time_augmentation
        else:
            raise ValueError("invalid augmentation {}".format(augmentation))

    # add resize
    if resize is not None:
        basic_transform += [Resize(resize)]

    # add to tensor and normalization
    basic_transform += to_tensor

    # tensor processing
    if copy_paste:
        basic_transform += [CopyPaste()]

    if augmentation is not None:
        if augmentation == "omni":
            basic_transform += omni_tensor_transformation
        elif augmentation == "seg":
            basic_transform += seg_tensor_augmentation 
        

    # add channel duplication
    if image_format == "greyscale":
        basic_transform += [CopyChannel()]
    return transforms.Compose(basic_transform)

