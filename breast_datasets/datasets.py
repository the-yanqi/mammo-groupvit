import logging, os, torch, itertools
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as F
from . import loading_mammogram, duke
from bisect import bisect_left

from PIL import Image
import numpy as np
import pickle
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]

def img_dl_mal_func(datapac):
    # A boolean function identify if a datum is malignant
    return datapac['image_cancer_label_bounding_box']['malignant'] == 1

def img_dl_pos_func(datapac):
    # A datum is considered positive if it has either matched pathology or annotations
    return len(datapac["lesions"]) > 0
    #return len(datapac["lesions"]) > 0 or (datapac["image_cancer_label_mml"] != "n/a" and datapac["image_cancer_label_mml"]["malignant"])

def load_us_image(img_dir):
    """
    Function that loads an image into PIL image format
    Always return 3-channel RGB format PIL image
    :param img_dir:
    :param format:
    :return:
    """
    img = np.load(img_dir)
    pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return pil_img


def load_mammogram_img(meta_data, img_dir, crop_size=(2944,1920)):
    """
    Function that loads a mammogram image using the meta data
    :param meta_data:
    :param img_dir:
    :param crop_size:
    :return:
    """
    img_path = os.path.join(img_dir, meta_data["hdf5_path"])
    loading_view = meta_data["View"][0] + "-" + meta_data["View"][1:]
    img = loading_mammogram.load_mammogram_img(img_path, crop_size, loading_view,
                                               meta_data["best_center"], meta_data["horizontal_flip"])
    img_pil = Image.fromarray(img / img.max())
    return img_pil


def load_us_img(meta_data, img_dir, crop_size=None):
    """
    loader function for ultrasound
    :param meta_data:
    :param img_dir:
    :param crop_size:
    :return:
    """
    img_path = os.path.join(img_dir, meta_data["StudyUID"], meta_data["filename"]+".npy")
    img = np.load(img_path)
    pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return pil_img


def load_segmentation_mammogram(meta_data, seg_dir, crop_size=(2944,1920)):
    """
    Load segmentation and return the numpy matrices
    :param meta_data:
    :param seg_dir:
    :param crop_size:
    :return:
    """
    # When there is no lesions, return zero masks.
    # Assumption: lesions is a field in the metadata and lesions is a list.
    if len(meta_data["lesions"]) == 0:
        return np.zeros(crop_size), np.zeros(crop_size)
    else:
        all_postfix = ["", "_rest", "_distortion"]
        short_file_path = meta_data["short_file_path"]
        loading_view = meta_data["View"][0] + "-" + meta_data["View"][1:]
        possible_benign_seg_paths = [os.path.join(seg_dir, f"{short_file_path}.benign{postfix}.hdf5") for postfix in all_postfix]
        possible_malignant_seg_paths = [os.path.join(seg_dir, f"{short_file_path}.malignant{postfix}.hdf5") for postfix in all_postfix]
        benign_seg_np = np.zeros(crop_size)
        malignant_seg_np = np.zeros(crop_size)
        for benign_seg_path in possible_benign_seg_paths:
            if os.path.exists(benign_seg_path):
                benign_seg_np += loading_mammogram.load_mammogram_img(benign_seg_path, crop_size, loading_view,
                                                                     meta_data["best_center"],
                                                                     meta_data["horizontal_flip"])
        for malignant_seg_path in possible_malignant_seg_paths:
            if os.path.exists(malignant_seg_path):
                malignant_seg_np += loading_mammogram.load_mammogram_img(malignant_seg_path, crop_size, loading_view,
                                                                     meta_data["best_center"],
                                                                     meta_data["horizontal_flip"])
        return benign_seg_np, malignant_seg_np


def load_segmentation_us(meta_data, seg_dir, crop_size=(2944,1920)):
    """
    Load segmentation and return the numpy matrices
    :param meta_data:
    :param seg_dir:
    :param crop_size:
    :return:
    """
    benign_seg_path = os.path.join(seg_dir, "{}_{}_ben.npy".format(meta_data["StudyUID"], meta_data["filename"]))
    malignant_seg_path = os.path.join(seg_dir, "{}_{}_mal.npy".format(meta_data["StudyUID"], meta_data["filename"]))
    if not os.path.exists(benign_seg_path):
        benign_seg_np = np.zeros(crop_size)
    else:
        benign_seg_np = np.load(benign_seg_path)
    if not os.path.exists(malignant_seg_path):
        malignant_seg_np = np.zeros(crop_size)
    else:
        malignant_seg_np = np.load(malignant_seg_path)
    return benign_seg_np, malignant_seg_np

def collate_func_img(batch):
    """
    Collate functions used to collapse a mini-batch for single modal dataset.
    :param batch:
    :return:
    """
    img_list = []
    cancer_label_list = []
    annotation_list = []
    seg_list = []
    meta_list = []
    for img, cancer_label, annotations, segmentations, meta_dict in batch:
        img_list.append(img.unsqueeze(0))
        cancer_label_list.append(cancer_label.unsqueeze(0))
        annotation_list.append(annotations)
        seg_list.append(segmentations.unsqueeze(0))
        meta_list.append(meta_dict)
    return torch.cat(img_list, dim=0), torch.cat(cancer_label_list, dim=0), annotation_list, torch.cat(seg_list, dim=0), meta_list


def collate_func_breast(batch):
    flatten = list(itertools.chain(*batch))
    return collate_func_img(flatten)

def collate_func_seg(batch):
    img_list = []
    seg_list = []
    for img, segmentations in batch:
        img_list.append(img.unsqueeze(0))
        seg_list.append(segmentations.unsqueeze(0))
    return torch.cat(img_list, dim=0), torch.cat(seg_list, dim=0) 

def resolve_cancer_label(datum, cancer_label_col="image_cancer_label_mml"):
    label = datum[cancer_label_col]
    if cancer_label_col == "image_cancer_label_mml":
        if label == "n/a":
            return torch.Tensor([0,0])
        else:
            return torch.Tensor([label["benign"], label["malignant"]])
    elif cancer_label_col == "birads":
        birads_mapping = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2}
        # missing birads
        if label == -1:
            if datum["image_cancer_label_mml"] == "n/a":
                return torch.Tensor([1]).long()
            elif datum["image_cancer_label_mml"]["malignant"] == 1:
                return torch.Tensor([2]).long()
            else:
                return torch.Tensor([1]).long()
        else:
            return torch.Tensor([birads_mapping[label]]).long()

def resolve_label(data_pac, label, breast_classes):
    if label == 'abnormality':
        text = data_pac['observation_reduced'].lower()
        label_list = []
        for i, bclass in enumerate(breast_classes):
            label_list.append((bclass in text))
           
        tensor_label = torch.Tensor(label_list)
    elif label == 'density':
        density_dict = {'extremely dense':0,'fibroglandular':1,'heterogeneous':2,'fatty':3}
        tensor_label = torch.Tensor([density_dict[data_pac['density']]]).long()

    elif label == 'cancer':
        tensor_label = resolve_cancer_label(data_pac)
    else:
        raise ValueError(f"bad label type: {label}") 

    return tensor_label

def load_single_image(data_pac, img_dir, seg_dir, transformations, index,
                      anno_prepare_func,
                      load_img_func=load_mammogram_img,
                      load_segmentation_func=load_segmentation_mammogram,
                      seg_only = False):
    """
    Template
    :param data_pac:
    :param img_dir:
    :param seg_dir:
    :param transformations:
    :param index:
    :param anno_prepare_func:
    :param load_img_func:
    :param load_segmentation_func:
    :return:
    """
    # step #1: load pil images
    img_pil = load_img_func(data_pac, img_dir)

    # step #2: load classification labels
    cancer_label = resolve_cancer_label(data_pac)

    # step #3: load segmentations
    bseg_np, mseg_np = load_segmentation_func(data_pac, seg_dir)
    bseg_pil = Image.fromarray(bseg_np.astype("uint8"))
    mseg_pil = Image.fromarray(mseg_np.astype("uint8"))

    # step #4: transformation
    sample = {"img": img_pil, "bseg": bseg_pil, "mseg": mseg_pil}
    res = transformations(sample)
    img, bseg, mseg = res["img"], res["bseg"], res["mseg"]
    bseg = (bseg>0).float()
    mseg = (mseg > 0).float()

    # if only need to output segmentation map
    if seg_only:
        return img, torch.cat([bseg, mseg], dim=0)

    # step #5: segmentation to bounding boxes
    raw_annotations = collect_annotations(bseg.data.numpy()[0, :, :], mseg.data.numpy()[0, :, :])
    # transform to coco format {"bbox":[x,y,width,height]}
    annotations = []
    for i in range(len(raw_annotations)):
        orig_anno = raw_annotations[i]
        anno = {"segmentation": [],  # TODO: add segmentation later
                "area": orig_anno["Width"] * orig_anno["Height"],  # TODO: adjust area later
                "bbox": [orig_anno["X"], orig_anno["Y"], orig_anno["Width"], orig_anno["Height"]],
                "category_id": orig_anno["Class"],  # 0 benign, 1 malignant
                "image_id": index,
                "iscrowd": 0,
                "id": data_pac["descriptive_path"] + "_" + str(i)
                }
        annotations.append(anno)
    _, h, w = img.size()
    annotations = anno_prepare_func((w, h), {"image_id": index, "annotations": annotations})
    return img, cancer_label, annotations, torch.cat([bseg, mseg], dim=0), data_pac

def load_single_image_text(data_pac,img_dir,seg_dir,image_transformations,text_transformations, 
                            load_img_func, load_segmentation_func, index, 
                            is_train=True, load_seg=False, cls_classes=['cancer'],label_type = 'abnormality'):
    
    img_pil = load_img_func(data_pac, img_dir)
    
    if load_seg:
        bseg_np, mseg_np = load_segmentation_func(data_pac, seg_dir)
        bseg_pil = Image.fromarray(bseg_np.astype("uint8"))
        mseg_pil = Image.fromarray(mseg_np.astype("uint8"))

        sample = {"image": img_pil, "bseg": bseg_pil, "mseg": mseg_pil}
        results = image_transformations(sample)
    else:
        img = image_transformations(img_pil)
        results = {'image': img}

    if is_train:
        
        label = resolve_label(data_pac, label_type, cls_classes)
        breast_classes = {0:'mass',1:'calcification',2:'architectural distortion', 3:'asymmetry'}
        ab_list = [breast_classes[i]  for i,c in enumerate(label) if c == 1]
        text = ",".join(ab_list) 

        #text = text_transformations(text)
        results['text'] = label[:1]
        # label = resolve_label(data_pac,'abnormality', cls_classes)
        # results['target'] = label
        # results['raw_text'] = data_pac['observation']
    else:
        label = resolve_label(data_pac, label_type, cls_classes)
        results['target'] = label[:1]

        #results['text'] = data_pac['density']
    return results

def collect_annotations(bseg, mseg):
    if np.sum(bseg) == 0:
        banno = []
    else:
        banno = duke.DDSMAugmenter.get_connected_components(bseg, 0, "benign")

    if np.sum(mseg) == 0:
        manno = []
    else:
        manno = duke.DDSMAugmenter.get_connected_components(mseg, 1, "malignant")
    return banno + manno


class ConvertCocoPolysToMask(object):
    """
    This function is copied from https://github.com/facebookresearch/detr/blob/091a817eca74b8b97e35e4531c1c39f89fbe38eb/datasets/coco.py
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, img_size, target):
        w, h = img_size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        # NOTE: original code transforms box to [x0, y0, x1, y1]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # NOTE: I made the change to change the box to [cx, cy, width, height] format
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        #target["original_box"] = [obj["bbox"] for obj in anno]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


class ImageDataset(Dataset):
    def __init__(self, data_list, img_dir, seg_dir, imaging_modality, transformations,
                 check_positive_func=img_dl_pos_func, pos_to_neg_ratio=None, purge=True, seg_only=False, num_positives = None):
        # purge datalist:
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.check_positive_func = check_positive_func
        self.seg_only = seg_only
        self.num_positives = num_positives

        self.all_data_list = data_list.copy()
        self.purged_data_list = [x for x in data_list if len(x["lesions"]) > 0 or
                                 x["image_cancer_label_mml"] == "n/a" or
                                 (x["image_cancer_label_mml"]["malignant"] == 0 and x["image_cancer_label_mml"][
                                     "benign"] == 0)]
        if purge:
            self.switch_mode("purge")
        else:
            self.switch_mode("all")

        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transformations = transformations
        if imaging_modality == "mammo":
            self.load_img_func = load_mammogram_img
            self.load_segmentation_func = load_segmentation_mammogram
        elif imaging_modality == "us":
            self.load_img_func = load_us_img
            self.load_segmentation_func = load_segmentation_us
        self.prepare = ConvertCocoPolysToMask()

    def switch_mode(self, mode):
        # Update self.data_list
        if mode == "purge":
            self.data_list = self.purged_data_list
        elif mode == "all":
            self.data_list = self.all_data_list
        else:
            raise ValueError(f"bad mode: {mode}")
        # Update pos and neg cases
        self.positive_cases = [x for x in self.data_list if self.check_positive_func(x)]
        self.negative_cases = [x for x in self.data_list if not self.check_positive_func(x)]
        # Resample if requested
       
        if self.pos_to_neg_ratio is not None:
            self.resample()

    def resample(self, pos_to_neg_ratio=None):
        """
        Resample self.data_list to include all positive samples + some randomly sampled negative samples.
        :param pos_to_neg_ratio:
        :return:
        """
        # Determine how many negative samples do we need.
        if pos_to_neg_ratio is None:
            pos_to_neg_ratio = self.pos_to_neg_ratio
        
        if self.num_positives is not None: 
            num_pos_cases = np.minimum(self.num_positives, len(self.positive_cases))
            random_idx = np.random.permutation(range(len(self.positive_cases)))[:num_pos_cases]
            self.positive_cases = [self.positive_cases[idx] for idx in random_idx] 
 
        neg_need_num = np.minimum(int(round(len(self.positive_cases) * pos_to_neg_ratio)), len(self.negative_cases))
        random_idx = np.random.permutation(range(len(self.negative_cases)))[:neg_need_num]
        need_negative_cases = [self.negative_cases[idx] for idx in random_idx]
        self.data_list = self.positive_cases + need_negative_cases
        print(
        "After upsampling: {} positive exams {} negative exams".format(len(self.positive_cases), len(need_negative_cases)))
        # Reshuffle datalist.
        np.random.shuffle(self.data_list)

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        return load_single_image(data_pac=data_pac, img_dir=self.img_dir, seg_dir=self.seg_dir,
                                 transformations=self.transformations, index=index,
                                 anno_prepare_func=self.prepare,
                                 load_img_func=self.load_img_func,
                                 load_segmentation_func=self.load_segmentation_func,
                                 seg_only=self.seg_only)

    def __len__(self):
        return len(self.data_list)


# TODO: update me to make it work with ultrasound
class BreastDataset(Dataset):
    def __init__(self, data_list, img_dir, seg_dir, transformations):
        self.data_list = data_list
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transformations = transformations
        self.prepare = ConvertCocoPolysToMask()

    @staticmethod
    def undersample(data_pac, max_num=6):
        if data_pac["num_imgs"] <= max_num:
            return data_pac
        else:
            pri_list = []
            for ele in data_pac["datalist"]:
                view = "cc" if "cc" in ele["View"] else "mlo"
                priority = 2 + np.random.rand() if len(ele["lesions"]) > 0 else np.random.rand()
                pri_list.append((view, priority, ele))
            cc_list = sorted([x for x in pri_list if x[0] == "cc"], key=lambda x: -x[1])
            mlo_list = sorted([x for x in pri_list if x[0] == "mlo"], key=lambda x: -x[1])
            # decide how many elements to take from both view
            if len(cc_list) < len(mlo_list):
                cc_num = np.minimum(max_num // 2, len(cc_list))
                mlo_num = max_num - cc_num
            else:
                mlo_num = np.minimum(max_num // 2, len(mlo_list))
                cc_num = max_num - mlo_num
            output_dl = [x[2] for x in cc_list[:cc_num]] + [x[2] for x in mlo_list[:mlo_num]]
            assert len(output_dl) == max_num
            has_anno = False
            has_pos = False
            for ele in output_dl:
                if len(ele["lesions"]) > 0:
                    has_anno = True
                if "image_cancer_label_mml" in ele and ele["image_cancer_label_mml"] != "n/a":
                    has_pos = ele["image_cancer_label_mml"]["benign"]==1 or ele["image_cancer_label_mml"]["malignant"]==1
                elif "image_cancer_label_bounding_box" in ele:
                    has_pos = ele["image_cancer_label_bounding_box"]["benign"]==1 or ele["image_cancer_label_bounding_box"]["malignant"]==1
            return {"num_imgs": max_num, "datalist": output_dl, "has_anno": has_anno, "has_pos":has_pos}

    @staticmethod
    def group_dl_for_breast(dl):
        output = {}
        for ele in dl:
            key = (ele["StudyUID"], ele["View"][0])
            if key not in output:
                output[key] = {"num_imgs": 0, "datalist": [], "has_anno": False}
            output[key]["num_imgs"] += 1
            output[key]["datalist"].append(ele)
            if len(ele["lesions"]) > 0:
                output[key]["has_anno"] = True
        return list(output.values())

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        return [load_single_image(x, self.img_dir, self.seg_dir, self.transformations, index, self.prepare)
                for x in data_pac["datalist"]]

    def __len__(self):
        return len(self.data_list)

def check_positive_abnormality(x):
    if np.array(x['ab_label'][0]).sum() == 0:
        return False
    else:
        return True

def check_positive_random(x):
    random_n = np.random.rand()
    if random_n < 0.5:
        return False
    else:
        return True

class ImageTextDataset(Dataset):
    def __init__(self, data_list, datalist_prefix, img_dir, seg_dir, imaging_modality, image_transformations, text_transformations,
                 check_positive_func=check_positive_abnormality, pos_to_neg_ratio=None, num_positives = None, is_train=True, load_seg = False, 
                 label_type = 'abnormality'):
        self.label_type = label_type
        if self.label_type == 'abnormality':
            self.cls_classes = ['mass','calcification','architectural distortion', 'asymmetr'] 
        elif self.label_type == 'density':
            self.cls_classes = ['extremely dense','fibroglandular','heterogeneous','fatty']
        # purge datalist:
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.check_positive_func = check_positive_func
        self.num_positives = num_positives
        self.data_list = data_list.copy()
        self.datalist_prefix = datalist_prefix
        self.is_train = is_train
        self.load_seg = load_seg

        # Need to modify for new datalist
        self.positive_cases = [x for x in self.data_list if self.check_positive_func(x)]
        self.negative_cases = [x for x in self.data_list if not self.check_positive_func(x)]
        # Resample if requested
        if self.pos_to_neg_ratio is not None:
            self.resample()
            
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.image_transformations = image_transformations
        self.text_transformations = text_transformations

        if imaging_modality == "mammo":
            self.load_img_func = load_mammogram_img
            self.load_segmentation_func = load_segmentation_mammogram
        elif imaging_modality == "us":
            self.load_img_func = load_us_img
            self.load_segmentation_func = load_segmentation_us

    def resample(self, pos_to_neg_ratio=None):
        """
        Resample self.data_list to include all positive samples + some randomly sampled negative samples.
        :param pos_to_neg_ratio:
        :return:
        """
        # Determine how many negative samples do we need.
        if pos_to_neg_ratio is None:
            pos_to_neg_ratio = self.pos_to_neg_ratio
        
        if self.num_positives is not None: 
            num_pos_cases = np.minimum(self.num_positives, len(self.positive_cases))
            pos_random_idx = np.random.permutation(range(len(self.positive_cases)))[:num_pos_cases]
            need_positive_cases = [self.positive_cases[idx] for idx in pos_random_idx]
        else:
            need_positive_cases = self.positive_cases 

        neg_need_num = np.minimum(int(round(len(need_positive_cases) * pos_to_neg_ratio)), len(self.negative_cases))
        neg_random_idx = np.random.permutation(range(len(self.negative_cases)))[:neg_need_num]
        need_negative_cases = [self.negative_cases[idx] for idx in neg_random_idx]
        self.data_list = need_positive_cases + need_negative_cases
        print('pos',pos_random_idx[:5])
        print('neg',neg_random_idx[:5])
        print("After upsampling: {} positive exams {} negative exams".format(len(need_positive_cases), len(need_negative_cases)))
        # Reshuffle datalist.
        np.random.shuffle(self.data_list)

   
    def __getitem__(self, index):
        meta_data_pac = self.data_list[index]
        with open(os.path.join(self.datalist_prefix, meta_data_pac['pkl']), "rb") as f:
            data = pickle.load(f)
        data_pac = data[meta_data_pac['idx']]
        
        return load_single_image_text(data_pac=data_pac, img_dir=self.img_dir, seg_dir = self.seg_dir,
                                 image_transformations=self.image_transformations,
                                 text_transformations=self.text_transformations, 
                                 load_img_func=self.load_img_func,
                                 load_segmentation_func=self.load_segmentation_func, 
                                 index=index,
                                 is_train=self.is_train,
                                 load_seg=self.load_seg,
                                 cls_classes=self.cls_classes,
                                 label_type=self.label_type)

    def __len__(self):
        return len(self.data_list)
    

class UpsampleLoader:
    """
    A wrapper of dataset and dataloader which resamples data list for every epoch
    """
    def __init__(self, dataset, collate_fn, num_workers,  batch_size=None, sampler=None, shuffle=False, 
                pin_memory=False,  persistent_workers=None, worker_init_fn=None,
                max_numel_per_batch=None, numel_col=None, upsample_shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.max_numel_per_batch = max_numel_per_batch
        self.numel_col = numel_col
        self.sampler = sampler
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.worker_init_fn = worker_init_fn
        self.upsample_shuffle = upsample_shuffle
        
        self.resample()

    def resample(self):
        self.dataset.resample()
        print('resample in dataloader')
        # create new data loader
        # TODO: check if this still works
        if self.max_numel_per_batch is not None:
            # create a batch sampler using the current data list
            assert self.numel_col is not None
            sampler = MaxImageNumberSampler(self.upsample_data_list, max_numel_per_batch=self.max_numel_per_batch, random=True,
                                                thresholds=range(2, self.max_numel_per_batch + 1), numel_col=self.numel_col)
            # create a data loader
            self.data_loader = DataLoader(self.dataset, collate_fn=self.collate_fn, num_workers=self.num_workers,
                                           pin_memory=True, batch_sampler=sampler)
        else:
            self.data_loader = DataLoader(self.dataset, 
                                        collate_fn=self.collate_fn, 
                                        batch_size=self.batch_size,
                                        sampler=self.sampler,
                                        shuffle=self.shuffle, 
                                        num_workers=self.num_workers, 
                                        pin_memory=self.pin_memory, 
                                        persistent_workers=self.persistent_workers,
                                        worker_init_fn=self.worker_init_fn)


    def __iter__(self):
        for batch in self.data_loader:
            yield batch
        # reshuffle and re-sample the negative cases every epoch
        if self.upsample_shuffle:
            self.resample()
            print("shuffle")

    def __len__(self):
        return len(self.data_loader)


class BucketQueue:
    """
    Object that queues each exam according to number of images per exam
    """
    def __init__(self, data_list, max_numel_per_batch, thresholds=None, numel_col="num_imgs"):
        self.data_list = data_list
        self.max_numel_per_batch = max_numel_per_batch
        self.numel_col = numel_col
        numel = np.array([x.get(numel_col) for x in data_list])
        indices = np.arange(len(data_list))
        # create thresholds
        if thresholds is None:
            self.bucket_thresholds = [np.percentile(numel, 30), np.percentile(numel, 60),
                                      np.percentile(numel, 75), np.percentile(numel, 90),
                                      np.max(numel)]
        else:
            self.bucket_thresholds = thresholds
        # create list of queues
        self.idx_queue_list = [[] for _ in range(len(self.bucket_thresholds))]
        for i in range(len(numel)):
            for j in range(len(self.bucket_thresholds)):
                if numel[i] <= self.bucket_thresholds[j]:
                    self.idx_queue_list[j].append((indices[i], numel[i]))
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

    def sample_a_batch(self, random=False):
        """
        Method that samples a minibatch from the queue list
        :param random:
        :return:
        """
        output = []
        current_limit = self.max_numel_per_batch
        need_sample = True
        while need_sample:
            # select the bucket with largest number of elements
            largest_threshold_idx = bisect_left(self.bucket_thresholds, current_limit)
            selected_queue = self.idx_queue_list[largest_threshold_idx - 1]
            # select an element out of the bucket
            if random:
                bucket_idx = np.random.randint(low=0, high=len(selected_queue))
                current_data_idx, numel_added = selected_queue[bucket_idx]
                del selected_queue[bucket_idx]
            else:
                current_data_idx, numel_added = selected_queue.pop()
            # update status
            output.append(current_data_idx)
            current_limit -= numel_added
            self.update()
            # check if we can take more images
            if self.deplete or current_limit <= self.bucket_thresholds[0]:
                need_sample = False
        return output

    def give_all_batches(self, random=False):
        """
        Method that creates all minibatch index
        :param random:
        :return:
        """
        all_batches = []
        while not self.deplete:
            all_batches.append(self.sample_a_batch(random))
        return all_batches


class MaxImageNumberSampler(Sampler):
    """
    Object that creates minibatches which has strictly less number of images than the input
    """
    def __init__(self, data_list, max_numel_per_batch=6, random=False, thresholds=None, numel_col="num_imgs"):
        super(MaxImageNumberSampler).__init__()
        self.random = random
        self.thresholds = thresholds
        self.numel_col = numel_col
        self.data_list = data_list
        self.max_numel_per_batch = max_numel_per_batch
        self.recompute_batches(data_list, max_numel_per_batch, random)

    def recompute_batches(self, data_list, max_numel_per_batch, random):
        # create a queuelist object
        self.bucket_queue = BucketQueue(data_list, max_numel_per_batch,
                                        thresholds=self.thresholds, numel_col=self.numel_col)
        # calculate all batch index
        self.all_batches = self.bucket_queue.give_all_batches(random)

    def __iter__(self):
        if self.random:
            self.recompute_batches(self.data_list, self.max_numel_per_batch, True)
            np.random.shuffle(self.all_batches)
        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)
