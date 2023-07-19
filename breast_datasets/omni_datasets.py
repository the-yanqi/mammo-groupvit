"""
File that contains code that is relevant to the Omni-supervised dataset class.
"""
import numpy as np
import torch, functools
from datasets.breast_data import datasets

class ReversibleWeakAugmentation:
    """
    An augmentation that is reversible: A = reverse_aug(aug(A))
    """
    def __init__(self, generator=None, p_hflip=0.5):
        """
        :param generator: numpy random state object
        :param p_hflip: probability for horizontal flip
        """
        if generator is None:
            self.generator = np.random.default_rng(seed=666)
        else:
            self.generator = generator
        self.has_hflip = False
        self.p_hflip = p_hflip

    def aug(self, img_tensor):
        self.has_hflip = False
        if self.generator.random() > self.p_hflip:
            self.has_hflip = True
            # Assumption: W is the last dimension of img_tensor
            img_tensor = torch.flip(img_tensor, dims=[-1])
        return img_tensor

    def reverse_aug(self, img_tensor):
        if self.has_hflip:
            img_tensor = torch.flip(img_tensor, dims=[-1])
        return img_tensor


def bbox_list_to_segmap(h, w, bbox_list):
    """
    Create a segmentation map by setting all pixels included in the bbox_list to 1.
    :param h: height of image
    :param w: width of image
    :param bbox_list: [bbox1, bbox2, ...], bbox in DETR format: (cx, cy, dx, dy), all four cord needs be < 1.
    :return: a torch Tensor of size (h,w)
    """
    seg_map = torch.zeros((h, w))
    for bbox in bbox_list:
        bbox = bbox.data.numpy()
        cx, cy, dx, dy = bbox
        x0 = int(round((cx - dx / 2) * w))
        x1 = int(round((cx + dx / 2) * w))
        y0 = int(round((cy - dy / 2) * h))
        y1 = int(round((cy + dy / 2) * h))
        seg_map[y0:y1, x0:x1] = 1
    return seg_map


def filter_pseudo_bbox_label(inf_res, benign_threshold, malignant_threshold, topk_box):
    """
    Filter the inference output of DETR to only contain bboxes that qualify:
    - at most topk_box per class
    - has cls_score > the class-sensitive threshold
    :param inf_res: output of DETR, must contains "pred_logits" and "pred_boxes".
    :param benign_threshold:
    :param malignant_threshold:
    :param topk_box:
    :return:
    """
    # fetch bbox and cls prob
    # Assumption: batch size = 1
    bbox_prob = torch.exp(inf_res["pred_logits"])[0, :, :]
    bbox = inf_res["pred_boxes"][0, :, :]
    # select top k box for both classes
    # Assumption: there is only one image in the batch
    top_val_ben, top_idx_ben = bbox_prob[:, 0].topk(topk_box)
    top_val_mal, top_idx_mal = bbox_prob[:, 1].topk(topk_box)
    benign_pseudo_bbox = bbox[top_idx_ben[torch.exp(top_val_ben) > benign_threshold], :]
    malignant_pseudo_bbox = bbox[top_idx_mal[torch.exp(top_val_mal) > malignant_threshold], :]
    return benign_pseudo_bbox, malignant_pseudo_bbox

def load_pseudo_segmentation_map(data_pac,
                                 img_dir,
                                 no_aug_transformation,
                                 load_img_func,
                                 weak_augmentor,
                                 teacher_network,
                                 device,
                                 benign_threshold=0.1,
                                 malignant_threshold=0.1,
                                 topk_box=2):
    """
    Top-level function for using teacher network to produce pseudo label.
    :param data_pac:
    :param img_dir:
    :param no_aug_transformation: transformation w/o any augmentation.
    :param load_img_func:
    :param weak_augmentor:
    :param teacher_network:
    :param benign_threshold:
    :param malignant_threshold:
    :param topk_box:
    :return:
    """
    # step 1: load pil images
    img_pil = load_img_func(data_pac, img_dir)
    # step 2: augmentation-free transformation
    # TODO: this is ugly
    transformed_img_dict = no_aug_transformation({"img": img_pil, "mseg": img_pil, "bseg": img_pil})
    img_tensor = transformed_img_dict["img"]
    _, h, w = img_tensor.size()
    # step 3: weak augmentation: random flip
    aug_img_tensor = weak_augmentor.aug(img_tensor).unsqueeze(0)
    # step 4: inference
    with torch.no_grad():
        aug_img_tensor = aug_img_tensor.to(device)
        inf_res = teacher_network(aug_img_tensor, None)
        inf_res["pred_logits"] = inf_res["pred_logits"].data.cpu()
        inf_res["pred_boxes"] = inf_res["pred_boxes"].data.cpu()
    # step 5: determine pseudo bbox
    benign_pseudo_bbox, malignant_pseudo_bbox = filter_pseudo_bbox_label(inf_res,
                                                                         benign_threshold,
                                                                         malignant_threshold,
                                                                         topk_box)
    # step 6: bbox to seg
    benign_seg_mask = bbox_list_to_segmap(h, w, benign_pseudo_bbox)
    malignant_seg_mask = bbox_list_to_segmap(h, w, malignant_pseudo_bbox)
    # step 7: revert the augmentation if augmentation is applied
    benign_seg_mask_reverse = weak_augmentor.reverse_aug(benign_seg_mask)
    malignant_seg_mask_reverse = weak_augmentor.reverse_aug(malignant_seg_mask)
    return benign_seg_mask_reverse.data.cpu().numpy(), malignant_seg_mask_reverse.data.cpu().numpy()


def determine_need_pseudo_label(data_pac):
    """
    Determines if we use the pseudo label or the gt label for each class.
    Returns: {"benign": T/F, "malignant":T/F}, T=need pseudo label, F=use gt.
    """
    need_infer_flag = {"benign": False, "malignant": False}
    if data_pac["image_cancer_label_mml"] != "n/a":
        all_lesion_class = [] if len(data_pac["lesions"]) == 0 else  [x["Class"] for x in data_pac["lesions"]]
        # use pseudo label if class label is positive but no annotation is associated with this class.
        if data_pac["image_cancer_label_mml"]["malignant"] == 1 and "malignant" not in all_lesion_class:
            need_infer_flag["malignant"] = True
        if data_pac["image_cancer_label_mml"]["benign"] == 1 and "benign" not in all_lesion_class:
            need_infer_flag["benign"] = True
    return need_infer_flag


def load_seg_func_both(data_pac,
                       seg_dir,
                       load_segmentation_func_clean,
                       load_segmentation_func_pseudo,
                       need_pseudo_label_flag):
    """
    Wrapper function that uses ground-truth label or pseudo label depending on need_pseudo_label_flag.
    :param data_pac:
    :param seg_dir:
    :param load_segmentation_func_clean:
    :param load_segmentation_func_pseudo:
    :param need_pseudo_label_flag:
    :return:
    """
    bseg_np_clean, mseg_np_clean = load_segmentation_func_clean(data_pac, seg_dir)
    bseg_np_pseudo_label, mseg_np_pseudo_label = load_segmentation_func_pseudo(data_pac)
    bseg_np = bseg_np_pseudo_label if need_pseudo_label_flag["benign"] else bseg_np_clean
    mseg_np = mseg_np_pseudo_label if need_pseudo_label_flag["malignant"] else mseg_np_clean
    return bseg_np, mseg_np

class PseudoLabelDataset(datasets.ImageDataset):
    """
    Class that selectively impute missing segmentation using pseudo labels generated by a teacher network.
    """
    def __init__(self,
                 data_list,
                 img_dir,
                 seg_dir,
                 imaging_modality,
                 strong_aug_transformation,
                 no_aug_transformation,
                 teacher_network=None,
                 check_positive_func=datasets.img_dl_pos_func,
                 pos_to_neg_ratio=None,
                 purge=False,
                 benign_threshold=0.1,
                 malignant_threshold=0.1,
                 topk_box=2
                 ):
        """
        :param data_list:
        :param img_dir:
        :param seg_dir:
        :param imaging_modality:
        :param strong_aug_transformation: augmentation that will be applied to all images during training.
        :param no_aug_transformation: transformation w/o augmentation for teacher network.
        :param teacher_network: currently only supports DETR.
        :param check_positive_func:
        :param pos_to_neg_ratio:
        :param purge: set default to false because this class is created to address positive imgs w/o annotation.
        :param benign_threshold:
        :param malignant_threshold:
        :param topk_box:
        """
        super().__init__(data_list=data_list,
                         img_dir=img_dir,
                         seg_dir=seg_dir,
                         imaging_modality=imaging_modality,
                         transformations=strong_aug_transformation,
                         check_positive_func=check_positive_func,
                         pos_to_neg_ratio=pos_to_neg_ratio,
                         purge=purge)
        self.no_aug_transformation = no_aug_transformation
        self.weak_augmentor = ReversibleWeakAugmentation()
        self.benign_threshold = benign_threshold
        self.malignant_threshold = malignant_threshold
        self.topk_box = topk_box
        self._teacher_network = teacher_network
        self._device = torch.device("cpu")
        if self._teacher_network is not None:
            self._teacher_network.eval()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def teacher_network(self):
        return self._teacher_network

    @teacher_network.setter
    def teacher_network(self, network):
        self._teacher_network = network
        self._teacher_network.eval()

    def update_img_seg_anno_with_pseudo_labels(self, img_tensor, annotation, segmentation, meta_data):
        """
        Method that update the image , annotation, and segmentation by selectively impute using pseudo labels.
        :param img_tensor: N,C,H,W tensor
        :param annotation: [{annotation}]
        :param segmentation: N,class,H,W tensor
        :param meta_data:
        :return:
        """
        for i in range(len(meta_data)):
            data_pac = meta_data[i]
            need_pseudo_label_flag = determine_need_pseudo_label(data_pac)
            if need_pseudo_label_flag["benign"] or need_pseudo_label_flag["malignant"]:
                new_img, _, new_annotations, new_seg, _ = self.load_sample_pseudo_label(data_pac)
                img_tensor[i,:,:,:] = new_img
                annotation[i] = new_annotations
                segmentation[i, :, : ,:] = new_seg

    def load_sample_pseudo_label(self, data_pac):
        """
        Method that loads an instance using pseudo labels.
        1. first check if malignant and benign class needs pseudo labels.
        2. compute pseudo labels for both benign and malignant classes.
        3. impute using pseudo labels.
        :param data_pac:
        :return:
        """
        # make sure teacher network is ready
        assert not self._teacher_network is None
        # decide the segmentation of this sample needs to be replaced with pseudo label
        need_pseudo_label_flag = determine_need_pseudo_label(data_pac)
        # partial function for loading pseudo label.
        load_pseudo_seg_partial = functools.partial(load_pseudo_segmentation_map,
                                                    img_dir=self.img_dir,
                                                    no_aug_transformation=self.no_aug_transformation,
                                                    load_img_func=self.load_img_func,
                                                    weak_augmentor=self.weak_augmentor,
                                                    teacher_network=self._teacher_network,
                                                    benign_threshold=self.benign_threshold,
                                                    malignant_threshold=self.malignant_threshold,
                                                    topk_box=self.topk_box,
                                                    device=self._device)
        # partial function for loading both gt and pseudo label.
        load_seg_func_partial = functools.partial(load_seg_func_both,
                                                  load_segmentation_func_clean=self.load_segmentation_func,
                                                  load_segmentation_func_pseudo=load_pseudo_seg_partial,
                                                  need_pseudo_label_flag=need_pseudo_label_flag)
        # replace the load_segmentation function with load_seg_func_partial.
        return datasets.load_single_image(data_pac=data_pac,
                                          img_dir=self.img_dir,
                                          seg_dir=self.seg_dir,
                                          transformations=self.transformations,
                                          index=0, # not sure if we can hard-code 0
                                          anno_prepare_func=self.prepare,
                                          load_img_func=self.load_img_func,
                                          load_segmentation_func=load_seg_func_partial)
