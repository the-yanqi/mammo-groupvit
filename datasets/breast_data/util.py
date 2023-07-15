def get_connected_components(seg, label_key):
    """
    return x1, y1, x2, y2 of connected components
    if width and height > 3
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(seg[:, :, 0])
    lesions = []
    accepted_masks = []
    for k, v in mask_pixels_dict.items():
        current_lesion = mask == k

        y_edge_top, y_edge_bottom = get_edge_values(current_lesion, "y")
        x_edge_left, x_edge_right = get_edge_values(current_lesion, "x")

        width = x_edge_right - x_edge_left
        height = y_edge_bottom - y_edge_top

        if (width > 3) and (height > 3):
            this_lesion_info = {
                'X': x_edge_left,
                'Y': y_edge_top,
                'Width': width,
                'Height': height,
                'combined_label': label_key,
            }
            lesions.append(this_lesion_info)
            accepted_masks.append(np.where(mask == k, 1, 0))

    return lesions, accepted_masks


def extract_lesions_from_seg(self, label_key, seg):
    lesions, accepted_masks = self.get_connected_components(seg, label_key)
    return lesions, accepted_masks


def extract_annot_from_seg(self, seg_dict):
    lesions = []
    seg_dict_new = {}
    for k in seg_dict.keys():
        extracted_lesions, accepted_masks = self.extract_lesions_from_seg(k, seg_dict[k])
        lesions.extend(extracted_lesions)
        if len(accepted_masks) > 0:
            seg_dict_new[k] = np.stack(accepted_masks)
            seg_dict_new[k] = np.transpose(seg_dict_new[k], (1, 2, 0))  # channel dim at the end
    return get_annotations_from_lesions_V2(lesions), seg_dict_new