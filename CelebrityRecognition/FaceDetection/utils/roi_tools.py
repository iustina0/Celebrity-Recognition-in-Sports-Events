import numpy as np
from FaceDetection.utils.bbox_util import bbox2loc, bbox_iou


def pos_hard_roi_selection(roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
    n_sample = 128
    pos_ratio = 0.25
    pos_iou_thresh = 0.7
    neg_iou_high = 0.3
    neg_iou_low = 0

    roi = np.concatenate((roi, bbox), axis=0)

    iou = bbox_iou(roi, bbox)
    gt_assignment = iou.argmax(axis=1)
    max_iou = iou.max(axis=1)
    gt_roi_label = label[gt_assignment] + 1

    pos_index_1 = np.where(max_iou >= pos_iou_thresh)[0]

    gt_max_iou = iou.max(axis=0)
    pos_index_2 = np.where(iou == gt_max_iou)[0]

    pos_index = np.unique(np.concatenate((pos_index_1, pos_index_2)))
    n_pos_rois = int(min(np.round(n_sample * pos_ratio), pos_index.size))

    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=n_pos_rois, replace=False)

    neg_index = np.where((max_iou < neg_iou_high) & (max_iou >= neg_iou_low))[0]
    n_neg_rois = n_sample - n_pos_rois
    n_neg_rois = int(min(n_neg_rois, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=n_neg_rois, replace=False)

    indexs = np.append(pos_index, neg_index)
    gt_roi_label = gt_roi_label[indexs]
    gt_roi_label[n_pos_rois:] = 0 
    sample_roi = roi[indexs]

    gt_roi_offset = bbox2loc(sample_roi, bbox[gt_assignment[indexs]])
    gt_roi_offset = ((gt_roi_offset - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

    return sample_roi, gt_roi_offset, gt_roi_label



def filter_roi(bbox, anchor, img_size):
    img_H, img_W = img_size

    n_anchor = len(anchor)
    inside_index = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= img_H) &
        (anchor[:, 3] <= img_W)
    )[0]
    anchor = anchor[inside_index]
    argmax_ious, label = _create_label(inside_index, anchor, bbox)

    loc = bbox2loc(anchor, bbox[argmax_ious])

    label = map_to_original(label, n_anchor, inside_index, fill=-1)
    loc = map_to_original(loc, n_anchor, inside_index, fill=0)

    return loc, label

def _create_label(valid_idx, anchors, gt_boxes):
    samples_per_image = 256
    positive_iou_threshold = 0.7
    negative_iou_threshold = 0.3
    positive_fraction = 0.5
    
    labels = np.full((len(valid_idx),), -1, dtype=np.int32)

    ious, max_ious, best_gt_idx = calc_ious(anchors, gt_boxes, valid_idx)

    labels[max_ious < negative_iou_threshold] = 0

    labels[best_gt_idx] = 1

    labels[max_ious >= positive_iou_threshold] = 1

    num_positive = int(positive_fraction * samples_per_image)
    positive_indices = np.where(labels == 1)[0]
    if len(positive_indices) > num_positive:
        disable_positive = np.random.choice(
            positive_indices, size=(len(positive_indices) - num_positive), replace=False)
        labels[disable_positive] = -1


    num_negative = samples_per_image - np.sum(labels == 1)
    negative_indices = np.where(labels == 0)[0]
    if len(negative_indices) > num_negative:
        disable_negative = np.random.choice(
            negative_indices, size=(len(negative_indices) - num_negative), replace=False)
        labels[disable_negative] = -1

    return ious, labels

def calc_ious(anchor, bbox, inside_index):
    ious = bbox_iou(anchor, bbox)
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(inside_index)), argmax_ious]
    gt_argmax_ious = ious.argmax(axis=0)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]

    return argmax_ious, max_ious, gt_argmax_ious


def map_to_original(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

