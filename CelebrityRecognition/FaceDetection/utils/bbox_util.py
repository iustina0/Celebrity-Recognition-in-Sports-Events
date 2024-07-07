import numpy as np

def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_center_y = src_bbox[:, 0] + 0.5 * src_height
    src_center_x = src_bbox[:, 1] + 0.5 * src_width

    delta_y, delta_x, delta_h, delta_w = loc[:, 0::4], loc[:, 1::4], loc[:, 2::4], loc[:, 3::4]

    pred_center_y = delta_y * src_height[:, np.newaxis] + src_center_y[:, np.newaxis]
    pred_center_x = delta_x * src_width[:, np.newaxis] + src_center_x[:, np.newaxis]
    pred_height = np.exp(delta_h) * src_height[:, np.newaxis]
    pred_width = np.exp(delta_w) * src_width[:, np.newaxis]

    pred_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    pred_bbox[:, 0::4] = pred_center_y - 0.5 * pred_height
    pred_bbox[:, 1::4] = pred_center_x - 0.5 * pred_width
    pred_bbox[:, 2::4] = pred_center_y + 0.5 * pred_height
    pred_bbox[:, 3::4] = pred_center_x + 0.5 * pred_width

    return pred_bbox

def bbox2loc(src_bbox, dst_bbox):
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_center_y = src_bbox[:, 0] + 0.5 * src_height
    src_center_x = src_bbox[:, 1] + 0.5 * src_width

    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_center_y = dst_bbox[:, 0] + 0.5 * dst_height
    dst_center_x = dst_bbox[:, 1] + 0.5 * dst_width

    eps = np.finfo(src_height.dtype).eps
    src_height = np.maximum(src_height, eps)
    src_width = np.maximum(src_width, eps)

    delta_y = (dst_center_y - src_center_y) / src_height
    delta_x = (dst_center_x - src_center_x) / src_width
    delta_h = np.log(dst_height / src_height)
    delta_w = np.log(dst_width / src_width)

    return np.vstack((delta_y, delta_x, delta_h, delta_w)).transpose()

def bbox_iou(box_a, box_b):
    tl = np.maximum(box_a[:, None, :2], box_b[:, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[:, 2:])

    intersection = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(box_a[:, 2:] - box_a[:, :2], axis=1)
    area_b = np.prod(box_b[:, 2:] - box_b[:, :2], axis=1)
    iou = intersection / (area_a[:, None] + area_b - intersection)
    return iou

def generate_anchor_base(ref_size=16, ratios=None, anchor_scales=None):
    if ratios is None:
        ratios = [0.5, 0.66, 0.75, 1, 1.33, 1.5, 2]
    if anchor_scales is None:
        anchor_scales = [2, 4, 8, 16, 32]

    py = ref_size * 0.5
    px = ref_size * 0.5

    num_anchors = len(ratios) * len(anchor_scales)
    anchor_base = np.zeros((num_anchors, 4), dtype=np.float32)
    
    for i, ratio in enumerate(ratios):
        for j, scale in enumerate(anchor_scales):
            h = ref_size * scale * np.sqrt(ratio)
            w = ref_size * scale * np.sqrt(1. / ratio)

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h * 0.5
            anchor_base[index, 1] = px - w * 0.5
            anchor_base[index, 2] = py + h * 0.5
            anchor_base[index, 3] = px + w * 0.5

    return anchor_base
