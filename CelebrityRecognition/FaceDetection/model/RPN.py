import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from FaceDetection.utils.bbox_util import generate_anchor_base, loc2bbox

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=None, anchor_scales=None, feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        if anchor_scales is None:
            anchor_scales = [2, 4, 8, 16, 32]
        if ratios is None:
            ratios = [0.5, 0.66, 0.75, 1, 1.33, 1.5, 2]

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride

        num_anchors = self.anchor_base.shape[0]
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv.weight.data.normal_(std=0.01)
        self.conv.bias.data.zero_()

        self.score = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1, stride=1, padding=0)
        self.score.weight.data.normal_(std=0.01)
        self.score.bias.data.zero_()

        self.loc = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0)
        self.loc.weight.data.normal_(std=0.01)
        self.loc.bias.data.zero_()

    def forward(self, features, img_size, scale=1.):
        batch_size, _, height, width = features.shape
        anchors = create_shifted_anchors(
            np.array(self.anchor_base),
            self.feat_stride, height, width
        )

        anchors[:, [0, 2]] = np.clip(anchors[:, [0, 2]], 0, img_size[0] - 1)
        anchors[:, [1, 3]] = np.clip(anchors[:, [1, 3]], 0, img_size[1] - 1)

        num_anchors = anchors.shape[0] // (height * width)
        conv_features = F.relu(self.conv(features))

        pred_locs = self.loc(conv_features)
        pred_locs = pred_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred_scores = self.score(conv_features)
        pred_scores = pred_scores.permute(0, 2, 3, 1).contiguous()
        softmax_scores = F.softmax(pred_scores.view(batch_size, height, width, num_anchors, 2), dim=4)
        foreground_scores = softmax_scores[:, :, :, :, 1].contiguous().view(batch_size, -1)
        pred_scores = pred_scores.view(batch_size, -1, 2)

        rois = []
        roi_indices = []
        for i in range(batch_size):
            single_rois = proposal_layer(
                pred_locs[i].cpu().data.numpy(),
                foreground_scores[i].cpu().data.numpy(),
                anchors, img_size,
                scale=scale
            )
            batch_index = i * np.ones((len(single_rois),), dtype=np.int32)
            rois.append(single_rois)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return pred_locs, pred_scores, rois, roi_indices, anchors


def create_shifted_anchors(base_anchors, stride, height, width):
    shift_y = np.arange(0, height * stride, stride)
    shift_x = np.arange(0, width * stride, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.stack((shift_y.ravel(), shift_x.ravel(),
                       shift_y.ravel(), shift_x.ravel()), axis=1)

    num_base_anchors = base_anchors.shape[0]
    num_shifts = shifts.shape[0]
    anchors = base_anchors.reshape((1, num_base_anchors, 4)) + shifts.reshape((1, num_shifts, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((num_shifts * num_base_anchors, 4)).astype(np.float32)
    return anchors


def proposal_layer(loc, score, anchor, img_size, scale=1.):
    nms_thresh = 0.7
    n_test_pre_nms = 6000
    n_test_post_nms = 300
    min_size = 16
    pre_nms_limit = n_test_pre_nms
    post_nms_limit = n_test_post_nms

    proposals = loc2bbox(anchor, loc)

    proposals[:, [0, 2]] = np.clip(proposals[:, [0, 2]], 0, img_size[0])
    proposals[:, [1, 3]] = np.clip(proposals[:, [1, 3]], 0, img_size[1])

    min_size = min_size * scale
    heights = proposals[:, 2] - proposals[:, 0]
    widths = proposals[:, 3] - proposals[:, 1]
    valid_indices = np.where((heights >= min_size) & (widths >= min_size))[0]
    proposals = proposals[valid_indices, :]
    score = score[valid_indices]

    order = score.ravel().argsort()[::-1]
    if pre_nms_limit > 0:
        order = order[:pre_nms_limit]
    proposals = proposals[order, :]
    score = score[order]

    keep_indices = nms( torch.from_numpy(proposals).cuda(), torch.from_numpy(score).cuda(), nms_thresh)

    if post_nms_limit > 0:
        keep_indices = keep_indices[:post_nms_limit]

    proposals = proposals[keep_indices.cpu().numpy()]
    return proposals
