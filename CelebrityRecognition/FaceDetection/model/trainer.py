from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from FaceDetection.utils.roi_tools import filter_roi, pos_hard_roi_selection
from torch import nn
import torch as t
from FaceDetection.utils.array_util import totensor, tonumpy, scalar
from FaceDetection.utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super().__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.optimizer = self.faster_rcnn.get_optimizer()

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, images, bboxes, labels, scale):
        _, _, img_height, img_width = images.shape
        img_size = (img_height, img_width)

        features = self.faster_rcnn.extractor(images)

        rpn_locs, rpn_scores, rois, _, anchors = self.faster_rcnn.rpn(features, img_size, scale)
        bbox = tonumpy(bboxes[0])
        label = tonumpy(labels[0])
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        sample_rois, gt_roi_locs, gt_roi_labels = pos_hard_roi_selection(roi, bbox, label)
        sample_roi_indices = t.zeros(len(sample_rois))
        roi_cls_locs, roi_scores = self.faster_rcnn.head(features, sample_rois, sample_roi_indices)

        gt_rpn_locs, gt_rpn_labels = filter_roi(bbox, anchors, img_size)
        gt_rpn_labels = totensor(gt_rpn_labels).long()
        gt_rpn_locs = totensor(gt_rpn_locs)
        rpn_loc_loss = loc_loss(rpn_loc, gt_rpn_locs, gt_rpn_labels.data, self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_labels.cuda(), ignore_index=-1)
        valid_rpn_labels = gt_rpn_labels[gt_rpn_labels > -1]
        valid_rpn_scores = tonumpy(rpn_score)[tonumpy(gt_rpn_labels) > -1]
        self.rpn_cm.add(totensor(valid_rpn_scores, False), valid_rpn_labels.data.long())

        num_samples = roi_cls_locs.shape[0]
        roi_cls_locs = roi_cls_locs.view(num_samples, -1, 4)
        roi_locs = roi_cls_locs[t.arange(0, num_samples).long().cuda(), totensor(gt_roi_labels).long()]
        gt_roi_labels = totensor(gt_roi_labels).long()
        gt_roi_locs = totensor(gt_roi_locs)

        roi_loc_loss = loc_loss(roi_locs.contiguous(), gt_roi_locs, gt_roi_labels.data, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_scores, gt_roi_labels.cuda())

        self.roi_cm.add(totensor(roi_scores, False), gt_roi_labels.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses.append(sum(losses))

        return LossTuple(*losses)

    def train_step(self, images, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(images, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self):
        save_dict = {
            'model': self.faster_rcnn.state_dict(),
            'config': opt.state_dict(),
        }

        timestr = time.strftime('%m%d%H%M')
        save_path = f'./FaceDetection/checkpoints/fasterrcnn_pretrained-{timestr}.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_dict = {k: scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_dict[key])

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def l1_loss(prediction, target, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (prediction - target)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    loss = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return loss.sum()


def loc_loss(pred_locs, gt_locs, gt_labels, sigma):
    in_weight = t.zeros(gt_locs.shape).cuda()
    in_weight[(gt_labels > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = l1_loss(pred_locs, gt_locs, in_weight.detach(), sigma)
    loc_loss /= ((gt_labels >= 0).sum().float())
    return loc_loss
