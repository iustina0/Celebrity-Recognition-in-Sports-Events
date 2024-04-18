import numpy as np
from torch.nn import functional as F
from torch import nn
from utils.bbox_util import generate_anchor_base
from utils.creators import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".


    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of int): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        proposal_creator_params (dict): Key valued parameters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    see also::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(self, in_channels=512, mid_channels=512, ratios=None,
                 anchor_scales=None, feat_stride=16, proposal_creator_params=None):

        super(RegionProposalNetwork, self).__init__()
        if proposal_creator_params is None:
            proposal_creator_params = dict()
        if anchor_scales is None:
            anchor_scales = [2, 4, 8, 16, 32]
        if ratios is None:
            ratios = [0.5, 0.66, 0.75, 1, 1.33, 1.5, 2]

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        len_anchor = self.anchor_base.shape[0]
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.conv.weight.data.normal_(std=0.01)
        self.conv.bias.data.zero_()

        self.score = nn.Conv2d(mid_channels, len_anchor * 2, 1, 1, 0)
        self.score.weight.data.normal_(std=0.01)
        self.score.bias.data.zero_()

        self.loc = nn.Conv2d(mid_channels, len_anchor * 4, 1, 1, 0)
        self.loc.weight.data.normal_(std=0.01)
        self.loc.bias.data.zero_()

    def forward(self, features, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            features (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = features.shape
        anchor = shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        anchor[:, [0, 2]] = np.clip(anchor[:, [0, 2]], 0, img_size[0] - 1)
        anchor[:, [1, 3]] = np.clip(anchor[:, [1, 3]], 0, img_size[1] - 1)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv(features))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def shifted_anchor(anchor_base, feat_stride, height, width):
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    a = anchor_base.shape[0]
    k = shift.shape[0]
    anchor = anchor_base.reshape((1, a, 4)) + shift.reshape((1, k, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((k * a, 4)).astype(np.float32)
    return anchor
