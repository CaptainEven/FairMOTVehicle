from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    """
    NCHW
    do max pooling operation
    """
    # print("heat.shape: ", heat.shape)  # default: torch.Size([1, 1, 152, 272])

    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    # print("hmax.shape: ", hmax.shape)  # default: torch.Size([1, 1, 152, 272])

    keep = (hmax == heat).float()  # 将boolean类型的Tensor转换成Float类型的Tensor
    # print("keep.shape: ", keep.shape, "keep:\n", keep)
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    """
    scores=heatmap by default
    """
    batch, cat, height, width = scores.size()
    # print("b_s, channels, h, w: ", batch, cat, height, width)

    # 2d feature map -> 1d feature map
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # print("topk_scores.shape: ", topk_scores.shape, ", topk_inds.shape: ", topk_inds.shape)  # 1×1×128

    topk_inds = topk_inds % (height * width)  # 这一步貌似没必要...
    # print("topk_inds.shape: ", topk_inds.shape)  # 1×1×128

    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # print("topk_ys.shape: ", topk_ys.shape, ", topk_xs.shape: ", topk_xs.shape)

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # print("topk_score.shape: ", topk_score.shape, ", topk_ind.shape: ", topk_ind.shape)  # 1×128
    # print("after view, topk_ind.shape:", topk_inds.view(batch, -1, 1).shape)

    topk_clses = (topk_ind / K).int()
    # print("topk_clses.shape", topk_clses.shape)  # 1×128

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)  # 1×128×1 -> 1×128?
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat,
               wh,
               reg=None,
               cat_spec_wh=False,
               K=100):
    """
    多目标检测结果解析
    """
    batch, cat, height, width = heat.size()  # N×C×H×W

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  # 默认应用3×3max pooling操作, 检测目标数变为feature map的1/9

    scores, inds, clses, ys, xs = _topk(scores=heat, K=K)

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()  # 目标类别
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,   # left
                        ys - wh[..., 1:2] / 2,   # top
                        xs + wh[..., 0:1] / 2,   # right
                        ys + wh[..., 1:2] / 2],  # down
                       dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
