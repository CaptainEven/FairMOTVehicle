from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.decode import mot_decode
from lib.models.losses import FocalLoss
from lib.models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.utils.post_process import ctdet_post_process

from .base_trainer import BaseTrainer


# 损失函数的定义
class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None  # L1 loss or smooth l1 loss
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg  # box size loss

        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        # 唯一包含可学习参数的层: 用于Re-ID的全连接层
        self.classifier = nn.Linear(self.emb_dim, self.nID)  # 不同的track id分类最后一层FC:将特征转换到概率得分
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)  # 不同的track id分类用交叉熵损失
        # self.TriLoss = TripletLoss()

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))  # 检测的损失缩放系数
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))  # track id分类的损失缩放系数

    def forward(self, outputs, batch):
        """
        :param outputs:
        :param batch:
        :return:
        """
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0.0, 0.0, 0.0, 0.0  # 初始化4个loss为0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            # 计算heatmap loss
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                             batch['dense_wh'] * batch['dense_wh_mask']) /
                                mask_weight) / opt.num_stacks
                else:  # 计算box尺寸的L1/Smooth L1 loss
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:  # 计算box中心坐标偏移的L1 loss
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            # 检测目标id分类的交叉熵损失
            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()  # 只有有目标的像素才计算id loss
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]  # 有目标的track id
                id_output = self.classifier.forward(id_head).contiguous()  # 用于检测目标分类的最后一层是FC?
                id_loss += self.IDLoss(id_output, id_target)
                # id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss \
                   + opt.wh_weight * wh_loss \
                   + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss \
               + torch.exp(-self.s_id) * id_loss \
               + (self.s_det + self.s_id)
        loss *= 0.5
        # print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss,
                      'hm_loss': hm_loss,
                      'wh_loss': wh_loss,
                      'off_loss': off_loss,
                      'id_loss': id_loss}
        return loss, loss_stats


# 核心训练类
class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
