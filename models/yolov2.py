import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv, reorg_layer
from models.backbone import build_backbone

import numpy as np
from .loss import iou_score, compute_loss


class YOLOv2(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes=20,
                 conf_thresh=0.001):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.stride = cfg['stride']

        # Anchor box config
        # self.anchor_size = torch.tensor(anchor_size)  # [KA, 2]
        self.anchor_size = torch.tensor(cfg['anchor_size']['voc'])
        self.num_anchors = len(self.anchor_size)

        # 主干网络
        self.backbone, feat_dims = build_backbone(
            cfg['backbone'], cfg['pretrained'])

        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(feat_dims[-1], cfg['head_dim'], k=3, p=1),
            Conv(cfg['head_dim'], cfg['head_dim'], k=3, p=1)
        )

        # 融合高分辨率的特征信息
        self.route_layer = Conv(feat_dims[-2], cfg['reorg_dim'], k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(
            cfg['head_dim']+cfg['reorg_dim']*4, cfg['head_dim'], k=3, p=1)

        # 预测层
        self.pred = nn.Conv2d(
            cfg['head_dim'], self.num_anchors*(1 + 4 + self.num_classes), 1)

        if self.training:
            self.init_bias()

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(
            self.pred.bias[..., 1*self.num_anchors:(1+self.num_classes)*self.num_anchors], bias_value)

    def forward(self, x, targets=None):
        # backbone主干网络
        feats = self.backbone(x)
        c4, c5 = feats['c4'], feats['c5']
        # 处理c5特征
        p5 = self.convsets_1(c5)
        # 融合c4特征
        p4 = self.reorg(self.route_layer(c4))
        p5 = torch.cat([p4, p5], dim=1)
        # 处理p5特征
        p5 = self.convsets_2(p5)
        # 预测
        prediction = self.pred(p5)

        batch_size, _, height, width = prediction.size()

        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(
            batch_size, height, width, self.num_anchors, 1+4+self.num_classes)

        return prediction