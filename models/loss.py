import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.metrics import *

class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        # 被忽略的先验框的mask都是-1，不参与loss计算
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size
            return loss
        else:
            return loss

def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    '''
    #input 
        pred_conf: [B, HW, 1]
        pred_cls: [B, HW, C]
        pred_txtytwth: [B, HW, 4]
        targets: [B, HW, 8]
    #output
        conf_loss: [1,]
        cls_loss: [1,]
        bbox_loss: [1,]
        total_loss: [1,]
    '''
    batch_size = pred_conf.size(0)
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 预测
    pred_conf = pred_conf[..., 0]           # [B, HW,]
    pred_cls = pred_cls.permute(0, 2, 1)    # [B, C, HW]
    pred_txty = pred_txtytwth[..., :2]      # [B, HW, 2]
    pred_twth = pred_txtytwth[..., 2:]      # [B, HW, 2]

    # 标签  
    gt_conf = targets[..., 0].float()                 # [B, HW,]
    gt_obj = targets[..., 1].float()                  # [B, HW,]
    gt_cls = targets[..., 2].long()                   # [B, HW,]
    gt_txty = targets[..., 3:5].float()               # [B, HW, 2]
    gt_twth = targets[..., 5:7].float()               # [B, HW, 2]
    gt_box_scale_weight = targets[..., 7]             # [B, HW,]
    gt_mask = (gt_box_scale_weight > 0.).float()      # [B, HW,]

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    conf_loss = conf_loss.sum() / batch_size
    
    # 类别损失，由于reduction是'none'，因此可以使用gt_mask筛选并手动求和
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_mask
    cls_loss = cls_loss.sum() / batch_size
    
    # 边界框txty的损失
    txty_loss = txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_mask * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth的损失
    twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_mask * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = txty_loss + twth_loss

    return conf_loss, cls_loss, bbox_loss

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.conf_loss_function = MSEWithLogitsLoss()
        self.cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        self.txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        self.twth_loss_function = nn.MSELoss(reduction='none')
        self.device = config['device']
        self.input_size = config['input_size']
        self.stride = config['stride']
        self.anchor_size = torch.tensor(config['anchor_size'][config['dataset']])
        self.num_anchors = len(self.anchor_size)
        self.anchor_boxes = self.create_grid(config['input_size'])

    def set_grid(self, input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(input_size)

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # 生成G矩阵
        fmp_w, fmp_h = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2]
        grid_xy = grid_xy[:, None, :].repeat(1, self.num_anchors, 1)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([grid_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
    def decode_boxes(self, anchors, txtytwth_pred):
        """将txtytwth预测换算成边界框的左上角点坐标和右下角点坐标 \n
            Input: \n
                txtytwth_pred : [B, H*W*KA, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*KA, 4] \n
        """
        # 获得边界框的中心点坐标和宽高
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + anchors[..., :2]
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * anchors[..., 2:]

        # [B, H*W*KA, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1) * self.stride

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[..., :2] = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred[..., 2:] = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5

        return x1y1x2y2_pred
    def forward(self, predictions, targets):
        batch_size, H, W, _, C = predictions.shape
        # reshape (batch, H, W, B*(5+C)) --> (batch, H, W, B, 5+C)
        predictions = predictions.view(batch_size, -1, C)

        # 分别取出
        pred_conf = predictions[..., 0].contiguous().view(batch_size, -1, 1)   # confidence
        # pred_box_xywh = predictions[..., 1:5].contiguous().view(batch_size, -1, 4)  # (x, y, w, h) 
        # pred_cls = predictions[..., 5:].contiguous().view(batch_size, -1, C - 5)   # classes
        pred_cls = predictions[..., 1:C-4].contiguous().view(batch_size, -1, C - 5)   # classes
        pred_box_xywh = predictions[..., C-4:].contiguous().view(batch_size, -1, 4)  # (x, y, w, h) 
        # 计算iou_scores
        box_xyxy_pred = (self.decode_boxes(self.anchor_boxes, pred_box_xywh) / self.input_size).view(-1, 4)
        box_xyxy_gt = targets[:, :, 7:].contiguous().view(-1, 4)
        iou_scores = iou_score(box_xyxy_pred, box_xyxy_gt).contiguous().view(batch_size, -1, 1)
        with torch.no_grad():
                gt_conf = iou_scores.clone()
        targets = torch.cat([gt_conf, targets[:, :, :7]], dim=2)

        # conf_loss, bbox_loss, cls_loss  = C(pred_conf, pred_box_xywh, pred_cls, targets)

        # 预测
        pred_conf = pred_conf[..., 0]           # [B, HW,]
        pred_cls = pred_cls.permute(0, 2, 1)    # [B, C, HW]
        pred_txty = pred_box_xywh[..., :2]      # [B, HW, 2]
        pred_twth = pred_box_xywh[..., 2:]      # [B, HW, 2]

        # 标签  
        gt_conf = targets[..., 0].float()                 # [B, HW,]
        gt_obj = targets[..., 1].float()                  # [B, HW,]
        gt_cls = targets[..., 2].long()                   # [B, HW,]
        gt_txty = targets[..., 3:5].float()               # [B, HW, 2]
        gt_twth = targets[..., 5:7].float()               # [B, HW, 2]
        gt_box_scale_weight = targets[..., 7]             # [B, HW,]
        gt_mask = (gt_box_scale_weight > 0.).float()      # [B, HW,]

        # 置信度损失
        conf_loss = self.conf_loss_function(pred_conf, gt_conf, gt_obj)
        conf_loss = conf_loss.sum() / batch_size
        
        # 类别损失
        cls_loss = self.cls_loss_function(pred_cls, gt_cls) * gt_mask
        cls_loss = cls_loss.sum() / batch_size
        # print(f'pred_twth {pred_twth}')
        # 边界框txty的损失
        txty_loss = self.txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_mask * gt_box_scale_weight
        txty_loss = txty_loss.sum() / batch_size
        # 边界框twth的损失
        twth_loss = self.twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_mask * gt_box_scale_weight
        twth_loss = twth_loss.sum() / batch_size
        bbox_loss = txty_loss + twth_loss

        # --- 总损失
        total_loss = conf_loss + bbox_loss + cls_loss
        # print(f'conf_loss = {conf_loss}, cls_loss = {cls_loss}, bbox_loss = {bbox_loss}')

        return total_loss




