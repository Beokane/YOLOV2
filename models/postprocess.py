import torch
import numpy as np
from utils.tools import decode_boxes, nms


# def postprocess(conf_pred, cls_pred, reg_pred, stride, input_size, anchors, conf_thresh, num_classes, topk):
#     """
#     Input:
#         conf_pred: (Tensor) [B, H*W*KA, 1]
#         cls_pred:  (Tensor) [B, H*W*KA, C]
#         reg_pred:  (Tensor) [B, H*W*KA, 4]
#         anchors:   (Tensor) [H*W*KA, 4]
#     """
#     # print(anchors.shape)
#     batch_size = conf_pred.size(0)
#     results = []
#     for i in range(batch_size):
#         cur_anchors = anchors.clone()
#         conf_b = conf_pred[i].clone()
#         cls_b = cls_pred[i].clone()
#         reg_b = reg_pred[i].clone()
#         # softmax 是对每个样本的每个类别的概率进行归一化，一般适用于多类别互斥任务。[H*W*KA, C]
#         scores = (torch.sigmoid(conf_b) * torch.softmax(cls_b, dim=-1)).flatten()

#         # Keep top k top scoring indices only.
#         num_topk = min(topk, reg_b.size(0))

#         # torch.sort is actually faster than .topk (at least on GPUs)
#         predicted_prob, topk_idxs = scores.sort(descending=True)
#         topk_scores = predicted_prob[:num_topk]
#         topk_idxs = topk_idxs[:num_topk]

#         # filter out the proposals with low confidence score
#         # keep_idxs = topk_scores > conf_thresh 说明是多标签
#         keep_idxs = topk_scores > conf_thresh
#         scores = topk_scores[keep_idxs]
#         topk_idxs = topk_idxs[keep_idxs]

#         anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
#         labels = topk_idxs % num_classes
#         # print(anchor_idxs)
#         reg_b = reg_b[anchor_idxs]
#         cur_anchors = cur_anchors[anchor_idxs]

#         # 解算边界框, 并归一化边界框: [n, 4]
#         bboxes = decode_boxes(stride, cur_anchors, reg_b)

#         # to cpu
#         scores = scores.cpu().numpy()
#         labels = labels.cpu().numpy()
#         bboxes = bboxes.cpu().numpy()

#         # NMS
#         keep = np.zeros(len(bboxes), dtype=int)
#         for i in range(num_classes):
#             inds = np.where(labels == i)[0]
#             if len(inds) == 0:
#                 continue
#             c_bboxes = bboxes[inds]
#             c_scores = scores[inds]
#             c_keep = nms(c_bboxes, c_scores)
#             keep[inds[c_keep]] = 1

#         keep = np.where(keep > 0)
#         bboxes = bboxes[keep]
#         scores = scores[keep]
#         labels = labels[keep]

#         # 归一化边界框
#         bboxes = bboxes / input_size
#         bboxes = np.clip(bboxes, 0., 1.)
#         result = np.concatenate([scores[:, np.newaxis], labels[:, np.newaxis], bboxes], axis=1)
#         result = np.concatenate([labels[:, np.newaxis], scores[:, np.newaxis], bboxes], axis=1)
#         results.append(result)

#     return results

def postprocess(conf_pred, cls_pred, reg_pred, stride, input_size, anchors, conf_thresh, num_classes, topk):
    """
    Input:
        conf_pred: (Tensor) [B, H*W*KA, 1]
        cls_pred:  (Tensor) [B, H*W*KA, C]
        reg_pred:  (Tensor) [B, H*W*KA, 4]
        anchors:   (Tensor) [H*W*KA, 4]
    """
    # print(anchors.shape)
    batch_size = conf_pred.size(0)
    results = []
    for i in range(batch_size):
        cur_anchors = anchors.clone()
        conf_b = conf_pred[i].clone()
        cls_b = cls_pred[i].clone()
        reg_b = reg_pred[i].clone()
        # softmax 是对每个样本的每个类别的概率进行归一化，一般适用于多类别互斥任务。[H*W*KA, C]
        scores, labels = torch.max(torch.softmax(cls_b, dim=-1), dim=-1)  # shape: [N], [N]
        scores = scores * torch.sigmoid(conf_b).squeeze(-1)  # 融合 objectness
        # print(scores.shape)
        # Keep top k top scoring indices only.
        num_topk = min(topk, reg_b.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, ids = scores.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = ids[:num_topk]

        # filter out the proposals with low confidence score
        # keep_idxs = topk_scores > conf_thresh 说明是多标签
        conf_idxs = topk_scores > conf_thresh
        keep_scores = topk_scores[conf_idxs]
        keep_idxs = topk_idxs[conf_idxs]

        # anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
        labels = labels[keep_idxs]
        anchor_idxs = keep_idxs
        # print(anchor_idxs)
        reg_b = reg_b[anchor_idxs]
        cur_anchors = cur_anchors[anchor_idxs]

        # 解算边界框, 并归一化边界框: [n, 4]
        bboxes = decode_boxes(stride, cur_anchors, input_size, reg_b)

        # to cpu
        scores = keep_scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 归一化边界框
        bboxes = bboxes / input_size
        bboxes = np.clip(bboxes, 0., 1.)
        # result = np.concatenate([scores[:, np.newaxis], labels[:, np.newaxis], bboxes], axis=1)
        result = np.concatenate([labels[:, np.newaxis], scores[:, np.newaxis], bboxes], axis=1)
        results.append(result)

    return results