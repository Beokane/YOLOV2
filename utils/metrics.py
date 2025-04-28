import torch
import numpy as np

def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)

class DetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.predictions = []
        self.ground_truths = []

    def add_sample(self, pred, gt):
        """
        pred: {'boxes': Tensor[N, 4], 'labels': Tensor[N], 'scores': Tensor[N]}
        gt:   {'boxes': Tensor[M, 4], 'labels': Tensor[M]}
        """
        self.predictions.append(pred)
        self.ground_truths.append(gt)

    def compute_iou(self, boxes1, boxes2):
        """IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou  # shape: [N, M]

    def compute(self):
        all_tp, all_fp, all_fn = 0, 0, 0

        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']

            gt_boxes = gt['boxes']
            gt_labels = gt['labels']

            matched_gt = set()
            tp, fp = 0, 0

            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue

            ious = self.compute_iou(pred_boxes, gt_boxes)

            for i in range(len(pred_boxes)):
                max_iou, max_j = 0, -1
                for j in range(len(gt_boxes)):
                    if j in matched_gt:
                        continue
                    if pred_labels[i] != gt_labels[j]:
                        continue
                    iou = ious[i, j]
                    if iou > max_iou:
                        max_iou = iou
                        max_j = j

                if max_iou >= self.iou_threshold:
                    matched_gt.add(max_j)
                    tp += 1
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            all_tp += tp
            all_fp += fp
            all_fn += fn

        precision = all_tp / (all_tp + all_fp + 1e-6)
        recall = all_tp / (all_tp + all_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou_threshold': self.iou_threshold
        }
