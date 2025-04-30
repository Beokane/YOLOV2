import torch
import numpy as np
from sklearn.metrics import auc

def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    eps = 1e-6
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i + eps)

def calculate_map(preds, targets, iou_threshold=0.01):
    batch_size = len(preds)
    aps = []

    for i in range(batch_size):
        preds_b = preds[i]
        gts_b = targets[i]

        preds_b = sorted(preds_b, key=lambda x: x[0], reverse=True)  # 按置信度排序
        # print(f'preds_b {preds_b}')
        tp = np.zeros(len(preds_b))
        fp = np.zeros(len(preds_b))
        gt_matched = set()

        for pred_idx, pred in enumerate(preds_b):
            pred_conf, pred_cls, pred_x1, pred_y1, pred_x2, pred_y2 = pred
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts_b):
                gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = gt

                if pred_cls != gt_cls:
                    continue

                iou = iou_score(
                    torch.tensor([pred_x1, pred_y1, pred_x2, pred_y2]),
                    torch.tensor([gt_x1, gt_y1, gt_x2, gt_y2])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                tp[pred_idx] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[pred_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (len(gts_b) + 1e-6)

        # print(f'tp_cumsum {tp_cumsum}, fp_cumsum {fp_cumsum}')

        # 修正顺序
        if len(recall) > 0 and len(precision) > 0:
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        else:
            ap = 0.0

        aps.append(ap)

    return np.mean(aps) if aps else 0.0

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
