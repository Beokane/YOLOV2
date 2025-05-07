import torch
import numpy as np
from collections import defaultdict

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

def compute_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# def calculate_map(preds, targets, iou_threshold=0.5):
#     image_size = len(preds)
#     class_aps = defaultdict(list)  # 存储每个类别的AP

#     for id in range(image_size):
#         preds_b = preds[id]
#         gts_b = targets[id]

#         # 按类别分组预测和GT
#         preds_by_class = defaultdict(list)
#         gts_by_class = defaultdict(list)

#         for pred in preds_b:
#             conf, cls, *box = pred
#             preds_by_class[cls].append((conf, *box))

#         for gt in gts_b:
#             # print(gts_b)
#             *box, cls = gt
#             gts_by_class[cls].append(box)

#         # 对每个类别单独计算AP
#         for cls in set(preds_by_class.keys()).union(gts_by_class.keys()):
#             cls_preds = preds_by_class.get(cls, [])
#             cls_gts = gts_by_class.get(cls, [])

#             if len(cls_preds) == 0 or len(cls_gts) == 0:
#                 class_aps[cls].append(0.0)
#                 continue

#             # 按置信度排序
#             cls_preds = sorted(cls_preds, key=lambda x: x[0], reverse=True)

#             tp = np.zeros(len(cls_preds))
#             fp = np.zeros(len(cls_preds))
#             gt_matched = set()

#             for pred_idx, pred in enumerate(cls_preds):
#                 conf, x1, y1, x2, y2 = pred
#                 best_iou = 0.0
#                 best_gt_idx = -1

#                 for gt_idx, gt in enumerate(cls_gts):
#                     gt_x1, gt_y1, gt_x2, gt_y2 = gt
#                     iou = iou_score(
#                         torch.tensor([[x1, y1, x2, y2]]),
#                         torch.tensor([[gt_x1, gt_y1, gt_x2, gt_y2]])
#                     ).item()

#                     if iou > best_iou and gt_idx not in gt_matched:
#                         best_iou = iou
#                         best_gt_idx = gt_idx

#                 if best_iou >= iou_threshold:
#                     tp[pred_idx] = 1
#                     gt_matched.add(best_gt_idx)
#                 else:
#                     fp[pred_idx] = 1

#             # 计算precision和recall
#             tp_cumsum = np.cumsum(tp)
#             fp_cumsum = np.cumsum(fp)
#             denom = tp_cumsum + fp_cumsum
#             precision = np.divide(tp_cumsum, denom, out=np.zeros_like(tp_cumsum), where=denom > 0)
#             recall = tp_cumsum / len(cls_gts)

#             # 计算AP（Area Under PR Curve）
#             ap = compute_ap(recall, precision, False)

#             class_aps[cls].append(ap)

#     # 计算每个类别的平均AP，然后对所有类别取平均
#     mean_ap = np.mean([np.mean(aps) for aps in class_aps.values()]) if class_aps else 0.0
#     return mean_ap

def calculate_map(preds, targets, iou_threshold=0.5, num_classes=20):
    def iou_caculation(box1, box2):
        """
        计算 IoU 输入是[x1, y1, x2, y2]格式
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    aps = []
    for cls_id in range(num_classes):
        # 筛选该类的 preds 和 targets
        cls_preds = [p for p in preds if p[1] == cls_id]
        cls_targets = [t for t in targets if t[1] == cls_id]
        npos = len(cls_targets)

        if npos == 0:
            aps.append(0)
            continue

        # 排序
        cls_preds = sorted(cls_preds, key=lambda x: -x[2])  # 按confidence降序
        image_gt = defaultdict(list)
        for t in cls_targets:
            image_gt[t[0]].append({'bbox': t[2:], 'used': False})

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, pred in enumerate(cls_preds):
            image_id, conf, pred_box = pred[0], pred[2], pred[3:]
            max_iou = 0
            max_gt = None
            for gt in image_gt[image_id]:
                iou = iou_caculation(pred_box, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt = gt

            if max_iou >= iou_threshold and max_gt is not None and not max_gt['used']:
                tp[i] = 1
                max_gt['used'] = True
            else:
                fp[i] = 1

        # 累积
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / npos
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = compute_ap(recall, precision, False)
        
        aps.append(ap)

    return np.mean(aps)



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
