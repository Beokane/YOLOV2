import torch
import numpy as np
import cv2
from models.yolov2 import YOLOv2
from argparse import ArgumentParser
from configs.yolov2_config import  yolov2_config
from data.datasets import DETECTION_CLASSES

# -----------------------------
# 参数设置
# -----------------------------
ANCHORS = [
    (1.19, 1.98), (2.79, 4.59), (4.53, 8.92), (8.06, 5.29), (10.32, 10.65)
]  # YOLOv2常用的5个anchors
NUM_CLASSES = 20
INPUT_SIZE = 416
CONF_THRESH = 0.1
NMS_THRESH = 0.45

# -----------------------------
# 解码函数
# -----------------------------
def load_weight(model, path_to_ckpt=None):
    # check
    if path_to_ckpt is None:
        print('no weight file ...')
        return model
        
    checkpoint_state_dict = torch.load(path_to_ckpt, map_location='cpu')
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    x = np.array(x)
    x_exp = np.exp(x - np.max(x))  # 减去最大值，防止数值爆炸
    return x_exp / np.sum(x_exp)


def decode_yolo_output(output, anchors, conf_thresh=0.01, input_dim=416):
    """
    output: (H, W, A, 5+C)
    return: [x1, y1, x2, y2, conf, cls_id]
    """
    H, W, A, _ = output.shape
    boxes = []
    for i in range(H):
        for j in range(W):
            for a in range(A):
                to = output[i, j, a, 0]
                tx, ty, tw, th = output[i, j, a, 21:]
                class_scores = output[i, j, a, 1:21]
                objectness = sigmoid(to)
                class_probs = softmax(class_scores)
                conf_scores = objectness * class_probs

                cls_id = np.argmax(conf_scores)
                conf = conf_scores[cls_id]
                print(f"i: {i}, j: {j}, a: {a}, conf: {conf}, cls_id: {cls_id}")
                if conf < conf_thresh:
                    continue

                cx = (j + sigmoid(tx)) / W
                cy = (i + sigmoid(ty)) / H
                bw = np.exp(tw) * anchors[a][0] / W
                bh = np.exp(th) * anchors[a][1] / H

                x1 = int((cx - bw / 2) * input_dim)
                y1 = int((cy - bh / 2) * input_dim)
                x2 = int((cx + bw / 2) * input_dim)
                y2 = int((cy + bh / 2) * input_dim)
                print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, cls_id: {cls_id}")
                boxes.append([x1, y1, x2, y2, conf, cls_id])
    return boxes

# -----------------------------
# NMS
# -----------------------------
def non_max_suppression(boxes, iou_thresh=0.45):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# -----------------------------
# 画图
# -----------------------------
def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, cls_id: {cls_id}")
        label = f"{DETECTION_CLASSES[int(cls_id)]}: {conf:.2f}"
        # cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # cv2.putText(img, label, (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        cv2.putText(img, label, (int(x1), max(int(y1) - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return img

# -----------------------------
# 主流程
# -----------------------------
def run_test(model, image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = img_resized[:, :, ::-1].copy()  # 修复负stride问题
    input_tensor = torch.tensor(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.

    with torch.no_grad():
        output = model(input_tensor)[0]  # (H, W, A, 5+C)

    output_np = output.cpu().numpy()
    boxes = decode_yolo_output(output_np, anchors=ANCHORS, conf_thresh=CONF_THRESH, input_dim=INPUT_SIZE)
    boxes_nms = non_max_suppression(boxes, NMS_THRESH)
    result = draw_boxes(img_resized, boxes_nms)

    cv2.imwrite("result.jpg", result)
    print("检测完成，已保存 result.jpg")

def parse_args():
    parser = ArgumentParser(description='YOLOv2 Training')
    # 基本参数
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval_epoch', type=int,
                        default=10, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')

    # 模型参数
    parser.add_argument('-v', '--version', default='yolov2',
                        help='build yolo')
    parser.add_argument('--conf_thresh', default=0.001, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.50, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk predicted candidates')

    # 训练配置
    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('-accu', '--accumulate', default=8, type=int,
                        help='gradient accumulate.')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[100, 150], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--input_size', default=416, type=int, help='input_size')

    # 优化器参数
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')

    # 数据集参数
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')

    return parser.parse_args()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # 需要根据你的输入大小调整
    img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)

if __name__ == '__main__':
    args = parse_args()
    cfg = yolov2_config[args.version]
    configs = {**cfg, **vars(args)}
    model = YOLOv2(configs)
    model = load_weight(model, '/home/oem/MachineLearning/detect/best_model.pth')
    image_path = '/home/oem/MachineLearning/detect/datasets/2008_001183.jpg'
    run_test(model, image_path)
