import os
from tqdm import tqdm
import torch
import torch.optim as optim
from argparse import ArgumentParser

from data.datasets import *
from data.transforms import *
from models.yolov2 import YOLOv2
from models.loss import Loss
from models.postprocess import postprocess
from models.matcher import *
from utils.tools import create_grid
from utils.metrics import calculate_map
from configs.yolov2_config import yolov2_config



def collate_fn(batch):
    """
    处理YOLO数据集的批量组合
    - 图像张量堆叠
    - 标签列表合并（保持变长特性）
    """
    images, labels = zip(*batch)  # 解压batch

    # 图像张量堆叠 (B, C, H, W)
    images = torch.stack(images, 0)  # 自动处理相同尺寸的张量

    # 标签处理：合并为一个列表，每个元素是(N_i, 5)
    labels = [torch.as_tensor(label, dtype=torch.float32) for label in labels]
    return images, labels


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
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=True,
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
    parser.add_argument('--input_size', default=416,
                        type=int, help='input_size')

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


def build_dataset(path, train_size, transforms):
    transform = transforms
    # 构建dataset类和dataloader类model
    data_root = os.path.join(path, 'VOCdevkit')
    # 加载voc数据集
    num_classes = 20
    dataset = VOCDetection(
        root=data_root,
        transform=transform
    )

    print('The dataset size:', len(dataset))
    return dataset, num_classes


def train(model, args, config, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    with tqdm(dataloader, desc='Training', unit='batch') as pbar:
        # epoch_size = len(dataloader)
        for iter_i, (images, targets) in enumerate(pbar):
            # 训练
            # ni = iter_i+epoch*epoch_size
            # if not config['no_warm_up']:
            #     if epoch < config['wp_epoch']:
            #         nw = config['wp_epoch']*epoch_size
            #         tmp_lr = base_lr * pow((ni)*1. / (nw), 4)
            #         set_lr(optimizer, tmp_lr)

            #     elif epoch == config['wp_epoch'] and iter_i == 0:
            #         tmp_lr = base_lr
            #         set_lr(optimizer, tmp_lr)
            images = images.to(args.device)
            # targets numpy shape (batch_size, box_num, 4+1)
            targets = [label.tolist() for label in targets]
            # targets shape (batch_size, grid_h*grid_w*anchor_num, 1+1+4+1+4）
            # 1: objectness    # 是否存在物体，用于区分正样本和负样本
            # 1: class
            # 4: tx, ty, tw, th
            # 1: box_scale_weight    # 用于平衡正样本和负样本的损失
            # 4: xmin, ymin, xmax, ymax
            targets = gt_creator(
                input_size=args.train_size,
                stride=config['stride'],
                label_lists=targets,
                anchor_size=config['anchor_size'][args.dataset],
                ignore_thresh=config['ignore_thresh']
            )
            targets = torch.tensor(
                targets, dtype=torch.float32).to(args.device)
            # 前向传播
            has_nan = torch.isnan(images).any()
            has_inf = torch.isinf(images).any()

            if has_nan or has_inf:
                print(
                    f"NaN or Inf detected in the input tensor.{has_nan}, {has_inf}")

            outputs = model(images)
            has_nan = torch.isnan(outputs).any()
            has_inf = torch.isinf(outputs).any()
            if has_nan or has_inf:
                print(
                    f"NaN or Inf detected in the output tensor. {has_nan}, {has_inf}")
            loss = criterion(outputs, targets)
            cur_loss = loss.item()
            total_loss += cur_loss
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印日志
            pbar.set_postfix({'Loss': cur_loss})
    total_loss = total_loss / len(dataloader)
    print(f"Total Loss: {total_loss:.4f}")
    return total_loss


def validate(model, args, config, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation', unit='batch') as pbar:
            for i, (b_images, b_targets) in enumerate(pbar):
                b_images = b_images.to(args.device)
                targets = [targets.tolist() for targets in b_targets]
                # 前向传播
                outputs = model(b_images)
                # 分离结果
                B, H, W, _, C = outputs.shape
                conf_pred = outputs[..., 0:1].contiguous().view(B, -1, 1)
                cls_pred = outputs[..., 1: C-4].contiguous().view(B, -1, C-5)
                txtytwth_pred = outputs[..., C-4:].contiguous().view(B, -1, 4)
                # anchors 是每个先验框 
                anchors_size = torch.tensor(config['anchor_size'][args.dataset])
                anchors = create_grid(config['stride'], args.train_size, anchors_size).to(args.device)
                # 计算解码边界框的窗格信息
                result = postprocess(conf_pred, cls_pred, txtytwth_pred, config['stride'], args.train_size,
                                                     anchors, 0.001, len(DETECTION_CLASSES), 1000)
                
                for b_idx, dets in enumerate(result):  # 遍历当前 batch 中每张图片的结果
                    for det in dets:
                        all_preds.append([i * B + b_idx] + det.tolist())  # [image_id, class_id, conf, x1, y1, x2, y2]
                for b_idx, target in enumerate(targets):  # target: [[class_id, x1, y1, x2, y2], ...]
                    for t in target:
                        all_targets.append([i * B + b_idx] + t[-1:] + t[:-1])  # [image_id, class_id, x1, y1, x2, y2]
                # 计算mAP
    map = calculate_map(all_preds, all_targets)
    return map

def check_nan_hook(module, input, output):
    if torch.is_tensor(output):
        if torch.isnan(output).any():
            print(f'[NaN Warning] {module.__class__.__name__} output has NaN!')
    elif isinstance(output, (tuple, list)):
        for o in output:
            if torch.is_tensor(o) and torch.isnan(o).any():
                print(
                    f'[NaN Warning] {module.__class__.__name__} output has NaN!')

# def set_lr(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def main():
    # 加载配置
    args = parse_args()
    model_config = yolov2_config[args.version]
    # configs = {**cfg, **vars(args)}

    # 是否使用cuda
    args.device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # 多尺度训练
    args.train_size, args.val_size = (
        640, 416) if args.multi_scale else (416, 416)
    # 数据预处理
    transform = Augmentation(args.train_size)

    # 数据加载
    train_dataset = DetectionDataset(
        'datasets/VOCdevkit', args.train_size, [('2012', 'train')], transform)
    val_dataset = DetectionDataset(
        'datasets/VOCdevkit', args.val_size, [('2012', 'val')], transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 模型构建
    model = YOLOv2(model_config)
    model = model.to(args.device)

    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

    # 优化器构建
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # 损失函数
    anchors = torch.tensor(model_config['anchor_size'][args.dataset])
    criterion = Loss(args, model_config['stride'], anchors)

    # 训练
    best_mAP = 0.0
    for epoch in range(args.start_epoch, args.max_epoch):
        train_loss = train(model, args, model_config,
                           train_dataloader, criterion, optimizer)
        # train_loss = 0.0
        # 验证
        mAP = validate(model, args, model_config, val_dataloader)
        print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {train_loss:.4f}, mAP: {mAP:.4f}")
        # print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {train_loss:.4f}")
        # 保存最好的模型
        if mAP >= best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), f"best_model.pth")
            print(f"Best model saved with mAP: {best_mAP:.4f}")


if __name__ == '__main__':
    main()
