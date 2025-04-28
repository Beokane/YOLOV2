import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# import utils
from utils.metrics import DetectionMetrics
from utils.logger import Logger

class DetectorTrainer:
    def __init__(self, model, optimizer, scheduler, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = Logger(config.log_dir)
        self.metrics = DetectionMetrics(config.num_classes)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            
            # 梯度裁剪
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.clip_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            total_loss += losses.item()
            pbar.set_postfix(loss=losses.item())
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        self.metrics.reset()
        
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            outputs = self.model(images)
            
            # 转换预测结果为标准格式
            preds = self._process_predictions(outputs)
            self.metrics.update(preds, targets)
        
        # 计算mAP等指标
        metrics = self.metrics.compute()
        return metrics
    
    def _process_predictions(self, outputs):
        """将模型输出转换为评估需要的格式"""
        processed = []
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            # 应用NMS
            keep = torchvision.ops.nms(
                torch.from_numpy(boxes),
                torch.from_numpy(scores),
                self.config.nms_thresh
            ).numpy()
            
            processed.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
        return processed
    
    def fit(self, train_loader, val_loader, epochs):
        best_map = 0.0
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self.train_epoch(train_loader)
            
            # 验证阶段
            val_metrics = self.validate(val_loader)
            
            # 记录日志
            self.logger.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mAP': val_metrics['map'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 保存最佳模型
            if val_metrics['map'] > best_map:
                best_map = val_metrics['map']
                self._save_checkpoint(epoch, is_best=True)
            
            # 定期保存
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
    
    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_map': self.metrics.best_map
        }
        filename = f"checkpoint_epoch{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        torch.save(state, os.path.join(self.config.save_dir, filename))