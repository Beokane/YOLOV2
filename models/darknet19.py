import torch
import torch.nn as nn
from collections import OrderedDict

class Darknet19(nn.Module):
    def __init__(self, pretrained=False):
        super(Darknet19, self).__init__()
        
        # 网络结构定义
        layers = [
            # Stage 1
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)),
            ('bn1', nn.BatchNorm2d(32)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            # Stage 2
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            # 继续添加所有层...
            # Stage 3-5 (示例，需补全)
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            
            # 最终输出1024通道
            ('conv17', nn.Conv2d(128, 1024, kernel_size=3, padding=1)),
            ('bn17', nn.BatchNorm2d(1024)),
            ('leaky17', nn.LeakyReLU(0.1, inplace=True))
        ]
        
        self.features = nn.Sequential(OrderedDict(layers))
        
        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights()

    def forward(self, x):
        return self.features(x)
    
    def _load_pretrained_weights(self):
        # 这里应该实现权重加载逻辑
        # 示例：从官方Darknet19权重文件加载
        try:
            state_dict = torch.load('weights/darknet19.pth')
            self.load_state_dict(state_dict)
            print("Loaded pretrained weights")
        except:
            print("Failed to load pretrained weights")
            pass

# 快捷函数
def darknet19(pretrained=False, **kwargs):
    return Darknet19(pretrained=pretrained, **kwargs)