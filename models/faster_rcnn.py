import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn

def build_faster_rcnn(config):
    # 骨干网络选择
    if config.backbone == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
    elif config.backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    
    # 锚点生成器
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # ROI Pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # 构建模型
    model = FasterRCNN(
        backbone,
        num_classes=config.num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=config.min_size,
        max_size=config.max_size,
        box_score_thresh=config.score_thresh
    )
    
    return model