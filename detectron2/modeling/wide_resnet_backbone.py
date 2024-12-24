from detectron2.modeling import Backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec
from torchvision.models import wide_resnet50_2
from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models

class WideResNetBackbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        self.body = wide_resnet50_2(pretrained=cfg.MODEL.WIDERESNET.PRETRAINED)
        
        # Extract layers corresponding to `res4` and `res5`
        self.res4 = nn.Sequential(OrderedDict(list(self.body.named_children())[:-3]))  # Adjust as necessary
        self.res5 = nn.Sequential(OrderedDict(list(self.body.named_children())[-2:]))  # Last block
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Output shapes
        self._out_features = ["res4", "res5"]
        self._out_feature_channels = {"res4": 1024, "res5": 2048}
        self._out_feature_strides = {"res4": 16, "res5": 32}  # Adjust strides as needed

    def forward(self, x):
        res4 = self.res4(x)  # Output: [B, 2048, 25, 25]
        res5_pooled = self.res5.avgpool(res4)  # Output: [B, 2048, 1, 1]
        res5_flattened = res5_pooled.view(res5_pooled.size(0), -1)  # Flatten to [B, 2048]
        res5 = self.res5.fc(res5_flattened)  # Pass to FC layer for classification
        features = {"res4": res4, "res5": res5}

        return features

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_wideresnet_backbone(cfg, input_shape):
    """
    Build the WideResNet backbone from config.
    """
    return WideResNetBackbone(cfg)

def add_wideresnet_config(cfg):
    """
    Add config for WideResNet backbone.
    """
    cfg.MODEL.WIDERESNET = CN()
    cfg.MODEL.WIDERESNET.DEPTH = 50  # Default depth (50 layers)
    cfg.MODEL.WIDERESNET.WIDTH_MULTIPLIER = 2  # Width multiplier
    cfg.MODEL.WIDERESNET.PRETRAINED = True  # Whether to load pretrained weights
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res4", "res5"]
