import torch
import torch.nn as nn
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

@BACKBONE_REGISTRY.register()
def build_wide_resnet_backbone(cfg, input_shape: ShapeSpec):
    """
    Build a Wide ResNet backbone. This function is invoked by Detectron2's config system
    when 'cfg.MODEL.BACKBONE.NAME' is set to 'build_wide_resnet_backbone'.
    """

    from torchvision.models import wide_resnet50_2

    model = wide_resnet50_2(pretrained=True)  

    backbone = WideResNet(model)  
    return backbone

class WideResNet(Backbone):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


    def forward(self, x):
        """
        Return a dict of feature maps, e.g. {"res2": ..., "res3": ..., ...}.
        The keys (like "res2") can follow the naming in the config.
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        res2 = self.model.layer1(x)
        res3 = self.model.layer2(res2)
        res4 = self.model.layer3(res3)
        res5 = self.model.layer4(res4)

        return {
            "res2": res2,
            "res3": res3,
            "res4": res4,
            "res5": res5,
        }

    def output_shape(self):
        """
        Return a dict of {feature_name: ShapeSpec}, describing
        each returned feature map’s spatial resolution & channels.
        This helps Detectron2's heads know what to expect.
        """
        return {
            "res2": ShapeSpec(channels=256, stride=4),
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        }

