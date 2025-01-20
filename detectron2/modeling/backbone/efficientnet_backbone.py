import torch
import torch.nn as nn
import torchvision.models as tv_models

from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY


class EfficientNetBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        model_name = cfg.MODEL.BACKBONE.EFFICIENTNET_TYPE  # e.g. "efficientnet_b0"
        base_model = getattr(tv_models, model_name)(pretrained=True)

        # ----------------------------
        # Group blocks by their stride
        # ----------------------------
        # stage0 -> stride=2 (internal; not returned as res2)
        self.stage0 = nn.Sequential(*base_model.features[0:2])
        # stage1 -> stride=4 => res2
        self.stage1 = nn.Sequential(base_model.features[2])
        # stage2 -> stride=8 => res3
        self.stage2 = nn.Sequential(base_model.features[3])
        # stage3 -> stride=16 => res4
        self.stage3 = nn.Sequential(*base_model.features[4:6])
        # stage4 -> stride=32 => res5
        self.stage4 = nn.Sequential(*base_model.features[6:])

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        # Run a dummy pass to record the number of channels in each stage:
        dummy_in = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        with torch.no_grad():
            r2, r3, r4, r5 = self._forward_impl(dummy_in)
            self._out_feature_channels = {
                "res2": r2.shape[1],
                "res3": r3.shape[1],
                "res4": r4.shape[1],
                "res5": r5.shape[1],
            }

    def _forward_impl(self, x):
        # stage0 => stride=2
        x = self.stage0(x)

        # stage1 => stride=4 => res2
        x = self.stage1(x)
        res2 = x

        # stage2 => stride=8 => res3
        x = self.stage2(x)
        res3 = x

        # stage3 => stride=16 => res4
        x = self.stage3(x)
        res4 = x

        # stage4 => stride=32 => res5
        x = self.stage4(x)
        res5 = x

        return res2, res3, res4, res5

    def forward(self, x):
        r2, r3, r4, r5 = self._forward_impl(x)
        return {
            "res2": r2,
            "res3": r3,
            "res4": r4,
            "res5": r5,
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, input_shape: ShapeSpec):
    return EfficientNetBackbone(cfg, input_shape)
