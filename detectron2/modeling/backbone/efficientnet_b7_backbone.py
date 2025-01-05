import torch
import torch.nn as nn
import torchvision.models as tv_models

from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY

class EfficientNetB7Backbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        # Hardcode EfficientNet-B7
        base_model = tv_models.efficientnet_b7(pretrained=True)

        # From your printed shapes:
        #   Block 0 -> [1, 64, 112, 112] (stride=2)
        #   Block 1 -> [1, 32, 112, 112] (still stride=2)
        #   Block 2 -> [1, 48, 56, 56]   (stride=4)
        #   Block 3 -> [1, 80, 28, 28]   (stride=8)
        #   Block 4 -> [1,160, 14, 14]   (stride=16)
        #   Block 5 -> [1,224, 14, 14]   (still stride=16)
        #   Block 6 -> [1,384,  7,  7]   (stride=32)
        #   Block 7 -> [1,640,  7,  7]   (still stride=32)
        #   Block 8 -> [1,2560, 7,  7]   (still stride=32)
        #
        # We want the final 4 stages to be:
        #   stage1 => stride=4  (res2)
        #   stage2 => stride=8  (res3)
        #   stage3 => stride=16 (res4)
        #   stage4 => stride=32 (res5)
        #
        # stage0 (not returned) is blocks 0..1 => stride=2
        # stage1 = block [2:3] => stride=4
        # stage2 = block [3:4] => stride=8
        # stage3 = block [4:6] => stride=16
        # stage4 = block [6:]  => stride=32

        self.stage0 = nn.Sequential(*base_model.features[0:2])  # blocks 0..1 => stride=2 (not returned)
        self.stage1 = nn.Sequential(*base_model.features[2:3])  # block 2 => stride=4 => res2
        self.stage2 = nn.Sequential(*base_model.features[3:4])  # block 3 => stride=8 => res3
        self.stage3 = nn.Sequential(*base_model.features[4:6])  # blocks 4..5 => stride=16 => res4
        self.stage4 = nn.Sequential(*base_model.features[6:])   # blocks 6..8 => stride=32 => res5

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        # Dummy pass to record out_channels
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
        # stage0 => stride=2 (ignored for output)
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
def build_efficientnet_b7_backbone(cfg, input_shape: ShapeSpec):
    return EfficientNetB7Backbone(cfg, input_shape)
