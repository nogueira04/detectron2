# file: detectron2/modeling/backbone/efficientnet_b7_fpn.py

from detectron2.modeling.backbone import FPN, BACKBONE_REGISTRY
from detectron2.modeling.backbone.efficientnet_b7_backbone import build_efficientnet_b7_backbone


@BACKBONE_REGISTRY.register()
def build_efficientnet_b7_fpn_backbone(cfg, input_shape):
    """
    Build an FPN on top of an EfficientNet-B7 backbone.
    This is used when cfg.MODEL.BACKBONE.NAME == "build_efficientnet_b7_fpn_backbone".
    """
    # 1. Build the bottom-up (EfficientNet B7) backbone:
    bottom_up = build_efficientnet_b7_backbone(cfg, input_shape)

    # 2. Build the FPN with the desired in_features, out_channels, fuse_type, etc.
    in_features = cfg.MODEL.FPN.IN_FEATURES     # e.g. ["res2", "res3", "res4", "res5"]
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS   # e.g. 256
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE         # "sum" or "avg", typically "sum"

    fpn = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        fuse_type=fuse_type,
    )
    return fpn
