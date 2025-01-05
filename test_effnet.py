# file: test_effnet.py

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model

# Import your custom FPN builder so that it's registered in BACKBONE_REGISTRY
import detectron2.modeling.backbone.efficientnet_fpn

def quick_test_efficientnet():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    # We use the standard "GeneralizedRCNN" meta-architecture,
    # which includes RPN + ROI heads, etc.
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024

    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
    cfg.INPUT.MIN_SIZE_TEST = 224   # or 256, etc.
    cfg.INPUT.MAX_SIZE_TEST = 224   # ensure no resizing beyond this
    cfg.INPUT.MIN_SIZE_TRAIN = 224  # if you are also training
    cfg.INPUT.MAX_SIZE_TRAIN = 224

    # Use the multi-level ROI heads (can handle p2..p5).
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"

    # Use the default box head for classification/regression.
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"

    # Typically you also specify other ROI box head params, e.g.:
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Set our custom FPN-based backbone:
    cfg.MODEL.BACKBONE.NAME = "build_efficientnet_fpn_backbone"
    cfg.MODEL.BACKBONE.EFFICIENTNET_TYPE = "efficientnet_b0"

    # The bottom-up EfficientNet will produce ["res2","res3","res4","res5"].
    # FPN will build ["p2","p3","p4","p5"] on top of those.
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.MODEL.FPN.FUSE_TYPE = "sum"

    # RPN and ROI heads should consume the FPN outputs named ["p2","p3","p4","p5"].
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # Number of classes in your dataset. For COCO, it's 80.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    # Build the model
    model = build_model(cfg)
    model.eval()

    # Create a dummy input (3, 224, 224)
    dummy_image = torch.randn(3, 224, 224)

    # Detectron2 expects a list[Dict], each containing an "image" key
    inputs = [
        {
            "image": dummy_image,
            "height": 224,
            "width": 224,
        }
    ]

    with torch.no_grad():
        outputs = model(inputs)

    print("Model output:", outputs)

if __name__ == "__main__":
    quick_test_efficientnet()
