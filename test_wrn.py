import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import detectron2.modeling.backbone.wide_resnet

def quick_test_wide_resnet():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_wide_resnet_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    model = build_model(cfg)
    model.eval()

    dummy_image = torch.randn(3, 224, 224)

    #   "image": (3, H, W)   # single image
    #   "height": <height of raw image>
    #   "width":  <width of raw image>
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
    quick_test_wide_resnet()
