# file: test_effnet_b7.py

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.modeling.backbone.efficientnet_b7_fpn import build_efficientnet_b7_fpn_backbone

def main():
    # 1) Load your custom YAML config
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/efficientnet_b7_fpn.yaml")  # adjust path if needed
    cfg.freeze()

    

    # 2) Build the entire model (Faster R-CNN + FPN + B7 backbone)
    model = build_model(cfg)
    model.eval()

    # 3) Create dummy inputs (two images, 224x224)
    images = torch.randn(2, 3, 224, 224)
    inputs = [
        {"image": images[0], "height": 224, "width": 224},
        {"image": images[1], "height": 224, "width": 224},
    ]

    # 4) Run a forward pass (inference mode)
    with torch.no_grad():
        outputs = model(inputs)

    print("\nModel outputs:")
    for i, out in enumerate(outputs):
        print(f"Image {i} predictions:\n", out)

if __name__ == "__main__":
    main()
