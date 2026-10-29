"""Independent Faster R-CNN model; does not reuse the LSTM or another detector."""
from __future__ import annotations

NUM_CLASSES = 1 + 4  # background + canonical classes
MODEL_NAME = "DriveMind-Emergency-FasterRCNN-ResNet50-FPN-v1"


def build_model():
    """Build from random initialization; no pretrained weights are downloaded."""
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except Exception as exc:
        raise RuntimeError("Emergency detector requires a torch/torchvision pair with detection ops; "
                           "install the versions listed in requirements.txt") from exc
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=640, max_size=1024)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, NUM_CLASSES)
    return model
