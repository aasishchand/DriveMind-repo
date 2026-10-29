"""Camera-frame inference for the separately trained emergency detector."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .labels import ID_TO_CLASS
from .model import MODEL_NAME, build_model

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights" / "emergency_detector.pt"


class EmergencyDetector:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, threshold: float = .5,
                 device: str | None = None):
        if not 0 < threshold < 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        self.checkpoint, self.threshold = Path(checkpoint), threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.status, self.error = "UNTRAINED", None
        self.model = None
        self.latency_ms = 0.0
        if self.checkpoint.is_file():
            try:
                payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
                self.model = build_model()
                self.model.load_state_dict(payload["model_state_dict"])
                self.model.to(self.device).eval()
                self.status = "READY"
            except Exception as exc:
                self.status, self.error = "ERROR", f"{type(exc).__name__}: {exc}"

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Accept RGB uint8 HxWx3; return only real model predictions."""
        if self.model is None:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera frame must have HxWx3 RGB shape")
        started = time.perf_counter()
        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().to(self.device) / 255
        try:
            with torch.inference_mode():
                result = self.model([image])[0]
        except Exception as exc:
            self.status, self.error, self.model = "ERROR", f"{type(exc).__name__}: {exc}", None
            return []
        self.latency_ms = (time.perf_counter()-started)*1000
        detections = []
        for box, label, score in zip(result["boxes"].cpu(), result["labels"].cpu(), result["scores"].cpu()):
            cls, confidence = ID_TO_CLASS.get(int(label)), float(score)
            if cls and confidence >= self.threshold:
                detections.append({"class": cls, "confidence": confidence,
                                   "bbox": [float(x) for x in box.tolist()]})
        return detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Render only model-produced boxes. Requires Pillow; input/output are RGB arrays."""
    from PIL import ImageDraw
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 80, 90), width=3)
        draw.text((x1, max(0, y1-16)), f"{item['class'].replace('_',' ').upper()} {item['confidence']:.2f}", fill=(255, 80, 90))
    return np.asarray(image)
