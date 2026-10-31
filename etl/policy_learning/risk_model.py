"""Small dependency-free collision-risk model used as a braking safety shield."""
from __future__ import annotations

import json
import math
from pathlib import Path


FEATURES = ("ego_speed_mps", "nearest_distance_m", "relative_speed_mps",
            "ttc_s", "risk_score", "emergency_detection_count")
MODEL_VERSION = 1


def _finite(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class CollisionRiskModel:
    def __init__(self, mean: list[float], scale: list[float], weights: list[float],
                 threshold: float = 0.55):
        self.mean, self.scale, self.weights = mean, scale, weights
        self.threshold = threshold

    def probability(self, features: dict) -> float:
        values = [_finite(features.get(name), 30.0 if name == "ttc_s" else 0.0)
                  for name in FEATURES]
        z = self.weights[0]
        for i, value in enumerate(values):
            z += self.weights[i+1] * ((value-self.mean[i]) / self.scale[i])
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_version") != MODEL_VERSION or payload.get("features") != list(FEATURES):
            raise ValueError(f"unsupported risk model format: {path}")
        return cls(payload["mean"], payload["scale"], payload["weights"],
                   float(payload.get("threshold", 0.55)))

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_version": MODEL_VERSION, "model_type": "logistic_collision_risk",
                   "features": list(FEATURES), "mean": self.mean, "scale": self.scale,
                   "weights": self.weights, "threshold": self.threshold,
                   "metadata": metadata or {}}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_active_model(repo_root: Path) -> CollisionRiskModel | None:
    path = repo_root / "models" / "risk_shield.json"
    return CollisionRiskModel.load(path) if path.exists() else None
