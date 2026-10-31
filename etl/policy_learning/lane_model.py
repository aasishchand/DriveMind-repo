"""Behavior-cloned adjacent-lane decision model."""
from __future__ import annotations

import json
import math
from pathlib import Path


FEATURES = ("ego_speed_mps", "current_front_gap_m", "current_front_closing_mps",
            "adjacent_front_gap_m", "adjacent_front_closing_mps",
            "adjacent_rear_gap_m", "adjacent_rear_closing_mps")
MODEL_VERSION = 1


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


class LaneChoiceModel:
    def __init__(self, mean, scale, weights, threshold=0.5):
        self.mean, self.scale, self.weights = mean, scale, weights
        self.threshold = threshold

    def change_probability(self, features: dict) -> float:
        values = [_number(features.get(name), 120.0 if "gap" in name else 0.0)
                  for name in FEATURES]
        z = self.weights[0] + sum(
            self.weights[i+1] * ((value-self.mean[i])/self.scale[i])
            for i, value in enumerate(values))
        z = max(-30.0, min(30.0, z))
        return 1.0/(1.0+math.exp(-z))

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_version") != MODEL_VERSION or payload.get("features") != list(FEATURES):
            raise ValueError(f"unsupported lane policy format: {path}")
        return cls(payload["mean"], payload["scale"], payload["weights"],
                   float(payload.get("threshold", 0.5)))

    def save(self, path: Path, *, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_version": MODEL_VERSION, "model_type": "binary_lane_choice",
                   "features": list(FEATURES), "mean": self.mean, "scale": self.scale,
                   "weights": self.weights, "threshold": self.threshold,
                   "metadata": metadata or {}}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_active_lane_model(repo_root: Path):
    path = repo_root/"models"/"lane_policy.json"
    return LaneChoiceModel.load(path) if path.exists() else None
