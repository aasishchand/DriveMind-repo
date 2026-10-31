"""Small ridge-regression steering model and shared CARLA road features."""
from __future__ import annotations

import json
import math
from pathlib import Path


FEATURES = ("speed_mps", "lateral_error_m", "heading_error_rad",
            "lookahead_error_rad", "curvature_rad_per_m")


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def steering_features(vehicle, target_waypoint, carla_map) -> dict[str, float]:
    """Describe how the ego pose differs from the target lane centerline."""
    transform = vehicle.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    lane_transform = target_waypoint.transform
    lane_loc = lane_transform.location
    lane_yaw = math.radians(lane_transform.rotation.yaw)
    # CARLA yaw=0 points along +x; the corresponding right vector is +y.
    dx, dy = loc.x-lane_loc.x, loc.y-lane_loc.y
    lateral_error = dx * -math.sin(lane_yaw) + dy * math.cos(lane_yaw)
    heading_error = _wrap(lane_yaw-yaw)

    next_points = target_waypoint.next(8.0)
    ahead = min(next_points, key=lambda wp: abs(_wrap(
        math.radians(wp.transform.rotation.yaw)-lane_yaw))) if next_points else None
    if ahead:
        ahead_loc = ahead.transform.location
        desired_yaw = math.atan2(ahead_loc.y-loc.y, ahead_loc.x-loc.x)
        lookahead_error = _wrap(desired_yaw-yaw)
        curvature = _wrap(math.radians(ahead.transform.rotation.yaw)-lane_yaw) / 8.0
    else:
        lookahead_error, curvature = heading_error, 0.0
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
    values = (speed, lateral_error, heading_error, lookahead_error, curvature)
    return dict(zip(FEATURES, values))


class SteeringModel:
    def __init__(self, mean, scale, weights, metadata=None):
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.metadata = metadata or {}

    def predict_features(self, features: dict[str, float]) -> float:
        values = [float(features[name]) for name in FEATURES]
        normalized = [(v-m)/s for v, m, s in zip(values, self.mean, self.scale)]
        # Include modest speed/error interactions for CARLA's speed-dependent
        # steering response while keeping inference dependency-free.
        x = [1.0, *normalized,
             normalized[0]*normalized[1], normalized[0]*normalized[2],
             normalized[1]*normalized[2]]
        return max(-1.0, min(1.0, sum(w*v for w, v in zip(self.weights, x))))

    def predict(self, vehicle, target_waypoint, carla_map) -> float:
        return self.predict_features(steering_features(vehicle, target_waypoint, carla_map))

    def save(self, path: Path, metadata=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_version": 1, "model_type": "carla_expert_ridge_steering",
                   "features": list(FEATURES), "mean": self.mean, "scale": self.scale,
                   "weights": self.weights, "metadata": metadata or self.metadata}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_version") != 1 or payload.get("features") != list(FEATURES):
            raise ValueError(f"unsupported steering model format: {path}")
        return cls(payload["mean"], payload["scale"], payload["weights"],
                   payload.get("metadata", {}))
