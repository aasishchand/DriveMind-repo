"""Train a candidate collision-risk shield from recorded CARLA experiences."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from etl.config import REPO_ROOT
from etl.policy_learning.risk_model import CollisionRiskModel, FEATURES


def _float(row: dict, key: str, default: float) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def load_examples(path: Path, lookahead_s: float = 2.0):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (r.get("run_id", ""), _float(r, "timestep_s", 0.0)))
    contacts: dict[str, list[float]] = {}
    collision_runs, clear_runs = set(), set()
    for row in rows:
        run = row.get("run_id", "")
        outcome = row.get("outcome", "")
        if outcome == "collision" or row.get("actual_collision", "").lower() == "true":
            contacts.setdefault(run, []).append(_float(row, "timestep_s", 0.0))
            collision_runs.add(run)
        elif outcome in {"clear", "near_miss", "near_miss_context", "collision_context"}:
            clear_runs.add(run)
    clear_runs.difference_update(collision_runs)
    examples = []
    for row in rows:
        run = row.get("run_id", "")
        t = _float(row, "timestep_s", 0.0)
        incident = row.get("outcome") in {"collision", "near_miss", "collision_context"}
        incident = incident or any(0 <= contact-t <= lookahead_s for contact in contacts.get(run, []))
        if row.get("outcome") == "post_collision":
            continue
        values = [_float(row, "ego_speed_mps", 0.0),
                  _float(row, "nearest_distance_m", 80.0),
                  _float(row, "relative_speed_mps", 0.0),
                  _float(row, "ttc_s", 30.0), _float(row, "risk_score", 0.0),
                  _float(row, "emergency_detection_count", 0.0)]
        examples.append((values, int(incident), run))
    return examples, collision_runs, clear_runs


def fit(examples, *, epochs: int = 1200, learning_rate: float = 0.04):
    n_features = len(FEATURES)
    mean = [sum(item[0][j] for item in examples)/len(examples) for j in range(n_features)]
    scale = []
    for j in range(n_features):
        variance = sum((item[0][j]-mean[j])**2 for item in examples)/len(examples)
        scale.append(max(math.sqrt(variance), 1e-6))
    x = [[1.0] + [(values[j]-mean[j])/scale[j] for j in range(n_features)]
         for values, _, _ in examples]
    y = [label for _, label, _ in examples]
    positives, negatives = sum(y), len(y)-sum(y)
    if not positives or not negatives:
        raise ValueError("training requires both incident and clear examples")
    weights = [0.0] * (n_features+1)
    pos_weight = len(y)/(2*positives)
    neg_weight = len(y)/(2*negatives)
    for _ in range(epochs):
        grad = [0.0] * len(weights)
        for row, label in zip(x, y):
            z = max(-30.0, min(30.0, sum(a*b for a, b in zip(weights, row))))
            p = 1/(1+math.exp(-z))
            error = (p-label) * (pos_weight if label else neg_weight)
            for j, value in enumerate(row):
                grad[j] += error*value
        for j in range(len(weights)):
            grad[j] = grad[j]/len(y) + (0.01*weights[j] if j else 0.0)
            weights[j] -= learning_rate*grad[j]
    return CollisionRiskModel(mean, scale, weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path,
                        default=REPO_ROOT/"data"/"policy_learning"/"experiences.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT/"models"/"risk_shield_candidate.json")
    parser.add_argument("--min-collision-runs", type=int, default=3)
    parser.add_argument("--min-clear-runs", type=int, default=3)
    args = parser.parse_args()
    examples, collision_runs, clear_runs = load_examples(args.experiences)
    if len(collision_runs) < args.min_collision_runs or len(clear_runs) < args.min_clear_runs:
        raise SystemExit("Not enough independent data to train safely: "
                         f"{len(collision_runs)} collision runs, {len(clear_runs)} clear runs; "
                         f"need at least {args.min_collision_runs} and {args.min_clear_runs}. "
                         "Collect more varied CARLA runs first.")
    model = fit(examples)
    model.save(args.output, metadata={"examples": len(examples),
        "collision_runs": len(collision_runs), "clear_runs": len(clear_runs),
        "positive_examples": sum(label for _, label, _ in examples),
        "training_source": str(args.experiences)})
    print(json.dumps({"candidate": str(args.output), "examples": len(examples),
        "collision_runs": len(collision_runs), "clear_runs": len(clear_runs),
        "positive_examples": sum(label for _, label, _ in examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
