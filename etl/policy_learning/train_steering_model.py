"""Fit and validate an expert steering regression model by held-out run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from etl.config import REPO_ROOT
from etl.policy_learning.steering_model import FEATURES, SteeringModel


def load_rows(directory):
    result = []
    for path in sorted(directory.glob("steering_expert_*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    values = [float(row[name]) for name in FEATURES]
                    target = float(row["expert_steer"])
                    if all(math.isfinite(value) for value in (*values, target)) and not row.get("collision") == "True":
                        result.append((values, max(-1.0, min(1.0, target)), row["run_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return result


def fit(rows, ridge=1.0):
    values = np.asarray([row[0] for row in rows], dtype=np.float64)
    targets = np.asarray([row[1] for row in rows], dtype=np.float64)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (values-mean)/scale
    x = np.column_stack((np.ones(len(z)), z, z[:, 0]*z[:, 1], z[:, 0]*z[:, 2], z[:, 1]*z[:, 2]))
    penalty = np.eye(x.shape[1])*ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T@x+penalty, x.T@targets)
    return SteeringModel(mean.tolist(), scale.tolist(), weights.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT/"data"/"policy_learning")
    parser.add_argument("--output", type=Path, default=REPO_ROOT/"models"/"steering_candidate.json")
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    rows = load_rows(args.data_dir)
    runs = sorted({row[2] for row in rows})
    if len(runs) < args.min_runs:
        raise SystemExit(f"Need at least {args.min_runs} independent steering captures; found {len(runs)}")
    validation_run = runs[-1]
    train_rows = [row for row in rows if row[2] != validation_run]
    validation = [row for row in rows if row[2] == validation_run]
    if len(train_rows) < 100 or len(validation) < 100:
        raise SystemExit("Need at least 100 samples in training and held-out capture")
    model = fit(train_rows, args.ridge)
    prediction = np.asarray([model.predict_features(dict(zip(FEATURES, row[0]))) for row in validation])
    target = np.asarray([row[1] for row in validation])
    mae = float(np.mean(np.abs(prediction-target)))
    rmse = float(np.sqrt(np.mean((prediction-target)**2)))
    baseline_mae = float(np.mean(np.abs(target-np.mean([row[1] for row in train_rows]))))
    final_model = fit(rows, args.ridge)
    metadata = {"runs": len(runs), "samples": len(rows), "validation_run": validation_run,
                "validation_mae": mae, "validation_rmse": rmse,
                "validation_mean_baseline_mae": baseline_mae, "ridge": args.ridge}
    final_model.save(args.output, metadata)
    print(json.dumps({"candidate": str(args.output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
