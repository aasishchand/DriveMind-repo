"""Train a binary keep-lane/change-adjacent behavior-cloning model."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from etl.config import REPO_ROOT
from etl.policy_learning.lane_model import FEATURES, LaneChoiceModel


def _value(row, name, default):
    try:
        x = float(row.get(name, ""))
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def load_rows(directory: Path):
    rows = []
    for path in sorted(directory.glob("lane_expert_*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows.extend(row for row in csv.DictReader(stream)
                        if row.get("expert_action") in {"stay", "change"})
    return rows


def _vectors(rows):
    result = []
    for row in rows:
        values = [_value(row, name, 120.0 if "gap" in name else 0.0) for name in FEATURES]
        result.append((values, int(row["expert_action"] == "change"), row["run_id"]))
    return result


def fit(rows, epochs=1000, learning_rate=0.035):
    n = len(FEATURES)
    mean = [sum(x[0][j] for x in rows)/len(rows) for j in range(n)]
    scale = []
    for j in range(n):
        variance = sum((x[0][j]-mean[j])**2 for x in rows)/len(rows)
        scale.append(max(math.sqrt(variance), 1e-6))
    samples = [[1.0]+[(values[j]-mean[j])/scale[j] for j in range(n)]
               for values, _, _ in rows]
    labels = [label for _, label, _ in rows]
    positives, negatives = sum(labels), len(labels)-sum(labels)
    if positives == 0 or negatives == 0:
        raise ValueError("expert data needs both lane changes and lane keeping")
    positive_weight, negative_weight = len(labels)/(2*positives), len(labels)/(2*negatives)
    weights = [0.0]*(n+1)
    for _ in range(epochs):
        gradient = [0.0]*(n+1)
        for sample, label in zip(samples, labels):
            z = max(-30, min(30, sum(w*x for w, x in zip(weights, sample))))
            probability = 1/(1+math.exp(-z))
            error = (probability-label)*(positive_weight if label else negative_weight)
            for j, value in enumerate(sample):
                gradient[j] += error*value
        for j in range(n+1):
            gradient[j] = gradient[j]/len(labels) + (0.01*weights[j] if j else 0.0)
            weights[j] -= learning_rate*gradient[j]
    return LaneChoiceModel(mean, scale, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT/"data"/"policy_learning")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT/"models"/"lane_policy_candidate.json")
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--min-changes", type=int, default=20)
    args = parser.parse_args()
    examples = _vectors(load_rows(args.data_dir))
    run_ids = sorted({row[2] for row in examples})
    changes = sum(row[1] for row in examples)
    stays = len(examples)-changes
    if len(run_ids) < args.min_runs:
        raise SystemExit(f"Need at least {args.min_runs} distinct expert runs; found {len(run_ids)}.")
    if changes < args.min_changes or stays < 50:
        raise SystemExit(f"Need at least {args.min_changes} lane-change labels and 50 keep labels; "
                         f"found {changes} and {stays}. Capture more expert drives.")

    validation_run = next((run_id for run_id in reversed(run_ids)
                           if any(row[2] == run_id and row[1] == 1 for row in examples)
                           and any(row[2] == run_id and row[1] == 0 for row in examples)), None)
    if validation_run is None:
        raise SystemExit("No single capture run contains both lane changes and lane keeping; "
                         "collect a more varied validation drive.")
    train_rows = [row for row in examples if row[2] != validation_run]
    validation_rows = [row for row in examples if row[2] == validation_run]
    model = fit(train_rows)
    predictions = [model.change_probability(dict(zip(FEATURES, row[0]))) >= model.threshold
                   for row in validation_rows]
    labels = [bool(row[1]) for row in validation_rows]
    correct = sum(p == y for p, y in zip(predictions, labels))
    tp = sum(p and y for p, y in zip(predictions, labels))
    recall = tp/max(1, sum(labels))
    precision = tp/max(1, sum(predictions))

    # Refit on all demonstration runs after reporting held-out-run metrics.
    final_model = fit(examples)
    final_model.save(args.output, metadata={"runs": len(run_ids), "samples": len(examples),
        "change_samples": changes, "stay_samples": stays,
        "validation_run": validation_run, "validation_accuracy": correct/max(1, len(labels)),
        "validation_precision": precision, "validation_recall": recall})
    print(json.dumps({"candidate": str(args.output), "runs": len(run_ids),
        "samples": len(examples), "change_samples": changes, "stay_samples": stays,
        "validation_run": validation_run, "validation_accuracy": correct/max(1, len(labels)),
        "validation_precision": precision, "validation_recall": recall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
