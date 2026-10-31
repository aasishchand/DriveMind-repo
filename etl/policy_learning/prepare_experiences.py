"""Normalize DriveMind CARLA controller logs into policy-learning examples."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from etl.config import REPO_ROOT


FIELDS = (
    "schema_version",
    "source",
    "run_id",
    "scenario",
    "timestep_s",
    "supervision_type",
    "ego_speed_mps",
    "ego_lane",
    "target_speed_mps",
    "nearest_distance_m",
    "relative_speed_mps",
    "ttc_s",
    "risk_score",
    "emergency_status",
    "emergency_detection_count",
    "nearest_emergency_detection_m",
    "max_detection_confidence",
    "action",
    "steering",
    "throttle",
    "brake",
    "near_miss",
    "actual_collision",
    "collision_with_actor_type",
    "outcome",
    "detections_json",
)


def _number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _detections(value: str | None) -> list[dict]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def normalize_row(row: dict[str, str], run_id: str) -> dict[str, object]:
    detections = _detections(row.get("emergency_detections"))
    emergency = [d for d in detections if d.get("class") in {
        "ambulance", "fire_truck", "police_vehicle"
    }]
    distance_values = [value for d in emergency
                       if (value := _number(str(d.get("distance_m", "")))) is not None]
    confidence_values = [value for d in detections
                         if (value := _number(str(d.get("confidence", "")))) is not None]
    actual_collision = _bool(row.get("actual_collision")) or (
        (_number(row.get("carla_collision_events")) or 0) > 0
    )
    near_miss = _bool(row.get("near_miss"))
    outcome = "collision" if actual_collision else "near_miss" if near_miss else "clear"
    return {
        "schema_version": "1",
        "source": row.get("source") or "drivemind_carla",
        "run_id": run_id,
        "scenario": row.get("scenario", ""),
        "timestep_s": _number(row.get("sim_time")),
        "supervision_type": "action_and_outcome",
        "ego_speed_mps": _number(row.get("ego_speed")),
        "ego_lane": row.get("ego_lane", ""),
        "target_speed_mps": _number(row.get("target_speed")),
        "nearest_distance_m": _number(row.get("nearest_distance")),
        "relative_speed_mps": _number(row.get("relative_speed")),
        "ttc_s": _number(row.get("ttc")),
        "risk_score": _number(row.get("risk_score")),
        "emergency_status": row.get("emergency_status", ""),
        "emergency_detection_count": int(_number(row.get("emergency_detection_count")) or 0),
        "nearest_emergency_detection_m": min(distance_values) if distance_values else None,
        "max_detection_confidence": max(confidence_values) if confidence_values else None,
        "action": row.get("action", ""),
        "steering": _number(row.get("steering")),
        "throttle": _number(row.get("throttle")),
        "brake": _number(row.get("brake")),
        "near_miss": near_miss,
        "actual_collision": actual_collision,
        "collision_with_actor_type": row.get("collision_with_actor_type", ""),
        "outcome": outcome,
        "detections_json": json.dumps(detections, separators=(",", ":")),
    }


def build(telemetry_dir: Path, output: Path,
          normalized_inputs: list[Path] | None = None) -> tuple[int, dict[str, int]]:
    examples: list[dict[str, object]] = []
    for path in sorted(telemetry_dir.glob("carla_control_*.csv")):
        run_id = path.stem.removeprefix("carla_control_")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            examples.extend(normalize_row(row, run_id) for row in csv.DictReader(stream))

    for path in normalized_inputs or []:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = set(FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing canonical columns: {', '.join(sorted(missing))}")
            for row in reader:
                if row.get("schema_version") != "1":
                    raise ValueError(f"{path} contains an unsupported schema_version")
                item: dict[str, object] = {field: row.get(field, "") for field in FIELDS}
                item["actual_collision"] = _bool(row.get("actual_collision"))
                item["near_miss"] = _bool(row.get("near_miss"))
                examples.append(item)

    examples.sort(key=lambda item: (str(item["source"]), str(item["run_id"]),
                                    _number(str(item["timestep_s"])) or 0.0))

    # A sustained bumper contact can be reported on many consecutive ticks.
    # Keep the first tick as the negative action outcome; later contact frames
    # are context, not additional independent crash examples or policy labels.
    collision_seen: set[str] = set()
    near_miss_active: dict[str, bool] = {}
    for example in examples:
        run_id = str(example["run_id"])
        if _bool(example["actual_collision"]):
            if run_id in collision_seen:
                example["outcome"] = "post_collision"
                example["supervision_type"] = "post_collision_context"
                example["action"] = ""
                example["steering"] = None
                example["throttle"] = None
                example["brake"] = None
            else:
                collision_seen.add(run_id)
            continue

        is_near_miss = _bool(example["near_miss"])
        if is_near_miss and near_miss_active.get(run_id, False):
            example["outcome"] = "near_miss_context"
            example["supervision_type"] = "near_miss_context"
            example["action"] = ""
            example["steering"] = None
            example["throttle"] = None
            example["brake"] = None
        elif is_near_miss:
            example["outcome"] = "near_miss"
        elif example["outcome"] in {"", "clear", "near_miss_context", "post_collision"}:
            example["outcome"] = "clear"
        near_miss_active[run_id] = is_near_miss

    # Mark the two seconds of decisions leading up to a physical collision as
    # risk examples too. Keep their actions: they are the decisions the risk
    # model should learn to flag before contact.
    collision_times: dict[str, list[float]] = {}
    for example in examples:
        if _bool(example["actual_collision"]):
            timestamp = _number(str(example["timestep_s"]))
            if timestamp is not None:
                collision_times.setdefault(str(example["run_id"]), []).append(timestamp)
    for example in examples:
        run_id = str(example["run_id"])
        timestamp = _number(str(example["timestep_s"]))
        if timestamp is None or _bool(example["actual_collision"]):
            continue
        if any(0.0 <= contact-timestamp <= 2.0 for contact in collision_times.get(run_id, [])):
            example["outcome"] = "collision_context"
            example["supervision_type"] = "collision_context"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(examples)

    outcomes: dict[str, int] = {}
    for example in examples:
        key = str(example["outcome"])
        outcomes[key] = outcomes.get(key, 0) + 1
    return len(examples), outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path,
                        default=REPO_ROOT / "data" / "simulation" / "telemetry")
    parser.add_argument("--normalized-input", type=Path, action="append", default=[],
                        help="additional CSV already converted to schema v1; repeat as needed")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "data" / "policy_learning" / "experiences.csv")
    args = parser.parse_args()
    count, outcomes = build(args.telemetry_dir, args.output, args.normalized_input)
    print(f"Wrote {count} experience rows to {args.output}")
    print(f"Outcome counts: {json.dumps(outcomes, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
