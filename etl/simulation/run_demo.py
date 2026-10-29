"""Generate a headless smoke-run log and matched planner evaluation reports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from etl.config import REPO_ROOT
from etl.simulation.core import (DrivingSimulation, SCENARIOS, telemetry_dict,
                                  run_experiments)


def run(duration: float = 25.0, scenario: str = SCENARIOS[0], mode: str = "DRIVEMIND") -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    telemetry_dir = REPO_ROOT / "data" / "simulation" / "telemetry"
    evaluation_dir = REPO_ROOT / "data" / "simulation" / "evaluation"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    sim = DrivingSimulation(scenario, mode)
    for _ in range(round(duration / 0.1)):
        sim.step(0.1)
    log_path = telemetry_dir / f"headless_{stamp}.csv"
    rows = [telemetry_dict(row) for row in sim.telemetry]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reports = run_experiments(duration)
    csv_path, json_path = write_evaluations(reports, stamp)
    return log_path, csv_path, json_path


def write_evaluations(reports: list[dict], stamp: str | None = None) -> tuple[Path, Path]:
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    evaluation_dir = REPO_ROOT / "data" / "simulation" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / f"planner_comparison_{stamp}.csv"
    json_path = evaluation_dir / f"planner_comparison_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)
    json_path.write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=25.0, help="simulated seconds per run")
    parser.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument("--mode", choices=("BASELINE", "RISK-AWARE", "DRIVEMIND"), default="DRIVEMIND")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    log_path, csv_path, json_path = run(args.duration, args.scenario, args.mode)
    print(f"simulation telemetry: {log_path}")
    print(f"planner comparison CSV: {csv_path}")
    print(f"planner comparison JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
