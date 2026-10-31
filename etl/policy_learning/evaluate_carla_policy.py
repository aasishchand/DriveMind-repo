"""Paired CARLA evaluation; promote a candidate only after it passes safety gates."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from etl.config import REPO_ROOT
from etl.policy_learning.risk_model import CollisionRiskModel
from etl.simulation.carla_demo import CAMERA_SCENARIOS, run
from etl.simulation.core import SCENARIOS


def evaluate(candidate_path: Path, scenario: str, episodes: int, duration: float,
             seed: int, host: str, port: int) -> dict:
    candidate = CollisionRiskModel.load(candidate_path)
    pairs = []
    for offset in range(episodes):
        episode_seed = seed + offset
        print(f"[policy evaluation] episode {offset+1}/{episodes}, seed={episode_seed}: baseline")
        _, _, baseline = run(host, port, duration, scenario, "DRIVEMIND", episode_seed,
                             camera_preview=False, risk_model=False)
        print(f"[policy evaluation] episode {offset+1}/{episodes}, seed={episode_seed}: candidate")
        _, _, learned = run(host, port, duration, scenario, "DRIVEMIND", episode_seed,
                            camera_preview=False, risk_model=candidate)
        pairs.append({"seed": episode_seed, "baseline": baseline, "candidate": learned})

    no_safety_regression = all(
        pair["candidate"]["collision_count"] <= pair["baseline"]["collision_count"]
        and pair["candidate"]["near_miss_count"] <= pair["baseline"]["near_miss_count"]
        and pair["candidate"]["path_length_m"] >= 0.8*pair["baseline"]["path_length_m"]
        for pair in pairs)
    baseline_collisions = sum(p["baseline"]["collision_count"] for p in pairs)
    candidate_collisions = sum(p["candidate"]["collision_count"] for p in pairs)
    baseline_near = sum(p["baseline"]["near_miss_count"] for p in pairs)
    candidate_near = sum(p["candidate"]["near_miss_count"] for p in pairs)
    baseline_risk = sum(p["baseline"]["average_risk"] for p in pairs)/episodes
    candidate_risk = sum(p["candidate"]["average_risk"] for p in pairs)/episodes
    improved = (candidate_collisions < baseline_collisions or candidate_near < baseline_near
                or candidate_risk <= baseline_risk-3.0)
    accepted = no_safety_regression and improved
    active_path = REPO_ROOT/"models"/"risk_shield.json"
    if accepted:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, active_path)
    report = {"scenario": scenario, "episodes": episodes, "duration_s": duration,
              "accepted": accepted, "promoted_model": str(active_path) if accepted else None,
              "baseline_totals": {"collisions": baseline_collisions, "near_misses": baseline_near,
                                  "average_risk": baseline_risk},
              "candidate_totals": {"collisions": candidate_collisions, "near_misses": candidate_near,
                                   "average_risk": candidate_risk},
              "no_safety_regression": no_safety_regression, "improved": improved,
              "pairs": pairs}
    report_path = REPO_ROOT/"data"/"policy_learning"/f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path,
                        default=REPO_ROOT/"models"/"risk_shield_candidate.json")
    parser.add_argument("--scenario", choices=(*SCENARIOS, *CAMERA_SCENARIOS),
                        default="Dense traffic lane changes")
    parser.add_argument("--episodes", type=int, default=5,
                        help="paired seeds; at least five are required for promotion")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    if args.episodes < 5 or args.duration <= 0:
        parser.error("promotion requires at least 5 paired episodes and a positive duration")
    report = evaluate(args.candidate, args.scenario, args.episodes, args.duration,
                      args.seed, args.host, args.port)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
