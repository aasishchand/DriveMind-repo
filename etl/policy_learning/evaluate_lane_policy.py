"""Evaluate lane-choice candidate against current planner over paired CARLA seeds."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from etl.config import REPO_ROOT
from etl.policy_learning.lane_model import LaneChoiceModel
from etl.simulation.carla_demo import CAMERA_SCENARIOS, run
from etl.simulation.core import SCENARIOS


def evaluate(candidate_path: Path, scenario: str, episodes: int, duration: float,
             seed: int, host: str, port: int) -> dict:
    candidate = LaneChoiceModel.load(candidate_path)
    pairs = []
    for offset in range(episodes):
        episode_seed = seed+offset
        print(f"[lane policy evaluation] episode {offset+1}/{episodes}, seed={episode_seed}: current planner")
        _, _, baseline = run(host, port, duration, scenario, "DRIVEMIND", episode_seed,
                             camera_preview=False, lane_model=False)
        print(f"[lane policy evaluation] episode {offset+1}/{episodes}, seed={episode_seed}: candidate")
        _, _, learned = run(host, port, duration, scenario, "DRIVEMIND", episode_seed,
                            camera_preview=False, lane_model=candidate)
        pairs.append({"seed": episode_seed, "baseline": baseline, "candidate": learned})

    safe = all(p["candidate"]["collision_count"] <= p["baseline"]["collision_count"]
               and p["candidate"]["near_miss_count"] <= p["baseline"]["near_miss_count"]
               and p["candidate"]["path_length_m"] >= 0.8*p["baseline"]["path_length_m"]
               for p in pairs)
    totals = {}
    for name in ("baseline", "candidate"):
        totals[name] = {
            "collisions": sum(p[name]["collision_count"] for p in pairs),
            "near_misses": sum(p[name]["near_miss_count"] for p in pairs),
            "average_risk": sum(p[name]["average_risk"] for p in pairs)/episodes,
        }
    old, new = totals["baseline"], totals["candidate"]
    improvement = (new["collisions"] < old["collisions"]
                   or new["near_misses"] < old["near_misses"]
                   or new["average_risk"] <= old["average_risk"]-3.0)
    accepted = safe and improvement
    active_path = REPO_ROOT/"models"/"lane_policy.json"
    if accepted:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, active_path)
    report_path = REPO_ROOT/"data"/"policy_learning"/f"lane_evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"scenario": scenario, "episodes": episodes, "duration_s": duration,
              "accepted": accepted, "promoted_model": str(active_path) if accepted else None,
              "baseline": old, "candidate": new, "no_safety_regression": safe,
              "improved": improvement, "pairs": pairs, "report_path": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path,
                        default=REPO_ROOT/"models"/"lane_policy_candidate.json")
    parser.add_argument("--scenario", choices=(*SCENARIOS, *CAMERA_SCENARIOS),
                        default="Dense traffic lane changes")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=161)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    if args.episodes < 5 or args.duration <= 0:
        parser.error("promotion requires at least 5 paired episodes and positive duration")
    result = evaluate(args.candidate, args.scenario, args.episodes, args.duration,
                      args.seed, args.host, args.port)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
