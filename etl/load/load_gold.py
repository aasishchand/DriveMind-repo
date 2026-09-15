"""Build Gold feature tables from Silver telemetry.

Two feature products, both derived only from CarlaTelemetrySchema rows:

  1. trajectory_windows - fixed-length past/future position windows per vehicle,
     the training samples for trajectory / emergency-vehicle prediction.

  2. risk_features - per (timestamp) ego-centric features for the risk engine:
     nearest-vehicle distance, closing speed, time-to-collision, neighbour
     counts, emergency-vehicle proximity, ego kinematics.

A deterministic, scenario-level train/val/test split is assigned so that all
ticks from one scenario stay in the same split (no leakage across the window
boundary).

STUB STATUS: the feature maths is a reasonable first cut. The risk-engine team
owns the final feature list and TTC definition; treat columns here as a
starting contract to iterate on.

Run:
    python -m etl.load.load_gold --silver-glob "data/silver/carla_telemetry/*.parquet"
"""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from etl.config import CARLA_DOMAIN, GOLD_DIR, SPLIT_RATIOS, domain_dir, get_logger
from etl.db.manifest import (
    append_manifest_row,
    build_manifest_row,
    cli_push_manifest_row,
    row_to_sql,
)

log = get_logger(__name__)

PAST_TICKS = 20
FUTURE_TICKS = 30
TTC_EPS = 1e-3
NEIGHBOUR_RADII_M = (10.0, 30.0)
EMERGENCY_RADIUS_M = 50.0


def assign_split(scenario: str) -> str:
    """Stable split from a hash of the scenario name (same name -> same split)."""
    bucket = int(hashlib.md5(scenario.encode()).hexdigest(), 16) % 1000 / 1000.0
    if bucket < SPLIT_RATIOS["train"]:
        return "train"
    if bucket < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def build_trajectory_windows(
    df: pd.DataFrame,
    scenario: str,
    *,
    past: int = PAST_TICKS,
    future: int = FUTURE_TICKS,
) -> pd.DataFrame:
    """One row per (vehicle, anchor tick) with past/future x,y arrays."""
    rows: list[dict] = []
    for vid, sub in df.sort_values("timestamp").groupby("vehicle_id"):
        sub = sub.reset_index(drop=True)
        n = len(sub)
        for t0 in range(past, n - future):
            hist = sub.iloc[t0 - past : t0]
            fut = sub.iloc[t0 : t0 + future]
            rows.append(
                dict(
                    scenario=scenario,
                    vehicle_id=vid,
                    vehicle_type=sub.at[t0, "vehicle_type"],
                    is_emergency=bool(sub.at[t0, "is_emergency"]),
                    anchor_timestamp=float(sub.at[t0, "timestamp"]),
                    past_x=hist["x"].to_list(),
                    past_y=hist["y"].to_list(),
                    past_speed=hist["speed"].to_list(),
                    past_heading=hist["heading"].to_list(),
                    future_x=fut["x"].to_list(),
                    future_y=fut["y"].to_list(),
                    split=assign_split(scenario),
                )
            )
    return pd.DataFrame(rows)


def build_risk_features(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Per-timestamp ego-centric risk features."""
    rows: list[dict] = []
    for t, sub in df.groupby("timestamp"):
        ego = sub[sub["vehicle_id"] == "ego"]
        others = sub[sub["vehicle_id"] != "ego"]
        if ego.empty:
            continue
        ego = ego.iloc[0]

        if others.empty:
            nearest_dist = np.nan
            nearest_closing = np.nan
            ttc = np.nan
            counts = {r: 0 for r in NEIGHBOUR_RADII_M}
            emergency_near = False
        else:
            idx = others["distance_to_ego"].astype(float).idxmin()
            nearest = others.loc[idx]
            nearest_dist = float(nearest["distance_to_ego"])
            nearest_closing = float(nearest["relative_speed"])
            ttc = nearest_dist / max(nearest_closing, TTC_EPS) if nearest_closing > 0 else np.inf
            counts = {
                r: int((others["distance_to_ego"].astype(float) <= r).sum())
                for r in NEIGHBOUR_RADII_M
            }
            emergency_near = bool(
                (
                    others["is_emergency"]
                    & (others["distance_to_ego"].astype(float) <= EMERGENCY_RADIUS_M)
                ).any()
            )

        rows.append(
            dict(
                scenario=scenario,
                timestamp=float(t),
                ego_speed=float(ego["speed"]),
                ego_acceleration=float(ego["acceleration"]) if pd.notna(ego["acceleration"]) else np.nan,
                ego_heading=float(ego["heading"]),
                nearest_vehicle_dist=nearest_dist,
                nearest_closing_speed=nearest_closing,
                time_to_collision=ttc,
                n_within_10m=counts[10.0],
                n_within_30m=counts[30.0],
                emergency_vehicle_within_50m=emergency_near,
                split=assign_split(scenario),
            )
        )
    return pd.DataFrame(rows)


def load_gold(silver_files: list[Path]) -> dict[str, tuple[Path, dict]]:
    traj_parts, risk_parts = [], []
    for path in silver_files:
        scenario = Path(path).stem
        sdf = pd.read_parquet(path)
        traj_parts.append(build_trajectory_windows(sdf, scenario))
        risk_parts.append(build_risk_features(sdf, scenario))
        log.info("features built for scenario %s (%d silver rows)", scenario, len(sdf))

    traj = pd.concat(traj_parts, ignore_index=True) if traj_parts else pd.DataFrame()
    risk = pd.concat(risk_parts, ignore_index=True) if risk_parts else pd.DataFrame()

    out_dir = domain_dir("gold", CARLA_DOMAIN)
    traj_out = out_dir / "trajectory_windows.parquet"
    risk_out = out_dir / "risk_features.parquet"
    traj.to_parquet(traj_out, engine="pyarrow", index=False)
    risk.to_parquet(risk_out, engine="pyarrow", index=False)

    result: dict[str, tuple[Path, dict]] = {}
    for out, name in ((traj_out, "trajectory_windows"), (risk_out, "risk_features")):
        row = build_manifest_row(
            out,
            layer="gold",
            domain=CARLA_DOMAIN,
            source_variant=name,
            schema_name=f"gold_{name}",
            validation_status="skipped",
        )
        append_manifest_row(row)
        result[name] = (out, row)

    # split summary for a quick sanity check
    if not risk.empty:
        log.info("risk_features split counts:\n%s", risk["split"].value_counts().to_string())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silver-glob",
        default=str(GOLD_DIR.parent / "silver" / "carla_telemetry" / "*.parquet"),
        help="glob for Silver telemetry parquet files (one file == one scenario)",
    )
    parser.add_argument(
        "--push", action="store_true", help="upsert the manifest rows into PostgreSQL"
    )
    parser.add_argument(
        "--emit-sql", action="store_true", help="print the manifest INSERTs instead of running them"
    )
    args = parser.parse_args(argv)

    files = [Path(p) for p in sorted(glob.glob(args.silver_glob))]
    if not files:
        parser.error(f"no Silver files matched: {args.silver_glob!r}")

    outputs = load_gold(files)
    for name, (path, row) in outputs.items():
        n = len(pd.read_parquet(path))
        print(f"[load_gold] {name}: {n} rows -> {path}")
        if args.emit_sql:
            print(row_to_sql(row))
        if args.push:
            manifest_id = cli_push_manifest_row(row)
            if manifest_id:
                print(f"[load_gold] data_manifest upserted: {name} -> {manifest_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
