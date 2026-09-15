"""Extract raw telemetry from a single CARLA scenario log.

STUB STATUS
-----------
The real ``scenario_runner`` / sensor-capture output format is not wired up yet.
This stub:

  * reads a per-scenario log if one is passed (``.json`` list-of-records or
    ``.csv``) and returns it untouched, or
  * emits a tiny synthetic raw frame so the rest of the pipeline is runnable.

The *raw* column names below intentionally differ from the common schema; the
mapping to the common schema happens in ``etl/transform/clean_carla.py``.

Reference for capture patterns: KevinLADLee/carla_dataset_tools.

Run:
    python -m etl.extract.extract_carla --scenario-name demo_overtake
    python -m etl.extract.extract_carla --scenario-log path/to/log.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from etl.config import get_logger

log = get_logger(__name__)

# Raw per-tick record as we expect to receive it from the simulation capture.
RAW_COLUMNS = (
    "scenario",
    "actor_id",
    "sim_time",
    "loc_x",
    "loc_y",
    "loc_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "yaw_deg",
    "lane_id",
    "blueprint_id",
    "emergency_light_on",
)


def _synthetic_raw(scenario: str, n_ticks: int = 60, dt: float = 0.05) -> pd.DataFrame:
    """Two vehicles (ego + one approaching car) over a short window."""
    rows = []
    for k in range(n_ticks):
        t = round(k * dt, 3)
        # ego: cruising along +x at ~8 m/s
        rows.append(
            dict(
                scenario=scenario,
                actor_id="ego",
                sim_time=t,
                loc_x=8.0 * t,
                loc_y=0.0,
                loc_z=0.0,
                vel_x=8.0,
                vel_y=0.0,
                vel_z=0.0,
                yaw_deg=0.0,
                lane_id=-1,
                blueprint_id="vehicle.tesla.model3",
                emergency_light_on=False,
            )
        )
        # other: starts 40 m ahead, closing at ~4 m/s
        rows.append(
            dict(
                scenario=scenario,
                actor_id="veh_042",
                sim_time=t,
                loc_x=40.0 + 4.0 * t,
                loc_y=0.0,
                loc_z=0.0,
                vel_x=4.0,
                vel_y=0.0,
                vel_z=0.0,
                yaw_deg=0.0,
                lane_id=-1,
                blueprint_id="vehicle.carlamotors.firetruck",
                emergency_light_on=True,
            )
        )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def _read_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text())
        return pd.DataFrame.from_records(records)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported scenario-log format: {path.suffix!r}")


def extract_carla(
    scenario_name: str,
    scenario_log: Path | None = None,
) -> pd.DataFrame:
    """Return a raw telemetry DataFrame for one scenario run.

    Parameters
    ----------
    scenario_name:
        Logical name of the scenario run; also used as the Bronze file stem.
    scenario_log:
        Optional path to an existing capture. When omitted, synthetic data is
        returned so downstream stages remain runnable.
    """
    if scenario_log is not None:
        log.info("reading CARLA scenario log: %s", scenario_log)
        df = _read_log(Path(scenario_log))
        if "scenario" not in df.columns:
            df["scenario"] = scenario_name
        log.info("read %d raw rows", len(df))
        return df

    log.warning("no --scenario-log given; emitting SYNTHETIC raw telemetry (stub)")
    df = _synthetic_raw(scenario_name)
    log.info("synthetic raw rows: %d", len(df))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-name", default="demo_scenario")
    parser.add_argument(
        "--scenario-log",
        type=Path,
        default=None,
        help="path to a CARLA capture (.json / .csv / .parquet); omit for synthetic stub data",
    )
    parser.add_argument(
        "--peek", type=int, default=5, help="rows to print to stdout for a quick look"
    )
    args = parser.parse_args(argv)

    df = extract_carla(args.scenario_name, args.scenario_log)
    print(df.head(args.peek).to_string(index=False))
    print(f"\n[stub] extracted {len(df)} raw rows for scenario {args.scenario_name!r}")
    print("[stub] next: python -m etl.load.load_bronze --domain carla_telemetry ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
