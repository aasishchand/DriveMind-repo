"""Clean and conform raw CARLA telemetry to the common schema (Bronze -> Silver).

Pipeline:

    1. map raw columns  -> common-schema columns
    2. derive           -> speed, acceleration, heading, distance_to_ego,
                           relative_speed
    3. data-quality gate -> the four checks Member 1 owns:
         a. missing values in required columns
         b. invalid coordinates / speeds (physical sanity bounds)
         c. broken trajectories (implausible position jumps between ticks)
         d. timestamp continuity (monotonic, roughly constant tick spacing)
    4. pandera validation against CarlaTelemetrySchema
    5. return (clean_df, quality_report)

The DQ checks currently *drop or clip* offending rows and record counts in the
report. Swap to "quarantine" (write rejects to a side table) once we decide how
strict Silver should be.

STUB STATUS: derivations and checks are real; they have only been exercised on
the synthetic extractor output so far.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from etl.config import CARLA_DOMAIN, domain_dir, get_logger
from etl.schemas.carla_telemetry import (
    CARLA_TELEMETRY_COLUMNS,
    CARLA_VEHICLE_TYPES,
    MAX_ABS_ACCEL_MPS2,
    MAX_SPEED_MPS,
    WORLD_COORD_LIMIT_M,
    validate as validate_carla,
)

log = get_logger(__name__)

REQUIRED_NON_NULL = ("vehicle_id", "timestamp", "x", "y", "z", "speed", "heading", "vehicle_type")

# Raw blueprint id substring -> coarse vehicle_type.
_BLUEPRINT_TYPE_HINTS = {
    "firetruck": "emergency",
    "ambulance": "emergency",
    "police": "emergency",
    "harley": "motorcycle",
    "yamaha": "motorcycle",
    "kawasaki": "motorcycle",
    "vespa": "motorcycle",
    "omafiets": "bicycle",
    "crossbike": "bicycle",
    "century": "bicycle",
    "diamondback": "bicycle",
    "carlacola": "truck",
    "cybertruck": "truck",
    "sprinter": "van",
    "t2": "van",
    "ambulance_van": "van",
    "bus": "bus",
}


@dataclass
class QualityReport:
    """Row-level accounting for one scenario's clean pass."""

    scenario: str
    rows_in: int = 0
    rows_out: int = 0
    null_required: int = 0
    invalid_coords: int = 0
    invalid_speed: int = 0
    accel_clipped: int = 0
    heading_normalised: int = 0
    broken_trajectory_rows: int = 0
    timestamp_gap_events: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "null_required": self.null_required,
            "invalid_coords": self.invalid_coords,
            "invalid_speed": self.invalid_speed,
            "accel_clipped": self.accel_clipped,
            "heading_normalised": self.heading_normalised,
            "broken_trajectory_rows": self.broken_trajectory_rows,
            "timestamp_gap_events": self.timestamp_gap_events,
            "notes": self.notes,
        }


def _map_vehicle_type(blueprint_id: object) -> str:
    bp = str(blueprint_id or "").lower()
    for hint, vtype in _BLUEPRINT_TYPE_HINTS.items():
        if hint in bp:
            return vtype
    if bp.startswith("vehicle."):
        return "car"
    return "unknown"


def _normalise_heading(deg: pd.Series) -> pd.Series:
    return ((deg + 180.0) % 360.0) - 180.0


def _raw_to_common(raw: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["vehicle_id"] = raw["actor_id"].astype("string")
    df["timestamp"] = pd.to_numeric(raw["sim_time"], errors="coerce")
    df["x"] = pd.to_numeric(raw["loc_x"], errors="coerce")
    df["y"] = pd.to_numeric(raw["loc_y"], errors="coerce")
    df["z"] = pd.to_numeric(raw["loc_z"], errors="coerce")

    vx = pd.to_numeric(raw.get("vel_x"), errors="coerce")
    vy = pd.to_numeric(raw.get("vel_y"), errors="coerce")
    vz = pd.to_numeric(raw.get("vel_z"), errors="coerce")
    df["speed"] = np.sqrt(vx.pow(2) + vy.pow(2) + vz.pow(2))

    heading = _normalise_heading(pd.to_numeric(raw["yaw_deg"], errors="coerce"))
    report.heading_normalised = int((heading != pd.to_numeric(raw["yaw_deg"], errors="coerce")).sum())
    df["heading"] = heading

    df["lane_id"] = pd.to_numeric(raw.get("lane_id"), errors="coerce").astype("Int64")
    df["vehicle_type"] = raw["blueprint_id"].map(_map_vehicle_type).astype("string")

    emergency_light = raw.get("emergency_light_on", False)
    if not isinstance(emergency_light, pd.Series):
        emergency_light = pd.Series(emergency_light, index=raw.index)
    df["is_emergency"] = (
        emergency_light.fillna(False).astype(bool) | df["vehicle_type"].eq("emergency")
    )

    # keep raw velocity components around for relative-speed maths, dropped later
    df["_vx"], df["_vy"], df["_vz"] = vx, vy, vz
    return df


def _derive_acceleration(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    df = df.sort_values(["vehicle_id", "timestamp"]).copy()
    grp = df.groupby("vehicle_id", sort=False)
    dt = grp["timestamp"].diff()
    dspeed = grp["speed"].diff()
    accel = dspeed / dt.replace(0.0, np.nan)
    clipped = accel.clip(-MAX_ABS_ACCEL_MPS2, MAX_ABS_ACCEL_MPS2)
    report.accel_clipped = int(((accel != clipped) & accel.notna()).sum())
    df["acceleration"] = clipped
    return df


def _derive_relative_to_ego(df: pd.DataFrame) -> pd.DataFrame:
    """distance_to_ego and relative_speed per timestamp, anchored on 'ego' rows.

    relative_speed convention: positive == the gap to ego is shrinking.
    """
    df = df.copy()
    df["distance_to_ego"] = pd.Series(np.nan, index=df.index)
    df["relative_speed"] = pd.Series(np.nan, index=df.index)

    ego = df[df["vehicle_id"] == "ego"]
    if ego.empty:
        return df

    ego_by_t = ego.set_index("timestamp")[["x", "y", "z", "_vx", "_vy", "_vz"]]
    for t, sub in df.groupby("timestamp"):
        if t not in ego_by_t.index:
            continue
        ex, ey, ez, evx, evy, evz = ego_by_t.loc[t]
        rx = sub["x"] - ex
        ry = sub["y"] - ey
        rz = sub["z"] - ez
        dist = np.sqrt(rx**2 + ry**2 + rz**2)
        df.loc[sub.index, "distance_to_ego"] = dist

        norm = dist.replace(0.0, np.nan)
        # closing speed = dot(v_ego - v_veh, unit(r)); positive => approaching
        closing = ((evx - sub["_vx"]) * rx + (evy - sub["_vy"]) * ry + (evz - sub["_vz"]) * rz) / norm
        closing = closing.clip(-MAX_SPEED_MPS, MAX_SPEED_MPS)
        df.loc[sub.index, "relative_speed"] = closing

    df.loc[df["vehicle_id"] == "ego", ["distance_to_ego", "relative_speed"]] = 0.0
    return df


def _check_missing(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    mask = df[list(REQUIRED_NON_NULL)].isna().any(axis=1)
    report.null_required = int(mask.sum())
    if mask.any():
        report.notes.append(f"dropped {int(mask.sum())} rows with nulls in required columns")
    return df[~mask]


def _check_coords_and_speed(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    coord_ok = (
        df[["x", "y", "z"]].abs().le(WORLD_COORD_LIMIT_M).all(axis=1)
        & np.isfinite(df[["x", "y", "z"]]).all(axis=1)
    )
    speed_ok = df["speed"].between(0.0, MAX_SPEED_MPS) & np.isfinite(df["speed"])
    report.invalid_coords = int((~coord_ok).sum())
    report.invalid_speed = int((~speed_ok).sum())
    return df[coord_ok & speed_ok]


def _check_broken_trajectories(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Flag ticks whose implied step speed exceeds MAX_SPEED_MPS by a wide margin."""
    df = df.sort_values(["vehicle_id", "timestamp"]).copy()
    grp = df.groupby("vehicle_id", sort=False)
    dt = grp["timestamp"].diff()
    step = np.sqrt(grp["x"].diff() ** 2 + grp["y"].diff() ** 2 + grp["z"].diff() ** 2)
    implied = step / dt.replace(0.0, np.nan)
    broken = implied > (MAX_SPEED_MPS * 1.5)
    report.broken_trajectory_rows = int(broken.fillna(False).sum())
    if broken.any():
        report.notes.append(
            f"dropped {report.broken_trajectory_rows} rows with implausible position jumps"
        )
    return df[~broken.fillna(False)]


def _check_timestamp_continuity(df: pd.DataFrame, report: QualityReport) -> None:
    """Report (do not drop) non-monotonic timestamps and large tick gaps."""
    for vid, sub in df.sort_values("timestamp").groupby("vehicle_id"):
        ts = sub["timestamp"].to_numpy()
        if len(ts) < 3:
            continue
        diffs = np.diff(ts)
        if (diffs <= 0).any():
            report.timestamp_gap_events += int((diffs <= 0).sum())
            report.notes.append(f"{vid}: {int((diffs <= 0).sum())} non-increasing timestamps")
        median_dt = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0.0
        if median_dt > 0:
            gaps = int((diffs > 3.0 * median_dt).sum())
            if gaps:
                report.timestamp_gap_events += gaps
                report.notes.append(f"{vid}: {gaps} tick gaps > 3x median dt ({median_dt:.3f}s)")


def clean_carla(raw: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, dict]:
    """Return ``(silver_df, quality_report_dict)`` for one scenario's raw frame."""
    report = QualityReport(scenario=scenario, rows_in=len(raw))

    df = _raw_to_common(raw, report)
    df = _derive_acceleration(df, report)
    df = _derive_relative_to_ego(df)

    df = _check_missing(df, report)
    df = _check_coords_and_speed(df, report)
    df = _check_broken_trajectories(df, report)
    _check_timestamp_continuity(df, report)

    df = df.drop(columns=["_vx", "_vy", "_vz"], errors="ignore")
    df = df[list(CARLA_TELEMETRY_COLUMNS)].reset_index(drop=True)

    # normalise dtypes ahead of pandera coercion
    df["vehicle_id"] = df["vehicle_id"].astype("string")
    df["vehicle_type"] = (
        df["vehicle_type"].where(df["vehicle_type"].isin(CARLA_VEHICLE_TYPES), "unknown")
    ).astype("string")
    df["is_emergency"] = df["is_emergency"].astype("boolean")
    df["lane_id"] = df["lane_id"].astype("Int64")

    validated = validate_carla(df, lazy=True)
    report.rows_out = len(validated)
    log.info("clean_carla[%s]: %d -> %d rows", scenario, report.rows_in, report.rows_out)
    return validated, report.as_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze-file",
        type=Path,
        required=True,
        help="Bronze telemetry parquet produced by load_bronze",
    )
    parser.add_argument("--scenario", default=None, help="defaults to the bronze file stem")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Silver parquet path; defaults to data/silver/carla_telemetry/<scenario>.parquet",
    )
    args = parser.parse_args(argv)

    scenario = args.scenario or args.bronze_file.stem
    raw = pd.read_parquet(args.bronze_file)
    silver, report = clean_carla(raw, scenario)

    out = args.out or (domain_dir("silver", CARLA_DOMAIN) / f"{scenario}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out, engine="pyarrow", index=False)

    print(silver.head().to_string(index=False))
    print(f"\n[clean_carla] wrote {len(silver)} rows -> {out}")
    print("[clean_carla] quality report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
