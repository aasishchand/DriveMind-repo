"""Tests for etl/transform/clean_carla.py - the raw->common mapping and the
four data-quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etl.extract.extract_carla import _synthetic_raw
from etl.transform.clean_carla import (
    MAX_ABS_ACCEL_MPS2,
    _map_vehicle_type,
    _normalise_heading,
    clean_carla,
)


def test_clean_carla_on_synthetic_data_round_trips():
    raw = _synthetic_raw("unit_test", n_ticks=30)
    clean, report = clean_carla(raw, "unit_test")

    assert report["rows_in"] == len(raw)
    assert report["rows_out"] == len(clean)
    assert report["rows_out"] > 0
    assert set(clean["vehicle_id"].unique()) == {"ego", "veh_042"}

    ego = clean[clean["vehicle_id"] == "ego"]
    assert (ego["distance_to_ego"] == 0.0).all()
    assert (ego["relative_speed"] == 0.0).all()


def test_missing_value_check_drops_and_counts():
    raw = _synthetic_raw("dq_missing", n_ticks=10)
    raw.loc[2, "loc_x"] = np.nan
    _, report = clean_carla(raw, "dq_missing")
    assert report["null_required"] == 1
    assert report["rows_out"] == report["rows_in"] - 1


def test_invalid_speed_check_drops_and_counts():
    raw = _synthetic_raw("dq_speed", n_ticks=10)
    raw.loc[4, "vel_x"] = 9999.0  # blows past MAX_SPEED_MPS after derivation
    _, report = clean_carla(raw, "dq_speed")
    assert report["invalid_speed"] >= 1


def test_invalid_coords_check_drops_and_counts():
    raw = _synthetic_raw("dq_coords", n_ticks=10)
    raw.loc[6, "loc_y"] = 5_000_000.0  # past WORLD_COORD_LIMIT_M
    _, report = clean_carla(raw, "dq_coords")
    assert report["invalid_coords"] >= 1


def test_broken_trajectory_check_drops_teleporting_row():
    raw = _synthetic_raw("dq_traj", n_ticks=10)
    # teleport one vehicle 500m sideways for a single tick - implausible speed
    raw.loc[8, "loc_y"] = raw.loc[8, "loc_y"] + 500.0
    _, report = clean_carla(raw, "dq_traj")
    assert report["broken_trajectory_rows"] >= 1


def test_acceleration_is_clipped_not_left_unbounded():
    raw = _synthetic_raw("dq_accel", n_ticks=10)
    raw.loc[3, "vel_x"] = raw.loc[3, "vel_x"] + 500.0  # huge speed jump -> huge accel
    clean, report = clean_carla(raw, "dq_accel")
    assert report["accel_clipped"] >= 1
    assert clean["acceleration"].abs().max() <= MAX_ABS_ACCEL_MPS2 + 1e-6


def test_map_vehicle_type_hints():
    assert _map_vehicle_type("vehicle.carlamotors.firetruck") == "emergency"
    assert _map_vehicle_type("vehicle.harley-davidson.low_rider") == "motorcycle"
    assert _map_vehicle_type("vehicle.tesla.model3") == "car"
    assert _map_vehicle_type("not_a_vehicle_blueprint") == "unknown"
    assert _map_vehicle_type(None) == "unknown"


def test_normalise_heading_wraps_to_range():
    s = pd.Series([0.0, 180.0, -180.0, 190.0, 350.0, -350.0])
    out = _normalise_heading(s)
    assert out.between(-180.0, 180.0).all()
    # 190 degrees should wrap to -170
    assert out.iloc[3] == -170.0
