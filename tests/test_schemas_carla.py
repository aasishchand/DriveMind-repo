"""Tests for the common CARLA telemetry schema (etl/schemas/carla_telemetry.py)."""

from __future__ import annotations

import pandera.errors as pa_errors
import pytest

from etl.schemas.carla_telemetry import (
    CARLA_TELEMETRY_COLUMNS,
    empty_frame,
    validate,
)


def test_valid_frame_passes(valid_carla_df):
    out = validate(valid_carla_df)
    assert len(out) == len(valid_carla_df)
    assert list(out.columns) == list(CARLA_TELEMETRY_COLUMNS)


def test_empty_frame_has_right_shape_and_passes():
    df = empty_frame()
    assert list(df.columns) == list(CARLA_TELEMETRY_COLUMNS)
    assert len(df) == 0
    validate(df)  # should not raise


def test_negative_speed_rejected(valid_carla_df):
    bad = valid_carla_df.copy()
    bad.loc[0, "speed"] = -1.0
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)


def test_heading_out_of_range_rejected(valid_carla_df):
    bad = valid_carla_df.copy()
    bad.loc[0, "heading"] = 270.0  # only [-180, 180] is valid
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)


def test_unknown_vehicle_type_rejected(valid_carla_df):
    bad = valid_carla_df.copy()
    bad.loc[0, "vehicle_type"] = "spaceship"
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)


def test_duplicate_vehicle_timestamp_rejected(valid_carla_df):
    bad = valid_carla_df.copy()
    bad.loc[1, "vehicle_id"] = bad.loc[0, "vehicle_id"]
    bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)


def test_relative_columns_require_an_ego_row(valid_carla_df):
    """distance_to_ego / relative_speed populated with no 'ego' row anchoring
    them should fail the ego_reference_columns_present dataframe check."""
    bad = valid_carla_df.copy()
    bad["vehicle_id"] = bad["vehicle_id"].replace({"ego": "veh_2"})
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)


def test_strict_schema_rejects_unexpected_column(valid_carla_df):
    bad = valid_carla_df.copy()
    bad["extra_column"] = 1
    with pytest.raises(pa_errors.SchemaErrors):
        validate(bad)
