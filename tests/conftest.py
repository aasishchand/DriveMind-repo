"""Shared fixtures for the DriveMind ETL test suite.

All fixtures build small in-memory DataFrames - no file I/O, no real IDD/CARLA
data, no database. Tests should run in a couple of seconds with nothing but
`pip install -r requirements.txt`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from etl.schemas.carla_telemetry import CARLA_TELEMETRY_COLUMNS


@pytest.fixture
def valid_carla_df() -> pd.DataFrame:
    """Two ticks, two vehicles (ego + one other), fully schema-conformant."""
    rows = [
        dict(
            vehicle_id="ego", timestamp=0.0, x=0.0, y=0.0, z=0.0, speed=5.0,
            acceleration=None, heading=0.0, lane_id=-1, vehicle_type="car",
            is_emergency=False, distance_to_ego=0.0, relative_speed=0.0,
        ),
        dict(
            vehicle_id="veh_1", timestamp=0.0, x=20.0, y=0.0, z=0.0, speed=3.0,
            acceleration=None, heading=180.0, lane_id=-1, vehicle_type="emergency",
            is_emergency=True, distance_to_ego=20.0, relative_speed=8.0,
        ),
        dict(
            vehicle_id="ego", timestamp=0.05, x=0.25, y=0.0, z=0.0, speed=5.0,
            acceleration=0.0, heading=0.0, lane_id=-1, vehicle_type="car",
            is_emergency=False, distance_to_ego=0.0, relative_speed=0.0,
        ),
        dict(
            vehicle_id="veh_1", timestamp=0.05, x=19.85, y=0.0, z=0.0, speed=3.0,
            acceleration=0.0, heading=180.0, lane_id=-1, vehicle_type="emergency",
            is_emergency=True, distance_to_ego=19.6, relative_speed=8.0,
        ),
    ]
    return pd.DataFrame(rows, columns=list(CARLA_TELEMETRY_COLUMNS))
