"""Common schema for CARLA telemetry rows (Silver layer contract).

One row == one tracked vehicle at one simulation tick within one scenario run.
Every Silver telemetry Parquet file must satisfy :class:`CarlaTelemetrySchema`;
Gold feature tables are derived exclusively from data in this shape.

The 13 columns match the agreed common schema:

    vehicle_id, timestamp, x, y, z, speed, acceleration, heading, lane_id,
    vehicle_type, is_emergency, distance_to_ego, relative_speed
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.20 moved the pandas API into a submodule
    import pandera.pandas as pa
    from pandera.typing.pandas import Series
except ModuleNotFoundError:  # pandera < 0.20
    import pandera as pa
    from pandera.typing import Series


# --- domain vocabulary -------------------------------------------------------

# Coarse vehicle categories. Cleaning maps raw CARLA blueprint ids onto these;
# anything unrecognised becomes "unknown" rather than being dropped.
CARLA_VEHICLE_TYPES = (
    "car",
    "truck",
    "bus",
    "van",
    "motorcycle",
    "bicycle",
    "emergency",
    "unknown",
)

# --- physical sanity bounds ------------------------------------------------
# Deliberately generous. These reject corrupt rows (NaN-coerced-to-0, unit
# mix-ups, exploded physics frames), not merely unusual-but-valid values.
MAX_SPEED_MPS = 150.0          # ~540 km/h
MAX_ABS_ACCEL_MPS2 = 50.0      # ~5 g
WORLD_COORD_LIMIT_M = 1_000_000.0

CARLA_TELEMETRY_COLUMNS = (
    "vehicle_id",
    "timestamp",
    "x",
    "y",
    "z",
    "speed",
    "acceleration",
    "heading",
    "lane_id",
    "vehicle_type",
    "is_emergency",
    "distance_to_ego",
    "relative_speed",
)

# pandas dtypes for an empty, schema-shaped frame (see ``empty_frame``).
_EMPTY_DTYPES = {
    "vehicle_id": "string",
    "timestamp": "float64",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "speed": "float64",
    "acceleration": "float64",
    "heading": "float64",
    "lane_id": "Int64",
    "vehicle_type": "string",
    "is_emergency": "boolean",
    "distance_to_ego": "float64",
    "relative_speed": "float64",
}


class CarlaTelemetrySchema(pa.DataFrameModel):
    """Canonical per-tick vehicle telemetry row."""

    vehicle_id: Series[str] = pa.Field(
        nullable=False,
        description="Stable id for the actor within a scenario run ('ego' for the ego vehicle).",
    )
    timestamp: Series[float] = pa.Field(
        ge=0.0,
        nullable=False,
        description="Seconds since the start of the scenario run (CARLA snapshot timestamp).",
    )
    x: Series[float] = pa.Field(
        ge=-WORLD_COORD_LIMIT_M,
        le=WORLD_COORD_LIMIT_M,
        nullable=False,
        description="World X position in metres (CARLA map frame).",
    )
    y: Series[float] = pa.Field(
        ge=-WORLD_COORD_LIMIT_M,
        le=WORLD_COORD_LIMIT_M,
        nullable=False,
        description="World Y position in metres (CARLA map frame).",
    )
    z: Series[float] = pa.Field(
        ge=-WORLD_COORD_LIMIT_M,
        le=WORLD_COORD_LIMIT_M,
        nullable=False,
        description="World Z position in metres (CARLA map frame).",
    )
    speed: Series[float] = pa.Field(
        ge=0.0,
        le=MAX_SPEED_MPS,
        nullable=False,
        description="Speed magnitude in m/s (Euclidean norm of the velocity vector).",
    )
    acceleration: Series[float] = pa.Field(
        ge=-MAX_ABS_ACCEL_MPS2,
        le=MAX_ABS_ACCEL_MPS2,
        nullable=True,
        description="Signed longitudinal acceleration in m/s^2; null on the first tick of a track.",
    )
    heading: Series[float] = pa.Field(
        ge=-180.0,
        le=180.0,
        nullable=False,
        description="Yaw in degrees normalised to [-180, 180] (CARLA rotation.yaw convention).",
    )
    lane_id: Series[pd.Int64Dtype] = pa.Field(
        nullable=True,
        description="CARLA waypoint lane id (sign encodes direction); null when off-road.",
    )
    vehicle_type: Series[str] = pa.Field(
        isin=CARLA_VEHICLE_TYPES,
        nullable=False,
        description="Coarse vehicle category; see CARLA_VEHICLE_TYPES.",
    )
    is_emergency: Series[bool] = pa.Field(
        nullable=False,
        description="True if the actor is an active emergency vehicle (lights/siren or emergency blueprint).",
    )
    distance_to_ego: Series[float] = pa.Field(
        ge=0.0,
        nullable=True,
        description="Euclidean distance in metres to the ego vehicle; 0.0 on the ego's own row.",
    )
    relative_speed: Series[float] = pa.Field(
        ge=-MAX_SPEED_MPS,
        le=MAX_SPEED_MPS,
        nullable=True,
        description=(
            "Closing speed in m/s along the ego->vehicle axis: positive = gap shrinking, "
            "0.0 on the ego's own row."
        ),
    )

    class Config:
        name = "carla_telemetry_silver"
        strict = True       # reject unexpected columns: Silver must match exactly
        coerce = True
        ordered = False
        unique = ["vehicle_id", "timestamp"]

    # A placeholder for cross-row data-quality rules that belong in the schema
    # rather than in clean_carla.py. Trajectory-continuity and timestamp-gap
    # checks currently live in etl/transform/clean_carla.py because they need
    # per-scenario parameters (tick rate, gap tolerance).
    @pa.dataframe_check(name="ego_reference_columns_present")
    def _ego_present_when_relative_cols_populated(cls, df: pd.DataFrame) -> bool:
        # If any relative measurement is populated, at least one 'ego' row must
        # exist to anchor it.
        has_relative = df["distance_to_ego"].notna().any() or df["relative_speed"].notna().any()
        return (not has_relative) or (df["vehicle_id"] == "ego").any()


def validate(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Validate ``df`` against the common schema, collecting all errors when lazy."""
    return CarlaTelemetrySchema.validate(df, lazy=lazy)


def empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the exact common-schema columns and dtypes."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in _EMPTY_DTYPES.items()}
    )
