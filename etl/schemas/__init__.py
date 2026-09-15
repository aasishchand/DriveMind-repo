"""pandera schema definitions for every validated DriveMind dataset.

    carla_telemetry -> CarlaTelemetrySchema  (the common Silver telemetry schema)
    idd_labels      -> IDDDetectionLabelSchema, IDDSegmentationLabelSchema
"""

from etl.schemas.carla_telemetry import (
    CARLA_TELEMETRY_COLUMNS,
    CarlaTelemetrySchema,
    empty_frame as empty_carla_frame,
    validate as validate_carla_telemetry,
)
from etl.schemas.idd_labels import (
    IDD_DETECTION_CLASSES,
    IDD_LEVEL3_CLASSES,
    IDD_SPLITS,
    IDDDetectionLabelSchema,
    IDDSegmentationLabelSchema,
    validate_detection_labels,
    validate_segmentation_labels,
)

__all__ = [
    "CARLA_TELEMETRY_COLUMNS",
    "CarlaTelemetrySchema",
    "empty_carla_frame",
    "validate_carla_telemetry",
    "IDD_DETECTION_CLASSES",
    "IDD_LEVEL3_CLASSES",
    "IDD_SPLITS",
    "IDDDetectionLabelSchema",
    "IDDSegmentationLabelSchema",
    "validate_detection_labels",
    "validate_segmentation_labels",
]
