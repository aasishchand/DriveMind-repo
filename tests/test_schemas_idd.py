"""Tests for the IDD label schemas (etl/schemas/idd_labels.py)."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from etl.schemas.idd_labels import (
    IDD_LEVEL3_CLASSES,
    IDD_LEVEL3_ID_MAP,
    IDDDetectionLabelSchema,
    IDDSegmentationLabelSchema,
    validate_detection_labels,
    validate_segmentation_labels,
)


def test_every_level3_class_has_an_official_id():
    """Guards against the two verified-against-real-data structures drifting
    apart if either is edited independently."""
    missing = [name for name in IDD_LEVEL3_CLASSES if name not in IDD_LEVEL3_ID_MAP]
    assert missing == []


def test_level3_id_map_values_are_known_ids():
    valid_ids = set(range(26)) | {255}
    bad = {name: idx for name, idx in IDD_LEVEL3_ID_MAP.items() if idx not in valid_ids}
    assert bad == {}


@pytest.fixture
def valid_detection_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                image_id="000001", image_path="JPEGImages/000001.jpg",
                image_width=1920, image_height=1080, split="train", variant="detection",
                sequence_id=None, frame_index=None, class_name="car", class_id=0,
                bbox_x=100.0, bbox_y=100.0, bbox_w=200.0, bbox_h=150.0,
                occluded=False, truncated=False,
            )
        ]
    )


@pytest.fixture
def valid_segmentation_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                image_id="0000000", image_path="train/images/0000000_leftImg8bit.jpg",
                image_width=1920, image_height=1080, split="train", variant="segmentation",
                gt_json_path="train/json/0000000_gtFine_polygons.json", label="road",
                level3_id=0, instance_id=1, polygon_area_px=1000.0, num_vertices=8,
            )
        ]
    )


def test_valid_detection_row_passes(valid_detection_row):
    out = validate_detection_labels(valid_detection_row)
    assert len(out) == 1


def test_bbox_exceeding_image_bounds_rejected(valid_detection_row):
    bad = valid_detection_row.copy()
    bad.loc[0, "bbox_w"] = 5000.0  # way past image_width
    with pytest.raises(pa_errors.SchemaErrors):
        validate_detection_labels(bad)


def test_valid_segmentation_row_passes(valid_segmentation_row):
    out = validate_segmentation_labels(valid_segmentation_row)
    assert len(out) == 1


def test_unknown_label_rejected(valid_segmentation_row):
    bad = valid_segmentation_row.copy()
    bad.loc[0, "label"] = "spaceship"
    with pytest.raises(pa_errors.SchemaErrors):
        validate_segmentation_labels(bad)


def test_degenerate_polygon_rejected_by_schema(valid_segmentation_row):
    """The schema itself still rejects <3 vertices - normalize_idd is expected
    to filter these out *before* validation (see test_normalize_idd.py)."""
    bad = valid_segmentation_row.copy()
    bad.loc[0, "num_vertices"] = 2
    with pytest.raises(pa_errors.SchemaErrors):
        validate_segmentation_labels(bad)
