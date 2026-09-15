"""Tests for etl/transform/normalize_idd.py."""

from __future__ import annotations

import pandas as pd

from etl.extract.extract_idd import RAW_SEGMENTATION_COLUMNS, _synthetic_detection, _synthetic_segmentation
from etl.transform.normalize_idd import normalize_idd


def test_synthetic_detection_round_trips():
    raw = _synthetic_detection("train", "detection")
    silver, report = normalize_idd(raw)
    assert report["dropped_unknown_class"] == 0
    assert len(silver) == report["rows_out"] == len(raw)


def test_synthetic_segmentation_round_trips():
    raw = _synthetic_segmentation("train", "segmentation")
    silver, report = normalize_idd(raw)
    assert report["dropped_unknown_class"] == 0
    assert len(silver) == report["rows_out"] == len(raw)
    # 'road' is level3Id 0 in the official AutoNUE mapping
    road_rows = silver[silver["label"] == "road"]
    assert (road_rows["level3_id"] == 0).all()


def test_unknown_segmentation_label_is_dropped_and_counted():
    raw = _synthetic_segmentation("train", "segmentation")
    raw.loc[0, "raw_label"] = "flying_saucer"
    _, report = normalize_idd(raw)
    assert report["dropped_unknown_class"] == 1
    assert report["rows_out"] == len(raw) - 1


def test_degenerate_polygon_is_dropped_and_counted():
    raw = _synthetic_segmentation("train", "segmentation")
    raw.loc[0, "num_vertices"] = 2  # below the 3-vertex minimum
    _, report = normalize_idd(raw)
    assert report["dropped_degenerate_polygon"] == 1
    assert report["rows_out"] == len(raw) - 1


def test_shared_level3_ids_are_assigned_correctly():
    """caravan / trailer / train / vehicle fallback all map to level3Id 12."""
    row = dict.fromkeys(RAW_SEGMENTATION_COLUMNS)
    row.update(
        image_id="img1", image_path="p.jpg", image_width=100, image_height=100,
        split="train", variant="segmentation", gt_json_path="p.json",
        raw_label="caravan", instance_id=1, polygon_area_px=10.0, num_vertices=4,
    )
    raw = pd.DataFrame([row])
    silver, report = normalize_idd(raw)
    assert report["dropped_unknown_class"] == 0
    assert silver.loc[0, "level3_id"] == 12
