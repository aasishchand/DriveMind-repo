"""Tests for etl/extract/extract_idd.py - the polygon-bbox derivation path used
when a real IDD Detection VOC-XML export isn't available (see extract_idd's
STATUS docstring: this is how real IDD Detection data was actually obtained,
derived from the real IDD Segmentation Part 2 polygons)."""

from __future__ import annotations

import json

from etl.extract.extract_idd import _parse_polygon_json_as_detection, _polygon_bbox


def test_polygon_bbox_axis_aligned():
    square = [[10.0, 20.0], [10.0, 50.0], [40.0, 50.0], [40.0, 20.0]]
    x, y, w, h = _polygon_bbox(square)
    assert (x, y, w, h) == (10.0, 20.0, 30.0, 30.0)


def test_polygon_bbox_empty_returns_none():
    assert _polygon_bbox([]) is None


def test_parse_polygon_json_as_detection_filters_to_thing_classes(tmp_path):
    payload = {
        "imgWidth": 1000,
        "imgHeight": 500,
        "objects": [
            {"label": "car", "id": 1, "polygon": [[10, 10], [10, 60], [60, 60], [60, 10]]},
            {"label": "road", "id": 2, "polygon": [[0, 400], [0, 500], [1000, 500], [1000, 400]]},
            {"label": "sky", "id": 3, "polygon": [[0, 0], [0, 50], [1000, 50], [1000, 0]]},
        ],
    }
    json_path = tmp_path / "0000000_gtFine_polygons.json"
    json_path.write_text(json.dumps(payload))

    rows = _parse_polygon_json_as_detection(json_path)

    # only the "thing" class (car) survives - road/sky are "stuff", dropped
    assert len(rows) == 1
    assert rows[0]["raw_class_name"] == "car"
    assert rows[0]["bbox_x"] == 10.0
    assert rows[0]["bbox_y"] == 10.0
    assert rows[0]["bbox_w"] == 50.0
    assert rows[0]["bbox_h"] == 50.0
    assert rows[0]["occluded"] is None
    assert rows[0]["truncated"] is None
    assert rows[0]["image_width"] == 1000
    assert rows[0]["image_height"] == 500
