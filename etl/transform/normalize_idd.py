"""Normalise raw IDD label rows to the canonical schema (Bronze -> Silver).

Steps:

    1. lower-case / strip raw class strings and remap known aliases
    2. drop labels outside the canonical class list (recorded in the report)
    3. attach integer class ids (index into the canonical tuple)
    4. de-duplicate exact-duplicate rows
    5. pandera validation (IDDDetectionLabelSchema / IDDSegmentationLabelSchema)

STATUS: the segmentation path (class list, official level3Id map, alias
handling) is verified against a real IDD Segmentation Part 2 train+val set
(~602k polygons, 0 unknown classes after canonicalisation). The detection
alias map is still a starting point - no real IDD Detection data has been run
through it yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from etl.config import IDD_DOMAIN, domain_dir, get_logger
from etl.schemas.idd_labels import (
    IDD_DETECTION_CLASSES,
    IDD_LEVEL3_CLASSES,
    IDD_LEVEL3_ID_MAP,
    validate_detection_labels,
    validate_segmentation_labels,
)

log = get_logger(__name__)

# raw string -> canonical detection class
_DETECTION_ALIASES = {
    "auto": "autorickshaw",
    "auto rickshaw": "autorickshaw",
    "autorickshaw": "autorickshaw",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "cycle": "bicycle",
    "traffic_sign": "traffic sign",
    "traffic_light": "traffic light",
    "vehicle_fallback": "vehicle fallback",
}

# raw string -> canonical level-3 class
_SEGMENTATION_ALIASES = {
    "drivable_fallback": "drivable fallback",
    "non_drivable_fallback": "non-drivable fallback",
    "vehicle_fallback": "vehicle fallback",
    "fallback_background": "fallback background",
    "guard_rail": "guard rail",
    "traffic_sign": "traffic sign",
    "traffic_light": "traffic light",
    "obs_str_bar_fallback": "obs-str-bar-fallback",
}

_DETECTION_ID = {name: i for i, name in enumerate(IDD_DETECTION_CLASSES)}
# level3_id uses the official AutoNUE mapping (not a local index) - see
# etl.schemas.idd_labels.IDD_LEVEL3_ID_MAP for where it comes from.
_LEVEL3_ID = IDD_LEVEL3_ID_MAP


def _canon(value: object, aliases: dict[str, str]) -> str:
    key = str(value or "").strip().lower()
    return aliases.get(key, key)


def normalize_idd(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return ``(silver_df, report_dict)``. Variant is read from the frame."""
    if raw.empty:
        raise ValueError("normalize_idd received an empty frame")

    variant = str(raw["variant"].iloc[0])
    is_detection = variant in {"detection", "temporal"}
    report: dict = {"variant": variant, "rows_in": len(raw)}

    df = raw.copy()

    if is_detection:
        df["class_name"] = df["raw_class_name"].map(lambda v: _canon(v, _DETECTION_ALIASES))
        known = df["class_name"].isin(IDD_DETECTION_CLASSES)
        report["dropped_unknown_class"] = int((~known).sum())
        df = df[known].copy()
        df["class_id"] = df["class_name"].map(_DETECTION_ID).astype("Int64")
        df["frame_index"] = pd.to_numeric(df.get("frame_index"), errors="coerce").astype("Int64")
        df["sequence_id"] = df.get("sequence_id").astype("string")
        for col in ("occluded", "truncated"):
            df[col] = df[col].astype("boolean")
        keep = [
            "image_id", "image_path", "image_width", "image_height", "split", "variant",
            "sequence_id", "frame_index", "class_name", "class_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "occluded", "truncated",
        ]
        df = df[keep].drop_duplicates().reset_index(drop=True)
        report["rows_out"] = len(df)
        validated = validate_detection_labels(df, lazy=True)
    else:
        df["label"] = df["raw_label"].map(lambda v: _canon(v, _SEGMENTATION_ALIASES))
        known = df["label"].isin(IDD_LEVEL3_CLASSES)
        report["dropped_unknown_class"] = int((~known).sum())
        df = df[known].copy()
        df["level3_id"] = df["label"].map(_LEVEL3_ID).astype("Int64")
        df["instance_id"] = pd.to_numeric(df.get("instance_id"), errors="coerce").astype("Int64")
        df["num_vertices"] = pd.to_numeric(df.get("num_vertices"), errors="coerce").astype("Int64")
        df["polygon_area_px"] = pd.to_numeric(df.get("polygon_area_px"), errors="coerce")

        # A handful of real annotations are degenerate (<3 vertices - a point or
        # a line, not a polygon); no meaningful shape/area, so drop and count.
        degenerate = df["num_vertices"].isna() | (df["num_vertices"] < 3)
        report["dropped_degenerate_polygon"] = int(degenerate.sum())
        df = df[~degenerate].copy()

        keep = [
            "image_id", "image_path", "image_width", "image_height", "split", "variant",
            "gt_json_path", "label", "level3_id", "instance_id",
            "polygon_area_px", "num_vertices",
        ]
        df = df[keep].drop_duplicates().reset_index(drop=True)
        report["rows_out"] = len(df)
        validated = validate_segmentation_labels(df, lazy=True)

    log.info("normalize_idd[%s]: %d -> %d rows", variant, report["rows_in"], report["rows_out"])
    return validated, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.bronze_file)
    silver, report = normalize_idd(raw)

    stem = args.bronze_file.stem
    out = args.out or (domain_dir("silver", IDD_DOMAIN) / f"{stem}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out, engine="pyarrow", index=False)

    print(silver.head().to_string(index=False))
    print(f"\n[normalize_idd] wrote {len(silver)} rows -> {out}")
    print(f"[normalize_idd] report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
