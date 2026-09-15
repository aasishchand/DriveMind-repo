"""Extract IDD annotations into flat label rows.

STATUS
------
  * ``_parse_polygon_json`` (segmentation/lite) is **verified** against a real
    IDD Segmentation Part 2 train+val set (602k polygons via ``--idd-root``,
    0 unknown classes after normalisation) - resolves each json's sibling
    image under ``images/`` too.
  * ``_parse_voc_xml`` (detection/temporal) is untested against real data -
    written from the Pascal-VOC spec, not yet run against an actual IDD
    Detection export.
  * With no ``--idd-root``, both fall back to a tiny synthetic frame so
    downstream stages stay runnable without any dataset on disk.

The official IDD site (https://idd.insaan.iiit.ac.in/, registration required)
was down when this was last checked; a real segmentation set was pulled from
a Kaggle mirror instead (search "IDD Segmentation Part 2" / gtFine polygon
JSON format - many other IDD mirrors ship preprocessed raster masks instead,
which this extractor does not read). Label parsing and the class hierarchy
follow AutoNUE/public-code.

Run:
    python -m etl.extract.extract_idd --variant segmentation --split train \\
        --idd-root /path/to/idd_part_2
    python -m etl.extract.extract_idd --variant detection --split train
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from etl.config import get_logger
from etl.schemas.idd_labels import IDD_SPLITS

log = get_logger(__name__)

# Raw (pre-normalisation) column sets. normalize_idd.py maps class strings onto
# the canonical vocabulary and attaches integer ids.
RAW_DETECTION_COLUMNS = (
    "image_id",
    "image_path",
    "image_width",
    "image_height",
    "split",
    "variant",
    "sequence_id",
    "frame_index",
    "raw_class_name",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "occluded",
    "truncated",
)
RAW_SEGMENTATION_COLUMNS = (
    "image_id",
    "image_path",
    "image_width",
    "image_height",
    "split",
    "variant",
    "gt_json_path",
    "raw_label",
    "instance_id",
    "polygon_area_px",
    "num_vertices",
)


def _polygon_area(points: list) -> float:
    """Shoelace area of a polygon given as ``[[x, y], ...]``."""
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse one Pascal-VOC annotation file into a list of raw box dicts."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    filename = root.findtext("filename", xml_path.stem)

    boxes: list[dict] = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        xmin = float(b.findtext("xmin", "0"))
        ymin = float(b.findtext("ymin", "0"))
        xmax = float(b.findtext("xmax", "0"))
        ymax = float(b.findtext("ymax", "0"))
        boxes.append(
            dict(
                image_id=xml_path.stem,
                image_path=filename,
                image_width=width,
                image_height=height,
                raw_class_name=(obj.findtext("name", "") or "").strip(),
                bbox_x=xmin,
                bbox_y=ymin,
                bbox_w=max(xmax - xmin, 0.0),
                bbox_h=max(ymax - ymin, 0.0),
                occluded=bool(int(obj.findtext("occluded", "0") or 0)),
                truncated=bool(int(obj.findtext("truncated", "0") or 0)),
            )
        )
    return boxes


def _find_sibling_image(json_path: Path) -> Path | None:
    """Given .../<split>/json/<id>_gtFine_polygons.json, find the matching
    .../<split>/images/<id>_leftImg8bit.{jpg,png} (the Cityscapes/IDD layout:
    json/ and images/ as sibling folders under each split)."""
    image_id = json_path.stem.replace("_gtFine_polygons", "")
    images_dir = json_path.parent.parent / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{image_id}_leftImg8bit{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_polygon_json(json_path: Path, idd_root: Path | None = None) -> list[dict]:
    """Parse one Cityscapes-style polygon JSON into a list of raw polygon dicts."""
    data = json.loads(json_path.read_text())
    width = int(data.get("imgWidth", 0))
    height = int(data.get("imgHeight", 0))
    image_id = json_path.stem.replace("_gtFine_polygons", "")

    image_path = ""
    sibling = _find_sibling_image(json_path)
    if sibling is not None:
        image_path = str(sibling.relative_to(idd_root)) if idd_root else str(sibling)

    rows: list[dict] = []
    for obj in data.get("objects", []):
        polygon = obj.get("polygon", [])
        rows.append(
            dict(
                image_id=image_id,
                image_path=image_path,
                image_width=width,
                image_height=height,
                gt_json_path=str(json_path),
                raw_label=(obj.get("label", "") or "").strip(),
                instance_id=obj.get("id"),
                polygon_area_px=_polygon_area(polygon),
                num_vertices=len(polygon),
            )
        )
    return rows


def _synthetic_detection(split: str, variant: str) -> pd.DataFrame:
    rows = [
        dict(
            image_id="000001",
            image_path=f"JPEGImages/frontFar/{split}/000001.jpg",
            image_width=1920,
            image_height=1080,
            split=split,
            variant=variant,
            sequence_id="seq_0000" if variant == "temporal" else None,
            frame_index=0 if variant == "temporal" else None,
            raw_class_name="car",
            bbox_x=812.0,
            bbox_y=443.0,
            bbox_w=190.0,
            bbox_h=120.0,
            occluded=False,
            truncated=False,
        ),
        dict(
            image_id="000001",
            image_path=f"JPEGImages/frontFar/{split}/000001.jpg",
            image_width=1920,
            image_height=1080,
            split=split,
            variant=variant,
            sequence_id="seq_0000" if variant == "temporal" else None,
            frame_index=0 if variant == "temporal" else None,
            raw_class_name="autorickshaw",
            bbox_x=1100.0,
            bbox_y=520.0,
            bbox_w=160.0,
            bbox_h=150.0,
            occluded=True,
            truncated=False,
        ),
    ]
    return pd.DataFrame(rows, columns=list(RAW_DETECTION_COLUMNS))


def _synthetic_segmentation(split: str, variant: str) -> pd.DataFrame:
    rows = [
        dict(
            image_id="frame0001",
            image_path=f"leftImg8bit/{split}/201/frame0001_leftImg8bit.png",
            image_width=1920,
            image_height=1080,
            split=split,
            variant=variant,
            gt_json_path=f"gtFine/{split}/201/frame0001_gtFine_polygons.json",
            raw_label="road",
            instance_id=None,
            polygon_area_px=904531.0,
            num_vertices=12,
        ),
        dict(
            image_id="frame0001",
            image_path=f"leftImg8bit/{split}/201/frame0001_leftImg8bit.png",
            image_width=1920,
            image_height=1080,
            split=split,
            variant=variant,
            gt_json_path=f"gtFine/{split}/201/frame0001_gtFine_polygons.json",
            raw_label="motorcycle",
            instance_id=1,
            polygon_area_px=5120.0,
            num_vertices=8,
        ),
    ]
    return pd.DataFrame(rows, columns=list(RAW_SEGMENTATION_COLUMNS))


def extract_idd(
    variant: str,
    split: str,
    idd_root: Path | None = None,
) -> pd.DataFrame:
    """Return a raw IDD label DataFrame for one (variant, split).

    With ``idd_root`` set, walks the tree for ``*.xml`` (detection/temporal) or
    ``*_polygons.json`` (segmentation/lite) and parses them. Without it, returns
    a small synthetic frame (stub).
    """
    is_detection = variant in {"detection", "temporal"}

    if idd_root is None:
        log.warning("no --idd-root given; emitting SYNTHETIC IDD labels (stub)")
        return (
            _synthetic_detection(split, variant)
            if is_detection
            else _synthetic_segmentation(split, variant)
        )

    idd_root = Path(idd_root)
    log.info("scanning IDD root %s for variant=%s split=%s", idd_root, variant, split)

    if is_detection:
        records: list[dict] = []
        for xml_path in sorted(idd_root.rglob("*.xml")):
            if split not in xml_path.parts:
                continue
            for box in _parse_voc_xml(xml_path):
                box.update(
                    split=split,
                    variant=variant,
                    sequence_id=None,
                    frame_index=None,
                )
                records.append(box)
        df = pd.DataFrame(records, columns=list(RAW_DETECTION_COLUMNS))
    else:
        records = []
        for json_path in sorted(idd_root.rglob("*_polygons.json")):
            if split not in json_path.parts:
                continue
            for poly in _parse_polygon_json(json_path, idd_root):
                poly.update(split=split, variant=variant)
                records.append(poly)
        df = pd.DataFrame(records, columns=list(RAW_SEGMENTATION_COLUMNS))

    log.info("parsed %d raw label rows", len(df))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("detection", "temporal", "segmentation", "lite"),
    )
    parser.add_argument("--split", required=True, choices=IDD_SPLITS)
    parser.add_argument(
        "--idd-root",
        type=Path,
        default=None,
        help="path to an unpacked IDD variant; omit for synthetic stub data",
    )
    parser.add_argument("--peek", type=int, default=5)
    args = parser.parse_args(argv)

    df = extract_idd(args.variant, args.split, args.idd_root)
    print(df.head(args.peek).to_string(index=False))
    print(f"\n[stub] extracted {len(df)} raw label rows ({args.variant}/{args.split})")
    print("[stub] next: python -m etl.load.load_bronze --domain idd ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
