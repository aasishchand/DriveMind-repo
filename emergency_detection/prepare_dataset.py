"""Merge Roboflow YOLO emergency labels with ordinary vehicles from COCO JSON."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .labels import normalize_label


def _paired_label(root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(root)
    parts = list(rel.parts)
    image_indices = [i for i, value in enumerate(parts) if value == "images"]
    if not image_indices:
        raise ValueError(f"Image path must contain an images directory: {image_path}")
    parts[image_indices[-1]] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def prepare(emergency_root: Path, coco_images: Path, coco_annotations: Path, output: Path,
            emergency_class_names: list[str]):
    """Copy emergency dataset and COCO car/bus/truck boxes into canonical YOLO IDs."""
    source_names = [normalize_label(n) for n in emergency_class_names]
    required = {"ambulance", "fire_truck", "police_vehicle"}
    if not required.issubset(source_names):
        raise ValueError(f"emergency class names must include {sorted(required)}")
    source_to_target = {i: ("ambulance", "fire_truck", "police_vehicle", "ordinary_vehicle").index(name)
                        for i, name in enumerate(source_names)}
    for image_dir in emergency_root.rglob("images"):
        for source_image in image_dir.iterdir():
            if source_image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            rel = source_image.relative_to(emergency_root)
            target_image = output/"emergency"/rel
            target_label = _paired_label(output/"emergency", target_image)
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_image)
            source_label = _paired_label(emergency_root, source_image)
            if not source_label.exists():
                raise FileNotFoundError(f"Missing label file {source_label}")
            rows = []
            for row in source_label.read_text(encoding="utf-8").splitlines():
                fields = row.split()
                if fields:
                    class_id = int(fields[0])
                    if class_id not in source_to_target:
                        raise ValueError(f"class id {class_id} is not in supplied --emergency-class-names")
                    rows.append(" ".join([str(source_to_target[class_id]), *fields[1:]]))
            target_label.write_text("\n".join(rows)+"\n", encoding="utf-8")

    coco = json.loads(coco_annotations.read_text(encoding="utf-8"))
    wanted = {c["id"] for c in coco["categories"] if c["name"] in {"car", "bus", "truck"}}
    category_ids = wanted
    by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in category_ids and ann.get("iscrowd", 0) == 0:
            by_image.setdefault(ann["image_id"], []).append(ann["bbox"])
    info = {x["id"]: x for x in coco["images"]}
    for image_id, boxes in by_image.items():
        item = info[image_id]
        src = coco_images/item["file_name"]
        if not src.is_file():
            continue
        dst = output/"coco"/"images"/item["file_name"]
        label = output/"coco"/"labels"/Path(item["file_name"]).with_suffix(".txt")
        dst.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows = []
        width, height = item["width"], item["height"]
        for x, y, w, h in boxes:
            cx, cy = (x+w/2)/width, (y+h/2)/height
            rows.append(f"3 {cx:.8f} {cy:.8f} {w/width:.8f} {h/height:.8f}")
        label.write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(f"Prepared YOLO image/label folders under {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emergency-root", type=Path, required=True,
                        help="Roboflow YOLO export containing images/ and labels/ folders")
    parser.add_argument("--emergency-class-names", required=True,
                        help="comma separated names in source class ID order, e.g. Ambulance,Fire Engine,Police Vehicle")
    parser.add_argument("--coco-images", type=Path, required=True, help="COCO train2017 image directory")
    parser.add_argument("--coco-annotations", type=Path, required=True, help="instances_train2017.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.emergency_root, args.coco_images, args.coco_annotations, args.output,
            [v.strip() for v in args.emergency_class_names.split(",")])


if __name__ == "__main__":
    main()
