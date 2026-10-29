"""Merge emergency, general-vehicle, and ITAWD YOLO exports to DriveMind IDs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .labels import normalize_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EMERGENCY_IDS = {"ambulance": 0, "fire_truck": 1, "police_vehicle": 2}
ORDINARY_NAMES = {
    "car", "truck", "bus", "motorbike", "motorcycle", "microbus",
    "pickup_van", "van", "rickshaw", "auto_rickshaw", "autorickshaw",
    "three_wheeler", "three_wheeler_auto", "taxi", "suv",
}


def _names(root: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install project dependencies with pip install -r requirements.txt") from exc
    configs = list(root.rglob("data.yaml"))
    if len(configs) != 1:
        raise ValueError(f"Expected exactly one data.yaml under {root}; found {len(configs)}")
    config = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[i]) for i in range(len(names))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"No class-name list found in {configs[0]}")


def _canonical_map(names: list[str], source: str) -> dict[int, int | None]:
    mapping = {}
    for index, name in enumerate(names):
        if source == "emergency":
            canonical = normalize_label(name)
            if canonical not in EMERGENCY_IDS:
                raise ValueError(f"Unexpected emergency class {name!r}; expected ambulance/firetruck/police")
            mapping[index] = EMERGENCY_IDS[canonical]
        else:
            key = "_".join(name.strip().lower().replace("-", " ").split())
            mapping[index] = 3 if key in ORDINARY_NAMES else None
    if source == "emergency" and set(mapping.values()) != set(EMERGENCY_IDS.values()):
        raise ValueError(f"Emergency export must contain all three emergency classes; found {names}")
    if source != "emergency" and 3 not in mapping.values():
        raise ValueError(f"No supported road-vehicle classes found in {names}")
    return mapping


def _label_for(root: Path, image: Path) -> Path:
    parts = list(image.relative_to(root).parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path must contain an images directory: {image}")
    parts[indices[-1]] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def _to_box(fields: list[str], label_path: Path, line_number: int) -> list[str]:
    """Keep YOLO boxes, or enclose a YOLO polygon in its axis-aligned box."""
    if len(fields) == 5:
        return fields
    coords = [float(value) for value in fields[1:]]
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError(f"Expected YOLO box or polygon at {label_path}:{line_number}")
    xs, ys = coords[0::2], coords[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [fields[0], f"{(x1+x2)/2:.8f}", f"{(y1+y2)/2:.8f}",
            f"{x2-x1:.8f}", f"{y2-y1:.8f}"]


def _merge(root: Path, output: Path, source_name: str, mapping: dict[int, int | None]) -> int:
    copied = 0
    for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        # Roboflow split export images are paired with labels through their images/labels path.
        source_label = _label_for(root, image)
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing YOLO label: {source_label}")
        target_image = output / source_name / image.relative_to(root)
        target_label = _label_for(output / source_name, target_image)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            source_id = int(fields[0])
            if source_id not in mapping:
                raise ValueError(f"Unknown class ID {source_id} at {source_label}:{line_number}")
            target_id = mapping[source_id]
            if target_id is not None:
                box = _to_box(fields, source_label, line_number)
                rows.append(" ".join([str(target_id), *box[1:]]))
        shutil.copy2(image, target_image)
        target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        copied += 1
    return copied


def combine(emergency_root: Path, vehicle_root: Path, itawd_root: Path, output: Path) -> dict[str, int]:
    """Copy all labeled images while mapping to 0 ambulance, 1 fire, 2 police, 3 ordinary."""
    sources = (
        ("emergency", emergency_root, "emergency"),
        ("general-vehicles", vehicle_root, "ordinary"),
        ("itawd", itawd_root, "ordinary"),
    )
    counts = {}
    for dest_name, root, source_kind in sources:
        names = _names(root)
        mapping = _canonical_map(names, source_kind)
        print(f"{dest_name} classes: {names}")
        counts[dest_name] = _merge(root, output, dest_name, mapping)
    print(f"Combined dataset written to {output}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emergency", type=Path, required=True, help="Roboflow emergency YOLOv8 export")
    parser.add_argument("--vehicles", type=Path, required=True, help="general vehicle YOLOv8 export")
    parser.add_argument("--itawd", type=Path, required=True, help="ITAWD YOLOv8 export")
    parser.add_argument("--output", type=Path, required=True, help="combined YOLO image/label root")
    args = parser.parse_args()
    combine(args.emergency, args.vehicles, args.itawd, args.output)


if __name__ == "__main__":
    main()
