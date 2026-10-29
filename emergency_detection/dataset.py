"""YOLO-format dataset reader and duplicate-safe train/val/test preparation."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .labels import CLASS_NAMES, CLASS_TO_ID, normalize_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
                  and "images" in p.parts and p.is_file())


def split_images(images: list[Path], seed: int = 23) -> dict[str, list[Path]]:
    """Hash-identical files stay together; deterministic 70/15/15 split.

    Group by file size first. Identical files necessarily have identical sizes,
    so only sizes shared by multiple images need full SHA-256 reads.
    """
    by_size: dict[int, list[Path]] = {}
    for path in images:
        by_size.setdefault(path.stat().st_size, []).append(path)
    groups: dict[str, list[Path]] = {}
    for size, paths in by_size.items():
        if len(paths) == 1:
            groups[f"unique:{size}:{paths[0]}"] = paths
            continue
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    a, b = int(n * .70), int(n * .85)
    return {name: [p for k in part for p in groups[k]] for name, part in
            (("train", keys[:a]), ("val", keys[a:b]), ("test", keys[b:]))}


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


class EmergencyVehicleDataset(Dataset):
    """Reads standard YOLO rows: class_id x_center y_center width height (normalized)."""
    def __init__(self, images: list[Path], *, augment: bool = False):
        self.images, self.augment = images, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            from PIL import Image, ImageEnhance, ImageOps
        except ImportError as exc:
            raise RuntimeError("Pillow is required for emergency detector training; install -r requirements.txt") from exc
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        rows = []
        label_path = _label_path(path)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO annotation: {label_path}")
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"{label_path}:{line_no}: expected class_id cx cy width height")
            cls, cx, cy, bw, bh = int(values[0]), *map(float, values[1:])
            if not 0 <= cls < len(CLASS_NAMES) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"Invalid annotation at {label_path}:{line_no}")
            rows.append((max(0, (cx-bw/2)*width), max(0, (cy-bh/2)*height),
                         min(width, (cx+bw/2)*width), min(height, (cy+bh/2)*height), cls+1))
        boxes = torch.tensor([r[:4] for r in rows], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([r[4] for r in rows], dtype=torch.int64)
        if self.augment and random.random() < .5:
            image = ImageOps.mirror(image)
            if len(boxes):
                old_left, old_right = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0], boxes[:, 2] = width-old_right, width-old_left
        if self.augment and random.random() < .5:
            image = ImageEnhance.Color(image).enhance(random.uniform(.75, 1.25))
        import numpy as np
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([index]),
                  "area": area, "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)}
        return tensor, target


def validate_class_names(names: list[str]) -> dict[int, int]:
    """Map source class IDs in names order to DriveMind IDs; reject missing classes."""
    normalized = [normalize_label(name) for name in names]
    missing = set(CLASS_TO_ID) - set(normalized)
    if missing:
        raise ValueError(f"Dataset is missing required classes: {sorted(missing)}")
    return {i: CLASS_TO_ID[label] for i, label in enumerate(normalized)}
