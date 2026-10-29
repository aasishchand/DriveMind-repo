"""Train the independent emergency detector from a prepared YOLO dataset."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import EmergencyVehicleDataset, _label_path, discover_images, split_images
from .evaluate import evaluate_model
from .model import MODEL_NAME, build_model

ROOT = Path(__file__).resolve().parents[1]


def collate(batch):
    return tuple(zip(*batch))


def train(data_root: Path, epochs: int = 20, batch_size: int = 2, learning_rate: float = .0005,
          seed: int = 23, output_dir: Path = ROOT/"emergency_detection"/"weights") -> dict:
    torch.manual_seed(seed)
    images = discover_images(data_root)
    if len(images) < 10:
        raise ValueError(f"Need at least 10 labeled images; found {len(images)} under {data_root}")
    present_classes = set()
    for image in images:
        label_path = _label_path(image)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing YOLO annotation: {label_path}")
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = line.split()
                if len(row) != 5 or not row[0].isdigit() or not 0 <= int(row[0]) < 4:
                    raise ValueError(f"Invalid canonical class/YOLO annotation at {label_path}:{line_no}")
                present_classes.add(int(row[0]))
    missing_classes = sorted(set(range(4))-present_classes)
    if missing_classes:
        raise ValueError(f"Dataset must include all four classes; missing canonical IDs {missing_classes}")
    splits = split_images(images, seed)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("Need nonempty train, validation, and test image splits")
    datasets = {name: EmergencyVehicleDataset(paths, augment=name == "train") for name, paths in splits.items()}
    loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=name == "train", num_workers=0,
                                collate_fn=collate) for name, ds in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=learning_rate, momentum=.9, weight_decay=.0005)
    history = []
    best_val = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir/"emergency_detector.pt"
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for images_batch, targets in loaders["train"]:
            loss_dict = model([im.to(device) for im in images_batch],
                              [{k: v.to(device) for k, v in t.items()} for t in targets])
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.train()
        val_losses = []
        for images_batch, targets in loaders["val"]:
            # torchvision detection models return losses in train mode.
            with torch.no_grad():
                loss_dict = model([im.to(device) for im in images_batch],
                                  [{k: v.to(device) for k, v in t.items()} for t in targets])
            val_losses.append(float(sum(loss_dict.values())))
        train_loss = sum(losses)/max(1, len(losses))
        val_loss = sum(val_losses)/max(1, len(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "model_name": MODEL_NAME,
                        "epoch": epoch, "class_names": ["background", "ambulance", "fire_truck",
                        "police_vehicle", "ordinary_vehicle"], "seed": seed,
                        "created_utc": datetime.now(timezone.utc).isoformat()}, checkpoint)
        print(f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} validation_loss={val_loss:.5f}")
    hist_path = output_dir/"training_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    best = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    metrics = evaluate_model(model, loaders["test"], device, output_dir/"evaluation.json")
    metadata = {"status": "trained_and_evaluated", "model_name": MODEL_NAME, "device": str(device),
                "images": {k: len(v) for k, v in splits.items()}, "checkpoint": str(checkpoint),
                "history": str(hist_path), "evaluation": str(output_dir/"evaluation.json"),
                "best_epoch": best["epoch"], "metrics": metrics}
    (output_dir/"training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset root containing images/ and labels/ split folders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT/"emergency_detection"/"weights")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.epochs, args.batch_size, output_dir=args.output), indent=2))


if __name__ == "__main__":
    main()
