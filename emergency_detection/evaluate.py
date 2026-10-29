"""Test-set precision, recall, AP50, AP50:95 and a detection confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .labels import CLASS_NAMES


def _iou(box, boxes):
    if len(boxes) == 0:
        return torch.zeros(0)
    lt, rb = torch.maximum(box[:2], boxes[:, :2]), torch.minimum(box[2:], boxes[:, 2:])
    inter = (rb-lt).clamp(min=0).prod(dim=1)
    area_a = (box[2:]-box[:2]).clamp(min=0).prod()
    area_b = (boxes[:, 2:]-boxes[:, :2]).clamp(min=0).prod(dim=1)
    return inter/(area_a+area_b-inter).clamp(min=1e-9)


def evaluate_model(model, loader, device: torch.device, output_path: Path | None = None) -> dict:
    model.eval()
    predictions, truths = [], []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([im.to(device) for im in images])
            for output, target in zip(outputs, targets):
                truths.append((target["boxes"], target["labels"]))
                predictions.append((output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()))

    thresholds = [i/100 for i in range(50, 100, 5)]
    labels = range(1, len(CLASS_NAMES)+1)
    gt_counts = {c: sum(int((ls == c).sum()) for _, ls in truths) for c in labels}
    aps = {c: [] for c in labels}
    confusion = [[0 for _ in range(len(CLASS_NAMES)+1)] for _ in range(len(CLASS_NAMES)+1)]
    summary = {}
    for threshold in thresholds:
        records = {c: [] for c in labels}
        matched = [{c: set() for c in labels} for _ in truths]
        for image_i, (boxes, cls, scores) in enumerate(predictions):
            gt_boxes, gt_cls = truths[image_i]
            for pi in torch.argsort(scores, descending=True).tolist():
                c = int(cls[pi])
                if c not in records:
                    continue
                candidates = torch.where(gt_cls == c)[0]
                overlap = _iou(boxes[pi], gt_boxes[candidates]) if len(candidates) else torch.zeros(0)
                bi = int(overlap.argmax()) if len(overlap) else -1
                gi = int(candidates[bi]) if bi >= 0 else -1
                tp = bi >= 0 and float(overlap[bi]) >= threshold and gi not in matched[image_i][c]
                if tp:
                    matched[image_i][c].add(gi)
                records[c].append((float(scores[pi]), bool(tp), image_i, gi))
        for c in labels:
            rows = sorted(records[c], key=lambda r: r[0], reverse=True)
            tp = fp = 0
            ps, rs = [], []
            for _, is_tp, _, _ in rows:
                tp += int(is_tp); fp += int(not is_tp)
                ps.append(tp/max(1, tp+fp)); rs.append(tp/max(1, gt_counts[c]))
            aps[c].append(sum((rs[i]-(rs[i-1] if i else 0))*max(ps[i:]) for i in range(len(rows))))
            if threshold == .5:
                summary[c] = (tp, fp, rows)
                for _, is_tp, _, _ in rows:
                    confusion[c][c if is_tp else 0] += 1
                for image_i, (_, gt_cls) in enumerate(truths):
                    found = {gi for _, is_tp, ii, gi in rows if is_tp and ii == image_i}
                    for gi, gt_c in enumerate(gt_cls.tolist()):
                        if gt_c == c and gi not in found:
                            confusion[c][0] += 1

    per_class = {}
    total_tp = total_fp = total_fn = 0
    for c, name in enumerate(CLASS_NAMES, 1):
        tp, fp, rows = summary[c]
        total_tp += tp; total_fp += fp; total_fn += max(0, gt_counts[c]-tp)
        per_class[name] = {"ground_truth_count": gt_counts[c], "predictions": len(rows),
                           "precision": tp/max(1, tp+fp), "recall": tp/max(1, gt_counts[c]),
                           "ap50": aps[c][0], "ap50_95": sum(aps[c])/len(thresholds)}
    result = {"status": "evaluated", "precision": total_tp/max(1, total_tp+total_fp),
              "recall": total_tp/max(1, total_tp+total_fn),
              "map50": sum(x["ap50"] for x in per_class.values())/len(per_class),
              "map50_95": sum(x["ap50_95"] for x in per_class.values())/len(per_class),
              "per_class": per_class, "iou_thresholds": thresholds,
              "confusion_matrix_labels": ["background", *CLASS_NAMES], "confusion_matrix": confusion}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
