"""Small IoU tracker and image-space motion/risk estimates for camera detections."""
from __future__ import annotations

import math


class DetectionTracker:
    def __init__(self, max_missed: int = 5, iou_threshold: float = .2):
        self.max_missed, self.iou_threshold = max_missed, iou_threshold
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1)*max(0, y2-y1)
        area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
        area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
        return inter/max(1e-9, area_a+area_b-inter)

    def update(self, detections: list[dict]) -> list[dict]:
        used = set()
        output = []
        for detection in detections:
            candidates = [(self._iou(detection["bbox"], tr["bbox"]), tid, tr)
                          for tid, tr in self.tracks.items()
                          if tid not in used and tr["class"] == detection["class"]]
            overlap, track_id, track = max(candidates, default=(0.0, None, None))
            if overlap < self.iou_threshold:
                track_id, track = self.next_id, {"class": detection["class"], "bbox": detection["bbox"],
                                                  "missed": 0, "distance_m": None, "previous_distance_m": None}
                self.next_id += 1
                self.tracks[track_id] = track
            used.add(track_id)
            item = dict(detection, track_id=track_id)
            track.update({"bbox": detection["bbox"], "missed": 0})
            output.append(item)
        for track_id in list(self.tracks):
            if track_id not in used:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed:
                    del self.tracks[track_id]
        return output


def estimate_distance_ttc(detection: dict, frame_width: int, frame_height: int,
                          horizontal_fov_deg: float, delta_s: float,
                          previous: dict[int, tuple[float, float]]) -> dict:
    """Estimate range from apparent vehicle height; TTC is returned only after two samples."""
    x1, y1, x2, y2 = detection["bbox"]
    box_height = max(1.0, y2-y1)
    fy = (frame_width/2) / math.tan(math.radians(horizontal_fov_deg)/2)
    # Approximate physical road vehicle height; this is a camera-model estimate, not ground truth.
    range_m = max(1.0, fy*2.3/box_height)
    prior = previous.get(detection["track_id"])
    closing = None
    ttc = None
    if prior and delta_s > 0:
        closing = (prior[0]-range_m)/delta_s
        if closing > .1:
            ttc = range_m/closing
    previous[detection["track_id"]] = (range_m, box_height)
    center_x = (x1+x2)/2
    return {"distance_m": range_m, "closing_speed_mps": closing, "ttc_s": ttc,
            "relative_position": "left" if center_x < frame_width*.4 else
                                "right" if center_x > frame_width*.6 else "ahead",
            "path_overlap": frame_width*.35 <= center_x <= frame_width*.65}
