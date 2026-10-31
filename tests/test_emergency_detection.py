import numpy as np
import pytest
import torch

from emergency_detection.inference import EmergencyDetector, draw_detections
from emergency_detection.labels import CLASS_NAMES, CLASS_TO_ID, normalize_label
from emergency_detection.tracking import DetectionTracker, estimate_distance_ttc
from emergency_detection.evaluate import evaluate_model
from etl.simulation.core import Actor, DrivingSimulation

CAMERA_SCENARIOS = {
    "Approaching ambulance", "Police vehicle crossing intersection", "Fire truck ahead",
    "Ordinary vehicle negative control", "Partially occluded emergency vehicle",
    "Low-confidence false-positive case",
}


def test_draw_detections_renders_boxes_without_changing_source():
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    result = draw_detections(source, [{"class": "ambulance", "confidence": 0.9,
                                       "bbox": [10, 20, 40, 40]}])
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert result.any()
    assert not source.any()


def test_dedicated_detector_is_explicitly_untrained_without_checkpoint(tmp_path):
    detector = EmergencyDetector(tmp_path/"missing.pt")
    assert detector.status == "UNTRAINED"
    assert detector.predict(np.zeros((32, 32, 3), dtype=np.uint8)) == []


def test_class_mapping_keeps_ordinary_vehicle_non_emergency():
    assert normalize_label("Fire Engine") == "fire_truck"
    assert normalize_label("Police Car") == "police_vehicle"
    assert normalize_label("car") == "ordinary_vehicle"
    assert set(CLASS_TO_ID) == set(CLASS_NAMES)
    with pytest.raises(ValueError):
        normalize_label("unknown")


def test_confidence_filter_and_detection_output_format(tmp_path):
    detector = EmergencyDetector(tmp_path/"missing.pt", threshold=.6)
    assert detector.predict(np.zeros((12, 20, 3), dtype=np.uint8)) == []
    class Model:
        def __call__(self, images):
            return [{"boxes": torch.tensor([[1., 2., 8., 9.], [0., 0., 4., 4.]]),
                     "labels": torch.tensor([1, 4]), "scores": torch.tensor([.9, .2])}]
    detector.model = Model()
    detections = detector.predict(np.zeros((12, 20, 3), dtype=np.uint8))
    assert detections == [{"class": "ambulance", "confidence": pytest.approx(.9),
                           "bbox": [1.0, 2.0, 8.0, 9.0]}]
    with pytest.raises(ValueError):
        detector.predict(np.zeros((12, 20), dtype=np.uint8))

    class FailedModel:
        def __call__(self, images):
            raise RuntimeError("inference failure")
    detector.model = FailedModel()
    assert detector.predict(np.zeros((12, 20, 3), dtype=np.uint8)) == []
    assert detector.status == "ERROR"


def test_tracker_and_relative_motion_estimate():
    tracker = DetectionTracker()
    first = tracker.update([{"class": "ambulance", "confidence": .8, "bbox": [100, 100, 180, 180]}])
    previous = {}
    estimate_distance_ttc(first[0], 1280, 720, 90, .1, previous)
    second = tracker.update([{"class": "ambulance", "confidence": .9, "bbox": [102, 110, 182, 198]}])
    motion = estimate_distance_ttc(second[0], 1280, 720, 90, .1, previous)
    assert first[0]["track_id"] == second[0]["track_id"]
    assert motion["distance_m"] > 0
    assert motion["closing_speed_mps"] is not None


def test_implausible_camera_range_jump_is_not_sent_to_planner():
    previous = {}
    detection = {"track_id": 1, "bbox": [100, 100, 180, 180]}
    estimate_distance_ttc(detection, 1280, 720, 90, .1, previous)
    detection["bbox"] = [102, 110, 182, 200]
    motion = estimate_distance_ttc(detection, 1280, 720, 90, .1, previous)
    assert motion["closing_speed_mps"] is None
    assert motion["ttc_s"] is None


def test_camera_emergency_actor_reaches_risk_planner_without_forced_maneuver():
    sim = DrivingSimulation("Normal traffic", "DRIVEMIND")
    sim.actors = [Actor("camera-1", "emergency", 5, 5, 0, emergency=True)]
    telemetry = sim.step(.1)
    assert telemetry.risk_level == "HIGH"
    assert telemetry.action == "BRAKE"


def test_ordinary_camera_vehicle_does_not_trigger_emergency_yield():
    sim = DrivingSimulation("Normal traffic", "DRIVEMIND")
    sim.actors = [Actor("camera-ordinary", "car", 20, 15, 0, emergency=False)]
    telemetry = sim.step(.1)
    assert telemetry.action != "YIELD — CLEAR EMERGENCY LANE"
    assert telemetry.emergency_status == "NONE"


def test_documented_camera_scenario_suite_is_complete():
    from pathlib import Path
    readme = Path(__file__).parents[1]/"README.md"
    content = " ".join(readme.read_text(encoding="utf-8").split())
    assert all(name in content for name in CAMERA_SCENARIOS)


def test_detection_metrics_and_confusion_matrix_from_predictions():
    class PerfectDetector:
        def eval(self):
            return self
        def __call__(self, images):
            return [{"boxes": torch.tensor([[1., 1., 9., 9.]]),
                     "labels": torch.tensor([1]), "scores": torch.tensor([.95])}]
    targets = {"boxes": torch.tensor([[1., 1., 9., 9.]]), "labels": torch.tensor([1])}
    metrics = evaluate_model(PerfectDetector(), [([torch.zeros(3, 10, 10)], [targets])], torch.device("cpu"))
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["map50"] == .25
    assert metrics["per_class"]["ambulance"]["ap50_95"] == 1
    assert metrics["confusion_matrix"][1][1] == 1
