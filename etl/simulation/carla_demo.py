"""Run the DriveMind risk planner against a live CARLA server.

Run with the CARLA-matched Python environment (the repository's .venv-carla):
    .venv-carla/Scripts/python -m etl.simulation.carla_demo --duration 20

The CARLA Traffic Manager drives surrounding traffic. DriveMind controls the
ego vehicle using local lane/actor state from CARLA, and writes actual CARLA
records plus planner telemetry for the existing ETL pipeline.
"""
from __future__ import annotations

import argparse
import base64
import csv
import itertools
from io import BytesIO
import json
import math
import os
import queue
from datetime import datetime
from pathlib import Path
import random
import signal
import time
import tkinter as tk
from tkinter import ttk

import carla

from etl.config import REPO_ROOT
from etl.prediction.inference import TrajectoryPredictor
from etl.simulation.core import Actor, DrivingSimulation, MODES, SCENARIOS, telemetry_dict
from etl.policy_learning.risk_model import CollisionRiskModel
from etl.policy_learning.lane_model import LaneChoiceModel
from etl.policy_learning.steering_model import SteeringModel
from emergency_detection.inference import EmergencyDetector
from emergency_detection.tracking import DetectionTracker, estimate_distance_ttc


DT = 0.05
TM_PORT = 8000
STOP_REQUESTED = False


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value
EMERGENCY_BITS = int(carla.VehicleLightState.Special1) | int(carla.VehicleLightState.Special2)
CAMERA_SCENARIOS = (
    "Approaching ambulance", "Police vehicle crossing intersection", "Fire truck ahead",
    "Ordinary vehicle negative control", "Partially occluded emergency vehicle",
    "Low-confidence false-positive case",
)


class CameraPreview:
    """Live camera and detector status panel; gracefully disabled on headless hosts."""
    def __init__(self):
        self.root = self.label = self.photo = None
        self.closed = False
        self.closed_by_user = False
        try:
            self.root = tk.Tk()
            self.root.title("DriveMind | Emergency Camera Detector")
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.root.configure(bg="#101827")
            self.canvas = tk.Canvas(self.root, width=1280, height=360, bg="#101827", highlightthickness=0)
            self.canvas.pack(padx=8, pady=8)
            self.status = ttk.Label(self.root, text="Waiting for camera", justify="left")
            self.status.pack(fill="x", padx=10, pady=8)
        except Exception:
            self.root = None

    def show(self, frames, detections, detector, telemetry=None, sim_truth=()):
        if self.root is None:
            return
        try:
            import numpy as np
            self.canvas.delete("all")
            photos = []
            for view, offset_x in (("front", 0), ("rear", 640)):
                frame = frames.get(view)
                if frame is None:
                    continue
                small = frame[np.linspace(0, frame.shape[0]-1, 360).astype(int)[:, None],
                              np.linspace(0, frame.shape[1]-1, 640).astype(int)]
                png = BytesIO()
                from PIL import Image
                Image.fromarray(small).save(png, format="PNG", compress_level=1)
                photo = tk.PhotoImage(data=base64.b64encode(png.getvalue()).decode("ascii"), format="PNG")
                photos.append(photo)
                self.canvas.create_image(offset_x, 0, image=photo, anchor="nw")
                self.canvas.create_text(offset_x+8, 8, anchor="nw", fill="white", text=view.upper())
                for detection in detections:
                    if detection.get("camera_view") != view:
                        continue
                    x1, y1, x2, y2 = [v*.5 for v in detection["bbox"]]
                    self.canvas.create_rectangle(offset_x+x1, y1, offset_x+x2, y2, outline="#ff505a", width=2)
                    self.canvas.create_text(offset_x+x1, max(8, y1-8), anchor="sw", fill="#ff505a",
                        text=f"{detection['class'].replace('_',' ').upper()} {detection['confidence']:.2f}")
            self.photos = photos
            self.photo = photos
            emergency = [d for d in detections if d["class"] != "ordinary_vehicle"]
            details = "\n".join(f"{d['class'].replace('_',' ').upper()} {d['confidence']:.3f} "
                                 f"bbox={tuple(round(x) for x in d['bbox'])}"
                                 for d in detections) or "No camera detections above threshold"
            risk = f"{telemetry.risk_level} {telemetry.risk_score:.0f}/100" if telemetry else "—"
            action = telemetry.action if telemetry else "—"
            timing = (f"{detector.latency_ms:.1f} ms | {1000/max(1.0, detector.latency_ms):.1f} FPS"
                      if detector.status == "READY" else "latency — | FPS —")
            truth_text = ("CARLA SIM GROUND TRUTH (not vision): " + ", ".join(sim_truth)
                          if sim_truth else "CARLA SIM GROUND TRUTH: no emergency actors")
            self.status.configure(text=f"{detector.status} | {detector.__class__.__name__} | "
                f"{len(emergency)} emergency vehicles | {timing} | Risk {risk} | Planner: {action}\n"
                f"{details}\n{truth_text}")
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError as exc:
            print(f"[camera preview] window unavailable: {exc}")
            self.closed = True
            self.root = None

    def close(self):
        self.closed = True
        self.closed_by_user = True
        if self.root is not None:
            try:
                self.root.destroy()
            except tk.TclError:
                pass


def _next(wp, distance: float, *, backwards: bool = False):
    points = wp.previous(distance) if backwards else wp.next(distance)
    return points[0] if points else None


def _driving_neighbor(wp, direction: str):
    candidate = wp.get_right_lane() if direction == "right" else wp.get_left_lane()
    if candidate and candidate.lane_type == carla.LaneType.Driving:
        yaw_delta = abs((candidate.transform.rotation.yaw-wp.transform.rotation.yaw+180) % 360-180)
        if yaw_delta > 45.0:
            return None
        return candidate
    return None


def _spawn_vehicle(world, blueprint, wp, role: str):
    transform = wp.transform
    transform.location.z += 0.5
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role)
    return world.try_spawn_actor(blueprint, transform)


def _choose_start(world, *, junction_ahead: bool = False):
    carla_map = world.get_map()
    points = list(carla_map.get_spawn_points())
    random.shuffle(points)
    for transform in points:
        wp = carla_map.get_waypoint(transform.location, project_to_road=True)
        has_junction = (not junction_ahead or any(
            (candidate := _next(wp, distance)) is not None and candidate.is_junction
            for distance in range(10, 121, 10))) if wp else False
        if wp and has_junction and _next(wp, 50) and _next(wp, 45, backwards=True) and (
            _driving_neighbor(wp, "right") or _driving_neighbor(wp, "left")
        ):
            return wp
    raise RuntimeError("could not find a CARLA spawn point with 45m of road and an adjacent lane")


def _choose_straight_start(world, length: float = 100.0, behind: float = 0.0):
    """Find a long, non-junction stretch suitable for repeated lane changes."""
    carla_map = world.get_map()
    points = list(carla_map.get_spawn_points())
    random.shuffle(points)
    for transform in points:
        wp = carla_map.get_waypoint(transform.location, project_to_road=True)
        if not wp or wp.is_junction or not (_driving_neighbor(wp, "left") or _driving_neighbor(wp, "right")):
            continue
        if behind and not _next(wp, behind, backwards=True):
            continue
        start_yaw = wp.transform.rotation.yaw
        straight = True
        for distance in range(10, int(length) + 1, 10):
            candidate = _next(wp, distance)
            if not candidate or candidate.is_junction:
                straight = False
                break
            yaw_delta = abs((candidate.transform.rotation.yaw-start_yaw+180) % 360-180)
            if yaw_delta > 12 or not (_driving_neighbor(candidate, "left") or
                                      _driving_neighbor(candidate, "right")):
                straight = False
                break
        if straight:
            return wp
    raise RuntimeError(f"could not find {length:.0f}m of straight road with an adjacent lane")


def _crossing_approach(world_map, start_wp):
    junction = next((_next(start_wp, distance) for distance in range(10, 121, 5)
                     if _next(start_wp, distance) and _next(start_wp, distance).is_junction), None)
    if junction is None:
        return None
    origin = junction.transform.location
    ego_yaw = start_wp.transform.rotation.yaw
    choices = []
    for candidate in world_map.generate_waypoints(3.0):
        if candidate.lane_type != carla.LaneType.Driving:
            continue
        loc = candidate.transform.location
        if loc.distance(origin) > 28 or not candidate.is_junction:
            continue
        yaw_delta = abs((candidate.transform.rotation.yaw-ego_yaw+180) % 360-180)
        if 55 <= yaw_delta <= 125:
            previous = _next(candidate, 8, backwards=True)
            entry = previous or candidate
            choices.append((loc.distance(origin), entry))
    return min(choices, key=lambda x: x[0])[1] if choices else None


def _relative_state(ego, actors, lane0_id, lane1_id):
    transform = ego.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    velocity = ego.get_velocity()
    ego_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    snapshots = []
    for actor_id, actor, emergency, obstacle in actors:
        if not actor.is_alive:
            continue
        other_loc = actor.get_location()
        dx, dy = other_loc.x-loc.x, other_loc.y-loc.y
        s = dx*fx + dy*fy
        v = actor.get_velocity()
        longitudinal_speed = v.x*fx + v.y*fy
        waypoint = ego.get_world().get_map().get_waypoint(other_loc, project_to_road=True)
        if obstacle:
            lane = 0
        elif waypoint and waypoint.lane_id == lane0_id:
            lane = 0
        elif waypoint and waypoint.lane_id == lane1_id:
            lane = 1
        else:
            continue
        # CARLA emergency flags/blueprints are retained only in ground-truth raw
        # telemetry; they are deliberately not passed to the planner as perception.
        kind = "obstacle" if obstacle else "car"
        snapshots.append((Actor(actor_id, kind, s, longitudinal_speed, lane, False), actor))
    return ego_speed, snapshots


def run(host: str = "127.0.0.1", port: int = 2000, duration: float | None = 20,
        scenario: str = "Emergency vehicle approaching", mode: str = "DRIVEMIND",
        seed: int = 41, camera_preview: bool = True, risk_model=None,
        lane_model=None, steering_model=None, stop_file: Path | None = None,
        pause_file: Path | None = None,
        web_frame_dir: Path | None = None) -> tuple[Path, Path, dict]:
    random.seed(seed)
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    tm = client.get_trafficmanager(TM_PORT)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    spawned: list[carla.Actor] = []
    sensors: list[carla.Actor] = []
    raw_rows: list[dict] = []
    ai_rows: list[dict] = []
    collision_frames: set[int] = set()
    collision_events: list[dict] = []
    try:
        bp_lib = world.get_blueprint_library()
        junction_case = scenario == "Police vehicle crossing intersection"
        if scenario == "Four emergency vehicles":
            base_wp = _choose_straight_start(world, length=100.0, behind=190.0)
            # Begin beyond the initial spawn area so fixed roadside map props
            # near the CARLA spawn point do not sit directly in the demo path.
            start_wp = _next(base_wp, 50.0) or base_wp
        elif junction_case:
            start_wp = _choose_start(world, junction_ahead=True)
        else:
            # The regular spawn-point selector can choose a tight curved road;
            # those are a poor fit for this demo's lane-change controller.
            start_wp = _choose_straight_start(world, length=100.0)
        adjacent = _driving_neighbor(start_wp, "right")
        adjacent_direction = "right"
        if adjacent is None:
            adjacent = _driving_neighbor(start_wp, "left")
            adjacent_direction = "left"
        if adjacent is None:
            raise RuntimeError("adjacent driving lane became unavailable")
        lane0, lane1 = start_wp.lane_id, adjacent.lane_id

        ego = _spawn_vehicle(world, bp_lib.find("vehicle.tesla.model3"), start_wp, "drivemind_ego")
        if ego is None:
            raise RuntimeError("could not spawn CARLA ego vehicle")
        spawned.append(ego)
        ego.set_autopilot(False)

        others: list[tuple[str, carla.Actor, bool, bool]] = []

        def spawn_at(distance: float, lane_wp, actor_name: str, *, emergency=False, obstacle=False,
                     slower=False, vehicle_type: str | None = None):
            target = lane_wp if distance == 0 else _next(
                lane_wp, abs(distance), backwards=distance < 0)
            if target is None:
                return
            bp_pattern = vehicle_type or ("vehicle.carlamotors.firetruck" if emergency else "vehicle.audi.a2")
            try:
                bp = bp_lib.find(bp_pattern)
            except Exception:
                bp = random.choice(bp_lib.filter("vehicle.*"))
            actor = _spawn_vehicle(world, bp, target, actor_name)
            if actor is None:
                return
            spawned.append(actor)
            if not obstacle:
                actor.set_autopilot(True, TM_PORT)
                if slower:
                    speed_difference = 20.0 if scenario == "Dense traffic lane changes" else 45.0
                    tm.vehicle_percentage_speed_difference(actor, speed_difference)
                if emergency:
                    speed_difference = -80.0 if scenario == "Four emergency vehicles" else -70.0
                    tm.vehicle_percentage_speed_difference(actor, speed_difference)
                    try:
                        actor.set_light_state(carla.VehicleLightState(EMERGENCY_BITS))
                    except Exception:
                        pass
            others.append((actor_name, actor, emergency, obstacle))

        lane1_wp = adjacent
        if scenario in ("Emergency vehicle approaching", "Approaching ambulance"):
            # Give perception and planning more time to recognize the siren
            # vehicle and clear a safe passing gap before it reaches the ego.
            # Bring the emergency actor into view sooner while leaving enough
            # room for detection and a controlled lane-clear maneuver.
            previous = _next(start_wp, 55, backwards=True)
            if previous:
                vehicle_type = "vehicle.ford.ambulance" if scenario == "Approaching ambulance" else "vehicle.carlamotors.firetruck"
                bp = bp_lib.find(vehicle_type)
                emergency_actor = _spawn_vehicle(world, bp, previous, "drivemind_emergency")
                if emergency_actor:
                    spawned.append(emergency_actor)
                    emergency_actor.set_autopilot(True, TM_PORT)
                    tm.vehicle_percentage_speed_difference(emergency_actor, -70.0)
                    try:
                        emergency_actor.set_light_state(carla.VehicleLightState(EMERGENCY_BITS))
                    except Exception:
                        pass
                    others.append(("E-01", emergency_actor, True, False))
            spawn_at(75, lane1_wp, "V-01")
            spawn_at(115, start_wp, "V-02")
        elif scenario == "Police vehicle crossing intersection":
            crossing_wp = _crossing_approach(carla_map, start_wp)
            if crossing_wp:
                spawn_at(0, crossing_wp, "E-POLICE", emergency=True,
                         vehicle_type="vehicle.dodge.charger_police")
            else:
                print("[carla_demo] no perpendicular junction approach found; spawning police vehicle on adjacent lane")
                spawn_at(25, lane1_wp, "E-POLICE", emergency=True,
                         vehicle_type="vehicle.dodge.charger_police")
        elif scenario == "Fire truck ahead":
            spawn_at(32, start_wp, "E-FIRE", emergency=True, slower=True)
        elif scenario == "Ordinary vehicle negative control":
            spawn_at(28, start_wp, "V-ORDINARY", slower=True)
        elif scenario == "Partially occluded emergency vehicle":
            spawn_at(42, start_wp, "E-OCCLUDED", emergency=True, slower=True)
            spawn_at(22, start_wp, "V-OCCLUDER", slower=True)
        elif scenario == "Low-confidence false-positive case":
            spawn_at(32, start_wp, "V-LOW-CONFIDENCE", vehicle_type="vehicle.audi.a2")
        elif scenario == "Slower vehicle ahead":
            spawn_at(26, start_wp, "V-01", slower=True)
            spawn_at(58, lane1_wp, "V-02")
        elif scenario == "Obstacle ahead":
            spawn_at(28, start_wp, "O-01", obstacle=True)
            spawn_at(55, lane1_wp, "V-01")
        elif scenario == "Traffic congestion":
            for i, distance in enumerate((18, 30, 42, 54, 66, 78)):
                spawn_at(distance, start_wp if i % 2 == 0 else lane1_wp, f"V-{i+1:02d}", slower=True)
        elif scenario == "Dense traffic lane changes":
            # Alternating slow traffic creates repeated, spaced opportunities
            # for the ego to pass between lanes without starting vehicles close
            # enough to overlap or force an immediate collision.
            positions = ((20, 0), (38, 1), (56, 0), (74, 1), (92, 0))
            for i, (distance, lane) in enumerate(positions):
                spawn_at(distance, start_wp if lane == 0 else lane1_wp,
                         f"V-{i+1:02d}", slower=True)
        elif scenario == "Four emergency vehicles":
            emergency_specs = (
                (-35, "E-AMBULANCE", "vehicle.ford.ambulance"),
                (-70, "E-POLICE", "vehicle.dodge.charger_police"),
                (-105, "E-FIRE", "vehicle.carlamotors.firetruck"),
                (-140, "E-AMBULANCE-2", "vehicle.ford.ambulance"),
            )
            for distance, name, blueprint in emergency_specs:
                # Keep emergency traffic in the adjacent lane so the faster
                # vehicles can pass visibly without forcing repeated ego turns.
                spawn_at(distance, lane1_wp, name, emergency=True, vehicle_type=blueprint)
            emergency_count = sum(is_emergency for _, _, is_emergency, _ in others)
            if emergency_count != 4:
                raise RuntimeError(f"four-emergency scenario spawned {emergency_count}/4 emergency vehicles")
        else:
            spawn_at(44, start_wp, "V-01")
            spawn_at(70, lane1_wp, "V-02")
            spawn_at(98, lane1_wp, "V-03", slower=True)

        print(f"[carla_demo] spawned {sum(is_emergency for _, _, is_emergency, _ in others)} "
              f"emergency and {sum(not is_emergency for _, _, is_emergency, _ in others)} "
              f"other traffic actors")

        def on_collision(event):
            collision_frames.add(event.frame)
            collision_events.append({
                "frame": event.frame,
                "other_actor_id": event.other_actor.id if event.other_actor else None,
                "other_actor_type": event.other_actor.type_id if event.other_actor else "unknown",
            })

        # Independent forward/rear RGB feeds detect hazards in both ego directions.
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        camera_frames = {view: queue.Queue(maxsize=1) for view in ("front", "rear")}
        for view, yaw in (("front", 0), ("rear", 180)):
            camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.8 if view == "front" else -1.8,
                z=1.35), carla.Rotation(yaw=yaw)), attach_to=ego)
            spawned.append(camera); sensors.append(camera)

            def on_camera(image, camera_view=view):
                import numpy as np
                pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
                rgb = pixels[:, :, 2::-1].copy()
                frames = camera_frames[camera_view]
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait((image.frame, rgb))
                except queue.Full:
                    pass

            camera.listen(on_camera)
        emergency_detector = EmergencyDetector(REPO_ROOT/"emergency_detection"/"weights"/"emergency_detector.pt")
        preview = CameraPreview() if camera_preview else None
        detection_trackers = {view: DetectionTracker() for view in camera_frames}
        detection_ranges: dict[str, dict[int, tuple[float, float]]] = {view: {} for view in camera_frames}
        latest_camera_frames = {view: None for view in camera_frames}
        latest_detections: list[dict] = []
        camera_frame_ids = {view: -1 for view in camera_frames}
        latest_detection_times = {view: -1.0 for view in camera_frames}

        predictor = TrajectoryPredictor(REPO_ROOT/"models"/"trajectory_lstm.pt")
        planner = DrivingSimulation(scenario, mode, seed=seed, predictor=predictor,
                                    risk_model=risk_model, lane_model=lane_model)
        planner.actors = []
        planner.ego_lane = 0
        current_lane_id, target_lane_id = lane0, lane0
        lane_ids = {0: lane0, 1: lane1}
        lane_change_in_progress = False
        lane_change_cooldown_until = 0.0
        safety_braking = False
        desired_lane_direction = adjacent_direction
        response_time = None
        collision_sensor = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        spawned.append(collision_sensor)
        sensors.append(collision_sensor)
        collision_sensor.listen(on_collision)
        ticks = None if duration is None else round(duration/DT)
        travelled = 0.0
        previous_location = ego.get_location()
        smoothed_control_target: float | None = None
        near_miss_count = 0
        in_near_miss = False
        min_ttc = math.inf
        risk_sum = 0.0
        collision_halt = False
        tick_indices = itertools.count() if ticks is None else range(ticks)
        for index in tick_indices:
            if STOP_REQUESTED or (stop_file is not None and stop_file.exists()):
                print("[carla_demo] stop requested; saving telemetry and restoring CARLA")
                break
            while pause_file is not None and pause_file.exists():
                if STOP_REQUESTED or (stop_file is not None and stop_file.exists()):
                    break
                time.sleep(0.05)
            if STOP_REQUESTED or (stop_file is not None and stop_file.exists()):
                print("[carla_demo] stop requested; saving telemetry and restoring CARLA")
                break
            world.tick()
            if collision_events and not collision_halt:
                first = collision_events[0]
                print("[carla_demo] physical collision with "
                      f"{first['other_actor_type']} (actor {first['other_actor_id']}); "
                      "holding ego vehicle stopped while camera feed continues")
                collision_halt = True
            for camera_view, frame_queue in camera_frames.items():
                try:
                    new_frame_id, new_frame = frame_queue.get_nowait()
                    if new_frame_id == camera_frame_ids[camera_view]:
                        continue
                    camera_frame_ids[camera_view], latest_camera_frames[camera_view] = new_frame_id, new_frame
                    latest_detection_times[camera_view] = index*DT
                    if web_frame_dir is not None:
                        try:
                            from PIL import Image
                            web_frame_dir.mkdir(parents=True, exist_ok=True)
                            frame_path = web_frame_dir / f"{camera_view}-{new_frame_id}.jpg"
                            Image.fromarray(new_frame[::2, ::2]).save(frame_path, format="JPEG", quality=78)
                            if new_frame_id % 20 == 0:
                                old_frames = sorted(web_frame_dir.glob(f"{camera_view}-*.jpg"),
                                                    key=lambda item: item.stat().st_mtime, reverse=True)
                                for old_frame in old_frames[10:]:
                                    old_frame.unlink(missing_ok=True)
                        except Exception as exc:
                            print(f"[carla_demo] web camera frame unavailable: {exc}")
                            web_frame_dir = None
                    outputs = detection_trackers[camera_view].update(emergency_detector.predict(new_frame))
                    for detection in outputs:
                        detection.update(estimate_distance_ttc(detection, 1280, 720, 90, DT,
                                                               detection_ranges[camera_view]))
                        detection["camera_view"] = camera_view
                    latest_detections = [d for d in latest_detections if d.get("camera_view") != camera_view] + outputs
                except queue.Empty:
                    pass
            latest_detections = [d for d in latest_detections
                                 if index*DT-latest_detection_times.get(d.get("camera_view"), -1.0) <= .5]
            if index % 20 == 0:
                print(f"[camera detector] {emergency_detector.status}; detections={len(latest_detections)}; "
                      f"latency={emergency_detector.latency_ms:.1f}ms")
            ego_speed, observed = _relative_state(ego, others, lane0, lane1)
            # Only model output adds emergency class labels to the planner; CARLA
            # blueprint identity and emergency-light state remain evaluation-only.
            visual_actors = []
            for detection in latest_detections:
                center_x = sum(detection["bbox"][::2])/2
                current_lane_index = 0 if current_lane_id == lane0 else 1
                if detection.get("path_overlap"):
                    lane = current_lane_index
                else:
                    camera_side = "left" if center_x < 640 else "right"
                    if detection.get("camera_view") == "rear":
                        camera_side = "right" if camera_side == "left" else "left"
                    if current_lane_index == 0 and camera_side == desired_lane_direction:
                        lane = 1
                    elif current_lane_index == 1 and camera_side != desired_lane_direction:
                        lane = 0
                    else:
                        # The camera may see a vehicle in a third lane or off
                        # the ego's two-lane planning corridor; don't mislabel it.
                        continue
                distance = detection["distance_m"] * (-1 if detection["camera_view"] == "rear" else 1)
                closing = detection.get("closing_speed_mps") or 0.0
                longitudinal_speed = ego_speed + closing if detection["camera_view"] == "rear" else ego_speed-closing
                longitudinal_speed = max(0.0, min(25.0, longitudinal_speed))
                visual_actors.append(Actor(f"camera-{detection['camera_view']}-{detection['track_id']}",
                    "emergency" if detection["class"] in ("ambulance", "fire_truck", "police_vehicle") else "car",
                    distance, longitudinal_speed, lane,
                    detection["class"] in ("ambulance", "fire_truck", "police_vehicle")))
            observed.extend((a, None) for a in visual_actors)
            planner.ego_s = 0.0
            planner.ego_speed = ego_speed
            planner.ego_lane = 0 if current_lane_id == lane0 else 1
            planner.actors = [actor for actor, _ in observed]
            telemetry = planner.step(DT)
            if preview is not None and any(frame is not None for frame in latest_camera_frames.values()):
                sim_truth = [f"{name}={('ambulance' if 'ambulance' in actor.type_id else 'fire truck' if 'firetruck' in actor.type_id else 'police vehicle')}"
                             for name, actor, is_emergency, _ in others
                             if is_emergency and actor.is_alive]
                preview.show(latest_camera_frames, latest_detections, emergency_detector,
                             telemetry, sim_truth)
                if preview.closed_by_user:
                    print("[carla_demo] camera preview closed; stopping continuous run")
                    break
            current_location = ego.get_location()
            travelled += current_location.distance(previous_location)
            previous_location = current_location
            physical_lane = 0 if current_lane_id == lane0 else 1
            lane_gaps = [abs(actor.s) for actor, _ in observed if actor.lane == physical_lane]
            closest_gap = min(lane_gaps, default=math.inf)
            frame_now = world.get_snapshot().frame
            near_now = closest_gap < 5.0 and frame_now not in collision_frames
            if near_now and not in_near_miss:
                near_miss_count += 1
            in_near_miss = near_now
            min_ttc = min(min_ttc, telemetry.ttc)
            risk_sum += telemetry.risk_score
            if telemetry.action != "MAINTAIN SPEED" and response_time is None:
                response_time = index*DT

            current_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
            sim_time = index*DT
            if current_wp and current_wp.lane_id in lane_ids.values():
                current_lane_id = current_wp.lane_id
            if lane_change_in_progress and current_lane_id == target_lane_id:
                lane_change_in_progress = False
                lane_change_cooldown_until = sim_time + 3.0
            requested_lane_id = lane_ids.get(telemetry.ego_lane, current_lane_id)
            # Commit to each maneuver until the ego reaches the target lane.
            # Planner lane requests can oscillate while a vehicle is crossing.
            if (not lane_change_in_progress and sim_time >= lane_change_cooldown_until
                    and requested_lane_id != current_lane_id):
                candidate = _driving_neighbor(current_wp, desired_lane_direction) if current_wp else None
                if candidate is None or candidate.lane_id != requested_lane_id:
                    opposite = "left" if desired_lane_direction == "right" else "right"
                    candidate = _driving_neighbor(current_wp, opposite) if current_wp else None
                if candidate and candidate.lane_id == requested_lane_id:
                    target_lane_id = requested_lane_id
                    lane_change_in_progress = True
            if not lane_change_in_progress:
                target_lane_id = current_lane_id
            target_wp = current_wp
            if current_wp and target_lane_id != current_wp.lane_id:
                neighbor = _driving_neighbor(current_wp, desired_lane_direction)
                if neighbor is None or neighbor.lane_id != target_lane_id:
                    opposite = "left" if desired_lane_direction == "right" else "right"
                    neighbor = _driving_neighbor(current_wp, opposite)
                if neighbor and neighbor.lane_id == target_lane_id:
                    target_wp = neighbor

            lookahead = max(8.0, min(18.0, 6.0+ego_speed*1.2))
            ahead = _next(target_wp, lookahead) if target_wp else None
            steer = 0.0
            # The steering demonstrations were collected with lane changes
            # disabled. Keep the learned controller in-distribution: use it
            # for lane keeping, and use waypoint geometry during a lane change.
            if (steering_model is not None and target_wp and not lane_change_in_progress
                    and sim_time >= lane_change_cooldown_until):
                steer = steering_model.predict(ego, target_wp, carla_map)
            elif ahead:
                origin = ego.get_location()
                target_loc = ahead.transform.location
                yaw_now = math.radians(ego.get_transform().rotation.yaw)
                yaw_target = math.atan2(target_loc.y-origin.y, target_loc.x-origin.x)
                delta_yaw = (yaw_target-yaw_now+math.pi) % (2*math.pi)-math.pi
                steer = max(-1.0, min(1.0, delta_yaw/math.radians(55)))
            if target_wp:
                # Heading-only steering drifts toward the edge on long curves.
                # Correct toward the selected lane center as well.
                location = ego.get_location()
                lane_center = target_wp.transform.location
                right = target_wp.transform.get_right_vector()
                lateral_error = ((location.x-lane_center.x)*right.x
                                 + (location.y-lane_center.y)*right.y)
                steer -= lateral_error*0.04
            steer = max(-0.3, min(0.3, steer))
            control_target_speed = telemetry.target_speed
            throttle_limit = 0.75
            if scenario in ("Dense traffic lane changes", "Four emergency vehicles"):
                # Keep closing speed modest so repeated passes remain visible
                # and the traffic manager has room to react around the ego.
                control_target_speed = min(control_target_speed,
                                           6.5 if scenario == "Four emergency vehicles" else 11.0)
                throttle_limit = 0.4
            # Keep the packaged simulator demo at a moderate, controllable speed.
            control_target_speed = min(control_target_speed, 10.0)
            if smoothed_control_target is None:
                smoothed_control_target = control_target_speed
            else:
                # Rate-limit routine target changes to avoid stop/start motion
                # when risk estimates or lane requests fluctuate between ticks.
                target_rate = 6.0 if telemetry.action == "BRAKE" else 1.8
                max_step = target_rate * DT
                change = control_target_speed-smoothed_control_target
                smoothed_control_target += max(-max_step, min(max_step, change))
            control_target_speed = smoothed_control_target
            speed_error = control_target_speed-ego_speed
            throttle = max(0.0, min(throttle_limit, speed_error*0.22))
            if telemetry.action == "BRAKE":
                # Preserve strong braking for rule-based imminent hazards.
                brake = max(0.0, min(1.0, -speed_error*0.24))
            else:
                # Learned/routine risk responses should bleed speed smoothly,
                # not turn every modest setpoint change into a hard stop.
                brake = max(0.0, min(0.45, -speed_error*0.10))
            # Independent CARLA safety guard uses simulator actor geometry,
            # not camera range estimates or the planner's lane labels.
            ego_transform = ego.get_transform()
            ego_location = ego_transform.location
            forward = ego_transform.get_forward_vector()
            right = ego_transform.get_right_vector()
            corridor_half_width = 4.5 if lane_change_in_progress else 2.5
            physical_front = None
            for actor_name, actor, _, _ in others:
                if not actor.is_alive:
                    continue
                location = actor.get_location()
                dx, dy = location.x-ego_location.x, location.y-ego_location.y
                longitudinal = dx*forward.x + dy*forward.y
                lateral = dx*right.x + dy*right.y
                if longitudinal > 0 and abs(lateral) <= corridor_half_width:
                    velocity = actor.get_velocity()
                    actor_speed = max(0.0, velocity.x*forward.x + velocity.y*forward.y)
                    if physical_front is None or longitudinal < physical_front[0]:
                        physical_front = (longitudinal, actor_speed, actor_name)
            safety_required_gap = 0.0
            safety_override = False
            if physical_front is not None:
                front_gap, front_speed, _ = physical_front
                closing_speed = max(0.0, ego_speed-front_speed)
                stopping_gap = (5.0 + 3.0 + closing_speed*0.8
                                + max(0.0, ego_speed**2-front_speed**2)/(2*7.0))
                safety_required_gap = stopping_gap
                if safety_braking:
                    safety_braking = front_gap <= stopping_gap + 4.0
                else:
                    safety_braking = front_gap <= stopping_gap
            else:
                safety_braking = False
            if safety_braking:
                throttle, brake = 0.0, 1.0
                safety_override = True
            if collision_halt:
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0,
                                                       steer=0.0, hand_brake=True))
            else:
                ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

            for actor_name, actor, is_emergency, is_obstacle in others:
                if not actor.is_alive:
                    continue
                tr, vel = actor.get_transform(), actor.get_velocity()
                wp = carla_map.get_waypoint(tr.location, project_to_road=True)
                light_on = bool(is_emergency and int(actor.get_light_state()) & EMERGENCY_BITS)
                raw_rows.append({
                    "scenario": f"carla_{scenario.lower().replace(' ', '_')}",
                    "actor_id": actor_name, "sim_time": round(sim_time, 3),
                    "loc_x": tr.location.x, "loc_y": tr.location.y, "loc_z": tr.location.z,
                    "vel_x": vel.x, "vel_y": vel.y, "vel_z": vel.z,
                    "yaw_deg": tr.rotation.yaw,
                    "lane_id": wp.lane_id if wp else None,
                    "blueprint_id": actor.type_id,
                    "emergency_light_on": light_on,
                })
            tr, vel = ego.get_transform(), ego.get_velocity()
            wp = carla_map.get_waypoint(tr.location, project_to_road=True)
            raw_rows.append({
                "scenario": f"carla_{scenario.lower().replace(' ', '_')}",
                "actor_id": "ego", "sim_time": round(sim_time, 3),
                "loc_x": tr.location.x, "loc_y": tr.location.y, "loc_z": tr.location.z,
                "vel_x": vel.x, "vel_y": vel.y, "vel_z": vel.z,
                "yaw_deg": tr.rotation.yaw, "lane_id": wp.lane_id if wp else None,
                "blueprint_id": ego.type_id, "emergency_light_on": False,
            })
            collision_count = len({(event["frame"], event["other_actor_id"])
                                   for event in collision_events})
            ai = telemetry_dict(telemetry)
            ai.update({"ego_s": travelled, "collision_count": collision_count,
                       "near_miss_count": near_miss_count,
                       "min_ttc": min_ttc if math.isfinite(min_ttc) else "inf"})
            ai.update({"scenario": scenario, "planner": mode,
                       "ego_speed": ego_speed, "ego_lane": int(telemetry.ego_lane),
                       "target_speed": control_target_speed, "steering": steer,
                       "throttle": throttle, "brake": brake,
                       "carla_target_speed_mps": control_target_speed,
                       "carla_safety_override": safety_override,
                       "carla_required_gap_m": safety_required_gap,
                       "carla_collision_events": collision_count, "source": "CARLA 0.9.16",
                       "actual_collision": False, "collision_with_actor_type": "",
                       "prediction_source":planner.prediction_source,
                       "prediction_latency_ms":planner.model_latency_ms,
                       "emergency_detector_status": emergency_detector.status,
                       "emergency_detector_latency_ms": emergency_detector.latency_ms,
                       "emergency_detection_count": len([d for d in latest_detections if d["class"] != "ordinary_vehicle"]),
                       "emergency_detections": json.dumps(latest_detections)})
            ai_rows.append(ai)
            if web_frame_dir is not None:
                try:
                    actor_state = []
                    actor_flags = {name: (is_emergency, is_obstacle)
                                   for name, _actor, is_emergency, is_obstacle in others}
                    for actor, _physical_actor in observed:
                        is_emergency, is_obstacle = actor_flags.get(actor.actor_id, (False, False))
                        actor_state.append({
                            "id": actor.actor_id, "kind": "obstacle" if is_obstacle else actor.kind,
                            "lane": int(actor.lane), "emergency": bool(is_emergency),
                            "delta_s": float(actor.s), "speed": float(actor.speed),
                            "prediction": [],
                        })
                    live_state = {
                        "source": "CARLA", "running": True, "paused": False,
                        "scenario": scenario, "mode": mode, "seed": seed,
                        "telemetry_file": "CARLA control telemetry (saved when run ends)",
                        "last": ai,
                        "metrics": {
                            "collision_count": collision_count,
                            "near_miss_count": near_miss_count,
                            "minimum_ttc_s": min_ttc if math.isfinite(min_ttc) else None,
                            "average_risk": risk_sum / max(1, len(ai_rows)),
                            "path_length_m": travelled,
                            "response_time_s": response_time,
                        },
                        "actors": actor_state,
                        "ego": {"speed": float(ego_speed), "lane": int(telemetry.ego_lane),
                                "position": float(travelled)},
                        "prediction_source": planner.prediction_source,
                    }
                    web_frame_dir.mkdir(parents=True, exist_ok=True)
                    state_tmp = web_frame_dir / f"live-state-{os.getpid()}.tmp"
                    state_tmp.write_text(json.dumps(_finite_json(live_state), allow_nan=False), encoding="utf-8")
                    for attempt in range(5):
                        try:
                            state_tmp.replace(web_frame_dir / "live-state.json")
                            break
                        except PermissionError:
                            if attempt == 4:
                                raise
                            time.sleep(0.01)
                except (OSError, TypeError, ValueError) as exc:
                    print(f"[carla_demo] web telemetry snapshot unavailable: {exc}")
            if index % 100 == 0:
                print(f"[carla_demo] t={sim_time:5.1f}s action={telemetry.action} "
                      f"speed={ego_speed*3.6:4.0f}km/h risk={telemetry.risk_level}:{telemetry.risk_score:.0f} "
                      f"collision_events={collision_count}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = REPO_ROOT/"data"/"raw_captures"/f"drivemind_carla_{stamp}.json"
        ai_path = REPO_ROOT/"data"/"simulation"/"telemetry"/f"carla_control_{stamp}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        ai_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
        metrics = planner.metrics()
        unique_collisions = {(row["frame"], row["other_actor_id"]) for row in collision_events}
        if collision_events and ai_rows:
            ai_rows[-1]["actual_collision"] = True
            ai_rows[-1]["carla_collision_events"] = len(unique_collisions)
            ai_rows[-1]["collision_count"] = len(unique_collisions)
            ai_rows[-1]["collision_with_actor_type"] = collision_events[0]["other_actor_type"]
        # Persist after applying the terminal contact label so the experience
        # builder sees the collision recorded by the sensor.
        with ai_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ai_rows[0].keys() if ai_rows else [])
            writer.writeheader()
            writer.writerows(ai_rows)
        metrics.update({"collision_count": len(unique_collisions), "collision_frames": len(collision_frames),
                        "collision_events": collision_events, "near_miss_count": near_miss_count,
                        "minimum_ttc_s": min_ttc if math.isfinite(min_ttc) else None,
                        "average_risk": risk_sum/max(1, len(ai_rows)), "path_length_m": travelled,
                        "planner_predicted_contacts": planner.collision_count,
                        "carla_collision_events": len(unique_collisions), "response_time_s": response_time,
                        "map": carla_map.name, "server_version": client.get_server_version(),
                        "emergency_detector_status": emergency_detector.status,
                        "emergency_detector_model": "DriveMind-Emergency-FasterRCNN-ResNet50-FPN-v1",
                        "emergency_detections_observed": sum(r.get("emergency_detection_count", 0) for r in ai_rows)})
        print(f"[carla_demo] raw CARLA telemetry -> {raw_path}")
        print(f"[carla_demo] DriveMind controls + risk -> {ai_path}")
        print(f"[carla_demo] results: {metrics}")
        return raw_path, ai_path, metrics
    finally:
        if "preview" in locals() and preview is not None:
            preview.close()
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        for actor in spawned:
            try:
                if actor.is_alive:
                    actor.set_autopilot(False, TM_PORT)
            except Exception:
                pass
        try:
            world.tick()
        except Exception:
            pass
        for actor in reversed(spawned):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        try:
            world.tick()
        except Exception:
            pass
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception:
            pass
        if stop_file is not None:
            stop_file.unlink(missing_ok=True)
        if pause_file is not None:
            pause_file.unlink(missing_ok=True)


def main() -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, _request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--continuous", action="store_true",
                        help="keep running until the camera window is closed or Ctrl+C is pressed")
    parser.add_argument("--scenario", choices=(*SCENARIOS, *CAMERA_SCENARIOS), default="Emergency vehicle approaching")
    parser.add_argument("--mode", choices=MODES, default="DRIVEMIND")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--risk-model", type=Path, default=None,
                        help="candidate risk model JSON; otherwise load an approved active model if present")
    parser.add_argument("--lane-policy", type=Path, default=None,
                        help="candidate lane-choice JSON; otherwise load an approved active policy if present")
    parser.add_argument("--steering-model", type=Path, default=None,
                        help="steering regression JSON trained from CARLA expert captures")
    parser.add_argument("--no-camera-preview", action="store_true", help="run headless without the camera window")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="stop gracefully when this file appears (used by the local web dashboard)")
    parser.add_argument("--pause-file", type=Path, default=None,
                        help="pause CARLA world ticks while this file exists")
    parser.add_argument("--web-frame-dir", type=Path, default=None,
                        help="write the latest ego front/rear camera frames for the browser dashboard")
    args = parser.parse_args()
    if not args.continuous and args.duration <= 0:
        parser.error("--duration must be positive")
    risk_model = CollisionRiskModel.load(args.risk_model) if args.risk_model else None
    lane_model = LaneChoiceModel.load(args.lane_policy) if args.lane_policy else None
    steering_model = SteeringModel.load(args.steering_model) if args.steering_model else None
    run(args.host, args.port, None if args.continuous else args.duration, args.scenario, args.mode, args.seed,
        camera_preview=not args.no_camera_preview, risk_model=risk_model, lane_model=lane_model,
        steering_model=steering_model, stop_file=args.stop_file, pause_file=args.pause_file,
        web_frame_dir=args.web_frame_dir)
    return 0


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


if __name__ == "__main__":
    raise SystemExit(main())
