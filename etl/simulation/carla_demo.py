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
import queue
from datetime import datetime
from pathlib import Path
import random
import signal
import tkinter as tk
from tkinter import ttk

import carla

from etl.config import REPO_ROOT
from etl.prediction.inference import TrajectoryPredictor
from etl.simulation.core import Actor, DrivingSimulation, MODES, SCENARIOS, telemetry_dict
from emergency_detection.inference import EmergencyDetector
from emergency_detection.tracking import DetectionTracker, estimate_distance_ttc


DT = 0.05
TM_PORT = 8000
STOP_REQUESTED = False
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

    def show(self, frames, detections, detector, telemetry=None):
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
            self.status.configure(text=f"{detector.status} | {detector.__class__.__name__} | "
                f"{len(emergency)} emergency vehicles | {timing} | Risk {risk} | Planner: {action}\n{details}")
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
        seed: int = 41, camera_preview: bool = True) -> tuple[Path, Path, dict]:
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
        start_wp = _choose_start(world, junction_ahead=junction_case)
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
            target = lane_wp if distance <= 0 else _next(lane_wp, distance)
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
                    tm.vehicle_percentage_speed_difference(actor, 45.0)
                if emergency:
                    tm.vehicle_percentage_speed_difference(actor, -20.0)
                    try:
                        actor.set_light_state(carla.VehicleLightState(EMERGENCY_BITS))
                    except Exception:
                        pass
            others.append((actor_name, actor, emergency, obstacle))

        lane1_wp = adjacent
        if scenario in ("Emergency vehicle approaching", "Approaching ambulance"):
            # Give perception and planning more time to recognize the siren
            # vehicle and clear a safe passing gap before it reaches the ego.
            previous = _next(start_wp, 85, backwards=True)
            if previous:
                vehicle_type = "vehicle.ford.ambulance" if scenario == "Approaching ambulance" else "vehicle.carlamotors.firetruck"
                bp = bp_lib.find(vehicle_type)
                emergency_actor = _spawn_vehicle(world, bp, previous, "drivemind_emergency")
                if emergency_actor:
                    spawned.append(emergency_actor)
                    emergency_actor.set_autopilot(True, TM_PORT)
                    tm.vehicle_percentage_speed_difference(emergency_actor, -20.0)
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
        else:
            spawn_at(44, start_wp, "V-01")
            spawn_at(70, lane1_wp, "V-02")
            spawn_at(98, lane1_wp, "V-03", slower=True)

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
        planner = DrivingSimulation(scenario, mode, seed=seed, predictor=predictor)
        planner.actors = []
        planner.ego_lane = 0
        current_lane_id, target_lane_id = lane0, lane0
        lane_ids = {0: lane0, 1: lane1}
        desired_lane_direction = adjacent_direction
        response_time = None
        collision_sensor = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        spawned.append(collision_sensor)
        sensors.append(collision_sensor)
        collision_sensor.listen(on_collision)
        ticks = None if duration is None else round(duration/DT)
        travelled = 0.0
        previous_location = ego.get_location()
        near_miss_count = 0
        in_near_miss = False
        min_ttc = math.inf
        risk_sum = 0.0
        tick_indices = itertools.count() if ticks is None else range(ticks)
        for index in tick_indices:
            if STOP_REQUESTED:
                print("[carla_demo] stop requested; saving telemetry and restoring CARLA")
                break
            world.tick()
            if collision_events:
                first = collision_events[0]
                print("[carla_demo] physical collision with "
                      f"{first['other_actor_type']} (actor {first['other_actor_id']}); stopping the run")
                break
            for camera_view, frame_queue in camera_frames.items():
                try:
                    new_frame_id, new_frame = frame_queue.get_nowait()
                    if new_frame_id == camera_frame_ids[camera_view]:
                        continue
                    camera_frame_ids[camera_view], latest_camera_frames[camera_view] = new_frame_id, new_frame
                    latest_detection_times[camera_view] = index*DT
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
                lane = (0 if current_lane_id == lane0 else 1) if detection.get("path_overlap") else (
                    0 if center_x < 512 else 1 if center_x > 768 else (0 if current_lane_id == lane0 else 1))
                closing = detection.get("closing_speed_mps") or 0.0
                distance = detection["distance_m"] * (-1 if detection["camera_view"] == "rear" else 1)
                closing = detection.get("closing_speed_mps") or 0.0
                longitudinal_speed = ego_speed + closing if detection["camera_view"] == "rear" else ego_speed-closing
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
                preview.show(latest_camera_frames, latest_detections, emergency_detector, telemetry)
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

            requested_lane_id = lane_ids.get(telemetry.ego_lane, current_lane_id)
            if requested_lane_id != target_lane_id:
                target_lane_id = requested_lane_id
            current_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
            target_wp = current_wp
            if current_wp and target_lane_id != current_wp.lane_id:
                neighbor = _driving_neighbor(current_wp, desired_lane_direction)
                if neighbor is None or neighbor.lane_id != target_lane_id:
                    opposite = "left" if desired_lane_direction == "right" else "right"
                    neighbor = _driving_neighbor(current_wp, opposite)
                if neighbor and neighbor.lane_id == target_lane_id:
                    target_wp = neighbor
            if current_wp and current_wp.lane_id in lane_ids.values():
                current_lane_id = current_wp.lane_id

            ahead = _next(target_wp, 8.0) if target_wp else None
            steer = 0.0
            if ahead:
                origin = ego.get_location()
                target_loc = ahead.transform.location
                yaw_now = math.radians(ego.get_transform().rotation.yaw)
                yaw_target = math.atan2(target_loc.y-origin.y, target_loc.x-origin.x)
                delta_yaw = (yaw_target-yaw_now+math.pi) % (2*math.pi)-math.pi
                steer = max(-1.0, min(1.0, delta_yaw/math.radians(35)))
            speed_error = telemetry.target_speed-ego_speed
            throttle = max(0.0, min(0.75, speed_error*0.22))
            brake = max(0.0, min(1.0, -speed_error*0.24))
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

            sim_time = index*DT
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
                       "carla_collision_events": collision_count, "source": "CARLA 0.9.16",
                       "prediction_source":planner.prediction_source,
                       "prediction_latency_ms":planner.model_latency_ms,
                       "emergency_detector_status": emergency_detector.status,
                       "emergency_detector_latency_ms": emergency_detector.latency_ms,
                       "emergency_detection_count": len([d for d in latest_detections if d["class"] != "ordinary_vehicle"]),
                       "emergency_detections": json.dumps(latest_detections)})
            ai_rows.append(ai)
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
        with ai_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ai_rows[0].keys() if ai_rows else [])
            writer.writeheader()
            writer.writerows(ai_rows)
        metrics = planner.metrics()
        unique_collisions = {(row["frame"], row["other_actor_id"]) for row in collision_events}
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
    parser.add_argument("--no-camera-preview", action="store_true", help="run headless without the camera window")
    args = parser.parse_args()
    if not args.continuous and args.duration <= 0:
        parser.error("--duration must be positive")
    run(args.host, args.port, None if args.continuous else args.duration, args.scenario, args.mode, args.seed,
        camera_preview=not args.no_camera_preview)
    return 0


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


if __name__ == "__main__":
    raise SystemExit(main())
