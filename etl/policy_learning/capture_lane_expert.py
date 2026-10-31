"""Collect CARLA Traffic Manager expert lane-choice examples as CSV."""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import carla

from etl.config import REPO_ROOT
from etl.policy_learning.lane_model import FEATURES


FIELDS = ("run_id", "scenario", "sim_time", "ego_speed_mps", "current_front_gap_m",
          "current_front_closing_mps", "adjacent_front_gap_m", "adjacent_front_closing_mps",
          "adjacent_rear_gap_m", "adjacent_rear_closing_mps", "current_lane_id",
          "adjacent_lane_ids", "expert_target_lane_id", "expert_action", "expert_steer",
          "expert_throttle", "expert_brake", "collision")


def _speed(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


def _project(ego_transform, ego_speed, actor):
    loc = actor.get_location()
    angle = math.radians(ego_transform.rotation.yaw)
    dx, dy = loc.x-ego_transform.location.x, loc.y-ego_transform.location.y
    gap = dx*math.cos(angle) + dy*math.sin(angle)
    velocity = actor.get_velocity()
    actor_speed = velocity.x*math.cos(angle) + velocity.y*math.sin(angle)
    return gap, actor_speed


def _nearest(rows, front: bool):
    if front:
        return min((r for r in rows if r[0] > 0), key=lambda r: r[0], default=None)
    return min((r for r in rows if r[0] < 0), key=lambda r: abs(r[0]), default=None)


def _feature_row(ego, actors, carla_map, run_id, scenario, sim_time, collision):
    transform = ego.get_transform()
    ego_wp = carla_map.get_waypoint(transform.location, project_to_road=True)
    if ego_wp is None:
        return None
    adjacent = []
    for direction in ("left", "right"):
        wp = ego_wp.get_left_lane() if direction == "left" else ego_wp.get_right_lane()
        if wp and wp.lane_type == carla.LaneType.Driving:
            adjacent.append((direction, wp.lane_id))
    lane_objects: dict[int, list[tuple[float, float]]] = {}
    ego_speed = _speed(ego)
    for actor in actors:
        if actor.id == ego.id or not actor.is_alive or not actor.type_id.startswith("vehicle."):
            continue
        loc = actor.get_location()
        wp = carla_map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            continue
        gap, other_speed = _project(transform, ego_speed, actor)
        lane_objects.setdefault(wp.lane_id, []).append((gap, other_speed))

    def front_features(lanes):
        candidates = [item for lane in lanes for item in lane_objects.get(lane, []) if item[0] > 0]
        nearest = min(candidates, key=lambda item: item[0], default=None)
        if nearest is None:
            return 120.0, 0.0
        return nearest[0], ego_speed-nearest[1]

    current_front, current_closing = front_features([ego_wp.lane_id])
    adjacent_lanes = [lane_id for _, lane_id in adjacent]
    adjacent_front, adjacent_front_closing = front_features(adjacent_lanes)
    rear_candidates = [item for lane in adjacent_lanes for item in lane_objects.get(lane, []) if item[0] < 0]
    rear = min(rear_candidates, key=lambda item: abs(item[0]), default=None)
    rear_gap = abs(rear[0]) if rear else 120.0
    rear_closing = max(0.0, rear[1]-ego_speed) if rear else 0.0
    control = ego.get_control()
    return {"run_id": run_id, "scenario": scenario, "sim_time": sim_time,
            "ego_speed_mps": ego_speed, "current_front_gap_m": current_front,
            "current_front_closing_mps": current_closing,
            "adjacent_front_gap_m": adjacent_front,
            "adjacent_front_closing_mps": adjacent_front_closing,
            "adjacent_rear_gap_m": rear_gap, "adjacent_rear_closing_mps": rear_closing,
            "current_lane_id": ego_wp.lane_id,
            "adjacent_lane_ids": ";".join(str(lane_id) for _, lane_id in adjacent),
            "expert_target_lane_id": "", "expert_action": "", "expert_steer": control.steer,
            "expert_throttle": control.throttle, "expert_brake": control.brake,
            "collision": collision}


def capture(host: str, port: int, duration: float, tick_hz: float, vehicles: int,
            seed: int, town: str | None, scenario: str) -> Path:
    random.seed(seed)
    client = carla.Client(host, port)
    client.set_timeout(15.0)
    world = client.load_world(town) if town else client.get_world()
    traffic = client.get_trafficmanager()
    traffic.set_random_device_seed(seed)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/tick_hz
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    traffic.set_synchronous_mode(True)
    spawned = []
    collisions = []
    rows = []
    try:
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        points = list(world.get_map().get_spawn_points())
        random.shuffle(points)
        if len(points) < vehicles+1:
            raise RuntimeError(f"need {vehicles+1} spawn points, found {len(points)}")
        ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(ego_bp, points.pop())
        if ego is None:
            raise RuntimeError("could not spawn expert ego vehicle")
        spawned.append(ego)
        ego.set_autopilot(True, traffic.get_port())
        traffic.auto_lane_change(ego, True)
        traffic.vehicle_percentage_speed_difference(ego, 0.0)
        for _ in range(vehicles):
            bp = random.choice(blueprints)
            actor = world.try_spawn_actor(bp, points.pop())
            if actor:
                actor.set_autopilot(True, traffic.get_port())
                traffic.vehicle_percentage_speed_difference(actor, random.uniform(15.0, 40.0))
                spawned.append(actor)
        print(f"[lane expert] spawned {len(spawned)} autopilot vehicles; TM lane changes enabled")

        sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                   carla.Transform(), attach_to=ego)
        spawned.append(sensor)
        sensor.listen(lambda event: collisions.append(event.frame))
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        carla_map = world.get_map()
        ticks = round(duration*tick_hz)
        for tick in range(ticks):
            world.tick()
            t = tick/tick_hz
            row = _feature_row(ego, spawned, carla_map, run_id, scenario, round(t, 3), bool(collisions))
            if row:
                rows.append(row)
            if collisions:
                print(f"[lane expert] collision at {t:.2f}s; ending capture")
                break
            if tick % int(tick_hz*10) == 0:
                print(f"[lane expert] t={t:.1f}s, lane={row['current_lane_id'] if row else 'unknown'}")

        horizon = max(1, round(2.0*tick_hz))
        for i, row in enumerate(rows):
            if i+horizon >= len(rows):
                continue
            current = row["current_lane_id"]
            neighbors = {int(value) for value in row["adjacent_lane_ids"].split(";") if value}
            future = rows[i+horizon]["current_lane_id"]
            if future == current:
                row["expert_action"] = "stay"
                row["expert_target_lane_id"] = current
            elif future in neighbors:
                row["expert_action"] = "change"
                row["expert_target_lane_id"] = future

        output = REPO_ROOT/"data"/"policy_learning"/f"lane_expert_{run_id}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if row["expert_action"])
        counts = {action: sum(row["expert_action"] == action for row in rows) for action in ("stay", "change")}
        print(f"[lane expert] wrote {sum(counts.values())} labeled samples {counts} -> {output}")
        return output
    finally:
        for actor in spawned:
            try:
                if actor.is_alive:
                    if actor.type_id.startswith("sensor."):
                        actor.stop()
                    else:
                        actor.set_autopilot(False, traffic.get_port())
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
            traffic.set_synchronous_mode(False)
            world.apply_settings(original)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--tick-hz", type=float, default=20.0)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--town", default=None)
    parser.add_argument("--scenario", default="traffic_manager_expert")
    args = parser.parse_args()
    if args.duration <= 0 or args.tick_hz <= 0 or args.vehicles < 1:
        parser.error("duration/tick rate must be positive and vehicles must be at least one")
    capture(args.host, args.port, args.duration, args.tick_hz, args.vehicles,
            args.seed, args.town, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
