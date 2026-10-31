"""Record CARLA Traffic Manager steering with lane-relative road features."""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import carla

from etl.config import REPO_ROOT
from etl.policy_learning.steering_model import FEATURES, steering_features


FIELDS = ("run_id", "sim_time", *FEATURES, "expert_steer", "expert_throttle",
          "expert_brake", "lane_id", "collision")


def capture(host="127.0.0.1", port=2000, duration=120.0, tick_hz=20.0,
            vehicles=8, seed=51, town=None):
    random.seed(seed)
    client = carla.Client(host, port)
    client.set_timeout(15.0)
    world = client.load_world(town) if town else client.get_world()
    traffic = client.get_trafficmanager(8000)
    traffic.set_random_device_seed(seed)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/tick_hz
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    traffic.set_synchronous_mode(True)
    spawned, collisions, rows = [], [], []
    try:
        points = list(world.get_map().get_spawn_points())
        random.shuffle(points)
        if len(points) < vehicles+1:
            raise RuntimeError(f"need {vehicles+1} spawn points, found {len(points)}")
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        ego = world.try_spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"), points.pop())
        if ego is None:
            raise RuntimeError("could not spawn steering expert ego vehicle")
        spawned.append(ego)
        ego.set_autopilot(True, traffic.get_port())
        traffic.auto_lane_change(ego, False)
        traffic.vehicle_percentage_speed_difference(ego, 10.0)
        for _ in range(vehicles):
            actor = world.try_spawn_actor(random.choice(blueprints), points.pop())
            if actor:
                actor.set_autopilot(True, traffic.get_port())
                traffic.vehicle_percentage_speed_difference(actor, random.uniform(10, 35))
                spawned.append(actor)
        sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                   carla.Transform(), attach_to=ego)
        spawned.append(sensor)
        sensor.listen(lambda event: collisions.append(event.frame))
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        carla_map = world.get_map()
        ticks = round(duration*tick_hz)
        print(f"[steering expert] spawned {len(spawned)-1} traffic vehicles; lane changes disabled")
        for tick in range(ticks):
            world.tick()
            wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
            if wp is None:
                continue
            control = ego.get_control()
            feature_row = steering_features(ego, wp, carla_map)
            rows.append({"run_id": run_id, "sim_time": round(tick/tick_hz, 3),
                         **feature_row, "expert_steer": control.steer,
                         "expert_throttle": control.throttle, "expert_brake": control.brake,
                         "lane_id": wp.lane_id, "collision": bool(collisions)})
            if collisions:
                print(f"[steering expert] collision at {tick/tick_hz:.2f}s; ending run")
                break
            if tick % int(tick_hz*10) == 0:
                print(f"[steering expert] t={tick/tick_hz:.0f}s samples={len(rows)}")
        output = REPO_ROOT/"data"/"policy_learning"/f"steering_expert_{run_id}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[steering expert] wrote {len(rows)} rows -> {output}")
        return output
    finally:
        for actor in spawned:
            try:
                if actor.is_alive and actor.type_id.startswith("sensor."):
                    actor.stop()
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
    parser.add_argument("--vehicles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--town", default=None)
    args = parser.parse_args()
    if args.duration <= 0 or args.tick_hz <= 0 or args.vehicles < 0:
        parser.error("duration/tick rate must be positive and vehicles must be non-negative")
    capture(args.host, args.port, args.duration, args.tick_hz, args.vehicles, args.seed, args.town)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
