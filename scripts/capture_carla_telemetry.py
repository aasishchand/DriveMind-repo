"""Capture real CARLA telemetry (+ optionally camera images and safety events)
to files that the ETL pipeline can read.

Runs against a live ``CarlaUE4.exe`` server. Spawns an ego vehicle, one emergency
vehicle, and N background vehicles on autopilot; steps the simulation in
synchronous mode at a fixed tick rate; records every actor's state each tick.

Three outputs, the latter two optional:

  * ``--out``                    telemetry JSON - keys match
                                  ``etl.extract.extract_carla.RAW_COLUMNS``,
                                  plugs straight into the main pipeline:
                                      python -m etl.load.load_bronze --domain carla_telemetry \\
                                          --name demo_overtake --scenario-log data/raw_captures/demo_overtake.json
  * ``--images-out DIR``         one PNG per sampled tick from an RGB camera
                                  mounted on the ego vehicle (``sensor.camera.rgb``)
  * ``--events-out FILE``        JSON list of collision
                                  (``sensor.other.collision``) and lane-invasion
                                  (``sensor.other.lane_invasion``) events on the
                                  ego vehicle - not yet wired into a Bronze/Silver
                                  schema (telemetry's common schema doesn't cover
                                  discrete events); this is a raw capture only.

IMPORTANT - this runs in a SEPARATE Python environment matching whatever
version your CARLA build's client wheel targets (check
PythonAPI/carla/dist/*.whl for the cp3X tag), not the project's main venv.
See scripts/README.md.

Run:
    python capture_carla_telemetry.py --scenario-name demo_overtake \\
        --duration 10 --out data/raw_captures/demo_overtake.json \\
        --images-out data/raw_captures/demo_overtake_images \\
        --events-out data/raw_captures/demo_overtake_events.json

Troubleshooting: if a blueprint id 404s ("blueprint not found"), run with
--list-blueprints to print what your CARLA build actually has, then pass
--ego-blueprint / --emergency-blueprint to override.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import carla

DEFAULT_EGO_BLUEPRINT = "vehicle.tesla.model3"
DEFAULT_EMERGENCY_BLUEPRINT = "vehicle.carlamotors.firetruck"

# carla.VehicleLightState is a Boost.Python flag enum: OR-ing two members with
# `|` degrades to a plain Python int, and passing a bare int to
# Vehicle.set_light_state() crashes the client (native ArgumentError, sometimes
# a hard access violation rather than a catchable exception). Always rewrap the
# combined bits back into the enum type before calling set_light_state().
_EMERGENCY_LIGHT_BITS = int(carla.VehicleLightState.Special1) | int(carla.VehicleLightState.Special2)
_EMERGENCY_LIGHT_STATE = carla.VehicleLightState(_EMERGENCY_LIGHT_BITS)

# Must match etl.extract.extract_carla.RAW_COLUMNS.
RAW_COLUMNS = (
    "scenario",
    "actor_id",
    "sim_time",
    "loc_x",
    "loc_y",
    "loc_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "yaw_deg",
    "lane_id",
    "blueprint_id",
    "emergency_light_on",
)


def _spawn(world, blueprint, transform, *, role_name: str | None = None):
    bp = blueprint
    if role_name and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    return world.try_spawn_actor(bp, transform)


def _waypoint_ahead(world, transform, distance: float):
    """A transform ``distance`` metres ahead of ``transform``, following the
    same lane (via the waypoint graph) - so a vehicle spawned there is on the
    same road, facing the same way, actually in front of the camera. Returns
    None if the road doesn't extend that far (e.g. spawn near a dead end)."""
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(transform.location, project_to_road=True)
    if wp is None:
        return None
    ahead = wp.next(distance)
    if not ahead:
        return None
    target = ahead[0].transform
    target.location.z += 0.5  # avoid spawning half-embedded in the road mesh
    return target


def spawn_actors(world, tm_port: int, args) -> list[tuple[str, "carla.Actor"]]:
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    if len(spawn_points) < args.num_vehicles + 2:
        print(
            f"[capture] warning: only {len(spawn_points)} spawn points for "
            f"{args.num_vehicles + 2} requested actors; some spawns may fail"
        )

    actors: list[tuple[str, "carla.Actor"]] = []

    ego_bp = bp_lib.find(args.ego_blueprint)
    ego_spawn_point = spawn_points.pop()
    ego = _spawn(world, ego_bp, ego_spawn_point, role_name="ego")
    if ego is None:
        raise RuntimeError(f"could not spawn ego blueprint {args.ego_blueprint!r}")
    ego.set_autopilot(True, tm_port)
    actors.append(("ego", ego))

    if args.emergency:
        em_bp = bp_lib.find(args.emergency_blueprint)
        em = None
        if args.emergency_ahead:
            ahead_transform = _waypoint_ahead(world, ego_spawn_point, args.emergency_ahead_distance)
            if ahead_transform is not None:
                em = _spawn(world, em_bp, ahead_transform)
                if em is not None:
                    print(
                        f"[capture] emergency vehicle placed {args.emergency_ahead_distance:.0f}m "
                        "ahead of ego, same lane"
                    )
        if em is None:
            # fall back to a random spawn point - road too short, or blocked
            em = _spawn(world, em_bp, spawn_points.pop())
            if em is not None and args.emergency_ahead:
                print("[capture] warning: couldn't place emergency vehicle ahead of ego; used a random spawn instead")
        if em is not None:
            em.set_autopilot(True, tm_port)
            try:  # not every blueprint exposes a light bar; best-effort only
                em.set_light_state(_EMERGENCY_LIGHT_STATE)
            except Exception:  # noqa: BLE001
                pass
            actors.append((f"veh_{em.id}", em))
        else:
            print(f"[capture] warning: could not spawn emergency blueprint {args.emergency_blueprint!r}")

    other_bps = bp_lib.filter("vehicle.*")
    while len(actors) < args.num_vehicles and spawn_points:
        bp = random.choice(other_bps)
        actor = _spawn(world, bp, spawn_points.pop())
        if actor is not None:
            actor.set_autopilot(True, tm_port)
            actors.append((f"veh_{actor.id}", actor))

    print(f"[capture] spawned {len(actors)} vehicles: {[label for label, _ in actors]}")
    return actors


def spawn_sensors(world, ego, args) -> tuple[list["carla.Actor"], dict]:
    """Attach an optional RGB camera and collision/lane-invasion sensors to
    ``ego``. Returns (sensor_actors, state) where state holds the mutable
    ``events`` list the sensor callbacks append to (callbacks fire on a CARLA
    background thread, driven by world.tick() in synchronous mode)."""
    bp_lib = world.get_blueprint_library()
    sensors: list["carla.Actor"] = []
    state = {"events": [], "images_written": 0}

    if args.images_out:
        args.images_out.mkdir(parents=True, exist_ok=True)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.image_width))
        cam_bp.set_attribute("image_size_y", str(args.image_height))
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)

        every_n = max(args.image_every_n_ticks, 1)

        def _on_image(image, _every_n=every_n, _dir=args.images_out, _state=state):
            if image.frame % _every_n == 0:
                image.save_to_disk(str(_dir / f"frame_{image.frame:06d}.png"))
                _state["images_written"] += 1

        camera.listen(_on_image)
        sensors.append(camera)
        print(f"[capture] camera attached -> {args.images_out} (every {every_n} ticks)")

    if args.events_out:
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)

        def _on_collision(event, _state=state):
            other = event.other_actor
            impulse = event.normal_impulse
            _state["events"].append(
                {
                    "event_type": "collision",
                    "frame": event.frame,
                    "sim_time": round(event.timestamp, 3),
                    "actor_id": "ego",
                    "other_actor_type": other.type_id if other is not None else None,
                    "impulse": [impulse.x, impulse.y, impulse.z],
                }
            )

        col_sensor.listen(_on_collision)
        sensors.append(col_sensor)

        lane_bp = bp_lib.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)

        def _on_lane_invasion(event, _state=state):
            _state["events"].append(
                {
                    "event_type": "lane_invasion",
                    "frame": event.frame,
                    "sim_time": round(event.timestamp, 3) if hasattr(event, "timestamp") else None,
                    "actor_id": "ego",
                    "crossed_lane_markings": [str(m.type) for m in event.crossed_lane_markings],
                }
            )

        lane_sensor.listen(_on_lane_invasion)
        sensors.append(lane_sensor)
        print("[capture] collision + lane-invasion sensors attached")

    return sensors, state


def capture(world, actors, args) -> tuple[list[dict], int]:
    """Returns (records, base_frame). ``base_frame`` is the server's absolute
    frame number just before the first tick - sensor event callbacks only get
    CARLA's absolute ``event.frame``/``event.timestamp`` (server uptime, not
    capture-relative), so callers subtract this off to get a sim_time that
    actually starts at 0 like the telemetry does."""
    records: list[dict] = []
    n_ticks = int(round(args.duration * args.tick_hz))
    carla_map = world.get_map()
    base_frame = world.get_snapshot().frame

    for tick in range(n_ticks):
        world.tick()
        sim_time = round(tick / args.tick_hz, 3)

        for label, actor in actors:
            if not actor.is_alive:
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            loc, rot = transform.location, transform.rotation

            waypoint = carla_map.get_waypoint(loc, project_to_road=False)
            lane_id = waypoint.lane_id if waypoint is not None else None

            try:
                # int() both sides - mixing a VehicleLightState with a raw int
                # in a bitwise op is the same trap as set_light_state above.
                emergency_on = bool(int(actor.get_light_state()) & _EMERGENCY_LIGHT_BITS)
            except Exception:  # noqa: BLE001
                emergency_on = False

            records.append(
                {
                    "scenario": args.scenario_name,
                    "actor_id": label,
                    "sim_time": sim_time,
                    "loc_x": loc.x,
                    "loc_y": loc.y,
                    "loc_z": loc.z,
                    "vel_x": velocity.x,
                    "vel_y": velocity.y,
                    "vel_z": velocity.z,
                    "yaw_deg": rot.yaw,
                    "lane_id": lane_id,
                    "blueprint_id": actor.type_id,
                    "emergency_light_on": emergency_on,
                }
            )

        if tick % args.tick_hz == 0:
            print(f"[capture] t={sim_time:6.2f}s  ({tick}/{n_ticks} ticks)")

    return records, base_frame


def list_blueprints(world, pattern: str) -> None:
    for bp in world.get_blueprint_library().filter(pattern):
        print(bp.id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="e.g. Town10HD; omit to use whatever is loaded")
    parser.add_argument("--scenario-name", default="demo_overtake")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds of sim time to capture")
    parser.add_argument("--tick-hz", type=float, default=20.0)
    parser.add_argument("--num-vehicles", type=int, default=6, help="including ego + emergency")
    parser.add_argument("--emergency", action="store_true", default=True)
    parser.add_argument("--no-emergency", dest="emergency", action="store_false")
    parser.add_argument(
        "--emergency-ahead",
        action="store_true",
        default=True,
        help="place the emergency vehicle N metres ahead of ego on the same lane (default), "
        "instead of a random spawn point",
    )
    parser.add_argument("--no-emergency-ahead", dest="emergency_ahead", action="store_false")
    parser.add_argument("--emergency-ahead-distance", type=float, default=25.0)
    parser.add_argument("--ego-blueprint", default=DEFAULT_EGO_BLUEPRINT)
    parser.add_argument("--emergency-blueprint", default=DEFAULT_EMERGENCY_BLUEPRINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--images-out",
        type=Path,
        default=None,
        help="directory to save ego-vehicle RGB camera frames as PNGs; omit to skip image capture",
    )
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument(
        "--image-every-n-ticks",
        type=int,
        default=10,
        help="save one frame every N ticks (default: every 0.5s at 20Hz)",
    )
    parser.add_argument(
        "--events-out",
        type=Path,
        default=None,
        help="path for a JSON log of ego collision/lane-invasion events; omit to skip",
    )
    parser.add_argument(
        "--list-blueprints",
        metavar="PATTERN",
        default=None,
        help="print vehicle blueprint ids matching PATTERN (e.g. 'vehicle.*') and exit",
    )
    parser.add_argument("--seed", type=int, default=None, help="traffic manager seed for repeatable runs")
    args = parser.parse_args(argv)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    if args.list_blueprints:
        list_blueprints(world, args.list_blueprints)
        return 0

    tm = client.get_trafficmanager()
    if args.seed is not None:
        tm.set_random_device_seed(args.seed)
        random.seed(args.seed)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.tick_hz
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    actors: list[tuple[str, "carla.Actor"]] = []
    sensors: list["carla.Actor"] = []
    sensor_state: dict = {"events": [], "images_written": 0}
    base_frame = 0
    try:
        actors = spawn_actors(world, tm.get_port(), args)
        ego = actors[0][1]  # spawn_actors always puts ego first
        sensors, sensor_state = spawn_sensors(world, ego, args)
        records, base_frame = capture(world, actors, args)
    finally:
        # Sensors first (attached children), then vehicles. Stop listening
        # before destroy so no callback fires on a half-destroyed actor.
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:  # noqa: BLE001
                pass
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:  # noqa: BLE001
                pass
        # Disabling autopilot before destroy, and ticking once more while
        # still synchronous, avoids a native crash in the traffic manager that
        # can happen when destroying an autopilot-controlled actor right
        # before switching sync mode off.
        for _, actor in actors:
            try:
                if actor.is_alive:
                    actor.set_autopilot(False, tm.get_port())
            except Exception:  # noqa: BLE001
                pass
        try:
            world.tick()
        except Exception:  # noqa: BLE001
            pass
        for _, actor in actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:  # noqa: BLE001
                pass
        try:
            world.tick()
        except Exception:  # noqa: BLE001
            pass
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records))
    print(f"[capture] wrote {len(records)} rows ({len(actors)} actors) -> {args.out}")

    if args.images_out:
        print(f"[capture] wrote {sensor_state['images_written']} images -> {args.images_out}")
    if args.events_out:
        args.events_out.parent.mkdir(parents=True, exist_ok=True)
        events = sorted(sensor_state["events"], key=lambda e: e["frame"])
        # event.timestamp/frame from CARLA are server-absolute (uptime since
        # server start), not capture-relative - rewrite sim_time to start at 0
        # like the telemetry does, so the two logs are comparable.
        for event in events:
            event["sim_time"] = round((event["frame"] - base_frame) / args.tick_hz, 3)
        args.events_out.write_text(json.dumps(events))
        print(f"[capture] wrote {len(events)} events -> {args.events_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
