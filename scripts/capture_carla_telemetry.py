"""Capture real CARLA telemetry to a JSON log that etl.extract.extract_carla can read.

Runs against a live ``CarlaUE4.exe`` server. Spawns an ego vehicle, one emergency
vehicle, and N background vehicles on autopilot; steps the simulation in
synchronous mode at a fixed tick rate; records every actor's state each tick.

The output is a flat JSON list of records with exactly the keys
``etl.extract.extract_carla.RAW_COLUMNS`` expects, so it plugs straight into the
main pipeline:

    python -m etl.load.load_bronze --domain carla_telemetry \\
        --name demo_overtake --scenario-log data/raw_captures/demo_overtake.json

IMPORTANT - this runs in a SEPARATE Python environment matching whatever
version your CARLA build's client wheel targets (check
PythonAPI/carla/dist/*.whl for the cp3X tag), not the project's main venv.
See scripts/README.md.

Run:
    python capture_carla_telemetry.py --scenario-name demo_overtake \\
        --duration 10 --out data/raw_captures/demo_overtake.json

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
    ego = _spawn(world, ego_bp, spawn_points.pop(), role_name="ego")
    if ego is None:
        raise RuntimeError(f"could not spawn ego blueprint {args.ego_blueprint!r}")
    ego.set_autopilot(True, tm_port)
    actors.append(("ego", ego))

    if args.emergency:
        em_bp = bp_lib.find(args.emergency_blueprint)
        em = _spawn(world, em_bp, spawn_points.pop())
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


def capture(world, actors, args) -> list[dict]:
    records: list[dict] = []
    n_ticks = int(round(args.duration * args.tick_hz))
    carla_map = world.get_map()

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

    return records


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
    parser.add_argument("--ego-blueprint", default=DEFAULT_EGO_BLUEPRINT)
    parser.add_argument("--emergency-blueprint", default=DEFAULT_EMERGENCY_BLUEPRINT)
    parser.add_argument("--out", type=Path, required=True)
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
    try:
        actors = spawn_actors(world, tm.get_port(), args)
        records = capture(world, actors, args)
    finally:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
