# scripts/ - CARLA telemetry capture

Standalone tool that talks to a live CARLA server and writes a raw telemetry
log for the main ETL pipeline to ingest. It runs in its **own** Python
environment because the CARLA client library is a version-pinned wheel bundled
inside the CARLA download, separate from `etl/`'s Python 3.14 venv.

## One-time setup

1. Install [CARLA](https://github.com/carla-simulator/carla/releases) and note
   the extraction folder.
2. Find the client wheel's Python tag - it changes between CARLA versions:
   ```powershell
   Get-ChildItem "<CARLA_ROOT>\PythonAPI\carla\dist"
   ```
   The filename looks like `carla-<version>-cp312-cp312-win_amd64.whl` - `cp312`
   means it needs **Python 3.12**. (For DriveMind's CARLA 0.9.16 build this is
   already confirmed as Python 3.12.)
3. Create the capture venv and install the wheel (repo root, project's own
   Python is untouched):
   ```powershell
   py -3.12 -m venv .venv-carla
   .venv-carla\Scripts\pip install "<CARLA_ROOT>\PythonAPI\carla\dist\carla-<version>-cp312-cp312-win_amd64.whl"
   ```
   If `py -3.12` isn't available, install it first: `py install 3.12` (the
   modern `py` launcher can fetch versions on its own), or grab it from
   python.org.

## Running a capture

1. **Start the CARLA server** - run `CarlaUE4.exe` in the CARLA folder and
   wait for the 3D window to appear (first launch is slow: shader compile).
2. **Run the capture script** in the CARLA venv:
   ```powershell
   .venv-carla\Scripts\python scripts\capture_carla_telemetry.py `
       --scenario-name demo_overtake `
       --duration 10 `
       --num-vehicles 6 `
       --out data\raw_captures\demo_overtake.json
   ```
   This spawns an ego vehicle (`vehicle.tesla.model3`), one emergency vehicle
   (`vehicle.carlamotors.firetruck`) with its light bar on, and background
   traffic, all on autopilot; ticks the simulation at 20 Hz for 10 sim-seconds;
   records every actor's transform/velocity each tick; writes JSON.

   If a blueprint id doesn't exist in your CARLA build, list what's available
   and override:
   ```powershell
   .venv-carla\Scripts\python scripts\capture_carla_telemetry.py --list-blueprints "vehicle.*"
   .venv-carla\Scripts\python scripts\capture_carla_telemetry.py ... --ego-blueprint vehicle.audi.a2
   ```

3. **Load the capture into the pipeline** - back in the *main* project venv
   (Python 3.14, from the repo root):
   ```powershell
   python -m etl.load.load_bronze --domain carla_telemetry `
       --name demo_overtake --scenario-log data\raw_captures\demo_overtake.json
   python -m etl.load.load_silver --domain carla_telemetry `
       --bronze-file data\bronze\carla_telemetry\demo_overtake.parquet
   python -m etl.load.load_gold --silver-glob "data/silver/carla_telemetry/*.parquet"
   ```
   Add `--push` to any of these to also write to PostgreSQL.

## Output format

A flat JSON array of per-tick, per-actor records with these keys (must match
`etl.extract.extract_carla.RAW_COLUMNS`):

```
scenario, actor_id, sim_time, loc_x, loc_y, loc_z,
vel_x, vel_y, vel_z, yaw_deg, lane_id, blueprint_id, emergency_light_on
```

`actor_id` is `"ego"` for the ego vehicle (matches the common schema's
`vehicle_id == "ego"` convention used throughout `clean_carla`) and
`"veh_<carla_actor_id>"` for everything else.

For lane-choice demonstrations, use the separate Traffic Manager expert
collector described in [the policy-learning notes](../etl/policy_learning/README.md).
It records lane-change labels and control values in a policy CSV without
changing this ETL capture schema.
