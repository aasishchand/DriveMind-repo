# DriveMind - ETL (Member 1: Data Engineering & ETL)

Data pipeline for the DriveMind self-driving simulation project. It turns two raw
data sources into validated, ML-ready feature tables:

| Source | What it is | Feeds |
|--------|-----------|-------|
| **IDD** (India Driving Dataset) | Static labelled road images (Detection / Temporal / Lite / Segmentation variants) | Perception model training |
| **CARLA telemetry** | Per-scenario simulation logs: vehicle positions, speeds, headings, distances, emergency-vehicle flags | Trajectory / emergency-vehicle prediction, risk engine |

> Status: **scaffold**. Every stage runs end to end on synthetic stub data.
> Real IDD parsing and real CARLA log ingestion are not wired up yet.

---

## Architecture: medallion (Bronze -> Silver -> Gold)

```
                 extract/                 transform/               load/
                 --------                 ----------               ----
IDD annotations ─┐                   ┌─ normalize_idd ─┐      ┌─ load_silver ─┐
                 ├─ extract_idd ─────┤                 ├──────┤               │
CARLA logs ──────┤                   └─ clean_carla ───┘      ├─ load_bronze  │
                 └─ extract_carla ──────────────────────────  └─ load_gold ───┘
                                                                     │
  data/bronze/            data/silver/                        data/gold/
  raw, as-ingested        cleaned + validated                 features + splits
  (no schema)             (common schema, pandera-checked)    (train/val/test)
```

- **Bronze** - raw extractor output written verbatim as Parquet. No schema
  guarantee. Append-only; a re-run overwrites the file for that scenario/variant.
- **Silver** - data mapped to the **common schema** and passed through pandera.
  A file only lands here if validation succeeds.
- **Gold** - feature tables derived from Silver, with a deterministic,
  scenario-level train/val/test split.

Structured metadata (which scenarios ran, which experiments used which data,
what files exist) lives in **PostgreSQL**; the bulk data stays in Parquet. The
DB holds pointers, not payloads.

---

## Repository layout

```
DriveMind/
  data/
    bronze/   {idd,carla_telemetry}/     # raw parquet (git-ignored)
    silver/   {idd,carla_telemetry}/     # validated parquet (git-ignored)
    gold/     carla_telemetry/           # feature tables (git-ignored)
    _manifest/manifest.jsonl             # local stand-in for data_manifest table
  etl/
    config.py            # paths, domain constants, logger
    run_demo.py          # run the whole synthetic pipeline end to end (+ --push)
    schemas/
      carla_telemetry.py # CarlaTelemetrySchema (the common schema)
      idd_labels.py      # IDDDetectionLabelSchema, IDDSegmentationLabelSchema
    extract/
      extract_idd.py     # parse IDD annotations  (stub + real VOC/JSON parsers)
      extract_carla.py   # read a CARLA scenario log (stub + synthetic data)
    transform/
      clean_carla.py     # raw -> common schema + 4 data-quality checks
      normalize_idd.py    # raw class strings -> canonical classes + ids
    load/
      load_bronze.py     # write raw output, register in manifest
      load_silver.py     # validate + write, register in manifest
      load_gold.py       # trajectory windows + risk features + split
    db/
      schema.sql         # Postgres DDL: scenario_runs, experiments, data_manifest
      connection.py      # psycopg connection seam (reads DRIVEMIND_PG_* env vars)
      manifest.py        # build/append/upsert data_manifest rows
      records.py         # insert/update helpers for scenario_runs + experiments
      verify_db.py       # connection + schema + round-trip smoke test
  tests/                  # pytest suite - schemas, clean_carla, normalize_idd, load_gold, manifest
  data_dictionary.md     # every column of every schema, plus the DDL
  requirements.txt
  pytest.ini
  scripts/
    capture_carla_telemetry.py  # runs against a live CARLA server, writes a raw JSON log
    README.md                   # separate Python 3.12 venv setup + capture -> load walkthrough
```

---

## How the pieces fit

1. **Extract** (`etl/extract/`) reads a source and returns a raw `DataFrame`. It
   does not reshape or validate. `extract_carla` emits raw CARLA-style columns;
   `extract_idd` emits raw label rows (with real Pascal-VOC and polygon-JSON
   parsers for when actual data is present).

2. **`load_bronze`** writes that raw frame to `data/bronze/<domain>/<name>.parquet`
   and appends a `data_manifest` row (`validation_status = skipped`). For carla,
   `--push` also upserts a `scenario_runs` row (keyed on the Bronze path) holding
   a telemetry summary - `num_frames`, `num_vehicles`, `duration_sec`,
   `tick_rate_hz`, `has_emergency_vehicle`, `status` - and stamps the manifest
   row's `run_id`.

3. **Transform** (`etl/transform/`) reads a Bronze file and produces
   common-schema rows:
   - `clean_carla` maps raw columns, derives `speed / acceleration / heading /
     distance_to_ego / relative_speed`, and runs the four checks Member 1 owns:
     **(a)** missing values in required columns, **(b)** invalid coordinates /
     speeds, **(c)** broken trajectories (implausible position jumps),
     **(d)** timestamp continuity (monotonic, roughly constant tick spacing).
     It returns the cleaned frame plus a quality report.
   - `normalize_idd` canonicalises raw class strings, drops out-of-vocabulary
     labels, attaches integer class ids, and de-duplicates.

4. **`load_silver`** runs the matching transform, validates the result against
   the pandera schema, writes `data/silver/<domain>/<name>.parquet`, and records
   the schema name + quality report in the manifest. For carla, `--push` links
   the silver manifest row to the same `scenario_runs` row as its Bronze source.

5. **`load_gold`** reads Silver telemetry and builds two products:
   - `trajectory_windows` - fixed-length past/future position windows per
     vehicle (prediction training samples).
   - `risk_features` - per-tick ego-centric features (nearest-vehicle distance,
     closing speed, time-to-collision, neighbour counts, emergency proximity).
   Each scenario is assigned to `train` / `val` / `test` by a hash of its name,
   so all ticks from one scenario stay in one split.

The **common schema** for CARLA telemetry (the contract between Silver and
everything downstream):

```
vehicle_id, timestamp, x, y, z, speed, acceleration, heading, lane_id,
vehicle_type, is_emergency, distance_to_ego, relative_speed
```

Full column definitions, types, units, and ranges are in
[data_dictionary.md](data_dictionary.md).

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: **pandas**, **pyarrow**, **pandera**, **psycopg[binary]** (the
PostgreSQL driver). Nothing else without team sign-off.

Run every script as a module **from the repo root** so `import etl...` resolves.

---

## Quickstart (synthetic data, no real sources)

Whole pipeline, both domains, all three layers:

```bash
python -m etl.run_demo               # files + local data/_manifest/manifest.jsonl
python -m etl.run_demo --push        # also upsert every manifest row into PostgreSQL
```

`--push` needs the DB reachable and `DRIVEMIND_PG_PASSWORD` set; check first with
`python -m etl.db.verify_db`.

The equivalent step-by-step, if you want to run one stage at a time:

```bash
python -c "from etl.config import ensure_data_dirs; ensure_data_dirs()"

# CARLA telemetry: bronze -> silver -> gold
python -m etl.load.load_bronze --domain carla_telemetry --name demo_overtake
python -m etl.load.load_silver --domain carla_telemetry \
    --bronze-file data/bronze/carla_telemetry/demo_overtake.parquet
python -m etl.load.load_gold --silver-glob "data/silver/carla_telemetry/*.parquet"

# IDD labels: bronze -> silver
python -m etl.load.load_bronze --domain idd --variant detection --split train
python -m etl.load.load_silver --domain idd \
    --bronze-file data/bronze/idd/idd_detection_train.parquet
```

Any `load_*` step also takes `--push` (upsert manifest row to Postgres) and
`--emit-sql` (print the INSERT instead). Individual stages run standalone too:

```bash
python -m etl.extract.extract_carla --scenario-name demo_overtake --peek 10
python -m etl.transform.clean_carla --bronze-file data/bronze/carla_telemetry/demo_overtake.parquet
```

---

## Tests

```bash
pytest
```

35 tests, no database or dataset files required - schemas, `clean_carla`'s four
DQ checks, `normalize_idd` (unknown-class and degenerate-polygon drops, the
official `level3_id` mapping), `load_gold` (split determinism/distribution,
risk features, trajectory windows), and the pure-Python half of the manifest
helpers. Runs in under a second.

One more test is DB-gated and skipped by default:

```bash
$env:DRIVEMIND_PG_PASSWORD = "your-postgres-password"   # PowerShell
pytest tests/test_db_live.py -v
```

It wraps `etl.db.verify_db` and rolls back its own transaction, so it never
leaves anything behind.

---

## Database

Create the DB and apply the schema (one time):

```bash
createdb drivemind
psql -d drivemind -f etl/db/schema.sql
```

Three tables (details in [data_dictionary.md](data_dictionary.md)):

- **`scenario_runs`** - one row per CARLA scenario execution.
- **`experiments`** - one row per model training / evaluation run.
- **`data_manifest`** - one row per file in any layer (path, checksum, row
  count, validating schema, `passed/failed/skipped`).

Connection settings come from environment variables (see
[etl/db/connection.py](etl/db/connection.py)). Set the password once per shell:

```powershell
$env:DRIVEMIND_PG_PASSWORD = "your-postgres-password"
# optional overrides: DRIVEMIND_PG_HOST / _PORT / _DB / _USER
```

Smoke-test the setup (connects, checks the three tables, does an
insert/read/rollback round-trip - leaves nothing behind):

```bash
python -m etl.db.verify_db
```

**Writing to the DB:**

- Every load step always appends to `data/_manifest/manifest.jsonl` (durable
  local log, no DB needed).
- Add `--push` to any `load_*.py` to also upsert the `data_manifest` row into
  PostgreSQL (keyed on `(layer, relative_path)`). Without a reachable DB it
  prints a one-line skip notice and carries on.
- Add `--emit-sql` to print the `INSERT` instead of running it.
- [etl/db/records.py](etl/db/records.py) has `insert_scenario_run` /
  `update_scenario_run` / `insert_experiment` / `update_experiment` for the
  other two tables, used from Python:

  ```python
  from etl.db.connection import connection
  from etl.db.records import insert_scenario_run, update_scenario_run

  with connection() as conn:
      run_id = insert_scenario_run(conn, scenario_name="demo_overtake",
                                   map_name="Town10HD", status="running")
      update_scenario_run(conn, run_id, status="completed", num_frames=1200)
  ```

---

## What is stubbed vs. real

| Piece | State |
|-------|-------|
| Medallion directory layout | real |
| pandera schemas (both domains) | real |
| Postgres DDL | real |
| `clean_carla` derivations + 4 DQ checks | real logic, only run on synthetic data |
| `normalize_idd` alias maps + id assignment | real logic, alias maps need extending |
| VOC / polygon-JSON parsers in `extract_idd` | real (stdlib) |
| IDD download | not implemented (manual download) |
| CARLA log format / field names | real capture script written ([scripts/capture_carla_telemetry.py](scripts/capture_carla_telemetry.py)); not yet run against a live server |
| `IDD_LEVEL3_CLASSES` / `IDD_LEVEL3_ID_MAP` | **verified** - 42 real class names from a full scan of IDD Segmentation Part 2 (train+val, 602k polygons, 0 unknown), ids transcribed from AutoNUE/public-code `anue_labels.py` |
| `IDD_DETECTION_CLASSES` | still a **placeholder** - no real IDD Detection data run through it yet |
| Gold feature list / TTC definition | first cut - risk-engine owner to finalise |
| DB writes (`data_manifest`, `scenario_runs`, `experiments`) | real - psycopg wired; `--push` + `db/records.py` |
| `scenario_runs` row on carla ingest | real - `load_bronze --push` upserts it (keyed on `bronze_path`) and links `data_manifest.run_id`; `load_silver --push` links the silver row to the same run |
| `scenario_runs` timing / config fields (`started_at`, `map_name`, `seed`, `weather`, `scenario_config`) | not populated - come from the scenario runner, not the telemetry file |

---

## Reference repos (patterns only, not vendored)

- **AutoNUE/public-code** - official IDD devkit; label parsing, class hierarchy
  (`helpers/anue_labels.py`).
- **KevinLADLee/carla_dataset_tools** - CARLA sensor / telemetry capture patterns.
- **carla-simulator/scenario_runner** - scenario metadata structure
  (feeds `scenario_runs.scenario_config`).

---

## Open decisions (ask the team before acting)

1. **Silver reject handling** - currently offending rows are dropped/clipped and
   counted. Switch to a quarantine table?
2. **`scenario_runs` lifecycle** - `load_bronze --push` already creates/updates
   the row on ingest (see below); confirm that's the right owner once the
   scenario runner (another member's side) exists, vs. it creating the row
   itself and `load_bronze` only linking to it.
3. **Orchestration** - deferred until the script pipeline is stable, per plan.

Resolved: real CARLA log schema (`extract_carla.RAW_COLUMNS`) and the IDD
segmentation class list are both now verified against real data instead of
assumed; `pytest` was added and a suite is in `tests/`.
