# DriveMind Data Dictionary

Single reference for every schema the ETL pipeline produces. Column contracts
here are enforced in code by the pandera classes in [etl/schemas/](etl/schemas/)
and by the PostgreSQL DDL in [etl/db/schema.sql](etl/db/schema.sql).

Layers:

| Layer  | Guarantee                                              | Format          |
|--------|-------------------------------------------------------|-----------------|
| bronze | raw, as-ingested; no schema guarantee                 | Parquet         |
| silver | cleaned, validated, conformed to the common schema    | Parquet         |
| gold   | ML / risk features + train/val/test split             | Parquet         |

---

## 1. CARLA telemetry - common schema (Silver)

File: `data/silver/carla_telemetry/<scenario>.parquet`
Schema: `CarlaTelemetrySchema` in [etl/schemas/carla_telemetry.py](etl/schemas/carla_telemetry.py)
Grain: one row per (vehicle, simulation tick) within one scenario run.

| Column            | Type        | Unit    | Nullable | Range / domain                        | Description |
|-------------------|-------------|---------|----------|---------------------------------------|-------------|
| `vehicle_id`      | string      | -       | no       | `"ego"` for the ego vehicle           | Stable actor id within a scenario run. |
| `timestamp`       | float64     | seconds | no       | `>= 0`                                | Seconds since scenario start (CARLA snapshot time). |
| `x`               | float64     | metres  | no       | `[-1e6, 1e6]`                         | World X position, CARLA map frame. |
| `y`               | float64     | metres  | no       | `[-1e6, 1e6]`                         | World Y position, CARLA map frame. |
| `z`               | float64     | metres  | no       | `[-1e6, 1e6]`                         | World Z position, CARLA map frame. |
| `speed`           | float64     | m/s     | no       | `[0, 150]`                            | Speed magnitude (norm of velocity vector). |
| `acceleration`    | float64     | m/s^2   | yes      | `[-50, 50]`                           | Signed longitudinal acceleration; null on a track's first tick; clipped to bounds. |
| `heading`         | float64     | degrees | no       | `[-180, 180]`                        | Yaw normalised to `[-180, 180]` (CARLA `rotation.yaw`). |
| `lane_id`         | Int64       | -       | yes      | any integer; sign = direction         | CARLA waypoint lane id; null when off-road. |
| `vehicle_type`    | string      | -       | no       | `car, truck, bus, van, motorcycle, bicycle, emergency, unknown` | Coarse category mapped from the blueprint id. |
| `is_emergency`    | boolean     | -       | no       | `{true, false}`                       | Active emergency vehicle (lights/siren flag or emergency blueprint). |
| `distance_to_ego` | float64     | metres  | yes      | `>= 0`                                | Euclidean distance to the ego vehicle; `0.0` on the ego's own row; null if no ego row at that tick. |
| `relative_speed`  | float64     | m/s     | yes      | `[-150, 150]`                        | Closing speed along the ego->vehicle axis: **positive = gap shrinking**; `0.0` on the ego's own row. |

Cross-row constraints (schema `Config`):

- `(vehicle_id, timestamp)` is unique.
- If any `distance_to_ego` / `relative_speed` value is populated, at least one
  `vehicle_id == "ego"` row must exist to anchor it.

### Raw CARLA telemetry (Bronze)

File: `data/bronze/carla_telemetry/<scenario>.parquet`. No enforced schema. The
stub extractor emits these columns, and
[etl/transform/clean_carla.py](etl/transform/clean_carla.py) maps them to the
common schema:

`scenario, actor_id, sim_time, loc_x, loc_y, loc_z, vel_x, vel_y, vel_z,
yaw_deg, lane_id, blueprint_id, emergency_light_on`

Derivations in the clean step: `speed` from the velocity vector norm;
`acceleration` from per-vehicle `diff(speed) / diff(timestamp)`; `heading` by
normalising `yaw_deg`; `vehicle_type` by matching `blueprint_id` substrings;
`distance_to_ego` / `relative_speed` per tick against the `ego` row (using the
raw velocity components before they are dropped).

---

## 2. IDD detection labels (Silver)

File: `data/silver/idd/idd_<variant>_<split>.parquet` for `variant in {detection, temporal}`
Schema: `IDDDetectionLabelSchema` in [etl/schemas/idd_labels.py](etl/schemas/idd_labels.py)
Grain: one row per annotated bounding box.

| Column         | Type   | Nullable | Range / domain                     | Description |
|----------------|--------|----------|------------------------------------|-------------|
| `image_id`     | string | no       | -                                  | Unique image identifier. |
| `image_path`   | string | no       | -                                  | Image path relative to the Bronze IDD root. |
| `image_width`  | int    | no       | `> 0`                              | Image width in pixels. |
| `image_height` | int    | no       | `> 0`                              | Image height in pixels. |
| `split`        | string | no       | `train, val, test`                 | Dataset split. |
| `variant`      | string | no       | `detection, temporal`              | Source IDD variant. |
| `sequence_id`  | string | yes      | -                                  | Video sequence id (Temporal only). |
| `frame_index`  | Int64  | yes      | `>= 0`                             | Frame position in the sequence (Temporal only). |
| `class_name`   | string | no       | `IDD_DETECTION_CLASSES` (placeholder) | Canonical object class. |
| `class_id`     | Int64  | no       | `>= 0`                             | Index into `IDD_DETECTION_CLASSES`. |
| `bbox_x`       | float64| no       | `>= 0`                             | Box top-left x, pixels. |
| `bbox_y`       | float64| no       | `>= 0`                             | Box top-left y, pixels. |
| `bbox_w`       | float64| no       | `> 0`                              | Box width, pixels. |
| `bbox_h`       | float64| no       | `> 0`                              | Box height, pixels. |
| `occluded`     | boolean| yes      | `{true, false}`                    | Annotator "occluded" flag. |
| `truncated`    | boolean| yes      | `{true, false}`                    | Annotator "truncated" flag. |

Cross-row constraint: `bbox_x + bbox_w <= image_width + 1` and
`bbox_y + bbox_h <= image_height + 1` (1 px rounding slack).

---

## 3. IDD segmentation labels (Silver)

File: `data/silver/idd/idd_<variant>_<split>.parquet` for `variant in {segmentation, lite}`
Schema: `IDDSegmentationLabelSchema` in [etl/schemas/idd_labels.py](etl/schemas/idd_labels.py)
Grain: one row per polygon / instance.

| Column            | Type   | Nullable | Range / domain                  | Description |
|-------------------|--------|----------|---------------------------------|-------------|
| `image_id`        | string | no       | -                               | Unique image identifier. |
| `image_path`      | string | no       | -                               | Image path relative to the Bronze IDD root. |
| `image_width`     | int    | no       | `> 0`                           | Image width, pixels. |
| `image_height`    | int    | no       | `> 0`                           | Image height, pixels. |
| `split`           | string | no       | `train, val, test`              | Dataset split. |
| `variant`         | string | no       | `segmentation, lite`            | Source IDD variant. |
| `gt_json_path`    | string | no       | -                               | Path to the source Cityscapes-style polygon JSON. |
| `label`           | string | no       | `IDD_LEVEL3_CLASSES` (verified, 42 names) | Canonical level-3 class name. |
| `level3_id`       | Int64  | no       | `0-25, 255`                     | Official AutoNUE level3Id (`IDD_LEVEL3_ID_MAP`); multiple names share an id (e.g. caravan/trailer/train/vehicle fallback -> 12); `255` = void/ignore. |
| `instance_id`     | Int64  | yes      | `>= 0`                          | Instance id within the image (null for stuff classes). |
| `polygon_area_px` | float64| yes      | `>= 0`                          | Polygon area, pixels (shoelace). |
| `num_vertices`    | Int64  | yes      | `>= 3`                          | Vertex count of the polygon. Rows with `< 3` are dropped during `normalize_idd` (degenerate annotation) and counted in the quality report. |

> `IDD_LEVEL3_CLASSES` / `IDD_LEVEL3_ID_MAP` are **verified**: the 42 names came
> from a full scan of a real IDD Segmentation Part 2 train+val set (602k
> polygons, 0 unknown classes), and the ids are transcribed from
> AutoNUE/public-code (`helpers/anue_labels.py`, `level3Id`). `IDD_DETECTION_CLASSES`
> is still a **placeholder** - no real IDD Detection data has gone through the
> pipeline yet.

---

## 4. Gold feature tables

### 4.1 `trajectory_windows`

File: `data/gold/carla_telemetry/trajectory_windows.parquet`
Grain: one row per (vehicle, anchor tick) with enough history and future.
Defaults: `PAST_TICKS = 20`, `FUTURE_TICKS = 30`.

| Column             | Type            | Description |
|--------------------|-----------------|-------------|
| `scenario`         | string          | Source scenario name. |
| `vehicle_id`       | string          | Actor id. |
| `vehicle_type`     | string          | Category at the anchor tick. |
| `is_emergency`     | boolean         | Emergency flag at the anchor tick. |
| `anchor_timestamp` | float64         | Timestamp of the split point between past and future. |
| `past_x`           | list<float64>   | `PAST_TICKS` historical X positions. |
| `past_y`           | list<float64>   | `PAST_TICKS` historical Y positions. |
| `past_speed`       | list<float64>   | `PAST_TICKS` historical speeds. |
| `past_heading`     | list<float64>   | `PAST_TICKS` historical headings. |
| `future_x`         | list<float64>   | `FUTURE_TICKS` future X positions (prediction target). |
| `future_y`         | list<float64>   | `FUTURE_TICKS` future Y positions (prediction target). |
| `split`            | string          | `train, val, test` (scenario-level). |

### 4.2 `risk_features`

File: `data/gold/carla_telemetry/risk_features.parquet`
Grain: one row per (scenario, timestamp) with an ego row present.

| Column                        | Type    | Unit   | Description |
|-------------------------------|---------|--------|-------------|
| `scenario`                    | string  | -      | Source scenario name. |
| `timestamp`                   | float64 | s      | Simulation tick. |
| `ego_speed`                   | float64 | m/s    | Ego speed. |
| `ego_acceleration`            | float64 | m/s^2  | Ego longitudinal acceleration (nullable). |
| `ego_heading`                 | float64 | deg    | Ego heading. |
| `nearest_vehicle_dist`        | float64 | m      | Distance to the closest non-ego vehicle (NaN if none). |
| `nearest_closing_speed`       | float64 | m/s    | `relative_speed` of that nearest vehicle. |
| `time_to_collision`           | float64 | s      | `nearest_vehicle_dist / max(nearest_closing_speed, 1e-3)`; `inf` when not closing. |
| `n_within_10m`                | int     | count  | Non-ego vehicles within 10 m. |
| `n_within_30m`                | int     | count  | Non-ego vehicles within 30 m. |
| `emergency_vehicle_within_50m`| boolean | -      | Any emergency vehicle within 50 m. |
| `split`                       | string  | -      | `train, val, test` (scenario-level). |

> These are a first cut. The risk-engine owner sets the final feature list and
> the authoritative TTC definition.

---

## 5. PostgreSQL metadata tables

Defined in [etl/db/schema.sql](etl/db/schema.sql). Stores metadata and pointers
only; bulk data stays in Parquet.

### `scenario_runs` - one row per CARLA scenario execution

| Column                  | Type          | Notes |
|-------------------------|---------------|-------|
| `run_id`                | UUID PK       | `gen_random_uuid()`. |
| `scenario_name`         | TEXT NOT NULL | Logical scenario name. |
| `scenario_config`       | JSONB         | scenario_runner params, route, triggers. |
| `carla_version`         | TEXT          | |
| `map_name`              | TEXT          | e.g. `Town10HD`. |
| `seed`                  | BIGINT        | RNG seed for reproducibility. |
| `weather`               | TEXT          | Weather preset / description. |
| `started_at` / `ended_at` | TIMESTAMPTZ | Wall-clock bounds. |
| `duration_sec`          | NUMERIC(10,3) | Simulated duration. |
| `tick_rate_hz`          | NUMERIC(6,2)  | Fixed-delta tick rate. |
| `num_frames`            | INTEGER       | `>= 0`. |
| `num_vehicles`          | INTEGER       | `>= 0`. |
| `has_emergency_vehicle` | BOOLEAN       | Default `FALSE`. |
| `status`                | TEXT          | `pending, running, completed, failed`. |
| `bronze_path`           | TEXT          | Repo-relative path to the raw telemetry Parquet. **Unique** (partial index, non-null); `load_bronze --push` upserts the run row on this key and stamps `num_frames` / `num_vehicles` / `duration_sec` / `tick_rate_hz` / `has_emergency_vehicle` / `status`. |
| `notes`                 | TEXT          | Free text. |
| `created_at` / `updated_at` | TIMESTAMPTZ | `updated_at` maintained by trigger. |

### `experiments` - one row per model training / evaluation run

| Column           | Type          | Notes |
|------------------|---------------|-------|
| `experiment_id`  | UUID PK       | `gen_random_uuid()`. |
| `name`           | TEXT NOT NULL | |
| `description`    | TEXT          | |
| `component`      | TEXT          | `perception, trajectory_prediction, emergency_vehicle, risk_engine, path_planning, control, other`. |
| `git_commit`     | TEXT          | Code revision. |
| `dataset_version`| TEXT          | Gold split tag trained on. |
| `hyperparams`    | JSONB         | |
| `metrics`        | JSONB         | Final / best metrics. |
| `artifact_path`  | TEXT          | Checkpoint / export location. |
| `started_at` / `ended_at` | TIMESTAMPTZ | |
| `status`         | TEXT          | `pending, running, completed, failed`. |
| `created_by`     | TEXT          | |
| `created_at` / `updated_at` | TIMESTAMPTZ | trigger-maintained. |

### `data_manifest` - one row per materialised file

| Column              | Type          | Notes |
|---------------------|---------------|-------|
| `manifest_id`       | UUID PK       | `gen_random_uuid()`. |
| `layer`             | TEXT NOT NULL | `bronze, silver, gold`. |
| `domain`            | TEXT NOT NULL | `idd, carla_telemetry`. |
| `source_variant`    | TEXT          | e.g. `detection`, or the scenario name. |
| `relative_path`     | TEXT NOT NULL | Path under the repo root. |
| `file_format`       | TEXT          | `parquet, json, png, csv, ...`. |
| `row_count`         | BIGINT        | `>= 0` (from Parquet metadata). |
| `file_size_bytes`   | BIGINT        | `>= 0`. |
| `checksum_sha256`   | TEXT          | Content hash. |
| `schema_name`       | TEXT          | pandera schema that validated the file. |
| `validation_status` | TEXT          | `passed, failed, skipped`. |
| `validation_report` | JSONB         | Quality-report payload. |
| `run_id`            | UUID FK       | -> `scenario_runs` (nullable). |
| `experiment_id`     | UUID FK       | -> `experiments` (nullable). |
| `created_at` / `updated_at` | TIMESTAMPTZ | trigger-maintained. |
| unique              | -             | `(layer, relative_path)`. |
