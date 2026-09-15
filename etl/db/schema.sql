-- DriveMind :: PostgreSQL metadata & experiment-tracking schema
-- Owner: Member 1 (Data Engineering & ETL)
--
-- Apply with:
--     createdb drivemind
--     psql -d drivemind -f etl/db/schema.sql
--
-- Scope: this database stores *metadata only* - pointers to bulk data, not the
-- bulk data itself. Telemetry / images / features live as Parquet + files under
-- data/{bronze,silver,gold}/ and are referenced here by relative_path.
--
-- Three tables:
--   scenario_runs  - one row per CARLA scenario execution (produces telemetry)
--   experiments    - one row per model training / evaluation run
--   data_manifest  - one row per file materialised in any layer

BEGIN;

-- gen_random_uuid() is built in on PostgreSQL 13+. On older servers:
--   CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- shared updated_at trigger
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- scenario_runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scenario_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_name       TEXT NOT NULL,
    scenario_config     JSONB,                       -- scenario_runner params, route, triggers
    carla_version       TEXT,
    map_name            TEXT,
    seed                BIGINT,
    weather             TEXT,
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    duration_sec        NUMERIC(10, 3),
    tick_rate_hz        NUMERIC(6, 2),
    num_frames          INTEGER CHECK (num_frames IS NULL OR num_frames >= 0),
    num_vehicles        INTEGER CHECK (num_vehicles IS NULL OR num_vehicles >= 0),
    has_emergency_vehicle BOOLEAN NOT NULL DEFAULT FALSE,
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    bronze_path         TEXT,                        -- relative path to raw telemetry parquet
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scenario_runs_name    ON scenario_runs (scenario_name);
CREATE INDEX IF NOT EXISTS idx_scenario_runs_status  ON scenario_runs (status);
CREATE INDEX IF NOT EXISTS idx_scenario_runs_started ON scenario_runs (started_at);

-- One Bronze telemetry file == one scenario run. Lets load_bronze upsert the
-- run row idempotently (re-running ingest updates instead of duplicating).
CREATE UNIQUE INDEX IF NOT EXISTS uq_scenario_runs_bronze_path
    ON scenario_runs (bronze_path) WHERE bronze_path IS NOT NULL;

DROP TRIGGER IF EXISTS trg_scenario_runs_updated_at ON scenario_runs;
CREATE TRIGGER trg_scenario_runs_updated_at
    BEFORE UPDATE ON scenario_runs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- experiments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    description         TEXT,
    component           TEXT CHECK (component IN
                            ('perception', 'trajectory_prediction', 'emergency_vehicle',
                             'risk_engine', 'path_planning', 'control', 'other')),
    git_commit          TEXT,
    dataset_version     TEXT,                        -- Gold split tag this run trained on
    hyperparams         JSONB,
    metrics             JSONB,                       -- final / best metrics as key-value
    artifact_path       TEXT,                        -- model checkpoint / export location
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_by          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiments_component ON experiments (component);
CREATE INDEX IF NOT EXISTS idx_experiments_status    ON experiments (status);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset   ON experiments (dataset_version);

DROP TRIGGER IF EXISTS trg_experiments_updated_at ON experiments;
CREATE TRIGGER trg_experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- data_manifest
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_manifest (
    manifest_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    layer               TEXT NOT NULL CHECK (layer IN ('bronze', 'silver', 'gold')),
    domain              TEXT NOT NULL CHECK (domain IN ('idd', 'carla_telemetry')),
    source_variant      TEXT,                        -- e.g. 'detection' / 'temporal' / scenario name
    relative_path       TEXT NOT NULL,               -- path under the repo root
    file_format         TEXT,                        -- 'parquet' | 'json' | 'png' | 'csv' | ...
    row_count           BIGINT CHECK (row_count IS NULL OR row_count >= 0),
    file_size_bytes     BIGINT CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    checksum_sha256     TEXT,
    schema_name         TEXT,                        -- pandera schema that validated the file
    validation_status   TEXT NOT NULL DEFAULT 'skipped'
                            CHECK (validation_status IN ('passed', 'failed', 'skipped')),
    validation_report   JSONB,
    run_id              UUID REFERENCES scenario_runs (run_id) ON DELETE SET NULL,
    experiment_id       UUID REFERENCES experiments (experiment_id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (layer, relative_path)
);

CREATE INDEX IF NOT EXISTS idx_manifest_layer_domain ON data_manifest (layer, domain);
CREATE INDEX IF NOT EXISTS idx_manifest_run          ON data_manifest (run_id);
CREATE INDEX IF NOT EXISTS idx_manifest_experiment   ON data_manifest (experiment_id);
CREATE INDEX IF NOT EXISTS idx_manifest_validation   ON data_manifest (validation_status);

DROP TRIGGER IF EXISTS trg_manifest_updated_at ON data_manifest;
CREATE TRIGGER trg_manifest_updated_at
    BEFORE UPDATE ON data_manifest
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
