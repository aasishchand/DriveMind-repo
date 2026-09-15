"""Insert / update helpers for the ``scenario_runs`` and ``experiments`` tables.

Thin wrappers over parameterised SQL - no ORM. Every function takes an open
psycopg connection as its first argument; use ``etl.db.connection.connection()``
to get one that commits and closes for you:

    from etl.db.connection import connection
    from etl.db.records import insert_scenario_run, update_scenario_run

    with connection() as conn:
        run_id = insert_scenario_run(conn, scenario_name="demo_overtake",
                                     map_name="Town10HD", status="running")
        ...
        update_scenario_run(conn, run_id, status="completed", num_frames=1200)

JSONB columns (``scenario_config``, ``hyperparams``, ``metrics``) accept plain
dicts.
"""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

_JSONB_COLUMNS = {"scenario_config", "hyperparams", "metrics"}

SCENARIO_RUN_COLUMNS = (
    "scenario_name",
    "scenario_config",
    "carla_version",
    "map_name",
    "seed",
    "weather",
    "started_at",
    "ended_at",
    "duration_sec",
    "tick_rate_hz",
    "num_frames",
    "num_vehicles",
    "has_emergency_vehicle",
    "status",
    "bronze_path",
    "notes",
)

EXPERIMENT_COLUMNS = (
    "name",
    "description",
    "component",
    "git_commit",
    "dataset_version",
    "hyperparams",
    "metrics",
    "artifact_path",
    "started_at",
    "ended_at",
    "status",
    "created_by",
)


def _coerce(col: str, value: Any) -> Any:
    if col in _JSONB_COLUMNS and value is not None and not isinstance(value, Jsonb):
        return Jsonb(value)
    return value


def _insert(conn, table: str, allowed: tuple[str, ...], data: dict, pk: str) -> str:
    cols = [c for c in allowed if c in data and data[c] is not None]
    if not cols:
        raise ValueError(f"no insertable columns supplied for {table}")
    values = [_coerce(c, data[c]) for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) RETURNING {pk}"
    with conn.cursor() as cur:
        cur.execute(sql, values)
        return str(cur.fetchone()[0])


def _update(conn, table: str, allowed: tuple[str, ...], pk: str, pk_value: str, fields: dict) -> None:
    cols = [c for c in allowed if c in fields]
    if not cols:
        return
    assignments = ", ".join(f"{c} = %s" for c in cols)
    values = [_coerce(c, fields[c]) for c in cols]
    values.append(pk_value)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {assignments} WHERE {pk} = %s", values)


# --- scenario_runs ---------------------------------------------------------

def insert_scenario_run(conn, *, scenario_name: str, **fields: Any) -> str:
    """Insert a row into ``scenario_runs``; return the new ``run_id``."""
    return _insert(
        conn,
        "scenario_runs",
        SCENARIO_RUN_COLUMNS,
        {"scenario_name": scenario_name, **fields},
        "run_id",
    )


def update_scenario_run(conn, run_id: str, **fields: Any) -> None:
    """Update selected columns of one ``scenario_runs`` row."""
    _update(conn, "scenario_runs", SCENARIO_RUN_COLUMNS, "run_id", run_id, fields)


def find_scenario_run_by_bronze_path(conn, bronze_path: str) -> str | None:
    """Return the ``run_id`` whose ``bronze_path`` matches, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id FROM scenario_runs WHERE bronze_path = %s LIMIT 1",
            (bronze_path,),
        )
        row = cur.fetchone()
    return str(row[0]) if row else None


def get_or_create_scenario_run(
    conn,
    *,
    scenario_name: str,
    bronze_path: str | None = None,
    run_id: str | None = None,
    **fields: Any,
) -> str:
    """Idempotent upsert of a ``scenario_runs`` row keyed on ``bronze_path``.

    One Bronze telemetry file == one scenario run. If ``run_id`` is given it is
    updated directly; otherwise the row matching ``bronze_path`` is updated, or a
    new row inserted. Returns the ``run_id``.
    """
    if run_id is None and bronze_path is not None:
        run_id = find_scenario_run_by_bronze_path(conn, bronze_path)

    if run_id is not None:
        update_scenario_run(
            conn, run_id, scenario_name=scenario_name, bronze_path=bronze_path, **fields
        )
        return run_id

    return insert_scenario_run(
        conn, scenario_name=scenario_name, bronze_path=bronze_path, **fields
    )


# --- experiments ---------------------------------------------------------

def insert_experiment(conn, *, name: str, **fields: Any) -> str:
    """Insert a row into ``experiments``; return the new ``experiment_id``."""
    return _insert(
        conn,
        "experiments",
        EXPERIMENT_COLUMNS,
        {"name": name, **fields},
        "experiment_id",
    )


def update_experiment(conn, experiment_id: str, **fields: Any) -> None:
    """Update selected columns of one ``experiments`` row."""
    _update(conn, "experiments", EXPERIMENT_COLUMNS, "experiment_id", experiment_id, fields)
