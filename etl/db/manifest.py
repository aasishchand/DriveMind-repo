"""Build and persist ``data_manifest`` rows.

Every file written to Bronze / Silver / Gold gets one manifest row recording its
location, size, checksum, row count, and which pandera schema validated it.

Rows are always appended to ``data/_manifest/manifest.jsonl`` (a durable local
log). ``upsert_manifest_row`` additionally writes to the PostgreSQL
``data_manifest`` table, keyed on ``(layer, relative_path)``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from etl.config import MANIFEST_DIR, REPO_ROOT

MANIFEST_JSONL = MANIFEST_DIR / "manifest.jsonl"

MANIFEST_FIELDS = (
    "layer",
    "domain",
    "source_variant",
    "relative_path",
    "file_format",
    "row_count",
    "file_size_bytes",
    "checksum_sha256",
    "schema_name",
    "validation_status",
    "run_id",
    "experiment_id",
    "created_at",
)


def rel_repo_path(path: Path | str) -> str:
    """Path as a forward-slash string relative to the repo root (falls back to
    the given path if it is outside the repo). This is the ``relative_path`` key
    used everywhere - manifest rows, scenario_runs.bronze_path, DB lookups."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parquet_row_count(path: Path) -> int | None:
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:  # noqa: BLE001 - non-parquet or unreadable
        return None


def build_manifest_row(
    path: Path,
    *,
    layer: str,
    domain: str,
    schema_name: str | None = None,
    validation_status: str = "skipped",
    source_variant: str | None = None,
    run_id: str | None = None,
    experiment_id: str | None = None,
) -> dict:
    path = Path(path)
    fmt = path.suffix.lstrip(".").lower()
    row_count = parquet_row_count(path) if fmt in {"parquet", "pq"} else None
    rel = rel_repo_path(path)
    return {
        "layer": layer,
        "domain": domain,
        "source_variant": source_variant,
        "relative_path": rel,
        "file_format": fmt,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "checksum_sha256": sha256_file(path) if path.exists() else None,
        "schema_name": schema_name,
        "validation_status": validation_status,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def append_manifest_row(row: dict) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# Columns eligible for the DB upsert (``created_at`` / ``updated_at`` are handled
# by table defaults + trigger; ``validation_report`` is JSONB).
MANIFEST_DB_COLUMNS = (
    "layer",
    "domain",
    "source_variant",
    "relative_path",
    "file_format",
    "row_count",
    "file_size_bytes",
    "checksum_sha256",
    "schema_name",
    "validation_status",
    "validation_report",
    "run_id",
    "experiment_id",
)


def row_to_sql(row: dict) -> str:
    """Render one manifest row as a literal PostgreSQL INSERT (debugging / dry-run)."""
    cols = [c for c in MANIFEST_FIELDS if c in row]
    vals = []
    for c in cols:
        v = row[c]
        if v is None:
            vals.append("NULL")
        elif isinstance(v, bool):
            vals.append("TRUE" if v else "FALSE")
        elif isinstance(v, (int, float)):
            vals.append(repr(v))
        else:
            vals.append("'" + str(v).replace("'", "''") + "'")
    return (
        "INSERT INTO data_manifest ("
        + ", ".join(cols)
        + ") VALUES ("
        + ", ".join(vals)
        + ") ON CONFLICT (layer, relative_path) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in {"layer", "relative_path"})
        + ";"
    )


def upsert_manifest_row(conn, row: dict) -> str:
    """Insert or update one ``data_manifest`` row; return its ``manifest_id``.

    ``conn`` is an open psycopg connection. The caller owns commit/close
    (``etl.db.connection.connection()`` does this for you).
    """
    from psycopg.types.json import Jsonb

    cols = [c for c in MANIFEST_DB_COLUMNS if row.get(c) is not None]
    values = []
    for c in cols:
        v = row[c]
        if c == "validation_report" and not isinstance(v, Jsonb):
            v = Jsonb(json.loads(v) if isinstance(v, str) else v)
        values.append(v)

    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in {"layer", "relative_path"}
    )
    sql = (
        f"INSERT INTO data_manifest ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (layer, relative_path) DO UPDATE SET {updates} "
        f"RETURNING manifest_id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, values)
        return str(cur.fetchone()[0])


def push_manifest_row(row: dict) -> str:
    """Open a connection, upsert one manifest row, commit, close. Return manifest_id."""
    from etl.db.connection import connection

    with connection() as conn:
        return upsert_manifest_row(conn, row)


def cli_push_manifest_row(row: dict) -> str | None:
    """``push_manifest_row`` for CLIs: on a config/connection error, print a short
    message and return None instead of raising a traceback."""
    try:
        return push_manifest_row(row)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as one line
        print(f"[db] --push skipped: {type(exc).__name__}: {exc}")
        return None


def push_manifest_with_scenario(
    row: dict,
    *,
    scenario_name: str | None = None,
    bronze_path: str | None = None,
    summary: dict | None = None,
    run_id: str | None = None,
    link_only: bool = False,
) -> tuple[str, str | None]:
    """Upsert a manifest row and link it to a ``scenario_runs`` row, in one
    transaction. Returns ``(manifest_id, run_id)``.

    * ``link_only=True``  - do not create/modify a run; just resolve one by
      ``bronze_path`` (or use the given ``run_id``) and stamp it on the manifest
      row. Used by load_silver/load_gold.
    * otherwise            - ensure a run exists for ``bronze_path`` (create it,
      or update the matching one) with ``scenario_name`` + ``summary`` fields.
      Used by load_bronze for the carla domain.
    """
    from etl.db.connection import connection
    from etl.db.records import (
        find_scenario_run_by_bronze_path,
        get_or_create_scenario_run,
    )

    with connection() as conn:
        rid = run_id
        if link_only:
            if rid is None and bronze_path:
                rid = find_scenario_run_by_bronze_path(conn, bronze_path)
        elif scenario_name is not None:
            rid = get_or_create_scenario_run(
                conn,
                scenario_name=scenario_name,
                bronze_path=bronze_path,
                run_id=rid,
                **(summary or {}),
            )
        if rid:
            row = {**row, "run_id": rid}
        manifest_id = upsert_manifest_row(conn, row)
        return manifest_id, rid


def cli_push_manifest_with_scenario(row: dict, **kwargs) -> tuple[str | None, str | None]:
    """CLI wrapper for :func:`push_manifest_with_scenario` - prints one line and
    returns ``(None, None)`` on any DB error instead of raising."""
    try:
        return push_manifest_with_scenario(row, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[db] --push skipped: {type(exc).__name__}: {exc}")
        return None, None
