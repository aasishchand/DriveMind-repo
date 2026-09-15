"""Smoke test for the DriveMind PostgreSQL setup.

Checks, in order:

    1. a connection can be opened with the current DRIVEMIND_PG_* environment
    2. the three expected tables exist
    3. a round-trip insert -> read -> delete works on scenario_runs + data_manifest
       (with a FK between them), all inside one transaction that is rolled back

Nothing is left behind. Run:

    python -m etl.db.verify_db
"""

from __future__ import annotations

import sys

from etl.db.connection import dsn, get_connection

EXPECTED_TABLES = ("scenario_runs", "experiments", "data_manifest")


def _check_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = ANY(%s)
            """,
            (list(EXPECTED_TABLES),),
        )
        found = {r[0] for r in cur.fetchall()}
    return [t for t in EXPECTED_TABLES if t not in found]


def _round_trip(conn) -> None:
    """Insert a scenario_run + a manifest row referencing it, read back, then roll back."""
    from etl.db.manifest import upsert_manifest_row
    from etl.db.records import insert_scenario_run

    run_id = insert_scenario_run(
        conn,
        scenario_name="__verify__",
        map_name="TownTest",
        status="completed",
        scenario_config={"verify": True},
        has_emergency_vehicle=False,
    )
    manifest_id = upsert_manifest_row(
        conn,
        {
            "layer": "bronze",
            "domain": "carla_telemetry",
            "relative_path": "data/bronze/carla_telemetry/__verify__.parquet",
            "file_format": "parquet",
            "row_count": 0,
            "validation_status": "skipped",
            "validation_report": {"verify": True},
            "run_id": run_id,
        },
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT scenario_name, status FROM scenario_runs WHERE run_id = %s", (run_id,)
        )
        assert cur.fetchone() == ("__verify__", "completed"), "scenario_runs read-back mismatch"
        cur.execute(
            "SELECT run_id FROM data_manifest WHERE manifest_id = %s", (manifest_id,)
        )
        assert str(cur.fetchone()[0]) == run_id, "data_manifest FK read-back mismatch"

    conn.rollback()  # leave the database exactly as we found it


def main() -> int:
    print(f"[verify_db] connecting: {dsn()}")
    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_db] FAIL - could not connect: {exc}")
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            print(f"[verify_db] server: {cur.fetchone()[0]}")

        missing = _check_tables(conn)
        if missing:
            print(f"[verify_db] FAIL - missing tables: {', '.join(missing)}")
            print("           apply etl/db/schema.sql against this database first")
            return 1
        print(f"[verify_db] tables present: {', '.join(EXPECTED_TABLES)}")

        _round_trip(conn)
        print("[verify_db] round-trip insert/read/rollback OK")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        print(f"[verify_db] FAIL - {type(exc).__name__}: {exc}")
        return 1
    finally:
        conn.close()

    print("[verify_db] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
