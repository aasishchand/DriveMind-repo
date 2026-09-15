"""Database connection seam - the one place the pipeline opens a PostgreSQL
connection. Everything else (load_*.py --push, db/records.py, db/verify_db.py)
goes through here.

Connection settings are read from the environment:

    DRIVEMIND_PG_HOST      (default: localhost)
    DRIVEMIND_PG_PORT      (default: 5432)
    DRIVEMIND_PG_DB        (default: drivemind)
    DRIVEMIND_PG_USER      (default: postgres)
    DRIVEMIND_PG_PASSWORD  (no default; set this to the postgres superuser password)

Set the password once per shell session, e.g. PowerShell:

    $env:DRIVEMIND_PG_PASSWORD = "your-postgres-password"
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg


def conn_kwargs() -> dict[str, str]:
    return {
        "host": os.environ.get("DRIVEMIND_PG_HOST", "localhost"),
        "port": os.environ.get("DRIVEMIND_PG_PORT", "5432"),
        "dbname": os.environ.get("DRIVEMIND_PG_DB", "drivemind"),
        "user": os.environ.get("DRIVEMIND_PG_USER", "postgres"),
        "password": os.environ.get("DRIVEMIND_PG_PASSWORD", ""),
    }


def dsn() -> str:
    """libpq-style DSN string (password redacted) - handy for logging."""
    kw = conn_kwargs()
    kw["password"] = "***" if kw["password"] else ""
    return " ".join(f"{k}={v}" for k, v in kw.items() if v)


def get_connection() -> psycopg.Connection:
    """Open a new connection. Caller is responsible for commit/close.

    Raises a clear error if DRIVEMIND_PG_PASSWORD is unset, which is the most
    common first-run mistake.
    """
    kw = conn_kwargs()
    if not kw["password"]:
        raise RuntimeError(
            "DRIVEMIND_PG_PASSWORD is not set. Set it to the postgres password, e.g.\n"
            '    PowerShell:  $env:DRIVEMIND_PG_PASSWORD = "your-password"\n'
            "    bash:        export DRIVEMIND_PG_PASSWORD=your-password"
        )
    return psycopg.connect(**kw)


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Context manager: commit on clean exit, rollback on exception, always close."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
