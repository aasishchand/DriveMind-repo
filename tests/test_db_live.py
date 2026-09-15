"""Optional live-database test.

Skipped unless DRIVEMIND_PG_PASSWORD is set (and a real DB is reachable) -
everything else in the suite runs with no Postgres involved at all. Reuses
etl.db.verify_db, which itself only inserts inside a transaction it rolls
back, so this leaves the database exactly as it found it.

Run just this one, with the DB up and the password set:
    pytest tests/test_db_live.py -m db -v
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.db

requires_db = pytest.mark.skipif(
    not os.environ.get("DRIVEMIND_PG_PASSWORD"),
    reason="DRIVEMIND_PG_PASSWORD not set - skipping live-database test",
)


@requires_db
def test_verify_db_round_trip():
    from etl.db.verify_db import main

    assert main() == 0
