"""Tests for etl/db/manifest.py - the parts that don't need a live database."""

from __future__ import annotations

import pandas as pd

from etl.config import REPO_ROOT
from etl.db.manifest import build_manifest_row, rel_repo_path, row_to_sql, sha256_file


def test_rel_repo_path_uses_forward_slashes(tmp_path):
    nested = REPO_ROOT / "data" / "bronze" / "carla_telemetry" / "unit_test.parquet"
    rel = rel_repo_path(nested)
    assert rel == "data/bronze/carla_telemetry/unit_test.parquet"
    assert "\\" not in rel


def test_build_manifest_row_computes_checksum_and_row_count(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)

    row = build_manifest_row(path, layer="bronze", domain="carla_telemetry", source_variant="unit")

    assert row["layer"] == "bronze"
    assert row["domain"] == "carla_telemetry"
    assert row["file_format"] == "parquet"
    assert row["row_count"] == 3
    assert row["checksum_sha256"] == sha256_file(path)
    assert row["file_size_bytes"] == path.stat().st_size
    assert row["validation_status"] == "skipped"


def test_row_to_sql_escapes_quotes_and_handles_none():
    row = {
        "layer": "bronze",
        "relative_path": "data/bronze/idd/o'brien.parquet",
        "row_count": 5,
        "schema_name": None,
    }
    sql = row_to_sql(row)
    assert "o''brien" in sql  # single quote escaped for SQL
    assert "NULL" in sql
    assert "INSERT INTO data_manifest" in sql
    assert "ON CONFLICT (layer, relative_path) DO UPDATE SET" in sql
