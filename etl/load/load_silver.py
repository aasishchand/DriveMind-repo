"""Validate cleaned/normalised data and persist it to the Silver layer.

Silver is the common-schema contract. A file only lands here if its pandera
schema passes; the manifest row records ``validation_status`` and the schema
name. This module ties the transform + validation + manifest steps together.

Run:
    python -m etl.load.load_silver --domain carla_telemetry \\
        --bronze-file data/bronze/carla_telemetry/demo_scenario.parquet
    python -m etl.load.load_silver --domain idd \\
        --bronze-file data/bronze/idd/idd_detection_train.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from etl.config import CARLA_DOMAIN, IDD_DOMAIN, domain_dir, get_logger
from etl.db.manifest import (
    append_manifest_row,
    build_manifest_row,
    cli_push_manifest_row,
    cli_push_manifest_with_scenario,
    rel_repo_path,
    row_to_sql,
)
from etl.transform.clean_carla import clean_carla
from etl.transform.normalize_idd import normalize_idd

log = get_logger(__name__)


def load_silver(
    bronze_file: Path, *, domain: str, name: str | None = None
) -> tuple[Path, dict]:
    bronze_file = Path(bronze_file)
    raw = pd.read_parquet(bronze_file)
    stem = name or bronze_file.stem

    if domain == CARLA_DOMAIN:
        silver, report = clean_carla(raw, stem)
        schema_name = "carla_telemetry_silver"
    elif domain == IDD_DOMAIN:
        silver, report = normalize_idd(raw)
        variant = str(raw["variant"].iloc[0]) if "variant" in raw.columns else "unknown"
        schema_name = (
            "idd_detection_labels_silver"
            if variant in {"detection", "temporal"}
            else "idd_segmentation_labels_silver"
        )
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(domain)

    out = domain_dir("silver", domain) / f"{stem}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out, engine="pyarrow", index=False)
    log.info("wrote %d validated rows -> %s", len(silver), out)

    row = build_manifest_row(
        out,
        layer="silver",
        domain=domain,
        source_variant=stem,
        schema_name=schema_name,
        validation_status="passed",
    )
    row["validation_report"] = json.dumps(report)
    append_manifest_row(row)
    return out, row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=(CARLA_DOMAIN, IDD_DOMAIN))
    parser.add_argument("--bronze-file", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--push", action="store_true", help="upsert the manifest row into PostgreSQL"
    )
    parser.add_argument(
        "--emit-sql", action="store_true", help="print the manifest INSERT instead of running it"
    )
    args = parser.parse_args(argv)

    out, row = load_silver(args.bronze_file, domain=args.domain, name=args.name)
    print(f"[load_silver] validated + wrote -> {out}")
    if args.emit_sql:
        print(row_to_sql(row))
    if args.push:
        if args.domain == CARLA_DOMAIN:
            manifest_id, run_id = cli_push_manifest_with_scenario(
                row, bronze_path=rel_repo_path(args.bronze_file), link_only=True
            )
            if run_id:
                print(f"[load_silver] linked to scenario_runs: {run_id}")
        else:
            manifest_id = cli_push_manifest_row(row)
        if manifest_id:
            print(f"[load_silver] data_manifest upserted: {manifest_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
