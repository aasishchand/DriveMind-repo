"""Persist raw extractor output to the Bronze layer, unchanged.

Bronze is append-only and schema-free: whatever the extractor produced is written
as Parquet and recorded in the manifest with ``validation_status = 'skipped'``.

For the carla domain, ``--push`` also upserts a ``scenario_runs`` row (keyed on
the Bronze file path) with a summary of the telemetry, and links the manifest
row to it via ``run_id``. Pass ``--run-id`` to attach to a run the scenario
runner already created instead.

Run:
    python -m etl.load.load_bronze --domain carla_telemetry --name demo_overtake
    python -m etl.load.load_bronze --domain idd --variant detection --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
from etl.extract.extract_carla import extract_carla
from etl.extract.extract_idd import extract_idd

log = get_logger(__name__)

_EMERGENCY_BLUEPRINT_HINTS = ("firetruck", "ambulance", "police")


def summarise_carla_raw(raw: pd.DataFrame) -> dict:
    """Derive scenario_runs summary fields from a raw telemetry frame.

    Tolerant of missing columns - real capture output may not match the stub's
    ``RAW_COLUMNS`` exactly.
    """
    out: dict = {}

    if "sim_time" in raw.columns:
        t = pd.to_numeric(raw["sim_time"], errors="coerce").dropna()
        if not t.empty:
            uniq = np.sort(t.unique())
            out["num_frames"] = int(uniq.size)
            out["duration_sec"] = round(float(uniq[-1] - uniq[0]), 3)
            if uniq.size > 2:
                dt = float(np.median(np.diff(uniq)))
                if dt > 0:
                    out["tick_rate_hz"] = round(1.0 / dt, 2)

    if "actor_id" in raw.columns:
        out["num_vehicles"] = int(raw["actor_id"].astype("string").nunique())

    emergency = False
    if "emergency_light_on" in raw.columns:
        emergency = bool(raw["emergency_light_on"].fillna(False).astype(bool).any())
    if not emergency and "blueprint_id" in raw.columns:
        bp = raw["blueprint_id"].astype("string").str.lower()
        pattern = "|".join(_EMERGENCY_BLUEPRINT_HINTS)
        emergency = bool(bp.str.contains(pattern, regex=True, na=False).any())
    out["has_emergency_vehicle"] = emergency

    return out


def load_bronze(df: pd.DataFrame, *, domain: str, name: str, source_variant: str | None) -> Path:
    out = domain_dir("bronze", domain) / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    log.info("wrote %d rows -> %s", len(df), out)

    row = build_manifest_row(
        out,
        layer="bronze",
        domain=domain,
        source_variant=source_variant,
        validation_status="skipped",
    )
    append_manifest_row(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=(CARLA_DOMAIN, IDD_DOMAIN))
    parser.add_argument("--name", default=None, help="output file stem")
    # carla
    parser.add_argument("--scenario-log", type=Path, default=None)
    # idd
    parser.add_argument("--variant", choices=("detection", "temporal", "segmentation", "lite"))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--idd-root", type=Path, default=None)
    parser.add_argument(
        "--from-polygons",
        action="store_true",
        help="idd detection/temporal: derive boxes from polygon JSON instead of VOC XML",
    )
    parser.add_argument(
        "--push", action="store_true", help="upsert the manifest row into PostgreSQL"
    )
    parser.add_argument(
        "--emit-sql", action="store_true", help="print the manifest INSERT instead of running it"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="carla only: link to this existing scenario_runs.run_id instead of creating one",
    )
    args = parser.parse_args(argv)

    if args.domain == CARLA_DOMAIN:
        name = args.name or (args.scenario_log.stem if args.scenario_log else "demo_scenario")
        df = extract_carla(name, args.scenario_log)
        variant = name
    else:
        if not (args.variant and args.split):
            parser.error("--variant and --split are required for --domain idd")
        name = args.name or f"idd_{args.variant}_{args.split}"
        df = extract_idd(
            args.variant, args.split, args.idd_root, from_polygons=args.from_polygons
        )
        variant = args.variant

    out = load_bronze(df, domain=args.domain, name=name, source_variant=variant)
    row = build_manifest_row(out, layer="bronze", domain=args.domain, source_variant=variant)

    print(f"[load_bronze] {len(df)} rows -> {out}")
    if args.emit_sql:
        print(row_to_sql(row))
    if args.push:
        if args.domain == CARLA_DOMAIN:
            summary = summarise_carla_raw(df)
            summary["status"] = "completed"
            manifest_id, run_id = cli_push_manifest_with_scenario(
                row,
                scenario_name=name,
                bronze_path=rel_repo_path(out),
                summary=summary,
                run_id=args.run_id,
            )
            if run_id:
                print(f"[load_bronze] scenario_runs: {run_id}  ({summary})")
        else:
            manifest_id = cli_push_manifest_row(row)
        if manifest_id:
            print(f"[load_bronze] data_manifest upserted: {manifest_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
