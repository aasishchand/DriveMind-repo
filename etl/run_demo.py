"""Run the whole synthetic pipeline end to end.

Bronze -> Silver -> Gold for both domains, using the stub extractors. This is the
plain-Python stand-in for an orchestrator until one is added; it just calls each
``load_*`` step's ``main()`` in order with the right arguments.

    python -m etl.run_demo             # files + local manifest only
    python -m etl.run_demo --push      # also upsert manifest rows into PostgreSQL

With --push you must have the DB reachable and DRIVEMIND_PG_PASSWORD set.
"""

from __future__ import annotations

import argparse

from etl.config import ensure_data_dirs, get_logger
from etl.load import load_bronze, load_gold, load_silver

log = get_logger("etl.run_demo")

CARLA_SCENARIO = "demo_overtake"
IDD_JOBS = (
    ("detection", "train"),
    ("segmentation", "val"),
)


def run(push: bool) -> int:
    ensure_data_dirs()
    extra = ["--push"] if push else []

    log.info("=== CARLA telemetry ===")
    load_bronze.main(["--domain", "carla_telemetry", "--name", CARLA_SCENARIO, *extra])
    load_silver.main(
        [
            "--domain",
            "carla_telemetry",
            "--bronze-file",
            f"data/bronze/carla_telemetry/{CARLA_SCENARIO}.parquet",
            *extra,
        ]
    )
    load_gold.main(["--silver-glob", "data/silver/carla_telemetry/*.parquet", *extra])

    log.info("=== IDD labels ===")
    for variant, split in IDD_JOBS:
        stem = f"idd_{variant}_{split}"
        load_bronze.main(
            ["--domain", "idd", "--variant", variant, "--split", split, *extra]
        )
        load_silver.main(
            ["--domain", "idd", "--bronze-file", f"data/bronze/idd/{stem}.parquet", *extra]
        )

    log.info("done - see data/_manifest/manifest.jsonl%s", " and the DB" if push else "")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push", action="store_true", help="also upsert manifest rows into PostgreSQL"
    )
    args = parser.parse_args(argv)
    return run(args.push)


if __name__ == "__main__":
    raise SystemExit(main())
