"""Central paths and constants for the DriveMind ETL pipeline.

Everything is derived from the repository root so the scripts work regardless of
the current working directory, as long as they are run as modules from the repo
root (``python -m etl.extract.extract_carla ...``).
"""

from __future__ import annotations

import logging
from pathlib import Path

# ``etl/config.py`` -> repo root is one directory up from the package.
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MANIFEST_DIR = DATA_DIR / "_manifest"

LAYERS = ("bronze", "silver", "gold")

# Per-domain sub-folders, kept identical across every layer so a file's domain is
# always ``data/<layer>/<domain>/...``.
IDD_DOMAIN = "idd"
CARLA_DOMAIN = "carla_telemetry"
DOMAINS = (IDD_DOMAIN, CARLA_DOMAIN)

# Deterministic scenario-level split ratios for the Gold layer.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def layer_dir(layer: str) -> Path:
    """Return the directory for ``layer`` ("bronze" | "silver" | "gold")."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
    return DATA_DIR / layer


def domain_dir(layer: str, domain: str) -> Path:
    """Return ``data/<layer>/<domain>/`` and create it if needed."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    path = layer_dir(layer) / domain
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dirs() -> None:
    """Create the full ``data/`` tree used by the pipeline."""
    for layer in LAYERS:
        for domain in DOMAINS:
            (DATA_DIR / layer / domain).mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format applied once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    return logging.getLogger(name)
