"""Tests for etl/load/load_gold.py - split assignment and feature building."""

from __future__ import annotations

from collections import Counter

from etl.config import SPLIT_RATIOS
from etl.extract.extract_carla import _synthetic_raw
from etl.load.load_gold import (
    FUTURE_TICKS,
    PAST_TICKS,
    assign_split,
    build_risk_features,
    build_trajectory_windows,
)
from etl.transform.clean_carla import clean_carla


def test_assign_split_is_deterministic():
    for name in ("demo_overtake", "night_merge", "intersection_01"):
        assert assign_split(name) == assign_split(name)


def test_assign_split_distribution_roughly_matches_ratios():
    names = [f"scenario_{i}" for i in range(3000)]
    counts = Counter(assign_split(n) for n in names)
    total = sum(counts.values())
    for split, ratio in SPLIT_RATIOS.items():
        observed = counts[split] / total
        assert abs(observed - ratio) < 0.05, f"{split}: {observed} vs expected {ratio}"


def _clean_silver_frame(n_ticks: int):
    raw = _synthetic_raw("gold_test", n_ticks=n_ticks)
    clean, _ = clean_carla(raw, "gold_test")
    return clean


def test_build_risk_features_sanity():
    df = _clean_silver_frame(n_ticks=20)
    risk = build_risk_features(df, "gold_test")

    assert len(risk) == df["timestamp"].nunique()
    assert (risk["nearest_vehicle_dist"] >= 0).all()
    assert risk["split"].isin(["train", "val", "test"]).all()
    # the synthetic scenario has the other vehicle closing on the ego
    assert (risk["nearest_closing_speed"] > 0).any()
    # time_to_collision should be finite wherever there's a positive closing speed
    closing = risk[risk["nearest_closing_speed"] > 0]
    assert (closing["time_to_collision"] > 0).all()


def test_build_trajectory_windows_count_matches_formula():
    n_ticks = 60
    df = _clean_silver_frame(n_ticks=n_ticks)
    windows = build_trajectory_windows(df, "gold_test")

    n_vehicles = df["vehicle_id"].nunique()
    expected_per_vehicle = max(n_ticks - PAST_TICKS - FUTURE_TICKS, 0)
    assert len(windows) == n_vehicles * expected_per_vehicle

    row = windows.iloc[0]
    assert len(row["past_x"]) == PAST_TICKS
    assert len(row["future_x"]) == FUTURE_TICKS
