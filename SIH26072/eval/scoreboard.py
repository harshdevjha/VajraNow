"""Evaluate probability forecasts from complete held-out event grids."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from vajranow.config import load_config


def evaluate(rows: list[dict], holdout_events: set[str], config: dict | None = None) -> dict:
    config = config or load_config()
    selected = [row for row in rows if row["event_id"] in holdout_events]
    if not selected:
        raise ValueError("none of the requested held-out events exist in the input")
    if any(row["event_id"] not in holdout_events for row in selected):
        raise ValueError("evaluation selection must use whole held-out events")
    threshold = config["alerts"]["p_on"]
    grid_km = config["region"]["grid_km"]
    radius = math.ceil(config["evaluation"]["neighborhood_km"] / grid_km)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["event_id"], row["valid_time"])].append(row)
    probabilities, baseline, observed, fss_pairs = [], [], [], []
    for key, frame in grouped.items():
        max_y = max(int(row["grid_y"]) for row in frame)
        max_x = max(int(row["grid_x"]) for row in frame)
        shape = (max_y + 1, max_x + 1)
        model_grid = np.full(shape, np.nan)
        baseline_grid = np.full(shape, np.nan)
        observed_grid = np.full(shape, np.nan)
        for row in frame:
            y, x = int(row["grid_y"]), int(row["grid_x"])
            model_grid[y, x] = float(row["probability"])
            baseline_grid[y, x] = float(row["baseline_probability"])
            observed_grid[y, x] = float(row["observed"])
        if np.isnan(model_grid).any() or np.isnan(baseline_grid).any() or np.isnan(observed_grid).any():
            raise ValueError(f"event/time grid {key} is incomplete; include every grid point")
        if np.any((model_grid < 0) | (model_grid > 1)) or np.any((baseline_grid < 0) | (baseline_grid > 1)):
            raise ValueError("forecast probabilities must lie in [0, 1]")
        if np.any((observed_grid != 0) & (observed_grid != 1)):
            raise ValueError("observations must be binary")
        probabilities.extend(model_grid.ravel())
        baseline.extend(baseline_grid.ravel())
        observed.extend(observed_grid.ravel())
        forecast_fraction = uniform_filter((model_grid >= threshold).astype(float), size=2 * radius + 1, mode="constant")
        observed_fraction = uniform_filter(observed_grid, size=2 * radius + 1, mode="constant")
        fss_pairs.append((forecast_fraction, observed_fraction))
    probability = np.asarray(probabilities)
    baseline_probability = np.asarray(baseline)
    truth = np.asarray(observed, dtype=bool)
    predicted = probability >= threshold
    hits = int(np.sum(predicted & truth))
    misses = int(np.sum(~predicted & truth))
    false_alarms = int(np.sum(predicted & ~truth))
    brier = float(np.mean((probability - truth) ** 2))
    baseline_brier = float(np.mean((baseline_probability - truth) ** 2))
    fss_numerator = sum(float(np.sum((forecast - actual) ** 2)) for forecast, actual in fss_pairs)
    fss_denominator = sum(float(np.sum(forecast**2 + actual**2)) for forecast, actual in fss_pairs)
    return {
        "status": "evaluated",
        "split": "whole event IDs explicitly selected as held out",
        "events": sorted(holdout_events),
        "grid_samples": len(probability),
        "threshold": threshold,
        "neighborhood_km": config["evaluation"]["neighborhood_km"],
        "CSI": _ratio(hits, hits + misses + false_alarms),
        "POD": _ratio(hits, hits + misses),
        "FAR": _ratio(false_alarms, hits + false_alarms),
        "bias": _ratio(hits + false_alarms, hits + misses),
        "FSS": None if fss_denominator == 0 else 1 - fss_numerator / fss_denominator,
        "Brier": brier,
        "Brier_skill_vs_baseline": None if baseline_brier == 0 else 1 - brier / baseline_brier,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--holdout-event", action="append", required=True, help="Repeat for each whole held-out event ID")
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"event_id", "valid_time", "grid_y", "grid_x", "probability", "baseline_probability", "observed"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"CSV must contain columns: {', '.join(sorted(required))}")
    try:
        result = evaluate(rows, set(args.holdout_event))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
