#!/usr/bin/env python3
"""Validate the checked-in earthquake verification artifacts.

This is a lightweight integrity check for the public verification package.
It does not rerun the model; it verifies that the saved JSON artifacts are
present, internally consistent, and aligned closely enough to each other to
support the documented claims.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "earthquake_honest"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _check_unit_interval(name: str, value: float, failures: list[str]) -> None:
    if not (0.0 <= value <= 1.0):
        failures.append(f"{name}={value!r} is outside [0, 1]")


def _check_finite(name: str, value: float, failures: list[str]) -> None:
    if not math.isfinite(value):
        failures.append(f"{name}={value!r} is not finite")


def _check_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
    failures: list[str],
) -> None:
    if abs(observed - expected) > tolerance:
        failures.append(
            f"{name} differs by more than {tolerance:.3f}: "
            f"observed={observed:.6f}, expected={expected:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the saved earthquake honest-mode artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing result JSONs (default: {DEFAULT_RESULTS_DIR})",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    v4_path = results_dir / "v4_regional_honest_results.json"
    same_loc_path = results_dir / "same_location_auc.json"

    failures: list[str] = []

    if not v4_path.exists():
        failures.append(f"Missing file: {v4_path}")
    if not same_loc_path.exists():
        failures.append(f"Missing file: {same_loc_path}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    v4_results = _load_json(v4_path)
    same_loc = _load_json(same_loc_path)

    global_combined = v4_results.get("global_combined", {})
    regional_results = v4_results.get("regional_results", {})
    regional_ensemble = global_combined.get("regional_ensemble", {})
    global_baseline = global_combined.get("global_baseline", {})
    rest_of_world = regional_results.get("rest_of_world", {})
    rest_metrics = rest_of_world.get("metrics", {})

    for label, metrics in (
        ("regional_ensemble", regional_ensemble),
        ("global_baseline", global_baseline),
    ):
        for key in ("auc", "pr_auc", "brier", "bss", "base_rate"):
            value = metrics.get(key)
            if value is None:
                failures.append(f"{label}.{key} is missing")
                continue
            _check_finite(f"{label}.{key}", float(value), failures)
        for key in ("auc", "pr_auc", "brier", "base_rate"):
            value = metrics.get(key)
            if value is not None:
                _check_unit_interval(f"{label}.{key}", float(value), failures)

        n_pos = metrics.get("n_positive")
        n_neg = metrics.get("n_negative")
        base_rate = metrics.get("base_rate")
        if isinstance(n_pos, int) and isinstance(n_neg, int) and n_pos + n_neg > 0:
            derived_rate = n_pos / (n_pos + n_neg)
            if base_rate is None:
                failures.append(f"{label}.base_rate is missing")
            else:
                _check_close(
                    f"{label}.base_rate",
                    float(base_rate),
                    derived_rate,
                    1e-9,
                    failures,
                )
        else:
            failures.append(f"{label} has invalid positive/negative counts")

    if "auc" not in rest_metrics:
        failures.append("regional_results.rest_of_world.metrics.auc is missing")
    else:
        _check_unit_interval("rest_of_world.metrics.auc", float(rest_metrics["auc"]), failures)

    for key in (
        "global_auc",
        "same_location_macro_auc",
        "same_location_weighted_auc",
        "same_location_median_auc",
    ):
        value = same_loc.get(key)
        if value is None:
            failures.append(f"same_location_auc.{key} is missing")
            continue
        _check_finite(f"same_location_auc.{key}", float(value), failures)
        _check_unit_interval(f"same_location_auc.{key}", float(value), failures)

    for key in ("n_locations_evaluated", "n_locations_pos_only", "n_locations_neg_only"):
        value = same_loc.get(key)
        if not isinstance(value, int) or value < 0:
            failures.append(f"same_location_auc.{key} must be a non-negative integer")

    dist = same_loc.get("distribution", {})
    ordered = [dist.get(k) for k in ("p10", "p25", "p50", "p75", "p90")]
    if any(value is None for value in ordered):
        failures.append("same_location_auc.distribution is incomplete")
    else:
        ordered_floats = [float(value) for value in ordered]
        for index, value in enumerate(ordered_floats):
            _check_unit_interval(f"same_location_auc.distribution[{index}]", value, failures)
        if ordered_floats != sorted(ordered_floats):
            failures.append("same_location_auc percentiles are not non-decreasing")

    rest_auc = rest_metrics.get("auc")
    global_auc = same_loc.get("global_auc")
    if rest_auc is not None and global_auc is not None:
        _check_close(
            "same_location global AUC vs rest_of_world AUC",
            float(global_auc),
            float(rest_auc),
            0.05,
            failures,
        )

    print("Earthquake honest-mode artifact verification")
    print(f"  Results dir:        {results_dir}")
    print(f"  Regional ensemble:  AUC {_fmt(float(regional_ensemble['auc']))}")
    print(f"  Global baseline:    AUC {_fmt(float(global_baseline['auc']))}")
    print(f"  Same-location:      macro {_fmt(float(same_loc['same_location_macro_auc']))}")
    print(f"  Same-location:      weighted {_fmt(float(same_loc['same_location_weighted_auc']))}")
    print(f"  Same-location test: global {_fmt(float(same_loc['global_auc']))}")
    print(f"  Locations checked:  {same_loc['n_locations_evaluated']}")

    if failures:
        print()
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print()
    print("[OK] Artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
