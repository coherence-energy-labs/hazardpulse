#!/usr/bin/env python3
"""Score frozen earthquake replay artifacts against realized events.

This script is designed for prospective benchmarking of the live
`fetch_and_score_earthquake.py` forecasts. It reads replay artifacts,
evaluates only forecasts whose 30-day windows have fully matured, and writes:

- `prospective_summary.json`
- `per_forecast_scores.jsonl`
- per-forecast forecast-grid CSV exports
- per-forecast observed-event CSV exports
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.core.metrics import average_precision, roc_auc  # noqa: E402
from hazardpulse.earthquake.coherence_engine import grid_cell_to_latlon, latlon_to_grid_cell  # noqa: E402
from hazardpulse.earthquake.prospective import (  # noqa: E402
    fetch_usgs_catalog_range,
    format_utc_z,
    parse_utc_datetime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "earthquake_prospective"

# Estimators behind the published numbers (recorded in the summary so a reader can
# tell these apart from the pre-2026-10 per-sample walk, which credited tied cells
# by sort order and inflated the mean AUC 0.6564 -> 0.6972).
AUC_ESTIMATOR = "mann_whitney_ties_half"
PR_AUC_ESTIMATOR = "average_precision_ties_grouped"


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tie-aware ROC-AUC; invariant to row order (see hazardpulse.core.metrics)."""
    return roc_auc(y_true, y_score)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tie-aware average precision; invariant to row order."""
    return average_precision(y_true, y_score)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    return float(np.mean((y_score - y_true) ** 2))


def poisson_log_likelihood(counts: np.ndarray, rates: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    rates = np.asarray(rates, dtype=np.float64)
    total = 0.0
    for observed, lam in zip(counts, rates, strict=False):
        lam = max(float(lam), 1e-12)
        if observed == 0:
            total += -lam
        else:
            total += observed * math.log(lam) - lam - math.lgamma(observed + 1.0)
    return float(total)


def load_replay_artifacts(replay_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    for path in sorted(replay_dir.glob("eq_fcst_*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    artifacts.sort(key=lambda artifact: artifact.get("issued_at", ""))
    return artifacts


def matured_artifacts(artifacts: list[dict], score_as_of: dt.datetime) -> list[dict]:
    matured: list[dict] = []
    for artifact in artifacts:
        issued_at = parse_utc_datetime(artifact["issued_at"])
        horizon_days = int(artifact.get("forecast_horizon_days", 30))
        mature_time = issued_at + dt.timedelta(days=horizon_days)
        if mature_time <= score_as_of:
            # Skip legacy minimal replay artifacts that lack scoreable data
            if "steps" in artifact and "forecast_domain" not in artifact:
                continue
            matured.append(artifact)
    return matured


def forecast_grid(artifact: dict) -> np.ndarray | None:
    """The full row-major probability grid of an artifact that publishes one, else None.

    Operational-model artifacts (docs/EARTHQUAKE_FORECAST_PROGRAM.md) carry a probability
    for every cell; older artifacts list active cells and default the rest.
    """
    grid = artifact.get("probability_grid")
    if grid is None:
        return None
    domain = artifact["forecast_domain"]
    if isinstance(grid, str):             # the scorer's compact form: comma-separated values
        grid = [float(v) for v in grid.split(",")]
    arr = np.asarray(grid, dtype=np.float64)
    if arr.size != int(domain["n_lat"]) * int(domain["n_lon"]):
        raise ValueError(f"{artifact.get('forecast_id')}: probability_grid has {arr.size} cells")
    return arr


def write_forecast_grid_csv(path: Path, artifact: dict) -> None:
    domain = artifact["forecast_domain"]
    n_lat = int(domain["n_lat"])
    n_lon = int(domain["n_lon"])
    default_probability = float(domain.get("default_probability", 0.0))
    active_probs = {
        (int(cell["row"]), int(cell["col"])): float(cell["probability"])
        for cell in artifact.get("active_cells", [])
    }
    grid = forecast_grid(artifact)
    if grid is not None:
        active_probs = {divmod(i, n_lon): float(p) for i, p in enumerate(grid)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "col",
                "lat",
                "lon",
                "probability_30d",
                "expected_count_30d",
            ]
        )
        for row in range(n_lat):
            for col in range(n_lon):
                lat, lon = grid_cell_to_latlon(row, col)
                probability = active_probs.get((row, col), default_probability)
                expected_count = -math.log(max(1.0 - probability, 1e-12))
                writer.writerow(
                    [
                        row,
                        col,
                        round(float(lat), 2),
                        round(float(lon), 2),
                        round(float(probability), 8),
                        round(float(expected_count), 8),
                    ]
                )


def write_observed_events_csv(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "latitude", "longitude", "depth", "mag", "id", "row", "col"])
        for event in events:
            row, col = latlon_to_grid_cell(event["latitude"], event["longitude"])
            writer.writerow(
                [
                    event["time"],
                    round(float(event["latitude"]), 4),
                    round(float(event["longitude"]), 4),
                    round(float(event["depth"]), 2),
                    round(float(event["mag"]), 2),
                    event.get("id", ""),
                    row,
                    col,
                ]
            )


def _accumulate_calibration(calib_acc: dict, y_score: np.ndarray, y_true: np.ndarray) -> None:
    """Pool per-cell (forecast probability -> #positive, #total) into a histogram.

    Rounding to 1e-6 keeps it compact (active cells share the same default
    probability), giving the honest calibration signal: what the model said vs
    what actually happened, over every scored grid cell.
    """
    rscore = np.round(y_score, 6)
    uniq, inv = np.unique(rscore, return_inverse=True)
    tot = np.bincount(inv)
    pos = np.bincount(inv, weights=y_true).astype(np.int64)
    for u, t, p in zip(uniq, tot, pos):
        key = float(u)
        slot = calib_acc.get(key)
        if slot is None:
            calib_acc[key] = [int(t), int(p)]
        else:
            slot[0] += int(t)
            slot[1] += int(p)


def write_calibration_dataset(output_dir: Path, calib_acc: dict, hazard: str = "earthquake",
                              model_version: str | None = None) -> Path:
    keys = sorted(calib_acc.keys())
    total = [int(calib_acc[k][0]) for k in keys]
    pos = [int(calib_acc[k][1]) for k in keys]
    n = int(sum(total))
    payload = {
        "hazard": hazard,
        # the ONE model whose forecasts were pooled; fit_calibration binds the calibrator to it
        "model_version": model_version,
        "n": n,
        "n_groups": len(keys),
        "base_rate": (sum(pos) / n) if n else 0.0,
        "scores": [round(float(k), 6) for k in keys],
        "pos": pos,
        "total": total,
    }
    path = output_dir / "calibration_dataset.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def score_single_forecast(
    artifact: dict,
    observed_events: list[dict],
    output_dir: Path,
    calib_acc: dict | None = None,
) -> dict | None:
    issued_at = parse_utc_datetime(artifact["issued_at"])
    horizon_days = int(artifact.get("forecast_horizon_days", 30))
    window_end = issued_at + dt.timedelta(days=horizon_days)

    domain = artifact.get("forecast_domain")
    if domain is None:
        # Legacy minimal replay artifact — cannot score
        return None
    n_lat = int(domain["n_lat"])
    n_lon = int(domain["n_lon"])
    default_probability = float(domain.get("default_probability", 0.0))
    n_cells = n_lat * n_lon

    cell_probs = {
        (int(cell["row"]), int(cell["col"])): float(cell["probability"])
        for cell in artifact.get("active_cells", [])
    }

    cell_counts = Counter()
    for event in observed_events:
        row, col = latlon_to_grid_cell(event["latitude"], event["longitude"])
        cell_counts[(row, col)] += 1

    y_true = np.zeros(n_cells, dtype=np.float64)
    y_score = np.full(n_cells, default_probability, dtype=np.float64)
    count_vec = np.zeros(n_cells, dtype=np.int64)
    active_mask = np.zeros(n_cells, dtype=bool)

    for row in range(n_lat):
        for col in range(n_lon):
            flat = row * n_lon + col
            y_score[flat] = cell_probs.get((row, col), default_probability)
            active_mask[flat] = (row, col) in cell_probs
            count = cell_counts.get((row, col), 0)
            count_vec[flat] = count
            y_true[flat] = 1.0 if count > 0 else 0.0

    grid = forecast_grid(artifact)
    if grid is not None:
        # The operational model publishes every cell; score what it said, not a default.
        y_score = grid

    if calib_acc is not None and grid is not None:
        _accumulate_calibration(calib_acc, grid, y_true)
    elif calib_acc is not None:
        # Pool the RAW model score (not the deployed/calibrated one) so re-fitting
        # the calibrator never double-calibrates. Before any calibrator exists,
        # raw_probability is absent and equals probability.
        raw_probs = {
            (int(cell["row"]), int(cell["col"])): float(
                cell.get("raw_probability", cell["probability"]))
            for cell in artifact.get("active_cells", [])
        }
        y_score_raw = np.full(n_cells, default_probability, dtype=np.float64)
        for (r_, c_), rp in raw_probs.items():
            y_score_raw[r_ * n_lon + c_] = rp
        _accumulate_calibration(calib_acc, y_score_raw, y_true)

    rate_vec = -np.log(np.clip(1.0 - y_score, 1e-12, 1.0))
    total_events = int(sum(cell_counts.values()))
    uniform_rate = total_events / n_cells if total_events > 0 else 1e-12
    uniform_rates = np.full(n_cells, uniform_rate, dtype=np.float64)

    active_sorted = sorted(
        artifact.get("active_cells", []),
        key=lambda cell: float(cell["probability"]),
        reverse=True,
    )

    top_hits: dict[str, bool] = {}
    for k in (1, 5, 10, 20):
        top_cells = active_sorted[:k]
        top_keys = {(int(cell["row"]), int(cell["col"])) for cell in top_cells}
        top_hits[f"top_{k}_hit"] = any(key in cell_counts for key in top_keys)

    forecast_id = artifact["forecast_id"]
    write_forecast_grid_csv(output_dir / "grid_forecasts" / f"{forecast_id}_forecast.csv", artifact)
    write_observed_events_csv(
        output_dir / "observed_events" / f"{forecast_id}_observed.csv",
        observed_events,
    )

    ll_model = poisson_log_likelihood(count_vec, rate_vec)
    ll_uniform = poisson_log_likelihood(count_vec, uniform_rates)
    info_gain_per_event = (
        (ll_model - ll_uniform) / total_events if total_events > 0 else 0.0
    )

    result = {
        "forecast_id": forecast_id,
        "issued_at": artifact["issued_at"],
        "window_end": format_utc_z(window_end),
        "n_cells": n_cells,
        "n_active_cells": len(active_sorted),
        "n_observed_events": total_events,
        "n_positive_cells": int(np.sum(y_true)),
        "n_negative_cells": int(n_cells - np.sum(y_true)),
        "auc": compute_auc(y_true, y_score),
        "pr_auc": compute_pr_auc(y_true, y_score),
        # The grid AUC mostly measures "active vs inactive" (every inactive cell ties
        # at default_probability). These two say how the model does where it actually
        # forecasts: ranking among its own active cells, and how many target events
        # fall in a cell it scored at all.
        "auc_active_cells": compute_auc(y_true[active_mask], y_score[active_mask]),
        "n_events_in_active_cells": int(count_vec[active_mask].sum()),
        "brier": brier_score(y_true, y_score),
        "poisson_log_likelihood": ll_model,
        "uniform_log_likelihood": ll_uniform,
        "information_gain_per_event": info_gain_per_event,
        **top_hits,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score matured earthquake replay artifacts.",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=DEFAULT_REPLAY_DIR,
        help=f"Replay artifact directory (default: {DEFAULT_REPLAY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--score-as-of",
        default=None,
        help="UTC timestamp for determining maturity. Defaults to now.",
    )
    parser.add_argument(
        "--emit-calibration",
        action="store_true",
        help="Pool per-cell (probability -> outcome) into calibration_dataset.json "
        "for the calibrator fitter (scripts/fit_calibration.py).",
    )
    args = parser.parse_args(argv)

    score_as_of = (
        parse_utc_datetime(args.score_as_of)
        if args.score_as_of
        else dt.datetime.now(dt.timezone.utc)
    )

    calib_acc: dict | None = {} if args.emit_calibration else None

    artifacts = load_replay_artifacts(args.replay_dir)
    matured = matured_artifacts(artifacts, score_as_of)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scored_as_of": format_utc_z(score_as_of),
        "replay_dir": str(args.replay_dir.resolve()),
        "n_replay_artifacts": len(artifacts),
        "n_matured_forecasts": len(matured),
        "forecast_horizon_days": 30,
        "target_magnitude_min": 6.0,
        "status": "ok",
    }

    per_forecast_path = output_dir / "per_forecast_scores.jsonl"
    per_forecast_results: list[dict] = []

    if matured:
        earliest = min(parse_utc_datetime(artifact["issued_at"]) for artifact in matured)
        latest = max(
            parse_utc_datetime(artifact["issued_at"])
            + dt.timedelta(days=int(artifact.get("forecast_horizon_days", 30)))
            for artifact in matured
        )
        observed_catalog = fetch_usgs_catalog_range(
            earliest,
            latest,
            min_magnitude=6.0,
            namespace="earthquake_prospective_score",
            verbose=False,
        )

        # A calibrator maps one model's scores: pool only the forecasts of the model that
        # issued the most recent matured forecast (a model change starts a new pool).
        calib_version = matured[-1].get("model_version")
        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                issued_at = parse_utc_datetime(artifact["issued_at"])
                horizon_days = int(artifact.get("forecast_horizon_days", 30))
                window_end = issued_at + dt.timedelta(days=horizon_days)
                observed_events = [
                    event
                    for event in observed_catalog
                    if issued_at <= parse_utc_datetime(event["time"]) < window_end
                ]
                pool = calib_acc if artifact.get("model_version") == calib_version else None
                result = score_single_forecast(
                    artifact, observed_events, output_dir, calib_acc=pool)
                if result is None:
                    continue
                result["model_version"] = artifact.get("model_version")
                per_forecast_results.append(result)
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="earthquake",
                                                   model_version=calib_version)
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_model_version"] = calib_version
            summary["calibration_n"] = int(sum(slot[0] for slot in calib_acc.values()))

        aucs = [result["auc"] for result in per_forecast_results if math.isfinite(result["auc"])]
        pr_aucs = [result["pr_auc"] for result in per_forecast_results if math.isfinite(result["pr_auc"])]
        active_aucs = [
            result["auc_active_cells"]
            for result in per_forecast_results
            if math.isfinite(result["auc_active_cells"])
        ]
        n_events_total = sum(result["n_observed_events"] for result in per_forecast_results)
        n_events_active = sum(result["n_events_in_active_cells"] for result in per_forecast_results)
        briers = [result["brier"] for result in per_forecast_results]
        info_gains = [result["information_gain_per_event"] for result in per_forecast_results]

        summary.update(
            {
                "observed_catalog_window": {
                    "start": format_utc_z(earliest),
                    "end": format_utc_z(latest),
                    "n_events": len(observed_catalog),
                },
                "total_observed_events": int(
                    sum(result["n_observed_events"] for result in per_forecast_results)
                ),
                "auc_estimator": AUC_ESTIMATOR,
                "pr_auc_estimator": PR_AUC_ESTIMATOR,
                "mean_auc": float(np.mean(aucs)) if aucs else None,
                "median_auc": float(np.median(aucs)) if aucs else None,
                "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
                "mean_auc_active_cells": float(np.mean(active_aucs)) if active_aucs else None,
                "n_forecasts_with_active_cell_auc": len(active_aucs),
                "fraction_events_in_active_cells": (
                    float(n_events_active / n_events_total) if n_events_total else None
                ),
                "mean_brier": float(np.mean(briers)) if briers else None,
                "mean_information_gain_per_event": (
                    float(np.mean(info_gains)) if info_gains else None
                ),
                "event_weighted_information_gain_per_event": (
                    float(
                        (
                            sum(
                                (
                                    result["poisson_log_likelihood"]
                                    - result["uniform_log_likelihood"]
                                )
                                for result in per_forecast_results
                            )
                        )
                        / max(
                            1,
                            sum(result["n_observed_events"] for result in per_forecast_results),
                        )
                    )
                ),
                "top_1_hit_rate": float(
                    np.mean([result["top_1_hit"] for result in per_forecast_results])
                ),
                "top_5_hit_rate": float(
                    np.mean([result["top_5_hit"] for result in per_forecast_results])
                ),
                "top_10_hit_rate": float(
                    np.mean([result["top_10_hit"] for result in per_forecast_results])
                ),
                "top_20_hit_rate": float(
                    np.mean([result["top_20_hit"] for result in per_forecast_results])
                ),
            }
        )
    else:
        summary["status"] = "waiting_for_matured_forecasts"
        summary["message"] = (
            "No replay artifacts have fully matured yet for the configured "
            "30-day forecast horizon."
        )
        per_forecast_path.write_text("", encoding="utf-8")

    summary_path = output_dir / "prospective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Earthquake prospective scoring")
    print(f"  Replay dir:          {args.replay_dir.resolve()}")
    print(f"  Matured forecasts:   {len(matured)} / {len(artifacts)}")
    print(f"  Summary:             {summary_path}")
    if matured:
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Mean AUC:            {summary.get('mean_auc')}")
        print(f"  Mean Brier:          {summary.get('mean_brier')}")
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
