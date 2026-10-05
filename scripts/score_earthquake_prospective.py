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
from hazardpulse.earthquake import live_record  # noqa: E402
from hazardpulse.earthquake.coherence_engine import grid_cell_to_latlon, latlon_to_grid_cell  # noqa: E402
from hazardpulse.earthquake.prospective import (  # noqa: E402
    AUDIT_MAX_MAGNITUDE,
    CATALOG_MAX_GAP_DAYS,
    fetch_target_catalog,
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
                              model_version: str | None = None, independent: dict | None = None) -> Path:
    """The pooled (score -> outcome) histogram the calibrator is fitted on.

    ``pos`` counts positive CELL-WINDOWS summed over overlapping windows (one earthquake is
    positive in ~100 of them: the retired model's pool held 5,931 positives from 64
    earthquakes), so ``independent`` (``live_record.version_counts`` of the pooled windows:
    ``n_distinct_events``, ``n_independent_windows``) is written beside it; the earthquake
    scorer applies a calibrator only on that independent evidence
    (fetch_and_score_earthquake.calibration_evidence)."""
    keys = sorted(calib_acc.keys())
    total = [int(calib_acc[k][0]) for k in keys]
    pos = [int(calib_acc[k][1]) for k in keys]
    n = int(sum(total))
    independent = independent or {}
    payload = {
        "hazard": hazard,
        # the ONE model whose forecasts were pooled; fit_calibration binds the calibrator to it
        "model_version": model_version,
        "n": n,
        "n_groups": len(keys),
        "base_rate": (sum(pos) / n) if n else 0.0,
        "n_distinct_events": independent.get("n_distinct_events"),
        "n_independent_windows": independent.get("n_independent_windows"),
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
        # events in THIS window; overlapping windows share events, so a version's record counts
        # them by id (observed_event_ids -> live_record.version_counts), never by summing these
        "n_observed_events": total_events,
        "observed_event_ids": sorted({live_record.event_key(event) for event in observed_events}),
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


def live_record_by_version(results: list[dict]) -> dict[str, dict]:
    """The live record of EACH model version on its own. The pooled means above mix every version
    that ever served; a page quoting them under the current model would show the replaced model's
    record (2026-10: 600 matured windows, mean AUC 0.697, information gain -14.6 per event -- all
    from eq_coherence_v1_0 -- beside the newly served C0).

    Windows overlap (a forecast every few hours, each 30 days long), so the counts are the
    independent ones (``live_record.version_counts``): ``n_distinct_events`` (by USGS id) and
    ``n_independent_windows``; ``n_event_windows`` is the per-window sum, which counts one
    earthquake once per window it falls in (613 windows: 6,955 event-windows, 64 earthquakes).
    The means are over overlapping windows and are quoted only through ``live_record.quotable``.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(str(r.get("model_version") or "unknown"), []).append(r)
    out: dict[str, dict] = {}
    for version, rs in groups.items():
        aucs = [r["auc"] for r in rs if math.isfinite(r["auc"])]
        active = [r["auc_active_cells"] for r in rs if math.isfinite(r["auc_active_cells"])]
        counts = live_record.version_counts(rs)
        n_event_windows = counts["n_event_windows"]
        out[version] = {
            "n_matured_forecasts": len(rs),
            "first_issued_at": min(r.get("issued_at", "") for r in rs) or None,
            "last_issued_at": max(r.get("issued_at", "") for r in rs) or None,
            **counts,
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "mean_auc_active_cells": float(np.mean(active)) if active else None,
            "mean_brier": float(np.mean([r["brier"] for r in rs])),
            # per (earthquake, window) pair: the Poisson log-likelihood gain of each window over a
            # uniform map, summed, divided by the event-windows
            "event_weighted_information_gain_per_event": float(
                sum(r["poisson_log_likelihood"] - r["uniform_log_likelihood"] for r in rs) / max(1, n_event_windows)),
            **{f"top_{k}_hit_rate": float(np.mean([bool(r[f"top_{k}_hit"]) for r in rs]))
               for k in (1, 5, 10, 20) if all(f"top_{k}_hit" in r for r in rs)},
        }
    return out


def summarize_results(results: list[dict]) -> dict:
    """Every aggregate of the summary, computed from the per-forecast results alone (so a stored
    ``per_forecast_scores.jsonl`` re-summarises exactly, ``--from-stored``)."""
    aucs = [r["auc"] for r in results if math.isfinite(r["auc"])]
    pr_aucs = [r["pr_auc"] for r in results if math.isfinite(r["pr_auc"])]
    active_aucs = [r["auc_active_cells"] for r in results if math.isfinite(r["auc_active_cells"])]
    n_event_windows = sum(r["n_observed_events"] for r in results)
    n_events_active = sum(r["n_events_in_active_cells"] for r in results)
    info_gains = [r["information_gain_per_event"] for r in results]
    pooled = live_record.version_counts(results)
    return {
        # every version pooled; the per-version record is by_model_version
        "n_distinct_observed_events": pooled["n_distinct_events"],
        "n_independent_windows": pooled["n_independent_windows"],
        "total_event_windows": int(n_event_windows),
        "auc_estimator": AUC_ESTIMATOR,
        "pr_auc_estimator": PR_AUC_ESTIMATOR,
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "median_auc": float(np.median(aucs)) if aucs else None,
        "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
        "mean_auc_active_cells": float(np.mean(active_aucs)) if active_aucs else None,
        "n_forecasts_with_active_cell_auc": len(active_aucs),
        "fraction_events_in_active_cells": (
            float(n_events_active / n_event_windows) if n_event_windows else None
        ),
        "mean_brier": float(np.mean([r["brier"] for r in results])) if results else None,
        "mean_information_gain_per_event": float(np.mean(info_gains)) if info_gains else None,
        "event_weighted_information_gain_per_event": float(
            sum(r["poisson_log_likelihood"] - r["uniform_log_likelihood"] for r in results)
            / max(1, n_event_windows)
        ),
        **{f"top_{k}_hit_rate": float(np.mean([r[f"top_{k}_hit"] for r in results])) if results else None
           for k in (1, 5, 10, 20)},
        "by_model_version": live_record_by_version(results),
    }


def _stored_event_ids(path: Path) -> list[str]:
    """The event keys of one window's observed-events CSV (written by write_observed_events_csv)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return sorted({
            live_record.event_key({"id": row.get("id"), "time": row.get("time"), "latitude": row["latitude"],
                                   "longitude": row["longitude"], "mag": row["mag"]})
            for row in csv.DictReader(handle)
        })


def resummarize_from_stored(output_dir: Path) -> dict:
    """Rebuild the summary's aggregates from what a previous run stored, with no catalog fetch.

    Reads ``per_forecast_scores.jsonl`` and, for results written before ``observed_event_ids``
    existed, each window's ``observed_events/<forecast_id>_observed.csv`` -- the very events
    that window was scored on. A window whose CSV is missing, or whose CSV holds a different
    number of events than its score says, stops the rebuild: nothing is guessed. Rewrites the
    per-forecast file (with the ids) and the summary (aggregates replaced, catalog window,
    calibration fields and timestamps kept, ``resummarized_at`` added).
    """
    per_forecast_path = output_dir / "per_forecast_scores.jsonl"
    summary_path = output_dir / "prospective_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = [json.loads(line) for line in per_forecast_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for r in results:
        if r.get("observed_event_ids") is not None:
            continue
        csv_path = output_dir / "observed_events" / f"{r['forecast_id']}_observed.csv"
        if not csv_path.is_file():
            raise SystemExit(f"{csv_path} is missing: cannot count {r['forecast_id']}'s events")
        ids = _stored_event_ids(csv_path)
        if len(ids) != int(r["n_observed_events"]):
            raise SystemExit(f"{csv_path} holds {len(ids)} distinct events, the score says "
                             f"{r['n_observed_events']}")
        r["observed_event_ids"] = ids
    for stale in ("total_observed_events",):     # the pre-2026-10 summed count, misread as events
        summary.pop(stale, None)
    summary.update(summarize_results(results) if results else {})
    summary["resummarized_at"] = format_utc_z(dt.datetime.now(dt.timezone.utc))
    with open(per_forecast_path, "w", encoding="utf-8", newline="\n") as handle:
        for r in results:
            handle.write(json.dumps(r) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


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
    parser.add_argument(
        "--from-stored",
        action="store_true",
        help="Do not fetch or score: rebuild the summary's aggregates from the stored "
        "per_forecast_scores.jsonl and observed_events/ (resummarize_from_stored).",
    )
    args = parser.parse_args(argv)

    if args.from_stored:
        summary = resummarize_from_stored(args.output_dir.resolve())
        record = summary.get("by_model_version") or {}
        print("Earthquake prospective summary rebuilt from stored scores")
        for version, r in sorted(record.items()):
            print(f"  {version}: {r['n_matured_forecasts']} windows ({r['n_independent_windows']} non-overlapping), "
                  f"{r['n_distinct_events']} distinct M6+ earthquakes ({r['n_event_windows']} event-windows)")
        return 0

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
        # The truth catalog, AUDITED: an M6+ pull cannot be checked for holes (one M6+ in
        # 2018-06), so the window is pulled at M4.5+, audited (an empty month or a multi-day gap
        # raises CatalogIncompleteError -- a missed M6+ would otherwise score as a correct "no"),
        # then filtered to M6.0+.
        observed_catalog = fetch_target_catalog(
            earliest,
            latest,
            target_min_magnitude=6.0,
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
            pooled = [r for r in per_forecast_results if r.get("model_version") == calib_version]
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="earthquake",
                                                   model_version=calib_version,
                                                   independent=live_record.version_counts(pooled))
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_model_version"] = calib_version
            summary["calibration_n"] = int(sum(slot[0] for slot in calib_acc.values()))

        summary.update(
            {
                "observed_catalog_window": {
                    "start": format_utc_z(earliest),
                    "end": format_utc_z(latest),
                    "n_events": len(observed_catalog),
                    "completeness_audit": (
                        f"pulled at M{AUDIT_MAX_MAGNITUDE:.1f}+ and audited (no empty month, no gap over "
                        f"{CATALOG_MAX_GAP_DAYS:g} d), then filtered to M6.0+"),
                },
                **summarize_results(per_forecast_results),
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
