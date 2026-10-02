#!/usr/bin/env python3
"""Score frozen tornado replay artifacts against observed SPC tornado reports.

Reads replay artifacts from dist/data/replay/to_fcst_*.json, evaluates
only forecasts whose 24-hour windows have fully matured, fetches actual
tornado reports from the SPC storm reports CSV feed, and writes:

- results/tornado_prospective/prospective_summary.json
- results/tornado_prospective/per_forecast_scores.jsonl
- results/tornado_prospective/calibration_dataset.json (with --emit-calibration)
- dist/data/tornado-recovery.json (worker-served recovery subset)

Outcome time convention (measured, 2026-10-01)
----------------------------------------------
SPC daily report files ``YYMMDD_rpts_filtered_torn.csv`` cover one CONVECTIVE
day, 12:00 UTC on YYMMDD to 11:59 UTC on the next calendar day, and their
``Time`` column is HHMM in UTC. Two witnesses: in ``240426`` a row timed
``1716`` reads "touched down at 1216 PM CDT" (12:16 CDT = 17:16 UTC); in
``240425`` the row ``0952,OK,35.33,-97.12`` is the SVRGIS record
``2024-04-26 03:49 tz=3`` (03:49 CST = 09:49 UTC on 26 April). So a row whose
HHMM is before 1200 happened on the NEXT UTC date. Placing it on the file's own
date (what this scorer did until 2026-10-01) moves ~a third of all reports 24 h
early, after which the +-4 h match window cannot see them.

Skill accounting
----------------
The headline is POOLED over every (forecast, storm) pair. Brier skill is
reported against two named references:

- ``causal`` climatology: for each forecast, the tornado base rate of every
  storm whose 24 h window had matured BEFORE that forecast was issued. It uses
  no outcome the forecaster could not have known, so it is the honest reference.
- ``sample`` climatology: the pooled base rate of the scored set itself. It is
  in-sample (the reference is fitted on the outcomes it is scored against) and
  is reported only alongside the causal number.

The per-forecast BSS against each forecast's OWN outcome mean is kept for
diagnostics only; it is undefined (NaN) when a forecast has no observed tornado,
and a mean of it is not a skill score.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.tornado.definitive_model import (  # noqa: E402
    LEGACY_MODEL_VERSION,
    model_version_of_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
# Reserved key in the calibration accumulator: the model version being pooled.
_CALIB_VERSION_KEY = "__model_version__"
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tornado_prospective"

# SPC storm reports CSV — daily (convective-day) archives
SPC_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/{date}_rpts_filtered_torn.csv"
# A convective day YYMMDD runs from this UTC hour on YYMMDD to the same hour next day.
SPC_CONVECTIVE_DAY_START_HOUR_UTC = 12

# Matching criteria
MATCH_RADIUS_KM = 40.0  # spatial proximity threshold
MATCH_WINDOW_HOURS = 4.0  # temporal proximity threshold

# Reliability-table bin edges (probabilities are rare-event; resolve the low end).
RELIABILITY_EDGES = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

SKILL_REFERENCE_NOTE = (
    "bss_vs_causal_climatology: reference = base rate of all storm-forecasts whose 24 h window "
    "matured before the forecast was issued (no outcome leak). bss_vs_sample_climatology: "
    "reference = pooled base rate of the scored set itself (in-sample, shown for comparison)."
)


def parse_utc(text: str) -> dt.datetime:
    """Parse timestamp to naive UTC datetime (handles multiple formats)."""
    text = text.strip()
    # ISO-8601: "2026-04-03T18:41:00Z"
    if text[:4].isdigit() and len(text) > 4 and text[4] == "-":
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    # SPC-style: "20260403_183040 UTC"
    text = text.replace(" UTC", "").replace(" utc", "")
    if "_" in text and len(text) >= 15:
        return dt.datetime.strptime(text[:15], "%Y%m%d_%H%M%S")
    if "_" in text and len(text) >= 13:
        return dt.datetime.strptime(text[:13], "%Y%m%d_%H%M")
    # Fallback
    return dt.datetime.strptime(text[:14], "%Y%m%d%H%M%S")


def format_utc_z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(path: Path) -> str:
    """Repo-relative POSIX path when under the project, else the absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# SPC outcomes
# ---------------------------------------------------------------------------

def convective_day_span(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """UTC [start, end) covered by the SPC daily report file for ``day``."""
    start = dt.datetime(day.year, day.month, day.day, SPC_CONVECTIVE_DAY_START_HOUR_UTC)
    return start, start + dt.timedelta(days=1)


def spc_report_time_utc(convective_day: dt.date, hhmm: str) -> dt.datetime:
    """UTC time of an SPC daily-file row.

    Rows timed before 1200 belong to the second half of the convective day,
    i.e. the next UTC calendar date (see module docstring for the witnesses).
    """
    hhmm = hhmm.strip()
    if len(hhmm) < 4 or not hhmm[:4].isdigit():
        raise ValueError(f"SPC time is not HHMM: {hhmm!r}")
    hour, minute = int(hhmm[:2]), int(hhmm[2:4])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"SPC time out of range: {hhmm!r}")
    t = dt.datetime(convective_day.year, convective_day.month, convective_day.day, hour, minute)
    if hour < SPC_CONVECTIVE_DAY_START_HOUR_UTC:
        t += dt.timedelta(days=1)
    return t


def parse_spc_reports_csv(raw: str, convective_day: dt.date) -> list[dict]:
    """Parse one SPC filtered-tornado daily CSV into UTC-timed reports.

    Returns dicts with keys: time (ISO Z, UTC), lat, lon, mag, location, state.
    SPC CSV format: Time,F_Scale,Location,County,State,Lat,Lon,Comments
    Rows that cannot be timed or located are dropped (a report without a
    usable time cannot be matched honestly).
    """
    reports: list[dict] = []
    reader = csv.reader(io.StringIO(raw))
    header = None
    for row in reader:
        if not row:
            continue
        if header is None:
            header = [col.strip().lower() for col in row]
            continue
        if [col.strip().lower() for col in row] == header:
            continue  # repeated header (multi-section files)
        if len(row) < len(header):
            continue
        rec = dict(zip(header, row))
        try:
            lat = float(rec.get("lat", 0))
            lon = float(rec.get("lon", 0))
            report_time = spc_report_time_utc(convective_day, rec.get("time", ""))
        except (ValueError, TypeError):
            continue
        if abs(lat) < 0.01 and abs(lon) < 0.01:
            continue
        mag_str = rec.get("f_scale", rec.get("mag", "-1")).strip()
        mag_str = mag_str.replace("EF", "").replace("ef", "")
        mag = int(mag_str) if mag_str.lstrip("-").isdigit() else -1
        reports.append({
            "time": format_utc_z(report_time),
            "lat": lat,
            "lon": lon,
            "mag": mag,
            "location": rec.get("location", ""),
            "state": rec.get("state", ""),
        })
    return reports


def fetch_spc_reports_for_date(date: dt.date) -> tuple[list[dict] | None, str | None]:
    """Fetch SPC filtered tornado reports for one convective day.

    Returns ``(reports, None)`` on success -- an empty list is a real "no
    tornadoes" day -- or ``(None, error)`` when the file could not be read.
    The two must never be confused: scoring a failed fetch as "no tornado"
    turns every positive that day into a negative.
    """
    date_str = date.strftime("%y%m%d")
    url = SPC_REPORTS_URL.format(date=date_str)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "HazardPulse/1.0 (research)"}
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  Warning: Could not fetch SPC reports for {date}: {exc}")
        return None, f"{type(exc).__name__}: {exc}"
    return parse_spc_reports_csv(raw, date), None


def fetch_spc_reports_range(
    start: dt.datetime, end: dt.datetime
) -> tuple[list[dict], list[str]]:
    """Fetch every convective day overlapping UTC [start, end].

    The convective day before ``start.date()`` is included because its file
    holds the 00-12 UTC reports of ``start.date()``. Returns the reports and the
    ISO dates of convective days whose file could not be fetched.
    """
    all_reports: list[dict] = []
    failed: list[str] = []
    current = start.date() - dt.timedelta(days=1)
    end_date = end.date()
    while current <= end_date:
        reports, error = fetch_spc_reports_for_date(current)
        if reports is None:
            failed.append(current.isoformat())
        else:
            all_reports.extend(reports)
        current += dt.timedelta(days=1)

    print(
        f"  Fetched {len(all_reports)} tornado reports from SPC "
        f"(convective days {start.date() - dt.timedelta(days=1)} to {end_date}; "
        f"{len(failed)} unavailable)"
    )
    return all_reports, failed


def window_overlaps_failed_days(
    issued_at: dt.datetime, window_end: dt.datetime, failed_days: list[str]
) -> bool:
    """True when any convective day with no outcome file overlaps the window."""
    for iso in failed_days:
        lo, hi = convective_day_span(dt.date.fromisoformat(iso))
        if lo <= window_end and issued_at < hi:
            return True
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC with tied scores counted as 1/2 (order-independent).

    Calibrated probabilities are step functions with many exact ties; an AUC
    that walks ties in sort order depends on that arbitrary order.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    first_rank = np.cumsum(counts) - counts + 1  # 1-based rank of each tie group's first element
    avg_rank = first_rank + (counts - 1) / 2.0
    ranks = avg_rank[inverse]
    return float((np.sum(ranks[y_true == 1]) - pos * (pos + 1) / 2.0) / (pos * neg))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))


def brier_skill_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """BSS against the forecast's OWN outcome mean (diagnostic only).

    Undefined (NaN) when the outcomes have no variance: against a reference
    that already knows "no tornado today" exactly, any nonzero forecast scores
    -inf, not 0.
    """
    bs = brier_score(y_true, y_score)
    clim = float(np.mean(y_true))
    bs_clim = float(np.mean((clim - np.asarray(y_true)) ** 2))
    if bs_clim < 1e-12:
        return float("nan")
    return 1.0 - bs / bs_clim


def reliability_table(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    rows: list[dict] = []
    edges = RELIABILITY_EDGES
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mask = (y_score >= lo) & ((y_score <= hi) if last else (y_score < hi))
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin": f"[{lo:g}, {hi:g}{']' if last else ')'}",
            "n": n,
            "mean_forecast": round(float(y_score[mask].mean()), 6),
            "observed_frequency": round(float(y_true[mask].mean()), 6),
            "n_positive": int(y_true[mask].sum()),
        })
    return rows


def causal_climatology(
    issued: list[dt.datetime],
    window_end: list[dt.datetime],
    n_storms: np.ndarray,
    n_pos: np.ndarray,
) -> np.ndarray:
    """Per-forecast reference rate from outcomes that had matured before issue.

    Element f = (positives / storms) over every forecast g with
    window_end[g] <= issued[f]. NaN when nothing had matured yet.
    """
    n_fc = len(issued)
    if n_fc == 0:
        return np.zeros(0)
    ends = np.array([np.datetime64(t, "s") for t in window_end])
    iss = np.array([np.datetime64(t, "s") for t in issued])
    order = np.argsort(ends, kind="mergesort")
    sorted_ends = ends[order]
    cum_n = np.concatenate([[0.0], np.cumsum(np.asarray(n_storms, dtype=np.float64)[order])])
    cum_p = np.concatenate([[0.0], np.cumsum(np.asarray(n_pos, dtype=np.float64)[order])])
    k = np.searchsorted(sorted_ends, iss, side="right")
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(cum_n[k] > 0, cum_p[k] / np.maximum(cum_n[k], 1.0), np.nan)
    return ref


def pooled_metrics(y_true: np.ndarray, y_score: np.ndarray, reference: np.ndarray) -> dict:
    """Pooled storm-level skill. ``reference`` is the per-row causal climatology."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n = int(y_true.size)
    if n == 0:
        return {"n_storm_forecasts": 0}
    base = float(y_true.mean())
    bs = brier_score(y_true, y_score)
    bs_sample = base * (1.0 - base)
    has_ref = np.isfinite(reference)
    n_ref = int(has_ref.sum())
    bs_model_ref = float(np.mean((y_score[has_ref] - y_true[has_ref]) ** 2)) if n_ref else float("nan")
    bs_causal = float(np.mean((reference[has_ref] - y_true[has_ref]) ** 2)) if n_ref else float("nan")
    rel = reliability_table(y_true, y_score)
    ece = sum(r["n"] / n * abs(r["mean_forecast"] - r["observed_frequency"]) for r in rel)

    def _r(x: float, nd: int = 6):
        return round(float(x), nd) if x is not None and math.isfinite(x) else None

    return {
        "n_storm_forecasts": n,
        "n_positive": int(y_true.sum()),
        "base_rate": _r(base, 8),
        "mean_forecast_probability": _r(float(y_score.mean()), 8),
        "brier": _r(bs, 8),
        "auc": _r(compute_auc(y_true, y_score), 4),
        "ece": _r(ece, 6),
        "brier_sample_climatology": _r(bs_sample, 8),
        "bss_vs_sample_climatology": _r(1.0 - bs / bs_sample, 4) if bs_sample > 0 else None,
        "n_with_causal_reference": n_ref,
        "brier_causal_climatology": _r(bs_causal, 8),
        "bss_vs_causal_climatology": (
            _r(1.0 - bs_model_ref / bs_causal, 4) if n_ref and bs_causal > 0 else None
        ),
        "reliability": rel,
    }


# ---------------------------------------------------------------------------
# Replay artifacts
# ---------------------------------------------------------------------------

def load_replay_artifacts(replay_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    for path in sorted(replay_dir.glob("to_fcst_*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    artifacts.sort(key=lambda a: a.get("issued_at", ""))
    return artifacts


def matured_artifacts(
    artifacts: list[dict], score_as_of: dt.datetime
) -> list[dict]:
    matured: list[dict] = []
    for artifact in artifacts:
        issued_at = parse_utc(artifact["issued_at"])
        horizon_hours = int(artifact.get("forecast_horizon_hours", 24))
        mature_time = issued_at + dt.timedelta(hours=horizon_hours)
        if mature_time <= score_as_of:
            matured.append(artifact)
    return matured


def _accumulate_calibration(calib_acc: dict, scores: np.ndarray, y_true: np.ndarray) -> None:
    """Pool (storm probability -> #positive, #total) into a histogram."""
    rscore = np.round(scores, 6)
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


def write_calibration_dataset(output_dir: Path, calib_acc: dict, hazard: str = "tornado") -> Path:
    keys = sorted(k for k in calib_acc.keys() if k != _CALIB_VERSION_KEY)
    total = [int(calib_acc[k][0]) for k in keys]
    pos = [int(calib_acc[k][1]) for k in keys]
    n = int(sum(total))
    payload = {
        "hazard": hazard,
        "model_version": calib_acc.get(_CALIB_VERSION_KEY),
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


def label_storms(artifact: dict, tornado_reports: list[dict]) -> dict:
    """Label every storm of one forecast against UTC-timed reports.

    A storm is positive when a report lies within MATCH_RADIUS_KM of it and
    within MATCH_WINDOW_HOURS of its valid time, counting only reports inside
    the forecast window [issued_at, issued_at + horizon].
    """
    issued_at = parse_utc(artifact["issued_at"])
    horizon_hours = int(artifact.get("forecast_horizon_hours", 24))
    window_end = issued_at + dt.timedelta(hours=horizon_hours)
    storms = artifact.get("storms") or []

    reports_in_window = []
    for report in tornado_reports:
        rtime = parse_utc(report["time"])
        if issued_at <= rtime <= window_end:
            reports_in_window.append((rtime, report))

    y_true = np.zeros(len(storms), dtype=np.float64)
    y_score = np.zeros(len(storms), dtype=np.float64)
    y_raw = np.zeros(len(storms), dtype=np.float64)
    calibrated = np.zeros(len(storms), dtype=bool)
    for i, storm in enumerate(storms):
        prob = float(storm.get("tornado_probability", 0.0))
        y_score[i] = prob
        y_raw[i] = float(storm.get("raw_probability", prob))
        calibrated[i] = bool(storm.get("calibrated", False))
        storm_lat = float(storm.get("lat", 0))
        storm_lon = float(storm.get("lon", 0))
        storm_time = parse_utc(storm.get("valid_time", artifact["issued_at"]))
        for rtime, report in reports_in_window:
            dist = haversine_km(storm_lat, storm_lon, report["lat"], report["lon"])
            dt_hours = abs((rtime - storm_time).total_seconds()) / 3600.0
            if dist <= MATCH_RADIUS_KM and dt_hours <= MATCH_WINDOW_HOURS:
                y_true[i] = 1.0
                break
    return {
        "issued_at": issued_at,
        "window_end": window_end,
        "y_true": y_true,
        "y_score": y_score,
        "y_raw": y_raw,
        "calibrated": calibrated,
        "n_reports_in_window": len(reports_in_window),
    }


def score_single_forecast(
    artifact: dict,
    tornado_reports: list[dict],
    calib_acc: dict | None = None,
    labels: dict | None = None,
) -> dict:
    """Score a single tornado forecast against observed (UTC-timed) reports."""
    if labels is None:
        labels = label_storms(artifact, tornado_reports)
    window_end = labels["window_end"]
    y_true, y_score = labels["y_true"], labels["y_score"]
    storms = artifact.get("storms") or []
    if not storms:
        return {
            "forecast_id": artifact["forecast_id"],
            "issued_at": artifact["issued_at"],
            "window_end": format_utc_z(window_end),
            "n_storms": 0,
            "n_reports_in_window": labels["n_reports_in_window"],
            "n_matched_storms": 0,
            "auc": float("nan"),
            "brier": float("nan"),
            "brier_skill_score": float("nan"),
            "scoring_tier": artifact.get("scoring_tier", "unknown"),
        }

    if calib_acc is not None:
        # Pool the RAW model score (not the deployed/calibrated one) so re-fitting
        # never double-calibrates. Before any calibrator exists raw == probability.
        # Only storms scored by the model the calibrator is FOR are pooled: a
        # calibrator is one model's curve, and pooling two models' raw scores
        # fits neither.
        version = calib_acc.get(_CALIB_VERSION_KEY)
        if version is None:
            keep = np.ones(len(storms), dtype=bool)
        else:
            keep = np.array([
                str(s.get("model_version", LEGACY_MODEL_VERSION)) == version for s in storms
            ], dtype=bool)
        if keep.any():
            _accumulate_calibration(calib_acc, labels["y_raw"][keep], y_true[keep])

    n_matched = int(np.sum(y_true))
    return {
        "forecast_id": artifact["forecast_id"],
        "issued_at": artifact["issued_at"],
        "window_end": format_utc_z(window_end),
        "n_storms": len(storms),
        "n_reports_in_window": labels["n_reports_in_window"],
        "n_matched_storms": n_matched,
        "match_rate": round(n_matched / len(storms), 4),
        "base_rate": round(n_matched / len(storms), 4),
        "auc": compute_auc(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        # Against this forecast's own outcome mean; NaN when no tornado matched.
        "brier_skill_score": brier_skill_score(y_true, y_score),
        "top_probability": float(np.max(y_score)),
        "scoring_tier": artifact.get("scoring_tier", "unknown"),
    }


def _finite_mean(values: list[float], nd: int = 4):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return round(float(np.mean(vals)), nd) if vals else None


def _finite_median(values: list[float], nd: int = 4):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return round(float(np.median(vals)), nd) if vals else None


def summarize(
    scored: list[tuple[dict, dict, dict]],
    score_as_of: dt.datetime,
) -> dict:
    """Aggregate (artifact, labels, per-forecast row) triples into summary fields.

    Pure: no I/O. Exposed for tests.
    """
    rows = [r for _, _, r in scored]
    issued = [lab["issued_at"] for _, lab, _ in scored]
    ends = [lab["window_end"] for _, lab, _ in scored]
    n_storms = np.array([lab["y_true"].size for _, lab, _ in scored], dtype=np.float64)
    n_pos = np.array([lab["y_true"].sum() for _, lab, _ in scored], dtype=np.float64)
    ref_fc = causal_climatology(issued, ends, n_storms, n_pos)

    def _pool(select) -> dict:
        ys, ps, refs = [], [], []
        for (art, lab, row), ref in zip(scored, ref_fc):
            mask = select(art, lab, row)
            if mask is None:
                continue
            if mask is True:
                mask = np.ones(lab["y_true"].size, dtype=bool)
            if not np.any(mask):
                continue
            ys.append(lab["y_true"][mask])
            ps.append(lab["y_score"][mask])
            refs.append(np.full(int(mask.sum()), ref))
        if not ys:
            return {"n_storm_forecasts": 0}
        return pooled_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(refs))

    pooled = _pool(lambda a, lab, r: True)
    tiers = sorted({r.get("scoring_tier", "unknown") for r in rows})
    pooled_by_tier = {
        t: _pool(lambda a, lab, r, t=t: True if r.get("scoring_tier", "unknown") == t else None)
        for t in tiers
    }
    pooled_by_served_mode = {
        "raw_model_probability": _pool(lambda a, lab, r: ~lab["calibrated"]),
        "calibrated_probability": _pool(lambda a, lab, r: lab["calibrated"]),
    }

    aucs = [r["auc"] for r in rows]
    briers = [r["brier"] for r in rows]
    bss_vals = [r["brier_skill_score"] for r in rows]
    total_storms = sum(r["n_storms"] for r in rows)
    total_matched = sum(r["n_matched_storms"] for r in rows)
    total_reports = sum(r["n_reports_in_window"] for r in rows)

    by_tier: dict[str, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r.get("scoring_tier", "unknown"), []).append(r)
    per_tier_metrics: dict[str, dict] = {}
    for tier, rs in by_tier.items():
        t_storms = sum(r["n_storms"] for r in rs)
        t_matched = sum(r["n_matched_storms"] for r in rs)
        pt = pooled_by_tier.get(tier, {})
        per_tier_metrics[tier] = {
            "n_forecasts": len(rs),
            "n_forecasts_with_valid_auc": sum(1 for r in rs if math.isfinite(r["auc"])),
            "total_storms_scored": t_storms,
            "total_matched_storms": t_matched,
            "match_rate": round(t_matched / max(t_storms, 1), 4),
            "mean_auc": _finite_mean([r["auc"] for r in rs]),
            "median_auc": _finite_median([r["auc"] for r in rs]),
            "mean_brier": _finite_mean([r["brier"] for r in rs]),
            "pooled_auc": pt.get("auc"),
            "pooled_brier": pt.get("brier"),
            "pooled_bss_vs_causal_climatology": pt.get("bss_vs_causal_climatology"),
            "pooled_bss_vs_sample_climatology": pt.get("bss_vs_sample_climatology"),
        }

    # Recovery buckets, anchored on score_as_of and on MATURITY (window_end):
    # a forecast can only be scored 24 h after issue, so an issue-time bucket
    # of "last 24 h" would be empty by construction.
    recency_windows = [
        ("last_24h", dt.timedelta(hours=24)),
        ("last_3d", dt.timedelta(days=3)),
        ("last_7d", dt.timedelta(days=7)),
        ("last_14d", dt.timedelta(days=14)),
    ]
    per_tier_recovery: dict[str, dict[str, dict]] = {}
    for tier, rs in by_tier.items():
        tier_recovery: dict[str, dict] = {}
        for label, delta in recency_windows:
            cutoff = score_as_of - delta
            window_rs = [r for r in rs if parse_utc(r["window_end"]) >= cutoff]
            tier_recovery[label] = {
                "n_forecasts": len(window_rs),
                "mean_auc": _finite_mean([r["auc"] for r in window_rs]),
                "mean_brier": _finite_mean([r["brier"] for r in window_rs]),
            }
        per_tier_recovery[tier] = tier_recovery

    n_defined_bss = sum(1 for v in bss_vals if math.isfinite(v))
    return {
        "n_scored_forecasts": len(rows),
        "total_storms_scored": total_storms,
        "total_matched_storms": total_matched,
        "total_tornado_reports_in_windows": total_reports,
        "overall_match_rate": round(total_matched / max(total_storms, 1), 4),
        "skill_reference": SKILL_REFERENCE_NOTE,
        "pooled": pooled,
        "pooled_by_tier": pooled_by_tier,
        "pooled_by_served_mode": pooled_by_served_mode,
        # Per-forecast means: diagnostics, NOT skill headlines (see module docstring).
        "mean_auc": _finite_mean(aucs),
        "median_auc": _finite_median(aucs),
        "mean_brier": _finite_mean(briers),
        "per_forecast_bss_own_climatology": {
            "mean": _finite_mean(bss_vals),
            "n_defined": n_defined_bss,
            "n_undefined_no_observed_tornado": len(rows) - n_defined_bss,
            "note": "Diagnostic only: each forecast's reference is its own outcome mean (in-sample); "
                    "undefined when no tornado matched. Use pooled.bss_vs_causal_climatology for skill.",
        },
        "n_forecasts_with_valid_auc": sum(1 for v in aucs if math.isfinite(v)),
        "n_forecasts_without_matches": sum(1 for r in rows if r["n_matched_storms"] == 0),
        "by_tier": per_tier_metrics,
        "by_tier_recovery": per_tier_recovery,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score matured tornado replay artifacts against SPC reports.",
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--score-as-of", default=None,
        help="UTC timestamp for determining maturity. Defaults to now.",
    )
    parser.add_argument(
        "--issued-after", default=None,
        help=("Only include forecasts issued AT or AFTER this UTC timestamp. "
              "Used to isolate metrics from a specific deployment / fix. "
              "Example: --issued-after 2026-04-28T13:00:00Z"),
    )
    parser.add_argument(
        "--emit-calibration", action="store_true",
        help="Pool per-storm (probability -> outcome) into calibration_dataset.json "
        "for the calibrator fitter (scripts/fit_calibration.py).",
    )
    args = parser.parse_args(argv)

    score_as_of = (
        parse_utc(args.score_as_of)
        if args.score_as_of
        else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    artifacts = load_replay_artifacts(args.replay_dir)
    matured = matured_artifacts(artifacts, score_as_of)

    issued_after_dt: dt.datetime | None = None
    if args.issued_after:
        issued_after_dt = parse_utc(args.issued_after)
        before = len(matured)
        matured = [
            a for a in matured
            if parse_utc(a["issued_at"]) >= issued_after_dt
        ]
        print(f"  Filter --issued-after {args.issued_after}: "
              f"kept {len(matured)} of {before} matured forecasts")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scored_as_of": format_utc_z(score_as_of),
        "replay_dir": _repo_relative(args.replay_dir),
        "n_replay_artifacts": len(artifacts),
        "n_matured_forecasts": len(matured),
        "forecast_horizon_hours": 24,
        "match_radius_km": MATCH_RADIUS_KM,
        "match_window_hours": MATCH_WINDOW_HOURS,
        "outcome_source": "SPC filtered tornado reports",
        "outcome_time_convention": (
            "SPC daily files are convective days (12 UTC to 12 UTC); HHMM is UTC; "
            "rows before 1200 are placed on the next UTC date"
        ),
        "status": "ok",
    }

    calib_acc: dict | None = None
    if args.emit_calibration:
        # Calibration data is pooled for the model being SERVED now.
        served = REPO_ROOT / "results" / "models" / "tornado_gbt_v1.json"
        calib_acc = {
            _CALIB_VERSION_KEY: (
                model_version_of_payload(served) if served.exists() else LEGACY_MODEL_VERSION
            )
        }

    per_forecast_path = output_dir / "per_forecast_scores.jsonl"

    if matured:
        earliest = min(parse_utc(a["issued_at"]) for a in matured)
        latest = max(
            parse_utc(a["issued_at"])
            + dt.timedelta(hours=int(a.get("forecast_horizon_hours", 24)))
            for a in matured
        )
        print(f"Fetching SPC tornado reports for {earliest.date()} to {latest.date()}...")
        all_reports, failed_days = fetch_spc_reports_range(earliest, latest)

        scored: list[tuple[dict, dict, dict]] = []
        unavailable: list[str] = []
        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                labels = label_storms(artifact, all_reports)
                if window_overlaps_failed_days(labels["issued_at"], labels["window_end"], failed_days):
                    # No outcome file for part of the window: unscorable, never "no tornado".
                    unavailable.append(artifact["forecast_id"])
                    continue
                result = score_single_forecast(artifact, all_reports, calib_acc=calib_acc, labels=labels)
                scored.append((artifact, labels, result))
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="tornado")
            summary["calibration_dataset"] = _repo_relative(calib_path)
            summary["calibration_n"] = int(sum(slot[0] for k, slot in calib_acc.items() if k != _CALIB_VERSION_KEY))
            summary["calibration_model_version"] = calib_acc.get(_CALIB_VERSION_KEY)

        summary.update({
            "observed_window": {
                "start": format_utc_z(earliest),
                "end": format_utc_z(latest),
                "n_tornado_reports": len(all_reports),
                "spc_days_unavailable": failed_days,
            },
            "n_forecasts_outcome_unavailable": len(unavailable),
        })
        summary.update(summarize(scored, score_as_of))
        summary["issued_after_filter"] = args.issued_after if issued_after_dt is not None else None
        if not scored:
            summary["status"] = "outcomes_unavailable"
            summary["message"] = "Matured forecasts exist but no SPC outcome file could be read for their windows."
    else:
        summary["status"] = "waiting_for_matured_forecasts"
        summary["n_scored_forecasts"] = 0
        summary["message"] = (
            "No tornado replay artifacts have fully matured yet."
        )
        per_forecast_path.write_text("", encoding="utf-8")

    summary_path = output_dir / "prospective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Also publish a worker-served subset focused on the recovery curve so
    # /verification/tornado/ can render it without loading the full payload.
    pooled = summary.get("pooled") or {}
    recovery_subset = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "scored_as_of": summary.get("scored_as_of"),
        "n_matured_forecasts": summary.get("n_matured_forecasts", 0),
        "n_scored_forecasts": summary.get("n_scored_forecasts", 0),
        "issued_after_filter": summary.get("issued_after_filter"),
        "pooled_auc": pooled.get("auc"),
        "pooled_bss_vs_causal_climatology": pooled.get("bss_vs_causal_climatology"),
        "by_tier": summary.get("by_tier", {}),
        "by_tier_recovery": summary.get("by_tier_recovery", {}),
        "fix_landed_at": "2026-04-28T13:50:00Z",  # HRRR + mlcape fix commit
    }
    worker_path = (
        Path(__file__).resolve().parents[1] / "dist" / "data" / "tornado-recovery.json"
    )
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(
        json.dumps(recovery_subset, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Tornado prospective scoring")
    print(f"  Replay dir:          {args.replay_dir.resolve()}")
    print(f"  Matured forecasts:   {len(matured)} / {len(artifacts)}")
    print(f"  Scored forecasts:    {summary.get('n_scored_forecasts')}")
    print(f"  Summary:             {summary_path}")
    if matured:
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Pooled AUC:          {pooled.get('auc')}")
        print(f"  Pooled Brier:        {pooled.get('brier')}")
        print(f"  Pooled BSS (causal): {pooled.get('bss_vs_causal_climatology')}")
        print(f"  Pooled BSS (sample): {pooled.get('bss_vs_sample_climatology')}")
        print(f"  Match rate:          {summary.get('overall_match_rate')}")
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
