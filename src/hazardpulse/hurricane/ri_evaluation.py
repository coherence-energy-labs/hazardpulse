"""Forecast-verification metrics for the hurricane RI models (temporal hold-out, storm bootstrap).

Everything here scores probabilities against 0/1 outcomes. Uncertainty comes from a
bootstrap over STORMS, not rows: the synoptic cases of one storm are strongly correlated,
and a row bootstrap would report intervals several times too narrow.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hazardpulse.hurricane.ri_model import CalibrationError, _logit, newton_logistic


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC as the Mann-Whitney statistic with tie-averaged ranks (ties count 1/2)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    # average ranks over tie groups (1-based)
    boundaries = np.flatnonzero(np.diff(sorted_p)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(p)]))
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of outcomes on the forecast's log-odds: perfect is (0, 1)."""
    X = np.column_stack([_logit(p), np.ones(len(y))])
    theta, _ = newton_logistic(X, np.asarray(y, dtype=float))
    return float(theta[1]), float(theta[0])


RELIABILITY_EDGES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0000001)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def reliability(y: np.ndarray, p: np.ndarray, edges=RELIABILITY_EDGES) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo_ci, hi_ci = wilson(k, n)
        rows.append({
            "bin": [round(lo, 4), round(min(hi, 1.0), 4)], "n": n,
            "mean_forecast": float(p[m].mean()), "observed": k / n,
            "observed_95ci": [lo_ci, hi_ci],
            "forecast_inside_ci": bool(lo_ci <= p[m].mean() <= hi_ci),
        })
    return rows


def point_metrics(y: np.ndarray, p: np.ndarray, climatology: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bs = brier(y, p)
    bs_clim = brier(y, np.full(len(y), climatology))
    try:
        intercept, slope = calibration_intercept_slope(y, p)
    except CalibrationError:
        intercept, slope = float("nan"), float("nan")
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "observed_rate": float(y.mean()),
        "mean_forecast": float(p.mean()),
        "auc": auc(y, p),
        "brier": bs,
        "brier_climatology": bs_clim,
        "bss_vs_climatology": float(1.0 - bs / bs_clim) if bs_clim > 0 else float("nan"),
        "log_loss": log_loss(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "prob_range": {
            "min": float(p.min()), "p01": float(np.quantile(p, 0.01)), "median": float(np.median(p)),
            "p99": float(np.quantile(p, 0.99)), "max": float(p.max()),
        },
    }


def storm_bootstrap(
    groups: np.ndarray,
    stat: Callable[[np.ndarray], dict[str, float]],
    *,
    reps: int = 2000,
    seed: int = 20261002,
) -> dict[str, list[float]]:
    """Percentile 95% intervals of ``stat(row_index)`` under resampling whole storms."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    draws: dict[str, list[float]] = {}
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        for key, value in stat(idx).items():
            draws.setdefault(key, []).append(value)
    out = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[key] = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else [float("nan")] * 2
    return out
