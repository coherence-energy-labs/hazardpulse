"""Candidate C's causal per-cell features (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 5).

The one implementation used by the program's evaluation (scripts/earthquake_program/
features_c.py re-exports it) and by the served model (operational_forecast). Moved here
byte-for-byte from the evaluated script after the CHOOSE decision selected C0; the served
artifact's build re-derives the evaluated forecasts through this module and refuses to
stand if they differ.

Every feature at issue time t reads events with origin time strictly before t (index
ranges from ``searchsorted(..., t, side="left")``) and the A/B rate maps at t, which are
themselves causal. ``tests/test_earthquake_operational_forecast.py`` checks that appending
events at or after t leaves every feature unchanged.

Windows: the longest M4.5+/M2.5+ window is 5 x 365.25 d (counts, depth); older history
enters only through M>=5 (10-year b-value) and M>=6 / M>=7 (time since the last one).
"""

from __future__ import annotations

import numpy as np

from hazardpulse.earthquake import operational_forecast as of


def cell_catalog_from_arrays(t, lat, lon, mag, depth=None) -> "CellCatalog":
    """CellCatalog in the program's canonical order (canonical precision, then lexsort on
    (t, lat, lon, mag)) -- the order ``scripts/earthquake_program/common.load_raw`` gives the
    evaluation, so ties in time resolve identically in serving."""
    t, lat, lon, mag = of._canon(t, lat, lon, mag)
    order = np.lexsort((mag, lon, lat, t))
    d = None if depth is None else np.asarray(depth, np.float64)[order]
    return CellCatalog(t[order], lat[order], lon[order], mag[order], d)

SEC_DAY = 86400.0
YEAR = 365.25 * SEC_DAY
NO_EVENT_DAYS = 1.0e5            # "never" (beyond the 1973 catalog start) for the time-since features

CORE_FEATURES = [
    "log10_lambda_A", "log10_lambda_B", "log10_lambda_short",
    "n45_cell_7d", "n45_cell_30d", "n45_cell_365d", "n45_cell_5y",
    "n45_nbr3_30d", "n45_nbr3_365d",
    "n25_cell_30d", "n25_cell_365d",
    "maxmag_nbr3_30d", "maxmag_nbr3_365d",
    "log10_days_since_m6_nbr3", "mag_last_m6_nbr3",
    "log10_days_since_m7_nbr5", "mag_last_m7_nbr5",
    "mean_depth_cell_5y", "frac_deep70_cell_5y",
    "b_value_nbr3_10y",
    "active",
]


class CellCatalog:
    """Time-sorted event arrays with their cell index (one magnitude band)."""

    def __init__(self, t, lat, lon, mag, depth=None) -> None:
        order = np.argsort(np.asarray(t), kind="stable")
        self.t = np.asarray(t, np.float64)[order]
        self.mag = np.asarray(mag, np.float64)[order]
        self.cell = of.cell_index(np.asarray(lat)[order], np.asarray(lon)[order])
        self.depth = None if depth is None else np.asarray(depth, np.float64)[order]

    def window(self, t: float, seconds: float) -> slice:
        lo = int(np.searchsorted(self.t, t - seconds, side="left"))
        hi = int(np.searchsorted(self.t, t, side="left"))
        return slice(lo, hi)

    def counts(self, t: float, seconds: float) -> np.ndarray:
        sl = self.window(t, seconds)
        return np.bincount(self.cell[sl], minlength=of.N_CELLS).astype(np.float64)


def _grid(x: np.ndarray) -> np.ndarray:
    return x.reshape(of.N_LAT, of.N_LON)


def nbr_reduce(x: np.ndarray, r: int, how: str, fill: float = 0.0) -> np.ndarray:
    """(2r+1) x (2r+1) neighbourhood sum/max on the grid; longitude wraps, latitude does not."""
    g = _grid(x)
    padded = np.full((of.N_LAT + 2 * r, of.N_LON), fill, dtype=np.float64)
    padded[r:r + of.N_LAT] = g
    out = None
    for dla in range(-r, r + 1):
        rows = padded[r + dla:r + dla + of.N_LAT]
        for dlo in range(-r, r + 1):
            sh = np.roll(rows, -dlo, axis=1)
            if out is None:
                out = sh.copy()
            elif how == "sum":
                out += sh
            else:
                out = np.maximum(out, sh)
    return out.reshape(-1)


def _last_event(cat: CellCatalog, t: float, min_mag: float, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Days since the most recent event >= min_mag in the (2r+1)^2 neighbourhood, and its magnitude."""
    n = int(np.searchsorted(cat.t, t, side="left"))
    sel = np.nonzero(cat.mag[:n] >= min_mag)[0]
    last_t = np.full(of.N_CELLS, -np.inf)
    last_m = np.zeros(of.N_CELLS)
    if sel.size:
        cells_rev = cat.cell[sel][::-1]
        uniq, first = np.unique(cells_rev, return_index=True)
        idx = sel[::-1][first]
        last_t[uniq] = cat.t[idx]
        last_m[uniq] = cat.mag[idx]
    g_t = _grid(last_t)
    g_m = _grid(last_m)
    pt = np.full((of.N_LAT + 2 * r, of.N_LON), -np.inf)
    pm = np.zeros((of.N_LAT + 2 * r, of.N_LON))
    pt[r:r + of.N_LAT] = g_t
    pm[r:r + of.N_LAT] = g_m
    best_t = np.full((of.N_LAT, of.N_LON), -np.inf)
    best_m = np.zeros((of.N_LAT, of.N_LON))
    for dla in range(-r, r + 1):
        for dlo in range(-r, r + 1):
            st = np.roll(pt[r + dla:r + dla + of.N_LAT], -dlo, axis=1)
            sm = np.roll(pm[r + dla:r + dla + of.N_LAT], -dlo, axis=1)
            better = st > best_t
            best_t = np.where(better, st, best_t)
            best_m = np.where(better, sm, best_m)
    days = np.where(np.isfinite(best_t), (t - best_t) / SEC_DAY, NO_EVENT_DAYS)
    return np.minimum(days, NO_EVENT_DAYS).reshape(-1), best_m.reshape(-1)


def core_features(t: float, c45: CellCatalog, c25: CellCatalog, lam_a: np.ndarray, lam_b: np.ndarray,
                  lam_short: np.ndarray) -> np.ndarray:
    """(N_CELLS x len(CORE_FEATURES)) float64 at issue time t."""
    f = np.empty((of.N_CELLS, len(CORE_FEATURES)), dtype=np.float64)
    f[:, 0] = np.log10(np.maximum(lam_a, 1e-30))
    f[:, 1] = np.log10(np.maximum(lam_b, 1e-30))
    f[:, 2] = np.log10(lam_short + 1e-9)
    n7 = c45.counts(t, 7 * SEC_DAY)
    n30 = c45.counts(t, 30 * SEC_DAY)
    n365 = c45.counts(t, 365 * SEC_DAY)
    n5y = c45.counts(t, 5 * YEAR)
    f[:, 3] = np.log1p(n7)
    f[:, 4] = np.log1p(n30)
    f[:, 5] = np.log1p(n365)
    f[:, 6] = np.log1p(n5y)
    f[:, 7] = np.log1p(nbr_reduce(n30, 1, "sum"))
    f[:, 8] = np.log1p(nbr_reduce(n365, 1, "sum"))
    m25_30 = c25.counts(t, 30 * SEC_DAY)
    f[:, 9] = np.log1p(m25_30)
    f[:, 10] = np.log1p(c25.counts(t, 365 * SEC_DAY))
    for col, secs in ((11, 30 * SEC_DAY), (12, 365 * SEC_DAY)):
        sl = c45.window(t, secs)
        mx = np.zeros(of.N_CELLS)
        np.maximum.at(mx, c45.cell[sl], c45.mag[sl])
        f[:, col] = nbr_reduce(mx, 1, "max")
    d6, m6 = _last_event(c45, t, 6.0, 1)
    f[:, 13] = np.log10(np.maximum(d6, 1e-3))
    f[:, 14] = m6
    d7, m7 = _last_event(c45, t, 7.0, 2)
    f[:, 15] = np.log10(np.maximum(d7, 1e-3))
    f[:, 16] = m7
    sl = c45.window(t, 5 * YEAR)
    dep = c45.depth[sl]
    ok = np.isfinite(dep)
    cells = c45.cell[sl][ok]
    nd = np.bincount(cells, minlength=of.N_CELLS).astype(np.float64)
    sd = np.bincount(cells, weights=dep[ok], minlength=of.N_CELLS)
    deep = np.bincount(cells, weights=(dep[ok] > 70.0).astype(float), minlength=of.N_CELLS)
    with np.errstate(invalid="ignore", divide="ignore"):
        f[:, 17] = np.where(nd > 0, sd / nd, np.nan)
        f[:, 18] = np.where(nd > 0, deep / nd, np.nan)
    sl = c45.window(t, 10 * YEAR)
    big = c45.mag[sl] >= 5.0
    cells = c45.cell[sl][big]
    nb = nbr_reduce(np.bincount(cells, minlength=of.N_CELLS).astype(float), 1, "sum")
    sx = nbr_reduce(np.bincount(cells, weights=c45.mag[sl][big] - 4.95, minlength=of.N_CELLS), 1, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        f[:, 19] = np.where(nb >= 30, np.log10(np.e) / (sx / nb), np.nan)
    f[:, 20] = (m25_30 >= 5).astype(float)
    return f


def active_cells(t: float, c25: CellCatalog) -> np.ndarray:
    """The live scorer's active set: >= 5 M2.5+ events in the cell in the 30 days before t."""
    return c25.counts(t, 30 * SEC_DAY) >= 5
