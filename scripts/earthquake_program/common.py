"""Shared pieces of the earthquake forecast program (docs/EARTHQUAKE_FORECAST_PROGRAM.md).

Issue times and splits (section 2), targets (section 1), metrics and the month-block
bootstrap (section 4). Every script of the program imports these, so a split boundary or
an estimator can only be defined once.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.core.metrics import roc_auc  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

PROGRAM_CACHE = REPO / ".cache" / "earthquake" / "program"
PRED_DIR = PROGRAM_CACHE / "preds"
RESULTS = REPO / "results" / "earthquake_program"
SEC_DAY = 86400.0

ISSUE_ORIGIN = dt.datetime(2005, 1, 3, tzinfo=dt.timezone.utc)
SPLITS = {
    "fit": (dt.datetime(2005, 1, 3, tzinfo=dt.timezone.utc), dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)),
    "choose": (dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)),
    "dev": (dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)),
    "final": (dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)),
}
CLIP = 1e-12
REL_EDGES = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]
N_BOOT = 2000


def all_issue_times() -> np.ndarray:
    """Every issue time that lies wholly inside some split (epoch seconds)."""
    out = []
    k = 0
    while True:
        t = ISSUE_ORIGIN + dt.timedelta(days=7 * k)
        if t >= SPLITS["final"][1]:
            break
        if split_of(t.timestamp()) is not None:
            out.append(t.timestamp())
        k += 1
    return np.array(out, dtype=np.float64)


def split_of(t: float) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo.timestamp() <= t and t + of.HORIZON_DAYS * SEC_DAY <= hi.timestamp():
            return name
    return None


def split_issue_times(name: str) -> np.ndarray:
    ts = all_issue_times()
    return ts[np.array([split_of(t) == name for t in ts])]


def month_key(t: float) -> int:
    d = dt.datetime.fromtimestamp(float(t), dt.timezone.utc)
    return d.year * 12 + (d.month - 1)


def quarter_key(t: float) -> int:
    d = dt.datetime.fromtimestamp(float(t), dt.timezone.utc)
    return d.year * 4 + (d.month - 1) // 3


def load_events(min_mag: float = 4.5) -> of.EventSet:
    z = np.load(PROGRAM_CACHE / "program_catalog.npz")
    return of.EventSet.from_arrays(z["t"], z["lat"], z["lon"], z["mag"], min_mag=min_mag)


def load_raw(name: str = "program_catalog.npz") -> dict[str, np.ndarray]:
    z = np.load(PROGRAM_CACHE / name)
    t, lat, lon, mag = of._canon(z["t"], z["lat"], z["lon"], z["mag"])
    order = np.lexsort((mag, lon, lat, t))
    return {"t": t[order], "lat": lat[order], "lon": lon[order], "mag": mag[order],
            "depth": z["depth"][order].astype(np.float64)}


def binary_targets(events: of.EventSet, issue: np.ndarray, min_mag: float = of.TARGET_MAG) -> np.ndarray:
    """Dense bool (n_issue x N_CELLS): at least one M>=min_mag epicentre in [t, t+30d)."""
    Y = of.target_matrix(events, issue, min_mag=min_mag)
    return (Y.toarray() > 0)


def save_pred(cand: str, split: str, P: np.ndarray) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{cand}_{split}.npy"
    np.save(path, P.astype(np.float64))
    return path


def load_pred(cand: str, split: str) -> np.ndarray:
    return np.load(PRED_DIR / f"{cand}_{split}.npy")


def write_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return path


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


# ---------------------------------------------------------------------------
# Metrics with month-block bootstrap
# ---------------------------------------------------------------------------

class SplitScorer:
    """Sufficient statistics per block (calendar month) for one split's outcomes.

    Built once per (split, outcome, cell mask); ``stats(P)`` turns a forecast matrix into
    per-block sums from which every resampled metric is a ratio of weighted sums. AUC uses
    the per-positive, per-block count of negatives scored below it (ties 1/2), so a
    resampled AUC is exact for the resampled population, not an approximation.
    """

    def __init__(self, issue: np.ndarray, Y: np.ndarray, *, p0: float, mask: np.ndarray | None = None,
                 block: str = "month", seed: int = 20261002, n_boot: int = N_BOOT) -> None:
        keyf = month_key if block == "month" else quarter_key
        keys = np.array([keyf(t) for t in issue])
        self.blocks, self.block_of_issue = np.unique(keys, return_inverse=True)
        self.nb = self.blocks.size
        self.Y = Y.astype(bool)
        self.mask = np.ones_like(self.Y, dtype=bool) if mask is None else mask.astype(bool)
        self.p0 = float(p0)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.nb, size=(n_boot, self.nb))
        self.W = np.zeros((n_boot, self.nb))
        for b in range(n_boot):
            self.W[b] = np.bincount(idx[b], minlength=self.nb)
        self.W0 = np.ones((1, self.nb))
        bo = self.block_of_issue[:, None] * np.ones((1, Y.shape[1]), dtype=np.int64)
        self.pos_block = bo[self.Y & self.mask]
        self.n_pos_b = np.bincount(self.pos_block, minlength=self.nb).astype(float)
        self.n_b = np.bincount(bo[self.mask], minlength=self.nb).astype(float)
        self.n_neg_b = self.n_b - self.n_pos_b
        self._bo = bo

    def stats(self, P: np.ndarray) -> dict[str, np.ndarray]:
        P = np.clip(np.asarray(P, np.float64), 0.0, 1.0)
        pc = np.clip(P, CLIP, 1 - CLIP)
        Y = self.Y
        m = self.mask
        bo = self._bo
        ll = np.where(Y, np.log(pc), np.log1p(-pc))
        ll0 = np.where(Y, np.log(self.p0), np.log1p(-self.p0))
        br = (P - Y) ** 2
        br0 = (self.p0 - Y) ** 2
        nb = self.nb
        out = {
            "ll": np.bincount(bo[m], weights=ll[m], minlength=nb),
            "ll0": np.bincount(bo[m], weights=ll0[m], minlength=nb),
            "brier": np.bincount(bo[m], weights=br[m], minlength=nb),
            "brier0": np.bincount(bo[m], weights=br0[m], minlength=nb),
            "sum_p": np.bincount(bo[m], weights=P[m], minlength=nb),
        }
        # AUC: for each positive, negatives below it in every block
        pos_scores = P[Y & m]
        C = np.zeros((pos_scores.size, nb))
        for b in range(nb):
            sel = (bo == b) & m & ~Y
            neg = np.sort(P[sel])
            lo = np.searchsorted(neg, pos_scores, side="left")
            hi = np.searchsorted(neg, pos_scores, side="right")
            C[:, b] = lo + 0.5 * (hi - lo)
        out["auc_C"] = C
        return out

    def _metrics(self, s: dict, W: np.ndarray) -> dict[str, np.ndarray]:
        npos = W @ self.n_pos_b
        n = W @ self.n_b
        nneg = W @ self.n_neg_b
        ll = W @ s["ll"]
        ll0 = W @ s["ll0"]
        brier = (W @ s["brier"]) / n
        brier0 = (W @ s["brier0"]) / n
        WC = W @ s["auc_C"].T                                  # (B, n_pos)
        num = (WC * W[:, self.pos_block]).sum(axis=1)
        return {
            "ig_per_target": (ll - ll0) / npos,
            "auc": num / (npos * nneg),
            "brier": brier,
            "bss": 1.0 - brier / brier0,
            "ll_per_cell_time": ll / n,
            "calib_ratio": (W @ s["sum_p"]) / npos,
        }

    def summary(self, s: dict) -> dict:
        point = {k: float(v[0]) for k, v in self._metrics(s, self.W0).items()}
        boot = self._metrics(s, self.W)
        out = {}
        for k, v in point.items():
            lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
            out[k] = {"value": v, "ci95": [float(lo), float(hi)]}
        out["n_cell_times"] = int(self.n_b.sum())
        out["n_positive"] = int(self.n_pos_b.sum())
        out["n_blocks"] = int(self.nb)
        return out

    def paired(self, s_x: dict, s_y: dict) -> dict:
        """Difference X - Y with the same resamples."""
        px = self._metrics(s_x, self.W0)
        py = self._metrics(s_y, self.W0)
        bx = self._metrics(s_x, self.W)
        by = self._metrics(s_y, self.W)
        out = {}
        for k in ("ig_per_target", "auc", "brier", "bss"):
            d = bx[k] - by[k]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            out[k] = {"diff": float(px[k][0] - py[k][0]), "ci95": [float(lo), float(hi)],
                      "frac_boot_le_0": float(np.mean(d <= 0))}
        return out


def reliability(P: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    m = np.ones_like(Y, dtype=bool) if mask is None else mask
    p = P[m]
    y = Y[m]
    rows = []
    for lo, hi in zip(REL_EDGES[:-1], REL_EDGES[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(sel.sum())
        rows.append({"bin": [lo, hi], "n": n, "mean_forecast": float(p[sel].mean()) if n else None,
                     "observed_freq": float(y[sel].mean()) if n else None, "n_positive": int(y[sel].sum())})
    return rows


def poisson_ig_site_reference(P: np.ndarray, counts: np.ndarray) -> float:
    """The site verifier's score: Poisson LL of counts under -ln(1-p) vs a per-cell uniform
    rate equal to the OBSERVED count / n_cells, per earthquake, pooled over issue times."""
    from math import lgamma
    lam = -np.log(np.clip(1.0 - P, 1e-12, 1.0))
    lam = np.maximum(lam, 1e-12)
    total = 0.0
    n_events = 0
    for k in range(P.shape[0]):
        c = counts[k]
        nev = int(c.sum())
        if nev == 0:
            continue
        u = nev / c.size
        lgam = np.array([lgamma(x + 1.0) for x in c[c > 0]])
        ll_m = float(np.sum(c * np.log(lam[k])) - lam[k].sum() - lgam.sum())
        ll_u = float(np.sum(c) * np.log(u) - u * c.size - lgam.sum())
        total += ll_m - ll_u
        n_events += nev
    return total / max(n_events, 1)


def auc_point(P: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.ones_like(Y, dtype=bool) if mask is None else mask
    return float(roc_auc(Y[m].astype(float), P[m]))
