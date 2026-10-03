#!/usr/bin/env python3
"""Program step 3: candidate D -- the probability the site publishes today, re-created at
every CHOOSE/DEV/FINAL issue time and scored on the contract.

Exactly the live primary path of ``fetch_and_score_earthquake.score_grid_cells`` (tier 1a)
followed by ``apply_trust_layer``: active cells (>= 5 M2.5+ events in the cell in the 30
days before t) are scored by the deep GRU K192
(``results/models/eq_deep_nowcast_m5.0_2025_K192.serve.npz``) on the M2.5+ events of the
1,827 days before t at the cell centre; the raw score is mapped by the trust-layer
calibrator (``results/calibration/earthquake_calibration.json``); every other cell is 0.0
(the replay artifact's ``default_probability``).

    python scripts/earthquake_program/score_d.py --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

HISTORY_DAYS = 1827
DEEP_NPZ = C.REPO / "results" / "models" / "eq_deep_nowcast_m5.0_2025_K192.serve.npz"
_RAW = None
_SCORER = None


class _Cat:
    pass


def _init():
    global _RAW, _SCORER
    from hazardpulse.earthquake.deep_serve import load_deep_eq_scorer
    _RAW = C.load_raw("program_catalog_m25.npz")
    _SCORER = load_deep_eq_scorer(DEEP_NPZ, calib_path=None)


def _one(k_t):
    k, t = k_t
    r = _RAW
    lo = int(np.searchsorted(r["t"], t - HISTORY_DAYS * C.SEC_DAY, side="left"))
    hi = int(np.searchsorted(r["t"], t, side="left"))
    cat = _Cat()
    cat.times = r["t"][lo:hi]
    cat.lats = r["lat"][lo:hi]
    cat.lons = r["lon"][lo:hi]
    cat.mags = r["mag"][lo:hi]
    d = r["depth"][lo:hi]
    cat.depths = np.where(np.isfinite(d), d, 10.0)
    recent = cat.times >= t - 30 * C.SEC_DAY
    cells = np.nonzero(np.bincount(of.cell_index(cat.lats[recent], cat.lons[recent]), minlength=of.N_CELLS) >= 5)[0]
    clat, clon = of.cell_centres()
    raw = np.full(cells.size, np.nan)
    for i, c in enumerate(cells):
        p = _SCORER.score(cat, float(clat[c]), float(clon[c]), float(t))
        if p is not None:
            raw[i] = p
    return k, cells, raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from hazardpulse.trust.scoring import load_forecaster

    fc = load_forecaster("earthquake")
    if fc is None:
        raise SystemExit("no earthquake trust calibrator (results/calibration/earthquake_calibration.json)")
    issue = C.all_issue_times()
    split = np.array([C.split_of(t) for t in issue])
    todo = [(k, issue[k]) for k in np.nonzero(np.isin(split, ["choose", "dev", "final"]))[0]]
    t0 = time.time()
    results = {}
    with Pool(args.workers, initializer=_init) as pool:
        for i, (k, cells, raw) in enumerate(pool.imap_unordered(_one, todo, chunksize=2)):
            results[k] = (cells, raw)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} issue times, {time.time() - t0:.0f}s", flush=True)
    report = {"model": DEEP_NPZ.name, "calibrator_model_version": fc.model_version, "splits": {}}
    for name in ("choose", "dev", "final"):
        idx = np.nonzero(split == name)[0]
        P = np.zeros((idx.size, of.N_CELLS))
        R = np.full((idx.size, of.N_CELLS), np.nan)
        n_active = n_none = 0
        for i, k in enumerate(idx):
            cells, raw = results[k]
            n_active += cells.size
            ok = np.isfinite(raw)
            n_none += int((~ok).sum())
            if ok.any():
                p, _, _ = fc.calibrator.predict(list(raw[ok]))
                P[i, cells[ok]] = np.asarray(p, np.float64)
                R[i, cells[ok]] = raw[ok]
        C.save_pred("D", name, P)
        np.save(C.PRED_DIR / f"D_raw_{name}.npy", R)
        report["splits"][name] = {"issue_times": int(idx.size), "active_cell_times": int(n_active),
                                  "deep_returned_none": n_none}
        print(f"  D {name}: {idx.size} issue times, {n_active} active cell-times, deep None {n_none}")
    report["seconds"] = round(time.time() - t0, 1)
    C.write_json("score_d.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
