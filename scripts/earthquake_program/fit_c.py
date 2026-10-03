#!/usr/bin/env python3
"""Program step 2b: candidates C0 and C1 -- LightGBM on causal per-cell features with A's
and B's rates as inputs (C1 adds Block S for active cells). Fitted on FIT only
(docs/EARTHQUAKE_FORECAST_PROGRAM.md section 5), forecasts written for CHOOSE/DEV/FINAL.

Training rows: every FIT cell-time that is positive or active, plus a seeded 5% sample of
the remaining negatives weighted 20 (inverse inclusion probability, so the weighted rows
estimate the full population and the log-loss fit stays calibrated). Rounds: early
stopping (100) on FIT issue times from 2015 when trained on issue times whose windows end
by 2015-01-01; then refitted on every FIT row with that round count.

    python scripts/earthquake_program/fit_c.py
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import features_c as F  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

SEED = 20261002
NEG_RATE = 0.05
PARAMS = {
    "objective": "binary", "num_leaves": 15, "learning_rate": 0.03, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1,
}
INTERNAL_SPLIT = dt.datetime(2015, 1, 1, tzinfo=dt.timezone.utc).timestamp()


def block_s_lookup():
    z = np.load(C.PROGRAM_CACHE / "block_s_active.npz")
    k, c, f = z["issue_idx"], z["cell"], z["feats"]
    out = {}
    bounds = np.searchsorted(k, np.arange(z["issue_times"].size + 1), side="left")
    for i in range(z["issue_times"].size):
        lo, hi = bounds[i], bounds[i + 1]
        out[i] = (c[lo:hi], f[lo:hi])
    return out, z["issue_times"]


def block_s_matrix(entry, n_cells=of.N_CELLS) -> np.ndarray:
    m = np.full((n_cells, 61), np.nan, dtype=np.float32)
    cells, feats = entry
    if cells.size:
        m[cells] = feats
    return m


def main() -> int:
    import lightgbm as lgb

    t_start = time.time()
    issue = C.all_issue_times()
    split = np.array([C.split_of(t) for t in issue])
    events = C.load_events(4.5)
    Y = C.binary_targets(events, issue)
    r45 = C.load_raw("program_catalog.npz")
    c45 = F.CellCatalog(r45["t"], r45["lat"], r45["lon"], r45["mag"], r45["depth"])
    r25 = C.load_raw("program_catalog_m25.npz")
    c25 = F.CellCatalog(r25["t"], r25["lat"], r25["lon"], r25["mag"])
    m_issue = np.load(C.PROGRAM_CACHE / "maps_issue_times.npy")
    if not np.array_equal(m_issue, issue):
        raise RuntimeError("saved A/B maps were written for different issue times")
    lam_a = np.load(C.PROGRAM_CACHE / "maps_lambda_A.npy")
    lam_s = np.load(C.PROGRAM_CACHE / "maps_lambda_short_B.npy")
    lam_b = np.empty((issue.size, of.N_CELLS))
    for name in ("fit", "choose", "dev", "final"):
        lam_b[split == name] = -np.log1p(-C.load_pred("B", name))
    bs, bs_issue = block_s_lookup()
    if not np.array_equal(bs_issue, issue):
        raise RuntimeError("Block S cache was written for different issue times")

    rng = np.random.default_rng(SEED)
    Xs, Bs, ys, ws, ks = [], [], [], [], []
    fit_idx = np.nonzero(split == "fit")[0]
    for k in fit_idx:
        f = F.core_features(issue[k], c45, c25, lam_a[k].astype(np.float64), lam_b[k], lam_s[k].astype(np.float64))
        y = Y[k]
        act = f[:, F.CORE_FEATURES.index("active")] == 1.0
        keep = y | act
        samp = (~keep) & (rng.random(of.N_CELLS) < NEG_RATE)
        rows = keep | samp
        Xs.append(f[rows].astype(np.float32))
        Bs.append(block_s_matrix(bs[k])[rows])
        ys.append(y[rows])
        ws.append(np.where(samp, 1.0 / NEG_RATE, 1.0)[rows])
        ks.append(np.full(rows.sum(), k))
    X = np.concatenate(Xs)
    Bm = np.concatenate(Bs)
    y = np.concatenate(ys).astype(np.float64)
    w = np.concatenate(ws)
    kk = np.concatenate(ks)
    print(f"FIT rows {X.shape[0]:,} ({int(y.sum())} positive, weighted population {w.sum():,.0f}); "
          f"features {time.time() - t_start:.0f}s", flush=True)
    t_row = issue[kk]
    tr = t_row + of.HORIZON_DAYS * C.SEC_DAY <= INTERNAL_SPLIT
    va = t_row >= INTERNAL_SPLIT
    from hazardpulse.earthquake.definitive_model import BLOCK_S_NAMES

    report = {"rows": int(X.shape[0]), "positives": int(y.sum()), "weighted_population": float(w.sum()),
              "params": PARAMS, "neg_rate": NEG_RATE, "variants": {}}
    boosters = {}
    for variant in ("C0", "C1"):
        Xv = X if variant == "C0" else np.concatenate([X, Bm], axis=1)
        names = list(F.CORE_FEATURES) + ([] if variant == "C0" else [f"blockS_{n}" for n in BLOCK_S_NAMES])
        dtr = lgb.Dataset(Xv[tr], y[tr], weight=w[tr], feature_name=names, free_raw_data=False)
        dva = lgb.Dataset(Xv[va], y[va], weight=w[va], reference=dtr, free_raw_data=False)
        b0 = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        best = int(b0.best_iteration)
        dall = lgb.Dataset(Xv, y, weight=w, feature_name=names, free_raw_data=False)
        booster = lgb.train(PARAMS, dall, num_boost_round=best)
        boosters[variant] = booster
        gain = booster.feature_importance(importance_type="gain")
        top = sorted(zip(names, gain), key=lambda x: -x[1])[:15]
        report["variants"][variant] = {
            "n_features": len(names), "best_iteration": best,
            "internal_val_logloss": float(b0.best_score["valid_0"]["binary_logloss"]),
            "top_gain": [[n, float(g)] for n, g in top],
        }
        model_path = C.PROGRAM_CACHE / f"{variant}_lightgbm.txt"
        booster.save_model(str(model_path))
        print(f"  {variant}: {len(names)} features, {best} rounds, val logloss "
              f"{report['variants'][variant]['internal_val_logloss']:.5f}; top: "
              + ", ".join(f"{n}" for n, _ in top[:6]), flush=True)
        C.write_json("fit_c.json", report)

    for name in ("choose", "dev", "final"):
        idx = np.nonzero(split == name)[0]
        P0 = np.empty((idx.size, of.N_CELLS))
        P1 = np.empty((idx.size, of.N_CELLS))
        for i, k in enumerate(idx):
            f = F.core_features(issue[k], c45, c25, lam_a[k].astype(np.float64), lam_b[k], lam_s[k].astype(np.float64))
            f32 = f.astype(np.float32)
            P0[i] = boosters["C0"].predict(f32)
            P1[i] = boosters["C1"].predict(np.concatenate([f32, block_s_matrix(bs[k])], axis=1))
        C.save_pred("C0", name, P0)
        C.save_pred("C1", name, P1)
        print(f"  wrote C0/C1 forecasts for {name} ({idx.size} issue times)", flush=True)
    report["seconds"] = round(time.time() - t_start, 1)
    C.write_json("fit_c.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
