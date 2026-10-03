#!/usr/bin/env python3
"""Audit what the HRRR tornado benchmark actually measures.

Two instruments, both offline (cached HRRR + SPC CSV, no network):

``--data D.npz [...]``  DECOMPOSE a built dataset. Trains the benchmark's xgboost on
    its temporal-split training set, then scores the test set whole and on subsets
    that remove the easy part: negatives restricted to storm cells (refc >= 35 / 40),
    both classes restricted to storm cells, positives held to the negatives' CAPE
    floor, and (hard datasets) near-miss negatives only. Also reports reflectivity-alone
    and CAPE-alone AUC and the share of positives with no echo at the analysis time.
    If most of the AUC lives in the easy part, the benchmark measures storm-vs-no-storm.

``--ablation``  FACTORIAL: rebuilds the benchmark under each labeling choice, one
    factor at a time, on two grid sources, and scores xgboost + the STP / reflectivity
    baselines on each with day-cluster bootstrap CIs.

    g1 = the pre-2026-10-01 cache files (native INDEX-space pooled; cell (i, j) is the
         31 x 28 native block, NOT the lat/lon cell (i, j)). Read only here and by the
         legacy reproduction.
    g3 = hazardpulse.data.hrrr's current grids (native points binned into the lat/lon
         cell that contains them). The benchmark.

      V1  g1 legacy (historical)   lat/lon box on g1 grids (mislocated), CST-day labels,
                                   easy negatives, raw features -- the published number
      V1s g1 legacy + sanitized    V1 with fill values -> NaN
      V2  g1 + true block          g1 read with its own (index-block) locator
      V3  g1 + time window         reports within [-2 h, +2 h), carried back
      V4  g1 hard                  + storm-population gate on both classes, all eligible
                                   non-tornadic storms, guard band (old-geometry baseline)
      E1  g3 legacy rule           the historical easy-negative recipe on correct grids
      E2  g3 easy + time window
      H   g3 HARD (default)        THE benchmark
      Hx  g3 hard, wrong locator   g3 grids read with the g1 block locator (control)
      Hn  g3 hard, no carry-back
      S1-S5 g3 hard sensitivity to the declared constants (window, refc, shear)

    python scripts/audit_tornado_hrrr_benchmark.py --data .cache/tornado/hrrr_env_2024.npz
    python scripts/audit_tornado_hrrr_benchmark.py --ablation --start 20220401 --end 20240831
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HAZARDPULSE_GPU", "0")       # deterministic CPU fits for an audit
os.environ.setdefault("HAZARDPULSE_HRRR_POOL", "max")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train = _load("train_tornado_hrrr_audit", "train_tornado_hrrr.py")
_build = _load("build_tornado_hrrr_dataset_audit", "build_tornado_hrrr_dataset.py")

from hazardpulse.data import hrrr as _hrrr  # noqa: E402
from hazardpulse.data.hrrr import HRRR_N_LAT, HRRR_N_LON, load_cached_hrrr  # noqa: E402
from hazardpulse.tornado import hrrr_env  # noqa: E402
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr  # noqa: E402
from hazardpulse.tornado.definitive_model import load_spc_tornado_reports  # noqa: E402

NAMES = hrrr_env.FEATURE_NAMES
I_REFC, I_MLCAPE, I_STP = NAMES.index("refc"), NAMES.index("mlcape"), NAMES.index("stp_eff")


def _fit_score(X, y, dates):
    Xtr, ytr, Xte, yte, cut = _train._temporal_split(X, y, dates)
    dte = dates[dates >= cut]
    raw_te = Xte.copy()
    Xtr, Xte = _train._impute(Xtr, Xte)
    clf = _train._tbt._fit_xgboost(Xtr, ytr, 0)
    p = np.asarray(clf.predict_proba(Xte))[:, 1]
    return {"p": p, "y": yte, "d": dte, "Xte": Xte, "raw_te": raw_te, "cut": cut,
            "n_train_pos": int(ytr.sum()), "n_train_neg": int(len(ytr) - ytr.sum())}


def _ci(y, scores, d):
    return _train.day_bootstrap(y, scores, d, reps=1000)


def decompose(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    X, y, dates = z["X"].astype(float), z["y"].astype(int), z["dates"]
    r = _fit_score(X, y, dates)
    p, yt, d, Xte, raw = r["p"], r["y"], r["d"], r["Xte"], r["raw_te"]
    refc, cape = raw[:, I_REFC], raw[:, I_MLCAPE]
    out = {"data": str(path), "split_cut": r["cut"], "n_test_pos": int(yt.sum()),
           "n_test_neg": int(len(yt) - yt.sum())}
    full = _ci(yt, {"xgboost": p, "refc_alone": Xte[:, I_REFC], "mlcape_alone": Xte[:, I_MLCAPE],
                    "stp_alone": Xte[:, I_STP]}, d)
    out["full"] = {k: v for k, v in full.items() if not k.startswith("_")}
    pos, neg = yt == 1, yt == 0
    with np.errstate(invalid="ignore"):
        out["pos_no_echo_frac"] = float(np.mean(refc[pos] <= -9.99))
        out["pos_refc_ge40_frac"] = float(np.mean(refc[pos] >= 40))
        out["neg_refc_ge40_frac"] = float(np.mean(refc[neg] >= 40))
        out["pos_mlcape_lt250_frac"] = float(np.mean(cape[pos] < 250))
        out["neg_mlcape_lt250_frac"] = float(np.mean(cape[neg] < 250))
        subsets = {
            "neg_refc>=35": pos | (refc >= 35),
            "neg_refc>=40": pos | (refc >= 40),
            "both_refc>=40": refc >= 40,
            "pos_mlcape>=250": neg | (cape >= 250),
            "both_refc>=40_and_pos_mlcape>=250": (refc >= 40) & (neg | (cape >= 250)),
        }
    if "cells" in z.files and int(np.asarray(z["cells"]).max()) >= 0:
        cells_te = np.asarray(z["cells"])[dates >= r["cut"]]
        near = np.zeros(len(yt), bool)
        for day in np.unique(d):
            on = d == day
            pc = cells_te[on & pos]
            if len(pc) == 0:
                continue
            idx = np.flatnonzero(on & neg)
            dist = np.abs(cells_te[idx][:, None, :] - pc[None, :, :]).max(axis=2).min(axis=1)
            near[idx[dist <= NEAR_MISS_RADIUS_BLOCKS]] = True
        subsets[f"near_miss_neg_within_{NEAR_MISS_RADIUS_BLOCKS}_blocks"] = pos | near
    out["subsets"] = {}
    for name, m in subsets.items():
        if yt[m].min() == yt[m].max():
            continue
        b = _ci(yt[m], {"xgboost": p[m], "stp": Xte[m, I_STP]}, d[m])
        out["subsets"][name] = {"n_pos": int(yt[m].sum()), "n_neg": int((yt[m] == 0).sum()),
                                "auc": b["xgboost"]["auc"], "ci95": b["xgboost"]["ci95"],
                                "stp_auc": b["stp"]["auc"], "stp_ci95": b["stp"]["ci95"]}
    return out


def g1_block(lat: float, lon: float):
    """Locator for a g1 (pre-2026-10-01, native-INDEX-pooled) grid: the 31 x 28 native
    block holding the point. Historical instrument -- wrong for g3 grids."""
    row, col = _hrrr.native_index_of_latlon(float(lat), float(lon))
    r, c = int(round(float(row))), int(round(float(col)))
    by, bx = _hrrr.NATIVE_NY // HRRR_N_LAT, _hrrr.NATIVE_NX // HRRR_N_LON
    if not (0 <= r < by * HRRR_N_LAT and 0 <= c < bx * HRRR_N_LON):
        return None
    return r // by, c // bx


LOCATORS = {"box": hrrr_env.locate_cell, "g1_block": g1_block}


def _label_day_easy(grids_raw, reports_utc, reports_cst, date, hour, *,
                    locate, labels, sanitize, rng):
    """Easy-negative rows (legacy negative rule) under a chosen locator/labeling."""
    grids = hrrr_env.sanitize_grids(grids_raw) if sanitize else grids_raw
    derived = compute_derived_hrrr(grids)
    pos = set()
    if labels == "cst_day":
        for rep in reports_cst.get(date, []):
            b, _ = _build.report_blocks(rep, 0.0, grids, carry_back=False, locate=locate)
            if b is not None:
                pos.add(b)
    else:
        lo, hi = _build.LABEL_WINDOW_H
        timed, _ = _build.reports_near_analysis(reports_utc, date, hour, max(abs(lo), abs(hi)))
        for rep, off in timed:
            if lo <= off < hi:
                _, b = _build.report_blocks(rep, off, grids, locate=locate)
                if b is not None:
                    pos.add(b)
    cape = grids["mlcape"]
    rows, labs = [], []
    for (i, j) in pos:
        rows.append(hrrr_env.cell_features(grids, derived, i, j))
        labs.append(1)
    cand = [(i, j) for i in range(HRRR_N_LAT) for j in range(HRRR_N_LON)
            if (i, j) not in pos and np.isfinite(cape[i, j])
            and cape[i, j] >= _build.LEGACY_NEG_MIN_CAPE]
    if cand:
        for idx in rng.choice(len(cand), min(_build.LEGACY_NEG_PER_DAY, len(cand)), replace=False):
            i, j = cand[idx]
            rows.append(hrrr_env.cell_features(grids, derived, i, j))
            labs.append(0)
    return rows, labs


VARIANTS = {
    "V1 g1 legacy (historical)": dict(src="g1", kind="legacy"),
    "V1s g1 legacy + sanitized": dict(src="g1", kind="easy", locate="box", labels="cst_day", sanitize=True),
    "V2 g1 + true block": dict(src="g1", kind="easy", locate="g1_block", labels="cst_day", sanitize=True),
    "V3 g1 + time window": dict(src="g1", kind="easy", locate="g1_block", labels="window", sanitize=True),
    "V4 g1 hard (old baseline)": dict(src="g1", kind="hard", locate="g1_block"),
    "E1 g3 legacy rule": dict(src="g3", kind="easy", locate="box", labels="cst_day", sanitize=True),
    "E2 g3 easy + window": dict(src="g3", kind="easy", locate="box", labels="window", sanitize=True),
    "H  g3 HARD (default)": dict(src="g3", kind="hard"),
    "Hx g3 hard, g1 locator": dict(src="g3", kind="hard", locate="g1_block"),
    "Hn g3 hard, no carry": dict(src="g3", kind="hard", carry_back=False),
    # sensitivity of H to its declared constants (robustness, not selection)
    "S1 g3 hard, win [-2,+3)": dict(src="g3", kind="hard", label_window_h=(-2.0, 3.0)),
    "S2 g3 hard, refc>=35": dict(src="g3", kind="hard", population_kw={"refc_min": 35.0}),
    "S3 g3 hard, refc>=45": dict(src="g3", kind="hard", population_kw={"refc_min": 45.0}),
    "S4 g3 hard, shear>=10": dict(src="g3", kind="hard", population_kw={"shear06_min": 10.0}),
    "S5 g3 hard, shear>=18": dict(src="g3", kind="hard", population_kw={"shear06_min": 18.0}),
}

# Near-miss stress subset: negatives within this many cells (~80 km each) of a
# same-day positive -- storms in the same mesoscale regime that did not produce one.
NEAR_MISS_RADIUS_BLOCKS = 6


def ablation(start: str, end: str, hour: int, sources=("g1", "g3")) -> dict:
    reports_utc = load_spc_tornado_reports(_build._SPC_CSV)
    reports_cst = _build.reports_by_local_date(reports_utc)
    days = list(_build._date_range(start, end))
    grids = {"g1": {d: g for d in days if (g := _build.load_old_geometry_cache(d, hour)) is not None},
             "g3": {d: g for d in days if (g := load_cached_hrrr(d, hour)) is not None}}
    common = sorted(set(grids["g1"]) & set(grids["g3"]))
    print(f"ablation: g1 days {len(grids['g1'])}, g3 days {len(grids['g3'])}, "
          f"scored on the {len(common)} days present in both ({start}..{end}, {hour}z)")
    results = {"days": len(common)}
    if len(common) < 20:
        raise SystemExit(f"ablation needs both grid sources on the same days; only {len(common)} "
                         "(set HAZARDPULSE_HRRR_CACHE to the current-geometry cache and keep the "
                         "pre-2026-10-01 files in .cache/hrrr)")
    for name, cfg in VARIANTS.items():
        if cfg["src"] not in sources:
            continue
        src = grids[cfg["src"]]
        locate = LOCATORS.get(cfg.get("locate", "box"))
        X, y, dd = [], [], []
        for d in common:
            if cfg["kind"] == "legacy":
                rows, labs = _build._cells_for(src[d], reports_cst.get(d, []),
                                               _build.LEGACY_NEG_PER_DAY, _build.LEGACY_NEG_MIN_CAPE,
                                               np.random.RandomState(int(d)))
            elif cfg["kind"] == "easy":
                rows, labs = _label_day_easy(src[d], reports_utc, reports_cst, d, hour,
                                             locate=locate, labels=cfg["labels"],
                                             sanitize=cfg["sanitize"], rng=np.random.RandomState(int(d)))
            else:
                lab = _build.label_day_hard(src[d], reports_utc, d, hour, locate=locate,
                                            carry_back=cfg.get("carry_back", True),
                                            label_window_h=cfg.get("label_window_h",
                                                                   _build.LABEL_WINDOW_H),
                                            population_kw=cfg.get("population_kw"))
                rows, labs = lab.rows, lab.labels
            X.extend(rows)
            y.extend(labs)
            dd.extend([int(d)] * len(rows))
        X = np.asarray(X, float).reshape(-1, len(NAMES))
        y = np.asarray(y, int)
        dd = np.asarray(dd)
        r = _fit_score(X, y, dd)
        b = _ci(r["y"], {"xgboost": r["p"], "stp": r["Xte"][:, I_STP], "refc": r["Xte"][:, I_REFC]},
                r["d"])
        results[name] = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                         "n_test_pos": int(r["y"].sum()), "n_test_neg": int(len(r["y"]) - r["y"].sum()),
                         "split_cut": r["cut"],
                         **{k: {"auc": v["auc"], "ci95": v["ci95"]} for k, v in b.items()
                            if not k.startswith("_")}}
        rr = results[name]
        print(f"  {name:26s} pos {rr['n_pos']:5d} neg {rr['n_neg']:6d} | test {rr['n_test_pos']:4d}/"
              f"{rr['n_test_neg']:5d} | xgb {rr['xgboost']['auc']:.3f} "
              f"[{rr['xgboost']['ci95'][0]:.3f},{rr['xgboost']['ci95'][1]:.3f}]  "
              f"STP {rr['stp']['auc']:.3f} [{rr['stp']['ci95'][0]:.3f},{rr['stp']['ci95'][1]:.3f}]  "
              f"refc {rr['refc']['auc']:.3f} [{rr['refc']['ci95'][0]:.3f},{rr['refc']['ci95'][1]:.3f}]")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--start", default="20220401")
    ap.add_argument("--end", default="20240831")
    ap.add_argument("--hour", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    report: dict = {}
    for path in args.data:
        dec = decompose(path)
        report.setdefault("decompose", []).append(dec)
        f = dec["full"]
        print(f"\n== {path}  (test {dec['n_test_pos']} pos / {dec['n_test_neg']} neg, cut {dec['split_cut']})")
        for k in ("xgboost", "refc_alone", "mlcape_alone", "stp_alone"):
            print(f"   {k:14s} AUC {f[k]['auc']:.4f}  [{f[k]['ci95'][0]:.4f},{f[k]['ci95'][1]:.4f}]")
        print(f"   positives with NO echo at the analysis (refc <= -10): {dec['pos_no_echo_frac']:.3f}; "
              f"refc>=40: pos {dec['pos_refc_ge40_frac']:.3f} / neg {dec['neg_refc_ge40_frac']:.3f}; "
              f"mlcape<250: pos {dec['pos_mlcape_lt250_frac']:.3f} / neg {dec['neg_mlcape_lt250_frac']:.3f}")
        for k, v in dec["subsets"].items():
            print(f"   subset {k:36s} pos {v['n_pos']:4d} neg {v['n_neg']:5d}  xgb AUC {v['auc']:.4f} "
                  f"[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]  STP {v['stp_auc']:.4f} "
                  f"[{v['stp_ci95'][0]:.4f},{v['stp_ci95'][1]:.4f}]")
    if args.ablation:
        report["ablation"] = ablation(args.start, args.end, args.hour)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
