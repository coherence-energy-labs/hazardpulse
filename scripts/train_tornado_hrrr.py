#!/usr/bin/env python3
"""Measure tornado-environment skill from the self-contained HRRR dataset.

Loads the .npz built by build_tornado_hrrr_dataset.py, does an HONEST temporal
holdout (train on earlier dates, test on later -- no leakage across the same
outbreak), and reports AUC + Brier for the incumbent-style GBT vs the SOTA
candidates, each with a DAY-CLUSTER bootstrap 95 % CI (cells on one day share one
weather regime, so the day -- not the cell -- is the resampling unit).

Every model is reported next to single-feature baselines on the SAME test cells
(STP, 0-1 km SRH, 0-3 km SRH, 0-6 km shear, CAPE, reflectivity) and a paired
bootstrap of model-minus-STP, so the model's real increment over the textbook
discriminator is visible -- not just its distance from 0.5.

The dataset's own population matters more than the model: a dataset whose negatives
include no-storm cells (``--negatives legacy-easy``) is scored mostly on
storm-vs-no-storm; the default hard dataset asks tornadic-vs-non-tornadic storm.

    python scripts/train_tornado_hrrr.py --data .cache/tornado/hrrr_env_dataset.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "Coherence" / "omega_one"))

# HAZARDPULSE_GPU=0 must mean NO GPU for every model here, not only xgboost/lightgbm:
# BestTabular's TabPFN members pick CUDA on their own whenever torch can see it.
if os.environ.get("HAZARDPULSE_GPU", "1") == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_tbt_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_tbt_spec)
_tbt_spec.loader.exec_module(_tbt)
# The shared stepwise roc_auc does not score a tied pos/neg pair as 1/2 (its result
# depends on the order argsort leaves ties in); single-feature baselines are full of
# ties (STP = 0, refc = -10). Every AUC here is auc_rank, which equals roc_auc on
# tie-free scores.
roc_auc, brier = _tbt.roc_auc, _tbt.brier

# Single-feature baselines (higher value = more tornadic) scored on the test cells.
BASELINE_FEATURES = ("stp_eff", "srh_01", "srh_03", "srh_05_est", "shear_06", "mlcape", "refc")
REFERENCE_BASELINE = "stp_eff"     # the textbook tornado discriminator
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0
LIVE_CEILING = 0.64


def _temporal_split(X, y, dates, test_frac=0.25):
    uniq = np.array(sorted(set(int(d) for d in dates)))
    cut = uniq[int((1.0 - test_frac) * len(uniq))]
    tr = dates < cut
    te = dates >= cut
    return X[tr], y[tr], X[te], y[te], int(cut)


def _impute(Xtr, Xte):
    mean = np.nanmean(Xtr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mean)
    Xte = np.where(np.isfinite(Xte), Xte, mean)
    return Xtr.astype(float), Xte.astype(float)


def auc_rank(y, s) -> float:
    """Mann-Whitney AUC with average ranks for ties (== trapezoidal ROC AUC)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    new = np.r_[True, ss[1:] != ss[:-1]]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, len(s)])
    avg = starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(len(s))
    ranks[order] = avg[np.cumsum(new) - 1]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def day_bootstrap(y, scores: dict, dates, *, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED,
                  reference: str | None = None) -> dict:
    """Day-cluster bootstrap 95 % CIs for each score's AUC (+ paired delta vs reference)."""
    y = np.asarray(y).astype(int)
    dates = np.asarray(dates)
    uniq = np.unique(dates)
    idx_by_day = [np.flatnonzero(dates == d) for d in uniq]
    rng = np.random.RandomState(seed)
    draws = {k: [] for k in scores}
    deltas = {k: [] for k in scores if reference and k != reference}
    for _ in range(reps):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_day[p] for p in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals = {k: auc_rank(yb, np.asarray(s)[idx]) for k, s in scores.items()}
        for k, v in vals.items():
            draws[k].append(v)
        for k in deltas:
            deltas[k].append(vals[k] - vals[reference])
    out = {}
    for k, s in scores.items():
        d = np.asarray(draws[k])
        out[k] = {"auc": auc_rank(y, s), "ci95": [float(np.percentile(d, 2.5)),
                                                    float(np.percentile(d, 97.5))]}
        if k in deltas:
            dd = np.asarray(deltas[k])
            out[k][f"delta_vs_{reference}"] = float(out[k]["auc"] - auc_rank(y, scores[reference]))
            out[k][f"delta_vs_{reference}_ci95"] = [float(np.percentile(dd, 2.5)),
                                                    float(np.percentile(dd, 97.5))]
    out["_reps_used"] = int(len(next(iter(draws.values()))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".cache/tornado/hrrr_env_dataset.npz")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", default="results/calibration/tornado_hrrr_report.json")
    ap.add_argument("--no-best-tabular", action="store_true",
                    help="skip the BestTabular ceiling (slow: a TabPFN portfolio below 10k rows)")
    ap.add_argument("--deploy", action="store_true",
                    help="train the forest on ALL data + export a signed tornado_forest_fp.json")
    ap.add_argument("--deploy-dir", default="results/calibration",
                    help="where --deploy writes the forest, sidecar and calibrator "
                         "(default: the SHIPPED location -- point it elsewhere for candidates)")
    args = ap.parse_args(argv)

    d = np.load(REPO / args.data if not Path(args.data).is_absolute() else args.data,
                allow_pickle=True)
    X, y, dates = d["X"], d["y"].astype(int), d["dates"]
    names = list(d["feature_names"])
    spec = str(d["feature_spec"]) if "feature_spec" in d.files else "hazardpulse/tornado/hrrr-env/v1"
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {"negatives": "legacy-easy(unlabelled)"}
    print(f"loaded {len(y)} cells  ({int(y.sum())} tornado / {len(y)-int(y.sum())} null)  "
          f"feat={X.shape[1]}  dates={len(set(int(x) for x in dates))}  "
          f"negatives={meta.get('negatives')}  spec={spec}")

    Xtr, ytr, Xte, yte, cut = _temporal_split(X, y, dates, args.test_frac)
    dte = dates[dates >= cut]                        # test-row dates, same order as Xte
    Xtr, Xte = _impute(Xtr, Xte)
    print(f"temporal split @ {cut}: train={len(ytr)} ({int(ytr.sum())} tor)  "
          f"test={len(yte)} ({int(yte.sum())} tor)  test days={len(set(dte.tolist()))}")
    if ytr.sum() < 5 or yte.sum() < 5 or len(set(yte)) < 2:
        print("  not enough tornado samples in a split for an honest AUC yet.")
        return 0

    report = {"n": int(len(y)), "n_pos": int(y.sum()), "n_features": int(X.shape[1]),
              "split_cut": cut, "n_test": int(len(yte)), "live_ceiling": LIVE_CEILING,
              "negatives": meta.get("negatives"), "feature_spec": spec, "dataset_meta": meta,
              "n_train_pos": int(ytr.sum()), "n_train_neg": int(len(ytr) - ytr.sum()),
              "n_test_pos": int(yte.sum()), "n_test_neg": int(len(yte) - yte.sum()),
              "test_base_rate": float(yte.mean()), "n_test_days": int(len(set(dte.tolist())))}
    scores: dict[str, np.ndarray] = {}

    # incumbent-style GBD baseline (xgboost CPU/GPU via the shared fitters)
    base = _tbt._fit_xgboost(Xtr, ytr, 0)
    pbase = np.asarray(base.predict_proba(Xte))[:, 1]
    report["xgboost"] = {"auc": auc_rank(yte, pbase), "brier": brier(yte, pbase)}
    scores["xgboost"] = pbase
    print(f"  xgboost      AUC {report['xgboost']['auc']:.4f}  Brier {report['xgboost']['brier']:.4f}")

    # servable + signed VerifiableForest (best of xgb/lgbm, honest selection)
    vf_holdout_proba = None
    try:
        vf = _tbt._verifiable_forest(Xtr, ytr, Xte, seed=0)
        vf_holdout_proba = np.asarray(vf["proba"], float)   # held-out -> honest calibration set
        report["verifiable_forest"] = {
            "auc": auc_rank(yte, vf["proba"]), "brier": brier(yte, vf["proba"]),
            "booster": vf["booster"], "self_reproduce": vf["self_reproduce"],
            "bit_exact": vf["bit_exact"], "n_trees": vf["n_trees"],
        }
        scores["verifiable_forest"] = vf_holdout_proba
        print(f"  forest[{vf['booster']}] AUC {report['verifiable_forest']['auc']:.4f}  "
              f"Brier {report['verifiable_forest']['brier']:.4f}  (servable+signed)")
    except Exception as exc:
        print(f"  verifiable_forest skipped: {exc}")

    # SOTA ceiling
    if not args.no_best_tabular:
        try:
            from omega.super_ensemble import BestTabular
            bt = BestTabular()
            bt.fit(Xtr, ytr)
            pbt = np.asarray(bt.predict_proba(Xte))[:, 1]
            report["best_tabular"] = {"auc": auc_rank(yte, pbt), "brier": brier(yte, pbt)}
            scores["best_tabular"] = pbt
            print(f"  BestTabular  AUC {report['best_tabular']['auc']:.4f}  "
                  f"Brier {report['best_tabular']['brier']:.4f}  (ceiling)")
        except Exception as exc:
            print(f"  best_tabular skipped: {exc}")

    model_keys = [k for k in ("xgboost", "verifiable_forest", "best_tabular") if k in scores]
    best_auc = max(report[k]["auc"] for k in model_keys)
    report["best_auc"] = best_auc
    report["beats_live_ceiling"] = bool(best_auc > LIVE_CEILING)

    # Single-feature baselines on the same (imputed) test cells, then day-cluster CIs.
    for f in BASELINE_FEATURES:
        if f in names:
            scores[f"feature:{f}"] = Xte[:, names.index(f)]
    ref = f"feature:{REFERENCE_BASELINE}" if f"feature:{REFERENCE_BASELINE}" in scores else None
    boot = day_bootstrap(yte, scores, dte, reference=ref)
    report["bootstrap"] = {"unit": "test day", "reps_requested": BOOTSTRAP_REPS,
                           "reps_used": boot.pop("_reps_used"), "seed": BOOTSTRAP_SEED}
    for k in model_keys:
        report[k].update({kk: v for kk, v in boot[k].items() if kk != "auc"})
    report["baselines"] = {k.split(":", 1)[1]: boot[k] for k in scores if k.startswith("feature:")}
    print(f"\n  day-cluster bootstrap ({report['bootstrap']['reps_used']} reps over "
          f"{report['n_test_days']} test days), test base rate {report['test_base_rate']:.4f}:")
    for k in list(model_keys) + [k for k in scores if k.startswith("feature:")]:
        b = boot[k]
        extra = ""
        if ref and k != ref and f"delta_vs_{ref}" in b:
            extra = (f"   minus STP {b[f'delta_vs_{ref}']:+.4f} "
                     f"[{b[f'delta_vs_{ref}_ci95'][0]:+.4f},{b[f'delta_vs_{ref}_ci95'][1]:+.4f}]")
        print(f"    {k:26s} AUC {b['auc']:.4f}  95% CI [{b['ci95'][0]:.4f},{b['ci95'][1]:.4f}]{extra}")

    # Deploy: train the signed forest on ALL data and export. Trains on RAW features
    # (with NaNs) -- the booster learns the missing-value routing, which VerifiableForest
    # freezes as default_left, so the live scorer can feed raw HRRR (gaps and all) and
    # the frozen forest routes them correctly. No imputation means/stds to drift.
    # The deployed booster is the one the holdout SELECTED and the calibrator below is
    # fit on (it used to be xgboost unconditionally, so a lightgbm selection shipped an
    # xgboost forest under a calibrator fit to lightgbm scores).
    if args.deploy and report.get("verifiable_forest"):
        dep = Path(args.deploy_dir)
        dep = dep if dep.is_absolute() else REPO / dep
        dep.mkdir(parents=True, exist_ok=True)
        booster = report["verifiable_forest"]["booster"]
        clf = dict(_tbt._FOREST_BOOSTERS)[booster](np.asarray(X, float), y, 0)   # raw, NaN-native
        vf_all, constants = _tbt._freeze(booster, clf)
        fp = dep / "tornado_forest_fp.json"
        fp.write_text(json.dumps(constants) + "\n", encoding="utf-8")
        # sidecar: feature order + spec the live scorer must reproduce (v2 = sanitized grids)
        (dep / "tornado_forest_features.json").write_text(
            json.dumps({"feature_names": names, "spec": spec,
                        "negatives": meta.get("negatives"),
                        "geometry": meta.get("geometry"),
                        "population": {k: meta[k] for k in ("refc_min_dbz", "shear06_min_ms",
                                                            "cape_min_jkg") if k in meta}}) + "\n",
            encoding="utf-8")
        report.update(deployed=True, deployed_n=int(len(y)), deploy_dir=str(dep), deployed_booster=booster,
                      model_sha256=vf_all.fp_model_sha256(),
                      n_trees=int(len(constants["tree_root"])))
        print(f"  SHIPPED signed tornado forest ({report['n_trees']} trees, "
              f"sha {report['model_sha256'][:12]}) -> {fp}")

        # Honest calibration: fit Venn-Abers on the HELD-OUT forest probs vs outcomes
        # so the live scorer turns the raw (overconfident) forest score into a
        # calibrated P(tornado) with a validity interval -- not a bare 0.99.
        if vf_holdout_proba is not None:
            from hazardpulse.trust.venn_abers import VennAbersCalibrator
            from hazardpulse.trust.calibration import (
                expected_calibration_error as _ece, brier_score as _bs)
            cal = VennAbersCalibrator().fit(vf_holdout_proba, yte.astype(int))
            cal_p, _, _ = cal.predict(vf_holdout_proba)
            rec = {"hazard": "tornado_hrrr",
                   "model_version": ("tornado_hrrr_env_v2" if spec.endswith("/v2")
                                     else "tornado_hrrr_env_v1"),
                   "n_calibration": int(len(yte)), "calibrator": cal.to_dict(),
                   "ece_before": round(float(_ece(vf_holdout_proba, yte)), 4),
                   "ece_after": round(float(_ece(cal_p, yte)), 4),
                   "brier_before": round(float(_bs(vf_holdout_proba, yte)), 4),
                   "brier_after": round(float(_bs(cal_p, yte)), 4)}
            (dep / "tornado_hrrr_calibration.json").write_text(
                json.dumps(rec) + "\n", encoding="utf-8")
            report["calibration"] = {k: rec[k] for k in
                                     ("ece_before", "ece_after", "brier_before", "brier_after")}
            print(f"  calibrated: ECE {rec['ece_before']:.3f} -> {rec['ece_after']:.3f}, "
                  f"Brier {rec['brier_before']:.3f} -> {rec['brier_after']:.3f}")

    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdict = "BEATS" if best_auc > LIVE_CEILING else "does NOT beat"
    print(f"\n  best AUC {best_auc:.4f} -> {verdict} the live ceiling (~{LIVE_CEILING})")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
