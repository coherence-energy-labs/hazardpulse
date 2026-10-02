#!/usr/bin/env python3
"""Honest temporal evaluation of the hurricane RI recipes, and the pre-registered serve decision.

Rolling origin, split by each storm's FIRST season (a storm never straddles a split):

    members      storms first seen 2000-2018
    calibration  storms first seen 2019-2021   (held out from the members)
    test         storms first seen 2022-2024   (held out from everything; scored once)

The served artifacts are the same recipes rolled forward one origin (members <= 2021,
calibration 2022-2024), so what is evaluated here is the procedure that produced them.

Truth on calibration and test rows is always the TRUE-CLOCK label (v8.2 set: wind 24 h
later minus wind now >= 30 kt) -- the event the site publishes -- and every model is scored
under its live serving contract (v8.1 models get true 6/12/24-h deltas, as live gives them,
and median-imputed speed/age). Candidates:

    A  v8.1 members + the legacy 500-step in-sample Platt   (what was served until 2026-10)
    B  v8.1 members + held-out converged calibration        (recipe hurricane_ri_v8_1_1)
    C  v8.2 members + held-out converged calibration        (recipe hurricane_ri_v8_2)
    P  persistence: logistic MLE on the past-24 h intensity change (members' years)
    K  climatology: the members' years' base rate

Intervals: 95% percentile bootstrap over TEST STORMS (2,000 replicates).

DECISION RULE (written before the evaluation was first run, 2026-10-02):
    serve hurricane_ri_v8_2   iff  AUC(C) >= AUC(B)  and  Brier(C) <= Brier(B)  and  C is calibrated
    else serve hurricane_ri_v8_1_1 iff B is calibrated
    else serve neither (the job keeps the last calibrated artifact; report why)
  where "calibrated" on TEST means: the storm-bootstrap 95% interval of
  (mean forecast - observed rate) contains 0 AND that of the calibration slope contains 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import operational_ri as ori  # noqa: E402
from hazardpulse.hurricane import ri_evaluation as ev  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402

ORIGIN = {"members": (2000, 2018), "calibration": (2019, 2021), "test": (2022, 2024)}
REPORT_PATH = ROOT / "results" / "calibration" / "hurricane_ri_evaluation.json"
BOOTSTRAP_REPS = 2000
SPEED_AGE = ["translation_speed_kmh", "storm_age_h"]


SMOKE_CONFIG = ori.OperationalRIConfig(  # --smoke: wiring check only, never a result
    feature_select_epochs=50, hgbt_d3_n_trees=3, hgbt_d4_n_trees=3,
    logistic_epochs_final=50, bag_n_bags_final=3, bag_epochs_final=30,
)


def _fit_members_job(dataset: str, years: tuple[int, int], legacy: bool, smoke: bool = False):
    rows = ri_model.select_years(ri_model.load_dataset(dataset), years)
    tag = f"[{dataset} members {years[0]}-{years[1]}]"
    t0 = time.perf_counter()
    out = ri_model.fit_members(rows, SMOKE_CONFIG if smoke else None,
                               log=lambda m: print(tag, m, flush=True),
                               return_training_ensemble=legacy)
    print(tag, f"fitted in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def _labels(rows: list[dict]) -> np.ndarray:
    return np.array([float(c["ri_label_30kt"]) for c in rows])


def calibrated(ci: dict[str, list[float]]) -> bool:
    lo, hi = ci["mean_minus_observed"]
    s_lo, s_hi = ci["calibration_slope"]
    return bool(lo <= 0.0 <= hi and s_lo <= 1.0 <= s_hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--smoke", action="store_true", help="tiny members + few replicates: wiring check only")
    args = parser.parse_args(argv)
    if args.smoke and args.out == REPORT_PATH:
        parser.error("--smoke must not overwrite the real report; pass --out")
    t_start = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        job81 = pool.submit(_fit_members_job, "v8.1", ORIGIN["members"], True, args.smoke)
        job82 = pool.submit(_fit_members_job, "v8.2", ORIGIN["members"], False, args.smoke)
        members81, p_in81, y_in81 = job81.result()
        members82 = job82.result()

    v81 = ri_model.load_dataset("v8.1")
    v82 = ri_model.load_dataset("v8.2")
    train82 = ri_model.select_years(v82, ORIGIN["members"])
    cal82 = ri_model.select_years(v82, ORIGIN["calibration"])
    test82 = ri_model.select_years(v82, ORIGIN["test"])
    test81 = ri_model.select_years(v81, ORIGIN["test"])
    y_cal, y_test = _labels(cal82), _labels(test82)

    def as_model(members, version, impute_live, calibration=None):
        return {"schema": ri_model.SCHEMA, "model_version": version, **members,
                "serving": {"impute_live": impute_live}, "calibration": calibration}

    # A: v8.1 members + the legacy Platt (500 GD steps on its own training predictions)
    _, a_legacy, b_legacy = ori.platt_calibrate(y_in81, p_in81, p_in81[:1])
    model_a = as_model(members81, "hurricane_ri_v8_1", SPEED_AGE,
                       {"method": "platt_prob_gd500_insample", "a": a_legacy, "b": b_legacy})
    # B and C: held-out converged calibration
    model_b = as_model(members81, "hurricane_ri_v8_1_1", SPEED_AGE)
    model_b["calibration"] = ri_model.fit_logit_calibrator(
        ri_model.member_probabilities(model_b, cal82)["ensemble"], y_cal)
    model_c = as_model(members82, "hurricane_ri_v8_2", [])
    model_c["calibration"] = ri_model.fit_logit_calibrator(
        ri_model.member_probabilities(model_c, cal82)["ensemble"], y_cal)

    # Baselines, fitted on the members' years only.
    y_train = _labels(train82)
    dv_train = np.array([c.get("analysis_dv_24h") for c in train82], dtype=float)
    dv_median = float(np.nanmedian(dv_train))

    def dv24(rows):
        x = np.array([c.get("analysis_dv_24h") for c in rows], dtype=float)
        return np.where(np.isnan(x), dv_median, x)

    theta_p, diag_p = ri_model.newton_logistic(np.column_stack([dv24(train82), np.ones(len(train82))]), y_train)
    climatology = float(y_train.mean())

    preds = {
        "A_v8_1_legacy_platt": ri_model.score_cases(model_a, test82)["calibrated"],
        "B_v8_1_1_heldout_newton": ri_model.score_cases(model_b, test82)["calibrated"],
        "C_v8_2_heldout_newton": ri_model.score_cases(model_c, test82)["calibrated"],
        "P_persistence_dv24": ori.sigmoid(theta_p[0] * dv24(test82) + theta_p[1]),
        "K_climatology": np.full(len(test82), climatology),
    }
    groups = np.array([c["storm_id"] for c in test82])

    results: dict[str, dict] = {}
    for name, p in preds.items():
        point = ev.point_metrics(y_test, p, climatology)
        is_const = name.startswith("K_")

        def stat(idx, p=p, is_const=is_const):
            yy, pp = y_test[idx], p[idx]
            out = {"auc": ev.auc(yy, pp), "brier": ev.brier(yy, pp),
                   "bss_vs_climatology": 1.0 - ev.brier(yy, pp) / ev.brier(yy, np.full(len(yy), climatology)),
                   "mean_minus_observed": float(pp.mean() - yy.mean())}
            if not is_const:
                try:
                    out["calibration_intercept"], out["calibration_slope"] = ev.calibration_intercept_slope(yy, pp)
                except ri_model.CalibrationError:
                    out["calibration_intercept"] = out["calibration_slope"] = float("nan")
            return out

        ci = ev.storm_bootstrap(groups, stat, reps=args.reps)
        results[name] = {**point, "ci95": ci, "reliability": ev.reliability(y_test, p),
                         "calibrated": None if is_const else calibrated(ci)}

    def paired(a: str, b: str) -> dict[str, list[float]]:
        pa, pb = preds[a], preds[b]
        return ev.storm_bootstrap(groups, lambda idx: {
            "delta_auc": ev.auc(y_test[idx], pa[idx]) - ev.auc(y_test[idx], pb[idx]),
            "delta_brier": ev.brier(y_test[idx], pa[idx]) - ev.brier(y_test[idx], pb[idx]),
        }, reps=args.reps)

    comparisons = {
        "C_minus_B": paired("C_v8_2_heldout_newton", "B_v8_1_1_heldout_newton"),
        "C_minus_P": paired("C_v8_2_heldout_newton", "P_persistence_dv24"),
        "B_minus_P": paired("B_v8_1_1_heldout_newton", "P_persistence_dv24"),
        "B_minus_A": paired("B_v8_1_1_heldout_newton", "A_v8_1_legacy_platt"),
    }
    # v8.1 judged on its own (12-hour) label, for transparency only.
    p81_self = ri_model.member_probabilities(model_a, test81)["ensemble"]
    v81_self_auc = ev.auc(_labels(test81), p81_self)

    rB, rC = results["B_v8_1_1_heldout_newton"], results["C_v8_2_heldout_newton"]
    if rC["auc"] >= rB["auc"] and rC["brier"] <= rB["brier"] and rC["calibrated"]:
        serve, why = "hurricane_ri_v8_2", "v8.2 is at least as good as v8.1.1 on AUC and Brier and is calibrated"
    elif rB["calibrated"]:
        serve, why = "hurricane_ri_v8_1_1", (
            "v8.2 failed the rule (AUC/Brier/calibration, see metrics); v8.1.1 is calibrated")
    else:
        serve, why = None, "neither held-out-calibrated candidate passed the calibration test on TEST"

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "origin": {k: list(v) for k, v in ORIGIN.items()},
        "truth": "true-clock 24 h RI (v8.2 set): wind(t+24h) - wind(t) >= 30 kt, synoptic times",
        "data_sha256": {k: ri_model.sha256_file(v) for k, v in ri_model.DATASETS.items()},
        "bootstrap": {"unit": "storm", "reps": args.reps, "interval": "percentile 95%"},
        "n_test_rows": int(len(test82)), "n_test_storms": int(len(np.unique(groups))),
        "n_test_events": int(y_test.sum()),
        "climatology_rate_members_years": climatology,
        "persistence_model": {"a_dv24": float(theta_p[0]), "b": float(theta_p[1]), **diag_p},
        "calibrators": {"B": model_b["calibration"], "C": model_c["calibration"],
                        "A_legacy": {"a": float(a_legacy), "b": float(b_legacy)}},
        "results": results,
        "paired": comparisons,
        "v8_1_on_its_own_12h_label_test_auc": v81_self_auc,
        "decision_rule": "serve v8.2 iff AUC(C)>=AUC(B) and Brier(C)<=Brier(B) and C calibrated; "
                         "else v8.1.1 iff B calibrated; calibrated = 95% storm-bootstrap CI of "
                         "(mean forecast - observed) contains 0 and of calibration slope contains 1",
        "decision": {"serve": serve, "why": why},
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, indent=1, allow_nan=True) + "\n")

    print()
    print(f"TEST: {report['n_test_rows']} rows, {report['n_test_storms']} storms, "
          f"{report['n_test_events']} RI events (rate {y_test.mean():.4f}); climatology {climatology:.4f}")
    hdr = f"{'model':26s} {'AUC [95% CI]':24s} {'Brier':>8s} {'BSS':>7s} {'mean fc':>8s} {'slope [CI]':>20s} {'range':>17s} cal"
    print(hdr)
    for name, r in results.items():
        ci = r["ci95"]
        slope = (f"{r['calibration_slope']:.2f} [{ci['calibration_slope'][0]:.2f},{ci['calibration_slope'][1]:.2f}]"
                 if "calibration_slope" in ci else "-")
        print(f"{name:26s} {r['auc']:.3f} [{ci['auc'][0]:.3f},{ci['auc'][1]:.3f}]   "
              f"{r['brier']:8.5f} {r['bss_vs_climatology']:7.3f} {r['mean_forecast']:8.4f} {slope:>20s} "
              f"[{r['prob_range']['min']:.4f},{r['prob_range']['max']:.3f}] {r['calibrated']}")
    for k, v in comparisons.items():
        print(f"  {k}: dAUC CI {v['delta_auc'][0]:+.4f}..{v['delta_auc'][1]:+.4f}  "
              f"dBrier CI {v['delta_brier'][0]:+.5f}..{v['delta_brier'][1]:+.5f}")
    print(f"  v8.1 on its own 12-h label, test AUC {v81_self_auc:.3f}")
    print(f"DECISION: serve {serve} -- {why}")
    print(f"Wrote {args.out} in {report['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
