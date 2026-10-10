#!/usr/bin/env python3
"""Amendment E4 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 12): GEAR1's weight by how active the cell is.

    python scripts/earthquake_program/gear1_e4.py      # on the published program data

  S1  logit p = a + c z + b g             (served since E1: the control)
  S2  logit p = a + c z + b g + d z g     (the challenger: GEAR1's weight is b + d z)

with z = logit(p_C0) and g = log10 of GEAR1's 30-day count (section 10's input), both fitted on CHOOSE, decided on
DEV (the 95% month-block interval of IG(S2) - IG(S1) entirely above 0), FINAL a declared second read.

Control (it stops the run): S1 refitted here reproduces section 10.1's coefficients within 1e-5 and its DEV
information gain within 1e-6.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import evaluate as ev  # noqa: E402
import gear1_stack as gs  # noqa: E402

SPLITS = ("choose", "dev", "final")
E1_S1 = {"a": 0.30891700168422964, "c": 0.9442332436931772, "b": 0.25669330473358914}
E1_S1_DEV_IG = 2.7362907775152046
COEF_TOL, IG_TOL = 1e-5, 1e-6


def design2(z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """S2's columns: 1, z, g, z g (g broadcast over issue times)."""
    gb = np.broadcast_to(g, z.shape)
    return np.column_stack([np.ones(z.size), z.ravel(), gb.ravel(), (z * gb).ravel()])


def fit_s2(z: np.ndarray, g: np.ndarray, y: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Maximum Bernoulli likelihood of sigmoid(design2 theta), from ``start`` (S1's coefficients, d = 0)."""
    X = design2(z, g)

    def nll(theta):
        eta = X @ theta
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return -np.sum(y * eta - np.logaddexp(0.0, eta)), -(X.T @ (y - p))
    r = optimize.minimize(nll, start, jac=True, method="L-BFGS-B")
    if not r.success:
        raise SystemExit(f"S2's fit did not converge: {r.message}")
    return r.x


def predict_s2(theta: np.ndarray, z: np.ndarray, g: np.ndarray) -> np.ndarray:
    eta = design2(z, g) @ theta
    return (0.5 * (1.0 + np.tanh(0.5 * eta))).reshape(z.shape)


def main() -> int:
    g_rep = json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))
    g = gs.gear1_log_map(g_rep["cells_per_year"])
    fit = json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))
    p0 = float(fit["fit"]["p0"])
    events = C.load_events(4.5)
    data = {}
    for split in SPLITS:
        issue = C.split_issue_times(split)
        data[split] = {"issue": issue, "Y": C.binary_targets(events, issue), "C0": C.load_pred("C0", split)}
        print(f"{split}: {issue.size} issue times, {int(data[split]['Y'].sum())} positive cell-windows")

    ch = data["choose"]
    y = ch["Y"].ravel().astype(float)
    z = gs.logit(ch["C0"])
    th1 = gs.fit_logistic(gs.design(z, g), y)
    got = {"a": float(th1[0]), "c": float(th1[1]), "b": float(th1[2])}
    off = max(abs(got[k] - E1_S1[k]) for k in E1_S1)
    if off > COEF_TOL:
        raise SystemExit(f"control failed: S1 refitted {got} differs from section 10.1's {E1_S1} by {off:.2e}")
    th2 = fit_s2(z, g, y, np.array([th1[0], th1[1], th1[2], 0.0]))
    coef = {"S1": got, "S2": {"a": float(th2[0]), "c": float(th2[1]), "b": float(th2[2]), "d": float(th2[3])}}
    pos = ch["Y"].astype(bool)
    zq = {"all_cell_times": np.percentile(z, (10, 50, 90)), "positive_cell_times": np.percentile(z[pos], (10, 50, 90))}
    weight = {k: {f"p{q}": {"z": float(v), "gear1_weight_S2": float(th2[2] + th2[3] * v)}
                  for q, v in zip((10, 50, 90), qs)} for k, qs in zq.items()}
    print("fitted on CHOOSE:", json.dumps(coef), "| GEAR1 weight b + d z:", json.dumps(weight))

    report = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 12 (amendment E4)", "prereg_tag": "prereg-earthquake-e4",
              "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "gear1_sha256": g_rep["sha256"], "control": {"S1_coefficients_max_abs_diff": off},
              "coefficients_fitted_on_choose": coef, "gear1_weight_by_activity": weight, "splits": {}}
    for split in SPLITS:
        d = data[split]
        zs = gs.logit(d["C0"])
        P = {"S1": gs.predict_logistic(th1, zs, g), "S2": predict_s2(th2, zs, g)}
        sc = C.SplitScorer(d["issue"], d["Y"], p0=p0)
        sc_act = C.SplitScorer(d["issue"], d["Y"], p0=p0, mask=ev.active_masks(d["issue"]))
        st = {k: sc.stats(v) for k, v in P.items()}
        res = {"role": {"choose": "fitted here (in-sample for the coefficients)", "dev": "decides",
                        "final": "declared second read: reported, never used to decide"}[split],
               "n_issue_times": int(d["issue"].size), "n_positive": int(d["Y"].sum()),
               "models": {k: sc.summary(s) for k, s in st.items()},
               "paired": {"S2-S1": sc.paired(st["S2"], st["S1"])}}
        for k in P:
            res["models"][k]["auc_active_cells"] = sc_act.summary(sc_act.stats(P[k]))["auc"]
        report["splits"][split] = res
        for k, m in res["models"].items():
            print(f"  {split:6s} {k} IG {m['ig_per_target']['value']:.6f} AUC {m['auc']['value']:.5f} "
                  f"BSS {m['bss']['value']:+.5f} sum p/sum y {m['calib_ratio']['value']:.4f}")
        pr = res["paired"]["S2-S1"]
        print(f"  {split:6s} S2-S1 dIG {pr['ig_per_target']['diff']:+.5f} {[round(v, 5) for v in pr['ig_per_target']['ci95']]}"
              f" dAUC {pr['auc']['diff']:+.5f} {[round(v, 5) for v in pr['auc']['ci95']]}")
    dev_ig = report["splits"]["dev"]["models"]["S1"]["ig_per_target"]["value"]
    if abs(dev_ig - E1_S1_DEV_IG) > IG_TOL:
        raise SystemExit(f"control failed: S1's DEV IG {dev_ig!r} != section 10.1's {E1_S1_DEV_IG!r}")
    report["control"]["S1_dev_ig_abs_diff"] = abs(dev_ig - E1_S1_DEV_IG)
    ci = report["splits"]["dev"]["paired"]["S2-S1"]["ig_per_target"]["ci95"]
    carried = bool(ci[0] > 0)
    report["carried_rule"] = {"S2_replaces_S1": carried, "dev_S2_minus_S1_ig_ci95": ci}
    report["prediction_d_negative"] = bool(th2[3] < 0)
    print(f"control passed (coefficients {off:.1e}, DEV IG {abs(dev_ig - E1_S1_DEV_IG):.1e}); d < 0: {th2[3] < 0}; "
          f"CARRIED RULE: S2 replaces S1 = {carried}")
    C.write_json("gear1_e4.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
