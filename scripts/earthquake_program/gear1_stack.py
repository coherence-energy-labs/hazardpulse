#!/usr/bin/env python3
"""Amendment E1 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10): does GEAR1 add to C0?

    python scripts/earthquake_program/gear1_stack.py      # after gear1_map.py

Fitted on CHOOSE (out of sample for both GEAR1 and C0), decided on DEV, FINAL a declared second
read -- exactly as registered:

  S0  logit p = a + c logit(p_C0)                         (C0 recalibrated: the control)
  S1  logit p = a + c logit(p_C0) + b log10(max(G, 1e-7))  (the challenger)
  AG  P = 1 - exp(-mu ((1 - eps) G/sum G + eps/11700))      (GEAR1 alone, descriptive)
  A_ch  A's own causal map with mu, eps refitted on CHOOSE (descriptive)

Controls: A's map is recovered from A's cached forecasts by inverting A's own formula -- the round
trip must reproduce them and every recovered map must sum to 1, or the run stops.
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
import fit_ab  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

G_FLOOR = 1e-7
SPLITS = ("choose", "dev", "final")


def gear1_log_map(cells_per_year) -> np.ndarray:
    """log10 of GEAR1's expected count per grid cell over the 30-day window, floored at 1e-7
    (the registered S1 input; the served stack stores exactly this)."""
    G = np.asarray(cells_per_year, np.float64) * of.HORIZON_DAYS / 365.25
    return np.log10(np.maximum(G, G_FLOOR))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, C.CLIP, 1 - C.CLIP)
    return np.log(p) - np.log1p(-p)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maximum Bernoulli likelihood of sigmoid(X theta) (X includes the intercept column)."""
    def nll(theta):
        eta = X @ theta
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return -ll, -(X.T @ (y - p))
    x0 = np.zeros(X.shape[1])
    x0[1] = 1.0                                   # start at "C0 as it is"
    r = optimize.minimize(nll, x0, jac=True, method="L-BFGS-B")
    if not r.success:
        raise SystemExit(f"logistic fit did not converge: {r.message}")
    return r.x


def design(z: np.ndarray, logg: np.ndarray | None) -> np.ndarray:
    cols = [np.ones(z.size), z.ravel()]
    if logg is not None:
        cols.append(np.broadcast_to(logg, z.shape).ravel())
    return np.column_stack(cols)


def predict_logistic(theta: np.ndarray, z: np.ndarray, logg: np.ndarray | None) -> np.ndarray:
    eta = design(z, logg) @ theta
    return (0.5 * (1.0 + np.tanh(0.5 * eta))).reshape(z.shape)


def long_term(mu: float, eps: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    return -np.expm1(-mu * ((1 - eps) * s + eps * u))


def main() -> int:
    g_rep = json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))
    G = np.asarray(g_rep["cells_per_year"]) * of.HORIZON_DAYS / 365.25
    logg = gear1_log_map(g_rep["cells_per_year"])
    g_pdf = G / G.sum()
    u_count = np.full(of.N_CELLS, 1.0 / of.N_CELLS)
    area_pdf = of.cell_areas() / of.cell_areas().sum()
    fit = json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))
    a_mu, a_eps = fit["A_chosen"]["long"]["mu"], fit["A_chosen"]["long"]["eps"]
    p0 = float(fit["fit"]["p0"])
    events = C.load_events(4.5)

    data = {}
    for split in SPLITS:
        issue = C.split_issue_times(split)
        Y = C.binary_targets(events, issue)
        p_c0, p_a = C.load_pred("C0", split), C.load_pred("A", split)
        s_a = (-np.log1p(-p_a) / a_mu - a_eps * area_pdf) / (1 - a_eps)       # A's causal map, recovered
        back = long_term(a_mu, a_eps, s_a, area_pdf)
        if np.max(np.abs(back - p_a)) > 1e-12 or np.max(np.abs(s_a.sum(axis=1) - 1.0)) > 1e-6:
            raise SystemExit(f"{split}: A's map does not round-trip ({np.max(np.abs(back - p_a)):.1e}, "
                             f"sums {s_a.sum(axis=1).min():.6f}..{s_a.sum(axis=1).max():.6f})")
        data[split] = {"issue": issue, "Y": Y, "C0": p_c0, "s_a": s_a,
                       **{k: C.load_pred(k, split) for k in ("A", "B", "D")}}
        print(f"{split}: {issue.size} issue times, {int(Y.sum())} positive cell-windows; A's map round-trips")

    ch = data["choose"]
    y = ch["Y"].ravel().astype(float)
    z = logit(ch["C0"])
    th0 = fit_logistic(design(z, None), y)
    th1 = fit_logistic(design(z, logg), y)
    pos = np.nonzero(ch["Y"])
    ag, ag_ll = fit_ab.fit_long_term(g_pdf[pos[1]], u_count[pos[1]], ch["issue"].size)
    ach, ach_ll = fit_ab.fit_long_term(ch["s_a"][pos], area_pdf[pos[1]], ch["issue"].size)
    coef = {"S0": {"a": th0[0], "c": th0[1]}, "S1": {"a": th1[0], "c": th1[1], "b": th1[2]},
            "AG": ag, "A_ch": ach}
    print("fitted on CHOOSE:", json.dumps(coef, default=float))

    report = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10 (amendment E1)",
              "prereg_tag": "prereg-earthquake-gear1",
              "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "gear1_sha256": g_rep["sha256"], "gear1_global_total_per_year": g_rep["global_total"],
              "coefficients_fitted_on_choose": coef, "splits": {}}
    for split in SPLITS:
        d = data[split]
        P = {"C0": d["C0"],
             "S0": predict_logistic(th0, logit(d["C0"]), None),
             "S1": predict_logistic(th1, logit(d["C0"]), logg),
             "A_ch": long_term(ach["mu"], ach["eps"], d["s_a"], area_pdf),
             "AG": np.broadcast_to(long_term(ag["mu"], ag["eps"], g_pdf, u_count), d["Y"].shape)}
        P.update({k: d[k] for k in ("A", "B", "D")})          # the program's own forecasts, for reference
        sc = C.SplitScorer(d["issue"], d["Y"], p0=p0)
        act = ev.active_masks(d["issue"])
        sc_act = C.SplitScorer(d["issue"], d["Y"], p0=p0, mask=act)
        st = {k: sc.stats(v) for k, v in P.items()}
        res = {"role": {"choose": "fitted here (in-sample for the coefficients)", "dev": "decides",
                        "final": "declared second read: reported, never used to decide"}[split],
               "n_issue_times": int(d["issue"].size), "n_positive": int(d["Y"].sum()),
               "n_cell_times": int(d["Y"].size), "positives_in_active_cells": int((d["Y"] & act).sum()),
               "first_issue": str(np.datetime64(int(d["issue"][0]), "s")),
               "last_issue": str(np.datetime64(int(d["issue"][-1]), "s")),
               "models": {k: sc.summary(s) for k, s in st.items()},
               "paired": {f"{x}-{y_}": sc.paired(st[x], st[y_])
                          for x, y_ in (("S1", "S0"), ("S1", "C0"), ("S0", "C0"), ("AG", "A_ch"),
                                        ("S1", "A"), ("S1", "B"), ("S1", "D"))}}
        for k in ("S1", "C0"):
            res["models"][k]["auc_active_cells"] = sc_act.summary(sc_act.stats(P[k]))["auc"]
        report["splits"][split] = res
        for k, m in res["models"].items():
            print(f"  {split:6s} {k:4s} IG {m['ig_per_target']['value']:.4f} {[round(v, 3) for v in m['ig_per_target']['ci95']]}"
                  f"  AUC {m['auc']['value']:.4f}  BSS {m['bss']['value']:+.4f}  sum p/sum y {m['calib_ratio']['value']:.3f}")
        for k, pr in res["paired"].items():
            print(f"  {split:6s} {k:8s} dIG {pr['ig_per_target']['diff']:+.4f} {[round(v, 4) for v in pr['ig_per_target']['ci95']]}"
                  f"  dAUC {pr['auc']['diff']:+.4f} {[round(v, 4) for v in pr['auc']['ci95']]}")
    dev = report["splits"]["dev"]["paired"]
    carried = bool(dev["S1-S0"]["ig_per_target"]["ci95"][0] > 0 and dev["S1-C0"]["ig_per_target"]["diff"] >= 0)
    report["carried_rule"] = {"S1_replaces_C0": carried,
                              "dev_S1_minus_S0_ig_ci95": dev["S1-S0"]["ig_per_target"]["ci95"],
                              "dev_S1_minus_C0_ig": dev["S1-C0"]["ig_per_target"]["diff"]}
    print(f"CARRIED RULE: S1 replaces C0 = {carried}")
    C.write_json("gear1_e1.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
