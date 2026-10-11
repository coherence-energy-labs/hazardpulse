#!/usr/bin/env python3
"""Amendment E6 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 14): the region waking up -- seismic moment over the decade.

    python scripts/earthquake_program/gear1_e6.py      # on the published program data

  S2  logit p = a + c z + b g + d z g             (served since E4: the control)
  S3  logit p = a + c z + b g + d z g + e h       (h = log10 of the 3x3 neighbourhood's moment in the year before over
                                                   its mean yearly moment in the decade before; shallow M4.5+)

Both fitted on CHOOSE, decided on DEV (the 95% month-block interval of IG(S3) - IG(S2) entirely above 0), FINAL a
declared second read. Control (it stops the run): S2 refitted here reproduces section 12.1's coefficients within 1e-5
and its DEV information gain within 1e-6.
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
import gear1_e4 as e4  # noqa: E402
import gear1_stack as gs  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

SPLITS = ("choose", "dev", "final")
DAY, YEAR = 86400.0, 365.25 * 86400.0
COEF_TOL, IG_TOL = 1e-5, 1e-6
FLOOR = 1e15


def nbr3(grid: np.ndarray) -> np.ndarray:
    """Each cell's sum over its 3x3 neighbourhood: longitude wraps, and the first and last rows have no row beyond."""
    out = np.zeros_like(grid)
    for dr in (-1, 0, 1):
        shifted = np.zeros_like(grid)
        if dr == 0:
            shifted = grid
        elif dr == 1:
            shifted[:-1] = grid[1:]
        else:
            shifted[1:] = grid[:-1]
        for dc in (-1, 0, 1):
            out = out + np.roll(shifted, dc, axis=1)
    return out


def moment_ratio(issue: np.ndarray) -> np.ndarray:
    """h (n_issue x N_CELLS): log10((M0 in [t - 1 y, t) + 1e15) / (M0 in [t - 11 y, t - 1 y) / 10 + 1e15)) summed over
    each cell's 3x3 neighbourhood (longitude wraps; the edge rows have no row beyond), shallow (<= 70 km) M4.5+."""
    r = C.load_raw("program_catalog.npz")
    keep = (r["mag"] >= 4.5) & np.isfinite(r["depth"]) & (r["depth"] <= 70.0)
    t, cell = r["t"][keep], of.cell_index(r["lat"][keep], r["lon"][keep])
    m0 = 10.0 ** (1.5 * r["mag"][keep] + 9.1)
    order = np.argsort(t, kind="stable")
    t, cell, m0 = t[order], cell[order], m0[order]
    cum_t = t

    def window_sum(a: float, b: float) -> np.ndarray:
        i, j = np.searchsorted(cum_t, a, "left"), np.searchsorted(cum_t, b, "left")
        grid = np.bincount(cell[i:j], weights=m0[i:j], minlength=of.N_CELLS).reshape(of.N_LAT, of.N_LON)
        return nbr3(grid).ravel()

    h = np.empty((issue.size, of.N_CELLS))
    for k, ti in enumerate(issue.astype(np.float64)):
        recent = window_sum(ti - YEAR, ti)
        decade = window_sum(ti - 11 * YEAR, ti - YEAR)
        h[k] = np.log10((recent + FLOOR) / (decade / 10.0 + FLOOR))
    return h


def design3(z: np.ndarray, g: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.column_stack([e4.design2(z, g), h.ravel()])


def fit_s3(z, g, h, y, start):
    X = design3(z, g, h)

    def nll(theta):
        eta = X @ theta
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return -np.sum(y * eta - np.logaddexp(0.0, eta)), -(X.T @ (y - p))
    r = optimize.minimize(nll, start, jac=True, method="L-BFGS-B")
    if not r.success:
        raise SystemExit(f"S3's fit did not converge: {r.message}")
    return r.x


def predict_s3(theta, z, g, h):
    return (0.5 * (1.0 + np.tanh(0.5 * (design3(z, g, h) @ theta)))).reshape(z.shape)


def main() -> int:
    e4rep = json.loads((C.RESULTS / "gear1_e4.json").read_text(encoding="utf-8"))
    want2 = e4rep["coefficients_fitted_on_choose"]["S2"]
    want2_dev_ig = e4rep["splits"]["dev"]["models"]["S2"]["ig_per_target"]["value"]
    g_rep = json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))
    g = gs.gear1_log_map(g_rep["cells_per_year"])
    p0 = float(json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))["fit"]["p0"])
    events = C.load_events(4.5)
    data = {}
    for split in SPLITS:
        issue = C.split_issue_times(split)
        data[split] = {"issue": issue, "Y": C.binary_targets(events, issue), "C0": C.load_pred("C0", split),
                       "h": moment_ratio(issue)}
        hh = data[split]["h"]
        print(f"{split}: {issue.size} issue times; h mean {hh.mean():+.3f}, share of cell-times with h > 0.5 "
              f"{(hh > 0.5).mean():.4f}")
    ch = data["choose"]
    y = ch["Y"].ravel().astype(float)
    z = gs.logit(ch["C0"])
    th1 = gs.fit_logistic(gs.design(z, g), y)
    th2 = e4.fit_s2(z, g, y, np.array([th1[0], th1[1], th1[2], 0.0]))
    got2 = dict(zip(("a", "c", "b", "d"), map(float, th2)))
    off = max(abs(got2[k] - want2[k]) for k in want2)
    if off > COEF_TOL:
        raise SystemExit(f"control failed: S2 refitted {got2} differs from section 12.1's {want2} by {off:.2e}")
    th3 = fit_s3(z, g, ch["h"], y, np.append(th2, 0.0))
    coef = {"S2": got2, "S3": dict(zip(("a", "c", "b", "d", "e"), map(float, th3)))}
    print("fitted on CHOOSE:", json.dumps(coef))
    report = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 14 (amendment E6)", "prereg_tag": "prereg-earthquake-e6",
              "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "control": {"S2_coefficients_max_abs_diff": off}, "coefficients_fitted_on_choose": coef, "splits": {}}
    for split in SPLITS:
        d = data[split]
        zs = gs.logit(d["C0"])
        P = {"S2": e4.predict_s2(th2, zs, g), "S3": predict_s3(th3, zs, g, d["h"])}
        sc = C.SplitScorer(d["issue"], d["Y"], p0=p0)
        sc_act = C.SplitScorer(d["issue"], d["Y"], p0=p0, mask=ev.active_masks(d["issue"]))
        st = {k: sc.stats(v) for k, v in P.items()}
        res = {"role": {"choose": "fitted here", "dev": "decides", "final": "declared second read"}[split],
               "n_issue_times": int(d["issue"].size), "n_positive": int(d["Y"].sum()),
               "models": {k: sc.summary(s) for k, s in st.items()}, "paired": {"S3-S2": sc.paired(st["S3"], st["S2"])}}
        for k in P:
            res["models"][k]["auc_active_cells"] = sc_act.summary(sc_act.stats(P[k]))["auc"]
        report["splits"][split] = res
        pr = res["paired"]["S3-S2"]
        print(f"  {split:6s} S2 IG {res['models']['S2']['ig_per_target']['value']:.6f} S3 IG "
              f"{res['models']['S3']['ig_per_target']['value']:.6f} | S3-S2 dIG {pr['ig_per_target']['diff']:+.5f} "
              f"{[round(v, 5) for v in pr['ig_per_target']['ci95']]} dAUC {pr['auc']['diff']:+.5f}")
    dev_ig = report["splits"]["dev"]["models"]["S2"]["ig_per_target"]["value"]
    if abs(dev_ig - want2_dev_ig) > IG_TOL:
        raise SystemExit(f"control failed: S2's DEV IG {dev_ig!r} != section 12.1's {want2_dev_ig!r}")
    report["control"]["S2_dev_ig_abs_diff"] = abs(dev_ig - want2_dev_ig)
    ci = report["splits"]["dev"]["paired"]["S3-S2"]["ig_per_target"]["ci95"]
    report["carried_rule"] = {"S3_replaces_S2": bool(ci[0] > 0), "dev_S3_minus_S2_ig_ci95": ci}
    report["prediction_e_positive"] = bool(th3[4] > 0)
    print(f"control passed; e > 0: {th3[4] > 0}; CARRIED RULE: S3 replaces S2 = {ci[0] > 0}")
    C.write_json("gear1_e6.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
