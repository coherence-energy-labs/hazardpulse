#!/usr/bin/env python3
"""Amendment E5 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13): the hours after a large earthquake.

    python scripts/earthquake_program/aftershock_e5.py diagnose     # D: the served S2 after every 2021-2025 mainshock

D, the diagnostic (reported, never decides): the served S2 by its live code path (the C0 artifact from its frozen
catalog plus the simulated live fetch, then the S2 stack), issued 0.5, 6, 24 and 168 h after every shallow (<= 70 km)
M7+ mainshock of 2021-2025, read in the mainshock's own cell. The outcome is an M6+ in that cell within 30 days of the
issue. Registered prediction: at +0.5 h, S2's mean forecast is below a third of the observed rate.
Writes results/earthquake_program/aftershock_e5_diagnostic.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_artifact as ba  # noqa: E402
import common as C  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402
from hazardpulse.earthquake import served  # noqa: E402

LAGS_H = (0.5, 6.0, 24.0, 168.0)
DAY = 86400.0
D_START, D_END = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
BASE = C.REPO / "results" / "models" / "earthquake_operational_v1.json"


def km(lat, lon, la, lo):
    p1, p2 = np.radians(lat), np.radians(la)
    x = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.radians(lo - lon))
    return 6371.0 * np.arccos(np.clip(x, -1.0, 1.0))


def mainshocks(raw: dict, t0: float, t1: float, mmin: float = 7.0) -> list[int]:
    """Shallow M7+ events in [t0, t1) with no event of equal or larger magnitude within 150 km in the 30 days before."""
    t, la, lo, m, d = raw["t"], raw["lat"], raw["lon"], raw["mag"], raw["depth"]
    out = []
    for i in np.nonzero((m >= mmin) & (t >= t0) & (t < t1) & np.isfinite(d) & (d <= 70.0))[0]:
        k = np.searchsorted(t, t[i] - 30 * DAY, "left")
        prev = slice(k, i)
        if np.any((m[prev] >= m[i]) & (km(la[i], lo[i], la[prev], lo[prev]) <= 150.0)):
            continue
        out.append(int(i))
    return out


def outcome(raw: dict, cell: int, t_issue: float) -> bool:
    t = raw["t"]
    i, j = np.searchsorted(t, t_issue, "right"), np.searchsorted(t, t_issue + 30 * DAY, "right")
    sel = slice(i, j)
    big = raw["mag"][sel] >= 6.0
    return bool(np.any(of.cell_index(raw["lat"][sel][big], raw["lon"][sel][big]) == cell)) if big.any() else False


def diagnose() -> int:
    raw = C.load_raw("program_catalog.npz")
    raw25 = C.load_raw("program_catalog_m25.npz")
    base = of.load_artifact(BASE)
    stack = of.load_stack(C.REPO / served.SERVED_STACK_RELPATH, base)
    idx = mainshocks(raw, D_START.timestamp(), D_END.timestamp())
    print(f"{len(idx)} shallow M7+ mainshocks 2021-2025; served {stack.model_version}")
    rows = []
    t_start = time.time()
    for n, i in enumerate(idx):
        t_k, la, lo, m = float(raw["t"][i]), float(raw["lat"][i]), float(raw["lon"][i]), float(raw["mag"][i])
        cell = int(of.cell_index(la, lo))
        rec = {"time": dt.datetime.fromtimestamp(t_k, dt.timezone.utc).isoformat()[:19], "lat": la, "lon": lo, "mag": m,
               "cell": cell, "lags": {}}
        for lag in LAGS_H:
            ti = t_k + lag * 3600.0
            out = of.forecast_with_stack(base, stack, ba._simulated_live(raw25, ti), ti)
            rec["lags"][f"{lag:g}"] = {"p_S2": float(out["probability"][cell]), "y": outcome(raw, cell, ti)}
        rows.append(rec)
        print(f"  [{n + 1}/{len(idx)}] {rec['time']} M{m:.1f} ({la:+.1f}, {lo:+.1f}): "
              + ", ".join(f"+{k}h {v['p_S2']:.3f}/{int(v['y'])}" for k, v in rec["lags"].items())
              + f" ({time.time() - t_start:.0f} s)")
    summary = {}
    for cls, (lo_m, hi_m) in (("M7.0-7.4", (7.0, 7.5)), ("M7.5+", (7.5, 10.0)), ("all M7+", (7.0, 10.0))):
        sel = [r for r in rows if lo_m <= r["mag"] < hi_m]
        summary[cls] = {"n": len(sel)}
        for lag in LAGS_H:
            p = np.array([r["lags"][f"{lag:g}"]["p_S2"] for r in sel])
            y = np.array([r["lags"][f"{lag:g}"]["y"] for r in sel], float)
            if not len(sel):
                continue
            ll = float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
            summary[cls][f"{lag:g}"] = {"mean_p_S2": float(p.mean()), "observed": float(y.mean()),
                                         "ratio_observed_over_forecast": float(y.mean() / p.mean()) if p.mean() > 0 else None,
                                         "log_loss_S2": ll}
    allm = summary["all M7+"]["0.5"]
    prediction = bool(allm["mean_p_S2"] < allm["observed"] / 3.0)
    res = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13.2 (amendment E5, diagnostic D)",
           "prereg_tag": "prereg-earthquake-e5", "served": stack.model_version,
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "prediction_mean_below_a_third_at_0p5h": prediction, "summary": summary, "rows": rows}
    C.write_json("aftershock_e5_diagnostic.json", res)
    for cls, s in summary.items():
        print(cls, f"n={s['n']}", " | ".join(f"+{k}h S2 {v['mean_p_S2']:.3f} vs observed {v['observed']:.3f}"
                                              for k, v in s.items() if k != "n"))
    print(f"PREDICTION (S2 below a third of the observed rate at +0.5 h): {prediction}")
    return 0


FIT_END = dt.datetime(2017, 11, 30, tzinfo=dt.timezone.utc)      # a trigger's outcome window stays inside 1990-2017
FIT_START = dt.datetime(1990, 2, 1, tzinfo=dt.timezone.utc)       # 30 days of catalog before the first trigger
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _days(t: dt.datetime) -> float:
    return (t - EPOCH).total_seconds() / DAY


def _comcat_m6():
    import aftershock_base_rates as abr
    rows = abr.fetch(dt.date.today().isoformat())
    return np.array([(_days(t), la, lo, de, m) for t, la, lo, de, m, _ in rows])


def _nbr3(cell: int) -> list[int]:
    r, c = divmod(int(cell), of.N_LON)
    return [rr * of.N_LON + (cc % of.N_LON) for rr in (r - 1, r, r + 1) if 0 <= rr < of.N_LAT for cc in (c - 1, c, c + 1)]


class FitData:
    """Section 13.2's fitting cell-times: every shallow M6.5+ trigger of 1990-2017, its cell and the 8 around it, at
    lags 0.5, 6, 24 and 168 h; each with its outcome and every shallow M6+ of the 30 days before it."""

    def __init__(self, ev: np.ndarray, t0: float, t1: float, layer):
        from hazardpulse.earthquake import aftershock as A
        self.grid_log = np.log(A.D_GRID_KM)
        shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
        all_cells = of.cell_index(ev[:, 1], ev[:, 2])
        trig = shallow[(shallow[:, 4] >= 6.5) & (shallow[:, 0] >= t0) & (shallow[:, 0] <= t1)]
        y, ct_cell, ct_lag, ct_trig_mag, ct_home = [], [], [], [], []
        contrib_ct, contrib_mag, contrib_dt, contrib_F = [], [], [], []
        for tk, la, lo, _, mk in trig:
            home = int(of.cell_index(la, lo))
            for lag in LAGS_H:
                t = tk + lag / 24.0
                prior = shallow[(shallow[:, 0] < t) & (shallow[:, 0] >= t - A.WINDOW_D)]
                win = (ev[:, 0] > t) & (ev[:, 0] <= t + A.WINDOW_D) & (ev[:, 4] >= 6.0)
                for cell in _nbr3(home):
                    i = len(y)
                    y.append(bool(np.any(all_cells[win] == cell)))
                    ct_cell.append(cell), ct_lag.append(lag), ct_trig_mag.append(mk), ct_home.append(cell == home)
                    for tj, laj, loj, _, mj in prior:
                        table = layer.trigger(tj, laj, loj, mj).table
                        contrib_ct.append(i), contrib_mag.append(mj), contrib_dt.append(t - tj)
                        contrib_F.append([d.get(cell, 0.0) for d in table])
        self.y = np.array(y, float)
        self.lag, self.trig_mag, self.home = np.array(ct_lag), np.array(ct_trig_mag), np.array(ct_home)
        self.ci, self.mag = np.array(contrib_ct), np.array(contrib_mag)
        self.dt, self.F = np.array(contrib_dt), np.array(contrib_F)

    def probs(self, theta: np.ndarray) -> np.ndarray:
        from hazardpulse.earthquake import aftershock as A
        K, alpha, p, c_t, d0 = math.exp(theta[0]), theta[1], theta[2], math.exp(theta[3]), math.exp(theta[4])
        d = np.log(np.clip(d0 * 10.0 ** (0.5 * (self.mag - 7.0)), A.D_GRID_KM[0], A.D_GRID_KM[-1]))
        j = np.clip(np.searchsorted(self.grid_log, d) - 1, 0, len(self.grid_log) - 2)
        w = (d - self.grid_log[j]) / (self.grid_log[j + 1] - self.grid_log[j])
        frac = (1 - w) * self.F[np.arange(len(j)), j] + w * self.F[np.arange(len(j)), j + 1]
        dom = A.omori_integral(self.dt + A.WINDOW_D, p, c_t) - A.omori_integral(self.dt, p, c_t)
        lam = np.bincount(self.ci, weights=K * 10.0 ** (alpha * (self.mag - 7.0)) * dom * frac, minlength=len(self.y))
        return 1.0 - np.exp(-lam)

    def nll(self, theta: np.ndarray) -> float:
        p = np.clip(self.probs(theta), 1e-12, 1 - 1e-12)
        return float(-np.sum(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))


def fit() -> int:
    from scipy import optimize
    from hazardpulse.earthquake import aftershock as A
    ev = _comcat_m6()
    layer = A.AftershockLayer(A.Params(1, 1, 1.1, 0.02, 40))
    data = FitData(ev, _days(FIT_START), _days(FIT_END), layer)
    print(f"fitting cell-times {len(data.y)} ({int(data.y.sum())} with an M6+), contributions {len(data.ci)}")
    x0 = np.array([math.log(0.1), 1.0, 1.1, math.log(0.02), math.log(40.0)])
    bounds = [(math.log(1e-4), math.log(1e3)), (0.3, 2.5), (0.7, 2.0), (math.log(1e-4), math.log(2.0)), (math.log(5.0), math.log(800.0))]
    r = optimize.minimize(data.nll, x0, method="L-BFGS-B", bounds=bounds)
    if not r.success:
        raise SystemExit(f"AS fit did not converge: {r.message}")
    th = r.x
    params = A.Params(K=math.exp(th[0]), alpha=float(th[1]), p=float(th[2]), c_t=math.exp(th[3]), d0=math.exp(th[4]))
    pr = data.probs(th)
    rel = {}
    for name, sel in (("trigger cell", data.home), ("neighbour cells", ~data.home)):
        for lag in LAGS_H:
            s = sel & (data.lag == lag)
            rel[f"{name}, +{lag:g} h"] = {"n": int(s.sum()), "mean_P_AS": float(pr[s].mean()), "observed": float(data.y[s].mean())}
    for lo_m, hi_m, nm in ((6.5, 7.0, "M6.5-6.9"), (7.0, 7.5, "M7.0-7.4"), (7.5, 10.0, "M7.5+")):
        s = data.home & (data.lag == 0.5) & (data.trig_mag >= lo_m) & (data.trig_mag < hi_m)
        rel[f"trigger cell, +0.5 h, {nm}"] = {"n": int(s.sum()), "mean_P_AS": float(pr[s].mean()), "observed": float(data.y[s].mean())}
    # the Panama sequence replayed at the moments the published forecasts were written
    layer = A.AftershockLayer(params)
    shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
    home = int(of.cell_index(7.587, -80.769))
    panama = {}
    for label, when in (("2026-10-09 18:13 (published 6.0%)", dt.datetime(2026, 10, 9, 18, 13, tzinfo=dt.timezone.utc)),
                        ("2026-10-10 00:00 (published 26.3%)", dt.datetime(2026, 10, 10, 0, 0, tzinfo=dt.timezone.utc)),
                        ("2026-10-11 00:00 (published 17.3%)", dt.datetime(2026, 10, 11, 0, 0, tzinfo=dt.timezone.utc))):
        t = _days(when)
        trig = [layer.trigger(*row[[0, 1, 2, 4]]) for row in shallow[(shallow[:, 0] < t) & (shallow[:, 0] >= t - 30)]]
        panama[label] = layer.probability(home, t, trig)
    res = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13.2 (amendment E5, AS fitted on 1990-2017)",
           "prereg_tag": "prereg-earthquake-e5",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": params.as_dict(), "nll": float(r.fun), "n_cell_times": int(len(data.y)),
           "n_with_m6": int(data.y.sum()), "in_sample_reliability": rel, "panama_replay_P_AS": panama,
           "d_grid_km": [float(x) for x in A.D_GRID_KM], "n_draws": A.N_DRAWS, "seed": A.SEED}
    C.write_json("aftershock_e5_fit.json", res)
    print("params", json.dumps(params.as_dict()))
    for k, v in rel.items():
        print(f"  {k}: n {v['n']}, mean P_AS {v['mean_P_AS']:.3f}, observed {v['observed']:.3f}")
    for k, v in panama.items():
        print(f"  Panama {k}: P_AS {v:.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("diagnose", "fit"))
    return {"diagnose": diagnose, "fit": fit}[ap.parse_args(argv).phase]()


if __name__ == "__main__":
    raise SystemExit(main())
