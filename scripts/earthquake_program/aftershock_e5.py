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

    def __init__(self, ev: np.ndarray, t0: float, t1: float, layer, trig_min: float = 6.5, s2_of=None):
        """``trig_min``: the triggers' minimum magnitude (E5: 6.5; E7: 7.0). ``s2_of``: for E7, a callable giving S2's
        grid at a time in days; its value in each cell-time becomes that cell-time's offset (background)."""
        from hazardpulse.earthquake import aftershock as A
        self.grid_log = np.log(A.D_GRID_KM)
        shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
        all_cells = of.cell_index(ev[:, 1], ev[:, 2])
        trig = shallow[(shallow[:, 4] >= trig_min) & (shallow[:, 0] >= t0) & (shallow[:, 0] <= t1)]
        y, ct_cell, ct_lag, ct_trig_mag, ct_home, ct_trig, ct_t, offset = [], [], [], [], [], [], [], []
        contrib_ct, contrib_mag, contrib_dt, contrib_F = [], [], [], []
        for n_trig, (tk, la, lo, _, mk) in enumerate(trig):
            home = int(of.cell_index(la, lo))
            for lag in LAGS_H:
                t = tk + lag / 24.0
                prior = shallow[(shallow[:, 0] < t) & (shallow[:, 0] >= t - A.WINDOW_D)]
                win = (ev[:, 0] > t) & (ev[:, 0] <= t + A.WINDOW_D) & (ev[:, 4] >= 6.0)
                s2 = s2_of(t) if s2_of is not None else None
                for cell in _nbr3(home):
                    i = len(y)
                    y.append(bool(np.any(all_cells[win] == cell)))
                    ct_cell.append(cell), ct_lag.append(lag), ct_trig_mag.append(mk), ct_home.append(cell == home)
                    ct_trig.append(n_trig), ct_t.append(t), offset.append(float(s2[cell]) if s2 is not None else 0.0)
                    for tj, laj, loj, _, mj in prior:
                        table = layer.trigger(tj, laj, loj, mj).table
                        contrib_ct.append(i), contrib_mag.append(mj), contrib_dt.append(t - tj)
                        contrib_F.append([d.get(cell, 0.0) for d in table])
        self.y = np.array(y, float)
        self.lag, self.trig_mag, self.home = np.array(ct_lag), np.array(ct_trig_mag), np.array(ct_home)
        self.trig, self.t, self.offset = np.array(ct_trig), np.array(ct_t), np.array(offset)
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
        return 1.0 - (1.0 - self.offset) * np.exp(-lam)       # E5: no offset (0); E7: S2 as the background

    def nll(self, theta: np.ndarray, sel: np.ndarray | None = None) -> float:
        p = np.clip(self.probs(theta), 1e-12, 1 - 1e-12)
        ll = self.y * np.log(p) + (1 - self.y) * np.log(1 - p)
        return float(-np.sum(ll if sel is None else ll[sel]))


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


TEST_START = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
N_BOOT, BOOT_SEED = 10_000, 20261011


def _snapshot_m6() -> np.ndarray:
    import aftershock_base_rates as abr
    return np.array([(_days(t), la, lo, de, m) for t, la, lo, de, m, _ in abr.load_snapshot()])


def test() -> int:
    """Section 13.2's test, held out: max(S2, AS) against S2 after every shallow M7+ mainshock of 2018 onward (their
    cells and the 8 around each, at 0.5, 6, 24 and 168 h), and non-inferiority over all DEV cell-times."""
    import aftershock_base_rates as abr
    import gear1_e4 as e4
    import gear1_stack as gs
    from hazardpulse.earthquake import aftershock as A
    # control 1: the base rates reproduce from the committed catalog snapshot
    committed = json.loads((C.RESULTS / "aftershock_base_rates.json").read_text(encoding="utf-8"))
    again = abr.compute(abr.load_snapshot())
    if again["classes"] != committed["classes"]:
        raise SystemExit("control failed: the base rates do not reproduce from the committed snapshot")
    # control 2: the served S2's live path reproduces its evaluated grid at three FINAL issue times
    base = of.load_artifact(BASE)
    stack = of.load_stack(C.REPO / served.SERVED_STACK_RELPATH, base)
    raw25 = C.load_raw("program_catalog_m25.npz")
    e4rep = json.loads((C.RESULTS / "gear1_e4.json").read_text(encoding="utf-8"))
    co2 = e4rep["coefficients_fitted_on_choose"]["S2"]
    th2 = np.array([co2["a"], co2["c"], co2["b"], co2["d"]])
    g = gs.gear1_log_map(json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))["cells_per_year"])
    fin = C.split_issue_times("final")
    want = e4.predict_s2(th2, gs.logit(C.load_pred("C0", "final")), g)
    worst = max(float(np.max(np.abs(of.forecast_with_stack(base, stack, ba._simulated_live(raw25, float(fin[k])),
                                                                  float(fin[k]))["probability"] - want[k])))
                for k in (0, len(fin) // 2, len(fin) - 1))
    if worst > 1e-9:
        raise SystemExit(f"control failed: S2's live path is {worst:.1e} from its evaluated grid")
    fitp = json.loads((C.RESULTS / "aftershock_e5_fit.json").read_text(encoding="utf-8"))["params"]
    layer = A.AftershockLayer(A.Params(**fitp))
    ev = _snapshot_m6()
    shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
    all_cells = of.cell_index(ev[:, 1], ev[:, 2])
    snap = abr.load_snapshot()
    main_idx = [i for i in abr.mainshocks(snap, 7.0, 10.0, snap[-1][0]) if snap[i][0] >= TEST_START]
    print(f"controls passed (base rates exact; S2 live parity {worst:.1e}); {len(main_idx)} held-out mainshocks from 2018")
    rows = []
    for n, i in enumerate(main_idx):
        t_k = _days(snap[i][0])
        la, lo, m = snap[i][1], snap[i][2], snap[i][4]
        home = int(of.cell_index(la, lo))
        for lag in LAGS_H:
            t = t_k + lag / 24.0
            s2 = of.forecast_with_stack(base, stack, ba._simulated_live(raw25, t * DAY), t * DAY)["probability"]
            trig = [layer.trigger(*row[[0, 1, 2, 4]]) for row in shallow[(shallow[:, 0] < t) & (shallow[:, 0] >= t - A.WINDOW_D)]]
            win = (ev[:, 0] > t) & (ev[:, 0] <= t + A.WINDOW_D) & (ev[:, 4] >= 6.0)
            for cell in _nbr3(home):
                p_as = layer.probability(cell, t, trig)
                rows.append({"mainshock": n, "mag": m, "lag": lag, "home": cell == home, "p_S2": float(s2[cell]),
                             "p_AS": p_as, "y": bool(np.any(all_cells[win] == cell))})
        first = next(r for r in rows if r["mainshock"] == n and r["home"] and r["lag"] == 0.5)
        print(f"  [{n + 1}/{len(main_idx)}] {snap[i][0].isoformat()[:16]} M{m:.1f}: home cell at +0.5 h "
              f"S2 {first['p_S2']:.3f}, AS {first['p_AS']:.3f}, outcome {int(first['y'])}")
    y = np.array([r["y"] for r in rows], float)
    p_s2 = np.clip(np.array([r["p_S2"] for r in rows]), 1e-12, 1 - 1e-12)
    p_mx = np.clip(np.maximum(p_s2, np.array([r["p_AS"] for r in rows])), 1e-12, 1 - 1e-12)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(p_mx) - ll(p_s2)
    clusters = np.array([r["mainshock"] for r in rows])
    per = np.array([d[clusters == k].sum() for k in range(len(main_idx))])
    cnt = np.array([(clusters == k).sum() for k in range(len(main_idx))])
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(main_idx), len(main_idx))
        boot.append(per[pick].sum() / cnt[pick].sum())
    lo95, hi95 = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    part1 = hi95 < 0
    rel = {}
    for lag in LAGS_H:
        for nm, sel in (("home", np.array([r["home"] for r in rows])), ("neighbours", ~np.array([r["home"] for r in rows]))):
            s = sel & (np.array([r["lag"] for r in rows]) == lag)
            rel[f"{nm} +{lag:g} h"] = {"n": int(s.sum()), "observed": float(y[s].mean()), "mean_S2": float(p_s2[s].mean()),
                                        "mean_S2+AS": float(p_mx[s].mean())}
    # part 2: non-inferiority over every DEV cell-time (weekly issue times, month blocks)
    dev = C.split_issue_times("dev")
    Y = C.binary_targets(C.load_events(4.5), dev)
    S2dev = e4.predict_s2(th2, gs.logit(C.load_pred("C0", "dev")), g)
    MX = S2dev.copy()
    for k, ti in enumerate(dev.astype(float) / DAY):
        trig = [layer.trigger(*row[[0, 1, 2, 4]]) for row in shallow[(shallow[:, 0] < ti) & (shallow[:, 0] >= ti - A.WINDOW_D)]]
        if trig:
            MX[k] = np.maximum(MX[k], layer.grid(ti, trig))
    p0 = float(json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))["fit"]["p0"])
    sc = C.SplitScorer(dev, Y, p0=p0)
    pair = sc.paired(sc.stats(MX), sc.stats(S2dev))["ig_per_target"]
    part2 = pair["ci95"][0] >= -0.005
    carried = bool(part1 and part2)
    res = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13.2 (amendment E5, the test)",
           "prereg_tag": "prereg-earthquake-e5", "served": stack.model_version, "as_params": fitp,
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "controls": {"base_rates_reproduce": True, "s2_live_parity_max_abs": worst},
           "n_mainshocks": len(main_idx), "n_cell_times": len(rows),
           "part1_logloss_S2+AS_minus_S2": {"mean": float(d.mean()), "ci95_cluster_by_mainshock": [lo95, hi95], "passes": part1},
           "part2_dev_ig_S2+AS_minus_S2": {"diff": pair["diff"], "ci95": pair["ci95"], "margin": -0.005, "passes": part2},
           "carried": carried, "reliability_held_out": rel, "rows": rows}
    C.write_json("aftershock_e5_test.json", res)
    print(f"PART 1 log loss (S2+AS) - S2: {d.mean():+.4f} [{lo95:+.4f}, {hi95:+.4f}] -> {'passes' if part1 else 'fails'}")
    print(f"PART 2 DEV IG (S2+AS) - S2: {pair['diff']:+.5f} {[round(x, 5) for x in pair['ci95']]} -> {'passes' if part2 else 'fails'}")
    for k, v in rel.items():
        print(f"  {k}: n {v['n']}, observed {v['observed']:.3f}, S2 {v['mean_S2']:.3f}, S2+AS {v['mean_S2+AS']:.3f}")
    print(f"CARRIED: {carried}")
    return 0


E7_START = dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc)        # section 15.1: S2's inputs are valid from 2005
E7_SPLIT = dt.datetime(2012, 1, 1, tzinfo=dt.timezone.utc)


def fit7() -> int:
    """Amendment E7 (section 15): Lambda fitted as an increment over S2, P = 1 - (1 - P_S2) exp(-Lambda), on the cell-times
    of every shallow M7+ of 2005-2017, S2 by its live path. Reported, never deciding: in-sample reliability against S2,
    and a time split (fit 2005-2011, read 2012-2017)."""
    from scipy import optimize
    from hazardpulse.earthquake import aftershock as A
    base = of.load_artifact(BASE)
    stack = of.load_stack(C.REPO / served.SERVED_STACK_RELPATH, base)
    raw25 = C.load_raw("program_catalog_m25.npz")
    grids: dict[float, np.ndarray] = {}

    def s2_of(t_days: float) -> np.ndarray:
        if t_days not in grids:
            ts = t_days * DAY
            grids[t_days] = of.forecast_with_stack(base, stack, ba._simulated_live(raw25, ts), ts)["probability"]
        return grids[t_days]

    ev = _snapshot_m6()
    e5 = json.loads((C.RESULTS / "aftershock_e5_fit.json").read_text(encoding="utf-8"))["params"]
    layer = A.AftershockLayer(A.Params(**e5))
    t_start = time.time()
    data = FitData(ev, _days(E7_START), _days(FIT_END), layer, trig_min=7.0, s2_of=s2_of)
    print(f"cell-times {len(data.y)} ({int(data.y.sum())} with an M6+), triggers {data.trig.max() + 1}, "
          f"S2 forecasts {len(grids)} ({time.time() - t_start:.0f} s)")
    x0 = np.array([math.log(e5["K"]), e5["alpha"], e5["p"], math.log(e5["c_t"]), math.log(e5["d0"])])
    bounds = [(math.log(1e-4), math.log(1e3)), (0.3, 2.5), (0.7, 2.0), (math.log(1e-4), math.log(2.0)), (math.log(5.0), math.log(800.0))]

    def fit_on(sel):
        r = optimize.minimize(lambda th: data.nll(th, sel), x0, method="L-BFGS-B", bounds=bounds)
        if not r.success:
            raise SystemExit(f"E7 fit did not converge: {r.message}")
        return r.x
    th = fit_on(None)
    params = A.Params(K=math.exp(th[0]), alpha=float(th[1]), p=float(th[2]), c_t=math.exp(th[3]), d0=math.exp(th[4]))
    s2 = np.clip(data.offset, 1e-12, 1 - 1e-12)
    pr = np.clip(data.probs(th), 1e-12, 1 - 1e-12)
    rel = {}
    for name, sel in (("mainshock cell", data.home), ("neighbour cells", ~data.home)):
        for lag in LAGS_H:
            s = sel & (data.lag == lag)
            rel[f"{name}, +{lag:g} h"] = {"n": int(s.sum()), "observed": float(data.y[s].mean()),
                                          "S2": float(s2[s].mean()), "E7": float(pr[s].mean())}
    ll = lambda p, s: -(data.y[s] * np.log(p[s]) + (1 - data.y[s]) * np.log(1 - p[s]))
    # the time split: fit on triggers before 2012, read those from 2012
    early = data.t < _days(E7_SPLIT)
    th_e = fit_on(early)
    pe = np.clip(data.probs(th_e), 1e-12, 1 - 1e-12)
    late = ~early
    d = ll(pe, late) - ll(s2, late)
    trig_late = data.trig[late]
    ks = np.unique(trig_late)
    per = np.array([d[trig_late == k].sum() for k in ks])
    cnt = np.array([(trig_late == k).sum() for k in ks])
    rng = np.random.default_rng(BOOT_SEED)
    boot = [per[p].sum() / cnt[p].sum() for p in (rng.integers(0, len(ks), len(ks)) for _ in range(N_BOOT))]
    split = {"fit_on": "2005-2011", "read_on": "2012-2017", "n_triggers_read": int(len(ks)),
             "logloss_E7_minus_S2": float(d.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
             "reliability_read": {f"{nm} +{lag:g} h": {"observed": float(data.y[late & s & (data.lag == lag)].mean()),
                                                       "S2": float(s2[late & s & (data.lag == lag)].mean()),
                                                       "E7": float(pe[late & s & (data.lag == lag)].mean())}
                                  for nm, s in (("mainshock cell", data.home), ("neighbours", ~data.home)) for lag in LAGS_H}}
    res = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 15 (amendment E7, the increment over S2)",
           "prereg_tag": "prereg-earthquake-e7", "served": stack.model_version,
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": params.as_dict(), "n_cell_times": int(len(data.y)), "n_with_m6": int(data.y.sum()),
           "in_sample_reliability": rel, "time_split_reported": split}
    C.write_json("aftershock_e7_fit.json", res)
    print("E7 params", json.dumps(params.as_dict()))
    for k, v in rel.items():
        print(f"  {k}: n {v['n']}, observed {v['observed']:.3f}, S2 {v['S2']:.3f}, E7 {v['E7']:.3f}")
    print(f"time split (reported): log loss E7 - S2 on 2012-2017 {split['logloss_E7_minus_S2']:+.4f} "
          f"{[round(x, 4) for x in split['ci95']]} ({split['n_triggers_read']} triggers)")
    return 0


def _e7_layer():
    from hazardpulse.earthquake import aftershock as A
    p = json.loads((C.RESULTS / "aftershock_e7_fit.json").read_text(encoding="utf-8"))["params"]
    return A.AftershockLayer(A.Params(**p)), p


def read7() -> int:
    """Section 15.3, part 1: E7 on exactly E5's held-out cell-times. S2's values are E5's recorded ones; E7 adds the
    increment from the committed snapshot. Controls (each stops the run): the cell-times rebuild in E5's order, with
    E5's mainshock, lag and home flag at every row, and each outcome recomputed from the snapshot equals E5's."""
    import aftershock_base_rates as abr
    from hazardpulse.earthquake import aftershock as A
    e5 = json.loads((C.RESULTS / "aftershock_e5_test.json").read_text(encoding="utf-8"))
    rows = e5["rows"]
    layer, params = _e7_layer()
    ev = _snapshot_m6()
    shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
    all_cells = of.cell_index(ev[:, 1], ev[:, 2])
    snap = abr.load_snapshot()
    main_idx = [i for i in abr.mainshocks(snap, 7.0, 10.0, snap[-1][0]) if snap[i][0] >= TEST_START]
    if len(main_idx) != e5["n_mainshocks"]:
        raise SystemExit(f"control failed: {len(main_idx)} mainshocks rebuilt, E5 tested {e5['n_mainshocks']}")
    k, p7, cells_of = 0, [], []
    for n, i in enumerate(main_idx):
        t_k = _days(snap[i][0])
        home = int(of.cell_index(snap[i][1], snap[i][2]))
        for lag in LAGS_H:
            t = t_k + lag / 24.0
            trig = [layer.trigger(*row[[0, 1, 2, 4]]) for row in shallow[(shallow[:, 0] < t) & (shallow[:, 0] >= t - A.WINDOW_D)]]
            win = (ev[:, 0] > t) & (ev[:, 0] <= t + A.WINDOW_D) & (ev[:, 4] >= 6.0)
            for cell in _nbr3(home):
                r = rows[k]
                if (r["mainshock"], r["lag"], r["home"]) != (n, lag, cell == home):
                    raise SystemExit(f"control failed: row {k} is {r['mainshock'], r['lag'], r['home']}, rebuilt "
                                     f"{n, lag, cell == home}")
                if bool(np.any(all_cells[win] == cell)) != bool(r["y"]):
                    raise SystemExit(f"control failed: row {k}'s outcome does not recompute")
                lam = layer.rate(cell, t, trig)
                p7.append(1.0 - (1.0 - r["p_S2"]) * math.exp(-lam))
                cells_of.append(cell)
                k += 1
    if k != len(rows):
        raise SystemExit(f"control failed: {k} cell-times rebuilt, E5 recorded {len(rows)}")
    y = np.array([r["y"] for r in rows], float)
    s2 = np.clip(np.array([r["p_S2"] for r in rows]), 1e-12, 1 - 1e-12)
    e7 = np.clip(np.array(p7), 1e-12, 1 - 1e-12)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(e7) - ll(s2)
    cl = np.array([r["mainshock"] for r in rows])
    K = int(cl.max()) + 1
    per = np.array([d[cl == j].sum() for j in range(K)])
    cnt = np.array([(cl == j).sum() for j in range(K)])
    rng = np.random.default_rng(BOOT_SEED)
    boot = [per[pk].sum() / cnt[pk].sum() for pk in (rng.integers(0, K, K) for _ in range(N_BOOT))]
    lo95, hi95 = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    home = np.array([r["home"] for r in rows])
    lagv = np.array([r["lag"] for r in rows])
    rel = {f"{nm} +{lag:g} h": {"n": int((sel & (lagv == lag)).sum()), "observed": float(y[sel & (lagv == lag)].mean()),
                                "S2": float(s2[sel & (lagv == lag)].mean()), "E7": float(e7[sel & (lagv == lag)].mean())}
           for nm, sel in (("mainshock cell", home), ("neighbour cells", ~home)) for lag in LAGS_H}
    part1 = hi95 < 0
    res = {"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 15.3 (amendment E7b, part 1)",
           "prereg_tag": "prereg-earthquake-e7b", "params": params,
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "controls": {"rows_rebuilt_in_order": True, "outcomes_recompute": True},
           "n_mainshocks": K, "n_cell_times": len(rows),
           "logloss_E7_minus_S2": {"mean": float(d.mean()), "ci95_cluster_by_mainshock": [lo95, hi95], "passes": part1},
           "by_cells": {nm: {"E7": float(ll(e7)[sel].mean()), "S2": float(ll(s2)[sel].mean())}
                        for nm, sel in (("mainshock", home), ("neighbours", ~home))},
           "reliability": rel, "p_E7": [float(x) for x in p7]}
    C.write_json("aftershock_e7_heldout.json", res)
    print(f"controls passed (3,120 rows rebuilt in order; outcomes recompute); {K} mainshocks")
    print(f"PART 1 log loss E7 - S2: {d.mean():+.4f} [{lo95:+.4f}, {hi95:+.4f}] -> {'passes' if part1 else 'fails'}")
    for kk, v in rel.items():
        print(f"  {kk}: n {v['n']}, observed {v['observed']:.3f}, S2 {v['S2']:.3f}, E7 {v['E7']:.3f}")
    return 0


def dev7() -> int:
    """Section 15.3, part 2: E7 against S2 over every DEV cell-time at the weekly issue times (month-block IG), and the
    same on FINAL as a second read. S2 from C0's cached forecasts and section 12.1's coefficients."""
    import gear1_e4 as e4
    import gear1_stack as gs
    from hazardpulse.earthquake import aftershock as A
    layer, params = _e7_layer()
    ev = _snapshot_m6()
    shallow = ev[(ev[:, 3] <= A.MAX_DEPTH_KM) & (ev[:, 4] >= A.MIN_MAG)]
    co2 = json.loads((C.RESULTS / "gear1_e4.json").read_text(encoding="utf-8"))["coefficients_fitted_on_choose"]["S2"]
    th2 = np.array([co2["a"], co2["c"], co2["b"], co2["d"]])
    g = gs.gear1_log_map(json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))["cells_per_year"])
    p0 = float(json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))["fit"]["p0"])
    events = C.load_events(4.5)
    out = {}
    for split in ("dev", "final"):
        issue = C.split_issue_times(split)
        Y = C.binary_targets(events, issue)
        S2 = e4.predict_s2(th2, gs.logit(C.load_pred("C0", split)), g)
        E7 = S2.copy()
        changed = 0
        for k, ti in enumerate(issue.astype(float) / DAY):
            trig = [layer.trigger(*row[[0, 1, 2, 4]]) for row in shallow[(shallow[:, 0] < ti) & (shallow[:, 0] >= ti - A.WINDOW_D)]]
            if trig:
                for c, p in layer.increment(ti, trig, S2[k]).items():
                    E7[k, c] = p
                    changed += 1
        sc = C.SplitScorer(issue, Y, p0=p0)
        st2, st7 = sc.stats(S2), sc.stats(E7)
        pair = sc.paired(st7, st2)
        out[split] = {"role": "decides (part 2)" if split == "dev" else "declared second read", "cells_changed": changed,
                      "S2": sc.summary(st2), "E7": sc.summary(st7), "E7-S2": pair}
        ig = pair["ig_per_target"]
        print(f"{split}: IG E7 - S2 {ig['diff']:+.5f} {[round(x, 5) for x in ig['ci95']]}; cells changed {changed}")
    lo = out["dev"]["E7-S2"]["ig_per_target"]["ci95"][0]
    out.update({"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 15.3 (amendment E7b, part 2)",
                "prereg_tag": "prereg-earthquake-e7b", "params": params, "margin": -0.005, "part2_passes": bool(lo >= -0.005),
                "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    C.write_json("aftershock_e7_dev.json", out)
    print(f"PART 2 (DEV non-inferiority at -0.005): {'passes' if lo >= -0.005 else 'fails'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("diagnose", "fit", "test", "fit7", "read7", "dev7"))
    return {"diagnose": diagnose, "fit": fit, "test": test, "fit7": fit7, "read7": read7,
            "dev7": dev7}[ap.parse_args(argv).phase]()


if __name__ == "__main__":
    raise SystemExit(main())
