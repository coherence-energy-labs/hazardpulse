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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("diagnose",))
    return {"diagnose": diagnose}[ap.parse_args(argv).phase]()


if __name__ == "__main__":
    raise SystemExit(main())
