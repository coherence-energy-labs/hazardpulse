#!/usr/bin/env python3
"""Precursor atlas, run 2 (docs/PRECURSOR_ATLAS.md, tag prereg-precursor-atlas-2): the texture of seismicity before
large earthquakes -- M4.5+ foreshock counts, b-value and its change, quiescence, moment acceleration, deeper
activity, the largest recent event, and the time since the last M6.5+ (renewal against clustering).

    python scripts/research/precursor_atlas_2.py      # after run 1's inputs; fetches ComCat M4.5+ by year if absent

Run 1's events, controls, statistic, instrument controls and rule, unchanged (imported, not copied).
Writes results/precursor_atlas/atlas_2.json.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precursor_atlas as a1  # noqa: E402

OUT = a1.ROOT / "results" / "precursor_atlas" / "atlas_2.json"
SIGNALS = ("F1", "F2", "F3", "B1", "B2", "Z1", "A1", "D1", "M1", "L1")
NAMES = {"F1": "M4.5+ within 100 km, 7 days before", "F2": "M4.5+ within 100 km, 30 days before",
         "F3": "M4.5+ within 200 km, 365 days before", "B1": "low b-value, 3 years before (200 km)",
         "B2": "b-value falling (3 y vs 3-10 y before)", "Z1": "rate, last year vs 5 years before (200 km)",
         "A1": "moment, last year vs 10 years before (200 km)", "D1": "deeper (70-300 km) activity, last year vs 5 before",
         "M1": "largest magnitude within 200 km, 365 days", "L1": "days since the last M6.5+ within 150 km (high = 'due')",
         "P+": "POSITIVE CONTROL: M6+ within 150 km, 30 days before", "P-": "NEGATIVE CONTROL: random noise"}
YEAR = 365.25


def load_m45() -> np.ndarray:
    """Every M4.5+ earthquake 1984-today: t (days since 1970), lat, lon, depth, mag."""
    rows = []
    for y in range(1984, dt.date.today().year + 1):
        p = a1.CACHE / f"m45_{y}.csv"
        if not p.exists() or p.stat().st_size < 100:
            q = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={y}-01-01"
                 f"&endtime={y + 1}-01-01&minmagnitude=4.5&orderby=time-asc&limit=20000")
            with urllib.request.urlopen(q, timeout=600) as r:
                p.write_bytes(r.read())
        n = 0
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                n += 1
                if r.get("type") != "earthquake" or not r["mag"]:
                    continue
                t = dt.datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
                rows.append((a1.days(t), float(r["latitude"]), float(r["longitude"]), float(r["depth"] or 0), float(r["mag"])))
        if n >= 20000:
            raise SystemExit(f"{p.name}: the query hit its 20,000-row limit, so the year is incomplete")
    return np.array(sorted(rows))


def b_value(mags: np.ndarray, mc: float = 4.5, dm: float = 0.1) -> float:
    """Aki-Utsu maximum-likelihood b-value for magnitudes binned at dm; NaN below 30 events."""
    m = mags[mags >= mc - 1e-9]
    if len(m) < 30:
        return math.nan
    return math.log10(math.e) / (m.mean() - (mc - dm / 2.0))


class Texture:
    def __init__(self, cat: np.ndarray, big: np.ndarray):
        self.c, self.big = cat, big

    def near(self, t, lat, lon, t0, t1, rmax, dmin=0.0, dmax=70.0):
        c = self.c
        i, j = np.searchsorted(c[:, 0], t0, "left"), np.searchsorted(c[:, 0], t1, "left")
        s = c[i:j]
        s = s[(s[:, 3] >= dmin) & (s[:, 3] <= dmax)]
        if not len(s):
            return s
        return s[a1.km_to(lat, lon, s[:, 1], s[:, 2]) <= rmax]

    def all(self, t, lat, lon, noise):
        yr = self.near(t, lat, lon, t - YEAR, t, 200.0)
        prev5 = self.near(t, lat, lon, t - 6 * YEAR, t - YEAR, 200.0)
        prev10m = self.near(t, lat, lon, t - 11 * YEAR, t - YEAR, 200.0)
        b3 = b_value(self.near(t, lat, lon, t - 3 * YEAR, t, 200.0)[:, 4])
        bold = b_value(self.near(t, lat, lon, t - 10 * YEAR, t - 3 * YEAR, 200.0)[:, 4])
        deep_yr = self.near(t, lat, lon, t - YEAR, t, 300.0, 70.0, 300.0)
        deep_prev = self.near(t, lat, lon, t - 6 * YEAR, t - YEAR, 300.0, 70.0, 300.0)
        mom = lambda s: float(np.sum(10 ** (1.5 * s[:, 4] + 9.1))) if len(s) else 0.0
        bg = self.big
        k = np.searchsorted(bg[:, 0], t, "left")
        prior = bg[max(0, k - 4000):k]
        prior = prior[(prior[:, 4] >= 6.5) & (prior[:, 3] <= 70.0)]
        near = prior[a1.km_to(lat, lon, prior[:, 1], prior[:, 2]) <= 150.0] if len(prior) else prior
        since = min(t - near[-1, 0], 10 * YEAR) if len(near) else 10 * YEAR
        return {
            "F1": float(len(self.near(t, lat, lon, t - 7, t, 100.0))),
            "F2": float(len(self.near(t, lat, lon, t - 30, t, 100.0))),
            "F3": float(len(yr)),
            "B1": -b3 if math.isfinite(b3) else math.nan,
            "B2": -(b3 - bold) if math.isfinite(b3) and math.isfinite(bold) else math.nan,
            "Z1": (len(yr) + 0.5) / (len(prev5) / 5.0 + 0.5),
            "A1": (mom(yr) + 1e15) / (mom(prev10m) / 10.0 + 1e15),
            "D1": (len(deep_yr) + 0.5) / (len(deep_prev) / 5.0 + 0.5),
            "M1": float(yr[:, 4].max()) if len(yr) else 4.4,
            "L1": float(since),
            "P+": a1.Signals.count_quakes(self._s1, t, lat, lon, 30.0, 0.0, 150.0, 6.0),
            "P-": noise,
        }


def main() -> int:
    q6 = a1.load_quakes()
    cat = load_m45()
    tex = Texture(cat, q6)
    tex._s1 = a1.Signals(q6, None, None, None, None)
    idx = a1.mainshocks(q6)
    rows = []
    for k, i in enumerate(idx):
        t, la, lo = q6[i, 0], q6[i, 1], q6[i, 2]
        rng = np.random.default_rng(a1.SEED + 9_000_000 + k)
        ev = tex.all(t, la, lo, float(rng.random()))
        cs = [tex.all(c, la, lo, float(rng.random())) for c in a1.controls(q6, i, k)]
        rec = {"t": (a1.EPOCH + dt.timedelta(days=float(t))).isoformat()[:19], "lat": float(la), "lon": float(lo),
               "mag": float(q6[i, 4]), "n_controls": len(cs), "u": {}, "top": {}}
        for s in (*SIGNALS, "P+", "P-"):
            cv = [c[s] for c in cs]
            rec["u"][s] = a1.percentile(ev[s], cv)
            good = [v for v in cv if math.isfinite(v)]
            rec["top"][s] = (bool(ev[s] > np.quantile(good, 0.8)) if math.isfinite(ev[s]) and len(good) >= 10 else None)
        rows.append(rec)
    split = a1.days(a1.SPLIT)

    def part(sel):
        return {s: a1.summary([r["u"][s] for r in sel], [r["top"][s] for r in sel]) for s in (*SIGNALS, "P+", "P-")}
    disc = [r for r in rows if a1.days(dt.datetime.fromisoformat(r["t"] + "+00:00")) < split]
    conf = [r for r in rows if a1.days(dt.datetime.fromisoformat(r["t"] + "+00:00")) >= split]
    res_all, res_d, res_c = part(rows), part(disc), part(conf)
    ctrl_ok = res_all["P+"]["mean_u"] > 0.5 and res_all["P+"]["p_two_sided"] < 0.01 and res_all["P-"]["p_two_sided"] >= 0.01
    discovered = a1.bh([res_d[s]["p_two_sided"] if res_d[s].get("n") else 1.0 for s in SIGNALS])
    verdict = {}
    for j, s in enumerate(SIGNALS):
        d, c = res_d[s], res_c[s]
        same = c.get("n") and d.get("n") and (c["mean_u"] - 0.5) * (d["mean_u"] - 0.5) > 0
        verdict[s] = {"discovered_1990_2009": j in discovered,
                      "confirmed_2010_2026": bool(j in discovered and same and c["p_one_sided_same_sign"] < 0.05)}
    panama = next((r for r in rows if r["t"].startswith("2026-10-09")), None)
    out = {"protocol": "docs/PRECURSOR_ATLAS.md (run 2)", "prereg_tag": "prereg-precursor-atlas-2",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "m45_events": len(cat),
           "n_events": len(rows), "n_discover": len(disc), "n_confirm": len(conf),
           "instrument_controls_pass": bool(ctrl_ok), "names": NAMES, "all": res_all, "discover_1990_2009": res_d,
           "confirm_2010_2026": res_c, "verdict": verdict, "panama_2026_10_09": panama}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"M4.5+ events {len(cat)}; mainshocks {len(rows)} (discover {len(disc)}, confirm {len(conf)})")
    print(f"INSTRUMENT: P+ mean u {res_all['P+']['mean_u']:.3f} p {res_all['P+']['p_two_sided']:.1e}; P- mean u "
          f"{res_all['P-']['mean_u']:.3f} p {res_all['P-']['p_two_sided']:.2f} -> {'PASS' if ctrl_ok else 'FAIL: results void'}")
    for s in SIGNALS:
        al, d, c, v = res_all[s], res_d[s], res_c[s], verdict[s]
        if not al.get("n"):
            print(f"{s} {NAMES[s]}: no values")
            continue
        print(f"{s} {NAMES[s][:46]:46s} all u {al['mean_u']:.3f} (n {al['n']}, p {al['p_two_sided']:.1e}, gain "
              f"{al['gain_top_quintile']:.2f}) | 1990-2009 {d['mean_u']:.3f} p {d['p_two_sided']:.1e} | 2010-2026 "
              f"{c['mean_u']:.3f} p {c['p_two_sided']:.1e} | "
              + ("CONFIRMED" if v["confirmed_2010_2026"] else ("discovered, not confirmed" if v["discovered_1990_2009"] else "no connection found")))
    if panama:
        print("Panama percentiles: " + ", ".join(f"{s} {panama['u'][s]:.2f}" for s in SIGNALS if math.isfinite(panama["u"][s])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
