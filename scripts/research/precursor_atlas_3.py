#!/usr/bin/env python3
"""Precursor atlas, run 3 (docs/PRECURSOR_ATLAS.md, tag prereg-precursor-atlas-3): heat radiated from the ground
(NOAA interpolated daily outgoing longwave radiation) over the epicentre before large earthquakes.

    python scripts/research/precursor_atlas_3.py      # fetches olr.day.mean.nc (~350 MB) into .cache/precursor/

Run 2b's design (past-only controls, imported), two signals: O1 the largest daily OLR in the 7 days before minus the
median of the 30 days before that; O2 the 7-day mean minus the same calendar days' mean over the 5 years before.
Writes results/precursor_atlas/atlas_3.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precursor_atlas as a1  # noqa: E402

a1.PAST_ONLY = True                                   # run 2b's controls
OUT = a1.ROOT / "results" / "precursor_atlas" / "atlas_3.json"
URL = "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc"
SIGNALS = ("O1", "O2")
NAMES = {"O1": "OLR spike: 7-day max minus the prior 30-day median (W/m2)",
         "O2": "OLR 7-day mean minus the same days of the 5 years before (W/m2)",
         "P+": "POSITIVE CONTROL: M6+ within 150 km, 30 days before", "P-": "NEGATIVE CONTROL: random noise"}


class OLR:
    def __init__(self, path: Path):
        import h5py
        f = h5py.File(path, "r")
        v = f["olr"]
        self.scale = float(np.ravel(v.attrs.get("scale_factor", [1.0]))[0])
        self.offset = float(np.ravel(v.attrs.get("add_offset", [0.0]))[0])
        fill = v.attrs.get("_FillValue", v.attrs.get("missing_value"))
        self.fill = None if fill is None else float(np.ravel(fill)[0])
        self.var = v
        self.lat = np.asarray(f["lat"][:], float)
        self.lon = np.asarray(f["lon"][:], float) % 360.0
        units = f["time"].attrs.get("units")
        units = units.decode() if isinstance(units, bytes) else str(units)
        # "hours since 1800-01-01 00:00:0.0": the epoch's date (its clock part is midnight)
        base = dt.datetime.fromisoformat(units.split("since")[1].strip().split()[0]).replace(tzinfo=dt.timezone.utc)
        if units.split("since")[1].strip().split()[1:2] not in ([], ["00:00:0.0"], ["00:00:00"], ["00:00"]):
            raise SystemExit(f"OLR time units {units!r}: an epoch not at midnight is not handled")
        hours = "hour" in units.split("since")[0]
        t = np.asarray(f["time"][:], float)
        self.t = np.array([a1.days(base + dt.timedelta(hours=x) if hours else base + dt.timedelta(days=x)) for x in t])
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def series(self, lat: float, lon: float) -> np.ndarray:
        i, j = int(np.argmin(np.abs(self.lat - lat))), int(np.argmin(np.abs(((self.lon - lon % 360.0) + 180) % 360 - 180)))
        if (i, j) not in self.cache:
            raw = np.asarray(self.var[:, i, j], float)
            if self.fill is not None:
                raw[raw == self.fill] = np.nan
            s = raw * self.scale + self.offset
            s[(s < 50) | (s > 450)] = np.nan          # physical range of daily OLR, W/m2
            self.cache[(i, j)] = s
        return self.cache[(i, j)]

    def window(self, s: np.ndarray, t0: float, t1: float) -> np.ndarray:
        a, b = np.searchsorted(self.t, t0, "left"), np.searchsorted(self.t, t1, "left")
        return s[a:b]

    def all(self, t, lat, lon):
        if t - 1 > self.t[-1] or t - 6 * 365.25 < self.t[0]:
            return {"O1": math.nan, "O2": math.nan}
        s = self.series(lat, lon)
        last7, base30 = self.window(s, t - 7, t), self.window(s, t - 37, t - 7)
        o1 = (np.nanmax(last7) - np.nanmedian(base30)) if np.isfinite(last7).sum() >= 5 and np.isfinite(base30).sum() >= 20 else math.nan
        past = [np.nanmean(self.window(s, t - 7 - 365.25 * k, t - 365.25 * k)) for k in range(1, 6)]
        past = [p for p in past if math.isfinite(p)]
        m7 = np.nanmean(last7) if np.isfinite(last7).sum() >= 5 else math.nan
        o2 = (m7 - float(np.mean(past))) if len(past) >= 4 and math.isfinite(m7) else math.nan
        return {"O1": float(o1), "O2": float(o2)}


def main() -> int:
    path = a1.CACHE / "olr.day.mean.nc"
    if not path.exists() or path.stat().st_size < 100_000_000:
        a1.CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=1800) as r:
            path.write_bytes(r.read())
    olr = OLR(path)
    q6 = a1.load_quakes()
    s1 = a1.Signals(q6, None, None, None, None)
    idx = a1.mainshocks(q6)
    rows = []
    for k, i in enumerate(idx):
        t, la, lo = q6[i, 0], q6[i, 1], q6[i, 2]
        rng = np.random.default_rng(a1.SEED + 11_000_000 + k)

        def sig(tt):
            out = olr.all(tt, la, lo)
            out["P+"] = s1.count_quakes(tt, la, lo, 30.0, 0.0, 150.0, 6.0)
            out["P-"] = float(rng.random())
            return out
        ev = sig(t)
        cs = [sig(c) for c in a1.controls(q6, i, k)]
        rec = {"t": (a1.EPOCH + dt.timedelta(days=float(t))).isoformat()[:19], "lat": float(la), "lon": float(lo),
               "mag": float(q6[i, 4]), "n_controls": len(cs), "u": {}, "top": {}}
        for s in (*SIGNALS, "P+", "P-"):
            cv = [c[s] for c in cs]
            rec["u"][s] = a1.percentile(ev[s], cv)
            good = [v for v in cv if math.isfinite(v)]
            rec["top"][s] = (bool(ev[s] > np.quantile(good, 0.8)) if math.isfinite(ev[s]) and len(good) >= 10 else None)
        rows.append(rec)
    split = a1.days(a1.SPLIT)
    part = lambda sel: {s: a1.summary([r["u"][s] for r in sel], [r["top"][s] for r in sel]) for s in (*SIGNALS, "P+", "P-")}
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
                      "confirmed_2010_on": bool(j in discovered and same and c["p_one_sided_same_sign"] < 0.05)}
    out = {"protocol": "docs/PRECURSOR_ATLAS.md (run 3)", "prereg_tag": "prereg-precursor-atlas-3",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "olr_last_day": (a1.EPOCH + dt.timedelta(days=float(olr.t[-1]))).isoformat()[:10],
           "n_events": len(rows), "instrument_controls_pass": bool(ctrl_ok), "names": NAMES, "all": res_all,
           "discover_1990_2009": res_d, "confirm_2010_on": res_c, "verdict": verdict}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"OLR to {out['olr_last_day']}; events {len(rows)}; INSTRUMENT P+ {res_all['P+']['mean_u']:.3f} "
          f"(p {res_all['P+']['p_two_sided']:.1e}), P- {res_all['P-']['mean_u']:.3f} (p {res_all['P-']['p_two_sided']:.2f}) -> "
          f"{'PASS' if ctrl_ok else 'FAIL'}")
    for s in SIGNALS:
        al, d, c, v = res_all[s], res_d[s], res_c[s], verdict[s]
        print(f"{s} {NAMES[s]}: all u {al['mean_u']:.3f} (n {al['n']}, p {al['p_two_sided']:.2e}, gain "
              f"{al['gain_top_quintile']:.2f}) | 1990-2009 {d['mean_u']:.3f} p {d['p_two_sided']:.2e} | 2010-on "
              f"{c['mean_u']:.3f} p {c['p_two_sided']:.2e} | "
              + ("CONFIRMED" if v["confirmed_2010_on"] else ("discovered, not confirmed" if v["discovered_1990_2009"] else "no connection found")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
