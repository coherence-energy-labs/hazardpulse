#!/usr/bin/env python3
"""Precursor atlas, run 4 (docs/PRECURSOR_ATLAS.md, tag prereg-precursor-atlas-4): does the ground move before large
earthquakes? Nevada Geodetic Laboratory daily GPS positions at the up to 5 nearest stations within 150 km.

    python scripts/research/precursor_atlas_4.py      # downloads ~1 GB of station series: run it on a runner

At time T a station's index is the horizontal length of its mean residual over the last W days, after steps are
removed and a trend plus annual and semiannual terms fitted on [T - 3 y, T - 90 d], divided by the standard deviation
of the same statistic over sliding W-day windows in the fit span. N1 / N2: the largest index over the stations for
W = 60 / 14 days; N3: the median for W = 60 (two or more stations). Run 2b's past-only controls; BH on events before
2016, confirmed on 2016 onward. Writes results/precursor_atlas/atlas_4.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precursor_atlas as a1  # noqa: E402

a1.PAST_ONLY = True
OUT = a1.ROOT / "results" / "precursor_atlas" / "atlas_4.json"
NGL = "https://geodesy.unr.edu"
LLH = f"{NGL}/NGLStationPages/llh.out"
SERIES = NGL + "/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3"
SIGNALS = ("N1", "N2", "N3")
NAMES = {"N1": "GPS: largest 60-day transient index, nearest 5 stations within 150 km",
         "N2": "GPS: largest 14-day transient index",
         "N3": "GPS: median 60-day transient index (2+ stations)",
         "P+": "POSITIVE CONTROL: M6+ within 150 km, 30 days before", "P-": "NEGATIVE CONTROL: random noise"}
START = dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc)
SPLIT = dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc)
YEAR = 365.25
GPS_DIR = a1.CACHE / "ngl"


def fetch(url: str, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                path.write_bytes(r.read())
            return path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                path.write_bytes(b"")                 # remembered as absent
                return None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    return None


def stations() -> np.ndarray:
    p = fetch(LLH, GPS_DIR / "llh.out")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 3:
            rows.append((f[0], float(f[1]), ((float(f[2]) + 180.0) % 360.0) - 180.0))
    return np.array(rows, dtype=object)


class Series:
    """One station's daily east/north positions (m) by day since 1970 (MJD - 40587)."""

    def __init__(self, path: Path):
        t, e, n = [], [], []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            f = line.split()
            if len(f) < 11 or not f[3].replace(".", "").isdigit():
                continue
            t.append(float(f[3]) - 40587.0)
            e.append(float(f[7]) + float(f[8]))
            n.append(float(f[9]) + float(f[10]))
        self.t, self.e, self.n = np.array(t), np.array(e) * 1000.0, np.array(n) * 1000.0     # mm

    def index(self, T: float, W: float) -> float:
        """The transient index at T for a W-day window; NaN when the station's record cannot support it."""
        m = (self.t >= T - 3 * YEAR) & (self.t < T)
        if m.sum() < 0.8 * 3 * YEAR:
            return math.nan
        t, e, n = self.t[m], self.e[m].copy(), self.n[m].copy()
        e, n = destep(t, e), destep(t, n)
        fit = t < T - 90.0
        ty = t / YEAR
        X = np.column_stack([np.ones_like(ty), ty - ty[-1], np.sin(2 * np.pi * ty), np.cos(2 * np.pi * ty),
                             np.sin(4 * np.pi * ty), np.cos(4 * np.pi * ty)])
        re = e - X @ np.linalg.lstsq(X[fit], e[fit], rcond=None)[0]
        rn = n - X @ np.linalg.lstsq(X[fit], n[fit], rcond=None)[0]
        last = t >= T - W
        if last.sum() < 0.6 * W:
            return math.nan
        L = math.hypot(re[last].mean(), rn[last].mean())
        hist = []
        for s in np.arange(t[0], T - 90.0 - W, 7.0):
            w = (t >= s) & (t < s + W)
            if w.sum() >= 0.6 * W:
                hist.append(math.hypot(re[w].mean(), rn[w].mean()))
        if len(hist) < 20 or np.std(hist) <= 0:
            return math.nan
        return L / float(np.std(hist))


def destep(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove offsets: a day-to-day change, after the series' median rate, larger than 5 times the median absolute
    change (and at least 3 mm) is taken as a step and replaced by the median-rate change."""
    if len(x) < 3:
        return x
    dt_ = np.diff(t)
    d = np.diff(x)
    v = float(np.median(d / np.maximum(dt_, 1.0)))
    r = d - v * dt_
    mad = float(np.median(np.abs(r - np.median(r)))) or 1e-9
    jump = np.abs(r) > max(5.0 * mad, 3.0)
    d = np.where(jump, v * dt_, d)
    return np.concatenate([[x[0]], x[0] + np.cumsum(d)])


def main() -> int:
    q6 = a1.load_quakes()
    s1 = a1.Signals(q6, None, None, None, None)
    st = stations()
    slat, slon = st[:, 1].astype(float), st[:, 2].astype(float)
    idx = [i for i in a1.mainshocks(q6) if q6[i, 0] >= a1.days(START)]
    rows, used = [], {}
    cache: dict[str, Series | None] = {}

    def series(sta: str) -> Series | None:
        if sta not in cache:
            p = fetch(SERIES.format(sta=sta), GPS_DIR / f"{sta}.tenv3")
            cache[sta] = Series(p) if p is not None and p.stat().st_size > 0 else None
        return cache[sta]

    for k, i in enumerate(idx):
        t, la, lo = q6[i, 0], q6[i, 1], q6[i, 2]
        d = a1.km_to(la, lo, slat, slon)
        near = [st[j, 0] for j in np.argsort(d) if d[j] <= 150.0][:5]
        rng = np.random.default_rng(a1.SEED + 13_000_000 + k)
        sers = [s for s in (series(x) for x in near) if s is not None and len(s.t)]

        def sig(T):
            i60 = [s.index(T, 60.0) for s in sers]
            i14 = [s.index(T, 14.0) for s in sers]
            f60 = [x for x in i60 if math.isfinite(x)]
            f14 = [x for x in i14 if math.isfinite(x)]
            return {"N1": max(f60) if f60 else math.nan, "N2": max(f14) if f14 else math.nan,
                    "N3": float(np.median(f60)) if len(f60) >= 2 else math.nan,
                    "P+": s1.count_quakes(T, la, lo, 30.0, 0.0, 150.0, 6.0), "P-": float(rng.random())}
        ev = sig(t)
        if not math.isfinite(ev["N1"]):
            continue                                       # no qualifying station at the event time
        cs = [sig(c) for c in a1.controls(q6, i, k)]
        rec = {"t": (a1.EPOCH + dt.timedelta(days=float(t))).isoformat()[:19], "lat": float(la), "lon": float(lo),
               "mag": float(q6[i, 4]), "stations": near, "event": ev, "u": {}, "top": {}}
        for s in (*SIGNALS, "P+", "P-"):
            cv = [c[s] for c in cs]
            rec["u"][s] = a1.percentile(ev[s], cv)
            good = [v for v in cv if math.isfinite(v)]
            rec["top"][s] = (bool(ev[s] > np.quantile(good, 0.8)) if math.isfinite(ev[s]) and len(good) >= 10 else None)
        rows.append(rec)
        print(f"  {rec['t'][:10]} M{rec['mag']:.1f} {len(sers)} stations: N1 {ev['N1']:.2f} (u {rec['u']['N1']})")
    split = a1.days(SPLIT)
    part = lambda sel: {s: a1.summary([r["u"][s] for r in sel], [r["top"][s] for r in sel]) for s in (*SIGNALS, "P+", "P-")}
    disc = [r for r in rows if a1.days(dt.datetime.fromisoformat(r["t"] + "+00:00")) < split]
    conf = [r for r in rows if a1.days(dt.datetime.fromisoformat(r["t"] + "+00:00")) >= split]
    res_all, res_d, res_c = part(rows), part(disc), part(conf)
    ctrl_ok = bool(res_all["P+"].get("n") and res_all["P+"]["mean_u"] > 0.5 and res_all["P+"]["p_two_sided"] < 0.01
                   and res_all["P-"]["p_two_sided"] >= 0.01)
    discovered = a1.bh([res_d[s]["p_two_sided"] if res_d[s].get("n") else 1.0 for s in SIGNALS])
    verdict = {}
    for j, s in enumerate(SIGNALS):
        d_, c_ = res_d[s], res_c[s]
        same = c_.get("n") and d_.get("n") and (c_["mean_u"] - 0.5) * (d_["mean_u"] - 0.5) > 0
        verdict[s] = {"discovered_before_2016": j in discovered,
                      "confirmed_2016_on": bool(j in discovered and same and c_["p_one_sided_same_sign"] < 0.05)}
    out = {"protocol": "docs/PRECURSOR_ATLAS.md (run 4)", "prereg_tag": "prereg-precursor-atlas-4",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_events": len(rows), "n_discover": len(disc), "n_confirm": len(conf),
           "instrument_controls_pass": ctrl_ok, "names": NAMES, "all": res_all, "discover_before_2016": res_d,
           "confirm_2016_on": res_c, "verdict": verdict, "events": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"events with GPS {len(rows)} (before 2016 {len(disc)}, from 2016 {len(conf)}); INSTRUMENT "
          f"{'PASS' if ctrl_ok else 'FAIL'} (P+ {res_all['P+'].get('mean_u')}, P- p {res_all['P-'].get('p_two_sided')})")
    for s in SIGNALS:
        al, v = res_all[s], verdict[s]
        if not al.get("n"):
            print(f"{s}: no values")
            continue
        print(f"{s} {NAMES[s]}: all u {al['mean_u']:.3f} (n {al['n']}, p {al['p_two_sided']:.2e}) | before 2016 "
              f"{res_d[s].get('mean_u', float('nan')):.3f} | 2016 on {res_c[s].get('mean_u', float('nan')):.3f} | "
              + ("CONFIRMED" if v["confirmed_2016_on"] else ("discovered, not confirmed" if v["discovered_before_2016"] else "no connection found")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
