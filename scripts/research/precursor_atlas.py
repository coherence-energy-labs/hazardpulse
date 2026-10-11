#!/usr/bin/env python3
"""Precursor atlas (docs/PRECURSOR_ATLAS.md, tag prereg-precursor-atlas-1): does anything in the Earth system change
before large earthquakes, against the same place at other times?

    python scripts/research/precursor_atlas.py      # fetches its public inputs into .cache/precursor/

Events: shallow M7+ mainshocks (ComCat). Controls: 20 same-epicentre times within +-3 y. Each signal is the event
value's mid-rank percentile u among its own controls; with no connection u is uniform. Benjamini-Hochberg on
1990-2009 (q = 0.10), confirmed on 2010-2026 (same sign, one-sided p < 0.05). P+ (a known connection) and P- (noise)
check the instrument itself. Writes results/precursor_atlas/atlas.json.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "earthquake_program"))
import aftershock_base_rates as abr  # noqa: E402

CACHE = ROOT / ".cache" / "precursor"
OUT = ROOT / "results" / "precursor_atlas" / ("atlas_1b.json" if "--past-only" in sys.argv else "atlas.json")
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
N_CTRL, HALF_WINDOW_D, EXCLUDE_D, SEED = 20, 3 * 365.25, 180.0, 20261011
EVENT_START, EVENT_END = dt.datetime(1990, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2026, 10, 9, 18, tzinfo=dt.timezone.utc)
SPLIT = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
URLS = {
    "kp.txt": "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt",
    "sn.csv": "https://www.sidc.be/SILSO/INFO/sndtotcsv.php",
    "eop.txt": "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now",
    "gvp_eruptions.json": ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
                           "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
                           "&outputFormat=application/json&CQL_FILTER=StartDateYear%3E%3D1984"),
    "gvp_volcanoes.json": ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
                           "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes"
                           "&outputFormat=application/json"),
    "comcat_m6_1984_1990.csv": ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=1984-01-01"
                                "&endtime=1990-01-01&minmagnitude=6&orderby=time-asc&limit=20000"),
}


def days(t: dt.datetime) -> float:
    return (t - EPOCH).total_seconds() / 86400.0


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists() or p.stat().st_size < 1000:
        with urllib.request.urlopen(URLS[name], timeout=600) as r:
            p.write_bytes(r.read())
    if p.read_bytes()[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise SystemExit(f"{name}: the source returned an HTML page, not data")
    return p


# --------------------------------------------------------------------------- inputs

def load_quakes() -> np.ndarray:
    """Every M6+ earthquake 1984-today: columns t (days), lat, lon, depth, mag."""
    rows = [(days(t), la, lo, de, m) for t, la, lo, de, m, _ in abr.fetch(dt.date.today().isoformat())]
    for r in csv.DictReader(io.StringIO(fetch("comcat_m6_1984_1990.csv").read_text(encoding="utf-8"))):
        if r.get("type") == "earthquake":
            t = dt.datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
            rows.append((days(t), float(r["latitude"]), float(r["longitude"]), float(r["depth"] or 0), float(r["mag"])))
    a = np.array(sorted(set(rows)))
    return a


def load_kp():
    t, v = [], []
    for line in fetch("kp.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        stamp = dt.datetime(int(f[0]), int(f[1]), int(f[2]), tzinfo=dt.timezone.utc) + dt.timedelta(hours=float(f[3]))
        k = float(f[7])
        if k >= 0:
            t.append(days(stamp) + 3 / 24.0)          # the 3-h interval is known at its end
            v.append(k)
    return np.array(t), np.array(v)


def load_sn():
    t, v = [], []
    for line in fetch("sn.csv").read_text(encoding="utf-8").splitlines():
        f = [x.strip() for x in line.split(";")]
        if len(f) < 5 or float(f[4]) < 0:
            continue
        t.append(days(dt.datetime(int(f[0]), int(f[1]), int(f[2]), tzinfo=dt.timezone.utc)) + 1.0)
        v.append(float(f[4]))
    return np.array(t), np.array(v)


def load_lod():
    t, v = [], []
    for line in fetch("eop.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        t.append(days(dt.datetime(int(f[0]), int(f[1]), int(f[2]), tzinfo=dt.timezone.utc)) + 1.0)
        v.append(float(f[12]))                         # LOD (s), the 13th column of the C04 1962-now format
    return np.array(t), np.array(v)


def load_eruptions():
    vol = json.loads(fetch("gvp_volcanoes.json").read_text(encoding="utf-8"))["features"]
    pos = {}
    for f in vol:
        p, c = f["properties"], (f.get("geometry") or {}).get("coordinates")
        lat, lon = p.get("Latitude", c[1] if c else None), p.get("Longitude", c[0] if c else None)
        if lat is not None and lon is not None:
            pos[p["Volcano_Number"]] = (float(lat), float(lon))
    out = []
    for f in json.loads(fetch("gvp_eruptions.json").read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        q, y = pos.get(p["Volcano_Number"]), p.get("StartDateYear")
        if q is None or y is None:
            continue
        m, d = int(p.get("StartDateMonth") or 0) or 1, int(p.get("StartDateDay") or 0) or 1
        try:
            out.append((days(dt.datetime(int(y), m, d, tzinfo=dt.timezone.utc)), q[0], q[1]))
        except ValueError:
            continue
    return np.array(sorted(out))


# --------------------------------------------------------------------------- the Moon (Meeus ch. 47, main terms)

def moon(tdays: float) -> tuple[float, float]:
    """(days to the nearest new or full moon, Earth-Moon distance in km) at ``tdays`` (days since 1970, UTC)."""
    T = (tdays - 10957.5) / 36525.0                   # Julian centuries from J2000.0
    r = math.radians
    D = r((297.8501921 + 445267.1114034 * T) % 360)
    M = r((357.5291092 + 35999.0502909 * T) % 360)
    Mp = r((134.9633964 + 477198.8675055 * T) % 360)
    F = r((93.2720950 + 483202.0175233 * T) % 360)
    elong = math.degrees(D) + 6.289 * math.sin(Mp) - 2.100 * math.sin(M) + 1.274 * math.sin(2 * D - Mp) \
        + 0.658 * math.sin(2 * D) + 0.214 * math.sin(2 * Mp) + 0.110 * math.sin(D)
    to_syzygy = min(elong % 180.0, 180.0 - elong % 180.0) / 12.19
    dist = (385000.56 - 20905.355 * math.cos(Mp) - 3699.111 * math.cos(2 * D - Mp) - 2955.968 * math.cos(2 * D)
            - 569.925 * math.cos(2 * Mp) + 48.888 * math.cos(M) - 3.149 * math.cos(2 * F)
            + 246.158 * math.cos(2 * D - 2 * Mp) - 152.138 * math.cos(2 * D - M - Mp) - 170.733 * math.cos(2 * D + Mp)
            - 204.586 * math.cos(2 * D - M) - 129.620 * math.cos(M - Mp) + 108.743 * math.cos(D)
            + 104.755 * math.cos(M + Mp))
    return to_syzygy, dist


def check_moon() -> dict:
    """The ephemeris against a known instant: the 2024-04-08 total solar eclipse (new moon 18:21 UTC; the Moon near
    perigee, about 360,000 km). The run stops if it is off by more than 6 h or 3,000 km."""
    t = days(dt.datetime(2024, 4, 8, 18, 21, tzinfo=dt.timezone.utc))
    s, d = moon(t)
    full = moon(days(dt.datetime(2024, 4, 23, 23, 49, tzinfo=dt.timezone.utc)))[0]   # full moon 2024-04-23 23:49
    if s > 0.25 or full > 0.25 or abs(d - 360_000) > 3_000:
        raise SystemExit(f"lunar ephemeris check failed: new {s:.2f} d, full {full:.2f} d, distance {d:.0f} km")
    return {"new_moon_2024_04_08_days_off": s, "full_moon_2024_04_23_days_off": full, "distance_km": d}


# --------------------------------------------------------------------------- geometry and windows

def km_to(lat, lon, la, lo):
    p1, p2 = np.radians(lat), np.radians(la)
    x = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(np.radians(lo - lon))
    return 6371.0 * np.arccos(np.clip(x, -1.0, 1.0))


def window(ts, vs, t, span, how):
    i, j = np.searchsorted(ts, t - span, "right"), np.searchsorted(ts, t, "right")
    seg = vs[i:j]
    if len(seg) == 0:
        return math.nan
    return float(seg.max() if how == "max" else seg.mean())


class Signals:
    def __init__(self, quakes, kp, sn, lod, eru):
        self.q, self.kp, self.sn, self.lod, self.eru = quakes, kp, sn, lod, eru

    def count_quakes(self, t, lat, lon, before_d, rmin, rmax, mmin):
        q = self.q
        i, j = np.searchsorted(q[:, 0], t - before_d, "left"), np.searchsorted(q[:, 0], t, "left")
        s = q[i:j]
        s = s[(s[:, 4] >= mmin) & (s[:, 3] <= 70.0)]
        if rmax is None:
            return float(len(s))
        d = km_to(lat, lon, s[:, 1], s[:, 2])
        return float(((d > rmin) & (d <= rmax)).sum())

    def count_eruptions(self, t, lat, lon, before_d, rmax):
        e = self.eru
        i, j = np.searchsorted(e[:, 0], t - before_d, "left"), np.searchsorted(e[:, 0], t, "left")
        s = e[i:j]
        return float((km_to(lat, lon, s[:, 1], s[:, 2]) <= rmax).sum()) if len(s) else 0.0

    def all(self, t, lat, lon, noise):
        to_syz, dist = moon(t)
        lod_now = window(*self.lod, t, 30.0, "mean")
        lod_year = window(*self.lod, t - 365.0, 30.0, "mean")
        return {
            "T1": -to_syz,                      # larger = nearer a spring tide (so "higher before quakes" reads one way)
            "T2": -dist,                        # larger = Moon nearer
            "Q1": self.count_quakes(t, lat, lon, 120.0, 150.0, 1500.0, 7.0),
            "Q2": self.count_quakes(t, lat, lon, 365.0, 0.0, 500.0, 6.0),
            "Q3": self.count_quakes(t, lat, lon, 30.0, 0.0, None, 7.0),
            "V1": self.count_eruptions(t, lat, lon, 365.0, 500.0),
            "V2": self.count_eruptions(t, lat, lon, 90.0, 1000.0),
            "G1": window(*self.kp, t, 3.0, "max"),
            "G2": window(*self.kp, t, 27.0, "mean"),
            "S1": window(*self.sn, t, 27.0, "mean"),
            "R1": lod_now,
            "R2": lod_now - lod_year if math.isfinite(lod_now) and math.isfinite(lod_year) else math.nan,
            "P+": self.count_quakes(t, lat, lon, 30.0, 0.0, 150.0, 6.0),
            "P-": noise,
        }


SIGNALS = ("T1", "T2", "Q1", "Q2", "Q3", "V1", "V2", "G1", "G2", "S1", "R1", "R2")
NAMES = {"T1": "spring tide (nearness to new or full moon)", "T2": "Moon near (perigee)",
         "Q1": "M7+ within 150-1,500 km, 120 days before", "Q2": "M6+ within 500 km, 365 days before",
         "Q3": "M7+ anywhere, 30 days before", "V1": "eruptions within 500 km, 365 days before",
         "V2": "eruptions within 1,000 km, 90 days before", "G1": "highest Kp, 72 h before",
         "G2": "mean Kp, 27 days before", "S1": "sunspot number, 27 days before", "R1": "length of day, 30 days before",
         "R2": "length of day, change over the year", "P+": "POSITIVE CONTROL: M6+ within 150 km, 30 days before",
         "P-": "NEGATIVE CONTROL: random noise"}


# --------------------------------------------------------------------------- events, controls, statistic

def mainshocks(q):
    out = []
    for i in range(len(q)):
        t, la, lo, de, m = q[i]
        if m < 7.0 or de > 70.0 or not days(EVENT_START) <= t <= days(EVENT_END):
            continue
        k = np.searchsorted(q[:, 0], t - 30.0, "left")
        prev = q[k:i]
        if len(prev) and ((prev[:, 4] >= m) & (km_to(la, lo, prev[:, 1], prev[:, 2]) <= 150.0)).any():
            continue
        out.append(i)
    return out


PAST_ONLY = "--past-only" in sys.argv          # run 2b: controls only before the event (docs/PRECURSOR_ATLAS.md)
PAST_SPAN_D = 6 * 365.25


def controls(q, i, k):
    """Run 1/2: 20 draws in [t - 3 y, t + 3 y] off +-180 d. Run 2b (``--past-only``): 20 draws in [t - 6 y, t - 180 d],
    each also refused when an M7+ lies within 150 km in the 30 days before it (the mainshock rule, mirrored). Both
    refuse a draw followed by an M7+ within 500 km within 30 days."""
    t, la, lo = q[i, 0], q[i, 1], q[i, 2]
    rng = np.random.default_rng(SEED + k + (50_000_000 if PAST_ONLY else 0))
    last = q[-1, 0] - 31.0
    big = q[(q[:, 4] >= 7.0)]
    big = big[km_to(la, lo, big[:, 1], big[:, 2]) <= 500.0]
    close = big[km_to(la, lo, big[:, 1], big[:, 2]) <= 150.0]
    out, tries = [], 0
    while len(out) < N_CTRL and tries < 5000:
        tries += 1
        c = t - rng.uniform(EXCLUDE_D, PAST_SPAN_D) if PAST_ONLY else t + rng.uniform(-HALF_WINDOW_D, HALF_WINDOW_D)
        if abs(c - t) < EXCLUDE_D or c > last or c < days(dt.datetime(1985, 1, 1, tzinfo=dt.timezone.utc)):
            continue
        if ((big[:, 0] > c) & (big[:, 0] <= c + 30.0)).any():
            continue
        if PAST_ONLY and ((close[:, 0] < c) & (close[:, 0] >= c - 30.0)).any():
            continue
        out.append(c)
    return out


def percentile(x, ctrl):
    c = np.array([v for v in ctrl if math.isfinite(v)])
    if not math.isfinite(x) or len(c) < 10:
        return math.nan
    return float(((c < x).sum() + 0.5 * (c == x).sum()) / len(c))


def summary(us, top):
    u = np.array([x for x in us if math.isfinite(x)])
    n = len(u)
    if n == 0:
        return {"n": 0}
    z = (u.mean() - 0.5) / math.sqrt(1.0 / (12.0 * n))
    p = math.erfc(abs(z) / math.sqrt(2.0))
    tq = np.array([x for x in top if x is not None])
    return {"n": n, "mean_u": float(u.mean()), "z": float(z), "p_two_sided": p,
            "p_one_sided_same_sign": 0.5 * math.erfc(z / math.sqrt(2.0)) if z > 0 else 0.5 * math.erfc(-z / math.sqrt(2.0)),
            "share_above_controls_p80": float(tq.mean()) if len(tq) else None,
            "gain_top_quintile": float(tq.mean() / 0.2) if len(tq) else None}


def bh(ps, q=0.10):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, passed = len(ps), set()
    kmax = 0
    for r, i in enumerate(order, 1):
        if ps[i] <= q * r / m:
            kmax = r
    for r, i in enumerate(order, 1):
        if r <= kmax:
            passed.add(i)
    return passed


def main() -> int:
    eph = check_moon()
    q = load_quakes()
    sig = Signals(q, load_kp(), load_sn(), load_lod(), load_eruptions())
    idx = mainshocks(q)
    rows = []
    for k, i in enumerate(idx):
        t, la, lo = q[i, 0], q[i, 1], q[i, 2]
        rng = np.random.default_rng(SEED + 7_000_000 + k)
        ev = sig.all(t, la, lo, float(rng.random()))
        cs = [sig.all(c, la, lo, float(rng.random())) for c in controls(q, i, k)]
        rec = {"t": (EPOCH + dt.timedelta(days=float(t))).isoformat()[:19], "lat": float(la), "lon": float(lo),
               "mag": float(q[i, 4]), "n_controls": len(cs), "u": {}, "top": {}}
        for s in (*SIGNALS, "P+", "P-"):
            cv = [c[s] for c in cs]
            rec["u"][s] = percentile(ev[s], cv)
            good = [v for v in cv if math.isfinite(v)]
            rec["top"][s] = (bool(ev[s] > np.quantile(good, 0.8)) if math.isfinite(ev[s]) and len(good) >= 10 else None)
        rows.append(rec)
    split = days(SPLIT)

    def part(sel):
        return {s: summary([r["u"][s] for r in sel], [r["top"][s] for r in sel]) for s in (*SIGNALS, "P+", "P-")}
    disc = [r for r in rows if days(dt.datetime.fromisoformat(r["t"] + "+00:00")) < split]
    conf = [r for r in rows if days(dt.datetime.fromisoformat(r["t"] + "+00:00")) >= split]
    res_all, res_d, res_c = part(rows), part(disc), part(conf)
    ctrl_ok = res_all["P+"]["mean_u"] > 0.5 and res_all["P+"]["p_two_sided"] < 0.01 and res_all["P-"]["p_two_sided"] >= 0.01
    discovered = bh([res_d[s]["p_two_sided"] for s in SIGNALS])
    verdict = {}
    for j, s in enumerate(SIGNALS):
        d, c = res_d[s], res_c[s]
        same = c.get("n") and (c["mean_u"] - 0.5) * (d["mean_u"] - 0.5) > 0
        confirmed = bool(j in discovered and same and c["p_one_sided_same_sign"] < 0.05)
        verdict[s] = {"discovered_1990_2009": j in discovered, "confirmed_2010_2026": confirmed}
    panama = next((r for r in rows if r["t"].startswith("2026-10-09")), None)
    out = {"protocol": "docs/PRECURSOR_ATLAS.md", "prereg_tag": "prereg-precursor-atlas-1",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "ephemeris_check": eph, "n_events": len(rows), "n_discover": len(disc), "n_confirm": len(conf),
           "instrument_controls_pass": bool(ctrl_ok), "names": NAMES,
           "all": res_all, "discover_1990_2009": res_d, "confirm_2010_2026": res_c, "verdict": verdict,
           "panama_2026_10_09": panama}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"events {len(rows)} (discover {len(disc)}, confirm {len(conf)}); ephemeris {eph}")
    print(f"INSTRUMENT: P+ mean u {res_all['P+']['mean_u']:.3f} p {res_all['P+']['p_two_sided']:.1e}; "
          f"P- mean u {res_all['P-']['mean_u']:.3f} p {res_all['P-']['p_two_sided']:.2f} -> {'PASS' if ctrl_ok else 'FAIL: results void'}")
    for s in SIGNALS:
        a, d, c, v = res_all[s], res_d[s], res_c[s], verdict[s]
        print(f"{s} {NAMES[s][:44]:44s} all u {a['mean_u']:.3f} (p {a['p_two_sided']:.3f}, gain {a['gain_top_quintile']:.2f}) | "
              f"1990-2009 {d['mean_u']:.3f} p {d['p_two_sided']:.3f} | 2010-2026 {c['mean_u']:.3f} p {c['p_two_sided']:.3f} | "
              f"{'CONFIRMED' if v['confirmed_2010_2026'] else ('discovered, not confirmed' if v['discovered_1990_2009'] else 'no connection found')}")
    if panama:
        print("Panama 2026-10-09 percentiles among its own controls: "
              + ", ".join(f"{s} {panama['u'][s]:.2f}" for s in SIGNALS if math.isfinite(panama["u"][s])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
