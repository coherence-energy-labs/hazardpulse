#!/usr/bin/env python3
"""How often an M6+ follows a large shallow mainshock in the same grid cell within 30 days (ledger E5, 2026-10-11).

    python scripts/earthquake_program/aftershock_base_rates.py      # fetches ComCat M6+ 1990-01-01 .. today if absent

Empirical, from ComCat (every M6+ of type "earthquake"). A mainshock is a shallow event (depth <= 70 km) with no
event of equal or larger magnitude within 150 km in the 30 days before it. For a forecast issued `lag` hours after it,
the outcome is an M6+ epicentre in the mainshock's own 2-degree cell (this program's grid: rows from 60 S, columns
from 180 W) within the 30 days from issue -- the published forecast's own event. Mainshocks in the last 31 days are
left out (their window is not over).

It measures the chance that matters most for lives -- after a large earthquake -- with no model at all, so a forecast
can be held to it. Written to results/earthquake_program/aftershock_base_rates.json.
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

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "earthquake" / "comcat_m6"
OUT = ROOT / "results" / "earthquake_program" / "aftershock_base_rates.json"
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CHUNKS = (("1990-01-01", "2004-01-01"), ("2004-01-01", "2015-01-01"), ("2015-01-01", None))
LAGS_H = (0.5, 6.0, 24.0, 168.0)
CLASSES = ((7.0, 7.5), (7.5, 10.0))
DEPTH_KM, FORE_KM, FORE_DAYS, WINDOW_D = 70.0, 150.0, 30, 30


def fetch(today: str) -> list[tuple]:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in CHUNKS:
        p = CACHE / f"m6_{a}_{b or today}.csv"
        if not p.exists():
            q = f"{FDSN}?format=csv&starttime={a}&endtime={b or today}&minmagnitude=6&orderby=time-asc&limit=20000"
            with urllib.request.urlopen(q, timeout=300) as r:
                p.write_bytes(r.read())
        for r in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))):
            if r.get("type") != "earthquake":
                continue
            t = dt.datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
            rows.append((t, float(r["latitude"]), float(r["longitude"]), float(r["depth"] or 0.0), float(r["mag"]), r["id"]))
    rows.sort()
    if any(x[5] == y[5] for x, y in zip(rows, rows[1:])):
        raise SystemExit("a ComCat id appears twice across the chunks")
    return rows


def km(a, b, c, d) -> float:
    p1, p2 = math.radians(a), math.radians(c)
    x = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(math.radians(d - b))
    return 6371.0 * math.acos(max(-1.0, min(1.0, x)))


def cell(lat: float, lon: float) -> tuple[int, int]:
    return (math.floor((lat + 60.0) / 2.0), math.floor((lon + 180.0) / 2.0))


def mainshocks(ev, lo_m, hi_m, end):
    for i, (t, la, lo, dep, m, eid) in enumerate(ev):
        if not (lo_m <= m < hi_m) or dep > DEPTH_KM or t > end - dt.timedelta(days=WINDOW_D + 1):
            continue
        if any(t - dt.timedelta(days=FORE_DAYS) <= u < t and v >= m and km(la, lo, a, b) <= FORE_KM
               for u, a, b, _, v, _ in ev[max(0, i - 400):i]):
            continue
        yield i


SNAPSHOT = ROOT / "results" / "earthquake_program" / "comcat_m6_snapshot"


def load_snapshot(folder: Path = SNAPSHOT) -> list[tuple]:
    """The committed ComCat M6+ snapshot the 2026-10-11 rates and amendment E5's fit were computed from."""
    rows = []
    for p in sorted(folder.glob("m6_*.csv")):
        for r in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))):
            if r.get("type") == "earthquake":
                t = dt.datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
                rows.append((t, float(r["latitude"]), float(r["longitude"]), float(r["depth"] or 0.0), float(r["mag"]), r["id"]))
    rows.sort()
    if not rows or any(x[5] == y[5] for x, y in zip(rows, rows[1:])):
        raise SystemExit(f"{folder}: empty, or a ComCat id appears twice")
    return rows


def main() -> int:
    ev = load_snapshot() if "--snapshot" in sys.argv else fetch(dt.date.today().isoformat())
    res = compute(ev)
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


def compute(ev: list[tuple]) -> dict:
    end = ev[-1][0]
    res = {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": FDSN,
           "events_m6": len(ev), "first": ev[0][0].isoformat(), "last": end.isoformat(),
           "definition": {"mainshock": f"depth <= {DEPTH_KM:g} km, no equal or larger event within {FORE_KM:g} km in "
                                       f"the {FORE_DAYS} days before", "outcome": "an M6+ epicentre in the mainshock's "
                                       "2-degree cell within 30 days of the forecast's issue (lag after the mainshock)"},
           "classes": {}}
    for lo_m, hi_m in CLASSES:
        idx = list(mainshocks(ev, lo_m, hi_m, end))
        name = f"M{lo_m:g}-{hi_m:g}" if hi_m < 10 else f"M{lo_m:g}+"
        rec = {"n_mainshocks": len(idx), "by_lag_h": {}}
        for lag in LAGS_H:
            hits = 0
            for i in idx:
                t, la, lo = ev[i][0], ev[i][1], ev[i][2]
                s, e = t + dt.timedelta(hours=lag), t + dt.timedelta(hours=lag, days=WINDOW_D)
                c0 = cell(la, lo)
                hits += any(s < u <= e and cell(a, b) == c0 for u, a, b, _, _, _ in ev[i + 1:i + 800])
            p = hits / len(idx)
            se = math.sqrt(p * (1 - p) / len(idx))
            rec["by_lag_h"][f"{lag:g}"] = {"hits": hits, "rate": p, "ci95": [max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)]}
            print(f"{name} (n={len(idx)}), forecast issued +{lag:g} h: M6+ in the same cell within 30 d "
                  f"{hits} ({p:.0%}, 95% {max(0, p - 1.96 * se):.0%}-{min(1, p + 1.96 * se):.0%})")
        res["classes"][name] = rec
    # the Panama state at its 00:00 forecast: an M7.5+ whose cell already had an M6+ within 6 h
    n = hit = 0
    for i in mainshocks(ev, 7.5, 10.0, end):
        t, la, lo = ev[i][0], ev[i][1], ev[i][2]
        c0 = cell(la, lo)
        nxt = ev[i + 1:i + 800]
        if not any(t < u <= t + dt.timedelta(hours=6) and cell(a, b) == c0 for u, a, b, _, _, _ in nxt):
            continue
        n += 1
        hit += any(t + dt.timedelta(hours=6) < u <= t + dt.timedelta(hours=6, days=WINDOW_D) and cell(a, b) == c0
                   for u, a, b, _, _, _ in nxt)
    res["M7.5+_with_an_M6+_in_the_cell_within_6h"] = {"n": n, "another_within_30d_from_6h": hit, "rate": hit / max(n, 1)}
    print(f"M7.5+ with an M6+ in its cell within 6 h (n={n}): yet another within 30 days from +6 h: {hit} ({hit / max(n, 1):.0%})")
    return res


if __name__ == "__main__":
    raise SystemExit(main())
