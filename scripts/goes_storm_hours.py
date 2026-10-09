"""The GOES storm-hour task list for program G1: where each storm was, hour by hour, as known at each RI cycle.

    PYTHONPATH=src python scripts/goes_storm_hours.py     # writes results/goes/tasks.jsonl.gz

For RI cycle t, the images are the hours t-12 .. t+2.
- Hours up to t are centred by interpolating the storm's CARQ fixes. Every fix up to t is known at t, and the
  interpolated centre of an hour between two fixes is the same for every later cycle, so such a task is shared.
- Hours t+1 and t+2 come after the last fix. They are centred by extrapolating the t-6 -> t motion, the same rule
  the GMGSI crops used. These tasks belong to cycle t alone.
Nothing uses a fix after t. The live scorer applies the same rule to its own deck.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane.atcf import _parse_lat, _parse_lon  # noqa: E402

OUT = ROOT / "results" / "goes" / "tasks.jsonl.gz"
WINDOW_BACK_H = 12
AHEAD_H = 2
MAX_GAP_H = 12


def carq_fixes(text: str) -> dict[dt.datetime, tuple[float, float]]:
    out = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8 or f[4] != "CARQ" or f[5] != "0":
            continue
        try:
            t = dt.datetime.strptime(f[2], "%Y%m%d%H")
        except ValueError:
            continue
        lat, lon = _parse_lat(f[6]), _parse_lon(f[7])
        if lat is not None and lon is not None:
            out.setdefault(t, (lat, lon))
    return out


def _wrap(dlon: float) -> float:
    return (dlon + 180.0) % 360.0 - 180.0


def centre_at(fixes: dict, hour: dt.datetime, t: dt.datetime) -> tuple[float, float, str] | None:
    """The centre of ``hour`` as known at cycle ``t`` (fixes after t are never read)."""
    known = sorted(u for u in fixes if u <= t)
    if not known:
        return None
    if hour <= t:
        before = [u for u in known if u <= hour]
        after = [u for u in known if u >= hour]
        if not before or not after:
            return None
        a, b = before[-1], after[0]
        if a == b:
            return (*fixes[a], "interp")
        if (b - a).total_seconds() > MAX_GAP_H * 3600:
            return None
        w = (hour - a).total_seconds() / (b - a).total_seconds()
        (la, lo), (lb, lob) = fixes[a], fixes[b]
        return la + w * (lb - la), _wrap(lo + w * _wrap(lob - lo)), "interp"
    last = known[-1]
    prev = [u for u in known if last - dt.timedelta(hours=MAX_GAP_H) <= u < last]
    (la, lo) = fixes[last]
    if prev:
        p = prev[-1]
        rate = (hour - last).total_seconds() / (last - p).total_seconds()
        (lp, lop) = fixes[p]
        return la + rate * (la - lp), _wrap(lo + rate * _wrap(lo - lop)), "extrap"
    return la, lo, "extrap"


def deck_text(aid: str, season: int) -> str:
    p = v9.adeck_path(aid) if season < 2026 else v9.Y26 / "adeck" / f"a{aid[:2].lower()}{aid[2:]}.dat.gz"
    return gzip.decompress(p.read_bytes()).decode("utf-8", "replace") if p.exists() else ""


def main() -> int:
    cases = json.loads((v9.WORK / "fusion_sources.json").read_text(encoding="utf-8"))   # every RI cycle 2020-2026
    by_storm: dict[str, list[dt.datetime]] = {}
    season = {}
    for c in cases:
        by_storm.setdefault(c["atcf_id"], []).append(dt.datetime.strptime(c["dtg"], "%Y%m%d%H"))
        season[c["atcf_id"]] = c["season"]
    tasks: dict[tuple, dict] = {}
    for aid, cycles in sorted(by_storm.items()):
        fixes = carq_fixes(deck_text(aid, season[aid]))
        for t in sorted(cycles):
            for k in range(-WINDOW_BACK_H, AHEAD_H + 1):
                hour = t + dt.timedelta(hours=k)
                got = centre_at(fixes, hour, t)
                if got is None:
                    continue
                lat, lon, kind = got
                key = (aid, hour.strftime("%Y%m%d%H"), kind, t.strftime("%Y%m%d%H") if kind == "extrap" else "")
                tasks.setdefault(key, {"storm": aid, "hour": key[1], "kind": kind, "cycle": key[3],
                                       "lat": round(lat, 4), "lon": round(lon, 4)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(tasks.values(), key=lambda r: (r["hour"], r["storm"], r["kind"], r["cycle"]))
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    hours = {r["hour"] for r in rows}
    print(f"{len(by_storm)} storms, {len(cases)} cycles -> {len(rows)} tasks over {len(hours)} distinct hours "
          f"({sum(r['kind'] == 'extrap' for r in rows)} extrapolated); wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
