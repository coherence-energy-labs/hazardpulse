"""Amendment 1 of docs/TORNADO_MODEL_PROGRAM.md: which storm label attributes tornado reports best?

    PYTHONPATH=src python scripts/audit_20261001/label_attribution.py 2021-01-01 2021-12-31

For every SPC tornado report whose preceding 60 min are covered by ProbSevere data, the set of
storm ids each candidate label credits (any observation at lead 0-60 min). Labels only; no model.
Writes results/lab/label_attribution_<start>_<end>.json.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from hazardpulse.data.probsevere import load_cached_probsevere
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import storm_features as sf

ROOT = Path(__file__).resolve().parents[2]
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", str(ROOT / ".cache" / "probsevere_v2")))
SPC = Path(os.environ.get("HAZARDPULSE_SPC_CSV", str(ROOT / ".cache" / "spc" / "1950-2025_actual_tornadoes.csv")))
UTC = dt.timezone.utc
LEAD_MAX_S = 3600.0
REACH_KM = 150.0      # centroid-to-report distance (plus advection) beyond which no candidate can match
CANDIDATES = ("L0_centroid_R", "L1_polygon_5km", "L2_polygon_10km", "T5_tracked_5km", "T10_tracked_10km",
              "nbhd_40km")
TRACK_MATCH_S = 900.0


def km(lat1, lon1, lat2, lon2):
    return 111.32 * math.hypot(lat1 - lat2, (lon1 - lon2) * math.cos(math.radians(0.5 * (lat1 + lat2))))


_DAY_CACHE: dict[str, list] = {}


def _slots(ds: str) -> list[tuple[float, dict]]:
    """[(valid time, step)] of one UTC day, memoised for the day and its neighbour."""
    if ds not in _DAY_CACHE:
        steps = load_cached_probsevere(ds, cache_dir=PS_V2) or []
        out = []
        for st in steps:
            t = dm.parse_probsevere_valid_time(st.get("valid_time", ""))
            if t is not None:
                out.append((t.timestamp(), st))
        _DAY_CACHE[ds] = out
        for k in sorted(_DAY_CACHE)[:-3]:
            del _DAY_CACHE[k]
    return _DAY_CACHE[ds]


def tracked_distance_km(track: list[tuple[float, dict]], r: dict) -> float:
    """Distance from report r to the polygon of the same storm id at its slot nearest the report
    time (within TRACK_MATCH_S), else at its last slot before the report, advected by the residual."""
    tr = r["time_utc"]
    near = min(track, key=lambda e: abs(e[0] - tr))
    if abs(near[0] - tr) > TRACK_MATCH_S:
        before = [e for e in track if e[0] <= tr]
        if not before:
            return float("inf")
        near = before[-1]
    ring = sf.polygon_ring_km(near[1])
    if ring is None:
        return float("inf")
    return sf.advected_polygon_distance_km(near[1], ring, r["slat"], r["slon"], tr - near[0])


def main() -> int:
    start = dt.date.fromisoformat(sys.argv[1])
    end = dt.date.fromisoformat(sys.argv[2])
    reports = dm.load_spc_tornado_reports(SPC)
    # every report of the period, by a stable id
    period = []
    d = start
    while d <= end:
        for r in reports.get(d.strftime("%Y%m%d"), []):
            period.append(r)
        d += dt.timedelta(days=1)
    rid = {id(r): i for i, r in enumerate(period)}
    credited = {c: collections.defaultdict(set) for c in CANDIDATES}
    covered = set()            # reports with at least one ProbSevere slot in [t - 60 min, t]
    nearest_any = {}           # report -> min advected-polygon distance over all obs at lead 0-60
    nearest_tracked = {}       # report -> min tracked-polygon distance over the ids observed at lead 0-60
    n_obs = n_poly_missing = 0
    day = start - dt.timedelta(days=1)
    while day <= end:
        ds = day.strftime("%Y%m%d")
        nxt = (day + dt.timedelta(days=1)).strftime("%Y%m%d")
        day += dt.timedelta(days=1)
        slots = _slots(ds)
        if not slots:
            continue
        window = [r for r in dm.reports_in_label_window(reports, ds) if id(r) in rid]
        if not window:
            continue
        tracks: dict[str, list] = collections.defaultdict(list)
        for ts, st in slots + _slots(nxt):
            for storm in st.get("storms", []):
                tracks[str(storm.get("id"))].append((ts, storm))
        tracked_memo: dict[tuple[str, int], float] = {}
        rt = np.array([r["time_utc"] for r in window])
        for ts, s in slots:
            sel = np.nonzero((rt >= ts) & (rt - ts <= LEAD_MAX_S))[0]
            for i in sel:
                covered.add(rid[id(window[i])])
            if len(sel) == 0:
                continue
            for storm in s.get("storms", []):
                n_obs += 1
                lat0, lon0 = float(storm["lat"]), float(storm["lon"])
                ue, vn = sf.storm_motion(storm)
                sid = str(storm.get("id"))
                ring = None
                for i in sel:
                    r = window[i]
                    dt_s = r["time_utc"] - ts
                    if km(lat0, lon0, r["slat"], r["slon"]) > REACH_KM + math.hypot(ue, vn) * dt_s / 1000.0:
                        continue
                    key = rid[id(r)]
                    lab, _, _ = sf.labels(storm, ts, [r])
                    if lab[sf.LABEL_NAMES.index("storm_60")]:
                        credited["L0_centroid_R"][key].add(sid)
                    if lab[sf.LABEL_NAMES.index("nbhd_60")]:
                        credited["nbhd_40km"][key].add(sid)
                    mk = (sid, key)
                    if mk not in tracked_memo:
                        tracked_memo[mk] = tracked_distance_km(tracks[sid], r)
                    dtk = tracked_memo[mk]
                    nearest_tracked[key] = min(nearest_tracked.get(key, float("inf")), dtk)
                    if dtk <= 5.0:
                        credited["T5_tracked_5km"][key].add(sid)
                    if dtk <= 10.0:
                        credited["T10_tracked_10km"][key].add(sid)
                    if ring is None:
                        ring = sf.polygon_ring_km(storm)
                        if ring is None:
                            n_poly_missing += 1
                            ring = False
                    if ring is False:
                        continue
                    dp = sf.advected_polygon_distance_km(storm, ring, r["slat"], r["slon"], dt_s)
                    nearest_any[key] = min(nearest_any.get(key, float("inf")), dp)
                    if dp <= 5.0:
                        credited["L1_polygon_5km"][key].add(sid)
                    if dp <= 10.0:
                        credited["L2_polygon_10km"][key].add(sid)
        print(ds, len(covered), flush=True)
    out = {"period": [str(start), str(end)], "reports": len(period), "covered": len(covered),
           "observations_scanned": n_obs, "observations_without_polygon": n_poly_missing, "candidates": {}}
    cov = sorted(covered)
    for c in CANDIDATES:
        n_cred = [len(credited[c].get(k, ())) for k in cov]
        hit = [n for n in n_cred if n > 0]
        out["candidates"][c] = {
            "coverage": len(hit) / max(len(cov), 1),
            "credited_reports": len(hit),
            "ambiguity_mean": float(np.mean(hit)) if hit else float("nan"),
            "ambiguity_median": float(np.median(hit)) if hit else float("nan"),
            "share_credited_to_2plus": float(np.mean([n >= 2 for n in hit])) if hit else float("nan"),
        }
    dists = np.array([nearest_any.get(k, np.inf) for k in cov])
    out["nearest_advected_polygon_km_quantiles"] = {
        q: float(np.quantile(dists[np.isfinite(dists)], q)) for q in (0.5, 0.75, 0.9, 0.95)
    } if np.isfinite(dists).any() else {}
    out["no_observation_within_reach"] = int(np.sum(~np.isfinite(dists)))
    dtr = np.array([nearest_tracked.get(k, np.inf) for k in cov])
    out["nearest_tracked_polygon_km_quantiles"] = {
        q: float(np.quantile(dtr[np.isfinite(dtr)], q)) for q in (0.5, 0.75, 0.9, 0.95)
    } if np.isfinite(dtr).any() else {}
    for ef_min in (0, 2):
        ks = [k for k in cov if float(period[k].get("mag", -1)) >= ef_min]
        out[f"coverage_ef{ef_min}plus"] = {
            c: float(np.mean([len(credited[c].get(k, ())) > 0 for k in ks])) if ks else float("nan")
            for c in CANDIDATES}
        out[f"n_ef{ef_min}plus"] = len(ks)
    path = ROOT / "results" / "lab" / f"label_attribution_{start}_{end}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
