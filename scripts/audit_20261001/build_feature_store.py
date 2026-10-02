"""Build the v3 storm-observation feature store, one compressed .npz per UTC day.

    PYTHONPATH=src python scripts/audit_20261001/build_feature_store.py [procs] [start] [end]
    PYTHONPATH=src python scripts/audit_20261001/build_feature_store.py --day 20240427

Every storm observation of the day with its full feature vector
(storm_features.FEATURE_NAMES), all labels (storm_features.LABEL_NAMES), and
metadata. Inputs: ProbSevere v2 cache, the 80 km g3 HRRR grids and the 9 km
native blocks of the day's 3-hourly analyses, SPC 1950-2025 (UTC instants).
A day is (re)built only when its output is missing; inputs missing for a day
are recorded in the file, never silently filled.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from hazardpulse.data import hrrr as H
from hazardpulse.data.probsevere import load_cached_probsevere
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado.coherence_engine import compute_coherence_fields, compute_derived_hrrr

ROOT = H.PROJECT_ROOT
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", str(ROOT / ".cache" / "probsevere_v2")))
STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", str(ROOT / ".cache" / "feature_store_v3")))
SPC = Path(os.environ.get("HAZARDPULSE_SPC_CSV", str(ROOT / ".cache" / "spc" / "1950-2025_actual_tornadoes.csv")))
HOURS = dm.HRRR_ANALYSIS_HOURS

_REPORTS = None


def _reports():
    global _REPORTS
    if _REPORTS is None:
        _REPORTS = dm.load_spc_tornado_reports(SPC)
    return _REPORTS


def build_day(d: str) -> str:
    out = STORE / f"{d}.npz"
    if out.exists():
        return f"{d} cached"
    t0 = time.time()
    steps = load_cached_probsevere(d, cache_dir=PS_V2)
    if steps is None:
        return f"{d} NO-PROBSEVERE"
    if not steps:
        return f"{d} empty"
    month = int(d[4:6])
    an80, an9 = {}, {}
    for h in HOURS:
        g = H.load_cached_hrrr(d, hour=h)
        if g is not None:
            an80[h] = (g, compute_derived_hrrr(g), compute_coherence_fields(g, month=month))
        b = H.load_lcc_blocks(d, h, sf.LCC_K)
        if b is not None:
            an9[h] = sf.analysis_fields(b)
    t_an = time.time() - t0
    window = dm.reports_in_label_window(_reports(), d)
    step_min = dm.probsevere_step_minutes(steps)
    idx = dm.index_storms_by_id(steps)
    X, Y, EF, LEAD, T, LAT, LON, SID, STEP, AH = [], [], [], [], [], [], [], [], [], []
    for si, ts in enumerate(steps):
        t = dm.parse_probsevere_valid_time(ts.get("valid_time", ""))
        if t is None:
            continue
        h80 = dm.select_analysis_hour(t, sorted(an80))
        h9 = dm.select_analysis_hour(t, sorted(an9))
        tsec = t.timestamp()
        for storm in ts.get("storms", []):
            hist = dm.build_storm_history(steps, storm.get("id"), si, id_index=idx)
            a80 = an80.get(h80) if h80 is not None else None
            X.append(sf.feature_vector(
                storm, hist, step_min,
                h80=(a80[0], a80[1]) if a80 else None,
                c80=a80[2] if a80 else None,
                fields9=an9.get(h9) if h9 is not None else None,
            ))
            lab, ef, lead = sf.labels(storm, tsec, window)
            Y.append(lab)
            EF.append(ef)
            LEAD.append(lead)
            T.append(int(tsec))
            LAT.append(float(storm.get("lat", 0.0)))
            LON.append(float(storm.get("lon", 0.0)))
            SID.append(str(storm.get("id", "")))
            STEP.append(si)
            AH.append(h9 if h9 is not None else (-1 if h80 is None else 100 + h80))
    if not X:
        return f"{d} no-storms"
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        str(tmp),
        X=np.stack(X).astype(np.float32), Y=np.stack(Y).astype(np.int8),
        ef=np.asarray(EF, np.float32), lead_min=np.asarray(LEAD, np.float32),
        t=np.asarray(T, np.int64), lat=np.asarray(LAT, np.float32), lon=np.asarray(LON, np.float32),
        sid=np.asarray(SID), step=np.asarray(STEP, np.int16), analysis=np.asarray(AH, np.int16),
        hours80=np.asarray(sorted(an80), np.int16), hours9=np.asarray(sorted(an9), np.int16),
        step_minutes=np.float32(step_min), schema=np.asarray(len(sf.FEATURE_NAMES)),
    )
    os.replace(tmp, out)
    ys = np.stack(Y)
    return (f"{d} n={len(X)} storm60={int(ys[:, 1].sum())} nbhd60={int(ys[:, 4].sum())} "
            f"an80={len(an80)} an9={len(an9)} {time.time() - t0:.0f}s (analyses {t_an:.0f}s)")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--day":
        print(build_day(sys.argv[2]), flush=True)
        return 0
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    start = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2020, 10, 15)
    end = dt.date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else dt.date(2025, 12, 31)
    days = [(start + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)]
    t0 = time.time()
    with Pool(procs, maxtasksperchild=20) as pool:
        for i, msg in enumerate(pool.imap_unordered(build_day, days), 1):
            print(f"[{i}/{len(days)} {time.time() - t0:6.0f}s] {msg}", flush=True)
    import json

    names = STORE / "feature_names.json"
    names.write_text(json.dumps({"features": list(sf.FEATURE_NAMES), "labels": list(sf.LABEL_NAMES),
                                 "blocks": sf.BLOCKS}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
