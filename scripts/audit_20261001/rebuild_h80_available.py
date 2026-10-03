"""Recompute the 80 km HRRR block (H80) of every feature-store row from the analysis a forecast
could actually have used (hazardpulse.data.hrrr_availability: published before the observation).

    PYTHONPATH=src python scripts/audit_20261001/rebuild_h80_available.py [procs] [start] [end]

The store gave each row the latest analysis VALID at or before it; ~56% of rows read one not yet
published (valid + 100 min). Per day this writes STORE/_h80avail/<day>.npz with
  H80 (n x 12, float32; NaN where no analysis was usable), age_min (analysis age at use),
  parity (rows re-derived under the OLD rule and compared with the stored H80 bit for bit).
A day whose old-rule parity fails is refused (the recomputation would not be the store's code).
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
from hazardpulse.data import hrrr_availability as ha
from hazardpulse.data.probsevere import load_cached_probsevere
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr

ROOT = H.PROJECT_ROOT
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", str(ROOT / ".cache" / "probsevere_v2")))
STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", str(ROOT / ".cache" / "feature_store_v3")))
OUT = STORE / "_h80avail"
H80_LO, H80_HI = sf.BLOCKS["H80"]
UTC = dt.timezone.utc


def _analysis(cache: dict, key: tuple[str, int]):
    if key not in cache:
        g = H.load_cached_hrrr(key[0], hour=key[1])
        cache[key] = None if g is None else (g, compute_derived_hrrr(g))
    return cache[key]


def build_day(d: str) -> str:
    src = STORE / f"{d}.npz"
    out = OUT / f"{d}.npz"
    if not src.exists():
        return f"{d} no-store"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return f"{d} cached"
    t0 = time.time()
    with np.load(src) as z:
        X_h80 = z["X"][:, H80_LO:H80_HI]
        sids, ts = z["sid"], z["t"]
    steps = load_cached_probsevere(d, cache_dir=PS_V2)
    prev = (dt.datetime.strptime(d, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")
    cache: dict = {}
    have = [(dd, h) for dd in (prev, d) for h in ha.ANALYSIS_HOURS if H._npz_path(dd, h).exists()]
    same_day_hours = sorted(h for dd, h in have if dd == d)
    n = len(sids)
    new = np.full((n, H80_HI - H80_LO), np.nan, np.float32)
    age = np.full(n, np.nan, np.float32)
    r = 0
    checked = mismatched = 0
    for st in steps:
        tv = dm.parse_probsevere_valid_time(st.get("valid_time", ""))
        if tv is None:
            continue
        for storm in st.get("storms", []):
            if str(storm.get("id", "")) != str(sids[r]) or int(tv.timestamp()) != int(ts[r]):
                raise RuntimeError(f"{d}: row {r} order differs from the store")
            # old rule (the store's): must reproduce the stored H80 exactly
            h_old = dm.select_analysis_hour(tv, same_day_hours)
            if h_old is not None:
                a = _analysis(cache, (d, h_old))
                old = dm.extract_block_h(storm, a[0], a[1]).astype(np.float32)
                checked += 1
                if not np.array_equal(old, X_h80[r], equal_nan=True):
                    mismatched += 1
            # the rule a live forecast is held to
            key = ha.usable_analysis(tv.astimezone(UTC).replace(tzinfo=None), have)
            if key is not None:
                a = _analysis(cache, key)
                new[r] = dm.extract_block_h(storm, a[0], a[1]).astype(np.float32)
                valid = dt.datetime.strptime(key[0], "%Y%m%d").replace(tzinfo=UTC) + dt.timedelta(hours=key[1])
                age[r] = (tv - valid).total_seconds() / 60.0
            r += 1
    if r != n:
        raise RuntimeError(f"{d}: {r} rows walked, store has {n}")
    if mismatched:
        return f"{d} PARITY-FAIL {mismatched}/{checked}"
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{d}.tmp.npz"
    np.savez_compressed(tmp, H80=new, age_min=age, parity_checked=np.int64(checked))
    os.replace(tmp, out)
    return f"{d} n={n} parity {checked}/{checked} usable={int(np.isfinite(age).sum())} {time.time() - t0:.0f}s"


def main() -> int:
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    start = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2020, 10, 15)
    end = dt.date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else dt.date(2025, 12, 31)
    days = [(start + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)]
    t0 = time.time()
    fails = 0
    with Pool(procs, maxtasksperchild=50) as pool:
        for i, msg in enumerate(pool.imap_unordered(build_day, days), 1):
            fails += "FAIL" in msg
            print(f"[{i}/{len(days)} {time.time() - t0:6.0f}s] {msg}", flush=True)
    print(f"DONE parity failures: {fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
