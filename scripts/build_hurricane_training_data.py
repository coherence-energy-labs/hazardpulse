#!/usr/bin/env python3
"""Build historical RI training dataset from IBTrACS best-track data.

Downloads IBTrACS CSV for multiple basins, extracts synoptic (6-hourly) observations,
labels each with RI (rapid intensification = 30+ kt in 24h), and writes
a JSONL file that the operational RI models are trained on.

  python scripts/build_hurricane_training_data.py                  # v8.2 (true clock)
  python scripts/build_hurricane_training_data.py --recipe v8.1    # reproduce the v8.1 file

v8.2 takes every time offset from real timestamps. The v8.1 recipe is kept, unchanged and
labelled LEGACY, only so the pinned v8.1 model stays reproducible for comparison.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The ONE definition of the derived features, shared with the live scorer so a served value
# can never drift from the trained one (tests re-derive committed rows from them).
from hazardpulse.hurricane.operational_ri import (  # noqa: E402
    climatological_mpi_features,
    translation_speed_kmh,
)
CACHE_DIR = PROJECT_ROOT / ".cache" / "ibtracs"
OUTPUT_PATHS = {
    "v8.2": PROJECT_ROOT / "results" / "hurricane_operational_ri_v8_2_2000_2024.jsonl",
    "v8.1": PROJECT_ROOT / "results" / "hurricane_operational_ri_2000_2024_al_sst.jsonl",
}
OUTPUT_PATH = OUTPUT_PATHS["v8.2"]

BASINS = {
    "NA": "Atlantic",
    "EP": "East Pacific",
    "WP": "West Pacific",
    "NI": "North Indian",
    "SI": "South Indian",
    "SP": "South Pacific",
}

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
)

RI_THRESHOLD_KT = 30
RI_WINDOW_HOURS = 24
RI_WINDOW_STEPS = 4  # LEGACY v8.1 only: "4 x 6h = 24h", but the rows are mostly 3-hourly
MIN_YEAR = 2000
MAX_YEAR = 2024


def fetch_cached(url: str) -> bytes | None:
    digest = hashlib.md5(url.encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{digest}.csv"
    if cache_path.exists():
        print(f"    Using cached: {cache_path.name}")
        return cache_path.read_bytes()
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "HazardPulse/1.0"})
        with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
        return data
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def parse_ibtracs(text: str) -> dict[str, dict]:
    """Parse IBTrACS CSV into storms keyed by SID."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    col = {h.strip().strip('"'): i for i, h in enumerate(header)}
    next(reader, None)  # skip units row

    storms: dict[str, dict] = {}
    for row in reader:
        if len(row) < 10:
            continue
        try:
            sid = row[col.get("SID", 0)].strip().strip('"')
            name = row[col.get("NAME", 1)].strip().strip('"')

            def sf(key):
                if key not in col:
                    return -999.0
                v = row[col[key]].strip().strip('"')
                if v in ("", " ", "NA", "na"):
                    return -999.0
                return float(v)

            wind = sf("USA_WIND")
            if wind <= 0:
                wind = sf("WMO_WIND")
            pres = sf("USA_PRES")
            if pres <= 0:
                pres = sf("WMO_PRES")

            time_str = row[col.get("ISO_TIME", 6)].strip().strip('"')
            year = int(time_str[:4]) if len(time_str) >= 4 else 0
            month = int(time_str[5:7]) if len(time_str) >= 7 else 0

            basin_str = ""
            if "BASIN" in col:
                basin_str = row[col["BASIN"]].strip().strip('"')

            if sid not in storms:
                storms[sid] = {"name": name, "basin": basin_str, "entries": []}

            storms[sid]["entries"].append({
                "wind": wind, "pres": pres,
                "lat": sf("LAT"), "lon": sf("LON"),
                "year": year, "month": month,
                "time_str": time_str,
            })
        except Exception:
            continue
    return storms


def _parse_track_time(time_str: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def extract_ri_cases(storms: dict[str, dict]) -> list[dict]:
    """v8.2: RI training cases whose every time offset is a REAL time offset.

    One case per storm per synoptic hour (00/06/12/18 UTC -- the cycles the live scorer
    runs on; IBTrACS' 3-hourly intermediate fixes are largely interpolated). Every
    "N hours ago / later" quantity is an exact timestamp lookup, never an index step:

    * ``ri_label_30kt``: wind(t + 24 h) - wind(t) >= 30 kt; no row without the t+24 h fix;
    * ``analysis_dv/dp_{6,12,24}h``: true 6/12/24-hour changes (None when that fix is absent);
    * ``translation_speed_kmh``: displacement since t - 6 h over 6 h;
    * ``storm_age_h``: hours since the track's first fix;

    which is exactly what scripts/fetch_and_score.py computes from a live a-deck.
    """
    cases: list[dict] = []
    for sid, sdata in storms.items():
        by_time: dict[dt.datetime, dict] = {}
        for e in sdata["entries"]:
            t = _parse_track_time(e["time_str"])
            if t is not None and t not in by_time:
                by_time[t] = e
        if not by_time:
            continue
        first_fix = min(by_time)
        for t in sorted(by_time):
            if t.hour % 6 or t.minute or t.second:
                continue
            e = by_time[t]
            w_now = e["wind"]
            if w_now <= 0 or e["year"] < MIN_YEAR or e["year"] > MAX_YEAR:
                continue
            future = by_time.get(t + dt.timedelta(hours=RI_WINDOW_HOURS))
            if future is None or future["wind"] <= 0:
                continue
            lat, lon = e["lat"], e["lon"]
            if lat == -999 or lon == -999:
                continue

            case: dict = {
                "storm_id": sid,
                "season_year": e["year"],
                "issue_time": e["time_str"],
                "basin": sdata["basin"],
                "storm_name": sdata["name"],
                "analysis_model": "BEST",
                "analysis_lat": lat,
                "analysis_lon": lon,
                "analysis_vmax_kt": w_now,
                "analysis_mslp_hpa": e["pres"] if e["pres"] > 800 else None,
                "ri_label_30kt": 1 if future["wind"] - w_now >= RI_THRESHOLD_KT else 0,
            }
            for hours in (6, 12, 24):
                prev = by_time.get(t - dt.timedelta(hours=hours))
                case[f"analysis_dv_{hours}h"] = (
                    w_now - prev["wind"] if prev is not None and prev["wind"] > 0 else None
                )
                case[f"analysis_dp_{hours}h"] = (
                    e["pres"] - prev["pres"]
                    if prev is not None and e["pres"] > 800 and prev["pres"] > 800 else None
                )
            case["abs_lat"] = abs(lat)
            case["issue_month_sin"] = float(np.sin(2 * np.pi * e["month"] / 12.0))
            case["issue_month_cos"] = float(np.cos(2 * np.pi * e["month"] / 12.0))
            mpi = climatological_mpi_features(lat, w_now, e["month"])
            case["mpi_deficit"] = mpi["mpi_deficit"]
            case["intensity_frac_mpi"] = mpi["intensity_frac_mpi"]
            prev6 = by_time.get(t - dt.timedelta(hours=6))
            case["translation_speed_kmh"] = (
                translation_speed_kmh(prev6["lat"], prev6["lon"], lat, lon, 6.0)
                if prev6 is not None and prev6["lat"] != -999 and prev6["lon"] != -999 else None
            )
            case["storm_age_h"] = (t - first_fix).total_seconds() / 3600.0
            cases.append(case)
    return cases


def extract_ri_cases_v81(storms: dict[str, dict]) -> list[dict]:
    """LEGACY (v8.1) extraction -- kept only to reproduce the pinned v8.1 training file.

    It steps through IBTrACS rows as if they were 6-hourly, but 94.1% of consecutive rows
    are 3 h apart, so its "24 h" RI label spans 12 h, its dv/dp_6/12/24h are 3/6/12-h
    changes, translation speed is halved and storm age doubled. Do not train new models
    on it; ``extract_ri_cases`` is the corrected builder. Reproduces
    results/hurricane_operational_ri_2000_2024_al_sst.jsonl byte for byte (LF sha256
    98c2b6b0..., verified 2026-10-02 from the 2026-04-13 IBTrACS cache).
    """
    cases: list[dict] = []

    for sid, sdata in storms.items():
        entries = sdata["entries"]
        n = len(entries)
        if n < RI_WINDOW_STEPS + 6:
            continue

        for i in range(4, n - RI_WINDOW_STEPS):
            e = entries[i]
            w_now = e["wind"]
            if w_now <= 0 or e["year"] < MIN_YEAR or e["year"] > MAX_YEAR:
                continue

            w_future = entries[i + RI_WINDOW_STEPS]["wind"]
            if w_future <= 0:
                continue

            dv_24h = w_future - w_now
            ri_label = 1 if dv_24h >= RI_THRESHOLD_KT else 0

            # Build feature dict with available best-track data
            lat = e["lat"]
            lon = e["lon"]
            if lat == -999 or lon == -999:
                continue

            case: dict = {
                "storm_id": sid,
                "season_year": e["year"],
                "issue_time": e["time_str"],
                "basin": sdata["basin"],
                "storm_name": sdata["name"],
                "analysis_model": "BEST",
                "analysis_lat": lat,
                "analysis_lon": lon,
                "analysis_vmax_kt": w_now,
                "analysis_mslp_hpa": e["pres"] if e["pres"] > 800 else None,
                "ri_label_30kt": ri_label,
            }

            # Wind change features
            for steps, suffix in [(1, "6h"), (2, "12h"), (4, "24h")]:
                if i >= steps and entries[i - steps]["wind"] > 0:
                    case[f"analysis_dv_{suffix}"] = w_now - entries[i - steps]["wind"]
                else:
                    case[f"analysis_dv_{suffix}"] = None

            # Pressure changes
            for steps, suffix in [(1, "6h"), (2, "12h"), (4, "24h")]:
                p_now = e["pres"]
                p_prev = entries[i - steps]["pres"] if i >= steps else -999
                if p_now > 800 and p_prev > 800:
                    case[f"analysis_dp_{suffix}"] = p_now - p_prev
                else:
                    case[f"analysis_dp_{suffix}"] = None

            # Location features
            case["abs_lat"] = abs(lat)
            case["issue_month_sin"] = float(np.sin(2 * np.pi * e["month"] / 12.0))
            case["issue_month_cos"] = float(np.cos(2 * np.pi * e["month"] / 12.0))

            # MPI estimate (latitude/season SST proxy) -- shared definition, see import.
            mpi = climatological_mpi_features(lat, w_now, e["month"])
            case["mpi_deficit"] = mpi["mpi_deficit"]
            case["intensity_frac_mpi"] = mpi["intensity_frac_mpi"]

            # Translation speed
            if i >= 1 and entries[i - 1]["lat"] != -999:
                dlat = math.radians(lat - entries[i - 1]["lat"])
                dlon = math.radians(lon - entries[i - 1]["lon"])
                a = (math.sin(dlat / 2) ** 2
                     + math.cos(math.radians(entries[i - 1]["lat"]))
                     * math.cos(math.radians(lat))
                     * math.sin(dlon / 2) ** 2)
                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                case["translation_speed_kmh"] = dist_km / 6.0
            else:
                case["translation_speed_kmh"] = None

            # Storm age
            case["storm_age_h"] = i * 6.0

            cases.append(case)

    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", choices=sorted(OUTPUT_PATHS), default="v8.2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    extract = extract_ri_cases if args.recipe == "v8.2" else extract_ri_cases_v81
    out_path = args.out or (OUTPUT_PATH if args.recipe == "v8.2" else OUTPUT_PATHS[args.recipe])

    print(f"Building hurricane RI training dataset from IBTrACS (recipe {args.recipe})")
    print(f"  Years: {MIN_YEAR}-{MAX_YEAR}")
    print(f"  RI threshold: {RI_THRESHOLD_KT} kt / 24h")
    print()

    all_cases: list[dict] = []

    for basin_code, basin_name in BASINS.items():
        print(f"  Loading IBTrACS {basin_name} ({basin_code})...")
        t0 = time.time()
        url = IBTRACS_URL.format(basin=basin_code)
        data = fetch_cached(url)
        if not data:
            continue
        text = data.decode("utf-8", errors="replace")
        storms = parse_ibtracs(text)
        cases = extract(storms)
        elapsed = time.time() - t0
        n_pos = sum(1 for c in cases if c["ri_label_30kt"] == 1)
        print(f"    {len(storms)} storms, {len(cases)} cases ({n_pos} RI+) in {elapsed:.1f}s")
        all_cases.extend(cases)

    n_pos = sum(1 for c in all_cases if c["ri_label_30kt"] == 1)
    n_neg = len(all_cases) - n_pos
    print()
    print(f"  Total: {len(all_cases)} cases ({n_pos} RI+, {n_neg} RI-, rate={n_pos / max(len(all_cases), 1):.1%})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": identical bytes on every OS (the model artifacts pin this file's sha256).
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in all_cases:
            fh.write(json.dumps(case) + "\n")

    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
