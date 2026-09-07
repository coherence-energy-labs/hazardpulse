#!/usr/bin/env python3
"""Download GNSS time series from Nevada Geodetic Laboratory (NGL).

Targets ~300-500 stations near plate boundaries and historical M6+ events.
Computes velocity, strain rate anomaly, and transient detection for each station.

Data source: https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/
Station list: https://geodesy.unr.edu/NGLStationPages/llh.out

Cache: .cache/gnss/{STATION}.tenv3
Output: .cache/gnss/gnss_summary.json

Usage:
    python scripts/download_gnss.py [--max-stations N] [--workers N] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NGL_STATION_LIST_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
NGL_TENV3_BASE = "https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "gnss"

MAX_WORKERS = 10
MAX_STATIONS = 500

# Plate boundary proximity threshold (km)
PLATE_BOUNDARY_RADIUS_KM = 200.0

# Major plate boundary segments (simplified PB2002 trace as lat/lon waypoints)
# Each segment is a list of (lat, lon) points. Stations within
# PLATE_BOUNDARY_RADIUS_KM of any segment are kept.
PLATE_BOUNDARY_SEGMENTS: list[list[tuple[float, float]]] = [
    # Pacific Ring of Fire - Japan/Kuril
    [(30, 130), (35, 140), (40, 143), (45, 150), (50, 155)],
    # Alaska-Aleutians
    [(51, 179), (52, -180), (54, -170), (56, -160), (58, -155), (60, -150)],
    # Cascadia
    [(40, -125), (43, -125), (46, -124), (48, -125)],
    # San Andreas
    [(32, -115), (34, -118), (36, -121), (38, -123)],
    # Mexico-Central America
    [(14, -93), (16, -98), (18, -103), (20, -106)],
    # Chile-Peru
    [(-45, -75), (-35, -73), (-25, -71), (-15, -76), (-5, -81)],
    # Indonesia-Philippines
    [(-8, 110), (-5, 120), (0, 125), (5, 127), (10, 126), (15, 121)],
    # New Zealand-Tonga
    [(-45, 167), (-42, 173), (-38, 178), (-30, -177), (-20, -175)],
    # Mediterranean (Turkey-Italy-Greece)
    [(36, 25), (37, 28), (38, 30), (39, 35), (40, 40), (38, 44)],
    # Himalayan belt (Iran-Pakistan-India)
    [(25, 62), (30, 67), (33, 70), (35, 73), (28, 85), (27, 90)],
    # New Madrid (Central US)
    [(35, -90), (36, -89), (37, -89)],
    # Caribbean
    [(10, -62), (12, -68), (15, -70), (18, -67), (19, -65)],
    # East Africa Rift
    [(-5, 36), (0, 37), (5, 38), (10, 42), (12, 44)],
]


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1.0 - a)))


def point_to_segment_dist_km(
    plat: float, plon: float,
    seg: list[tuple[float, float]],
) -> float:
    """Minimum distance from a point to a polyline segment (km)."""
    min_d = float("inf")
    for slat, slon in seg:
        d = haversine_km(plat, plon, slat, slon)
        if d < min_d:
            min_d = d
    return min_d


def near_plate_boundary(lat: float, lon: float, radius_km: float) -> bool:
    """Is this point within radius_km of any plate boundary segment?"""
    for seg in PLATE_BOUNDARY_SEGMENTS:
        if point_to_segment_dist_km(lat, lon, seg) <= radius_km:
            return True
    return False


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch URL content with a User-Agent header."""
    req = Request(url, headers={"User-Agent": "HazardPulse/1.0 research"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_station_list() -> list[dict]:
    """Download and parse the NGL station list (llh.out).

    Format: station_name  lat  lon  height
    """
    cache_path = CACHE_DIR / "llh.out"
    if cache_path.exists():
        data = cache_path.read_text(encoding="utf-8", errors="replace")
    else:
        print("  Downloading station list from NGL...")
        raw = _fetch_url(NGL_STATION_LIST_URL, timeout=60)
        data = raw.decode("utf-8", errors="replace")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(data, encoding="utf-8")

    stations = []
    for line in data.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            hgt = float(parts[3]) if len(parts) > 3 else 0.0
        except (ValueError, IndexError):
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 360:
            if lon > 180:
                lon -= 360
            stations.append({"name": name, "lat": lat, "lon": lon, "height": hgt})

    return stations


def filter_stations(
    stations: list[dict],
    max_count: int = MAX_STATIONS,
) -> list[dict]:
    """Filter stations to those near plate boundaries."""
    selected = []
    for s in stations:
        if near_plate_boundary(s["lat"], s["lon"], PLATE_BOUNDARY_RADIUS_KM):
            selected.append(s)

    # If too many, subsample by region to get geographic spread
    if len(selected) > max_count:
        # Grid-based subsampling: keep at most N per 2-degree cell
        grid: dict[tuple[int, int], list[dict]] = {}
        for s in selected:
            key = (int(s["lat"] / 2), int(s["lon"] / 2))
            grid.setdefault(key, []).append(s)

        # Take up to ceil(max_count / n_cells) per cell
        n_cells = len(grid)
        per_cell = max(1, math.ceil(max_count / n_cells))
        result = []
        for cell_stations in grid.values():
            result.extend(cell_stations[:per_cell])

        # If still too many, truncate
        if len(result) > max_count:
            result = result[:max_count]
        return result

    return selected


def download_tenv3(station_name: str, force: bool = False) -> Path | None:
    """Download a single station's .tenv3 file. Returns path or None."""
    cache_path = CACHE_DIR / f"{station_name}.tenv3"
    if cache_path.exists() and not force:
        return cache_path

    url = f"{NGL_TENV3_BASE}/{station_name}.tenv3"
    try:
        raw = _fetch_url(url, timeout=30)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        return cache_path
    except (HTTPError, URLError, OSError):
        return None


# ---------------------------------------------------------------------------
# Parse and analyze .tenv3 time series
# ---------------------------------------------------------------------------

def parse_tenv3(filepath: Path) -> dict | None:
    """Parse a .tenv3 file into numpy arrays.

    Returns dict with keys: decimal_year, N, E, V, sN, sE, sV
    or None if parsing fails.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    years = []
    north = []
    east = []
    vert = []
    sn = []
    se = []
    sv = []

    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 12:
            continue
        try:
            yr = float(parts[2])
            n_mm = float(parts[7])
            e_mm = float(parts[8])
            v_mm = float(parts[9])
            sn_mm = float(parts[10])
            se_mm = float(parts[11])
            sv_mm = float(parts[12]) if len(parts) > 12 else 1.0
        except (ValueError, IndexError):
            continue
        years.append(yr)
        north.append(n_mm)
        east.append(e_mm)
        vert.append(v_mm)
        sn.append(sn_mm)
        se.append(se_mm)
        sv.append(sv_mm)

    if len(years) < 30:
        return None

    return {
        "decimal_year": np.array(years),
        "N": np.array(north),
        "E": np.array(east),
        "V": np.array(vert),
        "sN": np.array(sn),
        "sE": np.array(se),
        "sV": np.array(sv),
    }


def compute_station_metrics(data: dict) -> dict:
    """Compute velocity, strain rate anomaly, and transients from parsed tenv3.

    Returns dict with computed metrics or None if insufficient data.
    """
    yr = data["decimal_year"]
    n = data["N"]
    e = data["E"]
    v = data["V"]

    n_pts = len(yr)
    span = yr[-1] - yr[0]
    if span < 1.0 or n_pts < 50:
        return {}

    # --- Linear velocity fit (mm/yr) via least squares ---
    A = np.column_stack([yr - yr[0], np.ones(n_pts)])

    # North
    coeff_n, _, _, _ = np.linalg.lstsq(A, n, rcond=None)
    vel_n = coeff_n[0]
    resid_n = n - A @ coeff_n

    # East
    coeff_e, _, _, _ = np.linalg.lstsq(A, e, rcond=None)
    vel_e = coeff_e[0]
    resid_e = e - A @ coeff_e

    # Vertical
    coeff_v, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
    vel_v = coeff_v[0]
    resid_v = v - A @ coeff_v

    # Horizontal velocity magnitude
    vel_h = math.sqrt(vel_n ** 2 + vel_e ** 2)

    # RMS of residuals (quality indicator)
    rms_n = float(np.sqrt(np.mean(resid_n ** 2)))
    rms_e = float(np.sqrt(np.mean(resid_e ** 2)))
    rms_total = math.sqrt(rms_n ** 2 + rms_e ** 2)

    # --- Strain rate anomaly: velocity in recent 1yr vs previous 3yr ---
    yr_max = yr[-1]
    recent_mask = yr >= yr_max - 1.0
    older_mask = (yr >= yr_max - 4.0) & (yr < yr_max - 1.0)

    velocity_change = 0.0
    if np.sum(recent_mask) >= 20 and np.sum(older_mask) >= 50:
        A_rec = np.column_stack([yr[recent_mask] - yr[recent_mask][0],
                                  np.ones(int(np.sum(recent_mask)))])
        A_old = np.column_stack([yr[older_mask] - yr[older_mask][0],
                                  np.ones(int(np.sum(older_mask)))])
        try:
            c_rec_n, _, _, _ = np.linalg.lstsq(A_rec, n[recent_mask], rcond=None)
            c_old_n, _, _, _ = np.linalg.lstsq(A_old, n[older_mask], rcond=None)
            c_rec_e, _, _, _ = np.linalg.lstsq(A_rec, e[recent_mask], rcond=None)
            c_old_e, _, _, _ = np.linalg.lstsq(A_old, e[older_mask], rcond=None)
            vel_rec = math.sqrt(c_rec_n[0] ** 2 + c_rec_e[0] ** 2)
            vel_old = math.sqrt(c_old_n[0] ** 2 + c_old_e[0] ** 2)
            velocity_change = vel_rec - vel_old
        except (np.linalg.LinAlgError, ValueError):
            pass

    # --- Transient detection: max deviation from linear trend ---
    def _max_transient(resid_h: np.ndarray, yr_arr: np.ndarray, window_days: int) -> float:
        window_yr = window_days / 365.25
        mask = yr_arr >= yr_arr[-1] - window_yr
        if np.sum(mask) < 5:
            return 0.0
        return float(np.max(np.abs(resid_h[mask])))

    resid_h = np.sqrt(resid_n ** 2 + resid_e ** 2)
    transient_30d = _max_transient(resid_h, yr, 30)
    transient_90d = _max_transient(resid_h, yr, 90)
    transient_180d = _max_transient(resid_h, yr, 180)

    # RMS in recent 90 days
    recent_90d_mask = yr >= yr[-1] - 90 / 365.25
    rms_recent = float(np.sqrt(np.mean(resid_h[recent_90d_mask] ** 2))) if np.sum(recent_90d_mask) >= 5 else rms_total

    # --- Strain rate (acceleration relative to long-term) ---
    # Fit quadratic to detect acceleration
    strain_rate = 0.0
    if n_pts >= 100:
        A_quad = np.column_stack([
            (yr - yr[0]) ** 2,
            yr - yr[0],
            np.ones(n_pts),
        ])
        try:
            c_h = np.linalg.lstsq(A_quad, np.sqrt(n ** 2 + e ** 2), rcond=None)[0]
            strain_rate = c_h[0]  # quadratic coefficient = acceleration
        except (np.linalg.LinAlgError, ValueError):
            pass

    return {
        "vel_n": round(float(vel_n), 4),
        "vel_e": round(float(vel_e), 4),
        "vel_v": round(float(vel_v), 4),
        "vel_h": round(float(vel_h), 4),
        "strain_rate": round(float(strain_rate), 6),
        "velocity_change": round(float(velocity_change), 4),
        "transient_30d": round(float(transient_30d), 3),
        "transient_90d": round(float(transient_90d), 3),
        "transient_180d": round(float(transient_180d), 3),
        "rms_total": round(float(rms_total), 3),
        "rms_recent": round(float(rms_recent), 3),
        "n_points": n_pts,
        "span_years": round(float(span), 2),
        "last_epoch": round(float(yr_max), 4),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def download_and_process_station(
    station: dict, force: bool = False,
) -> dict | None:
    """Download + parse + analyze a single GNSS station. Thread-safe."""
    name = station["name"]
    filepath = download_tenv3(name, force=force)
    if filepath is None:
        return None

    data = parse_tenv3(filepath)
    if data is None:
        return None

    metrics = compute_station_metrics(data)
    if not metrics:
        return None

    return {
        "station": name,
        "lat": station["lat"],
        "lon": station["lon"],
        "height": station.get("height", 0.0),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GNSS time series from Nevada Geodetic Laboratory"
    )
    parser.add_argument(
        "--max-stations", type=int, default=MAX_STATIONS,
        help=f"Maximum number of stations to download (default: {MAX_STATIONS})"
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Number of download threads (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cached"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GNSS Time Series Downloader (NGL)")
    print("=" * 60)

    # Step 1: Get station list
    print("\n[1] Fetching station list...")
    all_stations = download_station_list()
    print(f"    Total NGL stations: {len(all_stations)}")

    # Step 2: Filter to plate boundary stations
    print(f"\n[2] Filtering to stations near plate boundaries "
          f"(within {PLATE_BOUNDARY_RADIUS_KM} km)...")
    selected = filter_stations(all_stations, max_count=args.max_stations)
    print(f"    Selected stations: {len(selected)}")

    # Step 3: Download and process
    print(f"\n[3] Downloading .tenv3 files ({args.workers} workers)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    n_done = 0
    n_fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_and_process_station, s, args.force): s
            for s in selected
        }

        for future in as_completed(futures):
            n_done += 1
            station = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1

            if n_done % 50 == 0 or n_done == len(selected):
                elapsed = time.time() - t0
                print(
                    f"    Progress: {n_done}/{len(selected)} "
                    f"({len(results)} ok, {n_fail} fail) -- "
                    f"{elapsed:.0f}s"
                )

    # Step 4: Save summary
    summary = {
        "download_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_stations_attempted": len(selected),
        "n_stations_ok": len(results),
        "n_stations_failed": n_fail,
        "elapsed_seconds": round(time.time() - t0, 1),
        "stations": results,
    }

    summary_path = CACHE_DIR / "gnss_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[4] Summary saved to: {summary_path}")
    print(f"    Stations with data: {len(results)}")
    print(f"    Failed/skipped:     {n_fail}")
    print(f"    Total time:         {time.time() - t0:.0f}s")

    # Regional breakdown
    regions: dict[str, int] = {}
    for r in results:
        lat, lon = r["lat"], r["lon"]
        if 25 < lat < 50 and 125 < lon < 155:
            region = "Japan"
        elif 50 < lat < 65 and -170 < lon < -140:
            region = "Alaska"
        elif 30 < lat < 50 and -130 < lon < -110:
            region = "Western US"
        elif -50 < lat < -15 and -85 < lon < -65:
            region = "South America"
        elif -15 < lat < 15 and 95 < lon < 140:
            region = "Indonesia"
        elif 34 < lat < 42 and 22 < lon < 45:
            region = "Mediterranean"
        else:
            region = "Other"
        regions[region] = regions.get(region, 0) + 1

    print("\n    Regional breakdown:")
    for region, count in sorted(regions.items(), key=lambda x: -x[1]):
        print(f"      {region:20s}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
