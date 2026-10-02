#!/usr/bin/env python3
"""Download earthquake data sources for HazardPulse coherence model.

Downloads and caches:
  1. USGS global earthquake catalog (M2.5+, 2000-2025)
  2. GCMT moment tensor catalog
  3. PB2002 plate boundary model
  4. Nevada Geodetic Lab GPS time series (top 100 stations)

Usage:
  python download_earthquake_data.py          # full download
  python download_earthquake_data.py --test   # quick test (1 year + 5 stations)
  python download_earthquake_data.py --usgs-only  # just USGS catalog
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import re
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "earthquake"

USGS_DIR = CACHE_ROOT / "usgs"
GCMT_DIR = CACHE_ROOT / "gcmt"
PLATES_DIR = CACHE_ROOT / "plates"
GNSS_DIR = CACHE_ROOT / "gnss"

USER_AGENT = "hazardpulse/0.1 (+https://github.com/coherence-energy-labs/hazardpulse)"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_SSL_CTX = ssl.create_default_context()


def _fetch(url: str, *, timeout: int = 120) -> bytes:
    """Fetch raw bytes from *url*."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.read()


def _fetch_text(url: str, *, timeout: int = 120) -> str:
    return _fetch(url, timeout=timeout).decode("utf-8", errors="replace")


def _fetch_text_with_retry(url: str, *, timeout: int = 180, attempts: int = 4) -> str:
    """Fetch with exponential back-off; the last failure propagates (never swallowed)."""
    for attempt in range(attempts):
        try:
            return _fetch_text(url, timeout=timeout)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
    raise AssertionError("unreachable")


# ===================================================================
# 1.  USGS Global Earthquake Catalog
# ===================================================================

USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_MIN_MAGNITUDE = 2.5
USGS_REQUEST_PAUSE_SECONDS = 0.5      # politeness between FDSN requests
# A current-year file older than this is re-pulled so new events flow in.
USGS_CURRENT_YEAR_MAX_AGE = dt.timedelta(days=1)

# The shared limit-aware fetcher lives in the package (also used by the live scorer).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hazardpulse.data import usgs_fdsn  # noqa: E402


def usgs_year_path(year: int, usgs_dir: Path | None = None) -> Path:
    return (usgs_dir or USGS_DIR) / f"usgs_catalog_{year}.csv"


def usgs_manifest_path(csv_path: Path) -> Path:
    return usgs_fdsn.manifest_path_for(csv_path)


def _year_period(year: int, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return start, min(end, now)


def audit_usgs_year_file(path: Path, year: int, *, now: dt.datetime | None = None) -> list[str]:
    """Problems with a cached year file ([] = complete as far as can be checked).

    Uses the fetch manifest when present (written only after a limit-aware pull
    finished); legacy files without one are judged from their contents alone.
    """
    if not path.exists():
        return ["missing"]
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        times = [row.get("time", "") for row in csv.DictReader(fh)]
    return usgs_fdsn.audit_year_catalog(times, year, manifest=usgs_fdsn.read_manifest(path),
                                        now=now)


def _usgs_year_is_current(path: Path, year: int, now: dt.datetime) -> bool:
    """A cached file is reusable only if it is complete and, for the running year,
    was pulled recently enough."""
    if audit_usgs_year_file(path, year, now=now):
        return False
    if year == now.year:
        try:
            pulled = usgs_fdsn.parse_event_time(usgs_fdsn.read_manifest(path)["coverage_end"])
        except (ValueError, KeyError):
            return False
        return now - pulled <= USGS_CURRENT_YEAR_MAX_AGE
    return True


def download_usgs_year(
    year: int,
    *,
    usgs_dir: Path | None = None,
    fetch_text=None,
    now: dt.datetime | None = None,
    pause_seconds: float = USGS_REQUEST_PAUSE_SECONDS,
    force: bool = False,
) -> Path:
    """Download every USGS M2.5+ event of one calendar year (to ``now`` for the running year).

    Pages month by month and bisects any window whose response reaches the FDSN
    row limit, so the file is complete; the old single-request-per-year pull kept
    only the first 20,000 rows. Writes atomically with a fetch manifest, and RAISES
    on any failure -- no partial year is ever written or kept.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    dest = usgs_year_path(year, usgs_dir)
    if not force and _usgs_year_is_current(dest, year, now):
        print(f"  [USGS] {year} -- cached, complete ({dest.stat().st_size:,} bytes)")
        return dest
    start, end = _year_period(year, now)
    if start >= end:
        raise ValueError(f"year {year} has not started yet")

    fetch_text = fetch_text or _fetch_text_with_retry
    stats: dict = {}
    print(f"  [USGS] {year} -- downloading month by month ...")
    fieldnames, rows = usgs_fdsn.fetch_catalog(
        start, end, min_magnitude=USGS_MIN_MAGNITUDE, fetch_text=fetch_text,
        pause_seconds=pause_seconds, stats=stats,
    )
    if not rows:
        raise usgs_fdsn.USGSCatalogIncompleteError(f"USGS returned no events for {year}")
    problems = usgs_fdsn.audit_event_times(
        (r.get("time", "") for r in rows), period_start=start, period_end=end,
        proven_unclipped=stats.get("max_rows_per_response", 0) < usgs_fdsn.USGS_FDSN_ROW_LIMIT,
    )
    if problems:
        raise usgs_fdsn.USGSCatalogIncompleteError(f"USGS {year}: {'; '.join(problems)}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "year": year,
        "min_magnitude": USGS_MIN_MAGNITUDE,
        "coverage_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_events": len(rows),
        "row_limit": usgs_fdsn.USGS_FDSN_ROW_LIMIT,
        "requests": stats.get("requests", 0),
        "bisections": stats.get("splits", 0),
        "max_rows_per_response": stats.get("max_rows_per_response", 0),
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetcher": "hazardpulse.data.usgs_fdsn.fetch_catalog (monthly, bisect-on-limit)",
    }
    tmp.replace(dest)
    usgs_manifest_path(dest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  [USGS] {year} -- {len(rows):,} events ({manifest['requests']} requests, "
          f"{manifest['bisections']} bisections, max {manifest['max_rows_per_response']} rows/response)")
    return dest


def download_usgs_catalog(
    min_year: int = 2000,
    max_year: int = 2025,
    *,
    usgs_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Download (or repair) the USGS catalog year by year; raises if any year fails."""
    print("\n=== USGS Global Earthquake Catalog (M2.5+) ===")
    (usgs_dir or USGS_DIR).mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    paths: list[Path] = []
    failures: list[str] = []
    for year in range(min_year, min(max_year, now.year) + 1):
        try:
            paths.append(download_usgs_year(year, usgs_dir=usgs_dir, now=now, force=force))
        except Exception as exc:
            failures.append(f"{year}: {exc}")
            print(f"  [USGS] {year} FAILED: {exc}")
    if failures:
        raise RuntimeError("USGS catalog download incomplete -- " + " | ".join(failures))
    return paths


# ===================================================================
# 2.  GCMT Moment Tensor Catalog
# ===================================================================

GCMT_NDK_URLS = [
    # 1976-2017 monolithic file
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec17.ndk",
    # Monthly updates 2018-2024 (annual bundles on the GCMT site)
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2018/jan18.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2019/jan19.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2020/jan20.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2021/jan21.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2022/jan22.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2023/jan23.ndk",
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/2024/jan24.ndk",
]


def _parse_ndk_block(lines: list[str]) -> dict | None:
    """Parse a single 5-line NDK record into a dict.

    NDK format reference: https://www.globalcmt.org/CMTfiles.html
    Line 1: hypocenter reference
    Line 2: CMT info
    Line 3: centroid parameters
    Line 4: moment tensor elements
    Line 5: nodal planes
    """
    if len(lines) < 5:
        return None
    try:
        # Line 1: hypocenter
        h = lines[0]
        origin_time = _parse_ndk_origin_time(h)
        # Line 3: centroid -- lat, lon, depth
        c = lines[2].split()
        lat = float(c[3])
        lon = float(c[5])
        depth = float(c[7])

        # Line 4: exponent + moment tensor elements
        m = lines[3].split()
        exponent = float(m[0])

        # Line 5: nodal planes
        p = lines[4].split()
        # event id from line 2 (e.g. M010176A/C010576A). The trailing tokens are quality
        # metadata, not identifiers.
        event_id = lines[1].split()[0] if lines[1].split() else ""

        # Scalar moment from line 4 (last value * 10^exponent)
        # Actually the scalar moment isn't directly on line 4; we compute
        # from the eigenvalues.  For simplicity use the exponent and Mrr.
        scalar_moment = float(m[1]) * (10.0 ** exponent)
        mw = (2.0 / 3.0) * (math.log10(scalar_moment) - 9.1)

        # Nodal planes: strike/dip/rake for each
        # Line 5 format: strike1 dip1 rake1 strike2 dip2 rake2
        # The line starts with eigenvalue info then nodal planes at the end
        # Typical line 5: eigenvalues(6 vals) then strike1 dip1 rake1 strike2 dip2 rake2
        # Actually NDK line 5: "V10 ... V11 ... V12 strike1 dip1 rake1 strike2 dip2 rake2 ..."
        # The last 6 numeric tokens are the two nodal planes
        nums = [x for x in p if _is_float(x)]
        if len(nums) >= 6:
            strike1, dip1, rake1 = float(nums[-6]), float(nums[-5]), float(nums[-4])
            strike2, dip2, rake2 = float(nums[-3]), float(nums[-2]), float(nums[-1])
        else:
            strike1 = dip1 = rake1 = strike2 = dip2 = rake2 = float("nan")

        return {
            "event_id": event_id,
            "time": origin_time,
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "Mw": round(mw, 2),
            "strike1": strike1,
            "dip1": dip1,
            "rake1": rake1,
            "strike2": strike2,
            "dip2": dip2,
            "rake2": rake2,
            "scalar_moment": scalar_moment,
        }
    except (IndexError, ValueError):
        return None


def _parse_ndk_origin_time(line: str) -> str:
    """Parse the hypocenter date/time on NDK line 1 into an ISO-8601 UTC string."""
    parts = line.split()
    if len(parts) < 3:
        raise ValueError("NDK hypocenter line missing date/time")
    year, month, day = [int(x) for x in parts[1].split("/")]
    hour_s, minute_s, second_s = parts[2].split(":")
    base = dt.datetime(
        year,
        month,
        day,
        int(hour_s),
        int(minute_s),
        tzinfo=dt.timezone.utc,
    )
    origin = base + dt.timedelta(seconds=float(second_s))
    return origin.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_ndk_text(text: str) -> list[dict]:
    """Parse an entire NDK-format file into a list of dicts."""
    lines = text.strip().splitlines()
    records: list[dict] = []
    for i in range(0, len(lines) - 4, 5):
        rec = _parse_ndk_block(lines[i : i + 5])
        if rec is not None:
            records.append(rec)
    return records


def download_gcmt_catalog() -> Path:
    """Download GCMT NDK files and parse into a single CSV."""
    print("\n=== GCMT Moment Tensor Catalog ===")
    GCMT_DIR.mkdir(parents=True, exist_ok=True)
    dest = GCMT_DIR / "gcmt_catalog.csv"

    if dest.exists():
        header = dest.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if "time" in {h.strip() for h in header.split(",")}:
            print(f"  [GCMT] cached ({dest.stat().st_size:,} bytes)")
            return dest
        print("  [GCMT] cached catalog is legacy/no-time; rebuilding from NDK sources")

    all_records: list[dict] = []
    for url in GCMT_NDK_URLS:
        fname = url.rsplit("/", 1)[-1]
        ndk_cache = GCMT_DIR / fname
        if ndk_cache.exists():
            print(f"  [GCMT] {fname} -- cached")
            text = ndk_cache.read_text(encoding="utf-8", errors="replace")
        else:
            print(f"  [GCMT] {fname} -- downloading ...")
            try:
                text = _fetch_text(url, timeout=300)
                ndk_cache.write_text(text, encoding="utf-8")
            except Exception as exc:
                print(f"  [GCMT] {fname} FAILED: {exc}")
                continue
            time.sleep(1.0)

        records = _parse_ndk_text(text)
        all_records.extend(records)
        print(f"  [GCMT] {fname} -- {len(records):,} events parsed")

    if not all_records:
        print("  [GCMT] WARNING: no records parsed")
        return dest

    # Write CSV
    fieldnames = [
        "event_id", "time", "lat", "lon", "depth", "Mw",
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
        "scalar_moment",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_records)
    dest.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  [GCMT] total: {len(all_records):,} events -> {dest}")
    return dest


# ===================================================================
# 3.  Plate Boundary Model (PB2002)
# ===================================================================

PB2002_BOUNDARIES = (
    "https://raw.githubusercontent.com/fraxen/tectonicplates"
    "/master/GeoJSON/PB2002_boundaries.json"
)
PB2002_PLATES = (
    "https://raw.githubusercontent.com/fraxen/tectonicplates"
    "/master/GeoJSON/PB2002_plates.json"
)


def download_plate_boundaries() -> tuple[Path, Path]:
    """Download PB2002 plate boundary GeoJSON files."""
    print("\n=== PB2002 Plate Boundary Model ===")
    PLATES_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for url, fname in [
        (PB2002_BOUNDARIES, "pb2002_boundaries.json"),
        (PB2002_PLATES, "pb2002_plates.json"),
    ]:
        dest = PLATES_DIR / fname
        if dest.exists():
            print(f"  [PB2002] {fname} -- cached")
        else:
            print(f"  [PB2002] {fname} -- downloading ...")
            try:
                data = _fetch_text(url)
                dest.write_text(data, encoding="utf-8")
                print(f"  [PB2002] {fname} -- saved ({dest.stat().st_size:,} bytes)")
            except Exception as exc:
                print(f"  [PB2002] {fname} FAILED: {exc}")
        paths.append(dest)
    return paths[0], paths[1]


# ===================================================================
# 4.  Nevada Geodetic Lab GPS Time Series
# ===================================================================

# Correct NGL layout (verified 2026-06): /gps_timeseries/IGS14/tenv3/IGS14/<SSSS>.tenv3
# (the old /gps_timeseries/tenv3/IGS14/ path 404s -- that was the bug behind 99/99 fails).
NGL_BASE = "https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14"
NGL_STATION_LIST = "https://geodesy.unr.edu/NGLStationPages/llh.out"

# Hand-curated stations near major fault zones.
# San Andreas / Cascadia / Basin & Range
NA_WEST_STATIONS = [
    "P034", "P035", "P038", "P041", "P042", "P156", "P159", "P160",
    "P163", "P166", "GOLD", "GRAS", "JPLM", "MENT", "OJAI", "PVER",
    "VTIS", "WRHS", "VNDN", "BLYT", "BRIB", "CMBB", "DHLG", "ELKO",
    "FERN", "LIND", "MUSB", "QUIN", "SLID", "WDCB",
]

# Japan subduction zone
JAPAN_STATIONS = [
    "0183", "0203", "0550", "0595", "0946", "0947", "1199", "3009",
    "3023", "3051", "MIZU", "USUD", "TSKB", "KGNI", "AIRA",
]

# Chile / South America subduction
CHILE_STATIONS = [
    "ANTC", "AREQ", "CONZ", "COPO", "IQQE", "SANT", "UFPR", "VALP",
    "LPGS", "CORD",
]

# Sumatra / Southeast Asia
SUMATRA_STATIONS = [
    "BAKO", "COCO", "DGAR", "NTUS", "PIMO", "XMIS", "KUNM", "LHAZ",
    "IISC", "HYDE",
]

# Turkey / Mediterranean / Middle East
TURKEY_STATIONS = [
    "ANKR", "ISTA", "TUBI", "NICO", "RAMO", "DRAG", "TRAB", "IZMI",
    "MATE", "GRAZ",
]

# Italy / Alps
ITALY_STATIONS = [
    "CAGL", "GENO", "GRAS", "LAMP", "UNPG", "AQUI", "MEDI", "WTZR",
    "ZIMM", "POTS",
]

# New Zealand / SW Pacific
NZ_STATIONS = [
    "AUCK", "CHAT", "DNVK", "GISB", "HIKB", "MQZG", "WGTN", "VESL",
    "KARR", "TOW2",
]

# Alaska / Aleutians
ALASKA_STATIONS = [
    "PRIOR", "AB50", "AB48", "AC61", "AC60",
]

ALL_CURATED_STATIONS: list[str] = sorted(set(
    NA_WEST_STATIONS + JAPAN_STATIONS + CHILE_STATIONS +
    SUMATRA_STATIONS + TURKEY_STATIONS + ITALY_STATIONS +
    NZ_STATIONS + ALASKA_STATIONS
))


def _download_one_station(station: str) -> tuple[str, bool, str]:
    """Download a single GNSS .tenv3 file.  Returns (station, ok, msg)."""
    dest = GNSS_DIR / f"{station}.tenv3"
    if dest.exists():
        return station, True, "cached"

    url = f"{NGL_BASE}/{station}.tenv3"
    try:
        data = _fetch_text(url, timeout=60)
        if len(data) < 100:
            return station, False, "empty response"
        dest.write_text(data, encoding="utf-8")
        return station, True, f"{len(data):,} bytes"
    except urllib.error.HTTPError as exc:
        return station, False, f"HTTP {exc.code}"
    except Exception as exc:
        return station, False, str(exc)


def download_gnss_stations(
    stations: list[str] | None = None,
    max_workers: int = 10,
) -> dict[str, bool]:
    """Download GNSS time series for stations using a thread pool."""
    if stations is None:
        stations = ALL_CURATED_STATIONS

    print(f"\n=== Nevada Geodetic Lab GPS Time Series ({len(stations)} stations) ===")
    GNSS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    done_count = 0
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one_station, s): s
            for s in stations
        }
        for future in as_completed(futures):
            station, ok, msg = future.result()
            results[station] = ok
            done_count += 1
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            if done_count % 10 == 0 or done_count == len(stations):
                print(
                    f"  [GNSS] {done_count}/{len(stations)} "
                    f"(ok={ok_count}, fail={fail_count})"
                )

    print(f"  [GNSS] complete: {ok_count} downloaded, {fail_count} failed")
    return results


# ===================================================================
# 5.  Ionospheric TEC (placeholder)
# ===================================================================

def note_tec_status() -> None:
    """Print status for TEC data (requires Earthdata auth)."""
    print("\n=== Ionospheric TEC ===")
    print("  [TEC] skipped (requires NASA Earthdata auth)")
    print("  [TEC] future enhancement: download CODE/IGS GIM TEC maps")


# ===================================================================
# Summary
# ===================================================================

def print_summary(usgs_dir: Path | None = None) -> None:
    """Print cache summary statistics."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    # USGS
    usgs_root = usgs_dir or USGS_DIR
    usgs_files = sorted(usgs_root.glob("usgs_catalog_*.csv")) if usgs_root.exists() else []
    total_events = 0
    for f in usgs_files:
        lines = f.read_text(encoding="utf-8", errors="replace").count("\n") - 1
        total_events += max(0, lines)
    print(f"  USGS catalog:  {len(usgs_files)} year files, ~{total_events:,} events")

    # GCMT
    gcmt_path = GCMT_DIR / "gcmt_catalog.csv"
    if gcmt_path.exists():
        n = gcmt_path.read_text(encoding="utf-8", errors="replace").count("\n") - 1
        print(f"  GCMT catalog:  {max(0, n):,} moment tensor records")
    else:
        print("  GCMT catalog:  not downloaded")

    # Plates
    for fname in ["pb2002_boundaries.json", "pb2002_plates.json"]:
        p = PLATES_DIR / fname
        status = f"{p.stat().st_size:,} bytes" if p.exists() else "not downloaded"
        print(f"  PB2002 {fname}: {status}")

    # GNSS
    gnss_files = sorted(GNSS_DIR.glob("*.tenv3")) if GNSS_DIR.exists() else []
    print(f"  GNSS stations: {len(gnss_files)} cached")

    # TEC
    print("  TEC:           skipped (requires Earthdata auth)")

    total_bytes = 0
    if CACHE_ROOT.exists():
        for f in CACHE_ROOT.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
    print(f"\n  Total cache size: {total_bytes / 1_048_576:.1f} MB")
    print(f"  Cache location:  {CACHE_ROOT}")


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download earthquake data sources for HazardPulse",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode: 1 year of USGS + 5 GNSS stations",
    )
    parser.add_argument(
        "--usgs-only",
        action="store_true",
        help="Download only the USGS earthquake catalog",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Start year for USGS catalog (default: 2000)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2025,
        help="End year for USGS catalog (default: 2025)",
    )
    parser.add_argument(
        "--gnss-workers",
        type=int,
        default=10,
        help="Thread pool workers for GNSS downloads (default: 10)",
    )
    parser.add_argument(
        "--usgs-dir",
        type=Path,
        default=None,
        help="Write the USGS year files here instead of the default cache "
        "(e.g. build a staging copy, audit it, then swap it in)",
    )
    parser.add_argument(
        "--force-usgs",
        action="store_true",
        help="Re-pull every requested USGS year even if the cached file audits complete",
    )
    parser.add_argument(
        "--audit-usgs",
        action="store_true",
        help="Only audit the cached USGS year files and exit non-zero if any is incomplete",
    )
    args = parser.parse_args()
    usgs_kw = {"usgs_dir": args.usgs_dir, "force": args.force_usgs}

    if args.audit_usgs:
        bad = 0
        for year in range(args.min_year, args.max_year + 1):
            problems = audit_usgs_year_file(usgs_year_path(year, args.usgs_dir), year)
            status = "ok" if not problems else "INCOMPLETE: " + "; ".join(problems)
            bad += bool(problems)
            print(f"  {year}: {status}")
        raise SystemExit(1 if bad else 0)

    t0 = time.time()
    print("HazardPulse Earthquake Data Downloader")
    print(f"Cache directory: {CACHE_ROOT}")

    if args.test:
        # Quick smoke test
        print("\n*** TEST MODE: limited download ***")
        download_usgs_catalog(min_year=2024, max_year=2024, **usgs_kw)
        download_plate_boundaries()
        test_stations = ALL_CURATED_STATIONS[:5]
        download_gnss_stations(stations=test_stations, max_workers=3)
        note_tec_status()

    elif args.usgs_only:
        download_usgs_catalog(min_year=args.min_year, max_year=args.max_year, **usgs_kw)

    else:
        # Full download
        download_usgs_catalog(min_year=args.min_year, max_year=args.max_year, **usgs_kw)
        download_gcmt_catalog()
        download_plate_boundaries()
        download_gnss_stations(max_workers=args.gnss_workers)
        note_tec_status()

    elapsed = time.time() - t0
    print_summary(args.usgs_dir)
    print(f"\nCompleted in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
