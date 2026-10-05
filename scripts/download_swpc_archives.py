#!/usr/bin/env python3
"""Download canonical historical space-weather archives.

Pulls four datasets that together cover any quake/storm timestamp from
1995-present:

  - Kp/ap definitive (GFZ Potsdam, since 1932)            -> kp_definitive.npz
  - Dst hourly (WDC Kyoto, since 1957)                    -> dst_definitive.npz
  - OMNI hourly IMF + plasma (NASA SPDF, since 1963)      -> omni_hourly.npz
  - GOES X-ray flare events (NCEI, since 1986)            -> goes_xray_events.npz

Each .npz holds a single ``arr`` 2-D float64 table; column 0 is always
epoch seconds (UTC), column 1+ is the value(s).

Run once (or whenever you want to refresh the recent years):

    python scripts/download_swpc_archives.py --since 1995

The cache lives at ``.cache/swpc/`` (override with $HAZARDPULSE_SWPC_CACHE).
Total download size: ~250 MB. Idempotent — already-downloaded yearly
files are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.space_weather import SWPC_CACHE_ROOT  # noqa: E402


def _http_get(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "HazardPulse/1.0 (research)"}
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed for {url}: {last_exc}")


def _epoch(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    return calendar.timegm((year, month, day, hour, minute, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Kp definitive (GFZ Potsdam)
# Format: fixed-width text, one row per 3-hour Kp value back to 1932.
# ---------------------------------------------------------------------------

def download_kp_definitive(since_year: int) -> np.ndarray:
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt"
    print(f"  Downloading Kp definitive archive...")
    raw = _http_get(url)
    text = raw.decode("utf-8", errors="replace")

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Format (post-2018): YYYY MM DD HH.H DAYS DAYS_M Kp ap D
        if len(parts) < 7:
            continue
        try:
            year = int(parts[0])
            if year < since_year:
                continue
            month = int(parts[1])
            day = int(parts[2])
            hour_frac = float(parts[3])
            kp = float(parts[7]) if parts[7] not in ("-1", "-1.000") else None
            if kp is None or kp < 0:
                continue
            hour = int(hour_frac)
            t = _epoch(year, month, day, hour)
            rows.append((t, kp))
        except (ValueError, IndexError):
            continue

    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)
    print(f"  Kp definitive: {len(arr)} records since {since_year}")
    return arr


# ---------------------------------------------------------------------------
# Dst hourly (WDC Kyoto)
# Format: per-year text dumps. We pull the index file then the yearly files.
# ---------------------------------------------------------------------------

def download_dst_definitive(since_year: int, end_year: int) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for year in range(since_year, end_year + 1):
        url = f"https://wdc.kugi.kyoto-u.ac.jp/dst{('_provisional' if year >= end_year - 2 else '_final')}/{year}/index.html"
        # Try alternate ASCII data url (more reliable):
        ascii_url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}{1:02d}/index.html"
        # The actual reliable path is the per-month IAGA-2002 ascii dump:
        for month in range(1, 13):
            mm = f"{month:02d}"
            data_url = (
                "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/"
                f"{year}{mm}/dst{str(year)[2:]}{mm}.for.request"
            )
            try:
                raw = _http_get(data_url, timeout=30, retries=2)
                text = raw.decode("ascii", errors="replace")
            except Exception:
                continue
            # Format: lines like "DST9001*01RRX020   -16  -19 ..." 24 hourly values
            for line in text.splitlines():
                if not line.startswith("DST"):
                    continue
                try:
                    line_year = 1900 + int(line[3:5])
                    if line_year < 50:
                        line_year += 2000
                    if line_year > 2050:
                        line_year -= 100
                    line_month = int(line[5:7])
                    line_day = int(line[8:10])
                except (ValueError, IndexError):
                    continue
                # 24 hourly values starting at column 20, each 4 chars
                for h in range(24):
                    start = 20 + h * 4
                    if start + 4 > len(line):
                        break
                    val_str = line[start:start + 4].strip()
                    try:
                        dst_val = int(val_str)
                    except ValueError:
                        continue
                    if dst_val == 9999:
                        continue
                    t = _epoch(line_year, line_month, line_day, h)
                    rows.append((t, float(dst_val)))
        print(f"    Dst {year}: cumulative {len(rows)} hourly records")

    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)
    if len(arr) > 0:
        # Sort + dedup by time
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        _, idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[np.sort(idx)]
    print(f"  Dst hourly: {len(arr)} records since {since_year}")
    return arr


# ---------------------------------------------------------------------------
# OMNI hourly IMF + plasma (NASA SPDF / GSFC)
# Format: fixed-width per-year text files (omni2_YYYY.dat).
# Columns of interest: bz_gsm (col 17), sw_speed (col 25), proton_density (col 24)
# Reference: https://omniweb.gsfc.nasa.gov/html/ow_data.html#3
# ---------------------------------------------------------------------------

def download_omni_hourly(since_year: int, end_year: int) -> np.ndarray:
    # OMNI low-res hourly format documented at:
    #   https://omniweb.gsfc.nasa.gov/html/ow_data.html#4
    # Field positions (1-based) for the columns we care about:
    #   1: year, 2: doy, 3: hour, 17: Bz GSM (nT), 25: SW plasma speed (km/s),
    #   24: SW proton density (n/cc)
    rows: list[tuple[float, float, float, float]] = []
    for year in range(since_year, end_year + 1):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        try:
            raw = _http_get(url, timeout=120, retries=2)
        except Exception as exc:
            print(f"    OMNI {year}: skipped ({exc})")
            continue
        text = raw.decode("ascii", errors="replace")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 30:
                continue
            try:
                yr = int(parts[0])
                doy = int(parts[1])
                hh = int(parts[2])
                bz = float(parts[16])     # column 17 (0-indexed 16)
                density = float(parts[23])  # column 24
                speed = float(parts[24])    # column 25
            except (ValueError, IndexError):
                continue
            # Sentinel "no data" values:
            if bz >= 999.9:
                bz = float("nan")
            if speed >= 9999:
                speed = float("nan")
            if density >= 999:
                density = float("nan")
            try:
                t = _epoch(yr, 1, 1, hh) + (doy - 1) * 86400
            except Exception:
                continue
            rows.append((t, bz, speed, density))
        print(f"    OMNI {year}: cumulative {len(rows)} hourly records")
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)
    print(f"  OMNI hourly: {len(arr)} records since {since_year}")
    return arr


# ---------------------------------------------------------------------------
# GOES X-ray flare events (NCEI)
# Format: per-year text report, one row per identified M+/X-class flare.
# ---------------------------------------------------------------------------

def download_goes_xray_events(since_year: int, end_year: int) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for year in range(since_year, end_year + 1):
        # NCEI keeps two filename conventions; try both:
        urls = [
            f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{year}.txt",
            f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{year}-ytd.txt",
        ]
        text = None
        for url in urls:
            try:
                text = _http_get(url, timeout=30, retries=2).decode("ascii", errors="replace")
                break
            except Exception:
                continue
        if text is None:
            print(f"    GOES X-ray {year}: not available")
            continue
        # Each row is fixed-width with date in cols 6-11 (YYMMDD) and
        # peak class somewhere on the line (e.g. "X1.2" or "M3.5").
        for line in text.splitlines():
            if len(line) < 30:
                continue
            try:
                date_token = line[5:11]
                if not date_token.isdigit():
                    continue
                yy = int(date_token[:2])
                mm = int(date_token[2:4])
                dd = int(date_token[4:6])
                line_year = 2000 + yy if yy < 50 else 1900 + yy
            except ValueError:
                continue
            m = re.search(r"\b([MX])(\d{1,2}\.?\d?)\b", line)
            if not m:
                continue
            letter = m.group(1)
            try:
                mag = float(m.group(2))
            except ValueError:
                continue
            base = 3.0 if letter == "M" else 4.0
            class_num = base + max(0.0, np.log10(max(mag, 0.1)) - 1.0)
            t = _epoch(line_year, mm, dd, 12)  # noon as approx; flare peak unknown w/o full parse
            rows.append((t, float(class_num)))
        print(f"    GOES X-ray {year}: cumulative {len(rows)} M+/X+ flares")
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2), dtype=np.float64)
    if len(arr) > 0:
        order = np.argsort(arr[:, 0])
        arr = arr[order]
    print(f"  GOES X-ray events: {len(arr)} flares since {since_year}")
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", type=int, default=2005,
                        help="Earliest year to download (default 2005)")
    parser.add_argument("--end", type=int, default=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).year,
                        help="Latest year (default current year)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cache file exists")
    parser.add_argument("--skip", default="",
                        help="Comma-list of datasets to skip: kp,dst,omni,xray")
    args = parser.parse_args(argv)

    SWPC_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip().lower() for s in args.skip.split(",") if s.strip())

    print(f"SWPC archives -> {SWPC_CACHE_ROOT}")
    print(f"Window: {args.since}-{args.end}")
    print()

    if "kp" not in skip:
        out = SWPC_CACHE_ROOT / "kp_definitive.npz"
        if args.force or not out.exists():
            arr = download_kp_definitive(args.since)
            np.savez_compressed(str(out), arr=arr)
            print(f"  Saved {out} ({out.stat().st_size / 1024:.1f} KB)\n")
        else:
            print(f"  Kp cache exists, skipping ({out})\n")

    if "dst" not in skip:
        out = SWPC_CACHE_ROOT / "dst_definitive.npz"
        if args.force or not out.exists():
            arr = download_dst_definitive(args.since, args.end)
            np.savez_compressed(str(out), arr=arr)
            print(f"  Saved {out} ({out.stat().st_size / 1024:.1f} KB)\n")
        else:
            print(f"  Dst cache exists, skipping ({out})\n")

    if "omni" not in skip:
        out = SWPC_CACHE_ROOT / "omni_hourly.npz"
        if args.force or not out.exists():
            arr = download_omni_hourly(args.since, args.end)
            np.savez_compressed(str(out), arr=arr)
            print(f"  Saved {out} ({out.stat().st_size / 1024:.1f} KB)\n")
        else:
            print(f"  OMNI cache exists, skipping ({out})\n")

    if "xray" not in skip:
        out = SWPC_CACHE_ROOT / "goes_xray_events.npz"
        if args.force or not out.exists():
            arr = download_goes_xray_events(args.since, args.end)
            np.savez_compressed(str(out), arr=arr)
            print(f"  Saved {out} ({out.stat().st_size / 1024:.1f} KB)\n")
        else:
            print(f"  X-ray cache exists, skipping ({out})\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
