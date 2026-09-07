"""Earthquake data loaders for HazardPulse.

Reads cached files produced by ``scripts/download_earthquake_data.py``.

Functions
---------
load_usgs_catalog   -- USGS M2.5+ earthquake events (CSV per year)
load_gcmt_catalog   -- GCMT moment tensor catalog (single CSV)
load_plate_boundaries -- PB2002 plate boundary GeoJSON
load_gnss_timeseries  -- GPS displacement time series (.tenv3)
scan_gnss_cache     -- List available cached GPS station codes
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Cache layout (mirrors download_earthquake_data.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "earthquake"

USGS_DIR = CACHE_ROOT / "usgs"
GCMT_DIR = CACHE_ROOT / "gcmt"
PLATES_DIR = CACHE_ROOT / "plates"
GNSS_DIR = CACHE_ROOT / "gnss"

_SSL_CTX = ssl.create_default_context()
_USER_AGENT = "hazardpulse/0.1 (+https://github.com/Jphilbrick10/hazardpulse)"
_USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"


# ---------------------------------------------------------------------------
# 1. USGS Global Earthquake Catalog
# ---------------------------------------------------------------------------

_USGS_FLOAT_FIELDS = {"latitude", "longitude", "depth", "mag", "dmin", "gap", "rms"}
_USGS_INT_FIELDS = {"nst", "tsunami", "sig"}
_USGS_KEEP_FIELDS = {
    "time", "latitude", "longitude", "depth", "mag",
    "magType", "place", "type", "id",
}


def _fetch_text(url: str, timeout: int = 180) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _month_start(year: int, month: int) -> _dt.date:
    return _dt.date(year, month, 1)


def _next_month_start(year: int, month: int) -> _dt.date:
    if month == 12:
        return _dt.date(year + 1, 1, 1)
    return _dt.date(year, month + 1, 1)


def bootstrap_usgs_catalog(
    min_year: int = 2000,
    max_year: int = 2025,
    *,
    force: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Download yearly USGS catalog files into the local cache.

    The USGS CSV endpoint caps a single query at 20,000 rows, so we fetch one
    calendar month at a time and stitch the rows back into a yearly CSV.
    """
    USGS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for year in range(min_year, max_year + 1):
        dest = USGS_DIR / f"usgs_catalog_{year}.csv"
        if dest.exists() and not force:
            written.append(dest)
            continue

        header: str | None = None
        rows: list[str] = []
        total_rows = 0

        for month in range(1, 13):
            start = _month_start(year, month)
            end = _next_month_start(year, month)
            url = (
                f"{_USGS_API}?format=csv"
                f"&starttime={start.isoformat()}"
                f"&endtime={end.isoformat()}"
                f"&minmagnitude=2.5"
                f"&orderby=time-asc"
                f"&limit=20000"
            )
            if verbose:
                print(
                    f"  [USGS bootstrap] {year}-{month:02d} "
                    f"{start.isoformat()}..{end.isoformat()}"
                )
            text = _fetch_text(url)
            lines = text.splitlines()
            if not lines:
                continue

            month_header = lines[0]
            month_rows = lines[1:]
            if header is None:
                header = month_header
            rows.extend(month_rows)
            total_rows += len(month_rows)
            time.sleep(0.1)

        if header is None:
            continue

        dest.write_text(
            header + "\n" + "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        if verbose:
            print(f"  [USGS bootstrap] {year} wrote {total_rows:,} rows to {dest}")
        written.append(dest)

    return written


def _coerce_usgs_row(row: dict[str, str]) -> dict:
    """Convert string values from the CSV to appropriate Python types."""
    out: dict = {}
    for key, val in row.items():
        if key not in _USGS_KEEP_FIELDS:
            continue
        if val == "" or val is None:
            out[key] = None
        elif key in _USGS_FLOAT_FIELDS:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = None
        elif key in _USGS_INT_FIELDS:
            try:
                out[key] = int(float(val))
            except ValueError:
                out[key] = None
        else:
            out[key] = val
    return out


def load_usgs_catalog(
    min_year: int = 2000,
    max_year: int = 2025,
    min_mag: float = 2.5,
) -> list[dict]:
    """Load cached USGS earthquake catalog.

    Parameters
    ----------
    min_year, max_year : int
        Range of years to include (inclusive).
    min_mag : float
        Minimum magnitude filter (applied on load).

    Returns
    -------
    list[dict]
        Each dict has keys: time, latitude, longitude, depth, mag,
        magType, place, type, id.  Numeric fields are float/int.
    """
    expected_paths = [USGS_DIR / f"usgs_catalog_{year}.csv" for year in range(min_year, max_year + 1)]
    if any(not path.exists() for path in expected_paths):
        bootstrap_usgs_catalog(min_year=min_year, max_year=max_year, verbose=True)

    events: list[dict] = []
    for year in range(min_year, max_year + 1):
        path = USGS_DIR / f"usgs_catalog_{year}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rec = _coerce_usgs_row(row)
                mag = rec.get("mag")
                if mag is not None and mag >= min_mag:
                    events.append(rec)
    return events


# ---------------------------------------------------------------------------
# 2. GCMT Moment Tensor Catalog
# ---------------------------------------------------------------------------

_GCMT_FLOAT_FIELDS = {
    "lat", "lon", "depth", "Mw",
    "strike1", "dip1", "rake1",
    "strike2", "dip2", "rake2",
    "scalar_moment",
}


def _coerce_gcmt_row(row: dict[str, str]) -> dict:
    out: dict = {}
    for key, val in row.items():
        if val == "" or val is None:
            out[key] = None
        elif key in _GCMT_FLOAT_FIELDS:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = None
        else:
            out[key] = val
    return out


def load_gcmt_catalog() -> list[dict]:
    """Load cached GCMT moment tensor catalog.

    Returns
    -------
    list[dict]
        Keys: event_id, lat, lon, depth, Mw, strike1, dip1, rake1,
        strike2, dip2, rake2, scalar_moment.
    """
    path = GCMT_DIR / "gcmt_catalog.csv"
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(_coerce_gcmt_row(row))
    return records


# ---------------------------------------------------------------------------
# 3. PB2002 Plate Boundary Model
# ---------------------------------------------------------------------------

def load_plate_boundaries() -> dict:
    """Load PB2002 plate boundary GeoJSON.

    Returns
    -------
    dict
        A dict with keys ``"boundaries"`` and ``"plates"``, each containing
        the parsed GeoJSON FeatureCollection (or ``None`` if not cached).
    """
    result: dict = {"boundaries": None, "plates": None}
    for key, fname in [
        ("boundaries", "pb2002_boundaries.json"),
        ("plates", "pb2002_plates.json"),
    ]:
        path = PLATES_DIR / fname
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                result[key] = json.load(fh)
    return result


# ---------------------------------------------------------------------------
# 4. Nevada Geodetic Lab GPS Time Series
# ---------------------------------------------------------------------------

def load_gnss_timeseries(station: str) -> list[dict] | None:
    """Load GPS time series for a station from cached .tenv3 file.

    Parameters
    ----------
    station : str
        Four-character station code (e.g. ``"P034"``).

    Returns
    -------
    list[dict] | None
        Each dict has keys: date, year_fraction, north_mm, east_mm,
        up_mm, north_sig, east_sig, up_sig.  Returns ``None`` if the
        station file is not cached.

    Notes
    -----
    .tenv3 format columns (space-delimited):
      station  date  year_frac  reflon  reflat  dN  dE  dU  sigN  sigE  sigU  ...
    We extract the displacement components (dN, dE, dU) in mm and their
    uncertainties.
    """
    path = GNSS_DIR / f"{station}.tenv3"
    if not path.exists():
        return None

    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("station"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                records.append({
                    "date": parts[1],
                    "year_fraction": float(parts[2]),
                    "north_mm": float(parts[5]),
                    "east_mm": float(parts[6]),
                    "up_mm": float(parts[7]),
                    "north_sig": float(parts[8]),
                    "east_sig": float(parts[9]),
                    "up_sig": float(parts[10]),
                })
            except (ValueError, IndexError):
                continue
    return records if records else None


def scan_gnss_cache() -> list[str]:
    """List available cached GPS station codes.

    Returns
    -------
    list[str]
        Sorted list of station codes (e.g. ``["ANKR", "AREQ", ...]``).
    """
    if not GNSS_DIR.exists():
        return []
    return sorted(
        p.stem for p in GNSS_DIR.glob("*.tenv3") if p.stat().st_size > 100
    )
