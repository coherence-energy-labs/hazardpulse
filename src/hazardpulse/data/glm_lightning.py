"""NOAA GOES-19 GLM (Geostationary Lightning Mapper) L2 fetcher.

Reads 20-second GLM files from the public AWS Open Data bucket
``s3://noaa-goes19/GLM-L2-LCFA/`` and subsets to a lat/lon bounding box.
Each file contains arrays of ``event``, ``group``, and ``flash`` records
— we use ``flash`` (the coarsest aggregate, one record per lightning
discharge) as the unit of analysis.

Feature block L (lightning) — shared by tornado, hurricane scorers:

  - ``ltg_flash_count_Xh``     # total flashes in X-hour window over bbox
  - ``ltg_flash_rate``         # flashes / minute
  - ``ltg_area_coverage``      # fraction of bbox cells with >=1 flash
  - ``ltg_peak_intensity``     # 99th-percentile single-flash energy
  - ``ltg_flash_density``      # flashes / km^2
  - ``ltg_cg_percentage``      # cloud-ground vs total (when flags present)

The fetcher is bbox-aware: it never downloads the full continental GLM
stream. Subsetting happens at read time (filter by event_lat/lon before
keeping the record). Typical query for one storm bbox / 1h window is
~180 files × ~50-200 KB = ~15-35 MB.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import re
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GLM_CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_GLM_CACHE",
    str(PROJECT_ROOT / ".cache" / "glm"),
))

# Primary GOES-East GLM (active 2023-): GOES-19
# East coverage: ~75W (Atlantic/E. Caribbean/US East Coast/Central)
GLM_BUCKETS = {
    "goes19": "https://noaa-goes19.s3.amazonaws.com",
    "goes18": "https://noaa-goes18.s3.amazonaws.com",  # West Pacific coverage
    "goes16": "https://noaa-goes16.s3.amazonaws.com",  # Historical (2017-2024)
}

# Filename: OR_GLM-L2-LCFA_G19_sYYYYJJJHHMMSSS_eYYYYJJJHHMMSSS_cYYYYJJJHHMMSSS.nc
_GLM_FILE_RE = re.compile(
    r"OR_GLM-L2-LCFA_G\d+_s(\d{14})_e(\d{14})_c\d{14}\.nc$"
)


BLOCK_L_NAMES: list[str] = [
    "ltg_flash_count_1h",     # total in last 1 hour over bbox
    "ltg_flash_count_6h",     # 6-hour count
    "ltg_flash_rate_per_min", # flashes per minute (last hour)
    "ltg_area_coverage",      # fraction of 10km grid cells hit
    "ltg_peak_energy_J",      # 99th percentile flash energy
    "ltg_flash_density_per_km2",  # flashes / km^2
    "ltg_jump_ratio",         # last-15-min rate / 60-min rate (flash jump proxy)
]
N_FEAT_L: int = len(BLOCK_L_NAMES)


def _empty_block_l() -> dict[str, float]:
    return {name: float("nan") for name in BLOCK_L_NAMES}


# ---------------------------------------------------------------------------
# S3 listing + file download
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "HazardPulse/1.0 (research)"}
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def _list_glm_files(
    satellite: str,
    year: int,
    doy: int,
    hour: int,
) -> list[str]:
    """List all GLM files for (satellite, year, day-of-year, hour).

    Returns S3 object keys (without bucket prefix).
    """
    base = GLM_BUCKETS[satellite]
    prefix = f"GLM-L2-LCFA/{year}/{doy:03d}/{hour:02d}/"
    url = f"{base}/?list-type=2&prefix={prefix}&max-keys=1000"
    try:
        xml = _http_get(url, timeout=20).decode("utf-8", errors="replace")
    except Exception:
        return []
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def _filename_epoch(name: str) -> tuple[float, float]:
    """Return (start_epoch, end_epoch) parsed from GLM filename."""
    m = _GLM_FILE_RE.search(name)
    if not m:
        return 0.0, 0.0
    s_str, e_str = m.group(1), m.group(2)

    def _parse(ts: str) -> float:
        yr = int(ts[0:4])
        doy = int(ts[4:7])
        hh = int(ts[7:9])
        mm = int(ts[9:11])
        ss = int(ts[11:13])
        # decisecond
        d0 = dt.datetime(yr, 1, 1, hh, mm, ss) + dt.timedelta(days=doy - 1)
        return d0.replace(tzinfo=dt.timezone.utc).timestamp()

    return _parse(s_str), _parse(e_str)


def _cache_path_for_key(key: str) -> Path:
    """Local cache path for a GLM S3 key."""
    return GLM_CACHE_ROOT / key


def _download_glm_file(satellite: str, key: str) -> Path | None:
    """Download one GLM file to local cache. Returns path or None on failure."""
    local = _cache_path_for_key(key)
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    base = GLM_BUCKETS[satellite]
    url = f"{base}/{key}"
    try:
        data = _http_get(url, timeout=30)
    except Exception as exc:
        print(f"  GLM download failed for {key}: {exc}")
        return None
    local.write_bytes(data)
    return local


# ---------------------------------------------------------------------------
# NetCDF read helpers
# ---------------------------------------------------------------------------

def _read_glm_flashes(
    path: Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Read one GLM file, return dict of flash arrays (bbox-filtered)."""
    if not HAS_XARRAY:
        raise RuntimeError("xarray required to read GLM NetCDF files")
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        try:
            ds = xr.open_dataset(path, engine="netcdf4")
        except Exception as exc:
            print(f"  GLM read failed for {path.name}: {exc}")
            return {}

    try:
        lat = ds["flash_lat"].values.astype(np.float32)
        lon = ds["flash_lon"].values.astype(np.float32)
        energy = ds["flash_energy"].values.astype(np.float32)  # Joules
        area = ds["flash_area"].values.astype(np.float32)       # m^2
        # GLM time of occurrence is seconds since 2000-01-01 12:00:00 UTC
        time_offset_sec = ds["flash_time_offset_of_first_event"].values.astype(np.float64)
        epoch_base = dt.datetime(2000, 1, 1, 12, 0, 0).replace(
            tzinfo=dt.timezone.utc
        ).timestamp()
        flash_epoch = epoch_base + time_offset_sec
    finally:
        ds.close()

    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        lat = lat[mask]
        lon = lon[mask]
        energy = energy[mask]
        area = area[mask]
        flash_epoch = flash_epoch[mask]

    return {
        "lat": lat,
        "lon": lon,
        "energy": energy,
        "area": area,
        "epoch": flash_epoch,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FlashCollection:
    lat: np.ndarray
    lon: np.ndarray
    energy: np.ndarray   # Joules
    area: np.ndarray     # m^2
    epoch: np.ndarray    # UTC epoch seconds

    def __len__(self) -> int:
        return len(self.lat)

    @classmethod
    def empty(cls) -> "FlashCollection":
        z = np.zeros(0, dtype=np.float32)
        return cls(lat=z, lon=z, energy=z, area=z, epoch=np.zeros(0, dtype=np.float64))


def fetch_glm_flashes(
    end_time: dt.datetime,
    *,
    window_h: float = 1.0,
    bbox: tuple[float, float, float, float] | None = None,
    satellite: str = "goes19",
    max_files: int = 240,  # 240 × 20s = 80 minutes
) -> FlashCollection:
    """Fetch GLM flashes in (end_time - window_h, end_time], optionally bbox-filtered.

    Parameters
    ----------
    end_time : datetime (naive UTC)
    window_h : float
        Hours before end_time to include.
    bbox : (lat_min, lat_max, lon_min, lon_max) or None
    satellite : "goes19" (default) | "goes18" | "goes16"
    max_files : hard cap on files fetched (safety)
    """
    if not HAS_XARRAY:
        return FlashCollection.empty()

    end_epoch = end_time.replace(tzinfo=dt.timezone.utc).timestamp()
    start_epoch = end_epoch - window_h * 3600

    # Enumerate the (year, doy, hour) tuples the window spans.
    start_dt = dt.datetime.fromtimestamp(start_epoch, dt.timezone.utc).replace(tzinfo=None)
    cur = dt.datetime(start_dt.year, start_dt.month, start_dt.day, start_dt.hour)
    hours: list[tuple[int, int, int]] = []
    while cur.replace(tzinfo=dt.timezone.utc).timestamp() <= end_epoch:
        hours.append((cur.year, int(cur.strftime("%j")), cur.hour))
        cur = cur + dt.timedelta(hours=1)

    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    epochs: list[np.ndarray] = []
    n_files = 0

    for year, doy, hh in hours:
        keys = _list_glm_files(satellite, year, doy, hh)
        for key in keys:
            if n_files >= max_files:
                break
            s_ep, e_ep = _filename_epoch(key)
            # Skip files that don't overlap our window
            if e_ep < start_epoch or s_ep > end_epoch:
                continue
            local = _download_glm_file(satellite, key)
            if local is None:
                continue
            flashes = _read_glm_flashes(local, bbox=bbox)
            if not flashes:
                continue
            # Additional time filter on flash epochs
            m = (flashes["epoch"] >= start_epoch) & (flashes["epoch"] <= end_epoch)
            if np.any(m):
                lats.append(flashes["lat"][m])
                lons.append(flashes["lon"][m])
                energies.append(flashes["energy"][m])
                areas.append(flashes["area"][m])
                epochs.append(flashes["epoch"][m])
            n_files += 1
        if n_files >= max_files:
            break

    if not lats:
        return FlashCollection.empty()
    return FlashCollection(
        lat=np.concatenate(lats),
        lon=np.concatenate(lons),
        energy=np.concatenate(energies),
        area=np.concatenate(areas),
        epoch=np.concatenate(epochs),
    )


def compute_block_l(
    end_time: dt.datetime,
    center_lat: float,
    center_lon: float,
    *,
    bbox_halfwidth_deg: float = 2.0,
    satellite: str = "goes19",
) -> dict[str, float]:
    """Compute Block L (lightning) features around (center_lat, center_lon).

    Uses a ±2° bbox by default (~220 km half-width at mid-latitudes,
    roughly the domain of a supercell complex or tropical cyclone).
    """
    feat = _empty_block_l()
    bbox = (
        center_lat - bbox_halfwidth_deg,
        center_lat + bbox_halfwidth_deg,
        center_lon - bbox_halfwidth_deg,
        center_lon + bbox_halfwidth_deg,
    )
    end_epoch = end_time.replace(tzinfo=dt.timezone.utc).timestamp()

    # Fetch 6h window once; compute 1h / 15-min sub-window stats from it
    flashes_6h = fetch_glm_flashes(end_time, window_h=6.0, bbox=bbox, satellite=satellite)
    if len(flashes_6h) == 0:
        feat["ltg_flash_count_1h"] = 0.0
        feat["ltg_flash_count_6h"] = 0.0
        feat["ltg_flash_rate_per_min"] = 0.0
        feat["ltg_area_coverage"] = 0.0
        feat["ltg_peak_energy_J"] = 0.0
        feat["ltg_flash_density_per_km2"] = 0.0
        feat["ltg_jump_ratio"] = 1.0
        return feat

    feat["ltg_flash_count_6h"] = float(len(flashes_6h))

    # 1-hour window
    m_1h = flashes_6h.epoch >= (end_epoch - 3600)
    feat["ltg_flash_count_1h"] = float(np.sum(m_1h))
    feat["ltg_flash_rate_per_min"] = feat["ltg_flash_count_1h"] / 60.0

    # 15-minute window for jump ratio
    m_15m = flashes_6h.epoch >= (end_epoch - 900)
    n_15m = float(np.sum(m_15m))
    rate_15m = n_15m / 15.0
    rate_60m = feat["ltg_flash_rate_per_min"]
    feat["ltg_jump_ratio"] = float(rate_15m / rate_60m) if rate_60m > 0.05 else 1.0

    # Spatial stats (1-hour window)
    if feat["ltg_flash_count_1h"] > 0:
        lat_1h = flashes_6h.lat[m_1h]
        lon_1h = flashes_6h.lon[m_1h]
        energy_1h = flashes_6h.energy[m_1h]
        area_km2 = (2 * bbox_halfwidth_deg * 111.0) ** 2

        # Area coverage: unique 0.1° cells hit
        cells = set(zip(
            np.round(lat_1h * 10).astype(int).tolist(),
            np.round(lon_1h * 10).astype(int).tolist(),
        ))
        total_cells = int((2 * bbox_halfwidth_deg * 10) ** 2)
        feat["ltg_area_coverage"] = len(cells) / max(total_cells, 1)

        feat["ltg_peak_energy_J"] = float(np.percentile(energy_1h, 99))
        feat["ltg_flash_density_per_km2"] = feat["ltg_flash_count_1h"] / max(area_km2, 1)

    return feat


def compute_block_l_array(
    end_time: dt.datetime,
    center_lat: float,
    center_lon: float,
    **kwargs,
) -> np.ndarray:
    """Return Block L as a fixed-length (N_FEAT_L,) float32 ndarray."""
    feats = compute_block_l(end_time, center_lat, center_lon, **kwargs)
    return np.array([feats[n] for n in BLOCK_L_NAMES], dtype=np.float32)
