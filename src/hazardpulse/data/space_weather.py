"""Space-weather data fetcher (NOAA SWPC + NCEI archives + WDC Kyoto Dst).

Provides a single entry point — ``space_weather_features_for_window`` — that
computes a fixed dictionary of features for any (event_time, optional lat/lon)
input. Backed by:

  - NOAA SWPC live JSON endpoints (Kp, solar wind, GOES X-ray, Dst proxy)
  - NOAA NCEI historical archives (GOES X-ray flare events back to 1986)
  - WDC Kyoto definitive Dst archive (1957-)
  - NASA OMNI 1-min plasma + IMF archive (1981-, the gold standard for
    historical IMF/solar wind)

The architecture is layered: a low-level ``SpaceWeatherCache`` holds the
canonical historical tables (downloaded once via
``scripts/download_swpc_archives.py``); a ``LiveSpaceWeatherFetcher`` pulls
the most recent ~7 days from SWPC's live endpoints for real-time scoring.

This module is shared by the earthquake, hurricane, and tornado scorers:
all three use the same Block W (space weather) feature definitions so
ablation experiments are directly comparable.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SWPC_CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_SWPC_CACHE",
    str(PROJECT_ROOT / ".cache" / "swpc"),
))


# ---------------------------------------------------------------------------
# Source URLs (all free, public, no auth)
# ---------------------------------------------------------------------------

# NOAA SWPC live (last few hours/days, JSON):
LIVE_URLS = {
    "kp_3h":           "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "solar_wind_5min": "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json",
    "imf_5min":        "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json",
    "xray_6h":         "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
    "xray_1d":         "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json",
    "xray_3d":         "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json",
    "dst_kyoto":       "https://services.swpc.noaa.gov/products/kyoto-dst.json",
    "f107":            "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
}

# Historical archives (large tables, downloaded once into SWPC_CACHE_ROOT):
ARCHIVE_URLS = {
    "kp_definitive":   "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt",
    "dst_definitive":  "https://wdc.kugi.kyoto-u.ac.jp/dstdir/dst{year}.html",  # per-year HTML
    "omni_hourly":     "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "goes_xray_events":"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{year}.txt",
}


# ---------------------------------------------------------------------------
# Block W (space weather) feature schema  — used by EQ / HU / TO scorers
# ---------------------------------------------------------------------------

BLOCK_W_NAMES: list[str] = [
    "sw_kp_max_72h",          # peak Kp index in 72h before event
    "sw_kp_mean_72h",         # mean Kp in 72h
    "sw_kp_storm_count_72h",  # # of 3-hour windows with Kp >= 5
    "sw_dst_min_72h",         # most-negative Dst nT in 72h (storm intensity)
    "sw_dst_recovery",        # Dst recovery rate (nT/hr after min)
    "sw_bz_min_72h",          # most-negative IMF Bz in 72h (sustained southward)
    "sw_bz_mean_neg_72h",     # mean of negative-Bz values only
    "sw_speed_max_72h",       # peak solar wind bulk speed (km/s)
    "sw_density_max_72h",     # peak plasma density (cm^-3)
    "sw_xray_flare_count_72h",# # of M+/X+ class flares
    "sw_xray_max_class",      # peak GOES X-ray class as numeric (M=2, X=3, X10=4)
    "sw_f107_72h",            # 10.7cm radio flux (sfu)
]
N_FEAT_W: int = len(BLOCK_W_NAMES)


def _empty_block_w() -> dict[str, float]:
    """Default Block W feature dict (NaN means data missing)."""
    return {name: float("nan") for name in BLOCK_W_NAMES}


# ---------------------------------------------------------------------------
# Low-level HTTP fetch with caching
# ---------------------------------------------------------------------------

def _fetch_text(url: str, *, timeout: int = 60, retries: int = 3) -> str:
    """Fetch a URL with retry + exponential backoff. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "HazardPulse/1.0 (research)"}
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


def _cache_path(name: str) -> Path:
    SWPC_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return SWPC_CACHE_ROOT / name


# ---------------------------------------------------------------------------
# Live SWPC fetcher — pulls last few days for real-time scoring
# ---------------------------------------------------------------------------

@dataclass
class LiveSpaceWeather:
    """In-memory snapshot of recent space-weather observations.

    Each array is (timestamp, value) pairs sorted by time ascending.
    Timestamps are naive UTC datetimes.
    """
    kp: list[tuple[dt.datetime, float]]
    dst: list[tuple[dt.datetime, float]]
    bz: list[tuple[dt.datetime, float]]
    sw_speed: list[tuple[dt.datetime, float]]
    sw_density: list[tuple[dt.datetime, float]]
    xray_flares: list[tuple[dt.datetime, float]]  # (time, class numeric)
    f107: float | None
    fetched_at: dt.datetime


def _parse_iso(ts: str) -> dt.datetime | None:
    """Parse ISO 8601 / SWPC JSON timestamp (naive UTC)."""
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        try:
            return dt.datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            return None


def _xray_class_to_numeric(cls: str | None) -> float | None:
    """Map flare class string ('M3.5', 'X1.2', 'C9.0') to a numeric value.

    Encoding: A=0, B=1, C=2, M=3, X=4 + log10(magnitude/10).
    So X10 = 5.0, X1 = 4.0, M5 = 3.7, C1 = 2.0.
    """
    if not cls or len(cls) < 2:
        return None
    letter = cls[0].upper()
    base = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}.get(letter)
    if base is None:
        return None
    try:
        mag = float(cls[1:])
    except ValueError:
        return float(base)
    if mag <= 0:
        return float(base)
    return base + math.log10(mag) - 1.0  # log10(mag) but mag is the 1-9.9 prefix


def _parse_records(
    data, time_key: str, value_keys: tuple[str, ...]
) -> list[tuple[dt.datetime, float]]:
    """Parse SWPC JSON into (datetime, value) tuples.

    SWPC ships two formats:
      - list of dicts:   [{"time_tag": "...", "Kp": 2.67}, ...]
      - list of arrays:  [["time_tag", "Kp"], ["2026-...", "2.67"], ...]
    This handles both.
    """
    out: list[tuple[dt.datetime, float]] = []
    if not isinstance(data, list) or not data:
        return out

    if isinstance(data[0], dict):
        for item in data:
            t = _parse_iso(str(item.get(time_key, "")))
            if t is None:
                continue
            for k in value_keys:
                if k in item and item[k] is not None:
                    try:
                        out.append((t, float(item[k])))
                        break
                    except (ValueError, TypeError):
                        pass
    elif isinstance(data[0], list):
        header = [str(c).lower() for c in data[0]]
        time_idx = next((i for i, h in enumerate(header) if h == time_key.lower() or "time" in h), 0)
        val_idx = -1
        for k in value_keys:
            if k.lower() in header:
                val_idx = header.index(k.lower())
                break
        if val_idx < 0:
            val_idx = 1
        for row in data[1:]:
            if not isinstance(row, list) or len(row) <= max(time_idx, val_idx):
                continue
            t = _parse_iso(str(row[time_idx]))
            try:
                v = float(row[val_idx])
            except (ValueError, TypeError):
                continue
            if t is not None:
                out.append((t, v))
    return sorted(out)


def _fetch_live_kp() -> list[tuple[dt.datetime, float]]:
    try:
        data = json.loads(_fetch_text(LIVE_URLS["kp_3h"]))
    except Exception:
        return []
    return _parse_records(data, "time_tag", ("Kp", "kp", "kp_index"))


def _fetch_live_dst() -> list[tuple[dt.datetime, float]]:
    """Fetch SWPC's Kyoto Dst proxy (1-hour cadence, last ~7 days)."""
    try:
        data = json.loads(_fetch_text(LIVE_URLS["dst_kyoto"]))
    except Exception:
        return []
    return _parse_records(data, "time_tag", ("dst", "Dst"))


def _fetch_live_imf_solar_wind() -> tuple[
    list[tuple[dt.datetime, float]],  # bz
    list[tuple[dt.datetime, float]],  # speed
    list[tuple[dt.datetime, float]],  # density
]:
    """Fetch DSCOVR/ACE 5-minute IMF + plasma from SWPC."""
    bz: list[tuple[dt.datetime, float]] = []
    speed: list[tuple[dt.datetime, float]] = []
    density: list[tuple[dt.datetime, float]] = []
    try:
        raw_imf = _fetch_text(LIVE_URLS["imf_5min"])
        imf = json.loads(raw_imf)
    except Exception:
        imf = []
    if isinstance(imf, list) and len(imf) > 1:
        header = [str(c).lower() for c in imf[0]] if isinstance(imf[0], list) else []
        bz_idx = header.index("bz_gsm") if "bz_gsm" in header else (
            header.index("bz_gse") if "bz_gse" in header else 3
        )
        for row in imf[1:]:
            if not isinstance(row, list) or len(row) < bz_idx + 1:
                continue
            t = _parse_iso(str(row[0]))
            try:
                v = float(row[bz_idx])
            except (ValueError, TypeError):
                continue
            if t is not None:
                bz.append((t, v))

    try:
        raw_pl = _fetch_text(LIVE_URLS["solar_wind_5min"])
        pl = json.loads(raw_pl)
    except Exception:
        pl = []
    if isinstance(pl, list) and len(pl) > 1:
        header = [str(c).lower() for c in pl[0]] if isinstance(pl[0], list) else []
        speed_idx = header.index("speed") if "speed" in header else 2
        density_idx = header.index("density") if "density" in header else 1
        for row in pl[1:]:
            if not isinstance(row, list):
                continue
            t = _parse_iso(str(row[0]))
            if t is None:
                continue
            try:
                speed.append((t, float(row[speed_idx])))
            except (ValueError, TypeError, IndexError):
                pass
            try:
                density.append((t, float(row[density_idx])))
            except (ValueError, TypeError, IndexError):
                pass
    return sorted(bz), sorted(speed), sorted(density)


def _fetch_live_xray_flares() -> list[tuple[dt.datetime, float]]:
    """Fetch GOES X-ray flux past 3 days, identify M+/X+ flares."""
    try:
        raw = _fetch_text(LIVE_URLS["xray_3d"])
        data = json.loads(raw)
    except Exception:
        return []

    flares: list[tuple[dt.datetime, float]] = []
    # Each entry is {"time_tag":"...", "energy":"0.1-0.8nm", "flux": 1.23e-7, ...}
    samples: list[tuple[dt.datetime, float]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("energy") != "0.1-0.8nm":
            continue
        t = _parse_iso(str(item.get("time_tag", "")))
        try:
            flux = float(item.get("flux", 0))
        except (ValueError, TypeError):
            continue
        if t is not None and flux > 0:
            samples.append((t, flux))
    # Detect peaks: any 1-min sample above M class threshold (1e-5 W/m^2)
    M_THRESHOLD = 1e-5
    for t, flux in samples:
        if flux >= M_THRESHOLD:
            # Convert flux -> class numeric:  M = log10(flux/1e-6) base
            class_num = 3.0 + math.log10(max(flux / 1e-5, 1.0))
            flares.append((t, class_num))
    return sorted(flares)


def _fetch_live_f107() -> float | None:
    try:
        raw = _fetch_text(LIVE_URLS["f107"])
        data = json.loads(raw)
    except Exception:
        return None
    if isinstance(data, list) and data:
        last = data[-1]
        try:
            return float(last.get("flux") or last.get("f10.7") or last.get("observed_flux"))
        except (ValueError, TypeError, AttributeError):
            pass
    return None


def fetch_live_space_weather() -> LiveSpaceWeather:
    """One-shot fetch of all live space-weather streams. Cached per-process."""
    bz, speed, density = _fetch_live_imf_solar_wind()
    return LiveSpaceWeather(
        kp=_fetch_live_kp(),
        dst=_fetch_live_dst(),
        bz=bz,
        sw_speed=speed,
        sw_density=density,
        xray_flares=_fetch_live_xray_flares(),
        f107=_fetch_live_f107(),
        fetched_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    )


# ---------------------------------------------------------------------------
# Historical archive cache — populated by scripts/download_swpc_archives.py
# ---------------------------------------------------------------------------

class SpaceWeatherCache:
    """Lazy-loaded historical SWPC tables.

    Each table is loaded once on first access from the .npz cache and held
    in memory. Use .ensure_loaded() before bulk feature extraction.
    """

    def __init__(self, cache_root: Path = SWPC_CACHE_ROOT):
        self.cache_root = Path(cache_root)
        self._kp: np.ndarray | None = None         # cols: (epoch, kp)
        self._dst: np.ndarray | None = None         # cols: (epoch, dst_nT)
        self._omni: np.ndarray | None = None        # cols: (epoch, bz_gsm, speed, density)
        self._xray: np.ndarray | None = None        # cols: (epoch, class_numeric)
        self._loaded = False

    def _load_table(self, name: str) -> np.ndarray | None:
        path = self.cache_root / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            return data["arr"]

    def ensure_loaded(self) -> bool:
        """Load all available archive tables. Returns True if at least one
        table is present."""
        if self._loaded:
            return any(t is not None for t in (self._kp, self._dst, self._omni, self._xray))
        self._kp = self._load_table("kp_definitive")
        self._dst = self._load_table("dst_definitive")
        self._omni = self._load_table("omni_hourly")
        self._xray = self._load_table("goes_xray_events")
        self._loaded = True
        return any(t is not None for t in (self._kp, self._dst, self._omni, self._xray))

    @staticmethod
    def _slice(table: np.ndarray | None, t_start: float, t_end: float) -> np.ndarray | None:
        if table is None or len(table) == 0:
            return None
        # Column 0 is always epoch seconds
        mask = (table[:, 0] >= t_start) & (table[:, 0] <= t_end)
        if not np.any(mask):
            return None
        return table[mask]

    def kp_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        return self._slice(self._kp, t_start, t_end)

    def dst_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        return self._slice(self._dst, t_start, t_end)

    def omni_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        return self._slice(self._omni, t_start, t_end)

    def xray_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        return self._slice(self._xray, t_start, t_end)


# Single shared cache instance for the process.
_GLOBAL_CACHE: SpaceWeatherCache | None = None
_GLOBAL_LIVE: LiveSpaceWeather | None = None
_GLOBAL_LIVE_FETCHED_AT: float = 0.0
_LIVE_TTL_SEC = 600  # cache live data for 10 minutes


def get_cache() -> SpaceWeatherCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = SpaceWeatherCache()
        _GLOBAL_CACHE.ensure_loaded()
    return _GLOBAL_CACHE


def get_live(refresh: bool = False) -> LiveSpaceWeather:
    """Return process-cached live space weather, refetching every 10 minutes."""
    global _GLOBAL_LIVE, _GLOBAL_LIVE_FETCHED_AT
    now = time.time()
    if refresh or _GLOBAL_LIVE is None or (now - _GLOBAL_LIVE_FETCHED_AT) > _LIVE_TTL_SEC:
        _GLOBAL_LIVE = fetch_live_space_weather()
        _GLOBAL_LIVE_FETCHED_AT = now
    return _GLOBAL_LIVE


# ---------------------------------------------------------------------------
# Feature computation — the public API used by EQ / HU / TO scorers
# ---------------------------------------------------------------------------

def space_weather_features_for_window(
    event_time: dt.datetime,
    *,
    window_h: int = 72,
) -> dict[str, float]:
    """Compute Block W features for the given event time.

    Tries historical archive first, falls back to live SWPC if event is
    within the last few days. All fields are NaN if no data covers the
    requested window.
    """
    feat = _empty_block_w()
    t_end = event_time.replace(tzinfo=dt.timezone.utc).timestamp()
    t_start = t_end - window_h * 3600

    cache = get_cache()
    live = None
    # Decide whether to use live: anything in the last 3 days needs live data
    age_days = (dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - event_time).total_seconds() / 86400.0
    if age_days < 3.5:
        try:
            live = get_live()
        except Exception:
            live = None

    # ---- Kp ----
    kp_vals: list[float] = []
    arr = cache.kp_window(t_start, t_end)
    if arr is not None and len(arr) > 0:
        kp_vals = [float(x) for x in arr[:, 1]]
    elif live is not None:
        kp_vals = [v for t, v in live.kp if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end]
    if kp_vals:
        feat["sw_kp_max_72h"] = float(np.max(kp_vals))
        feat["sw_kp_mean_72h"] = float(np.mean(kp_vals))
        feat["sw_kp_storm_count_72h"] = float(sum(1 for v in kp_vals if v >= 5.0))

    # ---- Dst ----
    dst_vals: list[tuple[float, float]] = []
    arr = cache.dst_window(t_start, t_end)
    if arr is not None and len(arr) > 0:
        dst_vals = [(float(t), float(v)) for t, v in arr]
    elif live is not None:
        dst_vals = [
            (t.replace(tzinfo=dt.timezone.utc).timestamp(), v)
            for t, v in live.dst
            if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end
        ]
    if dst_vals:
        dst_vals.sort()
        vals = [v for _, v in dst_vals]
        feat["sw_dst_min_72h"] = float(np.min(vals))
        # Recovery: nT/hr from minimum to end of window
        min_idx = int(np.argmin(vals))
        if min_idx < len(dst_vals) - 1:
            t_min, v_min = dst_vals[min_idx]
            t_last, v_last = dst_vals[-1]
            dt_h = (t_last - t_min) / 3600.0
            if dt_h > 0:
                feat["sw_dst_recovery"] = (v_last - v_min) / dt_h

    # ---- IMF Bz / solar wind ----
    bz_vals: list[float] = []
    speed_vals: list[float] = []
    density_vals: list[float] = []
    arr = cache.omni_window(t_start, t_end)
    if arr is not None and len(arr) > 0:
        # cols: epoch, bz_gsm, speed, density
        bz_vals = [float(x) for x in arr[:, 1] if not math.isnan(x)]
        speed_vals = [float(x) for x in arr[:, 2] if not math.isnan(x)]
        if arr.shape[1] >= 4:
            density_vals = [float(x) for x in arr[:, 3] if not math.isnan(x)]
    elif live is not None:
        bz_vals = [v for t, v in live.bz if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end]
        speed_vals = [v for t, v in live.sw_speed if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end]
        density_vals = [v for t, v in live.sw_density if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end]
    if bz_vals:
        feat["sw_bz_min_72h"] = float(np.min(bz_vals))
        neg = [v for v in bz_vals if v < 0]
        feat["sw_bz_mean_neg_72h"] = float(np.mean(neg)) if neg else 0.0
    if speed_vals:
        feat["sw_speed_max_72h"] = float(np.max(speed_vals))
    if density_vals:
        feat["sw_density_max_72h"] = float(np.max(density_vals))

    # ---- X-ray flares ----
    flare_classes: list[float] = []
    arr = cache.xray_window(t_start, t_end)
    if arr is not None and len(arr) > 0:
        flare_classes = [float(x) for x in arr[:, 1]]
    elif live is not None:
        flare_classes = [v for t, v in live.xray_flares if t_start <= t.replace(tzinfo=dt.timezone.utc).timestamp() <= t_end]
    if flare_classes:
        feat["sw_xray_flare_count_72h"] = float(len(flare_classes))
        feat["sw_xray_max_class"] = float(np.max(flare_classes))

    # ---- F10.7 (current snapshot — slowly varying daily index) ----
    if live is not None and live.f107 is not None:
        feat["sw_f107_72h"] = float(live.f107)

    return feat


def space_weather_features_for_events(
    event_times: Iterable[dt.datetime],
    *,
    window_h: int = 72,
    progress_every: int = 100,
) -> list[dict[str, float]]:
    """Bulk feature extraction (used by retraining scripts)."""
    cache = get_cache()
    cache.ensure_loaded()
    out: list[dict[str, float]] = []
    for i, t in enumerate(event_times):
        if progress_every and i and i % progress_every == 0:
            print(f"  Block W: extracted {i} events...")
        out.append(space_weather_features_for_window(t, window_h=window_h))
    return out
