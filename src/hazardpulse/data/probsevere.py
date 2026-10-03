"""ProbSevere v3 storm-object data access with on-disk caching.

Downloads ProbSevere v3 JSON storm-object files from AWS S3 at 15-minute
intervals during convective hours (12 Z -- 06 Z) and caches as compressed
JSON.

Each time step contains storm objects with:
  - Storm ID, lat/lon, motion vectors
  - ProbSevere scores (PS, PStor, PShail, PSwind)
  - Atmospheric proxies (MUCAPE, MLCAPE, MLCIN, EBSHEAR, SRH, etc.)
  - Lightning (flash rate, flash density, MaxLLAz, etc.)
  - Radar-derived (MESH, VIL density, etc.)
"""

from __future__ import annotations

import gzip
import json
import os
import re as _re
from pathlib import Path

from hazardpulse.data.http import fetch_bytes

# ---------------------------------------------------------------------------
# Project paths — check env var HAZARDPULSE_PROBSEVERE_CACHE first
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_PROBSEVERE_CACHE",
    str(PROJECT_ROOT / ".cache" / "probsevere"),
))

# ---------------------------------------------------------------------------
# AWS ProbSevere v3 endpoints
# ---------------------------------------------------------------------------

# ProbSevere is available via NOAA MRMS on AWS Open Data
# Bucket: noaa-mrms-pds, prefix: ProbSevere/{YYYYMMDD}/
PS_S3_BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PS_FETCH_THREADS = int(os.environ.get("HAZARDPULSE_PS_THREADS", "8"))  # slot files fetched concurrently
_re_nonword = __import__("re").compile("[^0-9a-z]+")
PS_S3_PREFIX = "ProbSevere/{date_str}/"

# Convective hours to scan (12 Z to 06 Z next day, every 15 min)
CONVECTIVE_HOURS: list[int] = list(range(12, 24)) + list(range(0, 7))
SCAN_MINUTES: list[int] = [0, 15, 30, 45]

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    date_str: str,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return the local cache path for a given date."""
    root = cache_dir or CACHE_ROOT
    return root / f"{date_str}.json.gz"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cached_probsevere(
    date_str: str,
    *,
    cache_dir: Path | None = None,
) -> list[dict] | None:
    """Load ProbSevere storm objects from local cache only (no network).

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    list[dict] or None
        List of time-step dicts with storm objects, or *None* if not cached.
    """
    path = _cache_path(date_str, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8"))
        return data.get("time_steps", data) if isinstance(data, dict) else data
    except Exception:
        return None


def scan_probsevere_cache(
    cache_dir: Path | None = None,
) -> list[str]:
    """List available cached dates.

    Returns
    -------
    list[str]
        Sorted list of date strings (``YYYYMMDD``) present in cache.
    """
    root = cache_dir or CACHE_ROOT
    if not root.is_dir():
        return []
    dates: list[str] = []
    for path in root.iterdir():
        if path.name.endswith(".json.gz") and len(path.stem) >= 8:
            dates.append(path.stem[:8])
    dates.sort()
    return dates


def fetch_probsevere_day(
    date_str: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Fetch all ProbSevere v3 storm objects for a day.

    Downloads from the AWS S3 bucket at 15-minute intervals during
    convective hours (12 Z -- 06 Z).  Results are cached as compressed
    JSON for subsequent calls.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    cache_dir : Path, optional
        Override the default cache directory.
    refresh : bool
        Force re-download even if cached.

    Returns
    -------
    list[dict]
        List of time-step dicts.  Each dict contains:
        ``valid_time`` (ISO str) and ``storms`` (list of storm-object dicts).
    """
    if not refresh:
        cached = load_cached_probsevere(date_str, cache_dir=cache_dir)
        if cached is not None:
            return cached

    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    time_steps: list[dict] = []

    # First try listing actual files from S3 (more reliable than guessing)
    s3_files, s3_listing_ok = _list_s3_files(date_str)
    if s3_files:
        print(f"  ProbSevere: {len(s3_files)} files found in S3 for {date_str}")
        # One file per 30-minute slot (the first in each), downloaded
        # concurrently; time steps keep slot order.
        import re as _re
        from concurrent.futures import ThreadPoolExecutor

        picked: list[tuple[str, str]] = []
        seen_slots: set[str] = set()
        for key in s3_files:
            m = _re.search(r"_(\d{8})_(\d{6})\.json", key)
            if not m:
                continue
            hhmm = m.group(2)[:4]  # HHMM
            slot = hhmm[:2] + ("00" if int(hhmm[2:]) < 30 else "30")  # round to 30min
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            picked.append((key, hhmm))

        def _get(item: tuple[str, str]) -> dict | None:
            key, hhmm = item
            try:
                raw = fetch_bytes(f"{PS_S3_BUCKET}/{key}", namespace="probsevere", timeout=30, use_cache=False)
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                return None
            storms = _parse_storms(data)
            if storms is None:
                return None
            valid_time = data.get("validTime", f"{year}-{month}-{day}T{hhmm[:2]}:{hhmm[2:]}:00Z")
            return {"valid_time": valid_time, "storms": storms}

        with ThreadPoolExecutor(PS_FETCH_THREADS) as ex:
            time_steps = [ts for ts in ex.map(_get, picked) if ts is not None]
    elif s3_listing_ok:
        # The bucket answered and holds NO files under this day's prefix. That
        # is authoritative -- the archive has gaps (e.g. 2021-05-15/16) -- so
        # there is nothing to probe. The old code probed 76 guessed slot names
        # anyway, each 404 retried with back-off: ~35 minutes per empty day,
        # ending in the same empty answer.
        print(
            f"  ProbSevere: S3 listing OK but empty for {date_str} "
            "(archive gap, or data not yet posted). Not probing; not caching."
        )
        return []
    else:
        print(
            f"  ProbSevere: S3 listing FAILED for {date_str} "
            "(bucket unreachable / auth / path change). Trying known time slots."
        )
        for hour in CONVECTIVE_HOURS:
            for minute in SCAN_MINUTES:
                ts = _fetch_single_timestep(year, month, day, hour, minute)
                if ts is not None:
                    time_steps.append(ts)

    if not time_steps:
        # Never cache an empty day: load_cached_probsevere would return [] (not
        # None) forever after, so a transient failure would become a permanent
        # hole in the cache.
        return time_steps

    # Persist to cache
    out_path = _cache_path(date_str, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"date": date_str, "time_steps": time_steps},
        separators=(",", ":"),
    ).encode("utf-8")
    with gzip.open(out_path, "wb") as fh:
        fh.write(payload)

    return time_steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _list_s3_files(date_str: str) -> tuple[list[str], bool]:
    """List ProbSevere JSON files for a date from the NOAA MRMS S3 bucket.

    Returns
    -------
    tuple[list[str], bool]
        ``(keys, listing_ok)`` — ``listing_ok`` is True when the S3 API
        returned a parseable response (even if empty); False if the request
        itself failed (network, HTTP error, malformed XML).
    """
    import re
    import urllib.request as _urllib

    prefix = PS_S3_PREFIX.format(date_str=date_str)
    list_url = f"{PS_S3_BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
    try:
        req = _urllib.Request(list_url, headers={"User-Agent": "hazardpulse/0.1"})
        resp = _urllib.urlopen(req, timeout=15)
        xml = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  ProbSevere S3 listing failed: {exc}")
        return [], False
    # Basic structural validation — S3 ListBucketResult is always XML.
    if "<ListBucketResult" not in xml:
        print(
            "  ProbSevere S3 listing returned unexpected response "
            f"(first 200 bytes: {xml[:200]!r})."
        )
        return [], False
    return re.findall(r"<Key>(.*?)</Key>", xml), True


def _fetch_single_timestep(
    year: str,
    month: str,
    day: str,
    hour: int,
    minute: int,
) -> dict | None:
    """Fetch a single ProbSevere time step closest to the given hour:minute.

    Returns a dict with ``valid_time`` and ``storms`` or *None* on failure.
    """
    date_str = f"{year}{month}{day}"
    # Try the NOAA MRMS bucket (correct URL)
    target_hhmm = f"{hour:02d}{minute:02d}"
    filename = f"MRMS_PROBSEVERE_{date_str}_{target_hhmm}00.json"
    url = f"{PS_S3_BUCKET}/ProbSevere/{date_str}/{filename}"

    try:
        raw = fetch_bytes(url, namespace="probsevere", timeout=30, use_cache=False)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        # Try nearby timestamps (ProbSevere uses ~2min intervals, not exact 15min)
        for delta in range(1, 5):
            for m in [minute + delta, minute - delta]:
                if 0 <= m < 60:
                    fn = f"MRMS_PROBSEVERE_{date_str}_{hour:02d}{m:02d}00.json"
                    u = f"{PS_S3_BUCKET}/ProbSevere/{date_str}/{fn}"
                    try:
                        raw = fetch_bytes(u, namespace="probsevere", timeout=15, use_cache=False)
                        data = json.loads(raw.decode("utf-8", errors="replace"))
                        break
                    except Exception:
                        continue
            else:
                continue
            break
        else:
            return None

    valid_time = data.get("validTime", f"{year}-{month}-{day}T{hour:02d}:{minute:02d}:00Z")

    storms = _parse_storms(data)
    if storms is None:
        return None
    return {"valid_time": valid_time, "storms": storms}


_RATE_RE = _re.compile(r"^\s*(\d{2})(\d{2})Z\s+(-?\d+(?:\.\d+)?)\s*%?/min(?:\s*\((\w+)\))?")
_KM_RE = _re.compile(r"(-?\d+(?:\.\d+)?)\s*km")
_RATE_CATEGORY = {"weak": 1.0, "moderate": 2.0, "strong": 3.0}


def _valid_minute_of_day(data: dict) -> int | None:
    m = _re.search(r"_(\d{2})(\d{2})\d{2}", str(data.get("validTime", "")))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def parse_string_attributes(props: dict, valid_minute: int | None) -> dict[str, float]:
    """NOAA publishes three ProbSevere v3 attributes as strings with units, which a float()
    filter silently drops (all three were NaN in every stored row until 2026-10-02):

    ``MAXRC_EMISS '2251Z 1.5%/min (weak)'``  peak satellite cloud-top emissivity growth rate
    ``MAXRC_ICECF '2241Z 0.01/min (weak)'``  peak ice-cloud-fraction (glaciation) rate
    ``AVG_BEAM_HGT '4.09 kft / 1.25 km'``    radar beam height over the object

    -> ``maxrc_emiss`` / ``maxrc_icecf`` (the rate), ``*_age_min`` (minutes from the peak to the
    file's valid time, across midnight), ``*_cat`` (weak 1, moderate 2, strong 3), and
    ``avg_beam_hgt`` (km). An unparseable or 'N/A' value yields nothing, never a guess.
    """
    out: dict[str, float] = {}
    for key, name in (("MAXRC_EMISS", "maxrc_emiss"), ("MAXRC_ICECF", "maxrc_icecf")):
        m = _RATE_RE.match(str(props.get(key, "")))
        if not m:
            continue
        out[name] = float(m.group(3))
        if valid_minute is not None:
            peak = int(m.group(1)) * 60 + int(m.group(2))
            out[f"{name}_age_min"] = float((valid_minute - peak) % 1440)
        if m.group(4) and m.group(4).lower() in _RATE_CATEGORY:
            out[f"{name}_cat"] = _RATE_CATEGORY[m.group(4).lower()]
    m = _KM_RE.search(str(props.get("AVG_BEAM_HGT", "")))
    if m:
        out["avg_beam_hgt"] = float(m.group(1))
    return out


def _parse_storms(data: dict) -> list[dict] | None:
    """Parse ProbSevere GeoJSON features into storm dicts."""
    valid_minute = _valid_minute_of_day(data)
    features = data.get("features", [])
    if not features:
        return []

    storms: list[dict] = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        # Compute centroid lat/lon from geometry polygon
        lat, lon = 0.0, 0.0
        coords = geom.get("coordinates", [[]])
        if coords and coords[0]:
            ring = coords[0]
            if len(ring) > 0:
                lat = sum(p[1] for p in ring) / len(ring)
                lon = sum(p[0] for p in ring) / len(ring)

        def _float(key: str, default: float = 0.0) -> float:
            v = props.get(key)
            if v is None or v == "N/A":
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        def _float_fallback(primary: str, fallback: str) -> float:
            """Return _float(primary) if the key exists, else _float(fallback).

            Unlike ``or``, this correctly preserves 0.0 values.
            """
            val = props.get(primary)
            if val is not None and val != "N/A":
                return _float(primary)
            return _float(fallback)

        # ProbSevere v3 publishes its hazard models OUTSIDE ``properties``, as
        # feature["models"][<model>]["PROB"] (0-100). Until 2026-10-02 the
        # parser looked only in ``properties`` (PROBTOR / PS_TOR), found
        # nothing, and stored 0.0: NOAA's ProbTor, ProbHail and ProbWind were
        # zero for every storm in the cache and on the live site.
        models = feat.get("models") or {}

        def _model_prob(name: str, legacy_primary: str, legacy_fallback: str) -> float:
            m = models.get(name) if isinstance(models, dict) else None
            if isinstance(m, dict) and m.get("PROB") not in (None, "N/A"):
                try:
                    return float(m["PROB"])
                except (ValueError, TypeError):
                    pass
            return _float_fallback(legacy_primary, legacy_fallback)

        storm: dict = {
            "id": props.get("ID", 0),
            "lat": lat,
            "lon": lon,
            "ps": _float("PS"),
            "ps_tor": _model_prob("probtor", "PROBTOR", "PS_TOR"),
            "ps_hail": _model_prob("probhail", "PROBHAIL", "PS_HAIL"),
            "ps_wind": _model_prob("probwind", "PROBWIND", "PS_WIND"),
            "ps_severe": _model_prob("probsevere", "PROBSEVERE", "PS"),
            "mucape": _float("MUCAPE"),
            "mlcape": _float("MLCAPE"),
            "mlcin": _float("MLCIN"),
            "ebshear": _float("EBSHEAR"),
            "srh01": _float_fallback("SRH01KM", "SRH01"),
            "mesh": _float("MESH"),
            "vil_density": _float_fallback("VIL_DENSITY", "VILD"),
            "flash_rate": _float_fallback("FLASH_RATE", "FLASHRATE"),
            "flash_density": _float_fallback("FLASH_DENSITY", "FLASHDENSITY"),
            "maxllaz": _float("MAXLLAZ"),
            "p98llaz": _float("P98LLAZ"),
            "p98mlaz": _float("P98MLAZ"),
            "lja": _float("LJA"),
            "size": _float("SIZE"),
            "motion_east": _float("MOTION_EAST"),
            "motion_south": _float("MOTION_SOUTH"),
        }
        # Keep every other numeric attribute NOAA publishes (PWAT, CAPE_M10M30,
        # MEANWIND_1-3kmAGL, WETBULB_0C_HGT, ...) under a sanitised lowercase name,
        # so a model can use them and the live scorer -- which parses with this same
        # function -- will have them too. The string-valued ones are parsed first.
        storm.update(parse_string_attributes(props, valid_minute))
        for key, raw in props.items():
            name = _re_nonword.sub("_", str(key).lower()).strip("_")
            if not name or name in storm or name == "id":
                continue
            try:
                storm[name] = float(raw)
            except (TypeError, ValueError):
                continue
        if geom:
            storm["geometry"] = geom
        storms.append(storm)

    return storms
