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

import datetime as dt
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
        time_steps = _fetch_steps(slot_start_keys(s3_files))
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
# Which files make a day's time steps
# ---------------------------------------------------------------------------

SLOT_MINUTES = 30
_KEY_TIME_RE = _re.compile(r"_(\d{8})_(\d{6})\.json")


def key_time(key: str) -> dt.datetime | None:
    """The UTC valid time a ProbSevere file name carries (``MRMS_PROBSEVERE_20261004_235839.json``)."""
    m = _KEY_TIME_RE.search(str(key))
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _slot_of(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=t.minute - t.minute % SLOT_MINUTES, second=0, microsecond=0)


def slot_start_keys(keys: list[str]) -> list[str]:
    """The TRAINING selection (and the day cache's): the first file of each 30-minute slot, in listing order.

    Every training row and every archived track was built from these files, so the track-history features
    (block E: deltas, slopes, maxima over up to ``STORM_HISTORY_LOOKBACK`` steps, and the step cadence) mean
    "the storm at each of the last few half-hour slot starts"."""
    picked: list[str] = []
    seen: set[dt.datetime] = set()
    for key in keys:
        t = key_time(key)
        if t is None:
            continue
        slot = _slot_of(t)
        if slot in seen:
            continue
        seen.add(slot)
        picked.append(key)
    return picked


def live_keys(keys: list[str]) -> list[str]:
    """The LIVE selection: the NEWEST file for the storm's current state, and for every earlier 30-minute slot
    of the day the file at the same phase within its slot (nearest the newest file's offset from its slot
    start; ties to the earlier file) -- so the history keeps the 30-minute cadence it was trained on, anchored
    back from the newest file instead of from the slot starts.

    Measured 2026-10-05: the live scorer took the FIRST file of the latest slot, so its input was a median
    17.5 minutes old at issue over the 1,757 tornado records of 2026 (the 00:25Z run read 00:00:38 data while
    12 newer files existed). When the newest file IS a slot start the selection is exactly ``slot_start_keys``
    on complete days (tests/test_probsevere_live_selection.py checks every slot start of a recorded day; on
    real files the block P and E features of all 145 storms of the 2026-10-04 20:30 run were identical). The
    exception is a slot whose first file is missing: then the history keeps the newest file's phase."""
    timed = sorted((t, k) for k in keys if (t := key_time(k)) is not None)
    if not timed:
        return []
    newest_t, newest_key = timed[-1]
    newest_slot = _slot_of(newest_t)
    phase = newest_t - newest_slot
    by_slot: dict[dt.datetime, list[tuple[dt.datetime, str]]] = {}
    for t, k in timed:
        by_slot.setdefault(_slot_of(t), []).append((t, k))
    picked = []
    for slot in sorted(s for s in by_slot if s < newest_slot):
        target = slot + phase
        picked.append(min(by_slot[slot], key=lambda e: (abs((e[0] - target).total_seconds()), e[0]))[1])
    return picked + [newest_key]


def _fetch_one(key: str) -> tuple[dict | None, dict | None]:
    """(time step, raw document) of one file; (None, None) if it cannot be read or parsed."""
    try:
        raw = fetch_bytes(f"{PS_S3_BUCKET}/{key}", namespace="probsevere", timeout=30, use_cache=False)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None, None
    storms = _parse_storms(data)
    if storms is None:
        return None, None
    t = key_time(key)
    default = t.strftime("%Y-%m-%dT%H:%M:00Z") if t is not None else ""
    return {"valid_time": data.get("validTime", default), "storms": storms}, data


def _fetch_steps(keys: list[str]) -> list[dict]:
    """The time steps of ``keys``, downloaded concurrently, in the order given (unreadable files dropped)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(PS_FETCH_THREADS) as ex:
        return [ts for ts, _ in ex.map(_fetch_one, keys) if ts is not None]


def fetch_probsevere_live(date_str: str, *, max_newest_tries: int = 3) -> tuple[list[dict], dict | None]:
    """The day's time steps for a LIVE run (``live_keys``) and the attribute census of the newest file.

    Never cached: the day cache holds the training selection (slot starts), which the prospective scorer's
    track labels read. If the newest file cannot be read (still being written), the next newest is the
    anchor. If the S3 listing itself fails, this falls back to ``fetch_probsevere_day`` (slot starts, no
    census). The census (``property_census``) is what the scorer's input-format guard reads: which of the
    attributes the model was trained on the feed still carries."""
    keys, listing_ok = _list_s3_files(date_str)
    if not keys:
        if listing_ok:
            return [], None
        return fetch_probsevere_day(date_str, refresh=True), None
    keys = list(keys)
    for _ in range(max_newest_tries):
        picked = live_keys(keys)
        if not picked:
            return [], None
        newest, raw = _fetch_one(picked[-1])
        if newest is not None:
            steps = _fetch_steps(picked[:-1]) + [newest]
            print(f"  ProbSevere: {len(keys)} files for {date_str}; newest {picked[-1].rsplit('/', 1)[-1]} "
                  f"+ {len(steps) - 1} earlier slots at the same phase")
            return steps, property_census(raw)
        keys = [k for k in keys if k != picked[-1]]
    return [], None


_DOCUMENTED_FORMAT = {"MAXRC_EMISS": "rate", "MAXRC_ICECF": "rate", "AVG_BEAM_HGT": "km"}


def property_census(data: dict) -> dict:
    """Which attributes one ProbSevere document carries, over its storm objects: per ``properties`` key the
    number of objects carrying it (``N/A`` counts as absent), per ``models`` entry the number with a PROB, and
    for the three string-valued attributes how many are in the documented string format (``parse_string_
    attributes``). The parser itself never reports an absent key -- it reads it as 0.0 or leaves it missing --
    so this is how a run can know (NOAA's 2025-08-06 format change removed PS, VIL_DENSITY and MAXRC_ICECF)."""
    features = data.get("features") or []
    props: dict[str, int] = {}
    models: dict[str, int] = {}
    documented = {k: 0 for k in _DOCUMENTED_FORMAT}
    for feat in features:
        p = feat.get("properties") or {}
        for k, v in p.items():
            if v is None or v == "N/A":
                continue
            props[str(k)] = props.get(str(k), 0) + 1
        feat_models = feat.get("models") if isinstance(feat.get("models"), dict) else {}
        for name, m in feat_models.items():
            if isinstance(m, dict) and m.get("PROB") not in (None, "N/A"):
                models[str(name)] = models.get(str(name), 0) + 1
        for k, kind in _DOCUMENTED_FORMAT.items():
            text = str(p.get(k, ""))
            if (kind == "rate" and _RATE_RE.match(text)) or (kind == "km" and _KM_RE.search(text)):
                documented[k] += 1
    return {"valid_time": data.get("validTime"), "n_objects": len(features),
            "n_attributes": len(props), "properties": dict(sorted(props.items())),
            "models": dict(sorted(models.items())), "documented_string_format": documented}


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
