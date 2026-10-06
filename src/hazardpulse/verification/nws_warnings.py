"""NWS storm-based TORNADO WARNINGS (VTEC ``TO.W``) as a verification baseline.

What this module answers
========================
For a storm object at ``(lat, lon)`` and UTC instant ``t`` -- the tuple the
HazardPulse tornado model scores every 30 minutes -- what had NWS forecasters
done?  :func:`tor_warning_state` returns, per point:

``active_now``
    the point lies inside a tornado-warning polygon *in effect* at ``t``.  The
    polygon is the warning's state at ``t``: Severe Weather Statements (SVS)
    that shrink the polygon, cancel the warning or let it expire are applied,
    so a location dropped from a warning stops being warned at the SVS instant.
``issued_within_next_60min``
    a NEW tornado warning whose *issuance polygon* contains the storm is issued
    in the half-open window ``(t, t + 60 min]``.  With ``motion_east_ms`` /
    ``motion_south_ms`` (ProbSevere's motion convention) the storm is advected
    to the issuance instant before the containment test; without motion the
    current position is tested (a lower bound -- a warning polygon is drawn
    from the storm's position *at issuance* forward along its track, so a
    position 40 min upstream is frequently outside it).
``minutes_since_issue``
    for an ``active_now`` point, minutes from the issuance of the
    earliest-issued in-effect warning covering it to ``t``; NaN otherwise.

Source (verified 2026-10-02)
============================
Iowa Environmental Mesonet (IEM) watch/warning GIS service::

    https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py
        ?accept=shapefile&sts=YYYY-01-01T00:00Z&ets=YYYY+1-01-01T00:00Z
        &timeopt=1&phenomena=TO&significance=W&limitps=yes
        &limit1=yes&addsvs=yes

* ``limitps=yes`` + ``phenomena=TO`` + ``significance=W``: tornado warnings only.
* ``limit1=yes``: storm-based (``GTYPE='P'``) polygons only -- no county rows.
* ``addsvs=yes``: include the polygons of every follow-up statement.  In the
  service's SQL this is ``status != 'CAN'`` (without it: ``status = 'NEW'``).
  A CAN is not a polygon state of its own; IEM truncates the previous state's
  ``polygon_end`` to the cancellation instant.
* ``timeopt=1``: rows whose ``coalesce(issue, polygon_begin)`` lies in
  ``[sts, ets)`` -- each year file holds the events *issued* in that year, so
  per-year files partition the record with no overlap.

The annual pickup archive ``/pickup/wwa/{year}_tsmf_sbw.zip`` was checked and
rejected as the source: for 2024 it holds 34,748 rows, every one ``STATUS=NEW``
(MEASURED) -- issuance polygons only, no SVS updates.  Its 2025 file was also
generated at 2025-12-31 02:22 local, before the year ended.

Field semantics (IEM ``/info/datasets/vtec.html`` + the service source)
-----------------------------------------------------------------------
Every timestamp field is a 12-character ``YYYYMMDDHHMM`` string in **UTC**
("The presented timestamps are always in UTC timezone"; the service formats
each with ``to_char(... at time zone 'UTC', 'YYYYMMDDHH24MI')``), minute
precision.  The clock was witnessed independently -- see *Clock witness*.

==========  ============================================================
WFO         3-char issuing NWS office (``OUN`` = Norman OK)
ISSUED      event start; updated over the event's life             [UTC]
EXPIRED     event end; updated by cancellations                    [UTC]
INIT_ISS    issuance of the product that started the event         [UTC]
INIT_EXP    expiry stated at first issuance; never updated         [UTC]
PHENOM      VTEC phenomenon (``TO`` tornado)
SIG         VTEC significance (``W`` warning)
GTYPE       ``P`` storm-based polygon, ``C`` county/zone
ETN         VTEC event tracking number (per office, per phenomenon, per year)
STATUS      VTEC action of the product that carried this polygon:
            ``NEW`` issuance, ``CON`` continuation (possibly reduced polygon),
            ``EXP`` expiration statement, ``COR`` correction (``CAN`` excluded)
AREA_KM2    IEM-computed polygon area (km^2)
UPDATED     when the event's lifecycle was last updated               [UTC]
POLY_BEG    instant this polygon state takes effect (= its product time) [UTC]
POLY_END    instant this polygon state stops being in effect           [UTC]
HAILTAG     IBW hail tag (inches);  WINDTAG  IBW wind tag (MPH)
TORNTAG     IBW tornado tag: ``RADAR INDICATED`` / ``OBSERVED``
DAMAGTAG    IBW damage tag: ``CONSIDERABLE`` (PDS) / ``CATASTROPHIC`` (emergency)
EMERGENC    tornado emergency at any point in the event's life
PROD_ID     IEM product id ``YYYYMMDDHHMM-CCCC-TTAAII-AWIPSID[-BBB]``; the
            leading 12 digits are the product's issuance instant      [UTC]
FCSTER      product signature (not retained here)
VTEC_YR     VTEC year of the event (ETNs reset every year)
==========  ============================================================

One row per **polygon state**.  A state is in effect on the half-open interval
``[valid_from, valid_to)`` with ``valid_from = POLY_BEG`` and ``valid_to =
POLY_END``.  Measured on 2021-2025 (MEASURED): the states of one event tile
``[INIT_ISS, EXPIRED)`` end to end; every deviation is listed below and
handled explicitly.

Deviations in the IEM record and how they are handled
-----------------------------------------------------
* **EXP statement after expiry** (41-77 rows/yr): ``POLY_BEG > POLY_END``.
  An empty interval -- such a state is never in effect.  IEM also sets the
  *previous* state's ``POLY_END`` to that late statement's instant, so 370
  states (2020-2025) appear to outlive the warning by 1-11 min.  The warning
  legally ends at its VTEC end: ``valid_to = min(POLY_END, EXPIRED)``.  A clip
  larger than 20 min raises (it would mean ``EXPIRED`` changed meaning).
* **NEW state starting after its own product** (3 rows in 5 yr: LSX 20/21
  2024, CAE 9 2025): ``POLY_BEG`` later than the TOR product that issued it,
  leaving the warning with no polygon for 11-14 min.  Repaired: a NEW state is
  in effect from its product's issuance instant (``valid_from =
  min(POLY_BEG, product_time)``).  Counted in the build metadata.
* **Duplicate / same-minute states** (<= 4 per year): two CON (or NEW + RRA
  retransmission) states with the same ``POLY_BEG`` and different
  ``POLY_END``.  ``active_now`` is a union, so the effect is at most a few
  minutes of the older, larger polygon.  Counted (``chain_overlaps``).
* **ETN reused within a year** (EAX 1 2022; LIX 1-4 2025): rows of two
  different events share ``(WFO, ETN, VTEC_YR)``; some carry an empty
  ``ISSUED``.  Each state's warning issuance is resolved as the latest NEW
  product of the same key at or before the state (not from ``ISSUED``).
* **Event with no NEW row** (AKQ 49 2022): its issuance is unknown; its CON
  state is kept for ``active_now`` and its ``issue_time`` falls back to
  ``INIT_ISS`` / ``ISSUED`` / the state's own start (``issue_source``).

Clock witness (MEASURED 2026-10-02; encoded in ``tests/test_nws_warnings.py``)
----------------------------------------------------------------------------
NWS product ``202404272216-KOUN-WFUS54-TOROUN`` (OUN TO.W 0036, 2024-04-27,
Kingfisher/Logan/Garfield counties, Oklahoma) states its issuance three ways:
WMO header ``WFUS54 KOUN 272216`` (UTC), VTEC ``240427T2216Z-240427T2300Z``,
and the local line ``516 PM CDT Sat Apr 27 2024`` (CDT = UTC-5).  All three
give 2024-04-27 22:16 UTC; IEM's ``ISSUED`` / ``POLY_BEG`` read
``202404272216`` and ``EXPIRED`` ``202404272300`` (= ``Until 600 PM CDT``).
Offset 0 min.  Were IEM on CDT the fields would read 17:16; on CST, 16:16.

Point-in-polygon
================
Exact crossing-number (even-odd) test against every ring of a polygon state,
so multipolygons and holes are handled by construction.  Edges are straight in
(lon, lat) -- the interpretation of NWS ``LAT...LON`` polygons used by AWIPS
WarnGen, IEM's PostGIS geometry and the shapefile itself.  The orientation
predicate is evaluated in IEEE double without division; a point can only be
misclassified within ~1e-12 deg (~0.1 micrometre) of an edge, and a point
exactly on an edge is decided deterministically by the half-open rule.  No
bounding-box approximation decides membership: the boxes only prune
candidates.  Polygons crossing the antimeridian are rejected at build time
(none exist in 2021-2025: every TO.W office is CONUS or Honolulu).

Index
=====
Query points are sorted once by the composite key ``(time bin, longitude)``.
Each polygon state is expanded into the (state, time bin) pairs its validity
window overlaps; for each pair two ``searchsorted`` calls return the
contiguous run of points of that bin whose longitude lies in the state's box.
Only those candidates are filtered on exact time, latitude, and then the exact
polygon test -- all vectorised across every state at once (no Python loop over
warnings).  Cost ~ O(N log N + candidates x max_edges).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import re
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "IEM_WATCHWARN_URL",
    "CACHE_ROOT",
    "MAX_STATE_LIFETIME_S",
    "TorWarnings",
    "TorWarningQuery",
    "ProductClocks",
    "NwsWarningDataError",
    "iem_tor_sbw_url",
    "fetch_raw_year",
    "parse_iem_sbw_zip",
    "build_year",
    "load_tor_warnings",
    "warnings_from_iem_zip",
    "query_tor_warnings",
    "tor_warning_state",
    "to_epoch_seconds",
    "parse_product_clocks",
    "clock_witness",
    "summarize",
]

# --------------------------------------------------------------------------
# Paths, source, constants
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(
    os.environ.get(
        "HAZARDPULSE_NWS_WARNINGS_CACHE",
        str(PROJECT_ROOT / ".cache" / "nws_warnings"),
    )
)
RAW_SUBDIR = "raw"
COMPACT_SUBDIR = "compact"

IEM_WATCHWARN_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
IEM_NWSTEXT_URL = "https://mesonet.agron.iastate.edu/api/1/nwstext/{product_id}"
USER_AGENT = "hazardpulse-nws-warnings/0.1 (+https://github.com/coherence-energy-labs/hazardpulse)"

#: Compact-file format.  Bump on any change to the arrays or their semantics;
#: a compact file of another version is rebuilt from the raw download.
FORMAT_VERSION = 2

#: Upper bound on (state valid_to - warning issuance).  The longest observed in
#: 2021-2025 is 1.13 h (MEASURED); the build refuses data that exceeds this, so
#: the year-coverage rule in :func:`query_tor_warnings` cannot be silently wrong.
MAX_STATE_LIFETIME_S = 3 * 3600

#: Metres per degree of latitude on the mean-radius sphere (R = 6371008.8 m).
M_PER_DEG = 6371008.8 * np.pi / 180.0

_NO_TIME = np.iinfo(np.int64).min
_KEY_STRIDE = 512.0  # > 360: the longitude span inside one time bin
_KEY_EPS = 1e-6  # deg; widens candidate ranges past float rounding of the key
_CAND_CHUNK = 4_000_000  # candidate pairs materialised at once (memory bound)
#: Largest clip of a state's end back to its event's EXPIRED that is accepted
#: as the late-EXP artefact (observed max 11 min, 2020-2025).
_MAX_EXPIRY_CLIP_S = 20 * 60

#: issue_source codes
ISSUE_FROM_NEW_PRODUCT = 0
ISSUE_FROM_INIT_ISS = 1
ISSUE_FROM_ISSUED = 2
ISSUE_FROM_STATE_START = 3


class NwsWarningDataError(RuntimeError):
    """The downloaded record violates an invariant this module relies on."""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def iem_tor_sbw_url(year: int) -> str:
    """IEM service URL for every TO.W storm-based polygon state issued in ``year``."""

    if not 2002 <= int(year) <= 2100:
        raise ValueError(f"year {year} outside the storm-based-warning era (>= 2002)")
    q = [
        ("accept", "shapefile"),
        ("sts", f"{int(year):04d}-01-01T00:00Z"),
        ("ets", f"{int(year) + 1:04d}-01-01T00:00Z"),
        ("timeopt", "1"),
        ("phenomena", "TO"),
        ("significance", "W"),
        ("limitps", "yes"),
        ("limit1", "yes"),
        ("addsvs", "yes"),
    ]
    return IEM_WATCHWARN_URL + "?" + "&".join(f"{k}={v}" for k, v in q)


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else CACHE_ROOT


def _raw_path(year: int, cache_dir: str | os.PathLike | None = None) -> Path:
    return _cache_dir(cache_dir) / RAW_SUBDIR / f"watchwarn_TO_W_sbw_addsvs_{int(year)}.zip"


def _compact_path(year: int, cache_dir: str | os.PathLike | None = None) -> Path:
    return _cache_dir(cache_dir) / COMPACT_SUBDIR / f"tor_warnings_{int(year)}.npz"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_get(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        if attempt < 2:
            time.sleep(5.0 * (attempt + 1))  # polite back-off
    assert last is not None
    raise last


def _check_zip(data: bytes, url: str) -> None:
    try:
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    except zipfile.BadZipFile as exc:
        head = data[:200].decode("latin-1", "replace")
        raise NwsWarningDataError(f"{url}: response is not a zip ({head!r})") from exc
    if not any(n.endswith(".shp") for n in names) or not any(n.endswith(".dbf") for n in names):
        raise NwsWarningDataError(f"{url}: zip lacks .shp/.dbf members: {names}")


def fetch_raw_year(
    year: int,
    *,
    cache_dir: str | os.PathLike | None = None,
    refresh: bool = False,
    timeout: int = 300,
) -> Path:
    """Download (once) the IEM TO.W storm-based zip for ``year``; return its path.

    A sidecar ``.manifest.json`` records the URL, the UTC fetch instant, byte
    count and SHA-256.  A file already present without a manifest (e.g. fetched
    by hand with the same URL) is adopted, its fetch instant taken from the
    file's modification time and marked as such.
    """

    url = iem_tor_sbw_url(year)
    raw = _raw_path(year, cache_dir)
    manifest = raw.with_suffix(".manifest.json")
    if raw.exists() and not refresh:
        if not manifest.exists():
            data = raw.read_bytes()
            _check_zip(data, url)
            manifest.write_text(
                json.dumps(
                    {
                        "url": url,
                        "fetched_utc": _utc_iso(raw.stat().st_mtime),
                        "fetched_utc_source": "file_mtime",
                        "bytes": len(data),
                        "sha256": _sha256(data),
                    },
                    indent=1,
                )
            )
        return raw
    raw.parent.mkdir(parents=True, exist_ok=True)
    fetched = time.time()
    data = _http_get(url, timeout)
    _check_zip(data, url)
    tmp = raw.with_suffix(".zip.part")
    tmp.write_bytes(data)
    os.replace(tmp, raw)
    manifest.write_text(
        json.dumps(
            {
                "url": url,
                "fetched_utc": _utc_iso(fetched),
                "fetched_utc_source": "request",
                "bytes": len(data),
                "sha256": _sha256(data),
            },
            indent=1,
        )
    )
    return raw


# --------------------------------------------------------------------------
# Shapefile + DBF readers (no third-party GIS dependency)
# --------------------------------------------------------------------------


def _read_shp_polygons(shp: bytes) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Parse an ESRI .shp of polygons -> per record ``(parts, xy)`` or None (null shape)."""

    if len(shp) < 100:
        raise NwsWarningDataError("shp shorter than its 100-byte header")
    file_code = struct.unpack(">i", shp[0:4])[0]
    file_words = struct.unpack(">i", shp[24:28])[0]
    version, shape_type = struct.unpack("<ii", shp[28:36])
    if file_code != 9994 or version != 1000:
        raise NwsWarningDataError(f"not a shapefile (code={file_code}, version={version})")
    if file_words * 2 != len(shp):
        raise NwsWarningDataError(f"shp length {len(shp)} != header {file_words * 2}")
    if shape_type not in (0, 5, 15, 25):
        raise NwsWarningDataError(f"shapefile type {shape_type} is not Polygon")
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    pos = 100
    expect = 1
    while pos < len(shp):
        rec_no, words = struct.unpack(">ii", shp[pos : pos + 8])
        if rec_no != expect:
            raise NwsWarningDataError(f"shp record {rec_no} where {expect} expected")
        start = pos + 8
        end = start + 2 * words
        if end > len(shp):
            raise NwsWarningDataError(f"shp record {rec_no} overruns the file")
        stype = struct.unpack("<i", shp[start : start + 4])[0]
        if stype == 0:
            out.append(None)
        elif stype in (5, 15, 25):
            n_parts, n_points = struct.unpack("<ii", shp[start + 36 : start + 44])
            if n_parts < 1 or n_points < 4:
                raise NwsWarningDataError(f"shp record {rec_no}: {n_parts} parts, {n_points} points")
            p0 = start + 44
            parts = np.frombuffer(shp, dtype="<i4", count=n_parts, offset=p0).astype(np.int64)
            xy0 = p0 + 4 * n_parts
            if xy0 + 16 * n_points > end:
                raise NwsWarningDataError(f"shp record {rec_no}: points overrun the record")
            xy = np.frombuffer(shp, dtype="<f8", count=2 * n_points, offset=xy0).reshape(n_points, 2).copy()
            if parts[0] != 0 or np.any(np.diff(parts) <= 0) or parts[-1] >= n_points:
                raise NwsWarningDataError(f"shp record {rec_no}: bad part offsets {parts.tolist()}")
            out.append((parts, xy))
        else:
            raise NwsWarningDataError(f"shp record {rec_no}: shape type {stype}")
        pos = end
        expect += 1
    return out


def _read_dbf(dbf: bytes, encoding: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Parse a dBASE III table -> ({field: values}, deleted_mask)."""

    if len(dbf) < 32:
        raise NwsWarningDataError("dbf shorter than its header")
    n_rec, header_len, rec_len = struct.unpack("<IHH", dbf[4:12])
    fields: list[tuple[str, str, int]] = []
    off = 32
    while dbf[off] != 0x0D:
        desc = dbf[off : off + 32]
        name = desc[:11].split(b"\x00", 1)[0].decode("ascii").strip()
        fields.append((name, chr(desc[11]), desc[16]))
        off += 32
        if off >= header_len:
            raise NwsWarningDataError("dbf field descriptors run past the header")
    if 1 + sum(f[2] for f in fields) != rec_len:
        raise NwsWarningDataError("dbf record length disagrees with its fields")
    if header_len + n_rec * rec_len > len(dbf):
        raise NwsWarningDataError("dbf records overrun the file")
    dtype = np.dtype([("_del", "S1")] + [(n, f"S{w}") for n, _, w in fields])
    recs = np.frombuffer(dbf, dtype=dtype, count=n_rec, offset=header_len)
    deleted = recs["_del"] == b"*"
    cols: dict[str, np.ndarray] = {}
    for name, ftype, _w in fields:
        raw = recs[name]
        if ftype in ("C", "D"):
            cols[name] = np.array([v.decode(encoding).strip() for v in raw.tolist()], dtype=object)
        elif ftype in ("N", "F"):
            vals = np.full(n_rec, np.nan)
            for i, v in enumerate(raw.tolist()):
                s = v.strip()
                if s and not s.startswith(b"*"):
                    vals[i] = float(s)
            cols[name] = vals
        elif ftype == "L":
            cols[name] = np.array([v.strip().upper() in (b"T", b"Y") for v in raw.tolist()], dtype=bool)
        else:
            raise NwsWarningDataError(f"dbf field {name}: unsupported type {ftype}")
    return cols, deleted


_IEM_TS = re.compile(r"^\d{12}$")


def _parse_iem_ts(values: Sequence[str], field_name: str) -> np.ndarray:
    """``YYYYMMDDHHMM`` (UTC) strings -> int64 POSIX seconds; blank -> ``_NO_TIME``."""

    out = np.full(len(values), _NO_TIME, dtype=np.int64)
    for i, s in enumerate(values):
        if not s:
            continue
        if not _IEM_TS.match(s):
            raise NwsWarningDataError(f"{field_name}: {s!r} is not YYYYMMDDHHMM")
        dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]), tzinfo=timezone.utc)
        out[i] = int(dt.timestamp())
    return out


_PROD_ID = re.compile(r"^(\d{12})-([A-Z0-9]{4})-([A-Z0-9]{6})-([A-Z0-9]{3,6})(?:-([A-Z]{3}))?$")


def _ring_area_km2(lon: np.ndarray, lat: np.ndarray) -> float:
    """Signed area of a closed lon/lat ring on the mean-radius sphere (km^2)."""

    lam = np.radians(lon)
    phi = np.radians(lat)
    r_km = 6371.0088
    return float(0.5 * r_km * r_km * np.sum((lam[1:] - lam[:-1]) * (np.sin(phi[1:]) + np.sin(phi[:-1]))))


# --------------------------------------------------------------------------
# Parse + build
# --------------------------------------------------------------------------


def parse_iem_sbw_zip(data: bytes, *, expect_year: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Parse an IEM watchwarn.py TO.W zip into polygon-state arrays + build metadata.

    Returns ``(arrays, meta)``.  ``arrays`` holds one entry per polygon state
    (see :class:`TorWarnings` for the meaning of each) plus the ragged geometry
    (``ring_start``, ``poly_ring_start``, ``vx``, ``vy``).  ``meta`` records
    every integrity count; violations of a hard invariant raise
    :class:`NwsWarningDataError`.
    """

    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    shp_name = next(n for n in names if n.endswith(".shp"))
    stem = shp_name[:-4]
    encoding = "latin-1"
    if stem + ".cpg" in names:
        cpg = zf.read(stem + ".cpg").decode("ascii", "replace").strip().lower()
        encoding = {"iso-8859-1": "latin-1", "utf-8": "utf-8", "utf8": "utf-8"}.get(cpg, cpg or "latin-1")
    shapes = _read_shp_polygons(zf.read(shp_name))
    cols, deleted = _read_dbf(zf.read(stem + ".dbf"), encoding)
    n = len(deleted)
    if n != len(shapes):
        raise NwsWarningDataError(f"dbf has {n} records, shp {len(shapes)}")
    if stem + ".prj" in names:
        prj = zf.read(stem + ".prj").decode("ascii", "replace")
        if "WGS_1984" not in prj and "WGS 84" not in prj:
            raise NwsWarningDataError(f"unexpected projection: {prj[:120]}")

    keep = ~deleted & np.array([s is not None for s in shapes], dtype=bool)
    meta: dict = {
        "n_records": int(n),
        "n_deleted": int(deleted.sum()),
        "n_null_shapes": int(sum(s is None for s in shapes)),
    }
    for col, want in (("PHENOM", "TO"), ("SIG", "W"), ("GTYPE", "P")):
        bad = {str(v) for v in cols[col][keep] if v != want}
        if bad:
            raise NwsWarningDataError(f"{col} values {sorted(bad)} in a TO.W storm-based file")

    idx = np.flatnonzero(keep)
    wfo = np.array([cols["WFO"][i] for i in idx], dtype="U4")
    etn = cols["ETN"][idx].astype(np.int32)
    vtec_year = cols["VTEC_YR"][idx].astype(np.int16) if "VTEC_YR" in cols else np.full(len(idx), -1, np.int16)
    status = np.array([cols["STATUS"][i] for i in idx], dtype="U3")
    product_id = np.array([cols["PROD_ID"][i] for i in idx], dtype="U40")
    issued = _parse_iem_ts([cols["ISSUED"][i] for i in idx], "ISSUED")
    expired = _parse_iem_ts([cols["EXPIRED"][i] for i in idx], "EXPIRED")
    init_iss = _parse_iem_ts([cols["INIT_ISS"][i] for i in idx], "INIT_ISS")
    init_exp = _parse_iem_ts([cols["INIT_EXP"][i] for i in idx], "INIT_EXP")
    poly_beg = _parse_iem_ts([cols["POLY_BEG"][i] for i in idx], "POLY_BEG")
    poly_end = _parse_iem_ts([cols["POLY_END"][i] for i in idx], "POLY_END")
    if np.any(poly_beg == _NO_TIME) or np.any(poly_end == _NO_TIME):
        raise NwsWarningDataError("polygon state without POLY_BEG/POLY_END")

    product_time = np.full(len(idx), _NO_TIME, dtype=np.int64)
    for j, pid in enumerate(product_id):
        m = _PROD_ID.match(pid)
        if not m:
            raise NwsWarningDataError(f"PROD_ID {pid!r} does not parse")
        product_time[j] = _parse_iem_ts([m.group(1)], "PROD_ID")[0]

    # -- validity interval ---------------------------------------------------
    is_new = status == "NEW"
    valid_from = poly_beg.copy()
    repair = is_new & (product_time < poly_beg)
    valid_from[repair] = product_time[repair]
    valid_to = poly_end.copy()
    # A late EXP statement (issued after the warning's VTEC end) makes IEM set
    # the previous state's POLY_END to the statement's instant, so that state
    # appears to outlive the warning by 1-11 min (370 states in 2020-2025,
    # MEASURED).  The warning legally ends at its VTEC end (EXPIRED): clip.
    # An event cannot expire before it is issued. Where a row's EXPIRED precedes its own ISSUED, the expiry
    # belongs to ANOTHER event of a reused (WFO, ETN, year) key -- 2026: JKL 24 was issued at 22:05 and
    # cancelled at 22:20, then a second JKL 24 was issued at 22:36 (initial expiry 23:15, EXP at 23:15), and
    # IEM carried the first event's 22:20 onto the second's rows, which read as a polygon outliving its
    # event by 55 min. Such a row is clipped to its own event's initial expiry (INIT_EXP) instead.
    expiry = expired.copy()
    foreign = (expired != _NO_TIME) & (issued != _NO_TIME) & (expired < issued) & (init_exp != _NO_TIME)
    expiry[foreign] = init_exp[foreign]
    meta["expiry_before_issuance"] = int(foreign.sum())
    meta["expiry_before_issuance_rows"] = [f"{wfo[j]} {etn[j]} {vtec_year[j]} {product_id[j]}"
                                           for j in np.flatnonzero(foreign)]
    expired = expiry
    clip = (expired != _NO_TIME) & (valid_to > expired) & (valid_to > valid_from)
    clip_s = (valid_to - expired)[clip]
    if clip_s.size and clip_s.max() > _MAX_EXPIRY_CLIP_S:
        j = np.flatnonzero(clip)[int(np.argmax(clip_s))]
        raise NwsWarningDataError(
            f"{product_id[j]}: polygon state outlives its event's EXPIRED by "
            f"{clip_s.max() / 60:.0f} min -- not the late-EXP artefact; EXPIRED semantics changed?"
        )
    valid_to[clip] = np.maximum(expired[clip], valid_from[clip])
    meta["clip_to_event_expiry"] = int(clip.sum())
    meta["clip_to_event_expiry_max_min"] = float(clip_s.max() / 60.0) if clip_s.size else 0.0
    meta["repair_new_state_start"] = int(repair.sum())
    meta["repaired_rows"] = [f"{wfo[j]} {etn[j]} {vtec_year[j]} {product_id[j]}" for j in np.flatnonzero(repair)]
    meta["empty_states"] = int(np.sum(valid_to <= valid_from))
    meta["status_counts"] = {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))}
    meta["new_product_time_ne_issued"] = int(np.sum(is_new & (issued != _NO_TIME) & (product_time != issued)))
    meta["state_start_ne_product_time"] = int(np.sum(~is_new & (poly_beg != product_time)))

    # -- warning issuance per state: latest NEW product of the same key -----
    issue_time = np.full(len(idx), _NO_TIME, dtype=np.int64)
    issue_source = np.full(len(idx), -1, dtype=np.int8)
    keys = np.array([f"{w}|{e}|{y}" for w, e, y in zip(wfo, etn, vtec_year)])
    order = np.lexsort((valid_from, keys))
    chain_overlaps = 0
    keys_without_new = 0
    keys_with_multiple_new = 0
    start = 0
    while start < len(order):
        stop = start
        while stop < len(order) and keys[order[stop]] == keys[order[start]]:
            stop += 1
        grp = order[start:stop]
        new_rows = grp[is_new[grp]]
        new_times = np.sort(product_time[new_rows])
        if len(new_rows) == 0:
            keys_without_new += 1
        elif len(new_rows) > 1:
            keys_with_multiple_new += 1
        for j in grp:
            k = np.searchsorted(new_times, valid_from[j], side="right") - 1
            if k >= 0:
                issue_time[j] = new_times[k]
                issue_source[j] = ISSUE_FROM_NEW_PRODUCT
            elif init_iss[j] != _NO_TIME:
                issue_time[j] = init_iss[j]
                issue_source[j] = ISSUE_FROM_INIT_ISS
            elif issued[j] != _NO_TIME:
                issue_time[j] = issued[j]
                issue_source[j] = ISSUE_FROM_ISSUED
            else:
                issue_time[j] = valid_from[j]
                issue_source[j] = ISSUE_FROM_STATE_START
        live = grp[valid_to[grp] > valid_from[grp]]
        if len(live) > 1:
            lv = live[np.argsort(valid_from[live], kind="stable")]
            chain_overlaps += int(np.sum(valid_to[lv[:-1]] > valid_from[lv[1:]]))
        start = stop
    meta["chain_overlaps"] = chain_overlaps
    meta["keys"] = int(len(np.unique(keys)))
    meta["keys_without_new"] = keys_without_new
    meta["keys_with_multiple_new"] = keys_with_multiple_new
    meta["issue_source_counts"] = {
        name: int(np.sum(issue_source == code))
        for name, code in (
            ("new_product", ISSUE_FROM_NEW_PRODUCT),
            ("init_iss", ISSUE_FROM_INIT_ISS),
            ("issued", ISSUE_FROM_ISSUED),
            ("state_start", ISSUE_FROM_STATE_START),
        )
    }
    live_all = valid_to > valid_from
    lifetime = (valid_to - issue_time)[live_all]
    meta["max_state_lifetime_s"] = int(lifetime.max()) if lifetime.size else 0
    if lifetime.size and lifetime.max() > MAX_STATE_LIFETIME_S:
        raise NwsWarningDataError(
            f"a polygon state outlives its issuance by {lifetime.max() / 3600:.2f} h "
            f"> MAX_STATE_LIFETIME_S; the query's year-coverage rule would be unsound"
        )
    if np.any(live_all & (issue_time > valid_from)):
        raise NwsWarningDataError("a live state begins before its warning was issued")
    beyond = live_all & (expired != _NO_TIME) & (valid_to > expired)
    if beyond.any():  # unreachable after the clip above; kept as the invariant's gate
        raise NwsWarningDataError(f"{int(beyond.sum())} live states outlive their event's EXPIRED")

    # -- year partition (IEM selects on coalesce(issue, polygon_begin)) -----
    if expect_year is not None:
        sel_t = np.where(issued != _NO_TIME, issued, poly_beg)
        y0 = int(datetime(expect_year, 1, 1, tzinfo=timezone.utc).timestamp())
        y1 = int(datetime(expect_year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
        outside = int(np.sum((sel_t < y0) | (sel_t >= y1)))
        if outside:
            raise NwsWarningDataError(f"{outside} rows were issued outside {expect_year}")

    # -- geometry ------------------------------------------------------------
    ring_lens: list[int] = []
    poly_ring_counts: list[int] = []
    vx_parts: list[np.ndarray] = []
    vy_parts: list[np.ndarray] = []
    rings_closed_by_us = 0
    multipart = 0
    area_ratio = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        parts, xy = shapes[i]  # type: ignore[misc]
        bounds = list(parts) + [len(xy)]
        if len(parts) > 1:
            multipart += 1
        signed = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            ring = xy[a:b]
            if not np.array_equal(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[:1]])
                rings_closed_by_us += 1
            if len(ring) < 4:
                raise NwsWarningDataError(f"{product_id[j]}: ring with {len(ring)} vertices")
            vx_parts.append(ring[:, 0])
            vy_parts.append(ring[:, 1])
            ring_lens.append(len(ring))
            signed += _ring_area_km2(ring[:, 0], ring[:, 1])
        poly_ring_counts.append(len(bounds) - 1)
        a_iem = float(cols["AREA_KM2"][i])
        if np.isfinite(a_iem) and a_iem > 0:
            area_ratio[j] = abs(signed) / a_iem
    vx = np.concatenate(vx_parts) if vx_parts else np.zeros(0)
    vy = np.concatenate(vy_parts) if vy_parts else np.zeros(0)
    ring_start = np.concatenate([[0], np.cumsum(ring_lens)]).astype(np.int64)
    poly_ring_start = np.concatenate([[0], np.cumsum(poly_ring_counts)]).astype(np.int64)
    if np.any(~np.isfinite(vx)) or np.any(~np.isfinite(vy)):
        raise NwsWarningDataError("non-finite polygon vertex")
    if np.any(np.abs(vy) > 90) or np.any(np.abs(vx) > 180):
        raise NwsWarningDataError("polygon vertex outside lon/lat range")
    minlon = np.empty(len(idx))
    maxlon = np.empty(len(idx))
    minlat = np.empty(len(idx))
    maxlat = np.empty(len(idx))
    for j in range(len(idx)):
        a = ring_start[poly_ring_start[j]]
        b = ring_start[poly_ring_start[j + 1]]
        minlon[j], maxlon[j] = vx[a:b].min(), vx[a:b].max()
        minlat[j], maxlat[j] = vy[a:b].min(), vy[a:b].max()
    if np.any(maxlon - minlon >= 180.0):
        raise NwsWarningDataError("a polygon spans >= 180 deg of longitude (antimeridian not supported)")
    meta["multipart_states"] = multipart
    meta["rings_closed_by_parser"] = rings_closed_by_us
    finite = np.isfinite(area_ratio)
    meta["area_check_n"] = int(finite.sum())
    meta["area_ratio_median"] = float(np.median(area_ratio[finite])) if finite.any() else float("nan")
    meta["area_ratio_off_by_gt_2pct"] = int(np.sum(np.abs(area_ratio[finite] - 1.0) > 0.02))
    meta["n_states"] = int(len(idx))
    meta["n_issuances"] = int(is_new.sum())
    meta["n_unique_issuances"] = int(len({(a, b, c, d) for a, b, c, d, nw in zip(wfo, etn, vtec_year, product_time, is_new) if nw}))

    arrays = {
        "wfo": wfo,
        "etn": etn,
        "vtec_year": vtec_year,
        "status": status,
        "product_id": product_id,
        "product_time": product_time,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "issue_time": issue_time,
        "issue_source": issue_source,
        "is_issuance": is_new,
        "expired": expired,
        "init_exp": init_exp,
        "tornado_tag": np.array([cols["TORNTAG"][i] for i in idx], dtype="U16"),
        "damage_tag": np.array([cols["DAMAGTAG"][i] for i in idx], dtype="U16"),
        "hail_tag_in": cols["HAILTAG"][idx].astype(np.float32),
        "is_emergency": cols["EMERGENC"][idx].astype(bool),
        "area_km2": cols["AREA_KM2"][idx].astype(np.float32),
        "minlon": minlon,
        "maxlon": maxlon,
        "minlat": minlat,
        "maxlat": maxlat,
        "ring_start": ring_start,
        "poly_ring_start": poly_ring_start,
        "vx": vx,
        "vy": vy,
    }
    return arrays, meta


def build_year(
    year: int,
    *,
    cache_dir: str | os.PathLike | None = None,
    fetch: bool = True,
    refresh: bool = False,
) -> Path:
    """Fetch (if needed) and compact one year into ``compact/tor_warnings_{year}.npz``."""

    raw = _raw_path(year, cache_dir)
    if fetch or refresh:
        raw = fetch_raw_year(year, cache_dir=cache_dir, refresh=refresh)
    elif not raw.exists():
        raise FileNotFoundError(f"{raw} missing and fetch=False")
    data = raw.read_bytes()
    manifest_path = raw.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    arrays, meta = parse_iem_sbw_zip(data, expect_year=int(year))
    fetched = manifest.get("fetched_utc")
    year_end = datetime(int(year) + 1, 1, 2, tzinfo=timezone.utc)
    complete = bool(fetched) and datetime.strptime(fetched, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) >= year_end
    meta.update(
        {
            "format_version": FORMAT_VERSION,
            "year": int(year),
            "source_url": iem_tor_sbw_url(year),
            "raw_file": raw.name,
            "raw_sha256": _sha256(data),
            "raw_fetched_utc": fetched,
            "raw_fetched_utc_source": manifest.get("fetched_utc_source"),
            "year_complete_at_fetch": complete,
            "built_utc": _utc_iso(time.time()),
            "clock": "all instants are int64 POSIX seconds, UTC",
        }
    )
    out = _compact_path(year, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------
# Loaded record
# --------------------------------------------------------------------------

_ROW_FIELDS = (
    "wfo",
    "etn",
    "vtec_year",
    "status",
    "product_id",
    "product_time",
    "valid_from",
    "valid_to",
    "issue_time",
    "issue_source",
    "is_issuance",
    "expired",
    "init_exp",
    "tornado_tag",
    "damage_tag",
    "hail_tag_in",
    "is_emergency",
    "area_km2",
    "minlon",
    "maxlon",
    "minlat",
    "maxlat",
)


@dataclass(frozen=True, eq=False)
class TorWarnings:
    """Every TO.W polygon state of the loaded years (one row per state).

    Instants are int64 POSIX seconds, UTC.  A state is in effect on
    ``[valid_from, valid_to)``; ``issue_time`` is the issuance instant of the
    warning it belongs to; ``is_issuance`` marks NEW (issuance) states, whose
    polygon is the polygon as issued.  Geometry is ragged: state ``r`` owns
    rings ``poly_ring_start[r]:poly_ring_start[r+1]``; ring ``k`` owns vertices
    ``ring_start[k]:ring_start[k+1]`` of ``vx`` (lon) / ``vy`` (lat), closed.
    """

    years: tuple[int, ...]
    meta: dict
    wfo: np.ndarray
    etn: np.ndarray
    vtec_year: np.ndarray
    status: np.ndarray
    product_id: np.ndarray
    product_time: np.ndarray
    valid_from: np.ndarray
    valid_to: np.ndarray
    issue_time: np.ndarray
    issue_source: np.ndarray
    is_issuance: np.ndarray
    expired: np.ndarray
    init_exp: np.ndarray
    tornado_tag: np.ndarray
    damage_tag: np.ndarray
    hail_tag_in: np.ndarray
    is_emergency: np.ndarray
    area_km2: np.ndarray
    minlon: np.ndarray
    maxlon: np.ndarray
    minlat: np.ndarray
    maxlat: np.ndarray
    source_year: np.ndarray
    ring_start: np.ndarray
    poly_ring_start: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    def __len__(self) -> int:
        return int(len(self.valid_from))

    @property
    def incomplete_years(self) -> tuple[int, ...]:
        return tuple(y for y in self.years if not self.meta.get(str(y), {}).get("year_complete_at_fetch", False))

    def polygon(self, r: int) -> list[np.ndarray]:
        """Rings of state ``r`` as ``(n, 2)`` arrays of (lon, lat)."""

        rings = []
        for k in range(self.poly_ring_start[r], self.poly_ring_start[r + 1]):
            a, b = self.ring_start[k], self.ring_start[k + 1]
            rings.append(np.column_stack([self.vx[a:b], self.vy[a:b]]))
        return rings

    @functools.cached_property
    def _edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flattened ring edges grouped by state: (start, count, x1, y1, x2, y2)."""

        nv = len(self.vx)
        is_ring_last = np.zeros(nv, dtype=bool)
        is_ring_last[self.ring_start[1:] - 1] = True
        src = np.flatnonzero(~is_ring_last)
        x1, y1 = self.vx[src], self.vy[src]
        x2, y2 = self.vx[src + 1], self.vy[src + 1]
        ring_edges = np.diff(self.ring_start) - 1
        edge_ring_end = np.concatenate([[0], np.cumsum(ring_edges)])
        start = edge_ring_end[self.poly_ring_start[:-1]]
        count = edge_ring_end[self.poly_ring_start[1:]] - start
        return start.astype(np.int64), count.astype(np.int64), x1, y1, x2, y2

    def contains(self, rows: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Exact point-in-polygon for pairs ``(rows[i], (lons[i], lats[i]))``."""

        rows = np.asarray(rows, dtype=np.int64)
        start, count, x1, y1, x2, y2 = self._edges
        return _pip_pairs(
            np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64), rows, start, count, x1, y1, x2, y2
        )


def _pip_pairs(
    px: np.ndarray,
    py: np.ndarray,
    rows: np.ndarray,
    e_start: np.ndarray,
    e_count: np.ndarray,
    ex1: np.ndarray,
    ey1: np.ndarray,
    ex2: np.ndarray,
    ey2: np.ndarray,
) -> np.ndarray:
    """Even-odd crossing number, vectorised over (point, polygon) pairs.

    A ray from the point toward +lon crosses edge (x1,y1)-(x2,y2) iff the edge
    straddles the point's latitude half-openly (``(y1 > py) != (y2 > py)``)
    and the point is strictly on the ray's side of the edge.  That side test is
    the sign of the orientation determinant -- no division -- matched to the
    edge direction: upward edges need ``cross > 0``, downward ``cross < 0``.
    """

    n = len(rows)
    inside = np.zeros(n, dtype=bool)
    if n == 0:
        return inside
    cnt = e_count[rows]
    order = np.argsort(-cnt, kind="stable")
    cnt_o = cnt[order]
    base = e_start[rows[order]]
    xx_all = px[order]
    yy_all = py[order]
    res = np.zeros(n, dtype=bool)
    neg = -cnt_o  # ascending
    for k in range(int(cnt_o[0])):
        m = int(np.searchsorted(neg, -k, side="left"))  # pairs with > k edges
        if m == 0:
            break
        e = base[:m] + k
        xa, ya, xb, yb = ex1[e], ey1[e], ex2[e], ey2[e]
        xx = xx_all[:m]
        yy = yy_all[:m]
        straddle = (ya > yy) != (yb > yy)
        cross = (xb - xa) * (yy - ya) - (xx - xa) * (yb - ya)
        up = yb > ya
        hit = straddle & ((up & (cross > 0)) | (~up & (cross < 0)))
        res[:m] ^= hit
    inside[order] = res
    return inside


def _load_compact(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta


def load_tor_warnings(
    years: int | Iterable[int],
    *,
    cache_dir: str | os.PathLike | None = None,
    build_missing: bool = True,
    fetch_missing: bool = True,
) -> TorWarnings:
    """Load the polygon states of the warnings *issued* in ``years``.

    Compact files are (re)built when missing, of another ``FORMAT_VERSION``, or
    stale against the raw download (SHA-256 mismatch).  A query at instant
    ``t`` needs the years of ``t - MAX_STATE_LIFETIME_S`` through ``t + 60
    min``; :func:`query_tor_warnings` refuses a query the loaded years cannot
    answer, so load e.g. 2020-2025 to query all of 2021-2025.
    """

    ys = sorted({int(years)} if isinstance(years, (int, np.integer)) else {int(y) for y in years})
    if not ys:
        raise ValueError("no years requested")
    per_year: list[dict[str, np.ndarray]] = []
    metas: dict[str, dict] = {}
    for y in ys:
        path = _compact_path(y, cache_dir)
        need = not path.exists()
        if not need:
            arrays, meta = _load_compact(path)
            raw = _raw_path(y, cache_dir)
            if meta.get("format_version") != FORMAT_VERSION:
                need = True
            elif raw.exists() and _sha256(raw.read_bytes()) != meta.get("raw_sha256"):
                need = True
        if need:
            if not build_missing:
                raise FileNotFoundError(f"{path} missing or stale and build_missing=False")
            build_year(y, cache_dir=cache_dir, fetch=fetch_missing)
            arrays, meta = _load_compact(path)
        per_year.append(arrays)
        metas[str(y)] = meta
    return _assemble(per_year, ys, metas)


def warnings_from_iem_zip(data: bytes, *, year: int) -> TorWarnings:
    """A :class:`TorWarnings` straight from an IEM watchwarn.py zip (no cache).

    ``year`` labels the issuance year the zip covers; a zip of a shorter window
    (e.g. a test fixture) is marked ``partial_window`` in its metadata.
    """

    arrays, meta = parse_iem_sbw_zip(data, expect_year=int(year))
    meta = dict(meta, year=int(year), partial_window=True, year_complete_at_fetch=False)
    return _assemble([arrays], [int(year)], {str(int(year)): meta})


def _assemble(per_year: list[dict[str, np.ndarray]], ys: list[int], metas: dict[str, dict]) -> TorWarnings:
    merged: dict[str, np.ndarray] = {}
    for f in _ROW_FIELDS:
        merged[f] = np.concatenate([a[f] for a in per_year])
    # ragged geometry: offset each year's indices
    ring_parts, prs_parts, vx_parts, vy_parts = [], [], [], []
    v_off = 0
    r_off = 0
    for a in per_year:
        rs = a["ring_start"]
        prs = a["poly_ring_start"]
        ring_parts.append(rs[:-1] + v_off)
        prs_parts.append(prs[:-1] + r_off)
        vx_parts.append(a["vx"])
        vy_parts.append(a["vy"])
        v_off += int(rs[-1])
        r_off += int(prs[-1])
    merged["ring_start"] = np.concatenate(ring_parts + [np.array([v_off], dtype=np.int64)])
    merged["poly_ring_start"] = np.concatenate(prs_parts + [np.array([r_off], dtype=np.int64)])
    merged["vx"] = np.concatenate(vx_parts)
    merged["vy"] = np.concatenate(vy_parts)
    merged["source_year"] = np.concatenate([np.full(len(a["valid_from"]), y, dtype=np.int16) for a, y in zip(per_year, ys)])
    return TorWarnings(years=tuple(ys), meta=metas, **merged)


# --------------------------------------------------------------------------
# Time coercion (clock discipline)
# --------------------------------------------------------------------------

_EPOCH_MIN = int(datetime(1990, 1, 1, tzinfo=timezone.utc).timestamp())
_EPOCH_MAX = int(datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp())


def to_epoch_seconds(times_utc) -> np.ndarray:
    """Coerce UTC instants to int64 POSIX seconds (floor), refusing ambiguous clocks.

    Accepted: ``numpy.datetime64`` arrays (any unit; *defined* as UTC),
    tz-aware ``pandas`` datetimes (converted to UTC), tz-aware
    :class:`datetime.datetime` objects, ISO-8601 strings carrying ``Z`` or an
    explicit offset, and int/float POSIX seconds.  Refused: naive Python
    datetimes, naive strings, tz-naive pandas data, and numbers outside
    1990-2100 (catches milliseconds / nanoseconds passed as seconds).

    Flooring to whole seconds preserves every comparison with the
    minute-precision warning instants: for integer ``a``, ``floor(t) < a`` iff
    ``t < a``.
    """

    obj = times_utc
    if type(obj).__module__.startswith("pandas"):
        dtype = getattr(obj, "dtype", None)
        if getattr(dtype, "tz", None) is not None:
            import pandas as pd  # only reached when the caller already uses pandas

            obj = pd.DatetimeIndex(obj).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
        elif dtype is not None and getattr(dtype, "kind", "") == "M":
            raise ValueError("tz-naive pandas datetimes: tz_localize('UTC') first (the clock must be explicit)")
        else:
            obj = np.asarray(obj)
    arr = np.asarray(obj).ravel()
    if np.issubdtype(arr.dtype, np.datetime64):
        if np.any(np.isnat(arr)):
            raise ValueError("NaT in times_utc")
        # floor to seconds (astype truncates toward zero; equal to floor after 1970)
        sec = arr.astype("datetime64[s]").astype(np.int64)
    elif np.issubdtype(arr.dtype, np.integer):
        sec = arr.astype(np.int64)
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite POSIX seconds in times_utc")
        sec = np.floor(arr).astype(np.int64)
    elif arr.dtype == object or arr.dtype.kind in ("U", "S"):
        sec = np.empty(arr.shape, dtype=np.int64)
        for i, v in enumerate(arr.tolist()):
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                s = v.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    v = datetime.fromisoformat(s)
                except ValueError as exc:
                    raise ValueError(f"unparseable time {v!r}") from exc
            if isinstance(v, datetime):
                if v.tzinfo is None or v.utcoffset() is None:
                    raise ValueError(f"naive datetime {v!r}: the clock must be explicit (UTC)")
                sec[i] = int(np.floor(v.timestamp()))
            elif isinstance(v, np.datetime64):
                if np.isnat(v):
                    raise ValueError("NaT in times_utc")
                sec[i] = int(v.astype("datetime64[s]").astype(np.int64))
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                if not np.isfinite(v):
                    raise ValueError("non-finite POSIX seconds in times_utc")
                sec[i] = int(np.floor(v))
            else:
                raise ValueError(f"cannot interpret {v!r} as a UTC instant")
    else:
        raise ValueError(f"cannot interpret dtype {arr.dtype} as UTC instants")
    if sec.size and (sec.min() < _EPOCH_MIN or sec.max() > _EPOCH_MAX):
        raise ValueError("times outside 1990-2100 as POSIX seconds -- milliseconds/nanoseconds passed as seconds?")
    return sec


def _years_of(sec: np.ndarray) -> np.ndarray:
    """UTC calendar year of each POSIX second (vectorised)."""

    return np.asarray(sec, dtype=np.int64).astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TorWarningQuery:
    """Per-point result of :func:`query_tor_warnings` (arrays in input order)."""

    active_now: np.ndarray  # bool
    issued_within_lookahead: np.ndarray  # bool
    minutes_since_issue: np.ndarray  # float, NaN unless active_now
    minutes_to_next_issue: np.ndarray  # float, NaN unless issued_within_lookahead
    active_state: np.ndarray  # int64 row of the earliest-issued covering state, -1 if none
    issuing_state: np.ndarray  # int64 row of the soonest qualifying NEW state, -1 if none
    lookahead_min: float
    advected: bool
    stats: dict = field(default_factory=dict)


class _PointIndex:
    """Points sorted by (time bin, longitude) for range lookups."""

    def __init__(self, lon: np.ndarray, t: np.ndarray, bin_s: int) -> None:
        self.bin_s = int(bin_s)
        self.t0 = int(t.min())
        tb = (t - self.t0) // self.bin_s
        self.nbins = int(tb.max()) + 1
        key = tb.astype(np.float64) * _KEY_STRIDE + (lon + 180.0)
        self.order = np.argsort(key, kind="stable")
        self.key = key[self.order]

    def bin_of(self, sec: np.ndarray) -> np.ndarray:
        return np.floor_divide(sec - self.t0, self.bin_s)

    def candidates(
        self,
        owners: np.ndarray,
        t_lo: np.ndarray,
        t_hi_excl: np.ndarray,
        lon_lo: np.ndarray,
        lon_hi: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (owner, point) candidate pairs, chunked, for windows [t_lo, t_hi_excl)."""

        b_lo = np.maximum(self.bin_of(t_lo), 0)
        b_hi = np.minimum(self.bin_of(t_hi_excl - 1), self.nbins - 1)
        ok = (t_hi_excl > t_lo) & (b_hi >= b_lo)
        owners, b_lo, b_hi = owners[ok], b_lo[ok], b_hi[ok]
        lon_lo, lon_hi = lon_lo[ok], lon_hi[ok]
        if owners.size == 0:
            return
        nb = (b_hi - b_lo + 1).astype(np.int64)
        pair_owner_pos = np.repeat(np.arange(len(owners)), nb)
        pair_bin = _ragged_arange(b_lo.astype(np.int64), nb)
        base = pair_bin.astype(np.float64) * _KEY_STRIDE + 180.0
        lo = np.searchsorted(self.key, base + lon_lo[pair_owner_pos] - _KEY_EPS, side="left")
        hi = np.searchsorted(self.key, base + lon_hi[pair_owner_pos] + _KEY_EPS, side="right")
        cnt = (hi - lo).astype(np.int64)
        keep = cnt > 0
        pair_owner_pos, lo, cnt = pair_owner_pos[keep], lo[keep], cnt[keep]
        if cnt.size == 0:
            return
        csum = np.cumsum(cnt)
        start = 0
        while start < len(cnt):
            prior = csum[start - 1] if start else 0
            stop = int(np.searchsorted(csum, prior + _CAND_CHUNK, side="right"))
            stop = max(stop, start + 1)
            sl = slice(start, stop)
            pos = _ragged_arange(lo[sl].astype(np.int64), cnt[sl])
            yield owners[np.repeat(pair_owner_pos[sl], cnt[sl])], self.order[pos]
            start = stop


def _ragged_arange(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(s, s + c)`` for each (s, c) without a Python loop."""

    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    shift = np.repeat(starts - (np.cumsum(counts) - counts), counts)
    return np.arange(total, dtype=np.int64) + shift


def query_tor_warnings(
    lats,
    lons,
    times_utc,
    *,
    warnings: TorWarnings | None = None,
    lookahead_min: float = 60.0,
    motion_east_ms=None,
    motion_south_ms=None,
    bin_seconds: int = 900,
    require_coverage: bool = True,
    cache_dir: str | os.PathLike | None = None,
) -> TorWarningQuery:
    """Full NWS tornado-warning state for each (lat, lon, UTC instant).

    See the module docstring for the semantics of each output.  ``warnings``
    defaults to loading every year the query needs.  With ``require_coverage``
    (default) a query whose instants need a year that is not loaded raises
    instead of silently reporting "not warned".
    """

    lat = np.asarray(lats, dtype=np.float64).ravel()
    lon = np.asarray(lons, dtype=np.float64).ravel()
    t = to_epoch_seconds(times_utc)
    n = len(lat)
    if len(lon) != n or len(t) != n:
        raise ValueError(f"lats/lons/times lengths differ: {n}, {len(lon)}, {len(t)}")
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("non-finite lat/lon")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("lat/lon outside [-90, 90] x [-180, 180] (lon must be signed degrees east)")
    if not lookahead_min > 0:
        raise ValueError("lookahead_min must be > 0")
    look_s = int(round(lookahead_min * 60.0))
    advect = motion_east_ms is not None or motion_south_ms is not None
    if advect:
        if motion_east_ms is None or motion_south_ms is None:
            raise ValueError("give both motion_east_ms and motion_south_ms")
        ve = np.broadcast_to(np.asarray(motion_east_ms, dtype=np.float64), (n,)).ravel()
        vs = np.broadcast_to(np.asarray(motion_south_ms, dtype=np.float64), (n,)).ravel()
        if not (np.all(np.isfinite(ve)) and np.all(np.isfinite(vs))):
            raise ValueError("non-finite motion; pass 0 for stationary storms")

    empty_f = np.full(n, np.nan)
    if n == 0:
        z = np.zeros(0, dtype=bool)
        return TorWarningQuery(z, z.copy(), empty_f, empty_f.copy(), np.zeros(0, np.int64), np.zeros(0, np.int64), lookahead_min, advect)

    # Warnings that can touch instant t were issued in [t - lifetime, t + lookahead];
    # the files partition by issuance year, so those years must all be loaded.
    need_years = {
        int(y)
        for y in np.unique(
            np.concatenate([_years_of(t - MAX_STATE_LIFETIME_S), _years_of(t), _years_of(t + look_s)])
        )
    }
    if warnings is None:
        warnings = load_tor_warnings(sorted(need_years), cache_dir=cache_dir)
    elif require_coverage:
        missing = sorted(need_years - set(warnings.years))
        if missing:
            raise ValueError(
                f"query instants need warnings issued in {missing}, not loaded "
                f"(loaded {list(warnings.years)}); load them or pass require_coverage=False"
            )

    W = warnings
    idx = _PointIndex(lon, t, bin_seconds)
    active = np.zeros(n, dtype=bool)
    since = np.full(n, np.nan)
    active_state = np.full(n, -1, dtype=np.int64)
    best_issue = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    stats = {"n_points": n, "candidates_active": 0, "pip_tests_active": 0, "hits_active": 0}

    # ---- active_now: state in effect at t and containing the point ----------
    live = np.flatnonzero((W.valid_to > W.valid_from) & (W.valid_to > t.min()) & (W.valid_from <= t.max()))
    for own, pts in idx.candidates(live, W.valid_from[live], W.valid_to[live], W.minlon[live], W.maxlon[live]):
        stats["candidates_active"] += int(len(own))
        tp = t[pts]
        m = (tp >= W.valid_from[own]) & (tp < W.valid_to[own])
        m &= (lat[pts] >= W.minlat[own]) & (lat[pts] <= W.maxlat[own])
        m &= (lon[pts] >= W.minlon[own]) & (lon[pts] <= W.maxlon[own])
        own, pts = own[m], pts[m]
        stats["pip_tests_active"] += int(len(own))
        hit = W.contains(own, lon[pts], lat[pts])
        own, pts = own[hit], pts[hit]
        stats["hits_active"] += int(len(own))
        if own.size:
            active[pts] = True
            iss = W.issue_time[own]
            # earliest issuance wins; ties broken by lowest row for determinism
            o = np.lexsort((own, iss, pts))
            pts_o, iss_o, own_o = pts[o], iss[o], own[o]
            first = np.ones(len(pts_o), dtype=bool)
            first[1:] = pts_o[1:] != pts_o[:-1]
            p1, i1, r1 = pts_o[first], iss_o[first], own_o[first]
            better = (i1 < best_issue[p1]) | ((i1 == best_issue[p1]) & ((active_state[p1] < 0) | (r1 < active_state[p1])))
            best_issue[p1[better]] = i1[better]
            active_state[p1[better]] = r1[better]
    hit_pts = active_state >= 0
    since[hit_pts] = (t[hit_pts] - best_issue[hit_pts]) / 60.0

    # ---- issued within (t, t + lookahead]: NEW polygon contains the storm ---
    issued = np.zeros(n, dtype=bool)
    to_issue = np.full(n, np.nan)
    issuing_state = np.full(n, -1, dtype=np.int64)
    best_next = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    stats.update({"candidates_issue": 0, "pip_tests_issue": 0, "hits_issue": 0})
    new_rows = np.flatnonzero(W.is_issuance)
    tau = W.valid_from[new_rows]  # issuance instant (= product time, repaired)
    sel = (tau > t.min()) & (tau - look_s <= t.max())
    new_rows, tau = new_rows[sel], tau[sel]
    if advect:
        vmax = float(np.max(np.hypot(ve, vs))) if n else 0.0
        d_lat = vmax * look_s / M_PER_DEG
    else:
        d_lat = 0.0
    if new_rows.size:
        absmax = np.maximum(np.abs(W.minlat[new_rows]), np.abs(W.maxlat[new_rows])) + d_lat
        d_lon = d_lat / np.cos(np.radians(np.minimum(absmax, 89.0)))
        lon_lo = W.minlon[new_rows] - d_lon
        lon_hi = W.maxlon[new_rows] + d_lon
        for own, pts in idx.candidates(new_rows, tau - look_s, tau, lon_lo, lon_hi):
            stats["candidates_issue"] += int(len(own))
            tp = t[pts]
            ti = W.valid_from[own]
            m = (tp < ti) & (tp >= ti - look_s)
            m &= (lat[pts] >= W.minlat[own] - d_lat) & (lat[pts] <= W.maxlat[own] + d_lat)
            own, pts, tp, ti = own[m], pts[m], tp[m], ti[m]
            if advect:
                dt = (ti - tp).astype(np.float64)
                py = lat[pts] - vs[pts] * dt / M_PER_DEG
                px = lon[pts] + ve[pts] * dt / (M_PER_DEG * np.cos(np.radians(lat[pts])))
            else:
                py = lat[pts]
                px = lon[pts]
            m2 = (py >= W.minlat[own]) & (py <= W.maxlat[own]) & (px >= W.minlon[own]) & (px <= W.maxlon[own])
            own, pts, ti, px, py = own[m2], pts[m2], ti[m2], px[m2], py[m2]
            stats["pip_tests_issue"] += int(len(own))
            hit = W.contains(own, px, py)
            own, pts, ti = own[hit], pts[hit], ti[hit]
            stats["hits_issue"] += int(len(own))
            if own.size:
                issued[pts] = True
                o = np.lexsort((own, ti, pts))
                pts_o, ti_o, own_o = pts[o], ti[o], own[o]
                first = np.ones(len(pts_o), dtype=bool)
                first[1:] = pts_o[1:] != pts_o[:-1]
                p1, i1, r1 = pts_o[first], ti_o[first], own_o[first]
                better = (i1 < best_next[p1]) | ((i1 == best_next[p1]) & ((issuing_state[p1] < 0) | (r1 < issuing_state[p1])))
                best_next[p1[better]] = i1[better]
                issuing_state[p1[better]] = r1[better]
    nx = issuing_state >= 0
    to_issue[nx] = (best_next[nx] - t[nx]) / 60.0
    return TorWarningQuery(
        active_now=active,
        issued_within_lookahead=issued,
        minutes_since_issue=since,
        minutes_to_next_issue=to_issue,
        active_state=active_state,
        issuing_state=issuing_state,
        lookahead_min=float(lookahead_min),
        advected=advect,
        stats=stats,
    )


def tor_warning_state(
    lats,
    lons,
    times_utc,
    *,
    warnings: TorWarnings | None = None,
    lookahead_min: float = 60.0,
    motion_east_ms=None,
    motion_south_ms=None,
    require_coverage: bool = True,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(active_now, issued_within_next_60min, minutes_since_issue)`` per point.

    Vectorised over N points; see :func:`query_tor_warnings` for the full
    result (minutes to the next issuance, the matching warning rows).
    """

    q = query_tor_warnings(
        lats,
        lons,
        times_utc,
        warnings=warnings,
        lookahead_min=lookahead_min,
        motion_east_ms=motion_east_ms,
        motion_south_ms=motion_south_ms,
        require_coverage=require_coverage,
        cache_dir=cache_dir,
    )
    return q.active_now, q.issued_within_lookahead, q.minutes_since_issue


# --------------------------------------------------------------------------
# Clock witness: the NWS text product states its own time three ways
# --------------------------------------------------------------------------

#: Standard/daylight abbreviations used in NWS product time lines -> UTC offset (h).
NWS_TZ_OFFSETS_H = {
    "UTC": 0, "GMT": 0, "Z": 0,
    "AST": -4, "ADT": -3,
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
    "AKST": -9, "AKDT": -8,
    "HST": -10, "SST": -11, "CHST": 10,
}

_WMO_RE = re.compile(r"^([A-Z]{4}\d{2}) ([A-Z]{4}) (\d{2})(\d{2})(\d{2})(?: [A-Z]{3})?\s*$", re.M)
_VTEC_RE = re.compile(r"/([OTEX])\.([A-Z]{3})\.([A-Z]{4})\.([A-Z]{2})\.([A-Z])\.(\d{4})\.(\d{6}T\d{4}Z)-(\d{6}T\d{4}Z)/")
_LOCAL_RE = re.compile(
    r"^(\d{1,2})(\d{2}) (AM|PM) ([A-Z]{1,4}) (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +(\d{1,2}) (\d{4})\s*$",
    re.M,
)
_MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


@dataclass(frozen=True)
class ProductClocks:
    """Every clock statement found in one NWS text product (POSIX seconds UTC)."""

    wmo_utc: int | None
    vtec: tuple[dict, ...]
    local_utc: int | None
    local_tz: str | None
    local_text: str | None
    latlon: np.ndarray  # (n, 2) of (lat, lon), lon signed east


def _vtec_ts(s: str) -> int | None:
    if s.startswith("000000T0000"):
        return None
    dt = datetime(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[7:9]), int(s[9:11]), tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_product_clocks(text: str) -> ProductClocks:
    """Extract the WMO header time, VTEC begin/end, local issuance line and polygon."""

    vtec = []
    for m in _VTEC_RE.finditer(text):
        vtec.append(
            {
                "class": m.group(1),
                "action": m.group(2),
                "office": m.group(3),
                "phenomena": m.group(4),
                "significance": m.group(5),
                "etn": int(m.group(6)),
                "begin_utc": _vtec_ts(m.group(7)),
                "end_utc": _vtec_ts(m.group(8)),
            }
        )
    local_utc = local_tz = local_text = None
    m = _LOCAL_RE.search(text)
    if m:
        hh, mm, ampm, tz, mon, day, year = m.groups()
        if tz not in NWS_TZ_OFFSETS_H:
            raise ValueError(f"unknown time-zone abbreviation {tz!r} in {m.group(0)!r}")
        h = int(hh) % 12 + (12 if ampm == "PM" else 0)
        local = datetime(int(year), _MONTHS[mon], int(day), h, int(mm))
        local_utc = int((local - timedelta(hours=NWS_TZ_OFFSETS_H[tz])).replace(tzinfo=timezone.utc).timestamp())
        local_tz = tz
        local_text = m.group(0).strip()
    wmo_utc = None
    w = _WMO_RE.search(text)
    if w:
        dd, hh, mi = int(w.group(3)), int(w.group(4)), int(w.group(5))
        ref = local_utc if local_utc is not None else next((v["begin_utc"] for v in vtec if v["begin_utc"]), None)
        if ref is not None:
            r = datetime.fromtimestamp(ref, tz=timezone.utc)
            best = None
            for dm in (-1, 0, 1):
                y, mo = r.year, r.month + dm
                if mo == 0:
                    y, mo = y - 1, 12
                elif mo == 13:
                    y, mo = y + 1, 1
                try:
                    c = datetime(y, mo, dd, hh, mi, tzinfo=timezone.utc)
                except ValueError:
                    continue
                if best is None or abs(c.timestamp() - ref) < abs(best.timestamp() - ref):
                    best = c
            wmo_utc = int(best.timestamp()) if best else None
    latlon = np.zeros((0, 2))
    ll = re.search(r"LAT\.\.\.LON((?:\s+\d{4,5})+)", text)
    if ll:
        nums = [int(x) for x in ll.group(1).split()]
        if len(nums) % 2:
            raise ValueError("odd number of LAT...LON values")
        pairs = np.array(nums, dtype=np.float64).reshape(-1, 2) / 100.0
        pairs[:, 1] = -pairs[:, 1]  # NWS LAT...LON longitudes are degrees west
        latlon = pairs
    return ProductClocks(wmo_utc=wmo_utc, vtec=tuple(vtec), local_utc=local_utc, local_tz=local_tz, local_text=local_text, latlon=latlon)


def clock_witness(text: str, iem_issue_utc: int, iem_expire_utc: int | None = None) -> dict:
    """Offsets (minutes) between IEM's instants and the product's own clock statements.

    ``iem_issue_utc`` is IEM's ISSUED / POLY_BEG for the product's NEW polygon,
    as parsed by this module.  Every offset is 0 when IEM's clock is UTC and is
    -300 / -360 if IEM were silently on CDT / CST.
    """

    c = parse_product_clocks(text)
    new = [v for v in c.vtec if v["action"] == "NEW" and v["phenomena"] == "TO" and v["significance"] == "W"]
    out: dict = {
        "wmo_utc": _utc_iso(c.wmo_utc) if c.wmo_utc is not None else None,
        "vtec_begin_utc": _utc_iso(new[0]["begin_utc"]) if new and new[0]["begin_utc"] else None,
        "vtec_end_utc": _utc_iso(new[0]["end_utc"]) if new and new[0]["end_utc"] else None,
        "local_line": c.local_text,
        "local_as_utc": _utc_iso(c.local_utc) if c.local_utc is not None else None,
        "iem_issue_utc": _utc_iso(iem_issue_utc),
    }
    off = {}
    if c.wmo_utc is not None:
        off["iem_minus_wmo_min"] = (iem_issue_utc - c.wmo_utc) / 60.0
    if new and new[0]["begin_utc"]:
        off["iem_minus_vtec_begin_min"] = (iem_issue_utc - new[0]["begin_utc"]) / 60.0
    if c.local_utc is not None:
        off["iem_minus_local_line_min"] = (iem_issue_utc - c.local_utc) / 60.0
    if iem_expire_utc is not None and new and new[0]["end_utc"]:
        off["iem_expire_minus_vtec_end_min"] = (iem_expire_utc - new[0]["end_utc"]) / 60.0
    out["offsets"] = off
    out["agree"] = bool(off) and all(v == 0 for v in off.values())
    return out


# --------------------------------------------------------------------------
# Summary + CLI
# --------------------------------------------------------------------------


def summarize(W: TorWarnings) -> dict:
    """Per-year counts: polygon states, issuances, cancellations, durations."""

    out = {}
    for y in W.years:
        m = W.meta[str(y)]
        sel = W.source_year == y
        new = sel & W.is_issuance
        has_exp = new & (W.init_exp != _NO_TIME)
        dur = (W.init_exp[has_exp] - W.valid_from[has_exp]) / 60.0
        keys_new = {(a, b, c, d) for a, b, c, d in zip(W.wfo[new], W.etn[new], W.vtec_year[new], W.product_time[new])}
        out[str(y)] = {
            "polygon_states": int(m["n_states"]),
            "tor_warnings_new_rows": int(m["n_issuances"]),
            "tor_warnings_unique_issuances": int(len(keys_new)),
            "status_counts": m["status_counts"],
            "median_issued_duration_min": float(np.median(dur)) if dur.size else float("nan"),
            "observed_tag_at_issuance": int(np.sum(W.tornado_tag[new] == "OBSERVED")),
            "pds_considerable_at_issuance": int(np.sum(W.damage_tag[new] == "CONSIDERABLE")),
            "emergency_catastrophic_at_issuance": int(np.sum(W.damage_tag[new] == "CATASTROPHIC")),
            "repair_new_state_start": m["repair_new_state_start"],
            "empty_states": m["empty_states"],
            "chain_overlaps": m["chain_overlaps"],
            "keys_without_new": m["keys_without_new"],
            "keys_with_multiple_new": m["keys_with_multiple_new"],
            "area_ratio_median": m["area_ratio_median"],
            "area_ratio_off_by_gt_2pct": m["area_ratio_off_by_gt_2pct"],
            "multipart_states": m["multipart_states"],
            "year_complete_at_fetch": m["year_complete_at_fetch"],
            "raw_sha256": m["raw_sha256"][:16],
        }
    return out


def _parse_years(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif part:
            years.add(int(part))
    return sorted(years)


def _bench(W: TorWarnings, n_points: int, seed: int) -> dict:
    """Synthetic throughput: storm-like points on 30-min slots over the loaded years."""

    rng = np.random.default_rng(seed)
    t_lo = int(W.valid_from.min())
    t_hi = int(W.valid_to.max())
    # half the points: uniform over CONUS and time; half: drawn near real
    # polygon states at their in-effect instants (the expensive, hit-rich case)
    n_near = n_points // 2
    n_unif = n_points - n_near
    slots = np.arange(t_lo - t_lo % 1800, t_hi, 1800)
    t_u = rng.choice(slots, n_unif) + 42
    lat_u = rng.uniform(25.0, 49.0, n_unif)
    lon_u = rng.uniform(-125.0, -67.0, n_unif)
    live = np.flatnonzero(W.valid_to > W.valid_from)
    r = rng.choice(live, n_near)
    t_n = W.valid_from[r] + (rng.random(n_near) * (W.valid_to[r] - W.valid_from[r])).astype(np.int64)
    lat_n = rng.uniform(W.minlat[r] - 0.3, W.maxlat[r] + 0.3)
    lon_n = rng.uniform(W.minlon[r] - 0.3, W.maxlon[r] + 0.3)
    lat = np.concatenate([lat_u, lat_n])
    lon = np.concatenate([lon_u, lon_n])
    t = np.concatenate([t_u, t_n]).astype(np.int64)
    ve = rng.normal(10.0, 6.0, n_points)
    vs = rng.normal(-4.0, 6.0, n_points)
    t0 = time.perf_counter()
    q = query_tor_warnings(lat, lon, t, warnings=W, require_coverage=False)
    t1 = time.perf_counter()
    qa = query_tor_warnings(lat, lon, t, warnings=W, motion_east_ms=ve, motion_south_ms=vs, require_coverage=False)
    t2 = time.perf_counter()
    return {
        "n_points": n_points,
        "static_seconds": round(t1 - t0, 3),
        "static_points_per_s": round(n_points / (t1 - t0)),
        "advected_seconds": round(t2 - t1, 3),
        "advected_points_per_s": round(n_points / (t2 - t1)),
        "active_fraction": float(q.active_now.mean()),
        "issued_fraction_static": float(q.issued_within_lookahead.mean()),
        "issued_fraction_advected": float(qa.issued_within_lookahead.mean()),
        "stats": q.stats,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hazardpulse.verification.nws_warnings", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fetch + compact years (e.g. 2020-2025)")
    b.add_argument("years")
    b.add_argument("--refresh", action="store_true", help="re-download even if cached")
    s = sub.add_parser("stats", help="per-year counts and integrity figures")
    s.add_argument("years")
    k = sub.add_parser("bench", help="query throughput on synthetic storm points")
    k.add_argument("years")
    k.add_argument("--points", type=int, default=2_000_000)
    k.add_argument("--seed", type=int, default=0)
    for p in (b, s, k):
        p.add_argument("--cache-dir", default=None)
    a = ap.parse_args(argv)
    years = _parse_years(a.years)
    if a.cmd == "build":
        for i, y in enumerate(years):
            raw_existed = _raw_path(y, a.cache_dir).exists()
            path = build_year(y, cache_dir=a.cache_dir, refresh=a.refresh)
            print(f"{y}: {path}")
            if (a.refresh or not raw_existed) and i + 1 < len(years):
                time.sleep(3.0)  # polite spacing between service requests
        return 0
    W = load_tor_warnings(years, cache_dir=a.cache_dir)
    if a.cmd == "stats":
        print(json.dumps(summarize(W), indent=1, sort_keys=True))
    else:
        print(json.dumps(_bench(W, a.points, a.seed), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
