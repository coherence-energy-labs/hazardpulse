"""HRRR Zarr data access with on-disk .npz caching.

Downloads HRRR analysis fields from the AWS Open Data Zarr store,
subsamples to an 80 km CONUS grid (34 lat x 63 lon), and caches locally.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

# Transient-failure retry for the public HRRR Zarr archive (it drops connections
# mid-read under load). A poisoned variable becomes all-NaN and silently kills the
# tornado signal, so a brief retry is worth it.
_HRRR_FETCH_ATTEMPTS = int(os.environ.get("HAZARDPULSE_HRRR_ATTEMPTS", "3"))
_HRRR_FETCH_BACKOFF = 1.5  # seconds, multiplied by attempt index
_HRRR_HTTP_TIMEOUT = float(os.environ.get("HAZARDPULSE_HRRR_TIMEOUT", "30"))  # per-request seconds
_HRRR_FETCH_THREADS = int(os.environ.get("HAZARDPULSE_HRRR_THREADS", "6"))  # variables read concurrently

from hazardpulse.data.http import fetch_bytes

# ---------------------------------------------------------------------------
# Project paths — check env var HAZARDPULSE_HRRR_CACHE first
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_HRRR_CACHE",
    str(PROJECT_ROOT / ".cache" / "hrrr"),
))

# ---------------------------------------------------------------------------
# 80 km CONUS grid constants
# ---------------------------------------------------------------------------

GRID_DLAT: float = 0.72  # ~80 km
GRID_DLON: float = 0.94  # ~80 km at 37.5 N
LAT_MIN: float = 25.0
LAT_MAX: float = 50.0
LON_MIN: float = -125.0
LON_MAX: float = -65.0

HRRR_N_LAT: int = 34
HRRR_N_LON: int = 63

# Pre-computed grid cell centres
GRID_LATS: np.ndarray = np.array(
    [LAT_MIN + (i + 0.5) * GRID_DLAT for i in range(HRRR_N_LAT)],
    dtype=np.float32,
)
GRID_LONS: np.ndarray = np.array(
    [LON_MIN + (j + 0.5) * GRID_DLON for j in range(HRRR_N_LON)],
    dtype=np.float32,
)

# Physical grid spacing
DX_KM: float = GRID_DLON * 111.0 * math.cos(math.radians(37.5))
DY_KM: float = GRID_DLAT * 111.0
DX_M: float = DX_KM * 1000.0
DY_M: float = DY_KM * 1000.0

# ---------------------------------------------------------------------------
# HRRR variable mapping  (short name -> Zarr group/variable path)
# ---------------------------------------------------------------------------

# Variable name -> relative path inside the hrrrzarr store.
# The Utah hrrrzarr bucket nests paths as "{level}/{var}/{level}/{var}".
# Mixed-layer CAPE is approximated by the 180_0mb layer; MU CAPE by 255_0mb.
#
# IMPORTANT: hrrrzarr does NOT publish named MLCAPE/MUCAPE/SBCAPE arrays the
# way the deprecated noaa-hrrr-bdp-pds wrfprsf bucket did. It exposes raw
# layer-CAPE values at fixed pressure-thickness layers (90/180/255 mb above
# ground). Standard NWS conventions:
#   MLCAPE  ≈ 90 mb mixed-layer CAPE  ->  90_0mb_above_ground/CAPE
#   MUCAPE  ≈ 255 mb most-unstable    ->  255_0mb_above_ground/CAPE
#   MLCIN   ≈ 90 mb mixed-layer CIN   ->  90_0mb_above_ground/CIN
# These are the canonical equivalents; the trained tornado GBT was trained
# against surface/MLCAPE which was NCEP's name for the same 0-90 mb
# mixed-layer integration, so they should be numerically close.
HRRR_VARS: dict[str, str] = {
    "cape": "surface/CAPE/surface/CAPE",
    "cin": "surface/CIN/surface/CIN",
    "mlcape": "90_0mb_above_ground/CAPE/90_0mb_above_ground/CAPE",
    "mlcin": "90_0mb_above_ground/CIN/90_0mb_above_ground/CIN",
    "mucape": "255_0mb_above_ground/CAPE/255_0mb_above_ground/CAPE",
    "srh_01": "1000_0m_above_ground/HLCY/1000_0m_above_ground/HLCY",
    "srh_03": "3000_0m_above_ground/HLCY/3000_0m_above_ground/HLCY",
    "refc": "entire_atmosphere/REFC/entire_atmosphere/REFC",
    "ushear_01": "0_1000m_above_ground/VUCSH/0_1000m_above_ground/VUCSH",
    "vshear_01": "0_1000m_above_ground/VVCSH/0_1000m_above_ground/VVCSH",
    "ushear_06": "0_6000m_above_ground/VUCSH/0_6000m_above_ground/VUCSH",
    "vshear_06": "0_6000m_above_ground/VVCSH/0_6000m_above_ground/VVCSH",
    "ustorm": "0_6000m_above_ground/USTM/0_6000m_above_ground/USTM",
    "vstorm": "0_6000m_above_ground/VSTM/0_6000m_above_ground/VSTM",
    "t2m": "2m_above_ground/TMP/2m_above_ground/TMP",
    "td2m": "2m_above_ground/DPT/2m_above_ground/DPT",
    "pwat": "entire_atmosphere_single_layer/PWAT/entire_atmosphere_single_layer/PWAT",
}

# Utah hrrrzarr bucket (active, public HRRR analysis Zarr mirror)
HRRR_ZARR_ROOT = (
    "https://hrrrzarr.s3.amazonaws.com/sfc/{date}/{date}_{hour:02d}z_anl.zarr"
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _npz_path(date_str: str, hour: int, *, cache_dir: Path | None = None) -> Path:
    """Return the local .npz cache path for a given date/hour.

    The name carries the grid GEOMETRY version and the pooling MODE, so a grid
    built one way can never be read as if it were built another. Files from
    before 2026-10-01 (``{date}_{hh}z.npz``) were pooled in native index space
    and then read as a lat/lon grid -- every cell mislocated by a median 283 km
    -- and are deliberately never loaded again.
    """
    root = cache_dir or CACHE_ROOT
    return root / f"{date_str}_{hour:02d}z.{GEOMETRY_VERSION}-{HRRR_POOL_MODE}.npz"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cached_hrrr(
    date_str: str,
    hour: int = 18,
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load HRRR data from local .npz cache only (no network).

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    hour : int
        Analysis hour (default 18 Z).
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    dict or None
        Mapping of variable name to ``(34, 63)`` float32 array, or *None*
        if the file does not exist in cache.
    """
    path = _npz_path(date_str, hour, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return {key: data[key] for key in data.files}
    except Exception:
        return None


def fetch_hrrr_natives(
    date_str: str,
    hour: int,
    *,
    threads: int = _HRRR_FETCH_THREADS,
) -> dict[str, np.ndarray] | None:
    """Every HRRR_VARS field of one analysis on the NATIVE 1059 x 1799 grid.

    Float32, fill values verified absent (a chunk that fails to arrive comes
    back from zarr as its fill value with no error -- retried, then refused),
    winds rotated to earth-relative. Variables are read concurrently. Returns
    None if any variable cannot be fetched cleanly; nothing partial escapes.
    """
    import zarr
    import fsspec
    from concurrent.futures import ThreadPoolExecutor

    # Hard network timeout: the public archive can stall a read indefinitely (no
    # bytes, no error). A bounded client timeout turns a stall into an exception
    # the per-variable retry can handle.
    client_kwargs = {}
    try:
        import aiohttp
        client_kwargs = {"timeout": aiohttp.ClientTimeout(total=_HRRR_HTTP_TIMEOUT)}
    except Exception:
        pass

    zarr_root = HRRR_ZARR_ROOT.format(date=date_str, hour=hour)
    try:
        store = fsspec.get_mapper(zarr_root, client_kwargs=client_kwargs)
        root = zarr.open(store, mode="r")
    except Exception as exc:
        print(f"  HRRR zarr store unreachable ({zarr_root}): {exc}")
        return None

    def _one(item: tuple[str, str]) -> tuple[str, np.ndarray | None]:
        var_name, zarr_path = item
        for attempt in range(_HRRR_FETCH_ATTEMPTS):
            try:
                arr = root[zarr_path]
                got = _fill_to_nan(np.asarray(arr, dtype=np.float32), getattr(arr, "fill_value", None))
                n_bad = int(np.isnan(got).sum())
                if n_bad:
                    raise ValueError(
                        f"{n_bad} of {got.size} native points are fill/NaN "
                        "(a chunk did not arrive)"
                    )
                return var_name, got.reshape(NATIVE_NY, NATIVE_NX)
            except Exception as exc:
                if attempt == _HRRR_FETCH_ATTEMPTS - 1:
                    print(f"  HRRR fetch failed for {var_name} after "
                          f"{_HRRR_FETCH_ATTEMPTS} tries: {exc}")
                else:
                    time.sleep(_HRRR_FETCH_BACKOFF * (attempt + 1))
        return var_name, None

    with ThreadPoolExecutor(max(1, threads)) as ex:
        results = dict(ex.map(_one, HRRR_VARS.items()))
    missing = [k for k, v in results.items() if v is None]
    if missing:
        # A grid with ANY missing variable is never cached or returned: the old
        # code cached partial pulls (an all-NaN variable among real ones, or
        # -10000 fill values pooled as data -- 236 of 279 cached grids carried
        # fill values by 2026-10-01).
        print(f"  HRRR pull for {date_str} {hour}z abandoned: {', '.join(missing)} unavailable.")
        return None
    natives: dict[str, np.ndarray] = results  # type: ignore[assignment]

    # Winds are published GRID-relative; rotate every vector to EARTH-relative
    # (up to 17-20 degrees at the CONUS edges, ~0 at 97.5 W).
    for u_name, v_name in WIND_PAIRS:
        natives[u_name], natives[v_name] = rotate_grid_winds_to_earth(
            natives[u_name], natives[v_name]
        )
    return natives


# ---------------------------------------------------------------------------
# Storm-scale grid: native Lambert blocks (k x k native 3 km points)
# ---------------------------------------------------------------------------
#
# The 80 km lat/lon grid is too coarse for a storm: a mesocyclone is 2-10 km.
# The storm-scale product keeps the NATIVE projection and pools k x k native
# points (k = 3 -> 9 km), so there is no regridding at all and a storm's block
# is found exactly through native_index_of_latlon. MAX for _HRRR_MAX_FIELDS,
# MEAN otherwise, as for the 80 km grid. Stored float16 (relative precision
# ~1e-3, below HRRR's own analysis error for every field kept).
LCC_BLOCK_VARS: tuple[str, ...] = (
    "mlcape", "mucape", "mlcin", "srh_01", "srh_03", "refc",
    "ushear_01", "vshear_01", "ushear_06", "vshear_06", "ustorm", "vstorm",
    "t2m", "td2m", "pwat",
)
CACHE_ROOT_LCC = Path(os.environ.get(
    "HAZARDPULSE_HRRR_LCC_CACHE",
    str(PROJECT_ROOT / ".cache" / "hrrr_lcc"),
))


def pool_lcc_blocks(full2d: np.ndarray, var_name: str, k: int = 3) -> np.ndarray:
    """Pool a native field into k x k native blocks (trailing rows/cols dropped)."""
    ny, nx = (NATIVE_NY // k) * k, (NATIVE_NX // k) * k
    b = np.asarray(full2d[:ny, :nx], dtype=np.float32).reshape(ny // k, k, nx // k, k)
    return b.max(axis=(1, 3)) if var_name in _HRRR_MAX_FIELDS else b.mean(axis=(1, 3))


def lcc_block_of_latlon(lat, lon, k: int = 3):
    """(row, col) of the k x k native block containing a lat/lon point (vectorised)."""
    r, c = native_index_of_latlon(lat, lon)
    return np.floor(np.asarray(r) + 0.5).astype(np.int64) // k, np.floor(np.asarray(c) + 0.5).astype(np.int64) // k


def _lcc_path(date_str: str, hour: int, k: int, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_ROOT_LCC) / f"{date_str}_{hour:02d}z.lcc{k}-{GEOMETRY_VERSION}.npz"


def save_lcc_blocks(natives: dict[str, np.ndarray], date_str: str, hour: int, k: int = 3,
                    cache_dir: Path | None = None) -> Path:
    out = _lcc_path(date_str, hour, k, cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = {v: pool_lcc_blocks(natives[v], v, k).astype(np.float16) for v in LCC_BLOCK_VARS}
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), **blocks)
    os.replace(tmp, out)
    return out


def load_lcc_blocks(date_str: str, hour: int, k: int = 3,
                    cache_dir: Path | None = None) -> dict[str, np.ndarray] | None:
    path = _lcc_path(date_str, hour, k, cache_dir)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return {key: z[key].astype(np.float32) for key in z.files}
    except Exception:
        return None


def fetch_hrrr_grid(
    date_str: str,
    hour: int = 18,
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Fetch HRRR analysis fields subsampled to the 80 km CONUS grid.

    Checks the local ``.npz`` cache first; downloads from the AWS Zarr
    store when the cache misses.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    hour : int
        Analysis hour (default 18 Z).
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of variable name to ``(34, 63)`` float32 array.

    Raises
    ------
    RuntimeError
        If the HRRR data cannot be fetched from AWS.
    """
    cached = load_cached_hrrr(date_str, hour, cache_dir=cache_dir)
    if cached is not None:
        # Treat all-NaN cache as missing so we re-fetch
        n_nan = sum(
            float(np.isnan(a).mean()) for a in cached.values() if isinstance(a, np.ndarray)
        )
        if n_nan / max(len(cached), 1) < 0.9:
            return cached

    natives = fetch_hrrr_natives(date_str, hour)
    if natives is None:
        return None  # type: ignore[return-value]
    grids = {k: _subsample_to_grid(v.ravel(), k) for k, v in natives.items()}

    out_path = _npz_path(date_str, hour, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **grids)
    return grids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Fields where the 80 km cell's PEAK is the meaningful tornado signal (instability,
# storm-relative helicity, reflectivity) -> cell-MAX. Everything else (winds,
# temperature, moisture, inhibition) -> cell-MEAN. Striding (one native point per
# 80 km cell) discards 99.9% of the field and misses these peaks; pooling keeps them.
_HRRR_MAX_FIELDS = frozenset({
    "cape", "mlcape", "mucape", "srh_01", "srh_03", "refc",
})

# Extraction mode: "max" (alias "pool") pools every native point that falls in
# the 80 km lat/lon cell -- MAX for _HRRR_MAX_FIELDS, MEAN otherwise; "stride"
# reads the single native point nearest the cell centre. The default moved from
# "stride" to "max" with the geometry fix (2026-10-01): every served model is
# retrained on the corrected grid anyway, so there is no trained distribution
# left to protect, and pooling is the physically meaningful reduction.
HRRR_POOL_MODE: str = os.environ.get("HAZARDPULSE_HRRR_POOL", "max").strip().lower()
if HRRR_POOL_MODE == "pool":
    HRRR_POOL_MODE = "max"
if HRRR_POOL_MODE not in ("max", "stride"):
    raise ValueError(f"HAZARDPULSE_HRRR_POOL must be 'max' or 'stride', got {HRRR_POOL_MODE!r}")

NATIVE_NY, NATIVE_NX = 1059, 1799

# ---------------------------------------------------------------------------
# Native grid geometry -- NCEP HRRR CONUS, Lambert conformal conic
# ---------------------------------------------------------------------------
# LoV -97.5, Latin1 = Latin2 = 38.5 N (tangent cone), 3 km spacing, first grid
# point (21.138123 N, -122.719528 E), spherical earth R = 6371229 m, row 0 at
# the south edge. Verified 2026-10-01 against hrrrzarr's own published grid
# index (grid/HRRR_chunk_index.zarr): max |difference| 6e-14 deg in lat,
# 1.3e-13 deg in lon over all 1,905,141 points.
#
# GEOMETRY_VERSION "g1" (everything before 2026-10-01) pooled the native grid
# in INDEX space -- 31 rows x 28 columns per cell -- and then read the result as
# the regular 0.72 x 0.94 deg grid below. Rows of a Lambert grid are not
# parallels, so each cell's data came from a median 283 km (p90 481, max 833
# km) from where latlon_to_hrrr_cell placed it: Oklahoma City read the
# atmosphere at (36.31 N, -99.95 E), 238 km away. Every HRRR- and
# coherence-derived feature, in training and live, was mislocated. "g2" (a few
# hours on 2026-10-01, never trained on) bins each native point into the
# lat/lon cell that actually contains it; "g3" also rotates the grid-relative
# winds to earth-relative first, so wind vectors and lat/lon gradients share
# one frame.
GEOMETRY_VERSION: str = "g3"

# (u, v) pairs published grid-relative, rotated to earth-relative on fetch.
WIND_PAIRS: tuple[tuple[str, str], ...] = (
    ("ushear_01", "vshear_01"),
    ("ushear_06", "vshear_06"),
    ("ustorm", "vstorm"),
)

_LCC_R = 6371229.0
_LCC_LAT0 = math.radians(38.5)
_LCC_LON0 = math.radians(-97.5)
_LCC_FIRST = (21.138123, -122.719528)
_LCC_DX = 3000.0
_LCC_N = math.sin(_LCC_LAT0)
_LCC_F = math.cos(_LCC_LAT0) * math.tan(math.pi / 4 + _LCC_LAT0 / 2) ** _LCC_N / _LCC_N
_LCC_RHO0 = _LCC_R * _LCC_F / math.tan(math.pi / 4 + _LCC_LAT0 / 2) ** _LCC_N


def _lcc_forward(lat, lon):
    """(lat, lon) degrees -> projected (x, y) metres."""
    rho = _LCC_R * _LCC_F / np.tan(np.pi / 4 + np.radians(lat) / 2) ** _LCC_N
    th = _LCC_N * (np.radians(lon) - _LCC_LON0)
    return rho * np.sin(th), _LCC_RHO0 - rho * np.cos(th)


_LCC_X0, _LCC_Y0 = _lcc_forward(_LCC_FIRST[0], _LCC_FIRST[1])


def native_index_of_latlon(lat, lon):
    """Fractional native (row, col) of a lat/lon point (row 0 = south edge)."""
    x, y = _lcc_forward(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
    return (y - _LCC_Y0) / _LCC_DX, (x - _LCC_X0) / _LCC_DX


def native_latlon() -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude (degrees) of every native HRRR point, shape (1059, 1799)."""
    global _NATIVE_LATLON
    if _NATIVE_LATLON is None:
        jj, ii = np.meshgrid(np.arange(NATIVE_NX, dtype=np.float64),
                             np.arange(NATIVE_NY, dtype=np.float64))
        x = _LCC_X0 + jj * _LCC_DX
        y = _LCC_Y0 + ii * _LCC_DX
        rho = np.sign(_LCC_N) * np.hypot(x, _LCC_RHO0 - y)
        th = np.arctan2(x, _LCC_RHO0 - y)
        lat = np.degrees(2 * np.arctan((_LCC_R * _LCC_F / rho) ** (1 / _LCC_N)) - np.pi / 2)
        lon = np.degrees(_LCC_LON0 + th / _LCC_N)
        _NATIVE_LATLON = (lat, lon)
    return _NATIVE_LATLON


def rotate_grid_winds_to_earth(u_grid: np.ndarray, v_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate native grid-relative (u, v) to earth-relative (east, north).

    With theta = n * (lon - LoV), the grid's +x axis points at -theta from
    true east (verified against the projection itself to 0.001 deg), so

        u_e =  cos(theta) u_g + sin(theta) v_g
        v_e = -sin(theta) u_g + cos(theta) v_g
    """
    _lat, lon = native_latlon()
    theta = _LCC_N * np.radians(lon - math.degrees(_LCC_LON0))
    c, s = np.cos(theta), np.sin(theta)
    ug = np.asarray(u_grid, dtype=np.float64).reshape(NATIVE_NY, NATIVE_NX)
    vg = np.asarray(v_grid, dtype=np.float64).reshape(NATIVE_NY, NATIVE_NX)
    return (c * ug + s * vg).astype(np.float32), (-s * ug + c * vg).astype(np.float32)


_NATIVE_LATLON: tuple[np.ndarray, np.ndarray] | None = None
_CELL_OF_NATIVE: np.ndarray | None = None
_FILL_FROM: np.ndarray | None = None
_STRIDE_POINTS: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _cell_of_native() -> np.ndarray:
    """Flat 80 km cell index (i * HRRR_N_LON + j) of each native point; -1 outside."""
    global _CELL_OF_NATIVE
    if _CELL_OF_NATIVE is None:
        lat, lon = native_latlon()
        i = np.floor((lat - LAT_MIN) / GRID_DLAT).astype(np.int64)
        j = np.floor((lon - LON_MIN) / GRID_DLON).astype(np.int64)
        inside = (i >= 0) & (i < HRRR_N_LAT) & (j >= 0) & (j < HRRR_N_LON)
        _CELL_OF_NATIVE = np.where(inside, i * HRRR_N_LON + j, -1).ravel()
    return _CELL_OF_NATIVE


def domain_mask() -> np.ndarray:
    """True for 80 km cells that contain at least one native HRRR point."""
    counts = np.bincount(_cell_of_native()[_cell_of_native() >= 0],
                         minlength=HRRR_N_LAT * HRRR_N_LON)
    return (counts > 0).reshape(HRRR_N_LAT, HRRR_N_LON)


def _fill_from() -> np.ndarray:
    """For each cell, the flat index of the nearest in-domain cell (itself if inside).

    A handful of the regular grid's cells (open Atlantic south-east of the
    HRRR domain, the far north-east corner) contain no native point. They are
    filled from the nearest covered cell so the coherence PDE sees a finite
    field; no ProbSevere storm can sit in them (no radar coverage), so no
    storm feature is ever read from a filled cell.
    """
    global _FILL_FROM
    if _FILL_FROM is None:
        mask = domain_mask().ravel()
        ii, jj = np.divmod(np.arange(mask.size), HRRR_N_LON)
        yk, xk = ii * DY_KM, jj * DX_KM
        inside = np.flatnonzero(mask)
        src = np.arange(mask.size)
        for c in np.flatnonzero(~mask):
            d2 = (yk[inside] - yk[c]) ** 2 + (xk[inside] - xk[c]) ** 2
            src[c] = inside[int(np.argmin(d2))]
        _FILL_FROM = src
    return _FILL_FROM


def _stride_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native (row, col) nearest each cell centre, and which centres lie inside the grid."""
    global _STRIDE_POINTS
    if _STRIDE_POINTS is None:
        la, lo = np.meshgrid(GRID_LATS.astype(np.float64), GRID_LONS.astype(np.float64), indexing="ij")
        r, c = native_index_of_latlon(la, lo)
        r, c = np.rint(r).astype(np.int64), np.rint(c).astype(np.int64)
        ok = (r >= 0) & (r < NATIVE_NY) & (c >= 0) & (c < NATIVE_NX)
        _STRIDE_POINTS = (np.clip(r, 0, NATIVE_NY - 1), np.clip(c, 0, NATIVE_NX - 1), ok)
    return _STRIDE_POINTS


def _fill_to_nan(full: np.ndarray, fill_value) -> np.ndarray:
    """Replace the zarr fill value (and the -10000 HRRR sentinel) with NaN."""
    out = np.array(full, dtype=np.float32, copy=True)
    if fill_value is not None:
        try:
            fv = float(fill_value)
        except (TypeError, ValueError):
            fv = None
        if fv is not None and np.isfinite(fv):
            out[out == np.float32(fv)] = np.nan
    out[out <= -9999.0] = np.nan
    return out


def _regrid_to_latlon(full2d: np.ndarray, var_name: str, mode: str) -> np.ndarray:
    """Reduce the native grid to the 80 km lat/lon grid by TRUE location."""
    if mode == "stride":
        # Nearest native point to each cell centre (centres outside the native
        # grid clamp to its edge, the nearest point it has).
        r, c, _ok = _stride_points()
        return full2d[r, c].astype(np.float32)

    ncell = HRRR_N_LAT * HRRR_N_LON
    cells = _cell_of_native()
    vals = full2d.ravel().astype(np.float64)
    keep = (cells >= 0) & np.isfinite(vals)
    cells, vals = cells[keep], vals[keep]
    counts = np.bincount(cells, minlength=ncell)
    if var_name in _HRRR_MAX_FIELDS:
        out = np.full(ncell, np.nan)
        hit = np.full(ncell, -np.inf)
        np.maximum.at(hit, cells, vals)
        out[counts > 0] = hit[counts > 0]
    else:
        out = np.full(ncell, np.nan)
        sums = np.bincount(cells, weights=vals, minlength=ncell)
        out[counts > 0] = sums[counts > 0] / counts[counts > 0]
    # Out-of-domain cells take the nearest covered cell's value (see _fill_from).
    # Only GEOMETRY-empty cells are filled: a covered cell whose points are all
    # NaN stays NaN, so a data failure is never painted over.
    geom_empty = ~domain_mask().ravel()
    out[geom_empty] = out[_fill_from()[geom_empty]]
    return out.reshape(HRRR_N_LAT, HRRR_N_LON).astype(np.float32)


def _subsample_to_grid(flat: np.ndarray, var_name: str, *, mode: str | None = None) -> np.ndarray:
    """Reduce a flat native HRRR array to the 80 km lat/lon grid (geometry ``g2``)."""
    mode = (mode or HRRR_POOL_MODE)
    if mode == "pool":
        mode = "max"
    if flat.size != NATIVE_NY * NATIVE_NX:
        raise ValueError(
            f"expected the native {NATIVE_NY}x{NATIVE_NX} HRRR grid, got {flat.size} values"
        )
    return _regrid_to_latlon(np.asarray(flat, dtype=np.float32).reshape(NATIVE_NY, NATIVE_NX),
                             var_name, mode)


def latlon_to_hrrr_cell(lat: float, lon: float) -> tuple[int, int]:
    """Map a lat/lon coordinate to the 80 km cell that contains it.

    With geometry ``g2`` this cell's value IS pooled from the native points
    inside it, so the lookup and the data agree on location.

    Returns
    -------
    tuple[int, int]
        ``(i_lat, j_lon)`` indices clamped to valid grid bounds.
    """
    i = int(math.floor((lat - LAT_MIN) / GRID_DLAT))
    j = int(math.floor((lon - LON_MIN) / GRID_DLON))
    i = max(0, min(i, HRRR_N_LAT - 1))
    j = max(0, min(j, HRRR_N_LON - 1))
    return i, j
