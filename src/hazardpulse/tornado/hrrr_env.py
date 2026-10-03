"""Shared HRRR-environment tornado feature extraction (train == serve).

ONE definition of the 26-feature vector (17 raw HRRR analysis variables + 9 derived
tornado discriminators) used by BOTH the offline dataset builder and the live scorer,
so the served features are byte-identical to the trained ones -- no train/serve skew.
The derived block (bulk shear, LCL, 0-500 m SRH, Significant Tornado Parameter, RFD
warmth, streamwise vorticity) is what actually separates tornadic from non-tornadic
storm environments.

This module also owns the three definitions the train/serve contract needs beyond the
feature vector itself:

1. **Locating a report.** The grid is ``hazardpulse.data.hrrr``'s geometry ``g3``: every
   native Lambert point binned into the regular 0.72 x 0.94 degree lat/lon cell that
   contains it (winds rotated to earth-relative first), so ``hrrr.latlon_to_hrrr_cell``
   is THE locator (:func:`locate_cell` wraps it to refuse off-grid points instead of
   clamping them). Before 2026-10-01 the cache was pooled in native INDEX space and
   read as lat/lon -- every cell 185-375 km from where the locator put it over tornado
   country; those files are never read by the benchmark (only by ``--negatives
   legacy-easy``, to reproduce the historical number they produced).
   :func:`carry_latlon` moves a report along the earth-relative HRRR storm motion.

2. **Fill values.** Pre-g2 caches carry the archive's -10000 fill wherever a native zarr
   chunk did not arrive (and physically impossible partial means where a mean-pooled
   cell straddled one); the current fetcher refuses such grids. :func:`sanitize_grids`
   turns every value outside :data:`PHYSICAL_BOUNDS` into NaN regardless of where a
   grid came from. Feature spec v2 = sanitized grids.

3. **The storm population.** A tornado-vs-null benchmark whose negatives include cells
   with no storm measures storm-vs-no-storm. :func:`storm_population_mask` is the ONE
   eligibility gate -- a convective core in a supercell-capable environment -- applied
   identically to positives and negatives at train time and to every cell at serve
   time, so the model's population is the population it is served on.
"""

from __future__ import annotations

import math

import numpy as np

from hazardpulse.data import hrrr as _hrrr
from hazardpulse.data.hrrr import HRRR_VARS, HRRR_N_LAT, HRRR_N_LON
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr

RAW_NAMES = list(HRRR_VARS)
DERIVED_NAMES = [
    "shear_01", "shear_06", "storm_speed", "td_depression", "lcl_est",
    "srh_05_est", "stp_eff", "rfd_warmth", "streamwise_vort",
]
FEATURE_NAMES = RAW_NAMES + DERIVED_NAMES
N_FEATURES = len(FEATURE_NAMES)

# v1: raw pre-g2 grids (index-space pooled, mislocated; fill values fed to the model as
#     numbers). The forest shipped 2026-06-26 / 2026-07-25 was trained on v1.
# v2: hrrr geometry g3 grids (true lat/lon cells, earth-relative winds) with
#     sanitize_grids() applied (fill / impossible -> NaN).
FEATURE_SPEC_V1 = "hazardpulse/tornado/hrrr-env/v1"
FEATURE_SPEC_V2 = "hazardpulse/tornado/hrrr-env/v2"

# ---------------------------------------------------------------------------
# 1. Locating a report on the grid (one representation: hrrr's lat/lon cells)
# ---------------------------------------------------------------------------


def locate_cell(lat: float, lon: float) -> tuple[int, int] | None:
    """The 80 km cell holding ``(lat, lon)`` -- ``hrrr.latlon_to_hrrr_cell`` -- or None
    when the point is outside the 34 x 63 grid or in a cell with no native HRRR point.

    ``latlon_to_hrrr_cell`` clamps an off-grid point into an edge cell; a label must
    not be filed somewhere it did not happen, so this refuses instead.
    """
    lat, lon = float(lat), float(lon)
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    i = math.floor((lat - _hrrr.LAT_MIN) / _hrrr.GRID_DLAT)
    j = math.floor((lon - _hrrr.LON_MIN) / _hrrr.GRID_DLON)
    if not (0 <= i < HRRR_N_LAT and 0 <= j < HRRR_N_LON):
        return None
    cell = _hrrr.latlon_to_hrrr_cell(lat, lon)
    if not _domain_mask()[cell]:
        return None
    return cell


_DOMAIN_MASK: np.ndarray | None = None


def _domain_mask() -> np.ndarray:
    """hrrr.domain_mask(), computed once (it bins all 1.9 M native points per call)."""
    global _DOMAIN_MASK
    if _DOMAIN_MASK is None:
        _DOMAIN_MASK = np.asarray(_hrrr.domain_mask(), bool)
    return _DOMAIN_MASK


# The (u, v) frame of the grids this module reads. hrrr geometry g3 rotates the native
# grid-relative winds to earth-relative before pooling; any other geometry is refused
# by the label builder rather than carried in the wrong frame.
WIND_FRAME_BY_GEOMETRY = {"g3": "earth"}
EARTH_RADIUS_M = 6371229.0


def carry_latlon(lat: float, lon: float, u_ms: float, v_ms: float,
                 hours: float) -> tuple[float, float] | None:
    """Position ``hours`` earlier of a point moving with EARTH-relative ``(u, v)`` m/s
    (east, north), on the sphere's local tangent plane -- exact to well under a
    kilometre for the <= 2-3 h, <= 250 km carries the label window allows.
    None when the carried point is not finite.
    """
    if _hrrr.GEOMETRY_VERSION not in WIND_FRAME_BY_GEOMETRY:
        raise RuntimeError(f"wind frame of hrrr geometry {_hrrr.GEOMETRY_VERSION!r} unknown")
    lat, lon = float(lat), float(lon)
    dist_n = -float(v_ms) * float(hours) * 3600.0
    dist_e = -float(u_ms) * float(hours) * 3600.0
    new_lat = lat + math.degrees(dist_n / EARTH_RADIUS_M)
    new_lon = lon + math.degrees(dist_e / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    if not (math.isfinite(new_lat) and math.isfinite(new_lon)):
        return None
    return new_lat, new_lon


# ---------------------------------------------------------------------------
# 2. Fill values / physically impossible values -> NaN
# ---------------------------------------------------------------------------

# Inclusive physical envelope per raw field. Anything outside is a fill value (-10000)
# or a mean-pooled block contaminated by one. Bounds are deliberately wide: they must
# never clip a real HRRR analysis value.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "cape": (0.0, 10000.0), "mlcape": (0.0, 10000.0), "mucape": (0.0, 10000.0),
    "cin": (-2000.0, 1.0), "mlcin": (-2000.0, 1.0),
    "srh_01": (-2000.0, 3000.0), "srh_03": (-2000.0, 3000.0),
    "refc": (-40.0, 90.0),
    "ushear_01": (-100.0, 100.0), "vshear_01": (-100.0, 100.0),
    "ushear_06": (-100.0, 100.0), "vshear_06": (-100.0, 100.0),
    "ustorm": (-100.0, 100.0), "vstorm": (-100.0, 100.0),
    "t2m": (180.0, 340.0), "td2m": (150.0, 340.0),
    "pwat": (0.0, 150.0),
}


def sanitize_grids(grids: dict) -> dict:
    """Copy of ``grids`` with fill / out-of-envelope values set to NaN (spec v2)."""
    out = {}
    for k, v in grids.items():
        a = np.array(v, dtype=np.float32, copy=True)
        lo_hi = PHYSICAL_BOUNDS.get(k)
        if lo_hi is not None:
            with np.errstate(invalid="ignore"):
                bad = ~((a >= lo_hi[0]) & (a <= lo_hi[1]))
            a[bad] = np.nan
        out[k] = a
    return out


# ---------------------------------------------------------------------------
# 3. The storm population (eligibility gate, identical for both classes + serving)
# ---------------------------------------------------------------------------

# Convective core: cell-max composite reflectivity >= 40 dBZ. 40 dBZ is the
# conventional deep-convective-core threshold; widespread stratiform rain rarely
# reaches it in composite reflectivity. MEASURED (259 days 2022-2024, geometry g3):
# 93.7 % of tornado reports within the hour before the 20z analysis sit under it in
# their own cell, 97.6 % once carried back along the storm motion.
REFC_MIN_DBZ = 40.0
# Supercell-capable shear: 0-6 km bulk shear >= 12.5 m/s, the lower cutoff of the
# effective-layer Significant Tornado Parameter (its shear term is zero below it;
# coherence_engine.compute_derived_hrrr uses the same 12.5 m/s).
SHEAR06_MIN_MS = 12.5
# Some buoyancy for the storm: max(surface, mixed-layer, most-unstable) CAPE >= 100 J/kg.
CAPE_MIN_JKG = 100.0


def storm_population_mask(grids: dict, derived: dict | None = None, *,
                          refc_min: float = REFC_MIN_DBZ, shear06_min: float = SHEAR06_MIN_MS,
                          cape_min: float = CAPE_MIN_JKG) -> np.ndarray:
    """Boolean ``(34, 63)`` mask of cells in the benchmark / serving population.

    A cell is eligible iff it holds a convective core (``refc >= REFC_MIN_DBZ``) in a
    supercell-capable environment (``shear_06 >= SHEAR06_MIN_MS`` and
    ``max(cape, mlcape, mucape) >= CAPE_MIN_JKG``). A NaN in any gate field makes the
    cell ineligible: eligibility is never assumed for missing data. Pass sanitized
    grids (spec v2); fill values would otherwise pass or fail the gate arbitrarily.
    The keyword thresholds exist for sensitivity analysis; the defaults ARE the gate.
    """
    if derived is None:
        derived = compute_derived_hrrr(grids)
    refc = np.asarray(grids["refc"], np.float64)
    shear = np.asarray(derived["shear_06"], np.float64)
    capes = np.stack([np.asarray(grids[k], np.float64) for k in ("cape", "mlcape", "mucape")])
    cape_max = np.max(np.where(np.isnan(capes), -np.inf, capes), axis=0)
    cape_max[np.isneginf(cape_max)] = np.nan          # every CAPE flavour missing
    with np.errstate(invalid="ignore"):
        mask = (refc >= refc_min) & (shear >= shear06_min) & (cape_max >= cape_min)
    return np.asarray(mask & np.isfinite(refc) & np.isfinite(shear) & np.isfinite(cape_max), bool)


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

def cell_features(grids: dict, derived: dict, i: int, j: int) -> np.ndarray:
    """The 26-feature vector for HRRR cell (i, j)."""
    raw = [float(grids[v][i, j]) for v in RAW_NAMES]
    der = [float(derived[v][i, j]) for v in DERIVED_NAMES]
    return np.array(raw + der, dtype=np.float32)


def grid_feature_matrix(grids: dict, *, sanitize: bool = False):
    """Vectorized (N_cells, 26) feature matrix + the cape grid, for whole-grid scoring.

    Returns (X, cape) where X[i*HRRR_N_LON + j] is the feature vector for cell (i, j)
    in row-major order, and cape is the (34, 63) instability grid for masking.

    ``sanitize=False`` is feature spec v1 (what the shipped v1 forest was trained on);
    a v2 model MUST be served with ``sanitize=True`` (see FEATURE_SPEC_V2).
    """
    if sanitize:
        grids = sanitize_grids(grids)
    derived = compute_derived_hrrr(grids)
    layers = [np.asarray(grids[v], np.float32) for v in RAW_NAMES]
    layers += [np.asarray(derived[v], np.float32) for v in DERIVED_NAMES]
    # stack to (34, 63, 26) then flatten cells -> (34*63, 26)
    cube = np.stack(layers, axis=-1)
    X = cube.reshape(HRRR_N_LAT * HRRR_N_LON, N_FEATURES)
    cape = np.asarray(grids.get("mlcape", grids.get("cape")), np.float32)
    return X, cape
