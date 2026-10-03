# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational earthquake prediction system.
# It does NOT replace official USGS or national seismological agency warnings.
# Always follow guidance from your national geological survey.
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Coherence field theory engine for earthquake prediction.

Implements the Helmholtz PDE solver and all derived seismic coherence
quantities on a 2-degree global lat/lon grid.  Pure NumPy + SciPy --
no external ML or physics libraries required.

Physics summary
----------------
The seismic coherence field tau_c(x, t) satisfies the screened Poisson
(Helmholtz) equation derived from the Equation of One (S_One):

    D * nabla^2(tau_c)  -  Gamma * tau_c  +  S(x, t)  =  0

where:
  - S(x, t)     = tectonic loading rate (seismic moment release density)
  - Gamma(x, t) = healing/friction rate (inverse recurrence time)
  - D           = stress diffusivity (~10^-2 m^2/s tectonic, ~1 m^2/s fluid)
  - kappa       = sqrt(Gamma / D) = inverse screening length
  - ell         = 1 / kappa = coherence length

Key prediction (CFT):  ell(t) -> infinity before mainshock
    ell(t) ~ |t_c - t|^(-nu)    where nu ~ 0.5 (mean-field critical exponent)

This manifests as:
  1. b-value DECREASES (energy concentrates in larger events)
  2. Correlation length of seismicity INCREASES
  3. Inter-event time distribution shifts from exponential to Lorentzian
  4. Spatial clustering TIGHTENS
  5. Rate acceleration (AMR -- accelerating moment release)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover
    curve_fit = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Grid constants -- 2-degree global grid
# ---------------------------------------------------------------------------

GRID_DLAT: float = 2.0
GRID_DLON: float = 2.0
LAT_MIN: float = -60.0
LAT_MAX: float = 70.0
LON_MIN: float = -180.0
LON_MAX: float = 180.0

N_LAT: int = int((LAT_MAX - LAT_MIN) / GRID_DLAT)   # 65
N_LON: int = int((LON_MAX - LON_MIN) / GRID_DLON)    # 180

# Approximate km per degree at mid-latitudes
KM_PER_DEG: float = 111.0

# ---------------------------------------------------------------------------
# Physical constants and thresholds
# ---------------------------------------------------------------------------

# Stress diffusivity (dimensionless units on the 2-deg grid; controls
# how far stress perturbations propagate per Jacobi iteration).
BASE_D: float = 1.0

# Helmholtz solver iterations -- 2-deg grid is small enough for fast
# convergence; 300 iterations gives <1% residual on a 65x180 grid.
HELMHOLTZ_ITERS: int = 300

# SOR relaxation parameter (0 < omega <= 1).  0.8 balances speed and
# stability on the sparse seismicity source field.
SOR_OMEGA: float = 0.8

# Background b-value for the global Gutenberg-Richter relation.
# Well-established empirical value (Kanamori 1981).
B_VALUE_BACKGROUND: float = 1.0

# Magnitude of completeness floor -- USGS global catalog is complete
# above M2.5 since ~2000 (Wiemer & Wyss 2000).
MC_FLOOR: float = 2.5

# Earth radius for distance calculations (km).
EARTH_RADIUS_KM: float = 6371.0

# Minimum events required for reliable b-value estimate (Woessner &
# Wiemer 2005 recommend N >= 50 for MLE b-value).
MIN_EVENTS_B_VALUE: int = 50

# Minimum event pairs for correlation length estimate.
MIN_PAIRS_CORR_LENGTH: int = 100

# Background correlation length (km) -- typical value for global
# seismicity at M2.5+ in tectonically active regions.  Derived from
# mean nearest-neighbour distance in subduction zone catalogs.
ELL_BACKGROUND_KM: float = 50.0

# ---------------------------------------------------------------------------
# Singularity test thresholds (earthquake-specific)
# ---------------------------------------------------------------------------
# Each threshold has a physical justification documented inline.

# 1. Correlation length must exceed 1.5x background.  This factor is
#    the mean-field prediction for the onset of critical correlations
#    (ell/ell_bg ~ 1.5 at t_c - t ~ 2 years for M7+ events).
ELL_FACTOR_THRESHOLD: float = 1.5

# 2. b-value must drop 0.15 below background.  A 0.15 decrease in b
#    corresponds to a factor ~1.4x increase in the fraction of energy
#    released in the largest events (Schorlemmer et al. 2005).
B_VALUE_DROP_THRESHOLD: float = 0.15

# 3. IET distribution: delta-AIC < -2 favouring Lorentzian over
#    exponential.  AIC difference of 2 corresponds to ~2.7x likelihood
#    ratio (Burnham & Anderson 2002).
DELTA_AIC_THRESHOLD: float = -2.0

# 4. Rate acceleration > 1.5x.  A 50% increase in seismicity rate is
#    the minimum detectable above Poisson fluctuations for typical
#    catalog sizes (Reasenberg & Jones 1989).
RATE_ACCEL_THRESHOLD: float = 1.5

# 5. Source/Gamma ratio > 1.0 means tectonic loading exceeds fault
#    healing -- the fault is accumulating stress faster than it can
#    dissipate through aseismic creep.
S_OVER_GAMMA_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Utility: haversine distance
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _haversine_km_arrays(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance in km."""
    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def latlon_to_grid_cell(lat: float, lon: float) -> tuple[int, int]:
    """Convert lat/lon to (row, col) indices on the 2-degree grid.

    Returns clipped indices so out-of-range coordinates map to the
    nearest edge cell rather than raising.
    """
    row = int((lat - LAT_MIN) / GRID_DLAT)
    col = int((lon - LON_MIN) / GRID_DLON)
    row = max(0, min(row, N_LAT - 1))
    col = max(0, min(col, N_LON - 1))
    return row, col


def grid_cell_to_latlon(row: int, col: int) -> tuple[float, float]:
    """Convert grid indices to the cell-centre lat/lon."""
    lat = LAT_MIN + (row + 0.5) * GRID_DLAT
    lon = LON_MIN + (col + 0.5) * GRID_DLON
    return lat, lon


# ---------------------------------------------------------------------------
# Event time parsing
# ---------------------------------------------------------------------------


def _parse_event_time(time_str: str) -> float:
    """Parse ISO-8601 timestamp to Unix epoch seconds.

    Handles the common USGS format ``2011-03-11T05:46:24.120Z``.
    Falls back to 0.0 if unparseable.
    """
    import datetime as _dt

    if not time_str:
        return 0.0
    try:
        # Strip trailing Z and parse
        clean = time_str.rstrip("Z")
        # Handle fractional seconds
        if "." in clean:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
        else:
            fmt = "%Y-%m-%dT%H:%M:%S"
        dt = _dt.datetime.strptime(clean, fmt).replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _epoch_to_days(epoch: float) -> float:
    """Convert Unix epoch seconds to fractional days since epoch."""
    return epoch / 86400.0


# ---------------------------------------------------------------------------
# Seismic moment
# ---------------------------------------------------------------------------


def compute_seismic_moment(magnitude: float) -> float:
    """Convert magnitude to seismic moment M0 in Newton-metres.

    Uses the Hanks & Kanamori (1979) relation:
        log10(M0) = 1.5 * M + 9.05

    Parameters
    ----------
    magnitude : float
        Event magnitude (any scale approximately equivalent to Mw).

    Returns
    -------
    float
        Seismic moment in N*m.
    """
    return 10.0 ** (1.5 * magnitude + 9.05)


# ---------------------------------------------------------------------------
# Helmholtz PDE solver (reused pattern from tornado coherence engine)
# ---------------------------------------------------------------------------


def solve_helmholtz_2d(
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
    n_iter: int = HELMHOLTZ_ITERS,
    omega: float = SOR_OMEGA,
) -> np.ndarray:
    """Solve ``D nabla^2 tau - gamma tau + S = 0`` on a 2-D grid.

    Earthquake-grid wrapper around the shared
    ``hazardpulse.coherence.tau_c_solver.solve_helmholtz_2d``.
    Uses float64 (small global 2-degree grid; precision over speed).
    ``gamma`` is the damping rate Gamma(x) itself — this module's header
    has always stated the PDE as ``D*nabla^2(tau) - Gamma*tau + S = 0``
    with kappa = sqrt(Gamma/D) a DERIVED quantity; the solver input now
    matches that statement (FVCS W-2).
    """
    from hazardpulse.coherence.tau_c_solver import (
        solve_helmholtz_2d as _shared_solver,
    )
    return _shared_solver(
        source, gamma, dx,
        D=D, n_iter=n_iter, omega=omega, dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Gradient operators
# ---------------------------------------------------------------------------


def _gradient_2d(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient (d/dy, d/dx) using central differences."""
    grad_y = np.zeros_like(field, dtype=np.float64)
    grad_x = np.zeros_like(field, dtype=np.float64)

    grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)
    grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)

    # Forward/backward at edges
    grad_y[0, :] = (field[1, :] - field[0, :]) / dy
    grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dy
    grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
    grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx

    return grad_y, grad_x


# ---------------------------------------------------------------------------
# b-value (Gutenberg-Richter)
# ---------------------------------------------------------------------------


def _estimate_mc(magnitudes: np.ndarray) -> float:
    """Estimate magnitude of completeness via maximum curvature method.

    The maximum curvature method (Wiemer & Wyss 2000) finds the magnitude
    bin with the highest frequency in the non-cumulative frequency-
    magnitude distribution.  We add 0.2 as a conservative correction
    (Woessner & Wiemer 2005).

    Parameters
    ----------
    magnitudes : ndarray
        Array of event magnitudes.

    Returns
    -------
    float
        Estimated Mc.
    """
    if len(magnitudes) < 10:
        return MC_FLOOR
    # Bin at 0.1 intervals
    bins = np.arange(
        math.floor(magnitudes.min() * 10) / 10.0,
        math.ceil(magnitudes.max() * 10) / 10.0 + 0.1,
        0.1,
    )
    counts, edges = np.histogram(magnitudes, bins=bins)
    if len(counts) == 0:
        return MC_FLOOR
    peak_idx = int(np.argmax(counts))
    mc = edges[peak_idx] + 0.2  # conservative correction
    return max(mc, MC_FLOOR)


def compute_b_value(
    magnitudes: np.ndarray | Sequence[float],
    mc: float | None = None,
) -> tuple[float, float, float, int]:
    """Compute Gutenberg-Richter b-value using Aki-Utsu MLE.

    b = log10(e) / (M_mean - (Mc - delta_M/2))

    where delta_M = 0.1 is the magnitude binning interval (Aki 1965,
    Utsu 1965).

    Parameters
    ----------
    magnitudes : array-like
        Event magnitudes.
    mc : float or None
        Magnitude of completeness.  If None, estimated via maximum
        curvature method.

    Returns
    -------
    tuple of (b_value, mc_used, b_uncertainty, n_used)
        b_value : float -- the MLE b-value (NaN if insufficient data)
        mc_used : float -- magnitude of completeness used
        b_uncertainty : float -- Shi-Bolt uncertainty (b / sqrt(N))
        n_used : int -- number of events above Mc
    """
    mags = np.asarray(magnitudes, dtype=np.float64)
    if len(mags) == 0:
        return float("nan"), MC_FLOOR, float("nan"), 0

    if mc is None:
        mc = _estimate_mc(mags)

    above = mags[mags >= mc]
    n = len(above)
    if n < MIN_EVENTS_B_VALUE:
        return float("nan"), mc, float("nan"), n

    delta_m = 0.05  # half the 0.1 binning width
    m_mean = float(np.mean(above))
    denom = m_mean - (mc - delta_m)
    if denom <= 0:
        return float("nan"), mc, float("nan"), n

    b = math.log10(math.e) / denom
    # Shi & Bolt (1982) uncertainty
    b_unc = b / math.sqrt(n)

    return b, mc, b_unc, n


# ---------------------------------------------------------------------------
# Spatial correlation length
# ---------------------------------------------------------------------------


def compute_correlation_length(
    events: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float = 500.0,
    time_window_days: float = 365.0,
    ref_epoch: float | None = None,
) -> tuple[float, int]:
    """Compute spatial correlation length ell of seismicity.

    Method: two-point correlation function C(r) from event pair
    counting in distance bins, then fit C(r) = A * exp(-r / ell)
    to extract ell.

    This is the DIRECT measurement of the Helmholtz screening length.
    ell increasing -> approaching criticality -> earthquake likely.

    Uses pair counting (not full distance matrix) by processing events
    in chunks to keep memory bounded.

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog entries with 'latitude', 'longitude', 'time'.
    center_lat, center_lon : float
        Centre of the analysis region.
    radius_km : float
        Radius of the circular region to analyse.
    time_window_days : float
        Only use events within this many days before ref_epoch.
    ref_epoch : float or None
        Reference time (Unix epoch seconds).  If None, use latest event.

    Returns
    -------
    tuple of (ell_km, n_events)
        ell_km : float -- correlation length in km (NaN if insufficient data)
        n_events : int -- number of events used
    """
    # Filter events to spatial and temporal window
    filtered = _filter_events(events, center_lat, center_lon,
                              radius_km, time_window_days, ref_epoch)
    n = len(filtered)
    if n < 10:
        return float("nan"), n

    lats = np.array([e["latitude"] for e in filtered])
    lons = np.array([e["longitude"] for e in filtered])

    # Distance bins (km)
    max_dist = min(radius_km, 500.0)
    bin_width = max(5.0, max_dist / 50.0)
    bins = np.arange(0, max_dist + bin_width, bin_width)
    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype=np.int64)

    # Pair counting in chunks to avoid O(N^2) memory
    chunk_size = 2000
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        for j_start in range(i_start, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            # Only count each pair once (j > i)
            lat_i = lats[i_start:i_end, np.newaxis]
            lon_i = lons[i_start:i_end, np.newaxis]
            lat_j = lats[np.newaxis, j_start:j_end]
            lon_j = lons[np.newaxis, j_start:j_end]

            dists = _haversine_km_arrays(
                lat_i.ravel()[:, np.newaxis] * np.ones((1, j_end - j_start)),
                lon_i.ravel()[:, np.newaxis] * np.ones((1, j_end - j_start)),
                np.ones((i_end - i_start, 1)) * lat_j.ravel()[np.newaxis, :],
                np.ones((i_end - i_start, 1)) * lon_j.ravel()[np.newaxis, :],
            )

            # Mask self-pairs and double-counting
            if i_start == j_start:
                mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
                dists = dists[mask]
            elif j_start < i_start:
                continue  # already counted
            else:
                dists = dists.ravel()

            # Bin the distances
            bin_idx = np.searchsorted(bins, dists, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            for idx in bin_idx[valid]:
                counts[idx] += 1

    # Normalize to get correlation function C(r)
    # Normalize by shell area (annular ring area proportional to r)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_area = np.maximum(bin_centers, 1.0)  # proportional to r
    C_r = counts.astype(np.float64) / shell_area

    if C_r.max() <= 0:
        return float("nan"), n

    C_r /= C_r.max()  # normalize to [0, 1]

    # Fit C(r) = A * exp(-r / ell)
    valid_mask = C_r > 0.01
    if valid_mask.sum() < 3:
        return float("nan"), n

    r_fit = bin_centers[valid_mask]
    c_fit = C_r[valid_mask]

    if curve_fit is None:
        # Fallback: log-linear regression
        log_c = np.log(np.maximum(c_fit, 1e-10))
        # ell = -1/slope
        if len(r_fit) < 2:
            return float("nan"), n
        slope, _ = np.polyfit(r_fit, log_c, 1)
        if slope >= 0:
            return float("nan"), n
        ell = -1.0 / slope
    else:
        def _exp_decay(r: np.ndarray, A: float, inv_ell: float) -> np.ndarray:
            return A * np.exp(-r * inv_ell)
        try:
            popt, _ = curve_fit(
                _exp_decay, r_fit, c_fit,
                p0=[1.0, 1.0 / 50.0],
                bounds=([0.0, 1e-6], [10.0, 1.0]),
                maxfev=2000,
            )
            ell = 1.0 / popt[1]
        except (RuntimeError, ValueError):
            # Fallback to log-linear
            log_c = np.log(np.maximum(c_fit, 1e-10))
            slope, _ = np.polyfit(r_fit, log_c, 1)
            if slope >= 0:
                return float("nan"), n
            ell = -1.0 / slope

    return float(ell), n


# ---------------------------------------------------------------------------
# Event filtering helper
# ---------------------------------------------------------------------------


def _filter_events(
    events: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
    time_window_days: float,
    ref_epoch: float | None = None,
) -> list[dict]:
    """Filter events by spatial radius and time window.

    Parsing the catalog times is cached per event-list (it was re-parsed on every call,
    and _filter_events is called several times per sample -> millions of redundant
    string parses). The time + haversine filter is vectorized. Identical result.
    """
    evs, lats, lons, eps = _parsed_catalog_arrays(events)
    if len(evs) == 0:
        return []
    if ref_epoch is None:
        ref_epoch = float(eps.max())
    t_min = ref_epoch - time_window_days * 86400.0
    idx = np.where((eps >= t_min) & (eps <= ref_epoch))[0]
    if idx.size == 0:
        return []
    dist = _haversine_km_arrays(center_lat, center_lon, lats[idx], lons[idx])
    keep = idx[dist <= radius_km]
    return [evs[i] for i in keep]


_PARSED_CATALOG_CACHE: dict = {}


def _parsed_catalog_arrays(events):
    """Parse an event list to (events, lat[], lon[], epoch[]) ONCE; cache by list id."""
    key = id(events)
    cached = _PARSED_CATALOG_CACHE.get(key)
    if cached is None:
        evs, lats, lons, eps = [], [], [], []
        for e in events:
            lat = e.get("latitude"); lon = e.get("longitude")
            if lat is None or lon is None:
                continue
            t = e.get("_epoch")                      # memoized parse (strptime is the hot spot)
            if t is None:
                t_str = e.get("time", "")
                t = _parse_event_time(t_str) if isinstance(t_str, str) else float(t_str)
                e["_epoch"] = t
            evs.append(e); lats.append(lat); lons.append(lon); eps.append(t)
        _PARSED_CATALOG_CACHE.clear()       # keep only the latest list (bounded memory)
        cached = (evs, np.asarray(lats, float), np.asarray(lons, float), np.asarray(eps, float))
        _PARSED_CATALOG_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Inter-event time distribution
# ---------------------------------------------------------------------------


def compute_iet_distribution(
    events: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float = 300.0,
    time_window_days: float = 365.0,
    ref_epoch: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Compute inter-event time distribution and compare Lorentzian vs exponential.

    Lorentzian IET -> coherent (damped resonance) -> approaching criticality.
    Exponential IET -> Poissonian (random) -> background state.

    The Lorentzian distribution arises naturally from the Helmholtz PDE
    when the coherence field develops a resonance (pole in the Green's
    function).  An exponential IET indicates memoryless (Poisson) process
    with no spatial coherence.

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog entries.
    center_lat, center_lon : float
        Centre of analysis region.
    radius_km : float
        Spatial radius.
    time_window_days : float
        Temporal window.
    ref_epoch : float or None
        Reference time.

    Returns
    -------
    tuple of (iet_days, lorentzian_aic, exponential_aic, delta_aic)
        iet_days : ndarray -- inter-event times in days
        lorentzian_aic : float -- AIC for Lorentzian fit
        exponential_aic : float -- AIC for exponential fit
        delta_aic : float -- lorentzian_aic - exponential_aic
            (negative = Lorentzian preferred)
    """
    filtered = _filter_events(events, center_lat, center_lon,
                              radius_km, time_window_days, ref_epoch)

    # Sort by time
    times: list[float] = []
    for e in filtered:
        t_str = e.get("time", "")
        t = _parse_event_time(t_str) if isinstance(t_str, str) else float(t_str)
        if t > 0:
            times.append(t)

    times.sort()
    if len(times) < 20:
        return np.array([]), float("nan"), float("nan"), float("nan")

    iet = np.diff(times) / 86400.0  # convert to days
    iet = iet[iet > 0]  # remove zero intervals

    if len(iet) < 10:
        return iet, float("nan"), float("nan"), float("nan")

    n = len(iet)

    # --- Exponential MLE ---
    # f(t) = lambda * exp(-lambda * t), lambda = 1 / mean
    mean_iet = float(np.mean(iet))
    if mean_iet <= 0:
        return iet, float("nan"), float("nan"), float("nan")

    lam = 1.0 / mean_iet
    log_lik_exp = n * math.log(lam) - lam * float(np.sum(iet))
    aic_exp = -2.0 * log_lik_exp + 2.0 * 1  # 1 parameter (lambda)

    # --- Lorentzian (Cauchy) MLE ---
    # f(t) = (1/pi) * gamma / ((t - t0)^2 + gamma^2)
    # Use median as location parameter (robust), MAD as scale
    median_iet = float(np.median(iet))
    mad = float(np.median(np.abs(iet - median_iet)))
    gamma_lor = max(mad, 1e-6)
    t0_lor = median_iet

    log_lik_lor = float(np.sum(
        np.log(gamma_lor / (np.pi * ((iet - t0_lor) ** 2 + gamma_lor ** 2)))
    ))
    aic_lor = -2.0 * log_lik_lor + 2.0 * 2  # 2 parameters (t0, gamma)

    delta_aic = aic_lor - aic_exp  # negative = Lorentzian preferred

    return iet, aic_lor, aic_exp, delta_aic


# ---------------------------------------------------------------------------
# Correlation length evolution
# ---------------------------------------------------------------------------


def compute_correlation_length_evolution(
    events: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float = 300.0,
    window_days: float = 180.0,
    step_days: float = 30.0,
    n_windows: int | None = None,
) -> list[dict]:
    """Track ell(t) over time by computing correlation length in sliding windows.

    This is the KEY diagnostic -- if ell(t) is accelerating, rupture
    is approaching.

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog.
    center_lat, center_lon : float
        Centre of analysis region.
    radius_km : float
        Spatial radius (km).
    window_days : float
        Width of the sliding time window.
    step_days : float
        Step size between windows.
    n_windows : int or None
        Maximum number of windows (from most recent backward).  If None,
        covers all available data.

    Returns
    -------
    list[dict]
        Each dict has keys: epoch, ell_km, b_value, rate, moment_rate,
        n_events.  Sorted chronologically.
    """
    # Parse + spatial-filter via the cached/vectorized path. The per-event Python
    # parse+haversine here was the remaining block_c hot spot (a full catalog pass per
    # sample, bypassing the memo). _parsed_catalog_arrays memoizes the parse and
    # _haversine_km_arrays vectorizes the distance. Identical result: 'parsed' is sorted
    # by time below, so build order is irrelevant.
    evs, lats_a, lons_a, eps_a = _parsed_catalog_arrays(events)
    if len(evs) == 0:
        return []
    d = _haversine_km_arrays(center_lat, center_lon, lats_a, lons_a)
    keep = np.where((d <= radius_km) & (eps_a > 0))[0]
    parsed: list[tuple[dict, float]] = [(evs[i], float(eps_a[i])) for i in keep]

    if not parsed:
        return []

    parsed.sort(key=lambda x: x[1])
    t_min_data = parsed[0][1]
    t_max_data = parsed[-1][1]

    # Generate window endpoints
    window_sec = window_days * 86400.0
    step_sec = step_days * 86400.0

    endpoints: list[float] = []
    t_end = t_max_data
    while t_end - window_sec >= t_min_data:
        endpoints.append(t_end)
        t_end -= step_sec
        if n_windows is not None and len(endpoints) >= n_windows:
            break

    endpoints.reverse()

    results: list[dict] = []
    for t_end in endpoints:
        t_start = t_end - window_sec
        window_events = [
            e for e, t in parsed if t_start <= t <= t_end
        ]

        n_ev = len(window_events)
        if n_ev < 5:
            continue

        # Correlation length
        ell, _ = compute_correlation_length(
            window_events, center_lat, center_lon,
            radius_km=radius_km, time_window_days=window_days + 1,
            ref_epoch=t_end,
        )

        # b-value
        mags = np.array([
            e["mag"] for e in window_events
            if e.get("mag") is not None
        ])
        b, _, _, _ = compute_b_value(mags)

        # Rate (events per day)
        rate = n_ev / window_days

        # Moment rate
        moment_rate = sum(
            compute_seismic_moment(e["mag"])
            for e in window_events
            if e.get("mag") is not None
        ) / window_days

        results.append({
            "epoch": t_end,
            "ell_km": ell,
            "b_value": b,
            "rate": rate,
            "moment_rate": moment_rate,
            "n_events": n_ev,
        })

    return results


# ---------------------------------------------------------------------------
# Divergence fitting:  ell(t) ~ |t_c - t|^(-nu)
# ---------------------------------------------------------------------------


def fit_divergence(
    ell_timeseries: list[dict],
    method: str = "power_law",
) -> tuple[float, float, float]:
    """Fit ell(t) ~ |t_c - t|^(-nu) to estimate time of criticality t_c.

    This is the DIRECT test of CFT's claim that earthquakes are coherence
    phase transitions.  If the fitted t_c actually precedes real M6+
    events, that is a genuine discovery.

    Parameters
    ----------
    ell_timeseries : list[dict]
        Output of ``compute_correlation_length_evolution``.  Each dict
        must have 'epoch' (seconds) and 'ell_km' (float).
    method : str
        'power_law' -- fit ell = A * |t_c - t|^(-nu)
        'log_linear' -- fit log(ell) = -nu * log|t_c - t| + C

    Returns
    -------
    tuple of (t_c_epoch, nu, r_squared)
        t_c_epoch : float -- estimated rupture time (Unix epoch seconds)
        nu : float -- critical exponent (CFT predicts ~0.5)
        r_squared : float -- goodness of fit

    Notes
    -----
    If nu ~ 0.5 (mean-field), this is consistent with CFT.
    If t_c is within the next 30/90/365 days, flag as elevated risk.
    """
    # Filter to valid entries
    valid = [
        (d["epoch"], d["ell_km"])
        for d in ell_timeseries
        if not math.isnan(d.get("ell_km", float("nan")))
        and d["ell_km"] > 0
    ]

    if len(valid) < 5:
        return float("nan"), float("nan"), float("nan")

    epochs = np.array([v[0] for v in valid])
    ells = np.array([v[1] for v in valid])

    if curve_fit is None:
        return float("nan"), float("nan"), float("nan")

    # Initial guess: t_c just after last data point
    t_last = epochs[-1]
    t_range = epochs[-1] - epochs[0]

    if method == "power_law":
        def _model(t: np.ndarray, t_c: float, A: float, nu: float) -> np.ndarray:
            dt = np.maximum(t_c - t, 1.0)  # avoid zero/negative
            return A * dt ** (-nu)

        try:
            popt, _ = curve_fit(
                _model, epochs, ells,
                p0=[t_last + t_range * 0.1, ells.min(), 0.5],
                bounds=(
                    [t_last, 0.01, 0.01],
                    [t_last + t_range * 10, ells.max() * 100, 5.0],
                ),
                maxfev=5000,
            )
            t_c, A, nu = popt
            predicted = _model(epochs, t_c, A, nu)
        except (RuntimeError, ValueError):
            return float("nan"), float("nan"), float("nan")

    elif method == "log_linear":
        # Try a grid of t_c values and pick the best
        best_r2 = -1.0
        best_tc = float("nan")
        best_nu = float("nan")

        for tc_offset_frac in np.linspace(0.01, 5.0, 50):
            tc_trial = t_last + tc_offset_frac * t_range * 0.1
            dt = tc_trial - epochs
            if np.any(dt <= 0):
                continue
            log_dt = np.log(dt)
            log_ell = np.log(ells)
            # Linear fit: log(ell) = -nu * log(dt) + C
            coeffs = np.polyfit(log_dt, log_ell, 1)
            slope, intercept = coeffs
            nu_trial = -slope
            if nu_trial <= 0:
                continue
            predicted_log = slope * log_dt + intercept
            ss_res = float(np.sum((log_ell - predicted_log) ** 2))
            ss_tot = float(np.sum((log_ell - np.mean(log_ell)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if r2 > best_r2:
                best_r2 = r2
                best_tc = tc_trial
                best_nu = nu_trial

        return best_tc, best_nu, best_r2

    else:
        raise ValueError(f"Unknown method: {method}")

    # R-squared
    ss_res = float(np.sum((ells - predicted) ** 2))
    ss_tot = float(np.sum((ells - np.mean(ells)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    return float(t_c), float(nu), r_squared


# ---------------------------------------------------------------------------
# Accelerating moment release (AMR)
# ---------------------------------------------------------------------------


def compute_moment_acceleration(
    events: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float = 300.0,
    ref_epoch: float | None = None,
    total_window_days: float = 3650.0,
) -> tuple[float, float, float]:
    """Compute accelerating moment release (AMR).

    Sum cumulative moment in expanding time windows and fit a power-law
    acceleration:  sum(M0(t)) ~ |t_c - t|^(-m)

    AMR is one of the longest-studied earthquake precursors (Bufe &
    Varnes 1993).  The Helmholtz PDE predicts it through the divergence
    of the coherence field source term.

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog.
    center_lat, center_lon : float
        Centre of analysis region.
    radius_km : float
        Spatial radius.
    ref_epoch : float or None
        Reference time.
    total_window_days : float
        Total lookback period for cumulative moment.

    Returns
    -------
    tuple of (t_c_epoch, exponent_m, r_squared)
    """
    filtered = _filter_events(events, center_lat, center_lon,
                              radius_km, total_window_days, ref_epoch)

    if len(filtered) < 20:
        return float("nan"), float("nan"), float("nan")

    # Parse times and moments
    time_moment: list[tuple[float, float]] = []
    for e in filtered:
        t_str = e.get("time", "")
        t = _parse_event_time(t_str) if isinstance(t_str, str) else float(t_str)
        mag = e.get("mag")
        if t > 0 and mag is not None:
            time_moment.append((t, compute_seismic_moment(mag)))

    if len(time_moment) < 20:
        return float("nan"), float("nan"), float("nan")

    time_moment.sort()
    times = np.array([tm[0] for tm in time_moment])
    moments = np.array([tm[1] for tm in time_moment])

    # Cumulative Benioff strain (sqrt of cumulative moment)
    cum_moment = np.cumsum(moments)
    cum_strain = np.sqrt(cum_moment)

    if curve_fit is None:
        return float("nan"), float("nan"), float("nan")

    t_last = times[-1]
    t_range = times[-1] - times[0]

    def _amr_model(t: np.ndarray, t_c: float, A: float, m: float, C: float) -> np.ndarray:
        dt = np.maximum(t_c - t, 1.0)
        return C - A * dt ** (1.0 - m)

    try:
        popt, _ = curve_fit(
            _amr_model, times, cum_strain,
            p0=[t_last + t_range * 0.05, cum_strain[-1] * 0.01, 0.3, cum_strain[-1]],
            bounds=(
                [t_last, 0, 0.01, 0],
                [t_last + t_range * 5, cum_strain[-1] * 10, 2.0, cum_strain[-1] * 10],
            ),
            maxfev=5000,
        )
        t_c, A, m_exp, C = popt
        predicted = _amr_model(times, t_c, A, m_exp, C)
        ss_res = float(np.sum((cum_strain - predicted) ** 2))
        ss_tot = float(np.sum((cum_strain - np.mean(cum_strain)) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
        return float(t_c), float(m_exp), r_squared
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Seismic coherence field on global grid
# ---------------------------------------------------------------------------


def compute_seismic_coherence_field(
    events: list[dict],
    time_window_days: float = 365.0,
    ref_epoch: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute the seismic coherence field tau_c on a global 2-degree grid.

    Source S(x): seismic moment release density (total moment per cell
    normalised by cell area).
    Damping Gamma(x): inverse recurrence time (estimated from local
    inter-event rate history).
    Diffusion D: fixed at BASE_D (uniform stress diffusivity assumption).

    Solves the Helmholtz PDE: D nabla^2(tau) - Gamma tau + S = 0

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog with 'latitude', 'longitude', 'mag', 'time'.
    time_window_days : float
        Temporal window for computing source and damping fields.
    ref_epoch : float or None
        Reference time.  If None, use latest event time.

    Returns
    -------
    dict with keys:
        tau : ndarray (N_LAT, N_LON) -- coherence field
        grad_tau : ndarray -- gradient magnitude
        kappa_field : ndarray -- screening wavenumber
        S_field : ndarray -- source (moment density)
        Gamma_field : ndarray -- damping (inverse recurrence)
        S_over_Gamma : ndarray -- loading / healing ratio
        Da : ndarray -- Damkohler number
    """
    # Parse event times
    parsed: list[tuple[dict, float]] = []
    for e in events:
        lat = e.get("latitude")
        lon = e.get("longitude")
        mag = e.get("mag")
        t_str = e.get("time", "")
        if lat is None or lon is None or mag is None:
            continue
        t = _parse_event_time(t_str) if isinstance(t_str, str) else float(t_str)
        if t > 0:
            parsed.append((e, t))

    # Determine reference epoch
    if not parsed:
        return _empty_grid_result()

    if ref_epoch is None:
        ref_epoch = max(t for _, t in parsed)

    t_min = ref_epoch - time_window_days * 86400.0

    # --- Build source field S(x): moment release density ---
    S_field = np.zeros((N_LAT, N_LON), dtype=np.float64)
    event_count = np.zeros((N_LAT, N_LON), dtype=np.float64)

    # Also track inter-event times per cell for Gamma
    cell_times: dict[tuple[int, int], list[float]] = {}

    for e, t in parsed:
        if t < t_min or t > ref_epoch:
            continue
        row, col = latlon_to_grid_cell(e["latitude"], e["longitude"])
        m0 = compute_seismic_moment(e["mag"])
        # Normalise moment by log to avoid dynamic range issues
        # log10(M0) ranges from ~12 (M2) to ~23 (M9.5)
        S_field[row, col] += math.log10(m0)
        event_count[row, col] += 1.0

        key = (row, col)
        if key not in cell_times:
            cell_times[key] = []
        cell_times[key].append(t)

    # Normalise source by window length (rate of moment accumulation)
    S_field /= time_window_days

    # --- Build damping field Gamma(x): inverse recurrence time ---
    # Gamma = 1 / mean_recurrence_time.  High Gamma = fast healing.
    # Cells with no events get a large Gamma (fast healing = stable).
    Gamma_field = np.full((N_LAT, N_LON), 0.1, dtype=np.float64)

    for (row, col), times_list in cell_times.items():
        if len(times_list) < 2:
            # Single event -- use window length as recurrence
            Gamma_field[row, col] = 1.0 / time_window_days
            continue
        times_sorted = sorted(times_list)
        iets = np.diff(times_sorted) / 86400.0  # days
        mean_iet = float(np.mean(iets))
        if mean_iet > 0:
            Gamma_field[row, col] = 1.0 / mean_iet
        else:
            Gamma_field[row, col] = 1.0  # very frequent events

    # --- Solve Helmholtz ---
    D_field = np.full((N_LAT, N_LON), BASE_D, dtype=np.float64)
    # kappa is a DERIVED reporting quantity (returned below, feeds kappa_local);
    # the solver consumes Gamma directly (FVCS W-2).
    kappa_field = np.sqrt(
        Gamma_field / np.maximum(D_field, 1e-12)
    )

    # Passed as kappa_field**2 rather than Gamma_field: with BASE_D = 1 the two are
    # equal in exact arithmetic, but sqrt-then-square is not the identity in floats,
    # and kappa_field**2 reproduces BIT-FOR-BIT the value the solver used to compute
    # internally — the live earthquake ledger's scores must not move under a rename.
    tau = solve_helmholtz_2d(S_field, kappa_field ** 2, dx=1.0, D=D_field)

    # Gradient magnitude
    grad_y, grad_x = _gradient_2d(tau)
    grad_tau = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # S / Gamma ratio
    S_over_Gamma = S_field / np.maximum(Gamma_field, 1e-12)

    # Damkohler number: Da = Gamma * L^2 / (D * v_scale)
    # Use grid spacing as length scale, v_scale = plate velocity ~ 50 mm/yr
    L_km = GRID_DLAT * KM_PER_DEG  # ~222 km
    v_scale = 0.05 / 365.25  # 50 mm/yr in km/day (for dimensional consistency)
    Da = Gamma_field * (L_km ** 2) / np.maximum(D_field * 100.0, 1e-6)

    return {
        "tau": tau,
        "grad_tau": grad_tau,
        "kappa_field": kappa_field,
        "S_field": S_field,
        "Gamma_field": Gamma_field,
        "S_over_Gamma": S_over_Gamma,
        "Da": Da,
        "event_count": event_count,
    }


def _empty_grid_result() -> dict[str, np.ndarray]:
    """Return an all-zeros grid result when there are no events."""
    zeros = np.zeros((N_LAT, N_LON), dtype=np.float64)
    return {
        "tau": zeros.copy(),
        "grad_tau": zeros.copy(),
        "kappa_field": zeros.copy(),
        "S_field": zeros.copy(),
        "Gamma_field": np.full((N_LAT, N_LON), 0.1, dtype=np.float64),
        "S_over_Gamma": zeros.copy(),
        "Da": zeros.copy(),
        "event_count": zeros.copy(),
    }


# ---------------------------------------------------------------------------
# Full feature extraction at a point
# ---------------------------------------------------------------------------


def extract_coherence_features(
    events: list[dict],
    lat: float,
    lon: float,
    radius_km: float = 300.0,
    time_window_days: float = 365.0,
    ref_epoch: float | None = None,
    grid_fields: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Extract all coherence field theory features for a location.

    Combines correlation length, b-value, IET distribution, AMR, and
    grid-based Helmholtz fields into a single feature vector.

    Parameters
    ----------
    events : list[dict]
        Earthquake catalog.
    lat, lon : float
        Location to analyse.
    radius_km : float
        Spatial analysis radius.
    time_window_days : float
        Temporal window.
    ref_epoch : float or None
        Reference time.
    grid_fields : dict or None
        Pre-computed grid fields from ``compute_seismic_coherence_field``.
        If None, only point-based features are computed (no grid lookups).

    Returns
    -------
    dict[str, float]
        Feature dictionary with all coherence diagnostics.
    """
    features: dict[str, float] = {}

    # --- Correlation length ---
    ell, n_events = compute_correlation_length(
        events, lat, lon, radius_km=radius_km,
        time_window_days=time_window_days, ref_epoch=ref_epoch,
    )
    features["ell"] = ell
    features["n_events"] = float(n_events)

    # --- Correlation length evolution (last 6 months of windows) ---
    ell_evo = compute_correlation_length_evolution(
        events, lat, lon, radius_km=radius_km,
        window_days=180.0, step_days=30.0, n_windows=12,
    )

    if len(ell_evo) >= 3:
        ell_vals = [d["ell_km"] for d in ell_evo if not math.isnan(d["ell_km"])]
        if len(ell_vals) >= 3:
            # Trend: linear fit of ell over time (km/month)
            x = np.arange(len(ell_vals), dtype=np.float64)
            coeffs = np.polyfit(x, ell_vals, 1)
            features["ell_trend"] = coeffs[0]  # km per step (30 days)

            # Acceleration: second derivative
            if len(ell_vals) >= 5:
                coeffs2 = np.polyfit(x, ell_vals, 2)
                features["ell_acceleration"] = coeffs2[0] * 2.0  # d^2 ell / dt^2
            else:
                features["ell_acceleration"] = float("nan")
        else:
            features["ell_trend"] = float("nan")
            features["ell_acceleration"] = float("nan")
    else:
        features["ell_trend"] = float("nan")
        features["ell_acceleration"] = float("nan")

    # --- b-value ---
    filtered = _filter_events(events, lat, lon, radius_km,
                              time_window_days, ref_epoch)
    mags = np.array([
        e["mag"] for e in filtered if e.get("mag") is not None
    ])
    b, mc, b_unc, n_b = compute_b_value(mags)
    features["b_value"] = b
    features["b_uncertainty"] = b_unc

    # b-value trend (compare recent half vs older half)
    if len(ell_evo) >= 4:
        b_vals = [d["b_value"] for d in ell_evo if not math.isnan(d["b_value"])]
        if len(b_vals) >= 4:
            mid = len(b_vals) // 2
            b_recent = np.mean(b_vals[mid:])
            b_older = np.mean(b_vals[:mid])
            features["b_trend"] = float(b_recent - b_older)
        else:
            features["b_trend"] = float("nan")
    else:
        features["b_trend"] = float("nan")

    # --- IET distribution ---
    iet, aic_lor, aic_exp, delta_aic = compute_iet_distribution(
        events, lat, lon, radius_km=radius_km,
        time_window_days=time_window_days, ref_epoch=ref_epoch,
    )
    features["delta_aic_iet"] = delta_aic

    # --- AMR ---
    t_c_amr, m_exp, r2_amr = compute_moment_acceleration(
        events, lat, lon, radius_km=radius_km, ref_epoch=ref_epoch,
    )
    features["amr_exponent"] = m_exp
    features["amr_r_squared"] = r2_amr

    # --- Divergence fit ---
    if len(ell_evo) >= 5:
        t_c_div, nu, r2_div = fit_divergence(ell_evo, method="power_law")
        features["t_c_estimate"] = t_c_div
        features["t_c_confidence"] = r2_div
        features["nu_exponent"] = nu

        # Time to criticality in days
        if not math.isnan(t_c_div) and ref_epoch is not None:
            features["days_to_criticality"] = (t_c_div - ref_epoch) / 86400.0
        elif not math.isnan(t_c_div) and ell_evo:
            last_epoch = ell_evo[-1]["epoch"]
            features["days_to_criticality"] = (t_c_div - last_epoch) / 86400.0
        else:
            features["days_to_criticality"] = float("nan")
    else:
        features["t_c_estimate"] = float("nan")
        features["t_c_confidence"] = float("nan")
        features["nu_exponent"] = float("nan")
        features["days_to_criticality"] = float("nan")

    # --- Grid-based features ---
    if grid_fields is not None:
        row, col = latlon_to_grid_cell(lat, lon)
        features["tau_local"] = float(grid_fields["tau"][row, col])
        features["grad_tau_local"] = float(grid_fields["grad_tau"][row, col])
        features["kappa_local"] = float(grid_fields["kappa_field"][row, col])
        features["S_over_Gamma"] = float(grid_fields["S_over_Gamma"][row, col])
        features["Da_local"] = float(grid_fields["Da"][row, col])
    else:
        features["tau_local"] = float("nan")
        features["grad_tau_local"] = float("nan")
        features["kappa_local"] = float("nan")
        features["S_over_Gamma"] = float("nan")
        features["Da_local"] = float("nan")

    # --- Basic seismicity statistics ---
    features["rate"] = len(filtered) / max(time_window_days, 1.0)

    # Rate acceleration: recent quarter vs previous three quarters
    if len(filtered) >= 10:
        times_all: list[float] = []
        for e in filtered:
            t_str = e.get("time", "")
            t = _parse_event_time(t_str) if isinstance(t_str, str) else float(t_str)
            if t > 0:
                times_all.append(t)
        times_all.sort()
        if times_all:
            t_range = times_all[-1] - times_all[0]
            if t_range > 0:
                t_quarter = times_all[-1] - t_range * 0.25
                n_recent = sum(1 for t in times_all if t >= t_quarter)
                n_older = sum(1 for t in times_all if t < t_quarter)
                # Normalise by time fraction
                rate_recent = n_recent / (t_range * 0.25) if t_range > 0 else 0
                rate_older = n_older / (t_range * 0.75) if t_range > 0 else 0
                features["rate_acceleration"] = (
                    rate_recent / max(rate_older, 1e-12)
                )
            else:
                features["rate_acceleration"] = float("nan")
        else:
            features["rate_acceleration"] = float("nan")
    else:
        features["rate_acceleration"] = float("nan")

    # Max magnitude in window
    if len(mags) > 0:
        features["max_mag"] = float(np.max(mags))
    else:
        features["max_mag"] = float("nan")

    # Moment rate
    features["moment_rate"] = sum(
        compute_seismic_moment(m) for m in mags
    ) / max(time_window_days, 1.0)

    # Spatial concentration (std of distances from centroid)
    if len(filtered) >= 5:
        f_lats = np.array([e["latitude"] for e in filtered])
        f_lons = np.array([e["longitude"] for e in filtered])
        c_lat = float(np.mean(f_lats))
        c_lon = float(np.mean(f_lons))
        dists = _haversine_km_arrays(
            f_lats, f_lons,
            np.full_like(f_lats, c_lat), np.full_like(f_lons, c_lon),
        )
        features["spatial_concentration"] = float(np.std(dists))
    else:
        features["spatial_concentration"] = float("nan")

    # Depth trend (shallowing = upward stress migration)
    depths = np.array([
        e["depth"] for e in filtered
        if e.get("depth") is not None and e["depth"] > 0
    ])
    if len(depths) >= 10:
        # Compare mean depth of recent vs older events
        mid = len(depths) // 2
        # Assume events are roughly time-ordered from _filter_events
        depth_recent = float(np.mean(depths[mid:]))
        depth_older = float(np.mean(depths[:mid]))
        features["depth_trend"] = depth_recent - depth_older  # negative = shallowing
    else:
        features["depth_trend"] = float("nan")

    return features


# ---------------------------------------------------------------------------
# Singularity test (5-condition earthquake criterion)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarthquakeSingularityResult:
    """Result of the 5-condition seismic criticality test.

    Each condition is derived from the Helmholtz PDE for seismic
    coherence and has a direct physical interpretation.
    """

    conditions_met: int
    ell_elevated: bool           # condition 1: correlation length elevated
    b_depressed: bool            # condition 2: b-value depressed
    iet_lorentzian: bool         # condition 3: IET Lorentzian preferred
    rate_accelerating: bool      # condition 4: seismicity accelerating
    loading_exceeds_healing: bool  # condition 5: S/Gamma > 1
    details: dict


def test_earthquake_singularity(
    features: dict[str, float],
    ell_background: float = ELL_BACKGROUND_KM,
    b_background: float = B_VALUE_BACKGROUND,
) -> EarthquakeSingularityResult:
    """Test the 5 conditions for approaching seismic criticality.

    1. ell > ell_background x 1.5
       Correlation length elevated beyond 1.5x background.  This
       indicates spatial correlations extending over larger distances,
       which is the direct signature of the coherence field approaching
       a critical point (ell -> infinity at rupture).

    2. b < b_background - 0.15
       b-value depressed below background.  Lower b means energy is
       concentrating in larger events -- the magnitude distribution
       shifts toward higher magnitudes as the fault system approaches
       failure (Schorlemmer et al. 2005).

    3. delta_AIC(IET) < -2
       Lorentzian preferred over exponential for inter-event times.
       Lorentzian IET arises from damped resonance in the Helmholtz
       PDE, indicating the coherence field has developed a characteristic
       frequency (coherent oscillation rather than random Poisson).

    4. rate_acceleration > 1.5
       Seismicity rate in the recent quarter exceeds the earlier period
       by 50%.  This is the accelerating moment release (AMR) signature
       predicted by the divergence of the source term S(x,t) as the
       system approaches criticality.

    5. S/Gamma > 1.0
       Tectonic loading exceeds fault healing.  When the source term
       (plate-driven stress) exceeds the damping term (frictional
       healing / aseismic creep), the fault cannot dissipate stress
       fast enough, leading to stress accumulation and eventual rupture.

    Parameters
    ----------
    features : dict[str, float]
        Output of ``extract_coherence_features``.
    ell_background : float
        Background correlation length (km).
    b_background : float
        Background b-value.

    Returns
    -------
    EarthquakeSingularityResult
    """
    ell = features.get("ell", float("nan"))
    b = features.get("b_value", float("nan"))
    delta_aic = features.get("delta_aic_iet", float("nan"))
    rate_accel = features.get("rate_acceleration", float("nan"))
    s_gamma = features.get("S_over_Gamma", float("nan"))

    # Condition 1: correlation length elevated
    c1 = (
        not math.isnan(ell)
        and ell > ell_background * ELL_FACTOR_THRESHOLD
    )

    # Condition 2: b-value depressed
    c2 = (
        not math.isnan(b)
        and b < b_background - B_VALUE_DROP_THRESHOLD
    )

    # Condition 3: IET Lorentzian preferred
    c3 = (
        not math.isnan(delta_aic)
        and delta_aic < DELTA_AIC_THRESHOLD
    )

    # Condition 4: rate accelerating
    c4 = (
        not math.isnan(rate_accel)
        and rate_accel > RATE_ACCEL_THRESHOLD
    )

    # Condition 5: loading exceeds healing
    c5 = (
        not math.isnan(s_gamma)
        and s_gamma > S_OVER_GAMMA_THRESHOLD
    )

    conditions_met = int(c1) + int(c2) + int(c3) + int(c4) + int(c5)

    details = {
        "ell_km": ell,
        "ell_threshold_km": ell_background * ELL_FACTOR_THRESHOLD,
        "b_value": b,
        "b_threshold": b_background - B_VALUE_DROP_THRESHOLD,
        "delta_aic_iet": delta_aic,
        "delta_aic_threshold": DELTA_AIC_THRESHOLD,
        "rate_acceleration": rate_accel,
        "rate_accel_threshold": RATE_ACCEL_THRESHOLD,
        "S_over_Gamma": s_gamma,
        "S_over_Gamma_threshold": S_OVER_GAMMA_THRESHOLD,
        "days_to_criticality": features.get("days_to_criticality", float("nan")),
        "nu_exponent": features.get("nu_exponent", float("nan")),
    }

    return EarthquakeSingularityResult(
        conditions_met=conditions_met,
        ell_elevated=c1,
        b_depressed=c2,
        iet_lorentzian=c3,
        rate_accelerating=c4,
        loading_exceeds_healing=c5,
        details=details,
    )


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single location
# ---------------------------------------------------------------------------


def analyse_location(
    events: list[dict],
    lat: float,
    lon: float,
    radius_km: float = 300.0,
    time_window_days: float = 365.0,
    ref_epoch: float | None = None,
) -> tuple[dict[str, float], EarthquakeSingularityResult]:
    """Run the complete coherence analysis pipeline for one location.

    This is the main entry point for earthquake coherence analysis.
    It computes all features and runs the 5-condition singularity test.

    Parameters
    ----------
    events : list[dict]
        Full earthquake catalog.
    lat, lon : float
        Location to analyse.
    radius_km : float
        Spatial analysis radius.
    time_window_days : float
        Temporal window for feature computation.
    ref_epoch : float or None
        Reference time.

    Returns
    -------
    tuple of (features, singularity_result)
    """
    features = extract_coherence_features(
        events, lat, lon,
        radius_km=radius_km,
        time_window_days=time_window_days,
        ref_epoch=ref_epoch,
    )

    singularity = test_earthquake_singularity(features)

    return features, singularity
