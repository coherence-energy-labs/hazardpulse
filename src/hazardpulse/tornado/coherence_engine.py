# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational tornado warning system.
# It does NOT replace official NWS tornado warnings.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Coherence field theory engine for tornado prediction.

Implements the Helmholtz PDE solver and all derived coherence quantities
on the 80 km CONUS grid.  Pure NumPy -- no external ML or physics
libraries required.

Physics summary
----------------
The coherence field tau(x, t) satisfies the screened Poisson (Helmholtz)
equation derived from the Equation of One (S_One):

    D nabla^2 tau  -  Gamma tau  +  S  =  0

where:
  - S(x,t)     = source term (CAPE, SRH, shear, reflectivity)
  - Gamma(x,t) = damping (CIN, Td depression)
  - D(x,t)     = diffusivity (storm-relative flow)
  - kappa^2    = Gamma / D  (screening wavenumber — DERIVED reporting quantity;
                 the solver consumes Gamma directly, never kappa: FVCS W-2)

From tau we derive:
  - grad(tau)    -- coherence gradient (front strength)
  - curl(tau)    -- coherence rotation
  - torsion      -- shear x curl(tau)  (Form 16 coupling)
  - alignment    -- vorticity-coherence coupling
  - Damkohler Da -- reaction-diffusion regime
  - E_coh        -- coherence energy from KL divergence
  - singularity  -- 5-condition singularity test
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hazardpulse.data.hrrr import (
    DX_KM,
    DY_KM,
    HRRR_N_LAT,
    HRRR_N_LON,
    latlon_to_hrrr_cell,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CAPE_SCALE: float = 2000.0
CIN_SCALE: float = 200.0
BASE_D: float = 1.0
HELMHOLTZ_ITERS: int = 200
GAUSS_SIGMA_CELLS: float = 3.0

# Mesocyclone rotation threshold (s^-1)
ROTATION_THRESHOLD: float = 0.003


# ---------------------------------------------------------------------------
# Helmholtz PDE solver
# ---------------------------------------------------------------------------


def solve_helmholtz_2d(
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
    n_iter: int = HELMHOLTZ_ITERS,
    omega: float = 0.7,
) -> np.ndarray:
    """Solve ``D nabla^2 tau - gamma tau + S = 0`` on the CONUS grid.

    Tornado-grid wrapper around the shared
    ``hazardpulse.coherence.tau_c_solver.solve_helmholtz_2d``.
    Uses float32 (large 34x63 CONUS grid; speed over precision).
    ``gamma`` is the damping rate Gamma(x) itself — kappa = sqrt(gamma/D)
    is a derived reporting quantity, never the solver input (FVCS W-2).
    """
    from hazardpulse.coherence.tau_c_solver import (
        solve_helmholtz_2d as _shared_solver,
    )
    return _shared_solver(
        source, gamma, dx,
        D=D, n_iter=n_iter, omega=omega, dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Spatial derivative operators
# ---------------------------------------------------------------------------


def _gradient_2d(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient (d/dy, d/dx) using central differences."""
    ny, nx = field.shape
    grad_y = np.zeros_like(field, dtype=np.float32)
    grad_x = np.zeros_like(field, dtype=np.float32)

    grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)
    grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)

    # Forward/backward at edges
    grad_y[0, :] = (field[1, :] - field[0, :]) / dy
    grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dy
    grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
    grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx

    return grad_y, grad_x


def compute_gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """Compute |nabla field| using central differences."""
    gy, gx = _gradient_2d(field)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def compute_curl_2d(field: np.ndarray) -> np.ndarray:
    """Compute scalar curl of the gradient field.

    Returns ``d(gx)/dy - d(gy)/dx`` on the grid.
    """
    grad_y, grad_x = _gradient_2d(field)

    dgx_dy = np.zeros_like(field, dtype=np.float32)
    dgx_dy[1:-1, :] = (grad_x[2:, :] - grad_x[:-2, :]) / 2.0
    dgx_dy[0, :] = grad_x[1, :] - grad_x[0, :]
    dgx_dy[-1, :] = grad_x[-1, :] - grad_x[-2, :]

    dgy_dx = np.zeros_like(field, dtype=np.float32)
    dgy_dx[:, 1:-1] = (grad_y[:, 2:] - grad_y[:, :-2]) / 2.0
    dgy_dx[:, 0] = grad_y[:, 1] - grad_y[:, 0]
    dgy_dx[:, -1] = grad_y[:, -1] - grad_y[:, -2]

    return (dgx_dy - dgy_dx).astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian smoothing (control kernel)
# ---------------------------------------------------------------------------


def gaussian_smooth_2d(
    source: np.ndarray,
    sigma_cells: float = GAUSS_SIGMA_CELLS,
) -> np.ndarray:
    """Gaussian smoothing of a 2-D field (control kernel for Helmholtz)."""
    ny, nx = source.shape
    result = np.zeros_like(source, dtype=np.float32)
    radius = int(math.ceil(3.0 * sigma_cells))
    sigma2 = sigma_cells ** 2

    # Build kernel
    ks = 2 * radius + 1
    ii = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (ii[:, None] ** 2 + ii[None, :] ** 2) / sigma2)
    kernel /= kernel.sum()

    for i in range(ny):
        for j in range(nx):
            val = np.float32(0.0)
            wt = np.float32(0.0)
            for di in range(-radius, radius + 1):
                ii_idx = i + di
                if ii_idx < 0 or ii_idx >= ny:
                    continue
                for dj in range(-radius, radius + 1):
                    jj_idx = j + dj
                    if jj_idx < 0 or jj_idx >= nx:
                        continue
                    w = kernel[di + radius, dj + radius]
                    val += w * source[ii_idx, jj_idx]
                    wt += w
            if wt > 1e-12:
                result[i, j] = val / wt
    return result


# ---------------------------------------------------------------------------
# Derived HRRR parameters
# ---------------------------------------------------------------------------


def compute_derived_hrrr(hrrr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute derived parameters from raw HRRR fields.

    Parameters
    ----------
    hrrr : dict[str, ndarray]
        Raw HRRR variable grids keyed by short name.

    Returns
    -------
    dict[str, ndarray]
        Derived parameter grids on the same ``(34, 63)`` grid.
    """
    derived: dict[str, np.ndarray] = {}

    # Bulk shear magnitudes
    derived["shear_01"] = np.sqrt(
        hrrr["ushear_01"] ** 2 + hrrr["vshear_01"] ** 2
    ).astype(np.float32)

    derived["shear_06"] = np.sqrt(
        hrrr["ushear_06"] ** 2 + hrrr["vshear_06"] ** 2
    ).astype(np.float32)

    # Storm motion speed
    derived["storm_speed"] = np.sqrt(
        hrrr["ustorm"] ** 2 + hrrr["vstorm"] ** 2
    ).astype(np.float32)

    # Td depression and LCL estimate (Bolton 1980)
    derived["td_depression"] = (hrrr["t2m"] - hrrr["td2m"]).astype(np.float32)
    derived["lcl_est"] = (125.0 * derived["td_depression"]).astype(np.float32)

    # 0-500 m SRH estimate
    ratio = np.where(
        derived["lcl_est"] < 800,
        0.65,
        np.where(derived["lcl_est"] < 1200, 0.55, 0.45),
    ).astype(np.float32)
    derived["srh_05_est"] = (hrrr["srh_01"] * ratio).astype(np.float32)

    # Effective-layer STP (SPC-style clipping)
    cape = np.maximum(hrrr["mlcape"], 0).astype(np.float32)
    lcl_term = np.clip((2000 - derived["lcl_est"]) / 1000.0, 0.0, 1.0)
    shear_term = np.clip(derived["shear_06"] / 20.0, 0.0, 1.5)
    shear_term = np.where(derived["shear_06"] < 12.5, 0.0, shear_term)
    cin_term = np.clip((200 + hrrr["mlcin"]) / 150.0, 0.0, 1.0)
    srh_term = np.maximum(hrrr["srh_01"], 0) / 150.0
    derived["stp_eff"] = (
        (cape / 1500.0) * lcl_term * srh_term * shear_term * cin_term
    ).astype(np.float32)

    # RFD warmth proxy
    mid_rh = np.maximum(0, 1.0 - derived["td_depression"] / 20.0).astype(
        np.float32
    )
    lapse_proxy = np.float32(6.5)
    dcape_proxy = (lapse_proxy * (1 - mid_rh) * 100).astype(np.float32)
    lcl_warm = np.maximum(0, (2000 - derived["lcl_est"]) / 1000.0).astype(
        np.float32
    )
    derived["rfd_warmth"] = (
        lcl_warm * mid_rh / (1 + dcape_proxy / 500.0)
    ).astype(np.float32)

    # Streamwise vorticity estimate
    # omega_s = SRH_01 / (|V_sr| * depth), where depth = 1000 m (0-1 km layer)
    # Use actual 0-1 km shear as storm-relative wind proxy
    v_sr = np.maximum(derived["shear_01"], 5.0).astype(np.float32)  # m/s
    derived["streamwise_vort"] = (
        np.abs(hrrr["srh_01"]) / (v_sr * 1000.0)  # units: 1/s
    ).astype(np.float32)

    return derived


# ---------------------------------------------------------------------------
# Coherence field computation
# ---------------------------------------------------------------------------


def compute_coherence_fields(
    atm_grid: dict[str, np.ndarray],
    month: int = 5,
) -> dict[str, np.ndarray]:
    """Solve the Helmholtz PDE on atmospheric state and return all
    coherence-derived quantities.

    Parameters
    ----------
    atm_grid : dict[str, ndarray]
        Raw HRRR grids (keys matching ``HRRR_VARS``).
    month : int
        Calendar month (used for seasonal modulation, default May).

    Returns
    -------
    dict
        tau, grad_tau, torsion, alignment, S_field, Gamma_field,
        S_over_Gamma, Da, E_coh, singularity_count.
    """
    derived = compute_derived_hrrr(atm_grid)

    # Source S(x,t)
    mlcape = np.maximum(atm_grid["mlcape"], 0).astype(np.float32)
    srh_01 = atm_grid["srh_01"].astype(np.float32)
    shear_06 = derived["shear_06"]
    refc = np.maximum(atm_grid["refc"], 0).astype(np.float32)

    S_field = (
        (mlcape / CAPE_SCALE)
        + 0.3 * (np.abs(srh_01) / 200.0)
        + 0.2 * (shear_06 / 25.0)
        + 0.1 * np.minimum(refc / 40.0, 2.0)
    ).astype(np.float32)

    # Damping Gamma(x,t)
    mlcin = atm_grid["mlcin"].astype(np.float32)
    td_dep = derived["td_depression"]
    Gamma_field = (
        (np.abs(mlcin) / CIN_SCALE + 0.1)
        + 0.2 * (np.maximum(td_dep, 0) / 15.0)
        + 0.05
    ).astype(np.float32)

    # Spatially varying diffusivity
    storm_speed = derived["storm_speed"]
    D_field = (1.0 + 0.3 * (storm_speed / 20.0)).astype(np.float32)

    # Solve Helmholtz PDE: D nabla^2 tau - Gamma tau + S = 0. Gamma goes in
    # DIRECTLY — the old kappa = sqrt(Gamma/D) input understated screening by a
    # factor of D on this non-unit-D grid (FVCS W-2), letting hazard influence
    # over-diffuse past the model's intended coherence length sqrt(D/Gamma).
    tau = solve_helmholtz_2d(S_field, Gamma_field, dx=1.0, D=D_field)

    # Spatial derivatives
    grad_y, grad_x = _gradient_2d(tau)
    grad_tau = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)

    # Torsion: shear x curl(tau) -- Form 16 coupling
    curl_tau = compute_curl_2d(tau)
    torsion = (shear_06 * curl_tau / 25.0).astype(np.float32)

    # Alignment: shear-coherence coupling (Form 16)
    grad_mag_safe = np.maximum(grad_tau, 1e-6)
    e_x = grad_x / grad_mag_safe
    e_y = grad_y / grad_mag_safe
    shear_x = atm_grid["ushear_01"].astype(np.float32)
    shear_y = atm_grid["vshear_01"].astype(np.float32)
    alignment = (shear_x * e_x + shear_y * e_y).astype(np.float32)

    # S / Gamma ratio
    S_over_Gamma = (S_field / np.maximum(Gamma_field, 0.01)).astype(
        np.float32
    )

    # Damkohler number: Da = Gamma * L^2 / (D * v)
    Da = (
        Gamma_field * (DX_KM ** 2) / np.maximum(D_field * 100.0, 1e-6)
    ).astype(np.float32)

    # Coherence energy via KL divergence of ingredient partition
    denom_kl = (
        np.maximum(mlcape, 0)
        + np.maximum(np.abs(mlcin), 50)
        + np.maximum(shear_06, 5)
        + 1e-6
    ).astype(np.float32)
    cape_frac = (np.maximum(mlcape, 0) / denom_kl + 0.01).astype(np.float32)
    srh_frac = (np.maximum(np.abs(srh_01), 0) / denom_kl + 0.01).astype(
        np.float32
    )
    shear_frac = (np.maximum(shear_06, 0) / denom_kl + 0.01).astype(
        np.float32
    )
    p_sum = cape_frac + srh_frac + shear_frac
    p_cape = cape_frac / p_sum
    p_srh = srh_frac / p_sum
    p_shear = shear_frac / p_sum
    q = np.float32(1.0 / 3.0)
    D_KL = (
        p_cape * np.log(np.maximum(p_cape / q, 1e-12))
        + p_srh * np.log(np.maximum(p_srh / q, 1e-12))
        + p_shear * np.log(np.maximum(p_shear / q, 1e-12))
    ).astype(np.float32)
    T_sfc = atm_grid["t2m"].astype(np.float32)
    E_coh = (T_sfc * np.maximum(D_KL, 0.0) / 300.0).astype(np.float32)

    # 5-condition singularity test using absolute thresholds
    # Thresholds based on observed tornadic environments; require calibration
    # against a labelled training set for production use.
    cond1 = (S_over_Gamma > 1.0).astype(np.float32)       # source exceeds damping
    cond2 = (grad_tau > 0.5).astype(np.float32)            # steep coherence gradient
    cond3 = (np.abs(torsion) > 0.1).astype(np.float32)     # significant rotational coupling
    cond4 = (alignment > 0).astype(np.float32)             # vorticity aligned with gradient
    cond5 = (Da > 10.0).astype(np.float32)                 # decay-dominated regime
    singularity_count = (cond1 + cond2 + cond3 + cond4 + cond5).astype(
        np.float32
    )

    return {
        "tau": tau,
        "grad_tau": grad_tau,
        "torsion": torsion,
        "alignment": alignment,
        "S_field": S_field,
        "Gamma_field": Gamma_field,
        "S_over_Gamma": S_over_Gamma,
        "Da": Da,
        "E_coh": E_coh,
        "singularity_count": singularity_count,
        # Keep derived for downstream feature extraction
        "_derived": derived,
    }


def compute_gaussian_fields(
    atm_grid: dict[str, np.ndarray],
    month: int = 5,
) -> dict[str, np.ndarray]:
    """Gaussian-smoothed control fields (mirrors coherence fields).

    Uses a Gaussian kernel instead of the Helmholtz PDE to test whether
    the PDE structure adds skill beyond simple smoothing.
    """
    derived = compute_derived_hrrr(atm_grid)

    mlcape = np.maximum(atm_grid["mlcape"], 0).astype(np.float32)
    srh_01 = atm_grid["srh_01"].astype(np.float32)
    shear_06 = derived["shear_06"]
    refc = np.maximum(atm_grid["refc"], 0).astype(np.float32)

    S_field = (
        (mlcape / CAPE_SCALE)
        + 0.3 * (np.abs(srh_01) / 200.0)
        + 0.2 * (shear_06 / 25.0)
        + 0.1 * np.minimum(refc / 40.0, 2.0)
    ).astype(np.float32)

    mlcin = atm_grid["mlcin"].astype(np.float32)
    td_dep = derived["td_depression"]
    Gamma_field = (
        (np.abs(mlcin) / CIN_SCALE + 0.1)
        + 0.2 * (np.maximum(td_dep, 0) / 15.0)
        + 0.05
    ).astype(np.float32)

    storm_speed = derived["storm_speed"]
    D_field = (1.0 + 0.3 * (storm_speed / 20.0)).astype(np.float32)

    # Gaussian instead of PDE -- smooth S/Gamma (same effective source
    # as the Helmholtz solver) so the control has access to the same
    # information content.
    effective_source = S_field / (Gamma_field + 0.01)
    tau = gaussian_smooth_2d(effective_source, GAUSS_SIGMA_CELLS)

    grad_y, grad_x = _gradient_2d(tau)
    grad_tau = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)
    curl_tau = compute_curl_2d(tau)
    torsion = (shear_06 * curl_tau / 25.0).astype(np.float32)

    grad_mag_safe = np.maximum(grad_tau, 1e-6)
    e_x = grad_x / grad_mag_safe
    e_y = grad_y / grad_mag_safe
    shear_x = atm_grid["ushear_01"].astype(np.float32)
    shear_y = atm_grid["vshear_01"].astype(np.float32)
    alignment = (shear_x * e_x + shear_y * e_y).astype(np.float32)

    S_over_Gamma = (S_field / np.maximum(Gamma_field, 0.01)).astype(
        np.float32
    )
    Da = (
        Gamma_field * (DX_KM ** 2) / np.maximum(D_field * 100.0, 1e-6)
    ).astype(np.float32)

    denom_kl = (
        np.maximum(mlcape, 0)
        + np.maximum(np.abs(mlcin), 50)
        + np.maximum(shear_06, 5)
        + 1e-6
    ).astype(np.float32)
    cape_frac = (np.maximum(mlcape, 0) / denom_kl + 0.01).astype(np.float32)
    srh_frac = (np.maximum(np.abs(srh_01), 0) / denom_kl + 0.01).astype(
        np.float32
    )
    shear_frac = (np.maximum(shear_06, 0) / denom_kl + 0.01).astype(
        np.float32
    )
    p_sum = cape_frac + srh_frac + shear_frac
    p_cape = cape_frac / p_sum
    p_srh = srh_frac / p_sum
    p_shear = shear_frac / p_sum
    q = np.float32(1.0 / 3.0)
    D_KL = (
        p_cape * np.log(np.maximum(p_cape / q, 1e-12))
        + p_srh * np.log(np.maximum(p_srh / q, 1e-12))
        + p_shear * np.log(np.maximum(p_shear / q, 1e-12))
    ).astype(np.float32)
    T_sfc = atm_grid["t2m"].astype(np.float32)
    E_coh = (T_sfc * np.maximum(D_KL, 0.0) / 300.0).astype(np.float32)

    # Same absolute thresholds as Helmholtz for consistent comparison
    cond1 = (S_over_Gamma > 1.0).astype(np.float32)
    cond2 = (grad_tau > 0.5).astype(np.float32)
    cond3 = (np.abs(torsion) > 0.1).astype(np.float32)
    cond4 = (alignment > 0).astype(np.float32)
    cond5 = (Da > 10.0).astype(np.float32)
    singularity_count = (cond1 + cond2 + cond3 + cond4 + cond5).astype(
        np.float32
    )

    return {
        "tau": tau,
        "grad_tau": grad_tau,
        "torsion": torsion,
        "alignment": alignment,
        "S_field": S_field,
        "Gamma_field": Gamma_field,
        "S_over_Gamma": S_over_Gamma,
        "Da": Da,
        "E_coh": E_coh,
        "singularity_count": singularity_count,
        "_derived": derived,
    }


# ---------------------------------------------------------------------------
# Alignment term from Form 16
# ---------------------------------------------------------------------------


def compute_alignment(
    tau: np.ndarray,
    srh: np.ndarray,
    shear: np.ndarray,
) -> np.ndarray:
    """Alignment term: vorticity-coherence coupling (Form 16).

    Computes ``srh * |nabla tau| / (shear + eps)`` as a measure of how
    coherently the storm-relative helicity aligns with the coherence
    gradient.
    """
    grad_mag = compute_gradient_magnitude(tau)
    return (
        srh * grad_mag / (np.maximum(shear, 5.0) + 1e-6)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------------


def extract_coherence_at_point(
    coherence_fields: dict[str, np.ndarray],
    lat: float,
    lon: float,
) -> dict[str, float]:
    """Extract all coherence features at a specific lat/lon point.

    Parameters
    ----------
    coherence_fields : dict
        Output of ``compute_coherence_fields``.
    lat, lon : float
        Geographic coordinates.

    Returns
    -------
    dict[str, float]
        Coherence diagnostics at the requested point.
    """
    i, j = latlon_to_hrrr_cell(lat, lon)
    result: dict[str, float] = {}
    for key in (
        "tau",
        "grad_tau",
        "torsion",
        "alignment",
        "S_field",
        "Gamma_field",
        "S_over_Gamma",
        "Da",
        "E_coh",
        "singularity_count",
    ):
        arr = coherence_fields.get(key)
        if arr is not None:
            result[key] = float(arr[i, j])
        else:
            result[key] = 0.0
    return result


# ---------------------------------------------------------------------------
# RFD warmth proxy
# ---------------------------------------------------------------------------


def compute_rfd_warmth(
    td_depression: np.ndarray,
    lcl_est: np.ndarray,
) -> np.ndarray:
    """Rear-flank downdraft warmth proxy.

    Warm RFDs favour tornado genesis by maintaining near-surface
    vorticity through positive buoyancy in the outflow.
    """
    mid_rh = np.maximum(0, 1.0 - td_depression / 20.0).astype(np.float32)
    lapse_proxy = np.float32(6.5)
    dcape_proxy = (lapse_proxy * (1 - mid_rh) * 100).astype(np.float32)
    lcl_warm = np.maximum(0, (2000 - lcl_est) / 1000.0).astype(np.float32)
    return (lcl_warm * mid_rh / (1 + dcape_proxy / 500.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# SRH 0-500 m estimation
# ---------------------------------------------------------------------------


def estimate_srh_05(
    srh_01: np.ndarray,
    lcl_est: np.ndarray,
) -> np.ndarray:
    """Estimate 0-500 m SRH from 0-1 km SRH using LCL-based partitioning."""
    ratio = np.where(
        lcl_est < 800,
        0.65,
        np.where(lcl_est < 1200, 0.55, 0.45),
    ).astype(np.float32)
    return (srh_01 * ratio).astype(np.float32)


# ---------------------------------------------------------------------------
# Streamwise vorticity
# ---------------------------------------------------------------------------


def compute_streamwise_vorticity(
    srh_01: np.ndarray,
    shear_01: np.ndarray,
) -> np.ndarray:
    """Streamwise vorticity estimate from SRH and 0-1 km shear.

    omega_s = SRH_01 / (|V_sr| * 1000m)
    Units: (m^2/s^2) / (m/s * m) = 1/s
    """
    v_sr = np.maximum(shear_01, 5.0).astype(np.float32)
    return (np.abs(srh_01) / (v_sr * 1000.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# STP effective
# ---------------------------------------------------------------------------


def compute_stp_effective(
    mlcape: np.ndarray,
    mlcin: np.ndarray,
    srh_01: np.ndarray,
    shear_06: np.ndarray,
    lcl_est: np.ndarray,
) -> np.ndarray:
    """Significant Tornado Parameter (effective layer, SPC-style)."""
    cape = np.maximum(mlcape, 0).astype(np.float32)
    lcl_term = np.clip((2000 - lcl_est) / 1000.0, 0.0, 1.0)
    shear_term = np.clip(shear_06 / 20.0, 0.0, 1.5)
    shear_term = np.where(shear_06 < 12.5, 0.0, shear_term)
    cin_term = np.clip((200 + mlcin) / 150.0, 0.0, 1.0)
    srh_term = np.maximum(srh_01, 0) / 150.0
    return (
        (cape / 1500.0) * lcl_term * srh_term * shear_term * cin_term
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Damkohler number
# ---------------------------------------------------------------------------


def compute_damkohler(
    Gamma: np.ndarray,
    D: np.ndarray,
    L_km: float = DX_KM,
) -> np.ndarray:
    """Damkohler number Da = Gamma * L^2 / (D * v_scale).

    Da >> 1 : reaction-dominated (coherence can build locally)
    Da << 1 : transport-dominated (coherence is advected away)
    """
    return (
        Gamma * (L_km ** 2) / np.maximum(D * 100.0, 1e-6)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 5-condition singularity test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingularityResult:
    """Result of the 5-condition tornado singularity test."""

    count: int
    s_over_gamma: bool
    high_gradient: bool
    high_torsion: bool
    positive_alignment: bool
    high_damkohler: bool


def test_singularity_at_point(
    coherence_fields: dict[str, np.ndarray],
    lat: float,
    lon: float,
) -> SingularityResult:
    """Evaluate the 5-condition singularity test at a single point.

    The five conditions for a coherence singularity (potential tornado),
    using absolute physically-motivated thresholds (require calibration
    against a labelled training set for production use):
      1. S / Gamma > 1.0  (source exceeds damping)
      2. |nabla tau| > 0.5  (steep coherence gradient)
      3. |torsion| > 0.1  (significant rotational coupling)
      4. alignment > 0  (shear aligned with coherence gradient)
      5. Da > 10.0  (decay-dominated regime)
    """
    i, j = latlon_to_hrrr_cell(lat, lon)

    sg = float(coherence_fields["S_over_Gamma"][i, j])
    gt_val = float(coherence_fields["grad_tau"][i, j])
    tors_val = float(coherence_fields["torsion"][i, j])
    align_val = float(coherence_fields["alignment"][i, j])
    da_val = float(coherence_fields["Da"][i, j])

    c1 = sg > 1.0
    c2 = gt_val > 0.5
    c3 = abs(tors_val) > 0.1
    c4 = align_val > 0
    c5 = da_val > 10.0

    return SingularityResult(
        count=int(c1) + int(c2) + int(c3) + int(c4) + int(c5),
        s_over_gamma=c1,
        high_gradient=c2,
        high_torsion=c3,
        positive_alignment=c4,
        high_damkohler=c5,
    )
