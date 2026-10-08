"""The coherence equation as the vortex's balanced response to convective heating (hurricane program H8).

In a balanced vortex, heating spins the winds up efficiently only where it is held by the vortex's own inertial
stability: inside a local Rossby radius of deformation ``ell(r) = c / I(r)`` (Schubert and Hack 1982, Vigh and
Schubert 2009). ``I(r)`` is the inertial frequency of the storm's wind profile and ``c`` the speed of the deep
(first baroclinic) gravity wave. Inside the radius of maximum wind ``I`` is large and ``ell`` short, so heating
there is retained; far outside, ``ell`` approaches ``c / f`` (~1,000 km) and the energy radiates away.

That is the coherence equation of ``hazardpulse.coherence.tau_c_solver`` in its Rossby-adjustment form, with a
coherence length that varies in space:

    lap(tau) - tau / ell(x)^2 + S(x) / ell(x)^2 = 0

``S`` is the convective heating proxy -- the cold-cloud excess of the GMGSI longwave counts above the program's
COLD threshold -- and ``tau`` the balanced state it adjusts to: ``S`` smoothed by a kernel of width ``ell`` that
integrates to one. Where ``ell`` is short (a strong core) ``tau ~ S``, the heating is retained in place; where it
is long, ``tau`` is spread thin -- dispersed as gravity waves. (The plain-source form ``... + S = 0`` grows like
``ell^2`` and would read the weakest vortex as the most efficient: the tests pin the right one.) It is solved to a
certified residual on a storm-centred, isotropic 8 km grid. The features are what the core retains: the balanced
heating inside the radius of maximum wind, the heating there, and their ratio.

Every constant is declared below and fixed before any feature met an outcome (docs/HURRICANE_RI_V9_PROGRAM.md,
amendment 8).
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from hazardpulse.coherence.tau_c_solver import solve_helmholtz_2d_certified
from hazardpulse.hurricane import ir_features as ir

GRAVITY_WAVE_SPEED_MS = 50.0       # first baroclinic mode of a deep tropical troposphere
RANKINE_DECAY = 0.5                # modified Rankine: v = Vmax (R / r)^0.5 outside the radius of maximum wind
GRID_KM = 8.0                      # the GMGSI pixel (0.072 deg)
HALF_WIDTH_KM = 400.0              # the crop is +-4 deg; +-400 km stays inside it to 40 deg latitude
INNER_KM = 300.0                   # the heating budget: the program's IR features stop at 300 km
MIN_COVERAGE = 0.5                 # the IR features' rule: under half the pixels valid -> NaN
KT = 0.514444                      # m/s per knot
NM_KM = 1.852                      # km per nautical mile
OMEGA = 7.2921e-5                  # earth's rotation, 1/s
SOLVER_TOL = 1e-8

H8_STATIC = ("h8_core_balanced", "h8_core_heating", "h8_core_retention")
H8_NAMES = H8_STATIC + ("h8_core_balanced_d6",)


def coriolis(lat_deg: float) -> float:
    return 2.0 * OMEGA * math.sin(math.radians(abs(lat_deg)))


def inertial_frequency(r_km: np.ndarray, vmax_kt: float, rmw_km: float, lat_deg: float) -> np.ndarray:
    """``I(r)`` (1/s) of a modified Rankine vortex: solid body inside the radius of maximum wind ``R``,
    ``v = Vmax (R/r)^a`` outside. I^2 = (f + 2v/r)(f + zeta), zeta = (1/r) d(rv)/dr."""
    f = coriolis(lat_deg)
    vmax = max(float(vmax_kt), 0.0) * KT
    R = max(float(rmw_km), GRID_KM) * 1000.0
    r = np.maximum(np.asarray(r_km, np.float64), 1e-3) * 1000.0
    a = RANKINE_DECAY
    inside = r <= R
    v = np.where(inside, vmax * r / R, vmax * (R / r) ** a)
    zeta = np.where(inside, 2.0 * vmax / R, (1.0 - a) * v / r)
    i2 = (f + 2.0 * v / r) * (f + zeta)
    return np.sqrt(np.maximum(i2, f * f))


def coherence_length_km(r_km: np.ndarray, vmax_kt: float, rmw_km: float, lat_deg: float) -> np.ndarray:
    """The local Rossby radius ``c / I(r)`` in km -- the coherence length of the balanced response."""
    return GRAVITY_WAVE_SPEED_MS / inertial_frequency(r_km, vmax_kt, rmw_km, lat_deg) / 1000.0


def storm_grid(counts: np.ndarray, lat: np.ndarray, lon: np.ndarray, centre: tuple[float, float]):
    """The crop's counts on a storm-centred isotropic grid (x east, y north, km): the mean of the valid pixels
    in each cell; ``(grid, has, r_km)`` -- NaN and False where no valid pixel falls."""
    dist, bearing = ir.distances_km(lat, lon, centre)
    x = dist * np.sin(bearing)
    y = dist * np.cos(bearing)
    n = int(round(2 * HALF_WIDTH_KM / GRID_KM)) + 1
    ix = np.rint((x + HALF_WIDTH_KM) / GRID_KM).astype(np.int64)
    iy = np.rint((y + HALF_WIDTH_KM) / GRID_KM).astype(np.int64)
    raw = np.asarray(counts)
    c = raw.astype(np.float64)
    # ir.MISSING (255) marks no data -- and 255 is also the coldest count, so a missing pixel read as data
    # would be the strongest heating in the crop
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & np.isfinite(c) & (raw != ir.MISSING)
    flat = iy[ok] * n + ix[ok]
    s = np.bincount(flat, weights=c[ok], minlength=n * n)
    k = np.bincount(flat, minlength=n * n)
    grid = np.where(k > 0, s / np.maximum(k, 1), np.nan).reshape(n, n)
    axis = np.arange(n) * GRID_KM - HALF_WIDTH_KM
    gx, gy = np.meshgrid(axis, axis)
    return grid, (k > 0).reshape(n, n), np.hypot(gx, gy)


def static_response(crop: Mapping, vmax_kt: float, rmw_nm: float, lat_deg: float) -> dict[str, float]:
    """H8's static features for one image (NaN when the crop, the vortex or the coverage is missing)."""
    out = {k: float("nan") for k in H8_STATIC}
    if crop is None or not np.isfinite(vmax_kt) or not np.isfinite(rmw_nm) or rmw_nm <= 0:
        return out
    grid, has, r = storm_grid(crop["counts"], crop["lat"], crop["lon"], crop["centre"])
    inner = r <= INNER_KM
    if has[inner].mean() < MIN_COVERAGE:
        return out
    S = np.where(has, np.maximum(grid - ir.COLD, 0.0), 0.0)
    rmw_km = max(float(rmw_nm) * NM_KM, GRID_KM)
    ell = coherence_length_km(r, vmax_kt, rmw_km, lat_deg)
    tau, cert = solve_helmholtz_2d_certified(S / ell ** 2, 1.0 / ell ** 2, dx=GRID_KM, D=1.0, tol=SOLVER_TOL)
    core = r <= rmw_km
    balanced, heating = float(tau[core].mean()), float(S[core].mean())
    out["h8_core_balanced"] = balanced
    out["h8_core_heating"] = heating
    out["h8_core_retention"] = balanced / (heating + 1.0)
    return out


def features(now: Mapping | None, before: Mapping | None, vmax_kt: float, rmw_nm: float,
             lat_deg: float) -> dict[str, float]:
    """H8 for one cycle: the static features of the newer image (t + 2 h) and the 6-hour change of the
    balanced core heating (t + 2 h minus t - 4 h), with the vortex at the analysis time -- the IR features' own pairing."""
    a = static_response(now, vmax_kt, rmw_nm, lat_deg)
    b = static_response(before, vmax_kt, rmw_nm, lat_deg)
    out = dict(a)
    out["h8_core_balanced_d6"] = a["h8_core_balanced"] - b["h8_core_balanced"]
    return out


def carq_rmw_nm(adeck_text: str, dtg: str) -> float:
    """The radius of maximum wind (nm) of the CARQ analysis (tau 0) of cycle ``dtg`` in an ATCF a-deck: field 20.
    NaN when absent or 0 (ATCF writes 0 for unknown)."""
    for line in adeck_text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) > 19 and f[2] == dtg and f[4] == "CARQ" and f[5] == "0":
            try:
                v = float(f[19])
            except ValueError:
                continue
            if v > 0:
                return v
    return float("nan")
