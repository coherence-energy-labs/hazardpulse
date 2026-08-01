"""Canonical Helmholtz / screened-Poisson PDE solver for coherence fields.

Solves:

    D(x) * nabla^2 tau_c - Gamma(x) * tau_c + S(x) = 0

on a 2-D regular grid using a Jacobi iterative scheme with SOR
relaxation under Dirichlet (zero) boundary conditions.

PARAMETER SEMANTICS (FVCS W-2). The damping argument is GAMMA — the
coefficient that multiplies tau_c directly, with units of a rate. It was
historically named ``kappa`` and squared internally, which invited callers
to pass the screening wavenumber kappa = sqrt(Gamma/D); with a non-unit D
that solves ``D*nabla^2(tau) - (Gamma/D)*tau + S = 0`` — screening
understated by a factor of D (the tornado engine shipped exactly this).
kappa and the coherence length ell are DERIVED reporting quantities,
never solver inputs:

    kappa = sqrt(Gamma / D)        ell = 1 / kappa = sqrt(D / Gamma)

Both the earthquake and tornado coherence engines previously held
near-identical copies of this routine; this module is now the single
source of truth. The dtype argument lets callers pick float32 (faster,
less memory — appropriate for large CONUS-scale grids) or float64
(better precision — appropriate for small global 2-degree grids).

Numerically the routine converges within ~250-300 iterations for
typical hazard sources at omega=0.7. The denominator is floored to
1e-12 to avoid singular cells when D=0 and Gamma=0 simultaneously.
"""
from __future__ import annotations

import numpy as np

HELMHOLTZ_DEFAULT_ITERS: int = 300
SOR_OMEGA_DEFAULT: float = 0.7


def solve_helmholtz_2d(
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
    n_iter: int = HELMHOLTZ_DEFAULT_ITERS,
    omega: float = SOR_OMEGA_DEFAULT,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Solve ``D * nabla^2(tau) - gamma * tau + S = 0`` on a 2-D grid.

    Parameters
    ----------
    source : ndarray, shape (ny, nx)
        Source term S(x).
    gamma : float or ndarray
        Damping rate Gamma(x) — multiplies tau directly. Scalars are
        broadcast. NOT the screening wavenumber: kappa = sqrt(gamma/D)
        and ell = 1/kappa are derived from this, never passed in.
    dx : float
        Grid spacing (the routine assumes dx == dy).
    D : float or ndarray
        Diffusivity (scalar or per-cell field).
    n_iter : int
        Jacobi iterations (default 300).
    omega : float in (0, 1]
        SOR relaxation parameter.
    dtype : numpy dtype
        Working precision (np.float32 or np.float64). float32 saves
        memory on large grids; float64 reduces numerical noise on
        small grids.

    Returns
    -------
    ndarray, shape (ny, nx), dtype as requested
        Coherence field tau_c.
    """
    ny, nx = source.shape
    tau = np.zeros((ny, nx), dtype=dtype)
    src = np.asarray(source, dtype=dtype)
    gamma_arr = np.asarray(gamma, dtype=dtype)
    D_arr = np.asarray(D, dtype=dtype)
    dx2 = dtype(dx ** 2)

    eps = dtype(1e-12)
    one_minus = dtype(1.0 - omega)
    omega_d = dtype(omega)

    for _ in range(n_iter):
        # 5-point Laplacian neighbour sum, Dirichlet (zero) BCs at edges.
        neighbors = np.zeros_like(tau)
        neighbors[1:, :] += tau[:-1, :]   # from above
        neighbors[:-1, :] += tau[1:, :]   # from below
        neighbors[:, 1:] += tau[:, :-1]   # from left
        neighbors[:, :-1] += tau[:, 1:]   # from right

        denom = 4.0 * D_arr / dx2 + gamma_arr
        denom = np.maximum(denom, eps)
        tau_new = (D_arr * neighbors / dx2 + src) / denom

        # Re-enforce Dirichlet BCs each iteration (defensive)
        tau_new[0, :] = 0.0
        tau_new[-1, :] = 0.0
        tau_new[:, 0] = 0.0
        tau_new[:, -1] = 0.0

        tau = omega_d * tau_new + one_minus * tau

    return tau


def gradient_2d(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient (d/dy, d/dx) on a 2-D grid.

    Boundaries use forward / backward differences. Returns float32 by
    default; promote inputs if you need higher precision.
    """
    f = np.asarray(field)
    ny, nx = f.shape
    grad_y = np.zeros_like(f, dtype=f.dtype)
    grad_x = np.zeros_like(f, dtype=f.dtype)

    grad_y[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
    grad_y[0, :] = (f[1, :] - f[0, :]) / dy
    grad_y[-1, :] = (f[-1, :] - f[-2, :]) / dy

    grad_x[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
    grad_x[:, 0] = (f[:, 1] - f[:, 0]) / dx
    grad_x[:, -1] = (f[:, -1] - f[:, -2]) / dx

    return grad_y, grad_x


def laplacian_2d(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """5-point Laplacian on a 2-D grid (Dirichlet zero BCs implicit at edges)."""
    f = np.asarray(field)
    lap = np.zeros_like(f, dtype=f.dtype)
    lap[1:-1, 1:-1] = (
        (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dx * dx)
        + (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dy * dy)
    )
    return lap
