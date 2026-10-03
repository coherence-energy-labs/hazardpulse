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

CONVERGENCE (measured 2026-08-02, tests/test_helmholtz_analytic.py). The
default fixed-count path is NOT a converged solve: at its 300 sweeps it is
44.5% from the closed-form K0 solution of a point-source problem, and the
sweeps needed grow with resolution. It survives, bit-identical, because
served earthquake models were trained on its output. New code passes
``tol=`` and gets ``solve_helmholtz_2d_certified``: PCG to a residual
certificate (``helmholtz_residual``), refusing -- not returning -- a field
that misses it. The legacy path's denominator is floored to 1e-12 to avoid
singular cells when D=0 and Gamma=0 simultaneously.
"""
from __future__ import annotations

import numpy as np

HELMHOLTZ_DEFAULT_ITERS: int = 300
SOR_OMEGA_DEFAULT: float = 0.7

# Certified mode: relative residual target and the CG budget before refusing.
# 1e-8 is ~1e5x below the legacy default's residual on the analytic problem
# (1.5e-2) and far above float64 round-off.
CERTIFIED_TOL_DEFAULT: float = 1e-8
CERTIFIED_MAX_ITERS: int = 20_000


def solve_helmholtz_2d(
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
    n_iter: int = HELMHOLTZ_DEFAULT_ITERS,
    omega: float = SOR_OMEGA_DEFAULT,
    dtype: np.dtype = np.float32,
    tol: float | None = None,
    max_iter: int = CERTIFIED_MAX_ITERS,
) -> np.ndarray:
    """Solve ``D * nabla^2(tau) - gamma * tau + S = 0`` on a 2-D grid.

    Two modes:

    * ``tol=None`` (default) -- the legacy fixed-count damped Jacobi
      iteration, ``n_iter`` sweeps, bit-identical to every earlier release.
      It reports success by construction: at the default 300 sweeps it is
      44.5% from the closed-form solution of a point-source problem
      (tests/test_helmholtz_analytic.py). Callers whose served models were
      trained on its output keep it until they retrain.
    * ``tol=<float>`` -- CERTIFIED: preconditioned conjugate gradients run
      until ``helmholtz_residual(tau, ...) <= tol`` (max |PDE residual| over
      the interior, relative to max |S|), else ``HelmholtzNotConverged`` is
      raised. ``n_iter`` and ``omega`` are ignored. Requires ``D > 0`` and
      ``gamma >= 0`` on the interior (the operator is then symmetric
      positive definite after dividing each row by D).

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
        small grids. The certified mode always iterates in float64 and
        casts on return.
    tol : float, optional
        Relative residual target; selects the certified mode.
    max_iter : int
        Certified mode only: CG iteration budget before refusing.

    Returns
    -------
    ndarray, shape (ny, nx), dtype as requested
        Coherence field tau_c.
    """
    if tol is not None:
        tau64, _info = solve_helmholtz_2d_certified(
            source, gamma, dx, D=D, tol=tol, max_iter=max_iter,
        )
        return tau64.astype(dtype)

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


class HelmholtzNotConverged(RuntimeError):
    """The certified solve did not reach its residual target -- no field is returned."""


def helmholtz_residual(
    tau: np.ndarray,
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
) -> float:
    """The solve's certificate: ``max |D lap(tau) - gamma tau + S|`` over the
    interior, divided by ``max |S|`` there.

    It needs no reference solution, costs one stencil pass, and cannot be
    satisfied by a solver that has not solved the equation -- unlike an
    iteration count, which reports success by construction. Boundary cells
    are excluded: the solver pins them to zero (Dirichlet), so the PDE is
    not imposed there. Returns 0.0 for an all-zero interior source when tau
    is zero, and ``inf`` for a nonzero tau with no source.
    """
    t = np.asarray(tau, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    g = np.broadcast_to(np.asarray(gamma, dtype=np.float64), t.shape)
    d = np.broadcast_to(np.asarray(D, dtype=np.float64), t.shape)
    lap = (
        t[2:, 1:-1] + t[:-2, 1:-1] + t[1:-1, 2:] + t[1:-1, :-2] - 4.0 * t[1:-1, 1:-1]
    ) / (dx * dx)
    r = d[1:-1, 1:-1] * lap - g[1:-1, 1:-1] * t[1:-1, 1:-1] + s[1:-1, 1:-1]
    r_max = float(np.max(np.abs(r))) if r.size else 0.0
    s_max = float(np.max(np.abs(s[1:-1, 1:-1]))) if r.size else 0.0
    if s_max == 0.0:
        return 0.0 if r_max == 0.0 else float("inf")
    return r_max / s_max


def solve_helmholtz_2d_certified(
    source: np.ndarray,
    gamma: float | np.ndarray,
    dx: float = 1.0,
    *,
    D: float | np.ndarray = 1.0,
    tol: float = CERTIFIED_TOL_DEFAULT,
    max_iter: int = CERTIFIED_MAX_ITERS,
) -> tuple[np.ndarray, dict]:
    """Solve ``D lap(tau) - gamma tau + S = 0`` to a residual certificate.

    Same discrete problem as the Jacobi path (5-point stencil, D at the cell
    centre, tau pinned to zero on the outer ring), solved by Jacobi-
    preconditioned conjugate gradients on the row-scaled system

        (-lap + diag(gamma / D)) tau = S / D      (interior unknowns only)

    which is symmetric positive definite when D > 0 and gamma >= 0. CG needs
    O(sqrt(condition number)) iterations where damped Jacobi needs O(condition
    number) -- on the analytic point-source problem it reaches the stencil's
    own accuracy floor in a few hundred iterations, where Jacobi needs ~6,000.

    Returns ``(tau, info)`` with tau in float64 and
    ``info = {"method", "iterations", "residual_rel", "tol"}``, where
    ``residual_rel`` is ``helmholtz_residual`` of the returned field. Raises
    ``HelmholtzNotConverged`` rather than return a field that misses ``tol``,
    and ``ValueError`` for D <= 0, gamma < 0 or non-finite inputs on the
    interior.
    """
    s = np.asarray(source, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"source must be 2-D, got shape {s.shape}")
    ny, nx = s.shape
    tau = np.zeros((ny, nx), dtype=np.float64)
    info = {"method": "pcg", "iterations": 0, "residual_rel": 0.0, "tol": float(tol)}
    if ny < 3 or nx < 3:
        return tau, info  # no interior: the Dirichlet ring is the whole answer

    g = np.broadcast_to(np.asarray(gamma, dtype=np.float64), s.shape)[1:-1, 1:-1]
    d = np.broadcast_to(np.asarray(D, dtype=np.float64), s.shape)[1:-1, 1:-1]
    si = s[1:-1, 1:-1]
    if not (np.all(np.isfinite(si)) and np.all(np.isfinite(g)) and np.all(np.isfinite(d))):
        raise ValueError("source, gamma and D must be finite on the interior")
    if np.any(d <= 0.0):
        raise ValueError("certified solve needs D > 0 on every interior cell")
    if np.any(g < 0.0):
        raise ValueError("certified solve needs gamma >= 0 on every interior cell")
    if not np.any(si):
        return tau, info  # zero source, zero field, zero residual

    inv_dx2 = 1.0 / (dx * dx)
    shift = g / d
    b = si / d
    m_inv = 1.0 / (4.0 * inv_dx2 + shift)  # Jacobi preconditioner
    s_max = float(np.max(np.abs(si)))

    def apply_a(u: np.ndarray) -> np.ndarray:
        nb = np.zeros_like(u)
        nb[1:, :] += u[:-1, :]
        nb[:-1, :] += u[1:, :]
        nb[:, 1:] += u[:, :-1]
        nb[:, :-1] += u[:, 1:]
        return (4.0 * u - nb) * inv_dx2 + shift * u

    x = np.zeros_like(b)
    r = b.copy()  # b - A x with x = 0
    z = m_inv * r
    p = z.copy()
    rz = float(np.sum(r * z))
    for it in range(1, max_iter + 1):
        ap = apply_a(p)
        alpha = rz / float(np.sum(p * ap))
        x += alpha * p
        r -= alpha * ap
        # PDE residual of the original (unscaled) equation is D * r.
        if float(np.max(np.abs(d * r))) / s_max <= tol:
            tau[1:-1, 1:-1] = x
            res = helmholtz_residual(tau, s, gamma, dx, D=D)
            if res <= tol:
                info.update(iterations=it, residual_rel=res)
                return tau, info
            # Recurrence drifted from the true residual: restart from x.
            r = b - apply_a(x)
        z = m_inv * r
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new

    tau[1:-1, 1:-1] = x
    res = helmholtz_residual(tau, s, gamma, dx, D=D)
    raise HelmholtzNotConverged(
        f"PCG reached residual {res:.3e} > tol {tol:.1e} after {max_iter} iterations"
    )


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
