"""Shared coherence-field PDE solver and helpers.

This package consolidates the Helmholtz / screened-Poisson solver
that earthquake/, tornado/, and (when wired) hurricane/ scorers all
rely on. Single source of truth for the canonical equation:

    D * nabla^2(tau_c) - Gamma * tau_c + S = 0

where:
    tau_c   = coherence field (positive scalar)
    D       = diffusivity (m^2/s in physical units, dimensionless on grid)
    Gamma   = damping (a.k.a. healing rate, kappa^2 in screened-Poisson form)
    S       = source term

The same solver is also imported by Signalbook's science layer so the
two projects produce bit-identical fields when given the same inputs
(important for federation queries that span both atlases).

Public API:
    - solve_helmholtz_2d(source, gamma, dx, *, D, n_iter, omega, dtype)
    - gradient_2d(field, dx, dy)
    - laplacian_2d(field, dx, dy)
"""
from hazardpulse.coherence.tau_c_solver import (
    HELMHOLTZ_DEFAULT_ITERS,
    SOR_OMEGA_DEFAULT,
    gradient_2d,
    laplacian_2d,
    solve_helmholtz_2d,
)

__all__ = [
    "HELMHOLTZ_DEFAULT_ITERS",
    "SOR_OMEGA_DEFAULT",
    "gradient_2d",
    "laplacian_2d",
    "solve_helmholtz_2d",
]
