"""The solver, measured against the exact solution it is supposed to approximate.

`solve_helmholtz_2d` had no test of its numerics against a closed form. Every
existing test compares it to itself (determinism), to a prior snapshot
(regression), or to a downstream score. Those cannot detect a solver that is
converging to the wrong answer, or one that is not converging at all.

THE CLOSED FORM. For a point source of strength Q at the origin,

    D nabla^2 tau - Gamma tau + Q delta = 0    =>    tau(r) = Q/(2 pi D) K_0(kappa r)

with kappa = sqrt(Gamma/D) and K_0 the modified Bessel function of the second
kind. This is the free-space Green's function of the screened Helmholtz
operator; it is exact, it is independent of the discretization, and it is the
right yardstick because it is what the module docstring says the routine solves.

WHAT THIS MEASURED (grid 193x193, D=2.5, Gamma=0.1, dx=0.5, so ell = 1/kappa =
5 units = 10 cells; annulus 3 dx < r < 0.4 N dx skips the singular core and the
Dirichlet wall):

    iters    L2 rel error    max residual / max(tau)
      300        44.53%              1.5e-02     <-- HELMHOLTZ_DEFAULT_ITERS
     1000         9.32%              1.3e-03
     3000         0.3154%            1.3e-05
     6000         0.2289%            3.3e-08
    12000         0.2288%            4.5e-13

Two things follow, and the second is the reason this file exists.

1. The shipped default of 300 iterations carries ~45% L2 error on this problem.
   The docstring's "converges within ~250-300 iterations" is a statement about
   typical hazard sources, not about convergence, and the two readings differ by
   nearly 200x here. The floor of 0.229% is reached by 6000 iterations and does
   not move at 12000 -- that residue is the 5-point stencil's own discretization
   error with a discrete delta, i.e. the honest accuracy ceiling of the method,
   which no iteration count improves. Note the cost scales with resolution: at
   dx = 1 the same problem needed only ~3000 sweeps, so an iteration count
   calibrated at one grid spacing does not transfer to another.

2. THE RESIDUAL TRACKS THE ERROR. max|D lap(tau) - Gamma tau + S| falls from
   1.5e-02 to machine zero in lockstep with the true error. A residual check --
   one scatter pass, no reference solve, impossible for the solver to game --
   would have flagged the under-convergence at the shipped default immediately.
   A fixed iteration count cannot, because it reports success by construction.

So this file asserts two different things: an accuracy bound the solver must
meet when properly converged, and the diagnostic property that makes the first
one checkable in production without a closed form to compare against.
"""
from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.coherence.tau_c_solver import (
    HELMHOLTZ_DEFAULT_ITERS,
    solve_helmholtz_2d,
)

kn = pytest.importorskip("scipy.special", reason="scipy needed for K_0").kn

# D != 1 and DX != 1 DELIBERATELY. A first cut of this file used D = dx = 1,
# and mutation testing then MISSED two real defects -- dropping D from the
# neighbour term, and mis-scaling by dx^2 -- because at unity those factors are
# invisible in the answer. A test whose parameters hide a coefficient cannot
# detect an error in it. Mutation score went 3/5 -> 5/5 on this change alone.
D = 2.5
GAMMA = 0.1
N = 193
DX = 0.5
Q = 1.0
CONVERGED_ITERS = 6000


def _setup():
    c = N // 2
    source = np.zeros((N, N))
    source[c, c] = Q / (DX * DX)          # discrete delta of total strength Q
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot((xx - c) * DX, (yy - c) * DX)
    # skip the singular core (the closed form diverges at r=0, the grid does not)
    # and the outer band where the Dirichlet wall dominates
    mask = (r > 3 * DX) & (r < 0.40 * N * DX)
    kappa = np.sqrt(GAMMA / D)
    exact = Q / (2 * np.pi * D) * kn(0, kappa * r[mask])
    return source, mask, exact


def _residual(tau, source):
    lap = np.zeros_like(tau)
    lap[1:-1, 1:-1] = (
        tau[2:, 1:-1] + tau[:-2, 1:-1] + tau[1:-1, 2:] + tau[1:-1, :-2]
        - 4 * tau[1:-1, 1:-1]
    ) / DX**2
    return np.abs(D * lap - GAMMA * tau + source)[1:-1, 1:-1].max()


def test_converged_solver_matches_the_greens_function():
    """Properly converged, the solver reproduces K_0 to the stencil's own floor."""
    source, mask, exact = _setup()
    tau = solve_helmholtz_2d(source, GAMMA, DX, D=D, n_iter=CONVERGED_ITERS, dtype=np.float64)
    rel = np.linalg.norm(tau[mask] - exact) / np.linalg.norm(exact)
    assert rel < 0.005, f"converged solver is {rel:.2%} from the exact solution"


def test_discretization_floor_is_not_an_iteration_deficit():
    """Past convergence the error stops moving: what remains is the stencil."""
    source, mask, exact = _setup()
    errs = []
    for iters in (CONVERGED_ITERS, 2 * CONVERGED_ITERS):
        tau = solve_helmholtz_2d(source, GAMMA, DX, D=D, n_iter=iters, dtype=np.float64)
        errs.append(np.linalg.norm(tau[mask] - exact) / np.linalg.norm(exact))
    assert abs(errs[0] - errs[1]) < 1e-4, (
        f"error still moving between {CONVERGED_ITERS} and {2*CONVERGED_ITERS} sweeps: {errs} — the floor is "
        "not discretization"
    )


def test_the_residual_detects_under_convergence():
    """The property that makes this checkable in production.

    This is the load-bearing assertion. It does not merely observe that the
    default is under-converged; it establishes that a residual check SEPARATES
    the under-converged run from the converged one by orders of magnitude, so a
    caller who prints the residual cannot ship the first believing it is the
    second.
    """
    source, _, _ = _setup()
    tau_default = solve_helmholtz_2d(
        source, GAMMA, DX, D=D, n_iter=HELMHOLTZ_DEFAULT_ITERS, dtype=np.float64
    )
    tau_converged = solve_helmholtz_2d(
        source, GAMMA, DX, D=D, n_iter=CONVERGED_ITERS, dtype=np.float64
    )
    r_default = _residual(tau_default, source) / tau_default.max()
    r_converged = _residual(tau_converged, source) / tau_converged.max()
    assert r_converged < 1e-6, f"converged residual not near zero: {r_converged:.3e}"
    assert r_default > 1e-3, f"default residual too small to flag: {r_default:.3e}"
    assert r_default / r_converged > 1e4, (
        "the residual does not separate the two runs, so it could not be used as "
        f"a convergence gate: {r_default:.3e} vs {r_converged:.3e}"
    )


def test_default_iteration_count_is_calibrated_not_converged():
    """Pin the measured cost of the shipped default, so a change of it is visible.

    Not a demand that the default change — 300 sweeps may be the right trade for
    CONUS-scale hazard grids. It records what the trade costs on a problem whose
    answer is known, which is the thing no other test in this suite can say.
    """
    source, mask, exact = _setup()
    tau = solve_helmholtz_2d(
        source, GAMMA, DX, D=D, n_iter=HELMHOLTZ_DEFAULT_ITERS, dtype=np.float64
    )
    rel = np.linalg.norm(tau[mask] - exact) / np.linalg.norm(exact)
    assert 0.20 < rel < 0.70, (
        f"the default's distance from the exact solution moved to {rel:.2%}; it "
        "was 44.53% when measured. Investigate before re-pinning."
    )


def test_shipped_omega_is_under_relaxed_and_the_domain_is_necessary():
    """Two facts about `omega`, both measured against the closed form.

    Mutation testing of this file flagged one surviving mutant: forcing
    `omega = 1` (dropping the relaxation) changed nothing the other tests could
    see. That is not a gap in the tests -- it is a fact about the solver, and it
    deserves its own assertion rather than a shrug.

    At equal sweep counts, plain Jacobi (omega = 1) reaches a LOWER error than
    the shipped omega = 0.7: 0.232% against 0.315%. The shipped value is
    under-relaxation, so it buys stability margin the uniform screened problem
    does not need, at roughly a third more error per sweep budget.

    And the documented domain `(0, 1]` is empirically necessary, not stylistic:
    this is a damped JACOBI iteration, not Gauss-Seidel, so over-relaxation is
    unstable -- omega = 1.3 overflows to inf on this problem. A caller who reads
    "SOR" and reaches for the classical omega > 1 speedup gets a diverged field,
    not a faster one.
    """
    source, mask, exact = _setup()

    def err(omega):
        tau = solve_helmholtz_2d(
            source, GAMMA, DX, D=D, n_iter=3000, omega=omega, dtype=np.float64
        )
        return np.linalg.norm(tau[mask] - exact) / np.linalg.norm(exact)

    assert err(1.0) < err(0.7), (
        "omega=0.7 is no longer under-relaxed relative to plain Jacobi; the "
        "solver's iteration changed and this note is stale"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        diverged = err(1.3)
    assert not np.isfinite(diverged) or diverged > 1.0, (
        f"over-relaxation at omega=1.3 no longer diverges ({diverged}); the "
        "documented domain (0, 1] may have become conservative"
    )
