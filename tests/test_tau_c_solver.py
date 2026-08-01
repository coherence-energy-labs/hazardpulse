"""FVCS W-2 acceptance: the canonical solver consumes Gamma, and the physics is right.

The defect: the solver's damping argument was named ``kappa`` and squared
internally, while the tornado engine passed kappa = sqrt(Gamma/D) with a
non-unit D — so the solved equation was D*lap(tau) - (Gamma/D)*tau + S = 0,
screening understated by a factor of D. The decay-length test below FAILS on
that code (it measures ell = D/sqrt(Gamma) instead of sqrt(D/Gamma)); the
residual test fails on any solver that squares its damping argument.
"""
from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.coherence.tau_c_solver import solve_helmholtz_2d, laplacian_2d

# One analytic case used by both tests: constant coefficients, D deliberately != 1.
D0 = 2.25
GAMMA0 = 0.09
ELL_TRUE = np.sqrt(D0 / GAMMA0)          # 5.0 cells — the screened-Poisson decay length
ELL_OLD_BUG = D0 / np.sqrt(GAMMA0)       # 7.5 cells — what kappa=sqrt(G/D) + squaring produced


def _point_source_solution(n=81, n_iter=40000):
    S = np.zeros((n, n))
    S[n // 2, n // 2] = 1.0
    return solve_helmholtz_2d(
        S, GAMMA0, dx=1.0, D=D0, n_iter=n_iter, omega=0.9, dtype=np.float64,
    )


def _measured_decay_length(tau, n=81, r_lo=8.0, r_hi=18.0):
    """Fit log(tau * sqrt(r)) ~ -r/ell along radii (2-D screened Poisson:
    G ~ K0(r/ell) ~ exp(-r/ell)/sqrt(r) for r >> ell)."""
    c = n // 2
    ys, xs = np.mgrid[0:n, 0:n]
    r = np.hypot(ys - c, xs - c)
    mask = (r >= r_lo) & (r <= r_hi) & (tau > 0)
    logt = np.log(tau[mask] * np.sqrt(r[mask]))
    slope = np.polyfit(r[mask], logt, 1)[0]
    return -1.0 / slope


def test_decay_length_matches_analytic_screened_poisson():
    """Constant Gamma, constant D != 1, point source: the solved field must decay
    with ell = sqrt(D/Gamma). On the pre-W-2 code this measures ~D/sqrt(Gamma)."""
    tau = _point_source_solution()
    ell = _measured_decay_length(tau)
    assert ell == pytest.approx(ELL_TRUE, rel=0.12), (
        f"measured ell={ell:.3f}, analytic sqrt(D/Gamma)={ELL_TRUE:.3f}"
    )
    # and it must NOT be the old bug's decay length (7.5 vs 5.0 — well separated)
    assert abs(ell - ELL_OLD_BUG) > abs(ell - ELL_TRUE), (
        f"measured ell={ell:.3f} sits closer to the pre-fix value {ELL_OLD_BUG:.3f} "
        f"than to the analytic {ELL_TRUE:.3f} — screening is still rescaled by D"
    )


def test_solver_satisfies_documented_pde():
    """The residual of D*lap(tau) - Gamma*tau + S must vanish on the interior.
    A solver that squares its damping argument fails this with Gamma != Gamma**2."""
    rng = np.random.default_rng(7)
    n = 64
    S = rng.uniform(0.0, 1.0, (n, n))
    G = rng.uniform(0.2, 1.5, (n, n))          # varying Gamma, nowhere == Gamma**2
    D = rng.uniform(0.5, 3.0, (n, n))
    tau = solve_helmholtz_2d(S, G, dx=1.0, D=D, n_iter=30000, omega=0.9,
                             dtype=np.float64)
    resid = D * laplacian_2d(tau) - G * tau + S
    interior = resid[2:-2, 2:-2]
    rel = np.abs(interior).max() / np.abs(S).max()
    assert rel < 5e-3, f"PDE residual {rel:.2e} — solver is not solving the documented equation"


def test_engine_wrappers_delegate_with_gamma_semantics():
    """Both engine wrappers must hand gamma through to the shared solver unchanged."""
    from hazardpulse.tornado.coherence_engine import solve_helmholtz_2d as tor_solve
    from hazardpulse.earthquake.coherence_engine import solve_helmholtz_2d as eq_solve

    rng = np.random.default_rng(11)
    S = rng.uniform(0, 1, (20, 30)).astype(np.float32)
    G = rng.uniform(0.1, 0.8, (20, 30)).astype(np.float32)
    D = rng.uniform(1.0, 1.6, (20, 30)).astype(np.float32)

    ref32 = solve_helmholtz_2d(S, G, dx=1.0, D=D, n_iter=200, omega=0.7,
                               dtype=np.float32)
    assert np.array_equal(tor_solve(S, G, dx=1.0, D=D, n_iter=200, omega=0.7), ref32)

    ref64 = solve_helmholtz_2d(S.astype(np.float64), G.astype(np.float64), dx=1.0,
                               D=D.astype(np.float64), n_iter=300, omega=0.8,
                               dtype=np.float64)
    assert np.array_equal(
        eq_solve(S.astype(np.float64), G.astype(np.float64), dx=1.0,
                 D=D.astype(np.float64)),
        ref64,
    )
