"""The certified Helmholtz solve: a residual certificate instead of an iteration count.

tests/test_helmholtz_analytic.py established that the legacy fixed-count
Jacobi path is 44.5% from the closed-form K0 solution at its default 300
sweeps and that the PDE residual tracks the error. These tests pin the mode
built on that finding: conjugate gradients run to a residual target, and a
field that misses it is refused rather than returned.
"""
from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.coherence.tau_c_solver import (
    HelmholtzNotConverged,
    helmholtz_residual,
    solve_helmholtz_2d,
    solve_helmholtz_2d_certified,
)

kn = pytest.importorskip("scipy.special", reason="scipy needed for K_0").kn

D, GAMMA, N, DX, Q = 2.5, 0.1, 193, 0.5, 1.0


def _point_source():
    c = N // 2
    src = np.zeros((N, N))
    src[c, c] = Q / (DX * DX)
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot((xx - c) * DX, (yy - c) * DX)
    mask = (r > 3 * DX) & (r < 0.40 * N * DX)
    exact = Q / (2 * np.pi * D) * kn(0, np.sqrt(GAMMA / D) * r[mask])
    return src, mask, exact


def test_certified_solve_reaches_the_stencil_floor_with_its_certificate():
    src, mask, exact = _point_source()
    tau, info = solve_helmholtz_2d_certified(src, GAMMA, DX, D=D, tol=1e-8)
    rel = np.linalg.norm(tau[mask] - exact) / np.linalg.norm(exact)
    assert rel < 0.005  # the 5-point stencil's own floor is 0.229%
    assert info["residual_rel"] <= 1e-8
    assert info["residual_rel"] == pytest.approx(helmholtz_residual(tau, src, GAMMA, DX, D=D))
    # Measured 193 iterations; Jacobi needs ~6,000 sweeps for the same floor.
    assert info["iterations"] < 1000


def test_certified_solve_agrees_with_converged_jacobi():
    rng = np.random.RandomState(3)
    S = rng.rand(34, 63)
    G = 0.15 + rng.rand(34, 63)
    Dv = 1.0 + 0.6 * rng.rand(34, 63)
    tj = solve_helmholtz_2d(S, G, 1.0, D=Dv, n_iter=5000, dtype=np.float64)
    tc, _ = solve_helmholtz_2d_certified(S, G, 1.0, D=Dv, tol=1e-12)
    assert np.abs(tc - tj).max() <= 1e-9 * np.abs(tj).max()


def test_the_certificate_separates_the_legacy_default_from_a_solve():
    src, _, _ = _point_source()
    legacy = solve_helmholtz_2d(src, GAMMA, DX, D=D, dtype=np.float64)  # 300 sweeps
    cert, _ = solve_helmholtz_2d_certified(src, GAMMA, DX, D=D, tol=1e-8)
    assert helmholtz_residual(legacy, src, GAMMA, DX, D=D) > 1e-4
    assert helmholtz_residual(cert, src, GAMMA, DX, D=D) <= 1e-8


def test_a_missed_target_is_refused_not_returned():
    src, _, _ = _point_source()
    with pytest.raises(HelmholtzNotConverged):
        solve_helmholtz_2d_certified(src, GAMMA, DX, D=D, tol=1e-12, max_iter=10)


def test_invalid_coefficients_are_rejected():
    S = np.ones((8, 8))
    with pytest.raises(ValueError):
        solve_helmholtz_2d_certified(S, 0.1, 1.0, D=0.0)
    with pytest.raises(ValueError):
        solve_helmholtz_2d_certified(S, -0.1, 1.0, D=1.0)
    bad = S.copy()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        solve_helmholtz_2d_certified(bad, 0.1, 1.0, D=1.0)


def test_zero_source_is_the_zero_field():
    tau, info = solve_helmholtz_2d_certified(np.zeros((10, 12)), 0.2, 1.0, D=1.3)
    assert not tau.any() and info["iterations"] == 0


def test_tol_keyword_routes_through_the_certified_path():
    rng = np.random.RandomState(5)
    S = rng.rand(20, 25)
    a = solve_helmholtz_2d(S, 0.3, 1.0, D=1.2, tol=1e-10, dtype=np.float64)
    b, _ = solve_helmholtz_2d_certified(S, 0.3, 1.0, D=1.2, tol=1e-10)
    assert np.array_equal(a, b)


def test_legacy_path_is_unchanged_without_tol():
    """No tol -> the fixed-count Jacobi, bit-for-bit (the live earthquake ledger rests on it)."""
    rng = np.random.RandomState(7)
    S = rng.rand(16, 18)
    G = 0.05 + rng.rand(16, 18)
    tau = np.zeros_like(S)
    for _ in range(300):
        nb = np.zeros_like(tau)
        nb[1:, :] += tau[:-1, :]
        nb[:-1, :] += tau[1:, :]
        nb[:, 1:] += tau[:, :-1]
        nb[:, :-1] += tau[:, 1:]
        new = (1.0 * nb / 1.0 + S) / np.maximum(4.0 * 1.0 / 1.0 + G, 1e-12)
        new[0, :] = new[-1, :] = 0.0
        new[:, 0] = new[:, -1] = 0.0
        tau = 0.7 * new + 0.30000000000000004 * tau
    got = solve_helmholtz_2d(S, G, 1.0, D=1.0, dtype=np.float64)
    assert np.array_equal(got, tau)


def test_tornado_engine_fields_carry_a_certified_tau():
    from hazardpulse.tornado import coherence_engine as ce

    from test_tornado_torsion import _synthetic_atmosphere

    atm = _synthetic_atmosphere()
    fields = ce.compute_coherence_fields(atm, month=5)
    res = helmholtz_residual(
        fields["tau"].astype(np.float64), fields["S_field"], fields["Gamma_field"], 1.0,
        D=(1.0 + 0.3 * (np.hypot(atm["ustorm"], atm["vstorm"]) / 20.0)),
    )
    # float32 storage of a 1e-8 solve: the cast, not the solve, bounds this.
    assert res < 1e-5
