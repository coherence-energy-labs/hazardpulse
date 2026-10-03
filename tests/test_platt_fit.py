"""Platt calibration: the maximum-likelihood fit is found, or the fit refuses -- never a diverged one.

Plain Newton diverged on the EF2+ product's leave-one-year-out scores (a = 3.1e16), so every
served probability was 0 or 1 and the 2025 Brier skill was -57.7. The synthetic shape below is the
same mechanism: negatives near -10 with a heavy right tail, a rare positive class scored high."""
from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.tornado import definitive_model as dm


def _plain_newton(F, y, max_iter=100):
    """The pre-2026-10-03 fit, kept here as the reference it must equal when it converges."""
    a, b = 1.0, float(np.log(y.mean() / (1.0 - y.mean())) - np.mean(F))
    for _ in range(max_iter):
        z = np.clip(a * F + b, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        r = p - y
        g = np.array([np.sum(r * F), np.sum(r)])
        h = np.array([[np.sum(w * F * F) + 1e-9, np.sum(w * F)], [np.sum(w * F), np.sum(w) + 1e-9]])
        step = np.linalg.solve(h, g)
        a -= step[0]
        b -= step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return a, b


def _rare_heavy_tail(seed=0, n_neg=50000, n_pos=10, tail=0.02):
    rng = np.random.RandomState(seed)
    n_tail = int(tail * n_neg)
    neg = np.r_[rng.normal(-10.4, 1.3, n_neg - n_tail), rng.uniform(-6.0, 7.0, n_tail)]
    pos = rng.normal(4.0, 2.5, n_pos)
    return np.r_[neg, pos], np.r_[np.zeros(n_neg), np.ones(n_pos)]


def test_the_ef2_shape_diverges_under_plain_newton_and_converges_here():
    F, y = _rare_heavy_tail()
    a_old, _ = _plain_newton(F, y)
    assert abs(a_old) > 1e6                                 # the failure this guards against still happens
    cal = dm.fit_platt(F, y)
    assert 0.0 < cal["a"] < 10.0 and -60.0 < cal["b"] < 0.0
    p = dm.apply_calibration(F, cal)
    assert p.mean() == pytest.approx(y.mean(), rel=1e-6)    # an MLE with an intercept: mean forecast = base rate
    const = np.full_like(F, y.mean())
    ll = lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
    assert ll(p) < ll(const)


def test_a_well_posed_fit_is_plain_newton_bit_for_bit():
    rng = np.random.RandomState(1)
    F = rng.normal(0, 2, 20000)
    y = (rng.uniform(size=F.size) < 1 / (1 + np.exp(-(0.8 * F - 3.0)))).astype(float)
    a_old, b_old = _plain_newton(F, y)
    cal = dm.fit_platt(F, y)
    assert cal["a"] == a_old and cal["b"] == b_old


def test_separable_scores_refuse_instead_of_returning_a_calibration():
    F = np.r_[np.linspace(-5, -1, 500), np.linspace(1, 5, 20)]
    y = np.r_[np.zeros(500), np.ones(20)]
    with pytest.raises(dm.PlattNotConverged):
        dm.fit_platt(F, y)


def test_one_class_is_refused():
    with pytest.raises(ValueError):
        dm.fit_platt(np.zeros(10), np.zeros(10))
