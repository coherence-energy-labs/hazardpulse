"""AS, amendment E5's aftershock layer (src/hazardpulse/earthquake/aftershock.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.earthquake import aftershock as A
from hazardpulse.earthquake import operational_forecast as of


@pytest.mark.parametrize("p", [0.9, 1.0, 1.1, 1.4])
def test_the_omori_integral_is_the_integral(p):
    c_t, tau = 0.03, 12.5
    s = np.linspace(0.0, tau, 400_001)
    numeric = np.trapezoid((s + c_t) ** -p, s)
    assert float(A.omori_integral(tau, p, c_t)) == pytest.approx(numeric, rel=1e-6)
    assert float(A.omori_integral(0.0, p, c_t)) == 0.0


def test_the_unit_draws_follow_the_kernel():
    r, az = A.unit_draws()
    assert np.mean(r <= 1.0) == pytest.approx(1.0 - 2 ** -0.5, abs=0.01)     # CDF(1) = 1 - (1 + 1)^-1/2
    assert np.mean(r <= 3.0) == pytest.approx(1.0 - 10 ** -0.5, abs=0.01)
    assert abs(np.mean(np.cos(az))) < 0.02 and abs(np.mean(np.sin(az))) < 0.02


def test_a_narrow_kernel_stays_in_its_cell_and_fractions_never_exceed_one():
    tab = A.kernel_fractions(7.0, -81.0)                  # the centre of a 2-degree cell
    home = int(of.cell_index(7.0, -81.0))
    assert tab[0][home] > 0.97                            # 2 km wide: nearly all in its own cell
    for d in tab:
        assert sum(d.values()) <= 1.0 + 1e-12
    assert tab[-1].get(home, 0.0) < tab[0][home]           # 2,000 km wide: spread out
    assert A.fraction_at(tab, home, 2.0) == pytest.approx(tab[0][home])
    assert A.fraction_at(tab, home, 1.0) == pytest.approx(tab[0][home])    # clamped below the grid


def test_the_layer_rises_with_magnitude_and_decays_with_time():
    layer = A.AftershockLayer(A.Params(K=0.05, alpha=1.0, p=1.1, c_t=0.02, d0=40.0))
    home = int(of.cell_index(7.59, -80.77))
    big, small = layer.trigger(0.0, 7.59, -80.77, 7.7), layer.trigger(0.0, 7.59, -80.77, 6.5)
    p_big = [layer.probability(home, t, [big]) for t in (0.01, 0.25, 1.0, 7.0, 29.0)]
    assert all(a > b for a, b in zip(p_big, p_big[1:]))                  # Omori decay
    assert layer.probability(home, 0.25, [big]) > layer.probability(home, 0.25, [small])
    assert layer.probability(home, 31.0, [big]) == 0.0                    # older than the 30-day window
    assert layer.probability(home, -1.0, [big]) == 0.0                    # not yet happened
    deep = A.AftershockLayer.trigger(layer, 0.0, 7.59, -80.77, 5.9)       # below M6 contributes nothing
    assert layer.rate(home, 0.25, [deep]) == 0.0
    g = layer.grid(0.25, [big])
    assert g[home] == pytest.approx(layer.probability(home, 0.25, [big])) and g.max() == g[home]
