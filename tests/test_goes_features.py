"""G1's inner-core features on synthetic polar images: each one reads what it says, pinned before any outcome."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.hurricane import goes_abi as g
from hazardpulse.hurricane import goes_features as gf


def _image(ring_km=30.0, ring_k=190.0, eye_k=285.0, env_k=270.0, east_only=False):
    r = g.RADII_KM[:, None]
    az = g.AZIMUTHS[None, :]
    bt = env_k - (env_k - ring_k) * np.exp(-((r - ring_km) / 10.0) ** 2) * np.ones_like(az)
    bt = np.where(r < ring_km - 15, eye_k, bt)
    if east_only:
        bt = np.where(np.sin(az) > 0, bt, env_k)                 # cold cloud on the east side only
    return g.encode(bt)


def test_an_eyed_symmetric_core_reads_as_one():
    s = gf.hour_stats(_image(), eye=True)
    assert s["eye_contrast"] > 60 and s["cold_min"] == pytest.approx(190.0, abs=1.0)
    assert 15.0 <= s["eye_radius"] <= 30.0                         # the eyewall's inner edge, under the ring
    assert s["sym"] == pytest.approx(1.0, abs=0.02)
    assert s["core_cold"] > 0.2 and s["ring_cold"] < s["core_cold"]


def test_a_one_sided_core_is_asymmetric_and_has_no_eye_radius():
    s = gf.hour_stats(_image(east_only=True), eye=False)
    full = gf.hour_stats(_image(), eye=False)
    assert s["sym"] < full["sym"] - 0.2 and math.isnan(s["eye_radius"])


def _overcast(radius_km=60.0, cold_k=190.0, env_k=270.0):
    """A central dense overcast: deep cold cloud out to ``radius_km``, no eye."""
    r = g.RADII_KM[:, None] * np.ones((1, g.N_AZ))
    return g.encode(np.where(r <= radius_km, cold_k, env_k))


def test_missing_data_gives_nan_never_a_number():
    blank = np.full((g.N_R, g.N_AZ), g.MISSING, np.uint8)
    s = gf.hour_stats(blank, eye=False)
    assert all(math.isnan(s[k]) for k in ("core_cold", "ring_cold", "eye_contrast", "sym", "cold_min"))
    f = gf.features([(k, None) for k in range(-12, 3)])
    assert f["g_n"] == 0.0 and all(math.isnan(f[n]) for n in gf.G1_NAMES if n != "g_n")


def test_the_dynamics_read_the_window():
    cold = gf.hour_stats(_overcast(), eye=False)                               # deep convection over the core
    warm = gf.hour_stats(_overcast(radius_km=10.0, cold_k=230.0), eye=False)
    assert cold["core_cold"] >= gf.SUSTAIN > warm["core_cold"]
    hours = [(k, warm) for k in range(-12, -3)] + [(k, cold) for k in range(-3, 3)]
    f = gf.features(hours)
    assert f["g_core_run"] == 6.0 and f["g_n"] == 15.0              # sustained for the last six hours
    assert f["g_core_trend"] > 0 and f["g_eye_frac"] == 0.0
    sym_hours = [(k, gf.hour_stats(_image(east_only=k < 0), eye=False)) for k in range(-12, 3)]
    assert gf.features(sym_hours)["g_sym_trend"] > 0                 # symmetrizing
    assert gf.features([(-1, cold), (2, warm)])["g_core_run"] == 0.0  # the latest hour decides
    assert math.isnan(gf.features([(-1, cold), (0, warm)])["g_sym_trend"])   # too few points for a trend
