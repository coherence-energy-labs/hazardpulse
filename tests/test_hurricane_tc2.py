"""TC2's rule (TC1 program amendment 2): the curve brackets the median, TC1 inside the bracket is untouched, outside it
moves to the nearest edge, the move ramps in over 24 h and decays as TC1 decays the storm."""
from __future__ import annotations

import math

import pytest

from hazardpulse.hurricane import tc2


def test_the_bracket_reads_the_median_off_the_exceedance_curve():
    curve = {15: 0.9, 20: 0.8, 25: 0.7, 30: 0.66, 35: 0.4, 40: 0.2, 45: 0.1}       # Isaias, 7 Oct 06Z-like
    assert tc2.bracket(curve) == (30.0, 35.0)
    assert tc2.bracket({k: 0.1 for k in tc2.THRESHOLDS_KT}) == (-math.inf, 15.0)  # a steady storm: median < 15
    assert tc2.bracket({k: 0.9 for k in tc2.THRESHOLDS_KT}) == (45.0, math.inf)
    # a non-monotone curve is made non-increasing first (a running minimum)
    assert tc2.monotone({15: 0.4, 20: 0.6, 25: 0.3}) == {15: 0.4, 20: 0.4, 25: 0.3}
    assert tc2.bracket({"15": 0.6, "20": 0.4}) == (15.0, 20.0)                    # string keys, as records store them


def test_tc1_inside_the_bracket_is_untouched_outside_moves_to_the_nearest_edge():
    assert tc2.shift_24h(32.0, 30.0, 35.0) == 0.0
    assert tc2.shift_24h(18.0, 30.0, 35.0) == 12.0        # raised to the lower edge
    assert tc2.shift_24h(40.0, 30.0, 35.0) == -5.0        # lowered to the upper edge
    assert tc2.shift_24h(25.0, -math.inf, 15.0) == -10.0  # a rise our model calls unlikely


def test_the_shift_ramps_in_over_24h_and_decays_as_tc1_decays_the_storm():
    tc1 = {12: 45.0, 24: 50.0, 36: 52.0, 48: 50.0, 72: 30.0, 120: 20.0}
    curve = {15: 0.9, 20: 0.8, 25: 0.7, 30: 0.66, 35: 0.4, 40: 0.2, 45: 0.1}
    out, info = tc2.project(tc1, 35.0, curve)                    # TC1: +15 kt; our median: 30-35 kt -> +15 kt more
    assert info["applied"] and info["shift_24h"] == pytest.approx(15.0)
    assert out[12] == pytest.approx(45.0 + 7.5) and out[24] == pytest.approx(65.0)
    assert out[36] == pytest.approx(52.0 + 15.0)                  # TC1 holds the storm: the shift persists (capped at 1)
    assert out[48] == pytest.approx(50.0 + 15.0)
    f72 = (30.0 - tc2.BACKGROUND_KT) / (50.0 - tc2.BACKGROUND_KT)  # TC1 decays it toward the background
    assert out[72] == pytest.approx(30.0 + 15.0 * f72)
    assert out[120] == pytest.approx(20.0)                        # below the background: nothing left of the shift


def test_no_curve_or_no_analysis_means_tc2_is_tc1():
    tc1 = {12: 45.0, 24: 50.0}
    assert tc2.project(tc1, 35.0, None)[0] == tc1
    assert tc2.project(tc1, None, {15: 0.9})[0] == tc1
    assert tc2.project(tc1, float("nan"), {15: 0.9})[0] == tc1
    same, info = tc2.project(tc1, 35.0, {15: 0.6, 20: 0.4})      # TC1 +15 inside [15, 20)
    assert same == tc1 and not info["applied"]


def test_tc2b_tapers_the_shift_to_zero_by_72h():
    tc1 = {12: 45.0, 24: 50.0, 48: 50.0, 72: 50.0, 96: 50.0}
    curve = {15: 0.9, 20: 0.8, 25: 0.7, 30: 0.66, 35: 0.4, 40: 0.2, 45: 0.1}
    b, _ = tc2.project(tc1, 35.0, curve, tc2.TC2B_TAPER_END_H)
    full, _ = tc2.project(tc1, 35.0, curve)
    assert b[12] == full[12] and b[24] == full[24] == pytest.approx(65.0)    # the first day is TC2's
    assert b[48] == pytest.approx(50.0 + 15.0 * 0.5)                          # half the shift at 48 h
    assert b[72] == pytest.approx(50.0) and b[96] == pytest.approx(50.0)      # TC1 from 72 h
    assert full[72] == pytest.approx(65.0)                                    # TC2 kept it (the falsified part)


def test_the_second_day_rule_reproduces_amendment_7s_arithmetic():
    """Killed (amendment 7 outcome), kept so the recorded test stays reproducible: none at 24 h, full at 48 h, half at
    72 h, nothing from 96 h, times the decay factor."""
    base = {24: 60.0, 48: 70.0, 72: 70.0, 96: 70.0}
    curve48 = {15: 0.9, 20: 0.8, 25: 0.7, 30: 0.6, 35: 0.55, 40: 0.3, 45: 0.2, 50: 0.1}   # median 35-40 kt
    out, info = tc2.project_second_day(base, 25.0, curve48)                 # base +45 kt -> clipped to +40
    assert info["applied"] and info["shift_48h"] == pytest.approx(-5.0)
    assert out[24] == 60.0 and out[48] == pytest.approx(65.0)
    assert out[72] == pytest.approx(70.0 - 2.5) and out[96] == pytest.approx(70.0)
