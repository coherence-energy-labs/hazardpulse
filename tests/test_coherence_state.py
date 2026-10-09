"""A storm's coherence state: memory with forgetting, from its own past only -- pinned before any outcome."""
from __future__ import annotations

import datetime as dt
import math

import pytest

from hazardpulse.hurricane import coherence_state as cs

T = dt.datetime(2024, 9, 1, 12)


def _sig(conv, asym=10.0, hold=0.5, v=50.0):
    return cs.signals({"ir_vcold_0_100": conv, "ir_asym_50_200": asym, "h8_core_retention": hold, "v0": v})


def _hist(convs, vs=None):
    vs = vs or [50.0] * len(convs)
    return [(T - dt.timedelta(hours=6 * (len(convs) - 1 - i)), _sig(c, v=v)) for i, (c, v) in enumerate(zip(convs, vs))]


def test_the_memory_is_an_exponentially_forgotten_mean():
    h = _hist([0.0, 1.0])                                     # 6 h ago 0, now 1
    s = cs.state(h, T)
    w = math.exp(-6 / 36)
    assert s["coh_conv_36"] == pytest.approx(1.0 / (1.0 + w))
    assert s["coh_conv_12"] > s["coh_conv_36"]               # the shorter memory forgets the old zero faster
    assert s["coh_conv_rise"] == pytest.approx(1.0 - s["coh_conv_36"])
    assert s["coh_n"] == 2.0


def test_a_sustained_core_reads_differently_from_a_burst():
    sustained = cs.state(_hist([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), T)
    burst = cs.state(_hist([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]), T)
    assert sustained["coh_persist"] == 1.0 and burst["coh_persist"] == pytest.approx(1 / 7)
    assert sustained["coh_conv_36"] > burst["coh_conv_36"] and burst["coh_conv_rise"] > sustained["coh_conv_rise"]


def test_the_state_never_reads_the_future_or_beyond_its_window():
    h = _hist([0.2, 0.4, 0.9]) + [(T + dt.timedelta(hours=6), _sig(0.0))]
    assert cs.state(h, T) == cs.state(h[:3], T)
    old = [(T - dt.timedelta(hours=42), _sig(1.0))] + _hist([0.0])
    assert cs.state(old, T)["coh_conv_36"] == 0.0 and cs.state(old, T)["coh_n"] == 1.0


def test_spin_is_the_intensity_tendency_per_six_hours():
    s = cs.state(_hist([0.5, 0.5, 0.5], vs=[40.0, 50.0, 65.0]), T)
    w = math.exp(-6 / 36)
    assert s["coh_spin_36"] == pytest.approx((15.0 + 10.0 * w) / (1.0 + w))
    gap = [(T - dt.timedelta(hours=12), _sig(0.5, v=40.0)), (T, _sig(0.5, v=60.0))]
    assert cs.state(gap, T)["coh_spin_36"] == pytest.approx(10.0)      # 20 kt over 12 h = 10 per 6 h


def test_missing_signals_are_skipped_and_none_at_all_is_nan():
    h = _hist([math.nan, 0.8])
    assert cs.state(h, T)["coh_conv_36"] == 0.8
    blank = cs.state(_hist([math.nan, math.nan]), T)
    assert all(math.isnan(blank[k]) for k in ("coh_conv_12", "coh_conv_36", "coh_conv_rise", "coh_persist"))
    assert cs.signals({"ir_asym_50_200": 4.0})["sym"] == -4.0 and math.isnan(cs.signals({})["conv"])


def test_add_states_writes_each_row_from_its_own_storm_only():
    rows = [{"atcf_id": "AL012024", "dtg": "2024090106", "f": {"ir_vcold_0_100": 0.0, "v0": 40.0}},
            {"atcf_id": "AL012024", "dtg": "2024090112", "f": {"ir_vcold_0_100": 1.0, "v0": 45.0}},
            {"atcf_id": "AL022024", "dtg": "2024090112", "f": {"ir_vcold_0_100": 0.0, "v0": 30.0}}]
    assert cs.add_states(rows) == 3
    assert rows[1]["f"]["coh_n"] == 2.0 and rows[2]["f"]["coh_n"] == 1.0
    assert rows[2]["f"]["coh_conv_36"] == 0.0 and rows[0]["f"]["coh_conv_36"] == 0.0
    assert set(cs.COH_NAMES) <= set(rows[0]["f"])
