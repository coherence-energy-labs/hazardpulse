"""Amendment E5's diagnostic helpers (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13.2): which events are mainshocks,
and what counts as the outcome."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0


def _e5():
    sys.path.insert(0, str(ROOT / "scripts" / "earthquake_program"))
    spec = importlib.util.spec_from_file_location("aftershock_e5_t", ROOT / "scripts" / "earthquake_program" / "aftershock_e5.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _raw(rows):
    a = np.array(rows, dtype=float)
    return {"t": a[:, 0], "lat": a[:, 1], "lon": a[:, 2], "mag": a[:, 3], "depth": a[:, 4]}


def test_a_mainshock_is_shallow_m7_with_no_larger_event_close_before():
    e5 = _e5()
    t0 = 1.7e9
    raw = _raw([
        (t0, 7.5, -80.7, 7.7, 12.0),                  # 0: a mainshock
        (t0 + 2 * DAY, 7.6, -80.8, 7.1, 10.0),        # 1: an M7.1 two days after a larger event 10 km away: not one
        (t0 + 5 * DAY, 30.0, 140.0, 7.2, 400.0),      # 2: deep: not one
        (t0 + 9 * DAY, -30.0, -70.0, 7.0, 20.0),      # 3: shallow, nothing before it nearby: one
        (t0 + 9.5 * DAY, 30.0, 140.0, 7.0, 20.0),     # 4: shallow, but the deep M7.2 4.5 days before (any depth counts)
        (t0 + 40 * DAY, 7.5, -80.7, 7.2, 10.0),       # 5: 40 days after the M7.7: past the 30-day rule, one
    ])
    assert e5.mainshocks(raw, t0 - 1, t0 + 100 * DAY) == [0, 3, 5]


def test_the_outcome_is_an_m6_in_the_same_cell_within_30_days_after_issue():
    e5 = _e5()
    t0 = 1.7e9
    raw = _raw([(t0, 7.59, -80.77, 7.7, 12.0), (t0 + 2.5 * 3600, 7.72, -81.47, 6.6, 10.0),
                (t0 + 40 * DAY, 7.5, -80.6, 6.5, 10.0)])
    cell = int(e5.of.cell_index(7.59, -80.77))
    assert e5.outcome(raw, cell, t0 + 0.5 * 3600)               # the M6.6 at +2.5 h, same cell
    assert not e5.outcome(raw, cell, t0 + 6 * 3600)             # after it, nothing else within 30 days
    assert e5.outcome(raw, cell, t0 + 20 * DAY)                 # the +40 d M6.5 is within 30 days of a +20 d issue
    assert not e5.outcome(raw, int(e5.of.cell_index(30.0, 140.0)), t0)


def test_e7f_domain_is_the_3x3_around_a_shallow_m7_from_half_an_hour_to_a_week_after():
    """Section 15.7: E7f applies E7 only where it was fitted -- the 3x3 cells around a shallow M7+, 0.5-168 h after."""
    e5 = _e5()
    t0 = 20000.0                                               # days
    m77 = (t0, 7.587, -80.769, 12.6, 7.7)
    m66 = (t0 + 0.1, -30.0, -70.0, 10.0, 6.6)                  # an M6.6 elsewhere never opens the domain
    shallow = np.array([m77, m66])
    home = int(e5.of.cell_index(7.587, -80.769))
    nine = set(e5._nbr3(home))
    assert len(nine) == 9 and home in nine
    for lag_h in (0.5, 6.0, 24.0, 168.0):                      # the four fitted lags, both edges included
        assert e5.fitted_domain(t0 + lag_h / 24.0, shallow) == nine, lag_h
    for lag_h in (0.0, 0.25, 169.0, 30 * 24.0):                # before half an hour, or after a week: outside
        assert e5.fitted_domain(t0 + lag_h / 24.0, shallow) == set(), lag_h
    far = int(e5.of.cell_index(7.587 + 4.0, -80.769))          # two cells north: not in the 3x3
    assert far not in e5.fitted_domain(t0 + 1.0, shallow)
    assert e5.fitted_domain(t0 + 0.1 + 1.0, np.array([m66])) == set()
