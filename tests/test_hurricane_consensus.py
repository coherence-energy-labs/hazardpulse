"""Program TC1's consensus: what the weights learn, that they learn only from the past, and the arithmetic."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from hazardpulse.hurricane import consensus as cs

T0 = dt.datetime(2024, 8, 1, 0)


def _world(n_storms=6, cycles=28, sd=None, bias_km=None, seed=0, kind="track"):
    """Synthetic storms moving north-west; each member's forecast is truth plus its own noise (and bias)."""
    sd = sd or {"AVNI": 120.0, "GDMI": 30.0, "HCCA": 60.0}
    bias_km = bias_km or {}
    rng = np.random.default_rng(seed)
    decks = []
    for s in range(n_storms):
        start = T0 + dt.timedelta(days=4 * s)
        truth, carq, fcs = {}, {}, {}
        for c in range(cycles + 20):
            t = start + dt.timedelta(hours=6 * c)
            truth[t] = (15.0 + 0.15 * c, -40.0 - 0.3 * c, 40.0 + 2.0 * c)
        for c in range(cycles):
            t = start + dt.timedelta(hours=6 * c)
            carq[t] = truth[t]
            for m, s_km in sd.items():
                fc = {}
                for lead in cs.LEADS:
                    tl = t + dt.timedelta(hours=lead)
                    la, lo, v = truth[tl]
                    be, bn = bias_km.get(m, (0.0, 0.0))
                    e, n_ = rng.normal(0, s_km * lead / 48.0, 2)
                    nla, nlo = cs.shift_km(la, lo, e + be, n_ + bn)
                    fc[lead] = (nla, nlo, v + rng.normal(0, s_km / 6.0) + be / 6.0)
                fcs[(t, m)] = fc
        decks.append((cs.StormDeck.from_dicts(f"al{s + 1:02d}2024", "AL", fcs, carq), truth))
    return decks


def _errors(out, decks, lead, kind="track"):
    errs = []
    for d, truth in decks:
        for (s, t, ld), (f, _) in out.items():
            if s != d.storm or ld != lead:
                continue
            la, lo, v = truth[t + dt.timedelta(hours=lead)]
            errs.append(cs.great_circle_km(f[0], f[1], la, lo) if kind == "track" else abs(f - v))
    return float(np.mean(errs))


def test_the_weights_learn_which_aid_is_better_and_beat_the_equal_mean():
    decks = _world()
    eq = cs.run([d for d, _ in decks], cs.Config(half_life_days=None), "track")
    on = cs.run([d for d, _ in decks], cs.Config(half_life_days=30.0, shrink=1.0), "track")
    last = decks[-1][0].storm
    w = [w for (s, t, ld), (_, w) in on.items() if s == last and ld == 48][-1]
    assert w["GDMI"] > w["HCCA"] > w["AVNI"]
    # independent errors, variances 1 : 4 : 16 -> inverse-variance weights 16 : 4 : 1 of 21
    assert w["GDMI"] == pytest.approx(16 / 21, abs=0.08)
    assert _errors(on, decks, 72) < 0.8 * _errors(eq, decks, 72)


def test_a_constant_bias_is_learned_and_removed():
    decks = _world(sd={"AVNI": 40.0, "HFAI": 40.0}, bias_km={"AVNI": (150.0, 0.0), "HFAI": (150.0, 0.0)})
    plain = cs.run([d for d, _ in decks], cs.Config(half_life_days=60.0, shrink=1.0, prior_n=2.0), "track")
    fixed = cs.run([d for d, _ in decks], cs.Config(half_life_days=60.0, shrink=1.0, prior_n=2.0, debias=True), "track")
    assert _errors(fixed, decks, 48) < 0.6 * _errors(plain, decks, 48)


def test_a_forecast_never_sees_the_future():
    decks = _world(n_storms=3)
    ds = [d for d, _ in decks]
    cfg = cs.Config(half_life_days=30.0, shrink=0.5, debias=True, storm_boost=4.0)
    before = cs.run(ds, cfg, "track")
    cut = T0 + dt.timedelta(days=6)
    for d in ds:                                   # rewrite every analysis after the cut
        for i, t in enumerate(d.cycles):
            if t > cut:
                d.carq[i] += (3.0, -3.0, 50.0)
    after = cs.run(ds, cfg, "track")
    early = [k for k in before if k[1] <= cut]
    assert early and all(before[k] == after[k] for k in early)
    assert any(before[k] != after[k] for k in before if k[1] > cut)       # and the check can fail


def test_the_equal_weight_reference_is_the_plain_mean_and_one_member_is_not_a_consensus():
    m = cs.OnlineConsensus(cs.Config(half_life_days=None), "track")
    (lat, lon), w = m.combine(T0, "AL", "al012024", 24, {"AVNI": (20.0, -60.0, 50), "GDMI": (20.0, -59.0, 60)})
    assert w == {"AVNI": 0.5, "GDMI": 0.5}
    assert lat == pytest.approx(20.0, abs=0.01) and lon == pytest.approx(-59.5, abs=1e-6)
    assert m.combine(T0, "AL", "al012024", 24, {"AVNI": (20.0, -60.0, 50)}) == (None, {})
    iv = cs.OnlineConsensus(cs.Config(half_life_days=None), "intensity")
    assert iv.combine(T0, "AL", "x", 24, {"DSHP": (None, None, 50), "LGEM": (None, None, 70)})[0] == 60.0
    # the dateline: -179.5 and 179.5 average to 180, not 0
    (_, lon), _ = m.combine(T0, "AL", "x", 24, {"AVNI": (10.0, -179.5, 1), "GDMI": (10.0, 179.5, 1)})
    assert abs(abs(lon) - 180.0) < 1e-6


def test_intensity_weights_learn_too():
    decks = _world(sd={"DSHP": 90.0, "HFAI": 30.0})
    on = cs.run([d for d, _ in decks], cs.Config(half_life_days=30.0, shrink=1.0), "intensity")
    eq = cs.run([d for d, _ in decks], cs.Config(half_life_days=None), "intensity")
    assert _errors(on, decks, 48, "intensity") < _errors(eq, decks, 48, "intensity")


def test_minimum_variance_weights_are_non_negative_and_exact_where_known():
    w = cs._nonneg_min_variance(np.diag([1.0, 4.0]))
    assert w == pytest.approx([0.8, 0.2])
    # a highly correlated, worse aid would take a negative weight unconstrained; here it is dropped
    C = np.array([[1.0, 1.9], [1.9, 4.0]])
    w = cs._nonneg_min_variance(C)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0) and w[1] == 0.0


def test_the_deck_parser_keeps_the_first_line_of_each_lead_and_carq_at_tau_0():
    text = "\n".join([
        "AL, 09, 2026100712, 01, CARQ,   0, 222N,  939W,  40, 1002, TS,  34, NEQ,",
        "AL, 09, 2026100712, 01, CARQ,   0, 222N,  939W,  40, 1002, TS,  50, NEQ,",
        "AL, 09, 2026100712, 01, CARQ, -12, 215N,  930W,  35, 1004, TS,  34, NEQ,",
        "AL, 09, 2026100712, 03, GDMI,  24, 240N,  950W,  55,    0, TS,  34, NEQ,",
        "AL, 09, 2026100712, 03, GDMI,  24, 241N,  951W,  56,    0, TS,  50, NEQ,",
        "AL, 09, 2026100712, 03, GDMI,   6, 230N,  945W,  45,    0, TS,  34, NEQ,",
        "AL, 09, 2026100712, 03, XTRP,  24, 260N,  960W,   0,    0, XX,   0, NEQ,",
        "AL, 09, 2026100712, 03, AVNI,  24, 239N,  949W,   0,    0, XX,   0, NEQ,",
    ])
    d = cs.parse_deck("al092026", text, cs.TRACK_MEMBERS)
    t = dt.datetime(2026, 10, 7, 12)
    assert d.basin == "AL" and d.analysis_times == [t] and d.analysis(t) == (22.2, -93.9, 40.0)
    assert d.get(t, "GDMI", 24) == (24.0, -95.0, 55.0)               # the first of the two 24-h lines
    assert all(d.get(t, "GDMI", ld) is None for ld in cs.LEADS if ld != 24)   # tau 6 is not a program lead
    la, lo, v = d.get(t, "AVNI", 24)
    assert (la, lo) == (23.9, -94.9) and math.isnan(v)                # ATCF's 0 kt = no intensity
    assert "XTRP" not in d.techs and d.get(t, "XTRP", 24) is None


def test_geometry_round_trips():
    la, lo = cs.shift_km(25.0, -80.0, 100.0, -50.0)
    e = cs.error_vector_km(la, lo, 25.0, -80.0)
    assert e == pytest.approx([100.0, -50.0], rel=2e-3)
    assert cs.great_circle_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi * cs.EARTH_KM / 180, rel=1e-9)
