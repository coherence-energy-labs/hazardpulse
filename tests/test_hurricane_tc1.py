"""Program TC1's scoring rules: NHC's verification rules, whose area a forecast is, crossing storms, the paired test."""
from __future__ import annotations

import datetime as dt
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane import consensus as cs

ROOT = Path(__file__).resolve().parents[1]
T = dt.datetime(2022, 7, 1, 0)


@pytest.fixture(scope="module")
def tc():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("hurricane_tc1_t", ROOT / "scripts" / "hurricane_tc1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_forecast_belongs_to_the_area_the_storm_is_in(tc):
    assert tc.area_at(25.0, -75.0) == "AL" and tc.area_at(25.0, -95.0) == "AL"        # the Gulf
    assert tc.area_at(15.0, -105.0) == "EP" and tc.area_at(11.0, -88.0) == "EP"       # off Mexico, off Nicaragua
    assert tc.area_at(12.0, -82.0) == "AL"                                            # the western Caribbean
    assert tc.area_at(15.0, -141.0) == "CP" and tc.area_at(15.0, 179.0) == "CP"       # CPHC's, never scored


def test_nhc_verifies_only_a_cyclone_at_both_ends(tc):
    tr = {T: (15.0, -50.0, 40.0, "TS"), T + dt.timedelta(hours=24): (16.0, -52.0, 50.0, "TS"),
          T + dt.timedelta(hours=48): (17.0, -54.0, 45.0, "EX")}
    fc = (16.5, -52.0, 60.0)
    assert tc.verify(tr, T, 24, fc, "track") == pytest.approx(cs.great_circle_km(16.5, -52.0, 16.0, -52.0) / cs.KM_PER_NM)
    assert tc.verify(tr, T, 24, fc, "intensity") == 10.0
    assert tc.verify(tr, T, 48, fc, "track") is None                       # extratropical at the end
    assert tc.verify({**tr, T: (15.0, -50.0, 25.0, "LO")}, T, 24, fc, "track") is None   # a disturbance at the start
    assert tc.verify(tr, T, 24, (16.5, -52.0, math.nan), "intensity") is None
    assert tc.verify(tr, T, 72, fc, "track") is None                       # no best track at the end


def test_a_deck_renamed_after_a_crossing_verifies_against_the_track_it_follows(tc):
    """Bonnie 2022 was AL02, then EP04; IBTrACS keeps the whole track under AL02."""
    bonnie = {T + dt.timedelta(hours=6 * i): (11.0 + 0.1 * i, -88.0 - 1.0 * i, 60.0, "TS") for i in range(12)}
    other = {T + dt.timedelta(hours=6 * i): (15.0, -120.0, 40.0, "TS") for i in range(12)}
    truth = {"AL022022": bonnie, "EP052022": other}
    carq = {t: v[:3] for t, v in list(bonnie.items())[4:]}
    d = cs.StormDeck.from_dicts("ep042022", "EP", {}, carq)
    assert tc.truth_for(d, truth) is bonnie
    assert tc.truth_for(cs.StormDeck.from_dicts("ep052022", "EP", {}, carq), truth) is other     # its own id wins
    few = cs.StormDeck.from_dicts("ep042022", "EP", {}, dict(list(carq.items())[:2]))
    assert tc.truth_for(few, truth) == {}                                     # two fixes are not a match


def test_the_paired_difference_is_per_lead_on_shared_cases_then_averaged(tc):
    a = {24: {("s1", T): 10.0, ("s2", T): 30.0, ("s3", T): 99.0}, 48: {("s1", T): 40.0}}
    b = {24: {("s1", T): 20.0, ("s2", T): 30.0}, 48: {("s1", T): 50.0, ("s2", T): 1.0}}
    p = tc.paired(a, b, (24, 48), 0.95)
    assert p["per_lead"]["24"]["d"] == pytest.approx(-5.0) and p["per_lead"]["24"]["n"] == 2    # s3 not shared
    assert p["per_lead"]["48"]["d"] == pytest.approx(-10.0) and p["per_lead"]["48"]["n"] == 1
    assert p["mean_over_leads"]["d"] == pytest.approx(-7.5)
    lo, hi = p["mean_over_leads"]["ci"]
    assert lo <= -7.5 <= hi
    assert tc.paired({24: {}}, {24: {}}, (24,), 0.95) == {"n": 0}


def test_the_grid_is_the_registered_one(tc):
    assert len(tc.GRID) == 24 and len(set(tc.GRID)) == 24
    assert tc.EQUAL.half_life_days is None and not any(c.include_official for c in tc.GRID)
    assert tc.CLAIM_LEVEL == pytest.approx(1 - 0.05 / 4)
    assert set(tc.WARMUP + tc.CHOOSE + tc.DEV) == set(range(2020, 2026)) and not set(tc.CHOOSE) & set(tc.DEV)
