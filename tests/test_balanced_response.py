"""H8: the coherence equation as the vortex's balanced response to heating -- physics, before any outcome."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.hurricane import balanced_response as br


def _crop(blob_r_km: float, lat0: float = 20.0, lon0: float = -60.0, cold: float = 230.0, width_km: float = 24.0):
    """A synthetic GMGSI crop (+-4 deg at 0.072 deg): warm background, one cold ring at ``blob_r_km``."""
    step = 0.072
    la = lat0 + np.arange(-4, 4 + 1e-9, step)
    lo = lon0 + np.arange(-4, 4 + 1e-9, step)
    dist, _ = br.ir.distances_km(la, lo, (lat0, lon0))      # crops store 1-D lat and lon axes
    counts = np.full(dist.shape, 150, np.uint8)
    counts[np.abs(dist - blob_r_km) <= width_km / 2] = cold
    return {"counts": counts, "lat": la, "lon": lo, "centre": (lat0, lon0)}


def test_the_vortex_core_has_a_short_coherence_length_and_the_far_field_tends_to_c_over_f():
    lat, vmax, rmw = 20.0, 100.0, 30.0
    f = br.coriolis(lat)
    i_core = br.inertial_frequency(np.array([5.0, 15.0, 29.0]), vmax, rmw, lat)
    assert np.allclose(i_core, f + 2 * vmax * br.KT / (rmw * 1000.0))          # solid body: constant inside R
    ell = br.coherence_length_km(np.array([10.0, 3000.0]), vmax, rmw, lat)
    assert ell[0] < 20.0                                                     # ~14 km inside a 100-kt core
    assert abs(ell[1] - br.GRAVITY_WAVE_SPEED_MS / f / 1000.0) / ell[1] < 0.05  # -> c / f far away
    weak = br.coherence_length_km(np.array([10.0]), 30.0, 278.0, lat)[0]
    assert weak > 10 * ell[0]                                                # a weak, broad system holds little


def test_heating_inside_the_radius_of_maximum_wind_is_held_far_better_than_the_same_heating_outside():
    """The mechanism the feature exists for (Schubert and Hack 1982): where the heating sits, not how much."""
    vmax, rmw_nm, lat = 90.0, 15.0, 20.0                     # RMW ~28 km
    inside = br.static_response(_crop(15.0), vmax, rmw_nm, lat)
    outside = br.static_response(_crop(150.0), vmax, rmw_nm, lat)
    assert inside["h8_core_heating"] > 0 and outside["h8_core_heating"] == 0.0
    # the core retains the inner heating in balance; heating 150 km out barely reaches it
    assert inside["h8_core_balanced"] > 10 * outside["h8_core_balanced"]
    assert 0.5 < inside["h8_core_retention"] < 1.2


def test_a_stronger_tighter_vortex_holds_the_same_heating_more_coherently():
    crop = _crop(20.0)
    weak = br.static_response(crop, 40.0, 60.0, 20.0)
    strong = br.static_response(crop, 110.0, 12.0, 20.0)
    assert strong["h8_core_retention"] > weak["h8_core_retention"]


def test_no_vortex_or_no_coverage_gives_nan_never_a_number_that_looks_like_data():
    crop = _crop(20.0)
    assert all(math.isnan(v) for v in br.static_response(crop, float("nan"), 15.0, 20.0).values())
    assert all(math.isnan(v) for v in br.static_response(crop, 90.0, 0.0, 20.0).values())
    blank = dict(crop, counts=np.full_like(crop["counts"], br.ir.MISSING))      # 255 = no data (and the coldest)
    assert all(math.isnan(v) for v in br.static_response(blank, 90.0, 15.0, 20.0).values())
    assert all(math.isnan(v) for v in br.static_response(None, 90.0, 15.0, 20.0).values())


def test_the_six_hour_change_is_newer_minus_older():
    f = br.features(_crop(15.0), _crop(150.0), 90.0, 15.0, 20.0)
    older = br.static_response(_crop(150.0), 90.0, 15.0, 20.0)
    assert f["h8_core_balanced_d6"] == pytest.approx(f["h8_core_balanced"] - older["h8_core_balanced"])
    assert set(f) == set(br.H8_NAMES)


def test_the_radius_of_maximum_wind_is_read_from_the_cycles_carq_analysis():
    text = "\n".join([
        "AL, 09, 2026100712, 03, OFCL,   0, 222N,  939W,  40, 1002, TS,  34, NEQ, 60, 50, 0, 40, 1008, 150,  25,",
        "AL, 09, 2026100712, 01, CARQ,   0, 222N,  939W,  40, 1002, TS,  34, NEQ, 60, 50, 0, 40, 1008, 150,  20,",
        "AL, 09, 2026100718, 01, CARQ,   0, 225N,  933W,  50,  995, TS,  34, NEQ, 70, 60, 0, 50, 1008, 160,   0,",
    ])
    assert br.carq_rmw_nm(text, "2026100712") == 20.0          # CARQ, not OFCL
    assert math.isnan(br.carq_rmw_nm(text, "2026100718"))      # ATCF's 0 = unknown
    assert math.isnan(br.carq_rmw_nm(text, "2026100800"))
