"""Storm feature engine v3: labels, storm-scale geometry, coherence-at-scale invariants."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from hazardpulse.data import hrrr as H
from hazardpulse.tornado import storm_features as sf

UTC = dt.timezone.utc


def _storm(lat=35.0, lon=-97.0, me=10.0, ms=-10.0, size=100.0, **kw):
    s = {"id": "1", "lat": lat, "lon": lon, "motion_east": me, "motion_south": ms, "size": size,
         "maxllaz": 0.006, "ps_tor": 40.0, "ps": 60.0}
    s.update(kw)
    return s


def _rep(lat, lon, minutes_after, t0, mag=1):
    return {"slat": lat, "slon": lon, "time_utc": t0 + 60.0 * minutes_after, "mag": mag}


T0 = dt.datetime(2024, 4, 27, 22, 0, tzinfo=UTC).timestamp()


FAM = {f: i for i, f in enumerate(sf.LABEL_FAMILIES)}


def test_per_storm_label_follows_the_storm_motion():
    s = _square_storm(half_deg=0.05, me=15.0, ms=0.0)  # 15 m/s eastward
    # a report 45 min later, 40.5 km east (where the storm will be): every storm family yes
    lat, lon = 35.0, -97.0 + 40.5 / (111.32 * math.cos(math.radians(35.0)))
    lab, ef, lead = sf.labels(s, T0, [_rep(lat, lon, 45, T0, mag=3)])
    names = dict(zip(sf.LABEL_NAMES, lab))
    for fam in ("storm", "centroid", "poly5", "track10"):
        assert names[f"{fam}_60"] == 1 and names[f"{fam}_90"] == 1 and names[f"{fam}_30"] == 0, fam
    assert names["nbhd_60"] == 0  # 40.5 km from where the storm IS now
    assert ef[FAM["centroid"]] == 3.0 and lead[FAM["centroid"]] == pytest.approx(45.0)
    assert ef[FAM["nbhd"]] == -1.0 and not np.isfinite(lead[FAM["nbhd"]])


def test_storm_alias_is_the_declared_primary_family():
    assert sf.PRIMARY_FAMILY == "track10"
    s = _square_storm(half_deg=0.05, me=0.0, ms=0.0)
    for minutes in (10, 50, 80):
        lab, _, _ = sf.labels(s, T0, [_rep(35.0, -97.0 + 0.11, minutes, T0)])
        names = dict(zip(sf.LABEL_NAMES, lab))
        for h in sf.HORIZONS_MIN:
            assert names[f"storm_{h}"] == names[f"track10_{h}"]


def test_neighbour_tornado_is_not_this_storms_label():
    s = _square_storm(half_deg=0.05, me=0.0, ms=0.0)
    lab, _, _ = sf.labels(s, T0, [_rep(35.25, -97.0, 20, T0)])  # ~28 km north, storm not moving
    names = dict(zip(sf.LABEL_NAMES, lab))
    assert names["nbhd_60"] == 1
    assert names["storm_60"] == 0 and names["centroid_60"] == 0 and names["poly10_60"] == 0


def test_tracked_label_uses_where_the_storm_went_not_where_its_motion_said():
    # The observation claims eastward motion; the archived track went NORTH. A tornado under the
    # real track at +40 min is this storm's (track) but not its advected guess (centroid/poly).
    s0 = _square_storm(half_deg=0.05, me=15.0, ms=0.0)
    north = _square_storm(half_deg=0.05, lat=35.36, me=15.0, ms=0.0)     # 40 km north at +40 min
    track = [(T0, s0), (T0 + 2400.0, north)]
    lab, ef, lead = sf.labels(s0, T0, [_rep(35.36, -97.0, 40, T0, mag=2)], track=track)
    names = dict(zip(sf.LABEL_NAMES, lab))
    assert names["track5_60"] == 1 and names["track10_60"] == 1 and names["storm_60"] == 1
    assert names["centroid_60"] == 0 and names["poly10_60"] == 0
    assert ef[FAM["track10"]] == 2.0 and lead[FAM["track10"]] == pytest.approx(40.0)
    # without the track only the advected fallback is known, and it misses
    lab_nt, _, _ = sf.labels(s0, T0, [_rep(35.36, -97.0, 40, T0, mag=2)])
    assert dict(zip(sf.LABEL_NAMES, lab_nt))["track10_60"] == 0


def test_tracked_distance_falls_back_to_the_last_slot_when_the_track_ends():
    s0 = _square_storm(half_deg=0.05, me=10.0, ms=0.0)
    ring = sf.polygon_ring_km(s0)
    lon_far = -97.0 + 30.0 / (111.32 * math.cos(math.radians(35.0)))
    # no slot within 15 min of the report: advect the last slot before it (30 km in 50 min)
    d = sf.tracked_distance_km([(T0, s0)], 35.0, lon_far, T0 + 3000.0)
    assert d == sf.advected_polygon_distance_km(s0, ring, 35.0, lon_far, 3000.0) == 0.0
    assert sf.tracked_distance_km([(T0 + 4000.0, s0)], 35.0, -97.0, T0) == float("inf")  # track starts after


def test_past_and_late_reports_never_label():
    s = _square_storm(half_deg=0.05, me=0.0, ms=0.0)
    lab, ef, lead = sf.labels(s, T0, [_rep(35.0, -97.0, -5, T0), _rep(35.0, -97.0, 95, T0)])
    assert lab.sum() == 0 and (ef == -1.0).all() and not np.isfinite(lead).any()


def _square_storm(half_deg=0.1, lat=35.0, lon=-97.0, me=0.0, ms=0.0):
    ring = [[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
            [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg], [lon - half_deg, lat - half_deg]]
    return _storm(lat=lat, lon=lon, me=me, ms=ms, geometry={"type": "Polygon", "coordinates": [ring]})


def test_polygon_distance_is_zero_inside_and_euclidean_outside():
    s = _square_storm()
    ring = sf.polygon_ring_km(s)
    half_y = 0.1 * 111.32
    assert sf.point_polygon_distance_km(ring, 0.0, 0.0) == 0.0
    assert sf.point_polygon_distance_km(ring, 0.0, half_y - 0.5) == 0.0
    assert sf.point_polygon_distance_km(ring, 0.0, half_y + 3.0) == pytest.approx(3.0, abs=1e-9)
    # beyond a corner: distance to the corner itself
    half_x = 0.1 * 111.32 * math.cos(math.radians(35.0))
    assert sf.point_polygon_distance_km(ring, half_x + 3.0, half_y + 4.0) == pytest.approx(5.0, abs=1e-9)


def test_polygon_label_geometry_advects_with_the_storm():
    s = _square_storm(me=10.0, ms=0.0)          # 10 m/s east = 36 km/h
    ring = sf.polygon_ring_km(s)
    half_x = 0.1 * 111.32 * math.cos(math.radians(35.0))
    lon_far = -97.0 + (half_x + 30.0) / (111.32 * math.cos(math.radians(35.0)))   # 30 km east of the east edge
    assert sf.advected_polygon_distance_km(s, ring, 35.0, lon_far, 0.0) == pytest.approx(30.0, abs=0.05)
    assert sf.advected_polygon_distance_km(s, ring, 35.0, lon_far, 3600.0) == 0.0    # 36 km in 60 min: 6 km inside
    assert sf.polygon_ring_km(_storm()) is None                                       # no polygon -> None, not a guess


def test_garbage_motion_is_clipped():
    ue, vn = sf.storm_motion(_storm(me=0.0, ms=-240.0))
    assert math.hypot(ue, vn) == pytest.approx(sf.MAX_MOTION_MS)


def test_track_features_see_the_trend():
    hist = [_storm(maxllaz=v) for v in (0.002, 0.004, 0.006, 0.008)]
    e = dict(zip(sf.TRACK_NAMES, sf.block_e(hist[-1], hist, 30.0)))
    assert e["age_min"] == 90.0 and e["maxllaz_max"] == pytest.approx(0.008)
    assert e["maxllaz_slope"] == pytest.approx(0.002, rel=1e-6)
    assert e["maxllaz_delta"] == pytest.approx(0.002, rel=1e-6)


def test_feature_vector_shape_and_missing_inputs_are_nan():
    v = sf.feature_vector(_storm(), [_storm()], 30.0)
    assert v.shape == (sf.N_FEATURES,) and len(sf.FEATURE_NAMES) == sf.N_FEATURES
    lo, hi = sf.BLOCKS["H9"]
    assert np.isnan(v[lo:hi]).all()  # no storm-scale analysis given -> NaN, never 0


def test_window_stat_matches_brute_force():
    x = np.random.RandomState(1).rand(23, 31)
    from numpy.lib.stride_tricks import sliding_window_view as sw
    assert np.array_equal(sf._window_stat(x, 2, "max"), sw(np.pad(x, 2, mode="edge"), (5, 5)).max(axis=(-1, -2)))


def _synthetic_blocks(seed=0):
    k = sf.LCC_K
    ny, nx = H.NATIVE_NY // k, H.NATIVE_NX // k
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    bump = np.exp(-(((yy - 140) / 12.0) ** 2 + ((xx - 300) / 20.0) ** 2)).astype(np.float32)
    f = lambda a: np.broadcast_to(np.asarray(a, np.float32), (ny, nx)).copy()
    return {"mlcape": f(3000 * bump), "mucape": f(3500 * bump), "mlcin": f(-20 - 30 * (1 - bump)),
            "srh_01": f(250 * bump), "srh_03": f(350 * bump), "refc": f(50 * bump),
            "ushear_01": f(8.0), "vshear_01": f(12.0), "ushear_06": f(20.0), "vshear_06": f(5.0),
            "ustorm": f(10.0), "vstorm": f(5.0), "t2m": f(300 - 5 * (1 - bump)),
            "td2m": f(292 - 8 * (1 - bump)), "pwat": f(30 * bump + 10)}


def test_storm_scale_fields_are_finite_and_peak_where_the_storm_environment_is():
    fields = sf.analysis_fields(_synthetic_blocks())
    for k, v in fields.items():
        assert np.isfinite(v).all(), k
    for scale in (9, 27, 81):
        tau = fields[f"pde_{scale}_tau"]
        r, c = np.unravel_index(np.argmax(tau), tau.shape)
        assert abs(r - 140) <= 6 and abs(c - 300) <= 6, (scale, r, c)
    # the PDE and the Gaussian control are genuinely different fields
    assert not np.allclose(fields["pde_27_tau"], fields["gauss_27_tau"])
