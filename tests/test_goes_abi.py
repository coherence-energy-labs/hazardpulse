"""GOES ABI reader: the fixed-grid geometry (PUG worked example), satellite choice, and the storm-centred polar
sampling -- checked on a synthetic hurricane built in the satellite's own grid."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from hazardpulse.hurricane import goes_abi as g


def test_the_pug_worked_example_and_its_inverse():
    x, y = g.latlon_to_xy(33.846162, -84.690932, -75.0)
    assert float(x) == pytest.approx(-0.024052, abs=1e-6) and float(y) == pytest.approx(0.095340, abs=1e-6)
    lat, lon = g.xy_to_latlon(x, y, -75.0)
    assert float(lat) == pytest.approx(33.846162, abs=1e-6) and float(lon) == pytest.approx(-84.690932, abs=1e-6)
    hx, _ = g.latlon_to_xy(20.0, 105.0, -75.0)                    # the far side of the Earth
    assert np.isnan(hx)
    assert np.isnan(g.xy_to_latlon(0.2, 0.2, -75.0)[0])           # off the disk


def test_the_operational_satellite_for_each_place_and_date():
    assert g.satellite_for(dt.datetime(2024, 10, 7), -92.0) == "goes16"
    assert g.satellite_for(dt.datetime(2025, 9, 1), -60.0) == "goes19"
    assert g.satellite_for(dt.datetime(2022, 8, 1), -120.0) == "goes17"
    assert g.satellite_for(dt.datetime(2024, 8, 1), -150.0) == "goes18"
    assert g.satellite_for(dt.datetime(2024, 8, 1), -100.0) == "goes16"      # east of the split


def test_destination_and_scan_times():
    lat, lon = g.destination(20.0, -60.0, np.array([0.0, math.pi / 2]), np.array([111.195, 104.5]))
    assert lat[0] == pytest.approx(21.0, abs=1e-3) and lon[1] == pytest.approx(-59.0, abs=0.01)
    k = "ABI-L2-CMIPF/2024/281/12/OR_ABI-L2-CMIPF-M6C13_G16_s20242811210208_e20242811219527_c20242811219593.nc"
    assert g.scan_start(k) == dt.datetime(2024, 10, 7, 12, 10, 20)
    ks = [k.replace("s2024281121020", "s2024281120020"), k]
    assert g.first_scan_at_or_after(ks, dt.datetime(2024, 10, 7, 12)) == ks[0]


def test_encode_decode_round_trip_and_missing():
    bt = np.array([170.0, 200.3, 322.4, np.nan, 400.0])
    v = g.encode(bt)
    assert v[3] == g.MISSING and v[4] == 254
    back = g.decode(v)
    assert np.isnan(back[3]) and np.allclose(back[:3], bt[:3], atol=g.BT_STEP / 2 + 1e-6)   # within half a step


def _synthetic_disk(clat=20.0, clon=-60.0, ring_km=30.0, lon0=-75.0):
    """A fixed-grid patch (2 km pixels near nadir scale) holding a warm eye inside a cold ring at ``ring_km``."""
    step = 56e-6                                                   # rad, the ABI 2 km pixel
    cx, cy = g.latlon_to_xy(clat, clon, lon0)
    xs = float(cx) + (np.arange(-100, 101)) * step
    ys = float(cy) - (np.arange(-100, 101)) * step                  # y decreases with row, as in the files
    X, Y = np.meshgrid(xs, ys)
    lat, lon = g.xy_to_latlon(X, Y, lon0)
    p1, p2 = np.radians(clat), np.radians(lat)
    dl = np.radians(lon - clon)
    r = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
    # 190 K ring, 290 K environment, and an eye warmest at its centre (+6 K), as real eyes are
    bt = 290.0 - 100.0 * np.exp(-((r - ring_km) / 8.0) ** 2) + 6.0 * np.exp(-(r / 6.0) ** 2)
    d = g.FullDisk.__new__(g.FullDisk)
    d.x, d.y, d.lon0 = xs, ys, lon0
    d.cmi, d.scale, d.offset, d.fill = bt, 1.0, 0.0, -999
    d.dx, d.dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    return d


class _Var:
    """An h5py dataset stand-in: data plus attributes stored as 1-element arrays, as in the GOES files."""

    def __init__(self, data, **attrs):
        self.data = np.asarray(data)
        self.attrs = {k: np.array([v]) for k, v in attrs.items()}

    def __getitem__(self, idx):
        return self.data[idx]


def test_the_reader_opens_a_file_whose_attributes_are_one_element_arrays_with_warnings_as_errors():
    """The first runner collection failed every crop: float() of a 1-element attribute array raises in newer NumPy
    (it only warns in 2.3, where the local smoke passed). Every attribute read must survive warnings-as-errors."""
    import warnings
    n = 8
    f = {"x": _Var(np.arange(n), scale_factor=56e-6, add_offset=-0.0002),
         "y": _Var(np.arange(n)[::-1], scale_factor=56e-6, add_offset=-0.0002),
         "goes_imager_projection": _Var([0], longitude_of_projection_origin=-75.0),
         "CMI": _Var(np.full((n, n), 2000, np.int16), scale_factor=0.1, add_offset=0.0, _FillValue=-1)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = g.FullDisk(f)
    assert d.lon0 == -75.0 and d.scale == pytest.approx(0.1) and d.fill == -1 and d.dx > 0 > d.dy


def _bearing_deg(d, clat=20.0, clon=-60.0):
    X, Y = np.meshgrid(d.x, d.y)
    lat, lon = g.xy_to_latlon(X, Y, d.lon0)
    return (np.degrees(np.arctan2(np.radians(lon - clon) * np.cos(np.radians(clat)), np.radians(lat - clat))) + 360) % 360


def test_a_closed_ring_eye_is_found_around_the_eye_and_the_analysed_image_is_kept():
    ring = _synthetic_disk(ring_km=30.0)                          # a full cold ring: a real eye
    c = g.polar_eye(ring, 20.03, -60.02)                          # analysed centre ~4 km off the eye
    assert c.eye and abs(c.lat - 20.0) < 0.03 and abs(c.lon + 60.0) < 0.03
    assert g.decode(c.image)[0].mean() > 290.0                    # the feature image sits in the eye
    # the candidate is the warmest 2 km sample, ~2 km from the eye's centre, so the best ring's warmest point is a
    # few K above the 190 K ring floor
    assert c.tcw < 210.0 and c.teye - c.tcw > 80.0
    assert np.array_equal(c.analysed, ring.polar(20.03, -60.02))  # the analysed-centre image, as sampled


def test_an_exposed_centre_is_not_an_eye():
    half = _synthetic_disk(ring_km=30.0)                          # cold cloud on the east side only
    X, _ = np.meshgrid(half.x, half.y)
    half.cmi = np.where(X > float(g.latlon_to_xy(20.0, -60.0, half.lon0)[0]), half.cmi, 290.0)
    c = g.polar_eye(half, 20.03, -60.02)
    assert not c.eye and (c.lat, c.lon) == (20.03, -60.02)        # the analysed centre stays
    assert np.array_equal(c.image, c.analysed) and c.tcw > 280.0  # no ring is cold all the way round


def test_a_warm_gap_in_a_curved_band_is_not_an_eye():
    """The weak-storm defect: a band wrapped 320 degrees round a warm centre, open in one 40-degree sector. The old
    rule (warm spot + cold cloud in >= 75 % of directions) called it an eye; ADT's coldest-warmest ring cannot,
    because every ring crosses the gap."""
    band = _synthetic_disk(ring_km=30.0)
    band.cmi = np.where((_bearing_deg(band) > 160) & (_bearing_deg(band) < 200), 290.0, band.cmi)
    c = g.polar_eye(band, 20.0, -60.0)
    assert not c.eye and c.tcw > 280.0


def test_a_cold_overcast_with_a_slightly_warmer_centre_is_not_an_eye():
    cdo = _synthetic_disk(ring_km=30.0)
    X, Y = np.meshgrid(cdo.x, cdo.y)
    lat, lon = g.xy_to_latlon(X, Y, cdo.lon0)
    r = 111.2 * np.hypot(lat - 20.0, (lon + 60.0) * math.cos(math.radians(20.0)))
    cdo.cmi = np.where(r < 150.0, 200.0 + 15.0 * np.exp(-(r / 8.0) ** 2), 290.0)   # closed, cold, centre +15 K
    c = g.polar_eye(cdo, 20.0, -60.0)
    assert not c.eye and c.tcw == pytest.approx(200.0, abs=1.0) and c.teye - c.tcw < g.EYE_DELTA_K


def test_the_adt_temperatures_skip_a_ring_that_cannot_show_it_is_closed_and_the_thresholds_are_inclusive():
    bt = np.full((g.N_R, g.N_AZ), 220.0)
    bt[g.RADII_KM < 24.0] = 280.0
    bt[(g.RADII_KM >= 24.0) & (g.RADII_KM < 40.0)] = 200.0
    img = g.encode(bt)
    assert g.adt_temperatures(img)[1] == pytest.approx(200.0, abs=0.3)
    img[(g.RADII_KM >= 24.0) & (g.RADII_KM < 40.0), 5] = g.MISSING   # one missing sample on each cold ring
    assert g.adt_temperatures(img)[1] == pytest.approx(220.0, abs=0.3)
    blank = np.full((g.N_R, g.N_AZ), g.MISSING, np.uint8)
    te, tc = g.adt_temperatures(blank)
    assert math.isnan(te) and math.isnan(tc) and not g.is_eye(te, tc)
    assert g.is_eye(g.EYE_RING_MAX_K + g.EYE_DELTA_K, g.EYE_RING_MAX_K)
    assert not g.is_eye(g.EYE_RING_MAX_K + 0.6 + g.EYE_DELTA_K, g.EYE_RING_MAX_K + 0.6)
    assert not g.is_eye(g.EYE_RING_MAX_K + g.EYE_DELTA_K - 0.6, g.EYE_RING_MAX_K)


def test_the_polar_image_puts_the_ring_at_its_radius_in_every_direction():
    d = _synthetic_disk(ring_km=30.0)
    img = g.decode(d.polar(20.0, -60.0))
    assert img.shape == (g.N_R, g.N_AZ)
    ring_r = g.RADII_KM[np.nanargmin(img, axis=0)]
    assert np.all(np.abs(ring_r - 30.0) <= 2.0)                      # every azimuth: the ring within one bin
    # the 2 km bins sit at 29 and 31 km, 1 km off the ring's 30 km centre: the sampled minimum is 290 - 100 e^-(1/8)^2
    assert np.nanmin(img) == pytest.approx(290.0 - 100.0 * math.exp(-(1 / 8) ** 2), abs=0.5)
    assert img[0].mean() > 280.0
    # samples beyond the patch (its corners reach ~330 km) are missing, never a number
    far = d.polar(20.0, -60.0)[g.RADII_KM > 340]
    assert (far == g.MISSING).all()
