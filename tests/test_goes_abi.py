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


def test_an_enclosed_eye_recentres_and_an_exposed_centre_does_not():
    ring = _synthetic_disk(ring_km=30.0)                         # a full cold ring: a real eye
    img, la, lo, eye = g.polar_recentred(ring, 20.03, -60.02)     # analysed centre ~4 km off the eye
    assert eye and abs(la - 20.0) < 0.03 and abs(lo + 60.0) < 0.03
    assert g.decode(img)[0].mean() > 280.0
    half = _synthetic_disk(ring_km=30.0)                          # cold cloud on the east side only
    X, _ = np.meshgrid(half.x, half.y)
    half.cmi = np.where(X > float(g.latlon_to_xy(20.0, -60.0, half.lon0)[0]), half.cmi, 290.0)
    img2, la2, lo2, eye2 = g.polar_recentred(half, 20.03, -60.02)
    assert not eye2 and (la2, lo2) == (20.03, -60.02)              # exposed: the analysed centre stays


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
