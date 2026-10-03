"""IR structure features (hurricane RI amendment 5): hand-checkable synthetic storms."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.hurricane import ir_features as ir


def _grid(n=161, half=4.0, centre=(15.0, -60.0)):
    lat = np.linspace(centre[0] + half, centre[0] - half, n)
    lon = np.linspace(centre[1] - half, centre[1] + half, n)
    return lat, lon, centre


def test_distances_and_bearings_are_great_circle():
    lat, lon, c = _grid()
    d, b = ir.distances_km(lat, lon, c)
    i0, j0 = np.unravel_index(np.argmin(d), d.shape)
    assert d[i0, j0] < 1e-6
    # one degree of latitude due north is ~111.2 km at bearing 0
    i = np.argmin(np.abs(lat - (c[0] + 1.0)))
    assert d[i, j0] == pytest.approx(111.19, rel=1e-3) and b[i, j0] == pytest.approx(0.0, abs=1e-9)
    j = np.argmin(np.abs(lon - (c[1] + 1.0)))
    # due east along a parallel: the great circle starts poleward of east by ~sin(lat) * dlon / 2
    assert b[i0, j] == pytest.approx(math.pi / 2 - math.sin(math.radians(c[0])) * math.radians(1.0) / 2, abs=2e-5)


def test_a_symmetric_eyed_storm_by_hand():
    lat, lon, c = _grid()
    d, _ = ir.distances_km(lat, lon, c)
    counts = np.full(d.shape, 100, np.uint8)            # warm surroundings
    counts[d < 200] = 220                               # cold, symmetric cloud shield to 200 km
    counts[d < 20] = 120                                # a warm eye
    f = ir.static_features(counts, lat, lon, c)
    assert f["ir_asym_50_200"] == pytest.approx(0.0, abs=1e-9)          # symmetric
    assert f["ir_mean_50_200"] == pytest.approx(220.0) and f["ir_std_50_200"] == pytest.approx(0.0)
    assert f["ir_cold_50_200"] == 1.0 and f["ir_vcold_0_100"] == pytest.approx(np.mean(counts[d < 100] >= 215))
    assert f["ir_mean_200_300"] == pytest.approx(100.0)
    assert f["ir_eye"] == pytest.approx(np.mean(counts[(d >= 25) & (d < 75)]) - 120.0)
    assert f["ir_eye"] == pytest.approx(100.0)
    assert f["ir_max_0_50"] == 220.0


def test_a_lopsided_storm_is_asymmetric_and_missing_data_gives_nan():
    lat, lon, c = _grid()
    d, b = ir.distances_km(lat, lon, c)
    counts = np.full(d.shape, 100, np.uint8)
    counts[(d < 200) & (b < math.pi)] = 220             # cloud only on the eastern half
    f = ir.static_features(counts, lat, lon, c)
    assert f["ir_asym_50_200"] == pytest.approx(60.0, abs=1.0)          # octant means 220 x4, 100 x4 -> std 60
    gone = counts.copy()
    gone[d < 300] = ir.MISSING
    g = ir.static_features(gone, lat, lon, c)
    assert all(math.isnan(g[n]) for n in ("ir_mean_0_50", "ir_mean_50_200", "ir_asym_50_200", "ir_eye"))


def test_trends_are_now_minus_six_hours_earlier_and_a_missing_image_is_nan():
    lat, lon, c = _grid()
    d, _ = ir.distances_km(lat, lon, c)
    early = {"counts": np.where(d < 200, 180, 100).astype(np.uint8), "lat": lat, "lon": lon, "centre": c}
    late = {"counts": np.where(d < 200, 220, 100).astype(np.uint8), "lat": lat, "lon": lon, "centre": c}
    f = ir.features(late, early)
    assert set(f) == set(ir.IR_NAMES) and len(ir.IR_NAMES) == 14
    assert f["d_ir_mean_50_200"] == pytest.approx(40.0) and f["d_ir_cold_50_200"] == pytest.approx(1.0)
    g = ir.features(late, None)
    assert math.isnan(g["d_ir_mean_50_200"]) and g["ir_mean_50_200"] == pytest.approx(220.0)
    assert all(math.isnan(v) for v in ir.features(None, None).values())
