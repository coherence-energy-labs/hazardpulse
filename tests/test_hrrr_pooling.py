"""HRRR reduction to the 80 km grid (geometry g2/g3): every cell is pooled from the native
Lambert points that lie INSIDE it, so ``latlon_to_hrrr_cell`` and the data agree.

Before 2026-10-01 the native grid was pooled in INDEX space (31 x 28 native points
per cell) and read as a lat/lon grid -- every cell 185-375 km from its label over
tornado country. These tests pin the true-location semantics; each fails under the
index-space pooling.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.data import hrrr

CITIES = [(35.5, -97.5), (33.5, -86.8), (41.6, -93.6), (32.3, -90.2), (44.9, -93.2),
          (27.9, -82.5), (40.0, -75.2), (39.7, -105.0)]


def _nearest_native(lat, lon):
    r, c = hrrr.native_index_of_latlon(lat, lon)
    return int(round(float(r))), int(round(float(c)))


def test_peak_at_a_city_lands_in_that_citys_latlon_cell():
    for lat, lon in CITIES:
        native = np.zeros((hrrr.NATIVE_NY, hrrr.NATIVE_NX), dtype=np.float32)
        native[_nearest_native(lat, lon)] = 5000.0
        pooled = hrrr._subsample_to_grid(native.ravel(), "mlcape", mode="max")
        assert pooled.shape == (hrrr.HRRR_N_LAT, hrrr.HRRR_N_LON)
        peak = tuple(int(v) for v in np.unravel_index(np.argmax(pooled), pooled.shape))
        assert peak == hrrr.latlon_to_hrrr_cell(lat, lon), (lat, lon, peak)
        assert pooled[peak] >= 4999.0
        # index-space pooling would have put it in the 31 x 28 native block instead
        r, c = _nearest_native(lat, lon)
        assert (r // 31, c // 28) != peak


def test_stride_reads_the_native_point_nearest_each_cell_centre():
    native = np.arange(hrrr.NATIVE_NY * hrrr.NATIVE_NX, dtype=np.float64).astype(np.float32)
    out = hrrr._subsample_to_grid(native, "mlcape", mode="stride")
    rows, cols, inside = hrrr._stride_points()
    full = native.reshape(hrrr.NATIVE_NY, hrrr.NATIVE_NX)
    assert np.array_equal(out, full[rows, cols])
    # and that point really is the one nearest the centre of an in-grid cell
    i, j = 14, 29                                     # Oklahoma City's cell
    assert inside[i, j]
    assert (rows[i, j], cols[i, j]) == _nearest_native(float(hrrr.GRID_LATS[i]), float(hrrr.GRID_LONS[j]))
    assert out[0, 0] != full[0, 0]                    # not the native corner


def test_native_point_outside_the_grid_lands_in_no_cell():
    native = np.zeros((hrrr.NATIVE_NY, hrrr.NATIVE_NX), dtype=np.float32)
    native[15, 14] = 5000.0                           # ~21.5 N: south of the 25 N grid edge
    lat = float(hrrr.native_latlon()[0][15, 14])
    assert lat < hrrr.LAT_MIN
    pooled = hrrr._subsample_to_grid(native.ravel(), "mlcape", mode="max")
    assert float(np.nanmax(pooled)) < 1.0


def test_mean_pooling_a_constant_stays_constant():
    native = np.full((hrrr.NATIVE_NY, hrrr.NATIVE_NX), 12.5, dtype=np.float32)
    pooled = hrrr._subsample_to_grid(native.ravel(), "t2m", mode="max")   # t2m -> MEAN
    assert np.allclose(pooled, 12.5, atol=1e-4)


def test_mean_pooled_native_latitude_lies_inside_each_cells_band():
    """The defining property of true-location binning: a cell's mean native latitude is
    inside that cell's own latitude band (index-space pooling violates it by degrees)."""
    lat = hrrr.native_latlon()[0].astype(np.float32)
    pooled = hrrr._subsample_to_grid(lat.ravel(), "t2m", mode="max")
    inside = hrrr.domain_mask()
    lo = hrrr.LAT_MIN + np.arange(hrrr.HRRR_N_LAT)[:, None] * hrrr.GRID_DLAT
    lo = np.broadcast_to(lo, pooled.shape)
    assert inside.sum() > 1500
    assert np.all(pooled[inside] >= lo[inside] - 1e-3)
    assert np.all(pooled[inside] <= lo[inside] + hrrr.GRID_DLAT + 1e-3)


def test_nan_aware_pooling():
    native = np.full((hrrr.NATIVE_NY, hrrr.NATIVE_NX), np.nan, dtype=np.float32)
    lat, lon = CITIES[0]
    native[_nearest_native(lat, lon)] = 3000.0         # one valid point in OKC's cell
    pooled = hrrr._subsample_to_grid(native.ravel(), "mlcape", mode="max")
    cell = hrrr.latlon_to_hrrr_cell(lat, lon)
    assert pooled[cell] >= 2999.0                      # max ignores the NaNs around it
    others = np.ones(pooled.shape, bool)
    others[cell] = False
    assert np.all(np.isnan(pooled[others & hrrr.domain_mask()]))   # no data invented


def test_reduction_contract():
    assert hrrr.GEOMETRY_VERSION not in (None, "g1")          # never index-space pooling
    assert hrrr.HRRR_POOL_MODE in ("max", "stride")
    with pytest.raises(ValueError):
        hrrr._subsample_to_grid(np.zeros(hrrr.HRRR_N_LAT * hrrr.HRRR_N_LON, np.float32), "mlcape")
