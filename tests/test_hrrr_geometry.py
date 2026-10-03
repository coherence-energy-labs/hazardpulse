"""The 80 km HRRR grid is a TRUE lat/lon grid (geometry g2) -- regression tests.

Before 2026-10-01 the native Lambert-conformal grid was pooled in index space
(31 rows x 28 columns per cell) and the result read as the regular 0.72 x 0.94
degree grid that ``latlon_to_hrrr_cell`` indexes. Rows of a Lambert grid are
not parallels: each cell's data came from a median 283 km (p90 481 km) from
where the lookup placed it -- Oklahoma City read the atmosphere at (36.31 N,
-99.95 E), 238 km away. Every HRRR and coherence feature was mislocated, in
training and live. Measured against hrrrzarr's own grid index.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.data import hrrr

CITIES = {
    "Oklahoma City": (35.47, -97.52),
    "Birmingham": (33.52, -86.80),
    "Des Moines": (41.59, -93.62),
    "Jackson MS": (32.30, -90.18),
    "Dallas": (32.78, -96.80),
    "Chicago": (41.88, -87.63),
}


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(math.radians(lon2 - lon1))
    return 6371.0 * math.acos(min(1.0, max(-1.0, c)))


def test_native_grid_matches_the_published_hrrr_corners():
    lat, lon = hrrr.native_latlon()
    assert lat.shape == (hrrr.NATIVE_NY, hrrr.NATIVE_NX)
    assert lat[0, 0] == pytest.approx(21.138123, abs=1e-9)
    assert lon[0, 0] == pytest.approx(-122.719528, abs=1e-9)
    assert lat[-1, -1] == pytest.approx(47.842195, abs=1e-5)
    assert lon[-1, -1] == pytest.approx(-60.917193, abs=1e-5)


def test_native_index_round_trips():
    lat, lon = hrrr.native_latlon()
    for la, lo in CITIES.values():
        r, c = hrrr.native_index_of_latlon(la, lo)
        rr, cc = int(round(float(r))), int(round(float(c)))
        assert _km(la, lo, lat[rr, cc], lon[rr, cc]) < 2.2  # within half a 3 km cell diagonal


@pytest.mark.parametrize("field,axis_vals,half", [
    ("lat", lambda: hrrr.GRID_LATS[:, None], hrrr.GRID_DLAT / 2),
    ("lon", lambda: hrrr.GRID_LONS[None, :], hrrr.GRID_DLON / 2),
])
def test_pooled_cells_hold_the_points_they_claim(field, axis_vals, half):
    """Mean-pool the native coordinate itself: each covered cell must return a
    value inside its own band. Under the old index-space pooling this failed by
    up to ~7 degrees."""
    lat, lon = hrrr.native_latlon()
    native = (lat if field == "lat" else lon).astype(np.float32)
    pooled = hrrr._subsample_to_grid(native.ravel(), "t2m", mode="max")  # t2m -> MEAN
    mask = hrrr.domain_mask()
    err = np.abs(pooled - axis_vals())[mask]
    assert err.max() <= half + 1e-4


@pytest.mark.parametrize("name", list(CITIES))
def test_a_peak_at_a_city_lands_in_the_citys_cell(name):
    la, lo = CITIES[name]
    r, c = hrrr.native_index_of_latlon(la, lo)
    rr, cc = int(round(float(r))), int(round(float(c)))
    native = np.zeros((hrrr.NATIVE_NY, hrrr.NATIVE_NX), dtype=np.float32)
    native[rr, cc] = 5000.0
    pooled = hrrr._subsample_to_grid(native.ravel(), "mlcape", mode="max")
    peak = tuple(int(v) for v in np.unravel_index(np.argmax(pooled), pooled.shape))
    # The peak sits at the native point nearest the city (<= 2.2 km away); it
    # must land in the cell containing THAT point (Dallas sits exactly on a
    # cell edge, so the point can be across it) -- and within one cell of the
    # city's own cell, never the hundreds of km of the old index-space pooling.
    lat, lon = hrrr.native_latlon()
    assert peak == hrrr.latlon_to_hrrr_cell(float(lat[rr, cc]), float(lon[rr, cc]))
    ci, cj = hrrr.latlon_to_hrrr_cell(la, lo)
    assert abs(peak[0] - ci) <= 1 and abs(peak[1] - cj) <= 1


def test_stride_reads_the_point_nearest_the_cell_centre():
    lat, lon = hrrr.native_latlon()
    out_lat = hrrr._subsample_to_grid(lat.astype(np.float32).ravel(), "t2m", mode="stride")
    out_lon = hrrr._subsample_to_grid(lon.astype(np.float32).ravel(), "t2m", mode="stride")
    # Cells whose centre lies inside the native grid read a point within half a
    # 3 km cell diagonal of the centre; margin cells clamp to the grid edge.
    _r, _c, inside = hrrr._stride_points()
    d = [
        _km(out_lat[i, j], out_lon[i, j], hrrr.GRID_LATS[i], hrrr.GRID_LONS[j])
        for i, j in zip(*np.nonzero(inside))
    ]
    assert inside.sum() > 0.9 * inside.size
    assert max(d) < 3.0


def test_old_index_space_pooling_was_hundreds_of_km_off():
    """The witness, kept: what the g1 cell (index block) actually contained."""
    lat, lon = hrrr.native_latlon()
    by, bx = hrrr.NATIVE_NY // hrrr.HRRR_N_LAT, hrrr.NATIVE_NX // hrrr.HRRR_N_LON
    la, lo = CITIES["Oklahoma City"]
    i, j = hrrr.latlon_to_hrrr_cell(la, lo)
    g1_lat, g1_lon = lat[i * by + by // 2, j * bx + bx // 2], lon[i * by + by // 2, j * bx + bx // 2]
    assert _km(la, lo, g1_lat, g1_lon) > 200.0


def test_every_storm_region_cell_is_covered():
    mask = hrrr.domain_mask()
    for la, lo in CITIES.values():
        assert mask[hrrr.latlon_to_hrrr_cell(la, lo)]
    # The uncovered cells are few and on the margins (open Atlantic, far NE).
    assert mask.sum() >= 0.97 * mask.size


def test_uncovered_cells_are_filled_from_a_covered_neighbour():
    native = np.full((hrrr.NATIVE_NY, hrrr.NATIVE_NX), 7.0, dtype=np.float32)
    pooled = hrrr._subsample_to_grid(native.ravel(), "t2m", mode="max")
    assert np.isfinite(pooled).all() and np.allclose(pooled, 7.0)


def test_grid_axes_are_rotated_by_the_cone_angle():
    """Derived from the projection itself: the +x grid axis points at -theta
    from true east, theta = n (lon - LoV) -- the basis of the wind rotation."""
    lat, lon = hrrr.native_latlon()
    n = math.sin(math.radians(38.5))
    for r, c in [(500, 50), (500, 900), (500, 1750), (1000, 1700)]:
        dphi = math.radians(lat[r, c + 1] - lat[r, c - 1])
        dlam = math.radians(lon[r, c + 1] - lon[r, c - 1]) * math.cos(math.radians(lat[r, c]))
        theta = math.degrees(n * math.radians(lon[r, c] + 97.5))
        assert math.degrees(math.atan2(dphi, dlam)) == pytest.approx(-theta, abs=0.01)


def test_grid_relative_winds_rotate_to_earth_relative():
    lat, lon = hrrr.native_latlon()
    theta = math.sin(math.radians(38.5)) * np.radians(lon + 97.5)
    # a 10 m/s wind from the WEST (blowing east) everywhere, in grid components
    ug, vg = 10 * np.cos(theta), 10 * np.sin(theta)
    ue, ve = hrrr.rotate_grid_winds_to_earth(ug, vg)
    assert np.abs(ue - 10).max() < 1e-4 and np.abs(ve).max() < 1e-4
    # at the west edge the grid components differ from earth by ~15 degrees
    assert abs(math.degrees(float(theta[500, 50]))) > 14


def test_fill_value_becomes_nan():
    a = np.array([1.0, -10000.0, 3.0, -9999.5, np.nan], dtype=np.float32)
    out = hrrr._fill_to_nan(a, -10000.0)
    assert out[0] == 1.0 and out[2] == 3.0
    assert np.isnan(out[[1, 3, 4]]).all()


def test_non_native_input_is_refused():
    with pytest.raises(ValueError):
        hrrr._subsample_to_grid(np.zeros(hrrr.HRRR_N_LAT * hrrr.HRRR_N_LON), "t2m")


def test_cache_name_carries_geometry_and_mode(tmp_path):
    p = hrrr._npz_path("20240427", 18, cache_dir=tmp_path)
    assert p.name == f"20240427_18z.{hrrr.GEOMETRY_VERSION}-{hrrr.HRRR_POOL_MODE}.npz"
    # A pre-g2 file is never read as if it were geometry-correct.
    np.savez_compressed(tmp_path / "20240427_18z.npz", t2m=np.zeros((2, 2), dtype=np.float32))
    assert hrrr.load_cached_hrrr("20240427", 18, cache_dir=tmp_path) is None
