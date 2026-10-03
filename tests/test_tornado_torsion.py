"""Torsion is a real rotational coupling, not the curl of a gradient.

Until 2026-10-01 ``compute_curl_2d(tau)`` returned the curl of tau's own
gradient -- identically zero up to float32 round-off (max|curl|/max|grad| ~
1e-8) -- so torsion, singularity condition 3, the NPE torsion term and
``torsion_x_srh`` were inert and the GBT gave them importance 0.0.
"""
from __future__ import annotations

import numpy as np

from hazardpulse.data.hrrr import HRRR_N_LAT, HRRR_N_LON
from hazardpulse.tornado import coherence_engine as ce


def test_curl_of_solid_body_rotation_is_twice_the_rate():
    ny, nx = 21, 31
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    w = 0.7
    u = -w * (yy - ny // 2)
    v = w * (xx - nx // 2)
    curl = ce.compute_curl_2d(u, v)
    assert np.allclose(curl, 2 * w, atol=1e-5)


def test_curl_of_a_gradient_is_zero_so_it_cannot_be_the_torsion():
    """The old construction, made explicit: grad of a scalar has no curl."""
    rng = np.random.RandomState(0)
    tau = rng.rand(34, 63).astype(np.float32)
    gy, gx = ce._gradient_2d(tau)
    assert np.abs(ce.compute_curl_2d(gx, gy)).max() <= 1e-5 * np.abs(gx).max()


def test_torsion_is_the_shear_component_across_the_gradient():
    ny, nx = 9, 9
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    tau = 0.5 * xx  # grad tau = (0.5 east, 0 north)
    s = ce.TORSION_SHEAR_SCALE
    # shear parallel to the gradient: no tilting
    par = ce.compute_tilting_torsion(tau, np.full_like(tau, 10.0), np.zeros_like(tau))
    assert np.abs(par).max() < 1e-6
    # shear perpendicular (northward): |S| |grad tau| / scale, sign from S x grad tau
    perp = ce.compute_tilting_torsion(tau, np.zeros_like(tau), np.full_like(tau, 10.0))
    assert np.allclose(perp, -10.0 * 0.5 / s, atol=1e-6)
    anti = ce.compute_tilting_torsion(tau, np.zeros_like(tau), np.full_like(tau, -10.0))
    assert np.allclose(anti, 10.0 * 0.5 / s, atol=1e-6)


def _synthetic_atmosphere(seed=1):
    rng = np.random.RandomState(seed)
    ny, nx = HRRR_N_LAT, HRRR_N_LON
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    bump = np.exp(-(((yy - 15) / 6.0) ** 2 + ((xx - 35) / 9.0) ** 2)).astype(np.float32)
    f = lambda a: a.astype(np.float32)
    return {
        "mlcape": f(3000 * bump + 50 * rng.rand(ny, nx)),
        "mlcin": f(-20 - 30 * rng.rand(ny, nx)),
        "srh_01": f(250 * bump),
        "srh_03": f(350 * bump),
        "ushear_06": f(20 + 5 * bump),
        "vshear_06": f(5 + 0 * bump),
        "ushear_01": f(8 + 2 * rng.rand(ny, nx)),
        "vshear_01": f(12 + 2 * rng.rand(ny, nx)),
        "ustorm": f(10 + 0 * bump),
        "vstorm": f(5 + 0 * bump),
        "t2m": f(300 - 5 * (1 - bump)),
        "td2m": f(292 - 8 * (1 - bump)),
        "refc": f(45 * bump),
        "pwat": f(30 * bump + 10),
    }


def test_engine_torsion_is_not_inert():
    fields = ce.compute_coherence_fields(_synthetic_atmosphere(), month=5)
    tors = fields["torsion"]
    gy, gx = ce._gradient_2d(fields["tau"])
    grad_max = float(np.hypot(gx, gy).max())
    # Real coupling: same order as |S_01| |grad tau| / scale, far above round-off.
    assert float(np.abs(tors).max()) > 0.1 * 14.0 * grad_max / ce.TORSION_SHEAR_SCALE
    assert np.isfinite(tors).all()


def test_singularity_condition_3_uses_the_shared_threshold():
    fields = ce.compute_coherence_fields(_synthetic_atmosphere(), month=5)
    i, j = np.unravel_index(np.argmax(np.abs(fields["torsion"])), fields["torsion"].shape)
    from hazardpulse.data.hrrr import GRID_LATS, GRID_LONS

    res = ce.test_singularity_at_point(fields, float(GRID_LATS[i]), float(GRID_LONS[j]))
    assert res.high_torsion == (abs(float(fields["torsion"][i, j])) > ce.TORSION_SINGULARITY_THRESHOLD)
