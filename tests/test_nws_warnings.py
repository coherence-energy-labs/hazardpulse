"""NWS tornado-warning baseline: geometry, clocks, polygon-state semantics.

Every check here can fail: the synthetic cases have hand-derived answers, the
randomised case is compared against an independent brute-force oracle in exact
rational arithmetic, the clock witness is falsified by a shifted clock, and the
real-data cases use the IEM record and the NWS products themselves (fixtures
under ``tests/fixtures/nws_warnings``, fetched 2026-10-02).
"""

from __future__ import annotations

import io
import struct
import zipfile
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.verification import nws_warnings as nw

FIX = Path(__file__).parent / "fixtures" / "nws_warnings"
OUN_ZIP = FIX / "iem_watchwarn_TO_W_OUN_20240427T2200Z_2300Z.zip"
DDC_ZIP = FIX / "iem_watchwarn_TO_W_DDC_20240427T2300Z_20240428T0000Z.zip"


def utc(y, mo, d, h, mi, s=0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp())


# --------------------------------------------------------------------------
# Synthetic TorWarnings builder + exact oracle
# --------------------------------------------------------------------------


def ring(*pts):
    a = np.array(pts, dtype=np.float64)
    if not np.array_equal(a[0], a[-1]):
        a = np.vstack([a, a[:1]])
    return a


def make_warnings(states, year=2024) -> nw.TorWarnings:
    """states: dicts with rings (list of closed (n,2) lon/lat), vf, vt, issue, new."""

    R = len(states)
    arrays = {
        "wfo": np.array([s.get("wfo", "TST") for s in states], dtype="U4"),
        "etn": np.array([s.get("etn", i + 1) for i, s in enumerate(states)], dtype=np.int32),
        "vtec_year": np.full(R, year, dtype=np.int16),
        "status": np.array(["NEW" if s.get("new") else "CON" for s in states], dtype="U3"),
        "product_id": np.array([f"P{i}" for i in range(R)], dtype="U40"),
        "product_time": np.array([s["vf"] for s in states], dtype=np.int64),
        "valid_from": np.array([s["vf"] for s in states], dtype=np.int64),
        "valid_to": np.array([s["vt"] for s in states], dtype=np.int64),
        "issue_time": np.array([s.get("issue", s["vf"]) for s in states], dtype=np.int64),
        "issue_source": np.zeros(R, dtype=np.int8),
        "is_issuance": np.array([bool(s.get("new")) for s in states]),
        "expired": np.array([s["vt"] for s in states], dtype=np.int64),
        "init_exp": np.array([s["vt"] for s in states], dtype=np.int64),
        "tornado_tag": np.array(["RADAR INDICATED"] * R, dtype="U16"),
        "damage_tag": np.array(["None"] * R, dtype="U16"),
        "hail_tag_in": np.full(R, np.nan, dtype=np.float32),
        "is_emergency": np.zeros(R, dtype=bool),
        "area_km2": np.full(R, np.nan, dtype=np.float32),
    }
    rl, pr, vx, vy = [], [], [], []
    mins = {"minlon": [], "maxlon": [], "minlat": [], "maxlat": []}
    for s in states:
        pr.append(len(s["rings"]))
        allv = np.vstack(s["rings"])
        mins["minlon"].append(allv[:, 0].min())
        mins["maxlon"].append(allv[:, 0].max())
        mins["minlat"].append(allv[:, 1].min())
        mins["maxlat"].append(allv[:, 1].max())
        for rg in s["rings"]:
            rl.append(len(rg))
            vx.append(rg[:, 0])
            vy.append(rg[:, 1])
    for k, v in mins.items():
        arrays[k] = np.array(v)
    arrays["ring_start"] = np.concatenate([[0], np.cumsum(rl)]).astype(np.int64)
    arrays["poly_ring_start"] = np.concatenate([[0], np.cumsum(pr)]).astype(np.int64)
    arrays["vx"] = np.concatenate(vx)
    arrays["vy"] = np.concatenate(vy)
    meta = {str(year): {"year_complete_at_fetch": True}}
    return nw._assemble([arrays], [year], meta)


def pip_exact(px, py, rings) -> bool:
    """Crossing number in exact rational arithmetic (the oracle)."""

    X, Y = Fraction(px), Fraction(py)
    inside = False
    for rg in rings:
        for (x1, y1), (x2, y2) in zip(rg[:-1].tolist(), rg[1:].tolist()):
            if (y1 > py) != (y2 > py):
                fx1, fy1, fx2, fy2 = Fraction(x1), Fraction(y1), Fraction(x2), Fraction(y2)
                xint = fx1 + (Y - fy1) * (fx2 - fx1) / (fy2 - fy1)
                if X < xint:
                    inside = not inside
    return inside


def brute_force(states, lat, lon, t, look_s, ve=None, vs=None):
    n = len(lat)
    active = np.zeros(n, bool)
    since = np.full(n, np.nan)
    issued = np.zeros(n, bool)
    to_issue = np.full(n, np.nan)
    for i in range(n):
        for s in states:
            if s["vf"] <= t[i] < s["vt"] and pip_exact(lon[i], lat[i], s["rings"]):
                active[i] = True
                m = (t[i] - s.get("issue", s["vf"])) / 60.0
                since[i] = m if np.isnan(since[i]) else max(since[i], m)
            if s.get("new") and t[i] < s["vf"] <= t[i] + look_s:
                if ve is None:
                    px, py = lon[i], lat[i]
                else:
                    dt = float(s["vf"] - t[i])
                    py = lat[i] - vs[i] * dt / nw.M_PER_DEG
                    px = lon[i] + ve[i] * dt / (nw.M_PER_DEG * np.cos(np.radians(lat[i])))
                if pip_exact(px, py, s["rings"]):
                    issued[i] = True
                    m = (s["vf"] - t[i]) / 60.0
                    to_issue[i] = m if np.isnan(to_issue[i]) else min(to_issue[i], m)
    return active, since, issued, to_issue


T0 = utc(2024, 5, 6, 20, 0)
SQUARE = ring((-98.0, 35.0), (-97.0, 35.0), (-97.0, 36.0), (-98.0, 36.0))


# --------------------------------------------------------------------------
# 1. Synthetic polygon / time cases with hand-derived answers
# --------------------------------------------------------------------------


def test_synthetic_square_half_open_interval_and_issuance_window():
    W = make_warnings([{"rings": [SQUARE], "vf": T0, "vt": T0 + 1800, "new": True}])
    inside = (35.5, -97.5)
    cases = [  # (t, active, issued_within_next_60, minutes_since_issue)
        (T0 - 3601, False, False, np.nan),  # issuance 60 min + 1 s ahead: outside (t, t+60]
        (T0 - 3600, False, True, np.nan),  # exactly 60 min ahead: inside the closed end
        (T0 - 1, False, True, np.nan),
        (T0, True, False, 0.0),  # issued at t is "active", not "issued within next"
        (T0 + 1799, True, False, 1799 / 60.0),
        (T0 + 1800, False, False, np.nan),  # expiry instant: no longer in effect
    ]
    lat = np.full(len(cases), inside[0])
    lon = np.full(len(cases), inside[1])
    t = np.array([c[0] for c in cases], dtype=np.int64)
    a, iss, since = nw.tor_warning_state(lat, lon, t, warnings=W)
    assert a.tolist() == [c[1] for c in cases]
    assert iss.tolist() == [c[2] for c in cases]
    np.testing.assert_array_equal(np.isnan(since), [np.isnan(c[3]) for c in cases])
    np.testing.assert_allclose(since[~np.isnan(since)], [c[3] for c in cases if not np.isnan(c[3])])
    # outside the polygon at an in-effect instant -> nothing
    a2, i2, s2 = nw.tor_warning_state([35.5, 36.0001, 34.9999], [-96.9999, -97.5, -97.5], [T0 + 60] * 3, warnings=W)
    assert not a2.any() and not i2.any() and np.isnan(s2).all()


def test_synthetic_concave_hole_and_multipolygon():
    u_shape = ring((-98, 35), (-97, 35), (-97, 36), (-97.3, 36), (-97.3, 35.3), (-97.7, 35.3), (-97.7, 36), (-98, 36))
    donut = [ring((-96, 35), (-95, 35), (-95, 36), (-96, 36)), ring((-95.7, 35.3), (-95.3, 35.3), (-95.3, 35.7), (-95.7, 35.7))]
    multi = [ring((-94, 35), (-93.8, 35), (-93.8, 35.2), (-94, 35.2)), ring((-93.5, 35.5), (-93.3, 35.5), (-93.3, 35.7), (-93.5, 35.7))]
    W = make_warnings(
        [
            {"rings": [u_shape], "vf": T0, "vt": T0 + 3600},
            {"rings": donut, "vf": T0, "vt": T0 + 3600},
            {"rings": multi, "vf": T0, "vt": T0 + 3600},
        ]
    )
    pts = [  # (lat, lon, expected active)
        (35.15, -97.5, True),  # U base
        (35.8, -97.85, True),  # left arm
        (35.8, -97.15, True),  # right arm
        (35.8, -97.5, False),  # the notch (inside the bounding box!)
        (35.5, -95.9, True),  # donut ring
        (35.5, -95.5, False),  # donut hole
        (35.1, -93.9, True),  # first part
        (35.6, -93.4, True),  # second part
        (35.35, -93.65, False),  # between the parts
    ]
    a, _, _ = nw.tor_warning_state([p[0] for p in pts], [p[1] for p in pts], [T0 + 600] * len(pts), warnings=W)
    assert a.tolist() == [p[2] for p in pts]


def test_minutes_since_issue_takes_earliest_covering_warning_and_ignores_empty_states():
    W = make_warnings(
        [
            {"rings": [SQUARE], "vf": T0 + 600, "vt": T0 + 2400, "issue": T0},  # CON of a warning issued at T0
            {"rings": [SQUARE], "vf": T0 + 900, "vt": T0 + 3000, "new": True},  # a second, later warning
            {"rings": [SQUARE], "vf": T0 + 1200, "vt": T0 + 1100, "new": False},  # empty (late EXP) state
        ]
    )
    q = nw.query_tor_warnings([35.5], [-97.5], [T0 + 1000], warnings=W)
    assert q.active_now[0]
    assert q.minutes_since_issue[0] == pytest.approx(1000 / 60.0)  # from T0, not from T0+900
    assert q.active_state[0] == 0
    q2 = nw.query_tor_warnings([35.5], [-97.5], [T0 + 2500], warnings=W)
    assert q2.active_now[0] and q2.minutes_since_issue[0] == pytest.approx((2500 - 900) / 60.0)


def test_advection_moves_the_storm_into_the_issuance_polygon():
    W = make_warnings([{"rings": [SQUARE], "vf": T0, "vt": T0 + 1800, "new": True}])
    # storm 30 min before issuance, 0.3 deg west of the polygon's west edge at 35.5N
    lat, lon, t = [35.5], [-98.3], [T0 - 1800]
    dx_m = 0.4 * nw.M_PER_DEG * np.cos(np.radians(35.5))  # needs > 0.3 deg in 1800 s
    v_east = dx_m / 1800.0
    static = nw.query_tor_warnings(lat, lon, t, warnings=W)
    east = nw.query_tor_warnings(lat, lon, t, warnings=W, motion_east_ms=[v_east], motion_south_ms=[0.0])
    west = nw.query_tor_warnings(lat, lon, t, warnings=W, motion_east_ms=[-v_east], motion_south_ms=[0.0])
    assert not static.issued_within_lookahead[0]
    assert east.issued_within_lookahead[0] and east.minutes_to_next_issue[0] == pytest.approx(30.0)
    assert not west.issued_within_lookahead[0]
    # motion_south is southward-positive (ProbSevere convention): a storm south of
    # the box moving NORTH has negative motion_south
    q = nw.query_tor_warnings([34.7], [-97.5], [T0 - 1800], warnings=W, motion_east_ms=[0.0], motion_south_ms=[-0.5 * nw.M_PER_DEG / 1800.0])
    assert q.issued_within_lookahead[0]


@pytest.mark.parametrize("bin_seconds", [60, 900, 18000])
def test_randomised_against_exact_brute_force_oracle(bin_seconds):
    rng = np.random.default_rng(20261002 + bin_seconds)
    states = []
    for k in range(40):
        cx, cy = rng.uniform(-100, -96), rng.uniform(34, 38)
        nv = int(rng.integers(4, 13))
        ang = np.sort(rng.uniform(0, 2 * np.pi, nv))
        rad = rng.uniform(0.1, 0.6, nv)
        outer = ring(*zip(cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        rings = [outer]
        if k % 7 == 3:  # a hole
            rings.append(ring((cx - 0.03, cy - 0.03), (cx + 0.03, cy - 0.03), (cx + 0.03, cy + 0.03), (cx - 0.03, cy + 0.03)))
        if k % 11 == 5:  # a second, disjoint part
            rings.append(ring((cx + 1.0, cy), (cx + 1.2, cy), (cx + 1.2, cy + 0.2), (cx + 1.0, cy + 0.2)))
        vf = T0 + int(rng.integers(0, 6 * 3600))
        dur = int(rng.integers(-300, 3600))  # some empty states
        states.append({"rings": rings, "vf": vf, "vt": vf + dur, "issue": vf - int(rng.integers(0, 1800)) * (k % 2), "new": k % 3 == 0})
    W = make_warnings(states)
    n = 2500
    lat = rng.uniform(33.5, 38.5, n)
    lon = rng.uniform(-100.5, -95, n)
    t = T0 - 3600 + rng.integers(0, 8 * 3600, n)
    # half the points placed at in-effect instants near states, so hits are plentiful
    pick = rng.integers(0, len(states), n // 2)
    for j, r in enumerate(pick):
        c = states[r]["rings"][0][:-1].mean(axis=0)
        lon[j] = c[0] + rng.normal(0, 0.25)
        lat[j] = c[1] + rng.normal(0, 0.25)
        t[j] = states[r]["vf"] + int(rng.integers(-3700, 3700))
    ve = rng.normal(10, 8, n)
    vs = rng.normal(-3, 8, n)
    for adv in (False, True):
        kw = {"motion_east_ms": ve, "motion_south_ms": vs} if adv else {}
        q = nw.query_tor_warnings(lat, lon, t, warnings=W, bin_seconds=bin_seconds, **kw)
        a, since, iss, to_iss = brute_force(states, lat, lon, t, 3600, *((ve, vs) if adv else (None, None)))
        assert a.sum() > 50 and iss.sum() > 20, f"oracle must see real hits ({a.sum()}, {iss.sum()}), or the comparison is dark"
        np.testing.assert_array_equal(q.active_now, a)
        np.testing.assert_array_equal(q.issued_within_lookahead, iss)
        np.testing.assert_allclose(q.minutes_since_issue, since, equal_nan=True)
        np.testing.assert_allclose(q.minutes_to_next_issue, to_iss, equal_nan=True)


def test_near_edge_points_match_exact_arithmetic():
    poly = ring((-97.46, 36.30), (-97.46, 36.13), (-97.84, 35.95), (-97.90, 36.01), (-97.56, 36.31))
    W = make_warnings([{"rings": [poly], "vf": T0, "vt": T0 + 3600}])
    rng = np.random.default_rng(7)
    lats, lons = [], []
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        for f in rng.uniform(0.05, 0.95, 20):
            mx, my = x1 + f * (x2 - x1), y1 + f * (y2 - y1)
            nx, ny = -(y2 - y1), (x2 - x1)
            nrm = np.hypot(nx, ny)
            for d in (1e-9, -1e-9, 1e-7, -1e-7):
                lons.append(mx + d * nx / nrm)
                lats.append(my + d * ny / nrm)
    a, _, _ = nw.tor_warning_state(lats, lons, [T0 + 1] * len(lats), warnings=W)
    expect = [pip_exact(x, y, [poly]) for x, y in zip(lons, lats)]
    assert 0 < sum(expect) < len(expect)
    assert a.tolist() == expect


# --------------------------------------------------------------------------
# 2. Clock witness: a real Oklahoma tornado warning, 2024-04-27
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oun():
    return nw.warnings_from_iem_zip(OUN_ZIP.read_bytes(), year=2024)


def _row(W, product_id):
    rows = np.flatnonzero(W.product_id == product_id)
    assert len(rows) == 1, product_id
    return int(rows[0])


@pytest.mark.parametrize(
    "product_id, issue_utc, expire_utc, local_line",
    [
        ("202404272216-KOUN-WFUS54-TOROUN", utc(2024, 4, 27, 22, 16), utc(2024, 4, 27, 23, 0), "516 PM CDT Sat Apr 27 2024"),
        ("202404272235-KOUN-WFUS54-TOROUN", utc(2024, 4, 27, 22, 35), utc(2024, 4, 27, 23, 0), "535 PM CDT Sat Apr 27 2024"),
    ],
)
def test_clock_witness_iem_fields_are_utc(oun, product_id, issue_utc, expire_utc, local_line):
    text = (FIX / f"{product_id}.txt").read_text()
    r = _row(oun, product_id)
    assert oun.status[r] == "NEW"
    assert int(oun.valid_from[r]) == issue_utc and int(oun.issue_time[r]) == issue_utc
    assert int(oun.expired[r]) == expire_utc
    w = nw.clock_witness(text, int(oun.valid_from[r]), int(oun.expired[r]))
    assert w["local_line"] == local_line
    assert w["offsets"] == {
        "iem_minus_wmo_min": 0.0,
        "iem_minus_vtec_begin_min": 0.0,
        "iem_minus_local_line_min": 0.0,
        "iem_expire_minus_vtec_end_min": 0.0,
    }
    assert w["agree"]
    # the witness can fail: the same record read on CDT / CST is caught
    for hours in (5, 6):
        bad = nw.clock_witness(text, int(oun.valid_from[r]) - hours * 3600, int(oun.expired[r]) - hours * 3600)
        assert not bad["agree"]
        assert bad["offsets"]["iem_minus_local_line_min"] == -60.0 * hours


def test_shapefile_polygon_equals_the_products_latlon(oun):
    for pid in ("202404272216-KOUN-WFUS54-TOROUN", "202404272229-KOUN-WWUS54-SVSOUN", "202404272235-KOUN-WFUS54-TOROUN"):
        r = _row(oun, pid)
        text_ll = nw.parse_product_clocks((FIX / f"{pid}.txt").read_text()).latlon
        rings = oun.polygon(r)
        assert len(rings) == 1
        shp = rings[0][:-1]  # drop the closing vertex
        np.testing.assert_allclose(shp[:, 1], text_ll[:, 0], atol=1e-9)
        np.testing.assert_allclose(shp[:, 0], text_ll[:, 1], atol=1e-9)


def test_to_epoch_seconds_refuses_ambiguous_clocks():
    want = utc(2024, 4, 27, 22, 16)
    assert nw.to_epoch_seconds(np.array(["2024-04-27T22:16"], dtype="datetime64[m]"))[0] == want
    assert nw.to_epoch_seconds(["2024-04-27T22:16:00Z"])[0] == want
    assert nw.to_epoch_seconds(["2024-04-27T17:16:00-05:00"])[0] == want
    assert nw.to_epoch_seconds([datetime(2024, 4, 27, 17, 16, tzinfo=timezone(timedelta(hours=-5)))])[0] == want
    assert nw.to_epoch_seconds([float(want) + 0.9])[0] == want
    for bad in (["2024-04-27T22:16:00"], [datetime(2024, 4, 27, 22, 16)], [want * 1000], np.array(["NaT"], dtype="datetime64[s]")):
        with pytest.raises(ValueError):
            nw.to_epoch_seconds(bad)


# --------------------------------------------------------------------------
# 3. Reduced / cancelled / expired polygon states are not active
# --------------------------------------------------------------------------


def test_svs_reduction_removes_the_dropped_area(oun):
    new = _row(oun, "202404272216-KOUN-WFUS54-TOROUN")
    con = _row(oun, "202404272229-KOUN-WWUS54-SVSOUN")
    p = (35.99, -97.86)  # SW corner of the issued polygon, cut by the 22:29Z SVS
    assert oun.contains([new], [p[1]], [p[0]])[0]
    assert not oun.contains([con], [p[1]], [p[0]])[0]
    t = [utc(2024, 4, 27, 22, 28, 59), utc(2024, 4, 27, 22, 29), utc(2024, 4, 27, 22, 40)]
    a, _, since = nw.tor_warning_state([p[0]] * 3, [p[1]] * 3, t, warnings=oun)
    assert a.tolist() == [True, False, False]
    assert since[0] == pytest.approx(12 + 59 / 60)
    # a point the SVS kept stays warned, with minutes counted from the 22:16Z issuance
    k = (36.25, -97.50)
    a, _, since = nw.tor_warning_state([k[0]] * 2, [k[1]] * 2, t[1:], warnings=oun)
    assert a.all()
    np.testing.assert_allclose(since, [13.0, 24.0])


def test_expired_state_is_not_active(oun):
    # inside the final (EXP statement) polygon of OUN TO.W 0036, which ends 23:00Z
    p = (36.25, -97.50)
    t = [utc(2024, 4, 27, 22, 59, 59), utc(2024, 4, 27, 23, 0), utc(2024, 4, 27, 23, 20)]
    a, _, since = nw.tor_warning_state([p[0]] * 3, [p[1]] * 3, t, warnings=oun)
    assert a.tolist() == [True, False, False]
    assert since[0] == pytest.approx(43 + 59 / 60) and np.isnan(since[1:]).all()


def test_cancelled_warning_is_not_active_after_the_can():
    W = nw.warnings_from_iem_zip(DDC_ZIP.read_bytes(), year=2024)
    r = _row(W, "202404272324-KDDC-WFUS53-TORDDC")
    can = nw.parse_product_clocks((FIX / "202404272339-KDDC-WWUS53-SVSDDC.txt").read_text())
    assert [v["action"] for v in can.vtec] == ["CAN"] and can.vtec[0]["etn"] == 2
    # the warning was issued to run until 00:15Z and cancelled at 23:39Z
    assert int(W.init_exp[r]) == utc(2024, 4, 28, 0, 15)
    assert can.wmo_utc == utc(2024, 4, 27, 23, 39) == int(W.valid_to[r]) == int(W.expired[r])
    lat, lon = 38.58, -100.90  # inside the polygon (LAT...LON 3852 10094 3857 10098 3869 10085 3858 10077)
    assert W.contains([r], [lon], [lat])[0]
    t = [utc(2024, 4, 27, 23, 30), utc(2024, 4, 27, 23, 38, 59), utc(2024, 4, 27, 23, 39), utc(2024, 4, 28, 0, 10)]
    a, _, _ = nw.tor_warning_state([lat] * 4, [lon] * 4, t, warnings=W)
    assert a.tolist() == [True, True, False, False]


# --------------------------------------------------------------------------
# 4. Parse-level semantics on synthetic IEM-format zips
# --------------------------------------------------------------------------

_FIELDS = [  # IEM watchwarn.py DBF layout (subset the parser reads)
    ("WFO", "C", 3, 0), ("ISSUED", "C", 12, 0), ("EXPIRED", "C", 12, 0), ("INIT_ISS", "C", 12, 0),
    ("INIT_EXP", "C", 12, 0), ("PHENOM", "C", 2, 0), ("GTYPE", "C", 1, 0), ("SIG", "C", 1, 0),
    ("ETN", "N", 4, 0), ("STATUS", "C", 3, 0), ("AREA_KM2", "N", 24, 15), ("EMERGENC", "L", 1, 0),
    ("POLY_BEG", "C", 12, 0), ("POLY_END", "C", 12, 0), ("HAILTAG", "N", 24, 15), ("TORNTAG", "C", 16, 0),
    ("DAMAGTAG", "C", 16, 0), ("PROD_ID", "C", 36, 0), ("VTEC_YR", "N", 4, 0),
]


def _write_iem_zip(rows) -> bytes:
    """rows: dicts of DBF values (+ 'xy' closed ring) -> zip with .shp/.dbf/.prj/.cpg."""

    recs = b""
    for i, r in enumerate(rows, start=1):
        xy = np.asarray(r["xy"], dtype="<f8")
        content = struct.pack("<i4d2ii", 5, xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max(), 1, len(xy), 0)
        content += xy.tobytes()
        recs += struct.pack(">ii", i, len(content) // 2) + content
    total = 100 + len(recs)
    hdr = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, total // 2) + struct.pack("<2i8d", 1000, 5, -180, -90, 180, 90, 0, 0, 0, 0)
    shp = hdr + recs
    dbf = bytearray(struct.pack("<4BIHH20x", 3, 126, 10, 2, len(rows), 32 + 32 * len(_FIELDS) + 1, 1 + sum(f[2] for f in _FIELDS)))
    for name, typ, w, d in _FIELDS:
        dbf += name.encode().ljust(11, b"\x00") + typ.encode() + bytes(4) + bytes([w, d]) + bytes(14)
    dbf += b"\x0d"
    for r in rows:
        dbf += b" "
        for name, typ, w, _d in _FIELDS:
            v = r.get(name, "")
            if typ == "N":
                s = ("" if v == "" else str(v)).rjust(w)
            elif typ == "L":
                s = "T" if v else "F"
            else:
                s = str(v).ljust(w)
            dbf += s.encode("latin-1")[:w]
    dbf += b"\x1a"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("wwa.shp", shp)
        z.writestr("wwa.dbf", bytes(dbf))
        z.writestr("wwa.cpg", "ISO-8859-1")
        z.writestr("wwa.prj", 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]')
    return buf.getvalue()


def _iem_row(status, beg, end, pid, *, issued="202405062000", expired="202405062045", etn=7, xy=None):
    return {
        "WFO": "TST", "ISSUED": issued, "EXPIRED": expired, "INIT_ISS": issued, "INIT_EXP": "202405062045",
        "PHENOM": "TO", "GTYPE": "P", "SIG": "W", "ETN": etn, "STATUS": status, "AREA_KM2": "", "EMERGENC": False,
        "POLY_BEG": beg, "POLY_END": end, "HAILTAG": "", "TORNTAG": "RADAR INDICATED", "DAMAGTAG": "",
        "PROD_ID": pid, "VTEC_YR": 2024, "xy": SQUARE if xy is None else xy,
    }


def test_parser_applies_late_exp_clip_and_new_start_repair():
    rows = [
        # NEW whose POLY_BEG (20:10) is later than its own product (20:00): repaired to 20:00
        _iem_row("NEW", "202405062010", "202405062030", "202405062000-KTST-WFUS53-TORTST"),
        # CON whose POLY_END was pushed past EXPIRED (20:45) to a late EXP at 20:48: clipped
        _iem_row("CON", "202405062030", "202405062048", "202405062030-KTST-WWUS53-SVSTST"),
        # the late EXP statement itself: empty interval
        _iem_row("EXP", "202405062048", "202405062045", "202405062048-KTST-WWUS53-SVSTST"),
    ]
    arrays, meta = nw.parse_iem_sbw_zip(_write_iem_zip(rows), expect_year=2024)
    assert arrays["valid_from"].tolist() == [utc(2024, 5, 6, 20, 0), utc(2024, 5, 6, 20, 30), utc(2024, 5, 6, 20, 48)]
    assert arrays["valid_to"].tolist() == [utc(2024, 5, 6, 20, 30), utc(2024, 5, 6, 20, 45), utc(2024, 5, 6, 20, 45)]
    assert meta["repair_new_state_start"] == 1 and meta["clip_to_event_expiry"] == 1 and meta["empty_states"] == 1
    assert (arrays["issue_time"] == utc(2024, 5, 6, 20, 0)).all()
    W = nw._assemble([arrays], [2024], {"2024": meta})
    t = [utc(2024, 5, 6, 20, 5), utc(2024, 5, 6, 20, 44, 59), utc(2024, 5, 6, 20, 45), utc(2024, 5, 6, 20, 47)]
    a, _, _ = nw.tor_warning_state([35.5] * 4, [-97.5] * 4, t, warnings=W)
    assert a.tolist() == [True, True, False, False]


def test_parser_refuses_a_state_far_past_its_event_expiry():
    rows = [_iem_row("NEW", "202405062000", "202405062130", "202405062000-KTST-WFUS53-TORTST")]  # 45 min past EXPIRED
    with pytest.raises(nw.NwsWarningDataError, match="outlives"):
        nw.parse_iem_sbw_zip(_write_iem_zip(rows), expect_year=2024)


def test_a_reused_event_number_does_not_lend_its_expiry_to_the_next_warning():
    """2026-08-11, JKL 24 twice: issued 22:05 and cancelled 22:20, then issued again 22:36 (initial expiry
    23:15). IEM carried 22:20 onto the second warning's rows, which read as 55 min past its expiry and
    stopped the loader. An expiry earlier than the row's own issuance is another event's."""
    first = _iem_row("NEW", "202405062205", "202405062220", "202405062205-KTST-WFUS53-TORTST",
                     issued="202405062205", expired="202405062220", etn=24)
    second = _iem_row("NEW", "202405062236", "202405062315", "202405062236-KTST-WFUS53-TORTST",
                      issued="202405062236", expired="202405062220", etn=24)
    second["INIT_EXP"] = "202405062315"
    arrays, meta = nw.parse_iem_sbw_zip(_write_iem_zip([first, second]), expect_year=2024)
    assert meta["expiry_before_issuance"] == 1 and meta["clip_to_event_expiry"] == 0
    assert arrays["valid_to"].tolist() == [utc(2024, 5, 6, 22, 20), utc(2024, 5, 6, 23, 15)]
    W = nw._assemble([arrays], [2024], {"2024": meta})
    a, _, since = nw.tor_warning_state([35.5] * 3, [-97.5] * 3,
                                       [utc(2024, 5, 6, 22, 30), utc(2024, 5, 6, 22, 50), utc(2024, 5, 6, 23, 15)],
                                       warnings=W)
    assert a.tolist() == [False, True, False]                     # the second warning is in force, then ends
    # a COHERENT event far past its own expiry still stops the loader (the guard is narrowed, not removed)
    with pytest.raises(nw.NwsWarningDataError, match="outlives"):
        nw.parse_iem_sbw_zip(_write_iem_zip([_iem_row("NEW", "202405062000", "202405062130",
                                                      "202405062000-KTST-WFUS53-TORTST")]), expect_year=2024)


def test_parser_refuses_non_tornado_rows_and_wrong_year():
    sv = _iem_row("NEW", "202405062000", "202405062030", "202405062000-KTST-WFUS53-SVRTST")
    sv["PHENOM"] = "SV"
    with pytest.raises(nw.NwsWarningDataError, match="PHENOM"):
        nw.parse_iem_sbw_zip(_write_iem_zip([sv]), expect_year=2024)
    ok = _iem_row("NEW", "202405062000", "202405062030", "202405062000-KTST-WFUS53-TORTST")
    with pytest.raises(nw.NwsWarningDataError, match="outside 2023"):
        nw.parse_iem_sbw_zip(_write_iem_zip([ok]), expect_year=2023)


def test_query_refuses_instants_whose_warnings_are_not_loaded():
    W = make_warnings([{"rings": [SQUARE], "vf": T0, "vt": T0 + 1800, "new": True}], year=2024)
    with pytest.raises(ValueError, match="2023"):
        nw.tor_warning_state([35.5], [-97.5], [utc(2024, 1, 1, 0, 30)], warnings=W)  # a 2023-12-31 warning could be live
    with pytest.raises(ValueError, match="2025"):
        nw.tor_warning_state([35.5], [-97.5], [utc(2024, 12, 31, 23, 30)], warnings=W)  # a 2025 issuance is within 60 min
    a, _, _ = nw.tor_warning_state([35.5], [-97.5], [utc(2024, 1, 1, 0, 30)], warnings=W, require_coverage=False)
    assert not a[0]


# --------------------------------------------------------------------------
# 5. Real cache (skipped when the compact 2024 file is absent)
# --------------------------------------------------------------------------


def test_real_2024_record_if_cached():
    path = nw._compact_path(2024)
    if not path.exists():
        pytest.skip(f"no compact 2024 file at {path}")
    W = nw.load_tor_warnings([2024], build_missing=False)
    assert W.meta["2024"]["n_issuances"] == int(W.is_issuance.sum())
    assert W.meta["2024"]["format_version"] == nw.FORMAT_VERSION
    r = _row(W, "202404272216-KOUN-WFUS54-TOROUN")
    assert int(W.valid_from[r]) == utc(2024, 4, 27, 22, 16)
    a, _, since = nw.tor_warning_state([36.25], [-97.50], [utc(2024, 4, 27, 22, 40)], warnings=W)
    assert a[0] and since[0] == pytest.approx(24.0)
