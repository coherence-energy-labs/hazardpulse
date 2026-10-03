"""Live tornado-warning inputs (model block W) from api.weather.gov: parity with training.

No tornado warning was in force on api.weather.gov when these fixtures were
taken (2026-10-02 23:24Z; the active feed is saved, empty).  The alert
fixtures are therefore the API's own records of the 34 tornado warnings of
2026-09-25..30, from its ``/alerts`` archive -- the same ``wx:Alert`` GeoJSON
Features the active feed serves -- saved verbatim, next to the IEM record of
the same warnings (the training source).  Fixtures (``tests/fixtures/nws_live``):

* ``api_alerts_TO_W_recent_20261002T2324Z.json.gz``  gzip of the raw body of
  ``GET https://api.weather.gov/alerts?event=Tornado%20Warning&limit=500``
  (raw body SHA-256 pinned below): 119 messages, 34 events.
* ``iem_watchwarn_TO_W_20260925T0000Z_20261002T0000Z.zip``  IEM watchwarn.py
  with nws_warnings' parameters (``phenomena=TO significance=W limitps limit1
  addsvs timeopt=1``) over ``sts=2026-09-25T00:00Z ets=2026-10-02T00:00Z``.
* ``api_active_TO_W_20261002T2324Z.json``  ``GET /alerts/active?event=Tornado%20Warning``.

Each expectation below is hand-derived from the product times and checked
against an exact rational point-in-polygon oracle, and every real-data case is
also required to equal the training path (IEM record -> nws_warnings) at the
same point and instant.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import random
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.data import http as hp_http
from hazardpulse.verification import nws_live as nl
from hazardpulse.verification import nws_warnings as nw
from test_nws_warnings import _iem_row, _write_iem_zip, pip_exact, ring, utc

FIX = Path(__file__).parent / "fixtures" / "nws_live"
API_GZ = FIX / "api_alerts_TO_W_recent_20261002T2324Z.json.gz"
IEM_ZIP = FIX / "iem_watchwarn_TO_W_20260925T0000Z_20261002T0000Z.zip"
ACTIVE_JSON = FIX / "api_active_TO_W_20261002T2324Z.json"
API_RAW_SHA256 = "82543ae7048770fbcf2bf52c41e032ce01e5f0c4eec630c0b6e94e949aeb6bc1"
IEM_SHA256 = "5763b55f648ad9b20ad5317da01dfd5343368f95b6992a985c1085b2259c19d5"
ACTIVE_SHA256 = "b7bf7305f61966ec6685f623d44e9375e485437d1f0bb896f352bb5eaf93cb73"


@pytest.fixture(scope="module")
def api_features() -> list[dict]:
    return json.loads(gzip.decompress(API_GZ.read_bytes()))["features"]


@pytest.fixture(scope="module")
def iem() -> nw.TorWarnings:
    return nw.warnings_from_iem_zip(IEM_ZIP.read_bytes(), year=2026)


def snap(features, *, known_from=None, known_until=None) -> nl.AlertSnapshot:
    """A snapshot of the archive: every message present, answerable for 2026-09-25..10-01."""

    return nl.AlertSnapshot(
        features=tuple(features),
        known_from=utc(2026, 9, 25, 0, 0) if known_from is None else known_from,
        known_until=utc(2026, 10, 2, 23, 23, 17) if known_until is None else known_until,
    )


def training_block_w(active, since) -> np.ndarray:
    """Block W exactly as training stored it (warning_state_on_store.py -> tornado_lab.assemble)."""

    a = np.asarray(active, np.int8)
    m = np.where(np.asarray(active, bool), np.asarray(since, np.float32), np.nan).astype(np.float32)
    return np.column_stack([a.astype(np.float32), m])


def live_and_training(api_features, iem, lats, lons, times):
    """Block W from the API path and from the training path (IEM), asserted bit-identical."""

    res = nl.inputs_from_snapshot(lats, lons, times, snap(api_features))
    a, _, s = nw.tor_warning_state(lats, lons, times, warnings=iem, require_coverage=False)
    ref = training_block_w(a, s)
    assert np.array_equal(res.matrix(), ref, equal_nan=True), (res.matrix(), ref)
    return res


def geometry_of(features, office, etn, action, sent_utc):
    for f in features:
        p = f["properties"]
        m = nw._VTEC_RE.search(p["parameters"]["VTEC"][0])
        if (m.group(3), int(m.group(6)), m.group(2)) == (office, etn, action) and nl._iso_s(p["sent"]) == sent_utc:
            return [np.array(f["geometry"]["coordinates"][0], dtype=np.float64)]
    raise KeyError((office, etn, action, sent_utc))


def minutes(t, t0) -> float:
    return (t - t0) / 60.0


# --------------------------------------------------------------------------
# 0. The fixtures are the pinned downloads
# --------------------------------------------------------------------------


def test_fixtures_are_the_pinned_downloads(api_features):
    assert hashlib.sha256(gzip.decompress(API_GZ.read_bytes())).hexdigest() == API_RAW_SHA256
    assert hashlib.sha256(IEM_ZIP.read_bytes()).hexdigest() == IEM_SHA256
    assert hashlib.sha256(ACTIVE_JSON.read_bytes()).hexdigest() == ACTIVE_SHA256
    assert len(api_features) == 119


# --------------------------------------------------------------------------
# 1. Real record: the API reconstruction IS the IEM record
# --------------------------------------------------------------------------


def test_real_record_states_equal_the_iem_states_exactly(api_features, iem):
    W, rep = nl.warnings_from_api_alerts(api_features)
    assert rep["action_counts"] == {"CAN": 11, "CON": 57, "EXP": 17, "NEW": 34}
    assert rep["events"] == 34 and rep["events_without_new"] == 0 and rep["n_states"] == 108
    assert rep["issue_source_counts"] == {"new_product": 108, "references": 0, "state_start": 0}
    assert rep["empty_states"] == 2  # two EXP statements issued after their warning's end
    assert rep["null_geometry"] == rep["bad_geometry"] == rep["ends_ne_vtec_end"] == rep["sent_not_whole_minute"] == 0
    ka = {str(p): r for r, p in enumerate(W.product_id)}
    ki = {str(p): r for r, p in enumerate(iem.product_id)}
    assert len(iem) == 108 and set(ka) == set(ki)  # same products, same IEM product ids
    for pid, a in ka.items():
        i = ki[pid]
        for fld in ("wfo", "etn", "vtec_year", "status", "valid_from", "valid_to", "issue_time", "is_issuance"):
            assert getattr(W, fld)[a] == getattr(iem, fld)[i], (pid, fld)
        ra, ri = W.polygon(a), iem.polygon(i)
        assert len(ra) == len(ri) == 1 and ra[0].shape == ri[0].shape
        np.testing.assert_allclose(ra[0], ri[0], atol=1e-9, rtol=0)


def test_real_record_inputs_equal_the_training_inputs(api_features, iem):
    """Parity on the real record: 20,000 points near the 108 polygons, at instants spanning each state +-10 min."""

    rng = np.random.default_rng(20261002)
    live = np.flatnonzero(iem.valid_to > iem.valid_from)
    n = 20000
    r = rng.choice(live, n)
    t = iem.valid_from[r] + (rng.random(n) * (iem.valid_to[r] - iem.valid_from[r] + 1200) - 600).astype(np.int64)
    t[: n // 10] = iem.valid_from[r[: n // 10]]  # exact state boundaries
    t[n // 10 : n // 5] = iem.valid_to[r[n // 10 : n // 5]] - 1
    lat = rng.uniform(iem.minlat[r] - 0.05, iem.maxlat[r] + 0.05)
    lon = rng.uniform(iem.minlon[r] - 0.05, iem.maxlon[r] + 0.05)
    res = live_and_training(api_features, iem, lat, lon, t)
    warned = res.w_tor_warning_active == 1.0
    assert 3000 < warned.sum() < n - 3000, "both outcomes must be well represented, or the parity is dark"
    assert np.unique(res.w_minutes_since_issue[warned]).size > 500
    assert set(np.unique(res.event[warned])) >= {"ICT.TO.W.0030", "AMA.TO.W.0027", "GID.TO.W.0016"}


# --------------------------------------------------------------------------
# 2. Real cases, hand-derived
# --------------------------------------------------------------------------


def test_point_inside_and_outside_a_real_polygon(api_features, iem):
    # GID TO.W 0016, issued 2026-10-01 01:50Z (Cancelled 02:03Z)
    t0 = utc(2026, 10, 1, 1, 50)
    rings = geometry_of(api_features, "KGID", 16, "NEW", t0)
    inside, outside = (39.432, -97.957), (39.432, -97.0)
    assert pip_exact(inside[1], inside[0], rings) and not pip_exact(outside[1], outside[0], rings)
    t = t0 + 7 * 60 + 30
    res = live_and_training(api_features, iem, [inside[0], outside[0]], [inside[1], outside[1]], [t, t])
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0]
    assert res.w_minutes_since_issue[0] == np.float32(7.5) and np.isnan(res.w_minutes_since_issue[1])
    assert res.event.tolist() == ["GID.TO.W.0016", ""]
    # issued at t0: not in force one second before, in force at t0 with 0 minutes
    res = live_and_training(api_features, iem, [inside[0]] * 2, [inside[1]] * 2, [t0 - 1, t0])
    assert res.w_tor_warning_active.tolist() == [0.0, 1.0] and res.w_minutes_since_issue[1] == 0.0


def test_con_update_that_moved_the_polygon(api_features, iem):
    # AMA TO.W 0027: NEW 2026-09-30 03:07Z; SVS CON 03:24Z re-drew the polygon east
    t_new, t_con = utc(2026, 9, 30, 3, 7), utc(2026, 9, 30, 3, 24)
    new = geometry_of(api_features, "KAMA", 27, "NEW", t_new)
    con = geometry_of(api_features, "KAMA", 27, "CON", t_con)
    dropped, gained = (35.198, -101.847), (35.216, -101.671)
    assert pip_exact(dropped[1], dropped[0], new) and not pip_exact(dropped[1], dropped[0], con)
    assert not pip_exact(gained[1], gained[0], new) and pip_exact(gained[1], gained[0], con)
    ts = [t_con - 1, t_con, t_con + 600]
    res = live_and_training(api_features, iem, [dropped[0]] * 3, [dropped[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0, 0.0]
    assert res.w_minutes_since_issue[0] == np.float32(minutes(t_con - 1, t_new))
    res = live_and_training(api_features, iem, [gained[0]] * 3, [gained[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [0.0, 1.0, 1.0]
    # minutes count from the 03:07Z issuance, not from the 03:24Z statement (its onset)
    np.testing.assert_array_equal(res.w_minutes_since_issue[1:], np.float32([17.0, 27.0]))


def test_expired_warning_ends_at_the_vtec_end_not_at_expires(api_features, iem):
    # GID TO.W 0015: issued 00:45Z, EXP statement 01:29Z, VTEC end 01:30Z, CAP expires 01:39Z
    exp = [f for f in api_features if f["properties"]["parameters"]["VTEC"][0].startswith("/O.EXP.KGID.TO.W.0015.")]
    assert len(exp) == 1 and nl._iso_s(exp[0]["properties"]["expires"]) == utc(2026, 10, 1, 1, 39)
    rings = geometry_of(api_features, "KGID", 15, "EXP", utc(2026, 10, 1, 1, 29))
    p = (39.413, -98.196)
    assert pip_exact(p[1], p[0], rings)
    ts = [utc(2026, 10, 1, 1, 29, 59), utc(2026, 10, 1, 1, 30), utc(2026, 10, 1, 1, 35)]
    res = live_and_training(api_features, iem, [p[0]] * 3, [p[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0, 0.0]
    assert res.w_minutes_since_issue[0] == np.float32(minutes(ts[0], utc(2026, 10, 1, 0, 45)))
    assert np.isnan(res.w_minutes_since_issue[1:]).all()


def test_cancelled_warning_ends_at_the_can(api_features, iem):
    # GID TO.W 0016: issued 01:50Z to 02:15Z; cancelled 02:03Z (the CAN alert keeps the polygon and
    # its CAP expires at 02:18:23Z -- neither may keep the warning in force)
    can = [f for f in api_features if f["properties"]["parameters"]["VTEC"][0].startswith("/O.CAN.KGID.TO.W.0016.")]
    assert len(can) == 1 and can[0]["geometry"] and can[0]["properties"]["messageType"] == "Cancel"
    p = (39.432, -97.957)
    assert pip_exact(p[1], p[0], [np.array(can[0]["geometry"]["coordinates"][0])])
    ts = [utc(2026, 10, 1, 2, 2, 59), utc(2026, 10, 1, 2, 3), utc(2026, 10, 1, 2, 10)]
    res = live_and_training(api_features, iem, [p[0]] * 3, [p[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0, 0.0]


def test_partial_cancellation_keeps_the_remaining_polygon(api_features, iem):
    # ICT TO.W 0030 (issued 23:14Z): one SVS at 23:46Z = CAN segment (Russell Co.) + CON segment
    # (Lincoln Co.), both carrying the remaining polygon
    t46 = utc(2026, 9, 30, 23, 46)
    segs = [f for f in api_features if ".KICT.TO.W.0030." in f["properties"]["parameters"]["VTEC"][0]
            and nl._iso_s(f["properties"]["sent"]) == t46]
    assert sorted(f["properties"]["parameters"]["VTEC"][0][3:6] for f in segs) == ["CAN", "CON"]
    assert segs[0]["geometry"] == segs[1]["geometry"]
    cut, kept = (39.024, -98.526), (39.117, -98.34)
    before = geometry_of(api_features, "KICT", 30, "CON", utc(2026, 9, 30, 23, 40))
    after = geometry_of(api_features, "KICT", 30, "CON", t46)
    assert pip_exact(cut[1], cut[0], before) and not pip_exact(cut[1], cut[0], after)
    assert pip_exact(kept[1], kept[0], before) and pip_exact(kept[1], kept[0], after)
    ts = [t46 - 1, t46, t46 + 240]
    res = live_and_training(api_features, iem, [cut[0]] * 3, [cut[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0, 0.0]
    res = live_and_training(api_features, iem, [kept[0]] * 3, [kept[1]] * 3, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 1.0, 1.0]
    np.testing.assert_array_equal(res.w_minutes_since_issue, np.float32([31 + 59 / 60, 32.0, 36.0]))


def test_overlapping_warnings_take_the_earliest_issuance(api_features, iem):
    # AMA TO.W 0018 (issued 00:54Z, in force to 01:30Z) and AMA TO.W 0023 (issued 01:20Z) overlap
    p = (34.99, -101.822)
    a18 = geometry_of(api_features, "KAMA", 18, "CON", utc(2026, 9, 30, 1, 8))
    a23 = geometry_of(api_features, "KAMA", 23, "NEW", utc(2026, 9, 30, 1, 20))
    assert pip_exact(p[1], p[0], a18) and pip_exact(p[1], p[0], a23)
    ts = [utc(2026, 9, 30, 1, 25), utc(2026, 9, 30, 1, 30)]
    res = live_and_training(api_features, iem, [p[0]] * 2, [p[1]] * 2, ts)
    assert res.w_tor_warning_active.tolist() == [1.0, 1.0]
    assert res.w_minutes_since_issue.tolist() == [31.0, 10.0]  # from 00:54Z, then (0018 ended) from 01:20Z
    assert res.event.tolist() == ["AMA.TO.W.0018", "AMA.TO.W.0023"]
    # the answer does not depend on the order the API lists the messages in
    shuffled = list(api_features)
    random.Random(7).shuffle(shuffled)
    res2 = nl.inputs_from_snapshot([p[0]] * 2, [p[1]] * 2, ts, snap(shuffled[::-1] + shuffled[:40]))
    assert np.array_equal(res2.matrix(), res.matrix(), equal_nan=True)
    assert res2.report["duplicate_ids"] == 40


def test_original_issuance_needs_the_new_message_not_references(api_features, iem):
    """The EXP statement of GID 0015 is typed 'Alert' with references [] -- only the NEW gives the issuance."""

    exp = [f for f in api_features if f["properties"]["parameters"]["VTEC"][0].startswith("/O.EXP.KGID.TO.W.0015.")]
    assert exp[0]["properties"]["messageType"] == "Alert" and exp[0]["properties"]["references"] == []
    p, t = (39.413, -98.196), utc(2026, 10, 1, 1, 29, 30)
    alone = nl.inputs_from_snapshot([p[0]], [p[1]], [t], snap(exp), require_coverage=False)
    assert alone.w_tor_warning_active[0] == 1.0
    assert alone.w_minutes_since_issue[0] == np.float32(0.5)  # from the statement: wrong, and flagged
    assert alone.report["issue_source_counts"]["state_start"] == 1 and alone.report["events_without_new"] == 1
    with pytest.raises(nl.NwsLiveDataError, match="without their NEW"):
        nl.inputs_from_snapshot([p[0]], [p[1]], [t], snap(exp), require_coverage=False, strict=True)
    # a CON that does reference its NEW gives the exact issuance without the NEW itself
    con = [f for f in api_features if f["properties"]["parameters"]["VTEC"][0].startswith("/O.CON.KGID.TO.W.0015.")
           and nl._iso_s(f["properties"]["sent"]) == utc(2026, 10, 1, 1, 21)]
    q = (39.413, -98.196)
    assert len(con) == 1 and pip_exact(q[1], q[0], [np.array(con[0]["geometry"]["coordinates"][0])])
    assert min(r["sent"] for r in con[0]["properties"]["references"]) == "2026-09-30T19:45:00-05:00"  # the NEW
    res = nl.inputs_from_snapshot([q[0]], [q[1]], [utc(2026, 10, 1, 1, 25)], snap(con), require_coverage=False)
    assert res.report["issue_source_counts"]["references"] == 1
    full = live_and_training(api_features, iem, [p[0], q[0]], [p[1], q[1]], [t, utc(2026, 10, 1, 1, 25)])
    assert full.w_minutes_since_issue.tolist() == [44.5, 40.0]
    assert res.w_minutes_since_issue[0] == full.w_minutes_since_issue[1]


# --------------------------------------------------------------------------
# 3. Null geometry, non-operational messages
# --------------------------------------------------------------------------


def test_null_geometry_follow_up_inherits_new_is_dropped_and_both_are_counted(api_features):
    feats = copy.deepcopy(api_features)
    t_con = utc(2026, 9, 30, 3, 24)
    for f in feats:  # AMA 0027: blank the 03:24Z CON's polygon; GID 0016: blank the NEW's
        v = f["properties"]["parameters"]["VTEC"][0]
        if (".KAMA.TO.W.0027." in v and v.startswith("/O.CON") and nl._iso_s(f["properties"]["sent"]) == t_con) or (
            v.startswith("/O.NEW.KGID.TO.W.0016.")
        ):
            f["geometry"] = None
    dropped_by_con, gid = (35.198, -101.847), (39.432, -97.957)
    lat = [dropped_by_con[0], 35.443, gid[0]]
    lon = [dropped_by_con[1], -101.722, gid[1]]
    t = [t_con + 300, t_con + 300, utc(2026, 10, 1, 1, 55)]
    res = nl.inputs_from_snapshot(lat, lon, t, snap(feats))
    assert res.report["null_geometry"] == 2
    assert res.report["geometry_inherited"] == 1 and res.report["geometry_dropped"] == 1
    # the CON inherits the NEW's polygon: the area the CON cut stays warned (the documented
    # bias), the area it kept is warned with the exact minutes; the NEW without a polygon warns nothing
    assert res.w_tor_warning_active.tolist() == [1.0, 1.0, 0.0]
    np.testing.assert_array_equal(res.w_minutes_since_issue[:2], np.float32([22.0, 22.0]))
    with pytest.raises(nl.NwsLiveDataError, match="geometry null"):
        nl.inputs_from_snapshot(lat, lon, t, snap(feats), strict=True)


def test_test_and_non_operational_messages_are_ignored_and_counted(api_features, iem):
    p, t = (39.432, -97.957), utc(2026, 10, 1, 1, 55)
    base = [f for f in api_features if ".KGID.TO.W.0016." in f["properties"]["parameters"]["VTEC"][0]]
    ok = nl.inputs_from_snapshot([p[0]], [p[1]], [t], snap(base))
    assert ok.w_tor_warning_active[0] == 1.0
    for mutate, counter in (
        (lambda f: f["properties"].__setitem__("status", "Test"), "skipped_not_actual"),
        (lambda f: f["properties"]["parameters"].__setitem__("VTEC", ["/T" + f["properties"]["parameters"]["VTEC"][0][2:]]),
         "skipped_no_operational_tow_vtec"),
        (lambda f: f["properties"].__setitem__("event", "Severe Thunderstorm Warning"), "skipped_not_tornado_warning"),
    ):
        feats = copy.deepcopy(base)
        for f in feats:
            mutate(f)
        res = nl.inputs_from_snapshot([p[0]], [p[1]], [t], snap(feats))
        assert res.w_tor_warning_active[0] == 0.0 and res.report[counter] == len(base)


def test_sent_with_seconds_takes_iems_minute_precision(api_features):
    # IEM stores every instant as YYYYMMDDHHMM (truncated); a message stamped 01:50:42Z is in force from 01:50:00Z
    feats = copy.deepcopy([f for f in api_features if ".KGID.TO.W.0016." in f["properties"]["parameters"]["VTEC"][0]])
    new = [f for f in feats if f["properties"]["parameters"]["VTEC"][0].startswith("/O.NEW")][0]
    new["properties"]["sent"] = "2026-09-30T20:50:42-05:00"
    p = (39.432, -97.957)
    res = nl.inputs_from_snapshot([p[0]] * 2, [p[1]] * 2, [utc(2026, 10, 1, 1, 50, 10), utc(2026, 10, 1, 1, 49, 59)], snap(feats))
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0]
    assert res.w_minutes_since_issue[0] == np.float32(10 / 60) and res.report["sent_not_whole_minute"] == 1


# --------------------------------------------------------------------------
# 4. PARITY: one synthetic warning set, as IEM records it and as the API serves it
# --------------------------------------------------------------------------

T0 = utc(2024, 5, 6, 20, 0)
CDT = timezone(timedelta(hours=-5))
A0 = ring((-98.0, 35.0), (-97.0, 35.0), (-97.0, 36.0), (-98.0, 36.0))
A1 = ring((-97.6, 35.2), (-96.9, 35.2), (-96.9, 35.8), (-97.6, 35.8))  # moved east, partly outside A0
B0 = ring((-97.5, 35.5), (-96.5, 35.5), (-96.5, 36.2), (-97.5, 36.2))
C0 = ring((-96.8, 34.6), (-96.0, 34.6), (-96.0, 35.3), (-96.8, 35.3))
C1 = ring((-96.5, 34.6), (-96.0, 34.6), (-96.0, 35.0), (-96.5, 35.0))
D0 = ring((-98.2, 34.6), (-97.4, 34.6), (-97.4, 35.3), (-98.2, 35.3))
E0 = ring((-97.9, 35.1), (-97.1, 35.1), (-97.1, 35.9), (-97.9, 35.9))
E1 = ring((-97.8, 35.3), (-97.2, 35.3), (-97.2, 35.85), (-97.8, 35.85))


def _stamp(t: int) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y%m%d%H%M")


def _alert(action, etn, sent, end, rings, *, seg=1, stem=None, mtype=None, refs=(), expires=None):
    """One api.weather.gov alert Feature, in the schema measured on the real record."""

    vt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%y%m%dT%H%MZ")  # noqa: E731
    begin = vt(sent) if action == "NEW" else "000000T0000Z"
    tor = action == "NEW"
    stem = stem or hashlib.sha1(f"{etn}{action}{sent}".encode()).hexdigest()
    fid = f"urn:oid:2.49.0.1.840.0.{stem}.{seg:03d}.1"
    iso = lambda t: datetime.fromtimestamp(t, tz=CDT).isoformat()  # noqa: E731
    return {
        "id": f"https://api.weather.gov/alerts/{fid}",
        "type": "Feature",
        "geometry": None if rings is None else {"type": "Polygon", "coordinates": [[list(map(float, v)) for v in rings[0]]]},
        "properties": {
            "@id": f"https://api.weather.gov/alerts/{fid}",
            "@type": "wx:Alert",
            "id": fid,
            "references": [{"@id": r, "identifier": r, "sender": "w-nws.webmaster@noaa.gov", "sent": iso(s)} for r, s in refs],
            "sent": iso(sent),
            "effective": iso(sent),
            "onset": iso(sent),
            "expires": iso(end if expires is None else expires),
            "ends": iso(end),
            "status": "Actual",
            "messageType": mtype or {"NEW": "Alert", "CAN": "Cancel"}.get(action, "Update"),
            "event": "Tornado Warning",
            "parameters": {
                "AWIPSidentifier": ["TORTST" if tor else "SVSTST"],
                "WMOidentifier": [f"{'WFUS53' if tor else 'WWUS53'} KTST {_stamp(sent)[6:]}"],
                "tornadoDetection": ["RADAR INDICATED"],
                "VTEC": [f"/O.{action}.KTST.TO.W.{etn:04d}.{begin}-{vt(end)}/"],
            },
        },
    }


def _synthetic_sets():
    m = 60
    api = [
        # A (ETN 7): NEW 20:00 -> SVS 20:12 moves the polygon -> EXP 20:40 (typed Alert, no references); end 20:45
        _alert("NEW", 7, T0, T0 + 45 * m, [A0]),
        _alert("CON", 7, T0 + 12 * m, T0 + 45 * m, [A1], refs=[("a-new", T0)]),
        _alert("EXP", 7, T0 + 40 * m, T0 + 45 * m, [A1], mtype="Alert", expires=T0 + 51 * m),
        # B (ETN 8): NEW 20:05, fully cancelled 20:20 (the CAN keeps a polygon; CAP expires 20:35)
        _alert("NEW", 8, T0 + 5 * m, T0 + 50 * m, [B0]),
        _alert("CAN", 8, T0 + 20 * m, T0 + 50 * m, [B0], refs=[("b-new", T0 + 5 * m)], expires=T0 + 35 * m),
        # C (ETN 9): NEW 20:10; one SVS at 20:25 = CAN segment + CON segment, both with the remaining polygon
        _alert("NEW", 9, T0 + 10 * m, T0 + 55 * m, [C0]),
        _alert("CAN", 9, T0 + 25 * m, T0 + 55 * m, [C1], seg=1, stem="c" * 40, refs=[("c-new", T0 + 10 * m)]),
        _alert("CON", 9, T0 + 25 * m, T0 + 55 * m, [C1], seg=2, stem="c" * 40, refs=[("c-new", T0 + 10 * m)]),
        # D (ETN 10): NEW 20:30 until 21:00; EXP statement issued late, 21:02
        _alert("NEW", 10, T0 + 30 * m, T0 + 60 * m, [D0]),
        _alert("EXP", 10, T0 + 62 * m, T0 + 60 * m, [D0], expires=T0 + 70 * m),
        # E: ETN 7 REUSED by a new warning after A ended (IEM: EAX 2022, LIX 2025): NEW 20:50, SVS 20:58, end 21:20
        _alert("NEW", 7, T0 + 50 * m, T0 + 80 * m, [E0]),
        _alert("CON", 7, T0 + 58 * m, T0 + 80 * m, [E1], refs=[("e-new", T0 + 50 * m)]),
    ]
    s = _stamp
    pid = lambda t, tor: f"{s(t)}-KTST-{'WFUS53' if tor else 'WWUS53'}-{'TORTST' if tor else 'SVSTST'}"  # noqa: E731
    iem_rows = [  # what IEM records for the same products (addsvs: CAN is not a state; it truncates)
        _iem_row("NEW", s(T0), s(T0 + 12 * m), pid(T0, True), issued=s(T0), expired=s(T0 + 45 * m), etn=7, xy=A0),
        _iem_row("CON", s(T0 + 12 * m), s(T0 + 40 * m), pid(T0 + 12 * m, False), issued=s(T0), expired=s(T0 + 45 * m), etn=7, xy=A1),
        _iem_row("EXP", s(T0 + 40 * m), s(T0 + 45 * m), pid(T0 + 40 * m, False), issued=s(T0), expired=s(T0 + 45 * m), etn=7, xy=A1),
        _iem_row("NEW", s(T0 + 5 * m), s(T0 + 20 * m), pid(T0 + 5 * m, True), issued=s(T0 + 5 * m), expired=s(T0 + 20 * m), etn=8, xy=B0),
        _iem_row("NEW", s(T0 + 10 * m), s(T0 + 25 * m), pid(T0 + 10 * m, True), issued=s(T0 + 10 * m), expired=s(T0 + 55 * m), etn=9, xy=C0),
        _iem_row("CON", s(T0 + 25 * m), s(T0 + 55 * m), pid(T0 + 25 * m, False), issued=s(T0 + 10 * m), expired=s(T0 + 55 * m), etn=9, xy=C1),
        # late EXP: IEM pushes the NEW state's end to the statement (nws_warnings clips it back to EXPIRED)
        _iem_row("NEW", s(T0 + 30 * m), s(T0 + 62 * m), pid(T0 + 30 * m, True), issued=s(T0 + 30 * m), expired=s(T0 + 60 * m), etn=10, xy=D0),
        _iem_row("EXP", s(T0 + 62 * m), s(T0 + 60 * m), pid(T0 + 62 * m, False), issued=s(T0 + 30 * m), expired=s(T0 + 60 * m), etn=10, xy=D0),
        _iem_row("NEW", s(T0 + 50 * m), s(T0 + 58 * m), pid(T0 + 50 * m, True), issued=s(T0 + 50 * m), expired=s(T0 + 80 * m), etn=7, xy=E0),
        _iem_row("CON", s(T0 + 58 * m), s(T0 + 80 * m), pid(T0 + 58 * m, False), issued=s(T0 + 50 * m), expired=s(T0 + 80 * m), etn=7, xy=E1),
    ]
    return api, iem_rows


def test_parity_synthetic_warning_as_iem_record_and_as_api_alerts():
    api, rows = _synthetic_sets()
    arrays, meta = nw.parse_iem_sbw_zip(_write_iem_zip(rows), expect_year=2024)  # IEM's own parser
    assert meta["clip_to_event_expiry"] == 1 and meta["empty_states"] == 1  # the late EXP, as IEM records it
    W_iem = nw._assemble([arrays], [2024], {"2024": meta})
    gx, gy = np.meshgrid(np.linspace(-98.3, -95.9, 25), np.linspace(34.5, 36.3, 19))
    times = np.unique(np.concatenate([np.arange(T0 - 120, T0 + 84 * 60, 60), np.arange(T0 - 61, T0 + 84 * 60, 60)]))
    lat = np.repeat(gy.ravel()[None, :], len(times), 0).ravel()
    lon = np.repeat(gx.ravel()[None, :], len(times), 0).ravel()
    t = np.repeat(times, gx.size)
    a, _, since = nw.tor_warning_state(lat, lon, t, warnings=W_iem, require_coverage=False)
    res = nl.inputs_from_snapshot(lat, lon, t, nl.AlertSnapshot(tuple(api), known_from=T0 - 4 * 3600, known_until=T0 + 2 * 3600))
    assert np.array_equal(res.matrix(), training_block_w(a, since), equal_nan=True)
    assert {"TST.TO.W.0007", "TST.TO.W.0008", "TST.TO.W.0009", "TST.TO.W.0010"} <= set(res.event)
    assert res.report["empty_states"] == 1 and res.report["action_counts"]["CAN"] == 2

    def at(la, lo, tt):  # one point, both paths
        r = nl.inputs_from_snapshot([la], [lo], [tt], nl.AlertSnapshot(tuple(api), known_from=T0 - 4 * 3600, known_until=T0 + 2 * 3600))
        aa, _, ss = nw.tor_warning_state([la], [lo], [tt], warnings=W_iem, require_coverage=False)
        assert np.array_equal(r.matrix(), training_block_w(aa, ss), equal_nan=True)
        return float(r.w_tor_warning_active[0]), float(r.w_minutes_since_issue[0])

    m = 60
    assert at(35.6, -97.3, T0 + 15 * m) == (1.0, 15.0)  # A1 and B0 overlap: earliest issuance (A, 20:00)
    assert at(35.9, -97.8, T0 + 11 * m) == (1.0, 11.0)  # in A0 only; the 20:12 SVS removes it ...
    assert at(35.9, -97.8, T0 + 12 * m)[0] == 0.0  # ... at 20:12
    assert at(35.3, -96.95, T0 + 11 * m)[0] == 0.0  # outside A0 ...
    assert at(35.3, -96.95, T0 + 12 * m) == (1.0, 12.0)  # ... gained by the SVS: minutes from 20:00, not 20:12
    assert at(36.0, -96.7, T0 + 19 * m) == (1.0, 14.0)  # B only, before the CAN
    assert at(36.0, -96.7, T0 + 20 * m)[0] == 0.0  # cancelled at 20:20 (CAP expires 20:35)
    assert at(34.8, -96.2, T0 + 30 * m) == (1.0, 20.0)  # C's remaining polygon after the partial CAN
    assert at(35.2, -96.7, T0 + 30 * m)[0] == 0.0  # C's cancelled part
    assert at(35.5, -97.3, T0 + 44 * m) == (1.0, 44.0)  # A's EXP-statement state, until 20:45
    assert at(35.5, -97.3, T0 + 45 * m)[0] == 0.0  # ended at the VTEC end (CAP expires 20:51)
    assert at(34.8, -98.0, T0 + 59 * m) == (1.0, 29.0)  # D in force to 21:00 despite the late EXP
    assert at(34.8, -98.0, T0 + 60 * m)[0] == 0.0
    assert at(35.7, -97.7, T0 + 49 * m)[0] == 0.0  # A (ETN 7) ended 20:45; E (ETN 7 again) not yet issued
    assert at(35.7, -97.7, T0 + 60 * m) == (1.0, 10.0)  # E's SVS state: from E's 20:50 NEW, not A's 20:00


# --------------------------------------------------------------------------
# 5. Fetch: requests, merge, failure -> typed error (never zeros)
# --------------------------------------------------------------------------


class _Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeAPI:
    """api.weather.gov stand-in: /alerts/active, and /alerts?start= filtered on sent, paginated."""

    def __init__(self, active_doc: dict, archive: list[dict], fail: str | None = None) -> None:
        self.active_doc, self.archive, self.fail = active_doc, archive, fail
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, req, timeout=None, context=None):
        url = req.full_url
        self.calls.append((url, req.get_header("User-agent")))
        u = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qs(u.query)
        history = u.path == "/alerts"
        if self.fail == "urlerror" or (self.fail == "history_down" and history):
            raise urllib.error.URLError("connection refused")
        if self.fail == "503":
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)
        if self.fail == "html":
            return _Resp(b"<html>maintenance</html>")
        if self.fail == "not_a_collection":
            return _Resp(b'{"title": "Not Found", "status": 404}')
        if u.path == "/alerts/active":
            return _Resp(json.dumps(self.active_doc).encode())
        assert history and q["event"] == ["Tornado Warning"]
        if "cursor" in q:
            return _Resp(json.dumps({"type": "FeatureCollection", "features": []}).encode())
        start = nl._iso_s(q["start"][0])
        feats = [f for f in self.archive if nl._iso_s(f["properties"]["sent"]) >= start]
        return _Resp(json.dumps({"type": "FeatureCollection", "features": feats, "pagination": {"next": url + "&cursor=x"}}).encode())


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(hp_http.time, "sleep", lambda s: None)


@pytest.mark.parametrize(
    "fail, n_requests",  # transport failures are retried (hazardpulse.data.http: 3 attempts), bad bodies are not
    [("urlerror", 3), ("503", 3), ("history_down", 1 + 3), ("html", 1), ("not_a_collection", 1)],
)
def test_network_failure_raises_a_typed_error_never_zeros(monkeypatch, no_sleep, api_features, fail, n_requests):
    active = json.loads(ACTIVE_JSON.read_text())
    api = FakeAPI(active, api_features, fail=fail)
    monkeypatch.setattr(hp_http.urllib.request, "urlopen", api)
    with pytest.raises(nl.NwsLiveFetchError):
        nl.live_tor_warning_inputs([39.117], [-98.34], [utc(2026, 9, 30, 23, 50)])
    assert issubclass(nl.NwsLiveFetchError, nl.NwsLiveError)
    assert len(api.calls) == n_requests


def test_fetch_requests_merges_and_answers_like_training(monkeypatch, no_sleep, api_features, iem):
    active = json.loads(ACTIVE_JSON.read_text())
    assert active["features"] == []  # no tornado warning in force at fetch time
    # the active feed lists the current message of one event; the history repeats it (merged by id)
    current = [f for f in api_features if ".KICT.TO.W.0030." in f["properties"]["parameters"]["VTEC"][0]
               and not f["properties"].get("replacedBy")]
    api = FakeAPI(dict(active, features=current), api_features)
    monkeypatch.setattr(hp_http.urllib.request, "urlopen", api)
    pts = [(39.117, -98.34), (39.024, -98.526), (34.99, -101.822)]
    ts = [utc(2026, 9, 30, 23, 50), utc(2026, 9, 30, 23, 50), utc(2026, 9, 30, 23, 50)]
    res = nl.live_tor_warning_inputs([p[0] for p in pts], [p[1] for p in pts], ts)
    urls = [c[0] for c in api.calls]
    assert urls[0] == nl.ACTIVE_URL and len(urls) == 3  # active, history page 1, empty page 2
    assert urls[1] == "https://api.weather.gov/alerts?event=Tornado%20Warning&start=2026-09-30T20:50:00Z&limit=500"
    assert all(c[1] == "HazardPulse/1.0 (hazardpulse.com; josh@coherenceenergylabs.com)" for c in api.calls)
    start = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[1]).query)["start"][0]
    assert start == "2026-09-30T20:50:00Z"  # t_min - 3 h, the longest a state may outlive its issuance
    assert res.report["known_from"] == utc(2026, 9, 30, 20, 50)
    assert res.report["known_until"] == min(utc(2026, 10, 2, 23, 23, 17), res.report["fetched_at"])
    in_window = [f for f in api_features if nl._iso_s(f["properties"]["sent"]) >= utc(2026, 9, 30, 20, 50)]
    assert len(current) == 2 and all(f in in_window for f in current)
    assert res.report["features"] == len(in_window) and res.report["duplicate_ids"] == 0  # active U history, by id
    a, _, s = nw.tor_warning_state([p[0] for p in pts], [p[1] for p in pts], ts, warnings=iem, require_coverage=False)
    assert np.array_equal(res.matrix(), training_block_w(a, s), equal_nan=True)
    assert res.w_tor_warning_active.tolist() == [1.0, 0.0, 0.0] and res.w_minutes_since_issue[0] == 36.0
    assert res.matrix().dtype == np.float32 and nl.W_NAMES == ("w_tor_warning_active", "w_minutes_since_issue")


def test_no_warning_in_force_is_reported_only_from_a_successful_fetch(monkeypatch, no_sleep):
    active = json.loads(ACTIVE_JSON.read_text())
    api = FakeAPI(active, [])
    monkeypatch.setattr(hp_http.urllib.request, "urlopen", api)
    t = utc(2026, 10, 2, 23, 20)
    res = nl.live_tor_warning_inputs([35.0, 40.0], [-97.0, -90.0], [t, t])
    assert res.w_tor_warning_active.tolist() == [0.0, 0.0] and np.isnan(res.w_minutes_since_issue).all()
    assert res.report["messages"] == 0 and len(api.calls) == 2  # active + one (empty) history page


def test_coverage_refuses_instants_the_snapshot_cannot_answer():
    K, U = utc(2026, 9, 30, 20, 0), utc(2026, 9, 30, 23, 0)
    s = nl.AlertSnapshot((), known_from=K, known_until=U)
    lo = K + nw.MAX_STATE_LIFETIME_S
    for t in (lo, U + nl.DEFAULT_MAX_FUTURE_S):
        nl.inputs_from_snapshot([35.0], [-97.0], [t], s)
    for t in (lo - 1, U + nl.DEFAULT_MAX_FUTURE_S + 1):
        with pytest.raises(nl.NwsLiveCoverageError):
            nl.inputs_from_snapshot([35.0], [-97.0], [t], s)
    nl.inputs_from_snapshot([35.0], [-97.0], [lo - 1], s, require_coverage=False)
    active_only = nl.AlertSnapshot((), known_from=None, known_until=U)  # answers only from its own instant
    nl.inputs_from_snapshot([35.0], [-97.0], [U], active_only)
    with pytest.raises(nl.NwsLiveCoverageError):
        nl.inputs_from_snapshot([35.0], [-97.0], [U - 60], active_only)


@pytest.mark.skipif(os.environ.get("HAZARDPULSE_NETWORK_TESTS") != "1", reason="set HAZARDPULSE_NETWORK_TESTS=1 to hit api.weather.gov")
def test_live_api_answers_now():
    now = datetime.now(timezone.utc)
    t = int(now.timestamp()) - 120
    s = nl.fetch_alert_snapshot(t)
    assert s.known_from == (t - nw.MAX_STATE_LIFETIME_S) // 60 * 60
    assert 0 <= int(now.timestamp()) - s.known_until < 600, "the active feed's ingest instant is stale"
    res = nl.inputs_from_snapshot([35.0, 39.0], [-97.0, -98.0], [t, t], s)
    assert res.matrix().shape == (2, 2) and res.report["skipped_not_tornado_warning"] == 0
    # our history query selects exactly what the API's own pagination form selects (a broken event
    # filter would return nothing); meaningful whenever the API's ~week of retention holds a warning
    start = t - 6 * 86400
    ours = json.loads(hp_http.fetch_bytes(nl.history_url(start), use_cache=False, user_agent=nl.USER_AGENT))["features"]
    native = nl.API_ROOT + "/alerts?event%5B0%5D=Tornado%20Warning&limit=500&start=" + nl.history_url(start).split("start=")[1].split("&")[0]
    theirs = json.loads(hp_http.fetch_bytes(native, use_cache=False, user_agent=nl.USER_AGENT))["features"]
    assert sorted(f["id"] for f in ours) == sorted(f["id"] for f in theirs)
    assert all(f["properties"]["event"] == "Tornado Warning" for f in ours)
