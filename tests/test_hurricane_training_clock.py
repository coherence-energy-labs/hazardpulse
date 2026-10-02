"""The hurricane RI training set runs on the real clock (v8.2).

v8.1's builder stepped through IBTrACS rows as if they were 6-hourly, but 94.1% of its
consecutive rows are 3 h apart: its "24 h" RI label spanned 12 h, dv_6/12/24h were
3/6/12-h changes, translation speed was halved and storm age doubled. These tests pin the
corrected builder on synthetic tracks AND audit the committed files themselves: every
label and lag in the v8.2 file must agree with the file's own rows 24 h later / earlier --
an audit the v8.1 file fails, which is what proves the audit can fail.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.hurricane import operational_ri as ori  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "bhtd_clock_test", REPO / "scripts" / "build_hurricane_training_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(start: dt.datetime, hours: list[float], winds: list[float], step_lat: float = 0.2):
    entries = []
    for i, (h, w) in enumerate(zip(hours, winds)):
        t = start + dt.timedelta(hours=h)
        entries.append({"wind": w, "pres": 1010.0 - w / 2, "lat": 15.0 + step_lat * i,
                        "lon": -50.0 - 0.3 * i, "year": t.year, "month": t.month,
                        "time_str": t.strftime("%Y-%m-%d %H:%M:%S")})
    return {"S1": {"name": "TEST", "basin": "NA", "entries": entries}}


def test_labels_and_lags_are_real_time_offsets(builder):
    start = dt.datetime(2010, 9, 1, 0)
    hours = [3 * k for k in range(17)]                      # 3-hourly fixes over 48 h, like IBTrACS
    winds = [30.0 + 2.5 * k for k in range(17)]             # +5 kt per 6 h = +20 kt per 24 h
    winds[12] = 100.0                                       # t = 36 h: a 24-h jump from t = 12 h
    cases = builder.extract_ri_cases(_track(start, hours, winds))
    by_time = {c["issue_time"]: c for c in cases}
    assert all(dt.datetime.fromisoformat(t).hour % 6 == 0 for t in by_time), "synoptic times only"
    assert sorted(by_time) == [  # a row needs its t+24 h fix: 00/06/12/18 h and 24 h only
        "2010-09-01 00:00:00", "2010-09-01 06:00:00", "2010-09-01 12:00:00",
        "2010-09-01 18:00:00", "2010-09-02 00:00:00"]
    c12 = by_time["2010-09-01 12:00:00"]
    assert c12["ri_label_30kt"] == 1 and c12["analysis_vmax_kt"] == 40.0   # 100 - 40 >= 30 over 24 h
    assert by_time["2010-09-01 06:00:00"]["ri_label_30kt"] == 0            # 35 -> 55 kt: +20 only
    c24 = by_time["2010-09-02 00:00:00"]
    assert c24["analysis_dv_6h"] == winds[8] - winds[6]     # 24 h vs 18 h, not 24 h vs 21 h
    assert c24["analysis_dv_12h"] == winds[8] - winds[4]
    assert c24["analysis_dv_24h"] == winds[8] - winds[0]
    assert by_time["2010-09-01 06:00:00"]["analysis_dv_12h"] is None       # no fix 12 h earlier
    assert c24["storm_age_h"] == 24.0
    assert c24["translation_speed_kmh"] == ori.translation_speed_kmh(
        15.0 + 0.2 * 6, -50.0 - 0.3 * 6, 15.0 + 0.2 * 8, -50.0 - 0.3 * 8, 6.0)


def test_a_gap_in_the_track_yields_none_not_a_wrong_delta(builder):
    start = dt.datetime(2011, 8, 1, 0)
    hours = [0, 6, 12, 30, 36, 42, 48, 54, 60]              # 12-h gap after 12 h
    winds = [40.0, 45, 50, 55, 60, 65, 70, 75, 80]
    cases = {c["issue_time"]: c for c in builder.extract_ri_cases(_track(start, hours, winds))}
    c36 = cases["2011-08-02 12:00:00"]                      # t = 36 h
    assert c36["analysis_dv_6h"] == 5.0                     # 36 vs 30 h exists
    assert c36["analysis_dv_12h"] is None                   # 24 h fix missing
    assert c36["analysis_dv_24h"] == 10.0                   # 36 vs 12 h exists
    assert "2011-08-01 18:00:00" not in cases               # no 18 h fix at all


def _audit(path: Path) -> tuple[int, int, int]:
    """(rows checked, label disagreements, dv_24h disagreements) against the file's own rows."""
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                c = json.loads(line)
                rows[(c["storm_id"], c["issue_time"])] = c
    checked = bad_label = bad_dv = 0
    for (sid, t), c in rows.items():
        when = dt.datetime.fromisoformat(t)
        later = rows.get((sid, (when + dt.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")))
        earlier = rows.get((sid, (when - dt.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")))
        if later is not None:
            checked += 1
            truth = int(later["analysis_vmax_kt"] - c["analysis_vmax_kt"] >= 30)
            bad_label += truth != c["ri_label_30kt"]
        if earlier is not None and c.get("analysis_dv_24h") is not None:
            bad_dv += c["analysis_dv_24h"] != c["analysis_vmax_kt"] - earlier["analysis_vmax_kt"]
    return checked, bad_label, bad_dv


def test_committed_v8_2_set_agrees_with_its_own_clock():
    checked, bad_label, bad_dv = _audit(ri_model.DATASETS["v8.2"])
    assert checked > 40_000
    assert bad_label == 0 and bad_dv == 0, (bad_label, bad_dv)


def test_the_same_audit_fails_on_the_legacy_v8_1_set():
    checked, bad_label, bad_dv = _audit(ri_model.DATASETS["v8.1"])
    assert checked > 40_000
    assert bad_label > 1_000 and bad_dv > 10_000, (bad_label, bad_dv)


def test_committed_v8_2_set_is_what_the_builder_documents():
    rates = {"rows": 0, "events": 0}
    with ri_model.DATASETS["v8.2"].open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            rates["rows"] += 1
            rates["events"] += c["ri_label_30kt"]
            assert dt.datetime.fromisoformat(c["issue_time"]).hour % 6 == 0
    assert rates["rows"] == 69_722 and rates["events"] == 4_268
