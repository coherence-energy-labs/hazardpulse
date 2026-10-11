"""After a large earthquake (earthquake program section 13): the scorer records every shallow M7+ of the last 30
days, and the earthquake page shows what history says followed such earthquakes -- bound to the measured rates --
beside the forecast's own number for that cell."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hazardpulse.site import fmt
from hazardpulse.site.pages import earthquake
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]
NOW = dt.datetime(2026, 10, 10, 18, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def fse():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fse_after_t", ROOT / "scripts" / "fetch_and_score_earthquake.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ev(t, mag, depth, lat=7.587, lon=-80.769, eid="x"):
    return {"time": t, "latitude": lat, "longitude": lon, "depth": depth, "mag": mag, "id": eid, "place": "off Panama"}


def test_the_scorer_keeps_shallow_m7s_of_the_last_30_days_newest_first(fse):
    events = [_ev("2026-10-09T17:56:06Z", 7.7, 12.6, eid="main"), _ev("2026-10-09T20:25:39Z", 6.6, 10.0, eid="m66"),
              _ev("2026-09-15T00:00:00Z", 7.4, 300.0, eid="deep"), _ev("2026-09-15T00:00:00Z", 7.0, 20.0, eid="old7"),
              _ev("2026-08-01T00:00:00Z", 7.9, 20.0, eid="too_old"), _ev("2026-10-10T17:00:00Z", 7.1, 30.0, eid="recent")]
    got = fse.recent_large_earthquakes(events, NOW)
    assert [e["id"] for e in got] == ["recent", "main", "old7"]          # deep, M6.6 and >30 days left out
    assert got[1]["mag"] == 7.7 and got[1]["depth_km"] == 12.6 and got[1]["time"] == "2026-10-09T17:56:06Z"


class _D:
    def __init__(self, replay, evidence):
        self.earthquake, self.evidence = replay, evidence


def _replay(quakes, p_cell=0.173):
    grid = [0.0001] * 11700
    row, col = int((7.587 + 60) / 2), int((-80.769 + 180) / 2)
    grid[row * 180 + col] = p_cell
    return {"issued_at": "2026-10-11T00:00:00Z", "recent_large_earthquakes": quakes,
            "forecast_domain": {"lat_min": -60.0, "lon_min": -180.0, "dlat": 2.0, "dlon": 2.0, "n_lat": 65, "n_lon": 180},
            "probability_grid": ",".join(str(x) for x in grid)}


def test_the_panel_quotes_the_measured_rate_for_the_class_and_elapsed_time():
    hist = se.earthquake_aftershock_history()
    q = {"id": "us6000u18k", "time": "2026-10-09T17:56:06Z", "latitude": 7.587, "longitude": -80.769, "depth_km": 12.6,
         "mag": 7.7, "place": "off Panama"}
    html = earthquake._after_large(_D(_replay([q]), {"earthquake_aftershock": hist}))
    r = hist["classes"]["M7.5+"]["by_lag_h"][24.0]                      # 30 hours after: the 1-day figure
    assert f"<strong>{fmt.pct(r['rate'], 0)}</strong>" in html and "a day after it" in html
    assert fmt.pct(r["ci"][0], 0) in html and fmt.pct(r["ci"][1], 0) in html
    assert "<strong>17.3%</strong>" in html and "M7.7" in html and "not a forecast for this earthquake" in html
    q2 = dict(q, time="2026-10-10T23:40:00Z", mag=7.2)                   # 20 minutes old, M7.2: the half-hour figure
    html2 = earthquake._after_large(_D(_replay([q2]), {"earthquake_aftershock": hist}))
    r2 = hist["classes"]["M7-7.5"]["by_lag_h"][0.5]
    assert f"<strong>{fmt.pct(r2['rate'], 0)}</strong>" in html2 and "half an hour after it" in html2


def test_no_panel_without_a_large_earthquake_or_without_the_measured_rates():
    hist = se.earthquake_aftershock_history()
    assert earthquake._after_large(_D(_replay([]), {"earthquake_aftershock": hist})) == ""
    q = {"time": "2026-10-09T17:56:06Z", "latitude": 7.587, "longitude": -80.769, "depth_km": 12.6, "mag": 7.7}
    assert earthquake._after_large(_D(_replay([q]), {"earthquake_aftershock": None})) == ""
    assert earthquake._after_large(_D({k: v for k, v in _replay([q]).items() if k != "recent_large_earthquakes"},
                                      {"earthquake_aftershock": hist})) == ""


PANAMA_CELL = str(int((7.587 + 60) / 2) * 180 + int((-80.769 + 180) / 2))


def test_the_aftershock_model_in_test_is_shown_beside_the_published_number_never_instead():
    """Section 15.5: E7's live shadow for the quake's cell, labelled as in test, with its held-out read bound."""
    hist, held = se.earthquake_aftershock_history(), se.earthquake_aftershock_in_test()
    assert held is not None and held["n_mainshocks"] == 88
    q = {"time": "2026-10-09T17:56:06Z", "latitude": 7.587, "longitude": -80.769, "depth_km": 12.6, "mag": 7.7}
    replay = dict(_replay([q]), shadow_aftershock_e7={"cells": {PANAMA_CELL: 0.346}})
    html = earthquake._after_large(_D(replay, {"earthquake_aftershock": hist, "earthquake_aftershock_in_test": held}))
    assert "<strong>17.3%</strong>" in html                              # the published number stays
    assert "in test gives it <strong>34.6%</strong> (not the published number)" in html
    assert "checked on 88 large earthquakes it was not fitted on" in html
    assert fmt.pct(held["day_after"]["E7"], 0) in html and fmt.pct(held["day_after"]["S2"], 0) in html
    # without the shadow, or without the bound held-out read, nothing of the model in test is said
    for r, ev in ((_replay([q]), {"earthquake_aftershock": hist, "earthquake_aftershock_in_test": held}),
                  (replay, {"earthquake_aftershock": hist, "earthquake_aftershock_in_test": None}),
                  (dict(_replay([q]), shadow_aftershock_e7={"cells": {"0": 0.5}}),
                   {"earthquake_aftershock": hist, "earthquake_aftershock_in_test": held})):
        plain = earthquake._after_large(_D(r, ev))
        assert "<strong>17.3%</strong>" in plain and "in test" not in plain and "fitted on" not in plain


def _copy_e7(tmp_path, mutate=None):
    for rel in (se.AFTERSHOCK_E7_FIT, se.AFTERSHOCK_E7_HELDOUT):
        d = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        if mutate:
            mutate(rel, d)
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(d), encoding="utf-8")


def test_the_in_test_read_is_bound_to_the_fitted_parameters_and_to_its_pass(tmp_path):
    assert se.earthquake_aftershock_in_test(tmp_path) is None                       # no files: nothing shown
    _copy_e7(tmp_path)
    assert se.earthquake_aftershock_in_test(tmp_path) == se.earthquake_aftershock_in_test()

    def other_params(rel, d):
        if rel == se.AFTERSHOCK_E7_HELDOUT:
            d["params"]["K"] *= 1.01
    _copy_e7(tmp_path, other_params)
    with pytest.raises(se.EvidenceError):
        se.earthquake_aftershock_in_test(tmp_path)

    def failed(rel, d):
        if rel == se.AFTERSHOCK_E7_HELDOUT:
            d["logloss_E7_minus_S2"]["passes"] = False
    _copy_e7(tmp_path, failed)
    assert se.earthquake_aftershock_in_test(tmp_path) is None                       # a failed part 1 is never shown

    def no_day_after(rel, d):
        if rel == se.AFTERSHOCK_E7_HELDOUT:
            del d["reliability"]["mainshock cell +24 h"]
    _copy_e7(tmp_path, no_day_after)
    with pytest.raises(se.EvidenceError):
        se.earthquake_aftershock_in_test(tmp_path)


def test_a_malformed_rates_file_is_refused(tmp_path):
    d = json.loads((ROOT / se.AFTERSHOCK_BASE_RATES).read_text(encoding="utf-8"))
    d["classes"]["M7.5+"]["by_lag_h"]["0.5"]["rate"] = 1.7
    (tmp_path / se.AFTERSHOCK_BASE_RATES).parent.mkdir(parents=True)
    (tmp_path / se.AFTERSHOCK_BASE_RATES).write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(se.EvidenceError):
        se.earthquake_aftershock_history(tmp_path)
