"""Hurricane RI verification must not invent outcomes.

Regression for the 2026-10 findings:
* a storm with no best track (every West Pacific storm 404s in NHC's btk folder) was scored
  ri_occurred=False, and 41 no-storm forecasts were averaged in as perfect Brier 0.0 entries -- the
  published mean Brier 0.0014 rested on zero verified outcomes;
* the truth was aligned to the time the scorer RAN, not to the storm's synoptic time: a 04:54 run for
  the 00Z cycle was verified 04:54 -> 04:54 the next day (Nolo 2026-10-03 00Z, +30 kt, scored as no RI);
* every repeat run of one cycle was scored as another forecast (Nolo 2026-10-03 12Z: six times);
* a "mean AUC" averaged AUCs computed inside single forecasts of three storms or fewer.

The time constants below are deliberately OFF the synoptic hour for the runs: with on-the-hour issue
times the run-time alignment was invisible to every test.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane.atcf import ATCFRecord

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shp():
    name = "hp_shp_verif_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "score_hurricane_prospective.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rec(cycle: dt.datetime, vmax: float, basin: str = "WP") -> ATCFRecord:
    return ATCFRecord(basin=basin, storm_number=4, cycle=cycle, tau_hours=0, model="BEST",
                      lat=10.0, lon=150.0, vmax_kt=vmax, mslp_hpa=None)


T = dt.datetime(2026, 4, 13, 12, 0)          # the storm's synoptic time
RUN = "2026-04-13T15:41:07Z"                 # when the forecast file was made: off the hour, t + 3 h 41


def _z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _track(vals: dict[int, float], t0: dt.datetime = T, basin: str = "WP"):
    return [_rec(t0 + dt.timedelta(hours=h), v, basin) for h, v in vals.items()]


def test_calibration_pools_only_the_served_model(shp):
    """v8.1 and v8.2 forecasts must never share a calibrator: the dataset pools
    only the version behind the newest matured forecast, and says which."""
    cyc = dt.datetime(2026, 10, 2, 0)
    old = {"forecast_id": "a", "issued_at": "2026-05-01T12:00:00Z",
           "storms": [{"storm_id": "AL012026", "issue_time": "2026-05-01T06:00:00", "ri_probability": 0.2,
                       "model_version": "hurricane_ri_v8_1"}]}
    new = {"forecast_id": "b", "issued_at": "2026-10-02T04:07:31Z",
           "storms": [{"storm_id": "EP152026", "issue_time": _z(cyc), "ri_probability": 0.006,
                       "model_version": "hurricane_ri_v8_2"},
                      {"storm_id": "EP192026", "issue_time": _z(cyc), "ri_probability": 0.013}]}
    assert shp.newest_model_version([old, new]) == "hurricane_ri_v8_2"
    assert shp.newest_model_version([]) == shp.LEGACY_HURRICANE_MODEL

    track = (_track({0: 60.0, 24: 65.0}, cyc), "test")
    acc = {shp._CALIB_VERSION_KEY: "hurricane_ri_v8_2"}
    shp.score_single_forecast(new, calib_acc=acc, best_track_fetcher=lambda sid: track)
    pooled = {k: v for k, v in acc.items() if k != shp._CALIB_VERSION_KEY}
    # EP15 is v8.2; EP19 has no per-storm version and inherits none -> legacy, excluded
    assert sum(t for t, _ in pooled.values()) == 1
    shp.score_single_forecast(old, calib_acc=acc, best_track_fetcher=lambda sid: track)
    assert sum(v[0] for k, v in acc.items() if k != shp._CALIB_VERSION_KEY) == 1


def test_no_best_track_is_unverifiable_not_no_ri(shp):
    assert shp.check_ri_occurred([], T) == (None, None, None)


def test_track_ending_before_the_window_is_unverifiable(shp):
    ri, v0, v1 = shp.check_ri_occurred(_track({0: 55.0, 6: 60.0}), T)
    assert ri is None and v0 == 55.0 and v1 is None


def test_bracketing_track_gives_a_real_outcome(shp):
    assert shp.check_ri_occurred(_track({0: 60.0, 24: 95.0}), T)[0] is True
    assert shp.check_ri_occurred(_track({0: 60.0, 24: 90.0}), T)[0] is True           # exactly +30 kt
    assert shp.check_ri_occurred(_track({0: 145.0, 24: 115.0}), T)[0] is False


def test_the_outcome_is_the_storms_own_24_hours_never_the_run_times(shp):
    """Nolo 2026-10-03 00Z: 70 kt at 00Z, 100 kt at 00Z the next day (+30 kt, RI). The run that published
    it was made at 04:54; aligning the truth to the run time read 90 kt (the 06Z fix) -> 95 kt and
    scored it 'no RI'. The storm's own cycle decides."""
    t0 = dt.datetime(2026, 10, 3, 0)
    track = (_track({0: 70.0, 6: 90.0, 12: 95.0, 18: 100.0, 24: 100.0, 30: 95.0}, t0, "EP"), "btk")
    art = {"forecast_id": "hu_fcst_20261003_0454", "issued_at": "2026-10-03T04:54:56Z",
           "storms": [{"storm_id": "EP152026", "issue_time": "2026-10-03T00:00:00", "ri_probability": 0.45,
                       "model_version": "m"}]}
    p = shp.score_single_forecast(art, best_track_fetcher=lambda sid: track)["predictions"][0]
    assert (p["ri_occurred"], p["vmax_at_issue"], p["vmax_at_end"]) == (True, 70.0, 100.0)
    assert p["cycle"] == "2026-10-03T00:00:00Z" and p["lag_hours"] == pytest.approx(4.9156, abs=1e-4)
    # a fix missing at t + 24 h is undecided, even when a neighbouring fix exists
    gap = (_track({0: 70.0, 18: 100.0, 30: 95.0}, t0, "EP"), "btk")
    assert shp.score_single_forecast(art, best_track_fetcher=lambda sid: gap)["predictions"][0]["ri_occurred"] is None


def _cycle_records(sid="EP152026", cyc="2026-10-03T12:00:00"):
    """Six runs of ONE cycle (Nolo 12Z on 2026-10-03): one before the advisory, five after."""
    runs = [("hu_fcst_20261003_1248", "2026-10-03T12:48:02Z", 0.18),       # t + 48 min: preliminary
            ("hu_fcst_20261003_1641", "2026-10-03T16:41:17Z", 0.20),       # t + 4 h 41: the record
            ("hu_fcst_20261003_1804", "2026-10-03T18:04:32Z", 0.21),
            ("hu_fcst_20261003_1846", "2026-10-03T18:46:31Z", 0.22)]
    return [{"forecast_id": fid, "issued_at": at,
             "storms": [{"storm_id": sid, "issue_time": cyc, "ri_probability": p, "model_version": "m"}]}
            for fid, at, p in runs]


def test_a_storm_cycle_is_scored_once_from_its_first_record_after_the_advisory(shp, tmp_path):
    arts = _cycle_records()
    sel, aside = shp.select_published(arts)
    assert [c.ref for c in sel.chosen.values()] == [("hu_fcst_20261003_1641", 0)]
    assert aside[("hu_fcst_20261003_1248", 0)].startswith("made before t + 3 h 30 min")
    assert all(aside[(f, 0)].startswith("a later record") for f in ("hu_fcst_20261003_1804", "hu_fcst_20261003_1846"))
    track = (_track({0: 95.0, 24: 95.0}, dt.datetime(2026, 10, 3, 12), "EP"), "btk")
    results = [shp.score_single_forecast(a, best_track_fetcher=lambda sid: track, set_aside=aside) for a in arts]
    scored = [p for r in results for p in r["predictions"]]
    assert [p["predicted_ri_probability"] for p in scored] == [0.20]
    summary = shp.summarize_results(results, sel)
    assert summary["total_storm_predictions"] == 1 and summary["total_storms_scored"] == 1
    assert summary["set_aside_records"] == {"made before t + 3 h 30 min (preliminary inputs)": 1,
                                            "a later record of a cycle that already has one": 2}
    # a cycle whose every record is preliminary has no scored record, and says so
    only_early = _cycle_records(cyc="2026-10-03T18:00:00")[3:]               # 18:46 for the 18Z cycle
    sel2, _ = shp.select_published(only_early)
    assert not sel2.chosen and sel2.cycles_without_a_record() == [("EP152026", dt.datetime(2026, 10, 3, 18))]


def test_main_scores_each_storm_cycle_once(shp, tmp_path, monkeypatch):
    import json
    replay = tmp_path / "replay"
    replay.mkdir()
    for a in _cycle_records():
        (replay / f"{a['forecast_id']}.json").write_text(json.dumps(a), encoding="utf-8")
    track = (_track({0: 95.0, 24: 130.0}, dt.datetime(2026, 10, 3, 12), "EP"), "btk")
    monkeypatch.setattr(shp, "fetch_best_track_with_source", lambda sid: track)
    shp.main(["--replay-dir", str(replay), "--output-dir", str(tmp_path / "out"),
              "--score-as-of", "2026-10-07T00:00:00Z", "--emit-calibration"])
    s = json.loads((tmp_path / "out" / "prospective_summary.json").read_text(encoding="utf-8"))
    assert s["n_matured_forecasts"] == 4 and s["total_storms_scored"] == 1 and s["total_ri_events"] == 1
    assert s["calibration_n"] == 1                                           # the pool holds the cycle once
    assert "mean_auc" not in s and s["pooled"]["auc"] is None                # one cycle: no ranking score


def test_unverifiable_storms_and_null_forecasts_are_not_scored(shp):
    artifact_storms = {
        "forecast_id": "hu_fcst_20260413_1541",
        "issued_at": RUN,
        "storms": [
            {"storm_id": "WP042026", "issue_time": _z(T), "ri_probability": 0.19},
            {"storm_id": "WP052026", "issue_time": _z(T), "ri_probability": 0.30},
            {"storm_id": "WP042026", "issue_time": None, "ri_probability": 0.0},     # names no cycle
        ],
    }
    tracks = {
        "WP042026": (_track({0: 145.0, 24: 115.0}), "src"),
        "WP052026": ([], None),
    }
    calib: dict = {}
    res = shp.score_single_forecast(artifact_storms, calib_acc=calib,
                                    best_track_fetcher=lambda sid: tracks[sid])
    assert res["n_verified"] == 1 and res["n_unverifiable"] == 1
    assert res["set_aside"] == [{"storm_id": "WP042026", "issue_time": None,
                                 "why": "the record names no synoptic time"}]
    assert res["brier"] == pytest.approx(0.19 ** 2)                  # only the verified storm
    assert sum(slot[0] for slot in calib.values()) == 1               # calibration pool too

    null = shp.score_single_forecast({"forecast_id": "hu_fcst_x", "issued_at": "2026-05-01T12:00:00Z",
                                      "storms": []})
    assert null["null_forecast"] is True and null["brier"] is None

    summary = shp.summarize_results([res, null])
    assert summary["pooled"]["brier"] == pytest.approx(0.19 ** 2)   # not diluted by the null
    assert summary["n_unverifiable_predictions"] == 1
    assert summary["unverifiable_by_basin"] == {"wp": 1}
    assert summary["total_storms_scored"] == 1


def test_nothing_verified_means_no_brier(shp):
    res = shp.score_single_forecast(
        {"forecast_id": "hu_fcst_y", "issued_at": RUN,
         "storms": [{"storm_id": "WP042026", "issue_time": _z(T), "ri_probability": 0.0}]},
        best_track_fetcher=lambda sid: ([], None))
    summary = shp.summarize_results([res])
    assert res["brier"] is None and summary["pooled"]["brier"] is None
    assert summary["ri_rate"] is None


def _pred(sid, p, y, version="m"):
    return {"storm_id": sid, "predicted_ri_probability": p, "ri_occurred": bool(y), "model_version": version}


def test_a_ranking_score_needs_both_outcomes_and_more_than_a_handful_of_storms(shp):
    """The old summary averaged AUCs computed inside single forecasts of <= 3 storms. Pooled now, with its
    counts, and withheld when it cannot mean anything."""
    four = [_pred(f"EP{i:02d}2026", 0.1 * (i + 1), i == 3) for i in range(4)]
    m = shp.pooled_metrics(four)
    assert m["auc"] is None and "4 storms" in m["auc_withheld"] and m["n_storm_cycles"] == 4
    assert shp.pooled_metrics([_pred("EP012026", 0.2, 0)] * 6)["auc_withheld"].startswith("only one outcome")
    six = four + [_pred("EP052026", 0.05, 0), _pred("EP062026", 0.9, 1)]
    m6 = shp.pooled_metrics(six)
    # positives 0.4 and 0.9; negatives 0.1, 0.2, 0.3, 0.05: every pair ranked right
    assert m6["auc"] == pytest.approx(1.0) and m6["n_events"] == 2 and m6["n_storms"] == 6
    s = shp.summarize_results([{"predictions": six[:3]}, {"predictions": six[3:]}])
    assert "mean_auc" not in s and "median_auc" not in s
    assert s["pooled"]["auc"] == pytest.approx(1.0)                         # pooled over both files
    assert set(s["by_model_version"]) == {"m"}


def test_the_calibration_pool_takes_the_number_the_trust_layer_is_applied_to(shp):
    """The trust layer is applied to ``ri_probability`` and keeps what it was given as ``raw_probability``.
    ``ri_probability_raw`` is v8.2's ensemble BEFORE v8.2's own calibration -- never a calibrator input."""
    assert shp.calibrator_input({"ri_probability": 0.19, "ri_probability_raw": 0.0954}) == (0.19, "ri_probability")
    assert shp.calibrator_input({"ri_probability": 0.25, "raw_probability": 0.19,
                                 "ri_probability_raw": 0.0954}) == (0.19, "raw_probability")
    acc = {shp._CALIB_VERSION_KEY: "v"}
    art = {"forecast_id": "f", "issued_at": RUN, "storms": [
        {"storm_id": "WP042026", "issue_time": _z(T), "ri_probability": 0.1909, "ri_probability_raw": 0.0954,
         "model_version": "v"}]}
    p = shp.score_single_forecast(art, calib_acc=acc, best_track_fetcher=lambda sid: (_track({0: 60, 24: 60}), "s"))
    assert {k: v for k, v in acc.items() if k != shp._CALIB_VERSION_KEY} == {0.1909: [1, 0]}
    assert p["predictions"][0]["calibrator_input_key"] == "ri_probability"


def test_west_pacific_storms_have_a_best_track_source(shp):
    assert shp.best_track_urls("WP042026") == [
        "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/2026/bwp042026.dat"]
    urls = shp.best_track_urls("AL052026")
    assert urls[0].endswith("/btk/bal052026.dat") and "bdecks_open/2026/bal052026.dat" in urls[1]
