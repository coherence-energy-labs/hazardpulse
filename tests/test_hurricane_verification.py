"""Hurricane RI verification must not invent outcomes.

Regression for the 2026-10 finding: a storm with no best track (every West Pacific
storm 404s in NHC's btk folder) was scored ri_occurred=False, and 41 no-storm
forecasts were averaged in as perfect Brier 0.0 entries -- the published mean Brier
0.0014 rested on zero verified outcomes."""

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


def _rec(cycle: dt.datetime, vmax: float) -> ATCFRecord:
    return ATCFRecord(basin="WP", storm_number=4, cycle=cycle, tau_hours=0, model="BEST",
                      lat=10.0, lon=150.0, vmax_kt=vmax, mslp_hpa=None)


ISSUE = dt.datetime(2026, 4, 13, 15, 0)


def test_calibration_pools_only_the_served_model(shp):
    """v8.1 and v8.2 forecasts must never share a calibrator: the dataset pools
    only the version behind the newest matured forecast, and says which."""
    old = {"forecast_id": "a", "issued_at": "2026-05-01T12:00:00Z",
           "storms": [{"storm_id": "AL012026", "ri_probability": 0.2, "model_version": "hurricane_ri_v8_1"}]}
    new = {"forecast_id": "b", "issued_at": "2026-10-02T04:00:00Z",
           "storms": [{"storm_id": "EP152026", "ri_probability": 0.006, "model_version": "hurricane_ri_v8_2"},
                      {"storm_id": "EP192026", "ri_probability": 0.013}]}
    assert shp.newest_model_version([old, new]) == "hurricane_ri_v8_2"
    assert shp.newest_model_version([]) == shp.LEGACY_HURRICANE_MODEL

    issue = dt.datetime(2026, 10, 2, 4, 0)
    track = ([_rec(issue, 60.0), _rec(issue + dt.timedelta(hours=24), 65.0)], "test")
    acc = {shp._CALIB_VERSION_KEY: "hurricane_ri_v8_2"}
    shp.score_single_forecast(new, calib_acc=acc, best_track_fetcher=lambda sid: track)
    pooled = {k: v for k, v in acc.items() if k != shp._CALIB_VERSION_KEY}
    # EP15 is v8.2; EP19 has no per-storm version and inherits none -> legacy, excluded
    assert sum(t for t, _ in pooled.values()) == 1
    shp.score_single_forecast(old, calib_acc=acc, best_track_fetcher=lambda sid: track)
    assert sum(v[0] for k, v in acc.items() if k != shp._CALIB_VERSION_KEY) == 1


def test_no_best_track_is_unverifiable_not_no_ri(shp):
    assert shp.check_ri_occurred([], ISSUE) == (None, None, None)


def test_track_ending_before_the_window_is_unverifiable(shp):
    recs = [_rec(ISSUE - dt.timedelta(hours=3), 55.0)]
    ri, v0, v1 = shp.check_ri_occurred(recs, ISSUE)
    assert ri is None and v0 == 55.0 and v1 is None


def test_bracketing_track_gives_a_real_outcome(shp):
    recs = [_rec(ISSUE, 60.0), _rec(ISSUE + dt.timedelta(hours=24), 95.0)]
    assert shp.check_ri_occurred(recs, ISSUE)[0] is True
    recs = [_rec(ISSUE, 145.0), _rec(ISSUE + dt.timedelta(hours=24), 115.0)]
    assert shp.check_ri_occurred(recs, ISSUE)[0] is False


def test_west_pacific_storms_have_a_best_track_source(shp):
    assert shp.best_track_urls("WP042026") == [
        "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/2026/bwp042026.dat"]
    urls = shp.best_track_urls("AL052026")
    assert urls[0].endswith("/btk/bal052026.dat") and "bdecks_open/2026/bal052026.dat" in urls[1]


def test_unverifiable_storms_and_null_forecasts_are_not_scored(shp):
    artifact_storms = {
        "forecast_id": "hu_fcst_20260413_1500",
        "issued_at": "2026-04-13T15:00:00Z",
        "storms": [
            {"storm_id": "WP042026", "ri_probability": 0.19},
            {"storm_id": "WP052026", "ri_probability": 0.30},
        ],
    }
    tracks = {
        "WP042026": ([_rec(ISSUE, 145.0), _rec(ISSUE + dt.timedelta(hours=24), 115.0)], "src"),
        "WP052026": ([], None),
    }
    calib: dict = {}
    res = shp.score_single_forecast(artifact_storms, calib_acc=calib,
                                    best_track_fetcher=lambda sid: tracks[sid])
    assert res["n_verified"] == 1 and res["n_unverifiable"] == 1
    assert res["brier"] == pytest.approx(0.19 ** 2)                  # only the verified storm
    assert sum(slot[0] for slot in calib.values()) == 1               # calibration pool too

    null = shp.score_single_forecast({"forecast_id": "hu_fcst_x", "issued_at": "2026-05-01T12:00:00Z",
                                      "storms": []})
    assert null["null_forecast"] is True and null["brier"] is None

    summary = shp.summarize_results([res, null])
    assert summary["mean_brier"] == pytest.approx(round(0.19 ** 2, 4))   # not diluted by the null
    assert summary["n_unverifiable_predictions"] == 1
    assert summary["unverifiable_by_basin"] == {"wp": 1}
    assert summary["total_storms_scored"] == 1


def test_nothing_verified_means_no_brier(shp):
    res = shp.score_single_forecast(
        {"forecast_id": "hu_fcst_y", "issued_at": "2026-04-13T15:00:00Z",
         "storms": [{"storm_id": "WP042026", "ri_probability": 0.0}]},
        best_track_fetcher=lambda sid: ([], None))
    summary = shp.summarize_results([res])
    assert res["brier"] is None and summary["mean_brier"] is None
    assert summary["ri_rate"] is None
