"""The tornado live record scores the model as it was trained and tested (tornado audit item 1, 2026-10-05).

(a) the scorer read the ROUNDED published number: 1,404 of 1,602 v3 storm forecasts were exactly 0.0;
(b) reports before the run's issue time were dropped, while training labels count from the storm's VALID time
    (the issue time is a median 17.5 min later; 33.8% of 2025's positives had their first report within 15 min);
(c) the Brier-skill reference pooled the OLD model's base rate, so /data/verification/to.json published
    "BSS 0.9986" for 794 storm forecasts with zero tornadoes;
(d) the same storm, re-forecast by consecutive runs from the same hour of data, was counted once per run.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V3 = "tornado_v3-05a06c843c87"
OLD = "tornado_storm_v1_0"


@pytest.fixture(scope="module")
def ver():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("stp_live_record", ROOT / "scripts" / "score_tornado_prospective.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _square(lat, lon, half=0.05):
    ring = [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half]]
    return {"type": "Polygon", "coordinates": [ring]}


def _storm(sid, valid, published, p60, lat=35.0, lon=-97.0):
    return {"storm_id": sid, "lat": lat, "lon": lon, "motion_east": 0.0, "motion_south": 0.0,
            "geometry": _square(lat, lon), "valid_time": valid, "tornado_probability": published,
            "model_version": V3, "v3": {"probability_60min": p60, "model_version": V3}}


def _still_tracks(ver, storms_by_time):
    """An archive in which each storm stays where it is (so the label is about time, not motion)."""
    def fetch(day):
        steps = []
        for valid, storms in storms_by_time:
            if valid.startswith(day):
                steps.append({"valid_time": valid, "storms": [
                    {"id": s["storm_id"], "lat": s["lat"], "lon": s["lon"], "motion_east": 0.0, "motion_south": 0.0,
                     "geometry": s["geometry"]} for s in storms]})
        return steps or [{"valid_time": f"{day}_000039 UTC", "storms": []}]
    return ver.TrackSource(fetch=fetch)


def test_a_the_record_scores_the_models_full_precision_number_not_the_rounded_display(ver):
    storms = [_storm("1", "20250506_180039 UTC", 0.0, 3.0e-5), _storm("2", "20250506_180039 UTC", 0.0, 4.0e-5, lat=36.0),
              _storm("3", "20250506_180039 UTC", 0.0004, 4.2e-4, lat=37.0)]
    art = {"forecast_id": "to_fcst_a", "issued_at": "2025-05-06T18:20:00Z", "storms": storms}
    lab = ver.label_storms(art, [], tracks=_still_tracks(ver, [("20250506_180039 UTC", storms)]))
    assert lab["y_score"].tolist() == [3.0e-5, 4.0e-5, 4.2e-4]           # never the 0.0 the page displayed
    assert lab["y_published"].tolist() == [0.0, 0.0, 0.0004]
    assert lab["y_raw"].tolist() == [3.0e-5, 4.0e-5, 4.2e-4]              # what a calibrator would be fitted on


def test_b_a_report_after_the_valid_time_but_before_the_issue_time_counts(ver):
    s = _storm("77", "20250506_180039 UTC", 0.01, 0.0123)
    art = {"forecast_id": "to_fcst_b", "issued_at": "2025-05-06T18:20:00Z", "storms": [s]}
    tracks = _still_tracks(ver, [("20250506_180039 UTC", [s])])
    early = [{"lat": 35.0, "lon": -97.0, "time": "2025-05-06T18:10:00Z", "mag": 1}]       # 9.4 min after valid
    lab = ver.label_storms(art, early, tracks=tracks)
    assert lab["y_true"].tolist() == [1.0]                 # the training label: counted from the valid time
    assert lab["y_true_after_issue"].tolist() == [0.0]     # the old, run-time-truncated label, kept separately
    late = [{"lat": 35.0, "lon": -97.0, "time": "2025-05-06T18:40:00Z", "mag": 1}]
    lab2 = ver.label_storms(art, late, tracks=tracks)
    assert lab2["y_true"].tolist() == [1.0] and lab2["y_true_after_issue"].tolist() == [1.0]
    before = [{"lat": 35.0, "lon": -97.0, "time": "2025-05-06T17:50:00Z", "mag": 1}]      # before the data
    assert ver.label_storms(art, before, tracks=tracks)["y_true"].tolist() == [0.0]


def _lab(issued, ends_h, version, y, p):
    issued = dt.datetime.fromisoformat(issued)
    return {"issued_at": issued, "window_end": issued + dt.timedelta(hours=ends_h), "y_true": np.array(y, float),
            "y_score": np.array(p, float), "calibrated": np.zeros(len(y), bool),
            "model_versions": np.array([version] * len(y), dtype=object)}


def _row(lab, tier="tier1_v3"):
    return {"auc": float("nan"), "brier": float("nan"), "brier_skill_score": float("nan"), "n_storms": lab["y_true"].size,
            "n_matched_storms": int(lab["y_true"].sum()), "n_reports_in_window": 0, "scoring_tier": tier,
            "window_end": lab["window_end"].strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_c_with_no_event_brier_skill_is_null_never_a_number(ver):
    # the old model's record: matured first, 2 events in 4 storm forecasts
    old = _lab("2026-10-01T00:00:00", 24, OLD, [1, 1, 0, 0], [0.5, 0.4, 0.1, 0.1])
    # the new model's record: zero tornadoes, tiny probabilities (the 794-forecast record of 2026-10-04)
    new = [_lab(f"2026-10-0{d}T00:00:00", 24, V3, [0, 0, 0], [1e-5, 2e-5, 3e-5]) for d in (3, 4)]
    scored = [(None, lab, _row(lab)) for lab in [old] + new]
    s = ver.summarize(scored, dt.datetime(2026, 10, 6))
    v3 = s["pooled_by_model_version"][V3]
    assert v3["n_positive"] == 0
    assert v3["bss_vs_causal_climatology"] is None and v3["bss_vs_sample_climatology"] is None
    assert v3["skill_undefined"] == "no event observed"


def test_c_the_causal_reference_is_the_versions_own_base_rate(ver):
    old = _lab("2026-10-01T00:00:00", 24, OLD, [1, 1, 1, 0], [0.5, 0.4, 0.3, 0.1])       # base rate 0.75
    first = _lab("2026-10-01T00:00:00", 24, V3, [1, 0, 0, 0], [0.2, 0.01, 0.01, 0.01])   # base rate 0.25
    later = _lab("2026-10-03T00:00:00", 24, V3, [1, 0], [0.3, 0.05])                     # both matured by then
    scored = [(None, lab, _row(lab)) for lab in (old, first, later)]
    v3 = ver.summarize(scored, dt.datetime(2026, 10, 6))["pooled_by_model_version"][V3]
    # only `later` has a causal reference, and it is v3's OWN 1/4 -- not the pooled 4/8 of both versions
    y, p, ref = np.array([1.0, 0.0]), np.array([0.3, 0.05]), 0.25
    expected = 1.0 - np.mean((p - y) ** 2) / np.mean((ref - y) ** 2)
    assert v3["n_with_causal_reference"] == 2
    assert v3["brier_causal_climatology"] == pytest.approx(np.mean((ref - y) ** 2), abs=1e-8)
    assert v3["bss_vs_causal_climatology"] == pytest.approx(expected, abs=1e-4)


def test_d_one_forecast_per_storm_per_valid_hour(ver):
    s1 = _storm("77", "20250506_180039 UTC", 0.01, 0.0123)
    s2 = _storm("77", "20250506_183039 UTC", 0.02, 0.0234)            # the same storm, same hour, next run
    s3 = _storm("77", "20250506_190039 UTC", 0.03, 0.0345)            # the next hour: a new forecast
    arts = [{"forecast_id": f"to_fcst_d{i}", "issued_at": iss, "storms": [s]}
            for i, (iss, s) in enumerate([("2025-05-06T18:05:00Z", s1), ("2025-05-06T18:35:00Z", s2),
                                          ("2025-05-06T19:05:00Z", s3)])]
    tracks = _still_tracks(ver, [(s["valid_time"], [s]) for s in (s1, s2, s3)])
    seen: set = set()
    scored = []
    for art in arts:
        lab = ver.label_storms(art, [], tracks=tracks)
        ver.mark_first_occurrences(lab, seen)
        scored.append((art, lab, ver.score_single_forecast(art, [], labels=lab)))
    assert [lab["keep"].tolist() for _, lab, _ in scored] == [[True], [False], [True]]
    assert [r["n_repeat_storm_forecasts"] for _, _, r in scored] == [0, 1, 0]
    s = ver.summarize(scored, dt.datetime(2025, 5, 8))
    rec = s["pooled_by_model_version"][V3]
    assert rec["n_storm_forecasts"] == 2 and rec["n_repeat_storm_forecasts_dropped"] == 1
    assert s["repeat_storm_forecasts"]["n_dropped"] == 1
    # the calibration data counts each storm-hour once too
    acc = {ver._CALIB_VERSION_KEY: V3}
    for art, lab, _ in scored:
        ver.score_single_forecast(art, [], calib_acc=acc, labels=lab)
    assert sum(v[0] for k, v in acc.items() if k != ver._CALIB_VERSION_KEY) == 2
