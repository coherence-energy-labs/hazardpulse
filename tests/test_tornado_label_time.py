"""The tornado label compares absolute UTC instants -- regression tests.

Until 2026-10-01 the definitive model's label read SPC report times (CST,
``tz=3``) as UTC hours, keyed reports by their CST date while storms were
keyed by UTC date, and wrapped negative differences by +24 h. Each test
below pins one consequence with a concrete witness that the OLD code got
wrong, so reintroducing any of the three errors fails here.

Separately, the training-side parser for ProbSevere stamps keyed on the
letter ``T`` and matched the ``T`` in ``"UTC"``: every native stamp
(``"20240427_000042 UTC"``, stored by the fetcher since 2026-03-22) parsed to
an unknown hour, so every sample from a fresh cache was silently excluded.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from hazardpulse.tornado import definitive_model as dm

UTC = dt.timezone.utc


def _utc(y, mo, d, h, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=UTC)


def _write_spc(tmp_path, rows):
    header = "om,yr,mo,dy,date,time,tz,st,stf,stn,mag,inj,fat,loss,closs,slat,slon,elat,elon,len,wid"
    lines = [header]
    for i, (yr, mo, dy, hhmm, tz, lat, lon) in enumerate(rows, 1):
        lines.append(
            f"{i},{yr},{mo},{dy},{yr}-{mo:02d}-{dy:02d},{hhmm}:00,{tz},OK,40,0,1,0,0,0,0,"
            f"{lat},{lon},{lat},{lon},1.0,50"
        )
    p = tmp_path / "spc.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# SPC clock
# --------------------------------------------------------------------------

def test_spc_cst_is_converted_to_utc():
    assert dm.parse_spc_time_utc(2024, 4, 27, "23:30:00", 3) == _utc(2024, 4, 28, 5, 30)
    assert dm.parse_spc_time_utc(2024, 4, 27, "17:05:00", 3) == _utc(2024, 4, 27, 23, 5)


def test_spc_gmt_rows_are_not_shifted():
    assert dm.parse_spc_time_utc(1999, 5, 3, "23:30:00", 9) == _utc(1999, 5, 3, 23, 30)


def test_spc_unknown_zone_or_time_is_never_guessed():
    assert dm.parse_spc_time_utc(2024, 4, 27, "23:30:00", 0) is None
    assert dm.parse_spc_time_utc(2024, 4, 27, "", 3) is None
    assert dm.parse_spc_time_utc(2024, 4, 27, "garbage", 3) is None


def test_reports_are_keyed_by_utc_date(tmp_path):
    # 23:30 CST on Apr 27 is 05:30 UTC on Apr 28: it must be filed under Apr 28.
    p = _write_spc(tmp_path, [(2024, 4, 27, "23:30", 3, 35.0, -97.0)])
    reports = dm.load_spc_tornado_reports(p)
    assert list(reports) == ["20240428"]
    rec = reports["20240428"][0]
    assert rec["hour"] == 5.5
    assert rec["time_utc"] == _utc(2024, 4, 28, 5, 30).timestamp()
    assert rec["local_date"] == "20240427"


# --------------------------------------------------------------------------
# The label, with the witnesses the old code got wrong
# --------------------------------------------------------------------------

def _labels(tmp_path, storm_time, tor_rows, lat=35.0, lon=-97.0):
    reports = dm.load_spc_tornado_reports(_write_spc(tmp_path, tor_rows))
    window = dm.reports_in_label_window(reports, storm_time.strftime("%Y%m%d"))
    return dm.compute_label(lat, lon, storm_time, window)


def test_storm_30_min_before_a_tornado_is_positive(tmp_path):
    # Tornado 23:30 CST Apr 27 = 05:30 UTC Apr 28; storm seen 05:00 UTC Apr 28.
    # OLD: report filed under Apr 27 at "hour 23.5" -> no match -> label 0.
    assert _labels(tmp_path, _utc(2024, 4, 28, 5, 0), [(2024, 4, 27, "23:30", 3, 35.0, -97.0)]) == 1


def test_storm_six_and_a_half_hours_before_a_tornado_is_negative(tmp_path):
    # Tornado 17:00 CST = 23:00 UTC; storm at 16:30 UTC is 6.5 h early.
    # OLD: compared 17.0 with 16.5 -> "30 min ahead" -> label 1.
    assert _labels(tmp_path, _utc(2024, 5, 6, 16, 30), [(2024, 5, 6, "17:00", 3, 35.0, -97.0)]) == 0


def test_a_tornado_in_the_past_is_never_a_label(tmp_path):
    # Tornado 18:15 CST May 5 = 00:15 UTC May 6; storm at 23:30 UTC May 6 is
    # ~23 h LATER. OLD: +24 h wrap turned -23.25 h into +0.75 h -> label 1.
    assert _labels(tmp_path, _utc(2024, 5, 6, 23, 30), [(2024, 5, 5, "18:15", 3, 35.0, -97.0)]) == 0


def test_window_crosses_00_utc(tmp_path):
    # Storm 23:30 UTC May 6; tornado 18:10 CST May 6 = 00:10 UTC May 7.
    assert _labels(tmp_path, _utc(2024, 5, 6, 23, 30), [(2024, 5, 6, "18:10", 3, 35.0, -97.0)]) == 1


def test_distance_still_gates_the_label(tmp_path):
    # Same timing as the 30-minute witness, 100 km away: negative.
    assert _labels(
        tmp_path, _utc(2024, 4, 28, 5, 0), [(2024, 4, 27, "23:30", 3, 35.9, -97.0)]
    ) == 0


def test_unknown_storm_time_is_excluded(tmp_path):
    assert dm.compute_label(35.0, -97.0, None, []) == -1


# --------------------------------------------------------------------------
# ProbSevere stamps
# --------------------------------------------------------------------------

def test_native_probsevere_stamp_parses():
    assert dm.parse_probsevere_valid_time("20240427_000042 UTC") == dt.datetime(
        2024, 4, 27, 0, 0, 42, tzinfo=UTC
    )


def test_iso_probsevere_stamp_parses():
    assert dm.parse_probsevere_valid_time("2024-04-27T18:30:00Z") == _utc(2024, 4, 27, 18, 30)


def test_unparseable_stamp_is_unknown():
    for bad in ("", "UTC", "20240427", "2024-04-27", "C"):
        assert dm.parse_probsevere_valid_time(bad) is None, bad


def _storm(sid, lat=35.0, lon=-97.0):
    return {"id": sid, "lat": lat, "lon": lon, "maxllaz": 0.004, "srh01": 150.0}


def test_a_native_stamped_day_is_not_silently_excluded():
    """The T-in-UTC bug: with native stamps the old builder excluded EVERY sample."""
    steps = [
        {"valid_time": "20240427_200040 UTC", "storms": [_storm(1), _storm(2, lat=40.0)]},
        {"valid_time": "20240427_203038 UTC", "storms": [_storm(1), _storm(2, lat=40.0)]},
    ]
    tor = [{"slat": 35.0, "slon": -97.0, "time_utc": _utc(2024, 4, 27, 20, 45).timestamp()}]
    analyses = {18: _analysis()}
    X, y, n_excluded, n_no_an = dm.build_samples_for_date(
        "20240427", steps, tor, analyses, include_coherence=True
    )
    assert n_excluded == 0 and n_no_an == 0
    assert X.shape == (4, dm.N_FEAT_FULL)
    # storm 1 at 20:00 and 20:30 is within 60 min / 40 km of the 20:45 tornado
    assert y.tolist() == [1.0, 0.0, 1.0, 0.0]


def _analysis():
    from test_tornado_torsion import _synthetic_atmosphere

    from hazardpulse.tornado import coherence_engine as ce

    g = _synthetic_atmosphere()
    return g, ce.compute_derived_hrrr(g), ce.compute_coherence_fields(g, month=4)


def test_storms_read_the_latest_analysis_at_or_before_them():
    hours = dm.HRRR_ANALYSIS_HOURS
    assert dm.select_analysis_hour(_utc(2024, 4, 27, 17, 59), hours) == 15
    assert dm.select_analysis_hour(_utc(2024, 4, 27, 18, 0), hours) == 18
    assert dm.select_analysis_hour(_utc(2024, 4, 27, 2, 30), hours) == 0
    # never a later analysis, never one older than the limit
    assert dm.select_analysis_hour(_utc(2024, 4, 27, 17, 59), (18, 21)) is None
    assert dm.select_analysis_hour(_utc(2024, 4, 27, 17, 59), (12,)) is None
    assert dm.select_analysis_hour(None, hours) is None


def test_a_storm_before_the_only_analysis_is_excluded_not_fed_the_future():
    """The 18Z leak: a 16:00 storm used to read the 18:00 atmosphere."""
    steps = [{"valid_time": "20240427_160000 UTC", "storms": [_storm(1)]}]
    X, y, n_exc, n_no_an = dm.build_samples_for_date(
        "20240427", steps, [], {18: _analysis()}, include_coherence=True
    )
    assert X.shape[0] == 0 and n_no_an == 1
    # the explicit ablation reproduces the old leak, and only it
    X_leak, *_ = dm.build_samples_for_date(
        "20240427", steps, [], {18: _analysis()}, include_coherence=True,
        analysis_policy="fixed_18z",
    )
    assert X_leak.shape[0] == 1


def test_step_cadence_comes_from_the_stamps():
    steps30 = [{"valid_time": f"20240427_{h:02d}{m:02d}00 UTC"} for h in (20, 21) for m in (0, 30)]
    steps15 = [{"valid_time": f"20240427_20{m:02d}00 UTC"} for m in (0, 15, 30, 45)]
    assert dm.probsevere_step_minutes(steps30) == 30.0
    assert dm.probsevere_step_minutes(steps15) == 15.0
    assert dm.probsevere_step_minutes([]) == dm.PROBSEVERE_STEP_MIN_DEFAULT


def test_storm_age_uses_the_real_cadence():
    hist = [_storm(1)] * 3
    e30 = dm.extract_block_e(_storm(1), hist, 30.0)
    e15 = dm.extract_block_e(_storm(1), hist, 15.0)
    assert e30[0] == 60.0 and e15[0] == 30.0


def test_history_lookback_is_bounded_and_index_agrees():
    steps = [{"storms": [_storm(7)]} for _ in range(20)]
    idx = dm.index_storms_by_id(steps)
    h_scan = dm.build_storm_history(steps, 7, 19)
    h_idx = dm.build_storm_history(steps, 7, 19, id_index=idx)
    assert len(h_scan) == len(h_idx) == dm.STORM_HISTORY_LOOKBACK + 1
    assert all(a is b for a, b in zip(h_scan, h_idx))


def test_temporal_integrity_check_can_fail():
    import pytest

    dm.assert_temporal_integrity(["20220101"], ["20230101"], ["20240101"])
    with pytest.raises(AssertionError):
        dm.assert_temporal_integrity(["20230105"], ["20230101"], ["20240101"])
