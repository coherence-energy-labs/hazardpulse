"""The live tornado run reads the NEWEST ProbSevere file, with its track history on the trained cadence
(tornado audit items 2 and 3, 2026-10-05).

Item 2 (MEASURED): the fetcher took the FIRST file of each 30-minute slot, so the live input was a median 17.5
minutes old at issue (1,757 records of 2026) (the 00:25Z run read 00:00:38 data while 12 newer files existed). The live selection is
now the newest file plus, for every earlier slot, the file at the same phase -- and when the newest file IS a
slot start it is exactly the training selection, so the history features mean what they meant in training.

Item 3 (MEASURED): NOAA's 2025-08-06 format change removed PS, VIL_DENSITY and MAXRC_ICECF; the parser reads
the first two as 0.0 for every storm. The values are unchanged (a model change needs its own evaluation); the
run now records which inputs were absent and what the model received instead.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.data import probsevere as ps
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import input_guard
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado import v3_serving as vs

FIXTURE = Path(__file__).parent / "fixtures" / "probsevere" / "listings.json"
LISTINGS = json.loads(FIXTURE.read_text(encoding="utf-8"))["days"]


def _keys(day: str) -> list[str]:
    return [f"ProbSevere/{day}/MRMS_PROBSEVERE_{day}_{hms}.json" for hms in LISTINGS[day]]


def _upto(keys: list[str], last: str) -> list[str]:
    return [k for k in keys if k <= last]


def _at(day: str, hhmm: str) -> str:
    """The recorded file of ``day`` whose time starts with ``hhmm``."""
    (k,) = [k for k in _keys(day) if k.rsplit("_", 1)[-1].startswith(hhmm)]
    return k


def test_on_a_recorded_day_every_slot_start_run_selects_exactly_the_training_files():
    keys = _keys("20261004")                    # a complete day: 720 files, every 2 minutes
    starts = ps.slot_start_keys(keys)
    assert len(keys) == 720 and len(starts) == 48
    for s in starts:
        upto = _upto(keys, s)
        assert ps.live_keys(upto) == ps.slot_start_keys(upto), s


def test_on_a_day_with_missing_files_the_one_difference_keeps_the_newest_files_phase():
    keys = _keys("20260519")                    # 33 files missing
    differ = [s for s in ps.slot_start_keys(keys) if ps.live_keys(_upto(keys, s)) != ps.slot_start_keys(_upto(keys, s))]
    # the 12:00 file is missing, so the slot's first file is 12:02:37; the live history keeps that phase
    assert [k.rsplit("_", 1)[-1] for k in differ] == ["120237.json"]
    picked = ps.live_keys(_upto(keys, differ[0]))
    gaps = np.diff([ps.key_time(k).timestamp() for k in picked]) / 60.0
    assert np.all(np.abs(gaps - 30.0) < 1.0)    # a 30-minute cadence, to the second jitter of the files


def test_a_mid_slot_run_reads_the_newest_file_and_keeps_thirty_minute_steps():
    keys = _keys("20261004")
    last = _at("20261004", "0024")                    # a run at 00:25Z: its newest file is 00:24:42
    picked = ps.live_keys(_upto(keys, last))
    assert picked == [last]                           # nothing earlier that day (the old fetcher: 00:00:39)
    assert ps.slot_start_keys(_upto(keys, last))[-1].endswith("000039.json")
    last = _at("20261004", "2024")
    picked = ps.live_keys(_upto(keys, last))
    assert picked[-1] == last and len(picked) == 41
    gaps = np.diff([ps.key_time(k).timestamp() for k in picked]) / 60.0
    assert np.all(np.abs(gaps - 30.0) < 1.0)
    old = ps.slot_start_keys(_upto(keys, last))
    assert ps.key_time(old[-1]).strftime("%H%M") == "2000"                     # what the old fetcher served
    assert (ps.key_time(picked[-1]) - ps.key_time(old[-1])) > dt.timedelta(minutes=20)   # 24 min fresher


# ------------------------------------------------------------------------------------------------
# end to end through the fetchers: the history FEATURES are the same when the newest file is a slot start
# ------------------------------------------------------------------------------------------------

def _doc(key: str, *, props_v2025: bool = False) -> dict:
    """A ProbSevere document whose storms change with time, so a different file gives different features."""
    t = ps.key_time(key)
    m = (t.hour * 60 + t.minute) / 1440.0
    feats = []
    for sid, (lat, lon) in (("11", (35.0, -97.0)), ("22", (33.0, -90.0))):
        props = {"ID": sid, "MAXLLAZ": f"{0.002 + 0.01 * m:.5f}", "P98LLAZ": f"{0.001 + 0.008 * m:.5f}",
                 "MESH": f"{0.5 + m:.3f}", "FLASH_RATE": f"{int(40 * m)}", "SIZE": f"{100 + int(300 * m)}",
                 "MOTION_EAST": "10", "MOTION_SOUTH": "-3", "MUCAPE": "2000"}
        if props_v2025:
            props.update({"PS": "40", "VIL_DENSITY": "2.1", "MAXRC_ICECF": "2241Z 0.01/min (weak)"})
        ring = [[lon, lat], [lon + 0.1, lat], [lon + 0.1, lat + 0.1], [lon, lat]]
        feats.append({"properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "models": {"probtor": {"PROB": f"{int(60 * m)}"}, "probsevere": {"PROB": "50"}}})
    return {"validTime": t.strftime("%Y%m%d_%H%M%S UTC"), "features": feats}


@pytest.fixture
def feed(monkeypatch):
    state = {"keys": []}
    monkeypatch.setattr(ps, "_list_s3_files", lambda d: (list(state["keys"]), True))
    monkeypatch.setattr(ps, "fetch_bytes", lambda url, **k: json.dumps(_doc(url.rsplit("/", 1)[-1])).encode())
    return state


def _history_features(steps: list[dict]) -> dict[str, np.ndarray]:
    latest = len(steps) - 1
    idx = dm.index_storms_by_id(steps)
    step_min = dm.probsevere_step_minutes(steps)
    lo, hi = sf.BLOCKS["E"]
    out = {}
    for s in steps[-1]["storms"]:
        fv = sf.feature_vector(s, vs.storm_history(steps, s["id"], latest, id_index=idx), step_min)
        out[s["id"]] = np.concatenate([fv[slice(*sf.BLOCKS["P"])], fv[lo:hi]])
    return out


def test_history_features_equal_the_training_way_when_the_newest_file_is_a_slot_start(feed, tmp_path):
    keys = _keys("20261004")
    s = _at("20261004", "2030")                                                   # a slot start
    assert s in ps.slot_start_keys(keys)
    feed["keys"] = _upto(keys, s)
    old = ps.fetch_probsevere_day("20261004", cache_dir=tmp_path, refresh=True)
    new, census = ps.fetch_probsevere_live("20261004")
    assert [st["valid_time"] for st in new] == [st["valid_time"] for st in old]
    a, b = _history_features(old), _history_features(new)
    assert a.keys() == b.keys() and len(a) == 2
    for sid in a:
        np.testing.assert_array_equal(a[sid], b[sid])
    assert census["n_objects"] == 2
    # and the check can fail: a run 24 minutes into the slot reads newer data, so its features differ
    feed["keys"] = _upto(keys, _at("20261004", "2054"))
    newer, _ = ps.fetch_probsevere_live("20261004")
    assert newer[-1]["valid_time"] == "20261004_205439 UTC"
    old2 = ps.fetch_probsevere_day("20261004", cache_dir=tmp_path / "x", refresh=True)
    assert old2[-1]["valid_time"] == "20261004_203038 UTC"                        # the 24-minute-old file
    assert any(not np.array_equal(_history_features(old2)[k], _history_features(newer)[k]) for k in a)


def test_the_live_selection_is_never_written_to_the_day_cache(feed, tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "CACHE_ROOT", tmp_path)
    feed["keys"] = _upto(_keys("20261004"), _at("20261004", "2054"))
    ps.fetch_probsevere_live("20261004")
    assert list(tmp_path.iterdir()) == []      # the day cache holds the training selection the labels read


# ------------------------------------------------------------------------------------------------
# item 3: the format guard records the gap; the parser's values are unchanged
# ------------------------------------------------------------------------------------------------

SERVED = ("p_ps", "p_ps_tor", "p_vil_density", "p_maxllaz", "p_maxrc_icecf", "p_maxrc_emiss", "p_avg_beam_hgt")


def test_the_guard_names_the_inputs_the_2025_format_no_longer_carries_and_the_values_are_unchanged():
    new_format = _doc("ProbSevere/20261004/MRMS_PROBSEVERE_20261004_235839.json")
    for f in new_format["features"]:          # as the 2026-10-04 23:58:39 file carries them
        f["properties"].update({"MAXRC_EMISS": "0", "AVG_BEAM_HGT": "1.7"})
    gaps = input_guard.input_gaps(ps.property_census(new_format), SERVED, splits={"p_ps": 196, "p_vil_density": 475})
    absent = {g["input"]: g for g in gaps["absent"]}
    assert set(absent) == {"p_ps", "p_vil_density", "p_maxrc_icecf"}
    assert absent["p_ps"]["fed"] == "zero" and absent["p_ps"]["model_splits"] == 196
    assert absent["p_maxrc_icecf"]["fed"] == "missing"
    changed = {g["input"]: g for g in gaps["changed_format"]}
    assert set(changed) == {"p_maxrc_emiss", "p_avg_beam_hgt"}      # numbers where strings were documented
    assert {g["input"] for g in gaps["partial"]} == set()
    assert input_guard.has_gaps(gaps)
    # what the model receives is exactly what it received before this guard existed
    storm = ps._parse_storms(new_format)[0]
    assert storm["ps"] == 0.0 and storm["vil_density"] == 0.0 and "maxrc_icecf" not in storm
    fv = sf.feature_vector(storm, [storm], 30.0)
    names = list(sf.FEATURE_NAMES)
    assert fv[names.index("p_ps")] == 0.0 and fv[names.index("p_vil_density")] == 0.0
    assert np.isnan(fv[names.index("p_maxrc_icecf")])


def test_the_old_format_has_no_gap_and_an_unknown_census_says_so():
    old_format = _doc("ProbSevere/20250506/MRMS_PROBSEVERE_20250506_180039.json", props_v2025=True)
    for f in old_format["features"]:
        f["properties"].update({"MAXRC_EMISS": "2251Z 1.5%/min (weak)", "AVG_BEAM_HGT": "4.09 kft / 1.25 km"})
    gaps = input_guard.input_gaps(ps.property_census(old_format), SERVED)
    assert gaps["absent"] == [] and gaps["changed_format"] == [] and not input_guard.has_gaps(gaps)
    unknown = input_guard.input_gaps(None, SERVED)
    assert unknown["census"] == "unavailable" and not input_guard.has_gaps(unknown)
