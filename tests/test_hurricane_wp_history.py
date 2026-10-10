"""West Pacific storms are scored with their best-track history, not from one warning alone.

A JTWC warning holds one cycle, so v8.2 -- trained on the best track -- had 9 of its 17 inputs imputed on
every West Pacific storm until 2026-10-05: scored that way its 2022-2024 West Pacific log loss was 0.2589
against climatology's 0.2649 (0.2057 with every input). The scorer now adds the storm's real-time best track
from UCAR RAL (``ral_track_history``), never a fix after the warning's own cycle.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane.atcf import ATCFRecord
from hazardpulse.hurricane.operational_ri import build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
V82_FEATURES = json.loads((ROOT / "results" / "models" / "hurricane_ri_v8_2.json").read_text(encoding="utf-8"))[
    "feature_names"]
T = dt.datetime(2026, 10, 5, 12)


@pytest.fixture(scope="module")
def fs():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_and_score_wp_history_t", ROOT / "scripts" / "fetch_and_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bdeck_line(when: dt.datetime, lat10: int, lon10: int, vmax: int, mslp: int, radius: int = 34) -> str:
    # the format RAL serves (one line per wind-radius threshold, as bwp262026.dat has)
    return (f"WP, 26, {when:%Y%m%d%H},   , BEST,   0, {lat10}N, {lon10}E, {vmax:3d},  {mslp:3d}, XX,  {radius}, "
            "NEQ,  250,  310,  210,  220, 1004,  410,  12,   0,")


TRACK = [  # (cycle, lat*10, lon*10, vmax, mslp) -- Choi-Wan's real fixes 10-04 06Z .. 10-05 12Z, as in
    (dt.datetime(2026, 10, 4, 6), 229, 1470, 125, 924),           # tests/fixtures/ral/bwp262026_20261005.dat
    (dt.datetime(2026, 10, 4, 12), 233, 1469, 135, 914),
    (dt.datetime(2026, 10, 4, 18), 244, 1471, 135, 912),
    (dt.datetime(2026, 10, 5, 0), 252, 1465, 130, 916),
    (dt.datetime(2026, 10, 5, 6), 263, 1463, 115, 928),
    (dt.datetime(2026, 10, 5, 12), 273, 1459, 110, 935),
]
LATER = (dt.datetime(2026, 10, 5, 18), 280, 1455, 100, 945)          # invented: a fix AFTER the warning's cycle


def _bdeck(track=TRACK, later=True) -> str:
    rows = []
    for c, la, lo, v, p in track + ([LATER] if later else []):
        rows += [_bdeck_line(c, la, lo, v, p, r) for r in (34, 50, 64)]
    return "\n".join(rows) + "\n"


WARNING = [  # what _parse_jtwc_warning_to_atcf yields: the analysis and the forecast, for ONE cycle
    ATCFRecord("WP", 26, T, 0, "JTWC", 27.3, 145.9, 110.0, None, "CHOI-WAN"),
    ATCFRecord("WP", 26, T, 24, "JTWC", 31.0, 145.0, 95.0, None, "CHOI-WAN"),
]


def _missing(case) -> list[str]:
    X, _, _, _ = build_feature_matrix([case], feature_names=V82_FEATURES)
    return [V82_FEATURES[j] for j in range(len(V82_FEATURES)) if not np.isfinite(X[0, j])]


def test_the_history_is_the_best_track_up_to_the_warnings_cycle_never_later(fs):
    recs = fs.ral_track_history("WP262026", T, fetch=lambda url: _bdeck())
    assert {r.cycle for r in recs} == {c for c, *_ in TRACK}               # 10-05 18Z is not used
    assert all(r.model == "BEST" and r.tau_hours == 0 for r in recs)
    seen = []
    fs.ral_track_history("WP262026", T, fetch=lambda url: seen.append(url) or "")
    assert seen == ["https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/2026/bwp262026.dat"]


def test_an_unreadable_history_falls_back_to_the_warning_and_says_so(fs):
    def boom(url):
        raise OSError("503")
    assert fs.with_track_history("WP262026", WARNING, fetch=boom) == (WARNING, "jtwc_warning", "WP262026")


def test_another_storms_track_is_never_merged(fs):
    """A file under the id the warning implies can be another storm's: last season's SH01 when the season year
    is wrong, or a storm 2000 km away. Neither is merged; the warning is scored alone and says so."""
    last_season = [(c.replace(year=2025), la, lo, v, p) for c, la, lo, v, p in TRACK]
    for text in (_bdeck(last_season, later=False),
                 _bdeck([(c, la + 180, lo, v, p) for c, la, lo, v, p in TRACK], later=False)):   # 18 deg north
        recs, source, sid = fs.with_track_history("WP262026", WARNING, fetch=lambda url, t=text: t)
        assert (recs, source, sid) == (WARNING, "jtwc_warning", "WP262026")
    # the same storm's fix 6 h before, 60 km/h away, is it -- the limit is not a hair trigger
    near = [(c, la, lo, v, p) for c, la, lo, v, p in TRACK[:-1]]
    assert fs.with_track_history("WP262026", WARNING, fetch=lambda url: _bdeck(near, later=False))[1] == "ral_bdeck"
    assert fs.history_mismatch(fs.ral_track_history("WP262026", T, fetch=lambda u: _bdeck(near, later=False)),
                               WARNING[0]) is None


def test_a_storm_warned_across_the_turn_of_its_season_keeps_the_year_it_formed_in(fs):
    """WP31 formed 30 December and warned on 2 January is WP312026, not WP312027."""
    jan2 = dt.datetime(2027, 1, 2, 0)
    track = [(jan2 - dt.timedelta(hours=6 * k), 100 + 2 * k, 1500 - 3 * k, 60 - 5 * k, 980 + 4 * k)
             for k in range(8, 0, -1)]
    warning = [ATCFRecord("WP", 31, jan2, 0, "JTWC", 9.8, 150.2, 65.0, None, "X")]
    files = {"bdecks_open/2026/bwp312026.dat": _bdeck(track, later=False).replace("WP, 26,", "WP, 31,")}
    def fetch(url):
        hit = [v for k, v in files.items() if url.endswith(k)]
        if not hit:
            raise OSError("404")
        return hit[0]
    case = fs.jtwc_live_case("WP312027", warning, fetch=fetch)
    assert case["storm_id"] == "WP312026" and case["season_year"] == 2026 and case["track_source"] == "ral_bdeck"


def test_a_southern_hemisphere_storm_takes_the_year_its_season_ends_in(fs):
    """RAL: bsh012026 began 2025-07-16, bsh012025 2024-10-01 -- the SH season runs July to June."""
    assert fs.atcf_season_year("SH", dt.datetime(2026, 10, 5)) == 2027
    assert fs.atcf_season_year("SH", dt.datetime(2026, 6, 30, 18)) == 2026
    assert fs.atcf_season_year("WP", dt.datetime(2026, 10, 5)) == 2026
    assert fs.atcf_season_year("IO", dt.datetime(2026, 12, 31, 18)) == 2026
    text = (ROOT / "tests" / "fixtures" / "jtwc" / "wp2626web_20261002.txt").read_text(encoding="utf-8")
    sid, _ = fs._parse_jtwc_warning_to_atcf(text, product_id="sh2627", now=dt.datetime(2026, 10, 2, 2, 40))
    assert sid == "SH262027"
    # the year is the warning's cycle's, not the clock's: read at 00:30 on 1 January, a 31 December warning
    sid, recs = fs._parse_jtwc_warning_to_atcf(text.replace("020000Z", "311800Z"), product_id="wp2626",
                                               now=dt.datetime(2027, 1, 1, 0, 30))
    assert recs[0].cycle == dt.datetime(2026, 12, 31, 18) and sid == "WP262026"


def test_with_the_history_every_v82_input_is_known_and_from_the_best_track(fs):
    merged, source, sid = fs.with_track_history("WP262026", WARNING, fetch=lambda url: _bdeck())
    assert source == "ral_bdeck" and sid == "WP262026"
    case = fs.build_live_case("WP262026", merged, full_history=True)
    assert case["analysis_model"] == "BEST" and case["issue_time"] == T.isoformat()   # best track preferred
    assert _missing(case) == []
    assert case["analysis_dv_6h"] == 110 - 115 and case["analysis_dv_24h"] == 110 - 135
    assert case["analysis_dp_12h"] == 935 - 916 and case["analysis_mslp_hpa"] == 935
    assert case["storm_age_h"] == 30.0                                       # since 10-04 06Z, its first fix


def test_from_the_warning_alone_nine_of_seventeen_inputs_are_missing(fs):
    """The pattern the 0.2589 log loss was measured on -- pinned, so the site's caution states it truly."""
    case = fs.build_live_case("WP262026", WARNING, full_history=False)
    assert len(V82_FEATURES) == 17
    assert sorted(_missing(case)) == sorted([
        "analysis_dp_12h", "analysis_dp_24h", "analysis_dp_6h", "analysis_dv_12h", "analysis_dv_24h",
        "analysis_dv_6h", "analysis_mslp_hpa", "storm_age_h", "translation_speed_kmh"])


def test_before_the_current_fix_is_published_the_warning_is_the_analysis_and_the_track_the_history(fs):
    merged, source, _ = fs.with_track_history("WP262026", WARNING, fetch=lambda url: _bdeck(TRACK[:-1], later=False))
    case = fs.build_live_case("WP262026", merged, full_history=True)
    assert source == "ral_bdeck" and case["analysis_model"] == "JTWC"
    assert case["analysis_dv_6h"] == 110 - 115 and case["analysis_dv_24h"] == 110 - 135   # warning now, track before
    assert case["storm_age_h"] == 30.0
    # a JTWC warning states no central pressure, so pressure and its three trends are all that stay unknown
    assert sorted(_missing(case)) == ["analysis_dp_12h", "analysis_dp_24h", "analysis_dp_6h", "analysis_mslp_hpa"]


def test_the_served_record_lists_the_inputs_v82_could_not_have_from_the_matrix_it_scored(fs):
    """End to end on the pinned artifact: the scorer's JTWC path, then v8.2 itself."""
    model = fs.load_serving_model()
    assert list(model["feature_names"]) == V82_FEATURES
    with_history = fs.jtwc_live_case("WP262026", WARNING, fetch=lambda url: _bdeck())
    warning_only = fs.jtwc_live_case("WP262026", WARNING, fetch=lambda url: "")      # RAL served nothing
    assert (with_history["track_source"], warning_only["track_source"]) == ("ral_bdeck", "jtwc_warning")
    assert warning_only["storm_age_h"] is None                                     # unknown, not 0
    full, alone = fs.score_live_cases(model, [with_history, warning_only])
    assert full["ri_inputs"] == {"analysis_model": "BEST", "track_source": "ral_bdeck", "n_inputs": 17,
                                 "inputs_missing": [], "inputs_before_first_fix": []}
    assert alone["ri_inputs"]["track_source"] == "jtwc_warning" and len(alone["ri_inputs"]["inputs_missing"]) == 9
    assert full["mslp_hpa"] == 935 and alone["mslp_hpa"] is None
    # the history changes the number v8.2 publishes: the inputs are not decorative
    assert full["ri_probability"] != alone["ri_probability"]


def test_a_change_over_more_hours_than_the_storm_has_lived_is_not_a_missing_input(fs):
    """Koguma (WP27) was 12 h old on 2026-10-05 12Z: its 24-hour changes do not exist, and did not in the
    training set either (None when that fix is absent), so the page must not call them missing."""
    model = fs.load_serving_model()
    young = TRACK[-3:]                                                         # 10-05 00Z, 06Z, 12Z
    case = fs.jtwc_live_case("WP262026", WARNING, fetch=lambda url: _bdeck(young, later=False))
    assert case["storm_age_h"] == 12.0
    (s,) = fs.score_live_cases(model, [case])
    assert s["ri_inputs"]["inputs_missing"] == []
    assert sorted(s["ri_inputs"]["inputs_before_first_fix"]) == ["analysis_dp_24h", "analysis_dv_24h"]
    from hazardpulse.site.pages import hurricane
    assert not hurricane._single_warning(s)
    # with no known age (the warning alone) every gap is a deficit, storm age among them
    (w,) = fs.score_live_cases(model, [fs.jtwc_live_case("WP262026", WARNING, fetch=lambda url: "")])
    assert len(w["ri_inputs"]["inputs_missing"]) == 9 and w["ri_inputs"]["inputs_before_first_fix"] == []


def test_the_page_cautions_exactly_when_the_model_lacked_inputs():
    from hazardpulse.site.pages import hurricane
    base = {"storm_id": "WP262026", "storm_name": "Choi-Wan", "basin": "WP", "lat": 26.3, "lon": 146.0,
            "vmax_kt": 110, "category": "Category 3", "ri_probability": 0.03, "ri_source": "v8.2",
            "ri_source_label": "HazardPulse v8.2", "issue_time": "2026-10-05T12:00:00"}
    full = dict(base, ri_inputs={"analysis_model": "BEST", "track_source": "ral_bdeck", "n_inputs": 17,
                                 "inputs_missing": []})
    assert not hurricane._single_warning(full) and "Caution" not in hurricane._storm_card(full)
    partial = dict(base, ri_inputs={"analysis_model": "JTWC", "track_source": "ral_bdeck", "n_inputs": 17,
                                    "inputs_missing": ["analysis_mslp_hpa"]})
    assert hurricane._single_warning(partial)
    assert "1 of the model&rsquo;s 17 inputs" in hurricane._storm_card(partial)
    legacy = dict(base, ri_inputs={"analysis_model": "JTWC"})                  # a record from before 2026-10-05
    assert hurricane._single_warning(legacy)
    # the input list decides, not the analysis's source: an a-deck analysis that lacked an input is cautioned
    adeck = dict(base, ri_inputs={"analysis_model": "CARQ", "n_inputs": 17, "inputs_missing": ["storm_age_h"]})
    assert hurricane._single_warning(adeck) and "1 of the model&rsquo;s 17 inputs" in hurricane._storm_card(adeck)
    # the measured cost of a single warning is quoted only for a record that WAS one warning
    comp = {"single_jtwc_warning": {"n_missing": 9, "n_inputs": 17, "log_loss": 0.2589,
                                    "log_loss_climatology": 0.2649, "log_loss_full_inputs": 0.2057}}
    ob = {"composition": comp, "test": {"when": "2019-2021"}}       # the years come from the evidence, not the page
    nine = ["x"] * 9
    alone = dict(base, ri_inputs={"analysis_model": "JTWC", "track_source": "jtwc_warning", "n_inputs": 17,
                                  "inputs_missing": nine})
    assert "Scored from a single JTWC warning" in hurricane._storm_card(alone, ob)
    assert "0.259" in hurricane._storm_card(alone, ob)
    assert "the 2019&ndash;2021 West Pacific cycles" in hurricane._storm_card(alone, ob)
    not_one_warning = dict(alone, ri_inputs=dict(alone["ri_inputs"], track_source="ral_bdeck"))
    assert "single JTWC warning" not in hurricane._storm_card(not_one_warning, ob)
    assert "Scored from a single JTWC warning" in hurricane._storm_card(legacy, ob)


LATE = ["analysis_dp_12h", "analysis_dp_24h", "analysis_dp_6h", "analysis_mslp_hpa"]


def test_a_late_fix_gets_its_measured_note_not_a_caution():
    """History from RAL, this cycle's fix not yet out: 4 pressure inputs filled. Measured on the 2022-2024 West
    Pacific cycles at 0.2134 against 0.2057 with every input -- not 'rough', and the page must not say so."""
    from hazardpulse.site.pages import hurricane
    comp = {"single_jtwc_warning": {"n_missing": 9, "n_inputs": 17, "log_loss": 0.2589,
                                    "log_loss_climatology": 0.2649, "log_loss_full_inputs": 0.2057},
            "late_best_track_fix": {"inputs": LATE, "n_missing": 4, "log_loss": 0.2134}}
    s = {"storm_id": "WP262026", "storm_name": "Choi-Wan", "basin": "WP", "lat": 27.3, "lon": 145.9,
         "vmax_kt": 110, "category": "Category 3", "ri_probability": 0.004, "ri_source": "v8.2",
         "ri_source_label": "HazardPulse v8.2", "issue_time": "2026-10-05T12:00:00",
         "ri_inputs": {"analysis_model": "JTWC", "track_source": "ral_bdeck", "n_inputs": 17,
                       "inputs_missing": list(reversed(LATE))}}                       # order is not the pattern
    ob = {"composition": comp, "test": {"when": "2022-2024"}}
    card = hurricane._storm_card(s, ob)
    assert "Caution" not in card and "rough" not in card
    assert "not yet published" in card and "central pressure and its changes" in card
    assert "0.213" in card and "0.206" in card and "0.265" in card
    # one input more or less is not the measured pattern: the plain caution again
    for other in (LATE[:-1], LATE + ["storm_age_h"]):
        card = hurricane._storm_card(dict(s, ri_inputs=dict(s["ri_inputs"], inputs_missing=other)), ob)
        assert "Caution" in card and "0.213" not in card
    # without the measurement bound, no number is quoted
    assert "Caution" in hurricane._storm_card(
        s, {"composition": {"single_jtwc_warning": comp["single_jtwc_warning"]}, "test": {"when": "2022-2024"}})


def test_the_late_fix_measurement_is_the_live_codes_pattern_and_reaches_the_site():
    """The composition file's late-fix inputs are what the live code leaves missing (derived, not typed), and
    the evidence the site reads carries the same list and number."""
    from hazardpulse.verification import served_evidence as se
    ob = se.hurricane_evidence()["other_basins"]
    # every composition file -- v8.2's, and the served replacement's (amendment 15: v8.3) -- carries the live
    # code's pattern; the site reads the served model's
    served_rel = (f"results/calibration/{ob['model']}_test_composition.json" if ob.get("subject") == "model"
                  else se.HURRICANE_V82_COMPOSITION)
    assert served_rel != se.HURRICANE_V82_COMPOSITION and ob["model"] == "hurricane_ri_v8_3"
    for rel in (se.HURRICANE_V82_COMPOSITION, served_rel):
        c = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert c["late_best_track_fix"]["inputs_missing_live"] == LATE, rel
        assert 0 < c["late_best_track_fix"]["log_loss"] < c["single_jtwc_warning"]["log_loss_climatology"], rel
    comp = json.loads((ROOT / served_rel).read_text(encoding="utf-8"))
    lf = comp["late_best_track_fix"]
    ev = ob["composition"]["late_best_track_fix"]
    assert ev == {"inputs": LATE, "n_missing": 4, "log_loss": lf["log_loss"]}
    from hazardpulse.site.pages import hurricane
    text = hurricane.v82_test_text(se.hurricane_evidence()["other_basins"])
    assert f"{lf['log_loss']:.3f}" in text and "working best track" in text
