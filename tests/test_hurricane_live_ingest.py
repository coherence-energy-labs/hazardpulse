"""Live hurricane ingest: which storms are active, who they are, and what features they carry.

Each test pins a defect measured on the live feeds on 2026-10-02, which the scorer would
have published the moment its timeout was fixed:

* NHC's aid_public lists every a-deck of the season (53 files; 3 analysed in the last 12 h,
  AL01 last analysed 106 days earlier) and all 53 were scored as "active";
* the JTWC parser took a storm's identity from anywhere in the text, so Hurricane Rachel's
  warning (100 kt, 19.4N 109.4W) was published as "Nolo"; it dropped Tropical Storm 26W
  Choi-wan (a hyphenated name); it filed EP storms under WP and duplicated NHC's;
* the live case builder never set abs_lat / mpi_deficit / intensity_frac_mpi, so the model
  saw training medians instead.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "jtwc"
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.hurricane import operational_ri as ori  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402

NOW = dt.datetime(2026, 10, 2, 2, 40)


@pytest.fixture(scope="module")
def fas():
    spec = importlib.util.spec_from_file_location("fas_ingest_test", REPO / "scripts" / "fetch_and_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(pid: str) -> str:
    return (FIXTURES / f"{pid}web_20261002.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JTWC identity
# ---------------------------------------------------------------------------

def test_jtwc_identity_comes_from_the_subj_header_not_the_body(fas):
    # Rachel's warning body says "REFER TO TROPICAL STORM 15E (NOLO) WARNINGS".
    assert "15E (NOLO)" in _fixture("ep1826")
    sid, recs = fas._parse_jtwc_warning_to_atcf(_fixture("ep1826"), product_id="ep1826", now=NOW)
    assert sid == "EP182026"
    analysis = recs[0]
    assert (analysis.storm_name, analysis.lat, analysis.lon, analysis.vmax_kt) == ("Rachel", 19.4, -109.4, 100.0)
    assert all(r.storm_name == "Rachel" for r in recs)


def test_jtwc_parses_hyphenated_names_and_files_them_under_the_right_basin(fas):
    sid, recs = fas._parse_jtwc_warning_to_atcf(_fixture("wp2626"), product_id="wp2626", now=NOW)
    assert sid == "WP262026"
    assert recs[0].storm_name == "Choi-Wan" and recs[0].tau_hours == 0
    assert (recs[0].lat, recs[0].lon, recs[0].vmax_kt) == (17.0, 144.7, 55.0)
    assert len(recs) > 1 and all(r.tau_hours > 0 for r in recs[1:])
    # Without a product id the basin comes from the number's suffix -- E is EP, not WP.
    sid_e, _ = fas._parse_jtwc_warning_to_atcf(_fixture("ep1526"), now=NOW)
    assert sid_e == "EP152026"


def test_jtwc_final_warning_on_remnants_is_not_a_storm(fas):
    assert fas._parse_jtwc_warning_to_atcf(_fixture("ep1926"), product_id="ep1926", now=NOW) == ("", [])


def test_jtwc_discovery_skips_nhc_basins_and_area_advisories(fas, monkeypatch):
    rss = "<rss><item>" + "".join(
        f'<link>https://www.metoc.navy.mil/jtwc/products/{pid}web.txt</link>'
        for pid in ("wp2626", "ep1526", "ep1826", "ep1926", "abpw", "abio")
    ) + "</item></rss>"
    requested: list[str] = []

    def fake_fetch_text(url, **_):
        if url.endswith(".rss"):
            return rss
        pid = url.rsplit("/", 1)[1].replace("web.txt", "")
        requested.append(pid)
        return _fixture(pid)

    monkeypatch.setattr(fas, "fetch_text", fake_fetch_text)
    storms = fas._discover_jtwc_storms()
    assert list(storms) == [f"WP26{dt.datetime.now(dt.timezone.utc).year}"]
    assert requested == ["wp2626"], "EP/CP warnings duplicate NHC's a-decks; abpw/abio are not warnings"


def test_jtwc_day_of_month_resolves_across_a_month_boundary(fas):
    assert fas._jtwc_day_to_datetime("302100", dt.datetime(2026, 10, 1, 3, 0)) == dt.datetime(2026, 9, 30, 21, 0)
    assert fas._jtwc_day_to_datetime("010000", dt.datetime(2026, 10, 1, 3, 0)) == dt.datetime(2026, 10, 1, 0, 0)
    assert fas._jtwc_day_to_datetime("311800", dt.datetime(2026, 1, 1, 0, 30)) == dt.datetime(2025, 12, 31, 18, 0)


# ---------------------------------------------------------------------------
# Which a-decks are active
# ---------------------------------------------------------------------------

def _rec(fas, cycle, tau=0, model="CARQ", vmax=50.0):
    return fas.ATCFRecord(basin="AL", storm_number=1, cycle=cycle, tau_hours=tau, model=model,
                          lat=20.0, lon=-60.0, vmax_kt=vmax, mslp_hpa=1000.0, storm_name="X")


@pytest.mark.parametrize("age_h,active", [(0, True), (8.6, True), (12, True), (12.5, False), (2546, False)])
def test_activity_is_the_age_of_the_latest_usable_analysis(fas, age_h, active):
    recs = [_rec(fas, NOW - dt.timedelta(hours=age_h)), _rec(fas, NOW - dt.timedelta(hours=age_h + 6))]
    got, age = fas.is_active(recs, NOW)
    assert got is active and age == pytest.approx(age_h)


def test_activity_ignores_forecasts_and_non_analysis_models(fas):
    old = NOW - dt.timedelta(days=30)
    recs = [_rec(fas, old), _rec(fas, NOW, tau=24, model="OFCL"), _rec(fas, NOW, model="NOTANALYSIS")]
    assert fas.is_active(recs, NOW)[0] is False
    assert fas.is_active([], NOW) == (False, None)


def test_adeck_index_parsing_keeps_every_file_and_its_modified_time(fas):
    html = (
        '<a href="aal012026.dat.gz">aal012026.dat.gz</a>   2026-06-18 02:48  1.2M\n'
        '<a href="aep152026.dat.gz">aep152026.dat.gz</a>   2026-10-02 00:44  9.9M\n'
        '<a href="aep192026.dat.gz">aep192026.dat.gz</a>\n'
    )
    idx = fas.parse_adeck_index(html)
    assert idx == {
        "AL012026": dt.datetime(2026, 6, 18, 2, 48),
        "EP152026": dt.datetime(2026, 10, 2, 0, 44),
        "EP192026": None,
    }


# ---------------------------------------------------------------------------
# Live features
# ---------------------------------------------------------------------------

def test_live_case_carries_every_served_model_feature(fas):
    cycle = dt.datetime(2026, 9, 30, 12)
    recs = [_rec(fas, cycle - dt.timedelta(hours=h), vmax=70.0 - h) for h in (0, 6, 12, 24, 30)]
    case = fas.build_live_case("AL012026", recs)
    model = ri_model.load_model(fas.MODEL_ARTIFACT, verify_data=False)
    built = {k for k, v in case.items() if v is not None}
    missing = [f for f in model["feature_names"] if f not in built]
    assert missing == [] and model["serving"]["impute_live"] == [], missing
    want = ori.climatological_mpi_features(20.0, 70.0, 9)
    assert case["abs_lat"] == 20.0
    assert case["mpi_deficit"] == want["mpi_deficit"]
    assert case["intensity_frac_mpi"] == want["intensity_frac_mpi"]
    assert case["analysis_dv_24h"] == 24.0 and case["storm_age_h"] == 30.0   # true clock


def test_live_motion_and_age_use_the_training_sets_definitions(fas):
    cycle = dt.datetime(2026, 9, 30, 12)
    recs = [
        fas.ATCFRecord(basin="AL", storm_number=1, cycle=cycle - dt.timedelta(hours=h), tau_hours=0,
                       model="CARQ", lat=20.0 - 0.1 * h, lon=-60.0 + 0.15 * h, vmax_kt=60.0,
                       mslp_hpa=990.0, storm_name="X")
        for h in (0, 6, 18)
    ]
    case = fas.build_live_case("AL012026", recs)
    assert case["translation_speed_kmh"] == ori.translation_speed_kmh(19.4, -59.1, 20.0, -60.0, 6.0)
    assert 15.0 < case["translation_speed_kmh"] < 20.0      # ~105 km in 6 h
    assert case["storm_age_h"] == 18.0
    # A JTWC warning is one cycle: no motion from history, and an UNKNOWN (not zero) age.
    jtwc = fas.build_live_case("WP262026", recs[:1], full_history=False)
    assert jtwc["translation_speed_kmh"] is None and jtwc["storm_age_h"] is None


def test_live_analysis_takes_carq_whose_pressure_ofcl_does_not_carry(fas):
    """An NHC a-deck cycle has both a CARQ and an OFCL tau-0 line; OFCL's MSLP is 0 (parsed as
    None). With OFCL first, 88% of live cases lost the pressure and its 6/12/24 h tendencies."""
    cycle = dt.datetime(2026, 9, 30, 12)
    recs = []
    for h, p in ((0, 970.0), (6, 980.0), (12, 985.0), (24, 995.0)):
        t = cycle - dt.timedelta(hours=h)
        recs.append(fas.ATCFRecord(basin="AL", storm_number=1, cycle=t, tau_hours=0, model="OFCL",
                                   lat=20.0, lon=-60.0, vmax_kt=90.0 - h, mslp_hpa=None, storm_name="X"))
        recs.append(fas.ATCFRecord(basin="AL", storm_number=1, cycle=t, tau_hours=0, model="CARQ",
                                   lat=20.0, lon=-60.0, vmax_kt=90.0 - h, mslp_hpa=p, storm_name="X"))
    case = fas.build_live_case("AL012026", recs)
    assert case["analysis_model"] == "CARQ"
    assert case["analysis_mslp_hpa"] == 970.0
    assert (case["analysis_dp_6h"], case["analysis_dp_12h"], case["analysis_dp_24h"]) == (-10.0, -15.0, -25.0)
    # A cycle with only OFCL still builds (wind from OFCL), pressure honestly missing.
    only = fas.build_live_case("AL012026", [r for r in recs if r.model == "OFCL"])
    assert only["analysis_model"] == "OFCL" and only["analysis_mslp_hpa"] is None


def test_the_atcf_parser_reads_a_zero_pressure_as_missing():
    from hazardpulse.hurricane.atcf import parse_atcf_deck
    line = ("AL, 09, 2022092612, 03, OFCL,   0, 182N,  830W,  75,    0, HU,  34, NEQ,   80,   60,"
            "   40,   60,    0,    0,   0,   0,   0,   0,   0,   0,    ,   0,         IAN")
    carq = line.replace("OFCL", "CARQ").replace(",    0, HU", ",  979, HU")
    ofcl_rec, carq_rec = parse_atcf_deck(line + "\n" + carq)
    assert ofcl_rec.mslp_hpa is None and carq_rec.mslp_hpa == 979.0
    assert ofcl_rec.vmax_kt == carq_rec.vmax_kt == 75.0


def test_v8_1_artifacts_declare_their_unmatched_live_features():
    pin = ri_model.load_model(ri_model.ARTIFACTS["hurricane_ri_v8_1"], verify_data=False)
    assert pin["serving"]["impute_live"] == ["translation_speed_kmh", "storm_age_h"]
    case = {"analysis_lat": 20.0, "translation_speed_kmh": 999.0, "storm_age_h": 999.0}
    imputed = ri_model.member_probabilities(pin, [case])
    blank = ri_model.member_probabilities(pin, [{"analysis_lat": 20.0}])
    for key in imputed:   # the contract replaces the live values by the medians, as served
        assert imputed[key][0] == blank[key][0]


def test_shared_mpi_definition_reproduces_every_training_row():
    """Serving and training use ONE definition: re-derive it for all 133,882 rows."""
    path = ri_model.DEFAULT_TRAINING_DATA
    n = mismatches = 0
    first_bad = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            month = int(row["issue_time"][5:7])
            got = ori.climatological_mpi_features(row["analysis_lat"], row["analysis_vmax_kt"], month)
            n += 1
            for key in ("abs_lat", "mpi_deficit", "intensity_frac_mpi"):
                if got[key] != row[key]:
                    mismatches += 1
                    first_bad = first_bad or (n, key, got[key], row[key])
    assert n == 133_882
    assert mismatches == 0, first_bad


def test_mpi_features_tolerate_missing_inputs():
    assert ori.climatological_mpi_features(None, 50.0, 9) == {
        "abs_lat": None, "mpi_deficit": None, "intensity_frac_mpi": None}
    out = ori.climatological_mpi_features(-15.0, None, 2)
    assert out["abs_lat"] == 15.0 and out["mpi_deficit"] is None
