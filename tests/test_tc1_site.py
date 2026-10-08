"""TC1 on the site and in the scorer: the table beside NHC's forecast, the evidence bound to the served state, and
the scorer's strict season read."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane import atcf, tc1_live
from hazardpulse.site.pages import hurricane as page
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fs():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_and_score_tc1_test", ROOT / "scripts" / "fetch_and_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _storm():
    return {"storm_id": "AL092026", "tc1": {
        "cycle": "2026-10-08T12:00:00Z",
        "TC1": {"24": {"lat": 25.0, "lon": -80.0, "vmax_kt": 85.0}, "72": {"lat": 30.0, "lon": -78.0, "vmax_kt": 70.0}},
        "OFCL": {"24": {"lat": 25.0, "lon": -81.0, "vmax_kt": 90.0}}}}


def test_the_storm_card_shows_our_track_beside_nhcs_lead_by_lead():
    html = page._tc1_table(_storm())
    assert "HazardPulse position" in html and "NHC position" in html
    assert "25.0°N 80.0°W" in html and "25.0°N 81.0°W" in html and "85 kt" in html and "90 kt" in html
    assert "54 n mi" in html                         # 1 degree of longitude at 25 N, in n mi
    assert html.count("<tr>") == 3                   # header + 24 h + 72 h; leads with neither are left out
    assert page._tc1_table({"storm_id": "WP012026"}) == ""


def _root(tmp_path):
    for name in ("state.json", "selection.json", "dev.json", "season_2026.json"):
        src = ROOT / "results" / "hurricane_tc1" / name
        if not src.exists():
            pytest.skip(f"{name} not built")
        (tmp_path / "results" / "hurricane_tc1").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, tmp_path / "results" / "hurricane_tc1" / name)
    return tmp_path


def test_the_evidence_is_bound_to_the_state_the_scorer_serves(tmp_path):
    root = _root(tmp_path)
    ev = se.tc1_hurricane(root)
    dev = json.loads((ROOT / "results/hurricane_tc1/dev.json").read_text(encoding="utf-8"))
    assert ev["track"]["errors"]["TC1"] == dev["track"]["errors"]["TC1"]["mean_over_leads"]
    assert ev["track"]["vs_ofcl"]["claim"] is dev["track"]["claims"]["TC1-OFCL"]["claim"]
    p = root / "results/hurricane_tc1/dev.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["track"]["errors"]["TC1"]["mean_over_leads"] -= 5.0                           # a better number, typed in
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(se.EvidenceError):
        se.tc1_hurricane(root)


def test_the_page_says_tie_unless_a_claim_was_met():
    ev = se.tc1_hurricane(ROOT)
    if ev is None:
        pytest.skip("TC1 results not built")
    html = page._tc1_section(type("D", (), {"evidence": {"hurricane": {"tc1": ev}}})())
    if ev["track"]["vs_ofcl"]["claim"]:
        assert "beats the official forecast" in html
    else:
        assert "a tie at the confidence the test required" in html and "beats the official" not in html


def test_a_season_deck_that_cannot_be_read_stops_tc1_and_nothing_else(fs, capsys):
    now = dt.datetime(2026, 10, 8, 16, 0)
    index = {"AL092026": None, "EP182026": None, "AL952026": None, "WP242026": None, "AL092025": None}
    rec = atcf.parse_atcf_deck("AL, 09, 2026100812, 01, CARQ,   0, 250N,  800W,  80,  980, HU,  34, NEQ,")
    got = fs.season_decks_for_tc1(index, {"AL092026": rec}, now, fetch=lambda sid: rec)
    assert [d.storm for d in got] == ["al092026", "ep182026"]                       # no invest, JTWC or last season
    with pytest.raises(RuntimeError):
        fs.season_decks_for_tc1(index, {"AL092026": rec}, now, fetch=lambda sid: [])
    storm = {"storm_id": "AL092026", "ri_probability": 0.12, "ri_inputs": {"analysis_model": "CARQ"}}
    before = dict(storm)
    n = fs.attach_tc1([storm], index, {"AL092026": rec}, now, fetch=lambda sid: [])
    assert n == 0 and storm == before and "TC1: not issued" in capsys.readouterr().out
