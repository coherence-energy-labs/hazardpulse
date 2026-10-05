"""What the site says about the hurricane models must be what was measured (site audit 2026-10-05):

* v8.2's 84.7% was a test of its METHOD on held-out 2022-2024 cycles from EVERY basin, with best-track
  inputs; the published artifact's calibration was fitted on those same cycles. The site said
  "held-out Atlantic and East Pacific cases ... not tested in the basins where it publishes".
* "Live record of this version: 14 forecasts" pooled every version that ever published (all 14 were
  April West Pacific numbers of models no longer published).
* a West Pacific number is v8.2 scored from ONE JTWC warning, 9 of its 17 inputs unknown; the page gave
  it to 0.1% with no word of that.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hazardpulse.site.data import SiteData
from hazardpulse.site.pages import hurricane, record
from hazardpulse.verification import evidence_pages as ep
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / se.HURRICANE_V82_COMPOSITION
pytestmark = pytest.mark.skipif(not COMP.exists() or not (ROOT / se.HURRICANE_V82_SERVED).exists(),
                                reason="v8.2 evidence not in this checkout")

WRONG = ("Atlantic and East Pacific cases", "has not been tested in the basins", "tested only on Atlantic",
         "NHC cases only", "its test cases are NHC")


@pytest.fixture(scope="module")
def hu_ev():
    return se.hurricane_evidence()


def test_the_v82_test_is_described_as_measured(hu_ev):
    import types
    sources = hurricane._sources(types.SimpleNamespace(evidence={"hurricane": hu_ev}))      # the live page's text
    for w in WRONG:
        assert w not in sources, w
    assert "every basin" in sources
    comp = json.loads(COMP.read_text(encoding="utf-8"))
    ob = hu_ev["other_basins"]
    c = ob["composition"]
    assert sum(b["n"] for b in c["by_basin"]) == ob["test"]["n"] == comp["n"]
    assert {b["basin"] for b in c["by_basin"]} >= {"WP", "SI", "NA", "EP"}             # every basin, not NHC's
    wp = next(b for b in c["by_basin"] if b["basin"] == "WP")
    assert wp["n"] == comp["by_basin"]["WP"]["n"]                                       # never typed
    assert c["served_calibration_fitted_on_test_cases"] is True                       # read from the artifact
    assert c["single_jtwc_warning"]["n_missing"] == len(comp["single_jtwc_warning"]["inputs_missing_live"])
    text = hurricane.v82_test_text(ob)
    for w in WRONG:
        assert w not in text
    assert f"West Pacific {wp['n']:,}" in text and "best track" in text and "calibration fitted on those same" in text
    jt = c["single_jtwc_warning"]
    assert f"{jt['log_loss']:.3f}" in text and f"{jt['log_loss_climatology']:.3f}" in text
    ev = {"hurricane": hu_ev, "earthquake": None, "tornado": None}
    pages = {"methods": ep.methods_hurricane(ev), "methods_simple": ep.methods_simple(ev),
             "registry": ep.registry_active(ev)}
    for name, html in pages.items():
        for w in WRONG:
            assert w not in html, (name, w)
    assert "every basin" in pages["methods"] and "calibration fitted on the test" in pages["methods"]
    assert "every basin" in pages["methods_simple"] and "every basin" in pages["registry"]


def test_a_composition_file_of_another_evaluation_is_refused(tmp_path):
    for rel in (se.HURRICANE_V82_EVALUATION, se.HURRICANE_V82_COMPOSITION, se.HURRICANE_V82_SERVED):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    evaluation = json.loads((ROOT / se.HURRICANE_V82_EVALUATION).read_text(encoding="utf-8"))
    heldout = evaluation["results"]["C_v8_2_heldout_newton"]
    assert se._v82_composition(tmp_path, evaluation, heldout)["by_basin"]
    comp = json.loads(COMP.read_text(encoding="utf-8"))
    comp["evaluation"]["n"] = 8000
    (tmp_path / se.HURRICANE_V82_COMPOSITION).write_text(json.dumps(comp), encoding="utf-8")
    with pytest.raises(se.EvidenceError):
        se._v82_composition(tmp_path, evaluation, heldout)


def _site(tmp_path: Path, hu_ev: dict, by_version: dict) -> SiteData:
    out = tmp_path / "results" / "hurricane_prospective"
    out.mkdir(parents=True)
    pooled = sum(int(v["n_storm_cycles"]) for v in by_version.values())     # what the old card quoted
    (out / "prospective_summary.json").write_text(json.dumps({
        "status": "ok", "by_model_version": by_version, "total_storm_predictions": pooled,
        "total_ri_events": sum(int(v["n_events"]) for v in by_version.values())}), encoding="utf-8")
    (tmp_path / "dist").mkdir()
    d = SiteData(root=tmp_path, dist=tmp_path / "dist")
    d.__dict__["evidence"] = {"earthquake": None, "hurricane": hu_ev, "tornado": None, "_errors": []}
    d.__dict__["verification_by_key"] = {"hu": {"model_version": hu_ev["model_version"]}}
    return d


def test_the_live_record_counts_only_the_versions_published_now(tmp_path, hu_ev):
    old = {"hurricane_ri_v8_1": {"n_storm_cycles": 3, "n_events": 0},
           "jtwc_warning_ingest": {"n_storm_cycles": 11, "n_events": 0}}
    d = _site(tmp_path, hu_ev, old)
    live = record._published_live(d, "hu", hu_ev["model_version"])
    assert "14" not in live and "11" not in live and live.count("none closed yet") == 2
    served = {hu_ev["model_version"]: {"n_storm_cycles": 7, "n_events": 1},
              "hurricane_ri_v8_2": {"n_storm_cycles": 5, "n_events": 0}, **old}
    d2 = _site(tmp_path / "b", hu_ev, served)
    live2 = record._published_live(d2, "hu", hu_ev["model_version"])
    assert "7 storm-cycles" in live2 and "5 storm-cycles" in live2 and "14" not in live2
    rows = {r[1]: r[2] for r in record._live_rows(d2)}
    assert rows[f"<code>{hu_ev['model_version']}</code>"] == "published"
    assert rows["<code>hurricane_ri_v8_1</code>"] == "retired"
    card = record._hazard_record_card(d2, "hu")
    for w in WRONG:
        assert w not in card


def test_a_number_scored_from_one_jtwc_warning_carries_a_plain_caution(hu_ev):
    comp = hu_ev["other_basins"]["composition"]
    wp = {"storm_id": "WP262026", "storm_name": "Choi-Wan", "basin": "WP", "lat": 22.9, "lon": 147.0,
          "vmax_kt": 125, "category": "Category 4", "ri_probability": 0.0177, "ri_source": "v8.2",
          "ri_source_label": "HazardPulse v8.2", "issue_time": "2026-10-04T06:00:00",
          "ri_inputs": {"analysis_model": "JTWC", "noaa_aid_stack": {"status": "jtwc_basin: no guidance"}}}
    assert "Treat this number as rough" in hurricane._storm_card(wp)       # even without the evidence bound
    card = hurricane._storm_card(wp, comp)
    jt = comp["single_jtwc_warning"]
    assert "Caution" in card and "single JTWC warning" in card and "Treat this number as rough" in card
    assert f"{jt['n_missing']} of the model&rsquo;s {jt['n_inputs']} inputs" in card
    assert f"{jt['log_loss']:.3f}" in card and f"{jt['log_loss_climatology']:.3f}" in card
    assert "1.8%" in card                                                   # the number itself is unchanged
    nhc = dict(wp, storm_id="EP152026", basin="EP", lon=-175.9, ri_source="noaa_aid_stack",
               ri_inputs={"used": "DTOP"})
    assert "Caution" not in hurricane._storm_card(nhc, comp)
