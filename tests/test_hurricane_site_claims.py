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


@pytest.fixture(scope="module")
def hu_ev82(hu_ev):
    """The evidence as it reads while v8.2 is the served other-basins model (until amendment 15): its METHOD's
    temporal hold-out. Built by the same function the site uses for v8.2, from the committed files."""
    v82 = json.loads((ROOT / se.HURRICANE_V82_SERVED).read_text(encoding="utf-8"))["model_version"]
    return dict(hu_ev, other_basins=se._v82_other_basins(ROOT, v82, se.version_label(v82)))


V83_FILES = ("results/models/hurricane_ri_v8_3.json", "results/calibration/hurricane_ri_v8_3.json",
             "results/calibration/hurricane_ri_v8_3_test_composition.json", se.HURRICANE_SCORER)


def test_the_published_v83_test_is_described_as_measured(hu_ev):
    """Amendment 15: the served other-basins model is v8.3, and its bound test is ITS registered test -- the
    published artifact on 2025-2026 cycles in the basins where it publishes, none used to fit it -- not v8.2's
    method on every basin. Every number is the results files'."""
    import types
    ob = hu_ev["other_basins"]
    assert se.hurricane_scorer_models()["served"] == ob["model"] == "hurricane_ri_v8_3" and ob["subject"] == "model"
    res = json.loads((ROOT / "results/calibration/hurricane_ri_v8_3.json").read_text(encoding="utf-8"))
    comp = json.loads((ROOT / "results/calibration/hurricane_ri_v8_3_test_composition.json").read_text(encoding="utf-8"))
    c = ob["composition"]
    assert sum(b["n"] for b in c["by_basin"]) == ob["test"]["n"] == comp["n"] == res["test"]["n"]
    assert {b["basin"] for b in c["by_basin"]} == set(res["test"]["decision_basins"]) == {"WP", "NI", "SI", "SP"}
    assert c["served_calibration_fitted_on_test_cases"] is False                     # read from the artifact
    assert ob["test"]["auc"] == res["registered"]["v8_3"]["auc"]
    assert ob["replaced"]["log_loss"] == res["registered"]["v8_2"]["log_loss"]
    assert ob["vs_replaced"]["d_ll_ci"] == res["registered"]["v8_3_minus_v8_2"]["d_ll_ci"]
    sources = hurricane._sources(types.SimpleNamespace(evidence={"hurricane": hu_ev}))      # the live page's text
    text = hurricane.v82_test_text(ob)
    wp = next(b for b in c["by_basin"] if b["basin"] == "WP")
    for page in (sources, text):
        for w in WRONG:
            assert w not in page, w
        assert "every basin" not in page and "calibration fitted on those same" not in page
    assert f"West Pacific {wp['n']:,}" in text and "best track" in text and "none of them used to fit it" in text
    assert f"{ob['test']['log_loss']:.4f}" in text and f"{ob['replaced']['log_loss']:.4f}" in text
    jt = c["single_jtwc_warning"]
    assert f"{jt['log_loss']:.3f}" in text and f"{jt['log_loss_climatology']:.3f}" in text
    ev = {"hurricane": hu_ev, "earthquake": None, "tornado": None}
    pages = {"methods": ep.methods_hurricane(ev), "methods_simple": ep.methods_simple(ev),
             "registry": ep.registry_active(ev), "limits": ep.methods_limits(ev)}
    for name, html in pages.items():
        for w in WRONG:
            assert w not in html, (name, w)
    assert "Its registered test" in pages["methods"] and "calibration fitted on the test" not in pages["methods"]
    assert "basins where it publishes" in pages["methods_simple"] and "registered test" in pages["registry"]
    assert "every basin" not in pages["limits"] and "basins where it publishes" in pages["limits"]


def test_a_replacement_decision_that_did_not_adopt_the_served_artifact_is_refused(tmp_path):
    """The other-basins evidence for v8.3 is refused unless its decision file adopted exactly the served artifact
    by its rule, with every control passed, replacing the v8.2 this repository holds -- and its composition
    describes that test."""
    def root(name):
        r = tmp_path / name
        for rel in V83_FILES:
            (r / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, r / rel)
        return r

    ok = se._replacement_other_basins(root("ok"), "hurricane_ri_v8_3", "hurricane_ri_v8_2")
    assert ok["label"] == "v8.3" and ok["composition"]["by_basin"]
    res_rel, comp_rel = V83_FILES[1], V83_FILES[2]
    cases = {
        "another artifact": (res_rel, lambda d: d["candidate"].update(artifact_sha256="0" * 64)),
        "not adopted": (res_rel, lambda d: d["decision"].update(adopt=False, serve="hurricane_ri_v8_2")),
        "rule not met": (res_rel, lambda d: d["rule"].update(met=False)),
        "a failed control": (res_rel, lambda d: d["controls"].update(all_passed=False)),
        "replaced another model": (res_rel, lambda d: d["comparator"].update(version="hurricane_ri_v8_1")),
        "an upper bound that is not the interval's": (res_rel, lambda d: d["rule"].update(upper=0.001)),
        "a composition of another test": (comp_rel, lambda d: d["evaluation"].update(n=2000)),
        "a composition of other rows": (comp_rel, lambda d: d["dataset"].update(sha256="0" * 64)),
    }
    for name, (rel, fn) in cases.items():
        r = root(name.replace(" ", "_").replace("'", ""))
        d = json.loads((r / rel).read_text(encoding="utf-8"))
        fn(d)
        (r / rel).write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(se.EvidenceError):
            se._replacement_other_basins(r, "hurricane_ri_v8_3", "hurricane_ri_v8_2")
    with pytest.raises(se.EvidenceError):                     # the served file is not the one the decision names
        se._replacement_other_basins(root("v82"), "hurricane_ri_v8_3", "hurricane_ri_v8_1")


def test_the_v82_test_is_described_as_measured(hu_ev82):
    """While v8.2 was served (and whenever the scorer serves it), its test is its METHOD's hold-out from every
    basin with its published calibration fitted on those cycles -- the site audit of 2026-10-05."""
    import types
    hu_ev = hu_ev82
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
    # the other-basins model published now is the evidence's (v8.3 since amendment 15); v8.2's records are retired
    other = hu_ev["other_basins"]["model"]
    assert other == "hurricane_ri_v8_3"
    served = {hu_ev["model_version"]: {"n_storm_cycles": 7, "n_events": 1},
              other: {"n_storm_cycles": 5, "n_events": 0},
              "hurricane_ri_v8_2": {"n_storm_cycles": 9, "n_events": 0}, **old}
    d2 = _site(tmp_path / "b", hu_ev, served)
    live2 = record._published_live(d2, "hu", hu_ev["model_version"])
    assert "7 storm-cycles" in live2 and "5 storm-cycles" in live2 and "14" not in live2
    assert "9 storm-cycles" not in live2                                  # v8.2 no longer publishes
    rows = {r[1]: r[2] for r in record._live_rows(d2)}
    assert rows[f"<code>{hu_ev['model_version']}</code>"] == rows[f"<code>{other}</code>"] == "published"
    assert rows["<code>hurricane_ri_v8_1</code>"] == rows["<code>hurricane_ri_v8_2</code>"] == "retired"
    card = record._hazard_record_card(d2, "hu")
    for w in WRONG:
        assert w not in card


def test_a_number_scored_from_one_jtwc_warning_carries_a_plain_caution(hu_ev):
    ob = hu_ev["other_basins"]
    comp = ob["composition"]
    wp = {"storm_id": "WP262026", "storm_name": "Choi-Wan", "basin": "WP", "lat": 22.9, "lon": 147.0,
          "vmax_kt": 125, "category": "Category 4", "ri_probability": 0.0177, "ri_source": "v8.2",
          "ri_source_label": "HazardPulse v8.2", "issue_time": "2026-10-04T06:00:00",
          "ri_inputs": {"analysis_model": "JTWC", "noaa_aid_stack": {"status": "jtwc_basin: no guidance"}}}
    assert "Treat this number as rough" in hurricane._storm_card(wp)       # even without the evidence bound
    card = hurricane._storm_card(wp, ob)
    jt = comp["single_jtwc_warning"]
    assert "Caution" in card and "single JTWC warning" in card and "Treat this number as rough" in card
    assert f"{jt['n_missing']} of the model&rsquo;s {jt['n_inputs']} inputs" in card
    assert f"{jt['log_loss']:.3f}" in card and f"{jt['log_loss_climatology']:.3f}" in card
    assert "1.8%" in card                                                   # the number itself is unchanged
    nhc = dict(wp, storm_id="EP152026", basin="EP", lon=-175.9, ri_source="noaa_aid_stack",
               ri_inputs={"used": "DTOP"})
    assert "Caution" not in hurricane._storm_card(nhc, ob)
    assert f"the {ob['test']['when'].replace('-', '&ndash;')} West Pacific cycles" in card     # from the evidence
