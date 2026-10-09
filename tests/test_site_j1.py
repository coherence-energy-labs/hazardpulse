"""J1 on the site (hurricane RI amendments 13, 13a, 13b and 14): our satellite model for the JTWC basins, shown
beside the published v8.2 -- never instead of it -- and only where its registered scope says.

Every number and label is read from a committed artifact, and each check below is shown failing on a changed
fixture or a hand-typed witness:

* a storm card shows J1's row only for a storm in the library's scope (``ri_j1.scope``, ``ri_j1.storm_region``),
  with the artifact's label and the record's own probability, and leaves the published number untouched;
* the methods card's numbers are the results files' (a changed file changes the page; a file that contradicts
  the artifact, or a scope that is not the registered rule, is refused), and no number on it is typed;
* the research record and the registry carry J1 (and amendment 14's variant) from the same files.
"""
from __future__ import annotations

import ast
import html
import json
import re
import shutil
from pathlib import Path

import pytest

from hazardpulse.hurricane import ir_features, ir_source, ri_j1
from hazardpulse.site.pages import hurricane
from hazardpulse.verification import evidence_pages as ep
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]
MODEL = ri_j1.MODEL_PATH.relative_to(ri_j1.ROOT).as_posix()
SCOPE = ri_j1.SCOPE_PATH.relative_to(ri_j1.ROOT).as_posix()
FILES = (MODEL, se.HURRICANE_J1_RESULTS, SCOPE, se.HURRICANE_V82_SERVED, se.HURRICANE_J1_SCORER,
         se.HURRICANE_V9_PROGRAM)


def _json(rel: str, root: Path = ROOT):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _root(tmp_path: Path) -> Path:
    for rel in FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def _edit(root: Path, rel: str, fn) -> None:
    d = _json(rel, root)
    fn(d)
    (root / rel).write_text(json.dumps(d), encoding="utf-8")


@pytest.fixture(scope="module")
def hu():
    return se.hurricane_evidence()


@pytest.fixture(scope="module")
def j1(hu):
    assert hu["j1"] is not None
    return hu["j1"]


# ---------------------------------------------------------------------------------------------------------------
# what the evidence is bound to
# ---------------------------------------------------------------------------------------------------------------

def test_the_evidence_is_the_artifacts_and_the_results_files(j1):
    art, version = ri_j1.load()
    res, j2 = _json(se.HURRICANE_J1_RESULTS), _json(SCOPE)
    v82 = _json(se.HURRICANE_V82_SERVED)
    assert (j1["label"], j1["model_version"], j1["shadow_key"]) == (art["label"], version, ri_j1.SHADOW_KEY)
    assert j1["label"] == ri_j1.LABEL and j1["live_basins"] == list(ri_j1.LIVE_JTWC_BASINS)
    assert j1["against"] == se.version_label(v82["model_version"]) == art["provenance"]["dev_2024_2025_jtwc"]["champion_label"]
    dev = res["development"]
    assert j1["dev"]["model"]["log_loss"] == dev[art["label"]]["log_loss"] == art["provenance"]["dev_2024_2025_jtwc"]["log_loss"]
    assert j1["dev"]["bar"]["brier"] == dev["A"]["brier"] and j1["dev"]["control"]["auc"] == dev["B"]["auc"]
    assert j1["vs_bar"]["d_ll_ci"] == res[f"{art['label']}_minus_A"]["d_ll_ci"]
    assert j1["vs_control"]["d_ll"] == res[f"{art['label']}_minus_B"]["d_ll"]
    assert [x["d_ll"] for x in j1["noise"]] == [x["d_ll"] for x in res["noise_control"]]
    assert j1["further"]["model"]["log_loss"] == res["further_read_2026"][art["label"]]["log_loss"]
    assert j1["scope"] == list(ri_j1.scope()) == j2["scope_basins"]
    table = j2["development_descriptive"][f"{art['label']}_vs_v82"]
    assert {r["region"]: r["d_ll"] for r in j1["regions"]} == {g: v["b_minus_a"]["d_ll"] for g, v in table.items()}
    assert {"SI", "SP"} <= {r["region"] for r in j1["regions"]}               # the two southern regions apart
    assert j1["inputs"]["ir"] == len(ir_features.IR_NAMES)
    assert j1["inputs"]["against"] == len(set(art["feature_names"]) & set(v82["selected_features"]))
    assert j1["rule"]["looks"] and j1["rule"]["level"] is not None


def test_a_file_that_contradicts_the_artifact_is_refused(tmp_path):
    root = _root(tmp_path / "ok")
    assert se.j1_hurricane(root)["label"] == ri_j1.LABEL
    label = ri_j1.LABEL
    cases = {
        "dev log loss": (se.HURRICANE_J1_RESULTS, lambda d: d["development"][label].update(log_loss=0.13)),
        "the bar": (se.HURRICANE_J1_RESULTS, lambda d: d["development"]["A"].update(brier=0.04)),
        "the interval": (se.HURRICANE_J1_RESULTS, lambda d: d[f"{label}_minus_A"].update(d_ll_ci=[-0.02, 0.001])),
        "not carried": (se.HURRICANE_J1_RESULTS, lambda d: d.update(carried="v8.2")),
        "the further read": (se.HURRICANE_J1_RESULTS, lambda d: d["further_read_2026"][label].update(log_loss=0.2)),
        "another v8.2": (se.HURRICANE_V82_SERVED, lambda d: d.update(model_version="hurricane_ri_v8_3")),
        "another entrant": (SCOPE, lambda d: d.update(live_entrant="J2")),
        "a scope that is not the rule": (SCOPE, lambda d: d.update(scope_basins=["WP", "SI", "SP"])),
        "a region the outcome reports otherwise": (
            SCOPE, lambda d: d["development_descriptive"][f"{label}_vs_v82"]["WP"]["b_minus_a"].update(d_ll=-0.02)),
    }
    for name, (rel, fn) in cases.items():
        root = _root(tmp_path / re.sub(r"\W", "_", name))
        _edit(root, rel, fn)
        with pytest.raises(se.EvidenceError):
            se.j1_hurricane(root)
        assert name


def test_without_its_artifact_or_result_there_is_no_j1_and_without_its_scope_file_it_is_shown_nowhere(tmp_path):
    root = _root(tmp_path / "a")
    (root / MODEL).unlink()
    assert se.j1_hurricane(root) is None
    root = _root(tmp_path / "b")
    (root / se.HURRICANE_J1_RESULTS).unlink()
    assert se.j1_hurricane(root) is None
    root = _root(tmp_path / "c")
    (root / SCOPE).unlink()
    j = se.j1_hurricane(root)
    assert j["scope"] == [] and j["regions"] == []
    assert se.j1_shown_shadow(_storm("SH", 80.0, j), j) is None
    assert "No storm" in ep._j1_card(j)


# ---------------------------------------------------------------------------------------------------------------
# (a) the storm card
# ---------------------------------------------------------------------------------------------------------------

def _storm(basin: str, lon: float, j: dict | None, shadow: dict | None = None, **over) -> dict:
    s = {"storm_id": f"{basin}272026", "storm_name": "TEST", "basin": basin, "lat": -15.0 if basin == "SH" else 18.0,
         "lon": lon, "vmax_kt": 60.0, "category": "Tropical storm", "ri_probability": 0.2755, "ri_source": "v8.2",
         "ri_source_label": "HazardPulse v8.2",
         "ri_inputs": {"inputs_missing": [], "track_source": "ral_bdeck", "n_inputs": 17}}
    if j is not None and shadow is not False:
        s[j["shadow_key"]] = shadow if shadow is not None else {
            "status": "ok", "probability": 0.4321, "model_probability": 0.43213, "model_version": j["model_version"],
            "label": j["label"], "ir_features_read": "14 of 14"}
    return {**s, **over}


def _card(s: dict, hu: dict, j: dict | None) -> str:
    return hurricane._storm_card(s, hu.get("other_basins"), hu.get("ours"), hu.get("tc1"), j)


ROW = re.compile(r"<div><dt>Our satellite model \(.*?</dd></div>")


def test_a_storm_in_scope_shows_j1_beside_the_published_number_and_nothing_else_changes(hu, j1):
    label = _json(MODEL)["label"]
    for basin, lon in (("WP", 135.0), ("SH", 80.0)):                       # West Pacific; South Indian
        s = _storm(basin, lon, j1)
        assert ri_j1.in_scope(basin, lon)
        card = _card(s, hu, j1)
        row = ROW.search(card)
        assert row, basin
        assert f"Our satellite model ({label})" in row.group(0) and "43.2%" in row.group(0)
        assert "shown for comparison while it is tested on new forecasts" in row.group(0)
        assert '<a href="/methods/#hurricane-j1">' in row.group(0)
        without = _card(_storm(basin, lon, j1, shadow=False), hu, j1)
        assert card.replace(row.group(0), "") == without                  # the published number, its source: as before
        assert "27.6%" in card and "source: HazardPulse v8.2" in card
    # the number is the record's: another record, another number
    s = _storm("WP", 135.0, j1, {**_storm("WP", 135.0, j1)[j1["shadow_key"]], "probability": 0.0712})
    assert "7.1%" in ROW.search(_card(s, hu, j1)).group(0)
    # a cycle whose satellite images were missing says so
    s = _storm("WP", 135.0, j1, {**_storm("WP", 135.0, j1)[j1["shadow_key"]], "ir_features_read": "0 of 14"})
    assert "only 0 of its 14 satellite inputs were available" in ROW.search(_card(s, hu, j1)).group(0)
    # the label is the evidence's, not the page's: a relabelled artifact relabels the row
    assert "Our satellite model (K9)" in _card(_storm("WP", 135.0, j1), hu, dict(j1, label="K9"))


def test_no_j1_row_outside_its_scope_or_without_a_good_record_of_this_model(hu, j1):
    ok = _storm("WP", 135.0, j1)[j1["shadow_key"]]
    cases = {
        "no record": _storm("WP", 135.0, j1, shadow=False),
        "a failed record": _storm("WP", 135.0, j1, {"status": "error: OSError: no image"}),
        "no probability": _storm("WP", 135.0, j1, {**ok, "probability": None}),
        "another artifact's record": _storm("WP", 135.0, j1, {**ok, "model_version": "hurricane_ri_j1-000000000000"}),
        "South Pacific (east of 135 E)": _storm("SH", 170.0, j1),
        "South Pacific (across the dateline)": _storm("SH", -170.0, j1),
        "North Indian": _storm("IO", 88.0, j1),
        "a southern storm without a longitude": {**_storm("SH", 80.0, j1), "lon": None},
        "an NHC storm": _storm("EP", -110.0, j1, ri_source="noaa_aid_stack", ri_source_label="NOAA DTOPS"),
    }
    for name, s in cases.items():
        assert se.j1_shown_shadow(s, j1) is None, name
        if s.get("lon") is not None:
            assert "Our satellite model" not in _card(s, hu, j1), name
    s = _storm("WP", 135.0, j1)
    assert se.j1_shown_shadow(s, j1) is not None                           # the control: the same storm, in scope
    assert se.j1_shown_shadow(s, None) is None and se.j1_shown_shadow(s, dict(j1, scope=[])) is None
    assert "Our satellite model" not in _card(s, hu, None)


def test_the_hurricane_page_says_where_j1_is_shown_from_its_scope(j1):
    text = hurricane.j1_sources_text(j1, j1["against"])
    names = [r["name"] for r in j1["regions"] if r["in_scope"]]
    assert names and all(n in text for n in names) and j1["label"] in text and "/methods/#hurricane-j1" in text
    out = [r["name"] for r in j1["regions"] if not r["in_scope"]]
    assert out and not any(n in text for n in out)
    assert hurricane.j1_sources_text(dict(j1, regions=[]), j1["against"]) == ""


# ---------------------------------------------------------------------------------------------------------------
# (b) the methods card
# ---------------------------------------------------------------------------------------------------------------

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _formats(x, out: set) -> None:
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        out.update({str(x), f"{x:,}", str(abs(x))})
    if isinstance(x, (int, float)) and x == x and abs(x) != float("inf"):
        for d in range(6):
            out.update({f"{abs(x):.{d}f}", f"{abs(x):,.{d}f}", f"{abs(100 * x):.{d}f}"})


def _walk(obj, out: set) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(NUMBER.findall(str(k)))
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        _formats(len(obj), out)
        for v in obj:
            _walk(v, out)
    elif isinstance(obj, str):
        out.update(n.rstrip(",") for n in NUMBER.findall(obj))
    else:
        _formats(obj, out)


def allowed_numbers(root: Path = ROOT) -> set[str]:
    """Every number the J1 card may print: each number in its bound files, as any fixed-point format (and as a
    percentage), each digit run of their strings and keys, each list's length; the program passage it quotes; the
    library's satellite inputs and image hours; and the interval level, named once in served_evidence and checked
    against the run's code below."""
    out: set[str] = set()
    art = _json(MODEL, root)
    _walk({k: v for k, v in art.items() if k != "members"}, out)
    for rel in (se.HURRICANE_J1_RESULTS, SCOPE):
        _walk(_json(rel, root), out)
    _walk(_json(se.HURRICANE_V82_SERVED, root).get("selected_features"), out)
    _walk(se._script_constants(root, se.HURRICANE_J1_SCORER, ("LOOKS", "LEVEL")), out)
    passage = se.program_passage(root, se.HURRICANE_V9_PROGRAM, se.J1_FRAGILITY_LEAD) or {}
    _walk([passage.get("text"), passage.get("amendment"), *passage.get("bullets", [])], out)
    _walk([len(ir_features.IR_NAMES), *[abs(h) for h in ir_source.OFFSETS.values()], se.J1_INTERVAL_LEVEL], out)
    _walk(_json(SCOPE, root).get("program"), out)
    return out


def typed_numbers(card: str, allowed: set[str]) -> list[str]:
    """The numbers a reader sees on ``card`` (model versions and file links aside) that no bound file holds."""
    text = re.sub(r"<code>.*?</code>|<a [^>]*>.*?</a>", " ", card)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return [n for n in (m.rstrip(",") for m in NUMBER.findall(text)) if n not in allowed]


def test_every_number_on_the_methods_card_is_in_a_bound_file(j1):
    card = ep._j1_card(j1)
    allowed = allowed_numbers()
    assert typed_numbers(card, allowed) == []
    # the check can fail: a number typed into the card is caught, and so is one the files do not hold
    assert typed_numbers(card.replace("</dl>", "<div><dt>x</dt><dd>log loss 0.12345</dd></div></dl>"), allowed) == ["0.12345"]
    res = _json(se.HURRICANE_J1_RESULTS)
    for x in (res["development"][j1["label"]]["log_loss"], res["development"]["A"]["log_loss"],
              res["development"]["B"]["log_loss"]):
        assert f"{x:.5f}" in card
    lo, hi = res[f"{j1['label']}_minus_A"]["d_ll_ci"]
    assert f"[&minus;{abs(lo):.5f}, &minus;{abs(hi):.5f}]" in card
    assert f"{100 * j1['rule']['level']:g}%" in card and all(x in card for x in j1["rule"]["looks"])
    assert 'id="hurricane-j1"' in card and j1["model_version"] in card


def test_the_methods_card_follows_its_files(tmp_path):
    """Change a number in a bound file and the card changes with it."""
    base = ep._j1_card(se.j1_hurricane(_root(tmp_path / "base")))
    root = _root(tmp_path / "edit")
    _edit(root, se.HURRICANE_J1_RESULTS, lambda d: d["development"]["B"].update(log_loss=0.15))
    _edit(root, se.HURRICANE_J1_RESULTS, lambda d: d["noise_control"][0].update(d_ll=0.0042))
    _edit(root, SCOPE, lambda d: d["development_descriptive"][f"{ri_j1.LABEL}_vs_v82"]["SP"]["b_minus_a"].update(
        d_ll_ci=[0.0061, 0.0362]))
    card = ep._j1_card(se.j1_hurricane(root))
    assert card != base
    assert "0.15000" in card and "0.15000" not in base
    assert "+0.00420" in card                                               # the noise control's new maximum
    assert "[+0.0061, +0.0362]" in card and "[+0.0061, +0.0362]" not in base
    assert typed_numbers(card, allowed_numbers(root)) == []
    # the quoted program passage is the program's: rewrite it and the card quotes the new words
    doc = (root / se.HURRICANE_V9_PROGRAM).read_text(encoding="utf-8")
    (root / se.HURRICANE_V9_PROGRAM).write_text(doc.replace("moving the test crops 15 km", "moving the test crops 25 km"),
                                                encoding="utf-8")
    assert "moving the test crops 25 km" in ep._j1_card(se.j1_hurricane(root))


def test_the_card_states_the_claim_as_measured_and_the_scope_with_its_reasons(j1):
    text = ep.page_text(ep._j1_card(j1))
    assert "pooled" in text and "best-track positions" in text and "hindcast" in text
    assert "not made basin by basin, nor for live forecasts" in text
    names = {r["region"]: r["name"] for r in j1["regions"]}
    shown = [names[g] for g in j1["scope"]]
    assert f"{' and '.join(shown)} storms only" in text
    for r in j1["regions"]:
        if not r["in_scope"]:
            assert re.search(re.escape(r["name"]) + r"\s*: .{0,200}?, so it is not shown there", text), r["region"]
    split = [r["name"] for r in j1["regions"] if r["region"] not in j1["outcome_basins"]]
    assert split and f"merged the {' and '.join(split)} into one" in text       # SI and SP, apart here
    sp = next(r for r in j1["regions"] if r["region"] == "SP")
    assert sp["d_ll_ci"][0] > 0 and f"{j1['label']} tested worse than {j1['against']} on 2024–2025, beyond its interval" in text
    v26 = next(x for x in j1["variant"]["season_2026"] if x["region"] == "SP")
    assert v26["d_ll"] <= 0 and "the rule excludes it although that season points the other way" in text
    assert "It excludes 0 by" in text and "dropping one storm" in text      # the program's words, quoted
    ni = next(r for r in j1["regions"] if r["events"] == 0)
    assert f"The {ni['name']} basin had no RI event in the test seasons" in text
    assert "The live record starts with its first matured cycle" in text   # no prospective record yet
    assert "never the published number" in text and "on live cycles in its scope only" in text


def test_the_interval_level_named_once_is_the_runs():
    """served_evidence names the level of the J1 run's intervals (the results file does not carry it); it must be
    the level the run's own percentiles give."""
    def level(source: str) -> float:
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "paired")
        qs = sorted({float(c.args[1].value) for c in ast.walk(fn) if isinstance(c, ast.Call)
                     and getattr(c.func, "attr", "") == "percentile"})
        assert len(qs) == 2
        return round((qs[1] - qs[0]) / 100, 10)
    src = (ROOT / se.HURRICANE_J1_RUN).read_text(encoding="utf-8")
    assert level(src) == se.J1_INTERVAL_LEVEL
    assert level(src.replace("np.percentile(bl, 2.5)", "np.percentile(bl, 5.0)")
                 .replace("np.percentile(bl, 97.5)", "np.percentile(bl, 95.0)")
                 .replace("np.percentile(bb, 2.5)", "np.percentile(bb, 5.0)")
                 .replace("np.percentile(bb, 97.5)", "np.percentile(bb, 95.0)")) != se.J1_INTERVAL_LEVEL


def test_the_live_record_once_it_exists_is_read_on_the_same_scope_and_looks(tmp_path):
    root = _root(tmp_path / "a")
    k = se._script_constants(root, se.HURRICANE_J1_SCORER, ("LOOKS",))
    rec = {"entrant": ri_j1.LABEL, "look_dates": list(k["LOOKS"]), "scope": list(ri_j1.scope()), "records_with_j1": 40,
           "scored_as_of": "2026-10-20T06:00Z", "looks": {},
           "running": {"j1": {"n": 12, "events": 2, "log_loss": 0.211}, "v8_2": {"n": 12, "events": 2, "log_loss": 0.245},
                       "storms": 3, "level": 0.975,
                       "j1_minus_v8_2": {"d_log_loss": -0.034, "d_log_loss_ci": [-0.09, 0.02]}}}
    (root / se.HURRICANE_J1_PROSPECTIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / se.HURRICANE_J1_PROSPECTIVE).write_text(json.dumps(rec), encoding="utf-8")
    text = ep.page_text(ep._j1_card(se.j1_hurricane(root)))
    assert "Live so far, in its scope" in text and "12 cycles from 3 storms (2 RI events)" in text
    assert "0.211 against 0.245" in text and "The live record starts" not in text
    (root / se.HURRICANE_J1_PROSPECTIVE).write_text(json.dumps(dict(rec, running={"n": 0})), encoding="utf-8")
    assert "40 live records carry it; none in its scope has matured yet" in ep.page_text(ep._j1_card(se.j1_hurricane(root)))
    for bad in (dict(rec, look_dates=["2026-11-01"]), dict(rec, scope=["WP", "SI", "SP"]), dict(rec, entrant="J2")):
        (root / se.HURRICANE_J1_PROSPECTIVE).write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(se.EvidenceError):
            se.j1_hurricane(root)


def test_the_methods_page_carries_the_card_in_its_hurricane_block():
    page = (ROOT / "dist" / "methods" / "index.html").read_text(encoding="utf-8")
    block = re.search(r"<!-- hp-evidence:methods-hurricane -->(.*?)<!-- /hp-evidence:methods-hurricane -->", page, re.S)
    assert 'id="hurricane-j1"' in block.group(1)
    assert ep._j1_card(se.j1_hurricane()) in block.group(1)


# ---------------------------------------------------------------------------------------------------------------
# the research record and the registry
# ---------------------------------------------------------------------------------------------------------------

def test_the_research_record_carries_amendments_13_and_14_from_their_files():
    rows = {r["key"]: r for r in se.research_record()}
    res, j2 = _json(se.HURRICANE_J1_RESULTS), _json(SCOPE)
    label = _json(MODEL)["label"]
    r = rows["j1"]
    assert r["amendment"] == "13" and r["carried"] is True and r["became"] == label
    assert r["d_ll_ci"] == res[f"{label}_minus_A"]["d_ll_ci"] and r["ir"]["d_ll"] == res[f"{label}_minus_B"]["d_ll"]
    assert r["n"] == res["development"][label]["n"] and r["n_features"] == len(res["candidates"][label])
    v = rows["j2"]
    variant = next(k for k in j2["names"] if k != label)
    assert v["amendment"] == "14" and v["candidate"] == variant and v["carried"] is (j2["carried"] == variant)
    assert v["d_ll"] == j2["season_2026"][f"{variant}_minus_{label}"]["d_ll"]
    html_rows = ep.methods_research({"research": se.research_record()})
    assert f"Carried on the pooled hindcast: it is {label}, in test beside" in html_rows
    assert f"Not carried: its log loss is not below {label}&rsquo;s" in html_rows
    for p in (se.HURRICANE_J1_RESULTS, SCOPE):
        assert p in html_rows


def test_the_registry_carries_j1_from_its_artifact():
    reg = json.loads((ROOT / "dist" / "data" / "model-registry.json").read_text(encoding="utf-8"))
    _art, version = ri_j1.load()
    e = next(x for x in reg["entries"] if (x.get("output_schema") or {}).get("model_version") == version)
    lin = e["lineage"]
    assert e["output_schema"]["label"] == _json(MODEL)["label"] and e["weights_path"] == MODEL
    assert lin["role"] == "shadow" and lin["status"] == "in test (shadow)"
    assert lin["live_records"]["key"] == ri_j1.SHADOW_KEY and lin["prospective"]["file"] == se.HURRICANE_J1_PROSPECTIVE
    assert lin["scope"] == list(ri_j1.scope())
    assert {p["path"] for p in lin["selection_and_evaluation"]} == {se.HURRICANE_J1_RESULTS, SCOPE}
    assert e["benchmark"]["dev_log_loss_30kt"] == _json(se.HURRICANE_J1_RESULTS)["development"][ri_j1.LABEL]["log_loss"]
    page = (ROOT / "dist" / "registry" / "index.html").read_text(encoding="utf-8")
    assert f"<code>{version}</code>" in page
