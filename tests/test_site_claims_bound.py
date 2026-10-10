"""Every model label, year range and claim on the site is bound to an artifact -- including the prose AROUND the
evidence blocks and the text typed in page code (root fix, 2026-10-09).

Until then ``evidence_pages.check_pages`` compared only the text BETWEEN the hp-evidence markers, so three kinds of
typed text could drift from the artifacts without failing anything, and did:

* sentences that contradicted the block beside them: v8.2 "has no test in those basins" (home) and was "tested on
  Atlantic and East Pacific cases, not in the basins where it now publishes" (methods), while the bound test held
  8,317 held-out cycles from every basin; the tornado format change "on 6 August 2025", "an open question we are
  testing", while tornado amendment 10 had measured 2025-08-05 and decided it; TC1 "a gain beyond chance" for a
  95% descriptive comparison beside a 98.75% claim;
* model labels typed in page code ("v10.1" in five places), which a change of the shown model at a look would have
  left behind;
* year ranges typed in page code ("2022-2025", "2022-2024", "2020-2024", "2018-2020"), which a refit or a new
  evaluation would have left behind.

Each check below passes on the site as built and fails on the old text (the witnesses are the old strings,
verbatim).
"""
from __future__ import annotations

import ast
import html
import json
import re
import shutil
import types
from pathlib import Path

import pytest

from hazardpulse.site import build, shell
from hazardpulse.site.pages import hurricane, overview
from hazardpulse.verification import evidence_pages as ep
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

# ---------------------------------------------------------------------------------------------------------------
# the old text, verbatim (git show 645fdf9e7:...), as witnesses that each check can fail
# ---------------------------------------------------------------------------------------------------------------

OLD_HOME_V82 = ". Elsewhere we publish our v8.2 model, which has no test in those basins"
OLD_METHODS_COVERAGE = (
    "<li><strong>Coverage.</strong> Earthquake: the globe between 60&deg;S and 70&deg;N on a 2&deg; grid. Hurricane: "
    "every active tropical cyclone, with NOAA&rsquo;s guidance in the Atlantic, East and Central Pacific and our v8.2 "
    "model elsewhere &mdash; v8.2 was tested on Atlantic and East Pacific cases, not in the basins where it now "
    "publishes. Tornado: every storm NOAA&rsquo;s radar tracks over and near the contiguous US, including northern "
    "Mexico and the Gulf; tornado reports, and so the tests, cover the US only.</li>")
OLD_TORNADO_NOTICE = (
    '<div class="notice notice-warn"><p><strong>Known input gap.</strong> Since NOAA changed the format of its '
    "ProbSevere feed on 6 August 2025, some attributes have changed. Every forecast records which inputs were "
    "missing; whether this costs accuracy is an open question we are testing.</p></div>")
OLD_TC1_HCCA = "Against HCCA it is a gain beyond chance: -6.2 n mi [-11.1, -1.4]."
OLD_METHODS_DATA_ROW = ("<div><dt>Our models</dt><dd>The early intensity guidance in NHC&rsquo;s forecast files; for "
                        "the v10.3 variant, NOAA&rsquo;s GMGSI satellite infrared</dd></div>")
OLD_PAGE_CODE = '''
def _ours_section(d):
    body = ("<p>HazardPulse v10.1 is our own rapid-intensification model, built from NOAA&rsquo;s intensity guidance, "
            "its rapid-intensification probabilities and the official forecast. Over the 2022&ndash;2025 seasons, each ")
    rows.append(("Our experimental model (v10.1)", "x"))
    v = storm.get("ri_v10_shadow") or {}
    note = "Scored the same way, the 2022&ndash;2024 West Pacific cycles were barely better "
'''


# ---------------------------------------------------------------------------------------------------------------
# 1. contradicted sentences fail the check anywhere on a page
# ---------------------------------------------------------------------------------------------------------------

def test_each_old_contradicted_sentence_is_caught_and_the_built_site_carries_none():
    for old in (OLD_HOME_V82, OLD_METHODS_COVERAGE, OLD_TORNADO_NOTICE, OLD_TC1_HCCA):
        assert ep.contradictions(old), old[:60]
    for phrase, why in ep.CONTRADICTED:
        assert ep.contradictions(f"<p>{html.escape(phrase)}</p>") and why        # every entry can fire
    for p in sorted(DIST.rglob("*.html")):
        assert ep.contradictions(p.read_text(encoding="utf-8")) == [], p.relative_to(DIST).as_posix()


def _dist_copy(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    for rel in ep.BLOCKS:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DIST / rel, d / rel)
    return d


def test_a_contradicted_sentence_in_the_prose_around_the_blocks_fails_check_pages(tmp_path):
    """The hole itself: the old Coverage line sat in hand-kept prose, outside every block, and check_pages passed."""
    ev = {k: v for k, v in se.all_evidence().items()}
    d = _dist_copy(tmp_path)
    assert ep.check_pages(d, ev=ev) == []                                  # the committed pages are current
    page = (d / "methods/index.html").read_text(encoding="utf-8")
    old = page.replace('<section class="section" aria-labelledby="limits"><div class="container prose">',
                       '<section class="section" aria-labelledby="limits"><div class="container prose"><ul>'
                       + OLD_METHODS_COVERAGE + "</ul>", 1)
    assert old != page
    (d / "methods/index.html").write_text(old, encoding="utf-8")
    stale = ep.check_pages(d, ev=ev)
    assert stale and all(s.startswith("methods/index.html (contradicted:") for s in stale)
    # the blocks themselves are unchanged: the old check (blocks only) passes on this page -- that was the hole
    assert all(ep.apply_block(old, n, fn(ev)) == old for n, fn in ep.BLOCKS["methods/index.html"].items())


def test_a_contradicted_sentence_typed_in_page_code_fails_the_site_check(monkeypatch):
    """A generated page is checked as the page code writes it, so a sentence typed in a page module fails
    ``build.check_site`` (and so tests/test_site_pages.py) before it can ship."""
    assert not [s for s in build.check_site() if "contradicted" in s]
    monkeypatch.setattr(hurricane, "v82_short", lambda ob: OLD_HOME_V82.lstrip(". "))
    stale = build.check_site()
    assert any(s.startswith("/ (contradicted: 'which has no test in those basins'") for s in stale), stale


# ---------------------------------------------------------------------------------------------------------------
# 2. the shown model is named in ONE place and every page follows it
# ---------------------------------------------------------------------------------------------------------------

def _pointer_root(tmp_path: Path, shown: str) -> Path:
    for rel in [rel for _e, rel, _w, _b in se.HURRICANE_V10_FAMILY] + [
            se.HURRICANE_PROSPECTIVE, "results/calibration/hurricane_ri_v10_vs_all.json"]:
        if (ROOT / rel).exists():
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, tmp_path / rel)
    ptr = json.loads((ROOT / se.HURRICANE_SHOWN).read_text(encoding="utf-8"))
    (tmp_path / se.HURRICANE_SHOWN).write_text(json.dumps(dict(ptr, shown=shown)), encoding="utf-8")
    return tmp_path


def test_the_committed_pointer_is_bound_and_shows_v10_1_until_the_first_look():
    """Amendment 4: v10.1 is shown until the first look (the switch is a commit AT a look); the label is the
    pointer's, read through the artifact and the prospective scorer, never typed by a page."""
    import datetime as dt
    shown = se.shown_model()
    pros = json.loads((ROOT / se.HURRICANE_PROSPECTIVE).read_text(encoding="utf-8"))
    assert shown["label"] == se.entrant_label(shown["entrant"])
    assert shown["shadow_key"] == pros["entrants"][shown["entrant"]]["claim_rule"]["key"]
    assert se.hurricane_evidence()["ours"]["model_version"] == shown["model_version"]
    first_look = min(pros["look_dates"])
    if dt.date.today().isoformat() < first_look:
        assert shown["entrant"] == "v10_1", f"the shown model may change only at a look ({first_look})"


@pytest.mark.skipif(not (ROOT / "results/models/hurricane_ri_v10_4.json").exists(), reason="v10.4 not built")
def test_switching_the_pointer_switches_every_page_and_nothing_else_names_the_old_model(tmp_path):
    root = _pointer_root(tmp_path, "v10_4")
    ours = se.ours_hurricane(root)
    assert (ours["label"], ours["shadow_key"]) == ("v10.4", "ri_v10_4_shadow")
    assert ours["challengers"] == [] and ours["vs_all"] is None           # v10.1's comparison is not v10.4's
    hu = dict(se.hurricane_evidence(), ours=ours)
    ev = {"hurricane": hu, "earthquake": None, "tornado": None, "research": []}
    storm = {"storm_id": "EP012026", "storm_name": "TEST", "basin": "EP", "lat": 15.0, "lon": -110.0,
             "vmax_kt": 80.0, "ri_probability": 0.2, "ri_source": "noaa_aid_stack", "ri_source_label": "NOAA DTOPS",
             "ri_v10_shadow": {"status": "ok", "probability": 0.11, "gate_ok": True},
             "ri_v10_4_shadow": {"status": "ok", "probability": 0.22, "gate_ok": True}}
    pages = {
        "storm card": hurricane._storm_card(storm, hu.get("other_basins"), ours, hu.get("tc1")),
        "ours section": hurricane._ours_section(types.SimpleNamespace(evidence={"hurricane": hu})),
        "methods card": ep._ours_hurricane_card(ours),
        "methods simple": ep.methods_simple(ev),
        "registry simple": ep.registry_simple(ev),
        "registry active": ep.registry_active(ev),
    }
    assert "Our experimental model (v10.4)" in pages["storm card"] and "22.0%" in pages["storm card"]
    assert "11.0%" not in pages["storm card"]
    for name, text in pages.items():
        assert "v10.4" in text, name
        assert "v10.1" not in text, name                                    # nothing typed the old model
    # and back: the committed pointer gives the committed pages
    assert "v10.1" in ep._ours_hurricane_card(se.ours_hurricane(_pointer_root(tmp_path / "b", "v10_1")))


def test_a_pointer_the_artifacts_or_the_prospective_test_do_not_know_is_refused(tmp_path):
    with pytest.raises(se.EvidenceError):
        se.shown_model(_pointer_root(tmp_path / "a", "v10_9"))
    root = _pointer_root(tmp_path / "b", "v10_1")
    pros = json.loads((root / se.HURRICANE_PROSPECTIVE).read_text(encoding="utf-8"))
    del pros["entrants"]["v10_1"]
    (root / se.HURRICANE_PROSPECTIVE).write_text(json.dumps(pros), encoding="utf-8")
    with pytest.raises(se.EvidenceError):                                  # its live forecasts could not be found
        se.shown_model(root)
    art = json.loads((ROOT / "results/models/hurricane_ri_v10_2.json").read_text(encoding="utf-8"))
    assert se.family_label(art, "v10_2") == "v10.2"
    with pytest.raises(se.EvidenceError):                                  # an artifact labelled otherwise
        se.family_label(art, "v10_3")


# ---------------------------------------------------------------------------------------------------------------
# 3. page code types no model label and no year range; hand-kept prose neither
# ---------------------------------------------------------------------------------------------------------------

LABEL = re.compile(r"\bv\d+\.\d+\b")
YEARS = re.compile(r"\b(?:19|20)\d\d\s*(?:-|–|&ndash;|\.\.|to)\s*(?:19|20)\d\d\b")
# every live record key of our RI models: ri_v9_shadow, ri_v10_2_shadow, ri_j1_shadow (until 2026-10-09 the pattern
# knew only the v-family's, so a J-family key typed in page code would have passed)
SHADOW_KEY = re.compile(r"\bri_[a-z]+\d+(?:_\d+)?_shadow\b")


def artifact_labels(models_dir: Path = ROOT / "results" / "models") -> set[str]:
    """Every label a model artifact gives itself in its own ``label`` field that the ``vN.N`` pattern does not
    already cover -- J1's, and whichever label the next artifact carries -- read from the artifacts, never typed
    here."""
    out = set()
    for p in sorted(models_dir.glob("*.json")):
        try:
            art = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        lab = art.get("label") if isinstance(art, dict) else None
        if isinstance(lab, str) and lab.strip() and not LABEL.fullmatch(lab):
            out.add(lab.strip())
    return out


def label_pattern(labels: set[str]) -> re.Pattern:
    """A whole-word match of any of ``labels`` (never matches when there are none)."""
    if not labels:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True))
                      + r")(?![\w])")


ARTIFACT_LABEL = label_pattern(artifact_labels())
# the code that writes what pages say
PAGE_CODE = sorted((ROOT / "src/hazardpulse/site").rglob("*.py")) + [
    ROOT / "src/hazardpulse/verification/evidence_pages.py", ROOT / "src/hazardpulse/verification/served_evidence.py"]
SHADOW_KEY_FREE = sorted((ROOT / "src/hazardpulse/site").rglob("*.py")) + [
    ROOT / "src/hazardpulse/verification/evidence_pages.py"]


def _v82_label() -> str | None:
    return se.version_label(json.loads((ROOT / se.HURRICANE_V82_SERVED).read_text(encoding="utf-8"))["model_version"])


# A typed token is allowed only with the artifact that justifies it, checked here: if the artifact changes, the
# allowance fails and the text must be bound or changed.
# (Until amendment 15, data.py typed "v8.2" as the record token it mapped to a display name; it now derives the name
# from the token the scorer writes, fetch_and_score.model_label, so that allowance is gone.)
ALLOWED = {
    ("src/hazardpulse/verification/served_evidence.py", "v8.2"): (
        "a key of the evaluation file's data_sha256, not displayed",
        lambda: "v8.2" in (json.loads((ROOT / se.HURRICANE_V82_EVALUATION).read_text(encoding="utf-8"))
                           .get("data_sha256") or {})),
}


def _docstrings(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def typed_tokens(source: str, patterns=(LABEL, YEARS, ARTIFACT_LABEL)) -> list[tuple[int, str]]:
    """(line, token) for every model label or year range inside a string literal of ``source`` -- f-string parts
    included, docstrings and comments not."""
    tree = ast.parse(source)
    skip = _docstrings(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for rx in patterns:
                out += [(node.lineno, m.group(0)) for m in rx.finditer(node.value)]
    return out


def test_the_lint_fails_on_the_old_page_code():
    found = {tok for _line, tok in typed_tokens(OLD_PAGE_CODE)}
    assert {"v10.1", "2022&ndash;2025", "2022&ndash;2024"} <= found
    assert any(t == "ri_v10_shadow" for _l, t in typed_tokens(OLD_PAGE_CODE, (SHADOW_KEY,)))


def j1_page_code(label: str, key: str) -> str:
    """Page code that types our satellite model's label and live key, as a J1 row written by hand would."""
    return (f'\ndef _row(storm):\n    v = storm.get("{key}") or {{}}\n'
            f'    return ("Our satellite model ({label})", f"{{v}} -- {label} is in test")\n'
            f'def _other():\n    """A docstring may name {label}."""\n    return "{label}_minus_A"\n')


def test_the_lint_covers_every_label_an_artifact_gives_itself_and_every_live_key():
    """The vN.N pattern could not see J1. The lint now also matches every label a model artifact gives itself
    (read from results/models), and every ri_*_shadow key -- so a J1 row typed by hand fails, while a docstring or a
    results-file key that merely contains the label does not."""
    from hazardpulse.hurricane import ri_j1
    label = json.loads(ri_j1.MODEL_PATH.read_text(encoding="utf-8"))["label"]
    assert label in artifact_labels() and not LABEL.search(label)          # the old pattern was blind to it
    code = j1_page_code(label, ri_j1.SHADOW_KEY)
    found = [tok for _l, tok in typed_tokens(code)]
    assert found == [label, label]                                         # the row's two literals, nothing else
    assert [t for _l, t in typed_tokens(code, (SHADOW_KEY,))] == [ri_j1.SHADOW_KEY]
    assert not typed_tokens(code, (LABEL, YEARS))                          # what the lint saw before
    assert hand_typed(f"<main><p>Our satellite model, {label}, is in test.</p></main>") == [label]


def test_the_label_lint_follows_the_artifacts_not_a_list(tmp_path):
    """A new artifact's label is covered the moment the artifact exists; a label no artifact gives is not."""
    (tmp_path / "a.json").write_text(json.dumps({"label": "K7", "members": []}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"label": "v11.2"}), encoding="utf-8")   # LABEL covers v-labels
    (tmp_path / "c.json").write_text("not json", encoding="utf-8")
    labels = artifact_labels(tmp_path)
    assert labels == {"K7"}
    rx = label_pattern(labels)
    assert rx.findall('"K7 row" "K7" "K7_minus_A" "TK7" "K77" "hurricane_ri_k7"') == ["K7", "K7"]
    assert label_pattern(set()).findall("anything J1 K7") == []


def test_page_code_types_no_model_label_and_no_year_range():
    bad = []
    for path in PAGE_CODE:
        rel = path.relative_to(ROOT).as_posix()
        for line, tok in typed_tokens(path.read_text(encoding="utf-8")):
            if (rel, tok) not in ALLOWED:
                bad.append(f"{rel}:{line}: {tok}")
    assert not bad, "typed in page code (bind it to an artifact, or justify it in ALLOWED): " + "; ".join(bad)
    for key, (why, check) in ALLOWED.items():
        assert check(), f"{key} is allowed because {why}, which no longer holds"


def test_page_code_names_no_shadow_record_key():
    """Which shadow forecast a page shows comes from the pointer (served_evidence.shown_model), never from a key
    typed in page code (hurricane.py read storm["ri_v10_shadow"] until 2026-10-09)."""
    bad = [f"{p.relative_to(ROOT).as_posix()}:{line}: {tok}" for p in SHADOW_KEY_FREE
           for line, tok in typed_tokens(p.read_text(encoding="utf-8"), (SHADOW_KEY,))]
    assert not bad, bad


def _prose_outside_blocks(page: str) -> str:
    main = page[page.find("<main"):page.find("</main>")]
    main = re.sub(r"<!-- (hp-[a-z]+):([\w-]+) -->.*?<!-- /\1:\2 -->", "", main, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", main))


LICENCE = re.compile(r"License v\d+\.\d+")


def hand_typed(page: str) -> list[str]:
    text = LICENCE.sub("", _prose_outside_blocks(page))
    return LABEL.findall(text) + YEARS.findall(text) + ARTIFACT_LABEL.findall(text)


def test_hand_kept_prose_types_no_model_label_and_no_year_range():
    """The hand-kept pages' prose (outside their evidence blocks) names no model label and no year range: those
    belong in a block rendered from the artifacts. The old methods page typed "the v10.3 variant" here."""
    assert hand_typed("<main>" + OLD_METHODS_DATA_ROW + "</main>") == ["v10.3"]
    bad = {}
    for p in shell.PAGES.values():
        f = shell.static_page_file(p.path, DIST)
        if p.static and f.exists():
            hits = hand_typed(f.read_text(encoding="utf-8"))
            if hits:
                bad[p.path] = hits
    assert not bad, bad


def known_labels() -> set[str]:
    """Every model label an artifact names: the prospective test's entrants, our RI family's artifacts, v8.2's
    artifact (published until amendment 15, J1's base since), and the served other-basins model's own version --
    the one the scorer serves, named only when its artifact exists."""
    pros = json.loads((ROOT / se.HURRICANE_PROSPECTIVE).read_text(encoding="utf-8"))
    out = {se.entrant_label(k) for k in pros.get("entrants") or {}}
    for entrant, rel, _w, _b in se.HURRICANE_V10_FAMILY:
        if (ROOT / rel).exists():
            out.add(se.family_label(json.loads((ROOT / rel).read_text(encoding="utf-8")), entrant))
    out.add(_v82_label())
    served = se.hurricane_scorer_models().get("served")
    served_file = ROOT / "results" / "models" / f"{served}.json"
    if served and served_file.exists():
        out.add(se.version_label(json.loads(served_file.read_text(encoding="utf-8"))["model_version"]))
    return out


def test_the_new_evidence_is_read_from_its_files_and_left_out_when_it_describes_another_model(tmp_path):
    """Tornado amendment 10, earthquake E3, TC1's identity and the research record: every number from its file,
    and a file about another model is left out (never raised: the live tornado scorer reads this evidence)."""
    to, eq = se.tornado_evidence(), se.earthquake_evidence()
    t2 = json.loads((ROOT / se.TORNADO_T2).read_text(encoding="utf-8"))
    fc = to["format_change"]
    assert fc["auc"] == t2["candidates"]["a_served/plus_W"]["auc"] and fc["auc_2025"] == to["test"]["auc"]
    assert fc["mean_over_base"] == t2["candidates"]["a_served/plus_W"]["mean_forecast"] / (t2["pos"] / t2["n"])
    assert fc["date"] == "2025-08-05" and fc["served_stays"] is True     # amendment 10's measured switch
    other = dict(t2, candidates={**t2["candidates"], "a_served/plus_W": dict(
        t2["candidates"]["a_served/plus_W"], model_sha256="0" * 64)})
    (tmp_path / se.TORNADO_T2).parent.mkdir(parents=True)
    (tmp_path / se.TORNADO_T2).write_text(json.dumps(other), encoding="utf-8")
    assert se._tornado_format_change(tmp_path, {"model_version": to["model_version"], "test": to["test"]}) is None

    e3 = json.loads((ROOT / se.EARTHQUAKE_E3).read_text(encoding="utf-8"))
    e1_dev = json.loads((ROOT / se.EARTHQUAKE_E1).read_text(encoding="utf-8"))["splits"]["dev"]
    it = se._earthquake_input_types(ROOT, e3["stack"], e3["base"], e1_dev)      # E3's own model: its numbers
    assert (it["non_earthquakes"], it["frozen_events"]) == (e3["frozen_non_earthquakes_removed"], e3["frozen_events"])
    assert it["adopted"] is e3["rule"]["E3s_replaces_S1"] is False
    assert eq["input_types"] == (it if e3["stack"] == eq["model_version"] else None)   # S2 served: E3 is about S1
    assert se._earthquake_input_types(ROOT, "another-model", eq["base_model_version"], {}) is None

    from hazardpulse.hurricane import tc1_live
    tc = se.tc1_hurricane()
    assert tc["model_version"] == f"hurricane_tc1-{tc1_live.sha256(ROOT / se.TC1_DIR / 'state.json')[:12]}"
    rec = {"storm_id": "AL092026", "tc1": {"cycle": "2026-10-08T12:00:00Z", "TC1": {"24": {"lat": 25.0, "lon": -80.0}},
                                         "selection_sha256": tc["selection_sha256"], "season_state_end": tc["season_end"]}}
    assert tc["model_version"] in hurricane._tc1_table(rec, tc)
    rec["tc1"]["selection_sha256"] = "another selection"
    assert tc["model_version"] not in hurricane._tc1_table(rec, tc)       # a record of other models is not named

    rows = {r["key"]: r for r in se.research_record()}
    g1 = json.loads((ROOT / se.RESEARCH_HURRICANE["g1"]).read_text(encoding="utf-8"))
    assert rows["g1"]["d_brier4_ci"] == g1["development"]["paired"]["d_brier4_ci"] and rows["g1"]["carried"] is False
    assert rows["h8"]["carried"] is True and rows["h8"]["became"] == se.family_label(
        json.loads((ROOT / "results/models/hurricane_ri_v10_4.json").read_text(encoding="utf-8")), "v10_4")
    assert rows["fusion"]["claim"] is False and rows["adt"]["passed"] is True
    adt = json.loads((ROOT / se.RESEARCH_HURRICANE["adt"]).read_text(encoding="utf-8"))
    assert rows["adt"]["hss"] == adt["measured"]["hss_2024_2026"] and rows["adt"]["held_out"] == "2024-2026"


def test_the_registry_carries_tc1_and_v9_1_and_marks_the_shown_model_from_the_pointer():
    import hashlib
    reg = json.loads((DIST / "data" / "model-registry.json").read_text(encoding="utf-8"))
    by_version = {(e.get("output_schema") or {}).get("model_version"): e for e in reg["entries"]}
    tc = se.tc1_hurricane()
    e = by_version[tc["model_version"]]
    state = (ROOT / se.TC1_DIR / "state.json").read_bytes().replace(b"\r\n", b"\n")
    assert e["sha256"] == hashlib.sha256(state).hexdigest() and e["weights_path"] == "results/hurricane_tc1/state.json"
    assert e["benchmark"]["track_mean_error_nmi"]["TC1"] == tc["track"]["errors"]["TC1"]
    v9 = se.v9_hurricane()
    assert v9["model_version"] in by_version and v9["label"] == "v9.1"
    shown = [x for x in reg["entries"] if (x.get("lineage") or {}).get("shown_on_site")]
    # two authorities, one per area: the pointer names the one NHC-area model shown beside NOAA's number; the
    # JTWC-basin model is shown on the storm cards of its scope (ri_j1.scope), and on none when that is empty
    from hazardpulse.hurricane import ri_j1
    by_test = {}
    for x in shown:
        by_test.setdefault(x["lineage"]["prospective"]["file"], []).append(x["output_schema"]["model_version"])
    assert by_test.pop("results/hurricane_prospective/v9_shadow.json") == [se.shown_model()["model_version"]]
    assert by_test == ({se.HURRICANE_J1_PROSPECTIVE: [ri_j1.load()[1]]} if ri_j1.scope() else {})


def test_every_model_label_on_the_site_is_one_an_artifact_names():
    known = known_labels()
    assert {"v8.2", "v8.3", "v9.1", "v10.1"} <= known
    bad = {}
    for p in sorted(DIST.rglob("*.html")):
        text = LICENCE.sub("", ep.page_text(p.read_text(encoding="utf-8")))
        unknown = sorted(set(LABEL.findall(text)) - known)
        if unknown:
            bad[p.relative_to(DIST).as_posix()] = unknown
    assert not bad, bad
    assert "v10.5" not in known                     # a label no artifact names would fail the check above
