"""Every model number on the site comes from the results file bound to the SERVED artifact."""
from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import pytest

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.verification import evidence_pages as ep
from hazardpulse.verification import served_evidence as se

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# a tiny served tornado suite + lab results in a temporary repository
# ---------------------------------------------------------------------------

def _stump(names, auc, run, label="storm_60"):
    tree = {"feature": [0, -1, -1], "threshold": [0.5, 0.0, 0.0], "default_left": [True, False, False],
            "missing_type": [0, 0, 0], "left": [1, -1, -1], "right": [2, -1, -1], "value": [0.0, -3.0, -1.0],
            "node_value": [-2.0, -3.0, -1.0]}
    return {"schema": lp.SCHEMA, "feature_names": list(names), "n_trees": 1, "trees": [tree],
            "calibration": {"method": "platt", "a": 1.0, "b": 0.0},
            "provenance": {"final_run": run, "label": label, "forecasts": "P(a tornado from THIS storm within 60 min)",
                           "event": "a tornado report starts within 10 km of THIS storm's tracked polygon within 60 min",
                           "trained": "2020-10-15..2024-12-31",
                           "final_2025": {"auc": auc, "auc_ci": [auc - 0.006, auc + 0.005], "bss": 0.11,
                                          "pr_auc": 0.18, "n": 1469977, "pos": 1579}}}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


MAIN_AUC, FB_AUC = 0.96912345678, 0.96812345678


def _tornado_repo(tmp: Path, lab_auc: float | None = MAIN_AUC) -> Path:
    p_names = [f"p_x{i}" for i in range(28)]
    (tmp / se.TORNADO_SERVED).parent.mkdir(parents=True, exist_ok=True)
    lp.save(_stump(p_names + ["w_tor_warning_active", "w_minutes_since_issue"], MAIN_AUC, "v3_plus_W"),
            tmp / se.TORNADO_SERVED)
    lp.save(_stump(p_names, FB_AUC, "v3_primary"), tmp / se.TORNADO_FALLBACK)
    lab = tmp / se.TORNADO_RESULTS
    if lab_auc is None:
        return tmp
    _write(lab / "final_v3_plus_W.json", {"final_2025": {"auc": lab_auc}})
    _write(lab / "final_v3_primary.json", {"final_2025": {"auc": FB_AUC}})
    for b, a, d in (("v3_plus_W", "probtor_raw", 0.090), ("v3_plus_W", "probtor_tiebroken", 0.027),
                    ("v3_plus_W", "v2", 0.006), ("v3_primary", "probtor_raw", 0.089),
                    ("v3_primary", "probtor_tiebroken", 0.026)):
        _write(lab / f"compare_{b}_vs_{a}_final_storm_60.json",
               {"a": a, "b": b, "n_days": 365, "n_boot": 2000, "delta_auc": d, "delta_auc_ci": [d - 0.01, d + 0.01],
                "delta_brier": -2e-5, "delta_brier_ci": [-3e-5, -1e-5]})
    _write(lab / "nws_bar_v3_plus_W_final_storm_60.json",
           {"nws_pod": 0.227, "nws_pofd": 0.0009, "model_pod_at_nws_pofd": 0.271, "delta_pod": 0.044,
            "delta_pod_ci": [0.02, 0.06], "positives": 1579})
    _write(lab / "nws_bar_v3_primary_final_storm_60.json",
           {"nws_pod": 0.227, "nws_pofd": 0.0009, "model_pod_at_nws_pofd": 0.220, "delta_pod": -0.007,
            "delta_pod_ci": [-0.04, 0.02], "positives": 1579})
    _write(lab / "baselines_storm_60.json", {"final": {"probtor_raw": {"auc": 0.879, "auc_ci": [0.85, 0.90]},
                                                        "probtor_tiebroken": {"auc": 0.942, "auc_ci": [0.93, 0.95]}}})
    _write(lab / "reliability_v3_plus_W_final.json",
           {"bins": [{"lo": 0.0, "hi": 0.01, "n": 1400000, "pos": 300, "observed": 0.0002, "mean_forecast": 0.0003,
                      "observed_ci": [0.00019, 0.00024]},
                     {"lo": 0.01, "hi": 0.05, "n": 0, "pos": 0, "observed": None, "mean_forecast": None},
                     {"lo": 0.05, "hi": 1.0, "n": 900, "pos": 120, "observed": 0.133, "mean_forecast": 0.12,
                      "observed_ci": [0.112, 0.156]}]})
    return tmp


# ---------------------------------------------------------------------------
# tornado
# ---------------------------------------------------------------------------

def test_tornado_evidence_is_the_payloads_own_numbers_and_its_bound_comparisons(tmp_path):
    ev = se.tornado_evidence(_tornado_repo(tmp_path))
    assert ev["model_version"] == lp.model_version(lp.load(tmp_path / se.TORNADO_SERVED))
    assert ev["test"]["auc"] == MAIN_AUC and ev["results_bound"] is True
    assert ev["input_families"] == {"ProbSevere storm attributes": 28, "NWS tornado-warning state": 2}
    assert ev["inputs_use_nws"] and not ev["inputs_use_hrrr"]
    # the comparison quoted is the LEAST favourable ProbTor variant, not the flattering one
    assert ev["vs_probtor"]["against"] == "probtor_tiebroken" and ev["vs_probtor"]["delta_auc"] == 0.027
    assert ev["fallback"]["test"]["auc"] == FB_AUC
    assert ev["fallback"]["vs_nws_warnings"]["delta_pod"] == -0.007


def test_a_lab_result_that_is_not_this_payloads_is_refused(tmp_path):
    with pytest.raises(se.EvidenceError):
        se.tornado_evidence(_tornado_repo(tmp_path, lab_auc=MAIN_AUC + 1e-9))


def test_without_lab_results_no_comparison_is_quoted(tmp_path):
    ev = se.tornado_evidence(_tornado_repo(tmp_path, lab_auc=None))
    assert ev["test"]["auc"] == MAIN_AUC
    assert ev["results_bound"] is False
    assert ev["vs_probtor"] is None and ev["vs_nws_warnings"] is None and ev["stress"] is None


def test_no_served_payload_means_no_tornado_evidence(tmp_path):
    assert se.tornado_evidence(tmp_path) is None


def test_reliability_bin_lookup_never_invents_a_rate(tmp_path):
    ev = se.tornado_evidence(_tornado_repo(tmp_path))
    table = se.reliability_for(ev, "v3_w")
    assert se.reliability_bin(table, 0.004)["observed"] == 0.0002
    assert se.reliability_bin(table, 0.02) is None            # an empty bin: no rate
    assert se.reliability_bin(table, 0.30)["n"] == 900
    assert se.reliability_bin(table, 1.0)["n"] == 900         # the top edge belongs to the last bin
    assert se.reliability_bin(table, math.nan) is None
    assert se.reliability_for(ev, "v3") is None                # the fallback's own table is not in this repo


# ---------------------------------------------------------------------------
# earthquake / hurricane: the real files, and a tampered copy
# ---------------------------------------------------------------------------

def test_the_served_earthquake_and_hurricane_models_are_bound_to_their_final_tests():
    eq = se.earthquake_evidence()
    base = json.loads((ROOT / se.EARTHQUAKE_ARTIFACT_RECORD).read_text())["model_version"]
    if se.EARTHQUAKE_STACK:                               # a stack on C0 is served (E1: S1; E4: S2)
        rec = json.loads((ROOT / se.EARTHQUAKE_STACK_RECORD).read_text(encoding="utf-8"))
        meta = json.loads((ROOT / se.EARTHQUAKE_STACK).read_text(encoding="utf-8"))
        cand = (meta.get("provenance") or {}).get("candidate", "S1")
        rep = json.loads((ROOT / se.EARTHQUAKE_STACK_REPORTS[cand]).read_text(encoding="utf-8"))
        assert eq["candidate"] == cand and eq["model_version"] == rec["model_version"]
        assert eq["base_model_version"] == base == rec["base_model_version"]
        assert eq["test"]["auc"]["value"] == rep["splits"]["final"]["models"][cand]["auc"]["value"]
        assert eq["test"]["second_read"] is True                  # never presented as a first read
        assert (eq["gear1"]["activity"] is not None) == (cand == "S2")
    else:
        final = json.loads((ROOT / se.EARTHQUAKE_FINAL).read_text(encoding="utf-8"))
        assert eq["model_version"] == base
        assert eq["test"]["auc"]["value"] == final["candidates"][eq["candidate"]]["auc"]["value"]
    hu = se.hurricane_evidence()
    hfinal = json.loads((ROOT / se.HURRICANE_FINAL).read_text(encoding="utf-8"))
    assert hu["model_version"] == hfinal["artifact"]["model_version"]
    assert hu["test"]["auc"] == hfinal["results"]["all_cases"]["forecasts"][hu["candidate"]]["auc"]
    for ev in (eq, hu):
        assert (ROOT / ev["program"]).exists(), ev["program"]


def _copy(rel: str, tmp: Path) -> Path:
    dst = tmp / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / rel, dst)
    return dst


def test_an_earthquake_record_naming_another_artifact_is_refused(tmp_path):
    for rel in (se.EARTHQUAKE_SERVED, se.EARTHQUAKE_FINAL):
        _copy(rel, tmp_path)
    rec = json.loads((ROOT / se.EARTHQUAKE_ARTIFACT_RECORD).read_text(encoding="utf-8"))
    rec["model_version"] = rec["model_version"][:-1] + ("0" if rec["model_version"][-1] != "0" else "1")
    _write(tmp_path / se.EARTHQUAKE_ARTIFACT_RECORD, rec)
    with pytest.raises(se.EvidenceError):
        se.earthquake_evidence(tmp_path)


def _eq_stack_root(tmp_path: Path) -> Path:
    for rel in (se.EARTHQUAKE_SERVED, se.EARTHQUAKE_FINAL, se.EARTHQUAKE_ARTIFACT_RECORD, se.EARTHQUAKE_STACK,
                se.EARTHQUAKE_STACK_RECORD, se.EARTHQUAKE_E1):
        _copy(rel, tmp_path)
    for rel in (*se.EARTHQUAKE_STACK_REPORTS.values(), se.EARTHQUAKE_E4, se.EARTHQUAKE_STACK_S1):
        if (ROOT / rel).exists() and not (tmp_path / rel).exists():          # S2's report, decision, S1's file
            _copy(rel, tmp_path)
    return tmp_path


def _served_report_rel() -> str:
    """The served stack's evaluation report (section 10's format), by the candidate its provenance names."""
    meta = json.loads((ROOT / se.EARTHQUAKE_STACK).read_text(encoding="utf-8"))
    return se.EARTHQUAKE_STACK_REPORTS[(meta.get("provenance") or {}).get("candidate", "S1")]


@pytest.mark.skipif(not se.EARTHQUAKE_STACK or not (ROOT / se.EARTHQUAKE_STACK).exists(), reason="no earthquake stack served")
def test_the_earthquake_stack_is_shown_only_when_bound_to_its_evaluation_and_base(tmp_path, monkeypatch):
    root = _eq_stack_root(tmp_path)
    cand = se.earthquake_evidence(root)["candidate"]
    rel = _served_report_rel()
    rep = json.loads((root / rel).read_text(encoding="utf-8"))
    rep["coefficients_fitted_on_choose"][cand]["b"] += 1e-9                  # another fit
    _write(root / rel, rep)
    with pytest.raises(se.EvidenceError):
        se.earthquake_evidence(root)
    root = _eq_stack_root(tmp_path / "b")
    rec = json.loads((root / se.EARTHQUAKE_STACK_RECORD).read_text(encoding="utf-8"))
    rec["base_model_version"] = "eq_operational_C0_v1-000000000000"            # built on another C0
    _write(root / se.EARTHQUAKE_STACK_RECORD, rec)
    with pytest.raises(se.EvidenceError):
        se.earthquake_evidence(root)
    root = _eq_stack_root(tmp_path / "c")
    (root / se.EARTHQUAKE_STACK).unlink()                       # the named stack is missing: never described as C0
    with pytest.raises(se.EvidenceError, match="missing"):
        se.earthquake_evidence(root)
    monkeypatch.setattr(se, "EARTHQUAKE_STACK", None)                         # C0 served, by name: read once
    eq = se.earthquake_evidence(root)
    assert eq["candidate"] == "C0" and not eq["test"].get("second_read")


@pytest.mark.skipif(not (ROOT / se.EARTHQUAKE_STACK).exists(), reason="no earthquake stack served")
def test_the_stack_refuses_another_base_and_applies_its_formula_by_hand(tmp_path):
    import math

    import numpy as np

    from hazardpulse.earthquake import operational_forecast as of
    base = of.load_artifact(ROOT / se.EARTHQUAKE_SERVED)
    stack = of.load_stack(ROOT / se.EARTHQUAKE_STACK, base)
    p = np.full(of.N_CELLS, 0.01)
    p[0] = 1e-15                                                               # clipped, never log(0)
    got = of.apply_stack(stack, p)
    z = stack.a + stack.c * math.log(0.01 / 0.99) + stack.b * stack.g_log10[5]
    assert got[5] == pytest.approx(1 / (1 + math.exp(-z)), rel=1e-12) and np.all(np.isfinite(got))
    other = json.loads((ROOT / se.EARTHQUAKE_STACK).read_text(encoding="utf-8"))
    other["base_model_version"] = "eq_operational_C0_v1-000000000000"
    (tmp_path / "s.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(of.OperationalArtifactError):
        of.load_stack(tmp_path / "s.json", base)


def test_a_hurricane_final_naming_another_artifact_is_refused(tmp_path):
    _copy(se.HURRICANE_SERVED, tmp_path)
    final = json.loads((ROOT / se.HURRICANE_FINAL).read_text(encoding="utf-8"))
    final["artifact"]["model_version"] = "hurricane_ri_stack_v1-000000000000"
    _write(tmp_path / se.HURRICANE_FINAL, final)
    with pytest.raises(se.EvidenceError):
        se.hurricane_evidence(tmp_path)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def test_the_published_pages_are_a_fixed_point_of_the_results_files():
    """A results change that was not re-rendered fails here instead of shipping a stale number
    (python -c "from hazardpulse.verification import evidence_pages as ep; ep.render_pages(...)",
    or any scorer run via build_site_artifacts)."""
    assert ep.check_pages(ROOT / "dist") == []


def _pages(tmp: Path) -> Path:
    for rel in ep.BLOCKS:
        _copy(f"dist/{rel}", tmp)
    return tmp / "dist"


def test_a_changed_result_makes_the_page_stale_and_a_render_fixes_it(tmp_path):
    dist = _pages(tmp_path)
    ev = se.all_evidence()
    assert ep.check_pages(dist, ev=ev) == []
    moved = copy.deepcopy(ev)
    moved["hurricane"]["test"]["auc"] = 0.5
    assert "methods/index.html" in ep.check_pages(dist, ev=moved)
    ep.render_pages(dist, ev=moved)
    assert ep.check_pages(dist, ev=moved) == []
    assert "AUC 0.500" in (dist / "methods/index.html").read_text(encoding="utf-8")


def test_the_verification_workflow_re_renders_and_commits_the_evidence_pages(tmp_path, monkeypatch):
    """The verification workflow rewrites the prospective summaries that feed the evidence blocks and the
    track record, so its rebuild must re-render the site and its commit must carry every page -- else main
    fails its own fixed point between scorer runs (found 2026-10-03: a challenger's error budget appeared in
    the results but not on the page)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import build_site_artifacts as bsa
    calls = []
    pulse = tmp_path / "pulse.json"
    pulse.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bsa, "LIVE_PULSE_PATH", pulse)
    monkeypatch.setattr(bsa, "_build_verification_summary", lambda p: {"hazards": []})
    monkeypatch.setattr(bsa, "_render_site", lambda: calls.append("site"))
    bsa.build_verification_rollups()
    assert calls == ["site"]
    wf = (ROOT / ".github" / "workflows" / "verification-score.yml").read_text(encoding="utf-8")
    detect = wf.split("Check for changes", 1)[1].split("Commit and push", 1)[0]
    commit = wf.split("Commit and push", 1)[1]
    # every page (git's '*' spans directories), so each evidence page is covered
    assert "'dist/*.html'" in detect and "'dist/*.html'" in commit
    for page in ep.BLOCKS:
        assert page.endswith(".html"), page


def test_a_page_without_its_block_or_with_it_twice_is_an_error():
    with pytest.raises(ep.PageBlockError):
        ep.apply_block("<p>no markers</p>", "methods-simple", "x")
    twice = "<!-- hp-evidence:a -->x<!-- /hp-evidence:a -->" * 2
    with pytest.raises(ep.PageBlockError):
        ep.apply_block(twice, "a", "y")
    assert ep.apply_block("<!-- hp-evidence:a -->old<!-- /hp-evidence:a -->", "a", "new") == \
        "<!-- hp-evidence:a -->\nnew\n<!-- /hp-evidence:a -->"


def test_unbound_evidence_renders_words_not_numbers():
    none = {"tornado": None, "earthquake": None, "hurricane": None}
    for rel, blocks in ep.BLOCKS.items():
        for name, fn in blocks.items():
            out = fn(none)
            assert "AUC 0." not in out and "nats" not in out, (rel, name)
    assert "No final test is bound" in ep.methods_tornado(none)


def test_the_tornado_page_states_the_nws_result_with_and_without_the_warning_input(tmp_path):
    ev = {"tornado": se.tornado_evidence(_tornado_repo(tmp_path)), "earthquake": None, "hurricane": None}
    body = ep.tornado_body(ev)
    assert "using the warning state as an input" in body and "without that input it catches 22.0%" in body
    assert "tie-broken" in body and "0.942" in body
    card = ep.methods_tornado(ev)
    assert "28 ProbSevere storm attributes, 2 NWS tornado-warning state" in card
