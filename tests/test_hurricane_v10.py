"""v10.1 RI shadow: the live path reproduces the lab's exceedance curve; the gate serves NOAA's own values."""
from __future__ import annotations

import datetime as dt
import gzip
import math
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane import atcf, ri_v10

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "hurricane_v9"

pytestmark = pytest.mark.skipif(not ri_v10.MODEL_PATH.exists(), reason="v10 artifact not built")


@pytest.fixture(scope="module")
def fs():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_and_score_v10_test", ROOT / "scripts" / "fetch_and_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v10():
    art, version = ri_v10.load()
    return {"artifact": art, "model_version": version}


def _records():
    return atcf.parse_atcf_deck(gzip.decompress((FIX / "aal012026.dat.gz").read_bytes()).decode("utf-8", "replace"))


def _case(dtg: str) -> dict:
    return {"storm_id": "AL012026", "issue_time": dt.datetime.strptime(dtg, "%Y%m%d%H").isoformat(),
            "analysis_model": "CARQ"}


def _ships(name):
    return lambda sid, cyc: ((FIX / name).read_text(encoding="utf-8"), name)


def test_the_live_shadow_reproduces_the_labs_exceedance_curve(fs, v10):
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    for c in exp["cases"]:
        out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                                  adeck_fetcher=lambda sid: _records())["ri_v10_shadow"]
        assert out["status"] == "ok" and out["gate_ok"] and out["source"] == "v10.1", out
        for k, want in c["v10"].items():
            assert out["probabilities"][k] == pytest.approx(want, abs=5e-5)          # live == lab
        curve = [out["model_probabilities"][str(k)] for k in v10["artifact"]["thresholds_kt"]]
        assert all(a >= b for a, b in zip(curve, curve[1:]))                           # coherent in k
        assert out["noaa_24h"]["30"] == pytest.approx(c["a"])


def test_without_the_early_guidance_each_threshold_is_noaas_own(fs, v10):
    c = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"][0]
    stripped = [r for r in _records() if r.model not in ri_v10.GATE_AIDS]
    out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                              adeck_fetcher=lambda sid: stripped)["ri_v10_shadow"]
    assert out["gate_ok"] is False and out["source"].startswith("DTOPS")
    for k in ("25", "30", "35", "40"):
        assert out["probabilities"][k] == out["noaa_24h"][k]
    assert out["probabilities"]["15"] is None and out["probabilities"]["45"] is None   # NOAA has no 24-h value


@pytest.fixture(scope="module")
def v10_2():
    art, version = ri_v10.load(ri_v10.V10_2_PATH)
    return {"artifact": art, "model_version": version}


def test_every_shadow_comes_from_one_read(fs, v10, v10_2):
    import hazardpulse.hurricane.ri_v9 as ri_v9
    payload, version = ri_v9.load()
    calls = {"ships": 0, "adeck": 0}
    c = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"][0]

    def ships(sid, cyc):
        calls["ships"] += 1
        return _ships(c["ships_text"])(sid, cyc)

    def adeck(sid):
        calls["adeck"] += 1
        return _records()
    out = fs.shadow_forecasts(_case(c["dtg"]), {"payload": payload, "model_version": version}, v10, ships, adeck,
                              challengers={"ri_v10_2_shadow": v10_2})
    assert set(out) == {"ri_v9_shadow", "ri_v10_shadow", "ri_v10_2_shadow"} and calls == {"ships": 1, "adeck": 1}


def test_the_challenger_reproduces_the_labs_curve_under_its_own_label(fs, v10, v10_2):
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    for c in exp["cases"]:
        out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                                  adeck_fetcher=lambda sid: _records(), challengers={"ri_v10_2_shadow": v10_2})
        ch, champ = out["ri_v10_2_shadow"], out["ri_v10_shadow"]
        assert ch["status"] == "ok" and ch["source"] == "v10.2" and champ["source"] == "v10.1"
        assert ch["model_version"] == v10_2["model_version"] != champ["model_version"]
        for k, want in c["v10_2"].items():
            assert ch["probabilities"][k] == pytest.approx(want, abs=5e-5)          # live == lab
    stripped = [r for r in _records() if r.model not in ri_v10.GATE_AIDS]
    c = exp["cases"][0]
    gated = fs.shadow_forecasts(_case(c["dtg"]), None, None, ships_raw_fetcher=_ships(c["ships_text"]),
                                adeck_fetcher=lambda sid: stripped,
                                challengers={"ri_v10_2_shadow": v10_2})["ri_v10_2_shadow"]
    assert gated["source"] == "DTOPS (v10.2 gate: early guidance missing)"


def test_the_challenger_never_lowers_the_odds_when_the_guidance_rises(v10, v10_2):
    """The monotone constraint is a property of the frozen artifact, not of the training script:
    sweep each constrained input on real cycles and the 30-kt probability never falls."""
    from hazardpulse.hurricane import ri_v9_features as fx
    names = v10_2["artifact"]["feature_names"]
    up = v10_2["artifact"]["provenance"]["monotone_up"]
    assert {"dv24_HCCA", "ofcl_dv24", "ri_DTOP_30_24"} <= set(up)
    recs = _records()
    base = []
    for c in json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"]:
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        base.append(fx.vector(fx.adeck_features(fx.cycle_table(recs, cyc), "AL"), names))
    grid = np.linspace(-40.0, 80.0, 61)
    falls = {}
    for label, model in (("v10.2", v10_2), ("v10.1", v10)):
        worst = 0.0
        for x0 in base:
            for n in ("dv24_HCCA", "dv24_IVCN", "ofcl_dv24", "dv24_regional_max"):
                X = np.repeat(x0[None, :], len(grid), axis=0)
                X[:, names.index(n)] = grid
                p = ri_v10.predict_matrix(model["artifact"], X, 30)
                worst = max(worst, float(np.max(p[:-1] - p[1:])))
        falls[label] = worst
    assert falls["v10.2"] <= 1e-12, falls
    assert falls["v10.1"] > 1e-6, falls          # the sweep can see a fall: the unconstrained model has one


def test_each_entrant_spends_half_the_previous_error_budget():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v9_prosp_budget", ROOT / "scripts" / "score_hurricane_v9_prospective.py")
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    alphas = [1 - e["level"] for e in p.ENTRANTS.values()]
    assert alphas == pytest.approx([0.025, 0.0125, 0.00625, 0.003125])
    assert all(b == pytest.approx(a / 2) for a, b in zip(alphas, alphas[1:])) and sum(alphas) < 0.05
    assert p.ENTRANTS["v10_2"]["key"] == "ri_v10_2_shadow"
    # challenger minus champion on shared cycles, by hand: one cycle, dV 32 kt
    champ = [{"storm_id": "EP01", "cycle": "c1", "dv": 32.0, "p_k": {"25": 0.5, "30": 0.5, "35": 0.5, "40": 0.5}}]
    chall = [{"storm_id": "EP01", "cycle": "c1", "dv": 32.0, "p_k": {"25": 0.9, "30": 0.8, "35": 0.2, "40": 0.1}},
             {"storm_id": "EP02", "cycle": "c9", "dv": 0.0, "p_k": {"25": 0.0, "30": 0.0, "35": 0.0, "40": 0.0}}]
    v = p.versus(chall, champ)
    # challenger 0.01+0.04+0.04+0.01 = 0.10; champion 4 x 0.25 = 1.00; only the shared cycle counts
    assert v["n"] == 1 and v["d_brier4"] == pytest.approx(0.10 - 1.00)
    assert p.versus(chall, [])["n"] == 0


def test_the_site_shows_our_model_beside_the_published_number_with_its_bound_evidence():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_site_artifacts as b
    from hazardpulse.verification import evidence_pages as ep
    from hazardpulse.verification import served_evidence as se
    assert b._ours_cell({"ri_v10_shadow": {"status": "ok", "probability": 0.6123, "gate_ok": True}}) == "61.2%"
    assert "guidance missing" in b._ours_cell({"ri_v10_shadow": {"status": "ok", "probability": 0.45, "gate_ok": False}})
    assert "--" in b._ours_cell({})
    ours = se.hurricane_evidence()["ours"]
    art, version = ri_v10.load()
    assert ours["model_version"] == version
    assert ours["dev"]["log_loss"] == art["provenance"]["dev_2022_2025"]["log_loss"]      # never typed
    card = ep.methods_hurricane({"hurricane": se.hurricane_evidence()})
    assert version in card and "not instead of it" in card and f"{ours['dev']['log_loss']:.3f}" in card


def _vs_all_mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v10_vs_all_t", ROOT / "scripts" / "hurricane_ri_v10_vs_all.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_probability_meets_a_yes_no_call_at_the_calls_own_false_alarm_rate():
    import numpy as np
    m = _vs_all_mod()
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    call = np.array([1, 0, 0, 1, 0, 0, 0])                 # 1 of 3 events, 1 of 4 non-events
    p = np.array([0.9, 0.8, 0.1, 0.95, 0.5, 0.2, 0.1])
    pod_call, pod_ours, pofd = m.pod_at_matched_pofd(y, p, call)
    # one false alarm allowed: threshold above 0.5 -> ours flags 0.95 (false) and 0.9, 0.8 (hits)
    assert (pod_call, pofd) == (1 / 3, 0.25) and pod_ours == 2 / 3
    # a tie at the threshold is never resolved in our favour: equal scores everywhere catch nothing
    assert m.pod_at_matched_pofd(y, np.full(7, 0.5), call)[1] == 0.0


def _root_copy(tmp_path, mutate=None):
    for rel in ("results/models/hurricane_ri_v10.json", "results/calibration/hurricane_ri_v10_vs_all.json"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    if mutate:
        p = tmp_path / "results/calibration/hurricane_ri_v10_vs_all.json"
        rep = json.loads(p.read_text(encoding="utf-8"))
        mutate(rep)
        p.write_text(json.dumps(rep), encoding="utf-8")
    return tmp_path


def test_the_every_aid_comparison_is_bound_to_the_served_artifact(tmp_path):
    from hazardpulse.verification import evidence_pages as ep
    from hazardpulse.verification import served_evidence as se
    rep = json.loads((ROOT / "results/calibration/hurricane_ri_v10_vs_all.json").read_text(encoding="utf-8"))
    va = se.ours_hurricane(_root_copy(tmp_path))["vs_all"]
    assert {a["tech"] for a in va["aids"]} == {"RIOD", "RIOL", "RIOB", "RIOC", "DTOP"}
    for a in va["aids"]:
        assert a["aid_log_loss"] == rep["aids"][a["tech"]]["aid_30"]["log_loss"]           # never typed
    text = " ".join(v for _, v in ep._ours_vs_all_lines(va))
    for tech, r in rep["aids"].items():
        assert f"{r['label']} {r['aid_30']['log_loss']:.3f}" in text
    with pytest.raises(se.EvidenceError):                     # a file from another model
        se.ours_hurricane(_root_copy(tmp_path / "a", lambda r: r.update(control_V2_log_loss=0.15)))
    with pytest.raises(se.EvidenceError):                     # a file computed without the served gate
        se.ours_hurricane(_root_copy(tmp_path / "b", lambda r: r.update(model="V2 ungated")))


def test_the_challenger_line_is_bound_to_the_served_champion(tmp_path):
    from hazardpulse.verification import evidence_pages as ep
    from hazardpulse.verification import served_evidence as se
    root = _root_copy(tmp_path)
    (root / "results/models").joinpath(ri_v10.V10_2_PATH.name).write_bytes(ri_v10.V10_2_PATH.read_bytes())
    ch = se.ours_hurricane(root)["challenger"]
    prov = json.loads(ri_v10.V10_2_PATH.read_text(encoding="utf-8"))["provenance"]["dev_2022_2025"]
    assert ch["dev"]["log_loss"] == prov["log_loss"] and ch["label"] == "v10.2"           # never typed
    (line,) = ep._ours_challenger_lines(ch)
    assert f"{prov['log_loss']:.4f} vs {prov['champion_log_loss']:.4f}" in line[1]
    assert ("interval still includes zero" in line[1]) == (prov["d_log_loss_vs_champion_ci"][1] >= 0)
    art = json.loads(ri_v10.V10_2_PATH.read_text(encoding="utf-8"))
    art["provenance"]["dev_2022_2025"]["champion_log_loss"] = 0.2                   # selected against another model
    (root / "results/models" / ri_v10.V10_2_PATH.name).write_bytes(ri_v10.canonical_bytes(art))
    with pytest.raises(se.EvidenceError):
        se.ours_hurricane(root)
    (root / "results/models" / ri_v10.V10_2_PATH.name).unlink()
    assert se.ours_hurricane(root)["challenger"] is None


def test_a_comparison_whose_interval_straddles_zero_is_reported_as_a_tie():
    from hazardpulse.verification import evidence_pages as ep
    aid = {"label": "X", "ours_log_loss": 0.14, "aid_log_loss": 0.15}
    va = {"seasons": [2022, 2025],
          "aids": [dict(aid, tech="A", label="Alpha", d_log_loss_ci=[-0.02, -0.01]),
                   dict(aid, tech="B", label="Beta", d_log_loss_ci=[-0.02, 0.001])],
          "calls": [{"label": "Gamma", "d_pod": 0.1, "d_pod_ci": [0.02, 0.2]},
                    {"label": "Delta", "d_pod": -0.03, "d_pod_ci": [-0.1, 0.05]},
                    {"label": "Eps", "d_pod": -0.2, "d_pod_ci": [-0.3, -0.1]}]}
    (_, probs), (_, calls) = ep._ours_vs_all_lines(va)
    assert "lower at 95% than Alpha" in probs and "every one" not in probs and "Beta" in probs
    assert "more RI events than Gamma" in calls and "fewer than Eps" in calls and "within noise of Delta" in calls
    assert ep._ours_vs_all_lines(None) == []


def test_an_artifact_of_another_schema_is_refused(tmp_path):
    art = json.loads(ri_v10.MODEL_PATH.read_text(encoding="utf-8"))
    art["schema"] = "something/else"
    p = tmp_path / "a.json"
    p.write_bytes(ri_v10.canonical_bytes(art))
    with pytest.raises(ValueError):
        ri_v10.load(p)


# ---------------------------------------------------------------------------
# v10.3 (amendments 5-6): V5 + IR structure from GMGSI
# ---------------------------------------------------------------------------

v10_3_present = pytest.mark.skipif(not ri_v10.V10_3_PATH.exists(), reason="v10.3 artifact not built")


@pytest.fixture(scope="module")
def v10_3():
    art, version = ri_v10.load(ri_v10.V10_3_PATH)
    return {"artifact": art, "model_version": version}


def _fixture_ir(sid, cyc, records):
    """The IR features from the fixture storm's real crops (the ones training used)."""
    from hazardpulse.hurricane import ir_features

    def crop(tag):
        p = FIX / "ir" / f"{sid}_{cyc:%Y%m%d%H}_{tag}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return {"counts": z["counts"], "lat": z["lat"], "lon": z["lon"], "centre": tuple(z["centre"])}
    return ir_features.features(crop("p2"), crop("m4"))


@v10_3_present
def test_the_ir_challenger_reproduces_the_labs_curve_from_the_same_crops(fs, v10, v10_3):
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    assert ri_v10.needs_ir(v10_3["artifact"]) and not ri_v10.needs_ir(v10["artifact"])
    for c in exp["cases"]:
        out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                                  adeck_fetcher=lambda sid: _records(), challengers={"ri_v10_3_shadow": v10_3},
                                  ir_fetcher=_fixture_ir)
        ch = out["ri_v10_3_shadow"]
        assert ch["source"] == "v10.3" and ch["ir"] == "ok"
        assert ch["ir_inputs"]["ir_mean_50_200"] is not None
        for k, want in c["v10_3"].items():
            assert ch["probabilities"][k] == pytest.approx(want, abs=5e-5)           # live == lab
        assert "ir" not in out["ri_v10_shadow"]                                          # v10.1 never reads IR


@v10_3_present
def test_an_ir_failure_touches_only_the_ir_model(fs, v10, v10_2, v10_3):
    c = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"][0]

    def broken(*a):
        raise OSError("bucket down")
    out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                              adeck_fetcher=lambda sid: _records(),
                              challengers={"ri_v10_2_shadow": v10_2, "ri_v10_3_shadow": v10_3}, ir_fetcher=broken)
    assert out["ri_v10_3_shadow"]["status"] == "ok" and out["ri_v10_3_shadow"]["ir"].startswith("error: OSError")
    assert all(v is None for v in out["ri_v10_3_shadow"]["ir_inputs"].values())        # NaN inputs, as in training
    assert out["ri_v10_shadow"]["probability"] is not None and out["ri_v10_2_shadow"]["probability"] is not None


def test_live_ir_crops_like_training_and_downloads_each_hour_once(fs, monkeypatch):
    """The live path crops a full image with ir_source exactly as the collector did, and two storms
    at the same cycle share the two downloads."""
    from hazardpulse.hurricane import ir_features, ir_source
    lat = np.linspace(72.7, -72.7, 2001)
    lon = np.linspace(-180.0, 179.9, 5000)
    rng = np.random.default_rng(1)
    counts = rng.integers(60, 230, size=(lat.size, lon.size)).astype(np.uint8)
    calls = []

    def fake_fetch(hour):
        calls.append(hour)
        return "key", counts, lat, lon
    monkeypatch.setattr(ir_source, "fetch_image", fake_fetch)
    fs._IR_IMAGES.clear()
    recs = _records()
    cyc = dt.datetime(2026, 6, 16, 12)
    got = fs.live_ir_features("AL012026", cyc, recs)
    cen = ir_source.centres(recs, cyc)
    want = ir_features.features(ir_source.crop(counts, lat, lon, cen["p2"][1]), ir_source.crop(counts, lat, lon, cen["m4"][1]))
    assert got.keys() == want.keys() and all(
        (math.isnan(got[k]) and math.isnan(want[k])) or got[k] == want[k] for k in got)
    fs.live_ir_features("AL012026", cyc, recs)                                           # same hours: cached
    assert sorted(calls) == sorted({cen["p2"][0], cen["m4"][0]}) and len(calls) == 2
    fs._IR_IMAGES.clear()


@v10_3_present
def test_an_ir_model_is_not_scored_without_its_reader_and_the_reader_is_a_dependency(fs, monkeypatch):
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"h5py' in pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "ri_v10_3_shadow" in fs.load_challengers()                     # this environment has it
    monkeypatch.setattr(fs, "ir_reader_available", lambda: False)
    got = fs.load_challengers()
    assert "ri_v10_3_shadow" not in got and "ri_v10_2_shadow" in got     # never v10.3 without IR


def test_the_loader_accepts_only_the_two_registered_input_sets(tmp_path):
    art = json.loads(ri_v10.MODEL_PATH.read_text(encoding="utf-8"))
    sets = ri_v10.allowed_feature_sets()
    assert art["feature_names"] == sets["ONH"]
    bad = dict(art, feature_names=sets["ONH+IR"][::-1])                                    # reordered inputs
    p = tmp_path / "bad.json"
    p.write_bytes(ri_v10.canonical_bytes(bad))
    with pytest.raises(ValueError):
        ri_v10.load(p)


@v10_3_present
def test_the_ir_challenger_is_bound_to_the_model_it_was_selected_against(tmp_path):
    from hazardpulse.verification import served_evidence as se
    root = _root_copy(tmp_path)
    for p in (ri_v10.V10_2_PATH, ri_v10.V10_3_PATH):
        (root / "results/models" / p.name).write_bytes(p.read_bytes())
    chs = se.ours_hurricane(root)["challengers"]
    assert [c["label"] for c in chs] == ["v10.2", "v10.3"] and chs[1]["against"] == "v10.2"
    art = json.loads(ri_v10.V10_3_PATH.read_text(encoding="utf-8"))
    art["provenance"]["dev_2022_2025"]["champion_log_loss"] = chs[0]["dev"]["champion_log_loss"]   # v10.1's, not v10.2's
    (root / "results/models" / ri_v10.V10_3_PATH.name).write_bytes(ri_v10.canonical_bytes(art))
    with pytest.raises(se.EvidenceError):
        se.ours_hurricane(root)
