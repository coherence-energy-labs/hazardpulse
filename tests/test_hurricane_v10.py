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
    assert alphas == pytest.approx([0.025, 0.0125, 0.00625, 0.003125, 0.0015625])
    assert all(b == pytest.approx(a / 2) for a, b in zip(alphas, alphas[1:])) and sum(alphas) < 0.05
    assert p.ENTRANTS["v10_2"]["key"] == "ri_v10_2_shadow" and p.ENTRANTS["v10_4"]["key"] == "ri_v10_4_shadow"
    assert ("v10_4", "v10_3") in p.CHALLENGES                  # against the model it was selected against
    # challenger minus champion on shared cycles, by hand: one cycle, dV 32 kt
    champ = [{"storm_id": "EP01", "cycle": "c1", "dv": 32.0, "p_k": {"25": 0.5, "30": 0.5, "35": 0.5, "40": 0.5}}]
    chall = [{"storm_id": "EP01", "cycle": "c1", "dv": 32.0, "p_k": {"25": 0.9, "30": 0.8, "35": 0.2, "40": 0.1}},
             {"storm_id": "EP02", "cycle": "c9", "dv": 0.0, "p_k": {"25": 0.0, "30": 0.0, "35": 0.0, "40": 0.0}}]
    v = p.versus(chall, champ)
    # challenger 0.01+0.04+0.04+0.01 = 0.10; champion 4 x 0.25 = 1.00; only the shared cycle counts
    assert v["n"] == 1 and v["d_brier4"] == pytest.approx(0.10 - 1.00)
    assert p.versus(chall, [])["n"] == 0


def test_the_site_shows_our_model_beside_the_published_number_with_its_bound_evidence():
    from hazardpulse.site.pages import hurricane as page
    from hazardpulse.verification import evidence_pages as ep
    from hazardpulse.verification import served_evidence as se
    storm = {"storm_id": "EP012026", "storm_name": "TEST", "basin": "EP", "lat": 15.0, "lon": -110.0,
             "vmax_kt": 80.0, "ri_probability": 0.2, "ri_source": "noaa_aid_stack", "ri_source_label": "NOAA DTOPS"}
    card = page._storm_card({**storm, "ri_v10_shadow": {"status": "ok", "probability": 0.6123, "gate_ok": True}})
    assert "61.2%" in card and "Our experimental model (v10.1)" in card and "equals NOAA" not in card
    fallback = page._storm_card({**storm, "ri_v10_shadow": {"status": "ok", "probability": 0.45, "gate_ok": False}})
    assert "45.0%" in fallback and "guidance was missing" in fallback
    assert "experimental" not in page._storm_card(storm)                # no model output, no row
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


# v10.4 (amendments 8-9): V8 + the coherence equation's balanced response to the IR heating
# ---------------------------------------------------------------------------

v10_4_present = pytest.mark.skipif(not ri_v10.V10_4_PATH.exists(), reason="v10.4 artifact not built")


@pytest.fixture(scope="module")
def v10_4():
    art, version = ri_v10.load(ri_v10.V10_4_PATH)
    return {"artifact": art, "model_version": version}


def _fixture_crop(sid, cyc, tag):
    p = FIX / "ir" / f"{sid}_{cyc:%Y%m%d%H}_{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"counts": z["counts"], "lat": z["lat"], "lon": z["lon"], "centre": tuple(z["centre"])}


def _fixture_h8(fs):
    """H8 from the fixture storm's real crops (the ones training used), by the live path's own vortex read."""
    from hazardpulse.hurricane import balanced_response as br
    from hazardpulse.hurricane import ri_v9_features as v9fx

    def read(sid, cyc, records):
        f = v9fx.adeck_features(v9fx.cycle_table(records, cyc), sid[:2])
        feats = br.features(_fixture_crop(sid, cyc, "p2"), _fixture_crop(sid, cyc, "m4"), float(f["v0"]),
                            fs.carq_rmw_from_records(records, cyc), float(f["abs_lat"]))
        return feats, "ok"
    return read


@v10_4_present
def test_the_balanced_response_challenger_reproduces_the_labs_curve_and_inputs(fs, v10, v10_3, v10_4):
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    assert ri_v10.needs_h8(v10_4["artifact"]) and ri_v10.needs_ir(v10_4["artifact"])
    assert not ri_v10.needs_h8(v10_3["artifact"])
    for c in exp["cases"]:
        out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                                  adeck_fetcher=lambda sid: _records(),
                                  challengers={"ri_v10_3_shadow": v10_3, "ri_v10_4_shadow": v10_4},
                                  ir_fetcher=_fixture_ir, h8_fetcher=_fixture_h8(fs))
        ch = out["ri_v10_4_shadow"]
        assert ch["source"] == "v10.4" and ch["ir"] == "ok" and ch["h8"] == "ok"
        for n, want in c["h8"].items():                     # the live vortex read gives training's H8 exactly
            assert ch["h8_inputs"][n] == pytest.approx(want, rel=1e-12, abs=1e-12)
        for k, want in c["v10_4"].items():
            assert ch["probabilities"][k] == pytest.approx(want, abs=5e-5)           # live == lab
        assert "h8" not in out["ri_v10_3_shadow"] and "h8_inputs" not in out["ri_v10_3_shadow"]
        for k, want in c["v10_3"].items():                                              # v10.3 untouched by H8
            assert out["ri_v10_3_shadow"]["probabilities"][k] == pytest.approx(want, abs=5e-5)


@v10_4_present
def test_an_h8_failure_touches_only_the_h8_model(fs, v10, v10_3, v10_4):
    c = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"][0]

    def broken(*a):
        raise ValueError("solver refused")
    out = fs.shadow_forecasts(_case(c["dtg"]), None, v10, ships_raw_fetcher=_ships(c["ships_text"]),
                              adeck_fetcher=lambda sid: _records(),
                              challengers={"ri_v10_3_shadow": v10_3, "ri_v10_4_shadow": v10_4},
                              ir_fetcher=_fixture_ir, h8_fetcher=broken)
    ch = out["ri_v10_4_shadow"]
    assert ch["status"] == "ok" and ch["h8"].startswith("error: ValueError")
    assert all(v is None for v in ch["h8_inputs"].values())                            # NaN inputs, as in training
    assert ch["ir"] == "ok" and ch["probability"] is not None
    for k, want in c["v10_3"].items():
        assert out["ri_v10_3_shadow"]["probabilities"][k] == pytest.approx(want, abs=5e-5)


def test_the_live_rmw_is_the_one_training_read(fs):
    """Training read the RMW from the deck's text (``carq_rmw_nm``), the live path from its parsed records:
    the same number on every cycle of the fixture deck, NaN where ATCF writes 0."""
    from hazardpulse.hurricane import balanced_response as br
    text = gzip.decompress((FIX / "aal012026.dat.gz").read_bytes()).decode("utf-8", "replace")
    recs = atcf.parse_atcf_deck(text)
    cycles = sorted({r.cycle for r in recs if r.model == "CARQ" and r.tau_hours == 0})
    assert len(cycles) >= 3
    for cyc in cycles:
        a, b = br.carq_rmw_nm(text, cyc.strftime("%Y%m%d%H")), fs.carq_rmw_from_records(recs, cyc)
        assert (math.isnan(a) and math.isnan(b)) or a == b, cyc
    assert any(math.isfinite(fs.carq_rmw_from_records(recs, cyc)) for cyc in cycles)
    line = "AL, 09, 2026100712, 01, CARQ,   0, 222N,  939W,  40, 1002, TS,  34, NEQ, 60, 50, 0, 40, 1008, 150, {},"
    zero, twenty = (atcf.parse_atcf_deck(line.format(v))[0] for v in ("  0", " 20"))
    assert zero.rmw_nm is None and twenty.rmw_nm == 20.0
    cyc = dt.datetime(2026, 10, 7, 12)
    assert math.isnan(fs.carq_rmw_from_records([zero], cyc)) and fs.carq_rmw_from_records([zero, twenty], cyc) == 20.0


def test_live_h8_crops_like_training_from_the_runs_cached_images(fs, monkeypatch):
    """live_h8_read cuts the same two crops live_ir_read does, from the same cached downloads, and feeds the
    balanced response the training vortex (v0, abs_lat, the CARQ RMW)."""
    from hazardpulse.hurricane import balanced_response as br, ir_source
    from hazardpulse.hurricane import ri_v9_features as v9fx
    lat = np.linspace(72.7, -72.7, 2001)
    lon = np.linspace(-180.0, 179.9, 5000)
    rng = np.random.default_rng(2)
    counts = rng.integers(60, 230, size=(lat.size, lon.size)).astype(np.uint8)
    calls = []

    def fake_fetch(hour):
        calls.append(hour)
        return "key", counts, lat, lon
    monkeypatch.setattr(ir_source, "fetch_image", fake_fetch)
    fs._IR_IMAGES.clear()
    recs = _records()
    cyc = dt.datetime(2026, 6, 16, 12)
    fs.live_ir_read("AL012026", cyc, recs)
    got, status = fs.live_h8_read("AL012026", cyc, recs)
    assert len(calls) == 2                                            # H8 reused the IR read's two images
    cen = ir_source.centres(recs, cyc)
    f = v9fx.adeck_features(v9fx.cycle_table(recs, cyc), "AL")
    want = br.features(ir_source.crop(counts, lat, lon, cen["p2"][1]), ir_source.crop(counts, lat, lon, cen["m4"][1]),
                       float(f["v0"]), fs.carq_rmw_from_records(recs, cyc), float(f["abs_lat"]))
    assert status == "ok" and got == want and all(math.isfinite(v) for v in got.values())
    no_rmw = [atcf.ATCFRecord(**{**r.__dict__, "rmw_nm": None}) for r in recs]
    got, status = fs.live_h8_read("AL012026", cyc, no_rmw)
    assert all(math.isnan(v) for v in got.values()) and status.startswith("missing: 0 of 4") and "no CARQ RMW" in status
    fs._IR_IMAGES.clear()


@v10_4_present
def test_the_h8_model_is_not_scored_without_its_ir_reader(fs, monkeypatch):
    assert "ri_v10_4_shadow" in fs.load_challengers()
    monkeypatch.setattr(fs, "ir_reader_available", lambda: False)
    assert "ri_v10_4_shadow" not in fs.load_challengers()


@v10_4_present
def test_the_h8_challenger_is_bound_to_v10_3(tmp_path):
    from hazardpulse.verification import served_evidence as se
    root = _root_copy(tmp_path)
    for p in (ri_v10.V10_2_PATH, ri_v10.V10_3_PATH, ri_v10.V10_4_PATH):
        (root / "results/models" / p.name).write_bytes(p.read_bytes())
    chs = se.ours_hurricane(root)["challengers"]
    assert [c["label"] for c in chs] == ["v10.2", "v10.3", "v10.4"] and chs[2]["against"] == "v10.3"
    assert chs[2]["season_2026"]["log_loss"] is not None
    art = json.loads(ri_v10.V10_4_PATH.read_text(encoding="utf-8"))
    art["provenance"]["dev_2022_2025"]["champion_log_loss"] = chs[0]["dev"]["log_loss"]      # v10.2's, not v10.3's
    (root / "results/models" / ri_v10.V10_4_PATH.name).write_bytes(ri_v10.canonical_bytes(art))
    with pytest.raises(se.EvidenceError):
        se.ours_hurricane(root)
