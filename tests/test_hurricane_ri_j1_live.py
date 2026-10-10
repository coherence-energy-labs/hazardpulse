"""J1's live path (amendment 13b): the artifact reproduces the exported row, the live scorer builds EXACTLY the
training row from a live case, the IR centre comes from the storm's own fixes, the shadow never touches the
published number, and the record audit recomputes it."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane import ir_features, ir_source, ri_j1
from hazardpulse.hurricane.atcf import ATCFRecord

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "hurricane_ri_j1_live_row.json").read_text(encoding="utf-8"))


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(f"{name}_t", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_artifact_reproduces_the_exported_row_and_names_its_identity():
    art, version = ri_j1.load()
    assert version == FIXTURE["model_version"] and art["label"] == "J1"
    assert ri_j1.recompute(art, FIXTURE["inputs"]) == pytest.approx(FIXTURE["probability"], abs=1e-12)
    assert art["feature_names"][0] == "v82_logit" and set(ri_j1.BASIN_FLAGS) <= set(art["feature_names"])
    assert art["v82_dependency"]["model_version"] == "hurricane_ri_v8_2"


def test_basin_flags_read_training_and_live_codes_alike():
    assert ri_j1.basin_flags("NI") == ri_j1.basin_flags("IO") == {"is_wp": 0.0, "is_ni": 1.0, "is_sh": 0.0, "is_nhc": 0.0}
    assert ri_j1.basin_flags("SP") == ri_j1.basin_flags("SI") == ri_j1.basin_flags("SH")
    assert ri_j1.basin_flags("WP")["is_wp"] == 1.0 and ri_j1.basin_flags("AL")["is_nhc"] == 1.0


def _live_case_from_fixture():
    """A live case carrying the fixture row's v8.2 inputs under the live scorer's keys, with the SH code the live
    feed uses for a South Pacific storm."""
    art, _ = ri_j1.load()
    skip = {"v82_logit", *ri_j1.BASIN_FLAGS, *ir_features.IR_NAMES}
    case = {n: v for n, v in FIXTURE["inputs"].items() if n not in skip}
    case.update({"basin": "SH", "storm_id": "SH252026", "issue_time": "2026-03-17T06:00:00"})
    ir = {n: FIXTURE["inputs"][n] for n in ir_features.IR_NAMES}
    ens = 1.0 / (1.0 + math.exp(-FIXTURE["inputs"]["v82_logit"]))
    return art, case, ens, ir


def test_the_live_scorer_builds_exactly_the_training_row():
    art, case, ens, ir = _live_case_from_fixture()
    inputs = ri_j1.live_inputs(art, case, ens, ir)
    assert set(inputs) == set(art["feature_names"])
    for n, want in FIXTURE["inputs"].items():
        assert inputs[n] == pytest.approx(want, abs=1e-12), n
    assert ri_j1.recompute(art, inputs) == pytest.approx(FIXTURE["probability"], abs=1e-12)
    # a missing IR image stays missing (the trees' missing branch, as in training), never imputed
    gap = ri_j1.live_inputs(art, case, ens, {n: float("nan") for n in ir_features.IR_NAMES})
    assert all(gap[n] is None for n in ir_features.IR_NAMES)


def _rec(model, cycle, lat, lon):
    return ATCFRecord(basin="SH", storm_number=25, cycle=cycle, tau_hours=0, model=model, lat=lat, lon=lon,
                      vmax_kt=50.0, mslp_hpa=990.0)


def test_fix_positions_prefer_the_best_track_then_the_warning_and_carq_is_unchanged():
    t = dt.datetime(2026, 3, 17, 6)
    recs = [_rec("JTWC", t, -12.9, 156.0), _rec("BEST", t, -12.6, 156.2), _rec("JTWC", t - dt.timedelta(hours=6), -12.0, 156.9),
            _rec("CARQ", t, -13.5, 155.0)]
    p0, pm6 = ir_source.fix_positions(recs, t, ri_j1.FIX_MODELS)
    assert p0 == (-12.6, 156.2) and pm6 == (-12.0, 156.9)          # BEST at t; only the warning at t - 6 h
    assert ir_source.carq_positions(recs, t) == ((-13.5, 155.0), None)
    cen = ri_j1.ir_centres({"t": list(p0), "t_minus_6h": list(pm6)}, t)
    want = ir_source.centres(recs, t, ri_j1.FIX_MODELS)
    assert cen == want and set(cen) == set(ir_source.OFFSETS)
    assert ri_j1.ir_centres({"t": None}, t) is None


def test_the_shadow_is_recorded_beside_v82_never_published_and_the_audit_recomputes_it(monkeypatch):
    fs = _script("fetch_and_score")
    audit = _script("audit_hurricane_records")
    art, case, ens, ir = _live_case_from_fixture()
    case["fix_positions"] = {"models": list(ri_j1.FIX_MODELS), "t": [-12.6, 156.2], "t_minus_6h": [-12.4, 156.4]}
    lat = np.arange(-20.0, -5.0, 0.072)
    lon = np.arange(150.0, 163.0, 0.072)
    counts = np.full((lat.size, lon.size), 200, np.uint8)
    monkeypatch.setattr(fs, "_IR_IMAGES", {})
    monkeypatch.setattr(ir_source, "fetch_image", lambda hour: ("k", counts, lat, lon))
    j1 = {"artifact": art, "model_version": ri_j1.version_of(art)}
    sh = fs.j1_shadow(case, j1, ens)
    assert sh["status"] == "ok" and sh["label"] == "J1" and sh["ir_features_read"] == "14 of 14"
    assert 0.0 < sh["model_probability"] < 1.0 and sh["probability"] == round(sh["model_probability"], 4)
    models = audit.known_models()
    kind, payload = models[sh["model_version"]]
    assert kind == "j1" and audit.recompute_shadow(kind, payload, sh) <= 1e-12
    # the scorer records it under its own key on a JTWC storm and leaves the published number alone
    model = fs.ri_model.load_model(fs.MODEL_ARTIFACT)
    live = dict(case, analysis_model="BEST", track_source="ral_bdeck", storm_name="TEST")
    scored = fs.score_live_cases(model, [live], j1=dict(j1, base=fs.load_j1_base(art)))
    s = scored[0]
    assert s[ri_j1.SHADOW_KEY]["status"] == "ok" and s["ri_source"] == fs.SERVED_LABEL == "v8.3"
    # the live v8.2 ensemble, scored by the scorer from the case, is the training row's own v8.2 input
    assert s[ri_j1.SHADOW_KEY]["inputs"]["v82_logit"] == pytest.approx(FIXTURE["inputs"]["v82_logit"], abs=1e-9)
    no_j1 = fs.score_live_cases(model, [live])[0]
    assert ri_j1.SHADOW_KEY not in no_j1 and no_j1["ri_probability"] == s["ri_probability"]


# ---------------------------------------------------------------------------------------------------------------
# Amendment 15: v8.3 is published; J1 keeps reading v8.2 (its trained input) and its comparator stays v8.2
# ---------------------------------------------------------------------------------------------------------------

def _scored_with_j1(monkeypatch):
    """A JTWC case scored by the scorer as it runs live: the served model, J1 as ``load_j1`` builds it, and every
    number ``j1_shadow`` receives captured."""
    fs = _script("fetch_and_score")
    art, case, _ens, _ir = _live_case_from_fixture()
    case["fix_positions"] = {"models": list(ri_j1.FIX_MODELS), "t": [-12.6, 156.2], "t_minus_6h": [-12.4, 156.4]}
    lat, lon = np.arange(-20.0, -5.0, 0.072), np.arange(150.0, 163.0, 0.072)
    counts = np.full((lat.size, lon.size), 200, np.uint8)
    monkeypatch.setattr(fs, "_IR_IMAGES", {})
    monkeypatch.setattr(ir_source, "fetch_image", lambda hour: ("k", counts, lat, lon))
    monkeypatch.setattr(fs, "ir_reader_available", lambda: True)
    received: list[tuple[float, float | None]] = []
    real = fs.j1_shadow
    monkeypatch.setattr(fs, "j1_shadow", lambda c, j, e, cal=None: received.append((e, cal)) or real(c, j, e, cal))
    served = fs.load_serving_model()
    j1 = fs.load_j1()
    live = dict(case, analysis_model="BEST", track_source="ral_bdeck", storm_name="TEST")
    s = fs.score_live_cases(served, [live], j1=j1)[0]
    v82 = fs.ri_model.score_cases(fs.ri_model.load_model(fs.J1_BASE_ARTIFACT), [live])
    v83 = fs.ri_model.score_cases(served, [live])
    return fs, s, received, v82, v83


def test_j1_reads_v82s_numbers_never_the_served_v83s(monkeypatch):
    """The scorer serves v8.3 and feeds J1 v8.2's ensemble and calibrated probability, from v8.2's artifact, on the
    same live case. Fails if J1 receives the served model's numbers (the coupling amendment 15 removed)."""
    fs, s, received, v82, v83 = _scored_with_j1(monkeypatch)
    assert fs.SERVED_MODEL_VERSION == "hurricane_ri_v8_3" and fs.J1_BASE_MODEL_VERSION == "hurricane_ri_v8_2"
    assert s["model_version"] == "hurricane_ri_v8_3" and s["ri_source"] == "v8.3"
    # the case discriminates: v8.2 and v8.3 give different numbers on it
    assert float(v82["ensemble"][0]) != float(v83["ensemble"][0])
    assert float(v82["calibrated"][0]) != float(v83["calibrated"][0])
    # J1 received v8.2's numbers, bit for bit, and not v8.3's
    assert received == [(float(v82["ensemble"][0]), float(v82["calibrated"][0]))]
    sh = s[ri_j1.SHADOW_KEY]
    assert sh["status"] == "ok"
    assert sh["inputs"]["v82_logit"] == ri_j1.logit(float(v82["ensemble"][0]))
    assert sh["inputs"]["v82_logit"] != ri_j1.logit(float(v83["ensemble"][0]))
    assert sh["inputs"]["v82_logit"] == pytest.approx(FIXTURE["inputs"]["v82_logit"], abs=1e-9)  # J1's training row
    assert sh["v8_2_model_probability"] == float(v82["calibrated"][0])
    assert sh["v8_2_model_probability"] != float(v83["calibrated"][0])
    assert sh["v8_2_model_version"] == "hurricane_ri_v8_2"
    # the published number is v8.3's; v8.2's is recorded beside it, for J1's comparator
    assert s["ri_probability"] == round(float(v83["calibrated"][0]), 4)
    assert s["v8_2"] == {"ri_probability": round(float(v82["calibrated"][0]), 4), "model_version": "hurricane_ri_v8_2"}


def test_the_j1_record_passes_its_prospective_comparator_check_and_v83_would_not(monkeypatch):
    """The record the scorer writes is one the prospective test scores against v8.2; the same record with v8.3's
    numbers in J1's comparator slots is refused."""
    _fs, s, _received, _v82, v83 = _scored_with_j1(monkeypatch)
    jp = _script("score_hurricane_j1_prospective")
    assert jp.comparator_checks(s, s[ri_j1.SHADOW_KEY])
    p83 = float(v83["calibrated"][0])
    coupled = {**s[ri_j1.SHADOW_KEY], "v8_2_model_probability": p83, "v8_2_model_version": "hurricane_ri_v8_3"}
    assert not jp.comparator_checks({**s, "v8_2": {"ri_probability": round(p83, 4),
                                                   "model_version": "hurricane_ri_v8_3"}}, coupled)


def test_j1s_base_must_be_the_v82_it_was_exported_with(tmp_path):
    fs = _script("fetch_and_score")
    art, _ = ri_j1.load()
    assert fs.load_j1_base(art)["model_version"] == "hurricane_ri_v8_2"
    # the served v8.3 is not J1's base: refused by its bytes
    with pytest.raises(fs.ri_model.ModelArtifactError, match="exported against"):
        fs.load_j1_base(art, fs.MODEL_ARTIFACT)
    # nor is an artifact J1 does not name
    with pytest.raises(fs.ri_model.ModelArtifactError, match="built on"):
        fs.load_j1_base({**art, "v82_dependency": {**art["v82_dependency"], "model_version": "hurricane_ri_v8_3"}})


def test_a_j1_given_the_served_model_as_its_base_is_written_down_as_an_error_and_publishes_nothing(monkeypatch):
    fs, s, _received, _v82, _v83 = _scored_with_j1(monkeypatch)
    served = fs.load_serving_model()
    art, version = ri_j1.load()
    _a, case, _e, _i = _live_case_from_fixture()
    case.update(fix_positions={"models": list(ri_j1.FIX_MODELS), "t": [-12.6, 156.2], "t_minus_6h": [-12.4, 156.4]},
                analysis_model="BEST", track_source="ral_bdeck", storm_name="TEST")
    bad = fs.score_live_cases(served, [case], j1={"artifact": art, "model_version": version, "base": served})[0]
    assert bad[ri_j1.SHADOW_KEY]["status"].startswith("error:") and "v8_2" not in bad
    assert bad["ri_probability"] == s["ri_probability"] and bad["model_version"] == "hurricane_ri_v8_3"


def test_the_audit_now_covers_v10_4():
    audit = _script("audit_hurricane_records")
    kinds = {}
    for _version, (kind, art) in audit.known_models().items():
        kinds.setdefault(kind, []).append(art.get("label") or art.get("model_name"))
    assert "v10.4" in kinds["v10"] and kinds["j1"] == ["J1"]


def test_regions_split_the_southern_hemisphere_at_135e_and_the_scope_is_the_registered_one(tmp_path):
    assert ri_j1.storm_region("SH", 120.0) == "SI" and ri_j1.storm_region("SH", 135.0) == "SP"
    assert ri_j1.storm_region("SH", -170.0) == "SP" and ri_j1.storm_region("SH", None) is None
    assert ri_j1.storm_region("IO", 88.0) == "NI" and ri_j1.storm_region("WP", 140.0) == "WP"
    assert ri_j1.storm_region("AL", -60.0) is None
    # amendment 14's registered scope: development point estimate <= 0 against v8.2 -> WP and SI
    assert ri_j1.scope() == ("WP", "SI")
    assert ri_j1.in_scope("SH", 100.0) and not ri_j1.in_scope("SH", 160.0) and not ri_j1.in_scope("IO", 88.0)
    assert ri_j1.scope(tmp_path / "absent.json") == () and not ri_j1.in_scope("WP", 140.0, tmp_path / "absent.json")


def test_the_v82_dependency_hash_is_the_stored_bytes_on_every_checkout(tmp_path):
    """The first export hashed a Windows CRLF working copy (8a75...), not the bytes git stores (b9b6...)."""
    art, _ = ri_j1.load()
    v82 = ROOT / "results" / "models" / "hurricane_ri_v8_2.json"
    assert art["v82_dependency"]["artifact_sha256"] == ri_j1.lf_sha256(v82)
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    body = v82.read_bytes().replace(bytes([13, 10]), bytes([10]))
    lf.write_bytes(body)
    crlf.write_bytes(body.replace(bytes([10]), bytes([13, 10])))
    assert ri_j1.lf_sha256(lf) == ri_j1.lf_sha256(crlf) == art["v82_dependency"]["artifact_sha256"]
