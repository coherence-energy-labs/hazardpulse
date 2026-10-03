"""v10.1 RI shadow: the live path reproduces the lab's exceedance curve; the gate serves NOAA's own values."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import sys
from pathlib import Path

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


def test_both_shadows_come_from_one_read(fs, v10):
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
    out = fs.shadow_forecasts(_case(c["dtg"]), {"payload": payload, "model_version": version}, v10, ships, adeck)
    assert set(out) == {"ri_v9_shadow", "ri_v10_shadow"} and calls == {"ships": 1, "adeck": 1}


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
