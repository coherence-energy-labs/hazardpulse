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


def test_an_artifact_of_another_schema_is_refused(tmp_path):
    art = json.loads(ri_v10.MODEL_PATH.read_text(encoding="utf-8"))
    art["schema"] = "something/else"
    p = tmp_path / "a.json"
    p.write_bytes(ri_v10.canonical_bytes(art))
    with pytest.raises(ValueError):
        ri_v10.load(p)
