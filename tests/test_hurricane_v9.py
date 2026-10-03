"""v9.1 RI shadow: the live path reproduces the 2026 final's own forecasts; the gate falls back to DTOPS."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane import atcf, ri_v9
from hazardpulse.hurricane import ri_v9_features as fx
from hazardpulse.tornado import lgbm_payload as lp

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "hurricane_v9"

pytestmark = pytest.mark.skipif(not ri_v9.MODEL_PATH.exists(), reason="v9 artifact not built")


@pytest.fixture(scope="module")
def fs():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_and_score_v9_test", ROOT / "scripts" / "fetch_and_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v9():
    payload, version = ri_v9.load()
    return {"payload": payload, "model_version": version}


def _records():
    return atcf.parse_atcf_deck(gzip.decompress((FIX / "aal012026.dat.gz").read_bytes()).decode("utf-8", "replace"))


def _case(dtg: str) -> dict:
    t = dt.datetime.strptime(dtg, "%Y%m%d%H")
    return {"storm_id": "AL012026", "issue_time": t.isoformat(), "analysis_model": "CARQ"}


def _ships(name):
    return lambda sid, cyc: ((FIX / name).read_text(encoding="utf-8"), name)


def test_the_live_shadow_reproduces_the_finals_own_forecast(fs, v9):
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    assert exp["cases"]
    for c in exp["cases"]:
        out = fs.v9_shadow(_case(c["dtg"]), v9, ships_raw_fetcher=_ships(c["ships_text"]),
                           adeck_fetcher=lambda sid: _records())
        assert out["status"] == "ok" and out["gate_ok"], out
        assert out["model_probability"] == pytest.approx(c["d_gbt"], abs=5e-5)     # live == final
        assert out["probability"] == out["model_probability"] and out["source"] == "v9.1"
        assert out["dtops_pct"] / 100.0 == pytest.approx(c["a"])                    # the comparator, recorded


def test_without_the_early_aids_the_gate_serves_dtops(fs, v9):
    c = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"][0]
    stripped = [r for r in _records() if r.model not in ri_v9.GATE_AIDS]
    out = fs.v9_shadow(_case(c["dtg"]), v9, ships_raw_fetcher=_ships(c["ships_text"]),
                       adeck_fetcher=lambda sid: stripped)
    assert out["gate_ok"] is False and set(out["gate_missing"]) == set(ri_v9.GATE_AIDS)
    assert out["source"].startswith("DTOPS") and out["probability"] == pytest.approx(c["a"])


def test_only_nhc_adeck_cases_are_shadowed(fs, v9):
    assert fs.v9_shadow({"storm_id": "WP242026", "issue_time": "2026-10-03T00:00:00", "analysis_model": "CARQ"},
                        v9)["status"] == "not an NHC a-deck case"
    assert fs.v9_shadow({**_case("2026061612"), "analysis_model": "JTWC"}, v9)["status"] == "not an NHC a-deck case"


def test_a_payload_with_other_inputs_is_refused(tmp_path):
    payload = lp.load(ri_v9.MODEL_PATH)
    payload["feature_names"] = list(reversed(payload["feature_names"]))
    p = tmp_path / "m.json"
    p.write_bytes(lp.canonical_bytes(payload))
    with pytest.raises(ValueError):
        ri_v9.load(p)


def test_the_feature_builder_reads_carq_and_aid_changes():
    recs = _records()
    t = dt.datetime(2026, 6, 16, 12)
    f = fx.adeck_features(fx.cycle_table(recs, t), "AL")
    assert f["is_atlantic"] == 1.0 and f["v0"] > 0
    carq24 = next(r.vmax_kt for r in recs if r.model == "DSHP" and r.cycle == t and r.tau_hours == 24)
    assert f["dv24_DSHP"] == pytest.approx(carq24 - f["v0"])
