"""Hurricane forecast records: each one keeps the exact inputs that made it, joins a hash chain, and
can be audited later -- chain, agreement with its replay file, and recomputation from its inputs."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane import atcf, ri_v10
from hazardpulse.hurricane import ri_v9_features as fx

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "hurricane_v9"
pytestmark = pytest.mark.skipif(not ri_v10.MODEL_PATH.exists(), reason="v10 artifact not built")


def _load(name, rel):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fs():
    return _load("fetch_and_score_records_t", "scripts/fetch_and_score.py")


@pytest.fixture(scope="module")
def audit_mod():
    return _load("audit_hurricane_records_t", "scripts/audit_hurricane_records.py")


def _records():
    return atcf.parse_atcf_deck(gzip.decompress((FIX / "aal012026.dat.gz").read_bytes()).decode("utf-8", "replace"))


def _storm(fs, dtg):
    c = next(x for x in json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"] if x["dtg"] == dtg)
    case = {"storm_id": "AL012026", "issue_time": dt.datetime.strptime(dtg, "%Y%m%d%H").isoformat(),
            "analysis_model": "CARQ"}
    v10 = dict(zip(("artifact", "model_version"), ri_v10.load()))
    out = fs.shadow_forecasts(case, None, v10, ships_raw_fetcher=lambda s, cy: ((FIX / c["ships_text"]).read_text(
        encoding="utf-8"), c["ships_text"]), adeck_fetcher=lambda s: _records())
    return {"storm_id": "AL012026", "issue_time": case["issue_time"], "ri_probability": 0.1,
            "model_version": "x", "ri_source": "noaa_aid_stack", **out}


def test_record_inputs_give_back_the_model_row_bit_for_bit():
    f = {"v0": 55.0, "dv24_HCCA": float("nan"), "ri_DTOP_30_24": -2.1972245773362196}
    names = ["v0", "dv24_HCCA", "ri_DTOP_30_24", "absent"]
    rec = fx.record_inputs(f, names)
    assert rec == {"v0": 55.0, "dv24_HCCA": None, "ri_DTOP_30_24": -2.1972245773362196, "absent": None}
    back = json.loads(json.dumps(rec))                                  # through the JSON record
    a, b = fx.vector(f, names), np.array([np.nan if back[n] is None else back[n] for n in names])
    assert np.array_equal(a, b, equal_nan=True)


def test_a_shadow_record_recomputes_from_its_own_inputs(fs):
    sh = _storm(fs, "2026061612")["ri_v10_shadow"]
    assert len(sh["inputs"]) == len(ri_v10.load()[0]["feature_names"])
    got = ri_v10.recompute(ri_v10.load()[0], json.loads(json.dumps(sh["inputs"])))
    for k, v in sh["model_probabilities"].items():
        assert got[k] == pytest.approx(v, abs=5e-5)


def _root(tmp_path: Path) -> Path:
    (tmp_path / "results" / "models").mkdir(parents=True)
    (tmp_path / "dist" / "data" / "replay").mkdir(parents=True)
    shutil.copyfile(ri_v10.MODEL_PATH, tmp_path / "results" / "models" / ri_v10.MODEL_PATH.name)
    return tmp_path


def _issue(fs, root: Path, fid: str, storms: list[dict]):
    rp = root / "dist" / "data" / "replay" / f"{fid}.json"
    rp.write_text(json.dumps({"forecast_id": fid, "issued_at": "x", "storms": storms}, indent=2), encoding="utf-8")
    fs.append_hurricane_ledger(fid, dt.datetime(2026, 10, 3, 12), "m", storms,
                               path=root / "dist" / "data" / "hurricane-ledger.jsonl")


def test_the_audit_passes_clean_records_and_catches_each_kind_of_change(fs, audit_mod, tmp_path):
    root = _root(tmp_path)
    for i, dtg in enumerate(("2026061612", "2026061618", "2026061700")):
        _issue(fs, root, f"hu_fcst_2026100{i}_0000", [_storm(fs, dtg)])
    clean = audit_mod.audit(root)
    assert clean["ok"] and clean["ledger"]["entries"] == 3 and clean["recompute"]["matched"] == 3

    # 1. a stored probability edited after the fact: the replay no longer matches its ledger entry
    p = root / "dist" / "data" / "replay" / "hu_fcst_20261001_0000.json"
    good = p.read_text(encoding="utf-8")
    art = json.loads(good)
    art["storms"][0]["ri_v10_shadow"]["probability"] = 0.99
    p.write_text(json.dumps(art), encoding="utf-8")
    res = audit_mod.audit(root)
    assert not res["ok"] and res["ledger"]["content_mismatches"] == ["hu_fcst_20261001_0000"]
    p.write_text(good, encoding="utf-8")

    # 2. a stored input edited: the shadow no longer recomputes (and the content hash breaks too)
    art = json.loads(good)
    sh = art["storms"][0]["ri_v10_shadow"]
    k = next(n for n, v in sh["inputs"].items() if v is not None and n.startswith("ri_"))
    sh["inputs"][k] = sh["inputs"][k] + 3.0
    p.write_text(json.dumps(art), encoding="utf-8")
    res = audit_mod.audit(root)
    assert res["recompute"]["mismatched"] and "hu_fcst_20261001_0000" in res["ledger"]["content_mismatches"]
    p.write_text(good, encoding="utf-8")

    # 3. a ledger entry rewritten (even with its own hash recomputed): the chain breaks at the next link
    led = root / "dist" / "data" / "hurricane-ledger.jsonl"
    rows = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines()]
    rows[1]["storms"][0]["published"] = 0.5
    rows[1]["hash"] = audit_mod.sha({k: v for k, v in rows[1].items() if k != "hash"})
    led.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = audit_mod.audit(root)
    assert any("entry 2" in m and "prev_hash" in m for m in res["ledger"]["chain_mismatches"])


def test_records_from_before_the_ledger_and_inputs_are_counted_as_not_covered(fs, audit_mod, tmp_path):
    root = _root(tmp_path)
    old = _storm(fs, "2026061612")
    old["ri_v10_shadow"].pop("inputs")
    (root / "dist" / "data" / "replay" / "hu_fcst_20260901_0000.json").write_text(
        json.dumps({"forecast_id": "hu_fcst_20260901_0000", "storms": [old]}), encoding="utf-8")
    res = audit_mod.audit(root)
    assert res["recompute"]["no_inputs"] == 1 and res["recompute"]["checked"] == 0
    assert res["ledger"]["entries"] == 0 and res["ok"]           # nothing contradicted, nothing claimed


def test_the_ledger_follows_the_output_directory(fs, monkeypatch, tmp_path):
    """Found 2026-10-03: a path fixed at import let the serving tests (which point DIST at a temp
    directory) append five fake entries to the real ledger."""
    monkeypatch.setattr(fs, "DIST", tmp_path)
    fs.append_hurricane_ledger("hu_fcst_x", dt.datetime(2026, 10, 2, 12), "m", [])
    assert (tmp_path / "data" / "hurricane-ledger.jsonl").exists()


def test_the_verification_workflow_runs_the_audit_strictly():
    wf = (ROOT / ".github" / "workflows" / "verification-score.yml").read_text(encoding="utf-8")
    assert "python scripts/audit_hurricane_records.py --strict" in wf
    assert "results/hurricane_prospective/" in wf.split("Commit and push", 1)[1]
