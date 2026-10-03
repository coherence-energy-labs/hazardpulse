"""The v9.1 prospective scorer: one record per storm-cycle, only matured cycles, looks frozen once."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v9_prospective_t", ROOT / "scripts" / "score_hurricane_v9_prospective.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(d: Path, fid: str, storms: list[dict]) -> None:
    (d / f"{fid}.json").write_text(json.dumps({"forecast_id": fid, "storms": storms}), encoding="utf-8")


def _storm(sid, cycle, p, dtops=10, ok=True):
    return {"storm_id": sid, "ri_v9_shadow": {"status": "ok" if ok else "error: x", "probability": p,
                                              "cycle": cycle, "dtops_pct": dtops, "riod_pct": 12,
                                              "gate_ok": True, "source": "v9.1", "model_version": "hurricane_ri_v9-x"}}


def test_records_are_deduplicated_and_old_or_failed_shadows_ignored(tmp_path):
    m = _mod()
    _write(tmp_path, "hu_fcst_20261003_2130", [_storm("EP202026", "2026-10-03T18:00:00Z", 0.3)])     # before start
    _write(tmp_path, "hu_fcst_20261004_0330", [_storm("EP202026", "2026-10-04T00:00:00Z", 0.4),
                                              _storm("EP212026", "2026-10-04T00:00:00Z", 0.1, ok=False)])
    _write(tmp_path, "hu_fcst_20261004_0400", [_storm("EP202026", "2026-10-04T00:00:00Z", 0.9)])     # a re-run
    recs = m.collect(tmp_path)
    assert [(r["storm_id"], r["cycle"], r["p"]) for r in recs] == [("EP202026", "2026-10-04T00:00:00Z", 0.4)]
    assert recs[0]["a"] == 0.10


def test_only_matured_cycles_with_both_fixes_are_scored_and_the_event_is_30_kt():
    m = _mod()
    recs = [{"storm_id": "EP202026", "cycle": "2026-10-04T00:00:00Z", "p": 0.4, "a": 0.1, "gate_ok": True},
            {"storm_id": "EP202026", "cycle": "2026-10-04T06:00:00Z", "p": 0.2, "a": 0.1, "gate_ok": True},
            {"storm_id": "EP202026", "cycle": "2026-10-05T00:00:00Z", "p": 0.2, "a": 0.1, "gate_ok": True}]
    bt = {dt.datetime(2026, 10, 4, 0): 50.0, dt.datetime(2026, 10, 5, 0): 80.0, dt.datetime(2026, 10, 4, 6): 55.0}
    out = m.score(recs, dt.datetime(2026, 10, 5, 12), best_track=lambda sid: bt)
    assert [(r["cycle"], r["y"]) for r in out] == [("2026-10-04T00:00:00Z", 1)]   # 06Z lacks t+24; the 5th is unmatured


def test_a_look_is_evaluated_once_and_then_frozen(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "OUT", tmp_path / "v9.json")
    monkeypatch.setattr(m, "collect", lambda: [])
    m.main(["--as-of", "2026-12-02"])
    first = json.loads((tmp_path / "v9.json").read_text())
    assert "2026-12-01" in first["looks"] and first["looks"]["2026-12-01"]["n"] == 0
    first["looks"]["2026-12-01"]["marker"] = "frozen"
    (tmp_path / "v9.json").write_text(json.dumps(first))
    m.main(["--as-of", "2026-12-20"])
    assert json.loads((tmp_path / "v9.json").read_text())["looks"]["2026-12-01"]["marker"] == "frozen"
