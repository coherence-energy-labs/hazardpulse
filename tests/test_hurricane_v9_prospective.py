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
    monkeypatch.setattr(m, "collect", lambda **kw: [])
    m.main(["--as-of", "2026-12-02"])
    first = json.loads((tmp_path / "v9.json").read_text())
    for name in ("v9_1", "v10_1"):
        assert first["entrants"][name]["looks"]["2026-12-01"]["n"] == 0
    first["entrants"]["v10_1"]["looks"]["2026-12-01"]["marker"] = "frozen"
    (tmp_path / "v9.json").write_text(json.dumps(first))
    m.main(["--as-of", "2026-12-20"])
    assert json.loads((tmp_path / "v9.json").read_text())["entrants"]["v10_1"]["looks"]["2026-12-01"]["marker"] == "frozen"


def _scored(dv, p_k, a_k, sid):
    return {"storm_id": sid, "cycle": "2026-10-05T00:00:00Z", "p": p_k["30"], "a": a_k["30"], "gate_ok": True,
            "dv": dv, "y": int(dv >= 30), "p_k": p_k, "a_k": a_k}


def test_the_v10_claim_uses_the_four_threshold_brier_and_the_30_kt_log_loss():
    m = _mod()
    good = {"25": 0.9, "30": 0.8, "35": 0.2, "40": 0.1}
    noaa = {"25": 0.5, "30": 0.4, "35": 0.3, "40": 0.2}
    calm_ours = {"25": 0.02, "30": 0.01, "35": 0.01, "40": 0.0}
    calm_noaa = {"25": 0.10, "30": 0.08, "35": 0.05, "40": 0.03}
    rows = []
    for i in range(30):
        rows.append(_scored(32.0, good, noaa, f"EP{i:02d}2026"))        # RI: dV 32 kt -> 25 and 30 exceeded
        rows.append(_scored(0.0, calm_ours, calm_noaa, f"EP{i:02d}2026"))
    # hand check of one RI row: ours (0.9-1)^2+(0.8-1)^2+0.2^2+0.1^2 = 0.10; NOAA 0.25+0.36+0.09+0.04 = 0.74
    assert abs(m._multi_brier(rows[:1], "ours")[0] - 0.10) < 1e-12
    assert abs(m._multi_brier(rows[:1], "noaa")[0] - 0.74) < 1e-12
    res = m.evaluate(rows, 0.9875, "multi")
    assert res["multi_threshold_brier"]["d"] < 0 and res["multi_threshold_brier"]["d_ci"][1] < 0
    assert res["claim"] is True
    # ours worse on every threshold: no claim
    worse = [dict(r, p_k=r["a_k"], a_k=r["p_k"], p=r["a"], a=r["p"]) for r in rows]
    assert m.evaluate(worse, 0.9875, "multi")["claim"] is False
