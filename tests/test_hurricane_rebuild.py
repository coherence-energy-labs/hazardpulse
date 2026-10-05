"""Amendment 7 rule 3: a missed test cycle may be rebuilt only by a procedure that reproduces the live
records exactly (to 1e-12). The committed rebuild (EP15 and EP18 at 2026-10-04 00Z) carries its control,
its inputs and its own hash, and the prospective test reads it as a rule 3 record."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane import cycle_records as cr

ROOT = Path(__file__).resolve().parents[1]
REBUILT = ROOT / "results" / "hurricane_prospective" / "rebuilt"


def _load(name, rel):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rb():
    return _load("rebuild_t", "scripts/rebuild_hurricane_cycles.py")


def test_the_control_sees_any_difference_beyond_1e_12(rb):
    stored = {"probability": 0.0182, "model_probability": 0.0182, "model_version": "m",
              "probabilities": {"30": 0.0182, "15": None}, "inputs": {"v0": 55.0, "ofcl_dv24": None}}
    assert rb.compare(stored, json.loads(json.dumps(stored))) == []
    for path, value in ((("probability",), 0.0183), (("probabilities", "15"), 0.01), (("inputs", "ofcl_dv24"), 5.0),
                        (("inputs", "v0"), 55.000000001), (("model_version",), "other")):
        other = json.loads(json.dumps(stored))
        d = other
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = value
        assert rb.compare(stored, other), path
    near = dict(stored, probability=0.0182 + 5e-13)
    assert rb.compare(stored, near) == []


def test_a_ships_text_modified_after_the_advisory_or_missing_from_the_archive_is_refused(rb, monkeypatch, tmp_path):
    proc = object.__new__(rb.Procedure)
    proc.read = {}
    text = b"SHIPS text\n"
    proc.fs = type("FS", (), {"fetch_bytes": staticmethod(lambda url, **kw: text)})
    monkeypatch.setattr(rb, "ARCHIVE", tmp_path)
    (tmp_path / "2026").mkdir()
    (tmp_path / "2026" / "26100400EP1526_ships.txt.gz").write_bytes(gzip.compress(text, mtime=0))
    t = dt.datetime(2026, 10, 4, 0)
    proc.listing = {"26100400EP1526_ships.txt": "2026-10-04 03:30"}                       # at t + 3 h 30: refused
    with pytest.raises(SystemExit):
        proc.ships("EP152026", t)
    proc.listing = {"26100400EP1526_ships.txt": "2026-10-04 01:06"}
    got, name = proc.ships("EP152026", t)
    assert got == "SHIPS text\n" and proc.read[("EP152026", t)]["ships_text"]["sha256"] == hashlib.sha256(text).hexdigest()
    proc.fs = type("FS", (), {"fetch_bytes": staticmethod(lambda url, **kw: b"rewritten\n")})
    with pytest.raises(SystemExit):                                                          # not a kept version
        proc.ships("EP152026", t)


def test_a_rebuilt_record_counts_only_where_no_run_recorded_the_cycle(tmp_path):
    m = _load("v9p_rebuild_t", "scripts/score_hurricane_v9_prospective.py")
    replay, rebuilt = tmp_path / "replay", tmp_path / "rebuilt"
    replay.mkdir()
    rebuilt.mkdir()

    def rec(sid, cyc, p, **extra):
        return {"storm_id": sid, "issue_time": cyc, **extra,
                "ri_v9_shadow": {"status": "ok", "probability": p, "cycle": cyc + "Z", "dtops_pct": 1, "gate_ok": True}}
    (replay / "hu_fcst_20261004_0959.json").write_text(json.dumps({
        "forecast_id": "hu_fcst_20261004_0959", "issued_at": "2026-10-04T09:59:43Z",
        "storms": [rec("EP152026", "2026-10-04T06:00:00", 0.02)]}), encoding="utf-8")
    (rebuilt / "hu_rebuilt_20261005_0251.json").write_text(json.dumps({
        "forecast_id": "hu_rebuilt_20261005_0251", "issued_at": "2026-10-05T02:51:28Z",
        cr.REBUILT_KEY: [rec("EP152026", "2026-10-04T00:00:00", 0.0069, rebuilt=True),
                         rec("EP152026", "2026-10-04T06:00:00", 0.5, rebuilt=True)]}), encoding="utf-8")
    recs = m.collect(selection=m.select_test_records(replay, rebuilt_dir=rebuilt))
    assert [(r["cycle"], r["p"], r["rebuilt"]) for r in recs] == [
        ("2026-10-04T00:00:00Z", 0.0069, True), ("2026-10-04T06:00:00Z", 0.02, False)]   # the live 06Z stands
    assert [r["cycle"] for r in m.without_catch_up(recs)] == ["2026-10-04T06:00:00Z"]
    assert len(m.collect(replay)) == 1                                                      # another dir: none


def test_the_committed_rebuild_carries_its_exact_control_and_its_own_hash():
    files = sorted(REBUILT.glob("hu_rebuilt_*.json"))
    if not files:
        pytest.skip("no rebuild in this checkout")
    for p in files:
        art = json.loads(p.read_text(encoding="utf-8"))
        body = {k: v for k, v in art.items() if k != "content_sha256"}
        assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() \
            == art["content_sha256"]
        c = art["control"]
        assert c["tolerance"] == 1e-12 and c["records"] >= 6 and c["reproduced"] == c["shadows"] == 4 * c["records"]
        assert all(r["reproduced"] and not r["differences"] for r in c["results"])
        for r in art[cr.REBUILT_KEY]:
            assert r["rebuilt"] is True and r["published"] is False
            t = cr.parse_utc(r["issue_time"])
            listed = r["inputs_read"]["ships_text"]["listed_modified"]
            assert listed < (t + cr.ADVISORY_DELAY).strftime("%Y-%m-%d %H:%M")
            assert {"ri_v9_shadow", "ri_v10_shadow", "ri_v10_2_shadow", "ri_v10_3_shadow"} <= set(r)
    v9p = _load("v9p_rebuild_committed_t", "scripts/score_hurricane_v9_prospective.py")
    chosen = v9p.select_test_records().chosen
    assert all(chosen[(r["storm_id"], cr.parse_utc(r["issue_time"]))].rebuilt
               for p in files for r in json.loads(p.read_text(encoding="utf-8"))[cr.REBUILT_KEY])
