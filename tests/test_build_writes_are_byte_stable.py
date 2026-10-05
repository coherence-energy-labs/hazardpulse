"""The site build writes signed/hashed JSON as LF bytes on every platform and never rewrites an
unchanged file (a Windows build rewrote two frozen replays to CRLF; .gitattributes pins them)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bsa():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("bsa_byte_stable", ROOT / "scripts" / "build_site_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_json_is_written_as_lf_bytes_and_an_unchanged_file_is_left_alone(tmp_path):
    bsa = _bsa()
    p = tmp_path / "replay" / "to_fcst_x.json"
    payload = {"forecast_id": "to_fcst_x", "storms": [{"p": 0.1}]}
    bsa._write_json(p, payload)
    data = p.read_bytes()
    assert bytes([13, 10]) not in data and data.endswith(bytes([10]))
    os.utime(p, (1_000_000_000, 1_000_000_000))
    bsa._write_json(p, payload)                      # same content: not touched
    assert p.stat().st_mtime == 1_000_000_000
    bsa._write_json(p, {**payload, "storms": []})    # new content: written
    assert p.stat().st_mtime != 1_000_000_000 and b'"storms": []' in p.read_bytes()


def _tornado_fixture(tmp_path, monkeypatch, bsa, forecast_id, storms):
    import json
    data, replay = tmp_path / "data", tmp_path / "data" / "replay"
    replay.mkdir(parents=True, exist_ok=True)
    (data / "live-pulse.json").write_text(json.dumps({"hazards": [{"key": "to", "forecast_id": forecast_id}]}),
                                         encoding="utf-8")
    (data / "live-tornadoes.json").write_text(json.dumps({
        "disclaimer": "Always follow official NWS guidance.", "updated_at": "2026-10-05T00:25:23.302308Z",
        "forecast_id": forecast_id, "model_version": "tornado_v3-05a06c843c87", "scoring_tier": "tier1_v3",
        "n_active_storms": len(storms), "recent_predictions": [{"n_storms": 9}], "storms": storms}), encoding="utf-8")
    for name, value in {"DIST": tmp_path, "LIVE_PULSE_PATH": data / "live-pulse.json",
                        "LIVE_STORMS_PATH": data / "live-storms.json",
                        "LIVE_TORNADOES_PATH": data / "live-tornadoes.json", "REPLAY_DIR": replay,
                        "REPLAY_INDEX_PATH": data / "evidence" / "replay-index.json"}.items():
        monkeypatch.setattr(bsa, name, value)
    return data, replay


def test_a_tornado_forecast_is_stored_once_and_its_record_is_never_rewritten(tmp_path, monkeypatch):
    """The live file and the frozen record were two documents holding the same storms: every tornado run
    committed ~1.2 MB twice (74 KB of a ~170 KB commit, compressed). Now they are the same bytes, so git
    stores one object -- and an issued record is never rewritten, even to add fields."""
    import json
    bsa = _bsa()
    storms = [{"storm_id": "1", "lat": 35.5, "lon": -97.6, "tornado_probability": 0.4}]
    data, replay = _tornado_fixture(tmp_path, monkeypatch, bsa, "to_fcst_20261005_0025", storms)
    bsa._ensure_live_publish_artifacts()
    record = (replay / "to_fcst_20261005_0025.json").read_bytes()
    assert record == (data / "live-tornadoes.json").read_bytes()
    doc = json.loads(record)
    assert doc["issued_at"] == "2026-10-05T00:25:23Z" and doc["top_probability"] == 0.4
    assert doc["updated_at"] and doc["disclaimer"] and doc["storms"] == storms   # every field either file had
    # a later build (another scorer's run) changes nothing
    bsa._ensure_live_publish_artifacts()
    assert (replay / "to_fcst_20261005_0025.json").read_bytes() == record == (data / "live-tornadoes.json").read_bytes()

    # an issued record in the old format stays exactly as issued
    legacy = replay / "to_fcst_20261004_2059.json"
    legacy.write_bytes(b'{"forecast_id": "to_fcst_20261004_2059", "storms": []}\n')
    _tornado_fixture(tmp_path, monkeypatch, bsa, "to_fcst_20261004_2059", [])
    bsa._ensure_live_publish_artifacts()
    assert legacy.read_bytes() == b'{"forecast_id": "to_fcst_20261004_2059", "storms": []}\n'
    assert json.loads((data / "live-tornadoes.json").read_text())["updated_at"]          # the live file keeps its fields


def test_the_replay_and_evidence_files_are_pinned_against_line_ending_conversion():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "dist/data/replay/** -text" in attrs and "dist/data/evidence/** -text" in attrs
