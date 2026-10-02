"""Liveness must be per hazard, from the hazard's own artifact, with no pulse-level fallback.

The old inline check in liveness-check.yml read ``hazard.get("updated_at") or
pulse["updated_at"]``. No hazard carried its own timestamp and every scorer rewrites the
pulse-level one, so the check passed on every run while the hurricane scorer was dead
(last success 2026-05-26). These tests pin both halves of the fix: the checker, and the
producer that stamps each hazard from its own artifact.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOW = dt.datetime(2026, 10, 2, 3, 0, tzinfo=dt.timezone.utc)


def _load(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def liveness():
    return _load("check_liveness.py", "check_liveness_test")


def _pulse(**stamps):
    return {
        "updated_at": "2026-10-02T02:55:00Z",   # fresh, as any scorer leaves it
        "hazards": [{"key": k, **({"updated_at": v} if v is not None else {})} for k, v in stamps.items()],
    }


def test_all_hazards_fresh_passes(liveness):
    lines, failures = liveness.evaluate_freshness(
        _pulse(eq="2026-10-01T22:00:00Z", hu="2026-10-01T12:50:00Z", to="2026-10-02T00:17:40Z"), NOW)
    assert failures == []
    assert len(lines) == 3 and all(line.rstrip().endswith("OK") for line in lines)


def test_the_historical_blind_spot_now_fails(liveness):
    """Exactly the 2026-06..09 state: fresh pulse timestamp, hurricane without its own."""
    _, failures = liveness.evaluate_freshness(
        _pulse(eq="2026-10-01T22:00:00Z", hu=None, to="2026-10-02T00:17:40Z"), NOW)
    assert len(failures) == 1 and failures[0].startswith("hu: no per-hazard updated_at")


def test_a_stale_hazard_fails(liveness):
    _, failures = liveness.evaluate_freshness(
        _pulse(eq="2026-10-01T22:00:00Z", hu="2026-05-26T12:50:00Z", to="2026-10-02T00:17:40Z"), NOW)
    assert len(failures) == 1 and failures[0].startswith("hu is stale: 3086.2h")


def test_missing_garbled_and_future_stamps_fail(liveness):
    _, failures = liveness.evaluate_freshness(
        _pulse(eq="not a time", to="2026-10-02T09:00:00Z"), NOW)
    joined = "\n".join(failures)
    assert "eq: no per-hazard updated_at" in joined
    assert "hu: hazard missing" in joined
    assert "to claims a future update time" in joined


def test_cli_exit_code_follows_the_verdict(liveness, tmp_path, monkeypatch):
    fresh = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pulse_path = tmp_path / "live-pulse.json"
    monkeypatch.setattr(liveness, "PULSE_PATH", pulse_path)
    monkeypatch.setattr(liveness, "REPLAY_DIR", tmp_path)
    pulse_path.write_text(json.dumps(_pulse(eq=fresh, hu=fresh, to=fresh)), encoding="utf-8")
    assert liveness.main([]) == 0
    pulse_path.write_text(json.dumps(_pulse(eq=fresh, hu=None, to=fresh)), encoding="utf-8")
    assert liveness.main([]) == 1


def test_site_builder_stamps_each_hazard_from_its_own_artifact(tmp_path, monkeypatch):
    bsa = _load("build_site_artifacts.py", "bsa_liveness_test")
    data = tmp_path / "data"
    replay = data / "replay"
    replay.mkdir(parents=True)
    pulse = {
        "updated_at": "2026-10-02T02:55:00Z",
        "hazards": [
            {"key": "eq", "forecast_id": "eq_fcst_20261001_2200"},
            {"key": "hu", "forecast_id": "hu_fcst_20260526_1250"},
            {"key": "to", "forecast_id": "to_fcst_20261002_0017"},
        ],
    }
    (data / "live-pulse.json").write_text(json.dumps(pulse), encoding="utf-8")
    (data / "live-storms.json").write_text(json.dumps(
        {"updated_at": "2026-05-26T12:50:48.1Z", "forecast_id": "hu_fcst_20260526_1250", "storms": []}),
        encoding="utf-8")
    (data / "live-tornadoes.json").write_text(json.dumps(
        {"updated_at": "2026-10-02T00:17:40.3Z", "forecast_id": "to_fcst_20261002_0017", "storms": []}),
        encoding="utf-8")
    (replay / "eq_fcst_20261001_2200.json").write_text(json.dumps(
        {"forecast_id": "eq_fcst_20261001_2200", "issued_at": "2026-10-01T22:00:00Z"}), encoding="utf-8")
    for name, value in {
        "DIST": tmp_path,
        "LIVE_PULSE_PATH": data / "live-pulse.json",
        "LIVE_STORMS_PATH": data / "live-storms.json",
        "LIVE_TORNADOES_PATH": data / "live-tornadoes.json",
        "REPLAY_DIR": replay,
        "REPLAY_INDEX_PATH": data / "evidence" / "replay-index.json",
    }.items():
        monkeypatch.setattr(bsa, name, value)

    bsa._ensure_live_publish_artifacts()
    stamped = {h["key"]: h.get("updated_at") for h in json.loads((data / "live-pulse.json").read_text())["hazards"]}
    assert stamped == {"eq": "2026-10-01T22:00:00Z", "hu": "2026-05-26T12:50:48Z", "to": "2026-10-02T00:17:40Z"}

    liveness = _load("check_liveness.py", "check_liveness_test2")
    _, failures = liveness.evaluate_freshness(json.loads((data / "live-pulse.json").read_text()), NOW)
    assert [f.split(":")[0] for f in failures] == ["hu is stale"]
