"""Pages that summarise every hazard are re-rendered by every scorer, from the artifacts.

The homepage and the /live/ overview were rendered only by the tornado scorer, so an
earthquake cycle advanced live-pulse.json while dist/index.html kept the previous
forecast: on 2026-10-01 aedf08cd1 published eq_fcst_20261001_2200 (4.5%) and the homepage
still showed eq_fcst_20261001_1300 (4.6%). test_site_integrity caught the symptom on every
push since 2026-07-31; these tests pin the producer.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _publish_eq(dist: Path, forecast_id: str, probability: float) -> None:
    """What an earthquake cycle leaves behind: its replay and its pulse entry."""
    _write(dist / "data" / "replay" / f"{forecast_id}.json", {
        "forecast_id": forecast_id, "issued_at": "2026-10-01T22:00:00Z",
        "active_cells": [{"lat": 35.0, "lon": 140.0, "probability": probability, "risk_band": "critical"}],
    })
    pulse_path = dist / "data" / "live-pulse.json"
    pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
    for hazard in pulse["hazards"]:
        if hazard["key"] == "eq":
            hazard.update(forecast_id=forecast_id, probability=probability)
    pulse_path.write_text(json.dumps(pulse), encoding="utf-8")


def test_cross_hazard_pages_follow_the_pulse_whichever_hazard_published(tmp_path, monkeypatch):
    tornado = _load("fetch_and_score_tornado.py", "fst_cross_hazard_test")
    monkeypatch.setattr(tornado, "DIST", tmp_path)
    _write(tmp_path / "data" / "live-pulse.json", {
        "updated_at": "2026-10-01T22:00:00Z",
        "hazards": [
            {"key": "eq", "probability": 0.0, "risk_band": "critical", "forecast_id": None},
            {"key": "hu", "probability": 0.0, "risk_band": "none", "forecast_id": "hu_fcst_20260526_1250"},
            {"key": "to", "probability": 0.0057, "risk_band": "high", "forecast_id": "to_fcst_20261001_2217"},
        ],
    })
    _write(tmp_path / "data" / "live-storms.json", {"n_active_storms": 0, "storms": []})
    _write(tmp_path / "data" / "live-tornadoes.json", {
        "scoring_tier": "tier1_ml", "n_active_storms": 1,
        "storms": [{"storm_id": "7", "tornado_probability": 0.0057, "risk_band": "high",
                    "lat": 35.1, "lon": -97.2}],
    })
    _write(tmp_path / "data" / "verification-summary.json", {"hazards": []})
    now = dt.datetime(2026, 10, 1, 22, 51)

    _publish_eq(tmp_path, "eq_fcst_20261001_1300", 0.046)
    tornado.render_cross_hazard_pages_from_artifacts(now)
    home = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "eq_fcst_20261001_1300" in home and "4.6%" in home

    # An earthquake-only cycle: nothing tornado-specific changes, the pages must still move.
    _publish_eq(tmp_path, "eq_fcst_20261001_2200", 0.0454)
    tornado.render_cross_hazard_pages_from_artifacts(now)
    home = (tmp_path / "index.html").read_text(encoding="utf-8")
    overview = (tmp_path / "live" / "index.html").read_text(encoding="utf-8")
    assert "eq_fcst_20261001_2200" in home and "eq_fcst_20261001_1300" not in home
    assert "4.5%" in home and "4.6%" not in home
    assert "4.5%" in overview and "4.6%" not in overview
    # The tornado-run inputs were read back from live-tornadoes.json (tier + storms).
    assert "ML (pre-trained gradient-boosted trees)" in home


def test_site_builder_renders_cross_hazard_pages_after_the_pulse_is_final(monkeypatch):
    bsa = _load("build_site_artifacts.py", "bsa_cross_hazard_test")
    body = inspect.getsource(bsa.build_site_artifacts)
    order = [body.index(call) for call in (
        "_ensure_live_publish_artifacts()",     # writes the final live-pulse.json
        "_render_cross_hazard_pages()",
        "_normalize_html_accessibility_labels()",  # post-processes every page, homepage included
    )]
    assert order == sorted(order)

    calls: list[str] = []
    stub = types.ModuleType("fetch_and_score_tornado")
    stub.render_cross_hazard_pages_from_artifacts = lambda: calls.append("rendered")
    monkeypatch.setitem(sys.modules, "fetch_and_score_tornado", stub)
    bsa._render_cross_hazard_pages()
    assert calls == ["rendered"]
    assert not hasattr(bsa, "sync_homepage_forecast_refs"), \
        "the id-only regex rewrite stamped new ids onto old numbers; it must not come back"
