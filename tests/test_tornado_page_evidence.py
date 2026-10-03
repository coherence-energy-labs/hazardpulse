"""The live tornado pages: the ledger keeps updating, and a storm's numbers are measured, not invented."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from hazardpulse.verification import evidence_pages as ep

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("tornado_page_evidence_live", ROOT / "scripts" / "fetch_and_score_tornado.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(ts: str, version: str, h: str) -> str:
    return json.dumps({"timestamp": ts, "model_version": version, "n_storms": 3, "top_probability": 0.1,
                       "prev_hash": "0" * 64, "hash": h * 64}) + "\n"


def test_the_ledger_keeps_updating_a_page_it_already_baked(live, tmp_path, monkeypatch):
    page = tmp_path / "verification" / "tornado" / "index.html"
    page.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "dist" / "verification" / "tornado" / "index.html", page)
    ledger = tmp_path / "tornado-ledger.jsonl"
    monkeypatch.setattr(live, "DIST", tmp_path)
    monkeypatch.setattr(live, "LEDGER_PATH", ledger)
    ledger.write_text(_entry("2026-10-02T00:00:00Z", "tornado_v3-aaaaaaaaaaaa", "a"), encoding="utf-8")
    live.render_verification_ledger()
    first = page.read_text(encoding="utf-8")
    assert "2026-10-02T00:00:00Z" in first and "tornado_v3-aaaaaaaaaaaa" in first
    # the second run is the one that silently did nothing from 2026-03-31: the page was already baked
    with ledger.open("a", encoding="utf-8") as f:
        f.write(_entry("2026-10-03T00:00:00Z", "tornado_v3-bbbbbbbbbbbb", "b"))
    live.render_verification_ledger()
    second = page.read_text(encoding="utf-8")
    assert "2026-10-03T00:00:00Z" in second and "tornado_v3-bbbbbbbbbbbb" in second
    assert second.count('class="ledger-row ledger-header"') == 1
    assert second.count("hash: " + "b" * 64) == 1
    # nothing outside the ledger blocks moved
    strip = lambda s: ep._marker_re("rows", "hp-ledger").sub("", ep._marker_re("chain", "hp-ledger").sub("", s))
    assert strip(first) == strip(second)


def test_a_page_without_the_ledger_markers_is_an_error_not_a_silent_no_op(live, tmp_path, monkeypatch):
    page = tmp_path / "verification" / "tornado" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>no markers</body></html>", encoding="utf-8")
    monkeypatch.setattr(live, "DIST", tmp_path)
    monkeypatch.setattr(live, "LEDGER_PATH", tmp_path / "none.jsonl")
    with pytest.raises(ep.PageBlockError):
        live.render_verification_ledger()


def _storm(v3: dict | None) -> dict:
    s = {"storm_id": "77", "lat": 35.2, "lon": -97.4, "valid_time": "20250506_180039 UTC", "risk_band": "high",
         "tornado_probability": 0.12, "motion_east": 12.0, "motion_south": -3.0, "mucape": 2500, "srh01": 250,
         "maxllaz": 0.012, "flash_rate": 30, "model_version": "tornado_v3-aaaaaaaaaaaa", "track_length": 5}
    if v3 is not None:
        s["v3"] = v3
    return s


_TABLE = {"bins": [{"lo": 0.0, "hi": 0.05, "n": 1460000, "pos": 400, "observed": 0.0003, "mean_forecast": 0.0004,
                    "observed_ci": [0.00025, 0.0003]},
                   {"lo": 0.05, "hi": 1.0, "n": 900, "pos": 120, "observed": 0.1333, "mean_forecast": 0.12,
                    "observed_ci": [0.112, 0.156]}]}


def test_a_v3_storm_shows_its_measured_outcome_rate_never_an_invented_one(live, monkeypatch):
    ev = {"model_version": "tornado_v3-aaaaaaaaaaaa", "reliability": _TABLE, "fallback": None,
          "test": {"auc": 0.97, "auc_ci": [0.96, 0.98]}}
    monkeypatch.setattr(live, "tornado_evidence", lambda: ev)
    v3 = {"model": "v3_w", "probability_60min": 0.12, "probability_30min": 0.07, "probability_90min": 0.15,
          "probability_ef2plus_60min": 0.02, "nws_warning": {"active": True, "minutes_since_issue": 4.0},
          "drivers": [{"input": "p_maxllaz", "label": "low-level rotation (max azimuthal shear)", "log_odds": 0.84},
                      {"input": "p_size", "label": "storm size", "log_odds": -0.21}]}
    html = live._render_storm_rows([_storm(v3)])
    assert "Historical analogs" not in html and "percentile (approx.)" not in html
    assert "Of the 900 storm observations of 2025" in html and "13.3%" in html
    assert "low-level rotation (max azimuthal shear) raises the score (+0.84 log-odds)" in html
    assert "storm size lowers the score (-0.21 log-odds)" in html
    assert "AUC 0.970 [0.960, 0.980]" in html
    assert "not an input to the served model" in html or "coherence" not in html.lower()
    # a probability in a range the test never populated: no rate is shown
    empty = {"bins": [{"lo": 0.0, "hi": 0.05, "n": 10, "pos": 0, "observed": 0.0, "mean_forecast": 0.01},
                      {"lo": 0.05, "hi": 1.0, "n": 0, "pos": 0, "observed": None, "mean_forecast": None}]}
    monkeypatch.setattr(live, "tornado_evidence", lambda: {**ev, "reliability": empty})
    html = live._render_storm_rows([_storm(v3)])
    assert "No 2025 storm observation was scored in this range" in html


def test_a_legacy_storm_has_no_invented_analog_either(live, monkeypatch):
    monkeypatch.setattr(live, "tornado_evidence", lambda: None)
    s = _storm(None)
    s["model_version"] = "tornado_storm_v1_0"
    html = live._render_storm_rows([s])
    assert "Historical analogs" not in html and "Observed rate" not in html
    assert "legacy tier; no final test is bound to it" in html
