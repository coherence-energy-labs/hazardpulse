"""A published band must contain the probability published beside it (tornado audit item 6, 2026-10-05).

Once PR #22 stopped the inflating calibrator, the tornado scorer's 02:05Z run published the model's own 60-minute
probability (0.0006) beside the Venn-Abers pair of the same model (0.0007-0.0007), so the live pulse said
"0.06%, range 0.07%-0.07%" -- and tests/test_site_integrity.py, which every deploy runs, failed on main.
The pair is two calibrated estimates, not a coverage interval, and nothing keeps a Platt probability inside it:
971 of 1,602 v3 storm forecasts had their p60 outside their own pair.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("band_live_tornado", ROOT / "scripts" / "fetch_and_score_tornado.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bsa():
    spec = importlib.util.spec_from_file_location("band_bsa", ROOT / "scripts" / "build_site_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _storm(sid, p, lo, hi):
    return {"storm_id": sid, "lat": 35.0, "lon": -97.0, "tornado_probability": p, "risk_band": "minimal",
            "confidence_lo": lo, "confidence_hi": hi}


def test_a_band_that_excludes_the_published_probability_is_withheld_and_one_that_contains_it_is_kept(live):
    scored = [_storm("a", 0.0006, 0.0007, 0.0007),       # the 02:05Z headline
              _storm("b", 0.10, 0.08, 0.12),
              _storm("c", 0.30, 0.05, 0.25),
              _storm("d", 0.01, None, None)]
    assert live.withhold_bands_excluding_probability(scored) == 2
    a, b, c, d = scored
    assert a["confidence_lo"] is None and a["confidence_hi"] is None and a["band_withheld"]
    assert (b["confidence_lo"], b["confidence_hi"]) == (0.08, 0.12) and "band_withheld" not in b
    assert c["confidence_lo"] is None and c["band_withheld"]
    assert d["confidence_lo"] is None and "band_withheld" not in d


def test_the_pulse_never_carries_a_band_that_excludes_its_probability(live, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "data" / "live-pulse.json").write_text(json.dumps({"hazards": [{"key": "to"}]}), encoding="utf-8")
    monkeypatch.setattr(live, "DIST", dist)
    monkeypatch.setattr(live, "LEDGER_PATH", dist / "data" / "tornado-ledger.jsonl")
    scored = [_storm("a", 0.0006, 0.0007, 0.0007)]
    live.withhold_bands_excluding_probability(scored)
    live.write_outputs(scored, dt.datetime(2026, 10, 5, 2, 5))
    to = json.loads((dist / "data" / "live-pulse.json").read_text(encoding="utf-8"))["hazards"][0]
    assert to["probability"] == 0.0006 and to["conf_lo"] is None and to["conf_hi"] is None


def test_the_site_build_withholds_a_committed_pulse_band_that_excludes_its_probability():
    bsa = _bsa()
    pulse = {"hazards": [
        {"key": "to", "probability": 0.0006, "conf_lo": 0.0007, "conf_hi": 0.0007},
        {"key": "eq", "probability": 0.05, "conf_lo": 0.04, "conf_hi": 0.06},
        {"key": "hu", "probability": 0.2, "conf_lo": None, "conf_hi": None}]}
    assert bsa._withhold_pulse_bands_excluding_probability(pulse) is True
    to, eq, hu = pulse["hazards"]
    assert to["conf_lo"] is None and to["conf_hi"] is None
    assert (eq["conf_lo"], eq["conf_hi"]) == (0.04, 0.06)
    assert bsa._withhold_pulse_bands_excluding_probability(pulse) is False      # a fixed point
