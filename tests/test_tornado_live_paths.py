"""Live tornado scorer paths: analysis staleness, published band, susceptibility map."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fast_live_tornado", ROOT / "scripts" / "fetch_and_score_tornado.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_analyses_are_newest_first_and_never_older_than_training_saw(live):
    from hazardpulse.tornado.definitive_model import MAX_ANALYSIS_AGE_H

    now = dt.datetime(2026, 10, 1, 22, 16)
    c = live.live_analysis_candidates(now)
    assert c == [("20261001", 22), ("20261001", 21), ("20261001", 20)]
    for d, h in c:
        t = dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h)
        assert 0 <= (now - t).total_seconds() <= MAX_ANALYSIS_AGE_H * 3600
    # across midnight
    c2 = live.live_analysis_candidates(dt.datetime(2026, 10, 2, 1, 5))
    assert c2 == [("20261002", 1), ("20261002", 0), ("20261001", 23)]


def test_published_band_follows_the_published_probability(live):
    scored = [
        {"tornado_probability": 0.0057, "risk_band": "high"},      # live 2026-10-01 witness
        {"tornado_probability": 0.62, "risk_band": "minimal"},
        {"tornado_probability": 0.2, "risk_band": "very_high"},
    ]
    live.refresh_risk_bands(scored)
    assert [s["risk_band"] for s in scored] == ["minimal", "very_high", "moderate"]


def test_susceptibility_reads_the_real_hrrr_fields(live, tmp_path, monkeypatch):
    from test_tornado_torsion import _synthetic_atmosphere

    from hazardpulse.data.hrrr import GRID_LATS, GRID_LONS

    monkeypatch.setattr(live, "DIST", tmp_path)
    atm = _synthetic_atmosphere()  # environment peaks at grid cell (15, 35)
    top = live.compute_day_ahead_susceptibility(atm, None, dt.datetime(2026, 5, 6, 18), "20260506 18Z")
    probs = [c["probability"] for c in top]
    # Not the inert sigmoid(-2) = 0.1192 everywhere ...
    assert max(probs) > 0.2 and len(set(probs)) > 1
    # ... and the top cell is the high-STP cell, not the first cell in grid order.
    best = top[0]
    assert abs(best["lat"] - float(GRID_LATS[15])) <= 0.72 * 3
    assert abs(best["lon"] - float(GRID_LONS[35])) <= 0.94 * 4
    assert best["stp"] > 0
