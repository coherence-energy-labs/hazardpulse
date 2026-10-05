"""The live tornado scorer's v3 tier: routing between the +W model and the fallback, outputs."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402
from hazardpulse.tornado import storm_features as sf  # noqa: E402
from hazardpulse.tornado import v3_serving as vs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fast_live_tornado_v3", ROOT / "scripts" / "fetch_and_score_tornado.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(names, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(3000, len(names))
    y = (X[:, 1] + (X[:, -2] if names[-1] == "w_minutes_since_issue" else 0) + rng.randn(3000) > 1.5).astype(int)
    bst = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": seed}, lgb.Dataset(X, y), 20)
    return lp.export_booster(bst, list(names), calibration={"a": 1.0, "b": -1.0}, provenance={})


@pytest.fixture(scope="module")
def suite():
    lo, hi = sf.BLOCKS["P"][0], sf.BLOCKS["H80"][1]
    base = list(sf.FEATURE_NAMES[lo:hi])
    main = _payload(base + list(vs.W_NAMES), 1)
    return vs.V3Suite(main=main, fallback=_payload(base, 2),
                      products={"p30": _payload(base + list(vs.W_NAMES), 3)})


def _steps():
    t0 = dt.datetime(2025, 5, 6, 18, 0, 39, tzinfo=dt.timezone.utc)
    steps = []
    for k in range(3):
        t = t0 + dt.timedelta(minutes=30 * k)
        steps.append({"valid_time": t.strftime("%Y%m%d_%H%M%S UTC"), "storms": [
            {"id": "11", "lat": 35.0 + 0.05 * k, "lon": -97.5 + 0.1 * k, "ps": 60, "ps_tor": 20 + 10 * k,
             "maxllaz": 0.004 + 0.001 * k, "size": 300, "motion_east": 12, "motion_south": -8},
            {"id": "22", "lat": 33.0, "lon": -90.0, "ps": 10, "ps_tor": 1, "maxllaz": 0.001, "size": 80,
             "motion_east": 10, "motion_south": -2}]})
    return steps


class _Inputs:
    def __init__(self, m):
        self._m = m

    def matrix(self):
        return self._m


def test_v3_tier_uses_the_warnings_model_when_the_feed_answers(live, suite, monkeypatch):
    calls = {}

    def fake(lats, lons, times, **kw):
        calls["n"] = len(lats)
        return _Inputs(np.array([[1.0, 12.0], [0.0, np.nan]], np.float32))
    monkeypatch.setattr(live.nws_live, "live_tor_warning_inputs", fake)
    out = live.score_storms(_steps(), None, None, None, dt.datetime(2025, 5, 6, 19, 10),
                            scoring_tier="tier1_v3", v3_suite=suite)
    assert calls["n"] == 2 and len(out) == 2
    by = {s["storm_id"]: s for s in out}
    assert by["11"]["v3"]["model"] == "v3_w" and by["11"]["v3"]["nws_warning"]["active"] is True
    assert by["11"]["v3"]["nws_warning"]["minutes_since_issue"] == 12.0
    assert by["22"]["v3"]["nws_warning"]["active"] is False
    assert by["11"]["model_version"] == lp.model_version(suite.main)
    assert by["11"]["v3"]["probability_30min"] is not None                  # products served with the feed
    assert by["11"]["v3"]["probability_30min"] <= by["11"]["v3"]["probability_60min"] + 1e-12
    # score_storms keeps the model's full precision (the trust layer and the live record read it);
    # main() rounds for display afterwards (round_published)
    assert by["11"]["tornado_probability"] == min(by["11"]["v3"]["probability_60min"], 0.99)
    live.round_published(out)
    assert by["11"]["tornado_probability"] == round(by["11"]["v3"]["probability_60min"], 4)
    assert by["11"]["v3"]["drivers"] and by["11"]["v3"]["hrrr_analysis"] is None   # no HRRR in this test


def test_v3_tier_falls_back_to_the_no_warnings_model_when_the_feed_fails(live, suite, monkeypatch):
    def broken(*a, **k):
        raise live.nws_live.NwsLiveFetchError("api.weather.gov timed out")
    monkeypatch.setattr(live.nws_live, "live_tor_warning_inputs", broken)
    out = live.score_storms(_steps(), None, None, None, dt.datetime(2025, 5, 6, 19, 10),
                            scoring_tier="tier1_v3", v3_suite=suite)
    for s in out:
        assert s["v3"]["model"] == "v3" and s["v3"]["nws_warning"] is None
        assert "NwsLiveFetchError" in s["v3"]["nws_feed_error"]
        assert s["v3"]["probability_30min"] is None                      # never guessed without the feed
        assert s["model_version"] == lp.model_version(suite.fallback)
