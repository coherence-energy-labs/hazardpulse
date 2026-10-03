"""The prospective scorers report each model version's live record on its own."""
from __future__ import annotations

import datetime as dt
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_earthquake_live_record_is_split_by_model_version():
    eq = _script("eq_prospective_by_version", "scripts/score_earthquake_prospective.py")
    old = [{"model_version": "eq_coherence_v1_0", "issued_at": f"2026-09-0{i}T00:00:00Z", "auc": 0.6,
            "auc_active_cells": math.nan, "n_observed_events": 2, "brier": 0.002,
            "poisson_log_likelihood": -40.0, "uniform_log_likelihood": -10.0} for i in range(1, 4)]
    new = [{"model_version": "eq_operational_C0_v1-d28c62a2e35b", "issued_at": "2026-10-09T00:00:00Z",
            "auc": 0.95, "auc_active_cells": 0.8, "n_observed_events": 4, "brier": 0.001,
            "poisson_log_likelihood": -5.0, "uniform_log_likelihood": -15.0}]
    rec = eq.live_record_by_version(old + new)
    assert set(rec) == {"eq_coherence_v1_0", "eq_operational_C0_v1-d28c62a2e35b"}
    assert rec["eq_coherence_v1_0"]["n_matured_forecasts"] == 3
    assert rec["eq_coherence_v1_0"]["mean_auc"] == pytest.approx(0.6)
    assert rec["eq_coherence_v1_0"]["event_weighted_information_gain_per_event"] == pytest.approx(-90.0 / 6)
    c0 = rec["eq_operational_C0_v1-d28c62a2e35b"]
    assert c0["n_matured_forecasts"] == 1 and c0["mean_auc"] == pytest.approx(0.95)
    assert c0["event_weighted_information_gain_per_event"] == pytest.approx(10.0 / 4)
    assert c0["first_issued_at"] == "2026-10-09T00:00:00Z"


def _triple(version: str, issued: dt.datetime, y: list[int], p: list[float]):
    end = issued + dt.timedelta(hours=24)
    lab = {"issued_at": issued, "window_end": end, "y_true": np.array(y, float), "y_score": np.array(p, float),
           "y_raw": np.array(p, float), "calibrated": np.zeros(len(y), bool),
           "model_versions": np.array([version] * len(y), dtype=object), "n_reports_in_window": sum(y)}
    row = {"auc": 0.8, "brier": 0.01, "brier_skill_score": 0.1, "n_storms": len(y), "n_matched_storms": sum(y),
           "n_reports_in_window": sum(y), "scoring_tier": "tier1_v3", "window_end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    return ({"model_version": version}, lab, row)


def test_the_tornado_live_record_is_split_by_model_version():
    to = _script("to_prospective_by_version", "scripts/score_tornado_prospective.py")
    t0 = dt.datetime(2026, 10, 1, 0, 0)
    scored = [
        _triple("tornado_storm_v1_0", t0, [1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3]),
        _triple("tornado_storm_v1_0", t0 + dt.timedelta(hours=2), [0, 1, 0], [0.2, 0.6, 0.1]),
        _triple("tornado_v3-aaaaaaaaaaaa", t0 + dt.timedelta(hours=4), [0, 0, 1, 0, 0], [0.01, 0.02, 0.4, 0.03, 0.01]),
    ]
    s = to.summarize(scored, t0 + dt.timedelta(days=3))
    by = s["pooled_by_model_version"]
    assert set(by) == {"tornado_storm_v1_0", "tornado_v3-aaaaaaaaaaaa"}
    assert by["tornado_storm_v1_0"]["n_storm_forecasts"] == 7 and by["tornado_storm_v1_0"]["n_forecasts"] == 2
    assert by["tornado_v3-aaaaaaaaaaaa"]["n_storm_forecasts"] == 5 and by["tornado_v3-aaaaaaaaaaaa"]["n_forecasts"] == 1
    assert by["tornado_v3-aaaaaaaaaaaa"]["first_issued_at"] == "2026-10-01T04:00:00Z"
    assert s["pooled"]["n_storm_forecasts"] == 12      # the all-version pool is still there, as history
