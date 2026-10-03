"""Tie-aware ranking metrics (hazardpulse.core.metrics) and the prospective scorers
that publish them. Regression for the 2026-10 finding that the per-sample argsort walk
credited tied cells by row order (earthquake mean AUC 0.6972 published vs 0.6564)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.core.metrics import average_precision, roc_auc

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_constant_scorer_is_exactly_one_half_in_any_row_order():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    s = np.zeros_like(y)
    for perm in (np.arange(8), np.arange(8)[::-1], np.array([1, 2, 4, 5, 6, 0, 3, 7])):
        assert roc_auc(y[perm], s[perm]) == 0.5


def test_hand_computed_values_with_ties():
    # scores: pos {0.9, 0.5}, neg {0.5, 0.1}: pairs (0.9>0.5)=1, (0.9>0.1)=1, (0.5=0.5)=.5, (0.5>0.1)=1
    y = [1, 1, 0, 0]
    s = [0.9, 0.5, 0.5, 0.1]
    assert roc_auc(y, s) == pytest.approx(3.5 / 4)
    # AP: threshold 0.9 -> P=1, R=.5; threshold 0.5 -> P=2/3, R=1 => .5*1 + .5*(2/3)
    assert average_precision(y, s) == pytest.approx(0.5 + 1.0 / 3.0)


def test_metrics_are_invariant_to_row_order_on_heavily_tied_data():
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.01).astype(float)
    s = np.where(rng.random(3000) < 0.97, 0.0, rng.random(3000))   # 97% tied at 0.0
    ref_auc, ref_ap = roc_auc(y, s), average_precision(y, s)
    for _ in range(5):
        p = rng.permutation(y.size)
        assert roc_auc(y[p], s[p]) == pytest.approx(ref_auc, abs=1e-12)
        assert average_precision(y[p], s[p]) == pytest.approx(ref_ap, abs=1e-12)


def test_matches_sklearn_oracle():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(5, 300))
        s = rng.integers(0, 5, n) / 4.0
        y = (rng.random(n) < 0.3).astype(float)
        if 0 < y.sum() < n:
            assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
            assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)


def test_degenerate_inputs():
    assert np.isnan(roc_auc([0, 0], [0.1, 0.2]))
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1])
    with pytest.raises(ValueError):
        roc_auc([0, 2], [0.1, 0.2])


def _tied_grid_case():
    """A grid like the live one: a few active cells, the rest tied at 0.0, and the
    target events falling in the tied inactive block (where the walk was order-luck)."""
    n = 400
    y = np.zeros(n)
    s = np.zeros(n)
    s[:5] = [0.9, 0.8, 0.7, 0.6, 0.5]       # active cells, all misses
    y[[10, 200, 399]] = 1.0                  # events in inactive (tied) cells
    return y, s


@pytest.mark.parametrize("rel,name", [
    ("scripts/score_earthquake_prospective.py", "hp_sep_rank_test"),
    ("scripts/score_hurricane_prospective.py", "hp_shp_rank_test"),
])
def test_prospective_scorers_give_ties_half_credit_regardless_of_order(rel, name):
    mod = _script(name, rel)
    y, s = _tied_grid_case()
    # Exact value: each positive ties 392 negatives at 0.0 and loses to 5 -> (392/2)/397.
    expected = (392 / 2) / 397
    rng = np.random.default_rng(0)
    for _ in range(4):
        p = rng.permutation(y.size)
        assert mod.compute_auc(y[p], s[p]) == pytest.approx(expected, abs=1e-12)


def test_earthquake_scorer_reports_active_cell_skill(tmp_path):
    mod = _script("hp_sep_active_test", "scripts/score_earthquake_prospective.py")
    from hazardpulse.earthquake.coherence_engine import grid_cell_to_latlon

    artifact = {
        "forecast_id": "eq_fcst_20260201_0000",
        "issued_at": "2026-02-01T00:00:00Z",
        "forecast_horizon_days": 30,
        "forecast_domain": {"n_lat": 3, "n_lon": 3, "default_probability": 0.0},
        "active_cells": [
            {"row": 0, "col": 0, "probability": 0.8},
            {"row": 1, "col": 1, "probability": 0.1},
            {"row": 2, "col": 2, "probability": 0.3},
        ],
    }
    observed = []
    for i, (r, c) in enumerate([(1, 1), (0, 2)]):    # one active (low-scored) hit, one inactive hit
        lat, lon = grid_cell_to_latlon(r, c)
        observed.append({"time": "2026-02-10T00:00:00Z", "latitude": lat, "longitude": lon,
                         "depth": 10.0, "mag": 6.1, "id": f"e{i}"})
    res = mod.score_single_forecast(artifact, observed, tmp_path)
    assert res["n_events_in_active_cells"] == 1
    # among active cells the hit (0.1) is ranked below both misses -> AUC 0
    assert res["auc_active_cells"] == pytest.approx(0.0)
    # 7 negatives (0.8, 0.3, five 0.0). Positive (1,1) at 0.1 beats the five zeros;
    # positive (0,2) at 0.0 ties them (half credit each).
    assert res["auc"] == pytest.approx((5 + 2.5) / (2 * 7))
