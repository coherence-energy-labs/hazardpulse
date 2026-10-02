"""The tornado lab's statistics: sampling weights, out-of-fold calibration, paired comparison."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("tornado_lab_test", REPO / "scripts" / "audit_20261001" / "tornado_lab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_sample_keeps_every_positive_and_weights_back_to_the_population(lab):
    y = np.zeros(10_000, np.int8)
    y[::100] = 1                      # 100 positives
    y[5::1000] = -1                   # rows excluded by a learning-curve subset
    rows, w = lab.train_sample(None, y, 10, seed=0)
    assert set(np.flatnonzero(y == 1)) <= set(rows)
    assert not np.isin(np.flatnonzero(y == -1), rows).any()
    n_neg = int((y == 0).sum())
    assert w[y[rows] == 0].sum() == pytest.approx(n_neg)       # weighted negatives = the population's
    assert (w[y[rows] == 1] == 1.0).all()


def test_out_of_fold_calibration_never_sees_its_own_day(lab):
    rng = np.random.RandomState(0)
    days = np.repeat(np.arange(20230101, 20230161), 400)
    s = rng.randn(len(days))
    y = (rng.rand(len(days)) < 1 / (1 + np.exp(-(s - 3)))).astype(np.int8)
    for kind in ("platt", "venn_abers"):
        base = lab.out_of_fold_calibrated(kind, s, y, days)
        y2 = y.copy()
        target = days == 20230110
        y2[target] = 1 - y2[target]                              # rewrite one day's truth entirely
        moved = lab.out_of_fold_calibrated(kind, s, y2, days)
        assert np.array_equal(base[target], moved[target]), kind  # that day's forecasts cannot move
        assert not np.array_equal(base[~target], moved[~target])  # ...while other folds' calibrators do


def test_block_w_is_appended_after_the_store_columns_and_only_when_asked(lab):
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    W = np.full((10, 2), np.nan, np.float32)
    W[3] = [1.0, 12.5]
    cols = lab.cols_for(["W", "p_ps_tor"])
    assert cols == [lab.FIDX["p_ps_tor"]]
    rows = np.array([2, 3])
    with_w = lab.matrix(np.zeros((10, len(lab.NAMES)), np.float32), W, rows, cols, True)
    assert with_w.shape == (2, 1 + len(lab.W_NAMES)) and with_w[1, 1:].tolist() == [1.0, 12.5]
    assert lab.matrix(X, W, rows, [0, 2], False).tolist() == [[8.0, 10.0], [12.0, 14.0]]


def test_compare_is_zero_for_identical_forecasts_and_signed_for_a_better_one(lab, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LAB", tmp_path)
    monkeypatch.setattr(lab, "OUT", tmp_path)
    rng = np.random.RandomState(1)
    n = 6000
    days = np.repeat(np.arange(20240101, 20240131), n // 30)
    y = (rng.rand(n) < 0.05).astype(np.int8)
    good = np.clip(0.05 + 0.5 * y + 0.05 * rng.randn(n), 0.001, 0.999)
    noise = np.clip(0.05 + 0.05 * rng.randn(n), 0.001, 0.999)
    np.lib.format.open_memmap(tmp_path / "dev_Y.npy", mode="w+", dtype=np.int8,
                              shape=(n, len(lab.LIDX)))[:] = y[:, None]
    np.lib.format.open_memmap(tmp_path / "dev_X.npy", mode="w+", dtype=np.float32, shape=(n, 1))
    np.savez(tmp_path / "dev_meta.npz", day=days)
    (tmp_path / "preds").mkdir()
    np.save(tmp_path / "preds" / "a_dev.npy", noise.astype(np.float32))
    np.save(tmp_path / "preds" / "b_dev.npy", good.astype(np.float32))
    same = lab.compare("a", "a", "dev", n_boot=50)
    assert same["delta_auc"] == 0 and same["delta_auc_ci"] == [0.0, 0.0] and same["delta_brier"] == 0
    better = lab.compare("a", "b", "dev", n_boot=50)
    assert better["delta_auc_ci"][0] > 0.3 and better["delta_brier_ci"][1] < 0
    assert json.loads((tmp_path / "compare_b_vs_a_dev_storm_60.json").read_text())["b"] == "b"
