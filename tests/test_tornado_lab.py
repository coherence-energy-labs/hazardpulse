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


def test_balanced_weights_give_both_classes_equal_total_weight(lab):
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
    w = lab.balanced_weights(y)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    assert (w[y == 0] == 1.0).all() and (w[y == 1] == 4.0).all()


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


def test_stress_strata_follow_their_declared_definitions(lab):
    n = 8
    names = lab.NAMES
    X = np.zeros((n, len(names)), np.float32)
    X[:, lab.FIDX["p_size"]] = [10, 20, 30, 40, 50, 60, 70, 80]
    meta = {"lat": np.array([35, 42, 33, 45, 35, 35, 35, 35], float),
            "lon": np.array([-100, -88, -86, -120, -100, -100, -100, -100], float),
            "day": np.array([20240115, 20240415, 20240715, 20241015] * 2),
            # 23:00 UTC at -100 E is 16:20 solar (day); 08:00 UTC is 01:20 solar (night)
            "t": np.array([23 * 3600] * 4 + [8 * 3600] * 4),
            "analysis": np.array([0, 3, 103, -1, 0, 0, 0, 0])}
    g = lab.stress_groups(X, meta)
    assert g["region_plains"].tolist() == [True, False, False, False, True, True, True, True]
    assert g["region_midwest"][1] and g["region_southeast"][2] and g["region_elsewhere"][3]
    assert g["season_DJF"][0] and g["season_MAM"][1] and g["season_JJA"][2] and g["season_SON"][3]
    assert g["local_day"][:4].all() and g["local_night"][4:].all()
    assert g["analysis_9km"][:2].all() and g["analysis_80km_only"][2] and g["analysis_none"][3]
    assert g["size_small"].sum() + g["size_mid"].sum() + g["size_large"].sum() == n
    for k in ("region", "season", "local", "size", "analysis"):   # each family partitions the rows
        fam = [v for name, v in g.items() if name.startswith(k)]
        assert (np.sum(fam, axis=0) == 1).all(), k


def test_day_bootstrap_auc_by_quadratic_form_equals_resampling_the_rows(lab):
    """The fast compare must give, draw for draw, the AUC of the actually resampled rows."""
    rng = np.random.RandomState(3)
    days = np.repeat(np.arange(12), rng.randint(20, 60, size=12))
    y = (rng.rand(len(days)) < 0.15).astype(np.int8)
    s = np.round(rng.randn(len(days)) + 1.5 * y, 1)        # rounded: ties must count 1/2
    uniq, day_idx = np.unique(days, return_inverse=True)
    U, P, N = lab.day_pair_matrix(y, s, day_idx, len(uniq))
    M = lab.day_bootstrap_counts(len(uniq), 25, seed=9)
    for m in M:
        rows = np.concatenate([np.flatnonzero(day_idx == d) for d in range(len(uniq)) for _ in range(int(m[d]))])
        brute = lab.dm.compute_auc(y[rows].astype(np.float64), s[rows])
        fast = (m @ U @ m) / ((m @ P) * (m @ N))
        assert fast == pytest.approx(brute, abs=1e-12)
    ones = np.ones(len(uniq))
    assert (ones @ U @ ones) / ((ones @ P) * (ones @ N)) == pytest.approx(lab.dm.compute_auc(y.astype(float), s), abs=1e-12)


def test_nws_bar_matches_the_false_alarm_rate_and_scores_hits(lab, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LAB", tmp_path)
    monkeypatch.setattr(lab, "OUT", tmp_path)
    rng = np.random.RandomState(4)
    n = 20000
    days = np.repeat(np.arange(20240101, 20240141), n // 40)
    y = (rng.rand(n) < 0.02).astype(np.int8)
    p = np.clip(0.02 + 0.6 * y * rng.rand(n) + 0.05 * rng.rand(n), 0, 1)       # a skilful model
    warned = ((y == 1) & (rng.rand(n) < 0.4)) | ((y == 0) & (rng.rand(n) < 0.01))
    np.lib.format.open_memmap(tmp_path / "dev_Y.npy", mode="w+", dtype=np.int8, shape=(n, len(lab.LIDX)))[:] = y[:, None]
    Wm = np.lib.format.open_memmap(tmp_path / "dev_W.npy", mode="w+", dtype=np.float32, shape=(n, 2))
    Wm[:, 0] = warned
    Wm[:, 1] = np.nan
    Wm.flush()
    np.lib.format.open_memmap(tmp_path / "dev_X.npy", mode="w+", dtype=np.float32, shape=(n, 1))
    np.savez(tmp_path / "dev_meta.npz", day=days)
    (tmp_path / "preds").mkdir()
    np.save(tmp_path / "preds" / "m_dev.npy", p.astype(np.float32))
    r = lab.nws_bar("m", "dev", n_boot=100)
    assert abs(r["model_pofd"] - r["nws_pofd"]) < 2e-4                 # same false-alarm rate
    assert r["nws_pod"] == pytest.approx(warned[y == 1].mean())
    assert r["delta_pod"] > 0.3 and r["delta_pod_ci"][0] > 0          # the skilful model wins, provably


def test_final_refit_rows_skip_a_split_wholly_inside_the_held_out_year(lab):
    """LOYO holding out 2023 excludes every row of the validation split (all 2023); that split
    must be skipped, not divide by zero (the first final run crashed here before reading 2025)."""
    rng = np.random.RandomState(0)

    def part(days, n=4000):
        X = rng.randn(n, len(lab.NAMES)).astype(np.float32)
        Y = np.zeros((n, len(lab.LIDX)), np.int8)
        Y[rng.rand(n) < 0.05, lab.LIDX["storm_60"]] = 1
        W = np.full((n, 2), np.nan, np.float32)
        return X, Y, {"day": rng.choice(days, size=n)}, W
    parts = {"train": part([20210501, 20220601]), "val": part([20230501]), "dev": part([20240501])}
    cols = lab.cols_for(["P"])
    Xc, yc, wc = lab._rows_for(parts, {2021, 2022, 2024}, 30, 0, cols, False, lab.LIDX["storm_60"])
    n_pos_expected = sum(int(p[1][:, lab.LIDX["storm_60"]].sum()) for k, p in parts.items() if k != "val")
    assert int(yc.sum()) == n_pos_expected and Xc.shape[1] == len(cols)
    assert wc[yc == 1].sum() == pytest.approx(wc[yc == 0].sum())


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
