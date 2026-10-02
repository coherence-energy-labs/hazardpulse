"""Metrics, calibration and serving parity for the definitive tornado model."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hazardpulse.tornado import definitive_model as dm


def _pairwise_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def test_auc_scores_ties_one_half():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
    assert dm.compute_auc(y, np.ones(8)) == 0.5
    rng = np.random.RandomState(0)
    for _ in range(20):
        y = (rng.rand(300) < 0.2).astype(float)
        s = rng.randint(0, 6, 300).astype(float)  # heavy ties, like tree leaves
        assert dm.compute_auc(y, s) == pytest.approx(_pairwise_auc(y, s), abs=1e-12)


def test_old_sequential_auc_was_order_dependent_on_ties():
    """Witness for the replaced implementation: same scores, different AUC."""
    def old_auc(y, s):
        order = np.argsort(-s, kind="stable")
        ys = y[order]
        n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
        tp = fp = 0
        auc = tpr_p = fpr_p = 0.0
        for v in ys:
            tp += v == 1
            fp += v == 0
            tpr, fpr = tp / n_pos, fp / n_neg
            auc += (fpr - fpr_p) * (tpr + tpr_p) / 2
            tpr_p, fpr_p = tpr, fpr
        return auc

    # A scorer that says NOTHING (constant) -- the old walk reported a perfect
    # or a perfectly wrong model depending only on the input row order.
    s = np.ones(4)
    assert old_auc(np.array([1.0, 1, 0, 0]), s) == 1.0
    assert old_auc(np.array([0.0, 0, 1, 1]), s) == 0.0
    assert dm.compute_auc(np.array([1.0, 1, 0, 0]), s) == 0.5
    assert dm.compute_auc(np.array([0.0, 0, 1, 1]), s) == 0.5
    # Partial ties: the new statistic is order-free and equals the pairwise count.
    y = np.array([1.0, 0, 0, 1, 0])
    s = np.array([1.0, 1.0, 0.0, 2.0, 2.0])
    assert old_auc(y, s) != old_auc(y[::-1].copy(), s[::-1].copy())
    assert dm.compute_auc(y, s) == dm.compute_auc(y[::-1].copy(), s[::-1].copy())
    assert dm.compute_auc(y, s) == pytest.approx(_pairwise_auc(y, s))


def _small_model(seed=0, n=600):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6).astype(np.float32)
    logit = 1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3] - 2.0
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    gbt = dm.GradientBoostedTrees(n_trees=30, max_depth=3, min_samples_leaf=10)
    gbt.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])
    return gbt, X, y


def test_batch_tree_prediction_matches_row_walk():
    gbt, X, _ = _small_model()
    for tree in gbt.trees[:10]:
        rows = np.array([gbt._predict_tree_row(tree, x) for x in X], dtype=np.float32)
        assert np.array_equal(gbt._predict_tree_batch(tree, X), rows)


def test_platt_recovers_a_known_mapping():
    rng = np.random.RandomState(1)
    F = rng.randn(20000) * 2.0
    a_true, b_true = 0.6, -3.0
    y = (rng.rand(20000) < 1 / (1 + np.exp(-(a_true * F + b_true)))).astype(float)
    cal = dm.fit_platt(F, y)
    assert cal["a"] == pytest.approx(a_true, abs=0.08)
    assert cal["b"] == pytest.approx(b_true, abs=0.12)
    p = dm.apply_calibration(F, cal)
    assert p.mean() == pytest.approx(y.mean(), rel=0.02)  # MLE intercept matches the base rate


def test_balanced_training_overstates_probability_until_calibrated():
    """The served-number defect: sigmoid(F) of a class-balanced model sits at a 50/50 prior."""
    rng = np.random.RandomState(2)
    n = 6000
    X = rng.randn(n, 4).astype(np.float32)
    y = (rng.rand(n) < 1 / (1 + np.exp(-(1.2 * X[:, 0] - 4.0)))).astype(np.float32)
    gbt = dm.GradientBoostedTrees(n_trees=40, max_depth=3, min_samples_leaf=20)
    gbt.fit(X[:3000], y[:3000], X_val=X[3000:4500], y_val=y[3000:4500])
    raw = gbt.predict_proba(X[4500:])
    cal = dm.fit_platt(gbt.decision_function(X[3000:4500]), y[3000:4500])
    p = dm.apply_calibration(gbt.decision_function(X[4500:]), cal)
    base = float(y[4500:].mean())
    assert raw.mean() > 5 * base                     # uncalibrated: wildly too high
    assert abs(p.mean() - base) < 0.5 * base + 0.005  # calibrated: near the base rate
    assert dm.compute_bss(y[4500:], p) > dm.compute_bss(y[4500:], raw)


def test_day_clustered_interval_is_wider_on_clustered_data():
    rng = np.random.RandomState(4)
    days = np.repeat(np.arange(40), 50)
    day_effect = rng.randn(40)[days]  # shared within a day
    y = (rng.rand(len(days)) < 1 / (1 + np.exp(-(day_effect - 1.5)))).astype(float)
    s = day_effect + 0.3 * rng.randn(len(days))
    iid = dm.bootstrap_auc_ci(y, s, n_boot=300)
    clu = dm.cluster_bootstrap_auc_ci(y, s, days, n_boot=300)
    assert clu["n_clusters"] == 40
    assert (clu["ci_hi"] - clu["ci_lo"]) > 1.5 * (iid["ci_hi"] - iid["ci_lo"])


def test_saved_payload_scores_exactly_like_the_model(tmp_path):
    gbt, X, y = _small_model(seed=5)
    norm = dm.FeatureNormalizer()
    Xn = norm.fit_transform(X)
    gbt2 = dm.GradientBoostedTrees(n_trees=20, max_depth=3, min_samples_leaf=10)
    gbt2.fit(Xn[:400], y[:400], X_val=Xn[400:], y_val=y[400:])
    cal = dm.fit_platt(gbt2.decision_function(Xn[400:]), y[400:])
    names = [f"f{i}" for i in range(X.shape[1])]
    path = tmp_path / "m.json"
    dm.save_model(gbt2, norm, names, path, calibration=cal, provenance={"k": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["calibration"]["method"] == "platt"
    expect = dm.apply_calibration(gbt2.decision_function(Xn), cal)
    for i in range(0, len(X), 37):
        p, calibrated = dm.predict_proba_from_payload(
            payload, {n: float(X[i, k]) for k, n in enumerate(names)}
        )
        assert calibrated
        assert p == pytest.approx(float(expect[i]), abs=2e-6)


def test_serving_routes_tied_values_exactly_like_training(tmp_path):
    """Values sitting ON a split threshold (refc = -10 dBZ is the modal 'no
    echo' value) must take the same branch live as in training. Every row of
    the training set is scored both ways; one flipped split moves F by
    lr * leaf (>= ~1e-3), far above the tolerance."""
    rng = np.random.RandomState(9)
    n = 1500
    refc = rng.choice([-10.0, -10.0, -10.0, 0.0, 20.0, 35.0, 45.0, 55.0], size=n)
    cape = rng.choice([0.0, 250.0, 1000.0, 2500.0], size=n) + rng.rand(n)
    srh = rng.gamma(2.0, 60.0, size=n)
    X = np.column_stack([refc, cape, srh]).astype(np.float32)
    logit = 0.04 * (refc + 10) + 0.0008 * cape + 0.01 * srh - 4.0
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    norm = dm.FeatureNormalizer()
    Xn = norm.fit_transform(X)
    gbt = dm.GradientBoostedTrees(n_trees=40, max_depth=3, min_samples_leaf=15)
    gbt.fit(Xn[:1000], y[:1000], X_val=Xn[1000:], y_val=y[1000:])
    cal = dm.fit_platt(gbt.decision_function(Xn[1000:]), y[1000:])
    names = ["hrrr_refc", "mlcape", "srh01"]
    path = tmp_path / "m.json"
    dm.save_model(gbt, norm, names, path, calibration=cal)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expect = dm.apply_calibration(gbt.decision_function(Xn), cal)
    got = np.array([
        dm.predict_proba_from_payload(payload, {k: float(X[i, j]) for j, k in enumerate(names)})[0]
        for i in range(n)
    ])
    assert np.abs(got - expect).max() < 1e-5


def test_payload_identity_is_bound_to_the_weights(tmp_path):
    gbt, X, y = _small_model(seed=8)
    norm = dm.FeatureNormalizer()
    norm.fit(X)
    names = [f"f{i}" for i in range(6)]
    legacy = tmp_path / "legacy.json"
    dm.save_model(gbt, norm, names, legacy)
    assert dm.model_version_of_payload(legacy) == dm.LEGACY_MODEL_VERSION
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    dm.save_model(gbt, norm, names, a, calibration={"method": "platt", "a": 1.0, "b": -4.0})
    dm.save_model(gbt, norm, names, b, calibration={"method": "platt", "a": 1.0, "b": -4.1})
    va, vb = dm.model_version_of_payload(a), dm.model_version_of_payload(b)
    assert va.startswith("tornado_gbt_v2-") and vb.startswith("tornado_gbt_v2-")
    assert va != vb  # any change to the served bytes is a new identity


def test_legacy_payload_is_flagged_uncalibrated(tmp_path):
    gbt, X, y = _small_model(seed=6)
    norm = dm.FeatureNormalizer()
    norm.fit(X)
    path = tmp_path / "legacy.json"
    dm.save_model(gbt, norm, [f"f{i}" for i in range(6)], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _, calibrated = dm.predict_proba_from_payload(payload, {})
    assert calibrated is False
