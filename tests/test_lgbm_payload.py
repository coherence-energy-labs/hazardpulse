"""The LightGBM payload must score exactly as LightGBM does -- every branch type, NaN and zero."""
from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402


def _data(seed=0, n=6000):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    X[rng.rand(n) < 0.15, 1] = np.nan          # a feature with missing values (missing_type NaN)
    X[rng.rand(n) < 0.30, 2] = 0.0             # many exact zeros
    X[:, 3] = np.round(X[:, 3], 1)             # ties on thresholds
    logit = 1.5 * X[:, 0] - np.nan_to_num(X[:, 1], nan=2.0) + 0.8 * (X[:, 2] == 0) + 0.5 * X[:, 3]
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X.astype(np.float32), y


@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_payload_equals_lightgbm_raw_scores(tmp_path, zero_as_missing):
    X, y = _data()
    bst = lgb.train({"objective": "binary", "num_leaves": 31, "min_data_in_leaf": 20, "verbose": -1,
                     "zero_as_missing": zero_as_missing, "deterministic": True, "seed": 1},
                    lgb.Dataset(X, y), 60)
    names = [f"f{i}" for i in range(X.shape[1])]
    payload = lp.export_booster(bst, names, calibration={"a": 1.0, "b": 0.0}, provenance={"test": True})
    Xt, _ = _data(seed=5, n=3000)
    Xt[::17, 0] = np.nan                       # NaN in a feature that had none in training (missing_type None)
    Xt[::11, 2] = 0.0
    want = bst.predict(Xt, raw_score=True)
    got = lp.predict_raw(payload, Xt)
    assert np.max(np.abs(got - want)) < 1e-12
    # survives a save/load round trip byte-stably, with a content-bound identity
    path = tmp_path / "m.json"
    v = lp.save(payload, path)
    again = lp.load(path)
    assert np.max(np.abs(lp.predict_raw(again, Xt) - want)) < 1e-12
    assert lp.model_version(again) == v


def test_contributions_add_up_to_the_raw_score_exactly():
    X, y = _data()
    bst = lgb.train({"objective": "binary", "num_leaves": 31, "verbose": -1, "deterministic": True, "seed": 2},
                    lgb.Dataset(X, y), 40)
    payload = lp.export_booster(bst, [f"f{i}" for i in range(6)], calibration={"a": 1.0, "b": 0.0}, provenance={})
    Xt, _ = _data(seed=7, n=2000)
    Xt[::13, 1] = np.nan
    bias, contrib = lp.contributions(payload, Xt)
    raw = lp.predict_raw(payload, Xt)
    assert np.max(np.abs(bias + contrib.sum(axis=1) - raw)) < 1e-9
    # f0 drives the label strongly, f4 and f5 not at all: attribution must say so
    mean_abs = np.abs(contrib).mean(axis=0)
    assert mean_abs[0] > 5 * mean_abs[4] and mean_abs[0] > 5 * mean_abs[5]
    # LightGBM's own TreeSHAP has the same bias (expected value) and the same total
    shap = bst.predict(Xt, pred_contrib=True)
    assert np.max(np.abs(shap.sum(axis=1) - raw)) < 1e-9


def test_interval_band_comes_from_the_stored_venn_abers_calibrator():
    from hazardpulse.trust.venn_abers import VennAbersCalibrator
    X, y = _data()
    bst = lgb.train({"objective": "binary", "verbose": -1, "deterministic": True, "seed": 3}, lgb.Dataset(X, y), 30)
    payload = lp.export_booster(bst, [f"f{i}" for i in range(6)], calibration={"a": 1.0, "b": 0.0}, provenance={})
    Xc, yc = _data(seed=11, n=5000)
    va = VennAbersCalibrator(max_groups=128).fit(lp.predict_raw(payload, Xc), yc)
    payload["interval"] = va.to_dict()
    p0, p1 = lp.predict_interval(payload, X[:200])
    assert np.all(p0 <= p1) and np.all((p0 >= 0) & (p1 <= 1))
    served = VennAbersCalibrator.from_dict(va.to_dict())           # what the payload carries (rounded 1e-8)
    _, want0, want1 = served.predict(lp.predict_raw(payload, X[:200]))
    assert np.array_equal(p0, want0) and np.array_equal(p1, want1)
    _, raw0, raw1 = va.predict(lp.predict_raw(payload, X[:200]))
    assert np.max(np.abs(p0 - raw0)) < 1e-7 and np.max(np.abs(p1 - raw1)) < 1e-7
    no_band = dict(payload, interval=None)
    assert np.isnan(lp.predict_interval(no_band, X[:3])[0]).all()


def test_missing_type_semantics_are_lightgbms():
    # one hand-built stump: x <= 0.5 -> left (value 1), else right (value 2)
    def stump(missing_type, default_left):
        return {"schema": lp.SCHEMA, "feature_names": ["x"], "calibration": {"a": 1, "b": 0}, "trees": [
            {"feature": [0, -1, -1], "threshold": [0.5, 0, 0], "default_left": [default_left, False, False],
             "missing_type": [missing_type, 0, 0], "left": [1, -1, -1], "right": [2, -1, -1],
             "value": [0.0, 1.0, 2.0]}]}
    X = np.array([[np.nan], [0.0], [1.0]])
    # None: NaN becomes 0.0 and compares (0 <= 0.5 -> left)
    assert lp.predict_raw(stump(0, False), X).tolist() == [1.0, 1.0, 2.0]
    # NaN: NaN follows default_left
    assert lp.predict_raw(stump(2, False), X).tolist() == [2.0, 1.0, 2.0]
    assert lp.predict_raw(stump(2, True), X).tolist() == [1.0, 1.0, 2.0]
    # Zero: zero (and NaN-as-zero) follows default_left
    assert lp.predict_raw(stump(1, False), X).tolist() == [2.0, 2.0, 2.0]


def test_wrong_column_count_is_refused():
    X, y = _data(n=500)
    bst = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), 5)
    payload = lp.export_booster(bst, [f"f{i}" for i in range(6)], calibration={"a": 1, "b": 0}, provenance={})
    with pytest.raises(ValueError):
        lp.predict_raw(payload, X[:, :5])
    with pytest.raises(ValueError):
        lp.export_booster(bst, ["only", "two"], calibration={"a": 1, "b": 0}, provenance={})
