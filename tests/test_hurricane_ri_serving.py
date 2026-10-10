"""The hurricane scorer serves a pinned v8.1 artifact and never trains.

Regression guard for the outage of 2026-05-27..2026-10: the daily job retrained four
models on 133,882 cases with a pure-Python split scan whenever any storm was present,
ran past its 10-minute budget, and was cancelled 100+ days in a row. These tests pin:

* the vectorised split scan reproduces the original scalar scan's trees bit for bit
  (and the comparison can fail: a "corrected" scan is detected);
* fit + score_cases reproduce the scorer's former inline train-and-score arithmetic
  exactly, including through the JSON artifact;
* the committed artifact is bound to the committed training data, config and trainer
  source, and stale / tampered artifacts are refused;
* the scorer's main() scores storms without calling any training function.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.hurricane import operational_ri as ori  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402


# ---------------------------------------------------------------------------
# The original scalar split scan, verbatim (the code v8.1 was trained with).
# ---------------------------------------------------------------------------

def _reference_build_tree_recursive(
    X, gradients, hessians, indices, depth, max_depth, lambda_reg,
    min_samples_leaf=5, min_child_weight=1.0, gamma=0.0, colsample=1.0, rng=None,
):
    G = np.sum(gradients[indices])
    H = np.sum(hessians[indices]) + lambda_reg
    leaf_value = -G / H

    if depth >= max_depth or len(indices) < 2 * min_samples_leaf:
        return {"leaf": leaf_value}

    n_features = X.shape[1]
    if colsample < 1.0 and rng is not None:
        n_use = max(1, int(n_features * colsample))
        feature_subset = rng.choice(n_features, size=n_use, replace=False)
    else:
        feature_subset = np.arange(n_features)

    best_gain = -np.inf
    best_feat = -1
    best_thresh = 0.0

    for j in feature_subset:
        col = X[indices, j]
        sorted_idx = np.argsort(col)
        sorted_g = gradients[indices[sorted_idx]]
        sorted_h = hessians[indices[sorted_idx]]
        sorted_col = col[sorted_idx]

        G_left = 0.0
        H_left = 0.0
        G_right = G
        H_right = H - lambda_reg

        n_total = len(sorted_idx)

        for i in range(min_samples_leaf, n_total - min_samples_leaf):
            G_left += sorted_g[i - 1]
            H_left += sorted_h[i - 1]
            G_right -= sorted_g[i - 1]
            H_right -= sorted_h[i - 1]

            if sorted_col[i] == sorted_col[i - 1]:
                continue

            if H_left < min_child_weight or H_right < min_child_weight:
                continue

            gain = (
                (G_left ** 2) / (H_left + lambda_reg)
                + (G_right ** 2) / (H_right + lambda_reg)
                - (G ** 2) / (H)
            ) / 2.0 - gamma

            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thresh = (sorted_col[i - 1] + sorted_col[i]) / 2.0

    if best_gain <= 0 or best_feat < 0:
        return {"leaf": leaf_value}

    left_mask = X[indices, best_feat] <= best_thresh
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    if len(left_indices) < min_samples_leaf or len(right_indices) < min_samples_leaf:
        return {"leaf": leaf_value}

    left_child = _reference_build_tree_recursive(
        X, gradients, hessians, left_indices, depth + 1, max_depth, lambda_reg,
        min_samples_leaf, min_child_weight, gamma, colsample, rng,
    )
    right_child = _reference_build_tree_recursive(
        X, gradients, hessians, right_indices, depth + 1, max_depth, lambda_reg,
        min_samples_leaf, min_child_weight, gamma, colsample, rng,
    )
    return {"feature": best_feat, "threshold": best_thresh, "left": left_child, "right": right_child}


def _tree_diff(a: dict, b: dict, path: str = "tree") -> str | None:
    """First structural / bitwise difference between two trees (types included)."""
    if set(a) != set(b):
        return f"{path}: keys {sorted(a)} != {sorted(b)}"
    for key in a:
        if isinstance(a[key], dict):
            diff = _tree_diff(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        elif not (a[key] == b[key] and type(a[key]) is type(b[key])):
            return f"{path}.{key}: {a[key]!r} != {b[key]!r}"
    return None


def _adversarial_data(seed: int, n: int = 1500):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 7))
    X[:, 1] = np.round(X[:, 1])            # heavy ties
    X[:, 2] = rng.integers(0, 3, n)         # 3-level discrete
    X[:, 3] = X[:, 2]                       # exact duplicate column: cross-feature exact ties
    X[:, 4] = 1.0                           # constant column: no split at all
    X[:, 5] = rng.integers(0, 2, n) * 1e6   # binary with a huge scale
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] + X[:, 2] - 1.5)))).astype(float)
    return X, y


GBT_CONFIGS = [
    dict(n_trees=8, max_depth=3, learning_rate=0.08, lambda_reg=1.0, scale_pos_weight=5.0),
    dict(n_trees=8, max_depth=4, learning_rate=0.06, lambda_reg=2.0, min_samples_leaf=10,
         min_child_weight=3.0, colsample=0.8, subsample=0.8, gamma=0.1, scale_pos_weight=7.0),
    dict(n_trees=5, max_depth=5, min_samples_leaf=1, min_child_weight=0.0),
]


def _train_both(monkeypatch, X, y, kw):
    fast = ori.train_histogram_gbt(X, y, **kw)
    with monkeypatch.context() as m:
        m.setattr(ori, "_build_tree_recursive", _reference_build_tree_recursive)
        slow = ori.train_histogram_gbt(X, y, **kw)
    return fast, slow


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kw", GBT_CONFIGS)
def test_vectorised_split_scan_is_bit_identical_to_the_scalar_scan(monkeypatch, seed, kw):
    X, y = _adversarial_data(seed)
    fast, slow = _train_both(monkeypatch, X, y, kw)
    assert len(fast[0]) == len(slow[0]) == kw["n_trees"]
    for i, (tf, ts) in enumerate(zip(fast[0], slow[0])):
        diff = _tree_diff(tf, ts, f"tree{i}")
        assert diff is None, diff
    assert fast[1:] == slow[1:]  # init score, lr, per-tree losses, AUCs


def test_tree_comparison_detects_a_changed_scan(monkeypatch):
    """The bit-identity check can fail: 'fixing' the left-sum offset changes the trees."""

    def corrected_scan(col, g, h, G, H, lambda_reg, msl, mcw, gamma):
        order = np.argsort(col)
        sc, sg, sh = col[order], g[order], h[order]
        gl, hl = np.cumsum(sg)[:-1], np.cumsum(sh)[:-1]       # left = all samples before i
        gr, hr = G - gl, (H - lambda_reg) - hl
        i = np.arange(1, len(sc))
        ok = (i >= msl) & (i <= len(sc) - msl) & (sc[1:] != sc[:-1]) & (hl >= mcw) & (hr >= mcw)
        gain = (gl ** 2 / (hl + lambda_reg) + gr ** 2 / (hr + lambda_reg) - G ** 2 / H) / 2.0 - gamma
        if not ok.any():
            return -np.inf, 0.0
        k = int(np.argmax(np.where(ok, gain, -np.inf)))
        return gain[k], (sc[k] + sc[k + 1]) / 2.0

    X, y = _adversarial_data(3, n=600)
    kw = dict(n_trees=6, max_depth=4, min_samples_leaf=30, lambda_reg=1.0, scale_pos_weight=5.0)
    with monkeypatch.context() as m:
        m.setattr(ori, "_build_tree_recursive", _reference_build_tree_recursive)
        reference = ori.train_histogram_gbt(X, y, **kw)[0]
    with monkeypatch.context() as m:
        m.setattr(ori, "_best_split_for_feature", corrected_scan)
        mutant = ori.train_histogram_gbt(X, y, **kw)[0]
    assert any(_tree_diff(a, b) for a, b in zip(reference, mutant)), \
        "the comparison failed to see a semantically different split scan"


# ---------------------------------------------------------------------------
# fit + score == the scorer's former inline train_and_score
# ---------------------------------------------------------------------------

SMALL = ori.OperationalRIConfig(
    feature_select_epochs=200, hgbt_d3_n_trees=6, hgbt_d4_n_trees=5,
    logistic_epochs_final=200, bag_n_bags_final=4, bag_epochs_final=120,
)


def _reference_inline_train_and_score(historical_cases, live_cases, config):
    """The pre-2026-10 body of scripts/fetch_and_score.py::train_and_score (numbers part)."""
    X_hist, y_hist, years_hist, feature_names = ori.build_feature_matrix(historical_cases)
    X_live, _, _, _ = ori.build_feature_matrix(live_cases, feature_names=feature_names)
    X_hist_imp, X_live_imp = ori.train_median_impute(X_hist, X_live)
    X_hist_std, X_live_std = ori.standardize(X_hist_imp, X_live_imp)
    pos = int(np.sum(y_hist == 1))
    neg = int(np.sum(y_hist == 0))
    cw_ratio = min(neg / max(pos, 1), 20.0)
    scale_pw = cw_ratio
    sel_idx, sel_names, _ = ori.select_features_by_logistic(
        X_hist_std, y_hist, feature_names, n_select=config.n_features_select,
        lam=config.feature_select_lam, n_epochs=config.feature_select_epochs, cw_ratio=cw_ratio,
    )
    X_hist_sel = X_hist_imp[:, sel_idx]
    X_live_sel = X_live_imp[:, sel_idx]
    X_hist_s, X_live_s = ori.standardize(X_hist_sel, X_live_sel)
    gbt3_trees, gbt3_init, gbt3_lr, _, _ = ori.train_histogram_gbt(
        X_hist_sel, y_hist, n_trees=config.hgbt_d3_n_trees, max_depth=config.hgbt_d3_max_depth,
        learning_rate=config.hgbt_d3_lr, lambda_reg=config.hgbt_d3_lambda_reg,
        scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )
    gbt4_trees, gbt4_init, gbt4_lr, _, _ = ori.train_histogram_gbt(
        X_hist_sel, y_hist, n_trees=config.hgbt_d4_n_trees, max_depth=config.hgbt_d4_max_depth,
        learning_rate=config.hgbt_d4_lr, lambda_reg=config.hgbt_d4_lambda_reg,
        min_samples_leaf=config.hgbt_d4_min_samples_leaf,
        min_child_weight=config.hgbt_d4_min_child_weight,
        colsample=config.hgbt_d4_colsample, subsample=config.hgbt_d4_subsample,
        gamma=config.hgbt_d4_gamma, scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )
    w_lr, b_lr = ori.train_logistic(
        X_hist_s, y_hist, lam=config.logistic_lam, lr=config.logistic_lr,
        n_epochs=config.logistic_epochs_final, cw_ratio=cw_ratio,
    )
    bags = ori.train_bagged_logistic(
        X_hist_s, y_hist, n_bags=config.bag_n_bags_final, feat_fraction=config.bag_feat_fraction,
        lam=config.bag_lam, lr=config.bag_lr, n_epochs=config.bag_epochs_final, cw_ratio=cw_ratio,
    )
    p_gbt3 = ori.predict_histogram_gbt(X_live_sel, gbt3_trees, gbt3_init, gbt3_lr)
    p_gbt4 = ori.predict_histogram_gbt(X_live_sel, gbt4_trees, gbt4_init, gbt4_lr)
    p_lr = ori.predict_logistic(X_live_s, w_lr, b_lr)
    p_bag = ori.predict_bagged(X_live_s, bags)
    p_ensemble = (p_gbt3 + p_gbt4 + p_lr + p_bag) / 4.0
    p_hist_ens = (
        ori.predict_histogram_gbt(X_hist_sel, gbt3_trees, gbt3_init, gbt3_lr)
        + ori.predict_histogram_gbt(X_hist_sel, gbt4_trees, gbt4_init, gbt4_lr)
        + ori.predict_logistic(X_hist_s, w_lr, b_lr)
        + ori.predict_bagged(X_hist_s, bags)
    ) / 4.0
    p_calibrated, _, _ = ori.platt_calibrate(y_hist, p_hist_ens, p_ensemble)
    return {"gbt_d3": p_gbt3, "gbt_d4": p_gbt4, "logistic": p_lr, "bagged": p_bag,
            "ensemble": p_ensemble, "calibrated": p_calibrated}


def _synthetic_cases(seed: int, n: int, with_labels: bool = True) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        v = float(rng.integers(20, 140))
        case = {
            "storm_id": f"S{i % 40:03d}", "season_year": 2000 + i % 25, "issue_time": "x",
            "basin": "NA", "storm_name": "T", "analysis_model": "BEST",
            "analysis_lat": float(rng.uniform(-35, 40)), "analysis_lon": float(rng.uniform(-180, 180)),
            "analysis_vmax_kt": v,
            "analysis_mslp_hpa": None if rng.random() < 0.2 else float(1010 - v / 2),
            "ri_label_30kt": int(rng.random() < 0.08 + 0.002 * v) if with_labels else 0,
            "analysis_dv_6h": None if rng.random() < 0.1 else float(rng.normal(0, 5)),
            "analysis_dv_24h": float(rng.normal(0, 10)),
            "storm_age_h": float(rng.integers(0, 300)),
        }
        cases.append(case)
    return cases


def test_fit_and_score_reproduce_the_former_inline_train_and_score(tmp_path):
    hist = _synthetic_cases(0, 700)
    live = _synthetic_cases(1, 25, with_labels=False)
    for c in live[:5]:
        c["analysis_dv_6h"] = None          # live NaNs exercise the stored medians
        c["analysis_mslp_hpa"] = None
        c.pop("storm_age_h")                 # a feature a live source can lack
    ref = _reference_inline_train_and_score(hist, live, SMALL)

    model = ri_model.fit_operational_ri_model(hist, SMALL, log=lambda *_: None)
    model["serving"]["impute_live"] = []     # the reference imputed only what was missing
    got = ri_model.score_cases(model, live)
    for key, want in ref.items():
        assert np.array_equal(got[key], want), key

    # ...and through the serialised artifact.
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(c) for c in hist) + "\n", encoding="utf-8")
    ri_model.attach_provenance(model, training_data=data, config=SMALL)
    path = ri_model.save_model(model, tmp_path / "m.json")
    loaded = ri_model.load_model(path, config=SMALL)
    again = ri_model.score_cases(loaded, live)
    for key, want in ref.items():
        assert np.array_equal(again[key], want), f"{key} after JSON round trip"


def test_load_model_refuses_stale_mismatched_or_corrupt_artifacts(tmp_path):
    hist = _synthetic_cases(2, 300)
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(c) for c in hist) + "\n", encoding="utf-8")
    model = ri_model.fit_operational_ri_model(hist, SMALL, log=lambda *_: None)
    ri_model.attach_provenance(model, training_data=data, config=SMALL)
    good = ri_model.save_model(model, tmp_path / "good.json")
    ri_model.load_model(good, config=SMALL)   # the control: accepted

    with pytest.raises(ri_model.ModelArtifactError, match="missing"):
        ri_model.load_model(tmp_path / "absent.json", config=SMALL)
    with pytest.raises(ri_model.ModelArtifactError, match="different config"):
        ri_model.load_model(good)              # default (v8.1 hyper-parameter) config
    data.write_text(data.read_text(encoding="utf-8") + json.dumps(hist[0]) + "\n", encoding="utf-8")
    with pytest.raises(ri_model.ModelArtifactError, match="stale"):
        ri_model.load_model(good, config=SMALL)

    newton = copy.deepcopy(model)
    newton["calibration"] = {"method": "logistic_on_logit_newton", "a": 1.0, "b": -2.0,
                             "grad_inf": 1e-12, "tol_grad": 1e-10}
    for base, mutate, match in [
        (model, lambda m: m["gbt_d3"]["trees"][0].update(feature=99), "out of range"),
        (model, lambda m: m.update(selected_idx=[]), "selected_idx"),
        (model, lambda m: m["bagged"][0]["w"].pop(), "length mismatch"),
        (model, lambda m: m.update(schema="something/else"), "schema"),
        (model, lambda m: m["serving"].update(impute_live=["not_a_feature"]), "unknown feature"),
        (newton, lambda m: m["calibration"].update(grad_inf=3e-4), "convergence tolerance"),
        (newton, lambda m: m["calibration"].update(a=-0.5), "not increasing"),
        (newton, lambda m: m["calibration"].update(method="isotonic?"), "unknown calibration"),
    ]:
        bad = copy.deepcopy(base)
        node = bad["gbt_d3"]["trees"][0]
        if "leaf" in node and "out of range" in match:
            continue  # a stump has no split to corrupt
        mutate(bad)
        path = ri_model.save_model(bad, tmp_path / "bad.json")
        with pytest.raises(ri_model.ModelArtifactError, match=match):
            ri_model.load_model(path, verify_data=False, config=SMALL)


def test_training_data_digest_is_line_ending_invariant(tmp_path):
    """The model is trained on Windows (CRLF checkout) and served on Linux (LF checkout)."""
    import hashlib

    rows = b'{"a": 1}\n{"b": 2.5}\n' * 40 + b'lone\rcarriage return stays\n'
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(rows)
    crlf.write_bytes(rows.replace(b"\n", b"\r\n"))
    blob_digest = hashlib.sha256(rows).hexdigest()       # what git stores and Linux checks out
    for chunk in (1, 2, 3, 7, 64, 1 << 20):              # CRLF pairs split across reads
        assert ri_model.sha256_file(crlf, chunk) == blob_digest
        assert ri_model.sha256_file(lf, chunk) == blob_digest
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(rows.replace(b"2.5", b"2.6"))
    assert ri_model.sha256_file(changed) != blob_digest  # content changes are still seen


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["hurricane_ri_v8_3", "hurricane_ri_v8_2", "hurricane_ri_v8_1"])
def test_committed_artifacts_are_bound_to_their_data_config_and_trainer(version):
    model = ri_model.load_model(ri_model.ARTIFACTS[version])  # raises unless every data sha + config match
    prov = model["provenance"]
    assert prov["trainer_fingerprint"] == ri_model.trainer_fingerprint(), (
        "operational_ri's training code changed since the artifact was trained: rerun "
        f"`python scripts/train_hurricane_ri.py --recipe {version}` (and `--verify`), then `git add -f` it"
    )
    recipe = ri_model.RECIPES[version]
    assert prov["data"]["members"]["storm_years"] == list(recipe["members_years"])
    assert model["serving"]["impute_live"] == recipe["impute_live"]
    assert len(model["gbt_d3"]["trees"]) == 200 and len(model["gbt_d4"]["trees"]) == 150
    assert len(model["bagged"]) == 50
    assert model["feature_names"] == sorted(model["feature_names"])
    if version == "hurricane_ri_v8_1":   # the original pin, reproduced for comparison only
        assert model["training_summary"]["n_cases"] == 133_882
        assert model["calibration"]["method"] == "platt_prob_gd500_insample"
    else:                                # served: held-out, converged
        cal = model["calibration"]
        assert cal["method"] == "logistic_on_logit_newton" and cal["grad_inf"] <= cal["tol_grad"] <= 1e-10
        assert cal["fitted_on"]["storm_years"] == list(recipe["calibration"]["years"])
        assert cal["fitted_on"]["storm_years"][0] > recipe["members_years"][1], "calibration rows must be held out"
        assert cal["n_events"] >= 100
        for role, years in (("members", recipe["members_years"]), ("calibration", recipe["calibration"]["years"])):
            spec = prov["data"][role]
            assert spec["path"] == ri_model._rel(ri_model.DATASETS[recipe["dataset"]])
            assert spec["sha256"] == ri_model.sha256_file(ri_model.DATASETS[recipe["dataset"]])
            assert spec["storm_years"] == list(years)


def _load_script(name: str):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v8_3_is_v8_2s_recipe_on_exactly_the_deduplicated_v8_2_rows():
    """Amendment 15: v8.3 = v8.2's recipe with only the dataset changed, and that dataset is the frozen v8.2 file
    with exactly its 5,950 byte-identical (storm, issue time) copies removed by amendment 13a's rule -- every kept
    line byte-identical, in the frozen order."""
    v82, v83 = ri_model.RECIPES["hurricane_ri_v8_2"], ri_model.RECIPES["hurricane_ri_v8_3"]
    assert (v82["dataset"], v83["dataset"]) == ("v8.2", "v8.3")
    assert (v82["calibration"]["dataset"], v83["calibration"]["dataset"]) == ("v8.2", "v8.3")
    strip = lambda r: {**{k: v for k, v in r.items() if k != "dataset"},  # noqa: E731
                       "calibration": {k: v for k, v in r["calibration"].items() if k != "dataset"}}
    assert strip(v82) == strip(v83)

    v8_3 = _load_script("hurricane_ri_v8_3")
    d = v8_3.dedupe_frozen()
    assert (d["frozen_rows"], d["removed"], d["kept_rows"]) == (69_722, 5_950, 63_772)
    assert ri_model.DATASETS["v8.3"].read_bytes().replace(b"\r\n", b"\n") == d["kept_bytes"]
    assert ri_model.DATASETS["v8.2"] != ri_model.DATASETS["v8.3"]


def test_the_dedupe_refuses_two_different_rows_under_one_key():
    j1 = _load_script("hurricane_ri_j1")
    a = {"storm_id": "S", "issue_time": "2020-01-01 00:00:00", "ri_label_30kt": 0}
    assert j1.dedupe([a, dict(a)]) == ([a], 1)
    with pytest.raises(SystemExit, match="two different rows"):
        j1.dedupe([a, dict(a, ri_label_30kt=1)])


# ---------------------------------------------------------------------------
# Calibration: converged on held-out data, or refused
# ---------------------------------------------------------------------------

def _logistic_sample(seed: int, n: int, a: float, b: float):
    rng = np.random.default_rng(seed)
    p_ens = 1.0 / (1.0 + np.exp(-rng.normal(-1.0, 1.5, n)))      # an inflated "ensemble" score
    z = a * np.log(p_ens / (1 - p_ens)) + b
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    return p_ens, y


def test_newton_calibrator_converges_to_the_likelihood_optimum():
    p, y = _logistic_sample(0, 20_000, a=0.8, b=-2.0)
    cal = ri_model.fit_logit_calibrator(p, y)
    assert cal["grad_inf"] <= 1e-10 and cal["iterations"] <= 20
    assert abs(cal["a"] - 0.8) < 0.08 and abs(cal["b"] + 2.0) < 0.15   # recovers the truth
    q = ri_model.apply_calibration(cal, p)
    assert abs(q.mean() - y.mean()) < 1e-6       # calibration-in-the-large holds at the MLE


def test_calibrator_refuses_what_it_cannot_identify():
    p, y = _logistic_sample(1, 5_000, a=0.8, b=-2.0)
    with pytest.raises(ri_model.CalibrationError, match="did not converge"):
        ri_model.fit_logit_calibrator(p, y, max_iter=1)          # an iteration cap is not a fit
    with pytest.raises(ri_model.CalibrationError, match="too few"):
        ri_model.fit_logit_calibrator(p[:200], np.where(np.arange(200) < 5, 1.0, 0.0))
    sep = np.concatenate([np.full(300, 0.1), np.full(300, 0.9)])
    with pytest.raises(ri_model.CalibrationError, match="separable|did not converge"):
        ri_model.fit_logit_calibrator(sep, np.concatenate([np.zeros(300), np.ones(300)]))
    with pytest.raises(ri_model.CalibrationError, match="not increasing"):
        ri_model.fit_logit_calibrator(p, 1.0 - y)                # an anti-correlated score


def test_the_legacy_platt_fit_was_not_a_converged_fit():
    """The check that would have caught v8.1: its 500 fixed steps stop far from the optimum."""
    p, y = _logistic_sample(2, 20_000, a=0.8, b=-2.0)
    _, a, b = ori.platt_calibrate(y, p, p[:1])                    # v8.1's calibration routine
    z = a * p + b
    grad = np.array([np.mean((ori.sigmoid(z) - y) * p), np.mean(ori.sigmoid(z) - y)])
    assert np.max(np.abs(grad)) > 1e-3, "500 fixed steps happened to converge here"
    legacy = ri_model.apply_calibration({"method": "platt_prob_gd500_insample", "a": a, "b": b}, p)
    assert legacy.min() > 0.1 > y.mean(), "the legacy map's floor sits far above the base rate"
    converged = ri_model.apply_calibration(ri_model.fit_logit_calibrator(p, y), p)
    assert abs(converged.mean() - y.mean()) < 1e-6 and converged.min() < 0.01


# ---------------------------------------------------------------------------
# The scheduled job never trains
# ---------------------------------------------------------------------------

def _load_scorer():
    spec = importlib.util.spec_from_file_location("fas_serving_test", REPO / "scripts" / "fetch_and_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_scores_active_storms_without_training(monkeypatch):
    import datetime as dt

    fas = _load_scorer()

    def forbid(*_a, **_k):
        raise AssertionError("the scheduled hurricane job must not train")

    for name in ("train_histogram_gbt", "train_logistic", "train_bagged_logistic",
                 "select_features_by_logistic", "platt_calibrate"):
        monkeypatch.setattr(ori, name, forbid)
    for name in ("fit_operational_ri_model", "fit_members", "fit_recipe",
                 "fit_logit_calibrator", "newton_logistic"):
        monkeypatch.setattr(ri_model, name, forbid)

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cycle = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=3)
    old = now - dt.timedelta(days=90)

    def rec(cyc, tau, model, vmax, lat=15.0, lon=-110.0):
        return fas.ATCFRecord(basin="EP", storm_number=15, cycle=cyc, tau_hours=tau, model=model,
                              lat=lat, lon=lon, vmax_kt=vmax, mslp_hpa=990.0, storm_name="TEST")

    decks = {
        "EP152026": [rec(cycle - dt.timedelta(hours=h), 0, "CARQ", 60.0 - h) for h in (0, 6, 12, 24)]
        + [rec(cycle, 24, "OFCL", 85.0)],
        "AL012026": [rec(old, 0, "CARQ", 35.0)],      # long dead: must not be scored
    }
    index = "".join(
        f'<a href="a{sid.lower()}.dat.gz">a{sid.lower()}.dat.gz</a>  '
        f'{(old if sid == "AL012026" else now).strftime("%Y-%m-%d %H:%M")}  1.0M\n'
        for sid in decks
    )
    fetched: list[str] = []
    monkeypatch.setattr(fas, "fetch_text", lambda url, **k: index if "aid_public" in url else "")
    monkeypatch.setattr(fas, "fetch_realtime_adeck", lambda sid: fetched.append(sid) or decks[sid])
    monkeypatch.setattr(fas, "_discover_jtwc_storms", lambda: {})
    # no network: the IR model's satellite images (NOAA GMGSI on S3, ~7.5 MB each) are not read here --
    # this test times the scoring path, and an un-stubbed download made it time the connection instead
    from hazardpulse.hurricane import ir_source
    images: list = []
    monkeypatch.setattr(ir_source, "fetch_image", lambda hour: images.append(hour) or (None, None, None, None))
    captured: dict = {}
    monkeypatch.setattr(fas, "write_outputs",
                        lambda scored, now, version, **k: captured.update(scored=scored, version=version,
                                                                          catch_up=k.get("catch_up")))
    monkeypatch.setattr(fas, "build_site_artifacts", lambda: None)
    monkeypatch.setattr(fas, "DIST", REPO / "nonexistent-dist-for-test")

    t0 = time.perf_counter()
    fas.main()
    elapsed = time.perf_counter() - t0

    scored = captured["scored"]
    assert [s["storm_id"] for s in scored] == ["EP152026"]
    assert fetched == ["EP152026"], "an a-deck untouched for 90 days must not even be downloaded"
    s = scored[0]
    assert 0.0 <= s["ri_probability"] <= 1.0
    assert s["model_version"] == captured["version"] == fas.SERVED_MODEL_VERSION
    assert s["calibration"] == "logistic_on_logit_newton"
    assert set(s["model_scores"]) == {"gbt_d3", "gbt_d4", "logistic", "bagged"}
    # amendment 7 rule 2: the previous cycle (now ~9 h old, no record of it) is forecast in shadow, from the
    # deck already read -- never a second a-deck download, never published
    if fas._shadow_keys(fas.load_v9_model(), fas.load_v10_model(), fas.load_challengers()):
        assert [(c["storm_id"], c["issue_time"], c["catch_up"]) for c in captured["catch_up"]] == [
            ("EP152026", (cycle - dt.timedelta(hours=6)).isoformat(), True)]
        assert all(c["issue_time"] != s["issue_time"] for c in captured["catch_up"])
    # The serving path is milliseconds; the training path it replaced was > 30 minutes.
    assert elapsed < 20.0, f"scoring took {elapsed:.1f}s"


def test_the_scorer_serves_what_the_preregistered_evaluation_selected():
    fas = _load_scorer()
    report = json.loads((REPO / "results" / "calibration" / "hurricane_ri_evaluation.json").read_text(encoding="utf-8"))
    assert report["decision"]["serve"] == fas.SERVED_MODEL_VERSION
    assert report["data_sha256"]["v8.2"] == ri_model.sha256_file(ri_model.DATASETS["v8.2"]), \
        "the evaluation was run on a different v8.2 training set; re-run scripts/evaluate_hurricane_ri.py"
    served = ri_model.load_model(fas.MODEL_ARTIFACT)
    assert served["provenance"]["data"]["members"]["sha256"] == report["data_sha256"]["v8.2"]
    model = fas.load_serving_model()
    assert model["model_version"] == fas.SERVED_MODEL_VERSION


def test_the_scorer_refuses_the_legacy_calibration(monkeypatch):
    fas = _load_scorer()
    monkeypatch.setattr(fas, "SERVED_MODEL_VERSION", "hurricane_ri_v8_1")
    monkeypatch.setattr(fas, "MODEL_ARTIFACT", ri_model.ARTIFACTS["hurricane_ri_v8_1"])
    with pytest.raises(ri_model.ModelArtifactError, match="refusing to serve calibration"):
        fas.load_serving_model()


def test_pulse_risk_band_follows_the_calibrated_probability(tmp_path, monkeypatch):
    fas = _load_scorer()
    monkeypatch.setattr(fas, "DIST", tmp_path)
    (tmp_path / "data").mkdir()
    pulse = {"updated_at": "x", "hazards": [{"key": "hu", "model_auc": 0.938, "n_features": 65}]}
    (tmp_path / "data" / "live-pulse.json").write_text(json.dumps(pulse), encoding="utf-8")
    for probability, band in [(0.004, "none"), (0.07, "guarded"), (0.18, "watch"), (0.62, "critical")]:
        storm = {"storm_id": "EP152026", "ri_probability": probability, "lat": 15.0, "lon": -110.0}
        fas.write_outputs([storm], dt_now(), "hurricane_ri_v8_2")
        hu = json.loads((tmp_path / "data" / "live-pulse.json").read_text())["hazards"][0]
        assert (hu["probability"], hu["risk_band"], hu["model_version"]) == (probability, band, "hurricane_ri_v8_2")
        assert "model_auc" not in hu and "n_features" not in hu, "stale v8.1 benchmark fields must not ride along"


def dt_now():
    import datetime as dt
    return dt.datetime(2026, 10, 2, 12, 0)


def test_a_trust_calibrator_fitted_for_another_model_is_not_applied(monkeypatch):
    import hazardpulse.trust.scoring as trust

    class Foreign:
        model_version = "hurricane_ri_v8_1"

    calls: list = []   # recorded, not raised: main() deliberately swallows trust-layer errors
    monkeypatch.setattr(trust, "load_forecaster", lambda *a, **k: Foreign())
    monkeypatch.setattr(trust, "enrich_cells", lambda *a, **k: calls.append(a))
    test_main_scores_active_storms_without_training(monkeypatch)
    assert calls == [], "a v8.1 calibrator must not re-map, band or sign v8.2 probabilities"

    class Own:
        model_version = "hurricane_ri_v8_2"

    monkeypatch.setattr(trust, "load_forecaster", lambda *a, **k: Own())
    test_main_scores_active_storms_without_training(monkeypatch)
    assert len(calls) == 1, "the control: a calibrator for the served model IS applied"
