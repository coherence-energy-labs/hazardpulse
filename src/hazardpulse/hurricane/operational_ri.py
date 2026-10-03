"""Operational hurricane rapid intensification prediction — v8.1.

Pure NumPy implementation of the full ML pipeline: histogram gradient boosted
trees, logistic regression, bagged logistic, L2 feature selection, Platt
calibration, and ensemble scoring.

All algorithms implemented from scratch — no sklearn, no PyTorch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OperationalRIConfig:
    """Hyperparameters for the v8.1 operational RI model."""

    # Feature selection
    n_features_select: int = 25
    feature_select_lam: float = 0.05
    feature_select_epochs: int = 3000

    # Histogram GBT depth-3
    hgbt_d3_n_trees: int = 200
    hgbt_d3_max_depth: int = 3
    hgbt_d3_lr: float = 0.08
    hgbt_d3_lambda_reg: float = 1.0
    n_bins: int = 64

    # Histogram GBT depth-4
    hgbt_d4_n_trees: int = 150
    hgbt_d4_max_depth: int = 4
    hgbt_d4_lr: float = 0.06
    hgbt_d4_lambda_reg: float = 2.0
    hgbt_d4_min_samples_leaf: int = 10
    hgbt_d4_min_child_weight: float = 3.0
    hgbt_d4_colsample: float = 0.8
    hgbt_d4_subsample: float = 0.8
    hgbt_d4_gamma: float = 0.1

    # Logistic regression
    logistic_lam: float = 0.01
    logistic_lr: float = 0.05
    logistic_epochs_final: int = 3000

    # Bagged logistic
    bag_n_bags_final: int = 50
    bag_feat_fraction: float = 0.5
    bag_lam: float = 0.01
    bag_lr: float = 0.05
    bag_epochs_final: int = 1500


def best_known_operational_ri_config() -> OperationalRIConfig:
    """Return the best-known hyperparameters for the v8.1 model."""
    return OperationalRIConfig()


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_operational_ri_cases(path: str | Path) -> list[dict]:
    """Load JSONL file of operational RI cases."""
    path = Path(path)
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_feature_matrix(
    cases: list[dict],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, years, feature_names) from a list of case dicts.

    If feature_names is None, auto-discovers numeric features from the first
    case (excluding metadata keys).
    """
    METADATA_KEYS = {
        "storm_id", "season_year", "issue_time", "basin",
        "storm_number", "storm_name", "analysis_model",
        "ri_label_30kt", "forecast_id",
    }

    if feature_names is None:
        # Auto-discover from first case
        sample = cases[0]
        feature_names = []
        for k, v in sample.items():
            if k in METADATA_KEYS:
                continue
            if isinstance(v, (int, float)) or v is None:
                feature_names.append(k)
        feature_names.sort()

    n = len(cases)
    d = len(feature_names)
    X = np.full((n, d), np.nan)
    y = np.zeros(n)
    years = np.zeros(n, dtype=int)

    for i, case in enumerate(cases):
        for j, fname in enumerate(feature_names):
            val = case.get(fname)
            if val is not None:
                try:
                    X[i, j] = float(val)
                except (ValueError, TypeError):
                    pass
        y[i] = float(case.get("ri_label_30kt", 0))
        years[i] = int(case.get("season_year", 0))

    return X, y, years, feature_names


def climatological_mpi_features(
    lat: float | None,
    vmax_kt: float | None,
    month: int,
) -> dict[str, float | None]:
    """``abs_lat`` / ``mpi_deficit`` / ``intensity_frac_mpi`` exactly as the training set has them.

    The single definition shared by the training-set builder
    (``scripts/build_hurricane_training_data.py``) and the live case builder
    (``scripts/fetch_and_score.py``), so a served feature can never drift from the one
    the model learned. The arithmetic is the builder's, operation for operation; a test
    re-derives every row of the committed training file from it.

    The MPI is a crude latitude/season SST proxy (no SST data is read), which is what v8.1
    was trained on.
    """
    if lat is None:
        return {"abs_lat": None, "mpi_deficit": None, "intensity_frac_mpi": None}
    abs_lat = abs(lat)
    sst_est = 30.0 - 0.5 * max(0, abs_lat - 10)
    if lat >= 0:
        sst_est += 2.0 * np.exp(-((month - 9) ** 2) / 8.0)
    else:
        sst_est += 2.0 * np.exp(-((month - 3) ** 2) / 8.0)
    mpi = min(30.0 * max(sst_est - 26.0, 0) + 40.0, 185.0)
    if vmax_kt is None:
        return {"abs_lat": abs_lat, "mpi_deficit": None, "intensity_frac_mpi": None}
    return {
        "abs_lat": abs_lat,
        "mpi_deficit": mpi - vmax_kt,
        "intensity_frac_mpi": vmax_kt / mpi if mpi > 0 else None,
    }


def translation_speed_kmh(
    lat_prev: float | None,
    lon_prev: float | None,
    lat_now: float | None,
    lon_now: float | None,
    hours: float,
) -> float | None:
    """Great-circle distance between two fixes divided by the REAL time between them.

    Shared by the v8.2 training-set builder and the live case builder. (v8.1's builder
    divided a 3-hour displacement by 6 -- see LEGACY notes in that script.)
    """
    if None in (lat_prev, lon_prev, lat_now, lon_now) or not hours or hours <= 0:
        return None
    dlat = math.radians(lat_now - lat_prev)
    dlon = math.radians(lon_now - lon_prev)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat_prev)) * math.cos(math.radians(lat_now))
         * math.sin(dlon / 2) ** 2)
    dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return dist_km / float(hours)


# ---------------------------------------------------------------------------
# Imputation & standardization
# ---------------------------------------------------------------------------

def train_median_impute(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median imputation fitted on training data only."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    d = X_train.shape[1]
    for j in range(d):
        col = X_train[:, j]
        valid = col[~np.isnan(col)]
        median_val = float(np.median(valid)) if len(valid) > 0 else 0.0

        train_nans = np.isnan(X_train[:, j])
        X_train[train_nans, j] = median_val

        test_nans = np.isnan(X_test[:, j])
        X_test[test_nans, j] = median_val

    return X_train, X_test


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score standardization fitted on training data."""
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0) + 1e-10
    return (X_train - mu) / sd, (X_test - mu) / sd


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features_by_logistic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int = 25,
    lam: float = 0.05,
    n_epochs: int = 3000,
    cw_ratio: float = 1.0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """L2-regularized logistic regression for feature ranking.

    Returns (selected_indices, selected_names, weight_magnitudes).
    """
    w, _ = train_logistic(X, y, lam=lam, lr=0.05, n_epochs=n_epochs, cw_ratio=cw_ratio)
    magnitudes = np.abs(w)
    top_idx = np.argsort(-magnitudes)[:n_select]
    top_idx.sort()
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx, selected_names, magnitudes[top_idx]


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.01,
    lr: float = 0.05,
    n_epochs: int = 2000,
    cw_ratio: float | None = None,
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression via gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    sw = np.ones(n)
    if cw_ratio is not None:
        sw[y == 1] = cw_ratio

    for epoch in range(n_epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = (p - y) * sw
        grad_w = (X.T @ error) / n + lam * w
        grad_b = np.sum(error) / n
        clr = lr / (1 + epoch * 0.001)
        w -= clr * grad_w
        b -= clr * grad_b

    return w, b


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict probabilities with trained logistic model."""
    return sigmoid(X @ w + b)


# ---------------------------------------------------------------------------
# Bagged logistic regression
# ---------------------------------------------------------------------------

def train_bagged_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n_bags: int = 50,
    feat_fraction: float = 0.5,
    lam: float = 0.01,
    lr: float = 0.05,
    n_epochs: int = 1500,
    cw_ratio: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Train bagged logistic regression ensemble."""
    n, d = X.shape
    n_feat = max(int(d * feat_fraction), 5)
    models: list[tuple[np.ndarray, np.ndarray, float]] = []
    rng = np.random.RandomState(42)

    for _ in range(n_bags):
        feat_idx = rng.choice(d, size=n_feat, replace=False)
        feat_idx.sort()
        sample_idx = rng.choice(n, size=n, replace=True)
        X_bag = X[sample_idx][:, feat_idx]
        y_bag = y[sample_idx]
        w, b = train_logistic(X_bag, y_bag, lam=lam, lr=lr,
                              n_epochs=n_epochs, cw_ratio=cw_ratio)
        models.append((feat_idx, w, b))

    return models


def predict_bagged(
    X: np.ndarray,
    models: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Predict with bagged logistic ensemble."""
    preds = np.zeros(X.shape[0])
    for feat_idx, w, b in models:
        X_sub = X[:, feat_idx]
        preds += sigmoid(X_sub @ w + b)
    return preds / len(models)


# ---------------------------------------------------------------------------
# Histogram gradient boosted trees
# ---------------------------------------------------------------------------

def precompute_bins(
    X: np.ndarray,
    n_bins: int = 64,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Bin features into integer histograms for fast GBT splitting.

    Returns (X_binned, bin_edges_per_feature).
    """
    n, d = X.shape
    X_binned = np.zeros((n, d), dtype=np.int32)
    bin_edges: list[np.ndarray] = []

    for j in range(d):
        col = X[:, j]
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(col, percentiles))
        X_binned[:, j] = np.searchsorted(edges, col, side="right").astype(np.int32)
        bin_edges.append(edges)

    return X_binned, bin_edges


def _pow2_like_scalar(x: np.ndarray) -> np.ndarray:
    """Element-wise ``x ** 2`` through libm ``pow``, exactly as the scalar scan squared.

    The original split scan squared NumPy *scalars* (``np.float64 ** 2`` goes through C
    ``pow``). An array ``x ** 2`` instead takes NumPy's ``x * x`` fast path, and on some
    libms the two differ in the last bit (MSVC CRT: 196 of 400,000 random doubles,
    measured 2026-10-01). Passing the exponent as an array bypasses the fast path, so the
    vectorised gains below equal the scalar gains bit for bit on the same platform.
    """
    return np.power(x, np.full(x.shape, 2.0))


def _best_split_for_feature(
    col: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    G,
    H,
    lambda_reg: float,
    min_samples_leaf: int,
    min_child_weight: float,
    gamma: float,
) -> tuple[float, float]:
    """Best (gain, threshold) for one feature of one node; ``(-inf, 0.0)`` if none.

    A vectorised twin of the scalar scan that v8.1 was trained with, which ran a Python
    loop over every sample of every node for every feature (~3.7 s per depth-3 tree on
    133,882 cases, 798 s for the 200-tree member alone). It reproduces that scan's
    arithmetic exactly -- the same ``np.argsort`` order, sequential accumulation
    (``np.cumsum`` is a left-to-right add; ``a + (-b)`` is exactly ``a - b``), the same
    expression order and ``pow`` for the gain, and the same first-maximum tie rule
    (the scan replaced its best only on a strictly greater gain) -- so every tree, and
    therefore every published v8.1 probability, is unchanged.

    Preserved quirk (deliberately, for model identity): the scan started accumulating at
    position ``min_samples_leaf - 1``, so the "left" statistics it scored omit the first
    ``min_samples_leaf - 1`` sorted samples (and the "right" ones include them), while the
    split it then applies sends those samples left. Correcting it changes the trees; that
    is a model revision to be evaluated, not something to slip into a speed-up.
    """
    n_total = col.shape[0]
    lo = int(min_samples_leaf)
    hi = n_total - lo
    if hi <= lo:
        return -np.inf, 0.0
    if lo < 1:
        raise ValueError("min_samples_leaf must be >= 1")

    order = np.argsort(col)
    sorted_col = col[order]
    step_g = g[order][lo - 1:hi - 1]
    step_h = h[order][lo - 1:hi - 1]

    g_left = np.cumsum(step_g)
    h_left = np.cumsum(step_h)
    g_right = np.cumsum(np.concatenate(([G], -step_g)))[1:]
    h_right = np.cumsum(np.concatenate(([H - lambda_reg], -step_h)))[1:]

    distinct = sorted_col[lo:hi] != sorted_col[lo - 1:hi - 1]
    heavy_enough = ~((h_left < min_child_weight) | (h_right < min_child_weight))
    gain = (
        _pow2_like_scalar(g_left) / (h_left + lambda_reg)
        + _pow2_like_scalar(g_right) / (h_right + lambda_reg)
        - (G ** 2) / (H)
    ) / 2.0 - gamma
    # A NaN gain never won in the scalar scan (``nan > best`` is False).
    eligible = distinct & heavy_enough & ~np.isnan(gain)
    if not eligible.any():
        return -np.inf, 0.0
    candidates = np.where(eligible, gain, -np.inf)
    k = int(np.argmax(candidates))  # first maximum == the scan's strict-">" winner
    threshold = (sorted_col[lo - 1 + k] + sorted_col[lo + k]) / 2.0
    return candidates[k], threshold


def _build_tree_recursive(
    X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    indices: np.ndarray,
    depth: int,
    max_depth: int,
    lambda_reg: float,
    min_samples_leaf: int = 5,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    colsample: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> dict:
    """Recursively build a single gradient boosted tree."""
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

    node_g = gradients[indices]
    node_h = hessians[indices]
    for j in feature_subset:
        gain, thresh = _best_split_for_feature(
            X[indices, j], node_g, node_h, G, H, lambda_reg,
            min_samples_leaf, min_child_weight, gamma,
        )
        # Features are visited in order and a later one wins only on a strictly greater
        # gain -- the scalar scan's cross-feature tie rule.
        if gain > best_gain:
            best_gain = gain
            best_feat = j
            best_thresh = thresh

    if best_gain <= 0 or best_feat < 0:
        return {"leaf": leaf_value}

    left_mask = X[indices, best_feat] <= best_thresh
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    if len(left_indices) < min_samples_leaf or len(right_indices) < min_samples_leaf:
        return {"leaf": leaf_value}

    left_child = _build_tree_recursive(
        X, gradients, hessians, left_indices,
        depth + 1, max_depth, lambda_reg,
        min_samples_leaf, min_child_weight, gamma, colsample, rng,
    )
    right_child = _build_tree_recursive(
        X, gradients, hessians, right_indices,
        depth + 1, max_depth, lambda_reg,
        min_samples_leaf, min_child_weight, gamma, colsample, rng,
    )

    return {
        "feature": best_feat,
        "threshold": best_thresh,
        "left": left_child,
        "right": right_child,
    }


def _predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a single tree."""
    n = X.shape[0]
    out = np.zeros(n)

    if "leaf" in tree:
        out[:] = tree["leaf"]
        return out

    feat = tree["feature"]
    thresh = tree["threshold"]
    left_mask = X[:, feat] <= thresh

    if np.any(left_mask):
        out[left_mask] = _predict_tree(tree["left"], X[left_mask])
    if np.any(~left_mask):
        out[~left_mask] = _predict_tree(tree["right"], X[~left_mask])

    return out


def train_histogram_gbt(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.08,
    lambda_reg: float = 1.0,
    scale_pos_weight: float = 1.0,
    n_bins: int = 64,
    min_samples_leaf: int = 5,
    min_child_weight: float = 1.0,
    colsample: float = 1.0,
    subsample: float = 1.0,
    gamma: float = 0.0,
) -> tuple[list[dict], float, float, list[float], list[float]]:
    """Train histogram-based gradient boosted trees for binary classification.

    Returns (trees, init_score, learning_rate, train_losses, train_aucs).
    """
    n = X.shape[0]
    rng = np.random.RandomState(42)

    # Sample weights for class imbalance
    sw = np.ones(n)
    sw[y == 1] = scale_pos_weight

    # Initial prediction (log-odds of positive class)
    p_pos = np.sum(y * sw) / np.sum(sw)
    p_pos = np.clip(p_pos, 0.01, 0.99)
    init_score = float(np.log(p_pos / (1 - p_pos)))
    F = np.full(n, init_score)

    trees: list[dict] = []
    train_losses: list[float] = []
    train_aucs: list[float] = []

    for t in range(n_trees):
        p = sigmoid(F)

        # Gradients and hessians for log-loss
        gradients = (p - y) * sw
        hessians = p * (1 - p) * sw

        # Subsample
        if subsample < 1.0:
            n_sub = max(1, int(n * subsample))
            sub_idx = rng.choice(n, size=n_sub, replace=False)
        else:
            sub_idx = np.arange(n)

        tree = _build_tree_recursive(
            X, gradients, hessians, sub_idx,
            depth=0, max_depth=max_depth, lambda_reg=lambda_reg,
            min_samples_leaf=min_samples_leaf,
            min_child_weight=min_child_weight,
            gamma=gamma, colsample=colsample, rng=rng,
        )

        update = _predict_tree(tree, X)
        F += learning_rate * update
        trees.append(tree)

        # Track training loss
        p_check = sigmoid(F)
        loss = -np.mean(
            sw * (y * np.log(p_check + 1e-15) + (1 - y) * np.log(1 - p_check + 1e-15))
        )
        train_losses.append(float(loss))

        if (t + 1) % 50 == 0:
            auc, _, _ = compute_roc_auc(y, p_check)
            train_aucs.append(float(auc))
            print(f"        Tree {t + 1}/{n_trees}, train AUC={auc:.4f}, loss={loss:.4f}")

    return trees, init_score, learning_rate, train_losses, train_aucs


def predict_histogram_gbt(
    X: np.ndarray,
    trees: list[dict],
    init_score: float,
    learning_rate: float,
) -> np.ndarray:
    """Predict probabilities with trained histogram GBT."""
    n = X.shape[0]
    F = np.full(n, init_score)
    for tree in trees:
        F += learning_rate * _predict_tree(tree, X)
    return sigmoid(F)


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def platt_calibrate(
    y_train: np.ndarray,
    p_train: np.ndarray,
    p_test: np.ndarray,
    n_epochs: int = 500,
    lr: float = 0.01,
) -> tuple[np.ndarray, float, float]:
    """Platt scaling: fit sigmoid(a * p + b) to training labels.

    Returns (calibrated_test_probs, a, b).
    """
    a = 1.0
    b = 0.0
    n = len(y_train)

    for _ in range(n_epochs):
        z = a * p_train + b
        q = sigmoid(z)
        grad_a = np.sum((q - y_train) * p_train) / n
        grad_b = np.sum(q - y_train) / n
        a -= lr * grad_a
        b -= lr * grad_b

    calibrated = sigmoid(a * p_test + b)
    return calibrated, float(a), float(b)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    """Compute ROC AUC from scratch."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5, [], []

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += 0.5 * (tpr_list[i] + tpr_list[i - 1]) * (fpr_list[i] - fpr_list[i - 1])

    return auc, fpr_list, tpr_list


def brier_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Brier Skill Score relative to climatology baseline."""
    bs = np.mean((y_pred - y_true) ** 2)
    clim = np.mean(y_true)
    bs_clim = np.mean((clim - y_true) ** 2)
    if bs_clim == 0:
        return 0.0
    return 1.0 - bs / bs_clim


def rank_ensemble(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> list[tuple[str, float]]:
    """Rank ensemble members by AUC."""
    results = []
    for name, preds in predictions.items():
        auc, _, _ = compute_roc_auc(y_true, preds)
        results.append((name, auc))
    results.sort(key=lambda x: -x[1])
    return results


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_operational_ri(
    cases: list[dict],
    test_start_year: int = 2020,
    config: OperationalRIConfig | None = None,
) -> dict:
    """Run full train/test benchmark on historical cases."""
    if config is None:
        config = best_known_operational_ri_config()

    train_cases = [c for c in cases if c.get("season_year", 0) < test_start_year]
    test_cases = [c for c in cases if c.get("season_year", 0) >= test_start_year]

    if not train_cases or not test_cases:
        return {"error": "insufficient data for split"}

    X_train, y_train, _, feature_names = build_feature_matrix(train_cases)
    X_test, y_test, _, _ = build_feature_matrix(test_cases, feature_names=feature_names)

    X_train, X_test = train_median_impute(X_train, X_test)
    X_train_std, X_test_std = standardize(X_train, X_test)

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    cw_ratio = min(neg / max(pos, 1), 20.0)

    # Feature selection
    sel_idx, sel_names, _ = select_features_by_logistic(
        X_train_std, y_train, feature_names,
        n_select=config.n_features_select,
        lam=config.feature_select_lam,
        n_epochs=config.feature_select_epochs,
        cw_ratio=cw_ratio,
    )

    X_tr_sel = X_train[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]
    X_tr_s, X_te_s = standardize(X_tr_sel, X_te_sel)

    # Train GBT depth-3
    gbt3_trees, gbt3_init, gbt3_lr, _, _ = train_histogram_gbt(
        X_tr_sel, y_train,
        n_trees=config.hgbt_d3_n_trees, max_depth=config.hgbt_d3_max_depth,
        learning_rate=config.hgbt_d3_lr, lambda_reg=config.hgbt_d3_lambda_reg,
        scale_pos_weight=cw_ratio, n_bins=config.n_bins,
    )
    p_gbt3 = predict_histogram_gbt(X_te_sel, gbt3_trees, gbt3_init, gbt3_lr)

    # Logistic
    w_lr, b_lr = train_logistic(
        X_tr_s, y_train,
        lam=config.logistic_lam, lr=config.logistic_lr,
        n_epochs=config.logistic_epochs_final, cw_ratio=cw_ratio,
    )
    p_lr = predict_logistic(X_te_s, w_lr, b_lr)

    # Ensemble
    p_ens = (p_gbt3 + p_lr) / 2.0
    auc_ens, _, _ = compute_roc_auc(y_test, p_ens)
    bss = brier_skill_score(y_test, p_ens)

    return {
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "n_features_selected": len(sel_idx),
        "test_auc": float(auc_ens),
        "test_bss": float(bss),
        "selected_features": sel_names,
    }
