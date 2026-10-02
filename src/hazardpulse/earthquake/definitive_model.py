# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational earthquake prediction system.
# It does NOT replace official USGS or national seismological agency warnings.
# Always follow guidance from your national geological survey.
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Definitive leak-proof earthquake prediction model v2.

Design philosophy: ZERO LEAKAGE BY CONSTRUCTION.
    - ONE model type: Gradient Boosted Trees. No meta-stacker, no blending.
    - Strict temporal separation enforced with assertions.
    - Labels require temporal ordering in code.
    - All features verified causal (available at prediction time).
    - No hyperparameter tuning on test. Test touched ONCE.

Feature architecture (v2 -- mega-model):
    Block S: 61 seismicity features (full v8c port, numpy-vectorized)
    Block C: 12 coherence field theory features
    Block X:  8 cross-domain interaction terms

Three model variants (one GBT each, NO stacking):
    1. Baseline:  Block S only (61 features) -- full v8c seismicity
    2. Enhanced:  Block S + C (73 features) -- + coherence field theory
    3. Full:      Block S + C + X (81 features) -- + cross-domain interactions

Temporal splits (hardcoded, non-negotiable):
    Train: 2005-01-01 to 2017-12-31
    Val:   2018-01-01 to 2019-12-31
    Test:  2020-01-01 to 2024-12-31

Label definition:
    Positive: M6+ mainshock within 300km and 365 days FORWARD
    Negative: same location, different time, no M6+ within window
    Gardner-Knopoff aftershock declustering applied first.
    2 negatives per positive, time-offset +/-1.5 to +/-4.5 years.

Performance: numpy-vectorized feature extraction (~1000 samples/min).
Self-contained: copies GBT implementation fresh, imports only from
hazardpulse.data.earthquake and hazardpulse.earthquake.coherence_engine.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Internal imports (the ONLY allowed external dependencies)
# ---------------------------------------------------------------------------

from hazardpulse.data.earthquake import load_usgs_catalog
from hazardpulse.earthquake.coherence_engine import (
    extract_coherence_features,
    compute_seismic_coherence_field,
    test_earthquake_singularity,
    _parse_event_time,
    _haversine_km,
    _haversine_km_arrays,
    compute_b_value,
    compute_seismic_moment,
)


# ===================================================================
# TEMPORAL SPLITS -- HARDCODED, NON-NEGOTIABLE
# ===================================================================

TRAIN_START: int = 2005
TRAIN_END: int = 2017
VAL_START: int = 2018
VAL_END: int = 2019
TEST_START: int = 2020
TEST_END: int = 2024

# Label parameters. Env-overridable so we can build a SHORT-TERM LOCAL nowcast (small
# radius / short window) alongside the year-ahead regional one: shrinking the window
# restores label rarity (and thus quiet-control contrast) at low magnitudes, where the
# year-ahead label is trivially almost-always-true.
LABEL_RADIUS_KM: float = float(os.environ.get("HAZARDPULSE_LABEL_RADIUS_KM", "300.0"))
FORWARD_WINDOW_DAYS: float = float(os.environ.get("HAZARDPULSE_FORWARD_WINDOW_DAYS", "365.0"))
# Target magnitude. Env-overridable so we can test lower thresholds (M5+/M5.5+ give
# ~5-10x more positive samples -- denser precursory signal, more training data).
MIN_MAINSHOCK_MAG: float = float(os.environ.get("HAZARDPULSE_MIN_MAINSHOCK_MAG", "6.0"))
CONTROL_RATIO: int = 2  # negatives per positive
CONTROL_OFFSET_RANGE: tuple[float, float] = (1.5, 4.5)  # years

# Time constants
SEC_PER_DAY: float = 86400.0
SEC_PER_YEAR: float = 365.25 * SEC_PER_DAY
# Block S reads the 5 years (of 365.25 d) before the reference time; a live scorer
# must hand it a catalog at least this long or the features differ from training.
BLOCK_S_LOOKBACK_DAYS: float = 5.0 * 365.25


# ===================================================================
# FEATURE BLOCK SIZES
# ===================================================================

N_FEAT_S: int = 61   # Full v8c seismicity
N_FEAT_C: int = 12   # Coherence field theory
N_FEAT_X: int = 8    # Cross-domain interactions

N_FEAT_BASELINE: int = N_FEAT_S                          # 61
N_FEAT_ENHANCED: int = N_FEAT_S + N_FEAT_C               # 73
N_FEAT_FULL: int = N_FEAT_S + N_FEAT_C + N_FEAT_X        # 81


# ===================================================================
# FEATURE NAMES -- every feature has a causal validity comment
# ===================================================================

# Block S: Full v8c Seismicity (61 features)
# All derived from PAST seismicity at the sample location.
BLOCK_S_NAMES: list[str] = [
    # b-value features (6)
    "b_trend",          # b_late - b_early -- past
    "b_recent",         # b_90d - b_older -- past
    "mc_late",          # magnitude of completeness (recent half) -- past
    "mc_change",        # Mc_late - Mc_early -- past
    "rate_mc_norm",     # above-Mc rate ratio late/early -- past
    "b_near",           # b-value within 100km -- past
    # Rate acceleration (7)
    "rate_1m",          # 1-month rate acceleration -- past
    "rate_3m",          # 3-month rate acceleration -- past
    "rate_6m",          # 6-month rate acceleration -- past
    "rate_12m",         # 12-month rate acceleration -- past
    "rate_7_30",        # 7-day / 30-day rate ratio -- past
    "rate_30_90",       # 30-day / 90-day rate ratio -- past
    "rate_90_365",      # 90-day / 365-day rate ratio -- past
    # NN/clustering (6)
    "nn_change",        # NN distance change (recent vs older) -- past
    "st_nn",            # spatiotemporal NN metric -- past
    "st_nn_std",        # std of spatiotemporal NN -- past
    "frac_clust",       # fraction of clustered events -- past
    "centroid_mig",     # centroid migration (km) -- past
    "elong",            # elongation ratio (eigenvalue) -- past
    # Magnitude (5)
    "maxmag_30d",       # max magnitude in last 30 days -- past
    "maxmag_90d",       # max magnitude in last 90 days -- past
    "maxmag_180d",      # max magnitude in last 180 days -- past
    "mag_var_chg",      # magnitude variance change -- past
    "mag_range_30d",    # magnitude range in last 30 days -- past
    # Coulomb (4)
    "coulomb",          # Coulomb stress proxy (M5+, <100km, <180d) -- past
    "coulomb_moment",   # moment-weighted Coulomb -- past
    "coulomb_n",        # count of M5+ in Coulomb zone -- past
    "coulomb_broad",    # broad Coulomb (M5+, <300km, <365d) -- past
    # Foreshock (4)
    "inv_omori",        # inverse Omori correlation -- past
    "bath_ratio",       # Bath's law ratio -- past
    "foreshock_r",      # foreshock fraction -- past
    "quiescence_7d",    # 7-day quiescence signal -- past
    # Moment (5)
    "mom_90d",          # log10 moment in 90 days -- past
    "mom_180d",         # log10 moment in 180 days -- past
    "mom_1y",           # log10 moment in 1 year -- past
    "mom_accel",        # moment acceleration -- past
    "mom_deficit",      # moment deficit vs annual rate -- past
    # Depth (4)
    "depth_mean",       # mean depth -- past
    "depth_std",        # depth std -- past
    "depth_trend",      # depth trend (late - early mean) -- past
    "depth_bimod",      # depth bimodality -- past
    # Tectonic regime (1)
    "subduc",           # subduction indicator -- past
    # Counts (5)
    "n_events",         # total event count -- past
    "n_7d",             # events in last 7 days -- past
    "n_14d",            # events in last 14 days -- past
    "n_30d",            # events in last 30 days -- past
    "n_90d",            # events in last 90 days -- past
    # Recurrence (4)
    "m5_accel",         # M5+ acceleration -- past
    "spatial_conc",     # spatial concentration (std of dists) -- past
    "mag_range_90d",    # magnitude range in last 90 days -- past
    "m6_recur",         # M6+ recurrence interval (years) -- past
    # v8c interactions (4) -- reduced to non-redundant set
    "b_x_rate",         # b_trend * rate_3m -- past
    "coul_x_rate",      # coulomb * rate_1m -- past
    "mom_x_clust",      # mom_accel * frac_clust -- past
    "fore_x_bath",      # foreshock_r * bath_ratio -- past
    # v8c near-field (6)
    "near_n_90d",       # events <100km in 90 days -- past
    "near_mom_90d",     # log10 moment <100km in 90 days -- past
    "near_iet_cv",      # near-field interevent time CV -- past
    "near_iet_min",     # near-field min interevent time -- past
    "near_far_ratio",   # near/far rate ratio -- past
    "near_rate_x_b",    # near_n_90d * b_trend -- past
]

# Block C: Coherence Field Theory (12 features)
BLOCK_C_NAMES: list[str] = [
    "ell",                  # Correlation length (km) -- past
    "ell_trend",            # Correlation length change rate -- past
    "ell_acceleration",     # d^2(ell)/dt^2 -- past
    "days_to_criticality",  # Estimated days until t_c -- past divergence fit
    "nu_exponent",          # Critical exponent from ell(t) fit -- past
    "divergence_r2",        # R^2 of divergence fit (confidence) -- past
    "delta_aic_iet",        # Lorentzian vs exponential IET -- past
    "tau_local",            # Helmholtz coherence at location -- past
    "grad_tau_local",       # Coherence gradient -- past
    "S_over_Gamma",         # Source/damping ratio -- past
    "Da_local",             # Damkohler number -- past
    "singularity_count",    # Conditions met (0-5) -- past
]

# Block X: Cross-domain Interactions (8 features)
BLOCK_X_NAMES: list[str] = [
    "ell_x_b_trend",            # ell * b_trend -- past
    "ell_x_rate_1m",            # ell * rate_1m -- past
    "ell_x_coulomb",            # ell * coulomb -- past
    "tau_x_maxmag_90d",         # tau_local * maxmag_90d -- past
    "dtc_x_singularity",        # days_to_criticality * singularity_count -- past
    "sg_x_mom_accel",           # S_over_Gamma * mom_accel -- past
    "daic_x_frac_clust",        # delta_aic_iet * frac_clust -- past
    "ell_accel_x_rate_7_30",    # ell_acceleration * rate_7_30 -- past
]

ALL_FEATURE_NAMES_BASELINE: list[str] = BLOCK_S_NAMES
ALL_FEATURE_NAMES_ENHANCED: list[str] = BLOCK_S_NAMES + BLOCK_C_NAMES
ALL_FEATURE_NAMES_FULL: list[str] = (
    BLOCK_S_NAMES + BLOCK_C_NAMES + BLOCK_X_NAMES
)


# ===================================================================
# GBT HYPERPARAMETERS -- FIXED A PRIORI, NOT TUNED ON VALIDATION
# ===================================================================

GBT_N_TREES: int = 300
GBT_MAX_DEPTH: int = 4
GBT_LEARNING_RATE: float = 0.03
GBT_SUBSAMPLE: float = 0.6
GBT_COLSAMPLE: float = 0.7
GBT_MIN_SAMPLES_LEAF: int = 15
GBT_L2_REG: float = 1.0
GBT_GAMMA: float = 0.05

# Early stopping patience (on validation loss)
EARLY_STOP_PATIENCE: int = 50


# ===================================================================
# VECTORIZED HAVERSINE
# ===================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def haversine_vec(
    lat1: float, lon1: float,
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine: one point vs array of points. Returns km."""
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lats))
        * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def mag_to_moment(mag: np.ndarray) -> np.ndarray:
    """Convert magnitude(s) to seismic moment (N*m)."""
    return 10.0 ** (1.5 * np.asarray(mag, dtype=np.float64) + 9.05)


# ===================================================================
# SIGMOID (numerically stable)
# ===================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -88.0, 88.0)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    ).astype(np.float32)


# ===================================================================
# V8C HELPER: PROPER B-VALUE (MAXIMUM CURVATURE)
# ===================================================================

def estimate_mc_maxcurv(magnitudes: np.ndarray, bin_width: float = 0.1) -> float:
    """Estimate magnitude of completeness using maximum curvature method."""
    if len(magnitudes) < 10:
        return 4.0
    bins = np.arange(
        np.floor(np.min(magnitudes) * 10) / 10,
        np.ceil(np.max(magnitudes) * 10) / 10 + bin_width,
        bin_width,
    )
    if len(bins) < 2:
        return 4.0
    hist, edges = np.histogram(magnitudes, bins=bins)
    if len(hist) == 0:
        return 4.0
    max_idx = np.argmax(hist)
    mc = (edges[max_idx] + edges[max_idx + 1]) / 2.0 + 0.2
    return mc


def b_value_proper(magnitudes: np.ndarray) -> tuple[float, float]:
    """Compute b-value using Aki-Utsu MLE with max-curvature Mc."""
    mc = estimate_mc_maxcurv(magnitudes)
    above = magnitudes[magnitudes >= mc]
    if len(above) < 5:
        return 1.0, mc
    b = np.log10(np.e) / (np.mean(above) - (mc - 0.05))
    return b, mc


# ===================================================================
# GRADIENT BOOSTED TREES -- SELF-CONTAINED IMPLEMENTATION
# ===================================================================

class GradientBoostedTrees:
    """Memory-efficient gradient boosted trees for binary classification.

    Uses log-loss (cross-entropy) with Newton-Raphson leaf values.
    Supports subsample, colsample, balanced class weights, L2 regularization,
    and minimum complexity gain (gamma) pruning.
    """

    def __init__(
        self,
        n_trees: int = GBT_N_TREES,
        max_depth: int = GBT_MAX_DEPTH,
        learning_rate: float = GBT_LEARNING_RATE,
        min_samples_leaf: int = GBT_MIN_SAMPLES_LEAF,
        subsample: float = GBT_SUBSAMPLE,
        colsample: float = GBT_COLSAMPLE,
        l2_reg: float = GBT_L2_REG,
        gamma: float = GBT_GAMMA,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.trees: list[dict] = []
        self.init_pred: float = 0.0
        # Seeded in fit(); _build_tree must never touch global RNG state.
        self._rng: np.random.RandomState | None = None

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int = 0,
    ) -> dict:
        """Build a single regression tree on gradient/hessian targets."""
        N = len(gradients)
        G = float(np.sum(gradients))
        H = float(np.sum(hessians))

        leaf_val = -G / (H + self.l2_reg) if (H + self.l2_reg) > 1e-12 else 0.0

        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {"leaf": True, "val": float(leaf_val)}

        D = X.shape[1]
        n_try = max(5, int(D * self.colsample))
        rng = self._rng if self._rng is not None else np.random.RandomState(0)
        feat_idx = rng.choice(D, size=min(n_try, D), replace=False)

        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0

        score_no_split = G ** 2 / (H + self.l2_reg)

        for f in feat_idx:
            col = X[:, f]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > 20:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))
            else:
                thresholds = unique_vals[:-1]

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())

                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue

                GL = float(np.sum(gradients[left_mask]))
                HL = float(np.sum(hessians[left_mask]))
                GR = float(np.sum(gradients[right_mask]))
                HR = float(np.sum(hessians[right_mask]))

                gain = 0.5 * (
                    GL ** 2 / (HL + self.l2_reg)
                    + GR ** 2 / (HR + self.l2_reg)
                    - score_no_split
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)

        if best_gain <= 0:
            return {"leaf": True, "val": float(leaf_val)}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_child = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
        )
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": left_child,
            "right": right_child,
        }

    def _predict_tree_row(self, node: dict, x: np.ndarray) -> float:
        """Predict a single row through a single tree."""
        while not node["leaf"]:
            if x[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["val"]

    def _predict_tree_batch(self, node: dict, X: np.ndarray) -> np.ndarray:
        """Predict all rows through a single tree, row by row."""
        result = np.empty(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            result[i] = self._predict_tree_row(node, X[i])
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict:
        """Train the GBT ensemble on labelled data."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = len(y)

        # Balanced class weights
        n_pos = float(y.sum())
        n_neg = float(N - n_pos)
        assert n_pos > 0, "No positive samples in training set"
        assert n_neg > 0, "No negative samples in training set"
        sample_w = np.where(
            y == 1,
            N / (2.0 * n_pos),
            N / (2.0 * n_neg),
        ).astype(np.float32)

        # Initial prediction: log-odds of training prevalence
        p_avg = float(np.clip(y.mean(), 0.01, 0.99))
        self.init_pred = math.log(p_avg / (1.0 - p_avg))
        F_train = np.full(N, self.init_pred, dtype=np.float32)

        # Validation setup for early stopping
        use_early_stop = X_val is not None and y_val is not None
        F_val = None
        best_val_loss = float("inf")
        patience_counter = 0
        best_n_trees = 0

        if use_early_stop:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            F_val = np.full(len(y_val), self.init_pred, dtype=np.float32)

        rng = np.random.RandomState(42)
        self._rng = rng
        self.trees = []

        for t in range(self.n_trees):
            p_train = sigmoid(F_train)

            gradients = ((p_train - y) * sample_w).astype(np.float32)
            hessians = (p_train * (1.0 - p_train) * sample_w + 1e-6).astype(
                np.float32
            )

            if self.subsample < 1.0:
                n_sub = max(1, int(N * self.subsample))
                idx = rng.choice(N, size=n_sub, replace=False)
                tree = self._build_tree(
                    X[idx], gradients[idx], hessians[idx]
                )
            else:
                tree = self._build_tree(X, gradients, hessians)

            self.trees.append(tree)
            F_train += self.lr * self._predict_tree_batch(tree, X)

            if use_early_stop:
                F_val += self.lr * self._predict_tree_batch(tree, X_val)
                p_val = sigmoid(F_val)
                val_loss = -float(np.mean(
                    y_val * np.log(p_val + 1e-12)
                    + (1 - y_val) * np.log(1 - p_val + 1e-12)
                ))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_n_trees = t + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOP_PATIENCE:
                    if verbose:
                        print(
                            f"      Early stop at tree {t + 1}, "
                            f"best_val_loss={best_val_loss:.6f} at tree {best_n_trees}"
                        )
                        sys.stdout.flush()
                    self.trees = self.trees[:best_n_trees]
                    return {
                        "n_trees_used": best_n_trees,
                        "stopped_early": True,
                        "best_val_loss": best_val_loss,
                    }

            if verbose and (t + 1) % 25 == 0:
                train_auc = compute_auc(y, sigmoid(F_train))
                msg = f"      Tree {t + 1}/{self.n_trees}: train_auc={train_auc:.4f}"
                if use_early_stop:
                    msg += f"  val_loss={best_val_loss:.6f}"
                print(msg)
                sys.stdout.flush()

        return {
            "n_trees_used": len(self.trees),
            "stopped_early": False,
            "best_val_loss": best_val_loss if use_early_stop else None,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        F = np.full(N, self.init_pred, dtype=np.float32)
        for tree in self.trees:
            for i in range(N):
                F[i] += self.lr * self._predict_tree_row(tree, X[i])
        return sigmoid(F)

    def feature_importances(self, n_features: int) -> np.ndarray:
        """Compute feature importance from tree split counts."""
        counts = np.zeros(n_features, dtype=np.float64)

        def _walk(node: dict) -> None:
            if node["leaf"]:
                return
            f = node["feat"]
            if 0 <= f < n_features:
                counts[f] += 1
            _walk(node["left"])
            _walk(node["right"])

        for tree in self.trees:
            _walk(tree)

        total = counts.sum()
        if total > 0:
            counts /= total
        return counts


# ===================================================================
# EVALUATION METRICS
# ===================================================================

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via trapezoidal integration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) < 2:
        return 0.5
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC -- better metric for imbalanced data."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = y_true.sum()
    if n_pos == 0 or len(y_true) == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    pr_auc = 0.0
    recall_prev = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / n_pos
        pr_auc += (recall - recall_prev) * precision
        recall_prev = recall

    return float(pr_auc)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score: mean squared error between predicted proba and labels."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_bss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier Skill Score: improvement over climatological baseline."""
    brier = compute_brier(y_true, y_pred)
    base_rate = float(np.mean(y_true))
    brier_clim = base_rate * (1 - base_rate)
    if brier_clim < 1e-12:
        return 0.0
    return float(1.0 - brier / brier_clim)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Reliability diagram: predicted vs observed frequency per bin."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means: list[float] = []
    bin_trues: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        count = int(mask.sum())
        if count > 0:
            bin_means.append(float(y_pred[mask].mean()))
            bin_trues.append(float(y_true[mask].mean()))
            bin_counts.append(count)

    return {"predicted": bin_means, "observed": bin_trues, "counts": bin_counts}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    N = len(y_true)
    aucs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        aucs[b] = compute_auc(y_true[idx], y_score[idx])

    return {
        "mean": float(np.mean(aucs)),
        "ci_lo": float(np.percentile(aucs, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
        "std": float(np.std(aucs)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Complete evaluation suite for a single model."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "auc": compute_auc(y_true, y_pred),
        "pr_auc": compute_pr_auc(y_true, y_pred),
        "brier": compute_brier(y_true, y_pred),
        "bss": compute_bss(y_true, y_pred),
        "calibration": compute_calibration_curve(y_true, y_pred, n_bins=10),
        "bootstrap_ci": bootstrap_auc_ci(y_true, y_pred, n_boot=2000),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "base_rate": float(y_true.mean()),
    }


# ===================================================================
# SIGNIFICANCE TESTS
# ===================================================================

def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Test whether model B significantly outperforms model A."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64)
    y_pred_b = np.asarray(y_pred_b, dtype=np.float64)
    N = len(y_true)

    auc_a_full = compute_auc(y_true, y_pred_a)
    auc_b_full = compute_auc(y_true, y_pred_b)
    delta_full = auc_b_full - auc_a_full

    deltas = np.empty(n_boot, dtype=np.float64)
    n_a_wins = 0

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        auc_a = compute_auc(y_true[idx], y_pred_a[idx])
        auc_b = compute_auc(y_true[idx], y_pred_b[idx])
        deltas[b] = auc_b - auc_a
        if auc_a >= auc_b:
            n_a_wins += 1

    p_value = n_a_wins / n_boot

    return {
        "delta_auc": float(delta_full),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_value": float(p_value),
    }


# ===================================================================
# FEATURE NORMALIZER
# ===================================================================

class FeatureNormalizer:
    """Z-score normalizer. Fit on train, apply to val/test."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0, dtype=np.float32)
        self.std = X.std(axis=0, dtype=np.float32)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "FeatureNormalizer has not been fit"
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ===================================================================
# GARDNER-KNOPOFF AFTERSHOCK DECLUSTERING
# ===================================================================

def _gardner_knopoff_window(mag: float) -> tuple[float, float]:
    """Return (distance_km, time_days) window for aftershock removal."""
    d_km = 10.0 ** (0.1238 * mag + 0.983)
    if mag >= 6.5:
        t_days = 10.0 ** (0.032 * mag + 2.7389)
    else:
        t_days = 10.0 ** (0.5409 * mag - 0.547)
    return d_km, t_days


def decluster_gardner_knopoff(
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove aftershocks using Gardner-Knopoff (1974) windows."""
    parsed: list[tuple[float, dict]] = []
    for e in events:
        t_str = e.get("time", "")
        epoch = _parse_event_time(t_str) if isinstance(t_str, str) else 0.0
        if epoch > 0 and e.get("mag") is not None:
            parsed.append((epoch, e))

    parsed.sort(key=lambda x: x[1]["mag"], reverse=True)

    is_aftershock = set()
    n = len(parsed)

    epochs = np.array([p[0] for p in parsed])
    lats = np.array([p[1]["latitude"] for p in parsed])
    lons = np.array([p[1]["longitude"] for p in parsed])
    mags = np.array([p[1]["mag"] for p in parsed])

    for i in range(n):
        if i in is_aftershock:
            continue
        if mags[i] < 5.0:
            break
        d_km, t_days = _gardner_knopoff_window(float(mags[i]))
        t_sec = t_days * 86400.0

        dt = np.abs(epochs - epochs[i])
        time_mask = dt <= t_sec
        mag_mask = mags < mags[i]
        candidates = np.where(time_mask & mag_mask)[0]

        for j in candidates:
            if j == i or j in is_aftershock:
                continue
            dist = haversine_km(
                float(lats[i]), float(lons[i]),
                float(lats[j]), float(lons[j]),
            )
            if dist <= d_km:
                is_aftershock.add(j)

    mainshocks = [parsed[i][1] for i in range(n) if i not in is_aftershock]
    aftershocks = [parsed[i][1] for i in range(n) if i in is_aftershock]

    return mainshocks, aftershocks


# ===================================================================
# EVENT TIME UTILITIES
# ===================================================================

def _event_epoch(event: dict) -> float:
    """Extract epoch seconds from an event dict.

    The parsed epoch is memoized on the dict (`_epoch`) -- strptime is ~80% of block_c
    cost and the same ~494k events are re-parsed millions of times per sample. Memoizing
    makes every parse after the first O(1). Identical value; string events only.
    """
    t_str = event.get("time", "")
    if isinstance(t_str, str):
        cached = event.get("_epoch")
        if cached is not None:
            return cached
        v = _parse_event_time(t_str)
        event["_epoch"] = v
        return v
    return 0.0


def _event_year(event: dict) -> int:
    """Extract year from an event's ISO-8601 timestamp."""
    t_str = event.get("time", "")
    if isinstance(t_str, str) and len(t_str) >= 4:
        try:
            return int(t_str[:4])
        except ValueError:
            pass
    return 0


# ===================================================================
# SAMPLE GENERATION
# ===================================================================

def identify_target_mainshocks(
    mainshocks: list[dict],
    min_mag: float = MIN_MAINSHOCK_MAG,
) -> list[dict]:
    """Filter declustered catalog to M6+ mainshocks with valid locations."""
    targets = []
    for e in mainshocks:
        mag = e.get("mag")
        lat = e.get("latitude")
        lon = e.get("longitude")
        if (
            mag is not None
            and mag >= min_mag
            and lat is not None
            and lon is not None
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            targets.append(e)
    return targets


# Cache the qualifying-mainshock (mag >= MIN_MAINSHOCK_MAG) arrays once per mainshock
# list, so the per-target control generation is a vectorized proximity check instead of
# a Python loop over all ~256k mainshocks (the O(targets x mainshocks) bottleneck that
# made M4.5 build_samples take ~9h). Keyed by id() of the list (stable within a run).
_CONTROL_QUAL_CACHE: dict = {}


def _control_qual_arrays(mainshocks):
    key = id(mainshocks)
    cached = _CONTROL_QUAL_CACHE.get(key)
    if cached is None:
        lats, lons, eps = [], [], []
        for m in mainshocks:
            if m.get("mag", 0) >= MIN_MAINSHOCK_MAG:
                lats.append(m["latitude"]); lons.append(m["longitude"]); eps.append(_event_epoch(m))
        _CONTROL_QUAL_CACHE.clear()   # keep only the latest list (bounded memory)
        cached = (np.asarray(lats, float), np.asarray(lons, float), np.asarray(eps, float))
        _CONTROL_QUAL_CACHE[key] = cached
    return cached


def generate_control_samples(
    target: dict,
    mainshocks: list[dict],
    catalog_epochs: np.ndarray,
    n_controls: int = CONTROL_RATIO,
    offset_range: tuple[float, float] = CONTROL_OFFSET_RANGE,
    label_radius_km: float = LABEL_RADIUS_KM,
    forward_days: float = FORWARD_WINDOW_DAYS,
    rng: np.random.RandomState | None = None,
) -> list[dict]:
    """Generate negative control samples at the SAME location as a target."""
    if rng is None:
        rng = np.random.RandomState(42)

    target_epoch = _event_epoch(target)
    target_lat = target["latitude"]
    target_lon = target["longitude"]

    # Vectorized proximity precompute: epochs of qualifying mainshocks within the label
    # radius of this target (distance is fixed per target). The per-candidate check then
    # only tests the time window -- identical result to the old per-mainshock Python loop,
    # ~50x faster.
    ql, qo, qe = _control_qual_arrays(mainshocks)
    if len(ql):
        near_epochs = qe[haversine_vec(target_lat, target_lon, ql, qo) <= label_radius_km]
    else:
        near_epochs = np.empty(0, dtype=float)

    controls: list[dict] = []
    max_attempts = n_controls * 10

    for _ in range(max_attempts):
        if len(controls) >= n_controls:
            break

        offset_years = rng.uniform(offset_range[0], offset_range[1])
        if rng.random() < 0.5:
            offset_years = -offset_years
        offset_sec = offset_years * 365.25 * 86400.0
        control_epoch = target_epoch + offset_sec

        # any qualifying mainshock within radius AND in (control_epoch, +forward_days]?
        dt_days = (near_epochs - control_epoch) / 86400.0
        has_m6_forward = bool(((dt_days >= 0) & (dt_days <= forward_days)).any())

        if not has_m6_forward:
            ctrl_dt = _dt.datetime.fromtimestamp(
                control_epoch, tz=_dt.timezone.utc
            )
            ctrl_time_str = ctrl_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            controls.append({
                "latitude": target_lat,
                "longitude": target_lon,
                "time": ctrl_time_str,
                "ref_epoch": control_epoch,
                "label": 0,
            })

    return controls


def build_samples(
    mainshocks: list[dict],
    full_catalog: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Build positive + negative samples from declustered catalog."""
    targets = identify_target_mainshocks(mainshocks)
    if verbose:
        print(f"    M6+ mainshocks found: {len(targets)}")
        sys.stdout.flush()

    ms_epochs = np.array([_event_epoch(m) for m in mainshocks], dtype=np.float64)

    rng = np.random.RandomState(42)
    samples: list[dict] = []

    for target in targets:
        target_epoch = _event_epoch(target)
        target_year = _event_year(target)

        samples.append({
            "latitude": target["latitude"],
            "longitude": target["longitude"],
            "ref_epoch": target_epoch,
            "label": 1,
            "year": target_year,
        })

        controls = generate_control_samples(
            target, mainshocks, ms_epochs, rng=rng,
        )
        for ctrl in controls:
            ctrl_year = int(ctrl["time"][:4]) if isinstance(ctrl["time"], str) else 0
            samples.append({
                "latitude": ctrl["latitude"],
                "longitude": ctrl["longitude"],
                "ref_epoch": ctrl["ref_epoch"],
                "label": ctrl["label"],
                "year": ctrl_year,
            })

    if verbose:
        n_pos = sum(1 for s in samples if s["label"] == 1)
        n_neg = sum(1 for s in samples if s["label"] == 0)
        print(f"    Total samples: {len(samples)} ({n_pos} pos, {n_neg} neg)")
        sys.stdout.flush()

    return samples


# ===================================================================
# PRE-BUILT NUMPY ARRAYS FOR FAST FEATURE EXTRACTION
# ===================================================================

class CatalogArrays:
    """Pre-built numpy arrays from full catalog for vectorized features.

    Converts the full catalog to sorted numpy arrays ONCE at startup.
    Provides fast spatial pre-filtering via 5-degree lat/lon bins.
    """

    def __init__(self, full_catalog: list[dict], verbose: bool = True) -> None:
        if verbose:
            print("    Pre-building numpy arrays from catalog...")
            sys.stdout.flush()

        n = len(full_catalog)
        self.times = np.empty(n, dtype=np.float64)
        self.lats = np.empty(n, dtype=np.float64)
        self.lons = np.empty(n, dtype=np.float64)
        self.mags = np.empty(n, dtype=np.float64)
        self.depths = np.empty(n, dtype=np.float64)

        for i, e in enumerate(full_catalog):
            t_str = e.get("time", "")
            self.times[i] = _parse_event_time(t_str) if isinstance(t_str, str) else 0.0
            self.lats[i] = e.get("latitude", 0.0)
            self.lons[i] = e.get("longitude", 0.0)
            self.mags[i] = e.get("mag", 0.0)
            self.depths[i] = e.get("depth", 10.0) if e.get("depth") is not None else 10.0

        # Sort by time
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.lats = self.lats[order]
        self.lons = self.lons[order]
        self.mags = self.mags[order]
        self.depths = self.depths[order]

        # Pre-build M6+ arrays for recurrence features
        m6_mask = self.mags >= 6.0
        self.m6_times = self.times[m6_mask]
        self.m6_lats = self.lats[m6_mask]
        self.m6_lons = self.lons[m6_mask]

        if verbose:
            print(f"    Arrays built: {n} events, {int(m6_mask.sum())} M6+")
            sys.stdout.flush()


# ===================================================================
# BLOCK S: FULL V8C SEISMICITY FEATURES (61 features, numpy-vectorized)
# ===================================================================

def compute_block_s(
    ev_lat: float,
    ev_lon: float,
    ev_time: float,
    cat: CatalogArrays,
) -> np.ndarray | None:
    """Compute all 60 Block S features using v8c numpy-vectorized approach.

    Parameters
    ----------
    ev_lat, ev_lon : float
        Sample location (degrees).
    ev_time : float
        Reference time (epoch seconds). Only events BEFORE this are used.
    cat : CatalogArrays
        Pre-built numpy arrays from full catalog.

    Returns
    -------
    ndarray, shape (60,), dtype float32, or None if insufficient data.
    """
    # Select events: before ev_time, within 5 years, within ~500km box.
    # cat.times is sorted -> binary-search the [t_start, ev_time) window (O(log N))
    # instead of an O(N) boolean mask over all ~0.5M events, then spatial-mask the
    # (much smaller) slice. Identical event set, just faster.
    t_start = ev_time - BLOCK_S_LOOKBACK_DAYS * SEC_PER_DAY
    i0 = int(np.searchsorted(cat.times, t_start, side="left"))
    i1 = int(np.searchsorted(cat.times, ev_time, side="left"))
    box_deg = 4.5
    sm = (
        (np.abs(cat.lats[i0:i1] - ev_lat) < box_deg)
        & (np.abs(cat.lons[i0:i1] - ev_lon) < box_deg)
    )

    if np.sum(sm) < 10:
        return None

    t = cat.times[i0:i1][sm]
    la = cat.lats[i0:i1][sm]
    lo = cat.lons[i0:i1][sm]
    m = cat.mags[i0:i1][sm]
    d = cat.depths[i0:i1][sm]
    dists = haversine_vec(ev_lat, ev_lon, la, lo)

    # Precise radius: 500km
    rmask = dists < 500
    if np.sum(rmask) < 10:
        return None
    t = t[rmask]
    la = la[rmask]
    lo = lo[rmask]
    m = m[rmask]
    d = d[rmask]
    dists = dists[rmask]
    dt = (ev_time - t) / SEC_PER_DAY  # days before event
    N = len(t)

    feats: dict[str, float] = {}

    # ---- b-value features ----
    if N >= 30:
        half = N // 2
        b_early, mc_early = b_value_proper(m[:half])
        b_late, mc_late = b_value_proper(m[half:])
        feats["b_trend"] = b_late - b_early

        r90 = dt < 90
        if np.sum(r90) >= 10 and np.sum(~r90) >= 10:
            b_rec, _ = b_value_proper(m[r90])
            b_old, _ = b_value_proper(m[~r90])
            feats["b_recent"] = b_rec - b_old
        else:
            feats["b_recent"] = 0.0

        feats["mc_late"] = mc_late

        # Mc trend
        mc_e = estimate_mc_maxcurv(m[:half])
        mc_l = estimate_mc_maxcurv(m[half:])
        feats["mc_change"] = mc_l - mc_e
        above_e = np.sum(m[:half] >= mc_e)
        above_l = np.sum(m[half:] >= mc_l)
        ht = (t[half] - t[0]) / SEC_PER_YEAR
        lt = (t[-1] - t[half]) / SEC_PER_YEAR
        if ht > 0 and lt > 0:
            feats["rate_mc_norm"] = (above_l / lt) / (above_e / ht + 0.01)
        else:
            feats["rate_mc_norm"] = 1.0
    else:
        feats["b_trend"] = 0.0
        feats["b_recent"] = 0.0
        feats["mc_late"] = 4.0
        feats["mc_change"] = 0.0
        feats["rate_mc_norm"] = 1.0

    # Near-field b-value
    near = dists < 100
    near_m = m[near]
    if len(near_m) >= 15:
        b_near, _ = b_value_proper(near_m)
        feats["b_near"] = b_near
    else:
        feats["b_near"] = 1.0

    # ---- Rate acceleration ----
    for wname, wdays in [("1m", 30), ("3m", 90), ("6m", 180), ("12m", 365)]:
        n_win = np.sum(dt < wdays)
        n_base = np.sum((dt >= wdays) & (dt < wdays * 3))
        base_rate = n_base / (2.0 * wdays + 0.01)
        win_rate = n_win / (wdays + 0.01)
        feats[f"rate_{wname}"] = float(np.log1p(win_rate / (base_rate + 0.001)))

    # Fine-grained rate ratios
    r7d = float(np.sum(dt < 7))
    r30d = float(np.sum(dt < 30))
    r90d = float(np.sum(dt < 90))
    r365d = float(np.sum(dt < 365))
    feats["rate_7_30"] = float(np.log1p(r7d / (r30d / 30 * 7 + 0.1)))
    feats["rate_30_90"] = float(np.log1p(r30d / (r90d / 90 * 30 + 0.1)))
    feats["rate_90_365"] = float(np.log1p(r90d / (r365d / 365 * 90 + 0.1)))

    # ---- NN distance change ----
    r90_mask = dt < 90
    old_mask = dt >= 90
    if np.sum(r90_mask) >= 3 and np.sum(old_mask) >= 3:
        feats["nn_change"] = float(
            np.mean(np.sort(dists[r90_mask])[:3])
            - np.mean(np.sort(dists[old_mask])[:3])
        )
    else:
        feats["nn_change"] = 0.0

    # ---- Clustering ----
    if N >= 20:
        # Deterministic subsample: seed a LOCAL RNG from the query cell so the same
        # (lat, lon, ref_epoch) always yields identical clustering features. Previously
        # this drew from the global unseeded RNG, making st_nn / st_nn_std / frac_clust
        # non-reproducible -- the live scorer and the training pipeline computed
        # DIFFERENT features for the same cell (train/serve skew), and the result could
        # not be cached. Uses a stable integer spatial hash (NOT Python's per-process
        # salted hash()) so train, serve, cache, and parallel workers all agree.
        _seed = (
            (int(round(ev_lat * 1000)) * 73856093)
            ^ (int(round(ev_lon * 1000)) * 19349663)
            ^ (int(ev_time) * 83492791)
        ) & 0xFFFFFFFF
        sample_idx = np.random.RandomState(_seed).choice(N, min(N, 60), replace=False)

        # Both branches independently fixed the same non-determinism (an unseeded global RNG here
        # made st_nn / st_nn_std / frac_clust irreproducible, so the live scorer and the training
        # pipeline computed DIFFERENT features for the same cell). main's fix seeded a single
        # RandomState(42) for every cell; that is reproducible, but every cell with the same
        # neighbour count then draws the SAME 60 indices, which correlates the subsample across
        # cells rather than merely stabilising it. Keying the seed on the cell itself is
        # reproducible AND independent, so it is kept.

        st_dists_list = []
        for i in sample_idx:
            other = np.arange(N) != i
            sd = haversine_vec(la[i], lo[i], la[other], lo[other])
            td = np.abs(t[i] - t[other]) / SEC_PER_DAY + 0.01
            st = np.log10(sd + 0.1) + np.log10(td)
            st_dists_list.append(float(np.min(st)))
        feats["st_nn"] = float(np.mean(st_dists_list))
        feats["st_nn_std"] = float(np.std(st_dists_list))
        feats["frac_clust"] = float(np.mean(np.array(st_dists_list) < 2.0))
    else:
        feats["st_nn"] = 5.0
        feats["st_nn_std"] = 0.0
        feats["frac_clust"] = 0.0

    # Centroid migration
    if np.sum(r90_mask) >= 3 and np.sum(old_mask) >= 3:
        feats["centroid_mig"] = float(haversine_km(
            float(np.mean(la[r90_mask])), float(np.mean(lo[r90_mask])),
            float(np.mean(la[old_mask])), float(np.mean(lo[old_mask])),
        ))
    else:
        feats["centroid_mig"] = 0.0

    # Elongation
    if N >= 10:
        pos = np.column_stack([
            la - np.mean(la),
            (lo - np.mean(lo)) * np.cos(np.radians(ev_lat)),
        ])
        cov = np.cov(pos.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        feats["elong"] = float(evals[0] / (evals[1] + 1e-6)) if evals[1] > 0 else 1.0
    else:
        feats["elong"] = 1.0

    # ---- Max magnitude in windows ----
    for wname, wdays in [("30d", 30), ("90d", 90), ("180d", 180)]:
        wm = m[dt < wdays]
        feats[f"maxmag_{wname}"] = float(np.max(wm)) if len(wm) > 0 else 4.0

    # Mag variance change
    if N >= 20:
        half = N // 2
        feats["mag_var_chg"] = float(np.var(m[half:]) - np.var(m[:half]))
    else:
        feats["mag_var_chg"] = 0.0

    # Mag ranges
    m30 = m[dt < 30]
    feats["mag_range_30d"] = float(np.max(m30) - np.min(m30)) if len(m30) >= 2 else 0.0

    # ---- Coulomb stress proxy ----
    m5_near = (m >= 5.0) & (dt < 180) & (dists < 100) & (dists > 1)
    feats["coulomb"] = (
        float(np.sum(1.0 / (dists[m5_near] + 1.0)))
        if np.sum(m5_near) > 0 else 0.0
    )
    feats["coulomb_moment"] = (
        float(np.sum(mag_to_moment(m[m5_near]) / (dists[m5_near] + 1.0)) / 1e18)
        if np.sum(m5_near) > 0 else 0.0
    )
    feats["coulomb_n"] = float(np.sum(m5_near))

    m5_broad = (m >= 5.0) & (dt < 365) & (dists < 300) & (dists > 1)
    feats["coulomb_broad"] = (
        float(np.sum(
            mag_to_moment(m[m5_broad]) / (dists[m5_broad] + 10.0) ** 2
        ) / 1e15)
        if np.sum(m5_broad) > 0 else 0.0
    )

    # ---- Foreshock detection ----
    r90_events = int(np.sum(r90_mask))
    if r90_events >= 5:
        recent_dt = np.sort(dt[r90_mask])
        n_weeks = max(4, int(np.ceil(90 / 7)))
        week_counts = np.zeros(n_weeks)
        for dd in recent_dt:
            wk = min(int(dd / 7), n_weeks - 1)
            week_counts[wk] += 1
        x = np.arange(n_weeks)
        if np.std(week_counts) > 0:
            feats["inv_omori"] = float(-np.corrcoef(x, week_counts)[0, 1])
        else:
            feats["inv_omori"] = 0.0
    else:
        feats["inv_omori"] = 0.0

    # Bath's law
    if len(m30) >= 2:
        sm = np.sort(m30)[::-1]
        feats["bath_ratio"] = float(sm[1] / (sm[0] + 0.01))
    else:
        feats["bath_ratio"] = 0.0

    # Foreshock ratio
    if N >= 10:
        check = min(N, 150)
        fcount = 0
        for i in range(max(0, N - check), N):
            if np.any((t > t[i]) & (t < t[i] + 7 * SEC_PER_DAY) & (m > m[i])):
                fcount += 1
        feats["foreshock_r"] = fcount / check
    else:
        feats["foreshock_r"] = 0.0

    # Quiescence detection
    expected_7d = r365d / 365 * 7
    feats["quiescence_7d"] = float(np.log1p(expected_7d / (r7d + 0.1)))

    # ---- Moment release ----
    moments = mag_to_moment(m)
    for wname, wdays in [("90d", 90), ("180d", 180), ("1y", 365)]:
        wm = moments[dt < wdays]
        feats[f"mom_{wname}"] = float(
            np.log10(np.sum(wm) + 1e10)
        ) if len(wm) > 0 else 10.0

    rec_mom = float(np.sum(moments[dt < 180]))
    old_mom = float(np.sum(moments[(dt >= 180) & (dt < 365)]))
    feats["mom_accel"] = float(np.log1p(rec_mom / (old_mom + 1e10)))

    yrs = max(0.5, (np.max(t) - np.min(t)) / SEC_PER_YEAR)
    annual = float(np.sum(moments)) / yrs
    feats["mom_deficit"] = float(
        np.log10(np.sum(moments[dt < 365]) / (annual + 1e10) + 0.01)
    )

    # ---- Depth features ----
    feats["depth_mean"] = float(np.mean(d))
    feats["depth_std"] = float(np.std(d)) if N >= 5 else 0.0
    if N >= 20:
        half = N // 2
        feats["depth_trend"] = float(np.mean(d[half:]) - np.mean(d[:half]))
    else:
        feats["depth_trend"] = 0.0

    if N >= 20:
        dh, _ = np.histogram(d, bins=10)
        feats["depth_bimod"] = float(np.std(dh) / (np.mean(dh) + 0.1))
    else:
        feats["depth_bimod"] = 0.0

    # ---- Tectonic regime ----
    deep = (d > 70) & (dists < 200)
    feats["subduc"] = float(np.sum(deep)) / max(1, N)

    # ---- Event counts ----
    feats["n_events"] = float(N)
    feats["n_7d"] = float(np.sum(dt < 7))
    feats["n_14d"] = float(np.sum(dt < 14))
    feats["n_30d"] = float(np.sum(dt < 30))
    feats["n_90d"] = float(np.sum(dt < 90))

    # ---- M5+ acceleration ----
    m5_90 = float(np.sum((m >= 5.0) & (dt < 90)))
    m5_365 = float(np.sum((m >= 5.0) & (dt < 365)))
    feats["m5_accel"] = float(np.log1p(m5_90 / (m5_365 / 365 * 90 + 0.1)))

    # Spatial concentration
    if np.sum(r90_mask) >= 3:
        feats["spatial_conc"] = float(np.std(dists[r90_mask]))
    else:
        feats["spatial_conc"] = 100.0

    # Mag range 90d
    m90 = m[dt < 90]
    feats["mag_range_90d"] = float(np.max(m90) - np.min(m90)) if len(m90) >= 2 else 0.0

    # ---- M6+ recurrence ----
    if len(cat.m6_times) > 0:
        hd = haversine_vec(ev_lat, ev_lon, cat.m6_lats, cat.m6_lons)
        nearby_m6 = cat.m6_times[(hd < 300) & (cat.m6_times < ev_time)]
        if len(nearby_m6) >= 2:
            intervals = np.diff(np.sort(nearby_m6)) / SEC_PER_YEAR
            feats["m6_recur"] = float(np.mean(intervals))
        else:
            feats["m6_recur"] = 10.0
    else:
        feats["m6_recur"] = 10.0

    # ---- v8c interactions (4 non-redundant) ----
    feats["b_x_rate"] = feats["b_trend"] * feats["rate_3m"]
    feats["coul_x_rate"] = feats["coulomb"] * feats["rate_1m"]
    feats["mom_x_clust"] = feats["mom_accel"] * feats["frac_clust"]
    feats["fore_x_bath"] = feats["foreshock_r"] * feats["bath_ratio"]

    # ---- v8c near-field (6) ----
    near_90 = near & (dt < 90)
    feats["near_n_90d"] = float(np.sum(near_90))
    feats["near_mom_90d"] = float(
        np.log10(np.sum(mag_to_moment(m[near_90])) + 1e10)
    ) if np.sum(near_90) > 0 else 10.0

    # Near-field interevent time CV
    near_dt = np.sort(dt[near])
    if len(near_dt) >= 5:
        iet = np.diff(near_dt)
        feats["near_iet_cv"] = float(np.std(iet) / (np.mean(iet) + 0.01))
        feats["near_iet_min"] = float(np.min(iet))
    else:
        feats["near_iet_cv"] = 1.0
        feats["near_iet_min"] = 100.0

    # Near/far ratio
    far = (dists >= 250) & (dists < 500)
    n_near_90 = float(np.sum(near & (dt < 90)))
    n_far_90 = float(np.sum(far & (dt < 90)))
    feats["near_far_ratio"] = float(np.log1p(n_near_90 / (n_far_90 + 0.1)))

    # Near-field rate x b
    feats["near_rate_x_b"] = feats["near_n_90d"] * feats["b_trend"]

    # ---- Assemble into ordered array matching BLOCK_S_NAMES ----
    result = np.zeros(N_FEAT_S, dtype=np.float32)
    for i, name in enumerate(BLOCK_S_NAMES):
        result[i] = feats.get(name, 0.0)

    return result


# ===================================================================
# BLOCK C: COHERENCE FIELD THEORY FEATURES (12 features)
# ===================================================================

def compute_block_c(
    full_catalog: list[dict],
    lat: float,
    lon: float,
    ref_epoch: float,
) -> np.ndarray:
    """Extract Block C (coherence field theory) features.

    Uses extract_coherence_features from coherence_engine.
    All computations use ONLY events BEFORE ref_epoch.

    Returns
    -------
    ndarray, shape (12,), dtype float32
    """
    feats = np.full(N_FEAT_C, np.nan, dtype=np.float64)

    past_catalog = [e for e in full_catalog if _event_epoch(e) < ref_epoch]

    cft = extract_coherence_features(
        past_catalog,
        lat=lat,
        lon=lon,
        radius_km=LABEL_RADIUS_KM,
        time_window_days=FORWARD_WINDOW_DAYS,
        ref_epoch=ref_epoch,
    )

    feats[0] = cft.get("ell", np.nan)
    feats[1] = cft.get("ell_trend", np.nan)
    feats[2] = cft.get("ell_acceleration", np.nan)
    feats[3] = cft.get("days_to_criticality", np.nan)
    feats[4] = cft.get("nu_exponent", np.nan)
    feats[5] = cft.get("t_c_confidence", np.nan)
    feats[6] = cft.get("delta_aic_iet", np.nan)
    feats[7] = cft.get("tau_local", np.nan)
    feats[8] = cft.get("grad_tau_local", np.nan)
    feats[9] = cft.get("S_over_Gamma", np.nan)
    feats[10] = cft.get("Da_local", np.nan)

    sing = test_earthquake_singularity(cft)
    feats[11] = float(sing.conditions_met)

    return feats.astype(np.float32)


# ===================================================================
# BLOCK X: CROSS-DOMAIN INTERACTION FEATURES (8 features)
# ===================================================================

def compute_block_x(
    block_s: np.ndarray,
    block_c: np.ndarray,
) -> np.ndarray:
    """Extract Block X (cross-domain interaction) features.

    8 physically motivated cross-products between seismicity and CFT.

    Returns
    -------
    ndarray, shape (8,), dtype float32
    """
    feats = np.zeros(N_FEAT_X, dtype=np.float64)

    def _safe(val: float) -> float:
        return 0.0 if (math.isnan(val) or math.isinf(val)) else val

    # Block S feature indices (from BLOCK_S_NAMES order):
    # 0=b_trend, 6=rate_1m, 10=rate_7_30, 15=frac_clust,
    # 20=maxmag_90d, 24=coulomb, 33=mom_accel
    s_b_trend = _safe(float(block_s[0]))
    s_rate_1m = _safe(float(block_s[6]))
    s_rate_7_30 = _safe(float(block_s[10]))
    s_frac_clust = _safe(float(block_s[15]))
    s_maxmag_90d = _safe(float(block_s[20]))
    s_coulomb = _safe(float(block_s[24]))
    s_mom_accel = _safe(float(block_s[33]))

    # Block C feature indices (from BLOCK_C_NAMES order):
    # 0=ell, 1=ell_trend, 2=ell_acceleration, 3=days_to_criticality,
    # 6=delta_aic_iet, 7=tau_local, 9=S_over_Gamma, 11=singularity_count
    c_ell = _safe(float(block_c[0]))
    c_ell_accel = _safe(float(block_c[2]))
    c_dtc = _safe(float(block_c[3]))
    c_daic = _safe(float(block_c[6]))
    c_tau = _safe(float(block_c[7]))
    c_sg = _safe(float(block_c[9]))
    c_sing = _safe(float(block_c[11]))

    feats[0] = c_ell * s_b_trend              # ell_x_b_trend
    feats[1] = c_ell * s_rate_1m              # ell_x_rate_1m
    feats[2] = c_ell * s_coulomb              # ell_x_coulomb
    feats[3] = c_tau * s_maxmag_90d           # tau_x_maxmag_90d
    feats[4] = c_dtc * c_sing                 # dtc_x_singularity
    feats[5] = c_sg * s_mom_accel             # sg_x_mom_accel
    feats[6] = c_daic * s_frac_clust          # daic_x_frac_clust
    feats[7] = c_ell_accel * s_rate_7_30      # ell_accel_x_rate_7_30

    return feats.astype(np.float32)


# ===================================================================
# PARALLEL FEATURE EXTRACTION
# ===================================================================
# Per-sample feature extraction is independent and CPU-bound (catalog scans), so it
# fans out cleanly across cores. Each worker rebuilds the catalog + CatalogArrays
# from the local CSV cache in its initializer (no pickling of the 0.5M-event catalog),
# then maps the SAME compute_block_* functions the serial path uses -> byte-identical
# matrices, just ~Ncores faster. The serial path stays the reference.

_PAR_CAT = None        # per-worker CatalogArrays (read-only)
_PAR_CATALOG = None    # per-worker full catalog list


def _par_worker_init(max_year=2024):
    global _PAR_CAT, _PAR_CATALOG
    _PAR_CATALOG = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)
    _PAR_CAT = CatalogArrays(_PAR_CATALOG, verbose=False)


def _par_worker_extract(sample):
    """One sample -> (X_s, X_c, X_x, label, skipped). Mirrors the serial loop exactly."""
    lat = sample["latitude"]; lon = sample["longitude"]; ref_epoch = sample["ref_epoch"]
    s = compute_block_s(lat, lon, ref_epoch, _PAR_CAT)
    skipped = 1 if s is None else 0
    X_s = np.zeros(N_FEAT_S, np.float32) if s is None else np.asarray(s, np.float32)
    X_c = np.nan_to_num(compute_block_c(_PAR_CATALOG, lat, lon, ref_epoch), nan=0.0).astype(np.float32)
    X_x = np.nan_to_num(compute_block_x(X_s, X_c), nan=0.0).astype(np.float32)
    return X_s, X_c, X_x, np.float32(sample["label"]), skipped


def extract_features_parallel(split_samples, split_name="", n_workers=None, verbose=True,
                              max_year=2024):
    """Process-pool version of the per-split feature extraction.

    Returns ``(X_dict, y)`` with X_dict = {baseline, enhanced, full}, identical to the
    serial closure inside ``load_all_data``. ``max_year`` is passed to each worker so
    its catalog matches the parent's.
    """
    from concurrent.futures import ProcessPoolExecutor

    n = len(split_samples)
    X_s = np.zeros((n, N_FEAT_S), dtype=np.float32)
    X_c = np.zeros((n, N_FEAT_C), dtype=np.float32)
    X_x = np.zeros((n, N_FEAT_X), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    if n == 0:
        return ({"baseline": X_s.copy(),
                 "enhanced": np.hstack([X_s, X_c]),
                 "full": np.hstack([X_s, X_c, X_x])}, y)

    workers = int(n_workers) if n_workers else max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    n_skipped = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_par_worker_init,
                             initargs=(max_year,)) as ex:
        for idx, (s, c, x, lab, sk) in enumerate(
            ex.map(_par_worker_extract, split_samples, chunksize=8)
        ):
            X_s[idx] = s; X_c[idx] = c; X_x[idx] = x; y[idx] = lab; n_skipped += sk
            if verbose and (idx + 1) % 200 == 0:
                rate = (idx + 1) / max(time.time() - t0, 1e-9)
                eta = (n - idx - 1) / rate / 60.0
                print(f"      {split_name} (parallel x{workers}): {idx + 1}/{n} "
                      f"({100 * (idx + 1) / n:.0f}%, {rate:.0f}/s, ETA {eta:.1f}min)")
                sys.stdout.flush()
    if verbose:
        print(f"      {split_name}: {n}/{n} (100%) -- {time.time() - t0:.1f}s, "
              f"{n_skipped} skipped [{workers} workers]")
        sys.stdout.flush()
    return ({"baseline": X_s.copy(),
             "enhanced": np.hstack([X_s, X_c]),
             "full": np.hstack([X_s, X_c, X_x])}, y)


# ===================================================================
# DATA LOADING AND SAMPLE PIPELINE
# ===================================================================

def load_all_data(
    verbose: bool = True,
    parallel: bool = False,
    n_workers: int | None = None,
    max_year: int = 2024,
) -> tuple[
    dict[str, np.ndarray],  # X_train
    dict[str, np.ndarray],  # X_val
    dict[str, np.ndarray],  # X_test
    np.ndarray,             # y_train
    np.ndarray,             # y_val
    np.ndarray,             # y_test
    dict,                   # meta
]:
    """Load earthquake catalog, build samples, extract features, split.

    Steps:
    1. Load USGS catalog (2000-2024, M2.5+)
    2. Gardner-Knopoff aftershock declustering
    3. Identify M6+ mainshocks
    4. Generate same-location controls (2:1 neg ratio)
    5. Temporal split with assertions
    6. Pre-build numpy arrays for vectorized features
    7. Extract features for each sample
    """
    # Step 1: Load USGS catalog
    if verbose:
        print("    Loading USGS catalog (2000-2024, M2.5+)...")
        sys.stdout.flush()

    full_catalog = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)

    if verbose:
        print(f"    Loaded {len(full_catalog)} events")
        sys.stdout.flush()

    if len(full_catalog) == 0:
        raise RuntimeError(
            "No earthquake data found. Run scripts/download_earthquake_data.py first."
        )

    # Step 2: Gardner-Knopoff aftershock declustering
    if verbose:
        print("    Aftershock declustering (Gardner-Knopoff 1974)...")
        sys.stdout.flush()

    mainshocks, aftershocks = decluster_gardner_knopoff(full_catalog)

    if verbose:
        print(
            f"    Declustered: {len(mainshocks)} mainshocks, "
            f"{len(aftershocks)} aftershocks removed"
        )
        sys.stdout.flush()

    # Step 3-4: Build positive + negative samples
    if verbose:
        print("    Building samples...")
        sys.stdout.flush()

    samples = build_samples(mainshocks, full_catalog, verbose=verbose)

    # Step 5: Temporal split. The test window extends to max_year so newly captured
    # years (e.g. 2025+) flow into evaluation -- and, for deployment, get folded in.
    test_end = max(TEST_END, int(max_year))
    train_samples = [s for s in samples if TRAIN_START <= s["year"] <= TRAIN_END]
    val_samples = [s for s in samples if VAL_START <= s["year"] <= VAL_END]
    test_samples = [s for s in samples if TEST_START <= s["year"] <= test_end]

    # ASSERT temporal integrity
    for s in train_samples:
        assert s["year"] <= TRAIN_END, (
            f"LEAKAGE: Train sample year {s['year']} > TRAIN_END {TRAIN_END}"
        )
    for s in val_samples:
        assert VAL_START <= s["year"] <= VAL_END, (
            f"LEAKAGE: Val sample year {s['year']} outside [{VAL_START}, {VAL_END}]"
        )
    for s in test_samples:
        assert TEST_START <= s["year"] <= test_end, (
            f"LEAKAGE: Test sample year {s['year']} outside [{TEST_START}, {test_end}]"
        )

    # Cross-boundary assertion
    if train_samples and val_samples:
        max_train_year = max(s["year"] for s in train_samples)
        min_val_year = min(s["year"] for s in val_samples)
        assert max_train_year < min_val_year, (
            f"LEAKAGE: max train year {max_train_year} >= min val year {min_val_year}"
        )
    if val_samples and test_samples:
        max_val_year = max(s["year"] for s in val_samples)
        min_test_year = min(s["year"] for s in test_samples)
        assert max_val_year < min_test_year, (
            f"LEAKAGE: max val year {max_val_year} >= min test year {min_test_year}"
        )

    if verbose:
        print(f"\n    Temporal split:")
        print(f"      Train ({TRAIN_START}-{TRAIN_END}): {len(train_samples)} samples")
        print(f"      Val   ({VAL_START}-{VAL_END}):   {len(val_samples)} samples")
        print(f"      Test  ({TEST_START}-{TEST_END}):  {len(test_samples)} samples")
        sys.stdout.flush()

    # Step 6: Pre-build numpy arrays for FAST vectorized features
    if verbose:
        print("\n    Pre-building catalog arrays for vectorized features...")
        sys.stdout.flush()

    cat = CatalogArrays(full_catalog, verbose=verbose)

    # Step 7: Extract features
    if verbose:
        print("\n    Extracting features (numpy-vectorized)...")
        sys.stdout.flush()

    def _extract_features_for_split(
        split_samples: list[dict],
        split_name: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Extract all feature blocks for a list of samples."""
        n = len(split_samples)
        X_s = np.zeros((n, N_FEAT_S), dtype=np.float32)
        X_c = np.zeros((n, N_FEAT_C), dtype=np.float32)
        X_x = np.zeros((n, N_FEAT_X), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        t0 = time.time()
        n_skipped = 0

        for idx, sample in enumerate(split_samples):
            if verbose and (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"      {split_name}: {idx + 1}/{n} "
                    f"({100 * (idx + 1) / n:.0f}%, "
                    f"{rate:.0f} samples/min)",
                )
                sys.stdout.flush()

            lat = sample["latitude"]
            lon = sample["longitude"]
            ref_epoch = sample["ref_epoch"]

            # Block S: fast numpy-vectorized (v8c approach)
            s_feats = compute_block_s(lat, lon, ref_epoch, cat)
            if s_feats is None:
                n_skipped += 1
                # Leave as zeros -- will be imputed later
            else:
                X_s[idx] = s_feats

            # Block C: coherence engine
            c_feats = compute_block_c(full_catalog, lat, lon, ref_epoch)
            c_feats = np.nan_to_num(c_feats, nan=0.0)
            X_c[idx] = c_feats

            # Block X: cross-domain interactions
            x_feats = compute_block_x(X_s[idx], X_c[idx])
            x_feats = np.nan_to_num(x_feats, nan=0.0)
            X_x[idx] = x_feats

            y[idx] = sample["label"]

        if verbose:
            elapsed = time.time() - t0
            print(
                f"      {split_name}: {n}/{n} (100%) -- "
                f"{elapsed:.1f}s, {n_skipped} skipped"
            )
            sys.stdout.flush()

        # Assemble feature matrices for each variant
        X_dict = {
            "baseline": X_s.copy(),
            "enhanced": np.hstack([X_s, X_c]),
            "full": np.hstack([X_s, X_c, X_x]),
        }
        return X_dict, y

    if parallel:
        if verbose:
            print(f"    Parallel extraction across "
                  f"{n_workers or max(1, (os.cpu_count() or 2) - 2)} workers...")
            sys.stdout.flush()
        X_train, y_train = extract_features_parallel(train_samples, "train", n_workers, verbose, max_year)
        X_val, y_val = extract_features_parallel(val_samples, "val", n_workers, verbose, max_year)
        X_test, y_test = extract_features_parallel(test_samples, "test", n_workers, verbose, max_year)
    else:
        X_train, y_train = _extract_features_for_split(train_samples, "train")
        X_val, y_val = _extract_features_for_split(val_samples, "val")
        X_test, y_test = _extract_features_for_split(test_samples, "test")

    # Impute NaN with training mean (per feature)
    for variant in ["baseline", "enhanced", "full"]:
        train_mean = np.nanmean(X_train[variant], axis=0)
        train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)

        for X in [X_train[variant], X_val[variant], X_test[variant]]:
            nan_mask = np.isnan(X)
            for col in range(X.shape[1]):
                col_nans = nan_mask[:, col]
                if col_nans.any():
                    X[col_nans, col] = train_mean[col]

    # 5:1 downsample if needed
    n_pos_train = int(y_train.sum())
    n_neg_train = int(len(y_train) - n_pos_train)
    if n_neg_train > 5 * n_pos_train:
        if verbose:
            print(
                f"\n    Downsampling: {n_neg_train} neg >> {5 * n_pos_train} target"
            )
            sys.stdout.flush()
        rng = np.random.RandomState(42)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        keep_neg = rng.choice(neg_idx, size=5 * n_pos_train, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx, keep_neg]))

        y_train = y_train[keep_idx]
        for variant in ["baseline", "enhanced", "full"]:
            X_train[variant] = X_train[variant][keep_idx]

        if verbose:
            print(
                f"    After downsample: {len(y_train)} samples "
                f"({int(y_train.sum())} pos, "
                f"{int(len(y_train) - y_train.sum())} neg)"
            )

    meta = {
        "n_catalog_events": len(full_catalog),
        "n_mainshocks": len(mainshocks),
        "n_aftershocks_removed": len(aftershocks),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "train_pos": int(y_train.sum()),
        "train_neg": int(len(y_train) - y_train.sum()),
        "val_pos": int(y_val.sum()),
        "val_neg": int(len(y_val) - y_val.sum()),
        "test_pos": int(y_test.sum()),
        "test_neg": int(len(y_test) - y_test.sum()),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, meta


# ===================================================================
# TEMPORAL INTEGRITY ASSERTION
# ===================================================================

def assert_temporal_integrity(
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
) -> None:
    """Assert that no sample crosses temporal boundaries."""
    for y in train_years:
        assert y <= TRAIN_END, (
            f"LEAKAGE: Train year {y} > TRAIN_END {TRAIN_END}"
        )
    for y in val_years:
        assert y >= VAL_START, f"LEAKAGE: Val year {y} < VAL_START {VAL_START}"
        assert y <= VAL_END, f"LEAKAGE: Val year {y} > VAL_END {VAL_END}"
    for y in test_years:
        assert y >= TEST_START, f"LEAKAGE: Test year {y} < TEST_START {TEST_START}"
        assert y <= TEST_END, f"LEAKAGE: Test year {y} > TEST_END {TEST_END}"


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main(
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the complete definitive earthquake prediction pipeline v2.

    Steps:
        1. Load USGS catalog
        2. Gardner-Knopoff aftershock declustering
        3. Identify M6+ mainshocks
        4. Generate same-location controls (2:1 neg ratio)
        5. Pre-build numpy arrays for fast vectorized features
        6. For each sample: compute Block S (60 v8c features, numpy)
        7. For each sample: compute Block C (12 CFT features)
        8. For each sample: compute Block X (8 cross-domain interactions)
        9. Assert temporal split integrity
        10. Normalize features
        11. Train 3 GBT models
        12. Evaluate on test ONCE
        13. Significance tests
        14. Save results

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for results JSON.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Complete results with all metrics and audit guarantees.
    """
    t_start = time.time()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[3]
    if output_dir is None:
        output_dir = project_root / "results" / "earthquake_definitive"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 72)
        print("DEFINITIVE EARTHQUAKE PREDICTION MODEL v2 (MEGA-MODEL)")
        print("RESEARCH ONLY -- NOT OPERATIONAL")
        print("=" * 72)
        print()
        print(f"Feature architecture:")
        print(f"  Block S: {N_FEAT_S} seismicity features (full v8c, numpy-vectorized)")
        print(f"  Block C: {N_FEAT_C} coherence field theory features")
        print(f"  Block X: {N_FEAT_X} cross-domain interaction features")
        print(f"  Total:   {N_FEAT_FULL} features (full model)")
        print()
        print("Temporal splits:")
        print(f"  Train: {TRAIN_START} -- {TRAIN_END}")
        print(f"  Val:   {VAL_START} -- {VAL_END}")
        print(f"  Test:  {TEST_START} -- {TEST_END}")
        print()
        print(f"GBT config: {GBT_N_TREES} trees, depth={GBT_MAX_DEPTH}, "
              f"lr={GBT_LEARNING_RATE}, subsample={GBT_SUBSAMPLE}")
        print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
        print(f"Label: M{MIN_MAINSHOCK_MAG}+ within {LABEL_RADIUS_KM} km, "
              f"{FORWARD_WINDOW_DAYS:.0f} days forward")
        print()
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Steps 1-7: Load data, decluster, build samples, extract features
    # ---------------------------------------------------------------
    if verbose:
        print("[1-7] Loading data, building samples, extracting features...")
        sys.stdout.flush()

    X_train, X_val, X_test, y_train, y_val, y_test, meta = load_all_data(
        verbose=verbose,
    )

    assert len(y_train) > 0, "No training samples loaded"
    assert len(y_val) > 0, "No validation samples loaded"
    assert len(y_test) > 0, "No test samples loaded"

    # ---------------------------------------------------------------
    # Step 10: Normalize features (fit on train, apply to val/test)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[10] Normalizing features...")
        sys.stdout.flush()

    normalizers: dict[str, FeatureNormalizer] = {}
    for variant in ["baseline", "enhanced", "full"]:
        norm = FeatureNormalizer()
        X_train[variant] = norm.fit_transform(X_train[variant])
        X_val[variant] = norm.transform(X_val[variant])
        X_test[variant] = norm.transform(X_test[variant])
        normalizers[variant] = norm

    # ---------------------------------------------------------------
    # Step 11: Train 3 GBT models on train (with early stopping on val)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[11] Training GBT models...")
        sys.stdout.flush()

    models: dict[str, GradientBoostedTrees] = {}
    train_meta: dict[str, dict] = {}

    for variant in ["baseline", "enhanced", "full"]:
        if verbose:
            n_feat = X_train[variant].shape[1]
            feature_names = {
                "baseline": ALL_FEATURE_NAMES_BASELINE,
                "enhanced": ALL_FEATURE_NAMES_ENHANCED,
                "full": ALL_FEATURE_NAMES_FULL,
            }[variant]
            print(f"\n  --- {variant.upper()} ({n_feat} features) ---")
            sys.stdout.flush()

        gbt = GradientBoostedTrees(
            n_trees=GBT_N_TREES,
            max_depth=GBT_MAX_DEPTH,
            learning_rate=GBT_LEARNING_RATE,
            min_samples_leaf=GBT_MIN_SAMPLES_LEAF,
            subsample=GBT_SUBSAMPLE,
            colsample=GBT_COLSAMPLE,
            l2_reg=GBT_L2_REG,
            gamma=GBT_GAMMA,
        )

        tmeta = gbt.fit(
            X_train[variant], y_train,
            X_val=X_val[variant], y_val=y_val,
            verbose=verbose,
        )

        models[variant] = gbt
        train_meta[variant] = tmeta

        if verbose:
            print(f"    Trees used: {tmeta['n_trees_used']}, "
                  f"early_stop: {tmeta['stopped_early']}")
            sys.stdout.flush()

    # ---------------------------------------------------------------
    # Validation sanity check
    # ---------------------------------------------------------------
    if verbose:
        print("\n[VAL] Validation sanity check...")
        sys.stdout.flush()

    val_preds: dict[str, np.ndarray] = {}
    for variant in ["baseline", "enhanced", "full"]:
        p_val = models[variant].predict_proba(X_val[variant])
        val_preds[variant] = p_val
        val_auc = compute_auc(y_val, p_val)
        if verbose:
            print(f"  {variant}: val_auc = {val_auc:.4f}")

        if val_auc < 0.55:
            print(
                f"  WARNING: {variant} val AUC ({val_auc:.4f}) is very low. "
                f"Model may be broken."
            )

    # ---------------------------------------------------------------
    # Step 12: Evaluate on test ONCE
    # ---------------------------------------------------------------
    if verbose:
        print("\n[12] TEST EVALUATION (single pass, no going back)...")
        sys.stdout.flush()

    test_preds: dict[str, np.ndarray] = {}
    test_results: dict[str, dict] = {}

    for variant in ["baseline", "enhanced", "full"]:
        p_test = models[variant].predict_proba(X_test[variant])
        test_preds[variant] = p_test
        test_results[variant] = evaluate(y_test, p_test)

        if verbose:
            r = test_results[variant]
            print(f"\n  {variant.upper()} on TEST:")
            print(f"    AUC:    {r['auc']:.4f} "
                  f"[{r['bootstrap_ci']['ci_lo']:.4f}, "
                  f"{r['bootstrap_ci']['ci_hi']:.4f}]")
            print(f"    PR-AUC: {r['pr_auc']:.4f}")
            print(f"    Brier:  {r['brier']:.6f}")
            print(f"    BSS:    {r['bss']:.4f}")
            print(f"    N_pos:  {r['n_positive']}, N_neg: {r['n_negative']}, "
                  f"base_rate: {r['base_rate']:.6f}")
            sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 13: Significance tests
    # ---------------------------------------------------------------
    if verbose:
        print("\n[13] Paired bootstrap significance tests...")
        sys.stdout.flush()

    cft_lift = paired_bootstrap_test(
        y_test, test_preds["baseline"], test_preds["enhanced"]
    )
    interaction_lift = paired_bootstrap_test(
        y_test, test_preds["enhanced"], test_preds["full"]
    )

    if verbose:
        print(f"\n  CFT lift (enhanced vs baseline):")
        print(f"    delta_AUC = {cft_lift['delta_auc']:.4f} "
              f"[{cft_lift['ci_lo']:.4f}, {cft_lift['ci_hi']:.4f}] "
              f"p = {cft_lift['p_value']:.4f}")
        print(f"\n  Interaction lift (full vs enhanced):")
        print(f"    delta_AUC = {interaction_lift['delta_auc']:.4f} "
              f"[{interaction_lift['ci_lo']:.4f}, {interaction_lift['ci_hi']:.4f}] "
              f"p = {interaction_lift['p_value']:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Feature importance for full model
    # ---------------------------------------------------------------
    full_importance = models["full"].feature_importances(N_FEAT_FULL)
    importance_ranked = sorted(
        zip(ALL_FEATURE_NAMES_FULL, full_importance.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    if verbose:
        print(f"\n  Top 20 features (full model, split count importance):")
        for name, imp in importance_ranked[:20]:
            print(f"    {name:30s}  {imp:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 14: Assemble and save results
    # ---------------------------------------------------------------
    elapsed = time.time() - t_start

    results = {
        "model": "hazardpulse_earthquake_definitive_v2",
        "disclaimer": "RESEARCH ONLY. NOT operational. NOT a replacement for USGS.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "feature_architecture": {
            "block_s": f"{N_FEAT_S} seismicity features (full v8c port)",
            "block_c": f"{N_FEAT_C} coherence field theory features",
            "block_x": f"{N_FEAT_X} cross-domain interaction features",
            "baseline_total": N_FEAT_BASELINE,
            "enhanced_total": N_FEAT_ENHANCED,
            "full_total": N_FEAT_FULL,
        },
        "audit_guarantees": {
            "temporal_split": (
                f"Train {TRAIN_START}-{TRAIN_END}, "
                f"Val {VAL_START}-{VAL_END}, "
                f"Test {TEST_START}-{TEST_END}"
            ),
            "label_definition": (
                f"M{MIN_MAINSHOCK_MAG}+ within {LABEL_RADIUS_KM}km "
                f"and {FORWARD_WINDOW_DAYS:.0f} days FORWARD"
            ),
            "aftershock_declustering": "Gardner-Knopoff (1974)",
            "control_generation": (
                f"Same-location, {CONTROL_RATIO}:1 neg ratio, "
                f"time-offset {CONTROL_OFFSET_RANGE[0]}-{CONTROL_OFFSET_RANGE[1]} years"
            ),
            "meta_stacker": False,
            "hyperparameter_tuning_on_test": False,
            "single_test_evaluation": True,
            "feature_extraction": "numpy-vectorized (v8c approach)",
            "model_type": "GradientBoostedTrees (single per variant, no ensemble)",
            "gbt_config": {
                "n_trees": GBT_N_TREES,
                "max_depth": GBT_MAX_DEPTH,
                "learning_rate": GBT_LEARNING_RATE,
                "subsample": GBT_SUBSAMPLE,
                "colsample": GBT_COLSAMPLE,
                "min_samples_leaf": GBT_MIN_SAMPLES_LEAF,
                "l2_reg": GBT_L2_REG,
                "gamma": GBT_GAMMA,
                "early_stop_patience": EARLY_STOP_PATIENCE,
            },
        },
        "data_summary": meta,
        "training_metadata": {
            variant: {
                "n_trees_used": train_meta[variant]["n_trees_used"],
                "stopped_early": train_meta[variant]["stopped_early"],
                "best_val_loss": train_meta[variant].get("best_val_loss"),
            }
            for variant in ["baseline", "enhanced", "full"]
        },
        "validation_sanity_check": {
            variant: {
                "auc": compute_auc(y_val, val_preds[variant]),
            }
            for variant in ["baseline", "enhanced", "full"]
        },
        "baseline": test_results["baseline"],
        "enhanced": test_results["enhanced"],
        "full": test_results["full"],
        "theory_tests": {
            "cft_lift": cft_lift,
            "interaction_lift": interaction_lift,
        },
        "feature_importance": {
            "full_model": [
                {"feature": name, "importance": imp}
                for name, imp in importance_ranked
            ],
            "note": (
                "Importance is split frequency, NOT causal attribution. "
                "A feature that splits often may simply have high variance."
            ),
        },
    }

    # Save to JSON
    results_path = output_dir / "definitive_results_v2.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved to: {results_path}")
        print(f"  Total time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
        print("\n" + "=" * 72)
        print("DONE. Remember: this is RESEARCH ONLY.")
        print("=" * 72)
        sys.stdout.flush()

    return results


# ===================================================================
# CLI ENTRYPOINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Definitive leak-proof earthquake prediction model v2 "
            "(RESEARCH ONLY)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results JSON (default: results/earthquake_definitive/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
