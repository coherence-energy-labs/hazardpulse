# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational earthquake prediction system.
# It does NOT replace official USGS or national seismological agency warnings.
# Always follow guidance from your national geological survey.
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""V4 Regional Earthquake Prediction Model.

Design philosophy: REGIONAL MODELS OUTPERFORM GLOBAL MODELS.

Different tectonic settings have fundamentally different precursor physics:
  - Subduction zones: slow-slip precursors, depth-stratified seismicity,
    GNSS transients, deep-to-shallow migration
  - Transform faults: strike-slip clustering, shallow events, geodetic
    strain accumulation, Coulomb stress transfer
  - Continental collision: diffuse seismicity, complex geometry, crustal
    thickening signals

Architecture:
    Phase 1: Load ALL data (USGS catalog, GNSS, faults)
    Phase 2: Define 8 tectonic regions
    Phase 3: Extract 55 features per sample (Blocks S+G+C+T)
    Phase 4: Train separate GBT per region
    Phase 5: Ensemble: route prediction to correct regional model
    Phase 6: Evaluate per-region + global combined vs single global

Feature blocks:
    Block S: 30 seismicity features (tuned per tectonic type)
    Block G: 12 geodetic/GNSS features
    Block C:  8 coherence field theory features
    Block T:  5 tectonic context features
    Total:   55 features

Temporal splits (hardcoded, non-negotiable):
    Train: 2005-01-01 to 2017-12-31
    Val:   2018-01-01 to 2019-12-31
    Test:  2020-01-01 to 2024-12-31

Label definition:
    Positive: M6+ (or M5.5+ in sparse regions) mainshock within
              region-specific radius and 365 days FORWARD
    Negative: same location, time-offset 1.5-4.5 years, no target
              event within window
    Gardner-Knopoff aftershock declustering applied first.
    2 negatives per positive.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
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
    test_earthquake_singularity,
    _parse_event_time,
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

# Label parameters (defaults; overridden per region)
FORWARD_WINDOW_DAYS: float = 365.0
CONTROL_RATIO: int = 2
CONTROL_OFFSET_RANGE: tuple[float, float] = (1.5, 4.5)

# Honest mode parameters (comparable to CSEP/ETAS published systems)
HONEST_MAGNITUDE_THRESHOLD: float = 6.0   # M6.0+ only, no M5.5 fallback
HONEST_LABEL_RADIUS_KM: float = 100.0     # 100km, not 200-300km
HONEST_FORWARD_WINDOW_DAYS: float = 90.0  # 90 days, not 365
HONEST_CONTROL_RATIO: int = 5             # 5:1 neg ratio (~17% base rate)

# Global flag set by CLI --honest
_HONEST_MODE: bool = False

# Time constants
SEC_PER_DAY: float = 86400.0
SEC_PER_YEAR: float = 365.25 * SEC_PER_DAY


# ===================================================================
# REGION DEFINITIONS
# ===================================================================

REGIONS: dict[str, dict] = {
    "japan": {
        "lat": (30, 46), "lon": (128, 146),
        "type": "subduction", "label_radius_km": 200,
    },
    "cascadia": {
        "lat": (40, 52), "lon": (-130, -120),
        "type": "subduction", "label_radius_km": 250,
    },
    "san_andreas": {
        "lat": (32, 40), "lon": (-123, -114),
        "type": "transform", "label_radius_km": 150,
    },
    "alaska": {
        "lat": (55, 65), "lon": (-170, -140),
        "type": "subduction", "label_radius_km": 300,
    },
    "chile": {
        "lat": (-45, -15), "lon": (-76, -66),
        "type": "subduction", "label_radius_km": 250,
    },
    "mediterranean": {
        "lat": (34, 42), "lon": (-5, 30),
        "type": "collision", "label_radius_km": 200,
    },
    "indonesia": {
        "lat": (-10, 5), "lon": (95, 130),
        "type": "subduction", "label_radius_km": 300,
    },
    "rest_of_world": {
        "lat": (-90, 90), "lon": (-180, 180),
        "type": "mixed", "label_radius_km": 300,
    },
}

# Regions checked in order; first match wins. rest_of_world is the fallback.
REGION_ORDER: list[str] = [
    "japan", "cascadia", "san_andreas", "alaska",
    "chile", "mediterranean", "indonesia",
]

# Minimum samples required to train a region-specific model
MIN_TRAIN_SAMPLES: int = 50

# Magnitude threshold: use M5.5+ if a region has too few M6+ events
SPARSE_THRESHOLD_M6: int = 200  # if fewer M6+ events in region, use M5.5+


# ===================================================================
# FEATURE BLOCK SIZES
# ===================================================================

N_FEAT_S: int = 30   # Seismicity
N_FEAT_G: int = 12   # Geodetic/GNSS
N_FEAT_C: int = 8    # Coherence Field Theory
N_FEAT_T: int = 5    # Tectonic context
N_FEAT_TOTAL: int = N_FEAT_S + N_FEAT_G + N_FEAT_C + N_FEAT_T  # 55


# ===================================================================
# FEATURE NAMES
# ===================================================================

BLOCK_S_NAMES: list[str] = [
    # Rate features at multiple windows (6)
    "rate_7d", "rate_14d", "rate_30d", "rate_90d", "rate_180d", "rate_365d",
    # b-value (3)
    "b_overall", "b_shallow", "b_deep",
    # Nearest-neighbor (4)
    "nn_spatial", "nn_temporal", "st_nn", "nn_change",
    # Moment release (4)
    "mom_30d", "mom_90d", "mom_accel", "mom_deficit",
    # Foreshock (3)
    "inv_omori", "bath_ratio", "foreshock_frac",
    # Magnitude (3)
    "maxmag_30d", "maxmag_90d", "mag_var_chg",
    # Depth (3)
    "depth_mean", "depth_trend", "frac_shallow",
    # Clustering (3)
    "spatial_conc", "elong", "quiescence_7d",
    # IET delta-AIC (1)
    "iet_delta_aic",
]

BLOCK_G_NAMES: list[str] = [
    # Velocity anomaly (2)
    "vel_anom_h", "vel_anom_v",
    # Strain (2)
    "strain_rate", "strain_rate_anom",
    # Transient (3)
    "transient_30d", "transient_90d", "transient_180d",
    # Transient coherence (1)
    "transient_coherence",
    # Fault (2)
    "dist_nearest_fault", "fault_slip_rate",
    # Coverage (1)
    "n_gnss_100km",
    # Locking (1)
    "locking_proxy",
]

BLOCK_C_NAMES: list[str] = [
    "tau", "grad_tau", "S_over_Gamma", "Da",
    "divergence_rate", "t_c_estimate",
    "singularity_count", "tau_x_strain",
]

BLOCK_T_NAMES: list[str] = [
    "tectonic_subduction", "tectonic_transform", "tectonic_collision",
    "plate_boundary_dist", "regional_m6_rate",
]

ALL_FEATURE_NAMES: list[str] = (
    BLOCK_S_NAMES + BLOCK_G_NAMES + BLOCK_C_NAMES + BLOCK_T_NAMES
)
assert len(ALL_FEATURE_NAMES) == N_FEAT_TOTAL


# ===================================================================
# GBT HYPERPARAMETERS -- FIXED A PRIORI
# ===================================================================

GBT_N_TREES: int = 200
GBT_MAX_DEPTH: int = 4
GBT_LEARNING_RATE: float = 0.03
GBT_SUBSAMPLE: float = 0.6
GBT_COLSAMPLE: float = 0.7
GBT_MIN_SAMPLES_LEAF: int = 15
GBT_L2_REG: float = 1.0
GBT_GAMMA: float = 0.05
EARLY_STOP_PATIENCE: int = 40


# ===================================================================
# MATH UTILITIES
# ===================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1.0 - a)))


def haversine_vec(
    lat1: float, lon1: float,
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine: one point vs array. Returns km."""
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lats))
        * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0, 1.0 - a)))


def mag_to_moment(mag: np.ndarray) -> np.ndarray:
    """Magnitude(s) to seismic moment (N*m)."""
    return 10.0 ** (1.5 * np.asarray(mag, dtype=np.float64) + 9.05)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -88.0, 88.0)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    ).astype(np.float32)


# ===================================================================
# B-VALUE (Aki-Utsu MLE with max-curvature Mc)
# ===================================================================

def estimate_mc_maxcurv(magnitudes: np.ndarray, bin_width: float = 0.1) -> float:
    """Magnitude of completeness via maximum curvature method."""
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
    """B-value via Aki-Utsu MLE with max-curvature Mc."""
    mc = estimate_mc_maxcurv(magnitudes)
    above = magnitudes[magnitudes >= mc]
    if len(above) < 5:
        return 1.0, mc
    b = np.log10(np.e) / (np.mean(above) - (mc - 0.05))
    return b, mc


# ===================================================================
# GRADIENT BOOSTED TREES -- SELF-CONTAINED
# ===================================================================

class GradientBoostedTrees:
    """GBT for binary classification (log-loss, Newton-Raphson leaves)."""

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

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int = 0,
    ) -> dict:
        N = len(gradients)
        G = float(np.sum(gradients))
        H = float(np.sum(hessians))

        leaf_val = -G / (H + self.l2_reg) if (H + self.l2_reg) > 1e-12 else 0.0

        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {"leaf": True, "val": float(leaf_val)}

        D = X.shape[1]
        n_try = max(5, int(D * self.colsample))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)

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
        while not node["leaf"]:
            if x[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["val"]

    def _predict_tree_batch(self, node: dict, X: np.ndarray) -> np.ndarray:
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
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = len(y)

        n_pos = float(y.sum())
        n_neg = float(N - n_pos)
        assert n_pos > 0, "No positive samples"
        assert n_neg > 0, "No negative samples"
        sample_w = np.where(
            y == 1,
            N / (2.0 * n_pos),
            N / (2.0 * n_neg),
        ).astype(np.float32)

        p_avg = float(np.clip(y.mean(), 0.01, 0.99))
        self.init_pred = math.log(p_avg / (1.0 - p_avg))
        F_train = np.full(N, self.init_pred, dtype=np.float32)

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
                tree = self._build_tree(X[idx], gradients[idx], hessians[idx])
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
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        F = np.full(N, self.init_pred, dtype=np.float32)
        for tree in self.trees:
            for i in range(N):
                F[i] += self.lr * self._predict_tree_row(tree, X[i])
        return sigmoid(F)

    def feature_importances(self, n_features: int) -> np.ndarray:
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
    """Precision-Recall AUC."""
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
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_bss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    brier = compute_brier(y_true, y_pred)
    base_rate = float(np.mean(y_true))
    brier_clim = base_rate * (1 - base_rate)
    if brier_clim < 1e-12:
        return 0.0
    return float(1.0 - brier / brier_clim)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
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
    """Complete evaluation suite."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "auc": compute_auc(y_true, y_pred),
        "pr_auc": compute_pr_auc(y_true, y_pred),
        "brier": compute_brier(y_true, y_pred),
        "bss": compute_bss(y_true, y_pred),
        "bootstrap_ci": bootstrap_auc_ci(y_true, y_pred, n_boot=2000),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "base_rate": float(y_true.mean()) if len(y_true) > 0 else 0.0,
    }


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

    deltas = np.empty(n_boot, dtype=np.float64)
    n_a_wins = 0

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        auc_a = compute_auc(y_true[idx], y_pred_a[idx])
        auc_b = compute_auc(y_true[idx], y_pred_b[idx])
        deltas[b] = auc_b - auc_a
        if auc_a >= auc_b:
            n_a_wins += 1

    return {
        "delta_auc": float(auc_b_full - auc_a_full),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_value": float(n_a_wins / n_boot),
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
        assert self.mean is not None, "FeatureNormalizer not fit"
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ===================================================================
# GARDNER-KNOPOFF AFTERSHOCK DECLUSTERING
# ===================================================================

def _gardner_knopoff_window(mag: float) -> tuple[float, float]:
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
# EVENT UTILITIES
# ===================================================================

def _event_epoch(event: dict) -> float:
    t_str = event.get("time", "")
    if isinstance(t_str, str):
        return _parse_event_time(t_str)
    return 0.0


def _event_year(event: dict) -> int:
    t_str = event.get("time", "")
    if isinstance(t_str, str) and len(t_str) >= 4:
        try:
            return int(t_str[:4])
        except ValueError:
            pass
    return 0


# ===================================================================
# REGION IDENTIFICATION
# ===================================================================

def identify_region(lat: float, lon: float) -> str:
    """Identify which tectonic region a point belongs to.

    Checks named regions in REGION_ORDER first. If no match,
    returns 'rest_of_world'.
    """
    for rname in REGION_ORDER:
        rdef = REGIONS[rname]
        lat_lo, lat_hi = rdef["lat"]
        lon_lo, lon_hi = rdef["lon"]
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return rname
    return "rest_of_world"


def _in_region(lat: float, lon: float, region_name: str) -> bool:
    """Check if a point falls within a named region."""
    if region_name == "rest_of_world":
        # rest_of_world catches everything not in a specific region
        return identify_region(lat, lon) == "rest_of_world"
    rdef = REGIONS[region_name]
    lat_lo, lat_hi = rdef["lat"]
    lon_lo, lon_hi = rdef["lon"]
    return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi


# ===================================================================
# CATALOG ARRAYS (pre-built for speed)
# ===================================================================

class CatalogArrays:
    """Pre-built numpy arrays from full catalog for vectorized features."""

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
            dep = e.get("depth")
            self.depths[i] = dep if dep is not None else 10.0

        # Sort by time
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.lats = self.lats[order]
        self.lons = self.lons[order]
        self.mags = self.mags[order]
        self.depths = self.depths[order]

        # Pre-build M6+ and M5.5+ arrays for recurrence features
        m6_mask = self.mags >= 6.0
        self.m6_times = self.times[m6_mask]
        self.m6_lats = self.lats[m6_mask]
        self.m6_lons = self.lons[m6_mask]

        m55_mask = self.mags >= 5.5
        self.m55_times = self.times[m55_mask]
        self.m55_lats = self.lats[m55_mask]
        self.m55_lons = self.lons[m55_mask]

        if verbose:
            print(
                f"    Arrays built: {n} events, "
                f"{int(m6_mask.sum())} M6+, {int(m55_mask.sum())} M5.5+"
            )
            sys.stdout.flush()


class RegionalCatalogSlice:
    """Pre-sliced view of CatalogArrays for a specific region.

    Avoids re-scanning the full global catalog for every sample.
    """

    def __init__(
        self, cat: CatalogArrays, region_name: str,
        margin_deg: float = 5.0,
    ) -> None:
        rdef = REGIONS[region_name]
        lat_lo, lat_hi = rdef["lat"]
        lon_lo, lon_hi = rdef["lon"]

        # Expand bounds by margin for feature extraction
        mask = (
            (cat.lats >= lat_lo - margin_deg)
            & (cat.lats <= lat_hi + margin_deg)
            & (cat.lons >= lon_lo - margin_deg)
            & (cat.lons <= lon_hi + margin_deg)
        )

        self.times = cat.times[mask]
        self.lats = cat.lats[mask]
        self.lons = cat.lons[mask]
        self.mags = cat.mags[mask]
        self.depths = cat.depths[mask]

        m6_mask = self.mags >= 6.0
        self.m6_times = self.times[m6_mask]
        self.m6_lats = self.lats[m6_mask]
        self.m6_lons = self.lons[m6_mask]


# ===================================================================
# GNSS INDEX (global, loads from cache)
# ===================================================================

class GNSSIndex:
    """Spatial index for GNSS station data.

    Pre-loads GNSS station summaries from whichever public cache layout is
    available and builds a 2-degree grid for fast nearest-station lookups.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.stations: list[dict] = []
        self.lats: np.ndarray = np.array([], dtype=np.float64)
        self.lons: np.ndarray = np.array([], dtype=np.float64)
        self.grid: dict[tuple[int, int], list[int]] = {}
        self._loaded = False

        self._try_load(verbose=verbose)

    def _try_load(self, verbose: bool = True) -> None:
        """Try to load GNSS summary from cache."""
        project_root = Path(__file__).resolve().parents[3]
        candidate_paths = [
            project_root / ".cache" / "gnss" / "gnss_summary.json",
            project_root / ".cache" / "gnss" / "gnss_regional_summary.json",
            project_root / ".cache" / "earthquake" / "gnss" / "gnss_summary.json",
            project_root / ".cache" / "earthquake" / "gnss" / "gnss_regional_summary.json",
        ]
        summary_path = next((path for path in candidate_paths if path.exists()), None)

        if summary_path is None:
            if verbose:
                print("    GNSS data not found -- Block G will be NaN")
                sys.stdout.flush()
            return

        try:
            with open(summary_path) as f:
                data = json.load(f)
            self.stations = data.get("stations", [])
        except (json.JSONDecodeError, OSError):
            if verbose:
                print("    GNSS summary unreadable -- Block G will be NaN")
            return

        if not self.stations:
            return

        self.lats = np.array([s["lat"] for s in self.stations], dtype=np.float64)
        self.lons = np.array([s["lon"] for s in self.stations], dtype=np.float64)

        # Build 2-degree grid
        for idx, s in enumerate(self.stations):
            key = (int(s["lat"] / 2), int(s["lon"] / 2))
            self.grid.setdefault(key, []).append(idx)

        self._loaded = True
        if verbose:
            print(f"    GNSS index loaded: {len(self.stations)} stations from {summary_path}")
            sys.stdout.flush()

    def find_nearest(
        self, lat: float, lon: float, max_km: float = 300.0,
    ) -> list[tuple[int, float]]:
        """Find stations within max_km, sorted by distance.

        Returns list of (station_index, distance_km).
        """
        if not self._loaded:
            return []

        center_cell = (int(lat / 2), int(lon / 2))
        candidate_idx: set[int] = set()
        for dlat in range(-2, 3):
            for dlon in range(-2, 3):
                key = (center_cell[0] + dlat, center_cell[1] + dlon)
                if key in self.grid:
                    candidate_idx.update(self.grid[key])

        if not candidate_idx:
            # Fall back to full scan
            dists = haversine_vec(lat, lon, self.lats, self.lons)
            within = np.where(dists < max_km)[0]
            return sorted(
                [(int(i), float(dists[i])) for i in within],
                key=lambda x: x[1],
            )

        idx_arr = np.array(sorted(candidate_idx))
        dists = haversine_vec(lat, lon, self.lats[idx_arr], self.lons[idx_arr])
        within = dists < max_km
        result = [
            (int(idx_arr[i]), float(dists[i]))
            for i in range(len(idx_arr)) if within[i]
        ]
        result.sort(key=lambda x: x[1])
        return result

    def count_within(self, lat: float, lon: float, max_km: float) -> int:
        """Count stations within max_km (faster than find_nearest)."""
        return len(self.find_nearest(lat, lon, max_km=max_km))


# ===================================================================
# SAMPLE GENERATION
# ===================================================================

def identify_target_mainshocks(
    mainshocks: list[dict],
    min_mag: float = 6.0,
    region_name: str | None = None,
) -> list[dict]:
    """Filter declustered catalog to target mainshocks with valid locations.

    If region_name is specified, only includes events within that region.
    """
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
            if region_name is not None:
                if identify_region(lat, lon) != region_name:
                    continue
            targets.append(e)
    return targets


def generate_control_samples(
    target: dict,
    mainshocks: list[dict],
    n_controls: int = CONTROL_RATIO,
    offset_range: tuple[float, float] = CONTROL_OFFSET_RANGE,
    label_radius_km: float = 300.0,
    forward_days: float = FORWARD_WINDOW_DAYS,
    min_mag: float = 6.0,
    rng: np.random.RandomState | None = None,
) -> list[dict]:
    """Generate negative controls at same location, offset in time."""
    if rng is None:
        rng = np.random.RandomState(42)

    target_epoch = _event_epoch(target)
    target_lat = target["latitude"]
    target_lon = target["longitude"]

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

        has_target_forward = False
        for ms in mainshocks:
            ms_mag = ms.get("mag", 0)
            if ms_mag < min_mag:
                continue
            ms_epoch = _event_epoch(ms)
            dt_days = (ms_epoch - control_epoch) / 86400.0
            if dt_days < 0 or dt_days > forward_days:
                continue
            dist = haversine_km(
                target_lat, target_lon,
                ms["latitude"], ms["longitude"],
            )
            if dist <= label_radius_km:
                has_target_forward = True
                break

        if not has_target_forward:
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


def build_samples_for_region(
    mainshocks: list[dict],
    region_name: str,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Build positive + negative samples for a specific region.

    Automatically selects M5.5+ or M6.0+ threshold based on
    how many target events exist in the region.

    In honest mode, forces M6.0+, 100km radius, 90-day window, 5:1 ratio.

    Returns (samples, region_params) where region_params records the
    actual thresholds used.
    """
    rdef = REGIONS[region_name]

    if _HONEST_MODE:
        # Honest mode: strict parameters comparable to CSEP/ETAS
        label_radius_km = HONEST_LABEL_RADIUS_KM
        forward_days = HONEST_FORWARD_WINDOW_DAYS
        n_controls = HONEST_CONTROL_RATIO
        min_mag = HONEST_MAGNITUDE_THRESHOLD
        targets = identify_target_mainshocks(
            mainshocks, min_mag=min_mag, region_name=region_name,
        )
    else:
        # Default ("generous") mode
        label_radius_km = rdef["label_radius_km"]
        forward_days = FORWARD_WINDOW_DAYS
        n_controls = CONTROL_RATIO

        # First check how many M6+ events we have
        targets_m6 = identify_target_mainshocks(
            mainshocks, min_mag=6.0, region_name=region_name,
        )

        # Decide threshold
        if len(targets_m6) >= SPARSE_THRESHOLD_M6:
            min_mag = 6.0
            targets = targets_m6
        else:
            # Try M5.5+ for regions with fewer large events
            targets_m55 = identify_target_mainshocks(
                mainshocks, min_mag=5.5, region_name=region_name,
            )
            if len(targets_m55) > len(targets_m6) * 1.5:
                min_mag = 5.5
                targets = targets_m55
            else:
                min_mag = 6.0
                targets = targets_m6

    region_params = {
        "magnitude_threshold": min_mag,
        "label_radius_km": label_radius_km,
        "forward_window_days": forward_days,
        "control_ratio": n_controls,
    }

    if verbose:
        mode_tag = "[HONEST] " if _HONEST_MODE else ""
        print(
            f"    {mode_tag}{region_name}: {len(targets)} M{min_mag}+ mainshocks "
            f"(radius={label_radius_km}km, window={forward_days}d, "
            f"ctrl_ratio={n_controls}:1)"
        )
        sys.stdout.flush()

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
            "region": region_name,
        })

        controls = generate_control_samples(
            target, mainshocks, rng=rng,
            n_controls=n_controls,
            label_radius_km=label_radius_km,
            forward_days=forward_days,
            min_mag=min_mag,
        )
        for ctrl in controls:
            ctrl_year = int(ctrl["time"][:4]) if isinstance(ctrl["time"], str) else 0
            samples.append({
                "latitude": ctrl["latitude"],
                "longitude": ctrl["longitude"],
                "ref_epoch": ctrl["ref_epoch"],
                "label": ctrl["label"],
                "year": ctrl_year,
                "region": region_name,
            })

    if verbose:
        n_pos = sum(1 for s in samples if s["label"] == 1)
        n_neg = sum(1 for s in samples if s["label"] == 0)
        base_rate = n_pos / max(1, len(samples))
        print(
            f"      Samples: {len(samples)} ({n_pos} pos, {n_neg} neg, "
            f"base_rate={base_rate:.3f})"
        )
        sys.stdout.flush()

    region_params["base_rate"] = n_pos / max(1, len(samples))

    return samples, region_params


# ===================================================================
# BLOCK S: SEISMICITY FEATURES (30 features)
# ===================================================================

def compute_block_s(
    ev_lat: float,
    ev_lon: float,
    ev_time: float,
    cat: CatalogArrays | RegionalCatalogSlice,
    search_radius_km: float = 400.0,
) -> np.ndarray | None:
    """Compute 30 seismicity features.

    Uses only events BEFORE ev_time within search_radius_km and 5 years.
    """
    t_start = ev_time - 5.0 * SEC_PER_YEAR
    time_mask = (cat.times >= t_start) & (cat.times < ev_time)
    box_deg = 4.0
    spatial_mask = (
        (np.abs(cat.lats - ev_lat) < box_deg)
        & (np.abs(cat.lons - ev_lon) < box_deg)
    )
    mask = time_mask & spatial_mask

    if np.sum(mask) < 10:
        return None

    t = cat.times[mask]
    la = cat.lats[mask]
    lo = cat.lons[mask]
    m = cat.mags[mask]
    d = cat.depths[mask]
    dists = haversine_vec(ev_lat, ev_lon, la, lo)

    rmask = dists < search_radius_km
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

    # ---- Rate features at multiple windows (6) ----
    for wname, wdays in [
        ("7d", 7), ("14d", 14), ("30d", 30),
        ("90d", 90), ("180d", 180), ("365d", 365),
    ]:
        n_win = float(np.sum(dt < wdays))
        n_base = float(np.sum((dt >= wdays) & (dt < wdays * 3)))
        base_rate = n_base / (2.0 * wdays + 0.01)
        win_rate = n_win / (wdays + 0.01)
        feats[f"rate_{wname}"] = float(np.log1p(win_rate / (base_rate + 0.001)))

    # ---- b-value (3): overall, shallow (<30km), deep (>80km) ----
    if N >= 20:
        b_all, _ = b_value_proper(m)
        feats["b_overall"] = b_all
    else:
        feats["b_overall"] = 1.0

    shallow_mask = d < 30
    shallow_m = m[shallow_mask]
    feats["b_shallow"] = b_value_proper(shallow_m)[0] if len(shallow_m) >= 15 else 1.0

    deep_mask = d > 80
    deep_m = m[deep_mask]
    feats["b_deep"] = b_value_proper(deep_m)[0] if len(deep_m) >= 15 else 1.0

    # ---- Nearest-neighbor (4) ----
    near = dists < 100
    if np.sum(near) >= 3:
        near_dists = np.sort(dists[near])[:3]
        feats["nn_spatial"] = float(np.mean(near_dists))
    else:
        feats["nn_spatial"] = 100.0

    # Temporal NN
    if N >= 5:
        sorted_dt = np.sort(dt)
        iet = np.diff(sorted_dt)
        feats["nn_temporal"] = float(np.min(iet)) if len(iet) > 0 else 100.0
    else:
        feats["nn_temporal"] = 100.0

    # Spatiotemporal NN
    if N >= 20:
        rng_st = np.random.RandomState(0)
        sample_idx = rng_st.choice(N, min(N, 50), replace=False)
        st_vals = []
        for i in sample_idx:
            other = np.arange(N) != i
            sd = haversine_vec(la[i], lo[i], la[other], lo[other])
            td = np.abs(t[i] - t[other]) / SEC_PER_DAY + 0.01
            st = np.log10(sd + 0.1) + np.log10(td)
            st_vals.append(float(np.min(st)))
        feats["st_nn"] = float(np.mean(st_vals))
    else:
        feats["st_nn"] = 5.0

    # NN change (recent vs older)
    r90_mask = dt < 90
    old_mask = dt >= 90
    if np.sum(r90_mask) >= 3 and np.sum(old_mask) >= 3:
        feats["nn_change"] = float(
            np.mean(np.sort(dists[r90_mask])[:3])
            - np.mean(np.sort(dists[old_mask])[:3])
        )
    else:
        feats["nn_change"] = 0.0

    # ---- Moment release (4) ----
    moments = mag_to_moment(m)
    for wname, wdays in [("30d", 30), ("90d", 90)]:
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

    # ---- Foreshock (3) ----
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

    m30 = m[dt < 30]
    if len(m30) >= 2:
        sm = np.sort(m30)[::-1]
        feats["bath_ratio"] = float(sm[1] / (sm[0] + 0.01))
    else:
        feats["bath_ratio"] = 0.0

    if N >= 10:
        check = min(N, 100)
        fcount = 0
        for i in range(max(0, N - check), N):
            if np.any((t > t[i]) & (t < t[i] + 7 * SEC_PER_DAY) & (m > m[i])):
                fcount += 1
        feats["foreshock_frac"] = fcount / check
    else:
        feats["foreshock_frac"] = 0.0

    # ---- Magnitude (3) ----
    for wname, wdays in [("30d", 30), ("90d", 90)]:
        wm = m[dt < wdays]
        feats[f"maxmag_{wname}"] = float(np.max(wm)) if len(wm) > 0 else 4.0

    if N >= 20:
        half = N // 2
        feats["mag_var_chg"] = float(np.var(m[half:]) - np.var(m[:half]))
    else:
        feats["mag_var_chg"] = 0.0

    # ---- Depth (3) ----
    feats["depth_mean"] = float(np.mean(d))
    if N >= 20:
        half = N // 2
        feats["depth_trend"] = float(np.mean(d[half:]) - np.mean(d[:half]))
    else:
        feats["depth_trend"] = 0.0
    feats["frac_shallow"] = float(np.sum(d < 30)) / max(1, N)

    # ---- Clustering (3) ----
    if r90_events >= 3:
        feats["spatial_conc"] = float(np.std(dists[r90_mask]))
    else:
        feats["spatial_conc"] = 100.0

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

    # Quiescence
    r7d = float(np.sum(dt < 7))
    r365d = float(np.sum(dt < 365))
    expected_7d = r365d / 365 * 7
    feats["quiescence_7d"] = float(np.log1p(expected_7d / (r7d + 0.1)))

    # ---- IET delta-AIC (1) ----
    if N >= 30:
        iet = np.diff(np.sort(dt))
        iet = iet[iet > 0]
        if len(iet) >= 20:
            lam = 1.0 / (np.mean(iet) + 0.01)
            ll_exp = float(np.sum(np.log(lam * np.exp(-lam * iet) + 1e-30)))
            median_iet = float(np.median(iet))
            gamma_lor = median_iet / 2.0 + 0.01
            ll_lor = float(np.sum(
                np.log(gamma_lor / (np.pi * (iet ** 2 + gamma_lor ** 2)) + 1e-30)
            ))
            aic_exp = -2 * ll_exp + 2
            aic_lor = -2 * ll_lor + 2
            feats["iet_delta_aic"] = aic_exp - aic_lor
        else:
            feats["iet_delta_aic"] = 0.0
    else:
        feats["iet_delta_aic"] = 0.0

    # Assemble into ordered array matching BLOCK_S_NAMES
    result = np.zeros(N_FEAT_S, dtype=np.float32)
    for i, name in enumerate(BLOCK_S_NAMES):
        result[i] = feats.get(name, 0.0)

    return result


# ===================================================================
# BLOCK G: GEODETIC/GNSS FEATURES (12 features)
# ===================================================================

def compute_block_g(
    lat: float,
    lon: float,
    gnss: GNSSIndex,
    ref_epoch: float = 0.0,
) -> np.ndarray:
    """Extract 12 GNSS/geodetic features.

    Handles gracefully when no GNSS stations are available nearby.

    TODO: When GNSS time-series data is available (parsed .tenv3),
    use ref_epoch to compute:
      - velocity_anomaly: velocity in 180 days before ref_epoch vs long-term
      - transient_at_time: max deviation from linear trend in 90 days before
      - strain_rate_at_time: strain rate in the year before ref_epoch
    Currently the cached summary only has aggregate stats, so positive and
    control samples at the same location get identical GNSS features.
    """
    feats = np.full(N_FEAT_G, np.nan, dtype=np.float32)

    nearby = gnss.find_nearest(lat, lon, max_km=300.0)
    if not nearby:
        return feats

    # Nearest station
    idx0, dist0 = nearby[0]
    s0 = gnss.stations[idx0]

    # Velocity anomaly (2)
    vel_h = s0.get("vel_h", 0.0) or 0.0
    vel_v = s0.get("vel_v", 0.0) or 0.0
    feats[0] = vel_h                                      # vel_anom_h
    feats[1] = vel_v                                      # vel_anom_v

    # Strain rate (2)
    feats[2] = s0.get("strain_rate", 0.0) or 0.0          # strain_rate

    # Strain rate anomaly: velocity spread across nearby stations
    if len(nearby) >= 3:
        vels_h = []
        stn_dists = []
        for idx_i, dist_i in nearby[:5]:
            si = gnss.stations[idx_i]
            vels_h.append(si.get("vel_h", 0.0) or 0.0)
            stn_dists.append(dist_i)
        vel_spread = float(np.std(vels_h))
        mean_dist = float(np.mean(stn_dists)) + 1.0
        feats[3] = vel_spread / mean_dist                  # strain_rate_anom
    else:
        feats[3] = 0.0

    # Transient (3)
    feats[4] = s0.get("transient_30d", 0.0) or 0.0        # transient_30d
    feats[5] = s0.get("transient_90d", 0.0) or 0.0        # transient_90d
    feats[6] = s0.get("transient_180d", 0.0) or 0.0       # transient_180d

    # Transient coherence (1)
    if len(nearby) >= 2:
        n_transient = 0
        threshold_mm = 3.0
        for idx_i, dist_i in nearby[:10]:
            si = gnss.stations[idx_i]
            tr = si.get("transient_90d", 0.0) or 0.0
            if tr > threshold_mm:
                n_transient += 1
        feats[7] = n_transient / min(len(nearby), 10)      # transient_coherence
    else:
        feats[7] = 0.0

    # Distance to nearest fault (1) -- proxy: use distance to nearest
    # high-strain GNSS station as a fault proximity indicator
    feats[8] = dist0                                       # dist_nearest_fault (proxy)

    # Fault slip rate (1) -- proxy: highest strain rate within 200km
    if len(nearby) >= 1:
        max_strain = 0.0
        for idx_i, dist_i in nearby[:10]:
            si = gnss.stations[idx_i]
            sr = si.get("strain_rate", 0.0) or 0.0
            max_strain = max(max_strain, sr)
        feats[9] = max_strain                              # fault_slip_rate (proxy)
    else:
        feats[9] = 0.0

    # N GNSS stations within 100km (1)
    n_100 = sum(1 for _, d in nearby if d < 100.0)
    feats[10] = float(n_100)                               # n_gnss_100km

    # Locking proxy (1)
    # If velocity is much lower than expected plate motion, fault may be locked
    expected_vel = 40.0  # mm/yr generic plate motion
    feats[11] = max(0.0, 1.0 - vel_h / expected_vel)      # locking_proxy

    return feats


# ===================================================================
# BLOCK C: COHERENCE FIELD THEORY FEATURES (8 features)
# ===================================================================

def compute_block_c(
    cat: "CatalogArrays | RegionalCatalogSlice | list[dict]",
    lat: float,
    lon: float,
    ref_epoch: float,
    label_radius_km: float = 300.0,
    strain_rate: float = 0.0,
) -> np.ndarray:
    """Extract 8 CFT features.

    Includes one cross-physics interaction term (tau x strain).

    Accepts CatalogArrays / RegionalCatalogSlice (converts to list[dict]
    for the coherence engine) OR a plain list[dict].
    """
    feats = np.full(N_FEAT_C, np.nan, dtype=np.float64)

    # Convert CatalogArrays/RegionalCatalogSlice to list[dict] for the
    # coherence engine, which expects list[dict].  Only include nearby
    # events to keep memory bounded.
    if hasattr(cat, 'lats'):
        dlat = np.abs(cat.lats - lat)
        dlon = np.abs(cat.lons - lon)
        box_deg = label_radius_km / 111.0 * 2
        nearby = (dlat < box_deg) & (dlon < box_deg) & (cat.times < ref_epoch)
        n_nearby = int(nearby.sum())
        if n_nearby < 10:
            return feats.astype(np.float32)  # not enough data
        nearby_idx = np.where(nearby)[0]
        # Cap at 5000 events to save memory
        if len(nearby_idx) > 5000:
            nearby_idx = nearby_idx[-5000:]  # keep most recent
        events_list: list[dict] = []
        for idx in nearby_idx:
            epoch_val = float(cat.times[idx])
            dt_obj = _dt.datetime.fromtimestamp(epoch_val, tz=_dt.timezone.utc)
            events_list.append({
                'latitude': float(cat.lats[idx]),
                'longitude': float(cat.lons[idx]),
                'magnitude': float(cat.mags[idx]),
                'mag': float(cat.mags[idx]),
                'time': dt_obj.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                'depth': float(cat.depths[idx]) if hasattr(cat, 'depths') else 10.0,
            })
    else:
        events_list = cat  # already list[dict]

    fwd_days = HONEST_FORWARD_WINDOW_DAYS if _HONEST_MODE else FORWARD_WINDOW_DAYS
    try:
        cft = extract_coherence_features(
            events_list,
            lat=lat,
            lon=lon,
            radius_km=label_radius_km,
            time_window_days=fwd_days,
            ref_epoch=ref_epoch,
        )
    except Exception:
        # If coherence extraction fails, return NaN features
        return feats.astype(np.float32)

    feats[0] = cft.get("tau_local", np.nan)                # tau
    feats[1] = cft.get("grad_tau_local", np.nan)           # grad_tau
    feats[2] = cft.get("S_over_Gamma", np.nan)             # S_over_Gamma
    feats[3] = cft.get("Da_local", np.nan)                 # Da
    feats[4] = cft.get("ell_trend", np.nan)                # divergence_rate
    feats[5] = cft.get("days_to_criticality", np.nan)      # t_c_estimate

    sing = test_earthquake_singularity(cft)
    feats[6] = float(sing.conditions_met)                  # singularity_count

    tau = cft.get("tau_local", 0.0) or 0.0
    feats[7] = tau * strain_rate                            # tau_x_strain

    return feats.astype(np.float32)


# ===================================================================
# BLOCK T: TECTONIC CONTEXT FEATURES (5 features)
# ===================================================================

def compute_block_t(
    lat: float,
    lon: float,
    region_name: str,
    cat: CatalogArrays,
    ev_time: float,
) -> np.ndarray:
    """Extract 5 tectonic context features.

    One-hot for tectonic type + plate boundary distance proxy +
    regional historical M6+ rate.
    """
    feats = np.zeros(N_FEAT_T, dtype=np.float32)

    rdef = REGIONS[region_name]
    rtype = rdef["type"]

    # One-hot tectonic type (3)
    if rtype == "subduction":
        feats[0] = 1.0
    elif rtype == "transform":
        feats[1] = 1.0
    elif rtype == "collision":
        feats[2] = 1.0
    # "mixed" gets all zeros

    # Plate boundary distance proxy (1)
    # Use distance to nearest M6+ event as a proxy for plate boundary
    if len(cat.m6_lats) > 0:
        m6_dists = haversine_vec(lat, lon, cat.m6_lats, cat.m6_lons)
        m6_before = cat.m6_times < ev_time
        if np.sum(m6_before) > 0:
            feats[3] = float(np.min(m6_dists[m6_before]))  # plate_boundary_dist
        else:
            feats[3] = 500.0
    else:
        feats[3] = 500.0

    # Regional historical M6+ rate (events per year per 10,000 km2) (1)
    rdef_region = REGIONS[region_name]
    lat_lo, lat_hi = rdef_region["lat"]
    lon_lo, lon_hi = rdef_region["lon"]
    lat_span = lat_hi - lat_lo
    lon_span = lon_hi - lon_lo
    # Approximate area in km2
    mid_lat = (lat_lo + lat_hi) / 2.0
    area_km2 = (
        lat_span * 111.0 * lon_span * 111.0
        * math.cos(math.radians(mid_lat))
    )
    area_km2 = max(area_km2, 1.0)

    # Count M6+ events in region before ev_time
    m6_in_region = (
        (cat.m6_lats >= lat_lo) & (cat.m6_lats <= lat_hi)
        & (cat.m6_lons >= lon_lo) & (cat.m6_lons <= lon_hi)
        & (cat.m6_times < ev_time)
    )
    n_m6 = int(np.sum(m6_in_region))
    # Time span of available data
    if n_m6 > 0:
        t_span_yr = max(
            1.0,
            (ev_time - float(np.min(cat.m6_times[m6_in_region]))) / SEC_PER_YEAR,
        )
    else:
        t_span_yr = 20.0

    feats[4] = float(n_m6 / t_span_yr / (area_km2 / 10000.0))  # regional_m6_rate

    return feats


# ===================================================================
# FULL FEATURE EXTRACTION FOR ONE SAMPLE
# ===================================================================

def extract_all_features(
    lat: float,
    lon: float,
    ref_epoch: float,
    region_name: str,
    cat: CatalogArrays | RegionalCatalogSlice,
    global_cat: CatalogArrays,
    gnss: GNSSIndex,
    full_catalog: list[dict] | None = None,
) -> np.ndarray | None:
    """Extract all 55 features for one sample.

    Returns shape (55,) float32, or None if insufficient seismicity data.
    """
    rdef = REGIONS[region_name]

    # Block S: Seismicity (30 features)
    s_feats = compute_block_s(lat, lon, ref_epoch, cat)
    if s_feats is None:
        return None

    # Block G: Geodetic (12 features)
    g_feats = compute_block_g(lat, lon, gnss, ref_epoch=ref_epoch)

    # Block C: Coherence (8 features)
    # Use regional catalog instead of full catalog to save memory
    strain_val = float(g_feats[2]) if not np.isnan(g_feats[2]) else 0.0
    c_feats = compute_block_c(
        cat, lat, lon, ref_epoch,
        label_radius_km=rdef["label_radius_km"],
        strain_rate=strain_val,
    )
    c_feats = np.nan_to_num(c_feats, nan=0.0)

    # Block T: Tectonic context (5 features)
    t_feats = compute_block_t(lat, lon, region_name, global_cat, ref_epoch)

    # Combine all blocks
    combined = np.concatenate([s_feats, g_feats, c_feats, t_feats])
    assert combined.shape[0] == N_FEAT_TOTAL, (
        f"Feature vector has {combined.shape[0]} elements, expected {N_FEAT_TOTAL}"
    )

    return combined


# ===================================================================
# DATA LOADING AND PIPELINE
# ===================================================================

def load_all_data(
    verbose: bool = True,
) -> tuple[
    dict[str, dict],     # per-region feature/label data
    CatalogArrays,       # global catalog arrays
    list[dict],          # full catalog (list of event dicts)
    GNSSIndex,           # GNSS index
    dict,                # metadata
]:
    """Load all data, build samples per region, extract features.

    Returns per-region dictionaries with train/val/test splits.
    """
    # Step 1: Load USGS catalog
    if verbose:
        print("  [1] Loading USGS catalog (2000-2024, M2.5+)...")
        sys.stdout.flush()

    full_catalog = load_usgs_catalog(min_year=2000, max_year=2024, min_mag=2.5)

    if len(full_catalog) == 0:
        raise RuntimeError(
            "No earthquake data found after attempting to bootstrap the USGS cache. "
            "Check network access or run scripts/download_earthquake_data.py manually."
        )

    if verbose:
        print(f"      Loaded {len(full_catalog)} events")
        sys.stdout.flush()

    # Step 2: Gardner-Knopoff aftershock declustering
    if verbose:
        print("  [2] Aftershock declustering (Gardner-Knopoff 1974)...")
        sys.stdout.flush()

    mainshocks, aftershocks = decluster_gardner_knopoff(full_catalog)

    if verbose:
        print(
            f"      Declustered: {len(mainshocks)} mainshocks, "
            f"{len(aftershocks)} aftershocks removed"
        )
        sys.stdout.flush()

    # Step 3: Build global catalog arrays
    if verbose:
        print("  [3] Building catalog arrays...")
        sys.stdout.flush()

    global_cat = CatalogArrays(full_catalog, verbose=verbose)

    # Step 4: Load GNSS
    if verbose:
        print("  [4] Loading GNSS data...")
        sys.stdout.flush()

    gnss = GNSSIndex(verbose=verbose)

    # Step 5: Build samples per region, extract features, split
    if verbose:
        print("  [5] Building samples and extracting features per region...")
        sys.stdout.flush()

    all_regions = list(REGIONS.keys())
    region_data: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    for rname in all_regions:
        if verbose:
            print(f"\n  --- Region: {rname.upper()} ---")
            sys.stdout.flush()

        # Build samples for this region
        samples, region_params = build_samples_for_region(
            mainshocks, rname, verbose=verbose,
        )

        if len(samples) == 0:
            if verbose:
                print(f"    No samples for {rname}, skipping")
                sys.stdout.flush()
            continue

        # Temporal split
        train_s = [s for s in samples if TRAIN_START <= s["year"] <= TRAIN_END]
        val_s = [s for s in samples if VAL_START <= s["year"] <= VAL_END]
        test_s = [s for s in samples if TEST_START <= s["year"] <= TEST_END]

        # Temporal integrity assertions
        for s in train_s:
            assert s["year"] <= TRAIN_END
        for s in val_s:
            assert VAL_START <= s["year"] <= VAL_END
        for s in test_s:
            assert TEST_START <= s["year"] <= TEST_END

        if verbose:
            print(
                f"    Split: train={len(train_s)}, val={len(val_s)}, "
                f"test={len(test_s)}"
            )
            sys.stdout.flush()

        # CHECKPOINTING: skip regions whose features are already saved
        ckpt_suffix = "_honest" if _HONEST_MODE else ""
        checkpoint_dir = Path(f"results/earthquake_definitive/_checkpoints{ckpt_suffix}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{rname}_features.npz"

        if checkpoint_file.exists():
            if verbose:
                print(f"    [CHECKPOINT] Loading saved features from {checkpoint_file}")
                sys.stdout.flush()
            ckpt = np.load(checkpoint_file)
            X_train = ckpt["X_train"]
            y_train = ckpt["y_train"]
            X_val = ckpt["X_val"]
            y_val = ckpt["y_val"]
            X_test = ckpt["X_test"]
            y_test = ckpt["y_test"]

            region_data[rname] = {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
            }
            meta[rname] = {
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "train_pos": int(y_train.sum()) if len(y_train) > 0 else 0,
                "train_neg": int(len(y_train) - y_train.sum()) if len(y_train) > 0 else 0,
                "test_pos": int(y_test.sum()) if len(y_test) > 0 else 0,
                "test_neg": int(len(y_test) - y_test.sum()) if len(y_test) > 0 else 0,
                "tectonic_type": REGIONS[rname]["type"],
                **region_params,
            }
            continue  # skip to next region

        # Pre-slice catalog for this region (speed optimization)
        if rname != "rest_of_world":
            regional_cat = RegionalCatalogSlice(global_cat, rname)
        else:
            regional_cat = global_cat  # type: ignore[assignment]

        # Extract features for each split
        def _extract_split(
            split_samples: list[dict], split_name: str,
        ) -> tuple[np.ndarray, np.ndarray]:
            n = len(split_samples)
            if n == 0:
                return np.zeros((0, N_FEAT_TOTAL), dtype=np.float32), np.zeros(0, dtype=np.float32)

            X = np.zeros((n, N_FEAT_TOTAL), dtype=np.float32)
            y = np.zeros(n, dtype=np.float32)
            valid_mask = np.ones(n, dtype=bool)

            t0 = time.time()
            n_gnss_ok = 0

            for idx, sample in enumerate(split_samples):
                if verbose and (idx + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"      {rname}/{split_name}: {idx + 1}/{n} "
                        f"({rate:.0f} samples/min)"
                    )
                    sys.stdout.flush()

                feats = extract_all_features(
                    sample["latitude"],
                    sample["longitude"],
                    sample["ref_epoch"],
                    rname,
                    regional_cat,
                    global_cat,
                    gnss,
                )

                if feats is None:
                    valid_mask[idx] = False
                else:
                    X[idx] = feats
                    if not np.all(np.isnan(feats[N_FEAT_S:N_FEAT_S + N_FEAT_G])):
                        n_gnss_ok += 1

                y[idx] = sample["label"]

            # Remove samples with insufficient data
            X = X[valid_mask]
            y = y[valid_mask]

            if verbose:
                elapsed = time.time() - t0
                n_valid = int(valid_mask.sum())
                print(
                    f"      {rname}/{split_name}: {n_valid}/{n} valid, "
                    f"{n_gnss_ok} w/ GNSS, {elapsed:.1f}s"
                )
                sys.stdout.flush()

            return X, y

        X_train, y_train = _extract_split(train_s, "train")
        X_val, y_val = _extract_split(val_s, "val")
        X_test, y_test = _extract_split(test_s, "test")

        # Impute NaN with training mean
        if len(X_train) > 0:
            train_mean = np.nanmean(X_train, axis=0)
            train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)

            for X in [X_train, X_val, X_test]:
                nan_mask = np.isnan(X)
                for col in range(X.shape[1]):
                    col_nans = nan_mask[:, col]
                    if col_nans.any():
                        X[col_nans, col] = train_mean[col]

        region_data[rname] = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }

        # CHECKPOINT: save features so we can resume if crashed
        np.savez_compressed(
            checkpoint_file,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )
        if verbose:
            print(f"    [CHECKPOINT] Saved to {checkpoint_file}")
            sys.stdout.flush()

        # Free memory between regions
        gc.collect()

        meta[rname] = {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "train_pos": int(y_train.sum()) if len(y_train) > 0 else 0,
            "train_neg": int(len(y_train) - y_train.sum()) if len(y_train) > 0 else 0,
            "test_pos": int(y_test.sum()) if len(y_test) > 0 else 0,
            "test_neg": int(len(y_test) - y_test.sum()) if len(y_test) > 0 else 0,
            "tectonic_type": REGIONS[rname]["type"],
            **region_params,
        }

    return region_data, global_cat, full_catalog, gnss, meta


# ===================================================================
# TRAINING: REGIONAL MODELS + GLOBAL BASELINE
# ===================================================================

def train_regional_models(
    region_data: dict[str, dict],
    verbose: bool = True,
) -> tuple[
    dict[str, GradientBoostedTrees],   # regional models
    dict[str, FeatureNormalizer],       # normalizers
    dict[str, dict],                    # training meta per region
]:
    """Train one GBT per region with enough data.

    Regions with < MIN_TRAIN_SAMPLES training samples will be skipped
    (they'll use the rest_of_world model at prediction time).
    """
    models: dict[str, GradientBoostedTrees] = {}
    normalizers: dict[str, FeatureNormalizer] = {}
    train_meta: dict[str, dict] = {}

    for rname, rdata in region_data.items():
        X_train = rdata["X_train"]
        y_train = rdata["y_train"]
        X_val = rdata["X_val"]
        y_val = rdata["y_val"]

        n_train = len(y_train)
        n_pos = int(y_train.sum()) if n_train > 0 else 0
        n_neg = n_train - n_pos

        if n_train < MIN_TRAIN_SAMPLES or n_pos < 5 or n_neg < 5:
            if verbose:
                print(
                    f"\n  {rname.upper()}: SKIPPED "
                    f"(train={n_train}, pos={n_pos}, neg={n_neg})"
                )
                sys.stdout.flush()
            train_meta[rname] = {"skipped": True, "reason": "insufficient_data"}
            continue

        if verbose:
            print(
                f"\n  --- Training {rname.upper()} "
                f"({n_train} train, {len(y_val)} val, "
                f"{N_FEAT_TOTAL} features) ---"
            )
            sys.stdout.flush()

        # Normalize
        norm = FeatureNormalizer()
        X_train_n = norm.fit_transform(X_train)
        X_val_n = norm.transform(X_val) if len(X_val) > 0 else None
        y_val_n = y_val if len(y_val) > 0 else None

        normalizers[rname] = norm

        # Train GBT
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
            X_train_n, y_train,
            X_val=X_val_n, y_val=y_val_n,
            verbose=verbose,
        )

        models[rname] = gbt
        train_meta[rname] = {
            "skipped": False,
            "n_trees_used": tmeta["n_trees_used"],
            "stopped_early": tmeta["stopped_early"],
            "best_val_loss": tmeta.get("best_val_loss"),
        }

        if verbose:
            print(
                f"    Trees used: {tmeta['n_trees_used']}, "
                f"early_stop: {tmeta['stopped_early']}"
            )
            sys.stdout.flush()

    return models, normalizers, train_meta


def train_global_baseline(
    region_data: dict[str, dict],
    verbose: bool = True,
) -> tuple[GradientBoostedTrees, FeatureNormalizer, dict]:
    """Train a single global model on ALL data (for comparison).

    Combines train/val/test from all regions into one big pool.
    """
    if verbose:
        print("\n  --- Training GLOBAL BASELINE (all regions combined) ---")
        sys.stdout.flush()

    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []

    for rname, rdata in region_data.items():
        if len(rdata["X_train"]) > 0:
            all_X_train.append(rdata["X_train"])
            all_y_train.append(rdata["y_train"])
        if len(rdata["X_val"]) > 0:
            all_X_val.append(rdata["X_val"])
            all_y_val.append(rdata["y_val"])

    X_train = np.vstack(all_X_train) if all_X_train else np.zeros((0, N_FEAT_TOTAL))
    y_train = np.concatenate(all_y_train) if all_y_train else np.zeros(0)
    X_val = np.vstack(all_X_val) if all_X_val else None
    y_val = np.concatenate(all_y_val) if all_y_val else None

    if verbose:
        n_pos = int(y_train.sum()) if len(y_train) > 0 else 0
        print(
            f"    Global: {len(y_train)} train ({n_pos} pos), "
            f"{len(y_val) if y_val is not None else 0} val"
        )
        sys.stdout.flush()

    norm = FeatureNormalizer()
    X_train_n = norm.fit_transform(X_train)
    X_val_n = norm.transform(X_val) if X_val is not None and len(X_val) > 0 else None

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
        X_train_n, y_train,
        X_val=X_val_n, y_val=y_val,
        verbose=verbose,
    )

    if verbose:
        print(
            f"    Global trees used: {tmeta['n_trees_used']}, "
            f"early_stop: {tmeta['stopped_early']}"
        )
        sys.stdout.flush()

    return gbt, norm, tmeta


# ===================================================================
# EVALUATION: REGIONAL + GLOBAL COMPARISON
# ===================================================================

def evaluate_all(
    region_data: dict[str, dict],
    regional_models: dict[str, GradientBoostedTrees],
    regional_normalizers: dict[str, FeatureNormalizer],
    global_model: GradientBoostedTrees,
    global_normalizer: FeatureNormalizer,
    train_meta: dict[str, dict],
    verbose: bool = True,
) -> dict:
    """Evaluate regional models per region and combined, vs global baseline.

    Returns a comprehensive results dict.
    """
    results: dict = {
        "regional_results": {},
        "global_combined": {},
        "global_baseline": {},
        "feature_importance": {},
        "comparison": {},
    }

    # Minimum test positives to include in global combined AUC
    MIN_TEST_POSITIVES_FOR_GLOBAL: int = 20
    insufficient_data_regions: list[str] = []

    # Collect all test predictions for global combined evaluation
    all_y_test = []
    all_p_regional = []
    all_p_global = []

    for rname, rdata in region_data.items():
        X_test = rdata["X_test"]
        y_test = rdata["y_test"]

        if len(y_test) == 0:
            if verbose:
                print(f"\n  {rname.upper()}: No test samples, skipping evaluation")
            continue

        # Regional model prediction
        if rname in regional_models and not train_meta.get(rname, {}).get("skipped", True):
            # Use the regional model
            model_used = rname
            norm = regional_normalizers[rname]
            X_test_n = norm.transform(X_test)
            p_regional = regional_models[rname].predict_proba(X_test_n)
        elif "rest_of_world" in regional_models:
            # Fallback to rest_of_world model
            model_used = "rest_of_world (fallback)"
            norm = regional_normalizers["rest_of_world"]
            X_test_n = norm.transform(X_test)
            p_regional = regional_models["rest_of_world"].predict_proba(X_test_n)
        else:
            # No model available at all
            if verbose:
                print(f"\n  {rname.upper()}: No model available")
            continue

        # Global baseline prediction
        X_test_g = global_normalizer.transform(X_test)
        p_global = global_model.predict_proba(X_test_g)

        # Per-region evaluation
        reg_eval = evaluate(y_test, p_regional)
        glb_eval = evaluate(y_test, p_global)

        results["regional_results"][rname] = {
            "model_used": model_used,
            "metrics": reg_eval,
        }

        if verbose:
            print(f"\n  {rname.upper()} (model: {model_used}):")
            print(
                f"    Regional AUC: {reg_eval['auc']:.4f} "
                f"[{reg_eval['bootstrap_ci']['ci_lo']:.4f}, "
                f"{reg_eval['bootstrap_ci']['ci_hi']:.4f}]"
            )
            print(f"    Global  AUC: {glb_eval['auc']:.4f}")
            print(
                f"    Lift: {reg_eval['auc'] - glb_eval['auc']:+.4f} "
                f"(regional - global)"
            )
            print(
                f"    N_pos: {reg_eval['n_positive']}, "
                f"N_neg: {reg_eval['n_negative']}"
            )
            sys.stdout.flush()

        # Feature importance for this region
        if rname in regional_models:
            imp = regional_models[rname].feature_importances(N_FEAT_TOTAL)
            top_10 = np.argsort(imp)[::-1][:10]
            results["feature_importance"][rname] = {
                ALL_FEATURE_NAMES[i]: float(imp[i]) for i in top_10
            }

        # Accumulate for global combined (only if enough test positives)
        n_test_pos = int(y_test.sum())
        if n_test_pos < MIN_TEST_POSITIVES_FOR_GLOBAL:
            insufficient_data_regions.append(rname)
            if verbose:
                print(
                    f"    ** Excluded from global AUC: only {n_test_pos} "
                    f"test positives (need {MIN_TEST_POSITIVES_FOR_GLOBAL})"
                )
        else:
            all_y_test.append(y_test)
            all_p_regional.append(p_regional)
            all_p_global.append(p_global)

    # Global combined evaluation
    if all_y_test:
        y_combined = np.concatenate(all_y_test)
        p_reg_combined = np.concatenate(all_p_regional)
        p_glb_combined = np.concatenate(all_p_global)

        combined_reg = evaluate(y_combined, p_reg_combined)
        combined_glb = evaluate(y_combined, p_glb_combined)

        results["global_combined"]["regional_ensemble"] = combined_reg
        results["global_combined"]["global_baseline"] = combined_glb

        # Significance test: regional ensemble vs global baseline
        sig_test = paired_bootstrap_test(
            y_combined, p_glb_combined, p_reg_combined,
        )
        results["comparison"] = {
            "regional_ensemble_auc": combined_reg["auc"],
            "global_baseline_auc": combined_glb["auc"],
            "lift": combined_reg["auc"] - combined_glb["auc"],
            "significance": sig_test,
            "excluded_insufficient_data": insufficient_data_regions,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("GLOBAL COMBINED RESULTS")
            print(f"{'=' * 60}")
            print(
                f"  Regional ensemble AUC: {combined_reg['auc']:.4f} "
                f"[{combined_reg['bootstrap_ci']['ci_lo']:.4f}, "
                f"{combined_reg['bootstrap_ci']['ci_hi']:.4f}]"
            )
            print(
                f"  Global baseline AUC:   {combined_glb['auc']:.4f} "
                f"[{combined_glb['bootstrap_ci']['ci_lo']:.4f}, "
                f"{combined_glb['bootstrap_ci']['ci_hi']:.4f}]"
            )
            print(
                f"  Lift: {sig_test['delta_auc']:+.4f} "
                f"[{sig_test['ci_lo']:+.4f}, {sig_test['ci_hi']:+.4f}], "
                f"p={sig_test['p_value']:.4f}"
            )
            print(
                f"  Total test samples: {len(y_combined)} "
                f"({int(y_combined.sum())} pos, "
                f"{int(len(y_combined) - y_combined.sum())} neg)"
            )
            if insufficient_data_regions:
                print(
                    f"  Excluded from global AUC (<{MIN_TEST_POSITIVES_FOR_GLOBAL} "
                    f"test positives): {', '.join(insufficient_data_regions)}"
                )
            sys.stdout.flush()

    return results


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main(
    output_dir: str | Path | None = None,
    verbose: bool = True,
    honest: bool = False,
) -> dict:
    """Run the complete v4 regional earthquake prediction pipeline.

    Steps:
        1. Load USGS catalog (494K events)
        2. Gardner-Knopoff aftershock declustering
        3. Pre-build global catalog arrays
        4. Load GNSS station data
        5. For each of 8 regions:
           a. Build M6+/M5.5+ samples
           b. Extract 55 features (Blocks S+G+C+T)
           c. Temporal split
        6. Train one GBT per region (with early stopping)
        7. Train global baseline model
        8. Evaluate per-region + combined
        9. Compare regional ensemble vs global baseline
        10. Save results

    Parameters
    ----------
    honest : bool
        If True, use strict parameters comparable to CSEP/ETAS:
        M6.0+ only, 100km radius, 90-day window, 5:1 control ratio.
    """
    global _HONEST_MODE
    _HONEST_MODE = honest

    t_start = time.time()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[3]
    if output_dir is None:
        subdir = "earthquake_honest" if honest else "earthquake_definitive"
        output_dir = project_root / "results" / subdir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # In honest mode, clear old checkpoints (parameters changed)
    if honest:
        ckpt_dir = Path("results/earthquake_definitive/_checkpoints_honest")
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*.npz"):
                f.unlink()
            if verbose:
                print("[HONEST] Cleared old honest-mode checkpoints")
                sys.stdout.flush()

    if verbose:
        print("=" * 72)
        print("V4 REGIONAL EARTHQUAKE PREDICTION MODEL")
        if honest:
            print("*** HONEST MODE: M6.0+, 100km, 90 days (comparable to CSEP) ***")
        print("RESEARCH ONLY -- NOT OPERATIONAL")
        print("=" * 72)
        print()
        print("INSIGHT: Regional models outperform global models because")
        print("different tectonic settings have different precursor physics.")
        print()
        print(f"Regions: {len(REGIONS)}")
        for rname, rdef in REGIONS.items():
            print(
                f"  {rname:20s}: {rdef['type']:12s} "
                f"lat={rdef['lat']}, lon={rdef['lon']}, "
                f"radius={rdef['label_radius_km']}km"
            )
        print()
        print(f"Feature architecture:")
        print(f"  Block S: {N_FEAT_S} seismicity features")
        print(f"  Block G: {N_FEAT_G} geodetic/GNSS features")
        print(f"  Block C: {N_FEAT_C} coherence field theory features")
        print(f"  Block T: {N_FEAT_T} tectonic context features")
        print(f"  Total:   {N_FEAT_TOTAL} features per sample")
        print()
        print(f"Temporal splits:")
        print(f"  Train: {TRAIN_START} -- {TRAIN_END}")
        print(f"  Val:   {VAL_START} -- {VAL_END}")
        print(f"  Test:  {TEST_START} -- {TEST_END}")
        print()
        print(f"GBT: {GBT_N_TREES} trees, depth={GBT_MAX_DEPTH}, "
              f"lr={GBT_LEARNING_RATE}, subsample={GBT_SUBSAMPLE}")
        print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
        print(f"Min train samples per region: {MIN_TRAIN_SAMPLES}")
        print()
        sys.stdout.flush()

    # Phase 1-5: Load data, build samples, extract features
    if verbose:
        print("PHASE 1-5: DATA LOADING AND FEATURE EXTRACTION")
        print("-" * 50)
        sys.stdout.flush()

    region_data, global_cat, full_catalog, gnss, data_meta = load_all_data(
        verbose=verbose,
    )

    # Phase 6: Train regional models
    if verbose:
        print(f"\n{'=' * 50}")
        print("PHASE 6: TRAINING REGIONAL MODELS")
        print("-" * 50)
        sys.stdout.flush()

    regional_models, regional_normalizers, train_meta = train_regional_models(
        region_data, verbose=verbose,
    )

    # Phase 7: Train global baseline
    if verbose:
        print(f"\n{'=' * 50}")
        print("PHASE 7: TRAINING GLOBAL BASELINE")
        print("-" * 50)
        sys.stdout.flush()

    global_model, global_normalizer, global_tmeta = train_global_baseline(
        region_data, verbose=verbose,
    )

    # Phase 8-9: Evaluate
    if verbose:
        print(f"\n{'=' * 50}")
        print("PHASE 8-9: EVALUATION")
        print("-" * 50)
        sys.stdout.flush()

    results = evaluate_all(
        region_data,
        regional_models,
        regional_normalizers,
        global_model,
        global_normalizer,
        train_meta,
        verbose=verbose,
    )

    # Add metadata to results
    elapsed = time.time() - t_start
    results["metadata"] = {
        "model_version": "v4_regional",
        "honest_mode": honest,
        "evaluation_parameters": {
            "magnitude_threshold": HONEST_MAGNITUDE_THRESHOLD if honest else "per-region (M6.0 or M5.5)",
            "label_radius_km": HONEST_LABEL_RADIUS_KM if honest else "per-region (150-300)",
            "forward_window_days": HONEST_FORWARD_WINDOW_DAYS if honest else FORWARD_WINDOW_DAYS,
            "control_ratio": HONEST_CONTROL_RATIO if honest else CONTROL_RATIO,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "regions": list(REGIONS.keys()),
        "n_features": N_FEAT_TOTAL,
        "feature_names": ALL_FEATURE_NAMES,
        "gbt_config": {
            "n_trees": GBT_N_TREES,
            "max_depth": GBT_MAX_DEPTH,
            "learning_rate": GBT_LEARNING_RATE,
            "subsample": GBT_SUBSAMPLE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "temporal_splits": {
            "train": f"{TRAIN_START}-{TRAIN_END}",
            "val": f"{VAL_START}-{VAL_END}",
            "test": f"{TEST_START}-{TEST_END}",
        },
        "data_meta": data_meta,
        "train_meta": {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.floating)}
            for k, v in train_meta.items()
        },
        "global_baseline_meta": {
            "n_trees_used": global_tmeta["n_trees_used"],
            "stopped_early": global_tmeta["stopped_early"],
        },
    }

    # Phase 10: Save results
    fname = "v4_regional_honest_results.json" if honest else "v4_regional_results.json"
    output_path = output_dir / fname

    # Make results JSON-serializable
    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        return obj

    serializable_results = _make_serializable(results)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"Results saved to: {output_path}")
        print(f"Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
        print(f"{'=' * 72}")
        sys.stdout.flush()

    return results


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V4 Regional Earthquake Prediction Model (RESEARCH ONLY)",
    )
    parser.add_argument(
        "--honest",
        action="store_true",
        help=(
            "Honest evaluation mode: M6.0+ only, 100km radius, 90-day window, "
            "5:1 control ratio. Comparable to CSEP/ETAS published systems."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    if args.honest:
        print("=" * 72)
        print("HONEST MODE: M6.0+, 100km, 90 days (comparable to CSEP)")
        print("All regions use identical strict parameters.")
        print("=" * 72)
        print()
        sys.stdout.flush()

    main(
        output_dir=args.output_dir,
        verbose=not args.quiet,
        honest=args.honest,
    )
