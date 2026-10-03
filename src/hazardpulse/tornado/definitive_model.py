# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational tornado warning system.
# It does NOT replace official NWS tornado warnings.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Definitive leak-proof tornado prediction model.

Design philosophy: ZERO LEAKAGE BY CONSTRUCTION.
    - ONE model type: Gradient Boosted Trees. No meta-stacker, no blending.
    - Strict temporal separation enforced with assertions.
    - Labels require temporal ordering in code.
    - All features verified causal (available at prediction time).
    - No hyperparameter tuning on test. Test touched ONCE.

Three model variants (one GBT each, NO stacking):
    1. Baseline:  Block P only (13 features) -- ProbSevere raw
    2. Enhanced:  Block P + E + H (31 features) -- + evolution + HRRR
    3. Full:      Block P + E + H + C (41 features) -- + coherence field theory

Temporal splits (hardcoded, non-negotiable):
    Train: 2021-01-01 to 2022-12-31
    Val:   2023-01-01 to 2023-12-31
    Test:  2024-01-01 to 2024-12-31

Data sources:
    - ProbSevere v3 storm objects from AWS (cached .json.gz per day)
    - HRRR 18Z analysis fields (cached .npz per day, 80km grid)
    - SPC tornado reports 1950-2024 (labels only)

Self-contained: copies GBT implementation fresh, imports only from
hazardpulse.data.hrrr, hazardpulse.data.probsevere, and
hazardpulse.tornado.coherence_engine.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Internal imports (the ONLY allowed external dependencies)
# ---------------------------------------------------------------------------

from hazardpulse.data.hrrr import (
    GRID_LATS,
    GRID_LONS,
    HRRR_N_LAT,
    HRRR_N_LON,
    load_cached_hrrr,
    latlon_to_hrrr_cell,
)
from hazardpulse.data.probsevere import (
    load_cached_probsevere,
    scan_probsevere_cache,
)
from hazardpulse.tornado.coherence_engine import (
    compute_coherence_fields,
    extract_coherence_at_point,
)


# ===================================================================
# TEMPORAL SPLITS -- HARDCODED, NON-NEGOTIABLE
# ===================================================================

TRAIN_START: str = "20210101"
TRAIN_END: str = "20221231"
VAL_START: str = "20230101"
VAL_END: str = "20231231"
TEST_START: str = "20240101"
TEST_END: str = "20241231"


# ===================================================================
# FEATURE BLOCK SIZES
# ===================================================================

N_FEAT_P: int = 13   # ProbSevere raw storm-object features
N_FEAT_E: int = 6    # Storm evolution features
N_FEAT_H: int = 12   # HRRR atmospheric context
N_FEAT_C: int = 10   # Coherence field theory

N_FEAT_BASELINE: int = N_FEAT_P                                 # 13
N_FEAT_ENHANCED: int = N_FEAT_P + N_FEAT_E + N_FEAT_H           # 31
N_FEAT_FULL: int = N_FEAT_P + N_FEAT_E + N_FEAT_H + N_FEAT_C   # 41


# ===================================================================
# FEATURE NAMES -- every feature has a causal validity comment
# ===================================================================

# Block P: ProbSevere Raw (13 features)
# All are observed properties of the CURRENT storm at prediction time.
BLOCK_P_NAMES: list[str] = [
    "mucape",        # 1.  MUCAPE at storm (J/kg) -- observed, current storm
    "mlcape",        # 2.  MLCAPE at storm (J/kg) -- observed, current storm
    "mlcin",         # 3.  MLCIN at storm (J/kg, negative) -- observed
    "ebshear",       # 4.  Effective bulk shear (kt) -- observed
    "srh01",         # 5.  0-1km SRH (m2/s2) -- observed
    "mesh",          # 6.  Max estimated hail size (inches) -- observed radar
    "vil_density",   # 7.  VIL density (g/m3) -- observed radar
    "flash_rate",    # 8.  Total lightning flash rate -- observed GLM
    "maxllaz",       # 9.  Max low-level azimuthal shear (1/s) -- observed radar
    "p98llaz",       # 10. 98th pct low-level AzShear (1/s) -- observed radar
    "p98mlaz",       # 11. 98th pct mid-level AzShear (1/s) -- observed radar
    "ps",            # 12. ProbSevere score (0-100, any severe) -- observed
    "size",          # 13. Storm object size (km2) -- observed radar
]

# Block E: Storm Evolution (6 features)
# From tracking the same storm ID across ProbSevere time steps.
# All derived from PAST observations of the same storm.
BLOCK_E_NAMES: list[str] = [
    "storm_age_min",          # 14. Minutes since first appearance -- past
    "maxllaz_trend",          # 15. Current/15-min-ago maxllaz -- past
    "flash_rate_trend",       # 16. Current/15-min-ago flash rate -- past
    "ps_trend",               # 17. Current/15-min-ago ProbSevere score -- past
    "sustained_rotation_min", # 18. Minutes with maxllaz > 0.003 -- past
    "storm_speed_ms",         # 19. Translation speed from MOTION vectors -- past
]

# Block H: HRRR Atmospheric Context (12 features)
# Looked up at storm lat/lon from HRRR 18Z ANALYSIS grid.
# HRRR analysis is an observation product (not forecast), available at T.
BLOCK_H_NAMES: list[str] = [
    "hrrr_cape",      # 20. CAPE at storm location -- HRRR analysis
    "hrrr_cin",       # 21. CIN at storm location -- HRRR analysis
    "hrrr_srh01",     # 22. 0-1km SRH -- HRRR analysis
    "hrrr_srh03",     # 23. 0-3km SRH -- HRRR analysis
    "hrrr_shear06",   # 24. 0-6km bulk shear magnitude -- HRRR analysis
    "hrrr_shear01",   # 25. 0-1km shear magnitude -- HRRR analysis
    "hrrr_pwat",      # 26. Precipitable water -- HRRR analysis
    "hrrr_t2m",       # 27. 2m temperature -- HRRR analysis
    "hrrr_td2m",      # 28. 2m dewpoint -- HRRR analysis
    "hrrr_refc",      # 29. Composite reflectivity -- HRRR analysis
    "hrrr_stp",       # 30. STP effective (computed from HRRR fields)
    "hrrr_srh05_est", # 31. Estimated 0-500m SRH -- derived from HRRR
]

# Block C: Coherence Field Theory (10 features)
# Solved from HRRR atmospheric fields via Helmholtz PDE.
# NOT derived from tornado reports. Purely physical computation.
BLOCK_C_NAMES: list[str] = [
    "tau",              # 32. Coherence amplitude -- Helmholtz PDE on HRRR
    "grad_tau",         # 33. |grad(tau)| -- spatial derivative of PDE solution
    "torsion",          # 34. SRH x curl(tau) -- rotational coherence coupling
    "alignment",        # 35. Shear dot grad(tau) -- Form 16 alignment
    "S_over_Gamma",     # 36. Source/damping ratio -- PDE coefficients
    "Da",               # 37. Damkohler number -- reaction-diffusion regime
    "tau_x_maxllaz",    # 38. tau * maxllaz -- coherence x observed rotation
    "alignment_x_cape", # 39. alignment * CAPE -- coupling x instability
    "torsion_x_srh",    # 40. torsion * SRH -- rotational coupling x helicity
    "singularity_count",# 41. Singularity conditions met (0-5, absolute)
]

ALL_FEATURE_NAMES_BASELINE: list[str] = BLOCK_P_NAMES
ALL_FEATURE_NAMES_ENHANCED: list[str] = BLOCK_P_NAMES + BLOCK_E_NAMES + BLOCK_H_NAMES
ALL_FEATURE_NAMES_FULL: list[str] = (
    BLOCK_P_NAMES + BLOCK_E_NAMES + BLOCK_H_NAMES + BLOCK_C_NAMES
)


# ===================================================================
# GBT HYPERPARAMETERS -- FIXED A PRIORI, NOT TUNED ON VALIDATION
# ===================================================================

GBT_N_TREES: int = 200
GBT_MAX_DEPTH: int = 4
GBT_LEARNING_RATE: float = 0.05
GBT_SUBSAMPLE: float = 0.5
GBT_COLSAMPLE: float = 0.7
GBT_MIN_SAMPLES_LEAF: int = 20
GBT_L2_REG: float = 1.0
GBT_GAMMA: float = 0.1

# Early stopping patience (on validation loss)
EARLY_STOP_PATIENCE: int = 20

# Label parameters
LABEL_RADIUS_KM: float = 40.0
FORWARD_WINDOW_MIN: int = 60

# Negatives kept per positive in the TRAIN split only (GBT tractability).
# Val and test are never downsampled -- see load_all_data.
TRAIN_NEG_RATIO: int = 5

# Rotation threshold for sustained rotation tracking
ROTATION_THRESHOLD: float = 0.003

# Storm-evolution history: past steps searched for the same storm ID. Shared
# by training and live scoring -- the live scorer used to pass the WHOLE
# track, so storm_age_min / sustained_rotation_min were capped in training
# and unbounded live.
STORM_HISTORY_LOOKBACK: int = 8
# Cadence assumed when a day's own stamps cannot be parsed (the fetcher's
# S3 path keeps one file per 30-minute slot).
PROBSEVERE_STEP_MIN_DEFAULT: float = 30.0

# HRRR analysis matching. Each storm reads the LATEST analysis valid at or
# before its own observation time, at most MAX_ANALYSIS_AGE_H old -- never a
# later one. Until 2026-10-01 every storm of a day read the 18Z analysis: the
# 72-84% of samples observed before 18Z saw the atmosphere (and the
# radar-assimilated reflectivity) of up to 18 hours in their FUTURE, while the
# live scorer reads a 1-3 h old analysis. Training now samples the 3-hourly
# analyses so its staleness (0-3 h) matches what live scoring can have.
HRRR_ANALYSIS_HOURS: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21)
MAX_ANALYSIS_AGE_H: float = 3.0


# ===================================================================
# HAVERSINE DISTANCE
# ===================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two points.

    Uses the Haversine formula. Inputs in degrees.
    """
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ===================================================================
# TIME AXIS -- EVERYTHING IS COMPARED AS AN ABSOLUTE UTC INSTANT
# ===================================================================
#
# Two clocks meet in the label, and until 2026-10-01 they were compared as
# if they were one:
#
#   * ProbSevere storm objects are stamped in UTC and filed under their UTC
#     date (the S3 prefix ``ProbSevere/{YYYYMMDD}/`` is a UTC day).
#   * SPC tornado reports are stamped in LOCAL STANDARD TIME. The ``tz``
#     column of the SPC database says which: 3 = CST (UTC-6, never daylight
#     time), 9 = GMT. Every 2021-2024 row is tz=3.
#
# The old code read the SPC hour as a UTC hour, filed each report under its
# CST date, and wrapped negative differences by +24 h. A storm 30 minutes
# before a tornado could therefore be labelled 0, a storm ~6.5 h before it
# could be labelled 1, the evening peak (00-06 UTC = previous CST evening)
# was matched against the wrong day's reports, and the +24 h wrap matched
# tornadoes up to ~23 h in the PAST as if they were in the future.
#
# The fix converts both sides to absolute UTC instants and compares those.

# SPC ``tz`` code -> hours to ADD to the recorded local time to reach UTC.
# Source: SPC Severe Weather Database file description (field "tz").
SPC_TZ_TO_UTC_HOURS: dict[int, int] = {3: 6, 9: 0}

_PROBSEVERE_NATIVE_TIME = re.compile(r"^(\d{8})_(\d{6})\s*UTC$")
_ISO_UTC_TIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?Z?$")


def parse_spc_time_utc(
    yr: int,
    mo: int,
    dy: int,
    time_str: str,
    tz_code: int,
) -> dt.datetime | None:
    """Convert an SPC report's local date/time to an aware UTC datetime.

    Returns ``None`` when the time or the time zone is unknown -- such reports
    are never matched (the label refuses to guess a clock).
    """
    offset = SPC_TZ_TO_UTC_HOURS.get(int(tz_code))
    if offset is None or not time_str or ":" not in time_str:
        return None
    parts = time_str.strip().split(":")
    try:
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        ss = int(parts[2]) if len(parts) > 2 else 0
        local = dt.datetime(int(yr), int(mo), int(dy), hh, mm, ss)
    except (ValueError, IndexError):
        return None
    return (local + dt.timedelta(hours=offset)).replace(tzinfo=dt.timezone.utc)


def parse_probsevere_valid_time(valid_time: str) -> dt.datetime | None:
    """Parse a ProbSevere time-step stamp to an aware UTC datetime.

    Accepts both forms the cache has ever held:

    * ``"20240427_000042 UTC"`` -- the native v3 ``validTime``, stored by the
      fetcher since 2026-03-22;
    * ``"2024-04-27T00:00:00Z"`` -- the ISO stamp the first fetcher
      synthesised from its slot time.

    The old parser keyed on the letter ``T`` and so matched the ``T`` inside
    ``"UTC"``: every native stamp parsed to an unknown hour and every sample
    built from a cache filled after 2026-03-22 was silently excluded.
    Returns ``None`` for anything else.
    """
    s = str(valid_time or "").strip()
    m = _PROBSEVERE_NATIVE_TIME.match(s)
    if m:
        try:
            return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            return None
    m = _ISO_UTC_TIME.match(s)
    if m:
        y, mo, d, hh, mi, ss = m.groups()
        try:
            return dt.datetime(
                int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0),
                tzinfo=dt.timezone.utc,
            )
        except ValueError:
            return None
    return None


def select_analysis_hour(
    storm_time_utc: dt.datetime | None,
    available_hours,
    max_age_h: float = None,
) -> int | None:
    """Latest available analysis hour (same UTC day) at or before the storm.

    Returns ``None`` when no analysis is at most ``max_age_h`` old -- the
    sample then has no causal atmosphere and is excluded, never zero-filled.
    """
    if storm_time_utc is None:
        return None
    if max_age_h is None:
        max_age_h = MAX_ANALYSIS_AGE_H
    t = storm_time_utc.hour + storm_time_utc.minute / 60.0 + storm_time_utc.second / 3600.0
    ok = [h for h in available_hours if h <= t and (t - h) <= max_age_h]
    return max(ok) if ok else None


def reports_in_label_window(
    reports_by_utc_date: dict[str, list[dict]],
    date_str: str,
) -> list[dict]:
    """Reports that can label a storm observed on UTC date ``date_str``.

    The forward window crosses 00 UTC for storms observed after
    ``24h - FORWARD_WINDOW_MIN``, so the next UTC day's reports are included;
    ``compute_label`` compares absolute instants, so including them can never
    match a report outside the window.
    """
    day = dt.datetime.strptime(date_str, "%Y%m%d")
    nxt = (day + dt.timedelta(days=1)).strftime("%Y%m%d")
    return list(reports_by_utc_date.get(date_str, [])) + list(reports_by_utc_date.get(nxt, []))


# ===================================================================
# LABEL COMPUTATION -- STRICT TEMPORAL ORDERING
# ===================================================================

def compute_label(
    storm_lat: float,
    storm_lon: float,
    storm_time_utc: dt.datetime | None,
    candidate_reports: list[dict],
    forward_minutes: int = FORWARD_WINDOW_MIN,
) -> int:
    """Compute binary label: will this storm produce a tornado within forward_minutes?

    STRICT: requires temporal ordering. Tornado must occur AFTER storm observation.
    Samples with unknown storm timing are EXCLUDED (returns -1).

    Parameters
    ----------
    storm_lat, storm_lon : float
        Storm centroid coordinates.
    storm_time_utc : datetime or None
        Aware UTC instant of the storm observation (``None`` if unknown).
    candidate_reports : list[dict]
        SPC tornado reports that may fall in the window -- use
        ``reports_in_label_window``. Each dict carries ``slat``, ``slon`` and
        ``time_utc`` (POSIX seconds, or ``None`` when the report's clock is
        unknown).
    forward_minutes : int
        Maximum lookahead window in minutes.

    Returns
    -------
    int
        1 if a tornado starts within ``LABEL_RADIUS_KM`` and
        ``[0, forward_minutes]`` after the observation, 0 if none does,
        -1 if the sample must be EXCLUDED (unknown storm timing).
    """
    if storm_time_utc is None:
        return -1  # Can't verify temporal ordering without storm time
    t_storm = storm_time_utc.timestamp()
    window_s = forward_minutes * 60.0

    for tor in candidate_reports:
        t_tor = tor.get("time_utc")
        if t_tor is None:
            continue  # Skip tornadoes with unknown timing -- never guess

        # Temporal filter: tornado must start AFTER the observation, within
        # the window -- absolute instants, so no day boundary is special.
        dt_s = float(t_tor) - t_storm
        if not (0.0 <= dt_s <= window_s):
            continue

        # Spatial filter: within LABEL_RADIUS_KM of storm centroid
        dist_km = haversine_km(
            storm_lat, storm_lon,
            tor.get("slat", 0.0), tor.get("slon", 0.0),
        )
        if dist_km <= LABEL_RADIUS_KM:
            return 1

    return 0


# ===================================================================
# SIGMOID (numerically stable)
# ===================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Uses the two-branch trick to avoid overflow in exp().
    """
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -88.0, 88.0)  # float32 safe: exp(88) ~ 1.6e38 < FLT_MAX
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    ).astype(np.float32)


# ===================================================================
# GRADIENT BOOSTED TREES -- SELF-CONTAINED IMPLEMENTATION
# ===================================================================

class GradientBoostedTrees:
    """Memory-efficient gradient boosted trees for binary classification.

    Uses log-loss (cross-entropy) with Newton-Raphson leaf values.
    Supports subsample, colsample, balanced class weights, L2 regularization,
    and minimum complexity gain (gamma) pruning.

    Feature importance is via split counts (honest about limitations:
    this is a frequency-based proxy, not a causal importance measure).
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
        """Build a single regression tree on gradient/hessian targets.

        Uses exact greedy split finding with candidate thresholds from
        percentiles for continuous features.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix for this subsample.
        gradients : ndarray, shape (n_samples,)
            First-order gradients (g_i = p_i - y_i * w_i).
        hessians : ndarray, shape (n_samples,)
            Second-order hessians (h_i = p_i * (1 - p_i) * w_i).
        depth : int
            Current depth in the tree.

        Returns
        -------
        dict
            Tree node: either ``{"leaf": True, "val": float}`` or
            ``{"leaf": False, "feat": int, "thresh": float,
              "left": dict, "right": dict}``.
        """
        N = len(gradients)
        G = float(np.sum(gradients))
        H = float(np.sum(hessians))

        # Newton-Raphson optimal leaf value: -G / (H + lambda)
        leaf_val = -G / (H + self.l2_reg) if (H + self.l2_reg) > 1e-12 else 0.0

        # Stopping conditions
        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {"leaf": True, "val": float(leaf_val)}

        D = X.shape[1]
        n_try = max(5, int(D * self.colsample))
        rng = self._rng if self._rng is not None else np.random.RandomState(0)
        feat_idx = rng.choice(D, size=min(n_try, D), replace=False)

        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0

        # Score of current node (before splitting)
        score_no_split = G ** 2 / (H + self.l2_reg)

        for f in feat_idx:
            col = X[:, f]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue

            # Candidate thresholds: percentiles for efficiency. Thresholds are
            # float32 VALUES, the dtype of the data they split: every later
            # comparison (training partition, batch and row prediction, the
            # live payload walk) is then exact, whatever the scalar promotion
            # rules -- a float64 percentile compared against float32 data
            # rounds differently in different code paths.
            if len(unique_vals) > 20:
                thresholds = np.percentile(col, np.linspace(5, 95, 20)).astype(np.float32)
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

                # XGBoost-style gain:
                # Gain = 0.5 * [GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam)] - gamma
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
        """Predict a single row through a single tree. No array copies."""
        while not node["leaf"]:
            if x[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["val"]

    def _predict_tree_batch(self, node: dict, X: np.ndarray) -> np.ndarray:
        """Predict all rows through a single tree, one mask per node.

        Routes index sets, not rows: the cost is O(N * depth) NumPy work
        instead of O(N * depth) Python steps. Bit-identical to walking each
        row with ``_predict_tree_row`` (same ``<=`` comparison, same float32
        leaf values).
        """
        result = np.empty(X.shape[0], dtype=np.float32)
        stack: list[tuple[dict, np.ndarray]] = [(node, np.arange(X.shape[0]))]
        while stack:
            nd, idx = stack.pop()
            if idx.size == 0:
                continue
            if nd["leaf"]:
                result[idx] = nd["val"]
                continue
            go_left = X[idx, nd["feat"]] <= nd["thresh"]
            stack.append((nd["left"], idx[go_left]))
            stack.append((nd["right"], idx[~go_left]))
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
        """Train the GBT ensemble on labelled data.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_features)
            Training feature matrix.
        y : ndarray, shape (n_train,)
            Binary labels (0 or 1).
        X_val : ndarray, optional
            Validation feature matrix for early stopping.
        y_val : ndarray, optional
            Validation labels for early stopping.
        verbose : bool
            Print progress every 25 trees.

        Returns
        -------
        dict
            Training metadata: n_trees_used, stopped_early, best_val_loss.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = len(y)

        # Balanced class weights: w_pos = N / (2 * n_pos), w_neg = N / (2 * n_neg)
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
            # Score validation with the SAME class-balanced objective the trees
            # minimise. The validation split keeps its real base rate, while
            # the balanced weights make F a log-odds at a 50/50 prior; an
            # unweighted val loss would mostly measure that prior gap and stop
            # the ensemble for calibration reasons, not ranking ones.
            # Calibration to the real base rate is a separate, later step.
            nv = float(len(y_val))
            nv_pos = float(y_val.sum())
            nv_neg = nv - nv_pos
            assert nv_pos > 0 and nv_neg > 0, "validation split needs both classes"
            val_w = np.where(y_val == 1, nv / (2.0 * nv_pos), nv / (2.0 * nv_neg))

        rng = np.random.RandomState(42)
        self._rng = rng
        self.trees = []

        for t in range(self.n_trees):
            # Compute probabilities
            p_train = sigmoid(F_train)

            # Gradients and hessians (weighted)
            # g_i = (p_i - y_i) * w_i    (first derivative of weighted log-loss)
            # h_i = p_i * (1 - p_i) * w_i (second derivative)
            gradients = ((p_train - y) * sample_w).astype(np.float32)
            hessians = (p_train * (1.0 - p_train) * sample_w + 1e-6).astype(
                np.float32
            )

            # Subsample rows
            if self.subsample < 1.0:
                n_sub = max(1, int(N * self.subsample))
                idx = rng.choice(N, size=n_sub, replace=False)
                tree = self._build_tree(
                    X[idx], gradients[idx], hessians[idx]
                )
            else:
                tree = self._build_tree(X, gradients, hessians)

            self.trees.append(tree)

            # Update training predictions
            F_train += self.lr * self._predict_tree_batch(tree, X)

            # Early stopping check on validation
            if use_early_stop:
                F_val += self.lr * self._predict_tree_batch(tree, X_val)
                # Class-balanced log-loss on validation (the training objective)
                p_val = sigmoid(F_val).astype(np.float64)
                val_loss = -float(np.mean(val_w * (
                    y_val * np.log(p_val + 1e-12)
                    + (1 - y_val) * np.log(1 - p_val + 1e-12)
                )))
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
                    # Trim to best number of trees
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

        if use_early_stop and 0 < best_n_trees < len(self.trees):
            # Patience never ran out, but the best validation loss was still
            # reached before the last tree: keep the ensemble that earned it.
            self.trees = self.trees[:best_n_trees]
        return {
            "n_trees_used": len(self.trees),
            "stopped_early": False,
            "best_val_loss": best_val_loss if use_early_stop else None,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data.

        Row-by-row prediction through each tree to avoid large array copies.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted probabilities in [0, 1].
        """
        return sigmoid(self.decision_function(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw additive score F(x) (log-odds under the training weighting)."""
        X = np.asarray(X, dtype=np.float32)
        F = np.full(X.shape[0], self.init_pred, dtype=np.float32)
        for tree in self.trees:
            F += np.float32(self.lr) * self._predict_tree_batch(tree, X)
        return F

    def feature_importances(self, n_features: int) -> np.ndarray:
        """Compute feature importance from tree split counts.

        NOTE: This is a frequency-based proxy for importance, not a causal
        measure. A feature that splits often may not be the most predictive;
        it may simply have the most variance. Use with appropriate caveats.

        Parameters
        ----------
        n_features : int
            Total number of features.

        Returns
        -------
        ndarray, shape (n_features,)
            Normalized split frequency per feature.
        """
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

def _midranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of ``x`` with tied values sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # Boundaries of runs of equal values in sorted order
    new_run = np.concatenate(([True], xs[1:] != xs[:-1]))
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    run_end = np.concatenate((run_start[1:], [len(xs)]))
    avg_rank = (run_start + run_end + 1) / 2.0  # mean of 1-based ranks in the run
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = avg_rank[run_id]
    return ranks


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC as the Mann-Whitney statistic, with ties scored one half.

    ``P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)``.

    The previous implementation walked a ``np.argsort`` order one sample at
    a time, so within a block of tied scores it credited positives and
    negatives in whatever order the (unstable) sort happened to leave them.
    Tree ensembles emit many exactly-tied scores, so that order -- not the
    model -- moved the reported AUC; on a fully tied scorer it could return
    anything from 0 to 1 instead of 0.5. It was also an O(N) Python loop,
    which made a 2,000-draw bootstrap over a full test population
    impractical.

    Handles edge cases (no positives, no negatives) by returning 0.5.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) < 2:
        return 0.5
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _midranks(y_score)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC -- better metric for imbalanced data.

    Sorts by decreasing score, computes precision and recall at each
    threshold, integrates using the trapezoidal rule.
    """
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

        # Trapezoidal integration: area under PR curve
        pr_auc += (recall - recall_prev) * precision
        recall_prev = recall

    return float(pr_auc)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score: mean squared error between predicted proba and labels.

    Lower is better. Range [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_bss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier Skill Score: improvement over climatological baseline.

    BSS = 1 - Brier / Brier_clim, where Brier_clim uses the base rate.
    BSS = 1 means perfect, BSS = 0 means no better than climatology.
    """
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
    """Reliability diagram: predicted vs observed frequency per bin.

    Parameters
    ----------
    y_true : ndarray
        Binary labels.
    y_pred : ndarray
        Predicted probabilities.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        Keys: ``predicted`` (bin mean pred), ``observed`` (bin mean true),
        ``counts`` (samples per bin).
    """
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
            # Include right edge in last bin
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
    """Bootstrap confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true : ndarray
        Binary labels.
    y_score : ndarray
        Predicted probabilities.
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level (0.05 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``mean``, ``ci_lo``, ``ci_hi``, ``std``.
    """
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


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict:
    """Complete evaluation suite for a single model.

    Parameters
    ----------
    y_true : ndarray
        Binary labels.
    y_pred : ndarray
        Predicted probabilities. Brier, BSS and reliability are only
        meaningful if these are calibrated to the population being scored.
    groups : ndarray, optional
        Cluster label per sample (UTC day). When given, the AUC interval is
        a day-clustered bootstrap; otherwise an i.i.d. sample bootstrap.

    Returns
    -------
    dict
        All metrics: AUC, PR-AUC, Brier, BSS, calibration, bootstrap CI,
        sample counts, and base rate.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if groups is not None:
        ci = cluster_bootstrap_auc_ci(y_true, y_pred, groups, n_boot=2000)
    else:
        ci = bootstrap_auc_ci(y_true, y_pred, n_boot=2000)
        ci["unit"] = "sample"
    return {
        "auc": compute_auc(y_true, y_pred),
        "pr_auc": compute_pr_auc(y_true, y_pred),
        "brier": compute_brier(y_true, y_pred),
        "bss": compute_bss(y_true, y_pred),
        "calibration": compute_calibration_curve(y_true, y_pred, n_bins=10),
        "bootstrap_ci": ci,
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "base_rate": float(y_true.mean()),
        "mean_forecast": float(y_pred.mean()),
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
    """Test whether model B significantly outperforms model A.

    Uses paired bootstrap: the SAME resampled indices are used for both
    models on each iteration, so the comparison controls for sampling
    variability.

    Parameters
    ----------
    y_true : ndarray
        Binary labels.
    y_pred_a : ndarray
        Predictions from model A (baseline).
    y_pred_b : ndarray
        Predictions from model B (challenger).
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``delta_auc`` (B - A on full data), ``ci_lo``, ``ci_hi``,
        ``p_value`` (fraction of bootstraps where A >= B).
    """
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
# DAY-CLUSTERED BOOTSTRAP
# ===================================================================
#
# A storm object is observed every 15-30 minutes for as long as it lives, and
# one day's storms share one environment, so samples within a day are far
# from independent. Resampling individual samples treats a 48-step track as
# 48 independent draws and makes every interval too narrow. The honest unit
# is the day: resample whole days with replacement.

def _cluster_index(groups: np.ndarray) -> list[np.ndarray]:
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    g_sorted = groups[order]
    cut = np.flatnonzero(np.concatenate(([True], g_sorted[1:] != g_sorted[:-1], [True])))
    return [order[cut[k]:cut[k + 1]] for k in range(len(cut) - 1)]


def cluster_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for ROC-AUC resampling whole clusters (days)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    clusters = _cluster_index(groups)
    k = len(clusters)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.randint(0, k, size=k)
        idx = np.concatenate([clusters[c] for c in pick])
        aucs[b] = compute_auc(y_true[idx], y_score[idx])
    return {
        "mean": float(np.mean(aucs)),
        "ci_lo": float(np.percentile(aucs, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
        "std": float(np.std(aucs)),
        "unit": "day",
        "n_clusters": k,
    }


def paired_cluster_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """``paired_bootstrap_test`` with whole days as the resampling unit."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    a = np.asarray(y_pred_a, dtype=np.float64)
    b_ = np.asarray(y_pred_b, dtype=np.float64)
    clusters = _cluster_index(groups)
    k = len(clusters)
    delta_full = compute_auc(y_true, b_) - compute_auc(y_true, a)
    deltas = np.empty(n_boot, dtype=np.float64)
    n_a_wins = 0
    for i in range(n_boot):
        pick = rng.randint(0, k, size=k)
        idx = np.concatenate([clusters[c] for c in pick])
        d = compute_auc(y_true[idx], b_[idx]) - compute_auc(y_true[idx], a[idx])
        deltas[i] = d
        if d <= 0:
            n_a_wins += 1
    return {
        "delta_auc": float(delta_full),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_value": float(n_a_wins / n_boot),
        "unit": "day",
        "n_clusters": k,
    }


# ===================================================================
# PROBABILITY CALIBRATION -- FITTED ON VALIDATION, NEVER ON TEST
# ===================================================================
#
# The GBT trains with class-balanced weights on a 5:1 downsampled train
# split, so its sigmoid(F) is a probability at a 50/50 prior: on a
# population whose base rate is ~1-2% it overstates the odds by roughly
# (1 - base_rate) / base_rate, i.e. ~50-100x. The live scorer served that
# number as "tornado probability". Platt scaling, p = sigmoid(a F + b),
# fitted by maximum likelihood on the FULL validation population, maps F
# to the real base rate (b absorbs the prior shift, a any over- or
# under-confidence). It is monotone, so AUC is untouched.

class PlattNotConverged(ValueError):
    """The Platt maximum-likelihood fit did not converge; no calibration is returned."""


def _platt_loss(F: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    z = a * F + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_platt(F: np.ndarray, y: np.ndarray, max_iter: int = 200) -> dict:
    """Maximum-likelihood fit of ``p = sigmoid(a * F + b)``: Newton's method with a backtracking
    line search on the (convex) log-loss.

    The full Newton step is taken whenever it does not increase the loss -- the iterates are then
    exactly plain Newton's -- and halved until it does otherwise. Plain Newton DIVERGED on the rare
    EF2+ label (leave-one-year-out scores, 2026-10-03: a = 3.1e16, b = -1.1e16, so every
    probability was 0 or 1 and the 2025 Brier skill was -57.7); its 60/30/90-min fits converged.
    A fit that does not converge raises ``PlattNotConverged`` instead of returning a calibration.
    """
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("Platt calibration needs both classes")
    a, b = 1.0, float(np.log(y.mean() / (1.0 - y.mean())) - np.mean(F))
    ridge = 1e-9
    cur = _platt_loss(F, y, a, b)
    converged = False
    for _ in range(max_iter):
        z = np.clip(a * F + b, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        r = p - y
        g = np.array([np.sum(r * F), np.sum(r)])
        h = np.array([
            [np.sum(w * F * F) + ridge, np.sum(w * F)],
            [np.sum(w * F), np.sum(w) + ridge],
        ])
        step = np.linalg.solve(h, g)
        t = 1.0
        while True:
            na, nb = a - t * step[0], b - t * step[1]
            new = _platt_loss(F, y, na, nb)
            # rounding-level increases near the optimum are accepted, so a converging fit keeps
            # plain Newton's iterates bit for bit
            if np.isfinite(new) and new <= cur + 1e-12 * max(1.0, abs(cur)):
                break
            t *= 0.5
            if t < 1e-12:
                raise PlattNotConverged(f"no descent along the Newton direction at a={a!r}, b={b!r}")
        a, b, cur = na, nb, new
        if np.max(np.abs(t * step)) < 1e-10:
            converged = True
            break
    if not converged:
        raise PlattNotConverged(f"Platt fit did not converge in {max_iter} Newton steps "
                                f"(a={a!r}, b={b!r}; separable scores have no finite maximum)")
    return {"method": "platt", "a": float(a), "b": float(b)}


def apply_calibration(F: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Map raw additive scores to probabilities (identity sigmoid if None)."""
    F = np.asarray(F, dtype=np.float64)
    if calibration is None:
        z = F
    elif calibration.get("method") == "platt":
        z = calibration["a"] * F + calibration["b"]
    else:
        raise ValueError(f"unknown calibration method: {calibration.get('method')!r}")
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


LEGACY_MODEL_VERSION: str = "tornado_storm_v1_0"


def model_version_of_payload(path: str | Path) -> str:
    """The served model's IDENTITY, bound to its exact weights.

    Live forecasts, the prospective calibration dataset and the trust-layer
    calibrator all carry this string, and a calibrator is applied only to
    forecasts from the model it was fitted on. Until 2026-10-01 the identity
    was the constant ``"tornado_storm_v1_0"`` whatever the weights were, so a
    retrained model would have been "calibrated" by its predecessor's curve.
    Uncalibrated v1 payloads keep the legacy name (their calibrator stays
    valid); a calibrated payload is ``"tornado_gbt_v2-<sha256[:12]>"``.
    """
    import hashlib

    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("calibration") is None:
        return LEGACY_MODEL_VERSION
    return "tornado_gbt_v2-" + hashlib.sha256(raw).hexdigest()[:12]


def predict_proba_from_payload(payload: dict, raw_features: dict[str, float]) -> tuple[float, bool]:
    """Score one storm from a saved model payload -- the ONLY live scoring path.

    Normalises with the payload's stored train statistics, walks the trees,
    and applies the payload's calibration. Returns ``(probability,
    calibrated)``; ``calibrated`` is False for legacy payloads saved before
    calibration existed, whose probability is at the 50/50 training prior and
    must not be presented as a real-world probability.
    """
    names = payload["feature_names"]
    means = payload["normalization"]["means"]
    stds = payload["normalization"]["stds"]
    # Normalise in float32, element for element what FeatureNormalizer did to
    # the training matrix. The old live path normalised in float64, so a value
    # sitting exactly on a split threshold (e.g. hrrr_refc = -10 dBZ, the modal
    # "no echo" value) could route the other way than in training: 287 of 761
    # live storm-steps differed by > 1e-3, with risk-band flips.
    f32 = np.float32
    x = [
        float((f32(raw_features.get(n, 0.0)) - f32(means[i])) / f32(stds[i]))
        for i, n in enumerate(names)
    ]
    F = float(payload["init_pred"])
    lr = float(payload["learning_rate"])
    for tree in payload["trees"]:
        node = tree
        while not node.get("leaf", False):
            node = node["left"] if x[node["feat"]] <= node["thresh"] else node["right"]
        F += lr * float(node["val"])
    cal = payload.get("calibration")
    return float(apply_calibration(np.array([F]), cal)[0]), cal is not None


# ===================================================================
# SPC TORNADO REPORT LOADING
# ===================================================================

def load_spc_tornado_reports(
    spc_csv_path: Path,
) -> dict[str, list[dict]]:
    """Load SPC tornado reports from CSV, indexed by date string.

    Expected CSV format: SPC 1950-2024 actual tornadoes CSV with columns:
        om, yr, mo, dy, date, time, tz, st, stf, stn, mag, inj, fat, loss,
        closs, slat, slon, elat, elon, len, wid

    Parameters
    ----------
    spc_csv_path : Path
        Path to the SPC tornado CSV file.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from **UTC** date string (``YYYYMMDD``) to list of tornado
        report dicts with keys: ``slat``, ``slon``, ``mag``, ``time_utc``
        (POSIX seconds of the UTC start instant, or ``None`` if the report's
        time or time zone is unknown), ``hour`` (fractional **UTC** hour, -1
        if unknown) and ``local_date`` (the SPC row's own CST date).

        Reports with an unknown clock are filed under their local date and
        can never be matched by ``compute_label``.
    """
    import csv

    reports: dict[str, list[dict]] = {}

    if not spc_csv_path.exists():
        print(f"WARNING: SPC tornado report file not found: {spc_csv_path}")
        return reports

    with open(spc_csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = None
        for row in reader:
            if header is None:
                header = [col.strip().lower() for col in row]
                continue
            if len(row) < len(header):
                continue

            rec = dict(zip(header, row))

            try:
                yr = int(rec.get("yr", 0))
                mo = int(rec.get("mo", 0))
                dy = int(rec.get("dy", 0))
                slat = float(rec.get("slat", 0))
                slon = float(rec.get("slon", 0))
                mag = int(rec.get("mag", -1))
            except (ValueError, TypeError):
                continue

            # Skip obviously bad coordinates
            if abs(slat) < 1.0 or abs(slon) < 1.0:
                continue

            try:
                tz_code = int(rec.get("tz", -1))
            except (ValueError, TypeError):
                tz_code = -1
            t_utc = parse_spc_time_utc(yr, mo, dy, rec.get("time", ""), tz_code)
            local_date = f"{yr:04d}{mo:02d}{dy:02d}"
            if t_utc is None:
                key = local_date
                hour = -1.0
                time_utc = None
            else:
                key = t_utc.strftime("%Y%m%d")
                hour = t_utc.hour + t_utc.minute / 60.0
                time_utc = t_utc.timestamp()

            reports.setdefault(key, []).append({
                "slat": slat,
                "slon": slon,
                "hour": hour,
                "time_utc": time_utc,
                "mag": mag,
                "local_date": local_date,
            })

    return reports


# ===================================================================
# FEATURE EXTRACTION
# ===================================================================

def extract_block_p(storm: dict) -> np.ndarray:
    """Extract Block P: ProbSevere raw storm-object features (13).

    All 13 features are observed properties of the current storm object,
    directly from ProbSevere v3 JSON. Available at prediction time by
    definition.
    """
    feat = np.zeros(N_FEAT_P, dtype=np.float32)
    keys = [
        "mucape", "mlcape", "mlcin", "ebshear", "srh01",
        "mesh", "vil_density", "flash_rate",
        "maxllaz", "p98llaz", "p98mlaz", "ps", "size",
    ]
    assert len(keys) == N_FEAT_P, f"Block P expects {N_FEAT_P} features, got {len(keys)}"
    for idx, key in enumerate(keys):
        feat[idx] = np.float32(float(storm.get(key, 0)))
    return feat


def probsevere_step_minutes(time_steps: list[dict]) -> float:
    """Median spacing, in minutes, between consecutive ProbSevere time steps.

    The fetcher keeps one file per 30-minute slot when the S3 listing works
    and probes 15-minute slots when it does not, so the cadence is a property
    of each cached day, not a constant. Falls back to
    ``PROBSEVERE_STEP_MIN_DEFAULT`` when fewer than two stamps parse.
    """
    stamps = [parse_probsevere_valid_time(ts.get("valid_time", "")) for ts in time_steps]
    secs = sorted(s.timestamp() for s in stamps if s is not None)
    if len(secs) < 2:
        return PROBSEVERE_STEP_MIN_DEFAULT
    gaps = np.diff(np.asarray(secs)) / 60.0
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else PROBSEVERE_STEP_MIN_DEFAULT


def extract_block_e(
    storm: dict,
    storm_history: list[dict],
    step_minutes: float = None,
) -> np.ndarray:
    """Extract Block E: storm evolution features (6).

    All 6 features are derived from PAST observations of the same storm ID
    tracked across ProbSevere time steps. Available at prediction time
    because they use only historical data.

    ``storm_history`` must come from ``build_storm_history`` (at most
    ``STORM_HISTORY_LOOKBACK`` + 1 steps) in training AND in live scoring,
    and ``step_minutes`` from ``probsevere_step_minutes`` on the same day's
    steps. The "trend" features compare the current step with the previous
    one, whatever the cadence.
    """
    if step_minutes is None:
        step_minutes = PROBSEVERE_STEP_MIN_DEFAULT
    feat = np.zeros(N_FEAT_E, dtype=np.float32)
    n_steps = len(storm_history)

    # 14. storm_age_min: minutes since first appearance (within the lookback)
    feat[0] = np.float32(max(0, (n_steps - 1)) * step_minutes)

    if n_steps >= 2:
        current = storm_history[-1]
        prev = storm_history[-2]

        # 15. maxllaz_trend: current / 15-min-ago (rotation acceleration)
        prev_llaz = float(prev.get("maxllaz", 0))
        curr_llaz = float(current.get("maxllaz", 0))
        feat[1] = np.float32(curr_llaz / (prev_llaz + 1e-6) if prev_llaz > 1e-6 else 1.0)

        # 16. flash_rate_trend: current / 15-min-ago (lightning jump proxy)
        prev_fr = float(prev.get("flash_rate", 0))
        curr_fr = float(current.get("flash_rate", 0))
        feat[2] = np.float32(curr_fr / (prev_fr + 1e-6) if prev_fr > 1e-6 else 1.0)

        # 17. ps_trend: current / 15-min-ago ProbSevere score
        prev_ps = float(prev.get("ps", 0))
        curr_ps = float(current.get("ps", 0))
        feat[3] = np.float32(curr_ps / (prev_ps + 1e-6) if prev_ps > 1e-6 else 1.0)

        # 18. sustained_rotation_min: count of timesteps with maxllaz > threshold
        rot_count = sum(
            1 for s in storm_history
            if float(s.get("maxllaz", 0)) > ROTATION_THRESHOLD
        )
        feat[4] = np.float32(rot_count * step_minutes)
    else:
        feat[1] = np.float32(1.0)  # No trend data: neutral
        feat[2] = np.float32(1.0)
        feat[3] = np.float32(1.0)
        feat[4] = np.float32(0.0)

    # 19. storm_speed_ms: from MOTION_EAST and MOTION_SOUTH vectors
    me = float(storm.get("motion_east", 0))
    ms = float(storm.get("motion_south", 0))
    feat[5] = np.float32(math.sqrt(me ** 2 + ms ** 2))

    return feat


def extract_block_h(
    storm: dict,
    hrrr_grids: dict[str, np.ndarray],
    derived_hrrr: dict[str, np.ndarray],
) -> np.ndarray:
    """Extract Block H: HRRR atmospheric context features (12).

    All 12 features are looked up at the storm's lat/lon from the HRRR
    18Z analysis grid. HRRR analysis is an observation product (not a
    forecast), so all values are available at prediction time.

    Parameters
    ----------
    storm : dict
        Storm object with ``lat`` and ``lon`` keys.
    hrrr_grids : dict
        Raw HRRR variable grids.
    derived_hrrr : dict
        Derived HRRR parameters from ``compute_derived_hrrr``.
    """
    feat = np.zeros(N_FEAT_H, dtype=np.float32)
    lat = float(storm.get("lat", 0))
    lon = float(storm.get("lon", 0))
    i, j = latlon_to_hrrr_cell(lat, lon)

    def _get(grids: dict, key: str) -> float:
        arr = grids.get(key)
        if arr is not None and i < arr.shape[0] and j < arr.shape[1]:
            return float(arr[i, j])
        return 0.0

    feat[0] = np.float32(_get(hrrr_grids, "mlcape"))         # 20. hrrr_cape
    feat[1] = np.float32(_get(hrrr_grids, "mlcin"))           # 21. hrrr_cin
    feat[2] = np.float32(_get(hrrr_grids, "srh_01"))          # 22. hrrr_srh01
    feat[3] = np.float32(_get(hrrr_grids, "srh_03"))          # 23. hrrr_srh03
    feat[4] = np.float32(_get(derived_hrrr, "shear_06"))      # 24. hrrr_shear06
    feat[5] = np.float32(_get(derived_hrrr, "shear_01"))      # 25. hrrr_shear01
    feat[6] = np.float32(_get(hrrr_grids, "pwat"))            # 26. hrrr_pwat
    feat[7] = np.float32(_get(hrrr_grids, "t2m"))             # 27. hrrr_t2m
    feat[8] = np.float32(_get(hrrr_grids, "td2m"))            # 28. hrrr_td2m
    feat[9] = np.float32(_get(hrrr_grids, "refc"))            # 29. hrrr_refc
    feat[10] = np.float32(_get(derived_hrrr, "stp_eff"))      # 30. hrrr_stp
    feat[11] = np.float32(_get(derived_hrrr, "srh_05_est"))   # 31. hrrr_srh05_est

    return feat


def extract_block_c(
    storm: dict,
    coherence_fields: dict[str, np.ndarray],
    hrrr_grids: dict[str, np.ndarray],
) -> np.ndarray:
    """Extract Block C: coherence field theory features (10).

    All 10 features are computed from HRRR atmospheric fields via the
    Helmholtz PDE solver. They are NOT derived from tornado reports.
    The coherence field is a purely physical computation on the
    atmospheric state, available at prediction time.

    Parameters
    ----------
    storm : dict
        Storm object with ``lat``, ``lon``, ``maxllaz``, ``srh01``,
        ``mucape`` keys.
    coherence_fields : dict
        Output of ``compute_coherence_fields()`` on the HRRR grid.
    hrrr_grids : dict
        Raw HRRR variable grids (for CAPE lookup in interactions).
    """
    feat = np.zeros(N_FEAT_C, dtype=np.float32)
    lat = float(storm.get("lat", 0))
    lon = float(storm.get("lon", 0))

    # Extract coherence values at the storm location
    coh = extract_coherence_at_point(coherence_fields, lat, lon)

    tau = coh.get("tau", 0.0)
    grad_tau = coh.get("grad_tau", 0.0)
    torsion = coh.get("torsion", 0.0)
    alignment = coh.get("alignment", 0.0)
    s_over_gamma = coh.get("S_over_Gamma", 0.0)
    da = coh.get("Da", 0.0)
    sing = coh.get("singularity_count", 0.0)

    # Observed storm properties for interaction terms
    maxllaz = float(storm.get("maxllaz", 0))
    srh01 = float(storm.get("srh01", 0))
    mucape = float(storm.get("mucape", 0))

    feat[0] = np.float32(tau)                              # 32. tau
    feat[1] = np.float32(grad_tau)                         # 33. grad_tau
    feat[2] = np.float32(torsion)                          # 34. torsion
    feat[3] = np.float32(alignment)                        # 35. alignment
    feat[4] = np.float32(s_over_gamma)                     # 36. S_over_Gamma
    feat[5] = np.float32(da)                               # 37. Da
    feat[6] = np.float32(tau * maxllaz)                    # 38. tau_x_maxllaz
    feat[7] = np.float32(alignment * mucape / 1000.0)      # 39. alignment_x_cape
    feat[8] = np.float32(torsion * srh01 / 100.0)          # 40. torsion_x_srh
    feat[9] = np.float32(sing)                             # 41. singularity_count

    return feat


# ===================================================================
# FEATURE NORMALIZATION
# ===================================================================

class FeatureNormalizer:
    """Z-score normalizer. Fit on train, apply to val/test.

    Prevents any information leakage from val/test into normalization
    statistics.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Compute mean and std from training data only.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training feature matrix.
        """
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0, dtype=np.float32)
        self.std = X.std(axis=0, dtype=np.float32)
        # Prevent division by zero for constant features
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization using stored train statistics.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix to normalize.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Normalized features (float32).
        """
        assert self.mean is not None, "FeatureNormalizer has not been fit"
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (for training data only)."""
        self.fit(X)
        return self.transform(X)


# ===================================================================
# SAMPLE BUILDING -- STRICT TEMPORAL LABELING
# ===================================================================

def date_in_range(date_str: str, start: str, end: str) -> bool:
    """Check if a date string (YYYYMMDD) falls within [start, end] inclusive."""
    return start <= date_str <= end


def index_storms_by_id(time_steps: list[dict]) -> list[dict]:
    """Per-step ``{storm_id: storm}`` maps, for O(1) history lookups.

    The first storm with a given ID in a step wins, matching the linear
    scan in ``build_storm_history``.
    """
    out: list[dict] = []
    for ts in time_steps:
        m: dict = {}
        for s in ts.get("storms", []):
            m.setdefault(s.get("id"), s)
        out.append(m)
    return out


def build_storm_history(
    time_steps: list[dict],
    storm_id: int | str,
    current_step_idx: int,
    max_lookback: int = STORM_HISTORY_LOOKBACK,
    *,
    id_index: list[dict] | None = None,
) -> list[dict]:
    """Collect past observations of the same storm ID.

    Looks backward through time steps to build the storm's history
    for evolution feature extraction.

    Parameters
    ----------
    time_steps : list[dict]
        All ProbSevere time steps for the day.
    storm_id : int or str
        The storm ID to track.
    current_step_idx : int
        Index of the current time step.
    max_lookback : int
        Maximum number of past time steps to search.

    Returns
    -------
    list[dict]
        Chronologically ordered list of storm observations, ending with
        the current one.
    """
    history: list[dict] = []
    start_idx = max(0, current_step_idx - max_lookback)
    for idx in range(start_idx, current_step_idx + 1):
        if id_index is not None:
            s = id_index[idx].get(storm_id)
            if s is not None:
                history.append(s)
            continue
        ts = time_steps[idx]
        for s in ts.get("storms", []):
            if s.get("id") == storm_id:
                history.append(s)
                break
    return history


def build_samples_for_date(
    date_str: str,
    time_steps: list[dict],
    tornado_reports: list[dict],
    analyses: dict[int, tuple[dict, dict, dict | None]],
    include_coherence: bool = True,
    *,
    analysis_policy: str = "latest_at_or_before",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build feature/label samples from one day of ProbSevere data.

    Parameters
    ----------
    date_str : str
        UTC date in YYYYMMDD format (the ProbSevere S3 day).
    time_steps : list[dict]
        ProbSevere time steps for this date.
    tornado_reports : list[dict]
        SPC tornado reports that can fall in any of this day's label windows:
        ``reports_in_label_window(reports_by_utc_date, date_str)``.
    analyses : dict[int, (hrrr_grids, derived_hrrr, coherence_fields)]
        HRRR analyses of this UTC day keyed by analysis hour. Each storm reads
        the one chosen by ``analysis_policy``.
    include_coherence : bool
        Whether to include Block C features.
    analysis_policy : str
        ``"latest_at_or_before"`` (default): the latest analysis at most
        ``MAX_ANALYSIS_AGE_H`` before the observation; storms with none are
        EXCLUDED, never zero-filled. ``"fixed_18z"``: every storm reads the
        18Z analysis -- the pre-2026-10-01 behaviour, which leaks up to 18 h
        of future atmosphere; kept ONLY to measure that leak, never a default.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels.
    n_excluded : int
        Samples excluded due to unknown timing.
    n_no_analysis : int
        Samples excluded because no causal HRRR analysis was available.
    """
    if analysis_policy not in ("latest_at_or_before", "fixed_18z"):
        raise ValueError(f"unknown analysis_policy {analysis_policy!r}")
    n_feat = N_FEAT_FULL if include_coherence else N_FEAT_ENHANCED
    rows: list[np.ndarray] = []
    labels: list[float] = []
    n_excluded = 0
    n_no_analysis = 0
    step_minutes = probsevere_step_minutes(time_steps)
    id_index = index_storms_by_id(time_steps)
    hours = sorted(analyses)

    for step_idx, ts in enumerate(time_steps):
        storm_time_utc = parse_probsevere_valid_time(ts.get("valid_time", ""))
        if analysis_policy == "fixed_18z":
            hour = 18 if 18 in analyses else None
        else:
            hour = select_analysis_hour(storm_time_utc, hours)
        analysis = analyses.get(hour) if hour is not None else None

        storms = ts.get("storms", [])
        for storm in storms:
            # Compute label with STRICT temporal ordering
            label = compute_label(
                float(storm.get("lat", 0)),
                float(storm.get("lon", 0)),
                storm_time_utc,
                tornado_reports,
                forward_minutes=FORWARD_WINDOW_MIN,
            )

            if label == -1:
                n_excluded += 1
                continue  # EXCLUDED: cannot verify temporal ordering
            if analysis is None:
                n_no_analysis += 1
                continue  # EXCLUDED: no causal atmosphere for this observation
            hrrr_grids, derived_hrrr, coherence_fields = analysis

            # Build feature vector
            feat_p = extract_block_p(storm)

            # Build storm history for evolution features
            storm_id = storm.get("id", None)
            history = build_storm_history(time_steps, storm_id, step_idx, id_index=id_index)
            feat_e = extract_block_e(storm, history, step_minutes)

            feat_h = extract_block_h(storm, hrrr_grids, derived_hrrr)

            if include_coherence:
                if coherence_fields is None:
                    n_no_analysis += 1
                    continue
                feat_c = extract_block_c(storm, coherence_fields, hrrr_grids)
                feat = np.concatenate([feat_p, feat_e, feat_h, feat_c])
            else:
                feat = np.concatenate([feat_p, feat_e, feat_h])

            assert feat.shape[0] == n_feat, (
                f"Feature vector has {feat.shape[0]} elements, expected {n_feat}"
            )

            rows.append(feat)
            labels.append(float(label))

    if len(rows) == 0:
        return (
            np.zeros((0, n_feat), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            n_excluded,
            n_no_analysis,
        )

    X = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, n_excluded, n_no_analysis




# ===================================================================
# DATA LOADING ORCHESTRATOR
# ===================================================================

def load_all_data(
    spc_csv_path: Path,
    verbose: bool = True,
    *,
    train_neg_ratio: int = TRAIN_NEG_RATIO,
    analysis_policy: str = "latest_at_or_before",
) -> tuple[
    dict[str, np.ndarray],  # X_train variants
    dict[str, np.ndarray],  # X_val variants
    dict[str, np.ndarray],  # X_test variants
    np.ndarray,             # y_train
    np.ndarray,             # y_val
    np.ndarray,             # y_test
    dict,                   # metadata
]:
    """Load all data and build train/val/test feature matrices.

    Iterates over all cached ProbSevere dates, loads corresponding HRRR
    and coherence fields, builds samples with strict temporal labeling,
    and splits into train/val/test by date.

    Only the TRAIN split is negative-downsampled (to ``train_neg_ratio``:1,
    for GBT tractability). Validation and test keep their full storm-object
    populations, so the probabilities scored on them are scored at the real
    base rate. Until 2026-10-01 the test split was downsampled to 5:1 too,
    which left AUC unbiased but made the reported Brier, BSS, PR-AUC and
    reliability describe a population with a 1/6 base rate that does not
    exist anywhere outside the experiment.

    Returns
    -------
    X_train, X_val, X_test : dict[str, ndarray]
        Feature matrices keyed by variant: ``"baseline"``, ``"enhanced"``,
        ``"full"``.
    y_train, y_val, y_test : ndarray
        Binary labels for each split.
    metadata : dict
        Loading statistics. ``metadata["_groups"]`` maps each split to an
        int array (one entry per sample, aligned with ``y``) holding the
        sample's UTC day as ``YYYYMMDD`` -- the cluster unit for the day-block
        bootstrap. Keys starting with ``_`` are not JSON-serialised.
    """
    from hazardpulse.tornado.coherence_engine import (
        compute_coherence_fields,
        compute_derived_hrrr,
    )

    # Load SPC tornado reports (keyed by UTC date, absolute UTC instants)
    if verbose:
        print("Loading SPC tornado reports...")
        sys.stdout.flush()
    tornado_reports = load_spc_tornado_reports(spc_csv_path)
    n_unknown_clock = sum(
        1 for v in tornado_reports.values() for r in v if r.get("time_utc") is None
    )
    if verbose:
        total_reports = sum(len(v) for v in tornado_reports.values())
        print(f"  Loaded {total_reports} tornado reports across {len(tornado_reports)} UTC days "
              f"({n_unknown_clock} with an unknown clock, never matched)")

    # Scan available ProbSevere dates
    available_dates = scan_probsevere_cache()
    if verbose:
        print(f"  Found {len(available_dates)} cached ProbSevere dates")

    rows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    groups: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

    total_excluded = 0
    total_no_analysis = 0
    n_analyses_loaded = 0
    dates_processed = 0
    dates_with_hrrr = 0
    days_without_hrrr: list[str] = []

    for date_str in available_dates:
        # Determine which split this date belongs to
        if date_in_range(date_str, TRAIN_START, TRAIN_END):
            split = "train"
        elif date_in_range(date_str, VAL_START, VAL_END):
            split = "val"
        elif date_in_range(date_str, TEST_START, TEST_END):
            split = "test"
        else:
            continue  # Outside all splits

        # Load ProbSevere
        time_steps = load_cached_probsevere(date_str)
        if time_steps is None or len(time_steps) == 0:
            continue

        # Load this UTC day's HRRR analyses (each storm later reads the latest
        # one at or before its own observation -- see HRRR_ANALYSIS_HOURS).
        month = int(date_str[4:6])
        analyses: dict[int, tuple] = {}
        wanted = (18,) if analysis_policy == "fixed_18z" else HRRR_ANALYSIS_HOURS
        for hour in wanted:
            grids = load_cached_hrrr(date_str, hour=hour)
            if grids is None:
                continue
            analyses[hour] = (
                grids,
                compute_derived_hrrr(grids),
                compute_coherence_fields(grids, month=month),
            )
        n_analyses_loaded += len(analyses)
        if analyses:
            dates_with_hrrr += 1
        else:
            days_without_hrrr.append(date_str)

        # Reports that can fall in any of this UTC day's label windows
        tor_window = reports_in_label_window(tornado_reports, date_str)

        # Build samples (always build full feature vector)
        X_day, y_day, n_exc, n_no_an = build_samples_for_date(
            date_str,
            time_steps,
            tor_window,
            analyses,
            include_coherence=True,
            analysis_policy=analysis_policy,
        )

        total_excluded += n_exc
        total_no_analysis += n_no_an

        if X_day.shape[0] == 0:
            continue

        dates_processed += 1
        rows[split].append(X_day)
        labels[split].append(y_day)
        groups[split].append(np.full(len(y_day), int(date_str), dtype=np.int64))

        if verbose and dates_processed % 50 == 0:
            print(f"  Processed {dates_processed} dates...")
            sys.stdout.flush()

    def _stack(parts: list[np.ndarray], width: int | None, dtype) -> np.ndarray:
        if len(parts) == 0:
            shape = (0, width) if width is not None else (0,)
            return np.zeros(shape, dtype=dtype)
        if width is None:
            return np.concatenate(parts).astype(dtype)
        return np.vstack(parts).astype(dtype)

    X_full = {s: _stack(rows[s], N_FEAT_FULL, np.float32) for s in rows}
    y_all = {s: _stack(labels[s], None, np.float32) for s in rows}
    g_all = {s: _stack(groups[s], None, np.int64) for s in rows}

    if verbose:
        print(f"\n  === RAW DATA (full storm-object populations) ===")
        print(f"  Dates processed: {dates_processed}")
        print(f"  Dates with HRRR: {dates_with_hrrr} ({n_analyses_loaded} analyses, "
              f"policy {analysis_policy})")
        print(f"  Samples excluded (unknown timing): {total_excluded}")
        print(f"  Samples excluded (no causal HRRR analysis): {total_no_analysis}")
        for s in ("train", "val", "test"):
            y = y_all[s]
            rate = float(y.mean()) if len(y) else float("nan")
            print(f"  {s:5s}: {len(y)} samples ({int(y.sum())} positive, base rate {rate:.5f})")
        sys.stdout.flush()

    # Negative downsampling -- TRAIN ONLY. Keep all positives.
    rng = np.random.RandomState(42)
    y_tr = y_all["train"]
    n_pos_train = int((y_tr == 1).sum())
    n_neg_train = int((y_tr == 0).sum())
    neg_keep_rate = 1.0
    if n_pos_train > 0 and n_neg_train > train_neg_ratio * n_pos_train:
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        keep_neg = rng.choice(neg_idx, size=train_neg_ratio * n_pos_train, replace=False)
        keep = np.sort(np.concatenate([pos_idx, keep_neg]))
        X_full["train"] = X_full["train"][keep]
        y_all["train"] = y_tr[keep]
        g_all["train"] = g_all["train"][keep]
        neg_keep_rate = (train_neg_ratio * n_pos_train) / n_neg_train

    if verbose:
        print(f"\n  === AFTER TRAIN-ONLY {train_neg_ratio}:1 DOWNSAMPLING ===")
        print(f"  Train: {len(y_all['train'])} samples ({int(y_all['train'].sum())} positive, "
              f"negative keep rate {neg_keep_rate:.5f})")
        print(f"  Val:   {len(y_all['val'])} samples ({int(y_all['val'].sum())} positive) -- full population")
        print(f"  Test:  {len(y_all['test'])} samples ({int(y_all['test'].sum())} positive) -- full population")
        sys.stdout.flush()

    # Build variant-specific feature matrices by column slicing
    # Baseline: Block P only (columns 0..12)
    # Enhanced: Block P + E + H (columns 0..30)
    # Full: Block P + E + H + C (columns 0..40)
    def _variants(X: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "baseline": X[:, :N_FEAT_BASELINE],
            "enhanced": X[:, :N_FEAT_ENHANCED],
            "full": X[:, :N_FEAT_FULL],
        }

    X_train = _variants(X_full["train"])
    X_val = _variants(X_full["val"])
    X_test = _variants(X_full["test"])
    y_train, y_val, y_test = y_all["train"], y_all["val"], y_all["test"]

    def _days(g: np.ndarray) -> list[str]:
        return [str(d) for d in np.unique(g)]

    metadata = {
        "dates_processed": dates_processed,
        "dates_with_hrrr": dates_with_hrrr,
        "days_without_hrrr": len(days_without_hrrr),
        "hrrr_analyses_loaded": n_analyses_loaded,
        "hrrr_analysis_policy": analysis_policy,
        "hrrr_max_analysis_age_h": MAX_ANALYSIS_AGE_H,
        "total_excluded": total_excluded,
        "excluded_no_causal_analysis": total_no_analysis,
        "spc_reports_unknown_clock": n_unknown_clock,
        "label_time_axis": "absolute UTC instants (SPC tz-converted; ProbSevere validTime)",
        "train_negative_downsampling": {
            "neg_ratio": train_neg_ratio,
            "n_pos_population": n_pos_train,
            "n_neg_population": n_neg_train,
            "neg_keep_rate": neg_keep_rate,
            "population_base_rate": (n_pos_train / (n_pos_train + n_neg_train))
            if (n_pos_train + n_neg_train) else None,
        },
        "val_test_downsampled": False,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_train_pos": int(y_train.sum()) if len(y_train) > 0 else 0,
        "n_val_pos": int(y_val.sum()) if len(y_val) > 0 else 0,
        "n_test_pos": int(y_test.sum()) if len(y_test) > 0 else 0,
        "n_train_days": len(_days(g_all["train"])),
        "n_val_days": len(_days(g_all["val"])),
        "n_test_days": len(_days(g_all["test"])),
        "_groups": g_all,
    }

    # The anti-leakage guarantee, enforced on the data actually built.
    assert_temporal_integrity(_days(g_all["train"]), _days(g_all["val"]), _days(g_all["test"]))

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata




# ===================================================================
# TEMPORAL SPLIT ASSERTIONS
# ===================================================================

def assert_temporal_integrity(
    train_dates: list[str],
    val_dates: list[str],
    test_dates: list[str],
) -> None:
    """Assert that no sample crosses temporal boundaries.

    This is the core anti-leakage guarantee. If these assertions fail,
    the entire experiment is invalid.
    """
    for d in train_dates:
        assert d < VAL_START, (
            f"LEAKAGE: Train date {d} >= VAL_START {VAL_START}"
        )
        assert d <= TRAIN_END, (
            f"LEAKAGE: Train date {d} > TRAIN_END {TRAIN_END}"
        )

    for d in val_dates:
        assert d >= VAL_START, (
            f"LEAKAGE: Val date {d} < VAL_START {VAL_START}"
        )
        assert d < TEST_START, (
            f"LEAKAGE: Val date {d} >= TEST_START {TEST_START}"
        )

    for d in test_dates:
        assert d >= TEST_START, (
            f"LEAKAGE: Test date {d} < TEST_START {TEST_START}"
        )
        assert d <= TEST_END, (
            f"LEAKAGE: Test date {d} > TEST_END {TEST_END}"
        )


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main(
    spc_csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    *,
    analysis_policy: str = "latest_at_or_before",
) -> dict:
    """Run the complete definitive tornado prediction pipeline.

    ``analysis_policy`` exists to MEASURE the pre-2026-10-01 18Z leak
    (``"fixed_18z"``); every product run uses the default.

    Steps:
        1. Load ProbSevere cache
        2. Load HRRR cache
        3. Load SPC tornado reports (converted to absolute UTC instants)
        4. Build samples with STRICT temporal labeling
            - Temporal integrity asserted on the days actually built
            - Print: "X samples EXCLUDED due to unknown timing"
        5. Build feature matrices (P, E, H, C)
        6. Normalize features (fit on train, apply to val/test)
        7. Train 3 GBT models on train (early stopping on val)
        8. Calibrate each model on the full validation population
        9. Evaluate on test ONCE (full population, day-clustered CIs)
        10. Significance tests (day-clustered paired bootstrap)
        11. Save results JSON

    Parameters
    ----------
    spc_csv_path : str or Path, optional
        Path to SPC tornado CSV. Defaults to project cache.
    output_dir : str or Path, optional
        Directory for results JSON. Defaults to project root.
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
    if spc_csv_path is None:
        spc_csv_path = project_root / ".cache" / "spc" / "1950-2024_actual_tornadoes.csv"
    else:
        spc_csv_path = Path(spc_csv_path)

    if output_dir is None:
        output_dir = project_root / "results" / "definitive"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "enhanced", "full"]

    if verbose:
        print("=" * 72)
        print("DEFINITIVE TORNADO PREDICTION MODEL v2")
        print("RESEARCH ONLY -- NOT OPERATIONAL")
        print("=" * 72)
        print()
        print(f"Temporal splits:")
        print(f"  Train: {TRAIN_START} -- {TRAIN_END}")
        print(f"  Val:   {VAL_START} -- {VAL_END}")
        print(f"  Test:  {TEST_START} -- {TEST_END}")
        print()
        print(f"GBT config: {GBT_N_TREES} trees, depth={GBT_MAX_DEPTH}, "
              f"lr={GBT_LEARNING_RATE}, subsample={GBT_SUBSAMPLE}")
        print(f"Label: {LABEL_RADIUS_KM} km radius, {FORWARD_WINDOW_MIN} min forward, "
              f"absolute UTC instants")
        print()
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 1-4: Load data and build samples
    # ---------------------------------------------------------------
    if verbose:
        print("[1-4] Loading data and building samples...")
        sys.stdout.flush()

    X_train, X_val, X_test, y_train, y_val, y_test, meta = load_all_data(
        spc_csv_path, verbose=verbose, analysis_policy=analysis_policy,
    )
    groups = meta.pop("_groups")
    g_test = groups["test"]

    assert len(y_train) > 0, "No training samples loaded"
    assert len(y_val) > 0, "No validation samples loaded"
    assert len(y_test) > 0, "No test samples loaded"

    if verbose:
        print(f"\n  {meta['total_excluded']} samples EXCLUDED due to unknown timing")
        sys.stdout.flush()

    # Raw (physical-unit) test features, kept for the analytic NPE blend,
    # which applies physical thresholds and must never see z-scores.
    X_test_raw_full = X_test["full"].copy()

    # ---------------------------------------------------------------
    # Step 5-6: Normalize features (fit on train, apply to val/test)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[5-6] Normalizing features...")
        sys.stdout.flush()

    normalizers: dict[str, FeatureNormalizer] = {}
    for variant in variants:
        norm = FeatureNormalizer()
        X_train[variant] = norm.fit_transform(X_train[variant])
        X_val[variant] = norm.transform(X_val[variant])
        X_test[variant] = norm.transform(X_test[variant])
        normalizers[variant] = norm

    # ---------------------------------------------------------------
    # Step 7: Train 3 GBT models on train (with early stopping on val)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[7] Training GBT models...")
        sys.stdout.flush()

    models: dict[str, GradientBoostedTrees] = {}
    train_meta: dict[str, dict] = {}

    for variant in variants:
        if verbose:
            print(f"\n  --- {variant.upper()} ({X_train[variant].shape[1]} features) ---")
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
    # Step 8: Calibrate on the full validation population
    # ---------------------------------------------------------------
    if verbose:
        print("\n[8] Validation: sanity AUC + probability calibration...")
        sys.stdout.flush()

    calibrations: dict[str, dict] = {}
    val_preds: dict[str, np.ndarray] = {}
    val_auc: dict[str, float] = {}
    for variant in variants:
        F_val = models[variant].decision_function(X_val[variant])
        cal = fit_platt(F_val, y_val)
        cal["fitted_on"] = f"validation {VAL_START}-{VAL_END}, full population"
        cal["n"] = int(len(y_val))
        cal["base_rate"] = float(y_val.mean())
        calibrations[variant] = cal
        val_preds[variant] = apply_calibration(F_val, cal)
        val_auc[variant] = compute_auc(y_val, val_preds[variant])
        if verbose:
            print(f"  {variant}: val_auc = {val_auc[variant]:.4f}  "
                  f"platt a={cal['a']:.4f} b={cal['b']:.4f}")
        if val_auc[variant] < 0.55:
            print(
                f"  WARNING: {variant} val AUC ({val_auc[variant]:.4f}) is very low. "
                f"Model may be broken."
            )

    # ---------------------------------------------------------------
    # Step 9: Evaluate on test ONCE
    # ---------------------------------------------------------------
    if verbose:
        print("\n[9] TEST EVALUATION (single pass, full population, no going back)...")
        sys.stdout.flush()

    test_preds: dict[str, np.ndarray] = {}
    test_results: dict[str, dict] = {}
    for variant in variants:
        F_test = models[variant].decision_function(X_test[variant])
        p_test = apply_calibration(F_test, calibrations[variant])
        test_preds[variant] = p_test
        test_results[variant] = evaluate(y_test, p_test, groups=g_test)
        if verbose:
            r = test_results[variant]
            ci = r["bootstrap_ci"]
            print(f"\n  {variant.upper()} on TEST:")
            print(f"    AUC:    {r['auc']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
                  f"(day-clustered, {ci['n_clusters']} days)")
            print(f"    PR-AUC: {r['pr_auc']:.4f}")
            print(f"    Brier:  {r['brier']:.6f}")
            print(f"    BSS:    {r['bss']:.4f}")
            print(f"    N_pos:  {r['n_positive']}, N_neg: {r['n_negative']}, "
                  f"base_rate: {r['base_rate']:.6f}, mean forecast: {r['mean_forecast']:.6f}")
            sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 10: Significance tests
    # ---------------------------------------------------------------
    if verbose:
        print("\n[10] Paired day-clustered bootstrap significance tests...")
        sys.stdout.flush()

    hrrr_lift = paired_cluster_bootstrap_test(
        y_test, test_preds["baseline"], test_preds["enhanced"], g_test
    )
    cft_lift = paired_cluster_bootstrap_test(
        y_test, test_preds["enhanced"], test_preds["full"], g_test
    )

    if verbose:
        print(f"\n  HRRR lift (enhanced vs baseline):")
        print(f"    delta_AUC = {hrrr_lift['delta_auc']:.4f} "
              f"[{hrrr_lift['ci_lo']:.4f}, {hrrr_lift['ci_hi']:.4f}] "
              f"p = {hrrr_lift['p_value']:.4f}")
        print(f"\n  CFT lift (full vs enhanced):")
        print(f"    delta_AUC = {cft_lift['delta_auc']:.4f} "
              f"[{cft_lift['ci_lo']:.4f}, {cft_lift['ci_hi']:.4f}] "
              f"p = {cft_lift['p_value']:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 10b: NPE Fusion (analytic-learned blend with coherence physiology)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[10b] NPE Fusion: analytic-learned blend...")
        sys.stdout.flush()

    npe_lift = None
    npe_health = None
    try:
        from hazardpulse.tornado.tornado_npe import TornadoNPEEngine

        npe_engine = TornadoNPEEngine()
        # Coherence tracking learns from the VALIDATION outcomes first, then
        # predicts test -- the old order recorded val after predicting test,
        # so the tracker never informed a single test prediction.
        for idx in range(len(y_val)):
            npe_engine.record_outcome(float(val_preds["full"][idx]), bool(y_val[idx] > 0.5))

        col = {name: k for k, name in enumerate(ALL_FEATURE_NAMES_FULL)}
        Xr = X_test_raw_full  # PHYSICAL units -- the analytic model's thresholds are physical
        p_npe_test = np.zeros(len(y_test), dtype=np.float64)
        for idx in range(len(y_test)):
            pred = npe_engine.predict(
                tau=float(Xr[idx, col["tau"]]),
                grad_tau=float(Xr[idx, col["grad_tau"]]),
                torsion=float(Xr[idx, col["torsion"]]),
                alignment=float(Xr[idx, col["alignment"]]),
                s_over_gamma=float(Xr[idx, col["S_over_Gamma"]]),
                da=float(Xr[idx, col["Da"]]),
                maxllaz=float(Xr[idx, col["maxllaz"]]),
                srh01=float(Xr[idx, col["srh01"]]),
                gbt_probability=float(test_preds["full"][idx]),
            )
            p_npe_test[idx] = pred.probability

        test_results["npe_fusion"] = evaluate(y_test, p_npe_test, groups=g_test)
        test_preds["npe_fusion"] = p_npe_test
        npe_lift = paired_cluster_bootstrap_test(
            y_test, test_preds["full"], test_preds["npe_fusion"], g_test
        )
        npe_health = npe_engine.health()

        if verbose:
            r = test_results["npe_fusion"]
            print(f"\n  NPE FUSION on TEST:")
            print(f"    AUC:    {r['auc']:.4f} "
                  f"[{r['bootstrap_ci']['ci_lo']:.4f}, {r['bootstrap_ci']['ci_hi']:.4f}]")
            print(f"    PR-AUC: {r['pr_auc']:.4f}")
            print(f"    Coherence score: {npe_health.coherence_score:.3f}")
            print(f"    NPE lift vs Full GBT: {npe_lift['delta_auc']:.4f} "
                  f"(p={npe_lift['p_value']:.4f})")
            sys.stdout.flush()
    except ImportError:
        if verbose:
            print("  [SKIP] tornado_npe not available, skipping fusion")

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
        print(f"\n  Top 15 features (full model, split count importance):")
        for name, imp in importance_ranked[:15]:
            print(f"    {name:25s}  {imp:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 11: Assemble and save results
    # ---------------------------------------------------------------
    from hazardpulse.tornado import coherence_engine as _coh

    elapsed = time.time() - t_start

    results = {
        "model": "hazardpulse_tornado_definitive_v2",
        "disclaimer": "RESEARCH ONLY. NOT operational. NOT a replacement for NWS.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "audit_guarantees": {
            "temporal_split": f"Train {TRAIN_START}-{TRAIN_END}, "
                              f"Val {VAL_START}-{VAL_END}, "
                              f"Test {TEST_START}-{TEST_END}",
            "temporal_integrity_asserted_on_built_days": True,
            "label_temporal_ordering": True,
            "label_time_axis": "absolute UTC instants: SPC local time + tz offset "
                               "(tz=3 CST -> +6 h); ProbSevere validTime UTC; "
                               "reports from the observation's UTC day and the next",
            "unknown_timing_excluded": True,
            "hrrr_analysis_matching": (
                f"latest 3-hourly analysis at or before each observation, <= "
                f"{MAX_ANALYSIS_AGE_H:g} h old; storms without one excluded"
                if analysis_policy == "latest_at_or_before" else
                f"ABLATION {analysis_policy}: every storm reads the 18Z analysis (future leak)"
            ),
            "hrrr_grid_geometry": "g2: native Lambert points binned by true lat/lon",
            "n_excluded_unknown_timing": meta["total_excluded"],
            "negative_downsampling": f"train only, {TRAIN_NEG_RATIO}:1; val and test "
                                     "are full storm-object populations",
            "probabilities": "Platt-calibrated on the full validation population",
            "confidence_intervals": "bootstrap over whole UTC days",
            "meta_stacker": False,
            "hyperparameter_tuning_on_test": False,
            "single_test_evaluation": True,
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
                "early_stopping": "class-balanced val log-loss, patience "
                                  f"{EARLY_STOP_PATIENCE}",
            },
            "label_config": {
                "radius_km": LABEL_RADIUS_KM,
                "forward_window_min": FORWARD_WINDOW_MIN,
            },
            "coherence_solver": {
                "mode": "certified PCG" if _coh.HELMHOLTZ_TOL is not None else "legacy Jacobi",
                "tol": _coh.HELMHOLTZ_TOL,
                "damping_semantics": "Gamma passed directly (FVCS W-2)",
            },
            "torsion": "(S_01 x grad tau).k / "
                       f"{_coh.TORSION_SHEAR_SCALE} -- tilting of low-level horizontal "
                       "vorticity by the coherence gradient",
        },
        "data_summary": meta,
        "training_metadata": {
            variant: {
                "n_trees_used": train_meta[variant]["n_trees_used"],
                "stopped_early": train_meta[variant]["stopped_early"],
                "best_val_loss": train_meta[variant].get("best_val_loss"),
            }
            for variant in variants
        },
        "calibration": calibrations,
        "validation_sanity_check": {
            variant: {"auc": val_auc[variant]} for variant in variants
        },
        "baseline": test_results["baseline"],
        "enhanced": test_results["enhanced"],
        "full": test_results["full"],
        "npe_fusion": test_results.get("npe_fusion"),
        "theory_tests": {
            "hrrr_lift": hrrr_lift,
            "cft_lift": cft_lift,
            "npe_lift": npe_lift,
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
    results_path = output_dir / "definitive_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved to: {results_path}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print("\n" + "=" * 72)
        print("DONE. Remember: this is RESEARCH ONLY.")
        print("=" * 72)
        sys.stdout.flush()

    # Attach trained artifacts so callers can save them
    results["_models"] = models
    results["_normalizers"] = normalizers
    results["_calibrations"] = calibrations

    return results




# ===================================================================
# MODEL SERIALIZATION -- SAVE / LOAD FOR LIVE SCORING
# ===================================================================


def _tree_to_json(node: dict) -> dict:
    """Convert a GBT tree node to JSON-safe dict (recursive)."""
    if node.get("leaf", False):
        return {"leaf": True, "val": float(node["val"])}
    return {
        "leaf": False,
        "feat": int(node["feat"]),
        "thresh": float(node["thresh"]),
        "left": _tree_to_json(node["left"]),
        "right": _tree_to_json(node["right"]),
    }


def save_model(
    model: GradientBoostedTrees,
    normalizer: FeatureNormalizer,
    feature_names: list[str],
    path: str | Path,
    variant: str = "full",
    calibration: dict | None = None,
    provenance: dict | None = None,
) -> None:
    """Serialize a trained GBT + normalizer to a JSON file.

    ``calibration`` (from ``fit_platt`` on the validation population) is
    what makes the served number a probability; ``provenance`` records how
    the payload was produced (label time axis, splits, solver, metrics).

    The output is pure JSON with no numpy types, so it can be loaded
    by the live scoring pipeline without training dependencies.

    Parameters
    ----------
    model : GradientBoostedTrees
        Trained model instance.
    normalizer : FeatureNormalizer
        Fitted normalizer (means/stds from training set).
    feature_names : list[str]
        Ordered feature names matching the model's expected input.
    path : str or Path
        Output file path (.json).
    variant : str
        Model variant label (e.g. "full", "enhanced", "baseline").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "disclaimer": (
            "RESEARCH ONLY. NOT an operational warning system. "
            "Does NOT replace NWS tornado warnings."
        ),
        "model_format": "hazardpulse_gbt_v1",
        "variant": variant,
        "init_pred": float(model.init_pred),
        "learning_rate": float(model.lr),
        "n_trees": len(model.trees),
        "trees": [_tree_to_json(t) for t in model.trees],
        "feature_names": list(feature_names),
        "normalization": {
            "means": [float(x) for x in normalizer.mean],
            "stds": [float(x) for x in normalizer.std],
        },
    }
    if calibration is not None:
        payload["calibration"] = calibration
    if provenance is not None:
        payload["provenance"] = provenance

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  Saved model ({len(model.trees)} trees, {len(feature_names)} features) -> {path}")


def load_model(path: str | Path) -> tuple[GradientBoostedTrees, FeatureNormalizer, list[str]]:
    """Load a saved GBT model from JSON.

    Parameters
    ----------
    path : str or Path
        Path to the JSON model file produced by ``save_model()``.

    Returns
    -------
    tuple of (GradientBoostedTrees, FeatureNormalizer, list[str])
        The reconstructed model, normalizer, and feature name list.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data.get("model_format") == "hazardpulse_gbt_v1", (
        f"Unknown model format: {data.get('model_format')}"
    )

    gbt = GradientBoostedTrees(
        n_trees=data["n_trees"],
        learning_rate=data["learning_rate"],
    )
    gbt.init_pred = data["init_pred"]
    gbt.lr = data["learning_rate"]
    gbt.trees = data["trees"]  # Already in dict form, same as internal repr

    norm = FeatureNormalizer()
    norm.mean = np.array(data["normalization"]["means"], dtype=np.float32)
    norm.std = np.array(data["normalization"]["stds"], dtype=np.float32)

    feature_names = data["feature_names"]

    return gbt, norm, feature_names


# ===================================================================
# CLI ENTRYPOINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Definitive leak-proof tornado prediction model (RESEARCH ONLY)",
    )
    parser.add_argument(
        "--spc-csv",
        type=str,
        default=None,
        help="Path to SPC tornado CSV (default: .cache/spc/1950-2024_actual_tornadoes.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results JSON (default: results/definitive/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the trained full-variant GBT model to PATH (JSON). "
             "Example: results/models/tornado_gbt_v1.json",
    )

    args = parser.parse_args()
    results = main(
        spc_csv_path=args.spc_csv,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if args.save_model and "_models" in results and "_normalizers" in results:
        full = results["full"]
        save_model(
            model=results["_models"]["full"],
            normalizer=results["_normalizers"]["full"],
            feature_names=ALL_FEATURE_NAMES_FULL,
            path=args.save_model,
            variant="full",
            calibration=results["_calibrations"]["full"],
            provenance={
                "trained": results["timestamp"],
                "model": results["model"],
                "audit_guarantees": results["audit_guarantees"],
                "test_auc": full["auc"],
                "test_auc_ci_day_clustered": [
                    full["bootstrap_ci"]["ci_lo"], full["bootstrap_ci"]["ci_hi"],
                ],
                "test_bss_calibrated": full["bss"],
                "test_base_rate": full["base_rate"],
            },
        )
