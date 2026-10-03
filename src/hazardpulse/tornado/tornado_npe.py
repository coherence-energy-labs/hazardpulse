# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational tornado warning system.
# It does NOT replace official NWS tornado warnings.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Tornado Neural Predictive Engine -- Self-aware coherence-physics fusion.

Adapted from the Coherence Language NPE (Neural Predictive Engine).
Fuses analytic (Helmholtz PDE) predictions with learned (GBT) predictions,
tracks prediction accuracy via coherence physiology, and adapts confidence
based on system health.

Architecture
------------
1. TornadoNPEConfig       -- frozen configuration dataclass
2. TornadoCoherenceField  -- EMA-based accuracy tracker (from NPE field.py)
3. TornadoGatewayMode     -- operating mode enum (from NPE gateway.py)
4. analytic_tornado_probability -- physics-only P(tornado) from Helmholtz
5. TornadoFusionPredictor -- blends analytic + GBT with adaptive weights
6. TornadoNPEEngine       -- top-level orchestrator

Mathematical Model
------------------
The coherence field tracks prediction accuracy via exponential moving average:

    error_t = |predicted_prob - actual_tornado|
    error_ema = alpha * error_t + (1 - alpha) * error_ema_prev
    coherence_score = 1 / (1 + error_ema)

Fusion uses adaptive weighting:

    p_fused = alpha * p_analytic + (1 - alpha) * p_learned

where alpha increases as coherence degrades (trust physics more when ML
is unreliable).

RESEARCH ONLY. NOT an operational warning system.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from enum import Enum
from threading import RLock
from typing import Any

import numpy as np


# ===================================================================
# 1. CONFIGURATION
# ===================================================================


def _engine_torsion_threshold() -> float:
    from hazardpulse.tornado.coherence_engine import TORSION_SINGULARITY_THRESHOLD

    return float(TORSION_SINGULARITY_THRESHOLD)


# Default mode-specific uncertainty multipliers.
_DEFAULT_MODE_UNCERTAINTY: dict[str, float] = {
    "OPERATIONAL": 1.0,
    "CAUTIOUS": 1.3,
    "DEGRADED": 1.8,
    "REPAIR": 2.5,
    "LEARNING": 1.5,
}


@dataclass(frozen=True)
class TornadoNPEConfig:
    """Configuration for the Tornado NPE system.

    All thresholds are documented with physical justification.
    """

    # -- Fusion weights (physics vs ML) --
    # Default 30% physics / 70% ML.  In DEGRADED mode alpha rises to 1.0.
    base_analytic_weight: float = 0.3
    min_analytic_weight: float = 0.1   # even perfect ML keeps 10% physics
    max_analytic_weight: float = 1.0   # degraded = pure physics

    # -- Coherence field --
    # alpha=0.05: slower decay than NPE's 0.15 because tornado events are
    # rare (~1-2% base rate).  Longer memory avoids overreacting to
    # individual misses.
    coherence_alpha: float = 0.05
    coherence_target: float = 0.80     # desired coherence score
    coherence_warning: float = 0.60    # flag for review
    coherence_critical: float = 0.40   # switch to degraded
    coherence_emergency: float = 0.25  # repair mode

    # -- Gateway --
    learning_period: int = 100  # observations before leaving LEARNING mode

    # -- Analytic model thresholds --
    # These encode the 5-condition singularity criterion from Coherence
    # Field Theory.  Thresholds are HAND-TUNED initial estimates.
    # They MUST be validated against labeled training data before any
    # claim of calibration.  Do not treat them as theoretically derived.
    #
    # grad_tau > 0.3: steep coherence gradient indicates a frontal boundary
    # where vorticity concentrates.
    singularity_grad_threshold: float = 0.3
    # |torsion| above the coherence engine's condition-3 threshold: tilting
    # of low-level horizontal vorticity by grad(tau) (Form 16 coupling).
    # Shared with coherence_engine so the analytic model and the singularity
    # count can never disagree about what "significant" means. (The old
    # 0.05 was set against a torsion that was identically ~1e-8.)
    singularity_torsion_threshold: float = dc_field(
        default_factory=lambda: _engine_torsion_threshold()
    )
    # Da > 5.0: Damkohler number indicating reaction-dominated regime
    # where coherence energy release outpaces diffusion.
    singularity_da_threshold: float = 5.0
    # alignment > 0.1: vorticity vector within ~84 degrees of coherence
    # gradient (cos(84) ~ 0.1).
    singularity_alignment_threshold: float = 0.1
    # maxllaz > 0.003 s^-1: MRMS rotation rate threshold for mesocyclone
    # detection (standard operational value).
    singularity_rotation_threshold: float = 0.003
    # SRH > 100 m^2/s^2: storm-relative helicity threshold for
    # supercell-supportive environments (SPC operational guidance).
    singularity_srh_threshold: float = 100.0

    # -- Uncertainty --
    base_uncertainty: float = 0.15  # irreducible uncertainty floor
    mode_uncertainty_factor: dict[str, float] = dc_field(
        default_factory=lambda: dict(_DEFAULT_MODE_UNCERTAINTY)
    )


# ===================================================================
# 2. COHERENCE FIELD (adapted from NPE coherence/field.py)
# ===================================================================

# Minimum denominator to avoid division by zero.
EPSILON: float = 1e-9

# Max sliding-window length for drift detection.
MAX_WINDOW_SAMPLES: int = 1000

# Minimum samples before coherence score is considered confident.
MIN_CONFIDENT_SAMPLES: int = 20  # higher than NPE's 10 due to rarity


@dataclass(frozen=True)
class TornadoCoherenceSample:
    """A single prediction-vs-outcome observation.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp of the observation.
    predicted_prob : float
        Predicted tornado probability (0-1).
    actual_tornado : bool
        Whether a tornado actually occurred.
    error : float
        Absolute error |predicted - actual|.
    component : str
        Tracking component (detection, false_alarm, timing, location, severity).
    meta : dict
        Optional metadata (storm_id, lat, lon, etc.).
    """

    timestamp: str
    predicted_prob: float
    actual_tornado: bool
    error: float
    component: str = "detection"
    meta: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class TornadoCoherenceSnapshot:
    """Point-in-time snapshot of coherence field state.

    Attributes
    ----------
    coherence_score : float
        Aggregate coherence in (0, 1].
    error_ema : float
        Current EMA of prediction errors.
    sample_count : int
        Total observations processed.
    confident : bool
        Whether we have enough samples.
    min_error : float
        Minimum error observed.
    max_error : float
        Maximum error observed.
    last_sample_timestamp : str | None
        Most recent observation time.
    per_component_scores : dict[str, float]
        Per-component coherence scores.
    """

    coherence_score: float
    error_ema: float
    sample_count: int
    confident: bool
    min_error: float
    max_error: float
    last_sample_timestamp: str | None
    per_component_scores: dict[str, float] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "coherence_score": self.coherence_score,
            "error_ema": self.error_ema,
            "sample_count": self.sample_count,
            "confident": self.confident,
            "min_error": self.min_error,
            "max_error": self.max_error,
            "last_sample_timestamp": self.last_sample_timestamp,
            "per_component_scores": dict(self.per_component_scores),
        }


@dataclass
class _ComponentState:
    """Mutable internal state for one tracking component.  Not public."""

    error_ema: float = 0.0
    sample_count: int = 0
    min_error: float = float("inf")
    max_error: float = 0.0
    last_timestamp: str | None = None
    recent_errors: list[float] = dc_field(default_factory=list)


class TornadoCoherenceField:
    """Tracks how well tornado predictions match observed outcomes.

    Maintains per-component EMA of prediction errors:
      - detection: did we predict tornado where one occurred?
      - false_alarm: did we predict tornado where none occurred?
      - timing: did the timing match?
      - location: was the predicted location accurate?
      - severity: was the predicted intensity correct?

    coherence_score = 1 / (1 + error_ema)
    Range: 0 (terrible) to 1 (perfect).

    Thread-safe: all public methods are protected by an internal RLock.
    """

    _COMPONENTS = ("detection", "false_alarm", "timing", "location", "severity")

    __slots__ = (
        "_alpha",
        "_aggregate_error_ema",
        "_aggregate_sample_count",
        "_components",
        "_created_at",
        "_lock",
        "_window_samples",
    )

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha <= 1.0:
            alpha = 0.05
        self._lock = RLock()
        self._alpha = alpha
        self._created_at = datetime.now(UTC).isoformat()
        self._components: dict[str, _ComponentState] = {
            c: _ComponentState() for c in self._COMPONENTS
        }
        self._aggregate_error_ema: float = 0.0
        self._aggregate_sample_count: int = 0
        self._window_samples: list[tuple[str, float]] = []

    # ----- recording -----

    def record_observation(
        self,
        predicted_prob: float,
        actual_tornado: bool,
        component: str = "detection",
        timestamp: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> TornadoCoherenceSample:
        """Record a prediction vs outcome pair.

        Parameters
        ----------
        predicted_prob : float
            Predicted tornado probability (0-1).
        actual_tornado : bool
            Ground truth -- did a tornado occur?
        component : str
            Which tracking dimension.
        timestamp : str | None
            ISO-8601 timestamp (defaults to now).
        meta : dict | None
            Optional metadata.

        Returns
        -------
        TornadoCoherenceSample
        """
        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()

        # Clamp predicted_prob to [0, 1]
        predicted_prob = max(0.0, min(1.0, float(predicted_prob)))
        actual_val = 1.0 if actual_tornado else 0.0
        error = abs(predicted_prob - actual_val)

        sample = TornadoCoherenceSample(
            timestamp=timestamp,
            predicted_prob=predicted_prob,
            actual_tornado=actual_tornado,
            error=error,
            component=component,
            meta=meta or {},
        )

        with self._lock:
            # Ensure component exists
            if component not in self._components:
                self._components[component] = _ComponentState()

            cs = self._components[component]

            # Update EMA
            if cs.sample_count == 0:
                cs.error_ema = error
            else:
                cs.error_ema = self._alpha * error + (1 - self._alpha) * cs.error_ema

            cs.sample_count += 1
            cs.min_error = min(cs.min_error, error)
            cs.max_error = max(cs.max_error, error)
            cs.last_timestamp = timestamp

            # Sliding window for drift detection
            cs.recent_errors.append(error)
            if len(cs.recent_errors) > MAX_WINDOW_SAMPLES:
                cs.recent_errors.pop(0)

            # Aggregate
            if self._aggregate_sample_count == 0:
                self._aggregate_error_ema = error
            else:
                self._aggregate_error_ema = (
                    self._alpha * error
                    + (1 - self._alpha) * self._aggregate_error_ema
                )
            self._aggregate_sample_count += 1

            self._window_samples.append((timestamp, error))
            if len(self._window_samples) > MAX_WINDOW_SAMPLES:
                self._window_samples.pop(0)

        return sample

    # ----- queries -----

    def get_coherence_score(self) -> float:
        """Aggregate coherence score in (0, 1]."""
        with self._lock:
            return self._error_to_coherence(self._aggregate_error_ema)

    def snapshot(self) -> TornadoCoherenceSnapshot:
        """Point-in-time snapshot of all coherence state."""
        with self._lock:
            per_comp: dict[str, float] = {}
            for comp, state in self._components.items():
                if state.sample_count > 0:
                    per_comp[comp] = self._error_to_coherence(state.error_ema)
                else:
                    per_comp[comp] = 1.0

            min_err = float("inf")
            max_err = 0.0
            last_ts: str | None = None
            for state in self._components.values():
                if state.sample_count > 0:
                    min_err = min(min_err, state.min_error)
                    max_err = max(max_err, state.max_error)
                    if last_ts is None or (
                        state.last_timestamp and state.last_timestamp > last_ts
                    ):
                        last_ts = state.last_timestamp

            if min_err == float("inf"):
                min_err = 0.0

            return TornadoCoherenceSnapshot(
                coherence_score=self._error_to_coherence(
                    self._aggregate_error_ema
                ),
                error_ema=self._aggregate_error_ema,
                sample_count=self._aggregate_sample_count,
                confident=self._aggregate_sample_count >= MIN_CONFIDENT_SAMPLES,
                min_error=min_err,
                max_error=max_err,
                last_sample_timestamp=last_ts,
                per_component_scores=per_comp,
            )

    # ----- anomaly detection -----

    def detect_drift(self, window: int = 50, threshold: float = 0.2) -> bool:
        """True if recent errors are significantly higher than the EMA.

        Compares the mean of the last *window* errors to the long-term
        aggregate EMA.  If the ratio exceeds *threshold*, drift is declared.
        """
        with self._lock:
            if len(self._window_samples) < window:
                return False
            recent = [e for _, e in self._window_samples[-window:]]
            if self._aggregate_error_ema < EPSILON:
                return False
            recent_avg = sum(recent) / len(recent)
            drift = (recent_avg - self._aggregate_error_ema) / self._aggregate_error_ema
            return drift > threshold

    def detect_spike(self, threshold: float = 3.0) -> bool:
        """True if the most recent error is > threshold * EMA."""
        with self._lock:
            if len(self._window_samples) == 0:
                return False
            latest = self._window_samples[-1][1]
            if self._aggregate_error_ema < EPSILON:
                return False
            return latest > threshold * self._aggregate_error_ema

    # ----- serialization -----

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-safe dictionary."""
        with self._lock:
            comps: dict[str, Any] = {}
            for comp, state in self._components.items():
                comps[comp] = {
                    "error_ema": state.error_ema,
                    "sample_count": state.sample_count,
                    "min_error": (
                        state.min_error
                        if state.min_error != float("inf")
                        else None
                    ),
                    "max_error": state.max_error,
                    "last_timestamp": state.last_timestamp,
                    "coherence_score": self._error_to_coherence(state.error_ema),
                }
            return {
                "alpha": self._alpha,
                "created_at": self._created_at,
                "aggregate_error_ema": self._aggregate_error_ema,
                "aggregate_sample_count": self._aggregate_sample_count,
                "aggregate_coherence_score": self._error_to_coherence(
                    self._aggregate_error_ema
                ),
                "confident": self._aggregate_sample_count >= MIN_CONFIDENT_SAMPLES,
                "components": comps,
            }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TornadoCoherenceField:
        """Restore a field from a serialized dictionary."""
        alpha = float(d.get("alpha", 0.05))
        obj = cls(alpha=alpha)
        obj._created_at = str(d.get("created_at", obj._created_at))
        obj._aggregate_error_ema = float(d.get("aggregate_error_ema", 0.0))
        obj._aggregate_sample_count = int(d.get("aggregate_sample_count", 0))
        comps = d.get("components", {})
        for comp, cdict in comps.items():
            cs = _ComponentState(
                error_ema=float(cdict.get("error_ema", 0.0)),
                sample_count=int(cdict.get("sample_count", 0)),
                min_error=float(cdict.get("min_error") or float("inf")),
                max_error=float(cdict.get("max_error", 0.0)),
                last_timestamp=cdict.get("last_timestamp"),
            )
            obj._components[comp] = cs
        return obj

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            for state in self._components.values():
                state.error_ema = 0.0
                state.sample_count = 0
                state.min_error = float("inf")
                state.max_error = 0.0
                state.last_timestamp = None
                state.recent_errors.clear()
            self._aggregate_error_ema = 0.0
            self._aggregate_sample_count = 0
            self._window_samples.clear()

    # ----- internal -----

    @staticmethod
    def _error_to_coherence(error_ema: float) -> float:
        """Convert error EMA to coherence score: 1 / (1 + error_ema)."""
        if not math.isfinite(error_ema) or error_ema < 0:
            error_ema = 0.0
        return 1.0 / (1.0 + error_ema)


# ===================================================================
# 3. GATEWAY MODE SELECTION (adapted from NPE gateway.py)
# ===================================================================


class TornadoGatewayMode(str, Enum):
    """Operating mode for the Tornado NPE.

    OPERATIONAL : Full confidence, all features enabled.
    CAUTIOUS    : Reduced confidence, predictions flagged for review.
    DEGRADED    : Analytic-only, ML path disabled.
    REPAIR      : Minimal output, request human review.
    LEARNING    : First N predictions while building coherence baseline.
    """

    OPERATIONAL = "OPERATIONAL"
    CAUTIOUS = "CAUTIOUS"
    DEGRADED = "DEGRADED"
    REPAIR = "REPAIR"
    LEARNING = "LEARNING"


def _clamp01(x: float) -> float:
    """Clamp to [0, 1], treating non-finite as 0.0."""
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def decide_tornado_mode(
    coherence_score: float,
    n_observations: int,
    config: TornadoNPEConfig,
    *,
    data_quality_score: float = 1.0,
) -> TornadoGatewayMode:
    """Select gateway mode based on system health.

    Parameters
    ----------
    coherence_score : float
        Aggregate coherence from TornadoCoherenceField (0-1).
    n_observations : int
        Number of outcomes recorded so far.
    config : TornadoNPEConfig
        Configuration with threshold values.
    data_quality_score : float
        Quality of incoming data (0=garbage, 1=perfect).
        Reduces to CAUTIOUS/DEGRADED when data is poor.

    Returns
    -------
    TornadoGatewayMode
    """
    c = _clamp01(coherence_score)
    dq = _clamp01(data_quality_score)

    # Still building baseline -- LEARNING mode.
    if n_observations < config.learning_period:
        return TornadoGatewayMode.LEARNING

    # Severe degradation -> REPAIR.
    if c < config.coherence_emergency:
        return TornadoGatewayMode.REPAIR

    # Poor coherence -> DEGRADED (analytic only, no ML).
    if c < config.coherence_critical:
        return TornadoGatewayMode.DEGRADED

    # Moderate degradation or poor data -> CAUTIOUS.
    if c < config.coherence_warning or dq < 0.5:
        return TornadoGatewayMode.CAUTIOUS

    # Good coherence but marginal data -> CAUTIOUS.
    if dq < 0.8:
        return TornadoGatewayMode.CAUTIOUS

    # Everything healthy.
    return TornadoGatewayMode.OPERATIONAL


# ===================================================================
# 4. ANALYTIC TORNADO PROBABILITY
# ===================================================================


def analytic_tornado_probability(
    tau: float,
    grad_tau: float,
    torsion: float,
    alignment: float,
    s_over_gamma: float,
    da: float,
    maxllaz: float,
    srh01: float,
    config: TornadoNPEConfig | None = None,
) -> float:
    """Physics-only tornado probability from coherence field quantities.

    Combines the 5-condition singularity criterion from Coherence Field
    Theory with two observational signals via a sigmoid.

    CALIBRATION NOTE
    ~~~~~~~~~~~~~~~~
    The logit coefficients below are HAND-TUNED initial estimates based on
    physical reasoning about the relative importance of each condition.
    They have NOT been validated against labeled tornado data.  Before using
    this function in any evaluation, the coefficients MUST be calibrated
    (e.g., via logistic regression on training data) and cross-validated.
    Do not cite these coefficients as theoretically derived.

    Parameters
    ----------
    tau : float
        Coherence field value at the storm location.
    grad_tau : float
        |grad(tau)| -- coherence gradient magnitude.
    torsion : float
        Shear-curl coupling (Form 16 torsion).
    alignment : float
        Vorticity-coherence-gradient alignment (dot product).
    s_over_gamma : float
        Source / damping ratio from Helmholtz PDE.
    da : float
        Damkohler number (reaction rate / diffusion rate).
    maxllaz : float
        Maximum low-level azimuthal shear (s^-1) from MRMS.
    srh01 : float
        0-1 km storm-relative helicity (m^2/s^2).
    config : TornadoNPEConfig | None
        Configuration with thresholds (uses defaults if None).

    Returns
    -------
    float
        Estimated tornado probability in [0, 1].
    """
    if config is None:
        config = TornadoNPEConfig()

    logit = 0.0

    # --- Physics conditions (from S_One / Helmholtz PDE) ---

    # Condition 1: Source exceeds damping.  When S/Gamma > 1, coherence
    # energy is being injected faster than it dissipates -- a necessary
    # condition for singularity formation.
    # Coefficient 0.5: moderate weight; this is necessary but not sufficient.
    logit += 0.5 * max(0.0, s_over_gamma - 1.0)

    # Condition 2: Steep coherence gradient.  Large |grad(tau)| indicates
    # a sharp front where vorticity concentrates.
    # Coefficient 0.3: lower weight; gradients can be steep without tornado.
    logit += 0.3 * max(0.0, grad_tau - config.singularity_grad_threshold)

    # Condition 3: Rotational focusing (torsion).  Non-zero torsion means
    # shear and curl are coupled, focusing energy into a vortex.
    # Coefficient 0.4: moderate weight; torsion is strongly discriminative.
    logit += 0.4 * max(0.0, abs(torsion) - config.singularity_torsion_threshold)

    # Condition 4: Decay-dominated regime (high Da).  Da >> 1 means
    # coherence energy release outpaces diffusion -- explosive growth.
    # Coefficient 0.2/10: small weight per unit Da; Da can be very large.
    logit += 0.2 * max(0.0, da - config.singularity_da_threshold) / 10.0

    # Condition 5: Vorticity aligned with coherence gradient.
    # Alignment > threshold means the rotation axis is locked to the
    # coherence structure -- a focusing condition.
    # Coefficient 0.3: moderate weight.
    logit += 0.3 * max(0.0, alignment - config.singularity_alignment_threshold)

    # --- Observational conditions ---

    # Condition 6: Radar-detected rotation.  maxllaz > 0.003 s^-1 is the
    # standard operational mesocyclone detection threshold.
    # CAPPED at 1.5 to prevent dominating the score.
    rotation_contrib = min(
        2.0 * max(0.0, maxllaz - config.singularity_rotation_threshold) / 0.01,
        1.5,  # cap: rotation alone cannot push above ~50%
    )
    logit += rotation_contrib

    # Condition 7: Environmental helicity.  SRH > 100 m^2/s^2 indicates
    # a supercell-supportive wind profile.
    # CAPPED at 1.0 to prevent extreme SRH from dominating.
    srh_contrib = min(
        1.0 * max(0.0, srh01 - config.singularity_srh_threshold) / 200.0,
        1.0,  # cap: SRH alone cannot push above ~40%
    )
    logit += srh_contrib

    # Bias term: shifts the sigmoid so that when no conditions are met,
    # the base probability is ~7.6% (sigmoid(-2.5) ~ 0.076).
    logit -= 2.5

    prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, logit))))

    # Hard cap at 60% for analytic model (not ML-calibrated)
    return min(prob, 0.60)


# ===================================================================
# 5. TORNADO PREDICTION (output dataclass)
# ===================================================================


@dataclass(frozen=True)
class TornadoPrediction:
    """Result of a fused tornado prediction.

    Attributes
    ----------
    probability : float
        Fused P(tornado) in [0, 1].
    uncertainty : float
        Estimated confidence interval half-width.
    gateway_mode : TornadoGatewayMode
        Current system operating mode.
    coherence_score : float
        System health (aggregate coherence).
    analytic_probability : float
        Physics-only estimate.
    learned_probability : float
        GBT (or other ML) estimate.
    fusion_weight : float
        Alpha used for blending (1 = pure physics).
    singularity_conditions_met : int
        Count of the 5 physics conditions satisfied (0-5).
    timestamp : str
        ISO-8601 prediction time.
    meta : dict[str, Any]
        Optional additional metadata.
    """

    probability: float
    uncertainty: float
    gateway_mode: TornadoGatewayMode
    coherence_score: float
    analytic_probability: float
    learned_probability: float
    fusion_weight: float
    singularity_conditions_met: int
    timestamp: str
    meta: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "probability": self.probability,
            "uncertainty": self.uncertainty,
            "gateway_mode": self.gateway_mode.value,
            "coherence_score": self.coherence_score,
            "analytic_probability": self.analytic_probability,
            "learned_probability": self.learned_probability,
            "fusion_weight": self.fusion_weight,
            "singularity_conditions_met": self.singularity_conditions_met,
            "timestamp": self.timestamp,
            "meta": self.meta,
        }


# ===================================================================
# 6. FUSION PREDICTOR
# ===================================================================


class TornadoFusionPredictor:
    """Fuses analytic (physics) and learned (GBT) tornado predictions.

    The blending weight adapts based on coherence score:

        alpha = base + (max - base) * (1 - coherence_score)

    High coherence (learned path accurate) -> lower alpha (trust ML more).
    Low coherence (learned path unreliable) -> higher alpha (trust physics).
    DEGRADED mode -> alpha = 1.0 (pure physics).

    Thread-safe via the coherence field's internal lock.
    """

    def __init__(self, config: TornadoNPEConfig | None = None) -> None:
        self.config = config or TornadoNPEConfig()
        self.coherence_field = TornadoCoherenceField(
            alpha=self.config.coherence_alpha,
        )
        self._n_observations: int = 0
        self._lock = RLock()

    # ----- prediction -----

    def predict(
        self,
        *,
        tau: float,
        grad_tau: float,
        torsion: float,
        alignment: float,
        s_over_gamma: float,
        da: float,
        maxllaz: float,
        srh01: float,
        gbt_probability: float,
        storm_id: str | None = None,
    ) -> TornadoPrediction:
        """Fuse analytic and learned predictions.

        Parameters
        ----------
        tau, grad_tau, torsion, alignment, s_over_gamma, da
            Coherence field quantities at the storm location (from Helmholtz).
        maxllaz : float
            Maximum low-level azimuthal shear (s^-1).
        srh01 : float
            0-1 km storm-relative helicity (m^2/s^2).
        gbt_probability : float
            Learned (GBT) tornado probability (0-1).
        storm_id : str | None
            Optional storm identifier for metadata.

        Returns
        -------
        TornadoPrediction
        """
        cfg = self.config
        now = datetime.now(UTC).isoformat()

        # Compute analytic probability
        p_analytic = analytic_tornado_probability(
            tau=tau,
            grad_tau=grad_tau,
            torsion=torsion,
            alignment=alignment,
            s_over_gamma=s_over_gamma,
            da=da,
            maxllaz=maxllaz,
            srh01=srh01,
            config=cfg,
        )

        # Count singularity conditions met
        n_conditions = _count_singularity_conditions(
            grad_tau=grad_tau,
            torsion=torsion,
            alignment=alignment,
            s_over_gamma=s_over_gamma,
            da=da,
            config=cfg,
        )

        # Clamp GBT probability
        p_learned = _clamp01(gbt_probability)

        # Get current system state
        with self._lock:
            n_obs = self._n_observations

        coherence = self.coherence_field.get_coherence_score()
        mode = decide_tornado_mode(
            coherence_score=coherence,
            n_observations=n_obs,
            config=cfg,
        )

        # Compute adaptive fusion weight
        alpha = self._compute_fusion_weight(coherence, mode)

        # Fuse
        if mode == TornadoGatewayMode.DEGRADED or mode == TornadoGatewayMode.REPAIR:
            # Pure physics -- do not trust ML path.
            p_fused = p_analytic
            alpha = 1.0
        else:
            p_fused = alpha * p_analytic + (1.0 - alpha) * p_learned

        # Compute uncertainty
        uncertainty = self._compute_uncertainty(
            p_analytic=p_analytic,
            p_learned=p_learned,
            n_conditions=n_conditions,
            mode=mode,
        )

        meta: dict[str, Any] = {}
        if storm_id is not None:
            meta["storm_id"] = storm_id

        return TornadoPrediction(
            probability=p_fused,
            uncertainty=uncertainty,
            gateway_mode=mode,
            coherence_score=coherence,
            analytic_probability=p_analytic,
            learned_probability=p_learned,
            fusion_weight=alpha,
            singularity_conditions_met=n_conditions,
            timestamp=now,
            meta=meta,
        )

    # ----- outcome recording -----

    def record_outcome(
        self,
        predicted_prob: float,
        actual_tornado: bool,
        storm_id: str | None = None,
        timestamp: str | None = None,
    ) -> TornadoCoherenceSample:
        """Record a prediction outcome for coherence tracking.

        Call this after ground truth is known (typically next day).

        Parameters
        ----------
        predicted_prob : float
            The fused probability that was issued.
        actual_tornado : bool
            Whether a tornado actually occurred.
        storm_id : str | None
            Storm identifier for traceability.
        timestamp : str | None
            When the outcome was recorded.

        Returns
        -------
        TornadoCoherenceSample
        """
        meta: dict[str, Any] = {}
        if storm_id is not None:
            meta["storm_id"] = storm_id

        sample = self.coherence_field.record_observation(
            predicted_prob=predicted_prob,
            actual_tornado=actual_tornado,
            component="detection",
            timestamp=timestamp,
            meta=meta,
        )

        with self._lock:
            self._n_observations += 1

        return sample

    # ----- state -----

    @property
    def n_observations(self) -> int:
        """Number of outcomes recorded."""
        with self._lock:
            return self._n_observations

    @property
    def gateway_mode(self) -> TornadoGatewayMode:
        """Current gateway mode."""
        coherence = self.coherence_field.get_coherence_score()
        with self._lock:
            n_obs = self._n_observations
        return decide_tornado_mode(
            coherence_score=coherence,
            n_observations=n_obs,
            config=self.config,
        )

    # ----- serialization -----

    def to_dict(self) -> dict[str, Any]:
        """Serialize predictor state for persistence."""
        with self._lock:
            return {
                "n_observations": self._n_observations,
                "coherence_field": self.coherence_field.to_dict(),
                "config": {
                    "base_analytic_weight": self.config.base_analytic_weight,
                    "coherence_alpha": self.config.coherence_alpha,
                    "learning_period": self.config.learning_period,
                },
            }

    def save_state(self, path: str) -> None:
        """Save coherence state to a JSON file."""
        import pathlib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_state(self, path: str) -> None:
        """Load coherence state from a JSON file."""
        with open(path) as f:
            d = json.load(f)
        self.coherence_field = TornadoCoherenceField.from_dict(
            d.get("coherence_field", {}),
        )
        with self._lock:
            self._n_observations = int(d.get("n_observations", 0))

    # ----- internal -----

    def _compute_fusion_weight(
        self, coherence: float, mode: TornadoGatewayMode
    ) -> float:
        """Compute adaptive blending weight alpha.

        alpha interpolates from base_analytic_weight (high coherence) to
        max_analytic_weight (low coherence).

            alpha = base + (max - base) * (1 - coherence)

        In DEGRADED/REPAIR modes alpha is forced to 1.0.
        """
        cfg = self.config

        if mode in (TornadoGatewayMode.DEGRADED, TornadoGatewayMode.REPAIR):
            return cfg.max_analytic_weight

        c = _clamp01(coherence)
        alpha = cfg.base_analytic_weight + (
            cfg.max_analytic_weight - cfg.base_analytic_weight
        ) * (1.0 - c)

        return max(cfg.min_analytic_weight, min(cfg.max_analytic_weight, alpha))

    def _compute_uncertainty(
        self,
        p_analytic: float,
        p_learned: float,
        n_conditions: int,
        mode: TornadoGatewayMode,
    ) -> float:
        """Estimate prediction uncertainty.

        Components:
        1. Analytic uncertainty: decreases with more singularity conditions met.
           Fewer conditions -> less constrained -> wider uncertainty.
        2. Learned uncertainty: |p_analytic - p_learned| as a proxy for
           model disagreement.
        3. Mode factor: multiplier that inflates uncertainty in degraded modes.
        """
        cfg = self.config

        # Analytic uncertainty: fewer conditions met -> higher uncertainty
        # 5 conditions met -> 0.05 additional; 0 conditions -> 0.25 additional
        analytic_unc = cfg.base_uncertainty + 0.04 * (5 - n_conditions)

        # Learned uncertainty: disagreement between physics and ML
        disagreement = abs(p_analytic - p_learned)

        # Combined: take the larger signal
        raw_unc = max(analytic_unc, disagreement)

        # Mode factor
        mode_factor = cfg.mode_uncertainty_factor.get(mode.value, 1.0)

        return min(1.0, raw_unc * mode_factor)


# ===================================================================
# 7. TOP-LEVEL ENGINE
# ===================================================================


class TornadoNPEEngine:
    """Top-level orchestrator for the Tornado NPE system.

    Wraps TornadoFusionPredictor with convenience methods for batch
    prediction and diagnostics.

    Usage
    -----
    ::

        engine = TornadoNPEEngine()

        # Make predictions (one storm at a time)
        pred = engine.predict(
            tau=0.5, grad_tau=0.4, torsion=0.08,
            alignment=0.2, s_over_gamma=1.3, da=7.0,
            maxllaz=0.005, srh01=150.0,
            gbt_probability=0.35,
        )
        print(pred.probability, pred.gateway_mode)

        # After ground truth is known
        engine.record_outcome(predicted_prob=pred.probability, actual_tornado=True)

        # Check system health
        snap = engine.health()
        print(snap.coherence_score, snap.per_component_scores)
    """

    def __init__(self, config: TornadoNPEConfig | None = None) -> None:
        self.config = config or TornadoNPEConfig()
        self.predictor = TornadoFusionPredictor(config=self.config)

    def predict(
        self,
        *,
        tau: float,
        grad_tau: float,
        torsion: float,
        alignment: float,
        s_over_gamma: float,
        da: float,
        maxllaz: float,
        srh01: float,
        gbt_probability: float,
        storm_id: str | None = None,
    ) -> TornadoPrediction:
        """Produce a fused tornado prediction for a single storm.

        See TornadoFusionPredictor.predict for parameter details.
        """
        return self.predictor.predict(
            tau=tau,
            grad_tau=grad_tau,
            torsion=torsion,
            alignment=alignment,
            s_over_gamma=s_over_gamma,
            da=da,
            maxllaz=maxllaz,
            srh01=srh01,
            gbt_probability=gbt_probability,
            storm_id=storm_id,
        )

    def predict_batch(
        self,
        storms: list[dict[str, Any]],
    ) -> list[TornadoPrediction]:
        """Predict for multiple storms.

        Parameters
        ----------
        storms : list of dict
            Each dict must contain the keys accepted by ``predict()``.

        Returns
        -------
        list of TornadoPrediction
        """
        return [self.predict(**s) for s in storms]

    def record_outcome(
        self,
        predicted_prob: float,
        actual_tornado: bool,
        storm_id: str | None = None,
        timestamp: str | None = None,
    ) -> TornadoCoherenceSample:
        """Record a prediction outcome.  See TornadoFusionPredictor."""
        return self.predictor.record_outcome(
            predicted_prob=predicted_prob,
            actual_tornado=actual_tornado,
            storm_id=storm_id,
            timestamp=timestamp,
        )

    def health(self) -> TornadoCoherenceSnapshot:
        """Get the current coherence health snapshot."""
        return self.predictor.coherence_field.snapshot()

    @property
    def gateway_mode(self) -> TornadoGatewayMode:
        """Current gateway mode."""
        return self.predictor.gateway_mode

    def save_state(self, path: str) -> None:
        """Persist coherence state to disk."""
        self.predictor.save_state(path)

    def load_state(self, path: str) -> None:
        """Restore coherence state from disk."""
        self.predictor.load_state(path)

    def diagnostics(self) -> dict[str, Any]:
        """Return a diagnostic summary of the engine state."""
        snap = self.health()
        return {
            "gateway_mode": self.gateway_mode.value,
            "n_observations": self.predictor.n_observations,
            "coherence": snap.to_dict(),
            "drift_detected": self.predictor.coherence_field.detect_drift(),
            "spike_detected": self.predictor.coherence_field.detect_spike(),
            "config": {
                "base_analytic_weight": self.config.base_analytic_weight,
                "coherence_alpha": self.config.coherence_alpha,
                "learning_period": self.config.learning_period,
            },
        }


# ===================================================================
# INTERNAL HELPERS
# ===================================================================


def _count_singularity_conditions(
    *,
    grad_tau: float,
    torsion: float,
    alignment: float,
    s_over_gamma: float,
    da: float,
    config: TornadoNPEConfig,
) -> int:
    """Count how many of the 5 singularity conditions are met.

    The 5 conditions are:
    1. S/Gamma > 1       (source exceeds damping)
    2. grad_tau > thresh  (steep coherence gradient)
    3. |torsion| > thresh (rotational focusing active)
    4. Da > thresh        (decay-dominated regime)
    5. alignment > thresh (vorticity locked to coherence gradient)
    """
    count = 0
    if s_over_gamma > 1.0:
        count += 1
    if grad_tau > config.singularity_grad_threshold:
        count += 1
    if abs(torsion) > config.singularity_torsion_threshold:
        count += 1
    if da > config.singularity_da_threshold:
        count += 1
    if alignment > config.singularity_alignment_threshold:
        count += 1
    return count


# ===================================================================
# MODULE EXPORTS
# ===================================================================

__all__ = [
    # Configuration
    "TornadoNPEConfig",
    # Coherence field
    "TornadoCoherenceField",
    "TornadoCoherenceSample",
    "TornadoCoherenceSnapshot",
    # Gateway
    "TornadoGatewayMode",
    "decide_tornado_mode",
    # Analytic model
    "analytic_tornado_probability",
    # Prediction output
    "TornadoPrediction",
    # Fusion predictor
    "TornadoFusionPredictor",
    # Top-level engine
    "TornadoNPEEngine",
]
