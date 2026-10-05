"""The HazardPulse publish-gate engine.

Each gate evaluates one concrete, safety-relevant property of a forecast and
returns ``pass`` / ``degrade`` / ``block``. The engine runs them all and
aggregates to the worst outcome:

  pass    -> publish normally
  degrade -> publish with a constrained presentation + explicit warning
  block   -> do not publish forecast artifacts to public surfaces

Implemented gates (Truth-Surface spec §10):
  G0_SCHEMA_VALIDITY      probability/intervals are well-formed numbers
  G1_SOURCE_FRESHNESS     input data is recent enough for the hazard cadence
  G3_PROVENANCE_COMPLETE  model lineage + input hash + receipt + replay all present
  G4_CALIBRATION_FLOOR    the model's measured calibration is within band
  G5_SPATIOTEMPORAL_SANITY coords valid, prob in [0,1], interval ordered, cell granularity
  G6_ALERT_HARM_GUARD     no confident high-risk number on OOD / wide-uncertainty evidence
  G8_REPLAYABILITY_REQUIRED a replay artifact exists so the forecast can be re-run
  G9_PUBLIC_PROJECTION_POLICY  no reserved official-warning terminology

The design is data-light and pure so it is fully unit-testable and so the live
scorers and ``build_site_artifacts`` can call one engine instead of duplicating
ad-hoc stamping.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

PASS = "pass"
DEGRADE = "degrade"
BLOCK = "block"

_SEVERITY = {PASS: 0, DEGRADE: 1, BLOCK: 2}
GATE_SET_VERSION = "hp-gates-v1"

# Terms reserved for official agencies (18 U.S.C. §1038); never in our labels.
_RESERVED_TERMS = ("warning", "alert", "imminent", "evacuate", "take cover")


@dataclass(frozen=True)
class GateConfig:
    """Tunable thresholds for the gate spine (per hazard)."""
    freshness_soft_seconds: float = 12 * 3600.0
    freshness_hard_seconds: float = 36 * 3600.0
    ece_warn: float = 0.10
    ece_block: float = 0.25
    bss_warn: float = 0.0          # below this BSS -> degrade (worse than climatology)
    bss_block: float = -0.50       # catastrophically miscalibrated -> block
    min_cell_deg: float = 2.0      # public-granularity floor (no neighborhood claims)
    high_risk_prob: float = 0.50   # alert-harm guard trigger


_HAZARD_CONFIG = {
    "earthquake": GateConfig(freshness_soft_seconds=12 * 3600, freshness_hard_seconds=36 * 3600,
                             min_cell_deg=2.0),
    # A 60-minute forecast, issued every 2 h (every 30 min while risk is elevated; scripts/ci/schedule.py),
    # its age counted from its DATA's valid time: past 2.5 h a scheduled run has been missed (degrade);
    # past 6 h -- six of its own windows, and site.hazards.TORNADO's overdue age -- it is not published.
    # (These were 6 h / 12 h: a 60-minute forecast eleven hours old passed with a warning.)
    "tornado": GateConfig(freshness_soft_seconds=2.5 * 3600, freshness_hard_seconds=6 * 3600,
                          min_cell_deg=2.0),
    "hurricane": GateConfig(freshness_soft_seconds=24 * 3600, freshness_hard_seconds=48 * 3600,
                            min_cell_deg=0.0),  # storm-relative, not gridded
}


def default_config_for(hazard: str | None) -> GateConfig:
    return _HAZARD_CONFIG.get(str(hazard), GateConfig())


@dataclass
class GateContext:
    """Everything the gates need about one forecast. All optional -> a missing
    input is treated conservatively (a gate that cannot confirm a safety property
    degrades or blocks rather than silently passing)."""
    hazard: str | None = None
    forecast_id: str | None = None
    # provenance
    model_version: str | None = None
    model_sha256: str | None = None
    input_sha256: str | None = None
    receipt_sha256: str | None = None
    replay_artifact: str | None = None
    # forecast values
    probability: float | None = None
    confidence_lo: float | None = None
    confidence_hi: float | None = None
    abstained: bool = False
    ood_flag: bool = False
    uncertainty_class: str | None = None
    # geometry
    lat: float | None = None
    lon: float | None = None
    cell_size_deg: float | None = None
    # freshness
    issued_at: str | None = None
    data_age_seconds: float | None = None
    # calibration health (model-level, from the verification rollup)
    ece: float | None = None
    brier_skill_score: float | None = None
    calibration_known: bool = False
    # presentation
    risk_label: str | None = None


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    outcome: str
    reason: str | None = None


@dataclass
class GateDecision:
    gate_decision_id: str
    forecast_id: str | None
    hazard: str | None
    decision: str
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    gate_set_version: str = GATE_SET_VERSION
    emitted_at: str | None = None
    artifact_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gate_decision_id": self.gate_decision_id,
            "forecast_id": self.forecast_id,
            "hazard": self.hazard,
            "gate_set_version": self.gate_set_version,
            "decision": self.decision,
            "reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "emitted_at": self.emitted_at,
            "artifact_refs": list(self.artifact_refs),
            "gates": [{"gate_id": g.gate_id, "outcome": g.outcome, "reason": g.reason}
                      for g in self.gate_results],
        }


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------- #
# Individual gates
# --------------------------------------------------------------------------- #
def g0_schema_validity(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G0_SCHEMA_VALIDITY"
    if ctx.abstained:
        return GateResult(gid, PASS)
    if ctx.probability is None:
        return GateResult(gid, BLOCK, "probability missing on a non-abstained forecast")
    if not _is_number(ctx.probability):
        return GateResult(gid, BLOCK, "probability is not a finite number")
    for name, v in (("confidence_lo", ctx.confidence_lo), ("confidence_hi", ctx.confidence_hi)):
        if v is not None and not _is_number(v):
            return GateResult(gid, BLOCK, f"{name} is not a finite number")
    return GateResult(gid, PASS)


def g1_source_freshness(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G1_SOURCE_FRESHNESS"
    age = ctx.data_age_seconds
    if age is None:
        return GateResult(gid, DEGRADE, "source data age unknown")
    if age > cfg.freshness_hard_seconds:
        return GateResult(gid, BLOCK,
                          f"source data is stale ({age/3600:.1f}h > {cfg.freshness_hard_seconds/3600:g}h)")
    if age > cfg.freshness_soft_seconds:
        return GateResult(gid, DEGRADE,
                          f"source data is aging ({age/3600:.1f}h > {cfg.freshness_soft_seconds/3600:g}h)")
    return GateResult(gid, PASS)


def g3_provenance_complete(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G3_PROVENANCE_COMPLETE"
    # Model lineage must always be pinned (always available today) -> hard block.
    if not ctx.model_version:
        return GateResult(gid, BLOCK, "model_version missing")
    # Cryptographic provenance (signed re-runnable receipt) -> degrade if absent,
    # so the trust layer can roll out across scorers without darking the site;
    # clears to pass once a forecast carries its receipt + hashes.
    crypto_missing = [name for name, v in (
        ("model_sha256", ctx.model_sha256),
        ("input_sha256", ctx.input_sha256),
        ("receipt_sha256", ctx.receipt_sha256),
    ) if not v]
    if crypto_missing:
        return GateResult(gid, DEGRADE,
                          "cryptographic provenance incomplete: " + ", ".join(crypto_missing))
    return GateResult(gid, PASS)


def g4_calibration_floor(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G4_CALIBRATION_FLOOR"
    if not ctx.calibration_known:
        return GateResult(gid, DEGRADE, "model calibration not yet measured")
    if ctx.ece is not None and _is_number(ctx.ece):
        if ctx.ece > cfg.ece_block:
            return GateResult(gid, BLOCK, f"calibration error too high (ECE {ctx.ece:.3f} > {cfg.ece_block})")
        if ctx.ece > cfg.ece_warn:
            return GateResult(gid, DEGRADE, f"calibration error elevated (ECE {ctx.ece:.3f} > {cfg.ece_warn})")
    if ctx.brier_skill_score is not None and _is_number(ctx.brier_skill_score):
        if ctx.brier_skill_score < cfg.bss_block:
            return GateResult(gid, BLOCK,
                              f"Brier skill far below climatology (BSS {ctx.brier_skill_score:.3f})")
        if ctx.brier_skill_score < cfg.bss_warn:
            return GateResult(gid, DEGRADE,
                              f"Brier skill below climatology (BSS {ctx.brier_skill_score:.3f})")
    return GateResult(gid, PASS)


def g5_spatiotemporal_sanity(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G5_SPATIOTEMPORAL_SANITY"
    if ctx.lat is not None and (not _is_number(ctx.lat) or not (-90.0 <= ctx.lat <= 90.0)):
        return GateResult(gid, BLOCK, f"latitude out of range ({ctx.lat})")
    if ctx.lon is not None and (not _is_number(ctx.lon) or not (-180.0 <= ctx.lon <= 180.0)):
        return GateResult(gid, BLOCK, f"longitude out of range ({ctx.lon})")
    if not ctx.abstained and ctx.probability is not None and not (0.0 <= ctx.probability <= 1.0):
        return GateResult(gid, BLOCK, f"probability out of [0,1] ({ctx.probability})")
    lo, hi = ctx.confidence_lo, ctx.confidence_hi
    if lo is not None and hi is not None and _is_number(lo) and _is_number(hi) and lo > hi:
        return GateResult(gid, BLOCK, f"confidence interval inverted ({lo} > {hi})")
    if (cfg.min_cell_deg > 0 and ctx.cell_size_deg is not None
            and _is_number(ctx.cell_size_deg) and ctx.cell_size_deg < cfg.min_cell_deg):
        return GateResult(gid, DEGRADE,
                          f"cell finer than public-granularity floor ({ctx.cell_size_deg}deg < {cfg.min_cell_deg}deg)")
    return GateResult(gid, PASS)


def g6_alert_harm_guard(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G6_ALERT_HARM_GUARD"
    if ctx.abstained or ctx.probability is None:
        return GateResult(gid, PASS)
    if ctx.probability >= cfg.high_risk_prob:
        if ctx.ood_flag:
            return GateResult(gid, DEGRADE, "high-risk probability on out-of-distribution input")
        if ctx.uncertainty_class == "wide":
            return GateResult(gid, DEGRADE, "high-risk probability with wide uncertainty")
    return GateResult(gid, PASS)


def g8_replayability_required(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G8_REPLAYABILITY_REQUIRED"
    if not ctx.replay_artifact:
        return GateResult(gid, BLOCK, "no replay artifact: forecast cannot be re-run")
    return GateResult(gid, PASS)


def g9_public_projection_policy(ctx: GateContext, cfg: GateConfig) -> GateResult:
    gid = "G9_PUBLIC_PROJECTION_POLICY"
    label = (ctx.risk_label or "").lower()
    for term in _RESERVED_TERMS:
        if term in label:
            return GateResult(gid, BLOCK, f"risk label uses reserved official term '{term}'")
    return GateResult(gid, PASS)


_GATES = (
    g0_schema_validity,
    g1_source_freshness,
    g3_provenance_complete,
    g4_calibration_floor,
    g5_spatiotemporal_sanity,
    g6_alert_harm_guard,
    g8_replayability_required,
    g9_public_projection_policy,
)


class GateEngine:
    """Run the gate spine on a forecast context and aggregate to one decision."""

    def __init__(self, gates=_GATES):
        self.gates = tuple(gates)

    def evaluate(self, ctx: GateContext, *, config: GateConfig | None = None,
                 emitted_at: str | None = None) -> GateDecision:
        cfg = config or default_config_for(ctx.hazard)
        results = [gate(ctx, cfg) for gate in self.gates]
        worst = max((_SEVERITY[r.outcome] for r in results), default=0)
        decision = {0: PASS, 1: DEGRADE, 2: BLOCK}[worst]
        blocking = [r.reason for r in results if r.outcome == BLOCK and r.reason]
        warnings = [r.reason for r in results if r.outcome == DEGRADE and r.reason]
        refs = [ref for ref in (ctx.replay_artifact,) if ref]
        return GateDecision(
            gate_decision_id=f"gdec_{ctx.forecast_id}" if ctx.forecast_id else "gdec_unknown",
            forecast_id=ctx.forecast_id,
            hazard=ctx.hazard,
            decision=decision,
            blocking_reasons=blocking,
            warnings=warnings,
            gate_results=results,
            emitted_at=emitted_at,
            artifact_refs=refs,
        )
