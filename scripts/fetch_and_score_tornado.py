# Independent hazard intelligence platform.
# Always follow official NWS tornado warnings.
# See weather.gov for authoritative data.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

#!/usr/bin/env python3
"""Fetch ProbSevere + HRRR, score with coherence model, output JSON.

Designed to run every 15 minutes via GitHub Actions cron during severe
season (March -- September).  Outputs:

  - dist/data/live-tornadoes.json   (scored active storms)
  - dist/data/live-pulse.json       (updated tornado entry)
  - dist/data/tornado-ledger.jsonl  (append-only SHA-256 prediction chain)

If no ProbSevere data is available (e.g., off-season), writes empty state
with honest timestamps and exits cleanly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math as _math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _json_safe(obj):
    """JSON default handler: convert numpy types to Python."""
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if (_math.isnan(v) or _math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON output."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if (_math.isnan(v) or _math.isinf(v)) else v
    return obj

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_site_artifacts import build_site_artifacts

from hazardpulse.data.hrrr import (  # noqa: E402
    DX_KM,
    GRID_DLAT,
    GRID_DLON,
    GRID_LATS,
    GRID_LONS,
    HRRR_N_LAT,
    HRRR_N_LON,
    LAT_MIN,
    LON_MIN,
    fetch_hrrr_grid,
    load_cached_hrrr,
)
from hazardpulse.data.probsevere import (  # noqa: E402
    fetch_probsevere_day,
    load_cached_probsevere,
)
from hazardpulse.tornado.coherence_engine import (  # noqa: E402
    compute_coherence_fields,
    compute_derived_hrrr,
    extract_coherence_at_point,
    gaussian_smooth_2d,
    solve_helmholtz_2d,
    test_singularity_at_point,
)
try:
    from hazardpulse.tornado.operational_storm import (  # noqa: E402
        ALL_NAMES_FULL,
        GradientBoostedTrees,
        MetaStacker,
        TornadoStormConfig,
        build_storm_features,
        compute_auc,
        extract_block_a,
        extract_block_a_from_probsevere,
        extract_block_e,
        extract_block_s,
        extract_block_t,
        logistic_predict,
        logistic_train,
        predict_tornado_probability,
        sigmoid,
    )
    HAS_OPERATIONAL = True
except ImportError as _op_imp_err:
    HAS_OPERATIONAL = False
    # Print loudly so CI logs flag this — silently disabling the legacy ML
    # path has bitten us before.
    print(
        f"  WARNING: operational_storm module unavailable ({_op_imp_err}). "
        "Legacy tier1_ml path disabled; scoring will rely on pre-trained GBT "
        "only.",
        file=sys.stderr,
    )

from hazardpulse.tornado.tornado_npe import (  # noqa: E402
    analytic_tornado_probability,
)
from hazardpulse.tornado.definitive_model import (  # noqa: E402
    ALL_FEATURE_NAMES_FULL as DEFINITIVE_FEATURE_NAMES,
    LEGACY_MODEL_VERSION as DEFINITIVE_LEGACY_MODEL_VERSION,
    MAX_ANALYSIS_AGE_H as DEFINITIVE_MAX_ANALYSIS_AGE_H,
    build_storm_history as definitive_extract_history,
    model_version_of_payload as definitive_model_version,
    extract_block_c as definitive_extract_c,
    extract_block_e as definitive_extract_e,
    extract_block_h as definitive_extract_h,
    extract_block_p as definitive_extract_p,
    index_storms_by_id as definitive_index_storms,
    predict_proba_from_payload,
    probsevere_step_minutes as definitive_step_minutes,
    parse_probsevere_valid_time as definitive_parse_valid_time,
)
from hazardpulse.data.hrrr_availability import live_candidates as hrrr_live_candidates  # noqa: E402
from hazardpulse.tornado import lgbm_payload as v3_payload  # noqa: E402
from hazardpulse.tornado.v3_serving import V3Suite, storm_history as v3_storm_history  # noqa: E402
from hazardpulse.verification import served_evidence  # noqa: E402
try:
    from hazardpulse.verification import nws_live  # noqa: E402
    HAS_NWS_LIVE = True
except ImportError as _nws_imp_err:  # never silently: the +W model then cannot serve
    HAS_NWS_LIVE = False
    print(f"  WARNING: nws_live unavailable ({_nws_imp_err}); v3 serves without NWS warnings.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
LEDGER_PATH = DIST / "data" / "tornado-ledger.jsonl"

# Identity of the served model, bound to its exact weights (see
# definitive_model.model_version_of_payload): every forecast, the prospective
# calibration data and the trust-layer calibrator carry it, and a calibrator
# is only ever applied to forecasts from the model it was fitted on.
_SERVED_WEIGHTS = RESULTS / "models" / "tornado_gbt_v1.json"
# The v3 suite (docs/TORNADO_MODEL_PROGRAM.md) is served whenever its payloads are present; v2
# stays as the legacy path. Each storm carries the identity of the model that scored it; the
# page-level MODEL_VERSION is the suite's headline model (the +W model, else the fallback).
V3_SUITE = V3Suite.load(RESULTS / "models")
if V3_SUITE.available:
    MODEL_VERSION = v3_payload.model_version(V3_SUITE.main or V3_SUITE.fallback)
else:
    MODEL_VERSION = (
        definitive_model_version(_SERVED_WEIGHTS)
        if _SERVED_WEIGHTS.exists() else DEFINITIVE_LEGACY_MODEL_VERSION
    )
PRIMARY_DOMAIN = "https://hazardpulse.com"
SITE_PUBLISHER_NAME = "HazardPulse"
# ---------------------------------------------------------------------------
# Risk band
# ---------------------------------------------------------------------------


def _risk_band(prob: float) -> str:
    """Map probability to risk band.

    Band names deliberately avoid NWS terminology (e.g. "watch", "warning")
    to prevent confusion with official NWS products.
    """
    if prob >= 0.50:
        return "very_high"
    if prob >= 0.30:
        return "high"
    if prob >= 0.15:
        return "moderate"
    if prob >= 0.05:
        return "low"
    return "minimal"


_EVIDENCE_CACHE: dict = {}


def tornado_evidence() -> dict | None:
    """The SERVED model's read-once test, from served_evidence (bound to the payload's bytes) --
    never typed into a page and never another model's file. (The pages used to quote "AUC 0.894",
    then v2's 2024 results while v3 was serving.)"""
    if "to" not in _EVIDENCE_CACHE:
        try:
            _EVIDENCE_CACHE["to"] = served_evidence.tornado_evidence()
        except served_evidence.EvidenceError as exc:
            print(f"  Warning: tornado evidence not bound: {exc}")
            _EVIDENCE_CACHE["to"] = None
    return _EVIDENCE_CACHE["to"]


def refresh_risk_bands(scored: list[dict]) -> list[dict]:
    """Re-derive each storm's band from the probability it is PUBLISHED with.

    The band was computed from the pre-calibration score and never refreshed,
    so the trust layer's calibrated probability shipped under a stale band
    (live 2026-10-01: 0.57% labelled "high"; 77% of calibrated storms carried
    a band that contradicted their own probability).
    """
    for s in scored:
        s["risk_band"] = _risk_band(float(s.get("tornado_probability", 0.0)))
    return scored


BAND_WITHHELD_REASON = "the band does not contain the probability published beside it"


def withhold_bands_excluding_probability(scored: list[dict]) -> int:
    """Publish a storm's ``[confidence_lo, confidence_hi]`` only when it CONTAINS the probability published
    beside it; otherwise both are null and ``band_withheld`` says why. Returns how many were withheld.

    The v3 band is the Venn-Abers pair of the 60-min model (two calibrated estimates from leave-one-year-out
    scores), while the published 60-min probability is the payload's own Platt calibration: two different
    calibrations, so nothing keeps one inside the other (the raw p60 lay outside its own pair for 971 of the
    1,602 storm forecasts in the v3 records to 2026-10-05 02:05Z). Once PR #22 stopped the inflating calibrator, the
    02:05Z run of 2026-10-05 published a headline of 0.06% beside a "range" of 0.07%-0.07% (and the pulse
    failed tests/test_site_integrity.py, which deploys gate on). The pair itself stays in the record, at full
    precision, as ``v3.probability_band_60min``; only the published band is withheld.
    """
    withheld = 0
    for s in scored:
        lo, hi = s.get("confidence_lo"), s.get("confidence_hi")
        if lo is None or hi is None:
            continue
        p = s.get("tornado_probability")
        try:
            ok = p is not None and float(lo) <= float(p) <= float(hi)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            s["confidence_lo"] = s["confidence_hi"] = None
            s["band_withheld"] = BAND_WITHHELD_REASON
            withheld += 1
    return withheld


# ---------------------------------------------------------------------------
# Storm tracking from ProbSevere time steps
# ---------------------------------------------------------------------------


def build_storm_tracks(
    time_steps: list[dict],
) -> dict[int, list[tuple[int, str, dict]]]:
    """Group storms by ID across time steps to build tracks.

    Returns dict: storm_id -> list of (time_step_idx, valid_time, storm_dict).
    """
    tracks: dict[int, list[tuple[int, str, dict]]] = defaultdict(list)
    for ts_idx, ts in enumerate(time_steps):
        valid_time = ts.get("valid_time", "")
        for storm in ts.get("storms", []):
            sid = storm.get("id", 0)
            if sid == 0:
                continue
            tracks[sid].append((ts_idx, valid_time, storm))
    return dict(tracks)


# ---------------------------------------------------------------------------
# Model loading and analytic scoring
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km."""
    import math
    R = 6371.0  # Earth radius in km
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_coherence_from_probsevere(
    storms: list[dict],
) -> dict[str, np.ndarray] | None:
    """Build coherence fields from ProbSevere storm-embedded atmospheric data.

    When HRRR data is unavailable, each ProbSevere storm still carries
    MUCAPE, SRH01, EBSHEAR at its location.  This function interpolates
    those sparse point values onto the 80 km CONUS grid, builds a
    source term S, and solves the Helmholtz PDE to produce coherence
    fields identical in structure to the HRRR-derived ones.

    Returns None if no storms have usable atmospheric data.
    """
    if not storms:
        return None

    cape_field = np.zeros((HRRR_N_LAT, HRRR_N_LON), dtype=np.float32)
    srh_field = np.zeros_like(cape_field)
    shear_field = np.zeros_like(cape_field)
    count_field = np.zeros_like(cape_field)

    for storm in storms:
        lat = float(storm.get("lat", 0) or 0)
        lon = float(storm.get("lon", 0) or 0)
        i = int((lat - LAT_MIN) / GRID_DLAT)
        j = int((lon - LON_MIN) / GRID_DLON)
        if 0 <= i < HRRR_N_LAT and 0 <= j < HRRR_N_LON:
            cape_field[i, j] += float(storm.get("mucape", 0) or 0)
            srh_field[i, j] += float(storm.get("srh01", 0) or 0)
            shear_field[i, j] += float(storm.get("ebshear", 0) or 0)
            count_field[i, j] += 1

    # Average where multiple storms overlap
    mask = count_field > 0
    if not np.any(mask):
        return None
    cape_field[mask] /= count_field[mask]
    srh_field[mask] /= count_field[mask]
    shear_field[mask] /= count_field[mask]

    # Build source term: S = CAPE/2000 + 0.3*|SRH|/200 + 0.2*shear/25
    S_field = (
        cape_field / 2000.0
        + 0.3 * np.abs(srh_field) / 200.0
        + 0.2 * shear_field / 25.0
    ).astype(np.float32)

    # Smooth the sparse source field so the PDE has spatial structure
    S_field = gaussian_smooth_2d(S_field, sigma_cells=3.0)

    # Damping: uniform moderate value (no CIN available from ProbSevere)
    Gamma_field = np.full_like(S_field, 0.25, dtype=np.float32)

    # Diffusivity: uniform
    D_field = np.ones_like(S_field, dtype=np.float32)

    # Solve Helmholtz PDE: D nabla^2 tau - Gamma tau + S = 0 (Gamma goes in
    # directly; kappa = sqrt(Gamma/D) is derived-only — FVCS W-2)
    tau = solve_helmholtz_2d(S_field, Gamma_field, dx=1.0, D=D_field)

    # Spatial derivatives
    from hazardpulse.tornado.coherence_engine import (
        TORSION_SINGULARITY_THRESHOLD,
        _gradient_2d,
    )

    grad_y, grad_x = _gradient_2d(tau)
    grad_tau = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)

    # Torsion is the tilting coupling (S_01 x grad tau).k and needs the shear
    # VECTOR; ProbSevere carries only the effective-shear MAGNITUDE, so on this
    # fallback path torsion is undefined and is reported as exactly zero (the
    # old shear * curl(tau) was zero too, but by accident -- the curl of a
    # gradient vanishes identically).
    torsion = np.zeros_like(tau, dtype=np.float32)

    # Alignment: use SRH as proxy for shear direction alignment
    grad_mag_safe = grad_tau + 1e-6
    alignment = (np.abs(srh_field) * grad_tau / 200.0).astype(np.float32)

    # S / Gamma ratio
    S_over_Gamma = (S_field / np.maximum(Gamma_field, 0.01)).astype(
        np.float32
    )

    # Damkohler
    Da = (
        Gamma_field * (DX_KM ** 2) / np.maximum(D_field * 100.0, 1e-6)
    ).astype(np.float32)

    # E_coh: simplified (no T_sfc available)
    E_coh = np.zeros_like(tau)

    # Singularity count
    cond1 = (S_over_Gamma > 1.0).astype(np.float32)
    cond2 = (grad_tau > 0.5).astype(np.float32)
    cond3 = (np.abs(torsion) > TORSION_SINGULARITY_THRESHOLD).astype(np.float32)
    cond4 = (alignment > 0).astype(np.float32)
    cond5 = (Da > 10.0).astype(np.float32)
    singularity_count = (cond1 + cond2 + cond3 + cond4 + cond5).astype(
        np.float32
    )

    return {
        "tau": tau,
        "grad_tau": grad_tau,
        "torsion": torsion,
        "alignment": alignment,
        "S_field": S_field,
        "Gamma_field": Gamma_field,
        "S_over_Gamma": S_over_Gamma,
        "Da": Da,
        "E_coh": E_coh,
        "singularity_count": singularity_count,
    }


def live_analysis_candidates(now: dt.datetime) -> list[tuple[str, int]]:
    """HRRR analyses a live run may use, newest first -- the ONE availability rule training uses
    (hazardpulse.data.hrrr_availability): 3-hourly, published (valid + 100 min) before ``now``.
    Until 2026-10-02 this tried every whole hour back to 3 h, so a live run and a training row
    could read different analyses, and training read some not yet published (amendment 8)."""
    return hrrr_live_candidates(now)


def load_pretrained_model() -> dict | None:
    """Try to load a pre-trained model from results/models/tornado_gbt.json.

    Returns the model dict if found and loadable, otherwise None.
    """
    model_path = RESULTS / "models" / "tornado_gbt.json"
    if not model_path.exists():
        # Also check the old location
        model_path = RESULTS / "tornado_storm_model.json"
    if not model_path.exists():
        return None
    try:
        model_data = json.loads(model_path.read_text(encoding="utf-8"))
        if not HAS_OPERATIONAL:
            print("  Warning: operational_storm module not available, cannot use ML model")
            return None
        print(f"  Loaded pre-trained model from {model_path.name}")
        return model_data
    except Exception as e:
        print(f"  Warning: Failed to load model: {e}")
        return None


# ---------------------------------------------------------------------------
# Pre-trained GBT model loading (from definitive_model.save_model output)
# ---------------------------------------------------------------------------

PRETRAINED_GBT_PATH = RESULTS / "models" / "tornado_gbt_v1.json"


def load_pretrained_gbt() -> dict | None:
    """Load the pre-trained GBT model saved by definitive_model --save-model.

    Returns a dict with keys 'model_data', 'feature_names', 'normalization'
    if found, otherwise None. Falls through to Tier 2/3 gracefully.
    """
    if not PRETRAINED_GBT_PATH.exists():
        return None
    try:
        with open(PRETRAINED_GBT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("model_format") != "hazardpulse_gbt_v1":
            print(f"  Warning: Unknown model format in {PRETRAINED_GBT_PATH.name}")
            return None
        print(f"  Loaded pre-trained GBT ({data['n_trees']} trees, "
              f"{len(data['feature_names'])} features) from {PRETRAINED_GBT_PATH.name}")
        return data
    except Exception as e:
        print(f"  Warning: Failed to load pre-trained GBT: {e}")
        return None


def predict_with_pretrained(
    gbt_data: dict,
    raw_features: dict[str, float],
) -> float:
    """Score a single storm using the pre-trained GBT model.

    Parameters
    ----------
    gbt_data : dict
        Model payload from load_pretrained_gbt().
    raw_features : dict
        Feature name -> raw (unnormalized) value for this storm.

    Returns
    -------
    float
        Predicted tornado probability in [0, 1] -- calibrated to the real
        storm-object base rate when the payload carries a calibration (see
        ``definitive_model.predict_proba_from_payload``, the one scoring path
        shared with training-time evaluation). Use
        ``pretrained_is_calibrated`` to tell the two apart.
    """
    prob, _calibrated = predict_proba_from_payload(gbt_data, raw_features)
    return prob


def pretrained_is_calibrated(gbt_data: dict) -> bool:
    """True when the payload's probabilities are mapped to the real base rate.

    Payloads saved before 2026-10-01 have no calibration: they were trained
    with class-balanced weights on 5:1 downsampled data, so their sigmoid is a
    probability at a 50/50 prior and overstates the real-world odds ~50-100x.
    """
    return gbt_data.get("calibration") is not None


def score_storm_analytic(
    storm: dict,
    coherence_fields: dict[str, np.ndarray] | None,
) -> float:
    """Score a single storm using analytic probability (no ML needed).

    Uses coherence field theory + ProbSevere observational signals.
    Falls back to zero coherence fields if HRRR is unavailable.
    """
    lat = float(storm.get("lat", 0))
    lon = float(storm.get("lon", 0))

    # Extract coherence at storm location
    if coherence_fields is not None:
        coh = extract_coherence_at_point(coherence_fields, lat, lon)
        tau = coh.get("tau", 0)
        grad_tau = coh.get("grad_tau", 0)
        torsion = coh.get("torsion", 0)
        alignment = coh.get("alignment", 0)
        s_over_gamma = coh.get("S_over_Gamma", 0)
        da = coh.get("Da", 0)
    else:
        tau = grad_tau = torsion = alignment = s_over_gamma = da = 0.0

    maxllaz = float(storm.get("maxllaz", 0))
    srh01 = float(storm.get("srh01", 0))

    prob = analytic_tornado_probability(
        tau=tau, grad_tau=grad_tau, torsion=torsion,
        alignment=alignment, s_over_gamma=s_over_gamma, da=da,
        maxllaz=maxllaz, srh01=srh01,
    )
    return float(prob)


# ---------------------------------------------------------------------------
# Score storms
# ---------------------------------------------------------------------------


def score_storms(
    time_steps: list[dict],
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
    model: dict | None,
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
    coherence_source: str = "none",
    pretrained_gbt: dict | None = None,
    v3_suite: V3Suite | None = None,
    hrrr_label: str | None = None,
) -> list[dict]:
    """Score all active storms from the latest ProbSevere time step.

    scoring_tier controls which scoring method is used:
      - "tier1_v3": the v3 suite (hazardpulse.tornado.v3_serving): 60-min probability from the
        +W model when the live NWS warnings feed answered (else the no-warnings model), 30/90-min
        and EF2+ products, a Venn-Abers band and the storm's top drivers
      - "tier1_ml": Full ML model (pre-trained GBT)
      - "tier2_analytic": Analytic coherence probability (no ML)
      - "tier3_ps_only": ProbSevere raw scores only (minimal fallback)

    Returns list of scored storm dicts ready for JSON output.
    """
    if not time_steps:
        return []

    # Use the latest time step
    latest_ts = time_steps[-1]
    storms = latest_ts.get("storms", [])
    valid_time = latest_ts.get("valid_time", now.isoformat() + "Z")

    if not storms:
        return []

    # Build tracks for history
    tracks = build_storm_tracks(time_steps)
    # Block E must see exactly what training saw: the bounded lookback of
    # definitive_extract_history and the day's real step cadence (train==serve).
    latest_idx = len(time_steps) - 1
    id_index = definitive_index_storms(time_steps)
    step_minutes = definitive_step_minutes(time_steps)

    # Pre-compute derived HRRR grids once (needed for Block H feature extraction)
    derived_hrrr: dict[str, np.ndarray] = {}
    hrrr_usable = False
    if hrrr is not None and pretrained_gbt is not None:
        # Guard against all-NaN HRRR (stale Zarr bucket → meaningless ML output)
        nan_ratios = []
        for arr in hrrr.values():
            if isinstance(arr, np.ndarray) and arr.size > 0:
                nan_ratios.append(float(np.isnan(arr).mean()))
        mean_nan = float(np.mean(nan_ratios)) if nan_ratios else 1.0
        # 5% NaN threshold — HRRR analysis is essentially gap-free (0% NaN
        # on a healthy fetch). Anything above a couple percent indicates a
        # partial pull that will feed garbage into the GBT trees.
        if mean_nan > 0.05:
            print(
                f"  Warning: HRRR is {mean_nan:.1%} NaN (threshold 5%) — "
                "disabling tier1_ml GBT, falling back to tier2_analytic."
            )
            hrrr_usable = False
        else:
            try:
                derived_hrrr = compute_derived_hrrr(hrrr)
                hrrr_usable = True
            except Exception as exc:
                print(f"  Warning: compute_derived_hrrr failed: {exc}")
                derived_hrrr = {}
                hrrr_usable = False

    # v3: the HRRR environment is optional (measured on 2025: AUC 0.958 without it vs 0.971 with
    # it, ProbTor 0.879), the NWS warning state is fetched once for every storm of the step
    v3_derived: dict | None = None
    v3_warnings = None
    v3_warning_error: str | None = None
    if scoring_tier == "tier1_v3" and v3_suite is not None:
        if hrrr is not None:
            try:
                v3_derived = compute_derived_hrrr(hrrr)
            except Exception as exc:
                print(f"  Warning: compute_derived_hrrr failed ({exc}); v3 scores without HRRR")
        obs_time = definitive_parse_valid_time(valid_time) if isinstance(valid_time, str) else None
        if HAS_NWS_LIVE and obs_time is not None and v3_suite.main is not None:
            live_storms = [s for s in storms if s.get("id", 0) != 0]
            try:
                w = nws_live.live_tor_warning_inputs(
                    np.array([float(s.get("lat", 0)) for s in live_storms]),
                    np.array([float(s.get("lon", 0)) for s in live_storms]),
                    [obs_time] * len(live_storms))
                v3_warnings = {s.get("id"): (float(a), float(m)) for s, (a, m) in zip(live_storms, w.matrix())}
                print(f"  NWS warnings: {int(sum(a for a, _ in v3_warnings.values()))} of "
                      f"{len(v3_warnings)} storms inside an active tornado warning")
            except nws_live.NwsLiveError as exc:
                v3_warning_error = f"{type(exc).__name__}: {exc}"
                print(f"  Warning: NWS warnings feed failed ({v3_warning_error}); serving the no-warnings model")
        elif v3_suite.main is not None:
            v3_warning_error = "nws_live unavailable" if not HAS_NWS_LIVE else "no observation time"

    scored: list[dict] = []
    for storm in storms:
        sid = storm.get("id", 0)
        if sid == 0:
            continue

        lat = float(storm.get("lat", 0))
        lon = float(storm.get("lon", 0))

        # Get storm history
        track = tracks.get(sid, [])
        history = [t[2] for t in track]

        # --- Three-tier scoring ---
        top_features: list = []
        model_scores: dict = {}
        coherence_score: float = 0.0
        v3_out: dict | None = None

        if scoring_tier == "tier1_v3" and v3_suite is not None:
            v3_out = v3_suite.score_storm(
                storm,
                v3_storm_history(time_steps, sid, latest_idx, id_index=id_index),
                step_minutes,
                hrrr if v3_derived is not None else None,
                v3_derived,
                None if v3_warnings is None else v3_warnings.get(sid),
            )
            prob = round(min(max(v3_out["p60"], 0.0), 0.99), 4)
            risk = _risk_band(prob)
            model_scores = {"v3_p60": prob, "calibrated": True, "model": v3_out["model"]}
            top_features = [{"name": d["label"], "value": None if d["value"] is None else round(d["value"], 4),
                             "log_odds": d["log_odds"]} for d in v3_out.get("drivers", [])]
        elif (
            scoring_tier == "tier1_ml"
            and pretrained_gbt is not None
            and hrrr is not None
            and coherence_fields is not None
            and hrrr_usable
        ):
            # Tier 1 (pre-trained GBT): build 41-feature vector then score.
            try:
                f_p = definitive_extract_p(storm)
                f_e = definitive_extract_e(
                    storm,
                    definitive_extract_history(time_steps, sid, latest_idx, id_index=id_index),
                    step_minutes,
                )
                f_h = definitive_extract_h(storm, hrrr, derived_hrrr)
                f_c = definitive_extract_c(storm, coherence_fields, hrrr)
                full_vec = np.concatenate([f_p, f_e, f_h, f_c])
                raw_features = {
                    name: float(full_vec[i])
                    for i, name in enumerate(DEFINITIVE_FEATURE_NAMES)
                }
                prob = predict_with_pretrained(pretrained_gbt, raw_features)
                prob = round(min(max(prob, 0.0), 0.99), 4)
                risk = _risk_band(prob)
                # Surface top 5 features by importance * value magnitude
                model_scores = {
                    "gbt_prob": prob,
                    "calibrated": pretrained_is_calibrated(pretrained_gbt),
                }
                top_features = [
                    {"name": name, "value": round(raw_features[name], 4)}
                    for name in ("srh01", "hrrr_pwat", "alignment", "tau", "maxllaz")
                    if name in raw_features
                ]
                coherence_score = float(raw_features.get("tau", 0.0))
            except Exception as exc:
                print(f"  Warning: pre-trained GBT failed for storm {sid}: {exc}")
                # Fall through to tier 2 analytic
                prob = score_storm_analytic(storm, coherence_fields)
                prob = round(min(max(prob, 0.0), 0.99), 4)
                risk = _risk_band(prob)
                model_scores = {"analytic_prob": prob, "gbt_failed": True}
                coherence_score = prob
        elif scoring_tier == "tier1_ml" and model is not None and HAS_OPERATIONAL:
            # Tier 1 (legacy model format): Full ML model
            result = predict_tornado_probability(
                storm, history, hrrr, coherence_fields, model
            )
            prob = result["probability"]
            risk = result["risk_band"]
            top_features = result["top_features"]
            model_scores = result["model_scores"]
            coherence_score = result["coherence_score"]
        elif scoring_tier in ("tier1_ml", "tier2_analytic") and coherence_fields is not None:
            # Tier 2: Analytic coherence model (no ML needed)
            prob = score_storm_analytic(storm, coherence_fields)
            prob = round(min(max(prob, 0.0), 0.99), 4)
            risk = _risk_band(prob)
            model_scores = {"analytic_prob": prob}
            coherence_score = prob
        else:
            # Tier 3: ProbSevere-only composite (no ML, no HRRR)
            # Use available ProbSevere features directly
            import math as _math
            mucape = float(storm.get("mucape", 0) or 0)
            srh01 = float(storm.get("srh01", 0) or 0)
            ebshear = float(storm.get("ebshear", 0) or 0)
            maxllaz = float(storm.get("maxllaz", 0) or 0)
            mesh = float(storm.get("mesh", 0) or 0)
            flash_rate = float(storm.get("flash_rate", 0) or 0)

            # Composite: rotation matters but atmospheric support gates max
            cape_term = min(mucape / 2000.0, 1.5)
            srh_term = min(abs(srh01) / 200.0, 1.5)
            shear_term = min(ebshear / 30.0, 1.5)
            hail_term = min(mesh / 1.0, 1.0)
            lightning_term = min(flash_rate / 20.0, 1.0)

            # Atmospheric gate: storms need real instability + shear
            # to get above ~30% even with strong rotation
            atm_support = min(
                min(mucape / 1000.0, 1.0),       # need CAPE > 1000 for full support
                min(abs(srh01) / 100.0, 1.0),     # need SRH > 100 for full support
                min(ebshear / 20.0, 1.0),          # need shear > 20 for full support
            )

            # Rotation signal (still important, but gated)
            rotation_signal = min(maxllaz / 0.01, 2.0)

            # Combined: rotation matters but atmospheric support gates the max
            raw = atm_support * (
                rotation_signal * 0.4 +
                cape_term * srh_term * shear_term * 0.3 +
                hail_term * 0.15 +
                lightning_term * 0.15
            )
            prob = 1.0 / (1.0 + _math.exp(-4.0 * (raw - 0.8)))  # sigmoid centered at raw=0.8

            # Hard cap: never exceed 60% without ML model
            prob = round(min(max(prob, 0.0), 0.60), 4)
            risk = _risk_band(prob)
            model_scores = {"ps_composite_raw": round(raw, 4)}

        # Coherence diagnostics at storm location
        coherence_diag: dict = {}
        if coherence_fields is not None:
            coherence_diag = extract_coherence_at_point(
                coherence_fields, lat, lon
            )
            sing = test_singularity_at_point(coherence_fields, lat, lon)
            coherence_diag["singularity_conditions_met"] = sing.count
            coherence_diag["singularity_detail"] = {
                "s_over_gamma": sing.s_over_gamma,
                "high_gradient": sing.high_gradient,
                "high_torsion": sing.high_torsion,
                "positive_alignment": sing.positive_alignment,
                "high_damkohler": sing.high_damkohler,
            }

        entry: dict = {
            "storm_id": sid,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "motion_east": float(storm.get("motion_east", 0)),
            "motion_south": float(storm.get("motion_south", 0)),
            "valid_time": valid_time,
            "tornado_probability": prob,
            "risk_band": risk,
            "ps_tor": float(storm.get("ps_tor", 0)),
            "ps": float(storm.get("ps", 0)),
            "mucape": float(storm.get("mucape", 0)),
            "ebshear": float(storm.get("ebshear", 0)),
            "srh01": float(storm.get("srh01", 0)),
            "maxllaz": float(storm.get("maxllaz", 0)),
            "mesh": float(storm.get("mesh", 0)),
            "flash_rate": float(storm.get("flash_rate", 0)),
            "top_features": top_features,
            "model_scores": model_scores,
            "coherence_score": coherence_score,
            "coherence_diagnostics": coherence_diag,
            "scoring_tier": scoring_tier,
            "coherence_source": coherence_source,
            "model_version": MODEL_VERSION,
            "track_length": len(history),
        }
        if v3_out is not None:
            entry["model_version"] = v3_out["model_version"]      # the model that scored THIS storm
            if v3_out.get("band"):
                entry["confidence_lo"], entry["confidence_hi"] = (round(float(v), 4) for v in v3_out["band"])
            entry["v3"] = {
                "event": "a tornado report within 10 km of this storm's tracked radar polygon",
                "probability_60min": v3_out["p60"],
                "probability_30min": v3_out.get("p30"),
                "probability_90min": v3_out.get("p90"),
                "probability_ef2plus_60min": v3_out.get("p_ef2"),
                "probability_band_60min": v3_out.get("band"),
                # what the band IS: two calibrated estimates, not an interval with a stated coverage
                "probability_band_60min_kind": "Venn-Abers pair (not a coverage interval)",
                "model": v3_out["model"],
                "nws_warning": v3_out.get("warning"),
                "nws_feed_error": v3_warning_error,
                "hrrr_analysis": hrrr_label if v3_out.get("hrrr_used") else None,
                "coherence_clipped": v3_out.get("coherence_clipped", []),
                "drivers": v3_out.get("drivers", []),
                "model_version": v3_out["model_version"],
                "inputs": v3_out.get("inputs"),
            }

        # Include geometry for frontend polygon rendering
        geom = storm.get("geometry")
        if geom:
            entry["geometry"] = geom

        scored.append(entry)

    # Sort by probability descending
    scored.sort(key=lambda s: s["tornado_probability"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
    coherence_source: str = "none",
) -> None:
    """Write scored results to dist/data/."""
    forecast_id = f"to_fcst_{now.strftime('%Y%m%d_%H%M')}"

    # Determine scoring tier label for display
    tier_labels = TIER_LABELS

    # Read recent ledger entries to embed in output
    recent_predictions: list[dict] = []
    if LEDGER_PATH.exists():
        try:
            lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-10:]:
                if line.strip():
                    entry = json.loads(line)
                    recent_predictions.append({
                        "timestamp": entry.get("timestamp", ""),
                        "n_storms": entry.get("n_storms", 0),
                        "max_prob": entry.get("top_probability", 0),
                        "hash": entry.get("hash", "")[:16] + "...",
                    })
            recent_predictions.reverse()
        except Exception:
            pass

    # Write live-tornadoes.json
    output = {
        "disclaimer": (
            "Independent hazard intelligence platform. Always follow official NWS/USGS guidance."
        ),
        "updated_at": now.isoformat() + "Z",
        "forecast_id": forecast_id,
        "model_version": MODEL_VERSION,
        "scoring_tier": scoring_tier,
        "scoring_tier_label": tier_labels.get(scoring_tier, scoring_tier),
        "coherence_source": coherence_source,
        "n_active_storms": len(scored_storms),
        "recent_predictions": recent_predictions,
        "storms": scored_storms,
    }
    storms_path = DIST / "data" / "live-tornadoes.json"
    storms_path.parent.mkdir(parents=True, exist_ok=True)
    storms_path.write_text(
        json.dumps(_sanitize_for_json(output), indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(f"  Wrote {storms_path} ({len(scored_storms)} storms)")

    # Write GeoJSON for MapLibre
    geojson_path = DIST / "data" / "tornado-storms.geojson"
    geojson_path.write_text(_render_geojson(scored_storms), encoding="utf-8")
    print(f"  Wrote {geojson_path} ({len(scored_storms)} features)")

    # Update live-pulse.json tornado entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "to":
                if scored_storms:
                    top = scored_storms[0]  # Already sorted by probability
                    hazard["probability"] = top["tornado_probability"]
                    # Real uncertainty band from the calibrator (None until one
                    # exists) — removes the confidence_interval_unavailable gate warning.
                    hazard["conf_lo"] = top.get("confidence_lo")
                    hazard["conf_hi"] = top.get("confidence_hi")
                    hazard["uncertainty_class"] = top.get("uncertainty_class")
                    hazard["abstained"] = top.get("abstained", False)
                    hazard["receipt_sha256"] = top.get("receipt_sha256")
                    hazard["risk_band"] = top["risk_band"]
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = forecast_id
                    hazard["n_active_storms"] = len(scored_storms)
                    hazard["coherence_score"] = top.get("coherence_score", 0)
                    hazard["coherence_source"] = coherence_source
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "minimal"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = forecast_id
                    hazard["n_active_storms"] = 0
                    hazard["coherence_source"] = "none"
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(
            json.dumps(pulse, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Updated {pulse_path}")


def _v3_tier_label() -> str:
    """Name the served v3 model's input families from its payload (a typed label said "HRRR
    environment" after amendment 8 removed the HRRR block from the served model)."""
    payload = V3_SUITE.main or V3_SUITE.fallback
    if payload is None:
        return "v3 storm model"
    names = [str(n) for n in payload.get("feature_names", [])]
    parts = [label for prefix, label in (("p_", "ProbSevere storm attributes"), ("e_", "storm-track trends"),
                                         ("h80_", "HRRR environment"), ("w_", "NWS warning state"))
             if any(n.startswith(prefix) for n in names)]
    return "v3 storm model (LightGBM on " + ", ".join(parts) + ")"


TIER_LABELS = {
    "tier1_v3": _v3_tier_label(),
    "tier1_ml": "ML (pre-trained gradient-boosted trees)",
    "tier2_analytic": "Analytic coherence model (physics-only, no ML)",
    "tier3_ps_only": "ProbSevere-only fallback (no ML, no HRRR)",
}

RISK_COLORS = {
    "very_high": "#d32f2f",
    "high": "#e65100",
    "moderate": "#f9a825",
    "low": "#1976d2",
    "minimal": "#757575",
}

RISK_LABELS = {
    "very_high": "Very High",
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
    "minimal": "Minimal",
}

# ---------------------------------------------------------------------------
# Location name lookup for Simple mode
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pct(p: float) -> str:
    """Format a probability as a percentage string."""
    return f"{p * 100:.1f}%"


def _pct_fine(p: float) -> str:
    """A rate with enough digits to read at any scale (0.010%, 0.13%, 2.3%, 17%): one decimal
    printed the 2025 outcome rate of the lowest bin, 0.01%, as "0.0% [0.0%, 0.0%]"."""
    v = 100.0 * float(p)
    if v == 0.0:
        return "0%"
    if v >= 10.0:
        return f"{v:.0f}%"
    if v >= 1.0:
        return f"{v:.1f}%"
    if v >= 0.1:
        return f"{v:.2f}%"
    return f"{v:.3f}%"


def _format_time(ts: str) -> str:
    """Format a timestamp string for display."""
    if not ts:
        return "--"
    import re
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\s*UTC$", str(ts))
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)} UTC"
    try:
        d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d.strftime("%a, %d %b %Y %H:%M:%S UTC")
    except Exception:
        return str(ts)


def _read_json(path: Path) -> dict:
    """Read a JSON object from disk, returning an empty dict on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _lat_lon_to_svg(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to SVG coordinates for 960x480 equirectangular map."""
    x = ((lon + 180) / 360) * 960
    y = ((90 - lat) / 180) * 480
    return (x, y)


def _render_geojson(storms: list[dict]) -> str:
    """Render storms as a GeoJSON FeatureCollection for MapLibre."""
    features = []
    for s in storms:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {
                "storm_id": s["storm_id"],
                "probability": s["tornado_probability"],
                "risk_band": s["risk_band"],
                "mucape": s.get("mucape", 0),
                "srh01": s.get("srh01", 0),
                "maxllaz": s.get("maxllaz", 0),
            }
        })
    return json.dumps({"type": "FeatureCollection", "features": features})


def append_ledger(
    scored_storms: list[dict],
    now: dt.datetime,
) -> None:
    """Append prediction to SHA-256 chain ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read previous hash
    prev_hash = "0" * 64
    if LEDGER_PATH.exists():
        lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            try:
                last = json.loads(lines[-1])
                prev_hash = last.get("hash", prev_hash)
            except json.JSONDecodeError:
                pass

    # Build ledger entry
    entry = {
        "timestamp": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "n_storms": len(scored_storms),
        "top_probability": (
            scored_storms[0]["tornado_probability"] if scored_storms else 0.0
        ),
        "prev_hash": prev_hash,
    }
    # Add storm IDs and probabilities
    entry["storms"] = [
        {
            "id": s["storm_id"],
            "prob": s["tornado_probability"],
            "risk": s["risk_band"],
        }
        for s in scored_storms[:20]  # Cap at 20 for ledger size
    ]

    # SHA-256 hash of this entry (computed BEFORE adding "hash" key).
    # Verification: to recompute, exclude the "hash" key from the entry,
    # then json.dumps(entry_without_hash, sort_keys=True, separators=(",",":"))
    # and SHA-256 the result.
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    print(f"  Appended to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Day-ahead susceptibility scoring
# ---------------------------------------------------------------------------


def _climatological_stp(lat: float, lon: float, month: int) -> float:
    """Estimate STP from latitude, longitude, and month when HRRR unavailable.

    Uses a simple climatological proxy:
    - Peak tornado season (Apr-Jun) in central US (30-40N, -100 to -90W)
    - Returns a rough STP estimate in [0, 2].
    """
    # Seasonal factor: peaks in May
    month_weight = {
        1: 0.1, 2: 0.15, 3: 0.35, 4: 0.7, 5: 1.0, 6: 0.8,
        7: 0.4, 8: 0.3, 9: 0.2, 10: 0.15, 11: 0.2, 12: 0.1,
    }.get(month, 0.1)

    # Geographic factor: peak in central plains
    lat_factor = max(0.0, 1.0 - abs(lat - 35.0) / 15.0)
    lon_factor = max(0.0, 1.0 - abs(lon - (-95.0)) / 20.0)
    geo_weight = lat_factor * lon_factor

    return 2.0 * month_weight * geo_weight


def _sigmoid_scalar(z: float) -> float:
    """Numerically stable sigmoid for a single float."""
    import math as _m
    z = max(-88.0, min(88.0, z))
    if z >= 0:
        return 1.0 / (1.0 + _m.exp(-z))
    ef = _m.exp(z)
    return ef / (1.0 + ef)


def compute_day_ahead_susceptibility(
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
    now: dt.datetime,
    analysis_label: str | None = None,
) -> list[dict]:
    """Compute day-ahead tornado susceptibility on the 80km HRRR grid.

    For each grid cell, estimates P(tornado in next 24h) using STP.
    Falls back to climatological STP if HRRR is unavailable.

    Parameters
    ----------
    hrrr : dict or None
        HRRR grid arrays (cape, srh01, shear06, stp, etc.).
    coherence_fields : dict or None
        Coherence field arrays (tau, etc.).
    now : datetime
        Current UTC time.

    Returns
    -------
    list[dict]
        Top 10 grid cells ranked by susceptibility probability.
    """
    cells: list[dict] = []

    # The HRRR dict's keys are srh_01 / ushear_06 / ... ; STP and the 0-6 km
    # shear magnitude are DERIVED fields. The old lookups ("srh01", "shear06",
    # "stp") never existed, so every cell read 0, every probability was
    # sigmoid(-2) = 0.1192, and the published "top 10" were simply the first
    # ten cells in grid order -- Pacific Ocean off Baja California.
    derived = compute_derived_hrrr(hrrr) if hrrr is not None else None

    for i in range(HRRR_N_LAT):
        for j in range(HRRR_N_LON):
            lat = float(GRID_LATS[i])
            lon = float(GRID_LONS[j])

            if hrrr is not None:
                cape = float(hrrr["mlcape"][i, j])
                srh01 = float(hrrr["srh_01"][i, j])
                shear06 = float(derived["shear_06"][i, j])
                stp = float(derived["stp_eff"][i, j])
            else:
                # Climatological fallback
                stp = _climatological_stp(lat, lon, now.month)
                cape = 1500.0 * stp  # rough proxy
                srh01 = 150.0 * stp
                shear06 = 25.0 * stp

            # STP-based probability
            stp_prob = _sigmoid_scalar(2.0 * (stp - 1.0))

            # Get coherence tau if available
            tau = 0.0
            if coherence_fields is not None:
                tau_grid = coherence_fields.get("tau")
                if tau_grid is not None:
                    tau = float(tau_grid[i, j])

            # Risk band
            if stp_prob >= 0.50:
                risk = "very_high"
            elif stp_prob >= 0.30:
                risk = "high"
            elif stp_prob >= 0.15:
                risk = "elevated"
            elif stp_prob >= 0.05:
                risk = "marginal"
            else:
                risk = "minimal"

            cells.append({
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "probability": round(stp_prob, 4),
                "stp": round(stp, 2),
                "cape": round(cape, 0),
                "srh01": round(srh01, 0),
                "shear06": round(shear06, 0),
                "tau": round(tau, 4),
                "risk_band": risk,
            })

    # Sort by probability descending, take top 10
    cells.sort(key=lambda c: c["probability"], reverse=True)
    top_cells = cells[:10]

    # Write output
    output = {
        "disclaimer": (
            "Independent hazard intelligence platform. Always follow official NWS/USGS guidance."
        ),
        "updated_at": now.isoformat() + "Z",
        "model": "hp-tornado-susceptibility-v1",
        "forecast_period": "next 24 hours",
        "data_source": (
            f"HRRR analysis {analysis_label}" if hrrr is not None and analysis_label
            else "HRRR analysis" if hrrr is not None else "climatological estimates"
        ),
        "probability_kind": (
            "uncalibrated index: sigmoid(2 * (STP - 1)); ranks environments, "
            "not a verified probability"
        ),
        "top_cells": top_cells,
    }

    susc_path = DIST / "data" / "live-susceptibility.json"
    susc_path.parent.mkdir(parents=True, exist_ok=True)
    susc_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Wrote {susc_path} ({len(top_cells)} cells)")

    return top_cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the tornado scoring pipeline."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Date to score (YYYYMMDD). Default: today UTC.")
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    date_str = args.date if args.date else now.strftime("%Y%m%d")
    print(f"HazardPulse Tornado Scoring Pipeline -- {now.isoformat()}Z")
    print()

    # Step 1: Fetch ProbSevere data
    print("Step 1: Fetching ProbSevere storm objects...")
    try:
        time_steps = fetch_probsevere_day(date_str, refresh=True)
    except Exception as e:
        print(f"  Warning: ProbSevere fetch failed: {e}")
        time_steps = []

    # Fallback to cache
    if not time_steps:
        cached = load_cached_probsevere(date_str)
        if cached is not None:
            time_steps = cached
            print(f"  Loaded {len(time_steps)} time steps from cache")
        else:
            print("  No ProbSevere data available.")
            print("  Writing empty state...")
            write_outputs([], now, scoring_tier="tier3_ps_only", coherence_source="none")
            append_ledger([], now)
            build_site_artifacts()
            print()
            print("Done. No storms to score.")
            return

    n_storms_latest = len(time_steps[-1].get("storms", [])) if time_steps else 0
    print(f"  {len(time_steps)} time steps, {n_storms_latest} storms in latest")

    if n_storms_latest == 0:
        print("  No active storms in latest time step.")
        write_outputs([], now, scoring_tier="tier3_ps_only", coherence_source="none")
        append_ledger([], now)
        build_site_artifacts()
        print()
        print("Done. No storms to score.")
        return

    # Step 2: Fetch HRRR analysis (most recent available, falling back to yesterday)
    print()
    print("Step 2: Fetching most-recent available HRRR analysis...")
    hrrr: dict[str, np.ndarray] | None = None
    hrrr_hour_used: int | None = None
    hrrr_date_used: str | None = None

    # Newest analysis first, each tried in the cache and then on AWS, and
    # nothing older than the model was trained to see (MAX_ANALYSIS_AGE_H).
    # The old order tried EVERY candidate in the cache before fetching any,
    # so a cached analysis from yesterday beat a fetchable one from an hour
    # ago, and the window reached back 24 h while training never saw an
    # analysis more than 3 h old.
    # Anchored to the OBSERVATION time (the latest ProbSevere step), as every training row is: an
    # analysis published between the observation and this run would be one training never had.
    obs_time = None
    if time_steps:
        _vt = definitive_parse_valid_time(time_steps[-1].get("valid_time", ""))
        obs_time = _vt.replace(tzinfo=None) if _vt is not None else None
    candidates = live_analysis_candidates(obs_time or now)
    for cand_date, cand_hour in candidates:
        grid = load_cached_hrrr(cand_date, hour=cand_hour)
        source = "local cache"
        if grid is None:
            try:
                grid = fetch_hrrr_grid(cand_date, hour=cand_hour)
            except Exception as e:
                print(f"  HRRR fetch error for {cand_date} {cand_hour:02d}Z: {e}")
                grid = None
            source = "AWS"
        if grid is not None:
            hrrr = grid
            hrrr_hour_used = cand_hour
            hrrr_date_used = cand_date
            print(f"  HRRR {cand_date} {cand_hour:02d}Z from {source}")
            break

    if hrrr is None:
        print(f"  Warning: No HRRR analysis within {DEFINITIVE_MAX_ANALYSIS_AGE_H:g} h. "
              "Proceeding without HRRR (ProbSevere fallback mode).")
    else:
        # Sanity-check the data isn't all-NaN (defends against silent partial pulls)
        try:
            import numpy as _np_check
            nan_pcts = []
            for arr in hrrr.values():
                if isinstance(arr, _np_check.ndarray) and arr.size > 0:
                    nan_pcts.append(float(_np_check.isnan(arr).mean()))
            mean_nan = float(_np_check.mean(nan_pcts)) if nan_pcts else 1.0
            if mean_nan > 0.5:
                print(f"  Warning: HRRR is {mean_nan:.0%} NaN — discarding.")
                hrrr = None
        except Exception:
            pass

    # Step 3: Compute coherence fields
    print()
    print("Step 3: Computing coherence fields...")
    coherence_fields: dict[str, np.ndarray] | None = None
    coherence_source: str = "none"
    if hrrr is not None:
        try:
            coherence_fields = compute_coherence_fields(hrrr, month=now.month)
            tau_max = float(coherence_fields["tau"].max())
            sing_max = float(coherence_fields["singularity_count"].max())
            coherence_source = "hrrr"
            print(f"  HRRR coherence: tau_max={tau_max:.4f}, singularity_max={sing_max:.0f}")
        except Exception as e:
            print(f"  Warning: Coherence field computation failed: {e}")

    # Fallback: build coherence from ProbSevere atmospheric data
    # Trigger if coherence is None OR if tau is all zeros (HRRR had no useful data)
    tau_is_zero = (coherence_fields is not None and float(coherence_fields["tau"].max()) < 0.001)
    if (coherence_fields is None or tau_is_zero) and time_steps:
        latest_storms = time_steps[-1].get("storms", [])
        if latest_storms:
            print("  No HRRR -- building coherence fields from ProbSevere atmospheric data...")
            try:
                coherence_fields = build_coherence_from_probsevere(latest_storms)
                if coherence_fields is not None:
                    tau_max = float(coherence_fields["tau"].max())
                    sing_max = float(coherence_fields["singularity_count"].max())
                    coherence_source = "probsevere"
                    print(f"  ProbSevere coherence: tau_max={tau_max:.4f}, singularity_max={sing_max:.0f}")
                else:
                    print("  ProbSevere coherence: no storms with usable atmospheric data")
            except Exception as e:
                print(f"  Warning: ProbSevere coherence fallback failed: {e}")

    if coherence_fields is None:
        print("  No coherence fields available (neither HRRR nor ProbSevere)")

    # Step 4: Determine scoring tier
    print()
    print("Step 4: Determining scoring tier...")
    model: dict | None = None
    pretrained_gbt: dict | None = None
    scoring_tier: str = "tier3_ps_only"

    # Tier 1 v3: the pre-registered v3 suite whenever its payloads are present (it needs neither
    # coherence fields nor, at a measured cost, HRRR)
    if V3_SUITE.available:
        scoring_tier = "tier1_v3"
        print(f"  -> Tier 1 v3: {', '.join(f'{k} {v}' for k, v in V3_SUITE.versions().items())}")
    # Tier 1: Try loading pre-trained GBT (from definitive_model --save-model)
    elif (pretrained_gbt := load_pretrained_gbt()) is not None:
        scoring_tier = "tier1_ml"
        print(f"  -> Tier 1: Pre-trained GBT model ({pretrained_gbt['n_trees']} trees, "
              f"{len(pretrained_gbt['feature_names'])} features)")
    else:
        # Fallback: try legacy model format
        model = load_pretrained_model()
        if model is not None:
            scoring_tier = "tier1_ml"
            print(f"  -> Tier 1: Pre-trained ML model (legacy)")
        elif coherence_fields is not None:
            scoring_tier = "tier2_analytic"
            source_label = "HRRR" if coherence_source == "hrrr" else "ProbSevere fallback"
            print(f"  -> Tier 2: Analytic coherence model (no ML, coherence from {source_label})")
        else:
            scoring_tier = "tier3_ps_only"
            print(f"  -> Tier 3: ProbSevere-only fallback (no ML, no coherence fields)")

    # Step 5: Score active storms
    print()
    print("Step 5: Scoring active storms...")
    scored = score_storms(
        time_steps, hrrr, coherence_fields, model, now,
        scoring_tier=scoring_tier,
        coherence_source=coherence_source,
        pretrained_gbt=pretrained_gbt,
        v3_suite=V3_SUITE if scoring_tier == "tier1_v3" else None,
        hrrr_label=(f"{hrrr_date_used} {hrrr_hour_used:02d}Z" if hrrr_hour_used is not None else None),
    )

    # Step 5a: Block L (lightning) augmentation — score-time multiplier.
    # OFF by default until measured against SPC outcomes for >=200 forecasts.
    # Set HAZARDPULSE_ENABLE_LIGHTNING_AUG=1 to opt in (A/B test).
    # Diagnostics are still recorded on every storm so post-hoc analysis works.
    if scored and os.environ.get("HAZARDPULSE_ENABLE_LIGHTNING_AUG"):
        try:
            from hazardpulse.tornado.lightning_block import apply_lightning_augmentation
            print()
            print("Step 5a: Block L lightning augmentation (GLM, A/B enabled)...")
            apply_lightning_augmentation(scored, enable=True)
            n_lifted = sum(
                1 for s in scored
                if s.get("tornado_probability_pre_lightning", 0)
                != s.get("tornado_probability", 0)
            )
            print(f"  Lightning multiplier applied to {n_lifted}/{len(scored)} storms")
        except Exception as exc:
            print(f"  WARNING: Block L augmentation failed: {exc}; continuing with tier1_ml only.")
    elif scored:
        # Diagnostics-only: record lightning context per storm but do NOT
        # modify the GBT probability. Lets us A/B compare later.
        try:
            from hazardpulse.tornado.lightning_block import apply_lightning_augmentation
            apply_lightning_augmentation(scored, enable=False)
        except Exception:
            pass

    # Trust layer: calibrate tornado probabilities, attach honest [conf_lo,
    # conf_hi] bands + Ed25519-signed re-runnable receipts. Fail-safe: raw until
    # a calibrator (results/models/tornado_calibration.json) exists.
    try:
        from hazardpulse.trust.scoring import enrich_cells, load_forecaster, load_signer

        _signer = load_signer()
        _forecaster = load_forecaster("tornado", signer=_signer)
        if _forecaster is not None and _forecaster.model_version != MODEL_VERSION:
            # A calibrator maps ONE model's raw scores to probabilities; applied
            # to another model's scores it is a different, wrong curve.
            print(
                f"  Trust layer: calibrator was fitted for {_forecaster.model_version}, "
                f"serving {MODEL_VERSION} -- not applied; the model's own "
                "validation-fitted calibration stands until a matching one is fitted."
            )
            _forecaster = None
        if _forecaster is not None and scored:
            # only the storms the calibrator's own model scored: with v3, storms can come from the
            # +W model or the fallback, and a curve fitted for one is wrong for the other
            mine = [s for s in scored if s.get("model_version") == _forecaster.model_version]
            enrich_cells(mine, _forecaster, prob_key="tornado_probability",
                         issued_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(
                f"  Trust layer: calibrated {len(mine)} of {len(scored)} storms "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
        elif scored:
            print(
                "  Trust layer: no calibrator yet "
                "(results/models/tornado_calibration.json); emitting raw forecasts."
            )
    except Exception as exc:  # never let the trust layer break a live forecast
        print(f"  Trust layer: skipped ({exc})")

    refresh_risk_bands(scored)
    n_withheld = withhold_bands_excluding_probability(scored)
    if n_withheld:
        print(f"  Bands withheld: {n_withheld} of {len(scored)} storms' bands did not contain their "
              "published probability")

    for s in scored[:10]:
        print(
            f"  Storm {s['storm_id']}: P(tornado) = {s['tornado_probability']:.1%} "
            f"[{s['risk_band']}] -- CAPE={s['mucape']:.0f}, "
            f"SRH={s['srh01']:.0f}, MaxLLAz={s['maxllaz']:.4f}"
        )

    # Step 5b: Day-ahead susceptibility scoring
    print()
    print("Step 5b: Computing day-ahead susceptibility...")
    try:
        top_cells = compute_day_ahead_susceptibility(
            hrrr, coherence_fields, now,
            analysis_label=(f"{hrrr_date_used} {hrrr_hour_used:02d}Z"
                            if hrrr_hour_used is not None else None),
        )
        if top_cells:
            best = top_cells[0]
            print(f"  Top cell: ({best['lat']}, {best['lon']}) "
                  f"P={best['probability']:.1%} STP={best['stp']:.1f}")
    except Exception as e:
        print(f"  Warning: Susceptibility scoring failed: {e}")

    # Step 6: Write outputs
    print()
    print("Step 6: Writing outputs...")
    write_outputs(scored, now, scoring_tier=scoring_tier, coherence_source=coherence_source)
    append_ledger(scored, now)

    # Step 7: render every page from the published artifacts (hazardpulse.site.build)
    print()
    print("Step 7: Rendering the site from the published artifacts...")
    build_site_artifacts()

    # ---- Alert manager evaluation ----
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        try:
            from hazardpulse.alerts import build_default_manager
            mgr = build_default_manager(
                audit_path=RESULTS / "alerts" / "audit.ndjson",
                recent_path=DIST / "data" / "alerts-recent.json",
            )
            pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
            fired = mgr.evaluate(pulse)
            for a in fired:
                if a.severity != "suppressed":
                    print(f"  ALERT [{a.severity}] {a.rule_name}: {a.message}")
        except Exception as exc:
            print(f"  Warning: alert evaluation skipped: {exc}")

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
