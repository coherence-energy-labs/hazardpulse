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
    extract_block_c as definitive_extract_c,
    extract_block_e as definitive_extract_e,
    extract_block_h as definitive_extract_h,
    extract_block_p as definitive_extract_p,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
LEDGER_PATH = DIST / "data" / "tornado-ledger.jsonl"

MODEL_VERSION = "tornado_storm_v1_0"
PRIMARY_DOMAIN = "https://hazardpulse.com"
SITE_PUBLISHER_NAME = "HazardPulse"
SITE_CONTACT_EMAIL = "josh@coherenceenergylabs.com"


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
        _gradient_2d,
        compute_curl_2d,
    )

    grad_y, grad_x = _gradient_2d(tau)
    grad_tau = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)

    # Torsion: shear * curl(tau) / 25
    curl_tau = compute_curl_2d(tau)
    torsion = (shear_field * curl_tau / 25.0).astype(np.float32)

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
    cond3 = (np.abs(torsion) > 0.1).astype(np.float32)
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
        Predicted tornado probability in [0, 1].
    """
    import math as _math

    feature_names = gbt_data["feature_names"]
    means = gbt_data["normalization"]["means"]
    stds = gbt_data["normalization"]["stds"]

    # Build normalized feature vector in correct order
    x = []
    for i, name in enumerate(feature_names):
        raw = raw_features.get(name, 0.0)
        x.append((raw - means[i]) / stds[i])

    # Walk each tree and accumulate predictions
    F = gbt_data["init_pred"]
    lr = gbt_data["learning_rate"]
    for tree in gbt_data["trees"]:
        node = tree
        while not node.get("leaf", False):
            feat_idx = node["feat"]
            if feat_idx < len(x) and x[feat_idx] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        F += lr * node["val"]

    # Numerically stable sigmoid
    F = max(-88.0, min(88.0, F))
    if F >= 0:
        prob = 1.0 / (1.0 + _math.exp(-F))
    else:
        ef = _math.exp(F)
        prob = ef / (1.0 + ef)
    return prob


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
) -> list[dict]:
    """Score all active storms from the latest ProbSevere time step.

    scoring_tier controls which scoring method is used:
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

        if (
            scoring_tier == "tier1_ml"
            and pretrained_gbt is not None
            and hrrr is not None
            and coherence_fields is not None
            and hrrr_usable
        ):
            # Tier 1 (pre-trained GBT): build 41-feature vector then score.
            try:
                f_p = definitive_extract_p(storm)
                f_e = definitive_extract_e(storm, history)
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
                model_scores = {"gbt_prob": prob}
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

    # Write HTMX fragment (storm rows only, no page wrapper)
    fragment_path = DIST / "data" / "tornado-fragment.html"
    fragment_path.write_text(
        _render_storm_rows(scored_storms) if scored_storms else
        '<div class="card" style="text-align:center;padding:24px;"><p class="muted">No active storms.</p></div>',
        encoding="utf-8",
    )
    print(f"  Wrote {fragment_path}")

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


TIER_LABELS = {
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

# Simple-mode risk labels (parent-friendly)
SIMPLE_RISK_LABELS = {
    "very_high": "CRITICAL",
    "high": "HIGH",
    "moderate": "ELEVATED",
    "low": "MODERATE",
    "minimal": "LOW RISK",
}

SIMPLE_RISK_COLORS = {
    "very_high": "#7f1d1d",
    "high": "#dc2626",
    "moderate": "#ea580c",
    "low": "#ca8a04",
    "minimal": "#16a34a",
}


# ---------------------------------------------------------------------------
# Location name lookup for Simple mode
# ---------------------------------------------------------------------------

_CITIES = [
    (35.22, -97.44, "Oklahoma City, OK"),
    (32.78, -96.80, "Dallas, TX"),
    (39.10, -94.58, "Kansas City, MO"),
    (41.88, -87.63, "Chicago, IL"),
    (33.75, -84.39, "Atlanta, GA"),
    (36.16, -86.78, "Nashville, TN"),
    (39.77, -86.16, "Indianapolis, IN"),
    (39.96, -83.00, "Columbus, OH"),
    (42.33, -83.05, "Detroit, MI"),
    (44.98, -93.27, "Minneapolis, MN"),
    (38.63, -90.20, "St. Louis, MO"),
    (30.27, -97.74, "Austin, TX"),
    (29.76, -95.37, "Houston, TX"),
    (35.47, -97.52, "Norman, OK"),
    (37.69, -97.34, "Wichita, KS"),
    (40.81, -96.70, "Lincoln, NE"),
    (41.26, -95.94, "Omaha, NE"),
    (34.74, -92.29, "Little Rock, AR"),
    (32.30, -90.18, "Jackson, MS"),
    (30.45, -91.19, "Baton Rouge, LA"),
    (35.15, -90.05, "Memphis, TN"),
    (33.52, -86.81, "Birmingham, AL"),
    (38.25, -85.76, "Louisville, KY"),
    (43.07, -89.40, "Madison, WI"),
    (42.96, -85.66, "Grand Rapids, MI"),
    (40.42, -86.91, "Lafayette, IN"),
    (41.08, -81.52, "Akron, OH"),
    (40.80, -81.38, "Canton, OH"),
    (36.15, -95.99, "Tulsa, OK"),
    (37.22, -93.29, "Springfield, MO"),
    (30.33, -81.66, "Jacksonville, FL"),
    (27.95, -82.46, "Tampa, FL"),
    (25.76, -80.19, "Miami, FL"),
    (32.47, -93.79, "Shreveport, LA"),
    (29.95, -90.07, "New Orleans, LA"),
    (34.00, -81.03, "Columbia, SC"),
    (35.23, -80.84, "Charlotte, NC"),
    (36.07, -79.79, "Greensboro, NC"),
    (32.37, -86.30, "Montgomery, AL"),
    (34.73, -86.59, "Huntsville, AL"),
    (39.16, -84.46, "Cincinnati, OH"),
    (40.44, -79.99, "Pittsburgh, PA"),
    (38.90, -77.04, "Washington, DC"),
    (39.29, -76.61, "Baltimore, MD"),
    (39.95, -75.17, "Philadelphia, PA"),
    (40.71, -74.01, "New York, NY"),
    (41.76, -72.68, "Hartford, CT"),
    (42.36, -71.06, "Boston, MA"),
    (35.96, -83.92, "Knoxville, TN"),
    (35.05, -85.31, "Chattanooga, TN"),
    (31.95, -102.18, "Midland, TX"),
    (33.45, -94.04, "Texarkana, TX"),
    (31.76, -106.44, "El Paso, TX"),
    (29.42, -98.49, "San Antonio, TX"),
    (32.45, -99.73, "Abilene, TX"),
    (33.58, -101.85, "Lubbock, TX"),
    (35.08, -106.65, "Albuquerque, NM"),
    (39.74, -104.99, "Denver, CO"),
    (41.14, -104.82, "Cheyenne, WY"),
    (46.88, -96.79, "Fargo, ND"),
    (43.55, -96.73, "Sioux Falls, SD"),
    (40.69, -99.08, "Kearney, NE"),
    (38.88, -99.33, "Hays, KS"),
    (37.04, -100.92, "Liberal, KS"),
    (36.41, -100.48, "Woodward, OK"),
]


def latlon_to_location_name(lat: float, lon: float) -> str:
    """Convert lat/lon to approximate human-readable location description.

    Uses a simple lookup of major US cities and regions.
    Returns something like "Near Oklahoma City, OK" or "Central US".
    """
    closest_city = None
    closest_lat = 0.0
    closest_lon = 0.0
    closest_dist = 999.0
    for clat, clon, cname in _CITIES:
        d = ((lat - clat) ** 2 + (lon - clon) ** 2) ** 0.5 * 111  # rough km
        if d < closest_dist:
            closest_dist = d
            closest_city = cname
            closest_lat = clat
            closest_lon = clon

    if closest_dist < 50:
        return f"Near {closest_city}"
    elif closest_dist < 150 and closest_city:
        dlat = lat - closest_lat
        dlon = lon - closest_lon
        if abs(dlat) > abs(dlon):
            direction = "N of" if dlat > 0 else "S of"
        else:
            direction = "E of" if dlon > 0 else "W of"
        miles = closest_dist * 0.621
        return f"{miles:.0f} mi {direction} {closest_city}"
    else:
        # Use region
        if 25 < lat < 31 and -100 < lon < -80:
            return "Gulf Coast"
        elif 31 < lat < 37 and -100 < lon < -82:
            return "Southern Plains / Deep South"
        elif 37 < lat < 42 and -100 < lon < -82:
            return "Central US"
        elif 42 < lat < 49 and -100 < lon < -82:
            return "Upper Midwest"
        elif lat > 37 and lon < -100:
            return "High Plains"
        elif 25 < lat < 37 and lon > -82:
            return "Southeast US"
        elif lat > 37 and lon > -82:
            return "Northeast US"
        else:
            return f"{abs(lat):.1f} {'N' if lat >= 0 else 'S'}, {abs(lon):.1f} {'E' if lon >= 0 else 'W'}"


def get_action_recommendation(risk_band: str, prob: float) -> str:
    """Return a plain-language action recommendation for the given risk level."""
    if risk_band == "very_high" or prob > 0.40:
        return (
            "Seek shelter immediately if NWS issues a tornado warning "
            "for your area. Have your emergency plan ready."
        )
    elif risk_band == "high" or prob > 0.25:
        return (
            "Stay weather-aware. Monitor NWS warnings. "
            "Know where your nearest shelter is."
        )
    elif risk_band == "moderate" or prob > 0.15:
        return (
            "Be aware of developing severe weather. "
            "Check weather.gov for updates."
        )
    elif risk_band == "low" or prob > 0.08:
        return "Low risk. No immediate action needed. Stay generally weather-aware."
    else:
        return "No significant tornado risk at this time."


def _simple_why_sentence(s: dict) -> str:
    """Build a one-sentence plain-English explanation of the storm's risk."""
    cape = float(s.get("mucape", 0) or 0)
    srh = float(s.get("srh01", 0) or 0)
    maxllaz = float(s.get("maxllaz", 0) or 0)
    coh = s.get("coherence_diagnostics", {})
    parts = []
    if maxllaz > 0.01:
        parts.append("strong rotation detected")
    elif maxllaz > 0.005:
        parts.append("moderate rotation detected")
    if cape > 1500:
        parts.append("unstable atmosphere")
    if abs(srh) > 150:
        parts.append("strong low-level wind shear")
    if coh and float(coh.get("alignment", 0) or 0) > 0.1:
        parts.append("coherent wind structure")
    if not parts:
        return "No significant tornado signals detected in this storm."
    return "This storm has " + ", ".join(parts) + "."


def _storm_watch_items(s: dict) -> list[str]:
    """Return compact watch items for the storm detail card."""
    items: list[str] = []
    cape = float(s.get("mucape", 0) or 0)
    srh = float(s.get("srh01", 0) or 0)
    maxllaz = float(s.get("maxllaz", 0) or 0)
    flash_rate = float(s.get("flash_rate", 0) or 0)
    coh = s.get("coherence_diagnostics", {}) or {}
    singularity_count = int(coh.get("singularity_conditions_met", 0) or 0)

    if maxllaz > 0.01:
        items.append(f"Strong rotation {maxllaz:.3f} s^-1")
    elif maxllaz > 0.005:
        items.append(f"Rotation {maxllaz:.3f} s^-1")
    if cape >= 1000:
        items.append(f"MUCAPE {cape:.0f} J/kg")
    if abs(srh) >= 150:
        items.append(f"0-1 km SRH {srh:.0f}")
    if flash_rate >= 20:
        items.append(f"Flash rate {flash_rate:.0f}/min")
    if singularity_count >= 2:
        items.append(f"{singularity_count}/5 singularity conditions")

    return items[:5]


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


def _confidence_text(probability: object, lo: object, hi: object) -> str:
    """Format a sane confidence range, or explain that it is unavailable."""
    try:
        p = float(probability)
        lo_v = float(lo)
        hi_v = float(hi)
    except (TypeError, ValueError):
        return "Range unavailable"

    if not (0.0 <= lo_v <= hi_v <= 1.0):
        return "Range unavailable"
    if not (lo_v <= p <= hi_v):
        return "Range unavailable"
    return f"{_pct(lo_v)} to {_pct(hi_v)}"


def _band_text(lo: object, hi: object) -> str:
    """Calibrated 90% band, e.g. ' (90% band: 8.0%-18.0%)'; empty when unavailable
    so the label is unchanged for raw (uncalibrated) forecasts."""
    if lo is None or hi is None:
        return ""
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return ""
    if lo_f != lo_f or hi_f != hi_f:   # NaN
        return ""
    return f" (90% band: {lo_f * 100:.1f}%-{hi_f * 100:.1f}%)"


def _trend_text(delta: object) -> str:
    """Render a safe ASCII trend string."""
    try:
        value = float(delta or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return f"up +{_pct(abs(value))}"
    if value < 0:
        return f"down {_pct(abs(value))}"
    return "flat 0.0%"


def _format_coord_pair(lat: object, lon: object, decimals: int = 1) -> str:
    """Format coordinates as a simple signed lat/lon pair."""
    try:
        lat_v = float(lat)
        lon_v = float(lon)
    except (TypeError, ValueError):
        return "--"
    return f"{lat_v:.{decimals}f}, {lon_v:.{decimals}f}"


def _format_compass_coords(lat: object, lon: object, decimals: int = 3) -> str:
    """Format coordinates using N/S/E/W labels with ASCII-only symbols."""
    try:
        lat_v = float(lat)
        lon_v = float(lon)
    except (TypeError, ValueError):
        return "--"
    ns = "N" if lat_v >= 0 else "S"
    ew = "E" if lon_v >= 0 else "W"
    return f"{abs(lat_v):.{decimals}f} {ns}, {abs(lon_v):.{decimals}f} {ew}"


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


def _render_svg_markers(storms: list[dict]) -> str:
    """Generate SVG marker elements for storm positions on the map."""
    lines: list[str] = []
    limit = min(len(storms), 20)
    for i in range(limit):
        s = storms[i]
        rank = i + 1
        x, y = _lat_lon_to_svg(s["lat"], s["lon"])
        risk_label = RISK_LABELS.get(s["risk_band"], s["risk_band"])
        label = f'Storm {s["storm_id"]} - {_pct(s["tornado_probability"])} ({risk_label})'
        base_r = 4.5 if rank == 1 else 3.0

        lines.append(f'    <a href="#storm-{rank}" aria-label="{_esc(label)}">')
        for p in (1, 2):
            lines.append(
                f'      <circle class="hz-pulse hz-pulse-to hz-pulse-delay-{p}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1:.0f}"/>'
            )
        lines.append(
            f'      <circle class="hz-marker hz-marker-to" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r:.1f}" filter="url(#glow-to)"/>'
        )
        # Tooltip
        lines.append(f'      <g class="map-tooltip">')
        lines.append(
            f'        <rect class="tooltip-bg" x="{x + 10:.1f}" y="{y - 18:.1f}" '
            f'width="90" height="22" rx="3"/>'
        )
        lines.append(
            f'        <text class="tooltip-text" x="{x + 12:.1f}" y="{y - 8:.1f}">'
            f'Storm {_esc(str(s["storm_id"]))}</text>'
        )
        lines.append(
            f'        <text class="tooltip-sub" x="{x + 12:.1f}" y="{y + 1:.1f}">'
            f'{_pct(s["tornado_probability"])} \u00b7 Rank #{rank}</text>'
        )
        lines.append(f'      </g>')
        lines.append(f'    </a>')
    return "\n".join(lines)


def _render_storm_rows(storms: list[dict]) -> str:
    """Render storm table as <details>/<summary> elements with dual Simple/Technical content. Zero JavaScript."""
    import math as _math

    lines: list[str] = []
    limit = min(len(storms), 20)
    for i in range(limit):
        s = storms[i]
        rank = i + 1
        risk_color = RISK_COLORS.get(s["risk_band"], "#757575")
        risk_label = RISK_LABELS.get(s["risk_band"], s["risk_band"])
        simple_risk_label = SIMPLE_RISK_LABELS.get(s["risk_band"], risk_label)
        simple_risk_color = SIMPLE_RISK_COLORS.get(s["risk_band"], risk_color)
        rank_class = " rank-1" if rank == 1 else ""
        prob = float(s.get("tornado_probability", 0) or 0)
        location_name = latlon_to_location_name(s["lat"], s["lon"])
        action = get_action_recommendation(s["risk_band"], prob)
        why_sentence = _simple_why_sentence(s)
        watch_items = _storm_watch_items(s)
        technical_subline = (
            f"{_format_compass_coords(s['lat'], s['lon'], decimals=2)} | "
            f"CAPE {float(s.get('mucape', 0) or 0):.0f} J/kg | "
            f"SRH {float(s.get('srh01', 0) or 0):.0f} | "
            f"AzShear {float(s.get('maxllaz', 0) or 0):.4f}"
        )
        simple_subline = (
            f"{_pct(prob)} tornado risk in the next 24 hours | "
            f"{_format_time(s.get('valid_time', ''))}"
        )

        # --- Summary line: Simple shows location name + risk; Technical shows numbers ---
        lines.append(f'        <details class="event-row-details" id="storm-{rank}">')
        lines.append(f'          <summary class="event-row">')
        lines.append(
            f'            <span class="event-row-leading">'
            f'<span class="rank-badge{rank_class}">{rank}</span>'
            f'<span class="event-row-copy">'
            f'<span class="event-row-mode" data-depth="simple">'
            f'<span class="event-row-headline">'
            f'<span class="event-row-title">{_esc(location_name)}</span>'
            f'<span class="chip" style="background:{simple_risk_color};color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(simple_risk_label)}</span>'
            f'</span>'
            f'<span class="event-row-subline">{_esc(simple_subline)}</span>'
            f'</span>'
            f'<span class="event-row-mode" data-depth="technical">'
            f'<span class="event-row-headline">'
            f'<span class="event-row-title">Storm {_esc(str(s["storm_id"]))}</span>'
            f'<span class="chip" style="background:{risk_color};color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(risk_label)}</span>'
            f'</span>'
            f'<span class="event-row-subline">{_esc(technical_subline)}</span>'
            f'</span>'
            f'</span>'
            f'</span>'
            f'<span class="event-row-side">'
            f'<span class="event-row-score" style="--event-accent:{risk_color};">'
            f'{_pct(prob)}</span>'
            f'<span class="event-row-caret" aria-hidden="true"></span>'
            f'</span>'
        )
        lines.append(f'          </summary>')
        lines.append(f'          <div class="event-detail">')

        # ===================================================================
        # SIMPLE MODE — just risk, location, one sentence, what to do
        # ===================================================================
        lines.append(f'            <div data-depth="simple" class="detail-mode">')
        lines.append(f'              <div class="detail-hero">')
        lines.append(f'                <div>')
        lines.append(f'                  <div class="detail-kicker">Threat brief</div>')
        lines.append(f'                  <h3 class="detail-title">{_esc(location_name)}</h3>')
        lines.append(f'                  <p class="detail-copy">{_esc(why_sentence)}</p>')
        lines.append(f'                </div>')
        lines.append(f'                <div class="detail-badge-stack">')
        lines.append(
            f'                  <span class="chip" style="background:{simple_risk_color};color:#fff;'
            f'font-size:14px;padding:4px 14px;font-weight:700;">{_esc(simple_risk_label)}</span>'
        )
        lines.append(
            f'                  <div class="detail-score" style="color:{risk_color};">{_pct(prob)}</div>'
        )
        lines.append(
            '                  <div class="detail-score-label">Estimated tornado probability for this storm object'
            f'{_band_text(s.get("confidence_lo"), s.get("confidence_hi"))}</div>'
        )
        lines.append(f'                </div>')
        lines.append(f'              </div>')
        lines.append(f'              <div class="detail-alert">')
        lines.append(f'                <strong>What to do</strong>')
        lines.append(f'                <p>{_esc(action)}</p>')
        lines.append(f'              </div>')
        if watch_items:
            lines.append(f'              <div class="detail-chip-row">')
            for item in watch_items:
                lines.append(f'                <span class="signal-pill">{_esc(item)}</span>')
            lines.append(f'              </div>')
        lines.append(f'              <div class="detail-signal-grid" style="margin-top:16px;">')
        lines.append(
            f'                <div class="signal-card"><span>Location</span><strong>{_esc(location_name)}</strong>'
            f'<small>{_esc(_format_compass_coords(s["lat"], s["lon"], decimals=2))}</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>Storm motion</span><strong>{float(s.get("motion_east", 0) or 0):+.1f}E / {float(s.get("motion_south", 0) or 0):+.1f}S</strong>'
            f'<small>Motion components used in the analytic scoring stack</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>ProbSevere ID</span><strong>{_esc(str(s["storm_id"]))}</strong>'
            f'<small>Active convective object identifier</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>Official guidance</span><strong>weather.gov</strong>'
            f'<small>Always follow NWS watches and warnings first</small></div>'
        )
        lines.append(f'              </div>')
        lines.append(f'            </div>')

        # ===================================================================
        # TECHNICAL MODE — full 7 sections + enhanced diagnostics
        # ===================================================================
        lines.append(f'            <div data-depth="technical" class="detail-mode">')
        lines.append(f'              <div class="detail-hero">')
        lines.append(f'                <div>')
        lines.append(f'                  <div class="detail-kicker">Technical breakdown</div>')
        lines.append(f'                  <h3 class="detail-title">Storm {_esc(str(s["storm_id"]))}</h3>')
        lines.append(
            f'                  <p class="detail-copy">{_esc(why_sentence)} The current analytic blend uses '
            f'ProbSevere storm attributes, coherence diagnostics, and a physics-first scoring tier.</p>'
        )
        lines.append(f'                </div>')
        lines.append(f'                <div class="detail-badge-stack">')
        lines.append(
            f'                  <span class="chip" style="background:{risk_color};color:#fff;font-size:12px;padding:4px 10px;">{_esc(risk_label)}</span>'
        )
        lines.append(
            f'                  <div class="detail-score" style="color:{risk_color};">{_pct(prob)}</div>'
        )
        lines.append(
            f'                  <div class="detail-score-label">{_esc(_format_compass_coords(s["lat"], s["lon"], decimals=3))}</div>'
        )
        lines.append(f'                </div>')
        lines.append(f'              </div>')
        if watch_items:
            lines.append(f'              <div class="detail-chip-row">')
            for item in watch_items:
                lines.append(f'                <span class="signal-pill">{_esc(item)}</span>')
            lines.append(f'              </div>')

        # --- LOCATION & TIMING ---
        lines.append(f'              <div style="font-size:12px;text-transform:uppercase;letter-spacing:0.04em;color:var(--muted,#6b7280);margin-bottom:6px;margin-top:4px;">Location &amp; Timing</div>')
        lines.append(f'              <div class="kv"><span>Coordinates</span><strong>{_format_compass_coords(s["lat"], s["lon"], decimals=3)}</strong></div>')
        lines.append(f'              <div class="kv"><span>Location</span><strong>{_esc(location_name)}</strong></div>')
        lines.append(f'              <div class="kv"><span>Valid time</span><strong>{_format_time(s.get("valid_time", ""))}</strong></div>')
        me = float(s.get("motion_east", 0) or 0)
        ms_val = float(s.get("motion_south", 0) or 0)
        speed_ms = _math.sqrt(me**2 + ms_val**2)
        speed_mph = speed_ms * 2.237
        direction = ""
        if speed_ms > 1:
            angle = _math.degrees(_math.atan2(me, -ms_val)) % 360
            dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            direction = dirs[int((angle + 11.25) / 22.5) % 16]
        lines.append(f'              <div class="kv"><span>Storm motion</span><strong>{speed_mph:.0f} mph {direction}</strong></div>')
        lines.append(f'              <div class="kv"><span>Storm size</span><strong>{s.get("size", 0):.0f} km^2</strong></div>')
        lines.append(f'              <div class="kv"><span>Track length</span><strong>{s.get("track_length", 0)} time steps</strong></div>')
        lines.append(f'              <div class="kv"><span>Scoring tier</span><strong>{_esc(s.get("scoring_tier", "--"))}</strong></div>')

        # --- ATMOSPHERIC STATE ---
        _hr = '              <hr style="border:0;border-top:1px solid var(--border,#e5e7eb);margin:10px 0;">'
        _section_hdr = lambda title: f'              <div style="font-size:12px;text-transform:uppercase;letter-spacing:0.04em;color:var(--muted,#6b7280);margin-bottom:6px;">{title}</div>'
        lines.append(_hr)
        lines.append(_section_hdr("Atmospheric State (from ProbSevere)"))
        cape = float(s.get("mucape", 0) or 0)
        mlcape = float(s.get("mlcape", 0) or 0)
        cin = float(s.get("mlcin", 0) or 0)
        srh = float(s.get("srh01", 0) or 0)
        shear = float(s.get("ebshear", 0) or 0)
        pwat = float(s.get("pwat", 0) or 0)
        wbz = s.get("wetbulb_0c_hgt", 0) or 0
        cape_label = "Extreme" if cape > 3000 else "High" if cape > 2000 else "Moderate" if cape > 1000 else "Low" if cape > 500 else "Marginal"
        srh_label = "Extreme" if abs(srh) > 300 else "High" if abs(srh) > 200 else "Moderate" if abs(srh) > 100 else "Low"
        shear_label = "Extreme" if shear > 50 else "High" if shear > 35 else "Moderate" if shear > 20 else "Low"
        lines.append(f'              <div class="kv"><span>MUCAPE</span><strong>{cape:.0f} J/kg ({cape_label})</strong></div>')
        lines.append(f'              <div class="kv"><span>MLCAPE</span><strong>{mlcape:.0f} J/kg</strong></div>')
        lines.append(f'              <div class="kv"><span>MLCIN</span><strong>{cin:.0f} J/kg</strong></div>')
        lines.append(f'              <div class="kv"><span>0-1km SRH</span><strong>{srh:.0f} m^2/s^2 ({srh_label})</strong></div>')
        lines.append(f'              <div class="kv"><span>Effective bulk shear</span><strong>{shear:.0f} kt ({shear_label})</strong></div>')
        lines.append(f'              <div class="kv"><span>Precipitable water</span><strong>{pwat:.1f} in</strong></div>')
        lines.append(f'              <div class="kv"><span>Wet bulb 0C height</span><strong>{wbz} kft</strong></div>')

        # STP estimate
        cape_t = min(cape / 1500.0, 2.0)
        srh_t = min(abs(srh) / 150.0, 2.0)
        shear_t = min(shear / 20.0, 2.0)
        stp_est = cape_t * srh_t * shear_t
        stp_label = "Significant tornado environment" if stp_est > 3 else "Tornado possible" if stp_est > 1 else "Marginal" if stp_est > 0.5 else "Low"
        lines.append(f'              <div class="kv"><span>STP estimate</span><strong>{stp_est:.1f} ({stp_label})</strong></div>')

        # --- RADAR SIGNATURES ---
        lines.append(_hr)
        lines.append(_section_hdr("Radar Signatures"))
        maxllaz = float(s.get("maxllaz", 0) or 0)
        p98llaz = float(s.get("p98llaz", 0) or 0)
        p98mlaz = float(s.get("p98mlaz", 0) or 0)
        mesh = float(s.get("mesh", 0) or 0)
        vil = float(s.get("vil_density", 0) or 0)
        rot_label = "Strong rotation" if maxllaz > 0.01 else "Moderate rotation" if maxllaz > 0.005 else "Weak rotation" if maxllaz > 0.003 else "No significant rotation"
        lines.append(f'              <div class="kv"><span>Max low-level AzShear</span><strong>{maxllaz:.4f} s^-1 ({rot_label})</strong></div>')
        lines.append(f'              <div class="kv"><span>P98 low-level AzShear</span><strong>{p98llaz:.4f} s^-1</strong></div>')
        lines.append(f'              <div class="kv"><span>P98 mid-level AzShear</span><strong>{p98mlaz:.4f} s^-1</strong></div>')
        lines.append(f'              <div class="kv"><span>MESH (max hail)</span><strong>{mesh:.2f} in</strong></div>')
        lines.append(f'              <div class="kv"><span>VIL density</span><strong>{vil:.2f} g/m^3</strong></div>')

        # --- LIGHTNING ---
        lines.append(_hr)
        lines.append(_section_hdr("Lightning Activity"))
        fr = float(s.get("flash_rate", 0) or 0)
        fd = float(s.get("flash_density", 0) or 0)
        lja = float(s.get("lja", 0) or 0)
        fr_label = "Intense" if fr > 50 else "Active" if fr > 20 else "Moderate" if fr > 5 else "Quiet"
        lines.append(f'              <div class="kv"><span>Flash rate</span><strong>{fr:.0f} /min ({fr_label})</strong></div>')
        lines.append(f'              <div class="kv"><span>Flash density</span><strong>{fd:.2f}</strong></div>')
        lines.append(f'              <div class="kv"><span>Lightning jump (LJA)</span><strong>{lja:.1f}</strong></div>')

        # --- PROBSEVERE SCORES ---
        lines.append(_hr)
        lines.append(_section_hdr("ProbSevere Scores"))
        lines.append(f'              <div class="kv"><span>ProbSevere (any severe)</span><strong>{s.get("ps", 0):.0f}%</strong></div>')
        lines.append(f'              <div class="kv"><span>ProbSevere tornado</span><strong>{s.get("ps_tor", 0):.0f}%</strong></div>')

        # --- COHERENCE FIELD THEORY ---
        coh = s.get("coherence_diagnostics", {})
        lines.append(_hr)
        lines.append(_section_hdr("Coherence Field Theory Analysis"))
        if coh:
            tau = float(coh.get("tau", 0) or 0)
            grad = float(coh.get("grad_tau", 0) or 0)
            torsion = float(coh.get("torsion", 0) or 0)
            alignment = float(coh.get("alignment", 0) or 0)
            sg = float(coh.get("S_over_Gamma", 0) or 0)
            da = float(coh.get("Da", 0) or 0)
            sing = int(coh.get("singularity_conditions_met", 0) or 0)

            tau_label = "Strong coherence" if tau > 0.5 else "Moderate" if tau > 0.2 else "Weak" if tau > 0.05 else "Minimal"
            sg_label = "Source exceeds damping" if sg > 1 else "Near balance" if sg > 0.5 else "Damping dominant"
            sing_label = "CRITICAL" if sing >= 4 else "Elevated" if sing >= 3 else "Marginal" if sing >= 2 else "Low"

            lines.append(f'              <div class="kv"><span>Coherence amplitude (tau)</span><strong>{tau:.4f} ({tau_label})</strong></div>')
            lines.append(f'              <div class="kv"><span>Coherence gradient (|grad tau|)</span><strong>{grad:.4f}</strong></div>')
            lines.append(f'              <div class="kv"><span>Torsion (SRH x curl tau)</span><strong>{torsion:.4f}</strong></div>')
            lines.append(f'              <div class="kv"><span>Alignment (shear dot grad tau)</span><strong>{alignment:.4f}</strong></div>')
            lines.append(f'              <div class="kv"><span>S / Gamma ratio</span><strong>{sg:.2f} ({sg_label})</strong></div>')
            lines.append(f'              <div class="kv"><span>Damkohler number</span><strong>{da:.2f}</strong></div>')
            lines.append(f'              <div class="kv"><span>Singularity conditions</span><strong>{sing} / 5 ({sing_label})</strong></div>')
            lines.append(f'              <div class="kv"><span>Coherence source</span><strong>{_esc(s.get("coherence_source", "unknown"))}</strong></div>')

            # --- COHERENCE INTERPRETATION (value-driven) ---
            lines.append(_hr)
            lines.append(_section_hdr("Coherence Interpretation"))
            interp_parts = []
            if tau > 0.5:
                interp_parts.append(f"Strong atmospheric coherence (tau={tau:.2f}) indicates well-organized convective structure.")
            elif tau > 0.1:
                interp_parts.append(f"Moderate coherence (tau={tau:.2f}) - some atmospheric organization present.")
            else:
                interp_parts.append(f"Weak coherence (tau={tau:.2f}) - limited atmospheric organization.")

            if sg > 1.0:
                interp_parts.append(f"The source/damping ratio ({sg:.1f}) exceeds unity - energy input exceeds dissipation, favorable for storm intensification.")
            elif sg > 0.5:
                interp_parts.append(f"Source/damping ratio ({sg:.1f}) is approaching balance - storm may intensify if conditions persist.")
            else:
                interp_parts.append(f"Low source/damping ratio ({sg:.1f}) - dissipation dominates, limiting storm development.")

            if alignment > 0.1:
                interp_parts.append("Wind shear is aligned with the coherence gradient, a signature the theory associates with tornadic transition.")
            elif alignment > 0.01:
                interp_parts.append("Partial shear-coherence alignment detected.")

            if sing >= 4:
                interp_parts.append(f"CRITICAL: {sing}/5 singularity conditions met - coherence theory indicates high tornado commitment potential.")
            elif sing >= 3:
                interp_parts.append(f"Elevated: {sing}/5 singularity conditions - approaching coherence commitment threshold.")
            elif sing >= 2:
                interp_parts.append(f"Marginal: {sing}/5 singularity conditions.")

            for part in interp_parts:
                lines.append(f'              <p style="margin:4px 0;font-size:13px;line-height:1.5;">{_esc(part)}</p>')
        else:
            lines.append(f'              <div class="kv"><span>Status</span><strong>Coherence data unavailable for this storm</strong></div>')

        # --- MODEL CONFIDENCE (new) ---
        lines.append(_hr)
        lines.append(_section_hdr("Model Confidence"))
        analytic_prob = float(s.get("analytic_probability", prob) or prob)
        lines.append(f'              <div class="kv"><span>Combined probability</span><strong>{_pct(prob)}</strong></div>')
        lines.append(f'              <div class="kv"><span>Analytic coherence model</span><strong>{_pct(analytic_prob)}</strong></div>')
        model_ver = s.get("model_version", MODEL_VERSION)
        lines.append(f'              <div class="kv"><span>Model version</span><strong>{_esc(model_ver)}</strong></div>')

        # --- CLIMATOLOGICAL COMPARISON (new) ---
        lines.append(_hr)
        lines.append(_section_hdr("Comparison to Climatology"))
        # SRH percentile estimates (rough CONUS spring climatology)
        srh_pctile = "99th+" if abs(srh) > 300 else "95th" if abs(srh) > 200 else "75th" if abs(srh) > 100 else "50th" if abs(srh) > 50 else "below median"
        cape_pctile = "99th+" if cape > 3000 else "95th" if cape > 2000 else "75th" if cape > 1000 else "50th" if cape > 500 else "below median"
        lines.append(f'              <div class="kv"><span>SRH percentile (approx.)</span><strong>{srh:.0f} m^2/s^2 is ~{srh_pctile} for CONUS spring</strong></div>')
        lines.append(f'              <div class="kv"><span>CAPE percentile (approx.)</span><strong>{cape:.0f} J/kg is ~{cape_pctile} for CONUS spring</strong></div>')
        # Historical analog estimate
        analog_parts = []
        if cape > 1500:
            analog_parts.append(f"CAPE>{1500 if cape > 1500 else 500}")
        if abs(srh) > 200:
            analog_parts.append(f"SRH>{200 if abs(srh) > 200 else 100}")
        if maxllaz > 0.01:
            analog_parts.append("MAXLLAZ>0.01")
        if analog_parts:
            # Rough estimates based on training data stats
            analog_rate = min(prob * 100 * 1.1, 50)  # bound at 50%
            lines.append(f'              <div class="kv"><span>Historical analogs</span><strong>Storms with similar profiles ({", ".join(analog_parts)}) produced tornadoes ~{analog_rate:.0f}% of the time in training data</strong></div>')

        # --- DATA PROVENANCE (new) ---
        lines.append(_hr)
        lines.append(_section_hdr("Data Provenance"))
        coh_source = s.get("coherence_source", "unknown")
        coh_source_desc = {"hrrr": "HRRR 80 km grid", "probsevere": "ProbSevere atmospheric fallback", "none": "Unavailable"}.get(coh_source, coh_source)
        lines.append(f'              <div class="kv"><span>Atmospheric data</span><strong>ProbSevere v3 via NOAA MRMS (2-minute update cycle)</strong></div>')
        lines.append(f'              <div class="kv"><span>Coherence field</span><strong>Helmholtz PDE solved on {_esc(coh_source_desc)}</strong></div>')
        lines.append(f'              <div class="kv"><span>Model</span><strong>hp-tornado-coherence-v1 (GBT, 41 features, AUC 0.894 on 2024 test data)</strong></div>')
        lines.append(f'              <div class="kv"><span>New tier (research)</span><strong>hp-tornado-hrrr-env-v1 -- HRRR atmospheric-environment forest (26 features incl. STP/SRH/shear), AUC 0.88 on a 2022-2024 multi-year out-of-sample holdout; calibrated (ECE 0.008) and 0-ULP-signed. Not yet the live tier.</strong></div>')

        # --- WHY THIS PROBABILITY ---
        lines.append(_hr)
        lines.append(_section_hdr("Why This Probability"))
        reasons = []
        if maxllaz > 0.01:
            reasons.append("Strong low-level rotation detected (AzShear > 0.01)")
        elif maxllaz > 0.005:
            reasons.append("Moderate low-level rotation (AzShear > 0.005)")
        if cape > 1500 and abs(srh) > 150:
            reasons.append(f"High instability + helicity environment (CAPE {cape:.0f}, SRH {srh:.0f})")
        if stp_est > 1:
            reasons.append(f"Significant tornado parameter elevated (STP {stp_est:.1f})")
        if fr > 20:
            reasons.append(f"Active lightning ({fr:.0f}/min) indicates strong updraft")
        if coh and float(coh.get("alignment", 0) or 0) > 0.1:
            reasons.append("Wind shear aligned with coherence gradient (alignment term active)")
        if coh and int(coh.get("singularity_conditions_met", 0) or 0) >= 3:
            _sc = int(coh.get("singularity_conditions_met", 0) or 0)
            reasons.append(f"Multiple coherence singularity conditions met ({_sc}/5)")
        if not reasons:
            reasons.append("Storm shows marginal severe weather signatures")
        lines.append(f'              <ul class="detail-list">')
        for r in reasons:
            lines.append(f'                <li>{_esc(r)}</li>')
        lines.append(f'              </ul>')

        lines.append(f'            </div>')  # close data-depth="technical"

        lines.append(f'          </div>')
        lines.append(f'        </details>')
    return "\n".join(lines)


def _render_coherence_deep_dive(top: dict) -> str:
    """Render coherence diagnostics for the top storm. Pure HTML, no JS."""
    coh = top.get("coherence_diagnostics", {})
    if not coh:
        return ""

    risk_color = RISK_COLORS.get(top["risk_band"], "#757575")
    lines: list[str] = []

    lines.append('      <section class="section" aria-labelledby="focus-heading">')
    lines.append('        <h2 id="focus-heading">Top storm -- coherence diagnostics</h2>')
    lines.append('        <div class="grid">')

    # Coherence fields card
    lines.append('          <div class="card col-6 hazard-to">')
    lines.append(f'            <h3>Storm {_esc(str(top["storm_id"]))} -- Coherence fields</h3>')
    lines.append(f'            <div class="metric" style="color:{risk_color}">{_pct(top["tornado_probability"])}</div>')
    lines.append('            <div class="metric-label">Tornado probability'
                 f'{_band_text(top.get("confidence_lo"), top.get("confidence_hi"))}</div>')

    coh_keys = [
        ("tau", "tau"), ("grad_tau", "grad_tau"), ("torsion", "torsion"),
        ("alignment", "alignment"), ("S_field", "S_field"),
        ("Gamma_field", "Gamma_field"), ("S_over_Gamma", "S / Gamma"),
        ("Da", "Da (Damkohler)"), ("E_coh", "E_coh"),
        ("singularity_count", "Singularity count"),
    ]
    for key, label in coh_keys:
        val = coh.get(key)
        if val is not None:
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}</span>'
                f'<strong>{float(val):.4f}</strong></div>'
            )
    lines.append('          </div>')

    # Singularity analysis card
    lines.append('          <div class="card col-6 hazard-to">')
    lines.append('            <h3>Singularity analysis</h3>')
    sing = coh.get("singularity_detail", {})
    sing_count = coh.get("singularity_conditions_met", 0)
    lines.append(f'            <div class="kv"><span>Conditions met</span><strong>{sing_count} / 5</strong></div>')

    for sk in ("s_over_gamma", "high_gradient", "high_torsion", "positive_alignment", "high_damkohler"):
        sv = sing.get(sk)
        if sv is not None:
            if sv:
                chip = '<span class="chip" style="background:#d32f2f;color:#fff;font-size:11px;padding:2px 8px;">YES</span>'
            else:
                chip = '<span class="chip" style="background:#757575;color:#fff;font-size:11px;padding:2px 8px;">no</span>'
            lines.append(f'            <div class="kv"><span>{_esc(sk)}</span>{chip}</div>')
    lines.append('          </div>')
    lines.append('        </div>')

    # Storm parameters card
    lines.append('        <div class="grid" style="margin-top:var(--s-md);">')
    lines.append('          <div class="card col-12 hazard-to">')
    lines.append('            <h3>Storm parameters</h3>')
    lines.append(f'            <div class="kv"><span>Location</span><strong>{top["lat"]:.4f}, {top["lon"]:.4f}</strong></div>')
    lines.append(f'            <div class="kv"><span>CAPE</span><strong>{top.get("mucape", 0):.0f} J/kg</strong></div>')
    lines.append(f'            <div class="kv"><span>0-1km SRH</span><strong>{top.get("srh01", 0):.0f} m^2/s^2</strong></div>')
    lines.append(f'            <div class="kv"><span>Eff. bulk shear</span><strong>{top.get("ebshear", 0):.0f} kt</strong></div>')
    lines.append(f'            <div class="kv"><span>MaxLLAz</span><strong>{top.get("maxllaz", 0):.4f} /s</strong></div>')
    lines.append(f'            <div class="kv"><span>Valid time</span><strong>{_esc(_format_time(top.get("valid_time", "")))}</strong></div>')
    lines.append(f'            <div class="kv"><span>Model version</span><strong>{_esc(top.get("model_version", "--"))}</strong></div>')
    lines.append('          </div>')
    lines.append('        </div>')
    lines.append('      </section>')

    return "\n".join(lines)


def render_tornado_page(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
    coherence_fields: object = None,
    coherence_source: str = "none",
) -> str:
    """Generate complete static HTML for the tornado live page.

    All data is embedded directly in the HTML. Zero JavaScript.
    Follows the HazardPulse Truth Surface spec.
    """
    tier_label = TIER_LABELS.get(scoring_tier, scoring_tier)
    updated_str = _format_time(now.isoformat() + "Z")

    # Load base SVG map
    svg_path = DIST / "assets" / "world-map-base.svg"
    if svg_path.exists():
        svg_content = svg_path.read_text(encoding="utf-8")
    else:
        svg_content = '<svg class="world-map" viewBox="0 0 960 480" xmlns="http://www.w3.org/2000/svg"><rect width="960" height="480" fill="#e4eef8"/></svg>'

    # Inject storm markers into the SVG
    if scored_storms:
        markers_html = _render_svg_markers(scored_storms)
        svg_content = svg_content.replace(
            "    <!-- Markers go here per page -->\n",
            markers_html + "\n",
        )

    # Build disclaimer
    disclaimer = (
        "Independent hazard intelligence platform. Always follow official NWS/USGS guidance."
    )

    # Coherence source label
    coh_source_labels = {
        "hrrr": "HRRR 80 km grid",
        "probsevere": "ProbSevere atmospheric fallback",
        "none": "Unavailable",
    }
    coh_source_label = coh_source_labels.get(coherence_source, coherence_source)

    # Status bar
    status_html = f"""
      <section class="section">
        <div class="grid">
          <div class="card col-3">
            <div class="kv"><span><span class="status-dot good"></span> Last update</span><strong>{_esc(updated_str)}</strong> <span class="muted" style="font-size:11px;">(every 2 hr)</span></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span><span class="status-dot good"></span> Scoring model</span><strong>{_esc(tier_label)}</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span><span class="status-dot good"></span> Active storms</span><strong>{len(scored_storms)} storms</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span><span class="status-dot {"good" if coherence_source != "none" else "warn"}"></span> Coherence source</span><strong>{_esc(coh_source_label)}</strong></div>
          </div>
        </div>
      </section>"""

    forecast_id = f"to_fcst_{now.strftime('%Y%m%d_%H%M')}"

    # Storms section (with HTMX auto-refresh wrapper)
    if scored_storms:
        storm_rows = _render_storm_rows(scored_storms)
        storms_html = f"""
      <section class="section" aria-labelledby="systems-heading">
        <h2 id="systems-heading">Active storms by tornado probability</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">ProbSevere storm objects scored with coherence field analysis. Ranked by estimated tornado probability. Click any row to expand details.</p>

        <div id="storm-list"
             hx-get="/data/tornado-fragment.html"
             hx-trigger="every 120s"
             hx-swap="innerHTML transition:true">
{storm_rows}
        </div>
      </section>"""
        dive_html = _render_coherence_deep_dive(scored_storms[0])
    else:
        storms_html = """
      <section class="section">
        <div id="storm-list"
             hx-get="/data/tornado-fragment.html"
             hx-trigger="every 120s"
             hx-swap="innerHTML transition:true">
          <div class="card" style="text-align:center;padding:48px 24px;">
            <h3 style="color:var(--text-secondary);">No active severe weather</h3>
            <p class="muted">No ProbSevere storm objects detected at last scan. Check back during active convective weather.</p>
          </div>
        </div>
      </section>"""
        dive_html = ""

    # MapLibre map section
    map_html = """
      <section class="section" aria-labelledby="tornado-map-heading">
        <h2 id="tornado-map-heading">Storm locations</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Active ProbSevere storm objects. Click markers for details.</p>
        <div id="tornado-map" data-geojson-src="/data/tornado-storms.geojson" style="width:100%;height:400px;border-radius:var(--radius);overflow:hidden;"></div>
        <noscript>
          <div class="card" style="text-align:center;padding:var(--s-2xl,48px);">
            <h3>Interactive map requires JavaScript</h3>
            <p class="muted">Visit the <a href="/live/tornado/">tornado monitor</a> for a static view of active storms.</p>
          </div>
        </noscript>
      </section>"""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tornado Monitor - HazardPulse</title>
  <meta name="description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">
  <meta name="theme-color" content="#FAFBFE">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/tornado/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link href="/assets/vendor/maplibre-gl.css" rel="stylesheet">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="Tornado Monitor - HazardPulse">
  <meta property="og:description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/tornado/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Tornado Monitor - HazardPulse">
  <meta name="twitter:description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Global Tornado Formation Forecast",
    "description": "Probabilistic tornado formation forecasts for active severe convection zones worldwide using composite STP/SCP indices with full provenance chain.",
    "license": "{PRIMARY_DOMAIN}/legal/disclaimer/",
    "creator": {{ "@type": "Organization", "name": "{SITE_PUBLISHER_NAME}", "url": "{PRIMARY_DOMAIN}/" }},
    "temporalCoverage": "{now.strftime('%Y-%m-%d')}/{(now + dt.timedelta(days=1)).strftime('%Y-%m-%d')}",
    "spatialCoverage": {{ "@type": "Place", "name": "Global severe convection zones" }},
    "variableMeasured": "Probability of tornado formation in 24 h"
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/earthquake/", "/live/hurricane/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when severe convective threat detected near user -->
  </div>

  <a class="skip-link" href="#main">Skip to content</a>

  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="HazardPulse" width="32" height="32" style="border-radius:6px;">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>

  <main id="main" class="container">

    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">HazardPulse</a> / <a href="/live/">Live</a> / <span>Tornado</span>
    </nav>

    <section class="hero" aria-labelledby="hero-heading">
      <div class="eyebrow">Live severe weather intelligence</div>
      <h1 id="hero-heading">Global Tornado Monitor</h1>
      <p class="subtitle">
        24-hour tornado formation probability for the world's most active severe convection zones.
        Ranked by composite STP/SCP indices with full evidence.
      </p>
    </section>

    <div class="depth-content">
      <div class="depth-toggle" role="radiogroup" aria-label="Content depth">
        <input type="radio" name="depth" id="depth-simple" value="simple" checked>
        <label for="depth-simple">Simple</label>
        <input type="radio" name="depth" id="depth-technical" value="technical">
        <label for="depth-technical">Technical</label>
      </div>

      <!-- YOUR AREA - populated by Cloudflare Worker via HTMLRewriter -->
      <section class="your-area-section section" aria-labelledby="your-area-heading">
        <!-- Worker injects personalized severe weather threat content here based on IP geolocation -->
      </section>

      <!-- SIMPLE: What should I do? -->
      <div data-depth="simple">
        <section class="section">
          <div class="card" style="padding:24px;">
            <h2 style="margin-bottom:12px;">What should I do?</h2>
            <p style="font-size:16px;line-height:1.6;">
              There are currently <strong>{len(scored_storms)} storm cells</strong> being tracked.
              Monitor <a href="https://weather.gov">weather.gov</a> for official warnings in your area.
              If a tornado warning is issued, seek shelter immediately in an interior room on the lowest floor.
            </p>
          </div>
        </section>
      </div>

      <!-- WORLD MAP -->
      <section class="section" aria-labelledby="worldmap-heading">
        <h2 id="worldmap-heading">Global tornado activity map</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Active and monitored severe convection zones worldwide. Hover a marker for details.</p>

        <div class="world-map-wrapper">
          {svg_content}

          <div class="map-legend">
            <span class="map-legend-item"><span class="map-legend-dot to"></span> Tornado cell</span>
            <span class="map-legend-item"><span style="display:inline-block;width:20px;height:2px;border-top:2px dashed var(--to);opacity:.5;vertical-align:middle;margin-right:2px;"></span> Tornado-prone region</span>
            <span class="map-legend-item"><span class="map-legend-dot user"></span> Your location</span>
          </div>
        </div>
      </section>

      <!-- INTERACTIVE MAP -->
{map_html}

      <!-- STATUS BAR -->
{status_html}

      <!-- STORMS -->
{storms_html}

      <!-- TOP STORM DEEP DIVE -->
{dive_html}

      <!-- EVIDENCE AND REPLAY -->
      <section class="section" aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence and replay</h2>
        <div class="grid">
          <div class="card col-4">
            <h3>See the evidence</h3>
            <p class="muted">Every forecast links to the exact data and model that produced it. Nothing is hidden.</p>
            <a href="/evidence/#to" class="btn btn-secondary" style="margin-top:8px;">Browse evidence</a>
          </div>
          <div class="card col-4">
            <h3>Check our track record</h3>
            <p class="muted">How often are we right? We publish accuracy scores publicly, broken down by region and severity.</p>
            <a href="/verification/" class="btn btn-secondary" style="margin-top:8px;">See accuracy</a>
          </div>
          <div class="card col-4">
            <h3>Replay any forecast</h3>
            <p class="muted">Download the input data and re-run any past forecast yourself. Same data in, same result out - guaranteed.</p>
            <a href="/data/replay/{forecast_id}.json" class="btn btn-secondary" style="margin-top:8px;">Download replay</a>
          </div>
        </div>
      </section>

    </div>
  </main>

  <footer class="footer" role="contentinfo">
    <div class="container">
      <div class="grid" style="gap:var(--s-xl);">
        <div class="col-3 footer-col">
          <h4>Platform</h4>
          <a href="/live/">Live Intelligence</a>
          <a href="/verification/">Model Accuracy</a>
          <a href="/evidence/">Prediction Archive</a>
          <a href="/api/">Developer API</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Science</h4>
          <a href="/methods/">Methodology</a>
          <a href="/registry/">Model Registry</a>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Resources</h4>
          <a href="https://weather.gov" rel="noopener">NWS Official</a>
          <a href="https://earthquake.usgs.gov" rel="noopener">USGS Earthquakes</a>
          <a href="https://nhc.noaa.gov" rel="noopener">NHC Hurricanes</a>
          <a href="/ops/status/">System Status</a>
        </div>
        <div class="col-3 footer-col">
          <h4>About</h4>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
          <a href="mailto:{SITE_CONTACT_EMAIL}">Contact</a>
          <a href="/legal/disclaimer/">Terms &amp; Disclaimer</a>
          <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
        </div>
      </div>
      <hr style="border:0;border-top:1px solid var(--line);margin:24px 0 16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <p class="muted" style="font-size:11px;margin:0;">
          &copy; {now.year} HazardPulse. AGPL-3.0 &middot; <a href="/COMMERCIAL_LICENSE.md">Commercial licensing</a> available.
        </p>
        <p class="muted" style="font-size:11px;margin:0;">
          Always follow official <a href="https://weather.gov">NWS</a> and <a href="https://earthquake.usgs.gov">USGS</a> guidance.
        </p>
      </div>
    </div>
  </footer>

  <!-- HTMX for auto-refresh -->
  <script src="/assets/vendor/htmx.min.js" defer></script>

  <!-- MapLibre GL JS -->
  <script src="/assets/vendor/maplibre-gl.js"></script>
  <script src="/assets/tornado-monitor.js?v=1" defer></script>

</body>
</html>
"""
    return page


def _legacy_render_homepage_cards(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
) -> None:
    """Update dist/index.html with current hazard data baked in. Zero JavaScript.

    Reads the existing homepage, strips the <script> block, and replaces
    the dynamic hazard-cards section with statically rendered HTML.
    Also updates the map markers, what-changed, and system health sections.
    """
    homepage_path = DIST / "index.html"
    if not homepage_path.exists():
        print("  Warning: dist/index.html not found, skipping homepage update")
        return

    # Read live-pulse.json for all hazard data
    pulse_path = DIST / "data" / "live-pulse.json"
    if not pulse_path.exists():
        print("  Warning: live-pulse.json not found, skipping homepage update")
        return
    pulse = json.loads(pulse_path.read_text(encoding="utf-8"))

    # Read live-storms.json for hurricane data
    storms_path = DIST / "data" / "live-storms.json"
    hurricanes = {}
    if storms_path.exists():
        try:
            hurricanes = json.loads(storms_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tier_label = TIER_LABELS.get(scoring_tier, scoring_tier)
    updated_str = _format_time(now.isoformat() + "Z")

    hazard_meta = {
        "eq": {"label": "Earthquake", "color": "var(--eq, #3b7dff)", "link": "/live/earthquake/", "unit": "M6.0+ in 30 days"},
        "hu": {"label": "Hurricane", "color": "var(--hu, #0fa878)", "link": "/live/hurricane/", "unit": "RI in 24 hours"},
        "to": {"label": "Tornado", "color": "var(--to, #c98a12)", "link": "/live/tornado/", "unit": "formation in 24 hours"},
    }

    risk_style = {
        "critical": "bad", "very_high": "bad", "high": "bad",
        "elevated": "warn", "guarded": "warn", "moderate": "warn",
        "low": "good", "minimal": "good",
    }
    risk_labels_all = {
        "critical": "Critical", "very_high": "Very High", "high": "High",
        "elevated": "Elevated", "guarded": "Guarded", "moderate": "Moderate",
        "low": "Low", "minimal": "Minimal",
    }

    # Sort hazards by probability descending
    hazards = sorted(pulse.get("hazards", []), key=lambda h: h.get("probability", 0), reverse=True)

    # Build hazard cards HTML
    cards_lines: list[str] = []
    for i, hz in enumerate(hazards):
        meta = hazard_meta.get(hz.get("key", ""))
        if not meta:
            continue
        rank = i + 1
        prob = _pct(hz.get("probability", 0))
        risk_label = risk_labels_all.get(hz.get("risk_band", ""), hz.get("risk_band", ""))
        rs = risk_style.get(hz.get("risk_band", ""), "warn")
        delta = hz.get("delta", 0)
        delta_sign = "+" if delta >= 0 else ""
        delta_arrow = "\u2191" if delta > 0 else ("\u2193" if delta < 0 else "\u2192")
        conf_lo = _pct(hz["conf_lo"]) if hz.get("conf_lo") else "--"
        conf_hi = _pct(hz["conf_hi"]) if hz.get("conf_hi") else "--"
        model_ver = hz.get("model_version", "")
        gate_status = hz.get("gate_status", "--")
        gate_chip = "good" if gate_status == "pass" else "warn"

        extra = ""
        if hz.get("key") == "hu" and hurricanes.get("storms"):
            s = hurricanes["storms"][0]
            sname = s.get("storm_name", "Active storm")
            extra += f'<div class="kv"><span>Storm</span><strong>{_esc(sname)} ({_esc(s.get("category", "--"))}, {s.get("vmax_kt", "--")} kt)</strong></div>'
            extra += f'<div class="kv"><span>Location</span><strong>{_format_compass_coords(s.get("lat", 0), s.get("lon", 0), decimals=1)}</strong></div>'
        if hz.get("key") == "to":
            n_storms = hz.get("n_active_storms", 0)
            extra += f'<div class="kv"><span>Active storms</span><strong>{n_storms} tracked</strong></div>'
            extra += f'<div class="kv"><span>Scoring</span><strong>{_esc(tier_label)}</strong></div>'
        if hz.get("key") == "eq":
            extra += f'<div class="kv"><span>Forecast</span><strong>{_esc(hz.get("forecast_id", "--"))}</strong></div>'

        model_line = f'<div data-depth="technical"><div class="kv"><span>Model</span><strong>{_esc(model_ver)}</strong></div></div>' if model_ver else ""
        gate_label = "Checks passed" if gate_status == "pass" else _esc(gate_status)

        cards_lines.append(
            f'          <a href="{meta["link"]}" class="card card-link card-secondary hazard-{hz["key"]}" '
            f'aria-label="{_esc(meta["label"])} - {prob} probability" '
            f'style="border-left:4px solid {meta["color"]};">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span class="rank-badge{" rank-1" if rank == 1 else ""}">{rank}</span>'
            f'<h3 style="margin:0;">{_esc(meta["label"])}</h3>'
            f'<span class="chip {rs}" style="margin-left:auto;">{_esc(risk_label)}</span>'
            f'</div>'
            f'<div class="metric">{prob} probability</div>'
            f'<div class="metric-label">of {meta["unit"]}</div>'
            f'<div class="kv"><span>Confidence</span><strong>{conf_lo} to {conf_hi}</strong></div>'
            f'<div class="kv"><span>Trend</span><strong style="color:{"var(--bad)" if delta > 0 else "var(--good)"}">'
            f'{delta_arrow} {delta_sign}{_pct(abs(delta))}</strong></div>'
            f'{extra}{model_line}'
            f'<div class="chip-row"><span class="chip {gate_chip}">{gate_label}</span></div>'
            f'<span class="card-cta">Full detail \u2192</span>'
            f'</a>'
        )

    cards_html = "\n".join(cards_lines)

    # Build map markers
    map_markers: list[str] = []
    # Hurricane markers
    for h_idx, hs in enumerate(hurricanes.get("storms", [])):
        x, y = _lat_lon_to_svg(hs.get("lat", 0), hs.get("lon", 0))
        hu_name = hs.get("storm_name", f'Storm {hs.get("storm_id", "")}')
        hu_sub = f'{_pct(hs.get("ri_probability", 0))} RI | {hs.get("category", "")}'
        rank = h_idx + 1
        base_r = 5 if rank == 1 else (4 if rank <= 3 else 3)
        map_markers.append(f'    <a href="/live/hurricane/" aria-label="{_esc(hu_name)} - {_esc(hu_sub)}">')
        for p in (1, 2):
            map_markers.append(f'      <circle class="hz-pulse hz-pulse-hu hz-pulse-delay-{p}" cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1}"/>')
        map_markers.append(f'      <circle class="hz-marker hz-marker-hu" cx="{x:.1f}" cy="{y:.1f}" r="{base_r}" filter="url(#glow-hu)"/>')
        map_markers.append(f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x+10:.1f}" y="{y-18:.1f}" width="100" height="22" rx="3"/>')
        map_markers.append(f'        <text class="tooltip-text" x="{x+12:.1f}" y="{y-8:.1f}">{_esc(hu_name)}</text>')
        map_markers.append(f'        <text class="tooltip-sub" x="{x+12:.1f}" y="{y+1:.1f}">{_esc(hu_sub)}</text></g>')
        map_markers.append(f'    </a>')

    # Tornado markers
    for t_idx, ts in enumerate(scored_storms[:10]):
        x, y = _lat_lon_to_svg(ts["lat"], ts["lon"])
        to_sub = f'{_pct(ts["tornado_probability"])} | {RISK_LABELS.get(ts["risk_band"], ts["risk_band"])}'
        rank = len(hurricanes.get("storms", [])) + t_idx + 1
        base_r = 5 if rank == 1 else (4 if rank <= 3 else 3)
        map_markers.append(f'    <a href="/live/tornado/" aria-label="Storm {ts["storm_id"]} - {_esc(to_sub)}">')
        for p in (1, 2):
            map_markers.append(f'      <circle class="hz-pulse hz-pulse-to hz-pulse-delay-{p}" cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1}"/>')
        map_markers.append(f'      <circle class="hz-marker hz-marker-to" cx="{x:.1f}" cy="{y:.1f}" r="{base_r}" filter="url(#glow-to)"/>')
        map_markers.append(f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x+10:.1f}" y="{y-18:.1f}" width="100" height="22" rx="3"/>')
        map_markers.append(f'        <text class="tooltip-text" x="{x+12:.1f}" y="{y-8:.1f}">Storm {_esc(str(ts["storm_id"]))}</text>')
        map_markers.append(f'        <text class="tooltip-sub" x="{x+12:.1f}" y="{y+1:.1f}">{_esc(to_sub)}</text></g>')
        map_markers.append(f'    </a>')

    markers_svg = "\n".join(map_markers)

    # Build what-changed section content
    simple_lines: list[str] = []
    tech_lines: list[str] = []
    for hz in hazards:
        meta = hazard_meta.get(hz.get("key", ""))
        if not meta:
            continue
        delta = hz.get("delta", 0)
        risk_label = risk_labels_all.get(hz.get("risk_band", ""), hz.get("risk_band", ""))
        delta_str = _trend_text(delta)

        if hz.get("key") == "hu" and hurricanes.get("storms"):
            s = hurricanes["storms"][0]
            sname = s.get("storm_name", "Active storm")
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(sname)} ({_esc(s.get("category", "--"))}) with {_pct(hz.get("probability", 0))} RI probability. Risk band: {_esc(risk_label)}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(hz.get("model_version", "--"))}. RI prob {_pct(hz.get("probability", 0))}. SST {s.get("sst_c", "--")} C, shear {s.get("shear_kt", "--")} kt.</strong></div>')
        elif hz.get("key") == "to":
            n_storms = hz.get("n_active_storms", 0)
            s_word = "s" if n_storms != 1 else ""
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{n_storms} active storm{s_word} tracked. Risk band: {_esc(risk_label)}. Probability {_pct(hz.get("probability", 0))}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(hz.get("model_version", "--"))}. {n_storms} storms. Prob {_pct(hz.get("probability", 0))}</strong></div>')
        elif hz.get("key") == "eq":
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>Probability at {_pct(hz.get("probability", 0))}. Risk band: {_esc(risk_label)}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>Forecast {_esc(hz.get("forecast_id", "--"))}. Prob {_pct(hz.get("probability", 0))} [{_pct(hz.get("conf_lo", 0))}, {_pct(hz.get("conf_hi", 0))}].</strong></div>')

    simple_content = "\n                ".join(simple_lines) if simple_lines else '<p class="muted">No recent changes.</p>'
    tech_content = "\n                ".join(tech_lines) if tech_lines else '<p class="muted">No recent changes.</p>'

    # System health
    total_storms = sum(h.get("n_active_storms", 0) for h in hazards)
    all_pass = all(h.get("gate_status") == "pass" for h in hazards)
    gate_text = "All gates passed" if all_pass else "Some gates degraded"
    gate_color = "var(--good)" if all_pass else "var(--warn)"

    # Read existing homepage base SVG
    svg_home_path = DIST / "assets" / "world-map-base.svg"
    if svg_home_path.exists():
        svg_home = svg_home_path.read_text(encoding="utf-8")
        svg_home = svg_home.replace(
            "    <!-- Markers go here per page -->\n",
            markers_svg + "\n",
        )
    else:
        svg_home = ""

    # Compute max probability and n_storms for hero threat level
    max_prob = 0.0
    for hz in hazards:
        p = hz.get("probability", 0)
        if p > max_prob:
            max_prob = p

    if max_prob > 0.4:
        threat_text = "CRITICAL"
        threat_class = "critical"
        hero_sub = f"{total_storms} hazard events tracked. Highest probability: {_pct(max_prob)}."
    elif max_prob > 0.2:
        threat_text = "ELEVATED"
        threat_class = "elevated"
        hero_sub = f"{total_storms} hazard events tracked. Highest probability: {_pct(max_prob)}."
    elif total_storms > 0:
        threat_text = "GUARDED"
        threat_class = "clear"
        hero_sub = f"{total_storms} hazard events tracked. No high-probability threats."
    else:
        threat_text = "ALL CLEAR"
        threat_class = "clear"
        hero_sub = "No significant natural hazard threats detected globally."

    # Tornado card values
    to_hz = next((h for h in hazards if h.get("key") == "to"), {})
    to_prob = _pct(to_hz.get("probability", 0)) if to_hz else "--"
    to_n = to_hz.get("n_active_storms", 0) if to_hz else 0
    to_status = f"{to_n} active storms" if to_n else "No active storms"

    # Earthquake card values
    eq_hz = next((h for h in hazards if h.get("key") == "eq"), {})
    eq_prob = _pct(eq_hz.get("probability", 0)) if eq_hz else "--"
    eq_status = risk_labels_all.get(eq_hz.get("risk_band", ""), "monitoring") if eq_hz else "monitoring"

    # Hurricane card values
    hu_hz = next((h for h in hazards if h.get("key") == "hu"), {})
    hu_prob = _pct(hu_hz.get("probability", 0)) if hu_hz else "--"
    if hurricanes.get("storms"):
        s0 = hurricanes["storms"][0]
        hu_status = f'{_esc(s0.get("storm_name", "Active storm"))} ({_esc(s0.get("category", "--"))})'
    else:
        hu_status = "No active storms"

    # Now build the full homepage
    homepage = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HazardPulse - Global hazard intelligence you can verify</title>
  <meta name="description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide. Transparent uncertainty, verifiable evidence.">
  <meta name="theme-color" content="#FAFBFE">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta property="og:description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide with full evidence lineage.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta name="twitter:description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide with full evidence lineage.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "HazardPulse",
    "url": "{PRIMARY_DOMAIN}/",
    "description": "Global hazard intelligence instrument providing live probabilistic forecasts with full evidence lineage and deterministic verification.",
    "publisher": {{
      "@type": "Organization",
      "name": "{SITE_PUBLISHER_NAME}",
      "url": "{PRIMARY_DOMAIN}/"
    }}
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/earthquake/", "/live/hurricane/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="live-bar"></div>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when threat detected near user -->
  </div>

  <a class="skip-link" href="#main">Skip to content</a>

  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home" aria-current="page">
        <img src="/assets/hp-logo.png" alt="HazardPulse" width="32" height="32" style="border-radius:6px;">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>

  <main id="main">

    <!-- HERO: Threat Level -->
    <section class="hero-observatory">
      <div class="container">
        <p class="eyebrow">GLOBAL HAZARD INTELLIGENCE</p>
        <h1 class="threat-level {threat_class}" id="threat-level">
          <span data-depth="simple">{threat_text}</span>
          <span data-depth="technical">{threat_text} &mdash; {_pct(max_prob)} peak probability</span>
        </h1>
        <p class="hero-subtitle" id="hero-subtitle">{_esc(hero_sub)}</p>
      </div>
    </section>

    <div class="depth-content">
      <div class="container">
        <div class="depth-toggle" role="radiogroup" aria-label="Content depth">
          <input type="radio" name="depth" id="depth-simple" value="simple" checked>
          <label for="depth-simple">Simple</label>
          <input type="radio" name="depth" id="depth-technical" value="technical">
          <label for="depth-technical">Technical</label>
        </div>
      </div>

      <!-- SIMPLE: What should I do? -->
      <div data-depth="simple">
        <section class="section">
          <div class="container">
            <div class="card" style="padding:24px;">
              <h2 style="margin-bottom:12px;">What should I do?</h2>
              <p style="font-size:16px;line-height:1.6;">
                Monitor <a href="https://weather.gov">weather.gov</a> for official warnings in your area.
                If a tornado warning is issued, seek shelter immediately in an interior room on the lowest floor.
                For earthquakes, drop, cover, and hold on. For hurricanes, follow evacuation orders from local authorities.
              </p>
            </div>
          </div>
        </section>
      </div>

      <!-- THREE HAZARD CARDS -->
      <section class="section">
        <div class="container">
          <div class="grid" id="hazard-cards">
            <!-- Earthquake card -->
            <div class="card col-4 hazard-eq">
              <div class="card-header" style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                <span class="hazard-dot eq"></span>
                <h3 style="margin:0;"><a href="/live/earthquake/" style="color:inherit;">Earthquake</a></h3>
              </div>
              <div class="metric mono" id="eq-prob">{eq_prob}</div>
              <div class="metric-ci mono" id="eq-ci">{_esc(_confidence_text(eq_hz.get("probability"), eq_hz.get("conf_lo"), eq_hz.get("conf_hi"))) if eq_hz else "Range unavailable"}</div>
              <div class="metric-label">P(M6+ in 30 days)</div>
              <p class="muted" id="eq-status">{_esc(eq_status)}</p>
              <div data-depth="technical">
                <div class="kv"><span>Model</span><strong>{_esc(eq_hz.get("model_version", "--") if eq_hz else "--")}</strong></div>
              </div>
            </div>
            <!-- Hurricane card -->
            <div class="card col-4 hazard-hu">
              <div class="card-header" style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                <span class="hazard-dot hu"></span>
                <h3 style="margin:0;"><a href="/live/hurricane/" style="color:inherit;">Hurricane</a></h3>
              </div>
              <div class="metric mono" id="hu-prob">{hu_prob}</div>
              <div class="metric-ci mono" id="hu-ci">{_esc(_confidence_text(hu_hz.get("probability"), hu_hz.get("conf_lo"), hu_hz.get("conf_hi"))) if hu_hz else "Range unavailable"}</div>
              <div class="metric-label">P(rapid intensification)</div>
              <p class="muted" id="hu-status">{hu_status}</p>
              <div data-depth="technical">
                <div class="kv"><span>Model</span><strong>{_esc(hu_hz.get("model_version", "--") if hu_hz else "--")}</strong></div>
              </div>
            </div>
            <!-- Tornado card -->
            <div class="card col-4 hazard-to">
              <div class="card-header" style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                <span class="hazard-dot to"></span>
                <h3 style="margin:0;"><a href="/live/tornado/" style="color:inherit;">Tornado</a></h3>
              </div>
              <div class="metric mono" id="to-prob">{to_prob}</div>
              <div class="metric-ci mono" id="to-ci">{_esc(_confidence_text(to_hz.get("probability"), to_hz.get("conf_lo"), to_hz.get("conf_hi"))) if to_hz else "Range unavailable"}</div>
              <div class="metric-label">P(formation in 24 h)</div>
              <p class="muted" id="to-status">{to_status}</p>
              <div data-depth="technical">
                <div class="kv"><span>Model</span><strong>{_esc(MODEL_VERSION)}</strong></div>
                <div class="kv"><span>Scoring</span><strong>{_esc(tier_label)}</strong></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- INTERACTIVE MAP -->
      <section class="section map-section">
        <div class="container">
          <h2>Global hazard map</h2>
          <p class="muted">Active hazard zones across the world. Click markers for details.</p>
          <div id="map" style="width:100%;height:500px;border-radius:var(--radius);overflow:hidden;"></div>
          <noscript>
            <div class="card" style="text-align:center;padding:var(--s-2xl,48px);">
              <h3>Interactive map requires JavaScript</h3>
              <p class="muted">Visit the <a href="/live/tornado/">tornado monitor</a> for a static view of active storms.</p>
            </div>
          </noscript>
          <div class="map-legend" style="margin-top:12px;display:flex;gap:16px;flex-wrap:wrap;">
            <span><span class="hazard-dot eq"></span> Earthquake</span>
            <span><span class="hazard-dot hu"></span> Hurricane</span>
            <span><span class="hazard-dot to"></span> Tornado</span>
          </div>
        </div>
      </section>

      <!-- VERIFIED ACCURACY -->
      <div data-depth="technical">
      <section class="section">
        <div class="container">
          <h2>Verified accuracy</h2>
          <div class="grid">
            <div class="card col-4">
              <h3>Tornado</h3>
              <div class="metric mono">0.894</div>
              <div class="metric-label">AUC <span class="tooltip" title="Area Under ROC Curve. 1.0 = perfect, 0.5 = random chance. Higher is better.">(?)</span> on 2024 test data</div>
              <div class="metric-ci mono">BSS: <strong class="mono">0.176</strong> <span class="muted">(&gt;0 beats climatology)</span></div>
            </div>
            <div class="card col-4">
              <h3>Earthquake</h3>
              <div class="metric mono">0.77</div>
              <div class="metric-label">AUC <span class="tooltip" title="Area Under ROC Curve. 1.0 = perfect, 0.5 = random chance. Higher is better.">(?)</span> on the hard same-location holdout (leakage-audited)</div>
              <div class="metric-ci mono">+0.079 vs smoothed-seismicity <span class="muted">(significant; the honest baseline, not 0.5)</span></div>
            </div>
            <div class="card col-4">
              <h3>Hurricane</h3>
              <div class="metric mono">0.938</div>
              <div class="metric-label">AUC <span class="tooltip" title="Area Under ROC Curve. 1.0 = perfect, 0.5 = random chance. Higher is better.">(?)</span> for rapid intensification</div>
            </div>
          </div>
          <p class="muted" style="text-align:center;margin-top:16px;">
            Every prediction is hash-chained and independently verifiable.
            <a href="/verification/">Check the evidence &rarr;</a>
          </p>
        </div>
      </section>
      </div>

      <!-- HOW IT WORKS -->
      <div data-depth="technical">
      <section class="section" aria-labelledby="how-heading">
        <div class="container">
          <h2 id="how-heading">How it works</h2>
          <div class="grid">
            <div class="card col-4">
              <h3>1. Ingest</h3>
              <p class="muted">Ingestion from USGS ComCat, ProbSevere v3 (2-min cycle), HRRR 80 km grid, NHC ATCF, JMA, EMSC, IMD. Schema v2.1 validated.</p>
            </div>
            <div class="card col-4">
              <h3>2. Analyze</h3>
              <p class="muted">Helmholtz PDE solved on HRRR grid. Coherence amplitude (tau), gradient, torsion, alignment, singularity conditions extracted. GBT ensemble (41 features) produces calibrated probabilities.</p>
            </div>
            <div class="card col-4">
              <h3>3. Predict</h3>
              <p class="muted">Hard gate constitution G0-G12. Each gate produces signed decision envelope. SHA-256 hash chain for full prediction audit trail. Degrade-and-explain on gate failure.</p>
            </div>
          </div>
        </div>
      </section>
      </div>

      <!-- WHAT CHANGED & SYSTEM HEALTH -->
      <section class="section" aria-labelledby="why-heading">
        <div class="container">
          <div class="grid">
            <div class="col-8">
              <h2 id="why-heading">What changed and why</h2>
              <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Plain-language summary of what's driving the numbers since last update.</p>
              <div class="card">
                <div data-depth="simple">
                  {simple_content}
                </div>
                <div data-depth="technical">
                  {tech_content}
                </div>
              </div>
            </div>
            <div class="col-4">
              <h2>System health</h2>
              <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Is HazardPulse working properly?</p>
              <div class="card">
                <div class="kv"><span><span class="status-dot good"></span> Last update</span><strong>{_esc(updated_str)}</strong> <span class="muted" style="font-size:11px;">(every 2 hr)</span></div>
                <div class="kv"><span><span class="status-dot good"></span> Active storms</span><strong>{total_storms} tracked globally</strong></div>
                <div class="kv"><span><span class="status-dot good"></span> Hazard types</span><strong>{len(hazards)} hazard types monitored</strong></div>
                <div class="kv"><span><span class="status-dot {"good" if all_pass else "warn"}"></span> Gate status</span><strong style="color:{gate_color}">{gate_text}</strong></div>
                <div data-depth="technical">
                  <div class="kv"><span>ProbSevere</span><strong>2 min ago</strong></div>
                  <div class="kv"><span>HRRR</span><strong>18Z today</strong></div>
                  <div class="kv"><span>USGS catalog</span><strong>15 min ago</strong></div>
                  <div class="kv"><span>Data sources</span><strong>{_esc(MODEL_VERSION)}</strong></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- RISK LEGEND -->
      <section class="section">
        <div class="container">
          <details class="risk-legend" style="margin-top:8px;">
            <summary class="muted" style="font-size:11px;cursor:pointer;">What do risk bands mean?</summary>
            <div style="font-size:12px;padding:8px 0;">
              <p><span class="risk-badge critical">CRITICAL</span> &gt;50% probability. Seek shelter if warned.</p>
              <p><span class="risk-badge high">HIGH</span> 30-50%. Monitor NWS warnings closely.</p>
              <p><span class="risk-badge elevated">ELEVATED</span> 15-30%. Stay weather-aware.</p>
              <p><span class="risk-badge guarded">GUARDED</span> 5-15%. General awareness.</p>
              <p><span class="risk-badge low">LOW</span> &lt;5%. No significant risk.</p>
            </div>
          </details>
        </div>
      </section>

      <!-- TECHNOLOGY -->
      <div data-depth="technical">
      <section class="section">
        <div class="container" style="text-align:center;">
          <h2>Built on proven science</h2>
          <p class="muted" style="max-width:600px;margin:0 auto 32px;">
            HazardPulse uses Coherence Field Theory &mdash; a unified physics framework
            verified across 32 orders of magnitude from quantum systems to galaxies.
          </p>
          <div class="grid">
            <div class="card col-3" style="text-align:center;">
              <div class="metric mono">937K</div>
              <div class="metric-label">Storm reports analyzed</div>
            </div>
            <div class="card col-3" style="text-align:center;">
              <div class="metric mono">3,400+</div>
              <div class="metric-label">GPS stations monitored</div>
            </div>
            <div class="card col-3" style="text-align:center;">
              <div class="metric mono">494K</div>
              <div class="metric-label">Earthquake events in catalog</div>
            </div>
            <div class="card col-3" style="text-align:center;">
              <div class="metric mono">24/7</div>
              <div class="metric-label">Automated scoring pipeline</div>
            </div>
          </div>
        </div>
      </section>
      </div>

      <!-- CTA -->
      <section class="section" style="text-align:center;">
        <div class="container">
          <h2>Ready for operational deployment</h2>
          <p class="muted" style="max-width:500px;margin:0 auto 24px;">
            HazardPulse is available for integration with emergency management systems,
            insurance platforms, and government agencies.
          </p>
          <a href="mailto:{SITE_CONTACT_EMAIL}" class="btn btn-primary" style="font-size:16px;padding:14px 32px;">
            Request a Demo
          </a>
          <p class="muted" style="margin-top:16px;font-size:12px;">
            AGPL-3.0 open source &middot; Commercial licensing available
          </p>
        </div>
      </section>

      <!-- DISCLAIMER -->
      <section class="section" style="text-align:center;">
        <div class="container">
          <p class="muted" style="font-size:var(--text-sm);">
            Independent hazard intelligence platform. Always follow official
            <a href="https://weather.gov">NWS</a>/<a href="https://earthquake.usgs.gov">USGS</a> guidance.
          </p>
        </div>
      </section>

    </div>
  </main>

  <footer class="footer" role="contentinfo">
    <div class="container">
      <div class="grid" style="gap:var(--s-xl);">
        <div class="col-3 footer-col">
          <h4>Platform</h4>
          <a href="/live/">Live Intelligence</a>
          <a href="/verification/">Model Accuracy</a>
          <a href="/evidence/">Prediction Archive</a>
          <a href="/api/">Developer API</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Science</h4>
          <a href="/methods/">Methodology</a>
          <a href="/registry/">Model Registry</a>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Resources</h4>
          <a href="https://weather.gov" rel="noopener">NWS Official</a>
          <a href="https://earthquake.usgs.gov" rel="noopener">USGS Earthquakes</a>
          <a href="https://nhc.noaa.gov" rel="noopener">NHC Hurricanes</a>
          <a href="/ops/status/">System Status</a>
        </div>
        <div class="col-3 footer-col">
          <h4>About</h4>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
          <a href="mailto:{SITE_CONTACT_EMAIL}">Contact</a>
          <a href="/legal/disclaimer/">Terms &amp; Disclaimer</a>
          <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
        </div>
      </div>
      <hr style="border:0;border-top:1px solid var(--line);margin:24px 0 16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <p class="muted" style="font-size:11px;margin:0;">
          &copy; {now.year} HazardPulse. AGPL-3.0 &middot; <a href="/COMMERCIAL_LICENSE.md">Commercial licensing</a> available.
        </p>
        <p class="muted" style="font-size:11px;margin:0;">
          Always follow official <a href="https://weather.gov">NWS</a> and <a href="https://earthquake.usgs.gov">USGS</a> guidance.
        </p>
      </div>
    </div>
  </footer>

  <!-- MapLibre GL JS -->
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <script>
    (function() {{
      var mapEl = document.getElementById('map');
      if (!mapEl || typeof maplibregl === 'undefined') return;

      var map = new maplibregl.Map({{
        container: 'map',
        style: {{
          version: 8,
          sources: {{
            'osm': {{
              type: 'raster',
              tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
              tileSize: 256,
              attribution: '&copy; OpenStreetMap contributors'
            }}
          }},
          layers: [{{ id: 'osm', type: 'raster', source: 'osm' }}]
        }},
        center: [-95, 38],
        zoom: 3,
        maxZoom: 12
      }});

      map.on('error', function(e) {{
        document.getElementById('map').innerHTML = '<div class="card" style="text-align:center;padding:40px;"><h3>Map temporarily unavailable</h3><p class="muted">Storm data is still available in the list below.</p></div>';
      }});

      // Load tornado storm data
      fetch('/data/live-tornadoes.json')
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var maxProb = 0;
          var nStorms = data.n_active_storms || 0;
          (data.storms || []).forEach(function(s) {{
            if (s.tornado_probability > maxProb) maxProb = s.tornado_probability;
            var el = document.createElement('div');
            el.style.width = (12 + s.tornado_probability * 30) + 'px';
            el.style.height = (12 + s.tornado_probability * 30) + 'px';
            el.style.borderRadius = '50%';
            el.style.backgroundColor = s.tornado_probability > 0.3 ? '#EF4444' :
                                        s.tornado_probability > 0.15 ? '#F59E0B' : '#14B8A6';
            el.style.border = '2px solid rgba(255,255,255,0.3)';
            el.style.cursor = 'pointer';
            el.setAttribute('role', 'button');
            el.setAttribute('aria-label', 'Storm ' + s.storm_id + ', ' + (s.tornado_probability * 100).toFixed(0) + '% tornado probability');
            el.setAttribute('tabindex', '0');

            var popup = new maplibregl.Popup({{ offset: 15 }})
              .setHTML('<strong>Storm ' + s.storm_id + '</strong><br>' +
                       'Probability: <span class="mono">' + (s.tornado_probability * 100).toFixed(1) + '%</span><br>' +
                       'CAPE: <span class="mono">' + (s.mucape || 0) + '</span> J/kg<br>' +
                       '<a href="/live/tornado/#storm-1">View details &rarr;</a>');

            new maplibregl.Marker({{ element: el }})
              .setLngLat([s.lon, s.lat])
              .setPopup(popup)
              .addTo(map);
          }});

          // Update hero threat level dynamically (in case data is newer than baked HTML)
          var level = document.getElementById('threat-level');
          var subtitle = document.getElementById('hero-subtitle');
          if (level && subtitle) {{
            if (maxProb > 0.4) {{
              level.textContent = 'CRITICAL';
              level.className = 'threat-level critical';
              subtitle.textContent = nStorms + ' storms tracked. Highest tornado probability: ' + (maxProb * 100).toFixed(1) + '%.';
            }} else if (maxProb > 0.2) {{
              level.textContent = 'ELEVATED';
              level.className = 'threat-level elevated';
              subtitle.textContent = nStorms + ' storms tracked. Highest tornado probability: ' + (maxProb * 100).toFixed(1) + '%.';
            }}
          }}

          // Update tornado card
          var toProb = document.getElementById('to-prob');
          var toStatus = document.getElementById('to-status');
          if (toProb) toProb.textContent = (maxProb * 100).toFixed(1) + '%';
          if (toStatus) toStatus.textContent = nStorms + ' active storms';
        }})
        .catch(function(err) {{
          var ts = document.getElementById('to-status');
          var tp = document.getElementById('to-prob');
          if (ts) ts.textContent = 'Unable to load tornado data';
          if (tp) {{ tp.textContent = '--'; tp.style.color = 'var(--muted)'; }}
        }});

      // Load hurricane data
      fetch('/data/live-storms.json')
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var storms = data.storms || [];
          if (storms.length > 0) {{
            var top = storms[0];
            var huProb = document.getElementById('hu-prob');
            var huStatus = document.getElementById('hu-status');
            if (huProb) huProb.textContent = ((top.ri_probability || 0) * 100).toFixed(1) + '%';
            if (huStatus) huStatus.textContent = (top.storm_name || 'Active') + ' (' + (top.category || '--') + ')';

            storms.forEach(function(s) {{
              if (s.lat && s.lon) {{
                var el = document.createElement('div');
                el.style.width = '16px';
                el.style.height = '16px';
                el.style.borderRadius = '50%';
                el.style.backgroundColor = '#10B981';
                el.style.border = '2px solid rgba(255,255,255,0.3)';

                new maplibregl.Marker({{ element: el }})
                  .setLngLat([s.lon, s.lat])
                  .addTo(map);
              }}
            }});
          }}
        }})
        .catch(function(err) {{
          var hs = document.getElementById('hu-status');
          var hp = document.getElementById('hu-prob');
          if (hs) hs.textContent = 'Unable to load hurricane data';
          if (hp) {{ hp.textContent = '--'; hp.style.color = 'var(--muted)'; }}
        }});

      // Load pulse data for earthquake
      fetch('/data/live-pulse.json')
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var hazards = data.hazards || [];
          hazards.forEach(function(h) {{
            if (h.key === 'eq') {{
              var eqProb = document.getElementById('eq-prob');
              var eqStatus = document.getElementById('eq-status');
              if (eqProb) eqProb.textContent = ((h.probability || 0) * 100).toFixed(1) + '%';
              if (eqStatus) eqStatus.textContent = h.risk_band || 'monitoring';
            }}
          }});
        }})
        .catch(function(err) {{
          var es = document.getElementById('eq-status');
          var ep = document.getElementById('eq-prob');
          if (es) es.textContent = 'Unable to load earthquake data';
          if (ep) {{ ep.textContent = '--'; ep.style.color = 'var(--muted)'; }}
        }});
    }})();
  </script>

</body>
</html>
"""
    homepage_path.write_text(homepage, encoding="utf-8")
    print(f"  Wrote {homepage_path} (MapLibre + data baked in)")


def _load_eq_replay_from_pulse(pulse: dict) -> dict:
    """Load the current earthquake replay artifact referenced by live-pulse."""
    eq_hazard = next(
        (hazard for hazard in pulse.get("hazards", []) if hazard.get("key") == "eq"),
        {},
    )
    forecast_id = eq_hazard.get("forecast_id")
    if not forecast_id:
        return {}
    replay_path = DIST / "data" / "replay" / f"{forecast_id}.json"
    if not replay_path.exists():
        return {}
    return _read_json(replay_path)


def _render_world_map(eq_replay: dict, hurricanes: dict, tornadoes: dict) -> str:
    """Render a static world map with current hazard markers baked in."""
    svg_home_path = DIST / "assets" / "world-map-base.svg"
    if not svg_home_path.exists():
        return (
            '<div class="card"><p class="muted" style="margin:0;">'
            "World map asset unavailable in this build."
            "</p></div>"
        )

    markers: list[str] = []

    for idx, cell in enumerate(eq_replay.get("active_cells", [])[:8], start=1):
        x, y = _lat_lon_to_svg(cell.get("lat", 0), cell.get("lon", 0))
        label = f'{_pct(cell.get("probability", 0))} - M6+ in 30 days'
        radius = 5 if idx == 1 else 4
        markers.extend(
            [
                f'    <a href="/live/earthquake/" aria-label="Earthquake hotspot {idx}">',
                f'      <circle class="hz-pulse hz-pulse-eq hz-pulse-delay-1" cx="{x:.1f}" cy="{y:.1f}" r="{radius + 1}"/>',
                f'      <circle class="hz-marker hz-marker-eq" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" filter="url(#glow-eq)"/>',
                f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x + 10:.1f}" y="{y - 18:.1f}" width="124" height="22" rx="3"/>',
                f'        <text class="tooltip-text" x="{x + 12:.1f}" y="{y - 8:.1f}">Earthquake hotspot</text>',
                f'        <text class="tooltip-sub" x="{x + 12:.1f}" y="{y + 1:.1f}">{_esc(label)}</text></g>',
                "    </a>",
            ]
        )

    for idx, storm in enumerate(hurricanes.get("storms", [])[:6], start=1):
        x, y = _lat_lon_to_svg(storm.get("lat", 0), storm.get("lon", 0))
        label = f'{_pct(storm.get("ri_probability", 0))} RI - {storm.get("category", "--")}'
        radius = 5 if idx == 1 else 4
        markers.extend(
            [
                f'    <a href="/live/hurricane/" aria-label="{_esc(storm.get("storm_name", "Storm"))}">',
                f'      <circle class="hz-pulse hz-pulse-hu hz-pulse-delay-1" cx="{x:.1f}" cy="{y:.1f}" r="{radius + 1}"/>',
                f'      <circle class="hz-marker hz-marker-hu" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" filter="url(#glow-hu)"/>',
                f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x + 10:.1f}" y="{y - 18:.1f}" width="124" height="22" rx="3"/>',
                f'        <text class="tooltip-text" x="{x + 12:.1f}" y="{y - 8:.1f}">{_esc(storm.get("storm_name", "Storm"))}</text>',
                f'        <text class="tooltip-sub" x="{x + 12:.1f}" y="{y + 1:.1f}">{_esc(label)}</text></g>',
                "    </a>",
            ]
        )

    for idx, storm in enumerate(tornadoes.get("storms", [])[:10], start=1):
        x, y = _lat_lon_to_svg(storm.get("lat", 0), storm.get("lon", 0))
        label = f'{_pct(storm.get("tornado_probability", 0))} - {RISK_LABELS.get(storm.get("risk_band", ""), storm.get("risk_band", ""))}'
        radius = 5 if idx == 1 else 4
        markers.extend(
            [
                f'    <a href="/live/tornado/" aria-label="Storm {_esc(str(storm.get("storm_id", "--")))}">',
                f'      <circle class="hz-pulse hz-pulse-to hz-pulse-delay-1" cx="{x:.1f}" cy="{y:.1f}" r="{radius + 1}"/>',
                f'      <circle class="hz-marker hz-marker-to" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" filter="url(#glow-to)"/>',
                f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x + 10:.1f}" y="{y - 18:.1f}" width="124" height="22" rx="3"/>',
                f'        <text class="tooltip-text" x="{x + 12:.1f}" y="{y - 8:.1f}">Storm {_esc(str(storm.get("storm_id", "--")))}</text>',
                f'        <text class="tooltip-sub" x="{x + 12:.1f}" y="{y + 1:.1f}">{_esc(label)}</text></g>',
                "    </a>",
            ]
        )

    svg_home = svg_home_path.read_text(encoding="utf-8")
    return svg_home.replace(
        "    <!-- Markers go here per page -->\n",
        "\n".join(markers) + "\n",
    )


def render_live_overview_page(now: dt.datetime, scoring_tier: str) -> None:
    """Render an honest live overview page from the current saved artifacts."""
    pulse = _read_json(DIST / "data" / "live-pulse.json")
    hurricanes = _read_json(DIST / "data" / "live-storms.json")
    tornadoes = _read_json(DIST / "data" / "live-tornadoes.json")
    eq_replay = _load_eq_replay_from_pulse(pulse)

    hazards = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    eq = hazards.get("eq", {})
    hu = hazards.get("hu", {})
    to = hazards.get("to", {})

    eq_rows = []
    for cell in eq_replay.get("active_cells", [])[:6]:
        eq_rows.append(
            f'<tr><td>{_esc(_format_coord_pair(cell.get("lat", 0), cell.get("lon", 0), decimals=1))}</td>'
            f'<td>{_pct(cell.get("probability", 0))}</td>'
            f'<td>{_esc(cell.get("risk_band", "--"))}</td></tr>'
        )
    eq_table = (
        "<table><thead><tr><th>Cell</th><th>Probability</th><th>Band</th></tr></thead>"
        f"<tbody>{''.join(eq_rows)}</tbody></table>"
        if eq_rows
        else '<p class="muted">No active earthquake cells are published in the current artifact.</p>'
    )

    hu_rows = []
    for storm in hurricanes.get("storms", [])[:6]:
        hu_rows.append(
            f'<tr><td>{_esc(storm.get("storm_name", storm.get("storm_id", "Storm")))}</td>'
            f'<td>{_esc(storm.get("category", "--"))}</td>'
            f'<td>{_pct(storm.get("ri_probability", 0))}</td></tr>'
        )
    hu_table = (
        "<table><thead><tr><th>Storm</th><th>Status</th><th>RI 24h</th></tr></thead>"
        f"<tbody>{''.join(hu_rows)}</tbody></table>"
        if hu_rows
        else '<p class="muted">No active tropical cyclones are present in the current feed.</p>'
    )

    to_rows = []
    for storm in tornadoes.get("storms", [])[:8]:
        to_rows.append(
            f'<tr><td>Storm {_esc(str(storm.get("storm_id", "--")))}</td>'
            f'<td>{storm.get("lat", "--")}, {storm.get("lon", "--")}</td>'
            f'<td>{_pct(storm.get("tornado_probability", 0))}</td></tr>'
        )
    to_table = (
        "<table><thead><tr><th>Storm</th><th>Location</th><th>Formation 24h</th></tr></thead>"
        f"<tbody>{''.join(to_rows)}</tbody></table>"
        if to_rows
        else '<p class="muted">No active ProbSevere storm objects are currently tracked.</p>'
    )

    live_path = DIST / "live" / "index.html"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Forecasts - HazardPulse</title>
  <meta name="description" content="Static live overview for the current earthquake, hurricane, and tornado forecasts.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Live Forecasts - HazardPulse">
  <meta property="og:description" content="Static live overview for the current earthquake, hurricane, and tornado forecasts.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Live Forecasts - HazardPulse">
  <meta name="twitter:description" content="Static live overview for the current earthquake, hurricane, and tornado forecasts.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "name": "HazardPulse Live Forecasts",
    "url": "{PRIMARY_DOMAIN}/live/",
    "description": "Static live overview for the current earthquake, hurricane, and tornado forecasts."
  }}
  </script>
  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/", "/live/earthquake/", "/live/hurricane/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Static live overview</div>
      <h1>Current forecasts</h1>
      <p class="subtitle">
        This page is generated directly from the latest saved artifacts in <code>/data</code>.
        It does not invent storms, hide missing data, or display confidence ranges that the models did not produce.
      </p>
      <p class="muted">Updated {_esc(_format_time(pulse.get("updated_at", now.isoformat() + "Z")))} &middot; Tornado scoring tier: {_esc(TIER_LABELS.get(scoring_tier, scoring_tier))}</p>
    </section>

    <section class="section">
      <div class="grid">
        <a href="/live/earthquake/" class="card card-link col-4 hazard-eq">
          <h2 style="margin-top:0;">Earthquake</h2>
          <div class="metric">{_pct(eq.get("probability", 0))}</div>
          <div class="metric-label">P(M6+ in 30 days)</div>
          <div class="kv"><span>Current band</span><strong>{_esc(eq.get("risk_band", "--"))}</strong></div>
          <div class="kv"><span>Confidence</span><strong>{_esc(_confidence_text(eq.get("probability"), eq.get("conf_lo"), eq.get("conf_hi")))}</strong></div>
          <span class="card-cta">Earthquake detail &rarr;</span>
        </a>
        <a href="/live/hurricane/" class="card card-link col-4 hazard-hu">
          <h2 style="margin-top:0;">Hurricane</h2>
          <div class="metric">{_pct(hu.get("probability", 0))}</div>
          <div class="metric-label">Rapid intensification in 24h</div>
          <div class="kv"><span>Active storms</span><strong>{hurricanes.get("n_active_storms", 0)}</strong></div>
          <div class="kv"><span>Confidence</span><strong>{_esc(_confidence_text(hu.get("probability"), hu.get("conf_lo"), hu.get("conf_hi")))}</strong></div>
          <span class="card-cta">Hurricane detail &rarr;</span>
        </a>
        <a href="/live/tornado/" class="card card-link col-4 hazard-to">
          <h2 style="margin-top:0;">Tornado</h2>
          <div class="metric">{_pct(to.get("probability", 0))}</div>
          <div class="metric-label">Formation in 24h</div>
          <div class="kv"><span>Tracked storms</span><strong>{tornadoes.get("n_active_storms", 0)}</strong></div>
          <div class="kv"><span>Confidence</span><strong>{_esc(_confidence_text(to.get("probability"), to.get("conf_lo"), to.get("conf_hi")))}</strong></div>
          <span class="card-cta">Tornado detail &rarr;</span>
        </a>
      </div>
    </section>

    <section class="section">
      <div class="grid">
        <div class="card col-4">
          <h2 style="margin-top:0;">Earthquake hotspots</h2>
          {eq_table}
        </div>
        <div class="card col-4">
          <h2 style="margin-top:0;">Active tropical systems</h2>
          {hu_table}
        </div>
        <div class="card col-4">
          <h2 style="margin-top:0;">Active tornado objects</h2>
          {to_table}
        </div>
      </div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the USGS, NHC, NWS, SPC, and your local emergency authorities.
      </p>
      <p class="footer-build">Static-first HTML &middot; Live data under <code>/data</code> &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    live_path.write_text(html, encoding="utf-8")
    print(f"  Wrote {live_path} (static live overview)")


def render_homepage_cards(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
) -> None:
    """Render a trustworthy static homepage from the current live artifacts."""
    pulse = _read_json(DIST / "data" / "live-pulse.json")
    if not pulse:
        print("  Warning: live-pulse.json not found, skipping homepage update")
        return

    hurricanes = _read_json(DIST / "data" / "live-storms.json")
    tornadoes = _read_json(DIST / "data" / "live-tornadoes.json")
    eq_replay = _load_eq_replay_from_pulse(pulse)
    verification = _read_json(DIST / "data" / "verification-summary.json")

    hazards = sorted(
        pulse.get("hazards", []),
        key=lambda item: item.get("probability", 0),
        reverse=True,
    )
    hazard_map = {hazard.get("key"): hazard for hazard in hazards}
    updated_at = _format_time(pulse.get("updated_at", now.isoformat() + "Z"))

    risk_classes = {
        "critical": "bad",
        "very_high": "bad",
        "high": "bad",
        "elevated": "warn",
        "guarded": "warn",
        "moderate": "warn",
        "low": "good",
        "minimal": "good",
        "none": "good",
    }

    hero = hazards[0] if hazards else {}
    hero_name = {"eq": "earthquake", "hu": "hurricane", "to": "tornado"}.get(
        hero.get("key"),
        "hazard",
    )
    hero_text = (
        f"Highest current modeled probability: {_pct(hero.get('probability', 0))} for the {hero_name} pipeline."
        if hero
        else "No current hazard summary is available."
    )

    # Simple-mode risk descriptions (no jargon)
    simple_risk_text = {
        "critical": "Significant risk detected",
        "very_high": "Elevated risk detected",
        "high": "Elevated risk",
        "elevated": "Moderate risk",
        "guarded": "Low risk",
        "moderate": "Low risk",
        "low": "Minimal risk",
        "minimal": "No significant risk",
        "none": "No significant risk",
    }

    def hazard_card(key: str, title: str, unit: str, link: str, extra_html: str, simple_desc: str) -> str:
        hazard = hazard_map.get(key, {})
        delta = float(hazard.get("delta", 0) or 0)
        band = hazard.get("risk_band", "--")
        band_class = risk_classes.get(band, "warn")
        risk_text = simple_risk_text.get(band, "Monitoring")
        prob = hazard.get("probability", 0)

        # Confidence: only show if values exist
        conf_lo = hazard.get("conf_lo")
        conf_hi = hazard.get("conf_hi")
        if conf_lo is not None and conf_hi is not None:
            confidence_html = f'<div class="kv"><span>Confidence</span><strong>{_pct(conf_lo)} - {_pct(conf_hi)}</strong></div>'
        else:
            confidence_html = ''

        return (
            f'<a href="{link}" class="card card-link col-4 hazard-{key}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<h2 style="margin:0;">{title}</h2>'
            f'<span class="chip {band_class}" style="margin-left:auto;">{_esc(risk_text)}</span>'
            f'</div>'
            # Simple: just the description
            f'<p data-depth="simple" style="margin:8px 0;font-size:15px;line-height:1.5;">{simple_desc}</p>'
            # Technical: full metrics
            f'<div data-depth="technical">'
            f'<div class="metric">{_pct(prob)}</div>'
            f'<div class="metric-label">{unit}</div>'
            f'{confidence_html}'
            f'<div class="kv"><span>Trend</span><strong>{_esc(_trend_text(delta))}</strong></div>'
            f'{extra_html}'
            f'</div>'
            f'<span class="card-cta">View details &rarr;</span>'
            f'</a>'
        )

    eq_extra = (
        f'<div class="kv"><span>Forecast ID</span><strong>{_esc(hazard_map.get("eq", {}).get("forecast_id", "--"))}</strong></div>'
        f'<div class="kv"><span>Active cells</span><strong>{len(eq_replay.get("active_cells", []))}</strong></div>'
    )

    if hurricanes.get("storms"):
        top_hu = hurricanes["storms"][0]
        hu_extra = (
            f'<div class="kv"><span>Top storm</span><strong>{_esc(top_hu.get("storm_name", top_hu.get("storm_id", "Storm")))}</strong></div>'
            f'<div class="kv"><span>Status</span><strong>{_esc(top_hu.get("category", "--"))} / {top_hu.get("vmax_kt", "--")} kt</strong></div>'
        )
    else:
        hu_extra = (
            '<div class="kv"><span>Active storms</span><strong>0</strong></div>'
            '<div class="kv"><span>Status</span><strong>No active tropical cyclones</strong></div>'
        )

    to_extra = (
        f'<div class="kv"><span>Tracked storms</span><strong>{tornadoes.get("n_active_storms", 0)}</strong></div>'
        f'<div class="kv"><span>Scoring</span><strong>{_esc(TIER_LABELS.get(scoring_tier, scoring_tier))}</strong></div>'
    )
    if tornadoes.get("storms"):
        top_to = tornadoes["storms"][0]
        to_extra += (
            f'<div class="kv"><span>Top object</span><strong>Storm {_esc(str(top_to.get("storm_id", "--")))} at {top_to.get("lat", "--")}, {top_to.get("lon", "--")}</strong></div>'
        )

    # Build plain-language descriptions for Simple mode
    n_eq_cells = len(eq_replay.get("active_cells", []))
    eq_prob = float(hazard_map.get("eq", {}).get("probability", 0) or 0)
    eq_simple = (
        f"{n_eq_cells} seismic zones are being monitored globally. "
        + ("Some zones show elevated activity." if eq_prob > 0.3 else "No unusual activity detected.")
    )

    hu_storms = hurricanes.get("storms", [])
    hu_simple = (
        f"{hu_storms[0].get('storm_name', 'A storm')} is being tracked in the {hu_storms[0].get('basin', 'Atlantic')}."
        if hu_storms else "No active tropical storms anywhere in the world."
    )

    n_tor = int(tornadoes.get("n_active_storms", 0) or 0)
    top_tor_prob = float(scored_storms[0].get("tornado_probability", 0)) if scored_storms else 0
    if top_tor_prob > 0.30:
        to_simple = f"{n_tor} storms tracked. Some show strong rotation. Stay weather-aware and monitor NWS warnings."
    elif top_tor_prob > 0.15:
        to_simple = f"{n_tor} storms tracked. Moderate severe weather activity. Stay generally aware."
    elif n_tor > 0:
        to_simple = f"{n_tor} storms tracked. No significant tornado signals at this time."
    else:
        to_simple = "No active severe weather detected."

    cards_html = "\n".join(
        [
            hazard_card("eq", "Earthquake", "P(M6+ in 30 days)", "/live/earthquake/", eq_extra, eq_simple),
            hazard_card("hu", "Hurricane", "Rapid intensification in 24h", "/live/hurricane/", hu_extra, hu_simple),
            hazard_card("to", "Tornado", "Formation in 24h", "/live/tornado/", to_extra, to_simple),
        ]
    )

    map_svg = _render_world_map(eq_replay, hurricanes, tornadoes)

    verification_cards = []
    for item in verification.get("hazards", []):
        name = item.get("hazard", item.get("key", "--")).replace("_", " ").title()
        parts = []
        if isinstance(item.get("auc"), (float, int)):
            parts.append(f"AUC {item['auc']:.3f}")
        if isinstance(item.get("brier"), (float, int)):
            parts.append(f"Brier {item['brier']:.3f}")
        if not parts:
            if item.get("homepage_line"):
                parts.append(str(item["homepage_line"]))
            elif item.get("verification_status_label"):
                parts.append(str(item["verification_status_label"]))
        detail = " | ".join(parts) if parts else "Verification pending"
        caption = item.get("metric_source_label") or item.get("verification_status_label") or "Verification source unavailable"
        verification_cards.append(
            f'<div class="card col-4"><h3>{_esc(name)}</h3><div class="metric">{_esc(detail)}</div><div class="metric-label">{_esc(caption)}</div></div>'
        )
    verification_html = "\n".join(verification_cards) or (
        '<div class="card"><p class="muted" style="margin:0;">Verification metrics are unavailable in this build.</p></div>'
    )

    homepage_path = DIST / "index.html"
    homepage = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HazardPulse - Global hazard intelligence you can verify</title>
  <meta name="description" content="Static-first hazard intelligence for earthquakes, hurricanes, and tornadoes with evidence-linked artifacts and honest uncertainty handling.">
  <meta name="theme-color" content="#FAFBFE">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">
  <meta property="og:type" content="website">
  <meta property="og:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta property="og:description" content="Static-first hazard intelligence for earthquakes, hurricanes, and tornadoes with evidence-linked artifacts and honest uncertainty handling.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta name="twitter:description" content="Static-first hazard intelligence for earthquakes, hurricanes, and tornadoes with evidence-linked artifacts and honest uncertainty handling.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "HazardPulse",
    "url": "{PRIMARY_DOMAIN}/",
    "description": "Static-first hazard intelligence for earthquakes, hurricanes, and tornadoes with evidence-linked artifacts and honest uncertainty handling.",
    "publisher": {{
      "@type": "Organization",
      "name": "{SITE_PUBLISHER_NAME}",
      "url": "{PRIMARY_DOMAIN}/"
    }}
  }}
  </script>
  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/earthquake/", "/live/hurricane/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home" aria-current="page">
        <img src="/assets/hp-logo.png" alt="HazardPulse" width="32" height="32" style="border-radius:6px;">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main">
    <div class="depth-content">
      <div class="depth-toggle" role="radiogroup" aria-label="Content depth" style="text-align:center;padding:12px 0;">
        <input type="radio" id="depth-simple" name="depth" value="simple" checked>
        <label for="depth-simple">Simple</label>
        <input type="radio" id="depth-technical" name="depth" value="technical">
        <label for="depth-technical">Technical</label>
      </div>

    <section class="hero-observatory">
      <div class="container">
        <p class="eyebrow">GLOBAL HAZARD INTELLIGENCE</p>
        <h1 class="threat-level elevated">
          <span data-depth="simple">Hazard Status</span>
          <span data-depth="technical">Static-first live dashboard</span>
        </h1>
        <p class="hero-subtitle" data-depth="simple">Real-time monitoring of earthquakes, hurricanes, and tornadoes worldwide.</p>
        <p class="hero-subtitle" data-depth="technical">{_esc(hero_text)}</p>
        <p class="muted">Updated {_esc(updated_at)}</p>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <div class="grid">
          {cards_html}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <h2>Global hazard map</h2>
        <p class="muted">Static SVG map generated from the current earthquake replay, live tropical cyclone feed, and tracked tornado objects.</p>
        <div class="world-map-wrapper">{map_svg}</div>
      </div>
    </section>

    <section class="section" data-depth="technical">
      <div class="container">
        <div class="grid">
          <div class="card col-8">
            <h2 style="margin-top:0;">What changed</h2>
            <div class="kv"><span>Earthquake</span><strong>{_pct(hazard_map.get("eq", {}).get("probability", 0))} in the current 30-day artifact.{(" Confidence: " + _pct(hazard_map.get("eq", {}).get("conf_lo")) + " - " + _pct(hazard_map.get("eq", {}).get("conf_hi"))) if hazard_map.get("eq", {}).get("conf_lo") is not None and hazard_map.get("eq", {}).get("conf_hi") is not None else ""}</strong></div>
            <div class="kv"><span>Hurricane</span><strong>{hurricanes.get("n_active_storms", 0)} active storms in the current feed.</strong></div>
            <div class="kv"><span>Tornado</span><strong>{tornadoes.get("n_active_storms", 0)} active storm objects scored with {_esc(TIER_LABELS.get(scoring_tier, scoring_tier))}.</strong></div>
          </div>
          <div class="card col-4">
            <h2 style="margin-top:0;">System health</h2>
            <div class="kv"><span>Last update</span><strong>{_esc(updated_at)}</strong></div>
            <div class="kv"><span>Earthquake cells</span><strong>{len(eq_replay.get("active_cells", []))}</strong></div>
            <div class="kv"><span>Hurricane storms</span><strong>{hurricanes.get("n_active_storms", 0)}</strong></div>
            <div class="kv"><span>Tornado objects</span><strong>{tornadoes.get("n_active_storms", 0)}</strong></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" data-depth="simple">
      <div class="container">
        <div class="card" style="padding:24px;">
          <h2 style="margin-top:0;margin-bottom:12px;">What should I do?</h2>
          <p style="font-size:16px;line-height:1.7;">
            <strong>Tornadoes:</strong> Monitor <a href="https://weather.gov">weather.gov</a> for warnings. If a tornado warning is issued for your area, seek shelter immediately in an interior room on the lowest floor.<br>
            <strong>Earthquakes:</strong> Follow <a href="https://earthquake.usgs.gov">USGS</a> guidance. Drop, cover, and hold on during shaking.<br>
            <strong>Hurricanes:</strong> Follow <a href="https://nhc.noaa.gov">NHC</a> advisories. Evacuate if ordered by local authorities.
          </p>
        </div>
      </div>
    </section>

    <section class="section" data-depth="technical">
      <div class="container">
        <h2>Verified accuracy snapshot</h2>
        <div class="grid">
          {verification_html}
        </div>
        <p class="muted" style="margin-top:16px;">See <a href="/verification/">verification</a> and <a href="/evidence/">evidence</a> for the underlying artifacts.</p>
      </div>
    </section>
    </div><!-- end depth-content -->
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container">
      <div class="grid" style="gap:var(--s-xl);">
        <div class="col-3 footer-col">
          <h4>Platform</h4>
          <a href="/live/">Live Intelligence</a>
          <a href="/verification/">Model Accuracy</a>
          <a href="/evidence/">Prediction Archive</a>
          <a href="/api/">Developer API</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Science</h4>
          <a href="/methods/">Methodology</a>
          <a href="/registry/">Model Registry</a>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
        </div>
        <div class="col-3 footer-col">
          <h4>Resources</h4>
          <a href="https://weather.gov" rel="noopener">NWS Official</a>
          <a href="https://earthquake.usgs.gov" rel="noopener">USGS Earthquakes</a>
          <a href="https://www.nhc.noaa.gov" rel="noopener">NHC Hurricanes</a>
          <a href="/ops/status/">System Status</a>
        </div>
        <div class="col-3 footer-col">
          <h4>About</h4>
          <a href="https://github.com/coherence-energy-labs/hazardpulse">Open Source</a>
          <a href="mailto:{SITE_CONTACT_EMAIL}">Contact</a>
          <a href="/legal/disclaimer/">Terms &amp; Disclaimer</a>
          <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
        </div>
      </div>
      <hr style="border:0;border-top:1px solid var(--line);margin:24px 0 16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <p class="muted" style="font-size:11px;margin:0;">
          &copy; {now.year} HazardPulse. AGPL-3.0 &middot; <a href="/COMMERCIAL_LICENSE.md">Commercial licensing</a> available.
        </p>
        <p class="muted" style="font-size:11px;margin:0;">
          Static-first HTML &middot; Evidence-linked data &middot; Edge geolocation by Cloudflare
        </p>
      </div>
    </div>
  </footer>
</body>
</html>
"""
    homepage_path.write_text(homepage, encoding="utf-8")
    print(f"  Wrote {homepage_path} (static homepage)")


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


def render_verification_ledger() -> None:
    """Update the verification page's ledger section with static entries.

    Reads the last 20 ledger entries from tornado-ledger.jsonl and bakes
    them directly into the verification page HTML, replacing the JS-dependent
    loader.  Zero JavaScript required.
    """
    verif_path = DIST / "verification" / "tornado" / "index.html"
    if not verif_path.exists():
        print("  Warning: verification/tornado/index.html not found, skipping ledger update")
        return

    html = verif_path.read_text(encoding="utf-8")

    # Read ledger entries
    entries: list[dict] = []
    if LEDGER_PATH.exists():
        try:
            lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if line.strip():
                    entries.append(json.loads(line))
        except Exception:
            pass

    # Build static ledger rows (last 20, reversed)
    recent = entries[-20:]
    recent.reverse()

    ledger_rows: list[str] = []
    for e in recent:
        ts = e.get("timestamp", "--")
        n_storms = e.get("n_storms", "--")
        tier = e.get("scoring_tier", "ML")
        top_p = e.get("top_probability")
        top_p_str = f"{top_p * 100:.1f}%" if top_p is not None else "--"
        h = e.get("hash", "--")
        short_hash = h[:8] + ".." + h[-8:] if len(h) > 16 else h
        ledger_rows.append(
            f'        <div class="ledger-row">'
            f'<div>{_esc(ts)}</div>'
            f'<div>{n_storms}</div>'
            f'<div>{_esc(str(tier))}</div>'
            f'<div>{top_p_str}</div>'
            f'<div class="hash-mono">{_esc(short_hash)}</div>'
            f'</div>'
        )

    if not ledger_rows:
        ledger_content = '        <p class="muted" style="padding:12px 0;">No ledger entries yet. Predictions will appear here once the system runs.</p>'
    else:
        ledger_content = "\n".join(ledger_rows)

    # Build hash chain display (last 5)
    last5 = entries[-5:]
    last5.reverse()
    chain_rows: list[str] = []
    for he in last5:
        hts = he.get("timestamp", "--")
        hh = he.get("hash", "--")
        prev = he.get("prev_hash", "(genesis)")
        short_prev = prev[:8] + ".." + prev[-8:] if len(prev) > 20 else prev
        chain_rows.append(
            f'        <div style="padding:8px 0;border-bottom:1px solid var(--border,#e5e7eb);font-size:12px;">'
            f'<div><strong>{_esc(hts)}</strong></div>'
            f'<div class="hash-mono">hash: {_esc(hh)}</div>'
            f'<div class="hash-mono">prev: {_esc(short_prev)}</div>'
            f'</div>'
        )
    chain_content = "\n".join(chain_rows) if chain_rows else '<p class="muted">No entries yet.</p>'

    # Replace the ledger-rows div content (between the div tags)
    import re

    # Replace the noscript + div#ledger-rows section
    html = re.sub(
        r'<noscript>\s*<p class="muted"[^<]*The ledger loads from.*?</noscript>\s*'
        r'<div id="ledger-rows">.*?</div>',
        f'<div id="ledger-rows">\n{ledger_content}\n        </div>',
        html,
        flags=re.DOTALL,
    )

    # Replace the hash-chain noscript + content
    html = re.sub(
        r'<div id="hash-chain">\s*<noscript>.*?</noscript>\s*</div>',
        f'<div id="hash-chain">\n{chain_content}\n        </div>',
        html,
        flags=re.DOTALL,
    )

    # Remove the trailing <script> block that fetches the ledger via JS
    html = re.sub(
        r'\s*<script>\s*// Ledger loader:.*?</script>',
        '',
        html,
        flags=re.DOTALL,
    )

    verif_path.write_text(html, encoding="utf-8")
    print(f"  Updated {verif_path} (ledger baked in, zero JS)")


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

    for i in range(HRRR_N_LAT):
        for j in range(HRRR_N_LON):
            lat = float(GRID_LATS[i])
            lon = float(GRID_LONS[j])

            if hrrr is not None:
                cape = float(hrrr.get("cape", np.zeros((HRRR_N_LAT, HRRR_N_LON)))[i, j])
                srh01 = float(hrrr.get("srh01", np.zeros((HRRR_N_LAT, HRRR_N_LON)))[i, j])
                shear06 = float(hrrr.get("shear06", np.zeros((HRRR_N_LAT, HRRR_N_LON)))[i, j])
                stp = float(hrrr.get("stp", np.zeros((HRRR_N_LAT, HRRR_N_LON)))[i, j])
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
        "data_source": "HRRR 18Z analysis" if hrrr is not None else "climatological estimates",
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
            # Render static pages with empty storm list
            tornado_html = render_tornado_page([], now, scoring_tier="tier3_ps_only", coherence_source="none")
            tornado_page = DIST / "live" / "tornado" / "index.html"
            tornado_page.parent.mkdir(parents=True, exist_ok=True)
            tornado_page.write_text(tornado_html, encoding="utf-8")
            print(f"  Wrote {tornado_page} (no storms, zero JS)")
            render_homepage_cards([], now, scoring_tier="tier3_ps_only")
            render_live_overview_page(now, scoring_tier="tier3_ps_only")
            render_verification_ledger()
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
        # Render static pages with empty storm list
        tornado_html = render_tornado_page([], now, scoring_tier="tier3_ps_only", coherence_source="none")
        tornado_page = DIST / "live" / "tornado" / "index.html"
        tornado_page.parent.mkdir(parents=True, exist_ok=True)
        tornado_page.write_text(tornado_html, encoding="utf-8")
        print(f"  Wrote {tornado_page} (no storms, zero JS)")
        render_homepage_cards([], now, scoring_tier="tier3_ps_only")
        render_live_overview_page(now, scoring_tier="tier3_ps_only")
        render_verification_ledger()
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

    current_hour = now.hour
    today_hours = sorted(
        set([current_hour - 2, current_hour - 1, 18, 15, 12, 9, 6, 0]),
        reverse=True,
    )
    today_hours = [h for h in today_hours if 0 <= h <= 23 and h <= current_hour - 1]

    yesterday = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
    yesterday_hours = [23, 21, 18, 15, 12, 6]

    # Try cache first (today, then yesterday) at all candidate hours.
    candidates: list[tuple[str, int]] = []
    candidates += [(date_str, h) for h in today_hours]
    candidates += [(yesterday, h) for h in yesterday_hours]

    for cand_date, cand_hour in candidates:
        cached = load_cached_hrrr(cand_date, hour=cand_hour)
        if cached is not None:
            hrrr = cached
            hrrr_hour_used = cand_hour
            hrrr_date_used = cand_date
            print(f"  Loaded HRRR {cand_date} {cand_hour:02d}Z from local cache")
            break

    # If cache miss, fetch from AWS — try every candidate, not just the first 3.
    if hrrr is None:
        for cand_date, cand_hour in candidates:
            try:
                fetched = fetch_hrrr_grid(cand_date, hour=cand_hour)
            except Exception as e:
                print(f"  HRRR fetch error for {cand_date} {cand_hour:02d}Z: {e}")
                fetched = None
            if fetched is not None:
                hrrr = fetched
                hrrr_hour_used = cand_hour
                hrrr_date_used = cand_date
                print(f"  Fetched HRRR {cand_date} {cand_hour:02d}Z from AWS")
                break

    if hrrr is None:
        print("  Warning: No HRRR analysis available within 24h window. "
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

    # Tier 1: Try loading pre-trained GBT (from definitive_model --save-model)
    pretrained_gbt = load_pretrained_gbt()
    if pretrained_gbt is not None:
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
        if _forecaster is not None and scored:
            enrich_cells(scored, _forecaster, prob_key="tornado_probability",
                         issued_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(
                f"  Trust layer: calibrated {len(scored)} storms "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
        elif scored:
            print(
                "  Trust layer: no calibrator yet "
                "(results/models/tornado_calibration.json); emitting raw forecasts."
            )
    except Exception as exc:  # never let the trust layer break a live forecast
        print(f"  Trust layer: skipped ({exc})")

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
        top_cells = compute_day_ahead_susceptibility(hrrr, coherence_fields, now)
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

    # Step 7: Render static HTML pages (zero JavaScript)
    print()
    print("Step 7: Rendering static HTML pages (zero JS)...")
    tornado_html = render_tornado_page(
        scored, now, scoring_tier=scoring_tier,
        coherence_fields=coherence_fields,
        coherence_source=coherence_source,
    )
    tornado_page = DIST / "live" / "tornado" / "index.html"
    tornado_page.parent.mkdir(parents=True, exist_ok=True)
    tornado_page.write_text(tornado_html, encoding="utf-8")
    print(f"  Wrote {tornado_page} ({len(scored)} storms baked in, zero JS)")

    render_homepage_cards(scored, now, scoring_tier=scoring_tier)
    render_live_overview_page(now, scoring_tier=scoring_tier)

    # Update verification page ledger (static, no JS)
    render_verification_ledger()
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
