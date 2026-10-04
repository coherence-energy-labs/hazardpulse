# Independent hazard intelligence platform.
# Always follow official USGS and national seismological agency guidance.
# See earthquake.usgs.gov for authoritative data.
# Always follow guidance from your national geological survey (USGS, JMA, etc.).
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

#!/usr/bin/env python3
"""Fetch USGS earthquake catalog, score with coherence model, output static HTML.

Designed to run every 6 hours via GitHub Actions cron.  Outputs:

  - dist/live/earthquake/index.html   (static HTML page, zero JavaScript)
  - dist/data/live-pulse.json         (updated earthquake entry)
  - dist/data/earthquake-ledger.jsonl (append-only SHA-256 prediction chain)

All data is baked directly into HTML.  Zero JavaScript.
Same architecture as the tornado scorer.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_site_artifacts import build_site_artifacts

from hazardpulse.earthquake.coherence_engine import (  # noqa: E402
    GRID_DLAT,
    GRID_DLON,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    N_LAT,
    N_LON,
    ELL_BACKGROUND_KM,
    B_VALUE_BACKGROUND,
    compute_seismic_coherence_field,
    extract_coherence_features,
    grid_cell_to_latlon,
    latlon_to_grid_cell,
    test_earthquake_singularity,
)

# Optional ML model imports — gracefully degrade if not available
try:
    from hazardpulse.earthquake.definitive_model import (  # noqa: E402
        ALL_FEATURE_NAMES_ENHANCED as DEFINITIVE_EQ_FEATURE_NAMES,
        CatalogArrays,
        compute_block_s,
        compute_block_c,
    )
    HAS_EQ_ML = True
except ImportError as _eq_imp_err:
    HAS_EQ_ML = False
    print(
        f"  WARNING: definitive_model unavailable ({_eq_imp_err}). "
        "Earthquake ML path disabled; falling back to heuristic scorer.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIST = Path(__file__).resolve().parents[1] / "dist"
LEDGER_PATH = DIST / "data" / "earthquake-ledger.jsonl"

# The served probability: the operational 30-day M6+ forecast chosen by the pre-registered
# program (docs/EARTHQUAKE_FORECAST_PROGRAM.md), one artifact = parameters + frozen M5+
# catalog. Its model_version is bound to the artifact's content (CRLF-normalised SHA-256),
# so every forecast, ledger entry and replay names exactly the bytes that produced it.
from hazardpulse.earthquake import operational_forecast as eq_operational  # noqa: E402

OPERATIONAL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "models" / "earthquake_operational_v1.json"
)
# Amendment E1 (section 10): S1 = the C0 artifact above + GEAR1's long-term rate, a small stack
# bound to C0's exact model_version. When present it is what is published.
STACK_PATH = Path(__file__).resolve().parents[1] / "results" / "models" / "earthquake_gear1_stack_v1.json"
# Cells listed on the page / in the replay beyond the active ones: the highest-probability
# cells of the full grid, so a high forecast is never hidden for lack of recent M2.5+ events.
TOP_PROBABILITY_CELLS = 25


def _served_model_version() -> str:
    try:
        if STACK_PATH.exists():
            return eq_operational.stack_model_version(STACK_PATH)
        return eq_operational.artifact_model_version(OPERATIONAL_ARTIFACT_PATH)
    except Exception:  # missing at import time; run_pipeline refuses to publish without it
        return "eq_operational_unavailable"


MODEL_VERSION = _served_model_version()
PRIMARY_DOMAIN = "https://hazardpulse.com"
SITE_PUBLISHER_NAME = "HazardPulse"

# ---------------------------------------------------------------------------
# Risk band mapping
# ---------------------------------------------------------------------------

RISK_BANDS = [
    (0.50, "critical"),
    (0.30, "very_high"),
    (0.15, "elevated"),
    (0.08, "guarded"),
    (0.03, "low"),
    (0.00, "minimal"),
]


def _risk_band(prob: float) -> str:
    """Map probability to risk band.

    Band names deliberately avoid official seismological terminology
    to prevent confusion with authoritative products.
    """
    for threshold, band in RISK_BANDS:
        if prob >= threshold:
            return band
    return "minimal"


RISK_COLORS = {
    "critical": "#b71c1c",
    "very_high": "#d32f2f",
    "elevated": "#e65100",
    "guarded": "#f9a825",
    "low": "#1976d2",
    "minimal": "#757575",
}

RISK_LABELS = {
    "critical": "Critical",
    "very_high": "Very High",
    "elevated": "Elevated",
    "guarded": "Guarded",
    "low": "Low",
    "minimal": "Minimal",
}

# ---------------------------------------------------------------------------
# USGS earthquake catalog fetch
# ---------------------------------------------------------------------------

USGS_CSV_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=csv&starttime={start}&endtime={end}"
    "&minmagnitude=2.5&orderby=time"
)


def fetch_usgs_catalog(
    days: int = 30,
    end_time: dt.datetime | None = None,
) -> list[dict]:
    """Fetch USGS earthquake catalog (M2.5+) for the last N days.

    Returns list of event dicts with keys:
        time, latitude, longitude, depth, mag, magType, place, id
    """
    if end_time is None:
        end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(days=days)

    url = USGS_CSV_URL.format(
        start=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print(f"  Fetching USGS catalog: M2.5+, {days} days...")
    print(f"  URL: {url[:100]}...")

    req = Request(url, headers={"User-Agent": "HazardPulse/1.0 (research)"})
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except URLError as e:
        # Distinct failure mode from "API healthy but no events".
        print(f"  ERROR: USGS catalog fetch failed (network/HTTP): {e}")
        raise RuntimeError(f"USGS FDSNWS unreachable: {e}") from e

    # Validate we got a real CSV response, not an empty body or HTML error page.
    if not raw or not raw.strip():
        raise RuntimeError(
            "USGS returned HTTP 200 with empty body — API may be degraded."
        )
    first_line = raw.splitlines()[0].lower() if raw.splitlines() else ""
    if "time" not in first_line or "latitude" not in first_line:
        raise RuntimeError(
            f"USGS response missing expected CSV header (got first line: "
            f"{first_line[:120]!r})."
        )

    reader = csv.DictReader(io.StringIO(raw))
    events: list[dict] = []
    for row in reader:
        try:
            ev = {
                "time": row.get("time", ""),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "depth": float(row.get("depth", 0) or 0),
                "mag": float(row["mag"]),
                "magType": row.get("magType", ""),
                "place": row.get("place", ""),
                "id": row.get("id", ""),
            }
            events.append(ev)
        except (ValueError, KeyError):
            continue

    if not events:
        # Empty response but healthy API. Unusual for a 30-day global M2.5+
        # window (baseline ~2000/month); log distinctly from a fetch failure.
        print(
            "  WARNING: USGS returned zero events in {}-day window "
            "(API healthy, just no matches). This is unusual for global "
            "M2.5+; verify window parameters.".format(days)
        )
    else:
        print(f"  Fetched {len(events)} events from USGS catalog")
    return events


# ---------------------------------------------------------------------------
# Pre-trained GBT model (plus_cft variant: Block S + Block C = 73 features)
# ---------------------------------------------------------------------------

PRETRAINED_EQ_GBT_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "models" / "earthquake_gbt_v1.json"
)


def load_pretrained_eq_gbt() -> dict | None:
    """Load the pre-trained earthquake GBT if present."""
    if not PRETRAINED_EQ_GBT_PATH.exists():
        return None
    if not HAS_EQ_ML:
        print(
            "  WARNING: earthquake_gbt_v1.json exists but definitive_model is not "
            "importable. Cannot run ML path."
        )
        return None
    try:
        data = json.loads(PRETRAINED_EQ_GBT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARNING: Failed to parse earthquake GBT: {exc}")
        return None
    if data.get("model_format") != "hazardpulse_gbt_v1":
        print(f"  WARNING: Unknown earthquake GBT format in {PRETRAINED_EQ_GBT_PATH.name}")
        return None
    names = data.get("feature_names", [])
    if names != DEFINITIVE_EQ_FEATURE_NAMES:
        print(
            f"  WARNING: earthquake GBT feature_names ({len(names)}) don't match "
            f"definitive_model code ({len(DEFINITIVE_EQ_FEATURE_NAMES)}). "
            "Refusing to use — order mismatch would produce garbage predictions."
        )
        return None
    print(
        f"  Loaded pre-trained earthquake GBT "
        f"({data['n_trees']} trees, {len(names)} features) from "
        f"{PRETRAINED_EQ_GBT_PATH.name}"
    )
    return data


# Optional: the deployable accuracy champion (frozen VerifiableForest). Activates
# only when the offline trainer has exported a never-worse winner; pure-numpy serve.
try:
    from hazardpulse.trust.forest_serve import load_forest_scorer as _load_forest_scorer
    HAS_FOREST_SERVE = True
except Exception:  # pragma: no cover - trust package optional at import time
    HAS_FOREST_SERVE = False


def load_eq_forest(directory=None):
    """Load the exported earthquake VerifiableForest champion if present + compatible.

    The forest serves RAW enhanced features (Block S + Block C, 73 dims). Gate on the
    feature indices fitting that space so a stale/mismatched export can never produce
    garbage. Returns a ForestScorer or None.
    """
    if not HAS_FOREST_SERVE:
        return None
    if directory is None:
        directory = Path(__file__).resolve().parents[1] / "results" / "calibration"
    scorer = _load_forest_scorer("earthquake", directory)
    if scorer is None:
        return None
    max_feat = max((int(f) for f in scorer.constants.get("feat", []) if int(f) >= 0), default=-1)
    n_expected = len(DEFINITIVE_EQ_FEATURE_NAMES)
    if max_feat >= n_expected:
        print(
            f"  WARNING: earthquake forest references feature index {max_feat} >= "
            f"{n_expected} enhanced features. Refusing to use (train/serve mismatch)."
        )
        return None
    print(
        f"  Loaded earthquake VerifiableForest champion "
        f"({len(scorer.constants['tree_root'])} trees) - serves raw enhanced features."
    )
    return scorer


_MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"
# Year-ahead regional nowcast (M5+ within 300km/365d) and short-term local watch
# (M4.5+ within 50km/30d) -- two distinct products, both served torch-free.
DEEP_EQ_SERVE_NPZ = _MODELS_DIR / "eq_deep_nowcast_m5.0_2025_K192.serve.npz"
DEEP_EQ_SHORTTERM_NPZ = _MODELS_DIR / "eq_deep_shortterm_m4.5_r50_d30_ir100_K384.serve.npz"
# Operational forecaster -- the real "which region ruptures next" (M5+ within 100km/30d),
# trained on the operational objective: beats climatology by +0.14 (genuine temporal skill).
DEEP_EQ_OPERATIONAL_NPZ = _MODELS_DIR / "eq_operational_m5_30d_ir150_grid1.serve.npz"


def _load_deep_scorer(npz_path, label):
    """Load a deep GRU scorer for torch-free serving (pure-numpy forward). Served RAW
    (the static val-period calibrator did not transfer; periodic recalibration is the fix)."""
    try:
        from hazardpulse.earthquake.deep_serve import load_deep_eq_scorer
    except Exception as exc:  # pragma: no cover - numpy-only, should import
        print(f"  WARNING: deep_serve import failed ({exc}); deep tier disabled.")
        return None
    scorer = load_deep_eq_scorer(npz_path, calib_path=None)
    if scorer is not None:
        print(f"  Loaded deep {label} (K={scorer.K}, input radius={scorer.radius_km:.0f}km) "
              f"from {npz_path.name}")
    return scorer


def load_deep_eq_scorer_model():
    """Year-ahead regional nowcast champion (the primary tier-1 probability)."""
    return _load_deep_scorer(DEEP_EQ_SERVE_NPZ, "year-ahead regional nowcast")


def load_deep_eq_shortterm_model():
    """Short-term LOCAL watch: P(M4.5+ within 50km / 30 days). A second, distinct field."""
    return _load_deep_scorer(DEEP_EQ_SHORTTERM_NPZ, "short-term local watch (30d/50km)")


def load_deep_eq_operational_model():
    """Operational forecaster: P(M5+ within 100km / 30 days) -- the real WHERE-skill
    (beats climatology +0.14). A third, distinct field."""
    return _load_deep_scorer(DEEP_EQ_OPERATIONAL_NPZ, "operational forecaster (M5+/100km/30d)")


def _predict_eq_with_gbt(
    gbt: dict,
    raw_features_ordered: "np.ndarray",
) -> float:
    """Score a single grid cell using the pre-trained earthquake GBT.

    raw_features_ordered must be a 1-D ndarray in DEFINITIVE_EQ_FEATURE_NAMES
    order (73 values: Block S + Block C).
    """
    means = gbt["normalization"]["means"]
    stds = gbt["normalization"]["stds"]

    # Z-score normalize; treat NaN as missing → 0 post-normalization.
    x = np.asarray(raw_features_ordered, dtype=np.float64)
    x = (x - np.asarray(means)) / np.asarray(stds)
    x = np.where(np.isfinite(x), x, 0.0)

    F = float(gbt["init_pred"])
    lr = float(gbt["learning_rate"])
    for tree in gbt["trees"]:
        node = tree
        while not node.get("leaf", False):
            fi = node["feat"]
            if fi < len(x) and x[fi] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        F += lr * node["val"]

    F = max(-88.0, min(88.0, F))
    if F >= 0:
        return 1.0 / (1.0 + math.exp(-F))
    ef = math.exp(F)
    return ef / (1.0 + ef)


# ---------------------------------------------------------------------------
# Grid cell scoring
# ---------------------------------------------------------------------------


def bin_events_to_grid(events: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Bin earthquake events into 2-degree grid cells.

    Returns dict: (row, col) -> list of events in that cell.
    """
    grid: dict[tuple[int, int], list[dict]] = {}
    for ev in events:
        lat = ev["latitude"]
        lon = ev["longitude"]
        row, col = latlon_to_grid_cell(lat, lon)
        key = (row, col)
        if key not in grid:
            grid[key] = []
        grid[key].append(ev)
    return grid


def score_grid_cells(
    events: list[dict],
    grid_fields: dict[str, np.ndarray] | None = None,
    now: dt.datetime | None = None,
    pretrained_gbt: dict | None = None,
) -> list[dict]:
    """Score all active grid cells and return ranked list.

    For each 2-degree cell with recent seismicity, compute:
    - b-value and trend
    - Correlation length and trend
    - Rate acceleration
    - Singularity conditions (0-5)
    - Estimated days to criticality

    If ``pretrained_gbt`` is provided and the earthquake ML module is
    importable, cell probability comes from the trained GBT (Block S + C,
    73 features). Otherwise falls back to the heuristic scorer.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    ref_epoch = now.replace(tzinfo=dt.timezone.utc).timestamp()

    cell_bins = bin_events_to_grid(events)
    scored_cells: list[dict] = []

    # If ML tier available, pre-build the CatalogArrays once for the run.
    cat_arrays = None
    if pretrained_gbt is not None and HAS_EQ_ML:
        try:
            cat_arrays = CatalogArrays(events, verbose=False)
        except Exception as exc:
            print(f"  WARNING: CatalogArrays build failed ({exc}); disabling ML tier.")
            cat_arrays = None
            pretrained_gbt = None

    for (row, col), cell_events in cell_bins.items():
        if len(cell_events) < 5:
            continue

        lat, lon = grid_cell_to_latlon(row, col)

        # Extract coherence features (used for both tiers — diagnostics always
        # ride along with the output regardless of which tier produced prob).
        features = extract_coherence_features(
            events, lat, lon,
            radius_km=300.0,
            time_window_days=365.0,
            ref_epoch=ref_epoch,
            grid_fields=grid_fields,
        )

        # Test singularity conditions
        sing = test_earthquake_singularity(features)

        prob = None
        cell_tier = "tier2_heuristic"
        if pretrained_gbt is not None and cat_arrays is not None:
            try:
                block_s = compute_block_s(lat, lon, ref_epoch, cat_arrays)
                if block_s is not None:
                    block_c = compute_block_c(events, lat, lon, ref_epoch)
                    full_vec = np.concatenate([block_s, block_c])
                    if full_vec.shape[0] == len(DEFINITIVE_EQ_FEATURE_NAMES):
                        prob = float(_predict_eq_with_gbt(pretrained_gbt, full_vec))
                        cell_tier = "tier1_ml"
            except Exception as exc:
                print(f"  WARNING: ML scoring failed for cell ({row},{col}): {exc}")
                prob = None

        if prob is None:
            # Heuristic fallback (original scorer behaviour).
            base_prob = sing.conditions_met * 0.08
            rate_accel = features.get("rate_acceleration", 1.0)
            if not math.isnan(rate_accel) and rate_accel > 1.0:
                base_prob *= min(rate_accel, 3.0) / 1.5
            b_val = features.get("b_value", 1.0)
            if not math.isnan(b_val) and b_val < 0.85:
                base_prob *= 1.2
            prob = min(max(base_prob, 0.0), 0.95)

        # Max magnitude in cell in last 30 days
        max_mag = max(
            (e["mag"] for e in cell_events if e.get("mag") is not None),
            default=0.0,
        )

        risk = _risk_band(prob)

        entry = {
            "row": row,
            "col": col,
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "n_events": len(cell_events),
            "max_mag": round(max_mag, 1),
            "probability": round(prob, 4),
            "risk_band": risk,
            "scoring_tier": cell_tier,
            "b_value": round(features.get("b_value", float("nan")), 3),
            "b_trend": round(features.get("b_trend", float("nan")), 4),
            "ell_km": round(features.get("ell", float("nan")), 1),
            "ell_trend": round(features.get("ell_trend", float("nan")), 2),
            "rate_acceleration": round(
                features.get("rate_acceleration", float("nan")), 2
            ),
            "delta_aic_iet": round(
                features.get("delta_aic_iet", float("nan")), 2
            ),
            "S_over_Gamma": round(
                features.get("S_over_Gamma", float("nan")), 3
            ),
            "days_to_criticality": round(
                features.get("days_to_criticality", float("nan")), 1
            ),
            "conditions_met": sing.conditions_met,
            "singularity_detail": {
                "ell_elevated": sing.ell_elevated,
                "b_depressed": sing.b_depressed,
                "iet_lorentzian": sing.iet_lorentzian,
                "rate_accelerating": sing.rate_accelerating,
                "loading_exceeds_healing": sing.loading_exceeds_healing,
            },
            "tau_local": round(features.get("tau_local", float("nan")), 4),
            "grad_tau_local": round(
                features.get("grad_tau_local", float("nan")), 4
            ),
            "depth_trend": round(
                features.get("depth_trend", float("nan")), 2
            ),
            "spatial_concentration": round(
                features.get("spatial_concentration", float("nan")), 1
            ),
            "model_version": MODEL_VERSION,
        }
        scored_cells.append(entry)

    # Sort by probability descending, then by conditions_met
    scored_cells.sort(
        key=lambda c: (c["probability"], c["conditions_met"]),
        reverse=True,
    )
    return scored_cells


# ---------------------------------------------------------------------------
# HTML rendering helpers
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
    if math.isnan(p):
        return "--"
    return f"{p * 100:.1f}%"


def _fmt(val: float, fmt: str = ".2f") -> str:
    """Format a float, handling NaN."""
    if isinstance(val, float) and math.isnan(val):
        return "--"
    return f"{val:{fmt}}"


def _format_time(ts: str) -> str:
    """Format a timestamp string for display."""
    if not ts:
        return "--"
    try:
        d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d.strftime("%a, %d %b %Y %H:%M:%S UTC")
    except Exception:
        return str(ts)


def _lat_lon_to_svg(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to SVG coordinates for 960x480 equirectangular map."""
    x = ((lon + 180) / 360) * 960
    y = ((90 - lat) / 180) * 480
    return (x, y)


# ---------------------------------------------------------------------------
# SVG grid heatmap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cell row rendering (details/summary, no JS)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coherence deep dive for #1 cell
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full page rendering (zero JavaScript)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    scored_cells: list[dict],
    now: dt.datetime,
) -> None:
    """Write scored results to dist/data/."""
    # Update live-pulse.json earthquake entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "eq":
                if scored_cells:
                    top = scored_cells[0]
                    hazard["probability"] = top["probability"]
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = top["risk_band"]
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = (
                        f"eq_fcst_{now.strftime('%Y%m%d')}_"
                        f"{now.strftime('%H')}00"
                    )
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "minimal"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(
            json.dumps(pulse, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Updated {pulse_path}")


def append_ledger(
    scored_cells: list[dict],
    now: dt.datetime,
) -> None:
    """Append prediction to SHA-256 chain ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read previous hash
    prev_hash = "0" * 64
    if LEDGER_PATH.exists():
        lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
        if lines and lines[-1].strip():
            try:
                last = json.loads(lines[-1])
                prev_hash = last.get("hash", prev_hash)
            except json.JSONDecodeError:
                pass

    # Build ledger entry
    entry = {
        "timestamp": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "n_cells_scored": len(scored_cells),
        "top_probability": (
            scored_cells[0]["probability"] if scored_cells else 0.0
        ),
        "top_conditions": (
            scored_cells[0]["conditions_met"] if scored_cells else 0
        ),
        "prev_hash": prev_hash,
    }
    # Add top 5 cells summary
    entry["top_cells"] = [
        {
            "lat": c["lat"],
            "lon": c["lon"],
            "probability": c["probability"],
            "conditions_met": c["conditions_met"],
            "max_mag": c["max_mag"],
        }
        for c in scored_cells[:5]
    ]

    # Compute SHA-256 hash
    payload = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Append
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Appended to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

REPLAY_DIR = DIST / "data" / "replay"
REPLAY_INDEX_PATH = DIST / "data" / "evidence" / "replay-index.json"

def required_history_days() -> int:
    """Days of catalog every served tier needs before the issue time.

    The deep GRU tiers read 5 x 365 d (deep_serve.SEQUENCE_LOOKBACK_DAYS, mirroring the
    training builders) and the forest/GBT Block S reads 5 x 365.25 d
    (definitive_model.BLOCK_S_LOOKBACK_DAYS). The coherence features and field read
    365 d. Serving 400 d (the pre-2026-10 value) silently cut short the sequences of
    93% of active cells for the short-term model and 52% for the operational one
    (snapshot eq_fcst_20261001_2200).
    """
    from hazardpulse.earthquake.deep_serve import SEQUENCE_LOOKBACK_DAYS

    needs = [float(SEQUENCE_LOOKBACK_DAYS), 365.0]
    if HAS_EQ_ML:
        from hazardpulse.earthquake.definitive_model import BLOCK_S_LOOKBACK_DAYS

        needs.append(float(BLOCK_S_LOOKBACK_DAYS))
    else:
        needs.append(5.0 * 365.25)
    return int(math.ceil(max(needs)))


FEATURE_HISTORY_DAYS = required_history_days()  # 1827
RECENT_ACTIVITY_DAYS = 30
FORECAST_HORIZON_DAYS = 30
TARGET_MAGNITUDE = 6.0
MIN_CELL_EVENTS = 5


def fetch_usgs_catalog(
    days: int = RECENT_ACTIVITY_DAYS,
    end_time: dt.datetime | None = None,
    *,
    min_magnitude: float = 2.5,
) -> list[dict]:
    """Fetch USGS earthquake catalog for the last N days."""
    from hazardpulse.earthquake.prospective import fetch_usgs_catalog_range

    if end_time is None:
        end_time = dt.datetime.now(dt.timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=dt.timezone.utc)
    end_time = end_time.astimezone(dt.timezone.utc)
    start_time = end_time - dt.timedelta(days=days)

    print(f"  Fetching USGS catalog: M{min_magnitude:.1f}+, {days} days...")
    events = fetch_usgs_catalog_range(
        start_time,
        end_time,
        min_magnitude=min_magnitude,
        namespace="earthquake_live",
        verbose=False,
    )
    print(f"  Fetched {len(events)} events from USGS catalog")
    return events


def score_grid_cells(
    history_events: list[dict],
    candidate_events: list[dict] | None = None,
    grid_fields: dict[str, np.ndarray] | None = None,
    now: dt.datetime | None = None,
    pretrained_gbt: dict | None = None,
    eq_forest=None,
    deep_scorer=None,
    deep_scorer_st=None,
    deep_scorer_op=None,
    operational: dict | None = None,
) -> list[dict]:
    """Score active grid cells using causal history and recent activity.

    ``operational`` (the output of ``eq_operational.forecast_from_artifact`` for this issue
    time) is the served PRIMARY probability: when given, every listed cell publishes its
    full-grid value and no other tier is consulted, and the listed cells are the active
    ones plus the TOP_PROBABILITY_CELLS highest-probability cells of the whole grid. The
    pre-2026-10 precedence below (deep GRU year-ahead nowcast > forest > GBT > singularity
    heuristic) only applies without it -- those are case-control nowcasts of other
    quantities; scored on this page's contract the deep tier had no operational skill
    (docs/EARTHQUAKE_FORECAST_PROGRAM.md). If ``deep_scorer_st`` is given, each cell ALSO
    gets a second, independent field ``prob_30d_local`` = P(M4.5+ within 50km / 30 days)
    from the short-term local model -- a distinct product, not a fallback.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    ref_epoch = now.replace(tzinfo=dt.timezone.utc).timestamp()

    if candidate_events is None:
        candidate_events = history_events

    # Build CatalogArrays once per run if ANY ML model is active (deep/forest/GBT).
    cat_arrays = None
    if (pretrained_gbt is not None or eq_forest is not None or deep_scorer is not None
            or deep_scorer_st is not None or deep_scorer_op is not None) and HAS_EQ_ML:
        try:
            cat_arrays = CatalogArrays(history_events, verbose=False)
            tiers = [n for n, on in (("deep", deep_scorer is not None),
                                     ("op", deep_scorer_op is not None),
                                     ("forest", eq_forest is not None),
                                     ("gbt", pretrained_gbt is not None)) if on]
            print(f"  ML tier active ({'+'.join(tiers)}): CatalogArrays built from {len(history_events)} events")
        except Exception as exc:
            print(f"  WARNING: CatalogArrays build failed ({exc}); disabling ML tier.")
            cat_arrays = None
            pretrained_gbt = None
            eq_forest = None
            deep_scorer = None
            deep_scorer_st = None
            deep_scorer_op = None

    cell_bins = bin_events_to_grid(candidate_events)
    listed = {key: evs for key, evs in cell_bins.items() if len(evs) >= MIN_CELL_EVENTS}
    if operational is not None:
        grid_p = np.asarray(operational["probability"], dtype=np.float64)
        for flat in np.argsort(-grid_p, kind="stable")[:TOP_PROBABILITY_CELLS]:
            key = (int(flat) // N_LON, int(flat) % N_LON)
            listed.setdefault(key, cell_bins.get(key, []))
    scored_cells: list[dict] = []
    n_ml = n_heur = 0

    for (row, col), cell_events in listed.items():
        lat, lon = grid_cell_to_latlon(row, col)
        features = extract_coherence_features(
            history_events,
            lat,
            lon,
            radius_km=300.0,
            time_window_days=365.0,
            ref_epoch=ref_epoch,
            grid_fields=grid_fields,
        )
        sing = test_earthquake_singularity(features)

        prob = None
        cell_tier = "tier2_heuristic"
        model_id = None
        op_fields: dict = {}
        if operational is not None:
            flat = row * N_LON + col
            prob = float(operational["probability"][flat])
            model_id = MODEL_VERSION
            cell_tier = "tier1_operational"
            op_fields = {
                "lambda_long": _publish_prob(float(operational["lambda_long"][flat])),
                "lambda_short": _publish_prob(float(operational["lambda_short"][flat])),
            }
            n_ml += 1
        # Tier 1a: deep GRU nowcast (champion). Reads the raw event sequence directly --
        # no Block S/C needed. P(M5+ within radius/365d) precursory-state score.
        if prob is None and deep_scorer is not None and cat_arrays is not None:
            try:
                dp = deep_scorer.score(cat_arrays, lat, lon, ref_epoch)
                if dp is not None:
                    prob = float(dp)
                    model_id = "deep_gru_k192"
                    cell_tier = "tier1_deep"
                    n_ml += 1
            except Exception as exc:
                print(f"  WARNING: deep scoring failed for cell ({row},{col}): {exc}")
                prob = None
        # Tier 1b: forest / GBT on the 73 Block S + C features (fallback if deep absent/empty).
        if prob is None and (pretrained_gbt is not None or eq_forest is not None) and cat_arrays is not None:
            try:
                block_s = compute_block_s(lat, lon, ref_epoch, cat_arrays)
                if block_s is not None:
                    block_c = compute_block_c(history_events, lat, lon, ref_epoch)
                    full_vec = np.concatenate([block_s, block_c])
                    if full_vec.shape[0] == len(DEFINITIVE_EQ_FEATURE_NAMES):
                        if eq_forest is not None:
                            # Champion forest serves RAW features (no z-scoring).
                            prob = float(eq_forest.raw_proba_one(full_vec))
                            model_id = "verifiable_forest_fp"
                        else:
                            prob = float(_predict_eq_with_gbt(pretrained_gbt, full_vec))
                            model_id = "gbt_v1"
                        cell_tier = "tier1_ml"
                        n_ml += 1
            except Exception as exc:
                print(f"  WARNING: ML scoring failed for cell ({row},{col}): {exc}")
                prob = None

        if prob is None:
            base_prob = sing.conditions_met * 0.08
            rate_accel = features.get("rate_acceleration", 1.0)
            if not math.isnan(rate_accel) and rate_accel > 1.0:
                base_prob *= min(rate_accel, 3.0) / 1.5
            b_val = features.get("b_value", 1.0)
            if not math.isnan(b_val) and b_val < 0.85:
                base_prob *= 1.2
            prob = min(max(base_prob, 0.0), 0.95)
            n_heur += 1

        # Second, independent product: short-term local watch P(M4.5+ within 50km / 30d).
        prob_30d_local = None
        if deep_scorer_st is not None and cat_arrays is not None:
            try:
                stp = deep_scorer_st.score(cat_arrays, lat, lon, ref_epoch)
                if stp is not None:
                    prob_30d_local = round(float(stp), 4)
            except Exception as exc:
                print(f"  WARNING: short-term scoring failed for cell ({row},{col}): {exc}")

        # Third product: operational forecaster P(M5+ within 100km / 30d) -- the WHERE-skill.
        prob_op_m5_30d = None
        if deep_scorer_op is not None and cat_arrays is not None:
            try:
                opp = deep_scorer_op.score(cat_arrays, lat, lon, ref_epoch)
                if opp is not None:
                    prob_op_m5_30d = round(float(opp), 4)
            except Exception as exc:
                print(f"  WARNING: operational scoring failed for cell ({row},{col}): {exc}")

        max_mag = max(
            (event["mag"] for event in cell_events if event.get("mag") is not None),
            default=0.0,
        )
        risk = _risk_band(prob)
        scored_cells.append(
            {
                "row": row,
                "col": col,
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "n_events": len(cell_events),
                "max_mag": round(max_mag, 1),
                "probability": _publish_prob(prob),
                **op_fields,
                "prob_30d_local": prob_30d_local,
                "prob_op_m5_30d": prob_op_m5_30d,
                "risk_band": risk,
                "scoring_tier": cell_tier,
                "model_id": model_id,
                "b_value": round(features.get("b_value", float("nan")), 3),
                "b_trend": round(features.get("b_trend", float("nan")), 4),
                "ell_km": round(features.get("ell", float("nan")), 1),
                "ell_trend": round(features.get("ell_trend", float("nan")), 2),
                "rate_acceleration": round(
                    features.get("rate_acceleration", float("nan")),
                    2,
                ),
                "delta_aic_iet": round(
                    features.get("delta_aic_iet", float("nan")),
                    2,
                ),
                "S_over_Gamma": round(features.get("S_over_Gamma", float("nan")), 3),
                "days_to_criticality": round(
                    features.get("days_to_criticality", float("nan")),
                    1,
                ),
                "conditions_met": sing.conditions_met,
                "singularity_detail": {
                    "ell_elevated": sing.ell_elevated,
                    "b_depressed": sing.b_depressed,
                    "iet_lorentzian": sing.iet_lorentzian,
                    "rate_accelerating": sing.rate_accelerating,
                    "loading_exceeds_healing": sing.loading_exceeds_healing,
                },
                "tau_local": round(features.get("tau_local", float("nan")), 4),
                "grad_tau_local": round(
                    features.get("grad_tau_local", float("nan")),
                    4,
                ),
                "depth_trend": round(features.get("depth_trend", float("nan")), 2),
                "spatial_concentration": round(
                    features.get("spatial_concentration", float("nan")),
                    1,
                ),
                "model_version": MODEL_VERSION,
            }
        )

    scored_cells.sort(
        key=lambda cell: (cell["probability"], cell["conditions_met"]),
        reverse=True,
    )

    if n_ml or n_heur:
        print(f"  Scored cells by tier: tier1={n_ml}, tier2_heuristic={n_heur}")
    return scored_cells


def _publish_prob(p: float) -> float:
    """Six significant digits: the operational model's quiet cells sit at 1e-5..1e-3, which
    the old ``round(p, 4)`` would have published as 0."""
    return float(f"{float(p):.6g}")


def _make_json_serializable(obj):
    """Convert NumPy scalars and NaNs into plain JSON-safe values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(value) for value in obj]
    return obj


def write_outputs(
    scored_cells: list[dict],
    now: dt.datetime,
    *,
    forecast_id: str,
    pulse_path: Path = DIST / "data" / "live-pulse.json",
) -> None:
    """Write scored results to dist/data/."""
    from hazardpulse.earthquake.prospective import format_utc_z

    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "eq":
                if scored_cells:
                    top = scored_cells[0]
                    hazard["probability"] = top["probability"]
                    # Real uncertainty band from the calibrator (None until a
                    # calibrator exists) — populates HazardForecastV1 and removes
                    # the universal confidence_interval_unavailable gate warning.
                    hazard["conf_lo"] = top.get("confidence_lo")
                    hazard["conf_hi"] = top.get("confidence_hi")
                    hazard["uncertainty_class"] = top.get("uncertainty_class")
                    hazard["abstained"] = top.get("abstained", False)
                    hazard["receipt_sha256"] = top.get("receipt_sha256")
                    hazard["risk_band"] = top["risk_band"]
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = forecast_id
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "minimal"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = forecast_id
                break
        pulse["updated_at"] = format_utc_z(now)
        pulse_path.write_text(json.dumps(pulse, indent=2) + "\n", encoding="utf-8")
        print(f"  Updated {pulse_path}")


def write_replay_artifact(
    scored_cells: list[dict],
    now: dt.datetime,
    *,
    forecast_id: str,
    n_history_events: int,
    n_recent_events: int,
    replay_dir: Path = REPLAY_DIR,
    update_index: bool = True,
    probability_grid=None,
    operational_meta: dict | None = None,
) -> Path:
    """Write a frozen replay artifact for later prospective scoring."""
    from hazardpulse.earthquake.prospective import format_utc_z

    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / f"{forecast_id}.json"
    artifact = {
        "forecast_id": forecast_id,
        "hazard": "earthquake",
        "issued_at": format_utc_z(now),
        "model_version": MODEL_VERSION,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "target_magnitude_min": TARGET_MAGNITUDE,
        "feature_history_days": FEATURE_HISTORY_DAYS,
        "recent_activity_days": RECENT_ACTIVITY_DAYS,
        "min_events_per_active_cell": MIN_CELL_EVENTS,
        "forecast_domain": {
            "name": "global_2deg_grid",
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "dlat": GRID_DLAT,
            "dlon": GRID_DLON,
            "n_lat": N_LAT,
            "n_lon": N_LON,
            "default_probability": 0.0,
        },
        "source_catalog": {
            "provider": "USGS FDSNWS",
            "min_magnitude": 2.5,
            "window_start": format_utc_z(now - dt.timedelta(days=FEATURE_HISTORY_DAYS)),
            "window_end": format_utc_z(now),
            "n_events": n_history_events,
            "n_recent_events": n_recent_events,
        },
        "n_active_cells": len(scored_cells),
        "top_probability": scored_cells[0]["probability"] if scored_cells else 0.0,
        "active_cells": scored_cells,
    }
    if probability_grid is not None:
        grid = np.asarray(probability_grid, dtype=np.float64).ravel()
        if grid.size != N_LAT * N_LON:
            raise ValueError(f"probability_grid has {grid.size} cells, the domain has {N_LAT * N_LON}")
        # The forecast for EVERY cell (row-major, row = latitude band from lat_min), as ONE
        # comma-separated string of 6-significant-digit values: a JSON list would be written
        # one value per line by indent=2 (~210 KB per forecast). The verifier scores this
        # grid; default_probability applies only to artifacts without it.
        artifact["probability_grid"] = ",".join(f"{float(p):.6g}" for p in grid)
        artifact["forecast_domain"]["grid_order"] = "row_major_lat_then_lon"
    if operational_meta:
        artifact["operational_model"] = operational_meta
    replay_path.write_text(
        json.dumps(_make_json_serializable(artifact), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  Wrote {replay_path}")

    if update_index:
        update_replay_index(forecast_id, replay_path)
    return replay_path


def update_replay_index(
    forecast_id: str,
    replay_path: Path,
    *,
    replay_index_path: Path = REPLAY_INDEX_PATH,
) -> None:
    """Upsert the earthquake replay artifact into the shared replay index."""
    from hazardpulse.earthquake.prospective import format_utc_z

    replay_index_path.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    if replay_index_path.exists():
        try:
            payload = json.loads(replay_index_path.read_text(encoding="utf-8"))
            items = list(payload.get("items", []))
        except Exception:
            items = []

    try:
        artifact_ref = "/" + replay_path.relative_to(DIST).as_posix()
    except ValueError:
        artifact_ref = str(replay_path)

    items = [item for item in items if item.get("forecast_id") != forecast_id]
    items.append({"forecast_id": forecast_id, "replay_artifact": artifact_ref})
    items.sort(key=lambda item: item.get("forecast_id", ""))
    replay_index_path.write_text(
        json.dumps(
            {
                "generated_at": format_utc_z(dt.datetime.now(dt.timezone.utc)),
                "items": items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  Updated {replay_index_path}")


def append_ledger(
    scored_cells: list[dict],
    now: dt.datetime,
    *,
    forecast_id: str,
    ledger_path: Path = LEDGER_PATH,
    replay_path: Path | None = None,
) -> None:
    """Append prediction to the ledger without duplicate forecast ids."""
    from hazardpulse.earthquake.prospective import format_utc_z

    # FAIL CLOSED on the tracked production ledger: it is a 340+-entry hash chain committed
    # in git. If it is missing from disk the checkout is partial (e.g. sparse); appending
    # would silently fork a fresh chain from genesis and clobber the real one on the next
    # commit (71d56d53 shrank it 346 -> 2 entries this way). Explicit --ledger-path targets
    # (tests, replays) may still start fresh ledgers.
    if ledger_path == LEDGER_PATH and not ledger_path.exists():
        raise SystemExit(
            f"{ledger_path} is missing. The tracked prediction ledger must exist before "
            "appending -- run from a full checkout (git sparse-checkout disable). Refusing "
            "to fork a fresh chain from genesis."
        )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    prev_hash = "0" * 64
    if ledger_path.exists():
        existing_lines = [
            line
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in existing_lines:
            try:
                prior = json.loads(line)
            except json.JSONDecodeError:
                continue
            if prior.get("forecast_id") == forecast_id:
                print(
                    f"  Ledger already contains {forecast_id}; "
                    f"skipping duplicate append for {ledger_path}"
                )
                return
        if existing_lines:
            try:
                prev_hash = json.loads(existing_lines[-1]).get("hash", prev_hash)
            except json.JSONDecodeError:
                pass

    entry = {
        "forecast_id": forecast_id,
        "timestamp": format_utc_z(now),
        "model_version": MODEL_VERSION,
        "n_cells_scored": len(scored_cells),
        "top_probability": scored_cells[0]["probability"] if scored_cells else 0.0,
        "top_conditions": scored_cells[0]["conditions_met"] if scored_cells else 0,
        "top_receipt_sha256": (
            scored_cells[0].get("receipt_sha256") if scored_cells else None
        ),
        "prev_hash": prev_hash,
    }
    if replay_path is not None:
        try:
            entry["replay_artifact"] = "/" + replay_path.relative_to(DIST).as_posix()
        except ValueError:
            entry["replay_artifact"] = str(replay_path)
    entry["top_cells"] = [
        {
            "lat": cell["lat"],
            "lon": cell["lon"],
            "probability": cell["probability"],
            "conditions_met": cell["conditions_met"],
            "max_mag": cell["max_mag"],
        }
        for cell in scored_cells[:5]
    ]
    payload = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"  Appended to {ledger_path}")


class RiskBandContradiction(RuntimeError):
    """A published cell's risk band disagrees with its published probability."""


def attach_operational_receipts(
    scored: list[dict],
    *,
    model_version: str,
    model_sha256: str,
    input_sha256: str,
    issued_at: str,
    signer=None,
) -> list[dict]:
    """Receipt (spec hazardpulse/forecast/v1) for each listed operational cell.

    Binds the published probability to the artifact (``model_sha256``, CRLF-normalised)
    and to the exact model input (the canonical M5+ rows before the issue time) plus the
    cell. No calibrator is involved, so ``raw_probability == probability`` and the
    interval is absent (``uncertainty_class = "no_interval"``) -- nothing is claimed that
    was not computed. Signed when a signing key is configured, integrity-only otherwise.
    """
    from hazardpulse.trust.forecast import RECEIPT_SPEC, sign_forecast_receipt

    for cell in scored:
        cell_input = hashlib.sha256(
            f"{input_sha256}|{issued_at}|{cell['row']},{cell['col']}".encode("utf-8")).hexdigest()
        core = {
            "spec": RECEIPT_SPEC,
            "model_version": model_version,
            "model_sha256": model_sha256,
            "input_sha256": cell_input,
            "issued_at": issued_at,
            "raw_probability": cell["probability"],
            "probability": cell["probability"],
            "confidence_lo": None,
            "confidence_hi": None,
            "uncertainty_class": "no_interval",
            "ood_score": None,
            "ood_flag": False,
            "abstained": False,
            "abstain_reason": None,
            "gateway_mode": "NORMAL",
            "coverage_target": None,
        }
        receipt = sign_forecast_receipt(core, signer)
        cell["receipt"] = receipt
        cell["receipt_sha256"] = receipt["receipt_sha256"]
    return scored


def trust_layer_applies(forecaster, served_model_version: str) -> bool:
    """A calibrator maps ONE model's raw scores; it is bound to that model's version.

    The earthquake calibrator on disk was fitted (2026-10-01) to the deep GRU nowcast's
    raw outputs (model_version ``eq_coherence_v1_0``); feeding it the operational model's
    probabilities would replace calibrated numbers with a mapping learned for different
    inputs.
    """
    return forecaster is not None and getattr(forecaster, "model_version", None) == served_model_version


def apply_trust_layer(scored: list[dict], forecaster, *, issued_at: str) -> list[dict]:
    """Calibrate every scored cell in place and keep its risk band truthful.

    The band is first computed from the RAW score in score_grid_cells; the trust
    layer then replaces ``probability`` with the calibrated value. Re-deriving the
    band from the published probability (band_fn) is what stops a 4.54% cell from
    being labelled "critical". Raises if any contradiction survives.
    """
    from hazardpulse.trust.scoring import band_contradictions, enrich_cells

    enrich_cells(scored, forecaster, issued_at=issued_at, band_fn=_risk_band)
    bad = band_contradictions(scored, _risk_band)
    if bad:
        raise RiskBandContradiction(
            f"{len(bad)} earthquake cells carry a risk band that contradicts their "
            f"published probability (first: {scored[bad[0]].get('probability')} -> "
            f"{scored[bad[0]].get('risk_band')})"
        )
    return scored


def build_arg_parser():
    """Build the CLI argument parser for the replay-aware forecast runner."""
    parser = argparse.ArgumentParser(
        description="Generate a frozen earthquake forecast artifact.",
    )
    parser.add_argument("--issue-time", default=None, help="UTC issue time (ISO-8601).")
    parser.add_argument("--replay-dir", type=Path, default=REPLAY_DIR)
    parser.add_argument("--ledger-path", type=Path, default=LEDGER_PATH)
    parser.add_argument("--skip-site", action="store_true")
    parser.add_argument("--skip-live-pulse", action="store_true")
    parser.add_argument("--skip-replay-index", action="store_true")
    return parser


def run_pipeline(
    *,
    issue_time: dt.datetime | None = None,
    replay_dir: Path = REPLAY_DIR,
    ledger_path: Path = LEDGER_PATH,
    skip_site: bool = False,
    skip_live_pulse: bool = False,
    skip_replay_index: bool = False,
) -> dict:
    """Run the replay-aware earthquake scoring pipeline."""
    from hazardpulse.earthquake.prospective import (
        forecast_id_for_time,
        format_utc_z,
        parse_utc_datetime,
    )

    now = (
        issue_time.astimezone(dt.timezone.utc)
        if issue_time is not None
        else dt.datetime.now(dt.timezone.utc)
    ).replace(minute=0, second=0, microsecond=0)
    forecast_id = forecast_id_for_time(now)

    print(f"HazardPulse Earthquake Scoring Pipeline -- {format_utc_z(now)}")
    print(f"Forecast ID: {forecast_id}")
    print()
    print(
        f"Step 1: Fetching USGS earthquake catalog "
        f"(M2.5+, {FEATURE_HISTORY_DAYS} days history)..."
    )
    history_events = fetch_usgs_catalog(days=FEATURE_HISTORY_DAYS, end_time=now)
    recent_cutoff = now - dt.timedelta(days=RECENT_ACTIVITY_DAYS)
    recent_events = [
        event
        for event in history_events
        if parse_utc_datetime(event["time"]) >= recent_cutoff
    ]

    if not history_events:
        replay_path = write_replay_artifact(
            [],
            now,
            forecast_id=forecast_id,
            n_history_events=0,
            n_recent_events=0,
            replay_dir=replay_dir,
            update_index=not skip_replay_index,
        )
        if not skip_live_pulse:
            write_outputs([], now, forecast_id=forecast_id)
        append_ledger(
            [],
            now,
            forecast_id=forecast_id,
            ledger_path=ledger_path,
            replay_path=replay_path,
        )
        if not skip_site and not skip_live_pulse:
            build_site_artifacts()
        return {
            "forecast_id": forecast_id,
            "issued_at": format_utc_z(now),
            "n_history_events": 0,
            "n_recent_events": 0,
            "n_active_cells": 0,
            "replay_path": str(replay_path),
        }

    print(f"  {len(history_events)} history events fetched")
    print(f"  {len(recent_events)} recent events kept for active-cell discovery")
    print()
    print("Step 2: Computing seismic coherence field (Helmholtz PDE)...")
    grid_fields: dict[str, np.ndarray] | None = None
    try:
        grid_fields = compute_seismic_coherence_field(
            history_events,
            time_window_days=365.0,
        )
        print(f"  tau_max = {float(grid_fields['tau'].max()):.4f}")
    except Exception as exc:
        print(f"  Warning: Coherence field computation failed: {exc}")
        print("  Proceeding with point-based features only")

    print()
    print("Step 3: Operational forecast for every cell, diagnostics for the listed cells...")
    # FAIL CLOSED: the served probability is the artifact's model; without it nothing is
    # published (the old tiers are case-control nowcasts of other quantities).
    artifact = eq_operational.load_artifact(OPERATIONAL_ARTIFACT_PATH)
    stack = eq_operational.load_stack(STACK_PATH, artifact) if STACK_PATH.exists() else None
    served_version = stack.model_version if stack is not None else artifact.model_version
    if served_version != MODEL_VERSION:
        raise RuntimeError(f"the served earthquake model changed while running: {served_version} != {MODEL_VERSION}")
    if stack is not None:
        operational = eq_operational.forecast_with_stack(artifact, stack, history_events, now)
    else:
        operational = eq_operational.forecast_from_artifact(artifact, history_events, now)
    print(
        f"  {served_version} (base {artifact.model_version}): {operational['n_input_events']} M5+ inputs since "
        f"{eq_operational.CATALOG_START[:10]}; expected cells with an M6+ (sum of P) "
        f"{float(np.sum(operational['probability'])):.2f}; max P {float(np.max(operational['probability'])):.4f}"
    )
    deep_eq_scorer_st = load_deep_eq_shortterm_model()  # short-term local watch (2nd field)
    deep_eq_scorer_op = load_deep_eq_operational_model()  # M5+/100 km research field (3rd)
    scored = score_grid_cells(
        history_events,
        candidate_events=recent_events,
        grid_fields=grid_fields,
        now=now,
        deep_scorer_st=deep_eq_scorer_st,
        deep_scorer_op=deep_eq_scorer_op,
        operational=operational,
    )
    print(f"  {len(scored)} cells listed")

    # Trust layer: calibrate probabilities, attach honest [conf_lo, conf_hi]
    # bands + Ed25519-signed re-runnable receipts. Fails safe — if no calibrator
    # has been produced yet, forecasts stay raw (uncalibrated) and honest. A calibrator
    # fitted to ANOTHER model's scores is never applied (trust_layer_applies).
    try:
        from hazardpulse.trust.scoring import load_forecaster, load_signer

        _signer = load_signer()
        _forecaster = load_forecaster("earthquake", signer=_signer)
        if not trust_layer_applies(_forecaster, MODEL_VERSION):
            bound = _forecaster.model_version if _forecaster is not None else "none"
            attach_operational_receipts(
                scored, model_version=MODEL_VERSION, model_sha256=artifact.sha256,
                input_sha256=operational["input_sha256"], issued_at=format_utc_z(now), signer=_signer)
            print(
                f"  Trust layer: calibrator bound to {bound}, not the served {MODEL_VERSION}; "
                "not applied (the operational model is calibrated by its own likelihood fit). "
                f"Receipts attached to {len(scored)} cells (signed={_signer is not None})."
            )
        else:
            apply_trust_layer(scored, _forecaster, issued_at=format_utc_z(now))
            print(
                f"  Trust layer: calibrated {len(scored)} cells "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
    except RiskBandContradiction:
        raise  # never publish a label that contradicts its own probability
    except Exception as exc:  # never let the trust layer break a live forecast
        print(f"  Trust layer: skipped ({exc})")

    for cell in scored[:10]:
        print(
            f"  [{cell['lat']:.0f}N, {cell['lon']:.0f}E] "
            f"P={cell['probability']:.1%} conditions={cell['conditions_met']}/5 "
            f"b={_fmt(cell['b_value'], '.3f')} "
            f"ell={_fmt(cell['ell_km'], '.0f')}km "
            f"events={cell['n_events']} Mmax={cell['max_mag']:.1f}"
        )

    print()
    print("Step 4: Writing outputs...")
    replay_path = write_replay_artifact(
        scored,
        now,
        forecast_id=forecast_id,
        n_history_events=len(history_events),
        n_recent_events=len(recent_events),
        replay_dir=replay_dir,
        update_index=not skip_replay_index,
        probability_grid=operational["probability"],
        operational_meta={
            "model_version": served_version,
            "artifact": "results/models/" + (STACK_PATH.name if stack is not None else OPERATIONAL_ARTIFACT_PATH.name),
            "base_artifact": "results/models/" + OPERATIONAL_ARTIFACT_PATH.name,
            "base_model_version": artifact.model_version,
            "artifact_sha256_lf": artifact.sha256,
            "n_input_events_m5": operational["n_input_events"],
            # E[number of cells with an M6+ in the window] = sum of the published probabilities
            "expected_positive_cells": round(float(np.sum(operational["probability"])), 4),
            "protocol": "docs/EARTHQUAKE_FORECAST_PROGRAM.md",
        },
    )
    if not skip_live_pulse:
        write_outputs(scored, now, forecast_id=forecast_id)
    append_ledger(
        scored,
        now,
        forecast_id=forecast_id,
        ledger_path=ledger_path,
        replay_path=replay_path,
    )

    if not skip_site and not skip_live_pulse:
        print()
        print("Step 5: Rendering the site from the published artifacts...")
        build_site_artifacts()

    # ---- Alert manager evaluation (after live-pulse.json is fresh) ----
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        try:
            from hazardpulse.alerts import build_default_manager
            audit_path = DIST.parent / "results" / "alerts" / "audit.ndjson"
            recent_path = DIST / "data" / "alerts-recent.json"
            mgr = build_default_manager(
                audit_path=audit_path,
                recent_path=recent_path,
            )
            pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
            fired = mgr.evaluate(pulse)
            for a in fired:
                if a.severity != "suppressed":
                    print(f"  ALERT [{a.severity}] {a.rule_name}: {a.message}")
        except Exception as exc:
            print(f"  Warning: alert evaluation skipped: {exc}")

    print()
    print(
        f"Done. Scored {len(scored)} cells from {len(recent_events)} recent "
        f"events and {len(history_events)} history events."
    )
    return {
        "forecast_id": forecast_id,
        "issued_at": format_utc_z(now),
        "n_history_events": len(history_events),
        "n_recent_events": len(recent_events),
        "n_active_cells": len(scored),
        "top_probability": scored[0]["probability"] if scored else 0.0,
        "replay_path": str(replay_path),
    }


def main() -> None:
    """Run the replay-aware earthquake scoring pipeline."""
    from hazardpulse.earthquake.prospective import parse_utc_datetime

    parser = build_arg_parser()
    args = parser.parse_args()
    issue_time = parse_utc_datetime(args.issue_time) if args.issue_time else None
    run_pipeline(
        issue_time=issue_time,
        replay_dir=args.replay_dir,
        ledger_path=args.ledger_path,
        skip_site=args.skip_site,
        skip_live_pulse=args.skip_live_pulse,
        skip_replay_index=args.skip_replay_index,
    )


if __name__ == "__main__":
    main()
