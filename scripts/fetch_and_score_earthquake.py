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

MODEL_VERSION = "eq_coherence_v1_0"
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


def _band_text(lo, hi) -> str:
    """Calibrated uncertainty band, e.g. ' (90% band: 8.0%-18.0%)'.

    Empty string when no interval is available (raw, uncalibrated forecast) so
    the label gracefully reads as before until a calibrator exists.
    """
    if lo is None or hi is None:
        return ""
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return ""
    if math.isnan(lo_f) or math.isnan(hi_f):
        return ""
    return f" (90% band: {lo_f * 100:.1f}%-{hi_f * 100:.1f}%)"


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


def _compass_value(value: float, positive: str, negative: str) -> str:
    """Format a signed coordinate value with a compass suffix."""
    return f"{abs(value):.0f}{positive if value >= 0 else negative}"


def _format_cell_coords(lat: float, lon: float) -> str:
    """Format cell coordinates using N/S/E/W suffixes."""
    return f"{_compass_value(lat, 'N', 'S')}, {_compass_value(lon, 'E', 'W')}"


def _earthquake_summary_sentence(cell: dict) -> str:
    """Explain why a cell is ranking highly in plain language."""
    reasons: list[str] = []
    conditions = int(cell.get("conditions_met", 0) or 0)
    rate_acceleration = float(cell.get("rate_acceleration", 0) or 0)
    b_value = float(cell.get("b_value", 0) or 0)
    s_over_gamma = float(cell.get("S_over_Gamma", 0) or 0)

    if conditions >= 4:
        reasons.append(f"{conditions}/5 singularity conditions are active")
    elif conditions >= 3:
        reasons.append(f"{conditions}/5 precursor conditions are active")
    if rate_acceleration >= 2:
        reasons.append(f"event rate is {rate_acceleration:.1f}x the local background")
    if 0 < b_value < 0.8:
        reasons.append(f"the b-value is compressed at {b_value:.2f}")
    if s_over_gamma > 1:
        reasons.append(f"loading exceeds healing by {s_over_gamma:.1f}x")

    if not reasons:
        return (
            "This cell remains on the watchlist because multiple coherence and "
            "clustering signals are above background."
        )
    if len(reasons) == 1:
        return f"This cell stands out because {reasons[0]}."
    if len(reasons) == 2:
        return f"This cell stands out because {reasons[0]} and {reasons[1]}."
    return (
        f"This cell stands out because {reasons[0]}, {reasons[1]}, and "
        f"{reasons[2]}."
    )


def _earthquake_action_recommendation(cell: dict) -> str:
    """Operational guidance for the current seismic posture."""
    band = cell.get("risk_band", "minimal")
    if band in {"critical", "very_high"}:
        return (
            "Treat this as a heightened watch signal: review continuity plans, "
            "confirm response contacts, and monitor USGS or the relevant regional "
            "seismological agency for escalation."
        )
    if band in {"elevated", "guarded"}:
        return (
            "Keep this zone on watch, especially if you have facilities, shipping, "
            "or field operations nearby. Monitor official seismic catalogs for "
            "rapid changes in rate or magnitude."
        )
    return (
        "This is a monitoring signal rather than an emergency trigger. Keep "
        "official seismic feeds in view and reassess if nearby activity clusters."
    )


def _earthquake_watch_items(cell: dict) -> list[str]:
    """Short watch items for the simple detail view."""
    items: list[str] = []
    rate_acceleration = float(cell.get("rate_acceleration", 0) or 0)
    b_value = float(cell.get("b_value", 0) or 0)
    s_over_gamma = float(cell.get("S_over_Gamma", 0) or 0)
    ell_trend = float(cell.get("ell_trend", 0) or 0)
    max_mag = float(cell.get("max_mag", 0) or 0)
    conditions = int(cell.get("conditions_met", 0) or 0)

    if conditions:
        items.append(f"{conditions}/5 criticality conditions")
    if rate_acceleration >= 1.2:
        items.append(f"Rate acceleration {rate_acceleration:.1f}x")
    if 0 < b_value < 1:
        items.append(f"b-value {b_value:.2f}")
    if s_over_gamma > 0:
        items.append(f"S/Gamma {s_over_gamma:.1f}")
    if abs(ell_trend) >= 1:
        items.append(f"Corr. length trend {ell_trend:+.1f} km/mo")
    if max_mag >= 4:
        items.append(f"Mmax {max_mag:.1f} in-window")

    return items[:5]


def _lat_lon_to_svg(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to SVG coordinates for 960x480 equirectangular map."""
    x = ((lon + 180) / 360) * 960
    y = ((90 - lat) / 180) * 480
    return (x, y)


# ---------------------------------------------------------------------------
# SVG grid heatmap
# ---------------------------------------------------------------------------


def _render_grid_heatmap(cells: list[dict]) -> str:
    """Render 2-degree grid cells as colored rectangles on the SVG map."""
    lines: list[str] = []
    for c in cells:
        lat = c["lat"]
        lon = c["lon"]
        prob = c["probability"]
        risk = c["risk_band"]
        color = RISK_COLORS.get(risk, "#757575")

        # Cell corners
        x1, y1 = _lat_lon_to_svg(lat + GRID_DLAT / 2, lon - GRID_DLON / 2)
        x2, y2 = _lat_lon_to_svg(lat - GRID_DLAT / 2, lon + GRID_DLON / 2)
        w = x2 - x1
        h = y2 - y1

        opacity = min(0.15 + prob * 1.2, 0.8)
        label = (
            f"{_format_cell_coords(lat, lon)} - {_pct(prob)} "
            f"({c['conditions_met']}/5 conditions)"
        )
        lines.append(
            f'    <a href="#cell-{c["row"]}-{c["col"]}" '
            f'aria-label="{_esc(label)}">'
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="{opacity:.2f}" '
            f'stroke="{color}" stroke-width="0.5"/></a>'
        )
    return "\n".join(lines)


def _render_svg_markers(cells: list[dict]) -> str:
    """Generate SVG marker elements for top risk cells on the map."""
    lines: list[str] = []
    limit = min(len(cells), 10)
    for i in range(limit):
        c = cells[i]
        rank = i + 1
        x, y = _lat_lon_to_svg(c["lat"], c["lon"])
        risk_label = RISK_LABELS.get(c["risk_band"], c["risk_band"])
        cell_label = _format_cell_coords(c["lat"], c["lon"])
        label = (
            f"Cell {cell_label} - "
            f'{_pct(c["probability"])} ({risk_label})'
        )
        base_r = 5.0 if rank == 1 else 3.5

        lines.append(
            f'    <a href="#cell-{rank}" aria-label="{_esc(label)}">'
        )
        for p in (1, 2):
            lines.append(
                f'      <circle class="hz-pulse hz-pulse-eq hz-pulse-delay-{p}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1:.0f}"/>'
            )
        lines.append(
            f'      <circle class="hz-marker hz-marker-eq" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r:.1f}" '
            f'filter="url(#glow-eq)"/>'
        )
        # Tooltip
        lines.append(f'      <g class="map-tooltip">')
        lines.append(
            f'        <rect class="tooltip-bg" x="{x + 10:.1f}" '
            f'y="{y - 18:.1f}" width="120" height="22" rx="3"/>'
        )
        lines.append(
            f'        <text class="tooltip-text" x="{x + 12:.1f}" '
            f'y="{y - 8:.1f}">'
            f"{_esc(cell_label)}</text>"
        )
        lines.append(
            f'        <text class="tooltip-sub" x="{x + 12:.1f}" '
            f'y="{y + 1:.1f}">'
            f'{_pct(c["probability"])} | {c["conditions_met"]}/5</text>'
        )
        lines.append(f'      </g>')
        lines.append(f'    </a>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cell row rendering (details/summary, no JS)
# ---------------------------------------------------------------------------


def _render_cell_rows(cells: list[dict]) -> str:
    """Render top 10 grid cells as <details>/<summary> elements. Zero JS."""
    lines: list[str] = []
    limit = min(len(cells), 10)
    for i in range(limit):
        c = cells[i]
        rank = i + 1
        risk_color = RISK_COLORS.get(c["risk_band"], "#757575")
        risk_label = RISK_LABELS.get(c["risk_band"], c["risk_band"])
        rank_class = " rank-1" if rank == 1 else ""
        probability = float(c.get("probability", 0) or 0)
        cell_label = _format_cell_coords(c["lat"], c["lon"])
        summary_sentence = _earthquake_summary_sentence(c)
        action_text = _earthquake_action_recommendation(c)
        watch_items = _earthquake_watch_items(c)
        simple_subline = (
            f"{_pct(probability)} 30-day seismic watch probability with "
            f"{int(c.get('conditions_met', 0) or 0)}/5 criticality signals active"
        )
        technical_subline = (
            f"{int(c.get('n_events', 0) or 0)} events in-window | "
            f"Mmax {float(c.get('max_mag', 0) or 0):.1f} | "
            f"Rate {float(c.get('rate_acceleration', 0) or 0):.2f}x | "
            f"S/Gamma {float(c.get('S_over_Gamma', 0) or 0):.2f}"
        )
        sing = c.get("singularity_detail", {})
        cond_labels = [
            ("ell_elevated", "Correlation length elevated"),
            ("b_depressed", "b-value depressed"),
            ("iet_lorentzian", "IET Lorentzian"),
            ("rate_accelerating", "Rate accelerating"),
            ("loading_exceeds_healing", "Loading exceeds healing"),
        ]
        active_conditions = [
            label for key, label in cond_labels if sing.get(key)
        ]
        if not active_conditions:
            active_conditions = ["No singularity conditions are currently active"]

        lines.append(
            f'        <details class="event-row-details" id="cell-{rank}">'
        )
        lines.append('          <summary class="event-row">')
        lines.append(
            f'            <span class="event-row-leading">'
            f'<span class="rank-badge{rank_class}">{rank}</span>'
            f'<span class="event-row-copy">'
            f'<span class="event-row-mode" data-depth="simple">'
            f'<span class="event-row-headline">'
            f'<span class="event-row-title">{_esc(cell_label)}</span>'
            f'<span class="chip" style="background:{risk_color};'
            f'color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(risk_label)}</span>'
            f'</span>'
            f'<span class="event-row-subline">{_esc(simple_subline)}</span>'
            f'</span>'
            f'<span class="event-row-mode" data-depth="technical">'
            f'<span class="event-row-headline">'
            f'<span class="event-row-title">{_esc(cell_label)}</span>'
            f'<span class="chip" style="background:{risk_color};'
            f'color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(risk_label)}</span>'
            f'</span>'
            f'<span class="event-row-subline">{_esc(technical_subline)}</span>'
            f'</span>'
            f'</span>'
            f'</span>'
            f'<span class="event-row-side">'
            f'<span class="event-row-score" style="--event-accent:{risk_color};">'
            f'{_pct(probability)}</span>'
            f'<span class="event-row-caret" aria-hidden="true"></span>'
            f'</span>'
        )
        lines.append('          </summary>')
        lines.append('          <div class="event-detail">')
        lines.append('            <div data-depth="simple" class="detail-mode">')
        lines.append('              <div class="detail-hero">')
        lines.append('                <div>')
        lines.append('                  <div class="detail-kicker">Threat brief</div>')
        lines.append(
            f'                  <h3 class="detail-title">Grid cell {_esc(cell_label)}</h3>'
        )
        lines.append(
            f'                  <p class="detail-copy">{_esc(summary_sentence)}</p>'
        )
        lines.append('                </div>')
        lines.append('                <div class="detail-badge-stack">')
        lines.append(
            f'                  <span class="chip" style="background:{risk_color};'
            f'color:#fff;font-size:12px;padding:4px 10px;">{_esc(risk_label)}</span>'
        )
        lines.append(
            f'                  <div class="detail-score" style="color:{risk_color};">'
            f'{_pct(probability)}</div>'
        )
        lines.append(
            '                  <div class="detail-score-label">'
            'Estimated M6.0+ probability in the next 30 days'
            f'{_band_text(c.get("confidence_lo"), c.get("confidence_hi"))}'
            '</div>'
        )
        lines.append('                </div>')
        lines.append('              </div>')
        lines.append('              <div class="detail-alert">')
        lines.append('                <strong>Operational posture</strong>')
        lines.append(f'                <p>{_esc(action_text)}</p>')
        lines.append('              </div>')
        if watch_items:
            lines.append('              <div class="detail-chip-row">')
            for item in watch_items:
                lines.append(
                    f'                <span class="signal-pill">{_esc(item)}</span>'
                )
            lines.append('              </div>')
        lines.append('              <div class="detail-signal-grid" style="margin-top:16px;">')
        lines.append(
            f'                <div class="signal-card"><span>Recent events</span>'
            f'<strong>{int(c.get("n_events", 0) or 0)}</strong>'
            f'<small>Cataloged M2.5+ events in this cell</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>Largest event</span>'
            f'<strong>M{float(c.get("max_mag", 0) or 0):.1f}</strong>'
            f'<small>Strongest event in the current lookback window</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>Rate acceleration</span>'
            f'<strong>{float(c.get("rate_acceleration", 0) or 0):.2f}x</strong>'
            f'<small>Current rate versus local background</small></div>'
        )
        lines.append(
            f'                <div class="signal-card"><span>Conditions met</span>'
            f'<strong>{int(c.get("conditions_met", 0) or 0)} / 5</strong>'
            f'<small>Active coherence singularity conditions</small></div>'
        )
        lines.append('              </div>')
        lines.append('            </div>')

        lines.append('            <div data-depth="technical" class="detail-mode">')
        lines.append('              <div class="detail-hero">')
        lines.append('                <div>')
        lines.append('                  <div class="detail-kicker">Technical breakdown</div>')
        lines.append(
            f'                  <h3 class="detail-title">Cell {_esc(cell_label)}</h3>'
        )
        lines.append(
            f'                  <p class="detail-copy">{_esc(summary_sentence)} '
            f'The current window contains {int(c.get("n_events", 0) or 0)} '
            f'recent catalog events with a maximum magnitude of '
            f'{float(c.get("max_mag", 0) or 0):.1f}.</p>'
        )
        lines.append('                </div>')
        lines.append('                <div class="detail-badge-stack">')
        lines.append(
            f'                  <span class="chip" style="background:{risk_color};'
            f'color:#fff;font-size:12px;padding:4px 10px;">{_esc(risk_label)}</span>'
        )
        lines.append(
            f'                  <div class="detail-score" style="color:{risk_color};">'
            f'{_pct(probability)}</div>'
        )
        lines.append(
            f'                  <div class="detail-score-label">'
            f'{int(c.get("conditions_met", 0) or 0)} of 5 singularity conditions active'
            f'</div>'
        )
        lines.append('                </div>')
        lines.append('              </div>')
        lines.append('              <div class="detail-panel-grid">')
        lines.append('                <section class="detail-panel half">')
        lines.append('                  <div class="detail-section-label">Signal diagnostics</div>')
        lines.append(
            f'                  <div class="kv"><span>b-value</span>'
            f'<strong>{_fmt(c["b_value"], ".3f")}</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>b-value trend</span>'
            f'<strong>{_fmt(c["b_trend"], ".4f")}</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Correlation length</span>'
            f'<strong>{_fmt(c["ell_km"], ".1f")} km</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Corr. length trend</span>'
            f'<strong>{_fmt(c["ell_trend"], ".2f")} km/month</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Rate acceleration</span>'
            f'<strong>{_fmt(c["rate_acceleration"], ".2f")}x</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>IET delta-AIC</span>'
            f'<strong>{_fmt(c["delta_aic_iet"], ".2f")}</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>S / Gamma</span>'
            f'<strong>{_fmt(c["S_over_Gamma"], ".3f")}</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Days to criticality</span>'
            f'<strong>{_fmt(c["days_to_criticality"], ".0f")}</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Depth trend</span>'
            f'<strong>{_fmt(c["depth_trend"], ".2f")} km</strong></div>'
        )
        lines.append(
            f'                  <div class="kv"><span>Spatial concentration</span>'
            f'<strong>{_fmt(c["spatial_concentration"], ".1f")} km</strong></div>'
        )
        lines.append('                </section>')
        lines.append('                <section class="detail-panel half">')
        lines.append('                  <div class="detail-section-label">Singularity conditions</div>')
        for key, label in cond_labels:
            val = sing.get(key)
            if val is not None:
                chip = (
                    '<span class="chip" style="background:#d32f2f;'
                    'color:#fff;font-size:11px;padding:2px 8px;">YES</span>'
                    if val
                    else '<span class="chip" style="background:#757575;'
                    'color:#fff;font-size:11px;padding:2px 8px;">NO</span>'
                )
                lines.append(
                    f'                  <div class="kv"><span>{_esc(label)}</span>{chip}</div>'
                )
        lines.append('                </section>')
        lines.append('              </div>')
        lines.append('              <section class="detail-panel" style="margin-top:12px;">')
        lines.append('                <div class="detail-section-label">What is driving the rank</div>')
        lines.append('                <ul class="detail-list">')
        for item in active_conditions:
            lines.append(f'                  <li>{_esc(item)}</li>')
        lines.append(
            f'                  <li>Window probability: {_pct(probability)} over the next 30 days.</li>'
        )
        lines.append(
            f'                  <li>Catalog support: {int(c.get("n_events", 0) or 0)} recent events, '
            f'max magnitude {float(c.get("max_mag", 0) or 0):.1f}.</li>'
        )
        lines.append('                </ul>')
        lines.append('              </section>')
        lines.append('            </div>')
        lines.append('          </div>')
        lines.append(f'        </details>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coherence deep dive for #1 cell
# ---------------------------------------------------------------------------


def _render_coherence_deep_dive(top: dict) -> str:
    """Render coherence diagnostics for the top-risk cell. Pure HTML, no JS."""
    if not top:
        return ""

    risk_color = RISK_COLORS.get(top["risk_band"], "#757575")
    lines: list[str] = []

    lines.append(
        '      <section class="section" aria-labelledby="focus-heading">'
    )
    lines.append(
        '        <h2 id="focus-heading">'
        'Top risk cell -- coherence diagnostics</h2>'
    )
    lines.append('        <div class="grid">')

    # Coherence fields card
    lines.append('          <div class="card col-6 hazard-eq">')
    lines.append(
        f'            <h3>{_esc(_format_cell_coords(top["lat"], top["lon"]))} '
        f'-- Coherence fields</h3>'
    )
    lines.append(
        f'            <div class="metric" style="color:{risk_color}">'
        f'{_pct(top["probability"])}</div>'
    )
    lines.append(
        '            <div class="metric-label">'
        'Estimated M6.0+ probability (30 days)'
        f'{_band_text(top.get("confidence_lo"), top.get("confidence_hi"))}</div>'
    )

    coh_keys = [
        ("b_value", "b-value (GR)"),
        ("ell_km", "Correlation length (km)"),
        ("ell_trend", "Corr. length trend (km/mo)"),
        ("rate_acceleration", "Rate acceleration"),
        ("delta_aic_iet", "IET delta-AIC"),
        ("S_over_Gamma", "S / Gamma"),
        ("tau_local", "tau (coherence field)"),
        ("grad_tau_local", "grad(tau)"),
        ("days_to_criticality", "Days to criticality"),
    ]
    for key, label in coh_keys:
        val = top.get(key)
        if val is not None:
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}</span>'
                f'<strong>{_fmt(float(val), ".4f")}</strong></div>'
            )
    lines.append('          </div>')

    # Singularity analysis card
    lines.append('          <div class="card col-6 hazard-eq">')
    lines.append('            <h3>Singularity analysis</h3>')
    sing = top.get("singularity_detail", {})
    sing_count = top.get("conditions_met", 0)
    lines.append(
        f'            <div class="kv"><span>Conditions met</span>'
        f'<strong>{sing_count} / 5</strong></div>'
    )

    cond_labels = [
        ("ell_elevated", "Corr. length elevated (>1.5x bg)"),
        ("b_depressed", "b-value depressed (<0.85)"),
        ("iet_lorentzian", "IET Lorentzian (delta-AIC < -2)"),
        ("rate_accelerating", "Rate accelerating (>1.5x)"),
        ("loading_exceeds_healing", "Loading > healing (S/Gamma > 1)"),
    ]
    for sk, label in cond_labels:
        sv = sing.get(sk)
        if sv is not None:
            if sv:
                chip = (
                    '<span class="chip" style="background:#d32f2f;'
                    'color:#fff;font-size:11px;padding:2px 8px;">YES</span>'
                )
            else:
                chip = (
                    '<span class="chip" style="background:#757575;'
                    'color:#fff;font-size:11px;padding:2px 8px;">no</span>'
                )
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}'
                f'</span>{chip}</div>'
            )
    lines.append('          </div>')
    lines.append('        </div>')

    # Cell statistics card
    lines.append('        <div class="grid" style="margin-top:var(--s-md);">')
    lines.append('          <div class="card col-12 hazard-eq">')
    lines.append('            <h3>Cell statistics</h3>')
    lines.append(
        f'            <div class="kv"><span>Location</span>'
        f'<strong>{_esc(_format_cell_coords(top["lat"], top["lon"]))}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Events (30 days)</span>'
        f'<strong>{top["n_events"]}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Max magnitude</span>'
        f'<strong>M{top["max_mag"]:.1f}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>b-value trend</span>'
        f'<strong>{_fmt(top["b_trend"], ".4f")}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Depth trend</span>'
        f'<strong>{_fmt(top["depth_trend"], ".2f")} km</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Model version</span>'
        f'<strong>{_esc(top.get("model_version", "--"))}</strong></div>'
    )
    lines.append('          </div>')
    lines.append('        </div>')
    lines.append('      </section>')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full page rendering (zero JavaScript)
# ---------------------------------------------------------------------------


def render_earthquake_page(
    scored_cells: list[dict],
    now: dt.datetime,
    n_events_total: int = 0,
    *,
    forecast_id: str | None = None,
) -> str:
    """Generate complete static HTML for the earthquake live page.

    All data is embedded directly in the HTML. Zero JavaScript.
    Follows the HazardPulse Truth Surface spec.
    """
    updated_str = _format_time(now.isoformat() + "Z")

    # Load base SVG map. FAIL CLOSED: the base map is a tracked asset that carries the
    # user-marker node the edge worker rewrites. If it is missing the checkout is partial
    # (e.g. sparse) -- rendering the old silent fallback (an empty rectangle) produced a
    # degraded page that clobbered the deployed one when committed (71d56d53).
    svg_path = DIST / "assets" / "world-map-base.svg"
    if not svg_path.exists():
        raise SystemExit(
            f"{svg_path} is missing. This tracked asset must exist to render the live page -- "
            "run from a full checkout (git sparse-checkout disable). Refusing to render a "
            "degraded map-less page."
        )
    svg_content = svg_path.read_text(encoding="utf-8")

    # Inject grid heatmap and markers into the SVG
    if scored_cells:
        heatmap_html = _render_grid_heatmap(scored_cells)
        markers_html = _render_svg_markers(scored_cells)
        svg_content = svg_content.replace(
            "    <!-- Markers go here per page -->\n",
            heatmap_html + "\n" + markers_html + "\n",
        )

    # Build disclaimer
    disclaimer = (
        "Independent hazard intelligence platform. "
        "Always follow official USGS and national geological survey guidance. "
        "See earthquake.usgs.gov for authoritative data."
    )

    # Status bar
    n_active = len(scored_cells)
    top_cond = scored_cells[0]["conditions_met"] if scored_cells else 0
    status_html = f"""
      <section class="section">
        <div class="grid">
          <div class="card col-3">
            <div class="kv"><span>Last update</span><strong>{_esc(updated_str)}</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Model</span><strong>{_esc(MODEL_VERSION)}</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Active cells</span><strong>{n_active} cells scored</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Events (30 d)</span><strong>{n_events_total} M2.5+</strong></div>
          </div>
        </div>
      </section>"""

    # Cells section
    if scored_cells:
        cell_rows = _render_cell_rows(scored_cells)
        cells_html = f"""
      <section class="section" aria-labelledby="cells-heading">
        <h2 id="cells-heading">Top risk cells by seismic criticality</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">2-degree grid cells ranked by estimated M6.0+ probability (30 days). Click any row to expand coherence diagnostics.</p>

{cell_rows}
      </section>"""
        dive_html = _render_coherence_deep_dive(scored_cells[0])
    else:
        cells_html = """
      <section class="section">
        <div class="card" style="text-align:center;padding:48px 24px;">
          <h3 style="color:var(--text-secondary);">No active seismic zones</h3>
          <p class="muted">No grid cells with sufficient seismicity detected. Check back later.</p>
        </div>
      </section>"""
        dive_html = ""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Earthquake Monitor - HazardPulse</title>
  <meta name="description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/earthquake/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="Earthquake Monitor - HazardPulse">
  <meta property="og:description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/earthquake/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Earthquake Monitor - HazardPulse">
  <meta name="twitter:description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Global Earthquake Criticality Forecast",
    "description": "Probabilistic earthquake forecasts for global seismic zones using coherence field theory singularity conditions with full provenance chain.",
    "license": "{PRIMARY_DOMAIN}/legal/disclaimer/",
    "creator": {{ "@type": "Organization", "name": "{SITE_PUBLISHER_NAME}", "url": "{PRIMARY_DOMAIN}/" }},
    "temporalCoverage": "{now.strftime('%Y-%m-%d')}/{(now + dt.timedelta(days=30)).strftime('%Y-%m-%d')}",
    "spatialCoverage": {{ "@type": "Place", "name": "Global seismic zones" }},
    "variableMeasured": "Probability of M6.0+ earthquake in 30 days"
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/tornado/", "/live/hurricane/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when seismic threat detected near user -->
  </div>

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

    <section class="hero" aria-labelledby="hero-heading">
      <div class="eyebrow">Live seismic intelligence</div>
      <h1 id="hero-heading">Global Earthquake Monitor</h1>
      <p class="subtitle">
        30-day M6.0+ earthquake probability for the world's most active seismic zones.
        Grid cells ranked by coherence field singularity conditions with full evidence.
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
        <!-- Worker injects personalized seismic threat content here based on IP geolocation -->
      </section>

      <!-- WORLD MAP -->
      <section class="section" aria-labelledby="worldmap-heading">
        <h2 id="worldmap-heading">Global seismic activity map</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">2-degree grid cells colored by M6.0+ probability. Markers show top 10 risk zones. Hover for details.</p>

        <div class="world-map-wrapper">
          {svg_content}

          <div class="map-legend">
            <span class="map-legend-item"><span class="map-legend-dot eq"></span> Seismic zone</span>
            <span class="map-legend-item"><span style="display:inline-block;width:20px;height:2px;border-top:2px dashed var(--eq);opacity:.5;vertical-align:middle;margin-right:2px;"></span> Plate boundary</span>
            <span class="map-legend-item"><span class="map-legend-dot user"></span> Your location</span>
          </div>
        </div>
      </section>

      <!-- DISCLAIMER BANNER -->
      <section class="section">
        <div class="card" style="background:var(--warn-bg,#fff8e1);border-left:4px solid var(--warn,#c98a12);padding:12px 16px;">
          {_esc(disclaimer)}
        </div>
      </section>

      <!-- STATUS BAR -->
{status_html}

      <!-- TOP RISK CELLS -->
{cells_html}

      <!-- TOP CELL DEEP DIVE -->
{dive_html}

      <!-- EVIDENCE AND REPLAY -->
      <section class="section" aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence and replay</h2>
        <div class="grid">
          <div class="card col-4">
            <h3>See the evidence</h3>
            <p class="muted">Every forecast links to the exact data and model that produced it. Nothing is hidden.</p>
            <a href="/evidence/#eq" class="btn btn-secondary" style="margin-top:8px;">Browse evidence</a>
          </div>
          <div class="card col-4">
            <h3>Check our track record</h3>
            <p class="muted">How often are we right? We publish accuracy scores publicly, broken down by region and magnitude.</p>
            <a href="/verification/" class="btn btn-secondary" style="margin-top:8px;">See accuracy</a>
          </div>
          <div class="card col-4">
            <h3>Replay any forecast</h3>
            <p class="muted">Download the input data and re-run any past forecast yourself. Same data in, same result out - guaranteed.</p>
            <a href="/data/replay/{_esc(forecast_id or f'eq_fcst_{now.strftime("%Y%m%d")}_0300')}.json" class="btn btn-secondary" style="margin-top:8px;">Download replay</a>
          </div>
        </div>
      </section>

    </div>
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
        <a href="https://earthquake.usgs.gov/" rel="noopener">USGS</a>
        <a href="https://www.nhc.noaa.gov/" rel="noopener">NHC</a>
        <a href="https://www.spc.noaa.gov/" rel="noopener">SPC</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the USGS, NHC, NWS, SPC, JMA, and IMD.
        Probabilistic outputs represent model estimates with stated uncertainty.
      </p>
      <p class="footer-build">Static-first HTML &middot; Evidence-linked data &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>

</body>
</html>
"""
    return page


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
FEATURE_HISTORY_DAYS = 400
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
) -> list[dict]:
    """Score active grid cells using causal history and recent activity.

    Per-cell PRIMARY probability precedence: the deep GRU year-ahead nowcast
    (``deep_scorer``, AUC ~0.86 -- the measured champion) > the exported VerifiableForest
    champion (``eq_forest``) > the incumbent GBT (``pretrained_gbt``) > singularity
    heuristic. If ``deep_scorer_st`` is given, each cell ALSO gets a second, independent
    field ``prob_30d_local`` = P(M4.5+ within 50km / 30 days) from the short-term local
    model (AUC ~0.895) -- a distinct product, not a fallback.
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
    scored_cells: list[dict] = []
    n_ml = n_heur = 0

    for (row, col), cell_events in cell_bins.items():
        if len(cell_events) < MIN_CELL_EVENTS:
            continue

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
        # Tier 1a: deep GRU nowcast (champion). Reads the raw event sequence directly --
        # no Block S/C needed. P(M5+ within radius/365d) precursory-state score.
        if deep_scorer is not None and cat_arrays is not None:
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
                "probability": round(prob, 4),
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
        print(f"  Scored cells by tier: tier1_ml={n_ml}, tier2_heuristic={n_heur}")
    return scored_cells


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


def refresh_ledger_anchor() -> None:
    """Re-commit the chain's length and head after an append.

    The anchor (`evidence/ledger-anchors.json`) is what makes TRUNCATION detectable: a chain with
    its tail removed is still a perfectly linked chain. Refreshing it here, in the writer, is the
    only placement that cannot be forgotten by a new workflow. It RAISES if the chain no longer
    verifies -- publishing a forged ledger is worse than a failed run.
    """
    import importlib.util
    import sys as _sys

    name = "hazardpulse_ledger_anchor"
    module = _sys.modules.get(name)
    if module is None:
        module_path = Path(__file__).resolve().parent / "verify_ledger_chain.py"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        # MUST be registered BEFORE exec_module: @dataclass resolves cls.__module__ through
        # sys.modules and raises AttributeError on None if the module is not there yet.
        _sys.modules[name] = module
        spec.loader.exec_module(module)
    document = module.refresh_anchor()
    counts = {k.split("/")[-1]: v["n_entries"] for k, v in document["ledgers"].items()}
    print(f"  Re-anchored ledger chains: {counts}")


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
    # The digest covers EVERY key present at this point. Do not attach fields to `entry` after this
    # line: `forecast_id` was once added post-hash and 13 rows of the live chain still carry it
    # outside their own digest (see scripts/verify_ledger_chain.py).
    payload = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"  Appended to {ledger_path}")
    if ledger_path == LEDGER_PATH:
        refresh_ledger_anchor()


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
        if not skip_site:
            page_now = now.replace(tzinfo=None)
            eq_html = render_earthquake_page(
                [],
                page_now,
                n_events_total=0,
                forecast_id=forecast_id,
            )
            eq_page = DIST / "live" / "earthquake" / "index.html"
            eq_page.parent.mkdir(parents=True, exist_ok=True)
            eq_page.write_text(eq_html, encoding="utf-8")
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
    print("Step 3: Scoring active grid cells...")
    pretrained_eq_gbt = load_pretrained_eq_gbt()
    eq_forest = load_eq_forest()        # deployable champion; precedence over the GBT when present
    deep_eq_scorer = load_deep_eq_scorer_model()       # year-ahead regional nowcast (primary)
    deep_eq_scorer_st = load_deep_eq_shortterm_model()  # short-term local watch (2nd field)
    deep_eq_scorer_op = load_deep_eq_operational_model()  # operational forecaster (3rd, the WHERE-skill)
    scored = score_grid_cells(
        history_events,
        candidate_events=recent_events,
        grid_fields=grid_fields,
        now=now,
        pretrained_gbt=pretrained_eq_gbt,
        eq_forest=eq_forest,
        deep_scorer=deep_eq_scorer,
        deep_scorer_st=deep_eq_scorer_st,
        deep_scorer_op=deep_eq_scorer_op,
    )
    print(f"  {len(scored)} cells scored")

    # Trust layer: calibrate probabilities, attach honest [conf_lo, conf_hi]
    # bands + Ed25519-signed re-runnable receipts. Fails safe — if no calibrator
    # has been produced yet, forecasts stay raw (uncalibrated) and honest.
    try:
        from hazardpulse.trust.scoring import enrich_cells, load_forecaster, load_signer

        _signer = load_signer()
        _forecaster = load_forecaster("earthquake", signer=_signer)
        if _forecaster is not None:
            enrich_cells(scored, _forecaster, issued_at=format_utc_z(now))
            print(
                f"  Trust layer: calibrated {len(scored)} cells "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
        else:
            print(
                "  Trust layer: no calibrator yet "
                "(results/models/earthquake_calibration.json); emitting raw forecasts."
            )
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

    if not skip_site:
        print()
        print("Step 5: Rendering static HTML page (zero JS)...")
        page_now = now.replace(tzinfo=None)
        eq_html = render_earthquake_page(
            scored,
            page_now,
            n_events_total=len(recent_events),
            forecast_id=forecast_id,
        )
        eq_page = DIST / "live" / "earthquake" / "index.html"
        eq_page.parent.mkdir(parents=True, exist_ok=True)
        eq_page.write_text(eq_html, encoding="utf-8")
        print(f"  Wrote {eq_page} ({len(scored)} cells baked in, zero JS)")

    if not skip_site and not skip_live_pulse:
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
