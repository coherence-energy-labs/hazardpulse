#!/usr/bin/env python3
"""Live tornado forecast from the HRRR atmospheric environment + the signed forest.

Fetches the day's HRRR analysis, builds the 26-feature environment vector for every
CONUS cell (the SAME features the model trained on, via tornado.hrrr_env), and scores
each convective cell with the deployed, 0-ULP-signed VerifiableForest -> a calibrated
P(tornado-in-cell). Its published 0.81-0.85 AUC compared storms with storm-free cells;
storm-vs-storm (2026-10-01) a retrained model does not beat STP alone. Research only;
standalone -- does not touch the ProbSevere scorer.

    python scripts/score_tornado_hrrr_env.py --date 20240526 --hour 20 --top 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HAZARDPULSE_HRRR_POOL", "max")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data.hrrr import (  # noqa: E402
    GEOMETRY_VERSION, HRRR_N_LON, GRID_LATS, GRID_LONS, fetch_hrrr_grid, load_cached_hrrr,
)
from hazardpulse.tornado.hrrr_env import (  # noqa: E402
    FEATURE_NAMES,
    grid_feature_matrix,
    sanitize_grids,
    storm_population_mask,
)
from hazardpulse.trust.forest_serve import load_forest_scorer  # noqa: E402
from hazardpulse.trust.venn_abers import VennAbersCalibrator  # noqa: E402

_FP_DIR = REPO / "results" / "calibration"


def _load_calibrator():
    """The Venn-Abers calibrator fit at deploy time (raw forest score -> honest prob)."""
    path = _FP_DIR / "tornado_hrrr_calibration.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return VennAbersCalibrator.from_dict(rec["calibrator"])


def _risk_band(p: float) -> str:
    if p >= 0.50:
        return "high"
    if p >= 0.20:
        return "elevated"
    if p >= 0.05:
        return "marginal"
    return "low"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--hour", type=int, default=20)
    ap.add_argument("--min-cape", type=float, default=250.0,
                    help="only score convective cells (mlcape >= this)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    scorer = load_forest_scorer("tornado", _FP_DIR)
    if scorer is None:
        print("No tornado_forest_fp.json deployed yet (run train_tornado_hrrr.py --deploy).")
        return 1
    # A model may only score grids built the way its training grids were. The
    # sidecar records the feature spec and the HRRR grid geometry; a model
    # without a geometry was trained on g1 grids (index-space pooling read as
    # lat/lon -- every cell ~280 km from its label), which no longer exist.
    sidecar_path = _FP_DIR / "tornado_forest_features.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    model_spec = sidecar.get("spec", "hazardpulse/tornado/hrrr-env/v1")
    model_geometry = sidecar.get("geometry", "hrrr-g1")
    if model_geometry != f"hrrr-{GEOMETRY_VERSION}":
        print(f"Refusing: the deployed forest was trained on {model_geometry} grids; "
              f"this build serves hrrr-{GEOMETRY_VERSION}. Retrain on the current grids "
              "(scripts/build_tornado_hrrr_dataset.py + train_tornado_hrrr.py).")
        return 1
    spec_v2 = model_spec.endswith("/v2")
    # the served feature space must match what the forest references
    feat_idx = [int(f) for f in scorer.constants.get("feat", []) if int(f) >= 0]
    if feat_idx and max(feat_idx) >= len(FEATURE_NAMES):
        print(f"Refusing: forest references feature {max(feat_idx)} >= {len(FEATURE_NAMES)}.")
        return 1

    grids = (load_cached_hrrr(args.date, args.hour) if args.cached_only
             else fetch_hrrr_grid(args.date, args.hour))
    if grids is None:
        print(f"No HRRR for {args.date} {args.hour}z.")
        return 1

    X, cape = grid_feature_matrix(grids, sanitize=spec_v2)
    raw = scorer.raw_proba(X)                        # raw (overconfident) forest score
    cal = _load_calibrator()
    if cal is not None:
        proba, lo, hi = cal.predict(raw)             # honest calibrated prob + validity band
    else:
        proba, lo, hi = raw, raw, raw
        print("  WARNING: no calibrator -> emitting RAW (overconfident) probabilities.")
    cape_flat = np.asarray(cape, float).ravel()
    if spec_v2:
        # Score exactly the population the model was trained and benchmarked
        # on: storm cells (refc, shear and CAPE gates), not every warm cell --
        # a v1 forest put 83.5% of its probability mass on cells with no storm.
        convective = storm_population_mask(sanitize_grids(grids)).ravel()
    else:
        convective = np.isfinite(cape_flat) & (cape_flat >= args.min_cape)

    cells = []
    for flat in np.where(convective)[0]:
        i, j = divmod(int(flat), HRRR_N_LON)
        cells.append({
            "lat": round(float(GRID_LATS[i]), 2), "lon": round(float(GRID_LONS[j]), 2),
            "row": i, "col": j, "probability": round(float(proba[flat]), 4),
            "conf_lo": round(float(lo[flat]), 4), "conf_hi": round(float(hi[flat]), 4),
            "raw_probability": round(float(raw[flat]), 4),
            "risk_band": _risk_band(float(proba[flat])),
            "mlcape": round(float(cape_flat[flat]), 0),
        })
    cells.sort(key=lambda c: c["probability"], reverse=True)
    forecast_calibrated = cal is not None

    forecast = {
        "hazard": "tornado", "method": "hrrr_environment_forest",
        "date": args.date, "hour": args.hour, "calibrated": forecast_calibrated,
        "model_sha256": scorer.constants.get("model_sha256") or scorer.model_sha256,
        "n_convective_cells": int(convective.sum()),
        "max_probability": (cells[0]["probability"] if cells else 0.0),
        "top_cells": cells[: args.top],
    }
    print(f"HRRR-env tornado forecast {args.date} {args.hour}z "
          f"({'calibrated' if forecast_calibrated else 'RAW'}): "
          f"{forecast['n_convective_cells']} convective cells, "
          f"max P(tor)={forecast['max_probability']:.3f}")
    for c in cells[: args.top]:
        print(f"  ({c['lat']:.1f},{c['lon']:.1f})  P={c['probability']:.3f} "
              f"[{c['conf_lo']:.3f},{c['conf_hi']:.3f}]  {c['risk_band']:9s} "
              f"cape={c['mlcape']:.0f}  (raw {c['raw_probability']:.2f})")
    if args.out:
        Path(args.out).write_text(json.dumps(forecast, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
