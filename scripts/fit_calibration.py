#!/usr/bin/env python3
"""Fit a probability calibrator for a hazard from its own matured forecasts.

The most honest calibration signal is the platform's OWN predictions vs realized
outcomes. The prospective scorers pool, per hazard, a histogram of
(forecast_probability -> #positive, #total) over every matured grid cell / storm
and write it to ``results/<hazard>_prospective/calibration_dataset.json``. This
script fits a Venn-Abers calibrator to that histogram, measures the calibration
before and after (ECE / Brier / Brier-skill), and persists the fitted calibrator
to ``results/models/<hazard>_calibration.json`` for the live scorer to load.

This is the step that turns "we have a calibrator" into "we have a calibrator fit
on real outcomes", and the before/after numbers are the auditable proof the fix
works on live data (e.g. tornado Brier-skill from negative back above 0).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hazardpulse.trust.calibration import brier_from_counts, ece_from_counts  # noqa: E402
from hazardpulse.trust.venn_abers import VennAbersCalibrator  # noqa: E402

HAZARDS = ("earthquake", "tornado", "hurricane")
DEFAULT_MODEL_VERSION = {
    "earthquake": "eq_coherence_v1_0",
    "tornado": "tornado_storm_v1_0",
    "hurricane": "hurricane_ri_v8_1",
}


def _metrics(probs, pos, total) -> dict:
    base = float(np.sum(pos) / max(np.sum(total), 1.0))
    bs_clim = base * (1.0 - base)
    brier = brier_from_counts(probs, pos, total)
    bss = (1.0 - brier / bs_clim) if bs_clim > 0 else float("nan")
    return {
        "ece": round(ece_from_counts(probs, pos, total), 6),
        "brier": round(brier, 8),
        "brier_skill_score": round(bss, 6) if bss == bss else None,
        "base_rate": round(base, 8),
    }


def fit_one(dataset_path: Path, out_path: Path, *, model_version: str,
            min_calibration: int = 200, max_groups: int = 512) -> dict:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    scores = np.asarray(data["scores"], dtype=np.float64)
    pos = np.asarray(data["pos"], dtype=np.float64)
    total = np.asarray(data["total"], dtype=np.float64)

    cal = VennAbersCalibrator(min_calibration=min_calibration, max_groups=max_groups)
    cal.fit_grouped(scores, pos, total)

    before = _metrics(scores, pos, total)
    if cal.inflated:
        after = before
    else:
        cal_probs, _, _ = cal.predict(scores)
        after = _metrics(cal_probs, pos, total)

    payload = {
        "schema_version": 1,
        "hazard": data.get("hazard"),
        # The dataset names the model whose raw scores it pooled (tornado since
        # 2026-10-01); the calibrator is only valid for that model. The
        # per-hazard constant is the fallback for datasets that predate it.
        "model_version": data.get("model_version") or model_version,
        "fitted_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_calibration": int(np.sum(total)),
        "n_groups": int(scores.size),
        "inflated": bool(cal.inflated),
        "metrics_before": before,
        "metrics_after": after,
        "calibrator": cal.to_dict(),
        "source_dataset": str(dataset_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard", choices=HAZARDS, action="append", dest="hazards",
                        help="Hazard(s) to fit (default: all available).")
    parser.add_argument("--min-calibration", type=int, default=200)
    parser.add_argument("--max-groups", type=int, default=512)
    args = parser.parse_args(argv)

    hazards = args.hazards or list(HAZARDS)
    any_done = False
    for hazard in hazards:
        ds = REPO_ROOT / "results" / f"{hazard}_prospective" / "calibration_dataset.json"
        if not ds.is_file():
            print(f"  {hazard}: no calibration_dataset.json yet (run the prospective scorer with "
                  f"--emit-calibration); skipping.")
            continue
        out = REPO_ROOT / "results" / "calibration" / f"{hazard}_calibration.json"
        payload = fit_one(ds, out, model_version=DEFAULT_MODEL_VERSION[hazard],
                          min_calibration=args.min_calibration, max_groups=args.max_groups)
        b, a = payload["metrics_before"], payload["metrics_after"]
        print(f"  {hazard}: n={payload['n_calibration']}  inflated={payload['inflated']}")
        print(f"    ECE  {b['ece']:.4f} -> {a['ece']:.4f}")
        print(f"    BSS  {b['brier_skill_score']} -> {a['brier_skill_score']}")
        print(f"    wrote {out}")
        any_done = True
    if not any_done:
        print("No calibration datasets found. Run the prospective scorers with --emit-calibration.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
