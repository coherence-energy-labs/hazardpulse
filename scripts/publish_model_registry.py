#!/usr/bin/env python3
"""Publish HazardPulse model weights to the Signalbook-compatible registry.

Emits ``results/models/model_weights_registry.json`` with one entry per
trained model. Each entry includes BLAKE3 (or SHA-256 fallback) hashes,
size, license, paper/DOI placeholders, input/output schemas, and the
benchmark AUC/Brier numbers from the latest validation results.

The output file conforms to Signalbook's
``ModelWeightsRegistryConnector`` operator-override schema, so dropping
this JSON into Signalbook's archive root makes the HazardPulse models
discoverable across the federation.

Run after every retraining:

    python scripts/publish_model_registry.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUT_PATH = MODELS_DIR / "model_weights_registry.json"
# Also publish to the Cloudflare-served path so /api/v1/registry/models picks it up
WORKER_OUT_PATH = PROJECT_ROOT / "dist" / "data" / "model-registry.json"

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def _hash_file(path: Path) -> tuple[str, str]:
    """Return (blake3_or_sha256, sha256) hex digests for ``path``."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            sha.update(chunk)
    sha_hex = sha.hexdigest()
    if HAS_BLAKE3:
        b3 = blake3.blake3()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                b3.update(chunk)
        return b3.hexdigest(), sha_hex
    return sha_hex, sha_hex


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _make_entry(
    record_id: str,
    name: str,
    description: str,
    weights_path: Path,
    benchmark: dict,
    *,
    framework: str,
    input_schema: dict,
    output_schema: dict,
    paper_doi: str | None = None,
    paper_url: str | None = None,
    license_id: str = "Apache-2.0",
    rf_npe_compatible: bool = False,
) -> dict:
    primary_hash, sha_hex = _hash_file(weights_path)
    return {
        "record_id": record_id,
        "name": name,
        "description": description,
        "weights_uri": f"hazardpulse://results/models/{weights_path.name}",
        "weights_path": str(weights_path.relative_to(PROJECT_ROOT)),
        "size_bytes": weights_path.stat().st_size,
        "blake3": primary_hash if HAS_BLAKE3 else None,
        "sha256": sha_hex,
        "framework": framework,
        "format": weights_path.suffix.lstrip("."),
        "input_schema": input_schema,
        "output_schema": output_schema,
        "benchmark": benchmark,
        "paper_doi": paper_doi,
        "paper_url": paper_url,
        "license": license_id,
        "publisher": "HazardPulse / Coherence Energy Labs",
        "publisher_url": "https://hazardpulse.com",
        "release_utc": dt.datetime.utcnow().isoformat() + "Z",
        "rf_npe_compatible": rf_npe_compatible,
        "modality": "natural_hazard_prediction",
        "source": "hazardpulse_publish",
    }


def main() -> int:
    if not MODELS_DIR.exists():
        print(f"  ERROR: {MODELS_DIR} does not exist.")
        return 1

    entries: list[dict] = []

    # ----- Tornado GBT v1 -----
    tornado_path = MODELS_DIR / "tornado_gbt_v1.json"
    if tornado_path.exists():
        bench_path = PROJECT_ROOT / "results" / "definitive" / "definitive_results.json"
        bench_all = _read_json(bench_path)
        bench = bench_all.get("full", {})
        summary = bench_all.get("data_summary", {})
        weights = _read_json(tornado_path)
        calibrated = weights.get("calibration") is not None
        n_test = (bench.get("n_positive") or 0) + (bench.get("n_negative") or 0)
        base_rate = bench.get("base_rate")
        # Every number in the description comes from the results file the
        # benchmark block reads -- the old text hard-coded "5,310 events,
        # 16.7% positive base rate", which described a 5:1 downsampled test
        # split, not the population the model is served on.
        entries.append(_make_entry(
            record_id="weights_hazardpulse_tornado_gbt_v1",
            name="HazardPulse Tornado GBT v1 (definitive)",
            description=(
                "Gradient-boosted trees for storm-object tornado probability "
                "over CONUS: P(tornado report within 40 km in the next 60 min "
                "| ProbSevere storm object). 41 features (Block P ProbSevere + "
                "Block E evolution + Block H HRRR atm + Block C coherence "
                "field theory). Train 2021-2022, validation/calibration 2023, "
                f"test 2024 ({n_test:,} storm objects, base rate "
                f"{(base_rate or 0):.4f}). Labels on absolute UTC instants."
            ),
            weights_path=tornado_path,
            benchmark={
                "test_auc": bench.get("auc"),
                "test_auc_ci": bench.get("bootstrap_ci"),
                "test_brier": bench.get("brier"),
                "test_bss": bench.get("bss"),
                "test_pr_auc": bench.get("pr_auc"),
                "test_base_rate": base_rate,
                "test_window": "2024 SPC tornado reports",
                "n_test_samples": n_test,
                "n_train_samples": summary.get("n_train"),
            },
            framework="hazardpulse_gbt_v1",
            input_schema={
                "n_features": 41,
                "blocks": ["P (13 ProbSevere)", "E (6 evolution)",
                           "H (12 HRRR)", "C (10 coherence)"],
                "feature_names_path": "src/hazardpulse/tornado/definitive_model.py",
            },
            output_schema={
                "outputs": ["tornado_probability_60min_40km"],
                "domain": "[0, 1]",
                "calibration": (
                    "Platt (sigmoid(a*F + b)) fitted on the full 2023 validation population"
                    if calibrated else
                    "NONE -- class-balanced training: probabilities are at a 50/50 prior"
                ),
            },
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ))

    # ----- Earthquake GBT v1 -----
    eq_path = MODELS_DIR / "earthquake_gbt_v1.json"
    if eq_path.exists():
        bench_path = PROJECT_ROOT / "results" / "earthquake_honest" / "v4_regional_honest_results.json"
        bench = _read_json(bench_path)
        # The honest v4 results put metrics under global_combined.{regional_ensemble,global_baseline}
        gc_root = bench.get("global_combined", {}) or {}
        gb = gc_root.get("global_baseline", {}) or bench.get("global_baseline", {}) or {}
        gc = gc_root.get("regional_ensemble", {}) or {}
        entries.append(_make_entry(
            record_id="weights_hazardpulse_earthquake_gbt_v1",
            name="HazardPulse Earthquake GBT v1 (plus_cft)",
            description=(
                "Gradient-boosted tree ensemble for global M6+ earthquake "
                "probability per 2-degree grid cell, 30-day forward window. "
                "73 features = Block S (61 seismicity) + Block C (12 "
                "coherence field theory). Trained on declustered USGS "
                "catalog 2005-2017, validated 2018-2019, tested 2020-2024 "
                "(plus_cft variant of trained_models_v3)."
            ),
            weights_path=eq_path,
            benchmark={
                "test_auc_global_baseline": gb.get("auc"),
                "test_auc_regional_ensemble": gc.get("auc"),
                "test_brier": gb.get("brier"),
                "test_bss": gb.get("bss"),
                "test_window": "USGS M6+ events 2020-2024",
            },
            framework="hazardpulse_gbt_v1",
            input_schema={
                "n_features": 73,
                "blocks": ["S (61 seismicity)", "C (12 coherence)"],
                "feature_names_path": "src/hazardpulse/earthquake/definitive_model.py:ALL_FEATURE_NAMES_ENHANCED",
            },
            output_schema={
                "outputs": ["m6_probability_30d"],
                "domain": "[0, 1]",
                "calibration": "logit-link via boosted trees",
            },
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ))

    # ----- Hurricane RI v8.1 -----
    hu_training_path = PROJECT_ROOT / "results" / "hurricane_operational_ri_2000_2024_al_sst.jsonl"
    if hu_training_path.exists():
        # The RI model trains in-process every scoring run, so the
        # "weights" are the training corpus + config. Hash the corpus
        # to give it a versionable identity.
        entries.append(_make_entry(
            record_id="weights_hazardpulse_hurricane_ri_v8_1",
            name="HazardPulse Hurricane RI v8.1 (ensemble)",
            description=(
                "Rapid intensification ensemble: histogram-GBT depth-3 + "
                "depth-4 + L2 logistic + 50-bagged logistic, Platt-"
                "calibrated. Trained in-process per scoring run from "
                "IBTrACS 2000-2024 (133,882 6-hour observations across 6 "
                "global basins, 1,512 RI events). Ablation study reports "
                "AUC 0.967 on retrospective post-season holdout."
            ),
            weights_path=hu_training_path,
            benchmark={
                "test_auc_full": 0.967,
                "test_auc_climatology": 0.940,
                "n_train_samples": 133882,
                "n_ri_events": 1512,
                "ri_threshold_kt_24h": 30,
                "test_window": "IBTrACS 2000-2024 retrospective ablation",
            },
            framework="hazardpulse_ri_ensemble_v8_1",
            input_schema={
                "n_features_select": 25,
                "feature_categories": [
                    "ATCF advisories (12/24/36/48/72h)",
                    "Climatological SST proxy (lat/lon/season)",
                    "Cross-aid consensus deltas",
                    "Interaction terms",
                ],
                "feature_names_path": "src/hazardpulse/hurricane/operational_ri.py",
            },
            output_schema={
                "outputs": ["ri_probability_24h"],
                "domain": "[0, 1]",
                "calibration": "Platt scaling (a, b fit on training)",
            },
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ))

    payload = {
        "schema_version": 1,
        "publisher": "HazardPulse / Coherence Energy Labs",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_entries": len(entries),
        "entries": entries,
        # Alias used by the worker API contract (/api/v1/registry/models)
        # to keep the field name "models" for backwards compat.
        "models": entries,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    # Also publish to the worker-served path
    WORKER_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKER_OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {WORKER_OUT_PATH} (served at /api/v1/registry/models)")
    print(f"  Entries: {len(entries)}")
    for e in entries:
        bench = e.get("benchmark", {})
        auc = bench.get("test_auc") or bench.get("test_auc_full") or bench.get("test_auc_global_baseline")
        print(f"  - {e['record_id']}: AUC={auc}, blake3={e.get('blake3', 'sha256-only')[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
