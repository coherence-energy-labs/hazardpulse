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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUT_PATH = MODELS_DIR / "model_weights_registry.json"
# Also publish to the Cloudflare-served path so /api/v1/registry/models picks it up
WORKER_OUT_PATH = PROJECT_ROOT / "dist" / "data" / "model-registry.json"

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


TEXT_SUFFIXES = (".json", ".jsonl", ".txt", ".md", ".csv")


def _canonical_bytes(path: Path) -> bytes:
    """A text artifact's bytes with CRLF folded to LF -- the bytes git stores, so a hash is the same
    on a Windows checkout (core.autocrlf) and in Linux CI. Binary files are hashed as they are."""
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_SUFFIXES else data


def _hash_file(path: Path) -> tuple[str, str]:
    """Return (blake3_or_sha256, sha256) hex digests of ``path``'s canonical bytes."""
    data = _canonical_bytes(path)
    sha_hex = hashlib.sha256(data).hexdigest()
    if HAS_BLAKE3:
        return blake3.blake3(data).hexdigest(), sha_hex
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
        "weights_path": weights_path.relative_to(PROJECT_ROOT).as_posix(),      # the same on every OS
        "size_bytes": len(_canonical_bytes(weights_path)),
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
        "release_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "rf_npe_compatible": rf_npe_compatible,
        "modality": "natural_hazard_prediction",
        "source": "hazardpulse_publish",
    }


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(_canonical_bytes(path)).hexdigest() if path.exists() else None


# Our own hurricane RI models in shadow beside NOAA's published number: identity from each artifact,
# lineage declared here and checked by tests/test_model_registry.py (every file named must exist).
HU_SHADOWS = (
    {"file": "hurricane_ri_v9.json", "label": "v9.1", "kind": "v9", "shadow_key": "ri_v9_shadow",
     "entrant": "v9_1", "amendment": "protocol + amendment 1", "prereg_tag": "prereg-hurricane-ri-v9",
     "results": ["results/calibration/hurricane_ri_v9_selection.json", "results/calibration/hurricane_ri_v9_final.json"],
     "what": "LightGBM on the early intensity aids' 24-h changes, NOAA's RI probabilities at six thresholds "
             "and the official forecast; DTOPS where the early guidance is missing"},
    {"file": "hurricane_ri_v10.json", "label": "v10.1", "kind": "v10", "shadow_key": "ri_v10_shadow",
     "entrant": "v10_1", "amendment": "amendment 2", "prereg_tag": "prereg-hurricane-ri-v10",
     "results": ["results/calibration/hurricane_ri_v10_selection.json",
                 "results/calibration/hurricane_ri_v10_2026_second_read.json",
                 "results/calibration/hurricane_ri_v10_vs_all.json"],
     "what": "one threshold-stacked exceedance curve P(dV24 >= k), k = 15-45 kt, on v9's inputs; shown on the site"},
    {"file": "hurricane_ri_v10_2.json", "label": "v10.2", "kind": "v10", "shadow_key": "ri_v10_2_shadow",
     "entrant": "v10_2", "amendment": "amendments 3-4", "prereg_tag": "prereg-hurricane-ri-amend3",
     "results": ["results/calibration/hurricane_ri_v10_challengers.json"],
     "what": "v10.1 with monotone constraints on the guidance and NOAA's probabilities"},
    {"file": "hurricane_ri_v10_3.json", "label": "v10.3", "kind": "v10", "shadow_key": "ri_v10_3_shadow",
     "entrant": "v10_3", "amendment": "amendments 5-6", "prereg_tag": "prereg-hurricane-ri-amend5",
     "results": ["results/calibration/hurricane_ri_v10_ir.json"],
     "what": "v10.2 plus 14 cloud-top structure features from NOAA GMGSI infrared (t+2 h and t-4 h)"},
)
HU_PROGRAM = "docs/HURRICANE_RI_V9_PROGRAM.md"


def hurricane_shadow_entries(models_dir: Path = MODELS_DIR) -> list[dict]:
    """The hurricane RI models recorded in shadow on every live NHC cycle, with how each was made and
    where every forecast it issued is kept."""
    from hazardpulse.hurricane import ri_v9, ri_v10

    out = []
    for spec in HU_SHADOWS:
        path = models_dir / spec["file"]
        if not path.exists():
            continue
        if spec["kind"] == "v9":
            art, version = ri_v9.load(path)
            n_inputs = len(art["feature_names"])
            prov = art.get("provenance") or {}
        else:
            art, version = ri_v10.load(path)
            n_inputs = len(art["feature_names"])
            prov = art.get("provenance") or {}
        dev = prov.get("dev_2022_2025") or {}
        results = [{"path": r, "sha256": _file_sha256(PROJECT_ROOT / r)} for r in spec["results"]]
        out.append(_make_entry(
            record_id=f"weights_hazardpulse_{version}",
            name=f"HazardPulse Hurricane RI {spec['label']} (shadow, in prospective verification)",
            description=(f"{spec['what']}. Recorded on every live NHC cycle beside NOAA's published number, never "
                         f"instead of it; chosen by a pre-registered rule ({HU_PROGRAM}, {spec['amendment']})."),
            weights_path=path,
            benchmark={"dev_2022_2025_log_loss_30kt": dev.get("log_loss"), "dev_auc": dev.get("auc"),
                       "dev_brier4": dev.get("brier4"),
                       "claim": "the prospective test in " + HU_PROGRAM + " (looks 2026-12-01, 2027-12-01)"},
            framework="hazardpulse_lgbm_payload" if spec["kind"] == "v9" else "hazardpulse.hurricane_ri_v10/1",
            input_schema={"n_inputs": n_inputs, "feature_names_path": f"results/models/{spec['file']} (feature_names)"},
            output_schema={"outputs": ["ri_probability_24h" if spec["kind"] == "v9" else "exceedance_curve_24h"],
                           "domain": "[0, 1]", "model_version": version},
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ) | {"lineage": {
            "role": "shadow", "program": HU_PROGRAM, "amendment": spec["amendment"],
            "prereg_tag": spec["prereg_tag"], "candidate": prov.get("candidate"),
            "trained": prov.get("trained"), "rounds": prov.get("rounds"), "seeds": prov.get("seeds"),
            "selection_and_evaluation": results,
            "live_records": {"replay": "dist/data/replay/hu_fcst_*.json", "key": spec["shadow_key"],
                             "ledger": "dist/data/hurricane-ledger.jsonl",
                             "audit": "results/hurricane_prospective/record_audit.json"},
            "prospective": {"file": "results/hurricane_prospective/v9_shadow.json", "entrant": spec["entrant"]},
        }})
    return out


HU_STACK_PATH = MODELS_DIR / "hurricane_ri_stack_v1.json"
HU_STACK_FINAL_PATH = PROJECT_ROOT / "results" / "calibration" / "hurricane_ri_stack_final.json"


def hurricane_stack_entry(stack_path: Path = HU_STACK_PATH, final_path: Path = HU_STACK_FINAL_PATH) -> dict | None:
    """The NHC-basin RI model (scripts/hurricane_ri_stack.py) as its own registry entry.

    Identity and inputs come from the artifact (ri_stack.load_artifact refuses a malformed one);
    the benchmark is the program's read-once final season, and only when the final report names
    this very artifact -- otherwise the entry says it has none rather than borrow another's.
    """
    if not stack_path.exists():
        return None
    from hazardpulse.hurricane import ri_stack

    payload, version = ri_stack.load_artifact(stack_path)
    final = _read_json(final_path)
    bound = final.get("phase") == "final" and (final.get("artifact") or {}).get("model_version") == version
    choice = payload["candidate"]
    all_cases = ((final.get("results") or {}).get("all_cases") or {}) if bound else {}
    res = (all_cases.get("forecasts") or {}).get(choice) or {}
    paired = {}
    for other, d in (all_cases.get("paired_chosen_minus") or {}).items():
        paired[other] = {"delta_log_loss": d.get("delta_log_loss"), "delta_brier": d.get("delta_brier"),
                         "delta_auc": d.get("delta_auc"), "ci95": d.get("ci95")}
    v82_subset = ((final.get("results") or {}).get("v82_subset") or {}) if bound else {}
    if "v8_2" in (v82_subset.get("paired_chosen_minus") or {}):
        d = v82_subset["paired_chosen_minus"]["v8_2"]
        paired["v8_2_on_v82_subset"] = {"n": v82_subset.get("n"), "delta_log_loss": d.get("delta_log_loss"),
                                        "delta_brier": d.get("delta_brier"), "delta_auc": d.get("delta_auc"),
                                        "ci95": d.get("ci95")}
    raw = payload["kind"] == "raw_aid"
    season = final.get("final_season") if bound else None
    return _make_entry(
        record_id=f"weights_hazardpulse_{version}",
        name=f"HazardPulse Hurricane RI, NHC basins: NOAA aid stack v1 (candidate {choice})",
        description=(
            "P(>= 30 kt wind increase in 24 h) for Atlantic, East and Central Pacific storms from NOAA's "
            f"operational RI guidance: {payload['description']}. Read live from NHC's SHIPS text for "
            "the storm's cycle; chosen among six pre-registered candidates (DTOPS, SHIPS-RII, RI "
            "consensus, and three logistic stacks of the NOAA aids, one with HazardPulse v8.2) by "
            f"forward chaining on {payload['training']['seasons'][0]}-{payload['training']['seasons'][1]} "
            "under a parsimony rule (docs/HURRICANE_RI_PROGRAM.md)"
            + (f"; scored once on the {season} season." if bound else "; no final-season score is bound to this file.")
        ),
        weights_path=stack_path,
        benchmark={
            "benchmark_bound_to_this_artifact": bound,
            "test_window": f"{season} season, NHC-basin cycles with a SHIPS-RII record, read once" if bound else None,
            "test_auc": res.get("auc"), "test_auc_ci95": res.get("auc_ci95"),
            "test_brier": res.get("brier"), "test_brier_ci95": res.get("brier_ci95"),
            "test_bss_vs_fit_season_climatology": res.get("bss"), "test_bss_ci95": res.get("bss_ci95"),
            "test_log_loss": res.get("log_loss"), "test_log_loss_ci95": res.get("log_loss_ci95"),
            "calibration_slope": res.get("calibration_slope"),
            "n_test_cycles": res.get("n"), "n_test_events": res.get("events"), "n_test_storms": res.get("storms"),
            "paired_this_minus": paired,
            "selection_pooled_dev": {k: ((final.get("selection") or {}).get("choice_pooled_dev") or {}).get(k)
                                     for k in ("log_loss", "auc", "brier", "bss")} if bound else None,
        },
        framework="hazardpulse_noaa_ri_aid_stack_v1",
        input_schema={
            "source": "NHC SHIPS text 'Matrix of RI probabilities', column 30/24 (ATCF e-deck RI records in training)",
            "ships_text_rows": payload["ships_text_rows"],
            "aid": payload.get("aid"), "fallback": payload.get("fallback"), "pool_inputs": payload.get("inputs"),
            "representation": payload["representation"],
            "feature_names_path": "src/hazardpulse/hurricane/ri_stack.py",
        },
        output_schema={
            "outputs": ["ri_probability_24h"],
            "domain": "[0, 1]",
            "calibration": "none: NOAA's probability as issued" if raw else "per-availability-pattern logistic pool",
            "model_version": version,
        },
        license_id="Apache-2.0",
        paper_url="https://github.com/coherence-energy-labs/hazardpulse",
    )


EQ_OPERATIONAL_PATH = MODELS_DIR / "earthquake_operational_v1.json"
EQ_PROGRAM_FINAL_PATH = PROJECT_ROOT / "results" / "earthquake_program" / "final.json"
EQ_STACK_PATH = MODELS_DIR / "earthquake_gear1_stack_v1.json"
EQ_E1_PATH = PROJECT_ROOT / "results" / "earthquake_program" / "gear1_e1.json"


def earthquake_stack_entry(path: Path = EQ_STACK_PATH, e1_path: Path = EQ_E1_PATH,
                           base_path: Path = EQ_OPERATIONAL_PATH) -> dict | None:
    """The served earthquake model since amendment E1: C0 + GEAR1 (a stack bound to C0's version)."""
    if not path.exists():
        return None
    from hazardpulse.earthquake import operational_forecast as eq_op

    stack = eq_op.load_stack(path, eq_op.load_artifact(base_path))
    e1 = _read_json(e1_path)
    bound = bool(e1) and (e1.get("coefficients_fitted_on_choose") or {}).get("S1") == {
        k: stack.meta["coefficients"][k] for k in ("a", "c", "b")}
    fin = ((e1.get("splits") or {}).get("final") or {}) if bound else {}
    dev = ((e1.get("splits") or {}).get("dev") or {}) if bound else {}
    s1 = (fin.get("models") or {}).get("S1") or {}
    g1 = (stack.meta.get("provenance") or {}).get("gear1") or {}
    return _make_entry(
        record_id=f"weights_hazardpulse_{stack.model_version}",
        name="HazardPulse Earthquake operational forecast S1 (C0 + GEAR1, served)",
        description=(
            "P(at least one ComCat M6.0+ epicentre in the 2-degree cell within the next 30 days) for every cell: "
            f"logit p = a + c logit(p_C0) + b log10 G, with p_C0 from {stack.base_model_version} and G the GEAR1 "
            "long-term rate (Bird et al. 2015; geodetic strain + smoothed seismicity; CC-BY-4.0) on the grid. "
            "Weights fitted on 2018-2020, decided on 2021-2022 against C0 recalibrated on the same years "
            "(docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10); 2023-2025 is a declared second read."
            + ("" if bound else " No evaluation is bound to this file.")),
        weights_path=path,
        benchmark={
            "benchmark_type": "operational forward test on every cell-time of the grid (not case-control)",
            "benchmark_bound_to_this_artifact": bound,
            "decided_on": "weekly issue times 2021-2022" if bound else None,
            "dev_ig_vs_recalibrated_C0": ((dev.get("paired") or {}).get("S1-S0") or {}).get("ig_per_target"),
            "test_window": "weekly issue times 2023-2025, 30-day windows (a declared second read)" if bound else None,
            "test_information_gain_per_target_nats": (s1.get("ig_per_target") or {}).get("value"),
            "test_information_gain_ci95": (s1.get("ig_per_target") or {}).get("ci95"),
            "test_auc": (s1.get("auc") or {}).get("value"), "test_auc_ci95": (s1.get("auc") or {}).get("ci95"),
            "test_bss_vs_uniform": (s1.get("bss") or {}).get("value"),
            "paired_this_minus_C0": ((fin.get("paired") or {}).get("S1-C0")) if bound else None,
        },
        framework="hazardpulse_eq_operational_stack_v1",
        input_schema={"base_model_version": stack.base_model_version, "gear1_sha256": g1.get("sha256"),
                      "gear1_source": g1.get("source")},
        output_schema={"outputs": ["m6_probability_30d_every_cell"], "domain": "[0, 1]",
                       "calibration": "the stack's logistic weights (fitted by Bernoulli likelihood on 2018-2020)",
                       "model_version": stack.model_version, "coefficients": stack.meta["coefficients"]},
        paper_url="https://github.com/coherence-energy-labs/hazardpulse",
    )


def earthquake_operational_entry(path: Path = EQ_OPERATIONAL_PATH,
                                 final_path: Path = EQ_PROGRAM_FINAL_PATH) -> dict | None:
    """The served earthquake probability (docs/EARTHQUAKE_FORECAST_PROGRAM.md) as a registry entry.

    Identity is the artifact's content-bound model_version; the benchmark is the program's
    read-once FINAL split (2023-2025 issue times, every cell of the grid), bound only when
    the final report's chosen candidate is the one this artifact serves.
    """
    if not path.exists():
        return None
    from hazardpulse.earthquake import operational_forecast as eq_op

    art = eq_op.load_artifact(path)
    candidate = art.gbt["name"] if art.gbt else art.spec.name
    final = _read_json(final_path)
    bound = bool(final) and final.get("chosen_on_choose") == candidate
    res = ((final.get("candidates") or {}).get(candidate) or {}) if bound else {}
    ref = ((final.get("candidates") or {}).get("A") or {}) if bound else {}
    if art.gbt:
        kind = ("boosted trees (LightGBM, scored in NumPy) over its long-term smoothed-seismicity "
                "rate, an ETAS-style short-term clustering rate and recent catalog counts")
        calibration = ("none applied: the trees' log-loss fit over every FIT cell-time "
                       "(negatives sampled 5% at weight 20)")
    elif art.spec.short is not None:
        kind = "long-term smoothed seismicity plus ETAS-style short-term clustering"
        calibration = "none applied: P = 1 - exp(-rate), rates fitted by Bernoulli likelihood"
    else:
        kind = "long-term smoothed seismicity"
        calibration = "none applied: P = 1 - exp(-rate), rates fitted by Bernoulli likelihood"

    def _v(d, k):
        return (d.get(k) or {}).get("value")

    def _ci(d, k):
        return (d.get(k) or {}).get("ci95")

    return _make_entry(
        record_id=f"weights_hazardpulse_{art.model_version}",
        name=(f"HazardPulse Earthquake operational forecast v1 (candidate {candidate}, "
              + ("base of the served S1)" if EQ_STACK_PATH.exists() else "served)")),
        description=(
            "P(at least one ComCat M6.0+ epicentre in the 2-degree cell within the next 30 days) for "
            f"every cell of the global grid, from {kind} of ComCat M5+ events since 1973 strictly "
            "before the issue time. Parameters fitted by Bernoulli likelihood on weekly issue times "
            "2005-2017; chosen on 2018-2020 by a pre-registered information-gain rule with a parsimony "
            "clause; scored once on 2023-2025 (docs/EARTHQUAKE_FORECAST_PROGRAM.md). The artifact holds "
            "the parameters and the frozen M5+ catalog; live events after its cutoff come from the "
            "scorer's USGS fetch."
            + ("" if bound else " No final-split score is bound to this file.")
        ),
        weights_path=path,
        benchmark={
            "benchmark_type": "operational forward test on every cell-time of the grid (not case-control)",
            "benchmark_bound_to_this_artifact": bound,
            "test_window": "weekly issue times 2023-01-02 .. 2025-12-01, 30-day windows, read once" if bound else None,
            "test_information_gain_per_target_nats": _v(res, "ig_per_target"),
            "test_information_gain_ci95": _ci(res, "ig_per_target"),
            "test_auc": _v(res, "auc"), "test_auc_ci95": _ci(res, "auc"),
            "test_brier": _v(res, "brier"), "test_bss_vs_uniform": _v(res, "bss"), "test_bss_ci95": _ci(res, "bss"),
            "test_auc_active_cells": _v(res, "auc_active_cells"),
            "test_calibration_ratio_sum_p_over_sum_y": _v(res, "calib_ratio"),
            "reference_A_long_term_ig": _v(ref, "ig_per_target") if candidate != "A" else None,
            "paired_this_minus_A": ((final.get("paired") or {}).get(f"{candidate}-A") if bound and candidate != "A" else None),
            "n_test_positive_cell_windows": res.get("n_positive"),
        },
        framework="hazardpulse_eq_operational_v1",
        input_schema={
            "inputs": "ComCat events M>=5.0 with origin time strictly before the issue time (1973 onward)",
            "frozen_catalog_events": len(art.frozen),
            "frozen_catalog_cutoff": art.meta["frozen_catalog"]["cutoff"],
            "feature_names_path": "src/hazardpulse/earthquake/operational_forecast.py",
        },
        output_schema={
            "outputs": ["m6_probability_30d_every_cell"],
            "domain": "[0, 1]",
            "calibration": calibration,
            "model_version": art.model_version,
            "rate_model": art.spec.to_dict(),
            "trees": ({"n_trees": art.gbt["payload"]["n_trees"], "features": art.gbt["feature_names"]}
                      if art.gbt else None),
        },
        paper_url="https://github.com/coherence-energy-labs/hazardpulse",
    )


TORNADO_V3_FILES = {
    "tornado_v3_w.json": "served: 60-min probability (inputs include the live NWS warning state)",
    "tornado_v3.json": "served fallback when the NWS warnings feed is down",
    "tornado_v3_w_30.json": "served product: 30-min probability",
    "tornado_v3_w_90.json": "served product: 90-min probability",
    "tornado_v3_w_ef2.json": "served product: EF2+ tornado within 60 min",
}


def tornado_v3_entries(models_dir: Path = MODELS_DIR) -> list[dict]:
    """The v3 tornado suite (docs/TORNADO_MODEL_PROGRAM.md), one entry per served payload. Identity,
    inputs and the final-season numbers come from the payload itself (its provenance is written by
    export_v3_payload.py, which refuses to export a payload that does not reproduce its final run)."""
    from hazardpulse.tornado import lgbm_payload as lp

    out = []
    for fname, role in TORNADO_V3_FILES.items():
        path = models_dir / fname
        if not path.exists():
            continue
        payload = lp.load(path)
        version = lp.model_version(payload)
        prov = payload.get("provenance") or {}
        f = prov.get("final_2025") or {}
        out.append(_make_entry(
            record_id=f"weights_hazardpulse_{version}",
            name=f"HazardPulse Tornado v3 ({role.split(':')[0]})",
            description=(
                f"{role}. {prov.get('forecasts', '')}. Event: {prov.get('event', '')}. LightGBM "
                f"({payload['n_trees']} trees) exported to a NumPy payload; Platt on leave-one-year-out "
                f"scores; trained {prov.get('trained', '')}; scored once on 2025."),
            weights_path=path,
            benchmark={
                "test_window": "every ProbSevere storm observation of 2025 (read once by the final pipeline)",
                "test_auc": f.get("auc"), "test_auc_ci": f.get("auc_ci"), "test_bss": f.get("bss"),
                "test_pr_auc": f.get("pr_auc"), "n_test_samples": f.get("n"), "n_test_positive": f.get("pos"),
                "program": prov.get("program"),
            },
            framework=payload["schema"],
            input_schema={"n_features": len(payload["feature_names"]), "blocks": prov.get("blocks"),
                          "feature_names_path": f"results/models/{fname} (feature_names)"},
            output_schema={"outputs": [prov.get("label")], "domain": "[0, 1]",
                           "calibration": "Platt on leave-one-year-out scores; Venn-Abers band"
                           if payload.get("interval") else "Platt on leave-one-year-out scores",
                           "model_version": version},
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ))
    return out


VOLATILE_ENTRY_KEYS = ("release_utc", "blake3")      # differ run to run / by installed library


def normalized(payload: dict) -> list[dict]:
    """The registry's content without its timestamps and optional hashes: what the fixed-point test
    compares and what decides whether a regeneration rewrites the file."""
    return [{k: v for k, v in e.items() if k not in VOLATILE_ENTRY_KEYS} for e in payload.get("entries") or []]


def build_entries() -> list[dict]:
    if not MODELS_DIR.exists():
        raise SystemExit(f"{MODELS_DIR} does not exist")

    entries: list[dict] = []

    # ----- Tornado v3 suite (served) -----
    entries.extend(tornado_v3_entries())

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
            name="HazardPulse Tornado GBT v1 (definitive)" + (" -- SUPERSEDED by v3" if any(
                (MODELS_DIR / f).exists() for f in TORNADO_V3_FILES) else ""),
            description=(
                ("Superseded by the v3 suite (served); kept as the legacy path and the v3 benchmark. "
                 if any((MODELS_DIR / f).exists() for f in TORNADO_V3_FILES) else "") +
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

    # ----- Earthquake operational forecast (served) -----
    eq_op_entry = earthquake_operational_entry()
    eq_stack_entry = earthquake_stack_entry()
    eq_live = {"replay": "dist/data/replay/eq_fcst_*.json (the full 11,700-cell grid)",
               "ledger": "dist/data/earthquake-ledger.jsonl",
               "prospective": "results/earthquake_prospective/"}
    if eq_op_entry is not None:
        eq_op_entry["lineage"] = {
            "role": "base of the served S1" if eq_stack_entry is not None else "served",
            "program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md", "amendment": "sections 1-9",
            "prereg_tag": None,
            "registration": "sections 1-8 frozen before the first fit; their SHA-256 is recorded in section 9",
            "selection_and_evaluation": [{"path": p, "sha256": _file_sha256(PROJECT_ROOT / p)} for p in (
                "results/earthquake_program/fit_ab.json", "results/earthquake_program/fit_c.json",
                "results/earthquake_program/choose.json", "results/earthquake_program/dev.json",
                "results/earthquake_program/final.json", "results/earthquake_program/artifact.json")],
            "live_records": eq_live}
        entries.append(eq_op_entry)
    if eq_stack_entry is not None:
        eq_stack_entry["lineage"] = {
            "role": "served", "program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md", "amendment": "section 10 (E1)",
            "prereg_tag": "prereg-earthquake-gear1",
            "selection_and_evaluation": [{"path": p, "sha256": _file_sha256(PROJECT_ROOT / p)} for p in (
                "results/earthquake_program/gear1_cells.json", "results/earthquake_program/gear1_e1.json",
                "results/earthquake_program/stack_artifact.json")],
            "external_data": {"GEAR1": "Zenodo 7086053 GEAR1.dat (CC-BY-4.0), SHA-256 in gear1_cells.json"},
            "live_records": eq_live}
        entries.append(eq_stack_entry)
    # ----- Hurricane RI shadows (ours, in prospective verification) -----
    entries.extend(hurricane_shadow_entries())

    # ----- Earthquake GBT v1 (not served) -----
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
            name="HazardPulse Earthquake GBT v1 (plus_cft)" + (
                " -- NOT SERVED" if eq_op_entry is not None else ""),
            description=(
                ("Not served: the site's earthquake probability is the operational forecast entry "
                 "(docs/EARTHQUAKE_FORECAST_PROGRAM.md). " if eq_op_entry is not None else "") +
                "Gradient-boosted tree ensemble for global M6+ earthquake "
                "probability per 2-degree grid cell, 30-day forward window. "
                "73 features = Block S (61 seismicity) + Block C (12 "
                "coherence field theory). Trained on declustered USGS "
                "catalog 2005-2017, validated 2018-2019, tested 2020-2024 "
                "(plus_cft variant of trained_models_v3). The benchmark below "
                "is a CASE-CONTROL NOWCAST (positives scored at the mainshock "
                "setting against controls); as an operational forecaster on a "
                "gridded forward test this model scored AUC 0.509 -- no skill "
                "(docs/earthquake_nowcast_validation.md)."
            ),
            weights_path=eq_path,
            benchmark={
                "benchmark_type": "case-control nowcast, not an operational forecast",
                "test_auc_global_baseline": gb.get("auc"),
                "test_auc_regional_ensemble": gc.get("auc"),
                "test_brier": gb.get("brier"),
                "test_bss": gb.get("bss"),
                "test_window": "USGS M6+ events 2020-2024",
                "operational_auc_pooled": 0.509,
                "operational_source": "docs/earthquake_nowcast_validation.md (backtest_operational_grid.py)",
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

    # ----- Hurricane RI (the served, pinned model) -----
    # Everything below is read from the served model file and the held-out
    # evaluation report. The old entry described a model "trained in-process
    # per scoring run" and quoted AUC 0.967 -- measured on v8.1's own label,
    # which spanned 12 h, not the 24 h it was named for.
    hu_model_path = MODELS_DIR / "hurricane_ri_v8_2.json"
    hu_eval_path = PROJECT_ROOT / "results" / "calibration" / "hurricane_ri_evaluation.json"
    if hu_model_path.exists():
        hu_model = _read_json(hu_model_path)
        hu_eval = _read_json(hu_eval_path)
        served = (hu_eval.get("results") or {}).get("C_v8_2_heldout_newton", {})
        ci = served.get("ci95") or {}
        cal = hu_model.get("calibration") or {}
        entries.append(_make_entry(
            record_id=f"weights_hazardpulse_{hu_model.get('model_version', 'hurricane_ri_v8_2')}",
            name="HazardPulse Hurricane RI v8.2 (ensemble, pinned)",
            description=(
                "Rapid-intensification ensemble (histogram-GBT depth 3 + depth 4 + "
                "L2 logistic + bagged logistic) predicting a >= 30 kt wind increase "
                "in the next 24 h, on true synoptic timestamps. Served from a pinned "
                "file bound to its training-data hash; calibrated by a converged "
                "Newton logistic fit on seasons the members never trained on. "
                f"Held-out test: {hu_eval.get('n_test_storms')} storms first seen "
                f"2022-2024, {hu_eval.get('n_test_events')} RI events. Since 2026-10-02 it "
                "serves the JTWC basins and any NHC cycle without NOAA's SHIPS text; NHC-basin "
                "cycles with it are served the NOAA aid entry below."
            ),
            weights_path=hu_model_path,
            benchmark={
                "test_auc": served.get("auc"),
                "test_auc_ci95": ci.get("auc"),
                "test_brier": served.get("brier"),
                "test_bss_vs_climatology": served.get("bss_vs_climatology"),
                "test_bss_ci95": ci.get("bss_vs_climatology"),
                "calibration_slope": served.get("calibration_slope"),
                "persistence_auc": ((hu_eval.get("results") or {}).get("P_persistence_dv24") or {}).get("auc"),
                "n_test_rows": hu_eval.get("n_test_rows"),
                "test_window": "storms first seen 2022-2024 (bootstrap by storm)",
            },
            framework="hazardpulse_ri_ensemble_v8_2",
            input_schema={
                "n_features_select": len(hu_model.get("selected_features") or []),
                "feature_names_path": "src/hazardpulse/hurricane/operational_ri.py",
            },
            output_schema={
                "outputs": ["ri_probability_24h"],
                "domain": "[0, 1]",
                "calibration": f"{cal.get('method', '?')} fitted on held-out seasons",
            },
            paper_url="https://github.com/coherence-energy-labs/hazardpulse",
        ))

    # ----- Hurricane RI, NHC basins: NOAA's aids as the pre-registered program chose them -----
    stack_entry = hurricane_stack_entry()
    if stack_entry is not None:
        entries.append(stack_entry)

    return entries


def build_registry() -> dict:
    entries = build_entries()
    return {
        "schema_version": 1,
        "publisher": "HazardPulse / Coherence Energy Labs",
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "n_entries": len(entries),
        "entries": entries,
        # Alias used by the worker API contract (/api/v1/registry/models)
        # to keep the field name "models" for backwards compat.
        "models": entries,
    }


def main() -> int:
    payload = build_registry()
    entries = payload["entries"]
    old = json.loads(WORKER_OUT_PATH.read_text(encoding="utf-8")) if WORKER_OUT_PATH.exists() else {}
    if old and normalized(old) == normalized(payload):
        print(f"{WORKER_OUT_PATH} unchanged ({len(entries)} entries)")
        return 0
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
