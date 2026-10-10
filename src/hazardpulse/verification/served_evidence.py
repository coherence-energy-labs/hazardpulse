"""What each SERVED model has been measured to do -- read from the files its programme wrote.

Every page that states a served model's skill reads it here, so a number on the site is the
number in a results file, for the model that is actually serving. Each hazard's evidence is
returned only when its results file is BOUND to the exact artifact being served:

* tornado     the served payload (``results/models/tornado_v3_w.json``) carries its own
              read-once 2025 scores in its provenance; the lab's ``final_<run>.json`` in
              ``results/lab_avail`` must hold the identical AUC (the same float) before the lab's
              comparison, NWS and stress files are quoted beside it
* earthquake  ``results/earthquake_program/artifact.json`` names the served file's model_version
* hurricane   ``results/calibration/hurricane_ri_stack_final.json`` names the served artifact's
              model_version

Anything unbound is ``None`` (or the comparison is left out) and the page says so; nothing here
falls back to an older model's number. Until 2026-10 the site typed its model numbers by hand in
five places, and each copy had drifted to a different superseded model (0.894, 0.940, 0.971,
0.907 / 0.799, 0.77, 0.938).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

TORNADO_SERVED = "results/models/tornado_v3_w.json"
TORNADO_FALLBACK = "results/models/tornado_v3.json"
TORNADO_PRODUCTS = {
    "p30": "results/models/tornado_v3_w_30.json",
    "p90": "results/models/tornado_v3_w_90.json",
    "p_ef2": "results/models/tornado_v3_w_ef2.json",
}
TORNADO_RESULTS = "results/lab_avail"
TORNADO_PROGRAM = "docs/TORNADO_MODEL_PROGRAM.md"
# label each product run was scored on (tornado_lab final_product)
_PRODUCT_LABEL = {"p30": "storm_30", "p90": "storm_90", "p_ef2": "storm_60_ef2"}
# NOAA's ProbTor as the comparator. The tie-broken variant is the adversary's fix (amendment 8):
# raw ProbTor is integer percent, so its ties understate its AUC; the page quotes the comparison
# LEAST favourable to us among the variants that were run.
_PROBTOR_VARIANTS = ("probtor_tiebroken", "probtor_raw")
# the development-year (2024) run of each final recipe (tornado_lab experiments 09_dev)
_DEV_RUN = {"v3_plus_W": "dev_plus_W", "v3_primary": "dev_primary"}
_INPUT_FAMILIES = (("p_", "ProbSevere storm attributes"), ("e_", "storm-track trends"),
                   ("h80_", "HRRR environment fields"), ("c80_", "coherence-field features"),
                   ("w_", "NWS tornado-warning state"))
# What was measured and NOT served -- each line read from the run that measured it. (validation 2023;
# the coherence runs predate amendment 8's HRRR timing, which both of their arms share.)
_TORNADO_NOT_USED = (
    {"what": "the HRRR environment at the storm, timed as it is available live",
     "kind": "val_auc_difference", "with": "b_PE_H80", "without": "b_PE", "dir": TORNADO_RESULTS},
    {"what": "the coherence field (80 km) on top of those HRRR fields",
     "kind": "paired", "file": "results/lab/compare_b_PE_H80_C80_vs_b_PE_H80_val_storm_60.json",
     "note": "measured before amendment 8 corrected the HRRR timing, which both arms shared"},
    {"what": "the coherence PDE solution against a Gaussian-smoothing control (9 km)",
     "kind": "paired", "file": "results/lab/compare_b_all_C9pde_vs_b_all_C9gauss_val_storm_60.json",
     "note": "measured before amendment 8 corrected the HRRR timing, which both arms shared"},
)

EARTHQUAKE_SERVED = "results/models/earthquake_operational_v1.json"
EARTHQUAKE_ARTIFACT_RECORD = "results/earthquake_program/artifact.json"
EARTHQUAKE_FINAL = "results/earthquake_program/final.json"
EARTHQUAKE_PROGRAM = "docs/EARTHQUAKE_FORECAST_PROGRAM.md"
EARTHQUAKE_CANDIDATES = {
    "A": "long-term smoothed seismicity (the standard reference forecast)",
    "B": "smoothed seismicity + ETAS-style aftershock clustering",
    "C0": "gradient-boosted trees on causal seismicity features, with the smoothed-seismicity and aftershock-clustering rates as inputs",
    "C1": "C0 + the 61 coherence block-S features",
    "D": "the previous served model",
    "S1": "C0 plus GEAR1's long-term rate (geodetic strain and smoothed seismicity, Bird et al. 2015) as one extra term",
}
# Amendment E1 (section 10): the served S1 = the C0 artifact + a stack bound to its model_version
EARTHQUAKE_STACK = "results/models/earthquake_gear1_stack_v1.json"
EARTHQUAKE_STACK_RECORD = "results/earthquake_program/stack_artifact.json"
EARTHQUAKE_E1 = "results/earthquake_program/gear1_e1.json"

HURRICANE_SERVED = "results/models/hurricane_ri_stack_v1.json"
HURRICANE_FINAL = "results/calibration/hurricane_ri_stack_final.json"
HURRICANE_ADVERSARY = "results/calibration/hurricane_ri_stack_adversary.json"
# v8.2, the model served outside the NHC basins until amendment 15, and its own temporal hold-out
# (scripts/evaluate_hurricane_ri.py)
HURRICANE_V82_EVALUATION = "results/calibration/hurricane_ri_evaluation.json"
# what that hold-out was a test OF: its basins, its inputs, the served artifact's calibration, and the
# served model on West Pacific cycles with a single JTWC warning's inputs (scripts/hurricane_v82_test_composition.py)
HURRICANE_V82_COMPOSITION = "results/calibration/hurricane_ri_v8_2_test_composition.json"
# v8.2's artifact: published until amendment 15; since then J1's base and comparator, and the model the NHC
# program's claims and same-cycle scores name
HURRICANE_V82_SERVED = "results/models/hurricane_ri_v8_2.json"
# the live scorer: which HazardPulse model it serves outside the NHC basins (SERVED_MODEL_VERSION) and which model
# it feeds J1 (J1_BASE_MODEL_VERSION), read from its source. A model that replaced v8.2 by a registered rule
# carries its decision in results/calibration/<version>.json and its test's composition in
# results/calibration/<version>_test_composition.json (amendment 15: hurricane_ri_v8_3).
HURRICANE_SCORER = "scripts/fetch_and_score.py"
HURRICANE_PROGRAM = "docs/HURRICANE_RI_PROGRAM.md"
# our own RI models (docs/HURRICANE_RI_V9_PROGRAM.md): the prospective test's running record, the one file that
# names which of them the site shows, and v9.1 (amendment 1), whose claim would switch what is published
HURRICANE_V9_PROGRAM = "docs/HURRICANE_RI_V9_PROGRAM.md"
HURRICANE_PROSPECTIVE = "results/hurricane_prospective/v9_shadow.json"
HURRICANE_SHOWN = "results/hurricane_prospective/shown_model.json"
HURRICANE_V9_SERVED = "results/models/hurricane_ri_v9.json"
HURRICANE_CANDIDATES = {
    "A": "NOAA DTOPS (the NHC's deterministic-to-probabilistic RI guidance)",
    "B": "NOAA SHIPS-RII",
    "C": "NOAA RIOC (the RI consensus)",
}
_HURRICANE_CLAIM_NAMES = {
    "better_than_SHIPS_RII_B": "NOAA SHIPS-RII",
    "better_than_RIOC_C": "NOAA RIOC consensus",
    "better_than_v8_2": "our previous model",          # + the served v8.2 artifact's own label
}


def version_label(model_version: Any) -> str | None:
    """A model's short label from its own version: ``hurricane_ri_v8_2`` -> ``v8.2``,
    ``hurricane_ri_v10_2-3e9c1b766454`` -> ``v10.2``."""
    m = re.search(r"_v(\d+)_(\d+)(?:-[0-9a-f]+)?$", str(model_version or ""))
    return f"v{m.group(1)}.{m.group(2)}" if m else None


def entrant_label(entrant: Any) -> str:
    """The prospective scorer's entrant key as a label: ``v10_1`` -> ``v10.1``."""
    m = re.fullmatch(r"v(\d+)_(\d+)", str(entrant or ""))
    if not m:
        raise EvidenceError(f"{entrant!r} is not an entrant key of the prospective test")
    return f"v{m.group(1)}.{m.group(2)}"


class EvidenceError(ValueError):
    """A results file exists but contradicts the artifact it claims to describe."""


def _read(root: Path, rel: str) -> Any:
    path = root / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ci(x: Any) -> list[float] | None:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return None
    lo, hi = _finite(x[0]), _finite(x[1])
    return None if lo is None or hi is None else [lo, hi]


# ---------------------------------------------------------------------------
# Tornado
# ---------------------------------------------------------------------------

def _tornado_payload_summary(root: Path, rel: str) -> dict | None:
    from hazardpulse.tornado import lgbm_payload as lp

    path = root / rel
    if not path.exists():
        return None
    payload = lp.load(path)
    prov = payload.get("provenance") or {}
    f25 = prov.get("final_2025") or {}
    if _finite(f25.get("auc")) is None:
        raise EvidenceError(f"{rel}: the payload's provenance carries no 2025 AUC")
    names = [str(n) for n in payload.get("feature_names") or []]
    families = {label: sum(n.startswith(prefix) for n in names) for prefix, label in _INPUT_FAMILIES}
    return {
        "file": rel,
        "model_version": lp.model_version(payload),
        "input_families": {k: v for k, v in families.items() if v},
        "final_run": prov.get("final_run"),
        "label": prov.get("label"),
        "event": prov.get("event"),
        "forecasts": prov.get("forecasts"),
        "trained": prov.get("trained"),
        "n_inputs": len(payload.get("feature_names") or []),
        "inputs_use_nws": any(str(n).startswith("w_") for n in payload.get("feature_names") or []),
        "inputs_use_hrrr": any(str(n).startswith("h80_") for n in payload.get("feature_names") or []),
        "n_trees": payload.get("n_trees"),
        "has_band": bool(payload.get("interval")),
        "test": {
            "period": "2025 (every ProbSevere storm observation; read once)", "when": "2025",
            "auc": _finite(f25["auc"]), "auc_ci": _ci(f25.get("auc_ci")),
            "bss": _finite(f25.get("bss")), "pr_auc": _finite(f25.get("pr_auc")),
            "n": int(f25.get("n") or 0), "pos": int(f25.get("pos") or 0),
        },
    }


def _bound_lab_final(root: Path, summary: dict) -> dict | None:
    """The lab's final run for this payload -- only if it holds the payload's own AUC."""
    run = summary.get("final_run")
    if not run:
        return None
    final = _read(root, f"{TORNADO_RESULTS}/final_{run}.json")
    if final is None:
        return None
    lab_auc = _finite((final.get("final_2025") or {}).get("auc"))
    if lab_auc != summary["test"]["auc"]:
        raise EvidenceError(
            f"{TORNADO_RESULTS}/final_{run}.json has 2025 AUC {lab_auc!r} but the served payload "
            f"{summary['file']} says {summary['test']['auc']!r}: the results are not this model's")
    return final


def _compare(root: Path, b: str, a: str, label: str) -> dict | None:
    d = _read(root, f"{TORNADO_RESULTS}/compare_{b}_vs_{a}_final_{label}.json")
    if not d:
        return None
    return {
        "against": a, "n_days": d.get("n_days"), "n_boot": d.get("n_boot"),
        "delta_auc": _finite(d.get("delta_auc")), "delta_auc_ci": _ci(d.get("delta_auc_ci")),
        "delta_brier": _finite(d.get("delta_brier")), "delta_brier_ci": _ci(d.get("delta_brier_ci")),
    }


def _nws_bar(root: Path, run: str, label: str) -> dict | None:
    """The model at the NWS tornado warnings' own false-alarm rate (tornado_lab nws_bar)."""
    nws = _read(root, f"{TORNADO_RESULTS}/nws_bar_{run}_final_{label}.json")
    if not nws:
        return None
    return {
        "nws_pod": _finite(nws.get("nws_pod")), "nws_pofd": _finite(nws.get("nws_pofd")),
        "model_pod": _finite(nws.get("model_pod_at_nws_pofd")),
        "delta_pod": _finite(nws.get("delta_pod")), "delta_pod_ci": _ci(nws.get("delta_pod_ci")),
        "threshold": _finite(nws.get("threshold")), "positives": nws.get("positives"),
    }


def _vs_probtor(root: Path, run: str, label: str) -> dict | None:
    found = [c for a in _PROBTOR_VARIANTS if (c := _compare(root, run, a, label)) is not None
             and c["delta_auc"] is not None]
    if not found:
        return None
    worst = min(found, key=lambda c: c["delta_auc"])
    worst["variants_compared"] = [c["against"] for c in found]
    return worst


def _reliability_table(root: Path, run: str) -> dict | None:
    d = _read(root, f"{TORNADO_RESULTS}/reliability_{run}_final.json")
    if not d or not d.get("bins"):
        return None
    return d


def reliability_bin(table: dict | None, p: float) -> dict | None:
    """The 2025 final-test bin a live probability falls in: how often storms scored like this
    one were followed by the forecast event. ``None`` when no fine table is bound or the bin is
    empty (a rate is never invented for a range the test did not populate)."""
    if not table or not table.get("bins") or not math.isfinite(p):
        return None
    bins = table["bins"]
    for i, b in enumerate(bins):
        last = i == len(bins) - 1
        if b["lo"] <= p < b["hi"] or (last and b["lo"] <= p <= b["hi"]):
            return b if b.get("n", 0) > 0 else None
    return None


def reliability_for(ev: dict | None, model: str) -> dict | None:
    """The table of the model that scored a storm: ``v3_w`` (served) or ``v3`` (no-NWS fallback)."""
    if not ev:
        return None
    if model == "v3_w":
        return ev.get("reliability")
    return (ev.get("fallback") or {}).get("reliability")


def tornado_evidence(root: Path = ROOT) -> dict | None:
    main = _tornado_payload_summary(root, TORNADO_SERVED)
    if main is None:
        return None
    ev: dict[str, Any] = {"hazard": "tornado", "program": TORNADO_PROGRAM, **main}
    final = _bound_lab_final(root, main)
    run, label = main["final_run"], main.get("label") or "storm_60"
    ev["results_bound"] = final is not None
    ev["vs_probtor"] = _vs_probtor(root, run, label) if final else None
    ev["vs_v2"] = _compare(root, run, "v2", label) if final else None
    ev["vs_nws_warnings"] = _nws_bar(root, run, label) if final else None
    stress = _read(root, f"{TORNADO_RESULTS}/stress_{run}_final_{label}.json") if final else None
    ev["stress"] = None
    if stress:
        rows = []
        for name, s in stress.items():
            if not isinstance(s, dict) or run not in s:
                continue
            ref = s.get("probtor_platt") or {}
            rows.append({"stratum": name, "n": s.get("n"), "pos": s.get("pos"),
                         "auc": _finite(s[run].get("auc")), "auc_ci": _ci(s[run].get("ci")),
                         "probtor_auc": _finite(ref.get("auc")), "probtor_ci": _ci(ref.get("ci"))})
        if rows:
            beats = [r for r in rows if r["auc_ci"] and r["probtor_ci"] and r["auc_ci"][0] > r["probtor_ci"][1]]
            ev["stress"] = {"strata": rows, "min_auc": min(r["auc"] for r in rows if r["auc"] is not None),
                            "n_strata": len(rows), "n_clear_of_probtor": len(beats)}
    ev["reliability"] = _reliability_table(root, run) if final else None
    dev = (_read(root, f"{TORNADO_RESULTS}/{_DEV_RUN[run]}.json") or {}).get("dev") if final and run in _DEV_RUN else None
    ev["dev"] = None if not dev else {"period": "2024 (development year, before the refit)", "when": "2024",
                                      "auc": _finite(dev.get("auc")), "auc_ci": _ci(dev.get("auc_ci")),
                                      "bss": _finite(dev.get("bss")), "n": dev.get("n"), "pos": dev.get("pos")}
    base = ((_read(root, f"{TORNADO_RESULTS}/baselines_{label}.json") or {}).get("final") or {}) if final else {}
    ev["probtor_final"] = {k: {"auc": _finite(base[k].get("auc")), "auc_ci": _ci(base[k].get("auc_ci"))}
                           for k in ("probtor_raw", "probtor_tiebroken") if k in base} or None
    not_used = []
    for t in _TORNADO_NOT_USED:
        if t["kind"] == "val_auc_difference":
            a = _read(root, f"{t['dir']}/{t['with']}.json")
            b = _read(root, f"{t['dir']}/{t['without']}.json")
            if a and b:
                not_used.append({"what": t["what"], "delta_auc": _finite(a["val"]["auc"]) - _finite(b["val"]["auc"]),
                                 "delta_auc_ci": None, "paired": False,
                                 "source": [f"{t['dir']}/{t['with']}.json", f"{t['dir']}/{t['without']}.json"]})
        else:
            d = _read(root, t["file"])
            if d:
                not_used.append({"what": t["what"], "delta_auc": _finite(d.get("delta_auc")),
                                 "delta_auc_ci": _ci(d.get("delta_auc_ci")), "paired": True,
                                 "note": t.get("note"), "source": [t["file"]]})
    ev["tested_not_served"] = not_used
    fb = _tornado_payload_summary(root, TORNADO_FALLBACK)
    if fb is not None:
        fb_bound = _bound_lab_final(root, fb) is not None
        fb["reliability"] = _reliability_table(root, fb["final_run"]) if fb_bound else None
        # the model WITHOUT the warning input against the warnings: what it knows on its own
        fb["vs_nws_warnings"] = _nws_bar(root, fb["final_run"], fb.get("label") or "storm_60") if fb_bound else None
        fb["vs_probtor"] = _vs_probtor(root, fb["final_run"], fb.get("label") or "storm_60") if fb_bound else None
    ev["fallback"] = fb
    products = {}
    for key, rel in TORNADO_PRODUCTS.items():
        p = _tornado_payload_summary(root, rel)
        if p is None:
            continue
        pfinal = _bound_lab_final(root, p)
        p["vs_probtor"] = _vs_probtor(root, p["final_run"], _PRODUCT_LABEL[key]) if pfinal else None
        products[key] = p
    ev["products"] = products
    ev["trained_years"] = _years_of(main.get("trained"))
    ev["format_change"] = _tornado_format_change(root, main)
    return ev


TORNADO_T2 = "results/tornado_program/t2_2026.json"


def _years_of(period: Any) -> str | None:
    """``2020-10-15..2024-12-31, fixed rounds ...`` -> ``2020-2024``: the years a stated period spans."""
    m = re.match(r"\s*(\d{4})\S*\s*\.\.\s*(\d{4})", str(period or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _tornado_format_change(root: Path, main: dict) -> dict | None:
    """NOAA's 2025 ProbSevere format change and what it cost the served model (tornado program amendment 10,
    ledger T2): the date the guard records, the served model on the new format all through 2026-01..09 against
    its own read-once 2025 score, and its mean forecast against the base rate. Bound only when the run's served
    candidate is this payload (its full content digest begins with the served version's); a run about an earlier
    payload is not this model's result and is left out (None) -- never raised, because the live tornado scorer
    reads this evidence and a later payload must not cost it the rest."""
    from hazardpulse.tornado import input_guard

    t2 = _read(root, TORNADO_T2)
    if t2 is None:
        return None
    cand = (t2.get("candidates") or {}).get("a_served/plus_W") or {}
    digest = str(main["model_version"]).rsplit("-", 1)[-1]
    if not str(cand.get("model_sha256") or "").startswith(digest):
        return None
    n, pos = int(t2.get("n") or 0), int(t2.get("pos") or 0)
    base_rate = pos / n if n else None
    dec = t2.get("decision") or {}
    drop = (t2.get("paired_vs_a") or {}).get("d_drop2/plus_W") or {}
    period = list(t2.get("period") or [])
    return {
        "file": TORNADO_T2, "program": TORNADO_PROGRAM, "amendment": t2.get("amendment"),
        "date": input_guard.FORMAT_CHANGE,
        "period": ([f"{p[:4]}-{p[4:6]}-{p[6:]}" for p in period] if len(period) == 2 else None),
        "n": n, "pos": pos, "auc": _finite(cand.get("auc")), "auc_2025": main["test"]["auc"],
        "mean_over_base": (_finite(cand.get("mean_forecast")) / base_rate
                           if base_rate and _finite(cand.get("mean_forecast")) is not None else None),
        "served_stays": dec.get("served") == "a_served", "qualified": list(dec.get("qualified") or []),
        "without_inputs_d_auc": _finite(drop.get("delta_auc")), "without_inputs_d_auc_ci": _ci(drop.get("delta_auc_ci")),
    }


# ---------------------------------------------------------------------------
# Earthquake
# ---------------------------------------------------------------------------

def _earthquake_stack_evidence(root: Path, base_version: str, contract: dict, payload: dict, metric) -> dict:
    """The served S1 (C0 + GEAR1): bound when the stack's build record names this file and the
    served C0, and the evaluation's coefficients and GEAR1 hash are the stack's own."""
    from hazardpulse.earthquake import operational_forecast as eq

    path = root / EARTHQUAKE_STACK
    version = eq.stack_model_version(path)
    rec = _read(root, EARTHQUAKE_STACK_RECORD) or {}
    if rec.get("model_version") != version or rec.get("base_model_version") != base_version:
        raise EvidenceError(f"{EARTHQUAKE_STACK_RECORD} names {rec.get('model_version')!r} on "
                            f"{rec.get('base_model_version')!r}; served is {version!r} on {base_version!r}")
    stack = json.loads(path.read_text(encoding="utf-8"))
    if stack.get("base_model_version") != base_version:
        raise EvidenceError(f"{EARTHQUAKE_STACK} stacks on {stack.get('base_model_version')!r}, not {base_version!r}")
    e1 = _read(root, EARTHQUAKE_E1)
    if e1 is None:
        raise EvidenceError(f"{EARTHQUAKE_E1} missing for the served stack")
    if (e1.get("coefficients_fitted_on_choose") or {}).get("S1") != {
            k: stack["coefficients"][k] for k in ("a", "c", "b")} \
            or e1.get("gear1_sha256") != ((stack.get("provenance") or {}).get("gear1") or {}).get("sha256"):
        raise EvidenceError(f"{EARTHQUAKE_E1} does not describe the served stack's coefficients and GEAR1 map")
    fin, dev, choose = e1["splits"]["final"], e1["splits"]["dev"], e1["splits"].get("choose") or {}
    cand = fin["models"]["S1"]

    def pair(split, key, name):
        p = split["paired"].get(key)
        return None if p is None else {"name": name, **{k: {"diff": _finite(v.get("diff")), "ci": _ci(v.get("ci95"))}
                                                         for k, v in p.items()}}
    n_issue, n_pos = int(fin["n_issue_times"]), int(fin["n_positive"])
    return {
        "hazard": "earthquake", "program": EARTHQUAKE_PROGRAM, "file": EARTHQUAKE_STACK,
        "model_version": version, "base_model_version": base_version, "candidate": "S1",
        "candidate_name": EARTHQUAKE_CANDIDATES["S1"],
        "event": contract.get("event"), "horizon_days": contract.get("horizon_days"),
        "target_magnitude_min": contract.get("target_magnitude_min"),
        "test": {
            "period": f"issue times {fin['first_issue'][:7]} .. {fin['last_issue'][:7]} (a second read)",
            "when": f"{fin['first_issue'][:4]}-{fin['last_issue'][:4]}", "second_read": True,
            "n_issue_times": n_issue, "n_cell_times": fin.get("n_cell_times"), "n_positive": n_pos,
            "block": "month",
            "ig_per_target": metric(cand, "ig_per_target"), "auc": metric(cand, "auc"),
            "bss": metric(cand, "bss"), "auc_active_cells": metric(cand, "auc_active_cells"),
            "calib_ratio": metric(cand, "calib_ratio"),
            "n_cells": (int(fin["n_cell_times"]) // n_issue) if n_issue else None,
            "share_outside_active_cells": 1.0 - int(fin["positives_in_active_cells"]) / n_pos if n_pos else None,
        },
        "gear1": {"decided_on": f"{dev['first_issue'][:4]}-{dev['last_issue'][:4]}",
                  "fitted_on": (f"{choose['first_issue'][:4]}-{choose['last_issue'][:4]}"
                                if choose.get("first_issue") and choose.get("last_issue") else None),
                  "dev_vs_recalibrated": pair(dev, "S1-S0", "C0 recalibrated on the same years"),
                  "dev_vs_C0": pair(dev, "S1-C0", "C0"),
                  "global_rate_per_year": _finite(((stack.get("provenance") or {}).get("gear1") or {}).get("global_total_per_year"))},
        "n_trees": payload.get("n_trees"),
        "n_inputs": len(payload.get("feature_names") or []) + 1,
        "replaced_ig_per_target": metric(fin["models"]["C0"], "ig_per_target"),
        "replaced_name": "C0 alone",
        "vs": {o: v for o, v in (("A", pair(fin, "S1-A", EARTHQUAKE_CANDIDATES["A"])),
                                 ("B", pair(fin, "S1-B", EARTHQUAKE_CANDIDATES["B"])),
                                 ("C0", pair(fin, "S1-C0", "C0")),
                                 ("D", pair(fin, "S1-D", EARTHQUAKE_CANDIDATES["D"]))) if v},
        "grid": contract.get("grid"),
        "input_types": _earthquake_input_types(root, version, base_version, dev),
    }


EARTHQUAKE_E3 = "results/earthquake_program/e3_types.json"


def _earthquake_input_types(root: Path, version: str, base_version: str, e1_dev: dict) -> dict | None:
    """What the served map's inputs are, by ComCat event type (amendment E3, section 11): the frozen M5+ events
    that are not earthquakes, the cell holding the most of them with and without them, and the registered
    decision. Bound only when the run names the served stack and its base; a run about another model is left out
    (None), never raised, so a later model does not cost the page its other evidence. The decision years are E1's
    DEV years only when both runs scored the same number of DEV issue times."""
    e3 = _read(root, EARTHQUAKE_E3)
    if e3 is None:
        return None
    if e3.get("stack") != version or e3.get("base") != base_version:
        return None
    dev = (e3.get("splits") or {}).get("dev") or {}
    cells = dev.get("cells_with_most_removed_events") or {}
    top_key = max(cells, key=lambda k: int(cells[k].get("removed_m5_events") or 0)) if cells else None
    top = cells.get(top_key) or {}
    rc = re.fullmatch(r"r(\d+)c(\d+)", top_key or "")
    same_dev = int(dev.get("n_issue_times") or -1) == int(e1_dev.get("n_issue_times") or -2)
    ig = ((dev.get("paired_E3s_minus_S1") or {}).get("ig_per_target")) or {}
    rule = e3.get("rule") or {}
    return {
        "file": EARTHQUAKE_E3, "program_section": "section 11",
        "frozen_events": int(e3.get("frozen_events") or 0),
        "non_earthquakes": int(e3.get("frozen_non_earthquakes_removed") or 0),
        "top_cell": ({"row": int(rc.group(1)), "col": int(rc.group(2)),
                      "removed": int(top.get("removed_m5_events") or 0),
                      "with": _finite(top.get("S1_mean")), "without": _finite(top.get("E3s_mean"))} if rc else None),
        "dev_years": (f"{e1_dev['first_issue'][:4]}-{e1_dev['last_issue'][:4]}" if same_dev else None),
        "dev_d_ig": _finite(ig.get("diff")), "dev_d_ig_ci": _ci(ig.get("ci95")),
        "margin": _finite(rule.get("margin")), "adopted": bool(rule.get("E3s_replaces_S1")),
    }


def earthquake_evidence(root: Path = ROOT) -> dict | None:
    from hazardpulse.earthquake import operational_forecast as eq

    served = root / EARTHQUAKE_SERVED
    record = _read(root, EARTHQUAKE_ARTIFACT_RECORD)
    final = _read(root, EARTHQUAKE_FINAL)
    if not served.exists() or record is None or final is None:
        return None
    version = eq.artifact_model_version(served)
    if record.get("model_version") != version:
        raise EvidenceError(f"{EARTHQUAKE_ARTIFACT_RECORD} names {record.get('model_version')!r}, "
                            f"the served file is {version!r}")
    meta = json.loads(served.read_text(encoding="utf-8"))
    chosen = ((meta.get("provenance") or {}).get("decision") or {}).get("chosen")
    cand = (final.get("candidates") or {}).get(chosen)
    if not cand:
        raise EvidenceError(f"{EARTHQUAKE_FINAL} has no scores for the served candidate {chosen!r}")

    def metric(c, k):
        m = c.get(k) or {}
        return {"value": _finite(m.get("value")), "ci": _ci(m.get("ci95"))}

    def paired(key):
        p = (final.get("paired") or {}).get(key)
        if not p:
            return None
        return {k: {"diff": _finite(v.get("diff")), "ci": _ci(v.get("ci95"))} for k, v in p.items()}

    contract = meta.get("contract") or {}
    vs = {}
    for other in ("A", "B", "D"):
        if other == chosen:
            continue
        p = paired(f"{chosen}-{other}") or paired(f"{other}-{chosen}")
        if p is None:
            continue
        if f"{chosen}-{other}" not in (final.get("paired") or {}):      # stored as other - chosen: flip
            p = {k: {"diff": -v["diff"] if v["diff"] is not None else None,
                     "ci": [-v["ci"][1], -v["ci"][0]] if v["ci"] else None} for k, v in p.items()}
        vs[other] = {"name": EARTHQUAKE_CANDIDATES.get(other, other), **p}
    n_issue = int(final.get("n_issue_times") or 0)
    n_pos = int(final.get("n_positive_cell_windows") or 0)
    in_active = final.get("positives_in_active_cells")
    replaced = (final.get("candidates") or {}).get("D") if chosen != "D" else None
    payload = (meta.get("gbt") or {}).get("payload") or {}
    if (root / EARTHQUAKE_STACK).exists():
        return _earthquake_stack_evidence(root, version, contract, payload, metric)
    return {
        "hazard": "earthquake", "program": EARTHQUAKE_PROGRAM, "file": EARTHQUAKE_SERVED,
        "model_version": version, "candidate": chosen,
        "candidate_name": EARTHQUAKE_CANDIDATES.get(chosen, chosen),
        "event": contract.get("event"), "horizon_days": contract.get("horizon_days"),
        "target_magnitude_min": contract.get("target_magnitude_min"),
        "test": {
            "period": f"issue times {str(final.get('first_issue'))[:7]} .. {str(final.get('last_issue'))[:7]} (read once)",
            "when": f"{str(final.get('first_issue'))[:4]}-{str(final.get('last_issue'))[:4]}",
            "n_issue_times": final.get("n_issue_times"), "n_cell_times": final.get("n_cell_times"),
            "n_positive": final.get("n_positive_cell_windows"), "block": final.get("block"),
            "ig_per_target": metric(cand, "ig_per_target"), "auc": metric(cand, "auc"),
            "bss": metric(cand, "bss"), "auc_active_cells": metric(cand, "auc_active_cells"),
            "calib_ratio": metric(cand, "calib_ratio"),
            "n_cells": (int(final.get("n_cell_times") or 0) // n_issue) if n_issue else None,
            "share_outside_active_cells": (1.0 - int(in_active) / n_pos) if n_pos and in_active is not None else None,
        },
        "n_trees": payload.get("n_trees"),
        "n_inputs": len(payload.get("feature_names") or []),
        "replaced_ig_per_target": metric(replaced, "ig_per_target") if replaced else None,
        "vs": vs,
    }


# ---------------------------------------------------------------------------
# Hurricane
# ---------------------------------------------------------------------------

def hurricane_evidence(root: Path = ROOT) -> dict | None:
    from hazardpulse.hurricane import ri_stack

    served = root / HURRICANE_SERVED
    final = _read(root, HURRICANE_FINAL)
    if not served.exists() or final is None or final.get("phase") != "final":
        return None
    version = ri_stack.model_version_of(served)
    named = (final.get("artifact") or {}).get("model_version")
    if named != version:
        raise EvidenceError(f"{HURRICANE_FINAL} names {named!r}, the served file is {version!r}")
    choice = (final.get("selection") or {}).get("choice")
    res = ((((final.get("results") or {}).get("all_cases") or {}).get("forecasts")) or {}).get(choice)
    if not res:
        raise EvidenceError(f"{HURRICANE_FINAL} has no 2025 scores for the chosen candidate {choice!r}")
    v82_version = (_read(root, HURRICANE_V82_SERVED) or {}).get("model_version")
    v82_label = version_label(v82_version)
    claims = []
    for key, name in _HURRICANE_CLAIM_NAMES.items():
        c = (final.get("claims") or {}).get(key)
        if c is None:
            continue
        if key == "better_than_v8_2":
            name = f"our previous model ({v82_label})" if v82_label else "our previous model"
        claims.append({"against": name, "better": bool(c.get("claim")), "worse": bool(c.get("worse")),
                       "delta_log_loss_ci": _ci(c.get("delta_log_loss_ci95")),
                       "delta_brier_ci": _ci(c.get("delta_brier_ci95"))})
    adv = _read(root, HURRICANE_ADVERSARY) or {}
    cases = final.get("final_cases") or {}
    v82 = (((final.get("results") or {}).get("v82_subset") or {}).get("forecasts") or {}).get("v8_2") or {}
    other_basins = other_basins_evidence(root, v82_version, v82_label)
    return {
        "hazard": "hurricane", "program": HURRICANE_PROGRAM, "file": HURRICANE_SERVED,
        "model_version": version, "candidate": choice,
        "candidate_name": HURRICANE_CANDIDATES.get(choice, choice),
        "event": "rapid intensification: +30 kt or more in the next 24 h (NHC best track)",
        "test": {
            "period": f"the {final.get('final_season')} season (read once)", "when": str(final.get('final_season')),
            "n": int(res.get("n") or 0), "events": int(res.get("events") or 0),
            "storms": int(cases.get("storms") or res.get("storms") or 0),
            "auc": _finite(res.get("auc")), "auc_ci": _ci(res.get("auc_ci95")),
            "bss": _finite(res.get("bss")), "bss_ci": _ci(res.get("bss_ci95")),
            "brier": _finite(res.get("brier")), "log_loss": _finite(res.get("log_loss")),
            "unit": ((final.get("bootstrap") or {}).get("unit")),
        },
        "chosen_on_seasons": (final.get("fit") or {}).get("seasons"),
        "fallback_cycles": (((final.get("patterns_used") or {}).get(choice)) or {}).get("fallback_cycles"),
        # v8.2's scores in the NHC program's final test: named as v8.2 whatever is published now
        "v8_2_same_cases": ({"auc": _finite(v82.get("auc")), "bss": _finite(v82.get("bss")), "label": v82_label}
                            if v82 else None),
        "claims": claims,
        "adversary": {"verdict": adv.get("verdict"), "statement": adv.get("surviving_statement")} if adv else None,
        "other_basins": other_basins,
        "ours": ours_hurricane(root),
        "v9": v9_hurricane(root),
        "tc1": tc1_hurricane(root),
        "ir_models": ir_models(root),
        "j1": j1_hurricane(root),
    }


HURRICANE_BASIN_NAMES = {"WP": "West Pacific", "SI": "South Indian", "NA": "Atlantic", "EP": "East Pacific",
                         "SP": "South Pacific", "NI": "North Indian", "SA": "South Atlantic",
                         # live (ATCF) codes of the JTWC basins, and a results file's merged southern basin
                         "IO": "North Indian", "SH": "Southern Hemisphere"}


def hurricane_scorer_models(root: Path = ROOT) -> dict[str, str]:
    """``{"served": ..., "j1_base": ...}``: the HazardPulse model the live scorer serves outside the NHC basins and
    the model it feeds J1, read from the scorer's own constants. A scorer from before amendment 15 has no J1 base
    constant: it fed J1 the model it served. Empty when the scorer is not in ``root``."""
    c = _script_constants(root, HURRICANE_SCORER, ("SERVED_MODEL_VERSION", "J1_BASE_MODEL_VERSION"))
    out = {}
    if c.get("SERVED_MODEL_VERSION"):
        out["served"] = str(c["SERVED_MODEL_VERSION"])
        out["j1_base"] = str(c.get("J1_BASE_MODEL_VERSION") or c["SERVED_MODEL_VERSION"])
    return out


def _lf_sha256(path: Path) -> str:
    """sha256 of a text file's bytes with CRLF read as LF: the bytes git stores, on every checkout."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def other_basins_evidence(root: Path, v82_version: str | None, v82_label: str | None) -> dict | None:
    """The HazardPulse model published outside the NHC basins, and what its test was a test OF.

    The scorer names the model (``hurricane_scorer_models``). v8.2 (or no scorer in ``root``): its METHOD's
    temporal hold-out on 2022-2024 cycles from every basin (``_v82_other_basins``). A model that replaced v8.2 by
    a registered rule (amendment 15: v8.3): its own registered test, bound to the served artifact
    (``_replacement_other_basins``)."""
    served = hurricane_scorer_models(root).get("served")
    if served is None or served == v82_version:
        return _v82_other_basins(root, v82_version, v82_label)
    return _replacement_other_basins(root, served, v82_version)


def _v82_other_basins(root: Path, v82_version: str | None, v82_label: str | None) -> dict | None:
    evaluation = _read(root, HURRICANE_V82_EVALUATION) or {}
    v82_heldout = (evaluation.get("results") or {}).get("C_v8_2_heldout_newton")
    if not (v82_heldout and v82_version):
        return None
    test_years = (evaluation.get("origin") or {}).get("test") or ["?", "?"]
    return {
        "model": v82_version, "label": v82_label, "subject": "method", "scope": "every basin",
        "test": {"period": f"{test_years[0]}-{test_years[1]} cycles from every basin, held out from the "
                           "recipe's fit",
                 "when": f"{test_years[0]}-{test_years[1]}",
                 "n": int(v82_heldout.get("n") or 0), "auc": _finite(v82_heldout.get("auc")),
                 "auc_ci": _ci((v82_heldout.get("ci95") or {}).get("auc")),
                 "bss": _finite(v82_heldout.get("bss_vs_climatology"))},
        "composition": _v82_composition(root, evaluation, v82_heldout),
    }


def _replacement_other_basins(root: Path, served: str, replaced: str | None) -> dict:
    """A served model that replaced v8.2 by a registered rule: its decision file must have adopted exactly the
    served artifact (version and stored-bytes sha256), with every control passed and the rule met, and must have
    replaced the v8.2 this repository holds. Its test is the registered one: the published artifact itself on
    cycles never used to fit it, against the model it replaced."""
    art_rel, res_rel = f"results/models/{served}.json", f"results/calibration/{served}.json"
    art, res = _read(root, art_rel), _read(root, res_rel)
    if art is None or res is None:
        raise EvidenceError(f"the scorer serves {served}, but {art_rel} or its decision {res_rel} is not here")
    if art.get("model_version") != served:
        raise EvidenceError(f"{art_rel} holds {art.get('model_version')!r}, the scorer serves {served!r}")
    cand, dec, rule = res.get("candidate") or {}, res.get("decision") or {}, res.get("rule") or {}
    if dec.get("serve") != served or dec.get("adopt") is not True or rule.get("met") is not True:
        raise EvidenceError(f"{res_rel} does not adopt {served} by its rule (decision {dec}, rule met "
                            f"{rule.get('met')!r})")
    if cand.get("version") != served or cand.get("artifact_sha256") != _lf_sha256(root / art_rel):
        raise EvidenceError(f"{res_rel} scored another artifact than the served {art_rel}")
    if not (res.get("controls") or {}).get("all_passed"):
        raise EvidenceError(f"{res_rel}: its controls did not all pass")
    was = (res.get("comparator") or {}).get("version")
    if replaced is None or was != replaced:
        raise EvidenceError(f"{res_rel} replaced {was!r}; the model this repository published before is {replaced!r}")
    label, was_label = version_label(served), version_label(was)
    k, kw = label.replace(".", "_"), was_label.replace(".", "_")
    reg = res.get("registered") or {}
    m, w, d = reg.get(k) or {}, reg.get(kw) or {}, reg.get(f"{k}_minus_{kw}") or {}
    t = res.get("test") or {}
    seasons = sorted(int(s) for s in t.get("storm_first_seasons") or [])
    if not seasons or int(t.get("n") or -1) != int(reg.get("n") or -2) or int(m.get("n") or -3) != int(reg.get("n") or -2):
        raise EvidenceError(f"{res_rel}: its test and its registered numbers disagree")
    if _finite(rule.get("upper")) != _finite((d.get("d_ll_ci") or [None, None])[1]):
        raise EvidenceError(f"{res_rel}: the rule's upper bound is not the registered interval's")
    when = f"{seasons[0]}-{seasons[-1]}"
    program = str(res.get("program") or "")
    amendment = re.search(r"\bamendment (\d+\w*)", program)
    return {
        "model": served, "label": label, "subject": "model", "scope": "the JTWC basins",
        "regions": list(t.get("decision_basins") or []),
        "test": {"period": f"{when} cycles in the JTWC basins, none used to fit it", "when": when,
                 "n": int(t["n"]), "events": int(t.get("events") or 0), "storms": int(t.get("storms") or 0),
                 "auc": _finite(m.get("auc")), "auc_ci": None, "bss": None,
                 "log_loss": _finite(m.get("log_loss")), "brier": _finite(m.get("brier"))},
        "replaced": {"model": was, "label": was_label, "log_loss": _finite(w.get("log_loss")),
                     "brier": _finite(w.get("brier")), "auc": _finite(w.get("auc"))},
        "vs_replaced": {"d_ll": _finite(d.get("d_ll")), "d_ll_ci": _ci(d.get("d_ll_ci")),
                        "d_brier": _finite(d.get("d_brier")), "d_brier_ci": _ci(d.get("d_brier_ci"))},
        "rule": {"margin": _finite(rule.get("margin")), "upper": _finite(rule.get("upper")), "met": True},
        "program": HURRICANE_V9_PROGRAM, "amendment": amendment.group(1) if amendment else None,
        "results": res_rel, "prereg_tag": res.get("prereg_tag"),
        "composition": _replacement_composition(root, served, res, art, seasons),
    }


def _replacement_composition(root: Path, served: str, res: dict, art: dict, seasons: list[int]) -> dict | None:
    """What a replacement model's registered test was a test OF (``hurricane_v82_test_composition.py --model``):
    refused unless it describes that test -- its cycles, the candidate's AUC, the rows' sha256 and the model."""
    rel = f"results/calibration/{served}_test_composition.json"
    comp = _read(root, rel)
    if comp is None:
        return None
    ev, t = comp.get("evaluation") or {}, res.get("test") or {}
    k = version_label(served).replace(".", "_")
    auc = _finite(((res.get("registered") or {}).get(k) or {}).get("auc"))
    if (int(ev.get("n") or -1) != int(t.get("n") or 0) or _finite(ev.get("auc")) != auc
            or (comp.get("dataset") or {}).get("sha256") != t.get("rows_sha256")
            or (comp.get("served_artifact") or {}).get("model_version") != served):
        raise EvidenceError(f"{rel} does not describe the registered test in results/calibration/{served}.json")
    basins = comp.get("by_basin") or {}
    if sum(int(b.get("n") or 0) for b in basins.values()) != int(t.get("n") or 0):
        raise EvidenceError(f"{rel}: its basins do not add up to the test's cycles")
    cal = art.get("calibration") or {}
    fitted = [int(y) for y in (cal.get("fitted_on") or {}).get("storm_years") or []]
    on_test = bool(fitted) and fitted[0] <= seasons[-1] and seasons[0] <= fitted[-1]
    return _composition_view(comp, on_test)


def _composition_view(comp: dict, on_test: bool | None) -> dict:
    """A test-composition file as the pages read it."""
    basins = comp.get("by_basin") or {}
    jt = comp.get("single_jtwc_warning") or {}
    lf = comp.get("late_best_track_fix") or {}
    return {
        "by_basin": [{"basin": b, "name": HURRICANE_BASIN_NAMES.get(b, b), "n": int(v.get("n") or 0),
                      "events": int(v.get("events") or 0)} for b, v in basins.items()],
        "inputs": "best track",
        "served_calibration_fitted_on_test_cases": on_test,
        "single_jtwc_warning": {
            "n_inputs": int(jt.get("n_selected_inputs") or 0),
            "n_missing": len(jt.get("inputs_missing_live") or []),
            "n_cycles": int(jt.get("west_pacific_test_cycles") or 0),
            "log_loss": _finite(jt.get("log_loss_single_warning_inputs")),
            "log_loss_full_inputs": _finite(jt.get("log_loss_full_inputs")),
            "log_loss_climatology": _finite(jt.get("log_loss_climatology")),
        } if jt else None,
        # since 2026-10-05: RAL's history with the warning as the analysis (its cycle's fix not yet published)
        "late_best_track_fix": {
            "inputs": sorted(lf.get("inputs_missing_live") or []),
            "n_missing": len(lf.get("inputs_missing_live") or []),
            "log_loss": _finite(lf.get("log_loss")),
        } if lf else None,
    }


def _v82_composition(root: Path, evaluation: dict, heldout: dict) -> dict | None:
    """What v8.2's held-out figure tests (site audit 2026-10-05): the test's basins and inputs, whether
    the SERVED artifact's calibration was fitted on the test cases (read from the artifact itself), and
    the served model on West Pacific cycles with a single JTWC warning's inputs. Refused when the file
    describes another evaluation (its count, AUC or data hash differ)."""
    comp = _read(root, HURRICANE_V82_COMPOSITION)
    if comp is None:
        return None
    ev = comp.get("evaluation") or {}
    if (int(ev.get("n") or -1) != int(heldout.get("n") or 0) or _finite(ev.get("auc")) != _finite(heldout.get("auc"))
            or (comp.get("dataset") or {}).get("sha256") != (evaluation.get("data_sha256") or {}).get("v8.2")):
        raise EvidenceError(f"{HURRICANE_V82_COMPOSITION} does not describe the evaluation in {HURRICANE_V82_EVALUATION}")
    basins = comp.get("by_basin") or {}
    if sum(int(b.get("n") or 0) for b in basins.values()) != int(heldout.get("n") or 0):
        raise EvidenceError(f"{HURRICANE_V82_COMPOSITION}: its basins do not add up to the test's cycles")
    served = _read(root, HURRICANE_V82_SERVED) or {}
    cal = served.get("calibration") or {}
    on_test = (list((cal.get("fitted_on") or {}).get("storm_years") or []) == list(comp.get("test_storm_years") or [None])
               and int(cal.get("n") or -1) == int(heldout.get("n") or 0))
    return _composition_view(comp, bool(on_test) if served else None)


def _ours_vs_all(root: Path, dev: dict, prov: dict, label: str) -> dict | None:
    """Our shown model against every public RI aid on the development cases
    (``results/calibration/hurricane_ri_v10_vs_all.json``), bound to the shown artifact: the file names the model
    it compares ("v10.1 = V2 with the early-guidance gate ..."); when that is the shown model, its control log
    loss must be the artifact's own development log loss, bit for bit, and its gate the artifact's gate -- a file
    of this model that disagrees is refused. A file about another of our models is not shown for this one."""
    rep = _read(root, "results/calibration/hurricane_ri_v10_vs_all.json")
    if rep is None:
        return None
    named = str(rep.get("model", ""))
    if " = " in named and named.split(" = ", 1)[0].strip() != label:
        return None                                     # the comparison of another of our models
    if _finite(rep.get("control_V2_log_loss")) != _finite(dev.get("log_loss")):
        raise EvidenceError(f"hurricane_ri_v10_vs_all.json control LL {rep.get('control_V2_log_loss')} is not the "
                            f"shown artifact's development LL {dev.get('log_loss')}")
    gate = list(prov.get("gate_aids") or [])
    if not gate or any(a not in str(rep.get("model", "")) for a in gate):
        raise EvidenceError(f"hurricane_ri_v10_vs_all.json describes {rep.get('model')!r}, not the gate {gate}")
    aids = []
    for tech, r in (rep.get("aids") or {}).items():
        w, o, a = r.get("vs_30") or {}, r.get("ours_30") or {}, r.get("aid_30") or {}
        aids.append({"tech": tech, "label": r.get("label", tech), "n": r.get("n"),
                     "ours_log_loss": _finite(o.get("log_loss")), "aid_log_loss": _finite(a.get("log_loss")),
                     "d_log_loss_ci": _ci(w.get("d_log_loss_ci"))})
    calls = []
    for name, r in (rep.get("calls") or {}).items():
        calls.append({"name": name, "label": r.get("label", name), "calls": r.get("calls"),
                      "aid_pofd": _finite(r.get("aid_pofd")), "aid_pod": _finite(r.get("aid_pod")),
                      "ours_pod": _finite(r.get("ours_pod")), "d_pod": _finite(r.get("d_pod")),
                      "d_pod_ci": _ci(r.get("d_pod_ci"))})
    return {"n": rep.get("n"), "events": rep.get("events"), "seasons": rep.get("seasons") or [],
            "aids": aids, "calls": calls}


# Our RI models of the v10 schema, in the order each was carried: (prospective entrant, artifact, what it adds to
# the model it was selected against -- "{base}" is that model's label, read from its artifact -- and that model's
# entrant). Which ONE of them the site shows is named only by HURRICANE_SHOWN; the ones carried after it are its
# challengers. Each label is the artifact's own; v10.1's artifact predates labels, so its entrant key is its label.
HURRICANE_V10_FAMILY = (
    ("v10_1", "results/models/hurricane_ri_v10.json", None, None),
    ("v10_2", "results/models/hurricane_ri_v10_2.json",
     "the same inputs, never lowering the odds when the guidance or NOAA&rsquo;s probability rises", "v10_1"),
    ("v10_3", "results/models/hurricane_ri_v10_3.json",
     "{base} plus the storm&rsquo;s cloud-top structure from satellite infrared (NOAA GMGSI, two hours after the "
     "cycle and six hours earlier)", "v10_2"),
    ("v10_4", "results/models/hurricane_ri_v10_4.json",
     "{base} plus how much of that convective heating the vortex can hold: the coherence equation solved as the "
     "balanced response inside the local Rossby radius", "v10_3"),
)


def family_label(art: dict, entrant: str) -> str:
    """An artifact's label: its own ``label`` field, else (v10.1's artifact, which predates labels) its entrant
    key. A label that disagrees with the entrant it is recorded under is refused."""
    expected = entrant_label(entrant)
    own = art.get("label")
    if own is not None and str(own) != expected:
        raise EvidenceError(f"the artifact recorded as entrant {entrant!r} is labelled {own!r}")
    return expected


def dev_period(prov: dict) -> tuple[str | None, dict]:
    """The development period an artifact's provenance declares, and its numbers: ``dev_2022_2025`` ->
    ``("2022-2025", {...})``. None when the provenance names no such block."""
    for key, val in prov.items():
        m = re.fullmatch(r"dev_(\d{4})_(\d{4})", key)
        if m and isinstance(val, dict):
            return f"{m.group(1)}-{m.group(2)}", val
    return None, {}


def _season_2026_read(prov: dict) -> dict:
    """The declared later look at 2026 an artifact carries (second, third, fourth or further read)."""
    for key in sorted(prov):
        if re.fullmatch(r"season_2026_\w+_read", key) and isinstance(prov[key], dict):
            return prov[key]
    return {}


def shown_model(root: Path = ROOT) -> dict | None:
    """Which of our RI models the site shows beside NOAA's published number (amendment 4): the entrant named in
    ``HURRICANE_SHOWN`` -- the only place it is named -- with its artifact, its own label and version, and the
    key its shadow forecasts carry in each live record (the prospective scorer's own record of that entrant).
    None when the pointer is absent; refused when it names a model the family, the artifacts or the prospective
    test do not know."""
    from hazardpulse.hurricane import ri_v10

    ptr = _read(root, HURRICANE_SHOWN)
    if ptr is None:
        return None
    entrant = str(ptr.get("shown") or "")
    family = {e: (rel, what, base) for e, rel, what, base in HURRICANE_V10_FAMILY}
    if entrant not in family:
        raise EvidenceError(f"{HURRICANE_SHOWN} shows {entrant!r}, which is not one of our v10 models")
    rel = family[entrant][0]
    if not (root / rel).exists():
        raise EvidenceError(f"{HURRICANE_SHOWN} shows {entrant!r}, whose artifact {rel} is not in the repository")
    art, version = ri_v10.load(root / rel)
    pros = _read(root, HURRICANE_PROSPECTIVE) or {}
    rule = (((pros.get("entrants") or {}).get(entrant) or {}).get("claim_rule")) or {}
    if not rule.get("key"):
        raise EvidenceError(f"{HURRICANE_PROSPECTIVE} records no entrant {entrant!r}: the shown model's live "
                            "forecasts cannot be found")
    return {"entrant": entrant, "file": rel, "art": art, "model_version": version,
            "label": family_label(art, entrant), "shadow_key": str(rule["key"]), "pointer": HURRICANE_SHOWN}


def _ours_challengers(root: Path, shown_entrant: str, pros: dict) -> list[dict]:
    """Every model of the family carried after the shown one, running in shadow (amendments 3-9), each from its
    own artifact's provenance -- refused when the numbers of the model it was selected against are not that
    model's own, bit for bit (v10.2 against v10.1, v10.3 against v10.2, v10.4 against v10.3)."""
    from hazardpulse.hurricane import ri_v10

    out, carried, after_shown = [], {}, False
    for entrant_key, rel, what, base_key in HURRICANE_V10_FAMILY:
        path = root / rel
        if not path.exists():
            continue
        art, version = ri_v10.load(path)
        prov = art.get("provenance") or {}
        span, dev = dev_period(prov)
        label = family_label(art, entrant_key)
        if base_key is not None:
            base = carried.get(base_key)
            if base is None or _finite(dev.get("champion_log_loss")) != _finite(base["dev"].get("log_loss")):
                raise EvidenceError(f"{rel} was selected against LL {dev.get('champion_log_loss')}, not "
                                    f"{entrant_label(base_key)}'s {None if base is None else base['dev'].get('log_loss')}")
        carried[entrant_key] = {"dev": dev, "label": label}
        if entrant_key == shown_entrant:
            after_shown = True
            continue
        if not after_shown:
            continue
        base_label = carried[base_key]["label"]
        entrant = (pros.get("entrants") or {}).get(entrant_key) or {}
        versus = (pros.get("challenger_vs_champion") or {}).get(f"{entrant_key}_vs_{shown_entrant}")
        s26 = _season_2026_read(prov)
        out.append({"model_version": version, "label": label, "what": (what or "").format(base=base_label),
                    "against": base_label, "dev_period": span, "uses_ir": ri_v10.needs_ir(art),
                    "dev": {"log_loss": _finite(dev.get("log_loss")), "brier4": _finite(dev.get("brier4")),
                            "champion_log_loss": _finite(dev.get("champion_log_loss")),
                            "champion_brier4": _finite(dev.get("champion_brier4")),
                            "d_log_loss_ci": _ci(dev.get("d_log_loss_vs_champion_ci"))},
                    "season_2026": {"log_loss": _finite(s26.get("log_loss")),
                                    "champion_log_loss": _finite(s26.get("champion_log_loss")),
                                    "d_log_loss_ci": _ci(s26.get("d_log_loss_ci"))} if s26 else None,
                    "level": _finite((entrant.get("claim_rule") or {}).get("level")),
                    "matured_and_scored": entrant.get("matured_and_scored", 0),
                    "versus_champion": versus if isinstance(versus, dict) else None})
    return out


def ir_models(root: Path = ROOT) -> list[str]:
    """The labels of our RI models that read satellite infrared, from each artifact's own inputs."""
    from hazardpulse.hurricane import ri_v10

    out = []
    for entrant_key, rel, _what, _base in HURRICANE_V10_FAMILY:
        if (root / rel).exists():
            art, _version = ri_v10.load(root / rel)
            if ri_v10.needs_ir(art):
                out.append(family_label(art, entrant_key))
    return out


TC1_DIR = "results/hurricane_tc1"
TC1_PROGRAM = "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md"


def tc1_model_version(root: Path = ROOT) -> str | None:
    """TC1's identity: the content digest of the saved models the scorer replays each season from
    (``results/hurricane_tc1/state.json``, which itself records the digests of the selection and the DEV
    results it was made from). A new snapshot is a new version."""
    from hazardpulse.hurricane import tc1_live

    p = root / TC1_DIR / "state.json"
    return f"hurricane_tc1-{tc1_live.sha256(p)[:12]}" if p.exists() else None


def _tc1_diff(r: dict) -> dict | None:
    """One paired comparison (product minus another forecast): the mean over leads with its interval and level,
    and each lead's."""
    if not r or not r.get("mean_over_leads"):
        return None
    m = r["mean_over_leads"]
    return {"d": _finite(m.get("d")), "ci": _ci(m.get("ci")), "level": _finite(r.get("level")),
            "claim": bool(r.get("claim")) if "claim" in r else None, "n_storms": r.get("n_storms"),
            "per_lead": {str(k): {"d": _finite(v.get("d")), "ci": _ci(v.get("ci")), "n": v.get("n")}
                         for k, v in (r.get("per_lead") or {}).items()}}


def tc1_hurricane(root: Path = ROOT) -> dict | None:
    """Our track and intensity forecast (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md): its identity, its registered
    DEV numbers and the 2026 further read, bound to the saved models the scorer serves -- refused when the state
    was not made from this selection and these DEV results (content digests recorded in the state)."""
    from hazardpulse.hurricane import tc1_live

    d = root / TC1_DIR
    state_p, sel_p, dev_p = d / "state.json", d / "selection.json", d / "dev.json"
    if not (state_p.exists() and sel_p.exists() and dev_p.exists()):
        return None
    state = json.loads(state_p.read_text(encoding="utf-8"))
    if state.get("selection_sha256") != tc1_live.sha256(sel_p) or state.get("dev_sha256") != tc1_live.sha256(dev_p):
        raise EvidenceError("results/hurricane_tc1/state.json was not made from the selection and DEV results on disk")
    dev = json.loads(dev_p.read_text(encoding="utf-8"))
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    s26 = _read(root, f"{TC1_DIR}/season_2026.json") or {}
    out = {"model_version": tc1_model_version(root), "program": TC1_PROGRAM, "prereg_tag": dev.get("prereg_tag"),
           "file": f"{TC1_DIR}/state.json", "season_end": state.get("season_end"),
           "selection_sha256": state.get("selection_sha256"),
           "choose_seasons": (sel.get("seasons") or {}).get("choose"),
           "seasons": dev.get("seasons"), "claim_level": _finite(dev.get("claim_level")),
           "leads": dev.get("scored_leads"), "season_2026_at": s26.get("generated_at"),
           "season_2026_storms": len(s26.get("storms") or []) or None}
    for kind, second in (("track", "TVCN"), ("intensity", "IVCN")):
        k = dev[kind]
        err = {p: _finite((k["errors"].get(p) or {}).get("mean_over_leads")) for p in ("TC1", "TC1+O", "OFCL", "HCCA", second)}
        claim = k["claims"]["TC1-OFCL"]
        r26 = ((s26.get(kind) or {}).get("reported") or {})
        out[kind] = {"errors": err, "second": second,
                     "vs_ofcl": {"d": _finite(claim["mean_over_leads"]["d"]), "ci": _ci(claim["mean_over_leads"]["ci"]),
                                 "claim": bool(claim.get("claim")), "level": _finite(claim.get("level"))},
                     "vs_hcca": _tc1_diff(k["reported"]["TC1-HCCA"]),
                     "tc1o_vs_ofcl": _tc1_diff(k["claims"].get("TC1+O-OFCL") or {}),
                     "season_2026_vs_ofcl": _tc1_diff(r26.get("TC1-OFCL") or {}),
                     "season_2026_tc1o_vs_ofcl": _tc1_diff(r26.get("TC1+O-OFCL") or {})}
    out["tc2b"] = tc2b_hurricane(root)
    return out


TC2_DIR = "results/hurricane_tc2"


def tc2b_hurricane(root: Path = ROOT) -> dict | None:
    """TC2b (TC1 program amendments 2-4): TC1's intensity held to the median our RI model's 24-h curve implies. Its
    registered test (2026) and the DEV result of the same first-day rule, read from its results files; None when
    either is absent. ``ri_subset`` values are ``[mean error, cycles]`` as the test wrote them."""
    s26 = _read(root, f"{TC2_DIR}/season_2026.json")
    dev = _read(root, f"{TC2_DIR}/dev.json")
    if not s26 or not dev:
        return None

    def ri(doc, products):
        r = (doc.get("ri_subset") or {}).get("24") or {}
        return {p: _finite((r.get(p) or [None])[0]) for p in products}, (r.get("TC1") or [None, None])[1]

    ri26, n26 = ri(s26, ("TC1", "TC2b", "OFCL", "HCCA"))
    ridev, ndev = ri(dev, ("TC1", "TC2", "OFCL", "HCCA"))
    d24 = ((dev.get("TC2-TC1") or {}).get("per_lead") or {}).get("24") or {}
    dev_seasons = [int(x) for x in dev.get("seasons") or []]
    test_season = sorted({int(str(s)[-4:]) for s in s26.get("storms") or [] if str(s)[-4:].isdigit()})
    return {"label": "TC2b", "program": TC1_PROGRAM, "prereg_tag": s26.get("prereg_tag"),
            "dev_seasons": dev_seasons, "test_season": test_season[0] if len(test_season) == 1 else None,
            "carried": s26.get("carried_tc2b") == "TC2b", "files": [f"{TC2_DIR}/dev.json", f"{TC2_DIR}/season_2026.json"],
            "season_2026_vs_tc1": _tc1_diff(s26.get("TC2b-TC1") or {}),
            "dev_24h_vs_tc1": {"d": _finite(d24.get("d")), "ci": _ci(d24.get("ci"))},
            "dev_carried": dev.get("carried") == "TC2",
            "ri_2026_24h": ri26, "ri_2026_cycles": n26, "ri_dev_24h": ridev, "ri_dev_cycles": ndev}


def ours_hurricane(root: Path = ROOT) -> dict | None:
    """The one of our own RI models the site shows (``shown_model``) in prospective verification
    (docs/HURRICANE_RI_V9_PROGRAM.md, amendments 2 and 4): its identity and numbers from the frozen artifact's own
    provenance (bound by its bytes), and the prospective record so far. Shown beside NOAA's published number,
    never instead of it, until a look date's claim rule is met."""
    from hazardpulse.hurricane import ri_v10

    shown = shown_model(root)
    if shown is None:
        return None
    art, version, label = shown["art"], shown["model_version"], shown["label"]
    prov = art.get("provenance") or {}
    span, dev = dev_period(prov)
    s26 = _season_2026_read(prov)
    pros = _read(root, HURRICANE_PROSPECTIVE) or {}
    entrant = (pros.get("entrants") or {}).get(shown["entrant"]) or {}
    looks = entrant.get("looks") or {}
    claimed = any(bool(l.get("claim")) for l in looks.values())
    vs_all = _ours_vs_all(root, dev, prov, label)
    challengers = _ours_challengers(root, shown["entrant"], pros)
    family = {e: (what, base) for e, _rel, what, base in HURRICANE_V10_FAMILY}
    what, base = family[shown["entrant"]]
    return {"model_version": version, "label": label, "name": f"HazardPulse RI {label}", "entrant": shown["entrant"],
            "file": shown["file"], "shadow_key": shown["shadow_key"], "pointer": shown["pointer"],
            "adds": what.format(base=entrant_label(base)) if what and base else None,
            "uses_ir": ri_v10.needs_ir(art), "dev_period": span, "vs_all": vs_all,
            "challengers": challengers, "challenger": challengers[0] if challengers else None,
            "status": "claim met at a look" if claimed else "in prospective verification",
            "level": _finite((entrant.get("claim_rule") or {}).get("level")),
            "dev": {"log_loss": _finite(dev.get("log_loss")), "auc": _finite(dev.get("auc")),
                    "dtops_log_loss": _finite(dev.get("dtops_log_loss")), "d_log_loss_ci": _ci(dev.get("d_log_loss_ci")),
                    "multi_threshold": dev.get("multi_threshold")},
            "season_2026": {"log_loss": _finite(s26.get("log_loss")), "auc": _finite(s26.get("auc")),
                            "dtops_log_loss": _finite(s26.get("dtops_log_loss")), "declared": s26.get("declared")},
            "prospective": {"matured_and_scored": entrant.get("matured_and_scored", 0), "start": pros.get("start"),
                            "look_dates": pros.get("look_dates") or [], "looks": looks}}


def _better(lower: Any, higher: Any) -> bool:
    """Both numbers exist and the first is the smaller (log loss and Brier: lower is better)."""
    a, b = _finite(lower), _finite(higher)
    return a is not None and b is not None and a < b


def v9_hurricane(root: Path = ROOT) -> dict | None:
    """v9.1 (amendment 1), the entrant whose prospective claim would switch what is PUBLISHED in the NHC areas from
    NOAA's DTOPS to it: its identity and its read-once 2026 result from the frozen artifact's provenance, its claim
    level and running record from the prospective scorer. The running numbers are descriptive; a claim is made only
    at a look."""
    from hazardpulse.hurricane import ri_v9

    path = root / HURRICANE_V9_SERVED
    if not path.exists():
        return None
    payload, version = ri_v9.load(path)
    prov = payload.get("provenance") or {}
    pros = _read(root, HURRICANE_PROSPECTIVE) or {}
    entrants = pros.get("entrants") or {}
    key = next((k for k, e in entrants.items()
                if ((e.get("claim_rule") or {}).get("key") == "ri_v9_shadow")), None)
    if key is None:
        return None
    e = entrants[key]
    f26, d26 = prov.get("final_2026") or {}, prov.get("final_2026_dtops") or {}
    run = e.get("running_95") or {}
    vs = run.get("vs_dtops_30kt") or {}
    return {"model_version": version, "label": entrant_label(key), "entrant": key, "file": HURRICANE_V9_SERVED,
            "program": HURRICANE_V9_PROGRAM, "candidate": prov.get("candidate"), "trees": payload.get("n_trees"),
            "trained": prov.get("trained"), "gate_aids": list(prov.get("gate_aids") or []),
            "final_2026": {"n": f26.get("n"), "events": f26.get("events"), "log_loss": _finite(f26.get("log_loss")),
                           "brier": _finite(f26.get("brier")), "auc": _finite(f26.get("auc")),
                           "dtops_log_loss": _finite(d26.get("log_loss")), "dtops_brier": _finite(d26.get("brier")),
                           "dtops_auc": _finite(d26.get("auc")), "claim": bool(prov.get("claim_better_than_dtops")),
                           "better_on_every_point": (_better(f26.get("log_loss"), d26.get("log_loss"))
                                                     and _better(f26.get("brier"), d26.get("brier"))
                                                     and _better(d26.get("auc"), f26.get("auc")))},
            "level": _finite((e.get("claim_rule") or {}).get("level")),
            "matured_and_scored": int(e.get("matured_and_scored") or 0),
            "running": ({"n": (run.get("ours") or {}).get("n"), "storms": run.get("storms"),
                         "events": (run.get("ours") or {}).get("events"),
                         "log_loss": _finite((run.get("ours") or {}).get("log_loss")),
                         "dtops_log_loss": _finite((run.get("dtops") or {}).get("log_loss")),
                         "d_log_loss": _finite(vs.get("d_log_loss")),
                         "d_log_loss_ci": _ci(vs.get("d_log_loss_ci")), "level": _finite(run.get("level"))}
                        if (run.get("ours") or {}).get("n") else None),
            "look_dates": pros.get("look_dates") or [], "scored_as_of": pros.get("scored_as_of"),
            "claimed": any(bool(l.get("claim")) for l in (e.get("looks") or {}).values())}


# ---------------------------------------------------------------------------
# J1: our RI model for the JTWC basins, in test beside the published v8.2 (amendments 13, 13a, 13b, 14)
# ---------------------------------------------------------------------------

HURRICANE_J1_RESULTS = "results/calibration/hurricane_ri_j1.json"
HURRICANE_J1_PROSPECTIVE = "results/hurricane_prospective/j1_shadow.json"
HURRICANE_J1_SCORER = "scripts/score_hurricane_j1_prospective.py"
HURRICANE_J1_RUN = "scripts/hurricane_ri_j1.py"
# Amendment 13's arms in its results file: the bar (v8.2 exactly as served) and the mechanism control (the
# candidate without the satellite columns). The candidate's own key is its label, read from the artifact.
J1_BAR, J1_CONTROL = "A", "B"
# Amendment 13 registers paired 95% storm-bootstrap intervals for every J1 comparison, and the run's ``paired``
# takes the 2.5th and 97.5th percentiles (HURRICANE_J1_RUN). The results file does not carry the level, so it is
# named here once; tests/test_site_j1.py fails if the run's percentiles stop being this level's.
J1_INTERVAL_LEVEL = 0.95
# Amendment 14's account of how fragile the pooled interval is: the independent pass's numbers are recorded only
# in the program, so the page quotes that passage (the paragraph opening with this lead, and its bullets).
J1_FRAGILITY_LEAD = "**The fragility of the pooled interval.**"


def _script_constants(root: Path, rel: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Module-level literal constants of a script, read from its source without running it."""
    path = root / rel
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def program_passage(root: Path, doc: str, lead: str) -> dict | None:
    """The paragraph of a program document that opens with ``lead`` and the bullets that follow it, verbatim
    (markdown kept); None when the document or the passage is not there."""
    path = root / doc
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(lead):
            continue
        heading = next((h.lstrip("#").strip() for h in reversed(lines[:i]) if h.startswith("#")), "")
        amendment = re.search(r"\bAmendment (\d+\w*)\b", heading)
        text, j = [line[len(lead):].strip()], i + 1
        while j < len(lines) and lines[j].strip() and not lines[j].startswith("- "):
            text.append(lines[j].strip())
            j += 1
        bullets: list[str] = []
        while j < len(lines) and lines[j].strip() and (lines[j].startswith("- ") or (bullets and lines[j].startswith("  "))):
            if lines[j].startswith("- "):
                bullets.append(lines[j][2:].strip())
            else:
                bullets[-1] += " " + lines[j].strip()
            j += 1
        return {"doc": doc, "text": " ".join(t for t in text if t), "bullets": bullets,
                "amendment": amendment.group(1) if amendment else None}
    return None


def _brier_paired(p: dict | None) -> dict:
    p = p or {}
    return {"d_ll": _finite(p.get("d_ll")), "d_ll_ci": _ci(p.get("d_ll_ci")),
            "d_brier": _finite(p.get("d_brier")), "d_brier_ci": _ci(p.get("d_brier_ci"))}


def _scores(d: dict | None) -> dict:
    d = d or {}
    return {"n": int(d.get("n") or 0), "events": int(d.get("events") or 0), "log_loss": _finite(d.get("log_loss")),
            "brier": _finite(d.get("brier")), "auc": _finite(d.get("auc")),
            "mean_forecast": _finite(d.get("mean_forecast"))}


def _dev_block(prov: dict) -> tuple[str | None, dict]:
    """``dev_2024_2025_jtwc`` -> ``("2024-2025", {...})``: the development block an artifact's provenance names,
    with or without a suffix naming its basins."""
    for key, val in prov.items():
        m = re.fullmatch(r"dev_(\d{4})_(\d{4})(?:_[a-z]+)?", key)
        if m and isinstance(val, dict):
            return f"{m.group(1)}-{m.group(2)}", val
    return None, {}


def _span_of(keys: Any) -> str | None:
    years = sorted(str(k) for k in keys or [] if re.fullmatch(r"\d{4}", str(k)))
    return f"{years[0]}-{years[-1]}" if years else None


def j1_hurricane(root: Path = ROOT) -> dict | None:
    """J1 (docs/HURRICANE_RI_V9_PROGRAM.md, amendments 13, 13a, 13b, 14 and 15): our RI model for the West Pacific,
    North Indian and Southern Hemisphere, recorded live beside the published number (v8.2 until amendment 15,
    v8.3 since) and never instead of it; built on v8.2 and tested against it.

    * identity, label and live basins: the artifact (``hazardpulse.hurricane.ri_j1.load``), whose label must be the
      one its live records are written under, and the key of those records from the same library;
    * the registered result: ``HURRICANE_J1_RESULTS``, which must be the result the artifact's provenance records
      (log loss, Brier, AUC, cycles, events, the bar's numbers and the interval, bit for bit), must have carried
      this label, and must have been measured against the v8.2 the scorer feeds it (``hurricane_scorer_models``:
      its J1 base, by version and by the stored-bytes sha256 J1 was exported with);
    * where it is shown (amendment 14): the library's scope and its per-region development numbers;
    * the rule that would publish it: the prospective scorer's own constants (looks, level), and its running
      record when the scorer has written one.

    None without the artifact or its result; refused (EvidenceError) when they disagree."""
    from hazardpulse.hurricane import ir_features, ir_source, ri_j1

    rel = ri_j1.MODEL_PATH.relative_to(ri_j1.ROOT).as_posix()
    res = _read(root, HURRICANE_J1_RESULTS)
    if not (root / rel).exists() or res is None:
        return None
    art, version = ri_j1.load(root / rel)
    label = str(art.get("label") or "")
    if label != ri_j1.LABEL:
        raise EvidenceError(f"{rel} is labelled {art.get('label')!r}; its live records are written as {ri_j1.LABEL!r}")
    if list(art.get("live_basins") or []) != list(ri_j1.LIVE_JTWC_BASINS):
        raise EvidenceError(f"{rel} is live in {art.get('live_basins')}; the scorer scores {list(ri_j1.LIVE_JTWC_BASINS)}")
    if res.get("carried") != label:
        raise EvidenceError(f"{HURRICANE_J1_RESULTS} carried {res.get('carried')!r}, not {label!r}")
    v82 = _read(root, HURRICANE_V82_SERVED) or {}
    dep = art.get("v82_dependency") or {}
    against = dep.get("model_version")
    # J1 must be fed (and compared against) the model it was exported with: the scorer's J1 base (amendment 15),
    # or -- a scorer from before it, or none in ``root`` -- the v8.2 that was published then
    scorer = hurricane_scorer_models(root)
    base = scorer.get("j1_base") or v82.get("model_version")
    if against != base or against != v82.get("model_version"):
        raise EvidenceError(f"{rel} is built on {against!r}; the scorer feeds it {base!r} and "
                            f"{HURRICANE_V82_SERVED} holds {v82.get('model_version')!r}")
    if dep.get("artifact_sha256") and (root / HURRICANE_V82_SERVED).exists() \
            and dep["artifact_sha256"] != _lf_sha256(root / HURRICANE_V82_SERVED):
        raise EvidenceError(f"{rel} was exported against another {HURRICANE_V82_SERVED} (sha256 "
                            f"{str(dep['artifact_sha256'])[:12]}...)")
    against_label = version_label(against)
    published = scorer.get("served") or against
    published_label = version_label(published)
    prov = art.get("provenance") or {}
    period, pdev = _dev_block(prov)
    dev = res.get("development") or {}
    model, bar, control = _scores(dev.get(label)), _scores(dev.get(J1_BAR)), _scores(dev.get(J1_CONTROL))
    vs_bar = _brier_paired(res.get(f"{label}_minus_{J1_BAR}"))
    folds = _span_of(res.get("by_fold"))
    mine = {"log_loss": model["log_loss"], "brier": model["brier"], "auc": model["auc"], "n": model["n"],
            "events": model["events"], "champion_log_loss": bar["log_loss"], "champion_brier": bar["brier"],
            "champion_label": against_label, "d_log_loss_vs_champion_ci": vs_bar["d_ll_ci"], "period": folds}
    recorded = {k: (pdev.get(k) if k != "period" else period) for k in mine}
    for k in ("n", "events"):
        recorded[k] = int(recorded[k]) if isinstance(recorded[k], (int, float)) else recorded[k]
    for k in ("log_loss", "brier", "auc", "champion_log_loss", "champion_brier"):
        recorded[k] = _finite(recorded[k])
    recorded["d_log_loss_vs_champion_ci"] = _ci(recorded["d_log_loss_vs_champion_ci"])
    differ = sorted(k for k in mine if mine[k] is None or mine[k] != recorded[k])
    if differ:
        raise EvidenceError(f"{rel}'s provenance does not record the result in {HURRICANE_J1_RESULTS}: {differ}")
    further_key = next((k for k in res if re.fullmatch(r"further_read_\d{4}", k)), None)
    fr = res.get(further_key) or {}
    s26 = _season_2026_read(prov)
    further = None
    if fr:
        f_model, f_bar = _scores(fr.get(label)), _scores(fr.get(J1_BAR))
        if s26 and (_finite(s26.get("log_loss")) != f_model["log_loss"]
                    or _finite(s26.get("champion_log_loss")) != f_bar["log_loss"]):
            raise EvidenceError(f"{rel}'s further read is not the one in {HURRICANE_J1_RESULTS}")
        further = {"season": further_key.rsplit("_", 1)[-1], "model": f_model, "bar": f_bar,
                   "declared": fr.get("declared"), **_brier_paired(fr.get(f"{label}_minus_{J1_BAR}"))}
    names = list(art.get("feature_names") or [])
    v82_inputs = set(v82.get("selected_features") or [])
    hours = sorted(int(h) for h in ir_source.OFFSETS.values())
    rule = _script_constants(root, HURRICANE_J1_SCORER, ("LOOKS", "LEVEL"))
    looks, level = [str(x) for x in rule.get("LOOKS") or ()], _finite(rule.get("LEVEL"))
    pros = _read(root, HURRICANE_J1_PROSPECTIVE)
    running, claimed = None, False
    if pros is not None:
        if list(pros.get("look_dates") or []) != looks or pros.get("entrant") != label:
            raise EvidenceError(f"{HURRICANE_J1_PROSPECTIVE} records entrant {pros.get('entrant')!r} with looks "
                                f"{pros.get('look_dates')}; the scorer's are {label!r} and {looks}")
        claimed = any(bool(x.get("claim")) for x in (pros.get("looks") or {}).values())
        run = pros.get("running") or {}
        mk, bk = label.lower(), (against_label or "").replace(".", "_")
        if (run.get(mk) or {}).get("n"):
            d = run.get(f"{mk}_minus_{bk}") or {}
            running = {"n": int(run[mk]["n"]), "events": int(run[mk].get("events") or 0), "storms": run.get("storms"),
                       "log_loss": _finite(run[mk].get("log_loss")),
                       "bar_log_loss": _finite((run.get(bk) or {}).get("log_loss")),
                       "d_ll": _finite(d.get("d_log_loss")), "d_ll_ci": _ci(d.get("d_log_loss_ci")),
                       "level": _finite(run.get("level"))}
    return {
        "label": label, "model_version": version, "file": rel, "shadow_key": ri_j1.SHADOW_KEY,
        "live_basins": list(art["live_basins"]),
        "live_basin_names": [HURRICANE_BASIN_NAMES.get(b, b) for b in art["live_basins"]],
        "program": HURRICANE_V9_PROGRAM, "results": HURRICANE_J1_RESULTS, "prereg_tag": res.get("prereg_tag"),
        "amendment": (m.group(1) if (m := re.search(r"amendments? ([\w, ]+)\)", str(res.get("program") or "")))
                      else None),
        "against": against_label, "against_version": against,
        # the number published beside J1: v8.2 until amendment 15, v8.3 since (J1's comparator stays v8.2)
        "published": published_label, "published_version": published,
        "trained": prov.get("trained"), "seeds": len(prov.get("seeds") or []),
        "inputs": {"total": len(names), "against": sum(n in v82_inputs for n in names),
                   "ir": sum(n in ir_features.IR_NAMES for n in names),
                   "basin": sum(n in ri_j1.BASIN_FLAGS for n in names),
                   "ir_hours_after": [h for h in hours if h > 0], "ir_hours_before": [-h for h in hours if h < 0]},
        "dev_period": folds, "level": J1_INTERVAL_LEVEL,
        "dev": {"model": model, "bar": bar, "control": control},
        "vs_bar": vs_bar, "vs_control": _brier_paired(res.get(f"{label}_minus_{J1_CONTROL}")),
        "noise": [{"seed": s.get("seed"), "d_ll": _finite(s.get("d_ll"))} for s in res.get("noise_control") or []],
        "outcome_basins": list(res.get("by_basin") or {}),
        "folds": {str(k): {"model": _scores(v.get(label)), "bar": _scores(v.get(J1_BAR))}
                  for k, v in sorted((res.get("by_fold") or {}).items())},
        "further": further,
        "fragility": program_passage(root, HURRICANE_V9_PROGRAM, J1_FRAGILITY_LEAD),
        "rule": {"looks": looks, "level": level, "scorer": HURRICANE_J1_SCORER},
        "prospective": {"file": HURRICANE_J1_PROSPECTIVE, "exists": pros is not None,
                        "records": (pros or {}).get("records_with_j1"), "scored_as_of": (pros or {}).get("scored_as_of"),
                        "running": running, "claimed": claimed},
        **_j1_scope(root, res, label, against_label, pros),
    }


def j1_shown_shadow(storm: dict, j1: dict | None) -> dict | None:
    """The J1 forecast a storm card shows beside the published number, or None. Only from bound evidence (``j1``);
    only for a storm whose region (``ri_j1.storm_region``: live IO is NI, SH is SI west of 135 E and SP east of
    it) is in the scope -- J1 is recorded on every JTWC storm but shown only there (amendment 14); only a record
    the scorer marked "ok", with a probability, and made by the bound artifact (its version)."""
    from hazardpulse.hurricane import ri_j1

    if not j1 or not j1.get("scope"):
        return None
    if ri_j1.storm_region(str(storm.get("basin") or ""), _finite(storm.get("lon"))) not in j1["scope"]:
        return None
    v = storm.get(j1["shadow_key"]) or {}
    if (v.get("status") != "ok" or _finite(v.get("probability")) is None
            or v.get("model_version") != j1.get("model_version")):
        return None
    return v


def _j1_scope(root: Path, res: dict, label: str, against_label: str | None, pros: dict | None) -> dict:
    """Where the live entrant is shown and judged (amendment 14): ``ri_j1.scope()``, the regions (WP, NI, SI, SP)
    whose 2024-2025 development log loss against v8.2 has a point estimate <= 0, as the registered J2 test wrote
    them. Checked here: the file's live entrant is this artifact's label; the scope is that rule applied to the
    same file's per-region table (South Indian and South Pacific apart); every region the J1 outcome also reports
    carries the outcome's own numbers; the prospective record, when there is one, is judged on the same scope.
    Also, as the file has them: the variant with the two southern regions apart (J2) and its 2026 test, per region
    -- J1's own 2026 read is pooled only. No file: an empty scope, so J1 is shown nowhere."""
    from hazardpulse.hurricane import ri_j1

    rel = ri_j1.SCOPE_PATH.relative_to(ri_j1.ROOT).as_posix()
    scope = [str(g) for g in ri_j1.scope(root / rel)]
    d = _read(root, rel)
    if d is None:
        return {"scope": scope, "scope_file": None, "scope_amendment": None, "regions": [], "variant": None}
    if d.get("live_entrant") != label:
        raise EvidenceError(f"{rel} names the live entrant {d.get('live_entrant')!r}; the artifact is {label!r}")
    bar_key = (against_label or "").replace(".", "")
    table = (d.get("development_descriptive") or {}).get(f"{label}_vs_{bar_key}") or {}
    outcome = res.get("by_basin") or {}
    regions = []
    for g, v in table.items():
        a, b, p = _scores(v.get("a")), _scores(v.get("b")), _brier_paired(v.get("b_minus_a"))
        mine = outcome.get(g)
        if mine is not None and (_brier_paired(mine.get(f"{label}_minus_{J1_BAR}")) != p
                                 or _scores(mine.get(label)) != b or _scores(mine.get(J1_BAR)) != a):
            raise EvidenceError(f"{rel} reports {g} otherwise than {HURRICANE_J1_RESULTS}")
        regions.append({"region": g, "name": HURRICANE_BASIN_NAMES.get(g, g), "n": b["n"], "events": b["events"],
                        "mean_forecast": b["mean_forecast"], "bar_mean_forecast": a["mean_forecast"],
                        "in_scope": g in scope, **p})
    rule = sorted(r["region"] for r in regions if r["d_ll"] is not None and r["d_ll"] <= 0)
    if sorted(scope) != rule:
        raise EvidenceError(f"{rel} scopes {label} to {scope}; amendment 14's rule on its per-region table gives {rule}")
    if pros is not None and pros.get("scope") is not None and sorted(pros["scope"]) != sorted(scope):
        raise EvidenceError(f"{HURRICANE_J1_PROSPECTIVE} judges the claim on {pros['scope']}, the scope is {scope}")
    season_key = next((k for k in d if re.fullmatch(r"season_\d{4}", k)), None)
    season = d.get(season_key) or {}
    variant = None
    found = [(m.group(1), v) for k, v in season.items()
             if (m := re.fullmatch(r"by_region_(\w+)_vs_" + re.escape(bar_key), k)) and m.group(1) != label]
    if len(found) == 1:
        [(name, per)] = found
        vs_model = season.get(f"{name}_minus_{label}") or {}
        variant = {"label": name, "carried": d.get("carried") == name, "program": d.get("program"),
                   "prereg_tag": d.get("prereg_tag"), "season": season_key.rsplit("_", 1)[-1],
                   "vs_model": _brier_paired(vs_model),
                   "season_2026": [{"region": g, "name": HURRICANE_BASIN_NAMES.get(g, g),
                                    "n": _scores(v.get("b"))["n"], "events": _scores(v.get("b"))["events"],
                                    **_brier_paired(v.get("b_minus_a"))} for g, v in per.items()]}
    m = re.search(r"amendments? ([\w, ]+)\)", str(d.get("program") or ""))
    return {"scope": scope, "scope_file": rel, "scope_amendment": m.group(1) if m else None, "regions": regions,
            "variant": variant}


# ---------------------------------------------------------------------------
# The research record: every registered test since the served models were chosen, carried or not
# ---------------------------------------------------------------------------

RESEARCH_HURRICANE = {
    "h8": "results/calibration/hurricane_ri_h8.json",
    "h9": "results/calibration/hurricane_ri_h9.json",
    "fusion": "results/calibration/hurricane_ri_fusion.json",
    "g1": "results/calibration/hurricane_ri_g1.json",
    "adt": "results/goes/adt_check.json",
}


def _family_by_candidate(root: Path) -> dict[str, dict]:
    """Our carried RI models by the development candidate each froze (``V2`` -> v10.1, ``H8`` -> v10.4)."""
    from hazardpulse.hurricane import ri_v10

    out = {}
    for entrant_key, rel, _what, _base in HURRICANE_V10_FAMILY:
        if (root / rel).exists():
            art, version = ri_v10.load(root / rel)
            cand = (art.get("provenance") or {}).get("candidate")
            out[str(cand)] = {"label": family_label(art, entrant_key), "model_version": version,
                              "prereg_tag": (art.get("provenance") or {}).get("prereg_tag"),
                              "dev_period": dev_period(art.get("provenance") or {})[0]}
    return out


def _key_period(d: dict, prefix: str) -> str | None:
    """A period a results file states in a key name: ``hss_2024_2026`` -> ``2024-2026``."""
    for k in d:
        m = re.fullmatch(re.escape(prefix) + r"_(\d{4})_(\d{4})", k)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return None


def _paired(p: dict | None) -> dict:
    p = p or {}
    return {"d_ll": _finite(p.get("d_ll")), "d_ll_ci": _ci(p.get("d_ll_ci")),
            "d_brier4": _finite(p.get("d_brier4")), "d_brier4_ci": _ci(p.get("d_brier4_ci"))}


def research_record(root: Path = ROOT) -> list[dict]:
    """The registered tests run on the hurricane RI models after v10.1 was frozen, plus the input check G1 had to
    pass, each read from the file its run wrote. A test's decision is the file's own (``carried``, ``claim``,
    ``all_passed``); the model a carried candidate became is found by the candidate its artifact froze."""
    by_cand = _family_by_candidate(root)
    out = []
    for key in ("h8", "h9", "fusion", "g1"):
        rel = RESEARCH_HURRICANE[key]
        d = _read(root, rel)
        if d is None:
            continue
        m = re.search(r"amendments? ([\w, ]+)\)", str(d.get("program") or ""))
        row = {"key": key, "file": rel, "amendment": m.group(1) if m else None, "prereg_tag": d.get("prereg_tag")}
        if key == "fusion":
            dev = ((d.get("results") or {}).get("dev") or {})
            vb = dev.get("vs_best_single") or {}
            seasons = d.get("seasons") or {}
            dev_years = list(seasons.get("dev") or [])
            row.update({"kind": "fusion", "best_single": d.get("best_single_on_choose"),
                        "best_single_label": (by_cand.get(str(d.get("best_single_on_choose"))) or {}).get("label"),
                        "n": vb.get("n"), "d_ll": _finite(vb.get("d_ll")), "d_ll_ci": _ci(vb.get("d_ll_ci")),
                        "claim": bool(vb.get("claim")), "n_sources": len(d.get("sources") or []),
                        "dev_period": f"{dev_years[0]}-{dev_years[-1]}" if dev_years else None,
                        "choose": list(seasons.get("choose") or [])})
        else:
            dv = d.get("development") or {}
            ctl = [m.group(1) for k in d if (m := re.fullmatch(r"control_(\w+)_log_loss", k))]
            control = ctl[0] if len(ctl) == 1 else None
            arms = [k for k in dv if k not in ("paired", control)]
            if control not in dv or len(arms) != 1:
                raise EvidenceError(f"{rel}: cannot tell its control from its candidate ({sorted(dv)})")
            cand = arms[0]
            carried = d.get("carried")
            names = next((v for k, v in d.items() if k.endswith("_names") and isinstance(v, list)), None)
            row.update({"kind": "candidate", "control": control, "candidate": cand, "carried": carried == cand,
                        "control_label": (by_cand.get(str(control)) or {}).get("label"),
                        "dev_period": (by_cand.get(str(control)) or {}).get("dev_period"),
                        "n_features": len(names) if names else None,
                        "became": ((by_cand.get(str(cand)) or {}).get("label") if carried == cand else None),
                        "n": ((dv.get(cand) or {}).get("summary_30") or {}).get("n"),
                        "events": ((dv.get(cand) or {}).get("summary_30") or {}).get("events"),
                        **_paired(dv.get("paired")),
                        "further_2026": _paired(((d.get("further_read_2026") or {}).get("results") or {}).get("paired")),
                        "noise": [_finite(s.get("d_ll")) for s in d.get("noise_control") or []] or None})
        out.append(row)
    adt = _read(root, RESEARCH_HURRICANE["adt"])
    if adt is not None:
        meas, gates = adt.get("measured") or {}, adt.get("gates") or {}
        held = _key_period(meas, "hss")
        weak = next((m.group(1) for k in meas if (m := re.fullmatch(r"eye_frac_below_(\d+)kt", k))), None)
        hss_key = f"hss_{held.replace('-', '_')}" if held else None
        weak_key = f"eye_frac_below_{weak}kt" if weak else None
        out.append({"key": "adt", "kind": "input_check", "file": RESEARCH_HURRICANE["adt"],
                    "passed": bool(adt.get("all_passed")), "matched": meas.get("matched"),
                    "matched_gate": gates.get("matched"), "held_out": held,
                    "hss": _finite(meas.get(hss_key)), "hss_gate": _finite(gates.get(hss_key)),
                    "weak_kt": int(weak) if weak else None,
                    "weak_eye": _finite(meas.get(weak_key)), "weak_eye_gate": _finite(gates.get(weak_key)),
                    "centre_km": _finite(meas.get("centre_km_median")),
                    "centre_km_gate": _finite(gates.get("centre_km_median")),
                    "teye_spearman": _finite(meas.get("teye_spearman")),
                    "teye_spearman_gate": _finite(gates.get("teye_spearman"))})
    return out + _jtwc_research_rows(root)


def _jtwc_research_rows(root: Path) -> list[dict]:
    """The JTWC-basin tests: amendment 13's (J1 against the published v8.2) and amendment 14's (a variant against
    J1), each read from the file its run wrote. The label the carried candidate became is its artifact's, and the
    regions it is shown in are the library's scope (``ri_j1.scope``)."""
    from hazardpulse.hurricane import ri_j1

    out: list[dict] = []
    v82_label = version_label((_read(root, HURRICANE_V82_SERVED) or {}).get("model_version"))
    art_rel = ri_j1.MODEL_PATH.relative_to(ri_j1.ROOT).as_posix()
    became = None
    if (root / art_rel).exists():
        became = str(ri_j1.load(root / art_rel)[0].get("label") or "") or None
    scope_rel = ri_j1.SCOPE_PATH.relative_to(ri_j1.ROOT).as_posix()
    scope = [HURRICANE_BASIN_NAMES.get(str(g), str(g)) for g in ri_j1.scope(root / scope_rel)]

    def amendment(prog: Any) -> str | None:
        m = re.search(r"amendments? ([\w, ]+)\)", str(prog or ""))
        return m.group(1) if m else None
    d = _read(root, HURRICANE_J1_RESULTS)
    if d is not None:
        cands = d.get("candidates") or {}
        cand = next((k for k in cands if k != J1_CONTROL), None)
        dv = (d.get("development") or {}).get(cand) or {}
        carried = d.get("carried") == cand
        out.append({"key": "j1", "kind": "vs_published", "file": HURRICANE_J1_RESULTS,
                    "amendment": amendment(d.get("program")), "prereg_tag": d.get("prereg_tag"),
                    "candidate": cand, "carried": carried, "became": became if carried and became == cand else None,
                    "control_label": v82_label, "dev_period": _span_of(d.get("by_fold")),
                    "n_features": len(cands.get(cand) or []) or None, "n": dv.get("n"), "events": dv.get("events"),
                    **_brier_paired(d.get(f"{cand}_minus_{J1_BAR}")),
                    "ir": _brier_paired(d.get(f"{cand}_minus_{J1_CONTROL}")),
                    "noise": [_finite(s.get("d_ll")) for s in d.get("noise_control") or []] or None,
                    "scope": scope})
    j2 = _read(root, scope_rel)
    if j2 is not None and became:
        var = next((k for k in j2.get("names") or {} if k != became), None)
        season_key = next((k for k in j2 if re.fullmatch(r"season_\d{4}", k)), None)
        sea = j2.get(season_key) or {}
        met = _scores((sea.get("metrics") or {}).get(var))
        out.append({"key": "j2", "kind": "variant", "file": scope_rel, "amendment": amendment(j2.get("program")),
                    "prereg_tag": j2.get("prereg_tag"), "candidate": var, "incumbent": became,
                    "carried": j2.get("carried") == var, "live_entrant": j2.get("live_entrant"),
                    "season": season_key.rsplit("_", 1)[-1] if season_key else None,
                    "n": met["n"], "events": met["events"], **_brier_paired(sea.get(f"{var}_minus_{became}")),
                    "scope": scope})
    return out


def all_evidence(root: Path = ROOT, errors: list[str] | None = None) -> dict[str, Any]:
    """Every hazard's evidence, and the research record. A hazard whose results contradict its served artifact is
    ``None`` (the pages then say no result is bound) and its reason is appended to ``errors`` -- one bad file must
    not stop the live scorers publishing forecasts; the tests call each hazard directly, so CI still fails on it."""
    out: dict[str, Any] = {}
    for name, fn in (("tornado", tornado_evidence), ("earthquake", earthquake_evidence),
                     ("hurricane", hurricane_evidence), ("research", research_record)):
        try:
            out[name] = fn(root)
        except EvidenceError as exc:
            out[name] = None
            if errors is not None:
                errors.append(f"{name}: {exc}")
    return out
