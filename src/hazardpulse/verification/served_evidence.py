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

import json
import math
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
     "note": "measured before amendment 8 corrected the HRRR timing (both arms alike)"},
    {"what": "the coherence PDE solution against a Gaussian-smoothing control (9 km)",
     "kind": "paired", "file": "results/lab/compare_b_all_C9pde_vs_b_all_C9gauss_val_storm_60.json",
     "note": "measured before amendment 8 corrected the HRRR timing (both arms alike)"},
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
}

HURRICANE_SERVED = "results/models/hurricane_ri_stack_v1.json"
HURRICANE_FINAL = "results/calibration/hurricane_ri_stack_final.json"
HURRICANE_ADVERSARY = "results/calibration/hurricane_ri_stack_adversary.json"
# the model served outside the NHC basins, and its own temporal hold-out (scripts/evaluate_hurricane_ri.py)
HURRICANE_V82_EVALUATION = "results/calibration/hurricane_ri_evaluation.json"
HURRICANE_PROGRAM = "docs/HURRICANE_RI_PROGRAM.md"
HURRICANE_CANDIDATES = {
    "A": "NOAA DTOPS (the NHC's deterministic-to-probabilistic RI guidance)",
    "B": "NOAA SHIPS-RII",
    "C": "NOAA RIOC (the RI consensus)",
}
_HURRICANE_CLAIM_NAMES = {
    "better_than_SHIPS_RII_B": "NOAA SHIPS-RII",
    "better_than_RIOC_C": "NOAA RIOC consensus",
    "better_than_v8_2": "our previous model (v8.2)",
}


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
    return ev


# ---------------------------------------------------------------------------
# Earthquake
# ---------------------------------------------------------------------------

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
    claims = []
    for key, name in _HURRICANE_CLAIM_NAMES.items():
        c = (final.get("claims") or {}).get(key)
        if c is None:
            continue
        claims.append({"against": name, "better": bool(c.get("claim")), "worse": bool(c.get("worse")),
                       "delta_log_loss_ci": _ci(c.get("delta_log_loss_ci95")),
                       "delta_brier_ci": _ci(c.get("delta_brier_ci95"))})
    adv = _read(root, HURRICANE_ADVERSARY) or {}
    cases = final.get("final_cases") or {}
    v82 = (((final.get("results") or {}).get("v82_subset") or {}).get("forecasts") or {}).get("v8_2") or {}
    evaluation = _read(root, HURRICANE_V82_EVALUATION) or {}
    v82_heldout = (evaluation.get("results") or {}).get("C_v8_2_heldout_newton")
    origin = evaluation.get("origin") or {}
    other_basins = None
    if v82_heldout:
        test_years = origin.get("test") or ["?", "?"]
        other_basins = {
            "model": "hurricane_ri_v8_2",
            "test": {"period": f"{test_years[0]}-{test_years[1]} NHC cases (held out)",
                     "when": f"{test_years[0]}-{test_years[1]}",
                     "n": int(v82_heldout.get("n") or 0), "auc": _finite(v82_heldout.get("auc")),
                     "auc_ci": _ci((v82_heldout.get("ci95") or {}).get("auc")),
                     "bss": _finite(v82_heldout.get("bss_vs_climatology"))},
        }
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
        "v8_2_same_cases": {"auc": _finite(v82.get("auc")), "bss": _finite(v82.get("bss"))} if v82 else None,
        "claims": claims,
        "adversary": {"verdict": adv.get("verdict"), "statement": adv.get("surviving_statement")} if adv else None,
        "other_basins": other_basins,
    }


def all_evidence(root: Path = ROOT, errors: list[str] | None = None) -> dict[str, dict | None]:
    """Every hazard's evidence. A hazard whose results contradict its served artifact is ``None``
    (the pages then say no result is bound) and its reason is appended to ``errors`` -- one bad
    file must not stop the live scorers publishing forecasts; the tests call each hazard directly,
    so CI still fails on it."""
    out: dict[str, dict | None] = {}
    for name, fn in (("tornado", tornado_evidence), ("earthquake", earthquake_evidence),
                     ("hurricane", hurricane_evidence)):
        try:
            out[name] = fn(root)
        except EvidenceError as exc:
            out[name] = None
            if errors is not None:
                errors.append(f"{name}: {exc}")
    return out
