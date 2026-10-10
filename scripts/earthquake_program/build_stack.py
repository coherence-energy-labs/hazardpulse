#!/usr/bin/env python3
"""Serve the stack the program carried on C0 (docs/EARTHQUAKE_FORECAST_PROGRAM.md).

    python scripts/earthquake_program/build_stack.py                  # S1 = C0 + GEAR1 (section 10, amendment E1)
    python scripts/earthquake_program/build_stack.py --candidate S2   # S2 = S1 + d z g (section 12.2, amendment E4)

Writes the stack file (S1: ``results/models/earthquake_gear1_stack_v1.json``; S2: ``..._v2.json``): the coefficients
fitted on CHOOSE and GEAR1's 30-day log map, bound to the served C0 artifact by its model_version, and
``results/earthquake_program/stack_artifact.json``, the record of the stack just built. The C0 artifact is not touched.
Parity, enforced before the file is accepted (it is removed if any fails):

1. formula -- the stack applied to C0's cached forecasts reproduces the evaluated grid at every DEV issue time
              (max abs difference recorded; must be <= 1e-12);
2. live    -- the live code path (C0 artifact from its frozen catalog + the C0 build's simulated live fetch, then
              the stack) gives back the evaluated grid at three FINAL issue times (max abs difference <= LIVE_TOL).

For S2 it also writes ``results/earthquake_program/gear1_served.json``: section 10's report for the served
candidate (S2 against S0, S1, C0, A, B and D on every split), after checking that S2's DEV and FINAL information gain
reproduce the registered run (``gear1_e4.json``) and S1's reproduce E1's (``gear1_e1.json``), each to 1e-9.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_artifact as ba  # noqa: E402
import common as C  # noqa: E402
import evaluate as ev  # noqa: E402
import gear1_e4 as e4  # noqa: E402
import gear1_stack as gs  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

BASE = C.REPO / "results" / "models" / "earthquake_operational_v1.json"
# C0's live path matches its evaluation to 1e-12 (build_artifact.TREE_TOL); the logit stack can
# amplify that on near-zero probabilities, so the stack's live path is held to 1e-9
LIVE_TOL = 1e-9
REPRO_TOL = 1e-9
CANDIDATES = {
    "S1": {"report": "gear1_e1.json", "carried": ("S1_replaces_C0",), "out": "earthquake_gear1_stack_v1.json",
           "model_name": "eq_operational_S1_gear1_v1", "program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10 (amendment E1)",
           "prereg_tag": "prereg-earthquake-gear1", "decided_on": "DEV (2021-2022): S1 - S0 information gain interval above 0"},
    "S2": {"report": "gear1_e4.json", "carried": ("S2_replaces_S1",), "out": "earthquake_gear1_stack_v2.json",
           "model_name": "eq_operational_S2_gear1_v2", "program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 12 (amendment E4)",
           "prereg_tag": "prereg-earthquake-e4", "decided_on": "DEV (2021-2022): S2 - S1 information gain interval above 0"},
}


def _predict(cand: str, co: dict, p_c0: np.ndarray, g_log10: np.ndarray) -> np.ndarray:
    """The evaluated candidate's grid, by the evaluation's own code."""
    z = gs.logit(p_c0)
    if cand == "S1":
        return gs.predict_logistic(np.array([co["a"], co["c"], co["b"]]), z, g_log10)
    return e4.predict_s2(np.array([co["a"], co["c"], co["b"], co["d"]]), z, g_log10)


def served_report(co2: dict, e1: dict, e4_rep: dict, g_log10: np.ndarray) -> dict:
    """Section 10's report for S2, the served candidate: every split, S2 against S0, S1, C0, A, B and D. S2's and S1's
    information gain must reproduce their registered runs, or the build stops."""
    c1 = e1["coefficients_fitted_on_choose"]
    th0, th1 = np.array([c1["S0"]["a"], c1["S0"]["c"]]), np.array([c1["S1"]["a"], c1["S1"]["c"], c1["S1"]["b"]])
    p0 = float(json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))["fit"]["p0"])
    events = C.load_events(4.5)
    out = {"program": CANDIDATES["S2"]["program"], "prereg_tag": CANDIDATES["S2"]["prereg_tag"],
           "candidate": "S2", "decided_by": "results/earthquake_program/gear1_e4.json",
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "gear1_sha256": e4_rep["gear1_sha256"], "coefficients_fitted_on_choose": {"S0": c1["S0"], "S1": c1["S1"], "S2": co2},
           "splits": {}}
    for split in gs.SPLITS:
        issue = C.split_issue_times(split)
        Y = C.binary_targets(events, issue)
        p_c0 = C.load_pred("C0", split)
        z = gs.logit(p_c0)
        P = {"S2": _predict("S2", co2, p_c0, g_log10), "S1": gs.predict_logistic(th1, z, g_log10),
             "S0": gs.predict_logistic(th0, z, None), "C0": p_c0, **{k: C.load_pred(k, split) for k in ("A", "B", "D")}}
        sc = C.SplitScorer(issue, Y, p0=p0)
        act = ev.active_masks(issue)
        sc_act = C.SplitScorer(issue, Y, p0=p0, mask=act)
        st = {k: sc.stats(v) for k, v in P.items()}
        res = {"role": {"choose": "fitted here (in-sample for the coefficients)", "dev": "decided S2 (gear1_e4.json)",
                        "final": "declared second read: reported, never used to decide"}[split],
               "n_issue_times": int(issue.size), "n_positive": int(Y.sum()), "n_cell_times": int(Y.size),
               "positives_in_active_cells": int((Y & act).sum()),
               "first_issue": str(np.datetime64(int(issue[0]), "s")), "last_issue": str(np.datetime64(int(issue[-1]), "s")),
               "models": {k: sc.summary(s) for k, s in st.items()},
               "paired": {f"S2-{o}": sc.paired(st["S2"], st[o]) for o in ("S1", "S0", "C0", "A", "B", "D")}}
        for k in ("S2", "S1", "C0"):
            res["models"][k]["auc_active_cells"] = sc_act.summary(sc_act.stats(P[k]))["auc"]
        out["splits"][split] = res
        for k, want in (("S2", e4_rep["splits"][split]["models"]["S2"]), ("S1", e1["splits"][split]["models"]["S1"])):
            got = res["models"][k]["ig_per_target"]["value"]
            if abs(got - want["ig_per_target"]["value"]) > REPRO_TOL:
                raise SystemExit(f"{split}: {k}'s information gain {got!r} does not reproduce its registered run "
                                 f"{want['ig_per_target']['value']!r}")
        print(f"{split}: S2 IG {res['models']['S2']['ig_per_target']['value']:.6f} reproduces its run; "
              f"S2-A dAUC {res['paired']['S2-A']['auc']['diff']:+.5f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), default="S1")
    cand = ap.parse_args(argv).candidate
    cfg = CANDIDATES[cand]
    out_path = C.REPO / "results" / "models" / cfg["out"]
    rep = json.loads((C.RESULTS / cfg["report"]).read_text(encoding="utf-8"))
    if not all(rep["carried_rule"][k] for k in cfg["carried"]):
        raise SystemExit(f"the registered rule did not carry {cand}: nothing to serve")
    cells = json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))
    if cells["sha256"] != rep["gear1_sha256"]:
        raise SystemExit("gear1_cells.json is not the map the evaluation used")
    base = of.load_artifact(BASE)
    record = json.loads((C.RESULTS / "artifact.json").read_text(encoding="utf-8"))
    if record.get("model_version") != base.model_version:
        raise SystemExit(f"served base {base.model_version} is not the recorded {record.get('model_version')}")
    co = rep["coefficients_fitted_on_choose"][cand]
    g_log10 = gs.gear1_log_map(cells["cells_per_year"])
    dev, fin = rep["splits"]["dev"], rep["splits"]["final"]
    report = served_report(co, json.loads((C.RESULTS / "gear1_e1.json").read_text(encoding="utf-8")), rep, g_log10) \
        if cand == "S2" else None
    provenance = {"program": cfg["program"], "prereg_tag": cfg["prereg_tag"], "fitted_on": "CHOOSE (2018-2020 issue times)",
                  "decided_on": cfg["decided_on"], "evaluation": "results/earthquake_program/" + cfg["report"],
                  "candidate": cand,
                  "gear1": {"source": cells["source"], "sha256": cells["sha256"], "citation": cells["citation"],
                            "license": cells["license"], "global_total_per_year": cells["global_total"]},
                  "g_floor_per_30d": gs.G_FLOOR,
                  "dev_ig_per_target": dev["models"][cand]["ig_per_target"],
                  "final_ig_per_target_second_read": fin["models"][cand]["ig_per_target"]}
    if cand == "S1":                 # S1's file as published on 2026-10-03: its provenance is part of its hash
        provenance = {k: v for k, v in provenance.items() if k not in ("evaluation", "candidate")}
    version = of.write_stack(out_path, model_name=cfg["model_name"], base_model_version=base.model_version,
                             a=co["a"], c=co["c"], b=co["b"], d=co.get("d"), g_log10=g_log10, provenance=provenance)
    stack = of.load_stack(out_path, base)
    try:
        p_dev = C.load_pred("C0", "dev")
        worst_formula = float(np.max(np.abs(of.apply_stack(stack, p_dev) - _predict(cand, co, p_dev, g_log10))))
        if worst_formula > 1e-12:
            raise SystemExit(f"formula parity failed: {worst_formula:.2e}")
        issue = C.split_issue_times("final")
        want_fin = _predict(cand, co, C.load_pred("C0", "final"), g_log10)
        raw25 = C.load_raw("program_catalog_m25.npz")
        worst_live = 0.0
        for k in (0, len(issue) // 2, len(issue) - 1):
            t = float(issue[k])
            out = of.forecast_with_stack(base, stack, ba._simulated_live(raw25, t), t)
            worst_live = max(worst_live, float(np.max(np.abs(out["probability"] - want_fin[k]))))
        if worst_live > LIVE_TOL:
            raise SystemExit(f"live parity failed: {worst_live:.2e}")
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    C.write_json("stack_artifact.json", {"model_version": version, "file": "results/models/" + out_path.name,
                                         "base_model_version": base.model_version, "candidate": cand,
                                         "parity": {"formula_max_abs_dev": worst_formula,
                                                    "live_max_abs_final_3_issue_times": worst_live}})
    if report is not None:
        C.write_json("gear1_served.json", report)
    print(f"{out_path.name}: {version} on {base.model_version}; parity formula {worst_formula:.1e}, live {worst_live:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
