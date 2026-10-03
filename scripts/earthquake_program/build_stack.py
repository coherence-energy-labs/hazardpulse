#!/usr/bin/env python3
"""Amendment E1 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10): serve S1 = C0 + GEAR1.

    python scripts/earthquake_program/build_stack.py      # after gear1_stack.py carried S1

Writes ``results/models/earthquake_gear1_stack_v1.json``: the S1 coefficients fitted on CHOOSE
and GEAR1's 30-day log map, bound to the served C0 artifact by its model_version. The C0 artifact
is not touched. Parity, enforced before the file is accepted (it is removed if any fails):

1. formula -- the stack applied to C0's cached forecasts reproduces the evaluated S1 grid at
              every DEV issue time (max abs difference recorded; must be <= 1e-12);
2. live    -- the live code path (C0 artifact from its frozen catalog + the C0 build's simulated
              live fetch, then the stack) gives back the evaluated S1 grid at three FINAL issue
              times (max abs difference <= LIVE_TOL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_artifact as ba  # noqa: E402
import common as C  # noqa: E402
import gear1_stack as gs  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

BASE = C.REPO / "results" / "models" / "earthquake_operational_v1.json"
OUT = C.REPO / "results" / "models" / "earthquake_gear1_stack_v1.json"
# C0's live path matches its evaluation to 1e-12 (build_artifact.TREE_TOL); the logit stack can
# amplify that on near-zero probabilities, so the stack's live path is held to 1e-9
LIVE_TOL = 1e-9


def main() -> int:
    rep = json.loads((C.RESULTS / "gear1_e1.json").read_text(encoding="utf-8"))
    if not rep["carried_rule"]["S1_replaces_C0"]:
        raise SystemExit("the registered rule did not carry S1: nothing to serve")
    cells = json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))
    if cells["sha256"] != rep["gear1_sha256"]:
        raise SystemExit("gear1_cells.json is not the map the evaluation used")
    base = of.load_artifact(BASE)
    record = json.loads((C.RESULTS / "artifact.json").read_text(encoding="utf-8"))
    if record.get("model_version") != base.model_version:
        raise SystemExit(f"served base {base.model_version} is not the recorded {record.get('model_version')}")
    co = rep["coefficients_fitted_on_choose"]["S1"]
    g_log10 = gs.gear1_log_map(cells["cells_per_year"])
    dev, fin = rep["splits"]["dev"], rep["splits"]["final"]
    version = of.write_stack(
        OUT, model_name="eq_operational_S1_gear1_v1", base_model_version=base.model_version,
        a=co["a"], c=co["c"], b=co["b"], g_log10=g_log10,
        provenance={"program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10 (amendment E1)",
                    "prereg_tag": "prereg-earthquake-gear1", "fitted_on": "CHOOSE (2018-2020 issue times)",
                    "decided_on": "DEV (2021-2022): S1 - S0 information gain interval above 0",
                    "gear1": {"source": cells["source"], "sha256": cells["sha256"], "citation": cells["citation"],
                              "license": cells["license"], "global_total_per_year": cells["global_total"]},
                    "g_floor_per_30d": gs.G_FLOOR,
                    "dev_ig_per_target": dev["models"]["S1"]["ig_per_target"],
                    "final_ig_per_target_second_read": fin["models"]["S1"]["ig_per_target"]})
    stack = of.load_stack(OUT, base)
    try:
        theta = np.array([co["a"], co["c"], co["b"]])
        worst_formula = 0.0
        p_dev = C.load_pred("C0", "dev")
        want = gs.predict_logistic(theta, gs.logit(p_dev), g_log10)
        worst_formula = float(np.max(np.abs(of.apply_stack(stack, p_dev) - want)))
        if worst_formula > 1e-12:
            raise SystemExit(f"formula parity failed: {worst_formula:.2e}")
        issue = C.split_issue_times("final")
        p_fin = C.load_pred("C0", "final")
        want_fin = gs.predict_logistic(theta, gs.logit(p_fin), g_log10)
        raw25 = C.load_raw("program_catalog_m25.npz")
        worst_live = 0.0
        for k in (0, len(issue) // 2, len(issue) - 1):
            t = float(issue[k])
            out = of.forecast_with_stack(base, stack, ba._simulated_live(raw25, t), t)
            worst_live = max(worst_live, float(np.max(np.abs(out["probability"] - want_fin[k]))))
        if worst_live > LIVE_TOL:
            raise SystemExit(f"live parity failed: {worst_live:.2e}")
    except BaseException:
        OUT.unlink(missing_ok=True)
        raise
    C.write_json("stack_artifact.json", {"model_version": version, "file": "results/models/" + OUT.name,
                                         "base_model_version": base.model_version,
                                         "parity": {"formula_max_abs_dev": worst_formula,
                                                    "live_max_abs_final_3_issue_times": worst_live}})
    print(f"{OUT.name}: {version} on {base.model_version}; parity formula {worst_formula:.1e}, live {worst_live:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
