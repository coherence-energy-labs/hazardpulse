"""Ladder arm C (all fixes) and its ablations, on the rebuilt cache.

Run from the FIX worktree (hp) with PYTHONPATH=src:

    python ladder_new.py C   <out_dir> [--save-model <path>]   # every fix
    python ladder_new.py C0  <out_dir>                         # every fix, torsion forced to zero
    python ladder_new.py C18 <out_dir>                         # every fix, but every storm reads 18Z (the leak)
"""
import sys
from pathlib import Path

import numpy as np

import hazardpulse
from hazardpulse.tornado import coherence_engine as ce
from hazardpulse.tornado import definitive_model as dm

arm, out_dir = sys.argv[1], Path(sys.argv[2])
save = sys.argv[sys.argv.index("--save-model") + 1] if "--save-model" in sys.argv else None
print("hazardpulse from", hazardpulse.__file__, "arm", arm, flush=True)

policy = "latest_at_or_before"
if arm == "C0":
    ce.compute_tilting_torsion = lambda tau, us, vs: np.zeros_like(tau, dtype=np.float32)
elif arm == "C18":
    policy = "fixed_18z"

res = dm.main(output_dir=out_dir, verbose=True, analysis_policy=policy)
full = res["full"]
print("ARM", arm, "test AUC full/enhanced/baseline:",
      round(full["auc"], 4), round(res["enhanced"]["auc"], 4), round(res["baseline"]["auc"], 4),
      "CI", round(full["bootstrap_ci"]["ci_lo"], 4), round(full["bootstrap_ci"]["ci_hi"], 4),
      "BSS", round(full["bss"], 4), "excluded", res["data_summary"]["total_excluded"], flush=True)
imp = {d["feature"]: d["importance"] for d in res["feature_importance"]["full_model"]}
print("importance torsion", imp.get("torsion"), "torsion_x_srh", imp.get("torsion_x_srh"), flush=True)
if save:
    dm.save_model(
        res["_models"]["full"], res["_normalizers"]["full"], dm.ALL_FEATURE_NAMES_FULL, save,
        variant="full", calibration=res["_calibrations"]["full"],
        provenance={
            "trained": res["timestamp"], "model": res["model"],
            "audit_guarantees": res["audit_guarantees"],
            "test_auc": full["auc"],
            "test_auc_ci_day_clustered": [full["bootstrap_ci"]["ci_lo"], full["bootstrap_ci"]["ci_hi"]],
            "test_bss_calibrated": full["bss"], "test_base_rate": full["base_rate"],
        },
    )
