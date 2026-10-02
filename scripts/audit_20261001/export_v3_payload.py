"""Export the final v3 tornado model (tornado_lab.py final) as the served NumPy payload.

    PYTHONPATH=src python scripts/audit_20261001/export_v3_payload.py [primary|plus_W]

Reads the final run's booster and Platt calibration, writes results/models/tornado_v3[_w].json
(hazardpulse.tornado.lgbm_payload format: trees as node arrays, scored in pure NumPy) and
REFUSES to write unless the payload reproduces the final run's own 2025 probabilities for every
2025 storm observation (saved float32, so agreement is checked to 2e-6 absolute).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tornado_lab", Path(__file__).with_name("tornado_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "primary"
    name = "v3_primary" if which == "primary" else "v3_plus_W"
    final = json.loads((lab.OUT / f"final_{name}.json").read_text(encoding="utf-8"))
    import lightgbm as lgb
    bst = lgb.Booster(model_file=final["model_file"])
    exp = final["exp"]
    cols = lab.cols_for(exp["blocks"])
    names = [lab.NAMES[i] for i in cols] + (list(lab.W_NAMES) if "W" in exp["blocks"] else [])
    cal = {"method": "platt", "a": float(final["calibration"]["a"]), "b": float(final["calibration"]["b"]),
           "fitted_on": "leave-one-year-out scores 2021-2024 (2020-10..12 with 2021)"}
    f25 = final["final_2025"]
    payload = lp.export_booster(bst, names, calibration=cal, provenance={
        "program": "docs/TORNADO_MODEL_PROGRAM.md", "label": exp["label"], "label_family": sf.PRIMARY_FAMILY,
        "event": "a tornado report starts within 10 km of THIS storm's tracked polygon within 60 min",
        "trained": "2020-10-15..2024-12-31, fixed rounds chosen on validation 2023",
        "rounds": final["rounds"], "blocks": exp["blocks"], "params": exp["params"],
        "final_2025": {"auc": f25["auc"], "auc_ci": f25["auc_ci"], "bss": f25["bss"], "n": f25["n"], "pos": f25["pos"]},
    })
    # parity with the run's own 2025 probabilities, every row
    Xf, _, _ = lab.load("final")
    Wf = lab.load_w("final") if "W" in exp["blocks"] else None
    want = np.load(lab.LAB / "preds" / f"{name}_final.npy").astype(np.float64)
    got = np.concatenate([lp.predict_proba(payload, lab.matrix(Xf, Wf, slice(s, s + 200_000), cols, Wf is not None))
                          for s in range(0, Xf.shape[0], 200_000)])
    worst = float(np.max(np.abs(got - want)))
    if worst > 2e-6:
        raise SystemExit(f"payload disagrees with the final run by {worst:.2e}: not exported")
    out = ROOT / "results" / "models" / ("tornado_v3.json" if which == "primary" else "tornado_v3_w.json")
    version = lp.save(payload, out)
    print(f"{out.name}: {version}, {payload['n_trees']} trees, {len(names)} inputs, "
          f"max |payload - run| over {len(want)} rows = {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
