"""Export final v3 tornado models (tornado_lab.py final / final_product) as served NumPy payloads.

    PYTHONPATH=src python scripts/audit_20261001/export_v3_payload.py            # every served model

For each final run: the booster's trees (hazardpulse.tornado.lgbm_payload: node arrays incl. node
values for per-storm contributions), its Platt calibration, a Venn-Abers band fitted on the run's
leave-one-year-out scores when they were saved, and provenance with the run's 2025 numbers. Each
payload is written ONLY if it reproduces the final run's own 2025 probabilities on every 2025
storm observation (saved float32: agreement to 2e-6).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf
from hazardpulse.trust.venn_abers import VennAbersCalibrator

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tornado_lab", Path(__file__).with_name("tornado_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

# final run name -> (served file, what it forecasts)
SERVED = {
    "v3_plus_W": ("tornado_v3_w.json", "P(a tornado from THIS storm within 60 min); inputs include NWS warning state"),
    "v3_primary": ("tornado_v3.json", "P(a tornado from THIS storm within 60 min); no NWS input (fallback)"),
    "v3_plus_W_30": ("tornado_v3_w_30.json", "P(a tornado from THIS storm within 30 min)"),
    "v3_plus_W_90": ("tornado_v3_w_90.json", "P(a tornado from THIS storm within 90 min)"),
    "v3_plus_W_ef2": ("tornado_v3_w_ef2.json", "P(an EF2+ tornado from THIS storm within 60 min)"),
}
EVENT_BY_LABEL = {
    "storm_30": "a tornado report starts within 10 km of THIS storm's tracked polygon within 30 min",
    "storm_60": "a tornado report starts within 10 km of THIS storm's tracked polygon within 60 min",
    "storm_90": "a tornado report starts within 10 km of THIS storm's tracked polygon within 90 min",
    "storm_60_ef2": "an EF2+ tornado report starts within 10 km of THIS storm's tracked polygon within 60 min",
}


def export(name: str) -> str:
    out_name, meaning = SERVED[name]
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
        "program": "docs/TORNADO_MODEL_PROGRAM.md", "final_run": name, "forecasts": meaning,
        "label": exp["label"], "event": EVENT_BY_LABEL[exp["label"]], "label_family": sf.PRIMARY_FAMILY,
        "trained": "2020-10-15..2024-12-31, fixed rounds chosen by early stopping on validation 2023",
        "rounds": final["rounds"], "blocks": exp["blocks"], "params": exp["params"],
        "final_2025": {"auc": f25["auc"], "auc_ci": f25["auc_ci"], "bss": f25["bss"], "pr_auc": f25["pr_auc"],
                       "n": f25["n"], "pos": f25["pos"]},
    })
    oof = lab.LAB / "oof" / f"{name}.npz"
    if oof.exists():
        with np.load(oof) as z:
            va = VennAbersCalibrator(min_calibration=200, max_groups=512).fit(z["score"], z["y"])
        payload["interval"] = va.to_dict()
        payload["interval_fitted_on"] = "leave-one-year-out raw scores 2021-2024"
    Xf, _, _ = lab.load("final")
    Wf = lab.load_w("final") if "W" in exp["blocks"] else None
    want = np.load(lab.LAB / "preds" / f"{name}_final.npy").astype(np.float64)
    got = np.concatenate([lp.predict_proba(payload, lab.matrix(Xf, Wf, slice(s, s + 200_000), cols, Wf is not None))
                          for s in range(0, Xf.shape[0], 200_000)])
    worst = float(np.max(np.abs(got - want)))
    if worst > 2e-6:
        raise SystemExit(f"{name}: payload disagrees with the final run by {worst:.2e}: not exported")
    out = ROOT / "results" / "models" / out_name
    version = lp.save(payload, out)
    print(f"{out_name}: {version}, {payload['n_trees']} trees, {len(names)} inputs, band="
          f"{'yes' if payload.get('interval') else 'no'}, max |payload - run| over {len(want)} rows = {worst:.2e}",
          flush=True)
    return version


def main() -> int:
    names = sys.argv[1:] or list(SERVED)
    for n in names:
        if (lab.OUT / f"final_{n}.json").exists():
            export(n)
        else:
            print(f"{n}: no final run yet, skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
