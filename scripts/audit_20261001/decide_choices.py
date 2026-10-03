"""Apply the program's declared validation rules (amendment 4) to lab results and write the choice files.

    python decide_choices.py family_search     # steps 2-3 -> CHOICES/best_config.json
    python decide_choices.py steps48           # steps 4-8 -> best_config.json (+calibration, negatives,
                                               #   label) and chosen_on_validation.json

Reads results from HAZARDPULSE_LAB_OUT, writes into HAZARDPULSE_CHOICES_DIR. Mechanical rules only:
highest validation AUC (calibration: lowest validation Brier); an ensemble is carried only if its
paired interval over the single model excludes zero. An outcome the final pipeline cannot run (a
non-LightGBM family, Venn-Abers, an ensemble) STOPS with exit 3 for review instead of improvising.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ["HAZARDPULSE_LAB_OUT"])
CHOICES = Path(os.environ["HAZARDPULSE_CHOICES_DIR"])


def res(name: str) -> dict:
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def family_search() -> int:
    fam = {n: res(n)["val"]["auc"] for n in ("f_lgbm", "f_xgb", "f_cat", "f_logit", "f_numpy_gbt")}
    best_fam = max(fam, key=fam.get)
    print("families", {k: round(v, 5) for k, v in fam.items()}, "->", best_fam)
    if best_fam != "f_lgbm":
        print("STOP: the best family is not LightGBM; the search and final pipeline are LightGBM-only")
        return 3
    search = {}
    for i in range(40):
        p = OUT / f"s_lgbm_{i:02d}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            search[r["exp"]["name"]] = (r["val"]["auc"], r["exp"]["params"])
    cands = {"f_lgbm": (fam["f_lgbm"], {})}
    cands.update(search)
    best = max(cands, key=lambda k: cands[k][0])
    aucs = sorted(v[0] for v in search.values())
    cfg = {"model": "lgbm", "params": cands[best][1],
           "chosen_by": (f"amendment 4 on validation (corrected program): family LightGBM "
                         f"({fam['f_lgbm']:.5f}); search {len(search)} configurations, median "
                         f"{aucs[len(aucs) // 2]:.5f}, max {aucs[-1]:.5f}; carried {best} ({cands[best][0]:.5f})")}
    (CHOICES / "best_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("best_config:", cfg["chosen_by"])
    return 0


def steps48() -> int:
    cfg = json.loads((CHOICES / "best_config.json").read_text(encoding="utf-8"))
    cal = {k: res(n)["val"]["brier"] for k, n in (("none", "c_none"), ("platt", "c_platt"), ("venn_abers", "c_venn_abers"))}
    calibration = min(cal, key=cal.get)
    neg = {10: res("n_neg10")["val"]["auc"], 30: res("c_platt")["val"]["auc"], 100: res("n_neg100")["val"]["auc"]}
    neg_per_pos = max(neg, key=neg.get)
    lab = {"storm_60": res("c_platt")["val"]["auc"], "nbhd_60": res("l_train_nbhd60_eval_storm60")["val"]["auc"]}
    label = max(lab, key=lab.get)
    ens = {}
    for e in ("e_ens5", "e_fam4"):
        c = OUT / f"compare_{e}_vs_c_platt_val_storm_60.json"
        if c.exists():
            ens[e] = json.loads(c.read_text(encoding="utf-8"))["delta_auc_ci"]
    carried_ens = [e for e, ci in ens.items() if ci[0] > 0]
    print("calibration (Brier)", cal, "->", calibration)
    print("negatives (AUC)", neg, "->", neg_per_pos)
    print("training label (AUC on storm_60)", lab, "->", label)
    print("ensembles (paired dAUC CI vs single)", ens, "-> carried", carried_ens or "none")
    if calibration != "platt" or carried_ens:
        print("STOP: the chosen calibration or an ensemble is not what the final pipeline runs")
        return 3
    cfg.update({"calibration": calibration, "neg_per_pos": neg_per_pos, "label": label})
    (CHOICES / "best_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    blocks = json.loads((CHOICES / "best_blocks.json").read_text(encoding="utf-8"))["blocks"]
    chosen = {
        "primary": {"name": "v3_primary", "model": "lgbm", "params": cfg["params"], "blocks": blocks,
                    "calibration": calibration, "neg_per_pos": neg_per_pos, "label": label, "seed": 0,
                    "validation_auc": res("c_platt")["val"]["auc"], "validation_bss": res("c_platt")["val"]["bss"]},
        "secondary_with_warnings": {"name": "v3_plus_W",
                                    "same_as_primary_except": {"blocks": blocks + ["W"]},
                                    "why_secondary": "amendment 3 / 6: served by the owner's decision when the NWS feed answers"},
        "steps": {"calibration_brier": cal, "negatives_auc": {str(k): v for k, v in neg.items()},
                  "label_auc": lab, "ensembles_ci": ens},
    }
    (CHOICES / "chosen_on_validation.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print("chosen_on_validation written")
    return 0


if __name__ == "__main__":
    raise SystemExit({"family_search": family_search, "steps48": steps48}[sys.argv[1]]())
