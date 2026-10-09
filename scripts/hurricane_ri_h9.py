"""Hurricane RI amendment 10 (docs/HURRICANE_RI_V9_PROGRAM.md): H9 = H8 + the storm's coherence state.

    PYTHONPATH=src python scripts/hurricane_ri_h9.py        # after hurricane_ri_h8.py (its H8 feature cache)

Control: H8 recomputed here must reproduce amendment 8's development log loss to 1e-12, or the run stops.
Carried rule: H9's pooled 30/24 log loss AND four-threshold Brier both below H8's. Also reported, descriptive:
paired intervals, POD at the HCCA call's false-alarm rate, the drift check, and the coherence features' share of the
split gain. 2026 is a declared further read.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_h8 as h8  # noqa: E402
from hurricane_ri_h8 import ch, v9, v10, v10ir  # noqa: E402
from hazardpulse.hurricane import coherence_state as coh  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_h9.json"
H8_LL = 0.14161766094567663
ch.CANDS["H9"] = dict(names=ch.CANDS["H8"]["names"] + list(coh.COH_NAMES), mono=True)   # not in M_UP: unconstrained


def gain_share(models) -> dict:
    names = list(ch.CANDS["H9"]["names"]) + ["threshold_kt"]
    g = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    total = float(g.sum())
    per = {n: float(g[names.index(n)] / total) for n in coh.COH_NAMES}
    return {"total": float(sum(per.values())), "per_feature": per}


def prepare(rows: list[dict], adeck) -> dict:
    v10ir.add_ir(rows)
    cov = h8.add_h8(rows, adeck)
    cov["coherence_present"] = coh.add_states(rows)
    return cov


def select() -> int:
    rows = v10.load(v10.DEV10)
    cov = prepare(rows, h8.adeck_dev)
    P = {c: {k: [] for k in v10.MULTI} for c in ("H8", "H9")}
    test_rows = []
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        for c in P:
            models, _ = ch.fit(c, train)
            for k, v in ch.predict(c, models, test).items():
                P[c][k].append(v)
        test_rows.extend(test)
        v9.log(f"fold {season} done")
    P = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in P.items()}
    y = np.array([r["y"] for r in test_rows])
    h8_ll = v9.summary(y, P["H8"][30], False)["log_loss"]
    if abs(h8_ll - H8_LL) > 1e-12:
        raise SystemExit(f"control failed: H8 recomputed LL {h8_ll!r} != amendment 8's {H8_LL!r}")
    dev = v10ir.compare(test_rows, P, "H8", "H9")
    carried = bool(dev["H9"]["summary_30"]["log_loss"] < dev["H8"]["summary_30"]["log_loss"]
                   and dev["H9"]["brier4"] < dev["H8"]["brier4"])
    for c in ("H8", "H9"):
        d = dev[c]
        v9.log(f"DEV {c}: LL {d['summary_30']['log_loss']:.4f} AUC {d['summary_30']['auc']:.4f} Brier4 {d['brier4']:.4f}"
               f" | at HCCA's FAR POD {d['vs_hcca_call']['ours_pod']:.3f} vs {d['vs_hcca_call']['aid_pod']:.3f}")
    pr = dev["paired"]
    v9.log(f"H9 - H8: dLL {pr['d_ll']:+.5f} [{pr['d_ll_ci'][0]:+.5f}, {pr['d_ll_ci'][1]:+.5f}]  dBrier4 "
           f"{pr['d_brier4']:+.5f} [{pr['d_brier4_ci'][0]:+.5f}, {pr['d_brier4_ci'][1]:+.5f}]; CARRIED H9: {carried}")
    dr = v10ir.drift(rows, coh.COH_NAMES)
    v9.log(f"drift 2022-2024 -> 2025: flagged {[n for n, v in dr.items() if v and v['flag']] or 'none'}")

    c26 = v10.cases_2026()
    cov26 = prepare(c26, h8.adeck_2026)
    fits = {c: ch.fit(c, rows) for c in ("H8", "H9")}
    share = gain_share(fits["H9"][0])
    v9.log(f"coherence share of split gain: {share['total']:.3f} "
           + " ".join(f"{n} {s:.3f}" for n, s in share["per_feature"].items()))
    P26 = {c: ch.predict(c, fits[c][0], c26) for c in fits}
    further = v10ir.compare(c26, P26, "H8", "H9")
    v9.log(f"2026 (further read): H8 LL {further['H8']['summary_30']['log_loss']:.4f}, H9 LL "
           f"{further['H9']['summary_30']['log_loss']:.4f}, dLL {further['paired']['d_ll']:+.4f} "
           f"[{further['paired']['d_ll_ci'][0]:+.4f}, {further['paired']['d_ll_ci'][1]:+.4f}]")
    OUT.write_text(json.dumps({
        "phase": "amendment 10 select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 10)",
        "prereg_tag": "prereg-hurricane-ri-amend10",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": v9.sha(v10.DEV10), "control_H8_log_loss": h8_ll, "coherence_names": list(coh.COH_NAMES),
        "coverage_dev": cov, "development": dev, "carried": "H9" if carried else "H8", "drift_2025_vs_2022_2024": dr,
        "gain_share_all_dev": share,
        "further_read_2026": {"declared": "2026 informed v10's design and amendments 5 and 8; reported, no claim",
                              "coverage": cov26, "results": further}}, indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(select())
