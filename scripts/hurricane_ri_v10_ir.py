"""Hurricane RI amendment 5 (docs/HURRICANE_RI_V9_PROGRAM.md): V8 = V5 + IR convective structure.

    PYTHONPATH=src python scripts/hurricane_ri_v10_ir.py        # after scripts/hurricane_ir_crops.py

Control: V5 recomputed here must reproduce amendment 3's log loss bit for bit, or the run stops.
Carried rule: V8's pooled 30/24 log loss AND four-threshold Brier both below V5's. Also reported:
paired intervals, POD at the HCCA call's false-alarm rate, IR coverage, and the 2024 -> 2025
product-version drift check (descriptive). 2026 is a declared fourth read.
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

import hurricane_ir_crops as crops  # noqa: E402
import hurricane_ri_v10 as v10  # noqa: E402
import hurricane_ri_v10_challengers as ch  # noqa: E402
import hurricane_ri_v10_vs_all as vs_all  # noqa: E402
import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import ir_features as ir  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_v10_ir.json"
V5_LL = 0.14360848294226844
ch.CANDS["V8"] = dict(names=ch.BASE + list(ir.IR_NAMES), mono=True)   # IR names are not in M_UP: unconstrained


def _load(aid: str, dtg: str, tag: str):
    p = crops.crop_path(aid, dtg, tag)
    if not p.exists():
        return None
    z = np.load(p)
    return {"counts": z["counts"], "lat": z["lat"], "lon": z["lon"], "centre": tuple(z["centre"])}


def add_ir(rows: list[dict]) -> dict:
    have = {"p2": 0, "m4": 0}
    for r in rows:
        now, before = _load(r["atcf_id"], r["dtg"], "p2"), _load(r["atcf_id"], r["dtg"], "m4")
        have["p2"] += now is not None
        have["m4"] += before is not None
        r["f"].update(ir.features(now, before))
    return have


def compare(rows, P: dict, base: str, other: str) -> dict:
    y = np.array([r["y"] for r in rows])
    g = np.array([r["sid"] for r in rows])
    hcca = np.array([r["f"]["dv24_HCCA"] for r in rows])
    has = np.isfinite(hcca)
    out = {}
    for c in (base, other):
        out[c] = {"summary_30": v9.summary(y, P[c][30], False), "brier4": float(ch.brier4(rows, P[c]).mean()),
                  "vs_hcca_call": {k: v for k, v in vs_all.matched_paired(
                      y[has], P[c][30][has], (hcca[has] >= 30).astype(float), g[has]).items()}}
    lb, lo = v9._ll_vec(y, P[base][30], False), v9._ll_vec(y, P[other][30], False)
    bb, bo = ch.brier4(rows, P[base]), ch.brier4(rows, P[other])
    out["paired"] = {"d_ll": float(lo.mean() - lb.mean()), "d_ll_ci": ch.paired_vec(lb, lo, g),
                     "d_brier4": float(bo.mean() - bb.mean()), "d_brier4_ci": ch.paired_vec(bb, bo, g)}
    return out


def drift(rows) -> dict:
    out = {}
    for n in ir.IR_NAMES:
        old = np.array([r["f"][n] for r in rows if 2022 <= r["season"] <= 2024])
        new = np.array([r["f"][n] for r in rows if r["season"] == 2025])
        old, new = old[np.isfinite(old)], new[np.isfinite(new)]
        if old.size < 30 or new.size < 30:
            out[n] = None
            continue
        sd = float(np.std(np.concatenate([old, new])))
        shift = (float(np.median(new)) - float(np.median(old))) / sd if sd > 0 else 0.0
        out[n] = {"median_2022_2024": float(np.median(old)), "median_2025": float(np.median(new)),
                  "shift_sd": shift, "flag": bool(abs(shift) > 1.0)}
    return out


FIXTURE = ROOT / "tests" / "fixtures" / "hurricane_v9"


def export() -> int:
    """Freeze V8 as results/models/hurricane_ri_v10_3.json (label "v10.3", inputs ONH + IR).
    Refused unless the refit reproduces the fourth read's 2026 log loss and the payloads reproduce
    the boosters. The fixture storm's real IR crops are copied into the test fixtures, with the
    lab's v10.3 curve, so the live path can be tested without the network."""
    import shutil

    from hazardpulse.hurricane import ri_v10
    from hazardpulse.tornado import lgbm_payload as lp
    rep = json.loads(OUT.read_text(encoding="utf-8"))
    if rep["carried"] != "V8":
        raise SystemExit("amendment 5 did not carry V8: nothing to export")
    rows = v10.load(v10.DEV10)
    add_ir(rows)
    models, rounds = ch.fit("V8", rows)
    c26 = v10.cases_2026()
    add_ir(c26)
    p = ch.predict("V8", models, c26)
    y = np.array([r["y"] for r in c26])
    ll = v9.summary(y, p[30], False)["log_loss"]
    want = rep["fourth_read_2026"]["results"]["V8"]["summary_30"]["log_loss"]
    if abs(ll - want) > 1e-12:
        raise SystemExit(f"refit 2026 LL {ll!r} != the fourth read's {want!r}")
    names = list(ch.CANDS["V8"]["names"])
    if names != ri_v10.allowed_feature_sets()["ONH+IR"]:
        raise SystemExit("V8's inputs are not the serving module's ONH+IR set")
    members = [lp.export_booster(m, names + ["threshold_kt"], calibration={"method": "identity", "a": 1.0, "b": 0.0},
                                 provenance={}) for m in models]
    dev = rep["development"]
    v5_art = json.loads(ri_v10.V10_2_PATH.read_text(encoding="utf-8"))
    art = {"schema": ri_v10.SCHEMA, "model_name": "hurricane_ri_v10_3", "label": "v10.3",
           "thresholds_kt": list(v10.K), "feature_names": names, "members": members,
           "provenance": {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 5)", "candidate": "V8",
                          "prereg_tag": rep["prereg_tag"], "trained": "NHC cycles 2020-2025 (IR from 2021-07-12)",
                          "event": "V(t+24 h) - V(t) >= k kt", "rounds": rounds, "seeds": list(v10.SEEDS),
                          "gate_aids": list(ri_v10.GATE_AIDS), "ir_source": "NOAA GMGSI longwave, hour t+2h and t-4h",
                          "monotone_up": [n for n in names if n in ch.M_UP],
                          "dev_2022_2025": {"log_loss": dev["V8"]["summary_30"]["log_loss"],
                                            "auc": dev["V8"]["summary_30"]["auc"], "brier4": dev["V8"]["brier4"],
                                            "champion_label": "v10.2",
                                            "champion_log_loss": dev["V5"]["summary_30"]["log_loss"],
                                            "champion_brier4": dev["V5"]["brier4"],
                                            "d_log_loss_vs_champion_ci": dev["paired"]["d_ll_ci"]},
                          "season_2026_fourth_read": {"log_loss": ll,
                                                      "champion_log_loss": rep["fourth_read_2026"]["results"]["V5"]["summary_30"]["log_loss"],
                                                      "d_log_loss_ci": rep["fourth_read_2026"]["results"]["paired"]["d_ll_ci"],
                                                      "declared": "2026 informed v10's design; no claim"}}}
    if json.loads(ri_v10.V10_2_PATH.read_text(encoding="utf-8"))["provenance"]["dev_2022_2025"]["log_loss"] != \
            dev["V5"]["summary_30"]["log_loss"]:
        raise SystemExit("V5 here is not the served v10.2")
    del v5_art
    ri_v10.V10_3_PATH.write_bytes(ri_v10.canonical_bytes(art))
    loaded, version = ri_v10.load(ri_v10.V10_3_PATH)
    X = v9.design(c26, names)
    worst = max(float(np.max(np.abs(ri_v10.predict_matrix(loaded, X, k) - p[k]))) for k in v10.MULTI)
    if worst > 1e-9:
        ri_v10.V10_3_PATH.unlink()
        raise SystemExit(f"artifact disagrees with the boosters by {worst:.2e}")
    fix = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    at = {r["dtg"]: i for i, r in enumerate(c26) if r["atcf_id"] == fix["storm"]}
    (FIXTURE / "ir").mkdir(exist_ok=True)
    for case in fix["cases"]:
        i = at[case["dtg"]]
        case["v10_3"] = {str(k): float(p[k][i]) for k in v10.MULTI}
        for tag in ("p2", "m4"):
            src = crops.crop_path(fix["storm"], case["dtg"], tag)
            if src.exists():
                shutil.copyfile(src, FIXTURE / "ir" / src.name)
    (FIXTURE / "expected.json").write_text(json.dumps(fix, indent=1) + "\n", encoding="utf-8")
    v9.log(f"{ri_v10.V10_3_PATH.name}: {version}; rounds {rounds}; 2026 LL {ll:.4f} reproduced; artifact = boosters "
           f"to {worst:.1e}; fixture gained the IR crops and the lab's v10.3 curve")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="select", choices=("select", "export"))
    return export() if ap.parse_args(argv).phase == "export" else select()


def select() -> int:
    rows = v10.load(v10.DEV10)
    have = add_ir(rows)
    P = {c: {k: [] for k in v10.MULTI} for c in ("V5", "V8")}
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
    v5_ll = v9.summary(y, P["V5"][30], False)["log_loss"]
    if abs(v5_ll - V5_LL) > 1e-12:
        raise SystemExit(f"control failed: V5 recomputed LL {v5_ll!r} != amendment 3's {V5_LL!r}")
    dev = compare(test_rows, P, "V5", "V8")
    carried = bool(dev["V8"]["summary_30"]["log_loss"] < dev["V5"]["summary_30"]["log_loss"]
                   and dev["V8"]["brier4"] < dev["V5"]["brier4"])
    covered = int(sum(math.isfinite(r["f"]["ir_mean_50_200"]) for r in test_rows))
    for c in ("V5", "V8"):
        d = dev[c]
        v9.log(f"DEV {c}: LL {d['summary_30']['log_loss']:.4f} AUC {d['summary_30']['auc']:.4f} Brier4 {d['brier4']:.4f}"
               f" | at HCCA's FAR POD {d['vs_hcca_call']['ours_pod']:.3f} vs {d['vs_hcca_call']['aid_pod']:.3f}")
    pr = dev["paired"]
    v9.log(f"V8 - V5: dLL {pr['d_ll']:+.4f} [{pr['d_ll_ci'][0]:+.4f}, {pr['d_ll_ci'][1]:+.4f}]  dBrier4 "
           f"{pr['d_brier4']:+.4f} [{pr['d_brier4_ci'][0]:+.4f}, {pr['d_brier4_ci'][1]:+.4f}]; IR on {covered} of "
           f"{len(test_rows)} scored cycles; CARRIED V8: {carried}")
    dr = drift(rows)
    flags = [n for n, v in dr.items() if v and v["flag"]]
    v9.log(f"drift 2022-2024 -> 2025: flagged {flags or 'none'}")

    c26 = v10.cases_2026()
    have26 = add_ir(c26)
    P26 = {c: ch.predict(c, ch.fit(c, rows)[0], c26) for c in ("V5", "V8")}
    third = compare(c26, P26, "V5", "V8")
    v9.log(f"2026 (fourth read): V5 LL {third['V5']['summary_30']['log_loss']:.4f}, V8 LL "
           f"{third['V8']['summary_30']['log_loss']:.4f}, dLL {third['paired']['d_ll']:+.4f} "
           f"[{third['paired']['d_ll_ci'][0]:+.4f}, {third['paired']['d_ll_ci'][1]:+.4f}]")
    OUT.write_text(json.dumps({
        "phase": "amendment 5 select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 5)",
        "prereg_tag": "prereg-hurricane-ri-amend5",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": v9.sha(v10.DEV10), "control_V5_log_loss": v5_ll, "ir_names": list(ir.IR_NAMES),
        "crops_present_dev": have, "scored_cycles_with_ir": covered, "development": dev, "carried": "V8" if carried else "V5",
        "drift_2025_vs_2022_2024": dr,
        "fourth_read_2026": {"declared": "2026 informed v10's design; reported, no claim", "crops_present": have26,
                             "results": third}}, indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
