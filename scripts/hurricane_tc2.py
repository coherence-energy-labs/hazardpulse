"""Program TC1, amendment 2 (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md): TC2 = TC1's intensity held to the median our
RI model's 24-h curve implies.

    PYTHONPATH=src python scripts/hurricane_tc2.py curves     # out-of-fold H8 curves (control: amendment 8's LL)
    PYTHONPATH=src python scripts/hurricane_tc2.py dev        # the registered test, DEV 2023-2025
    PYTHONPATH=src python scripts/hurricane_tc2.py read2026   # the further read

Controls (each stops the run): the curves' pooled 30/24 log loss over the RI folds reproduces amendment 8's
0.14161766094567663 to 1e-12; the 2026 curves reproduce amendment 8's further read (0.11715873924102127); the TC1
intensity run reproduces TC1's registered DEV mean error (``results/hurricane_tc1/dev.json``) to 1e-9.
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

import hurricane_tc1 as tc1  # noqa: E402
from hazardpulse.hurricane import consensus as cs  # noqa: E402
from hazardpulse.hurricane import tc2  # noqa: E402

OUT_DIR = ROOT / "results" / "hurricane_tc2"
CURVES_DEV = OUT_DIR / "ri_curves_dev.json"
CURVES_2026 = OUT_DIR / "ri_curves_2026.json"
DEV_OUT = OUT_DIR / "dev.json"
READ_2026 = OUT_DIR / "season_2026.json"
H8_LL = 0.14161766094567663
H8_LL_2026 = 0.11715873924102127
REPORT_LEADS = (12,) + tc1.SCORED_LEADS
GATE_AIDS = ("DSHP", "IVCN", "NNIC")


def _ri():
    import hurricane_ri_g1 as g1
    return g1


def _predict_all(models, rows, names):
    import hurricane_ri_v9 as v9
    X = v9.design(rows, names)
    return {k: np.mean([m.predict(np.hstack([X, np.full((len(X), 1), float(k))])) for m in models], axis=0)
            for k in tc2.THRESHOLDS_KT}


def _key(storm: str, t: dt.datetime | str) -> str:
    t = t if isinstance(t, str) else t.strftime("%Y%m%d%H")
    return f"{storm.upper()}|{t}"


def _gate(r) -> bool:
    return all(math.isfinite(float(r["f"].get(f"dv24_{a}", math.nan))) for a in GATE_AIDS)


def curves() -> int:
    """Out-of-fold H8 curves for the RI folds (2022-2025) and H8 fitted on 2020-2025 for 2026."""
    g1 = _ri()
    ch, v9, v10, h9 = g1.ch, g1.v9, g1.v10, g1.h9
    rows = v10.load(v10.DEV10)
    h9.prepare(rows, h9.h8.adeck_dev)
    names = list(ch.CANDS["H8"]["names"]) + ["threshold_kt"]
    out, y30, p30 = {}, [], []
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        models, _ = ch.fit("H8", train)
        P = _predict_all(models, test, names[:-1])
        for i, r in enumerate(test):
            out[_key(r["atcf_id"], r["dtg"])] = {"curve": {str(k): float(P[k][i]) for k in tc2.THRESHOLDS_KT},
                                                  "gate_ok": _gate(r), "season": r["season"]}
        y30.extend(r["y"] for r in test)
        p30.extend(P[30])
    ll = v9.summary(np.array(y30), np.array(p30), False)["log_loss"]
    if abs(ll - H8_LL) > 1e-12:
        raise SystemExit(f"control failed: out-of-fold H8 log loss {ll!r} != amendment 8's {H8_LL!r}")
    models, _ = ch.fit("H8", rows)
    c26 = v10.cases_2026()
    h9.prepare(c26, h9.h8.adeck_2026)
    P26 = _predict_all(models, c26, names[:-1])
    ll26 = v9.summary(np.array([r["y"] for r in c26]), P26[30], False)["log_loss"]
    if abs(ll26 - H8_LL_2026) > 1e-12:
        raise SystemExit(f"control failed: 2026 H8 log loss {ll26!r} != amendment 8's further read {H8_LL_2026!r}")
    out26 = {_key(r["atcf_id"], r["dtg"]): {"curve": {str(k): float(P26[k][i]) for k in tc2.THRESHOLDS_KT},
                                            "gate_ok": _gate(r)} for i, r in enumerate(c26)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 2)", "prereg_tag": "prereg-tc2",
            "model": "H8 (the RI program's v10.4 model), out of fold per season"}
    CURVES_DEV.write_text(json.dumps({**meta, "control_log_loss": ll, "curves": out}), encoding="utf-8")
    CURVES_2026.write_text(json.dumps({**meta, "model": "H8 fitted on 2020-2025 (the v10.4 artifact's model)",
                                       "control_log_loss": ll26, "curves": out26}), encoding="utf-8")
    tc1.log(f"controls passed: out-of-fold LL {ll!r}, 2026 LL {ll26!r}; {len(out)} dev and {len(out26)} 2026 curves")
    return 0


def tc2_product(decks, tc1_run: dict, curve_of: dict, taper_end: float | None = None) -> tuple[dict, dict]:
    """TC2 in the shape of a TC1 run, ``{(storm, t, lead): (forecast, None)}``, and the shift statistics."""
    by_storm = {d.storm: d for d in decks}
    cycles: dict = {}
    for (storm, t, lead), (fc, _w) in tc1_run.items():
        cycles.setdefault((storm, t), {})[lead] = fc
    out, stats = {}, {"cycles": 0, "with_curve": 0, "up": 0, "down": 0, "shifts": []}
    for (storm, t), by_lead in cycles.items():
        stats["cycles"] += 1
        c = curve_of.get(_key(storm, t))
        curve = {int(k): v for k, v in c["curve"].items()} if c and c.get("gate_ok") else None
        an = by_storm[storm].analysis(t)
        v0 = an[2] if an is not None and math.isfinite(an[2]) else None
        proj, info = tc2.project(by_lead, v0, curve, taper_end)
        if curve is not None:
            stats["with_curve"] += 1
        if info.get("applied"):
            s = info["shift_24h"]
            stats["up" if s > 0 else "down"] += 1
            stats["shifts"].append(s)
        for lead, v in proj.items():
            out[(storm, t, lead)] = (v, None)
    sh = np.array(stats.pop("shifts") or [0.0])
    stats.update({"mean_abs_shift_kt": float(np.abs(sh).mean()) if stats["up"] + stats["down"] else 0.0,
                  "max_up_kt": float(sh.max()), "max_down_kt": float(sh.min())})
    return out, stats


def _ri_subset(case_list) -> set:
    """Cycles whose best track rose >= 30 kt over the next 24 h."""
    keys = set()
    for d, tr, t in case_list:
        a, b = tr.get(t), tr.get(t + dt.timedelta(hours=24))
        if a and b and math.isfinite(a[2]) and math.isfinite(b[2]) and b[2] - a[2] >= 30:
            keys.add((d.storm, t))
    return keys


def _evaluate(decks, truth_of, seasons, curve_of) -> dict:
    case_list = tc1.cases(decks, truth_of, seasons)
    run = cs.run(decks, tc1.chosen("intensity"), "intensity", keep_weights=False)
    prod, stats = tc2_product(decks, run, curve_of)
    prod_b, stats_b = tc2_product(decks, run, curve_of, tc2.TC2B_TAPER_END_H)       # amendment 3
    products = {"TC1": run, "TC2": prod, "TC2b": prod_b, "OFCL": "OFCL", "HCCA": "HCCA", "IVCN": "IVCN"}
    errs = tc1.score(case_list, "intensity", REPORT_LEADS, products)
    res = {"errors": {p: tc1.mean_error(e, tc1.SCORED_LEADS) for p, e in errs.items()},
           "errors_12h": {p: (float(np.mean(list(e[12].values()))) if e[12] else None, len(e[12])) for p, e in errs.items()},
           # the registered primary is the mean over 24-120 h (TC1's scored leads); 12 h is reported on its own
           "TC2-TC1": tc1.paired(errs["TC2"], errs["TC1"], tc1.SCORED_LEADS, tc1.REPORT_LEVEL),
           "TC2-TC1_12h": tc1.paired(errs["TC2"], errs["TC1"], (12,), tc1.REPORT_LEVEL),
           "TC2b-TC1": tc1.paired(errs["TC2b"], errs["TC1"], tc1.SCORED_LEADS, tc1.REPORT_LEVEL),
           "TC2b-TC1_12h": tc1.paired(errs["TC2b"], errs["TC1"], (12,), tc1.REPORT_LEVEL),
           "reported": {f"{p}-{q}": tc1.paired(errs[p], errs[q], tc1.SCORED_LEADS, tc1.REPORT_LEVEL)
                        for p in ("TC2", "TC2b") for q in ("OFCL", "HCCA", "IVCN")},
           "shift_stats": stats, "shift_stats_tc2b": stats_b}
    ri = _ri_subset(case_list)
    res["ri_subset"] = {"cycles": len(ri), **{
        str(ld): {p: (float(np.mean([v for k, v in errs[p][ld].items() if k in ri])) if any(k in ri for k in errs[p][ld]) else None,
                      sum(1 for k in errs[p][ld] if k in ri)) for p in ("TC1", "TC2", "TC2b", "OFCL", "HCCA")}
        for ld in (12, 24)}}
    return res, errs


def dev() -> int:
    if not CURVES_DEV.exists():
        raise SystemExit("run `hurricane_tc2.py curves` first")
    curve_of = json.loads(CURVES_DEV.read_text(encoding="utf-8"))["curves"]
    seasons = tc1.WARMUP + tc1.CHOOSE + tc1.DEV
    tc1.log(f"loading decks and truth {seasons[0]}-{seasons[-1]}")
    truth = tc1.load_truth(seasons)
    decks = tc1.load_decks(tc1.ALL_TECHS, seasons)
    res, _ = _evaluate(decks, lambda d: tc1.truth_for(d, truth), tc1.DEV, curve_of)
    want = json.loads(tc1.DEV_OUT.read_text(encoding="utf-8"))["intensity"]["errors"]["TC1"]["mean_over_leads"]
    got = res["errors"]["TC1"]["mean_over_leads"]
    if abs(got - want) > 1e-9:
        raise SystemExit(f"control failed: TC1 DEV intensity {got!r} != the registered {want!r}")
    d = res["TC2-TC1"]
    carried = bool(d.get("n_storms") and d["mean_over_leads"]["d"] < 0 and d["per_lead"]["24"]["d"] < 0)
    DEV_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEV_OUT.write_text(json.dumps({
        "phase": "TC2 dev", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 2)",
        "prereg_tag": "prereg-tc2", "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": tc1.DEV, "control_tc1_dev_intensity": got, "carried": "TC2" if carried else "TC1", **res},
        indent=1), encoding="utf-8")
    e = res["errors"]
    tc1.log(f"control passed: TC1 DEV intensity {got:.6f} reproduced")
    tc1.log("mean over 24-120 h (kt): " + ", ".join(f"{p} {v['mean_over_leads']:.3f}" for p, v in e.items() if v["mean_over_leads"]))
    tc1.log(f"TC2 - TC1: mean {d['mean_over_leads']['d']:+.3f} [{d['mean_over_leads']['ci'][0]:+.3f}, "
            f"{d['mean_over_leads']['ci'][1]:+.3f}]; " + ", ".join(
                f"{ld} h {v['d']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for ld, v in d["per_lead"].items())
            + f"; CARRIED TC2: {carried}")
    for q, v in res["reported"].items():
        tc1.log(f"{q}: mean {v['mean_over_leads']['d']:+.3f} [{v['mean_over_leads']['ci'][0]:+.3f}, {v['mean_over_leads']['ci'][1]:+.3f}]")
    tc1.log(f"RI subset: {json.dumps(res['ri_subset'])}")
    tc1.log(f"shifts: {json.dumps(res['shift_stats'])}")
    return 0


def read2026() -> int:
    if not CURVES_2026.exists():
        raise SystemExit("run `hurricane_tc2.py curves` first")
    curve_of = json.loads(CURVES_2026.read_text(encoding="utf-8"))["curves"]
    storms = tc1.fetch_2026()
    seasons = tc1.WARMUP + tc1.CHOOSE + tc1.DEV + (2026,)
    decks = tc1.load_decks(tc1.ALL_TECHS, seasons)
    t26 = {s: tc1.btk_truth(s) for s in storms}
    res, _ = _evaluate(decks, lambda d: t26.get(d.storm, {}), (2026,), curve_of)
    db = res["TC2b-TC1"]
    carried = bool(db.get("n_storms") and db["per_lead"]["24"]["d"] < 0 and db["mean_over_leads"]["d"] < 0)
    READ_2026.write_text(json.dumps({
        "phase": "TC2b test (amendment 3) and TC2 further read, 2026",
        "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendments 2, 3)", "prereg_tag": "prereg-tc2b",
        "declared": "the season in progress, operational best tracks: TC2b's registered test; TC2 reported",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "storms": storms,
        "carried_tc2b": "TC2b" if carried else "TC1", **res}, indent=1), encoding="utf-8")
    e = res["errors"]
    tc1.log("2026 mean over 24-120 h (kt): " + ", ".join(f"{p} {v['mean_over_leads']:.3f}" for p, v in e.items() if v["mean_over_leads"]))
    tc1.log(f"2026 TC2b - TC1 (amendment 3, the registered test): mean {db['mean_over_leads']['d']:+.3f} "
            f"[{db['mean_over_leads']['ci'][0]:+.3f}, {db['mean_over_leads']['ci'][1]:+.3f}]; " + ", ".join(
                f"{ld} h {v['d']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for ld, v in db["per_lead"].items())
            + f"; CARRIED TC2b: {carried}")
    d = res["TC2-TC1"]
    tc1.log(f"2026 TC2 - TC1: mean {d['mean_over_leads']['d']:+.3f} [{d['mean_over_leads']['ci'][0]:+.3f}, "
            f"{d['mean_over_leads']['ci'][1]:+.3f}]; RI subset {json.dumps(res['ri_subset'])}; shifts {json.dumps(res['shift_stats'])}")
    return 0


TC2BO_OUT = OUT_DIR / "tc2bo.json"


def _evaluate_o(decks, truth_of, seasons, curve_of) -> dict:
    """Amendment 5: TC1+O and TC2b+O (TC2b's rule on TC1+O) against each other and NHC's aids."""
    case_list = tc1.cases(decks, truth_of, seasons)
    run_o = cs.run(decks, tc1.chosen("intensity", official=True), "intensity", keep_weights=False)
    prod_o, stats = tc2_product(decks, run_o, curve_of, tc2.TC2B_TAPER_END_H)
    products = {"TC1+O": run_o, "TC2b+O": prod_o, "OFCL": "OFCL", "HCCA": "HCCA", "IVCN": "IVCN"}
    errs = tc1.score(case_list, "intensity", REPORT_LEADS, products)
    ri = _ri_subset(case_list)
    return {"errors": {p: tc1.mean_error(e, tc1.SCORED_LEADS) for p, e in errs.items()},
            "errors_12h": {p: (float(np.mean(list(e[12].values()))) if e[12] else None, len(e[12])) for p, e in errs.items()},
            "TC2b+O-TC1+O": tc1.paired(errs["TC2b+O"], errs["TC1+O"], tc1.SCORED_LEADS, tc1.REPORT_LEVEL),
            "TC2b+O-TC1+O_12h": tc1.paired(errs["TC2b+O"], errs["TC1+O"], (12,), tc1.REPORT_LEVEL),
            "reported": {f"{p}-{q}": tc1.paired(errs[p], errs[q], tc1.SCORED_LEADS, tc1.REPORT_LEVEL)
                         for p in ("TC2b+O", "TC1+O") for q in ("OFCL", "HCCA", "IVCN")},
            "shift_stats": stats,
            "ri_subset": {"cycles": len(ri), **{
                str(ld): {p: (float(np.mean([v for k, v in errs[p][ld].items() if k in ri])) if any(k in ri for k in errs[p][ld]) else None,
                              sum(1 for k in errs[p][ld] if k in ri)) for p in ("TC1+O", "TC2b+O", "OFCL", "HCCA")}
                for ld in (12, 24)}}}


def tc2bo() -> int:
    """Amendment 5's registered test: TC2b+O against TC1+O on DEV and on 2026; carried iff both seasons agree."""
    dev_curves = json.loads(CURVES_DEV.read_text(encoding="utf-8"))["curves"]
    c26 = json.loads(CURVES_2026.read_text(encoding="utf-8"))["curves"]
    seasons = tc1.WARMUP + tc1.CHOOSE + tc1.DEV
    truth = tc1.load_truth(seasons)
    decks = tc1.load_decks(tc1.ALL_TECHS, seasons)
    dev_res = _evaluate_o(decks, lambda d: tc1.truth_for(d, truth), tc1.DEV, dev_curves)
    want = json.loads(tc1.DEV_OUT.read_text(encoding="utf-8"))["intensity"]["errors"]["TC1+O"]["mean_over_leads"]
    got = dev_res["errors"]["TC1+O"]["mean_over_leads"]
    if abs(got - want) > 1e-9:
        raise SystemExit(f"control failed: TC1+O DEV intensity {got!r} != the registered {want!r}")
    storms = tc1.fetch_2026()
    decks26 = tc1.load_decks(tc1.ALL_TECHS, seasons + (2026,))
    t26 = {s: tc1.btk_truth(s) for s in storms}
    res26 = _evaluate_o(decks26, lambda d: t26.get(d.storm, {}), (2026,), c26)

    def ok(r):
        d = r["TC2b+O-TC1+O"]
        return bool(d.get("n_storms") and d["per_lead"]["24"]["d"] < 0 and d["mean_over_leads"]["d"] < 0)
    carried = ok(dev_res) and ok(res26)
    TC2BO_OUT.write_text(json.dumps({
        "phase": "TC2b+O test (amendment 5)", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 5)",
        "prereg_tag": "prereg-tc2bo", "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "control_tc1o_dev_intensity": got, "storms_2026": storms, "carried": "TC2b+O" if carried else "TC1+O",
        "dev": dev_res, "season_2026": res26}, indent=1), encoding="utf-8")
    tc1.log(f"control passed: TC1+O DEV intensity {got:.6f}")
    for name, r in (("DEV", dev_res), ("2026", res26)):
        d = r["TC2b+O-TC1+O"]
        e = r["errors"]
        tc1.log(f"{name}: " + ", ".join(f"{p} {v['mean_over_leads']:.3f}" for p, v in e.items() if v["mean_over_leads"])
                + f" | TC2b+O - TC1+O mean {d['mean_over_leads']['d']:+.3f} [{d['mean_over_leads']['ci'][0]:+.3f}, "
                f"{d['mean_over_leads']['ci'][1]:+.3f}], 24 h {d['per_lead']['24']['d']:+.3f} "
                f"[{d['per_lead']['24']['ci'][0]:+.2f}, {d['per_lead']['24']['ci'][1]:+.2f}]"
                + f" | vs OFCL {r['reported']['TC2b+O-OFCL']['mean_over_leads']['d']:+.3f} "
                f"[{r['reported']['TC2b+O-OFCL']['mean_over_leads']['ci'][0]:+.3f}, {r['reported']['TC2b+O-OFCL']['mean_over_leads']['ci'][1]:+.3f}]"
                + f" | RI {json.dumps(r['ri_subset'])}")
    tc1.log(f"CARRIED TC2b+O: {carried}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("curves", "dev", "read2026", "tc2bo"))
    a = ap.parse_args(argv)
    return {"curves": curves, "dev": dev, "read2026": read2026, "tc2bo": tc2bo}[a.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
