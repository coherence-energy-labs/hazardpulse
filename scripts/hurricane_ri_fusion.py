"""Hurricane RI amendment 11 (docs/HURRICANE_RI_V9_PROGRAM.md): coherence fusion of every RI probability source.

    PYTHONPATH=src python scripts/hurricane_ri_fusion.py table   # the sources at every cycle (no fusion, no score)
    PYTHONPATH=src python scripts/hurricane_ri_fusion.py run     # the registered selection, claim and 2026 read

Sources per cycle and threshold (25/30/35/40 kt in 24 h):
- our models V2, V5, V8 and H8, each OUT OF FOLD -- a season is forecast only by models trained on earlier
  seasons, as live -- for 2022-2025, and fitted on 2020-2025 for 2026;
- NOAA's DTOPS, SHIPS-RII (RIOD), RIOC, RIOB and RIOL, as issued (whole percent).

Fusion (``hazardpulse.verification.coherence_fusion``) runs causally over 2020-2026 in time order, each threshold
on its own; a cycle verifies 24 h after it is issued.
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

import hurricane_ri_h9 as h9  # noqa: E402
from hurricane_ri_h9 import ch, v9, v10  # noqa: E402
from hazardpulse.verification import coherence_fusion as cf  # noqa: E402

TABLE = v9.WORK / "fusion_sources.json"
OUT = ROOT / "results" / "calibration" / "hurricane_ri_fusion.json"
MODELS = ("V2", "V5", "V8", "H8")
NOAA = ("DTOP", "RIOD", "RIOC", "RIOB", "RIOL")
SOURCES = list(MODELS) + list(NOAA)
WARMUP, CHOOSE, DEV = (2020, 2021, 2022), (2023,), (2024, 2025)
GRID = tuple(cf.FusionConfig(eta=e, tau_days=t, pool=p, align=a)
             for e in (0.05, 0.2, 1.0) for t in (30.0, 120.0, 365.0) for p in ("linear", "log") for a in (1.0, 1.3))
REFERENCES = {"equal_linear": cf.FusionConfig(equal=True), "equal_log": cf.FusionConfig(equal=True, pool="log")}
EPOCH = dt.datetime(2000, 1, 1)


def _days(dtg: str) -> float:
    return (dt.datetime.strptime(dtg, "%Y%m%d%H") - EPOCH).total_seconds() / 86400.0


def _noaa(row: dict, tech: str, k: int) -> float:
    v = (row.get("pcts") or {}).get(f"{tech}_{k}/24")
    return math.nan if v is None else float(v) / 100.0


def table() -> int:
    """Every cycle's sources at each threshold (models out of fold), with its truth. No fusion, no score."""
    rows = v10.load(v10.DEV10)
    h9.prepare(rows, h9.h8.adeck_dev)
    preds = {m: {} for m in MODELS}
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        for m in MODELS:
            models, _ = ch.fit(m, train)
            p = ch.predict(m, models, test)
            for i, r in enumerate(test):
                preds[m][(r["atcf_id"], r["dtg"])] = {k: float(p[k][i]) for k in v10.MULTI}
        v9.log(f"fold {season} done")
    c26 = v10.cases_2026()
    h9.prepare(c26, h9.h8.adeck_2026)
    for m in MODELS:
        models, _ = ch.fit(m, rows)
        p = ch.predict(m, models, c26)
        for i, r in enumerate(c26):
            preds[m][(r["atcf_id"], r["dtg"])] = {k: float(p[k][i]) for k in v10.MULTI}
    v9.log("2026 done")
    out = []
    for r in rows + c26:
        key = (r["atcf_id"], r["dtg"])
        src = {}
        for k in v10.MULTI:
            src[str(k)] = {**{m: (preds[m][key][k] if key in preds[m] else None) for m in MODELS},
                           **{t: (None if math.isnan(_noaa(r, t, k)) else _noaa(r, t, k)) for t in NOAA}}
        out.append({"atcf_id": r["atcf_id"], "dtg": r["dtg"], "season": int(r["season"]), "dv": float(r["dv"]),
                    "sources": src})
    TABLE.write_text(json.dumps(out), encoding="utf-8")
    v9.log(f"wrote {TABLE}: {len(out)} cycles")
    return 0


def load_cases(k: int) -> tuple[list[dict], list[dict]]:
    meta = json.loads(TABLE.read_text(encoding="utf-8"))
    cases = [{"t": _days(r["dtg"]), "y": int(r["dv"] >= k),
              "probs": {s: (math.nan if v is None else v) for s, v in r["sources"][str(k)].items()}} for r in meta]
    return cases, meta


def _ll(ps, ys) -> float:
    return float(np.mean([cf.log_loss(p, y) for p, y in zip(ps, ys)]))


def evaluate(fused: list, cases: list, meta: list, seasons) -> dict:
    """Mean log loss of the fusion and of each single source (on the cases it covers) over ``seasons``."""
    idx = [i for i, r in enumerate(meta) if r["season"] in seasons and fused[i] is not None]
    out = {"n": len(idx), "fused": _ll([fused[i] for i in idx], [cases[i]["y"] for i in idx]), "sources": {}}
    for s in SOURCES:
        have = [i for i in idx if math.isfinite(cases[i]["probs"][s])]
        out["sources"][s] = {"n": len(have), "ll": _ll([cases[i]["probs"][s] for i in have],
                                                        [cases[i]["y"] for i in have]) if have else None}
    return out


def paired(fused: list, cases: list, meta: list, seasons, source: str) -> dict:
    """fusion minus ``source`` on the cases both cover, storm-bootstrap 95% interval."""
    idx = [i for i, r in enumerate(meta) if r["season"] in seasons and fused[i] is not None
           and math.isfinite(cases[i]["probs"][source])]
    a = np.array([cf.log_loss(cases[i]["probs"][source], cases[i]["y"]) for i in idx])
    b = np.array([cf.log_loss(fused[i], cases[i]["y"]) for i in idx])
    g = np.array([meta[i]["atcf_id"] for i in idx])
    return {"n": len(idx), "d_ll": float(b.mean() - a.mean()), "d_ll_ci": ch.paired_vec(a, b, g)}


def brier4(fused_by_k: dict, meta: list, seasons) -> float:
    idx = [i for i, r in enumerate(meta) if r["season"] in seasons
           and all(fused_by_k[k][i] is not None for k in v10.MULTI)]
    return float(np.mean([sum((fused_by_k[k][i] - int(meta[i]["dv"] >= k)) ** 2 for k in v10.MULTI) for i in idx]))


def _cfg(c: cf.FusionConfig) -> dict:
    return {"eta": c.eta, "tau_days": c.tau_days, "pool": c.pool, "align": c.align, "equal": c.equal}


def run_phase() -> int:
    cases30, meta = load_cases(30)
    table_rows = []
    for i, cfg in enumerate(GRID):
        fused = cf.run(cases30, cfg, SOURCES, verify_after_days=1.0)
        table_rows.append({"index": i, "config": _cfg(cfg), "choose_ll": evaluate(fused, cases30, meta, CHOOSE)["fused"]})
    chosen = min(table_rows, key=lambda r: (r["choose_ll"], r["index"]))
    cfg = GRID[chosen["index"]]
    choose_eval = evaluate(cf.run(cases30, cfg, SOURCES, 1.0), cases30, meta, CHOOSE)
    eligible = {s: v for s, v in choose_eval["sources"].items() if v["n"] >= 0.9 * choose_eval["n"]}
    best = min(eligible, key=lambda s: (eligible[s]["ll"], SOURCES.index(s)))
    v9.log(f"chosen config {chosen['index']} {chosen['config']} (CHOOSE LL {chosen['choose_ll']:.4f}); "
           f"best single source on CHOOSE: {best} ({eligible[best]['ll']:.4f})")
    fused_by_k, refs = {}, {}
    for k in v10.MULTI:
        cases_k, _ = load_cases(k)
        fused_by_k[k] = cf.run(cases_k, cfg, SOURCES, 1.0)
    for name, rc in REFERENCES.items():
        refs[name] = cf.run(cases30, rc, SOURCES, 1.0)
    f30 = fused_by_k[30]
    res = {}
    for label, seasons in (("dev", DEV), ("season_2026", (2026,))):
        ev = evaluate(f30, cases30, meta, seasons)
        res[label] = {"eval": ev,
                      "vs_best_single": paired(f30, cases30, meta, seasons, best),
                      "vs_each": {s: paired(f30, cases30, meta, seasons, s) for s in SOURCES},
                      "references": {n: evaluate(f, cases30, meta, seasons)["fused"] for n, f in refs.items()},
                      "brier4_fused": brier4(fused_by_k, meta, seasons)}
    claim = res["dev"]["vs_best_single"]
    claim["claim"] = bool(claim["d_ll_ci"][1] < 0)
    v9.log(f"DEV fused LL {res['dev']['eval']['fused']:.4f} vs best single {best} "
           f"{res['dev']['eval']['sources'][best]['ll']:.4f}: d {claim['d_ll']:+.4f} "
           f"[{claim['d_ll_ci'][0]:+.4f}, {claim['d_ll_ci'][1]:+.4f}] -> claim {claim['claim']}")
    s26 = res["season_2026"]["vs_best_single"]
    v9.log(f"2026 (further read): fused LL {res['season_2026']['eval']['fused']:.4f}, d vs {best} {s26['d_ll']:+.4f} "
           f"[{s26['d_ll_ci'][0]:+.4f}, {s26['d_ll_ci'][1]:+.4f}]")
    OUT.write_text(json.dumps({
        "phase": "amendment 11", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 11)",
        "prereg_tag": "prereg-hurricane-ri-amend11",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": SOURCES, "seasons": {"warmup": WARMUP, "choose": CHOOSE, "dev": DEV},
        "grid": table_rows, "chosen": chosen, "best_single_on_choose": best, "results": res},
        indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("table", "run"))
    return table() if ap.parse_args(argv).phase == "table" else run_phase()


if __name__ == "__main__":
    raise SystemExit(main())
