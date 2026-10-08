"""Hurricane RI amendment 8 (docs/HURRICANE_RI_V9_PROGRAM.md): H8 = V8 + the coherence equation's balanced response.

    PYTHONPATH=src python scripts/hurricane_ri_h8.py        # after scripts/hurricane_ir_crops.py

Control: V8 recomputed here must reproduce amendment 5's development log loss to 1e-12, or the run stops.
Carried rule: H8's pooled 30/24 log loss AND four-threshold Brier both below V8's. Also reported, descriptive:
paired intervals, POD at the HCCA call's false-alarm rate, the drift check, and the H8 features' share of the
split gain. 2026 is a declared further read.

The balanced response is solved once per image (~0.2 s each) and cached under the cache's hurricane_ri_v9
directory, keyed by the bytes of the code that computes it and by each cycle's inputs.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v10 as v10  # noqa: E402
import hurricane_ri_v10_challengers as ch  # noqa: E402
import hurricane_ri_v10_ir as v10ir  # noqa: E402
import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import balanced_response as br  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_h8.json"
V8_LL = 0.1417825586152726
ch.CANDS["H8"] = dict(names=ch.CANDS["V8"]["names"] + list(br.H8_NAMES), mono=True)   # H8 is not in M_UP: unconstrained
WORKERS = 2
# the code H8 is computed by: a change to any of these is a different feature, so a different cache
H8_CODE = ("src/hazardpulse/hurricane/balanced_response.py", "src/hazardpulse/hurricane/ir_features.py",
           "src/hazardpulse/coherence/tau_c_solver.py")


def code_digest() -> str:
    h = hashlib.sha256()
    for rel in H8_CODE:
        h.update(rel.encode() + b"\0" + (ROOT / rel).read_bytes())
    return h.hexdigest()


def adeck_dev(aid: str) -> Path:
    return v9.adeck_path(aid)


def adeck_2026(aid: str) -> Path:
    return v9.Y26 / "adeck" / f"a{aid[:2].lower()}{aid[2:]}.dat.gz"


def _one(job: tuple) -> dict:
    aid, dtg, vmax, rmw, lat = job
    now, before = v10ir._load(aid, dtg, "p2"), v10ir._load(aid, dtg, "m4")
    return br.features(now, before, vmax, rmw, lat)


def add_h8(rows: list[dict], adeck) -> dict:
    """Write H8 into each row's features: the vortex (``v0``, the CARQ RMW, ``abs_lat``) at t and amendment 5's
    two images. Returns coverage counts."""
    texts: dict[str, str] = {}
    jobs = []
    for r in rows:
        aid = r["atcf_id"]
        if aid not in texts:
            p = adeck(aid)
            texts[aid] = gzip.decompress(p.read_bytes()).decode("utf-8", "replace") if p.exists() else ""
        rmw = br.carq_rmw_nm(texts[aid], r["dtg"])
        jobs.append((aid, r["dtg"], float(r["f"]["v0"]), rmw, float(r["f"]["abs_lat"])))
    path = v9.WORK / f"h8_features_{code_digest()[:16]}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = [json.dumps(j) for j in jobs]
    todo = sorted({k for k in key if k not in cache})
    if todo:
        v9.log(f"H8: solving {len(todo)} cycles ({2 * len(todo)} images) on {WORKERS} workers")
        with ProcessPoolExecutor(WORKERS) as pool:
            for k, f in zip(todo, pool.map(_one, [tuple(json.loads(k)) for k in todo], chunksize=16)):
                cache[k] = f
        path.write_text(json.dumps(cache), encoding="utf-8")
    for r, k in zip(rows, key):
        r["f"].update(cache[k])
    return {"cycles": len(rows), "rmw_present": int(sum(math.isfinite(j[3]) for j in jobs)),
            "h8_present": int(sum(math.isfinite(r["f"]["h8_core_balanced"]) for r in rows)),
            "d6_present": int(sum(math.isfinite(r["f"]["h8_core_balanced_d6"]) for r in rows)),
            "cache": path.name}


def gain_share(models) -> dict:
    """The H8 features' share of the split gain, per feature and in total, averaged over the seeds."""
    names = list(ch.CANDS["H8"]["names"]) + ["threshold_kt"]
    g = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    total = float(g.sum())
    per = {n: float(g[names.index(n)] / total) for n in br.H8_NAMES}
    return {"total": float(sum(per.values())), "per_feature": per}


def select() -> int:
    rows = v10.load(v10.DEV10)
    v10ir.add_ir(rows)
    cov = add_h8(rows, adeck_dev)
    P = {c: {k: [] for k in v10.MULTI} for c in ("V8", "H8")}
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
    v8_ll = v9.summary(y, P["V8"][30], False)["log_loss"]
    if abs(v8_ll - V8_LL) > 1e-12:
        raise SystemExit(f"control failed: V8 recomputed LL {v8_ll!r} != amendment 5's {V8_LL!r}")
    dev = v10ir.compare(test_rows, P, "V8", "H8")
    carried = bool(dev["H8"]["summary_30"]["log_loss"] < dev["V8"]["summary_30"]["log_loss"]
                   and dev["H8"]["brier4"] < dev["V8"]["brier4"])
    covered = int(sum(math.isfinite(r["f"]["h8_core_balanced"]) for r in test_rows))
    for c in ("V8", "H8"):
        d = dev[c]
        v9.log(f"DEV {c}: LL {d['summary_30']['log_loss']:.4f} AUC {d['summary_30']['auc']:.4f} Brier4 {d['brier4']:.4f}"
               f" | at HCCA's FAR POD {d['vs_hcca_call']['ours_pod']:.3f} vs {d['vs_hcca_call']['aid_pod']:.3f}")
    pr = dev["paired"]
    v9.log(f"H8 - V8: dLL {pr['d_ll']:+.4f} [{pr['d_ll_ci'][0]:+.4f}, {pr['d_ll_ci'][1]:+.4f}]  dBrier4 "
           f"{pr['d_brier4']:+.4f} [{pr['d_brier4_ci'][0]:+.4f}, {pr['d_brier4_ci'][1]:+.4f}]; H8 on {covered} of "
           f"{len(test_rows)} scored cycles; CARRIED H8: {carried}")
    dr = v10ir.drift(rows, br.H8_NAMES)
    flags = [n for n, v in dr.items() if v and v["flag"]]
    v9.log(f"drift 2022-2024 -> 2025: flagged {flags or 'none'}")

    c26 = v10.cases_2026()
    v10ir.add_ir(c26)
    cov26 = add_h8(c26, adeck_2026)
    fits = {c: ch.fit(c, rows) for c in ("V8", "H8")}
    share = gain_share(fits["H8"][0])
    v9.log(f"H8 share of split gain (all development seasons): {share['total']:.3f} "
           + " ".join(f"{n} {s:.3f}" for n, s in share["per_feature"].items()))
    P26 = {c: ch.predict(c, fits[c][0], c26) for c in fits}
    further = v10ir.compare(c26, P26, "V8", "H8")
    v9.log(f"2026 (further read): V8 LL {further['V8']['summary_30']['log_loss']:.4f}, H8 LL "
           f"{further['H8']['summary_30']['log_loss']:.4f}, dLL {further['paired']['d_ll']:+.4f} "
           f"[{further['paired']['d_ll_ci'][0]:+.4f}, {further['paired']['d_ll_ci'][1]:+.4f}]")
    OUT.write_text(json.dumps({
        "phase": "amendment 8 select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 8)",
        "prereg_tag": "prereg-hurricane-ri-amend8",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": v9.sha(v10.DEV10), "h8_code_sha256": code_digest(), "control_V8_log_loss": v8_ll,
        "h8_names": list(br.H8_NAMES), "coverage_dev": cov, "scored_cycles_with_h8": covered,
        "development": dev, "carried": "H8" if carried else "V8", "drift_2025_vs_2022_2024": dr,
        "gain_share_all_dev": share,
        "further_read_2026": {"declared": "2026 informed v10's design and amendment 5's read; reported, no claim",
                              "coverage": cov26, "results": further}}, indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(select())
