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
    # only each deck's CARQ lines are kept -- the only lines carq_rmw_nm reads. Holding every whole deck took the
    # first run past 2.4 GB, and rescanning one per row was most of its CPU.
    carq: dict[str, str] = {}
    jobs = []
    for r in rows:
        aid = r["atcf_id"]
        if aid not in carq:
            p = adeck(aid)
            text = gzip.decompress(p.read_bytes()).decode("utf-8", "replace") if p.exists() else ""
            carq[aid] = "\n".join(line for line in text.splitlines() if "CARQ" in line)
        rmw = br.carq_rmw_nm(carq[aid], r["dtg"])
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


FIXTURE = ROOT / "tests" / "fixtures" / "hurricane_v9"


def export() -> int:
    """Freeze H8 as results/models/hurricane_ri_v10_4.json (label "v10.4", inputs ONH + IR + H8). Refused unless
    the refit reproduces the further read's 2026 log loss, V8 here is the served v10.3, and the payloads
    reproduce the boosters. The fixture storm gains the lab's v10.4 curve and H8 inputs, so the live path is
    tested without the network (its IR crops are already there, from v10.3's export)."""
    from hazardpulse.hurricane import ri_v10
    from hazardpulse.tornado import lgbm_payload as lp
    rep = json.loads(OUT.read_text(encoding="utf-8"))
    if rep["carried"] != "H8":
        raise SystemExit("amendment 8 did not carry H8: nothing to export")
    rows = v10.load(v10.DEV10)
    v10ir.add_ir(rows)
    add_h8(rows, adeck_dev)
    models, rounds = ch.fit("H8", rows)
    c26 = v10.cases_2026()
    v10ir.add_ir(c26)
    add_h8(c26, adeck_2026)
    p = ch.predict("H8", models, c26)
    y = np.array([r["y"] for r in c26])
    ll = v9.summary(y, p[30], False)["log_loss"]
    further = rep["further_read_2026"]["results"]
    want = further["H8"]["summary_30"]["log_loss"]
    if abs(ll - want) > 1e-12:
        raise SystemExit(f"refit 2026 LL {ll!r} != the further read's {want!r}")
    names = list(ch.CANDS["H8"]["names"])
    if names != ri_v10.allowed_feature_sets()["ONH+IR+H8"]:
        raise SystemExit("H8's inputs are not the serving module's ONH+IR+H8 set")
    dev = rep["development"]
    v10_3 = json.loads(ri_v10.V10_3_PATH.read_text(encoding="utf-8"))
    if v10_3["provenance"]["dev_2022_2025"]["log_loss"] != dev["V8"]["summary_30"]["log_loss"]:
        raise SystemExit("V8 here is not the served v10.3")
    members = [lp.export_booster(m, names + ["threshold_kt"], calibration={"method": "identity", "a": 1.0, "b": 0.0},
                                 provenance={}) for m in models]
    art = {"schema": ri_v10.SCHEMA, "model_name": "hurricane_ri_v10_4", "label": "v10.4",
           "thresholds_kt": list(v10.K), "feature_names": names, "members": members,
           "provenance": {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 8)", "candidate": "H8",
                          "prereg_tag": rep["prereg_tag"], "h8_code_sha256": rep["h8_code_sha256"],
                          "trained": "NHC cycles 2020-2025 (IR from 2021-07-12)",
                          "event": "V(t+24 h) - V(t) >= k kt", "rounds": rounds, "seeds": list(v10.SEEDS),
                          "gate_aids": list(ri_v10.GATE_AIDS), "ir_source": "NOAA GMGSI longwave, hour t+2h and t-4h",
                          "h8_source": "balanced response lap(tau) - tau/ell^2 + S/ell^2 = 0, ell = c/I(r); "
                                       "vortex from CARQ v0, RMW and latitude at t",
                          "monotone_up": [n for n in names if n in ch.M_UP],
                          "dev_2022_2025": {"log_loss": dev["H8"]["summary_30"]["log_loss"],
                                            "auc": dev["H8"]["summary_30"]["auc"], "brier4": dev["H8"]["brier4"],
                                            "champion_label": "v10.3",
                                            "champion_log_loss": dev["V8"]["summary_30"]["log_loss"],
                                            "champion_brier4": dev["V8"]["brier4"],
                                            "d_log_loss_vs_champion_ci": dev["paired"]["d_ll_ci"]},
                          "season_2026_further_read": {"log_loss": ll,
                                                       "champion_log_loss": further["V8"]["summary_30"]["log_loss"],
                                                       "d_log_loss_ci": further["paired"]["d_ll_ci"],
                                                       "declared": "2026 informed v10's design and amendment 5's "
                                                                   "read; no claim"}}}
    ri_v10.V10_4_PATH.write_bytes(ri_v10.canonical_bytes(art))
    loaded, version = ri_v10.load(ri_v10.V10_4_PATH)
    X = v9.design(c26, names)
    worst = max(float(np.max(np.abs(ri_v10.predict_matrix(loaded, X, k) - p[k]))) for k in v10.MULTI)
    if worst > 1e-9:
        ri_v10.V10_4_PATH.unlink()
        raise SystemExit(f"artifact disagrees with the boosters by {worst:.2e}")
    fix = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    at = {r["dtg"]: i for i, r in enumerate(c26) if r["atcf_id"] == fix["storm"]}
    for case in fix["cases"]:
        i = at[case["dtg"]]
        case["v10_4"] = {str(k): float(p[k][i]) for k in v10.MULTI}
        case["h8"] = {n: float(c26[i]["f"][n]) for n in br.H8_NAMES}
    (FIXTURE / "expected.json").write_text(json.dumps(fix, indent=1) + "\n", encoding="utf-8")
    v9.log(f"{ri_v10.V10_4_PATH.name}: {version}; rounds {rounds}; 2026 LL {ll:.4f} reproduced; artifact = boosters "
           f"to {worst:.1e}; fixture gained the lab's v10.4 curve and H8 inputs")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="select", choices=("select", "export"))
    return export() if ap.parse_args(argv).phase == "export" else select()


if __name__ == "__main__":
    raise SystemExit(main())
