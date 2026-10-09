"""Hurricane RI amendments 12 and 12b (docs/HURRICANE_RI_V9_PROGRAM.md): G1 = H8 + the inner core from GOES 2 km, hourly.

    PYTHONPATH=src python scripts/hurricane_ri_g1.py stats <shard dir>   # per-hour inner-core stats from the crops
    PYTHONPATH=src python scripts/hurricane_ri_g1.py select              # the registered test

Control: H8 recomputed must reproduce amendment 8's development log loss to 1e-12, or the run stops.
Carried rule: G1's pooled 30/24 log loss AND four-threshold Brier both below H8's.
Pre-registered descriptive checks:
- paired intervals, and POD at the HCCA call's false-alarm rate;
- the drift check;
- the G1 features' share of split gain;
- a NOISE CONTROL: the same features shuffled within each season, three seeds.
2026 is a declared further read.
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
from hurricane_ri_h9 import ch, v9, v10, v10ir  # noqa: E402
from hazardpulse.hurricane import goes_features as gf  # noqa: E402

STATS = v9.WORK / "goes_hour_stats.json"
OUT = ROOT / "results" / "calibration" / "hurricane_ri_g1.json"
H8_LL = 0.14161766094567663
ch.CANDS["G1"] = dict(names=ch.CANDS["H8"]["names"] + list(gf.G1_NAMES), mono=True)   # not in M_UP: unconstrained
NOISE_SEEDS = (1, 2, 3)


def stats(shard_dir: str, tasks_path: Path | None = None) -> int:
    """Per-hour inner-core stats for every collected task (``status == "ok"``), keyed by task key -- only from a
    collection ``goes_collect.verify`` accepts (amendment 12b's fields, aligned, every task key exactly once)."""
    import goes_collect
    status_counts = goes_collect.verify(shard_dir, tasks_path)
    out, n_ok, n_all = {}, 0, 0
    for p in sorted(Path(shard_dir).rglob("goes_shard_*.npz")):
        with np.load(p) as z:
            for k, img, eye, te, tc, st in zip(z["keys"], z["images"], z["eye"], z["teye"], z["tcw"], z["status"]):
                n_all += 1
                if str(st) != "ok":
                    continue
                n_ok += 1
                s = gf.hour_stats(img, bool(eye), float(te), float(tc))
                out[str(k)] = {n: (None if isinstance(v, float) and math.isnan(v) else v) for n, v in s.items()}
    STATS.write_text(json.dumps({"tasks": n_all, "ok": n_ok, "status": status_counts, "stats": out}), encoding="utf-8")
    v9.log(f"{n_ok} of {n_all} tasks ok ({status_counts}); wrote {STATS}")
    return 0


def _hour_key(storm: str, t: dt.datetime, k: int) -> str:
    h = (t + dt.timedelta(hours=k)).strftime("%Y%m%d%H")
    return f"{storm}_{h}_interp" if k <= 0 else f"{storm}_{h}_extrap_{t:%Y%m%d%H}"


def add_g1(rows: list[dict], table: dict) -> dict:
    """G1's features on each row from the per-hour stats of its window (t-12 .. t+2)."""
    n_any = 0
    for r in rows:
        t = dt.datetime.strptime(r["dtg"], "%Y%m%d%H")
        hours = []
        for k in range(-12, 3):
            s = table.get(_hour_key(r["atcf_id"], t, k))
            hours.append((k, None if s is None else {n: (math.nan if v is None else v) for n, v in s.items()}))
        f = gf.features(hours)
        r["f"].update(f)
        n_any += f["g_n"] > 0
    return {"cycles": len(rows), "with_any_hour": n_any,
            "with_latest_hour": int(sum(1 for r in rows if math.isfinite(r["f"]["g_core_cold"])))}


def _folds(rows, names_cand: str):
    P = {c: {k: [] for k in v10.MULTI} for c in ("H8", names_cand)}
    test_rows = []
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        for c in P:
            models, _ = ch.fit(c, train)
            for k, v in ch.predict(c, models, test).items():
                P[c][k].append(v)
        test_rows.extend(test)
    return test_rows, {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in P.items()}


def noise_control(rows) -> list[dict]:
    """G1's features shuffled within each season (marginals kept, meaning destroyed): what adding any 12 columns
    of this kind costs or gains."""
    shuf = [f"{n}_shuf" for n in gf.G1_NAMES]
    ch.CANDS["G1S"] = dict(names=ch.CANDS["H8"]["names"] + shuf, mono=True)
    out = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        for season in sorted({r["season"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["season"] == season]
            perm = rng.permutation(idx)
            for n, s in zip(gf.G1_NAMES, shuf):
                vals = [rows[j]["f"][n] for j in perm]
                for i, v in zip(idx, vals):
                    rows[i]["f"][s] = v
        test_rows, P = _folds(rows, "G1S")
        pr = v10ir.compare(test_rows, P, "H8", "G1S")["paired"]
        out.append({"seed": seed, **pr})
        v9.log(f"noise control seed {seed}: dLL {pr['d_ll']:+.5f} [{pr['d_ll_ci'][0]:+.5f}, {pr['d_ll_ci'][1]:+.5f}]")
    return out


def gain_share(models) -> dict:
    names = list(ch.CANDS["G1"]["names"]) + ["threshold_kt"]
    gsum = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    total = float(gsum.sum())
    per = {n: float(gsum[names.index(n)] / total) for n in gf.G1_NAMES}
    return {"total": float(sum(per.values())), "per_feature": per}


def select() -> int:
    doc = json.loads(STATS.read_text(encoding="utf-8"))
    table = doc["stats"]
    rows = v10.load(v10.DEV10)
    cov = h9.prepare(rows, h9.h8.adeck_dev)
    cov["goes"] = add_g1(rows, table)
    test_rows, P = _folds(rows, "G1")
    y = np.array([r["y"] for r in test_rows])
    h8_ll = v9.summary(y, P["H8"][30], False)["log_loss"]
    if abs(h8_ll - H8_LL) > 1e-12:
        raise SystemExit(f"control failed: H8 recomputed LL {h8_ll!r} != amendment 8's {H8_LL!r}")
    dev = v10ir.compare(test_rows, P, "H8", "G1")
    carried = bool(dev["G1"]["summary_30"]["log_loss"] < dev["H8"]["summary_30"]["log_loss"]
                   and dev["G1"]["brier4"] < dev["H8"]["brier4"])
    for c in ("H8", "G1"):
        d = dev[c]
        v9.log(f"DEV {c}: LL {d['summary_30']['log_loss']:.5f} AUC {d['summary_30']['auc']:.4f} Brier4 {d['brier4']:.5f}"
               f" | at HCCA's FAR POD {d['vs_hcca_call']['ours_pod']:.3f} vs {d['vs_hcca_call']['aid_pod']:.3f}")
    pr = dev["paired"]
    v9.log(f"G1 - H8: dLL {pr['d_ll']:+.5f} [{pr['d_ll_ci'][0]:+.5f}, {pr['d_ll_ci'][1]:+.5f}]  dBrier4 "
           f"{pr['d_brier4']:+.5f} [{pr['d_brier4_ci'][0]:+.5f}, {pr['d_brier4_ci'][1]:+.5f}]; CARRIED G1: {carried}")
    dr = v10ir.drift(rows, gf.G1_NAMES)
    noise = noise_control(rows)
    c26 = v10.cases_2026()
    cov26 = h9.prepare(c26, h9.h8.adeck_2026)
    cov26["goes"] = add_g1(c26, table)
    fits = {c: ch.fit(c, rows) for c in ("H8", "G1")}
    share = gain_share(fits["G1"][0])
    v9.log(f"G1 share of split gain: {share['total']:.3f}")
    P26 = {c: ch.predict(c, fits[c][0], c26) for c in fits}
    further = v10ir.compare(c26, P26, "H8", "G1")
    v9.log(f"2026 (further read): H8 LL {further['H8']['summary_30']['log_loss']:.4f}, G1 LL "
           f"{further['G1']['summary_30']['log_loss']:.4f}, dLL {further['paired']['d_ll']:+.4f} "
           f"[{further['paired']['d_ll_ci'][0]:+.4f}, {further['paired']['d_ll_ci'][1]:+.4f}]")
    OUT.write_text(json.dumps({
        "phase": "amendment 12b select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendments 12, 12b)",
        "prereg_tag": "prereg-hurricane-ri-amend12b",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": v9.sha(v10.DEV10), "control_H8_log_loss": h8_ll, "g1_names": list(gf.G1_NAMES),
        "crop_status": doc["status"], "coverage_dev": cov, "development": dev, "carried": "G1" if carried else "H8",
        "noise_control": noise, "drift_2025_vs_2022_2024": dr, "gain_share_all_dev": share,
        "further_read_2026": {"declared": "2026 informed earlier amendments; reported, no claim", "coverage": cov26,
                              "results": further}}, indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("stats", "select"))
    ap.add_argument("shard_dir", nargs="?")
    a = ap.parse_args(argv)
    return stats(a.shard_dir) if a.phase == "stats" else select()


if __name__ == "__main__":
    raise SystemExit(main())
