"""Hurricane RI amendment 3 (docs/HURRICANE_RI_V9_PROGRAM.md): challengers to v10.1.

    PYTHONPATH=src python scripts/hurricane_ri_v10_challengers.py

V5 = V2 + M (monotone constraints), V6 = V2 + R (revisions since the previous cycle), V7 = V2 + R + M,
on V2's development table, folds, seeds and LightGBM settings. Control: V2 recomputed here must
reproduce its selection log loss bit for bit, or the run stops. The carried rule is the protocol's;
2026 is a declared third read (reported, no claim).
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

import hurricane_ri_v10 as v10  # noqa: E402
import hurricane_ri_v10_vs_all as vs_all  # noqa: E402
import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import ri_v9_features as fx  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_v10_challengers.json"
V2_LL = 0.1443435895033004
BASE = list(v10.V9_NAMES)
# R: what changed since the previous cycle (t - 6 h) of the same storm
REV_DV = ("dv24_DSHP", "dv24_LGEM", "dv24_IVCN", "dv24_HCCA", "dv24_NNIC", "dv24_regional_mean", "ofcl_dv24",
          "frac_ge30", "ri_DTOP_30_24", "ri_RIOD_30_24", "ri_RIOC_30_24")
R_NAMES = tuple(f"rev_{n}" for n in REV_DV) + ("dv_past6",)
# M: features whose increase may never lower the probability
M_UP = frozenset((*(f"dv24_{a}" for a in fx.INDIVIDUAL), "dv24_regional_mean", "dv24_regional_max",
                  "dv24_global_mean", "frac_ge30", "ofcl_dv24", "ofcl_dv12", *fx.N_NAMES))
CANDS = {"V2": dict(names=BASE, mono=False), "V5": dict(names=BASE, mono=True),
         "V6": dict(names=BASE + list(R_NAMES), mono=False), "V7": dict(names=BASE + list(R_NAMES), mono=True)}


def add_revisions(rows: list[dict]) -> int:
    """Write R into each row's features from the same storm's row at t - 6 h (NaN without one)."""
    index = {(r["atcf_id"], r["dtg"]): r for r in rows}
    found = 0
    for r in rows:
        prev_dtg = (dt.datetime.strptime(r["dtg"], "%Y%m%d%H") - dt.timedelta(hours=6)).strftime("%Y%m%d%H")
        prev = index.get((r["atcf_id"], prev_dtg))
        found += prev is not None
        for n in REV_DV:
            now = float(r["f"].get(n, math.nan))
            before = float(prev["f"].get(n, math.nan)) if prev is not None else math.nan
            r["f"][f"rev_{n}"] = now - before
    return found


def monotone(names: list[str], mono: bool) -> list[int]:
    return [(1 if (mono and n in M_UP) else 0) for n in names] + [-1]      # the threshold column


def fit(cand: str, train: list[dict]):
    """v10.fit_models with this candidate's names and constraints (V2 must reproduce v10's own)."""
    import lightgbm as lgb
    spec = CANDS[cand]
    names, mono = spec["names"], monotone(spec["names"], spec["mono"])
    last = max(r["season"] for r in train)
    Xa, ya, _ = v10._rows(train, names, v10.K, False)
    Xi, yi, _ = v10._rows([r for r in train if r["season"] < last], names, v10.K, False)
    Xv, yv, _ = v10._rows([r for r in train if r["season"] == last], names, v10.K, False)
    models, rounds = [], []
    for s in v10.SEEDS:
        p = dict(v9.GBT_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s,
                 monotone_constraints=mono)
        b = lgb.train(p, lgb.Dataset(Xi, yi), 2000, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        r = max(1, int(b.best_iteration or 1))
        rounds.append(r)
        models.append(lgb.train(p, lgb.Dataset(Xa, ya), r))
    return models, rounds


def predict(cand: str, models, rows: list[dict]) -> dict[int, np.ndarray]:
    X = v9.design(rows, CANDS[cand]["names"])
    return {k: np.mean([m.predict(np.hstack([X, np.full((len(X), 1), float(k))])) for m in models], axis=0)
            for k in v10.MULTI}


def brier4(rows, probs: dict) -> np.ndarray:
    return v10.multi_brier(rows, probs, False)


def paired_vec(a: np.ndarray, b: np.ndarray, groups, reps=v9.REPS, seed=v9.SEED) -> list[float]:
    """Storm-bootstrap 95% interval of mean(b - a)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d = [(b[idx] - a[idx]).mean() for idx in
         (np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(reps))]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def evaluate(rows, P: dict, groups) -> dict:
    y = np.array([r["y"] for r in rows])
    hcca = np.array([r["f"]["dv24_HCCA"] for r in rows])
    has = np.isfinite(hcca)
    out = {}
    ll2 = v9._ll_vec(y, P["V2"][30], False)
    b2 = brier4(rows, P["V2"])
    for c in P:
        llc = v9._ll_vec(y, P[c][30], False)
        bc = brier4(rows, P[c])
        call = vs_all.matched_paired(y[has], P[c][30][has], (hcca[has] >= 30).astype(float), groups[has])
        out[c] = {"summary_30": v9.summary(y, P[c][30], False), "brier4": float(bc.mean()),
                  "d_ll_vs_V2": float(llc.mean() - ll2.mean()), "d_ll_vs_V2_ci": paired_vec(ll2, llc, groups),
                  "d_brier4_vs_V2": float(bc.mean() - b2.mean()), "d_brier4_vs_V2_ci": paired_vec(b2, bc, groups),
                  "vs_hcca_call": {k: call[k] for k in ("aid_pod", "ours_pod", "d_pod", "d_pod_ci", "aid_pofd")}}
    return out


FIXTURE = ROOT / "tests" / "fixtures" / "hurricane_v9" / "expected.json"
LABELS = {"V5": "v10.2", "V6": "v10.2", "V7": "v10.2"}


def export() -> int:
    """Freeze the carried challenger as results/models/hurricane_ri_v10_2.json (v10.1's schema and
    inputs, label "v10.2"). Refused unless the refit reproduces the third read's 2026 log loss and the
    payloads reproduce the boosters; the live-path fixture gains the lab's v10.2 curve."""
    from hazardpulse.hurricane import ri_v10
    from hazardpulse.tornado import lgbm_payload as lp
    rep = json.loads(OUT.read_text(encoding="utf-8"))
    c = rep["carried"]
    if c == "V2" or CANDS[c]["names"] != BASE:
        raise SystemExit(f"carried {c}: nothing to export as a v10-schema challenger")
    rows = v10.load(v10.DEV10)
    add_revisions(rows)
    models, rounds = fit(c, rows)
    c26 = v10.cases_2026()
    add_revisions(c26)
    p = predict(c, models, c26)
    y = np.array([r["y"] for r in c26])
    ll = v9.summary(y, p[30], False)["log_loss"]
    want = rep["third_read_2026"]["candidates"][c]["summary_30"]["log_loss"]
    if abs(ll - want) > 1e-12:
        raise SystemExit(f"refit 2026 LL {ll!r} != the third read's {want!r}")
    names = BASE + ["threshold_kt"]
    members = [lp.export_booster(m, names, calibration={"method": "identity", "a": 1.0, "b": 0.0}, provenance={})
               for m in models]
    dev = rep["development"]
    art = {"schema": ri_v10.SCHEMA, "model_name": "hurricane_ri_v10_2", "label": LABELS[c],
           "thresholds_kt": list(v10.K), "feature_names": BASE, "members": members,
           "provenance": {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 3)", "candidate": c,
                          "prereg_tag": rep["prereg_tag"], "trained": "NHC cycles 2020-2025",
                          "event": "V(t+24 h) - V(t) >= k kt", "rounds": rounds, "seeds": list(v10.SEEDS),
                          "gate_aids": list(ri_v10.GATE_AIDS),
                          "monotone_up": [n for n in BASE if n in M_UP],
                          "dev_2022_2025": {"log_loss": dev[c]["summary_30"]["log_loss"],
                                            "auc": dev[c]["summary_30"]["auc"], "brier4": dev[c]["brier4"],
                                            "champion_log_loss": dev["V2"]["summary_30"]["log_loss"],
                                            "champion_brier4": dev["V2"]["brier4"],
                                            "d_log_loss_vs_champion_ci": dev[c]["d_ll_vs_V2_ci"]},
                          "season_2026_third_read": {"log_loss": ll, "champion_log_loss":
                                                     rep["third_read_2026"]["candidates"]["V2"]["summary_30"]["log_loss"],
                                                     "declared": "2026 informed v10's design; no claim"}}}
    ri_v10.V10_2_PATH.write_bytes(ri_v10.canonical_bytes(art))
    loaded, version = ri_v10.load(ri_v10.V10_2_PATH)
    X = v9.design(c26, BASE)
    worst = max(float(np.max(np.abs(ri_v10.predict_matrix(loaded, X, k) - p[k]))) for k in v10.MULTI)
    if worst > 1e-9:
        raise SystemExit(f"artifact disagrees with the boosters by {worst:.2e}")
    fix = json.loads(FIXTURE.read_text(encoding="utf-8"))
    at = {r["dtg"]: i for i, r in enumerate(c26) if r["atcf_id"] == fix["storm"]}
    for case in fix["cases"]:
        i = at[case["dtg"]]
        case["v10_2"] = {str(k): float(p[k][i]) for k in v10.MULTI}
    FIXTURE.write_text(json.dumps(fix, indent=1) + "\n", encoding="utf-8")
    v9.log(f"{ri_v10.V10_2_PATH.name}: {version}; {len(members)} members, rounds {rounds}; 2026 LL {ll:.4f} "
           f"reproduced; artifact = boosters to {worst:.1e}; fixture gained the lab's v10.2 curve")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="select", choices=("select", "export"))
    return export() if ap.parse_args(argv).phase == "export" else select()


def select() -> int:
    rows = v10.load(v10.DEV10)
    found = add_revisions(rows)
    v9.log(f"{len(rows)} development rows, {found} with the previous cycle")
    P = {c: {k: [] for k in v10.MULTI} for c in CANDS}
    test_rows, rounds = [], {c: {} for c in CANDS}
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        for c in CANDS:
            models, rnd = fit(c, train)
            rounds[c][season] = rnd
            for k, v in predict(c, models, test).items():
                P[c][k].append(v)
        test_rows.extend(test)
        v9.log(f"fold {season} done")
    P = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in P.items()}
    y = np.array([r["y"] for r in test_rows])
    groups = np.array([r["sid"] for r in test_rows])
    v2_ll = v9.summary(y, P["V2"][30], False)["log_loss"]
    if abs(v2_ll - V2_LL) > 1e-12:
        raise SystemExit(f"control failed: V2 recomputed LL {v2_ll!r} != selection's {V2_LL!r}")
    dev = evaluate(test_rows, P, groups)
    eligible = [c for c in ("V5", "V6", "V7")
                if dev[c]["summary_30"]["log_loss"] < dev["V2"]["summary_30"]["log_loss"]
                and dev[c]["brier4"] < dev["V2"]["brier4"]]
    carried = min(eligible, key=lambda c: (dev[c]["summary_30"]["log_loss"], ("V5", "V6", "V7").index(c))) \
        if eligible else "V2"
    for c in CANDS:
        d = dev[c]
        v9.log(f"DEV {c}: LL {d['summary_30']['log_loss']:.4f} AUC {d['summary_30']['auc']:.4f} "
               f"Brier4 {d['brier4']:.4f} | vs V2 dLL {d['d_ll_vs_V2']:+.4f} [{d['d_ll_vs_V2_ci'][0]:+.4f}, "
               f"{d['d_ll_vs_V2_ci'][1]:+.4f}] dBrier4 {d['d_brier4_vs_V2']:+.4f} [{d['d_brier4_vs_V2_ci'][0]:+.4f}, "
               f"{d['d_brier4_vs_V2_ci'][1]:+.4f}] | at HCCA's FAR POD {d['vs_hcca_call']['ours_pod']:.3f} vs "
               f"{d['vs_hcca_call']['aid_pod']:.3f}")
    v9.log(f"CARRIED: {carried}")

    # 2026, a declared third read: every candidate fitted on all development seasons
    c26 = v10.cases_2026()
    found26 = add_revisions(c26)
    P26 = {c: predict(c, fit(c, rows)[0], c26) for c in CANDS}
    g26 = np.array([r["sid"] for r in c26])
    third = evaluate(c26, P26, g26)
    y26 = np.array([r["y"] for r in c26])
    dtops26 = {"log_loss": v9.summary(y26, v9.a_forecast(c26), True)["log_loss"],
               "brier4": float(v10.multi_brier(c26, None, True).mean())}
    for c in CANDS:
        d = third[c]
        v9.log(f"2026 {c}: LL {d['summary_30']['log_loss']:.4f} Brier4 {d['brier4']:.4f} | vs V2 dLL "
               f"{d['d_ll_vs_V2']:+.4f} [{d['d_ll_vs_V2_ci'][0]:+.4f}, {d['d_ll_vs_V2_ci'][1]:+.4f}]")
    v9.log(f"2026 DTOPS: LL {dtops26['log_loss']:.4f} Brier4 {dtops26['brier4']:.4f}")
    OUT.write_text(json.dumps({
        "phase": "amendment 3 select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 3)",
        "prereg_tag": "prereg-hurricane-ri-amend3",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": v9.sha(v10.DEV10), "control_V2_log_loss": v2_ll, "rows_with_previous_cycle": found,
        "r_names": list(R_NAMES), "m_up": sorted(M_UP), "rounds": rounds, "development": dev,
        "eligible": eligible, "carried": carried,
        "third_read_2026": {"declared": "2026 informed v10's design; reported, no claim", "n": len(c26),
                            "rows_with_previous_cycle": found26, "dtops": dtops26, "candidates": third}},
        indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
