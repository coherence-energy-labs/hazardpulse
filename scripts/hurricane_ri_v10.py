"""Hurricane RI v10 (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 2).

    PYTHONPATH=src python scripts/hurricane_ri_v10.py build-dev    # v9's cases + E features + dV + raw aid %
    PYTHONPATH=src python scripts/hurricane_ri_v10.py select       # forward chaining 2022-2025: R0, V1-V4
    PYTHONPATH=src python scripts/hurricane_ri_v10.py second-read  # 2026, DECLARED second read (no claim)

Control: R0 is v9's D_gbt recomputed by this script; its pooled development log loss must equal
the frozen selection's (0.1459...) or the run stops.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import atcf  # noqa: E402
from hazardpulse.hurricane import ri_v9_features as fx  # noqa: E402

DEV10 = v9.WORK / "dev_cases_v10.jsonl"
SEL10 = ROOT / "results" / "calibration" / "hurricane_ri_v10_selection.json"
SECOND = ROOT / "results" / "calibration" / "hurricane_ri_v10_2026_second_read.json"
K = (15, 20, 25, 30, 35, 40, 45)
MULTI = (25, 30, 35, 40)          # the 24-h thresholds DTOPS publishes
SEEDS = (0, 1, 2, 3, 4)
V9_NAMES = fx.names_for("ONH")
V10_NAMES = fx.names_for("ONHE")
CANDS = {"V1": dict(names=V10_NAMES, stacked=False, mask=False),
         "V2": dict(names=V9_NAMES, stacked=True, mask=False),
         "V3": dict(names=V10_NAMES, stacked=True, mask=False),
         "V4": dict(names=V10_NAMES, stacked=True, mask=True)}
log = v9.log


def raw_pcts_edeck(by_th, aid, dtg):
    out = {}
    for th in fx.THRESHOLDS:
        slot = by_th[th].get((aid, dtg), {})
        for tech in fx.RI_TECHS:
            if tech in slot:
                out[f"{tech}_{th}"] = slot[tech].prob_pct
    return out


def features_for(records, cyc, basin, pcts_tuple):
    f = fx.adeck_features(fx.cycle_table(records, cyc), basin)
    f.update(fx.ri_features(pcts_tuple))
    v0 = f["v0"] if math.isfinite(f["v0"]) else None
    f.update(fx.error_features(records, cyc, v0))
    return f


def build_dev(_a) -> int:
    cases = [json.loads(l) for p in v9.STACK_CASES for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    records, _ = v9.bench.load_edecks(v9.DECKS, range(2020, 2026))
    by_th = {}
    for th in fx.THRESHOLDS:
        kt, hr = fx.threshold_key(th)
        by_th[th], _ = v9.bench.index_ri(records, tau=hr, dv_kt=kt, start_tau=0, stop_tau=hr)
    adecks, out, st = {}, [], Counter()
    for c in cases:
        aid = c["atcf_id"]
        if aid not in adecks:
            p = v9.adeck_path(aid)
            adecks[aid] = atcf.parse_atcf_deck(gzip.decompress(p.read_bytes()).decode("utf-8", "replace")) if p.exists() else []
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        raw = raw_pcts_edeck(by_th, aid, c["dtg"])
        tup = {(k.split("_")[0], k.split("_", 1)[1]): v for k, v in raw.items()}
        f = features_for(adecks[aid], cyc, c["basin"], tup)
        dv = float(c["v_t24"]) - float(c["v_t"])
        if int(dv >= 30) != int(c["y"]):
            st["label_mismatch"] += 1
        st["e_err12_mean_present"] += int(math.isfinite(f["err12_mean"]))
        out.append({"atcf_id": aid, "dtg": c["dtg"], "sid": c["sid"], "season": c["season"], "basin": c["basin"],
                    "y": int(c["y"]), "dv": dv, "a_dtop": c["aids"].get("DTOP"), "a_riod": c["aids"].get("RIOD"),
                    "pcts": raw, "f": f})
    DEV10.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out), encoding="utf-8")
    log(f"{len(out)} cases; {dict(st)}; wrote {DEV10} sha256 {v9.sha(DEV10)}")
    return 0


def load(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["f"] = {k: (float("nan") if v is None else float(v)) for k, v in r["f"].items()}
    return rows


def dtops_at(rows, k: int) -> np.ndarray:
    """DTOPS's own published probability at k kt / 24 h (SHIPS-RII where DTOPS is missing)."""
    out = []
    for r in rows:
        d, s = r["pcts"].get(f"DTOP_{k}/24"), r["pcts"].get(f"RIOD_{k}/24")
        out.append((d if d is not None else s) / 100.0 if (d is not None or s is not None) else np.nan)
    return np.array(out)


def _params(seed: int, n: int, stacked: bool) -> dict:
    p = dict(v9.GBT_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    if stacked:
        p["monotone_constraints"] = [0] * n + [-1]
    return p


def _rows(rows, names, ks, mask: bool):
    X = v9.design(rows, names)
    dv = np.array([r["dv"] for r in rows])
    seasons = np.array([r["season"] for r in rows])
    if mask:
        Xm = X.copy()
        Xm[:, [names.index(n) for n in fx.GUIDANCE_NAMES if n in names]] = np.nan
        X, dv, seasons = np.vstack([X, Xm]), np.concatenate([dv, dv]), np.concatenate([seasons, seasons])
    if ks is None:
        return X, (dv >= 30).astype(float), seasons
    Xs = np.vstack([np.hstack([X, np.full((len(X), 1), float(k))]) for k in ks])
    ys = np.concatenate([(dv >= k).astype(float) for k in ks])
    return Xs, ys, np.tile(seasons, len(ks))


def fit_models(cand: str, train):
    """The candidate's boosters (one per seed) fitted on ``train``, and the rounds each used."""
    import lightgbm as lgb
    spec = CANDS[cand]
    names = list(spec["names"])
    ks = K if spec["stacked"] else None
    last = max(r["season"] for r in train)
    inner_tr = [r for r in train if r["season"] < last]
    inner_va = [r for r in train if r["season"] == last]
    Xa, ya, _ = _rows(train, names, ks, spec["mask"])
    Xi, yi, _ = _rows(inner_tr, names, ks, spec["mask"])
    Xv, yv, _ = _rows(inner_va, names, ks, False)               # validation on real rows only
    models, rounds = [], []
    for s in SEEDS:
        p = _params(s, len(names), spec["stacked"])
        b = lgb.train(p, lgb.Dataset(Xi, yi), 2000, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        r = max(1, int(b.best_iteration or 1))
        rounds.append(r)
        models.append(lgb.train(p, lgb.Dataset(Xa, ya), r))
    return models, rounds


def predict_models(cand: str, models, test, ks_out=(30,)):
    spec = CANDS[cand]
    Xt = v9.design(test, list(spec["names"]))
    preds = {}
    for k in ks_out:
        if spec["stacked"]:
            Xk = np.hstack([Xt, np.full((len(Xt), 1), float(k))])
            preds[k] = np.mean([m.predict(Xk) for m in models], axis=0)
        elif k == 30:
            preds[k] = np.mean([m.predict(Xt) for m in models], axis=0)
    return preds


def fit_predict(cand: str, train, test, ks_out=(30,)):
    """``{k: P(dV >= k)}`` on ``test`` (averaged over SEEDS), and the rounds each seed used."""
    models, rounds = fit_models(cand, train)
    return predict_models(cand, models, test, ks_out), rounds


MODEL10 = ROOT / "results" / "models" / "hurricane_ri_v10.json"


def export(_a) -> int:
    """Freeze the carried stacked model as hurricane_ri_v10.json: every seed's booster as a NumPy
    payload. Refused unless the refit reproduces the second read's reported 2026 log loss and the
    payloads reproduce the boosters."""
    from hazardpulse.hurricane import ri_v10
    from hazardpulse.tornado import lgbm_payload as lp
    sel = json.loads(SEL10.read_text(encoding="utf-8"))
    rep = json.loads(SECOND.read_text(encoding="utf-8"))
    c = sel["carried"]
    if not CANDS[c]["stacked"]:
        raise SystemExit(f"{c} is not a stacked candidate")
    dev = load(DEV10)
    models, rounds = fit_models(c, dev)
    if rounds != rep["models"][c]["rounds"]:
        raise SystemExit(f"refit rounds {rounds} != the second read's {rep['models'][c]['rounds']}")
    test = cases_2026()
    y = np.array([r["y"] for r in test])
    p = predict_models(c, models, test, MULTI)
    ll = v9.summary(y, p[30], False)["log_loss"]
    if abs(ll - rep["models"][c]["all"]["log_loss"]) > 1e-12:
        raise SystemExit(f"refit 2026 LL {ll} != the reported {rep['models'][c]['all']['log_loss']}")
    names = list(CANDS[c]["names"]) + ["threshold_kt"]
    members = [lp.export_booster(m, names, calibration={"method": "identity", "a": 1.0, "b": 0.0}, provenance={})
               for m in models]
    art = {"schema": ri_v10.SCHEMA, "model_name": "hurricane_ri_v10", "thresholds_kt": list(K),
           "feature_names": list(CANDS[c]["names"]), "members": members,
           "provenance": {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 2)", "candidate": c,
                          "trained": "NHC cycles 2020-2025", "event": "V(t+24 h) - V(t) >= k kt", "rounds": rounds,
                          "seeds": list(SEEDS), "gate_aids": list(ri_v10.GATE_AIDS),
                          "dev_2022_2025": {"log_loss": sel["pooled"][c]["log_loss"], "auc": sel["pooled"][c]["auc"],
                                            "dtops_log_loss": sel["pooled"]["A"]["log_loss"],
                                            "d_log_loss_ci": sel["vs_DTOPS"][c]["d_log_loss_ci"],
                                            "multi_threshold": sel["multi_threshold"][c]},
                          "season_2026_second_read": {"log_loss": ll, "auc": rep["models"][c]["all"]["auc"],
                                                      "dtops_log_loss": v9.summary(y, v9.a_forecast(test), True)["log_loss"],
                                                      "declared": "design informed by the 2026 read; no claim"}}}
    MODEL10.write_bytes(ri_v10.canonical_bytes(art))
    loaded, version = ri_v10.load(MODEL10)
    X = np.vstack([v9.design(test, list(CANDS[c]["names"]))])
    worst = 0.0
    for k in MULTI:
        got = ri_v10.predict_matrix(loaded, X, k)
        worst = max(worst, float(np.max(np.abs(got - p[k]))))
    if worst > 1e-9:
        raise SystemExit(f"artifact disagrees with the boosters by {worst:.2e}")
    log(f"{MODEL10.name}: {version}; {len(members)} members, rounds {rounds}; 2026 LL {ll:.4f} reproduced; "
        f"artifact = boosters to {worst:.1e}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build-dev", "select", "second-read", "export"))
    a = ap.parse_args(argv)
    return {"build-dev": build_dev, "select": select, "second-read": second_read, "export": export}[a.phase](a)


def multi_brier(rows, probs: dict, base: bool) -> np.ndarray:
    """Per-row sum over MULTI thresholds of the Brier score (DTOPS's own if ``base``)."""
    dv = np.array([r["dv"] for r in rows])
    tot = np.zeros(len(rows))
    for k in MULTI:
        p = dtops_at(rows, k) if base else probs[k]
        tot += (np.nan_to_num(p, nan=0.0) - (dv >= k)) ** 2
    return tot


def paired_multi(rows, ours: dict, groups, level=0.95, reps=v9.REPS, seed=v9.SEED) -> dict:
    a, b = multi_brier(rows, None, True), multi_brier(rows, ours, False)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d = []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        d.append(b[idx].mean() - a[idx].mean())
    q = (1 - level) / 2
    return {"ours": float(b.mean()), "dtops": float(a.mean()), "d": float(b.mean() - a.mean()),
            "d_ci": [float(np.quantile(d, q)), float(np.quantile(d, 1 - q))]}


def select(_a) -> int:
    rows = load(DEV10)
    sel9 = json.loads(v9.SELECTION_PATH.read_text(encoding="utf-8"))
    preds = {c: {k: [] for k in MULTI} for c in ("R0", *CANDS)}
    ys, groups, test_rows, info = [], [], [], {c: {} for c in ("R0", *CANDS)}
    for y_season in v9.FOLDS:
        train = [r for r in rows if r["season"] < y_season]
        test = [r for r in rows if r["season"] == y_season]
        p0, _ = v9.fit_predict("D_gbt", train, test)                # R0, exactly v9's routine
        preds["R0"][30].append(p0)
        for c, spec in CANDS.items():
            out, rounds = fit_predict(c, train, test, ks_out=MULTI if spec["stacked"] else (30,))
            info[c][y_season] = rounds
            for k, v in out.items():
                preds[c][k].append(v)
        ys.append(np.array([r["y"] for r in test]))
        groups.extend(r["sid"] for r in test)
        test_rows.extend(test)
        log(f"fold {y_season} done")
    y, groups = np.concatenate(ys), np.array(groups)
    P = {c: {k: np.concatenate(v) for k, v in d.items() if v} for c, d in preds.items()}
    pa = v9.a_forecast(test_rows)
    r0_ll = v9.summary(y, P["R0"][30], False)["log_loss"]
    if abs(r0_ll - sel9["pooled"]["D_gbt"]["log_loss"]) > 1e-9:
        raise SystemExit(f"control failed: R0 pooled LL {r0_ll} != v9's {sel9['pooled']['D_gbt']['log_loss']}")
    res = {"A": v9.summary(y, pa, True)}
    vs_a, vs_r0, multi = {}, {}, {}
    for c in P:
        res[c] = v9.summary(y, P[c][30], False)
        vs_a[c] = v9.paired(y, pa, True, P[c][30], False, groups)
        if c != "R0":
            vs_r0[c] = v9.paired(y, P["R0"][30], False, P[c][30], False, groups)
        if all(k in P[c] for k in MULTI):
            multi[c] = paired_multi(test_rows, P[c], groups)
    eligible = [c for c in CANDS if res[c]["log_loss"] < res["R0"]["log_loss"]]
    carried = min(eligible, key=lambda c: (res[c]["log_loss"], list(CANDS).index(c))) if eligible else "R0"
    for c in ("A", "R0", *CANDS):
        line = f"POOLED {c:3s} LL {res[c]['log_loss']:.4f} Brier {res[c]['brier']:.5f} AUC {res[c]['auc']:.4f}"
        if c in vs_a:
            w = vs_a[c]
            line += f" | vs DTOPS dLL {w['d_log_loss']:+.4f} [{w['d_log_loss_ci'][0]:+.4f}, {w['d_log_loss_ci'][1]:+.4f}]"
        if c in vs_r0:
            w = vs_r0[c]
            line += f" | vs R0 dLL {w['d_log_loss']:+.4f} [{w['d_log_loss_ci'][0]:+.4f}, {w['d_log_loss_ci'][1]:+.4f}]"
        if c in multi:
            m = multi[c]
            line += f" | 4-threshold Brier {m['ours']:.4f} vs DTOPS {m['dtops']:.4f} d {m['d']:+.4f} [{m['d_ci'][0]:+.4f}, {m['d_ci'][1]:+.4f}]"
        log(line)
    log(f"CARRIED: {carried}")
    SEL10.write_text(json.dumps({"phase": "select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 2)",
                                 "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "dev_table_sha256": v9.sha(DEV10), "control_R0_log_loss": r0_ll,
                                 "pooled": res, "vs_DTOPS": vs_a, "vs_R0": vs_r0, "multi_threshold": multi,
                                 "rounds": info, "carried": carried}, indent=1, default=float), encoding="utf-8")
    return 0


def cases_2026() -> list[dict]:
    """v9's 2026 final cases (identity and truth), re-featured for v10 from the files cached by the
    final: the SHIPS texts (every threshold, raw percent) and the a-decks (incl. E features)."""
    from hazardpulse.hurricane import ships_text
    base = [json.loads(l) for l in (v9.Y26 / "final_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    adecks, out = {}, []
    for c in base:
        sid = c["atcf_id"]
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        low = sid[:2].lower() + sid[2:]
        if sid not in adecks:
            adecks[sid] = atcf.parse_atcf_deck(gzip.decompress((v9.Y26 / "adeck" / f"a{low}.dat.gz").read_bytes()).decode("utf-8", "replace"))
        name = cyc.strftime("%y%m%d%H") + sid[:4] + sid[-2:] + "_ships.txt"
        text = (v9.Y26 / "stext" / name).read_text(encoding="utf-8", errors="replace")
        raw, tup = {}, {}
        for th in fx.THRESHOLDS:
            ri = ships_text.parse_ships_text(text, filename=name, threshold=th)
            for tech in fx.RI_TECHS:
                v = ri.whole_percent.get(tech)
                if v is not None:
                    raw[f"{tech}_{th}"] = v
                    tup[(tech, th)] = v
        f = features_for(adecks[sid], cyc, c["basin"], tup)
        out.append({**{k: c[k] for k in ("atcf_id", "dtg", "sid", "season", "basin", "y", "a_dtop", "a_riod")},
                    "dv": float(c["v_t24"]) - float(c["v_t"]), "pcts": raw, "f": f})
    return out


def second_read(_a) -> int:
    """2026, a DECLARED second read for v10 (amendment 2): reported, never a claim."""
    dev = load(DEV10)
    sel = json.loads(SEL10.read_text(encoding="utf-8"))
    test = cases_2026()
    y = np.array([r["y"] for r in test])
    groups = np.array([r["atcf_id"] for r in test])
    has = np.array([math.isfinite(r["f"]["dv24_DSHP"]) for r in test])
    pa = v9.a_forecast(test)
    out = {"phase": "second-read", "declared": "v10 was designed after the 2026 read; no claim is made from these numbers",
           "n": int(len(y)), "events": int(y.sum()), "carried": sel["carried"], "models": {}}
    for c in sorted({sel["carried"], "V4"}):
        p, rounds = fit_predict(c, dev, test, ks_out=MULTI if CANDS[c]["stacked"] else (30,))
        m = {"rounds": rounds, "all": v9.summary(y, p[30], False), "vs_dtops": v9.paired(y, pa, True, p[30], False, groups)}
        if CANDS[c]["stacked"]:
            m["multi_threshold"] = paired_multi(test, p, groups)
        for nm, sub in (("with_guidance", has), ("without_guidance", ~has)):
            m[nm] = {"n": int(sub.sum()), "events": int(y[sub].sum()),
                     "log_loss": v9.summary(y[sub], p[30][sub], False)["log_loss"],
                     "dtops_log_loss": v9.summary(y[sub], pa[sub], True)["log_loss"]}
        out["models"][c] = m
        w = m["vs_dtops"]
        log(f"2026 (second read) {c}: LL {m['all']['log_loss']:.4f} vs DTOPS {v9.summary(y, pa, True)['log_loss']:.4f} "
            f"dLL {w['d_log_loss']:+.4f} [{w['d_log_loss_ci'][0]:+.4f}, {w['d_log_loss_ci'][1]:+.4f}] "
            f"dBrier {w['d_brier']:+.5f} [{w['d_brier_ci'][0]:+.5f}, {w['d_brier_ci'][1]:+.5f}] AUC {m['all']['auc']:.4f}"
            + (f" | 4-thr Brier d {m['multi_threshold']['d']:+.4f} [{m['multi_threshold']['d_ci'][0]:+.4f}, "
               f"{m['multi_threshold']['d_ci'][1]:+.4f}]" if "multi_threshold" in m else "")
            + f" | without guidance LL {m['without_guidance']['log_loss']:.4f} vs DTOPS {m['without_guidance']['dtops_log_loss']:.4f}")
    SECOND.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
