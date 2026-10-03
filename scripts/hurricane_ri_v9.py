"""Hurricane RI v9 -- the pre-registered programme to beat DTOPS (docs/HURRICANE_RI_V9_PROGRAM.md).

    PYTHONPATH=src python scripts/hurricane_ri_v9.py build-dev     # enrich the 2020-2025 cases
    PYTHONPATH=src python scripts/hurricane_ri_v9.py select        # forward chaining 2022-2025
    PYTHONPATH=src python scripts/hurricane_ri_v9.py final         # 2026, read once

Development reads only 2020-2025 decks and IBTrACS truth; `final` is the only code that reads a
2026 best track, and it refuses to run twice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_hurricane_vs_ships as bench  # noqa: E402
from hazardpulse.hurricane import atcf  # noqa: E402
from hazardpulse.hurricane import ri_v9_features as fx  # noqa: E402

CACHE = Path("C:/Users/Josh/Projects/hazardpulse/.cache")
DECKS = CACHE / "atcf_adecks"
STACK_CASES = (CACHE / "hurricane_ri_stack" / "cases_2020_2024.jsonl", CACHE / "hurricane_ri_stack" / "cases_2025.jsonl")
WORK = CACHE / "hurricane_ri_v9"
DEV_TABLE = WORK / "dev_cases.jsonl"
SELECTION_PATH = ROOT / "results" / "calibration" / "hurricane_ri_v9_selection.json"
FINAL_PATH = ROOT / "results" / "calibration" / "hurricane_ri_v9_final.json"
PROGRAM_COMMIT = "85a6cf9eb"
FOLDS = (2022, 2023, 2024, 2025)
SEED = 20261003
REPS = 2000
CANDIDATES = ("A", "B_lr", "B_gbt", "C_lr", "C_gbt", "D_lr", "D_gbt")
GROUPS = {"B": "O", "C": "ON", "D": "ONH"}
GBT_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=30,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, seed=0,
                  verbose=-1, num_threads=4, deterministic=True, force_row_wise=True)


def log(*a) -> None:
    print(*a, flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# build-dev
# ---------------------------------------------------------------------------

def adeck_path(atcf_id: str) -> Path:
    return DECKS / atcf_id[-4:] / f"a{atcf_id[:2].lower()}{atcf_id[2:4]}{atcf_id[-4:]}.dat.gz"


def build_dev(_args) -> int:
    cases = [json.loads(l) for p in STACK_CASES for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    log(f"{len(cases)} development cases from the previous programme's tables")
    records, _ = bench.load_edecks(DECKS, range(2020, 2026))
    by_th = {}
    for th in fx.THRESHOLDS:
        kt, hr = fx.threshold_key(th)
        by_th[th], stats = bench.index_ri(records, tau=hr, dv_kt=kt, start_tau=0, stop_tau=hr)
        log(f"  e-deck {th}: {stats}")
    adecks: dict[str, list] = {}
    stats = Counter()
    out = []
    for c in cases:
        aid = c["atcf_id"]
        if aid not in adecks:
            p = adeck_path(aid)
            adecks[aid] = atcf.parse_atcf_deck(gzip.decompress(p.read_bytes()).decode("utf-8", "replace")) if p.exists() else []
            stats["adeck_files" if p.exists() else "adeck_missing"] += 1
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        table = fx.cycle_table(adecks[aid], cyc)
        f = fx.adeck_features(table, c["basin"])
        pcts = {}
        for th in fx.THRESHOLDS:
            slot = by_th[th].get((aid, c["dtg"]), {})
            for tech in fx.RI_TECHS:
                if tech in slot:
                    pcts[(tech, th)] = slot[tech].prob_pct
        f.update(fx.ri_features(pcts))
        # the previous builder's 30/24 RIOD must be this builder's
        if pcts.get(("RIOD", "30/24")) != c["aids"].get("RIOD"):
            stats["riod_30_24_mismatch"] += 1
        stats["v0_missing"] += int(not math.isfinite(f["v0"]))
        a_pct = c["aids"].get("DTOP")
        out.append({"atcf_id": aid, "dtg": c["dtg"], "sid": c["sid"], "season": c["season"], "basin": c["basin"],
                    "y": int(c["y"]), "a_dtop": a_pct, "a_riod": c["aids"].get("RIOD"), "f": f})
    WORK.mkdir(parents=True, exist_ok=True)
    DEV_TABLE.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=True) + "\n" for r in out), encoding="utf-8")
    cov = {}
    for season in sorted({r["season"] for r in out}):
        rows = [r for r in out if r["season"] == season]
        cov[season] = {"n": len(rows), "events": sum(r["y"] for r in rows),
                       **{n: round(float(np.mean([math.isfinite(r["f"][n]) for r in rows])), 3)
                          for n in fx.O_NAMES + fx.H_NAMES + ("ri_DTOP_30_24", "ri_RIOD_25_24")}}
    log(json.dumps(dict(stats)))
    for s, c in cov.items():
        log(s, json.dumps(c))
    log(f"wrote {DEV_TABLE} sha256 {sha(DEV_TABLE)}")
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def a_forecast(rows) -> np.ndarray:
    """A: DTOPS 30/24 as issued, SHIPS-RII where DTOPS is missing (whole percent)."""
    return np.array([(r["a_dtop"] if r["a_dtop"] is not None else r["a_riod"]) / 100.0 for r in rows])


def design(rows, names) -> np.ndarray:
    return np.vstack([fx.vector(r["f"], names) for r in rows])


class LR:
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        self.miss_cols = np.flatnonzero(np.isnan(X).any(axis=0))
        Z = self._z(X, fit=True)
        self.m = LogisticRegression(C=1.0, max_iter=10000).fit(Z, y)
        return self

    def _z(self, X, fit=False):
        Xi = np.where(np.isnan(X), self.med, X)
        M = np.isnan(X[:, self.miss_cols]).astype(float)
        Z = np.hstack([Xi, M])
        if fit:
            self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0)
            self.sd = np.where(self.sd > 0, self.sd, 1.0)
        return (Z - self.mu) / self.sd

    def predict(self, X):
        return self.m.predict_proba(self._z(X))[:, 1]


class GBT:
    def fit(self, X, y, seasons):
        import lightgbm as lgb
        last = max(seasons)
        tr, va = seasons < last, seasons == last
        rounds = 1
        if tr.any() and va.any() and y[tr].min() != y[tr].max():
            b = lgb.train(GBT_PARAMS, lgb.Dataset(X[tr], y[tr]), 2000, valid_sets=[lgb.Dataset(X[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds = max(1, int(b.best_iteration or 1))
        self.rounds = rounds
        self.b = lgb.train(GBT_PARAMS, lgb.Dataset(X, y), rounds)
        return self

    def predict(self, X):
        return self.b.predict(X)


def fit_predict(cand: str, train, test):
    if cand == "A":
        return a_forecast(test), {}
    family, klass = cand.split("_")
    names = fx.names_for(GROUPS[family])
    Xtr, Xte = design(train, names), design(test, names)
    y = np.array([r["y"] for r in train])
    if klass == "lr":
        return LR().fit(Xtr, y).predict(Xte), {}
    m = GBT().fit(Xtr, y, np.array([r["season"] for r in train]))
    return m.predict(Xte), {"rounds": m.rounds}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _ll_vec(y, p, quantised):
    q = np.clip(p, 0.005, 0.995) if quantised else np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def auc(y, p) -> float:
    from scipy.stats import rankdata
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary(y, p, quantised) -> dict:
    return {"n": int(len(y)), "events": int(y.sum()), "log_loss": float(_ll_vec(y, p, quantised).mean()),
            "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p), "mean_forecast": float(p.mean()),
            "event_rate": float(y.mean())}


def paired(y, pa, qa, pb, qb, groups, reps=REPS, seed=SEED, level=0.95) -> dict:
    """Storm-bootstrap ``level`` intervals (default 95%) of (b - a) in log loss, Brier and AUC."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lla, llb = _ll_vec(y, pa, qa), _ll_vec(y, pb, qb)
    ba, bb = (pa - y) ** 2, (pb - y) ** 2
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d_ll, d_br, d_auc = [], [], []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        d_ll.append(llb[idx].mean() - lla[idx].mean())
        d_br.append(bb[idx].mean() - ba[idx].mean())
        d_auc.append(auc(y[idx], pb[idx]) - auc(y[idx], pa[idx]))

    def ci(v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))]
    return {"d_log_loss": float(llb.mean() - lla.mean()), "d_log_loss_ci": ci(d_ll),
            "d_brier": float(bb.mean() - ba.mean()), "d_brier_ci": ci(d_br),
            "d_auc": auc(y, pb) - auc(y, pa), "d_auc_ci": ci(d_auc)}


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def load_dev():
    rows = [json.loads(l) for l in DEV_TABLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["f"] = {k: (float("nan") if v is None else float(v)) for k, v in r["f"].items()}
    return rows


def select(_args) -> int:
    rows = load_dev()
    preds = {c: [] for c in CANDIDATES}
    ys, groups, seasons, info = [], [], [], {c: {} for c in CANDIDATES}
    per_fold = {}
    for y_season in FOLDS:
        train = [r for r in rows if r["season"] < y_season]
        test = [r for r in rows if r["season"] == y_season]
        yt = np.array([r["y"] for r in test])
        per_fold[y_season] = {}
        for c in CANDIDATES:
            p, meta = fit_predict(c, train, test)
            preds[c].append(p)
            info[c][y_season] = meta
            per_fold[y_season][c] = summary(yt, p, c == "A")
        ys.append(yt)
        groups.extend(r["sid"] for r in test)
        seasons.extend(r["season"] for r in test)
        log(f"fold {y_season}: n={len(test)} events={int(yt.sum())} | " +
            " ".join(f"{c} LL {per_fold[y_season][c]['log_loss']:.4f} AUC {per_fold[y_season][c]['auc']:.3f}"
                     for c in CANDIDATES))
    y = np.concatenate(ys)
    groups = np.array(groups)
    pooled = {c: summary(y, np.concatenate(preds[c]), c == "A") for c in CANDIDATES}
    pa = np.concatenate(preds["A"])
    vs_a = {c: paired(y, pa, True, np.concatenate(preds[c]), False, groups) for c in CANDIDATES if c != "A"}
    order = [c for c in CANDIDATES if c != "A"]
    chosen = min(order, key=lambda c: (pooled[c]["log_loss"], order.index(c)))
    alone = min(("B_lr", "B_gbt"), key=lambda c: (pooled[c]["log_loss"], ("B_lr", "B_gbt").index(c)))
    for c in CANDIDATES:
        v = vs_a.get(c, {})
        log(f"POOLED {c:6s} LL {pooled[c]['log_loss']:.4f} Brier {pooled[c]['brier']:.5f} AUC {pooled[c]['auc']:.4f}"
            + (f" | vs A dLL {v['d_log_loss']:+.4f} [{v['d_log_loss_ci'][0]:+.4f}, {v['d_log_loss_ci'][1]:+.4f}]"
               f" dBrier {v['d_brier']:+.5f} [{v['d_brier_ci'][0]:+.5f}, {v['d_brier_ci'][1]:+.5f}]"
               f" dAUC {v['d_auc']:+.4f} [{v['d_auc_ci'][0]:+.4f}, {v['d_auc_ci'][1]:+.4f}]" if v else ""))
    log(f"CARRIED to the final: {chosen}; ours alone: {alone}")
    SELECTION_PATH.write_text(json.dumps({
        "phase": "select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md", "program_commit": PROGRAM_COMMIT,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_table_sha256": sha(DEV_TABLE), "folds": list(FOLDS), "per_fold": per_fold, "pooled": pooled,
        "vs_A": vs_a, "fit_info": info, "carried": chosen, "ours_alone": alone,
        "bootstrap": {"unit": "IBTrACS SID", "reps": REPS, "seed": SEED},
    }, indent=1, default=float), encoding="utf-8")
    log(f"wrote {SELECTION_PATH}")
    return 0


# ---------------------------------------------------------------------------
# final (2026, read once)
# ---------------------------------------------------------------------------

NHC = "https://ftp.nhc.noaa.gov/atcf"
Y26 = WORK / "2026"
UA = {"User-Agent": "HazardPulse-research/1.0 (RI v9 final; sequential, cached)"}


def fetch(url: str, path: Path) -> bytes:
    import time
    import urllib.request
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            time.sleep(0.2)
            return data
        except Exception as exc:  # noqa: BLE001 -- retried, then raised
            if attempt == 3:
                raise
            log(f"  retry {url}: {exc}")
            time.sleep(2 + 3 * attempt)


def listing(url: str) -> list[str]:
    import re
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"?/][^"]*)"', html)))


def best_track(text: str) -> dict[dt.datetime, float]:
    out = {}
    for rec in atcf.parse_atcf_deck(text):
        if rec.model == "BEST" and rec.tau_hours == 0 and rec.cycle not in out and rec.vmax_kt:
            out[rec.cycle] = float(rec.vmax_kt)
    return out


def build_2026() -> tuple[list[dict], dict]:
    from hazardpulse.hurricane import ships_text
    stats = Counter()
    names = [n for n in listing(f"{NHC}/stext/") if n.endswith("_ships.txt")]
    cases = []
    adecks, btks = {}, {}
    for name in names:
        try:
            sid, cyc = ships_text.parse_filename(name)
        except Exception:  # noqa: BLE001
            stats["unparsed_filename"] += 1
            continue
        basin, num = sid[:2], int(sid[2:4])
        if basin not in ("AL", "EP", "CP") or not 1 <= num <= 49 or sid[-4:] != "2026" or cyc.hour % 6:
            stats["not_a_numbered_storm_cycle"] += 1
            continue
        stats["stext_files"] += 1
        text = fetch(f"{NHC}/stext/{name}", Y26 / "stext" / name).decode("utf-8", "replace")
        pcts = {}
        try:
            for th in fx.THRESHOLDS:
                ri = ships_text.parse_ships_text(text, filename=name, threshold=th)
                for tech in fx.RI_TECHS:
                    v = ri.whole_percent.get(tech)
                    if v is not None:
                        pcts[(tech, th)] = v
        except Exception as exc:  # noqa: BLE001
            stats[f"stext_unparsed:{type(exc).__name__}"] += 1
            continue
        if ("RIOD", "30/24") not in pcts:
            stats["no_ships_rii_30_24"] += 1
            continue
        low = sid[:2].lower() + sid[2:]
        if sid not in adecks:
            raw = fetch(f"{NHC}/aid_public/a{low}.dat.gz", Y26 / "adeck" / f"a{low}.dat.gz")
            adecks[sid] = atcf.parse_atcf_deck(gzip.decompress(raw).decode("utf-8", "replace"))
            braw = fetch(f"{NHC}/btk/b{low}.dat", Y26 / "btk" / f"b{low}.dat")
            btks[sid] = best_track(braw.decode("utf-8", "replace"))
        v_t, v_24 = btks[sid].get(cyc), btks[sid].get(cyc + dt.timedelta(hours=24))
        if v_t is None or v_24 is None:
            stats["no_truth_at_t_or_t24"] += 1
            continue
        f = fx.adeck_features(fx.cycle_table(adecks[sid], cyc), basin)
        f.update(fx.ri_features(pcts))
        cases.append({"atcf_id": sid, "dtg": cyc.strftime("%Y%m%d%H"), "sid": sid, "season": 2026, "basin": basin,
                      "y": int(v_24 - v_t >= fx.RI_KT), "v_t": v_t, "v_t24": v_24,
                      "a_dtop": pcts.get(("DTOP", "30/24")), "a_riod": pcts[("RIOD", "30/24")],
                      "a_rioc": pcts.get(("RIOC", "30/24")), "f": f})
        stats["cases"] += 1
    return cases, dict(stats)


def final(_args) -> int:
    if FINAL_PATH.exists():
        raise SystemExit(f"{FINAL_PATH} exists: 2026 is read once")
    sel = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    chosen, alone = sel["carried"], sel["ours_alone"]
    dev = load_dev()
    log(f"final: carried {chosen}, ours alone {alone}; refit on {len(dev)} development cases (2020-2025)")
    cases, stats = build_2026()
    log(f"2026 cases: {json.dumps(stats)}")
    table = Y26 / "final_cases.jsonl"
    table.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in cases), encoding="utf-8")
    y = np.array([r["y"] for r in cases])
    groups = np.array([r["atcf_id"] for r in cases])
    p = {"A": a_forecast(cases)}
    meta = {}
    for c in (chosen, alone):
        p[c], meta[c] = fit_predict(c, dev, cases)
    p["SHIPS_RII"] = np.array([r["a_riod"] / 100.0 for r in cases])
    p["RIOC"] = np.array([(r["a_rioc"] if r["a_rioc"] is not None else r["a_riod"]) / 100.0 for r in cases])
    quant = {"A": True, "SHIPS_RII": True, "RIOC": True}
    res = {k: summary(y, v, quant.get(k, False)) for k, v in p.items()}
    vs_a = {k: paired(y, p["A"], True, v, quant.get(k, False), groups) for k, v in p.items() if k != "A"}
    v = vs_a[chosen]
    claim = ((v["d_log_loss_ci"][1] < 0 or v["d_brier_ci"][1] < 0) and v["d_log_loss"] <= 0 and v["d_brier"] <= 0)
    for k in p:
        line = f"2026 {k:9s} LL {res[k]['log_loss']:.4f} Brier {res[k]['brier']:.5f} AUC {res[k]['auc']:.4f}"
        if k in vs_a:
            w = vs_a[k]
            line += (f" | vs DTOPS dLL {w['d_log_loss']:+.4f} [{w['d_log_loss_ci'][0]:+.4f}, {w['d_log_loss_ci'][1]:+.4f}]"
                     f" dBrier {w['d_brier']:+.5f} [{w['d_brier_ci'][0]:+.5f}, {w['d_brier_ci'][1]:+.5f}]"
                     f" dAUC {w['d_auc']:+.4f} [{w['d_auc_ci'][0]:+.4f}, {w['d_auc_ci'][1]:+.4f}]")
        log(line)
    log(f"CLAIM 'better than DTOPS' ({chosen}): {claim}")
    np.savez(Y26 / "final_predictions.npz", y=y, groups=groups, **{k: v for k, v in p.items()})
    FINAL_PATH.write_text(json.dumps({
        "phase": "final", "program": "docs/HURRICANE_RI_V9_PROGRAM.md", "program_commit": PROGRAM_COMMIT,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_sha256": sha(SELECTION_PATH), "dev_table_sha256": sha(DEV_TABLE),
        "final_table_sha256": sha(table), "carried": chosen, "ours_alone": alone, "fit_info": meta,
        "cases": stats, "season": 2026, "truth": "2026 b-deck BEST tau 0 (operational best track) at t and t+24 h",
        "results": res, "vs_DTOPS": vs_a, "claim_better_than_DTOPS": bool(claim),
        "claim_rule": "paired storm-bootstrap 95% interval of dLL or dBrier wholly below 0 AND both point differences <= 0",
        "bootstrap": {"unit": "ATCF storm id", "reps": REPS, "seed": SEED},
    }, indent=1, default=float), encoding="utf-8")
    log(f"wrote {FINAL_PATH}")
    return 0


MODEL_PATH = ROOT / "results" / "models" / "hurricane_ri_v9.json"


def export(_args) -> int:
    """Amendment 1: freeze the D_gbt the final scored, as a NumPy payload. Refused unless the refit
    reproduces the final's own 2026 predictions and the payload reproduces the booster."""
    from hazardpulse.tornado import lgbm_payload as lp
    fin = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
    if fin["carried"] != "D_gbt":
        raise SystemExit(f"the final carried {fin['carried']}, not D_gbt")
    dev = load_dev()
    names = list(fx.names_for(GROUPS["D"]))
    X, y = design(dev, names), np.array([r["y"] for r in dev])
    m = GBT().fit(X, y, np.array([r["season"] for r in dev]))
    if m.rounds != fin["fit_info"]["D_gbt"]["rounds"]:
        raise SystemExit(f"refit chose {m.rounds} trees, the final {fin['fit_info']['D_gbt']['rounds']}")
    cases = [json.loads(l) for l in (Y26 / "final_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in cases:
        r["f"] = {k: (float("nan") if v is None else float(v)) for k, v in r["f"].items()}
    X26 = design(cases, names)
    saved = np.load(Y26 / "final_predictions.npz", allow_pickle=True)["D_gbt"]
    d_refit = float(np.max(np.abs(m.predict(X26) - saved)))
    if d_refit > 1e-12:
        raise SystemExit(f"the refit is not the model the final scored (max diff {d_refit:.2e})")
    payload = lp.export_booster(m.b, names, calibration={"method": "identity", "a": 1.0, "b": 0.0}, provenance={
        "program": "docs/HURRICANE_RI_V9_PROGRAM.md", "candidate": "D_gbt", "trained": "NHC cycles 2020-2025",
        "event": "V(t+24 h) - V(t) >= 30 kt", "rounds": m.rounds, "gate_aids": ["DSHP", "IVCN", "NNIC"],
        "final_2026": {k: fin["results"]["D_gbt"][k] for k in ("n", "events", "log_loss", "brier", "auc")},
        "final_2026_dtops": {k: fin["results"]["A"][k] for k in ("log_loss", "brier", "auc")},
        "claim_better_than_dtops": fin["claim_better_than_DTOPS"]})
    d_payload = float(max(np.max(np.abs(lp.predict_proba(payload, Xs) - m.predict(Xs))) for Xs in (X, X26)))
    if d_payload > 1e-9:
        raise SystemExit(f"payload disagrees with the booster by {d_payload:.2e}")
    version = lp.save(payload, MODEL_PATH)
    version = lp.model_version(payload, prefix="hurricane_ri_v9")
    log(f"{MODEL_PATH.name}: {version}, {payload['n_trees']} trees, {len(names)} inputs; refit = final to "
        f"{d_refit:.1e}, payload = booster to {d_payload:.1e}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build-dev", "select", "final", "export"))
    args = ap.parse_args(argv)
    if args.phase == "build-dev":
        return build_dev(args)
    if args.phase == "select":
        return select(args)
    if args.phase == "export":
        return export(args)
    return final(args)


if __name__ == "__main__":
    raise SystemExit(main())
