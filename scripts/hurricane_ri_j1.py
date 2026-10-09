"""Hurricane RI amendment 13 (docs/HURRICANE_RI_V9_PROGRAM.md): J1 = v8.2 + GMGSI IR for the JTWC basins.

    PYTHONPATH=src python scripts/hurricane_ri_j1.py rows                      # rows, v8.2 scores, IR tasks
    PYTHONPATH=src python scripts/hurricane_ri_j1.py collect --shard 3 --of 20 --out j1_out   # (runners)
    PYTHONPATH=src python scripts/hurricane_ri_j1.py verify <shard dir>         # every IR task exactly once
    PYTHONPATH=src python scripts/hurricane_ri_j1.py select <shard dir>         # the registered test

Rows: storms of 2022-2024 from the frozen v8.2 file, 2025-2026 from the same builder on the IBTrACS release of
2026-10-08 (2026 is the declared further read). Controls, both stop the run when they fail: v8.2 recomputed must
reproduce its artifact's recorded calibration log loss to 1e-12 on its 8,317 calibration cases, and the builder
must reproduce the frozen file's 2022-2024 rows on >= 99 % of cases.

IR: amendment 5's features (``ir_features``) from GMGSI images at t + 2 h and t - 4 h, each crop +-4 degrees
around the best-track position at t extrapolated along the t - 6 h -> t motion (``ir_source.extrapolate``, the
live path's own function). A collection shard reduces each crop to ``ir_features.static_features`` on the runner;
``features`` is the same arithmetic ``ir_features.features`` does (static now, now minus before for the trends).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.hurricane import ir_features as irf  # noqa: E402
from hazardpulse.hurricane import ir_source as irs  # noqa: E402

TASKS = ROOT / "results" / "j1" / "ir_tasks.jsonl.gz"
ROWS_NAME = "rows.jsonl.gz"
OUT = ROOT / "results" / "calibration" / "hurricane_ri_j1.json"
FRESH_IBTRACS_TAG = "ibtracs_20261008"
SERVED = "hurricane_ri_v8_2"           # the JTWC-basin model scripts/fetch_and_score.py serves (SERVED_MODEL_VERSION)
FIRST, LAST_DEV, FURTHER = 2022, 2025, 2026
FOLDS = (2024, 2025)
JTWC = ("WP", "NI", "SI", "SP")
EQUIV_MIN = 0.99
SEEDS = (0, 1, 2, 3, 4)
NOISE_SEEDS = (1, 2, 3)
N_BOOT = 2000
EARLY_HOURS = 10
MIN_OK_FRACTION = 0.5


def _work() -> Path:
    import hurricane_ri_v9 as v9
    w = v9.WORK / "j1"
    w.mkdir(parents=True, exist_ok=True)
    return w


def _t(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def row_key(r: dict) -> str:
    return f"{r['storm_id']}|{r['issue_time']}"


def task_key(t: dict) -> str:
    return f"{t['row']}|{t['tag']}"


# ---------------------------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------------------------

def _fresh_cases() -> list[dict]:
    import build_hurricane_training_data as b
    b.CACHE_DIR = _work() / FRESH_IBTRACS_TAG
    b.MAX_YEAR = FURTHER
    out = []
    for code in b.BASINS:
        data = b.fetch_cached(b.IBTRACS_URL.format(basin=code))
        if not data:
            raise SystemExit(f"IBTrACS {code} could not be fetched")
        out.extend(b.extract_ri_cases(b.parse_ibtracs(data.decode("utf-8", errors="replace"))))
    return out


def rows_phase() -> int:
    from hazardpulse.hurricane import ri_model as rm
    model = rm.load_model(rm.ARTIFACTS[SERVED])
    if model["model_version"] != "hurricane_ri_v8_2":
        raise SystemExit(f"the served model is {model['model_version']}, not v8.2")
    frozen = rm.load_dataset("v8.2")
    calib = rm.select_years(frozen, tuple(model["calibration"]["fitted_on"]["storm_years"]))
    p = rm.score_cases(model, calib)["calibrated"]
    y = np.array([c["ri_label_30kt"] for c in calib], float)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    want = model["calibration"]["mean_log_loss"]
    if len(calib) != model["calibration"]["n"] or abs(ll - want) > 1e-12:
        raise SystemExit(f"control failed: v8.2 log loss {ll!r} on {len(calib)} cases, artifact {want!r} on "
                         f"{model['calibration']['n']}")
    print(f"control: v8.2 reproduces its calibration log loss {ll!r} on {len(calib)} cases", flush=True)

    fresh = _fresh_cases()
    first_f = rm.storm_first_year(frozen)
    first_n = rm.storm_first_year(fresh)
    fz = {row_key(r): r for r in frozen if FIRST <= first_f[r["storm_id"]] <= 2024}
    fr = {row_key(r): r for r in fresh if FIRST <= first_n[r["storm_id"]] <= 2024}
    same = sum(1 for k, r in fz.items() if fr.get(k) == r)
    frac = same / max(1, len(fz))
    diff_fields: dict[str, int] = {}
    for k, r in fz.items():
        g = fr.get(k)
        if g is None:
            diff_fields["<row absent>"] = diff_fields.get("<row absent>", 0) + 1
        elif g != r:
            for f in r:
                if g.get(f) != r[f]:
                    diff_fields[f] = diff_fields.get(f, 0) + 1
    print(f"equivalence: {same} of {len(fz)} frozen 2022-2024 rows reproduced exactly ({frac:.4f}); "
          f"{len(fr) - len(set(fr) & set(fz))} fresh rows not in the frozen file; differing fields {diff_fields}",
          flush=True)
    if frac < EQUIV_MIN:
        raise SystemExit(f"equivalence control failed: {frac:.4f} < {EQUIV_MIN}")

    rows = [dict(r, j1_season=first_f[r["storm_id"]]) for r in frozen if FIRST <= first_f[r["storm_id"]] <= 2024]
    rows += [dict(r, j1_season=first_n[r["storm_id"]]) for r in fresh if 2025 <= first_n[r["storm_id"]] <= FURTHER]
    sc = rm.score_cases(model, rows)
    for r, e, c in zip(rows, sc["ensemble"], sc["calibrated"]):
        r["v82_ensemble"], r["v82_calibrated"] = float(e), float(c)

    pos = {row_key(r): (r["analysis_lat"], r["analysis_lon"]) for r in list(frozen) + list(fresh)}
    tasks = []
    for r in rows:
        t = _t(r["issue_time"])
        p0 = (float(r["analysis_lat"]), float(r["analysis_lon"]))
        pm6 = pos.get(f"{r['storm_id']}|{(t - dt.timedelta(hours=6)):%Y-%m-%d %H:%M:%S}")
        pm6 = (float(pm6[0]), float(pm6[1])) if pm6 is not None else None
        for tag, h in irs.OFFSETS.items():
            la, lo = irs.extrapolate(p0, pm6, h)
            tasks.append({"row": row_key(r), "tag": tag, "hour": f"{t + dt.timedelta(hours=h):%Y%m%d%H}",
                          "lat": round(la, 4), "lon": round(lo, 4)})
    w = _work()
    with gzip.open(w / ROWS_NAME, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    TASKS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(TASKS, "wb", mtime=0) as gz:
        gz.write("".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks).encode("utf-8"))
    by = {}
    for r in rows:
        k = (r["j1_season"], "JTWC" if r["basin"] in JTWC else "NHC")
        n, e = by.get(k, (0, 0))
        by[k] = (n + 1, e + int(r["ri_label_30kt"]))
    print(f"rows {len(rows)}; by (season, region): {dict(sorted(by.items()))}; tasks {len(tasks)} over "
          f"{len({t['hour'] for t in tasks})} image hours; wrote {w / ROWS_NAME} and {TASKS}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# collect / verify
# ---------------------------------------------------------------------------------------------------------------

def shard_of(hour: str, n: int) -> int:
    t = dt.datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() // 3600) % n


def collect(shard: int, of: int, out: str, max_hours: int = 0) -> int:
    with gzip.open(TASKS, "rt", encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh]
    by_hour: dict[str, list[dict]] = {}
    for t in tasks:
        if shard_of(t["hour"], of) == shard:
            by_hour.setdefault(t["hour"], []).append(t)
    hours = sorted(by_hour)[: max_hours or None]
    keys, stats, status, image = [], [], [], []
    t0, done = time.time(), 0
    for h in hours:
        hour = dt.datetime.strptime(h, "%Y%m%d%H")
        try:
            key, counts, lat, lon = irs.fetch_image(hour)
            opened = "missing: no image" if key is None else ""
        except Exception as exc:  # noqa: BLE001 -- the hour's tasks are recorded missing
            key, counts, opened = None, None, f"missing: {type(exc).__name__}: {str(exc)[:120]}"
        for t in by_hour[h]:
            if counts is None:
                s, st = {n: math.nan for n in irf.STATIC}, opened
            else:
                try:
                    c = irs.crop(counts, lat, lon, (t["lat"], t["lon"]))
                    s, st = irf.static_features(c["counts"], c["lat"], c["lon"], tuple(c["centre"])), "ok"
                except Exception as exc:  # noqa: BLE001 -- this task only
                    s, st = {n: math.nan for n in irf.STATIC}, f"missing: {type(exc).__name__}: {str(exc)[:120]}"
            keys.append(task_key(t)), stats.append([s[n] for n in irf.STATIC]), status.append(st)
            image.append(key or "")
        done += 1
        if done == EARLY_HOURS and not any(s == "ok" for s in status):
            print(f"ABORT: no crop succeeded in the first {EARLY_HOURS} image hours: {status[:1]}", flush=True)
            return 1
        if done % 25 == 0:
            print(f"{done}/{len(hours)} image hours, {len(keys)} tasks, {(time.time() - t0) / 60:.1f} min", flush=True)
    if len({len(keys), len(stats), len(status), len(image)}) != 1:
        raise SystemExit("misaligned records")
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    path = o / f"j1_shard_{shard:02d}_of_{of:02d}.npz"
    np.savez_compressed(path, keys=np.array(keys), stats=np.array(stats, float).reshape(-1, len(irf.STATIC)),
                        static_names=np.array(irf.STATIC), status=np.array(status), image=np.array(image))
    ok = sum(s == "ok" for s in status)
    print(f"shard {shard}/{of}: {len(keys)} tasks, {ok} ok; {(time.time() - t0) / 60:.1f} min; wrote {path}", flush=True)
    if status and ok < MIN_OK_FRACTION * len(status):
        print(f"FAIL: only {ok} of {len(status)} tasks ok", flush=True)
        return 1
    return 0


def load_collection(shard_dir: str) -> tuple[dict[str, list[float]], dict[str, int]]:
    """``{task key: static features}`` for every task, verified: each task exactly once, the static feature names
    the library's, and every key in the task list. Raises SystemExit naming the first violation."""
    with gzip.open(TASKS, "rt", encoding="utf-8") as fh:
        want = {task_key(json.loads(line)) for line in fh}
    got: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for p in sorted(Path(shard_dir).rglob("j1_shard_*.npz")):
        with np.load(p) as z:
            if tuple(str(n) for n in z["static_names"]) != tuple(irf.STATIC):
                raise SystemExit(f"{p}: static feature names differ from the library's")
            for k, s, st in zip(z["keys"], z["stats"], z["status"]):
                k = str(k)
                if k in got:
                    raise SystemExit(f"{p}: task {k} collected twice")
                got[k] = list(s) if str(st) == "ok" else [math.nan] * len(irf.STATIC)
                head = str(st).split(":")[0]
                counts[head] = counts.get(head, 0) + 1
    if set(got) != want:
        raise SystemExit(f"collection holds {len(got)} tasks, the list {len(want)}: {len(want - set(got))} missing, "
                         f"{len(set(got) - want)} not in the list")
    return got, counts


# ---------------------------------------------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------------------------------------------

def ir_row_features(stats: dict[str, list[float]], rk: str) -> dict[str, float]:
    """``ir_features.features`` arithmetic on stored statics: the t + 2 h image's static features, and t + 2 h minus
    t - 4 h for the trend features."""
    now = dict(zip(irf.STATIC, stats.get(f"{rk}|p2", [math.nan] * len(irf.STATIC))))
    bef = dict(zip(irf.STATIC, stats.get(f"{rk}|m4", [math.nan] * len(irf.STATIC))))
    out = dict(now)
    for n in irf.TREND:
        out[f"d_{n}"] = now[n] - bef[n]
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(r[n]) if r.get(n) is not None else np.nan for n in names] for r in rows], float)


def fit_predict(train: list[dict], test: list[dict], names: list[str]) -> np.ndarray:
    """The program's learner: v9.GBT_PARAMS, seeds averaged, early stopping on the last training season, refit on
    all training rows at the found rounds."""
    import lightgbm as lgb
    import hurricane_ri_v9 as v9
    last = max(r["j1_season"] for r in train)
    inner = [r for r in train if r["j1_season"] < last]
    valid = [r for r in train if r["j1_season"] == last]
    ya = np.array([r["ri_label_30kt"] for r in train], float)
    Xa = design(train, names)
    Xi, yi = design(inner, names), np.array([r["ri_label_30kt"] for r in inner], float)
    Xv, yv = design(valid, names), np.array([r["ri_label_30kt"] for r in valid], float)
    Xt = design(test, names)
    preds = []
    for s in SEEDS:
        p = dict(v9.GBT_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
        b = lgb.train(p, lgb.Dataset(Xi, yi), 2000, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(1, int(b.best_iteration or 1))
        preds.append(lgb.train(p, lgb.Dataset(Xa, ya), rounds).predict(Xt))
    return np.mean(preds, axis=0)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())) if 0 < pos.sum() < len(y) else math.nan
    return {"n": int(len(y)), "events": int(y.sum()), "log_loss": ll, "brier": float(np.mean((p - y) ** 2)),
            "auc": auc, "mean_forecast": float(p.mean())}


def paired(y, pa, pb, storms, n_boot=N_BOOT, seed=0) -> dict:
    """b minus a: log loss and Brier, with a 95 % storm-bootstrap interval (storms resampled, cycles kept)."""
    pa, pb = np.clip(pa, 1e-12, 1 - 1e-12), np.clip(pb, 1e-12, 1 - 1e-12)
    dll = -(y * np.log(pb) + (1 - y) * np.log(1 - pb)) + (y * np.log(pa) + (1 - y) * np.log(1 - pa))
    dbr = (pb - y) ** 2 - (pa - y) ** 2
    ids = np.unique(storms, return_inverse=True)[1]
    k = ids.max() + 1
    s_ll, s_br, s_n = np.bincount(ids, dll, k), np.bincount(ids, dbr, k), np.bincount(ids, None, k)
    rng = np.random.default_rng(seed)
    bl, bb = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        n = (w * s_n).sum()
        bl.append((w * s_ll).sum() / n), bb.append((w * s_br).sum() / n)
    return {"d_ll": float(dll.mean()), "d_ll_ci": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
            "d_brier": float(dbr.mean()), "d_brier_ci": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}


def _basin_flags(r: dict) -> dict:
    b = r["basin"]
    return {"is_wp": float(b == "WP"), "is_ni": float(b == "NI"), "is_sh": float(b in ("SI", "SP")),
            "is_nhc": float(b in ("NA", "EP"))}


def select(shard_dir: str) -> int:
    from hazardpulse.hurricane import ri_model as rm
    stats, counts = load_collection(shard_dir)
    with gzip.open(_work() / ROWS_NAME, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    model = rm.load_model(rm.ARTIFACTS[SERVED])
    v82_names = list(model["feature_names"])
    ir_names = list(irf.IR_NAMES)
    for r in rows:
        r.update(ir_row_features(stats, row_key(r)))
        r.update(_basin_flags(r))
        r["v82_logit"] = float(_logit(r["v82_ensemble"]))
    base = ["v82_logit"] + v82_names + ["is_wp", "is_ni", "is_sh", "is_nhc"]
    cands = {"J1": base + ir_names, "B": base}
    ir_cov = float(np.mean([math.isfinite(r["ir_mean_0_50"]) for r in rows]))
    print(f"IR collection {counts}; rows with the t + 2 h image's features: {ir_cov:.3f}", flush=True)

    P = {c: [] for c in ("A", "J1", "B")}
    noise = {s: [] for s in NOISE_SEEDS}
    test_all = []
    for fold in FOLDS:
        train = [r for r in rows if FIRST <= r["j1_season"] < fold]
        test = [r for r in rows if r["j1_season"] == fold and r["basin"] in JTWC]
        P["A"].append(np.array([r["v82_calibrated"] for r in test]))
        for c, names in cands.items():
            P[c].append(fit_predict(train, test, names))
        for s in NOISE_SEEDS:
            rng = np.random.default_rng(s)
            shuf_tr, shuf_te = [dict(r) for r in train], [dict(r) for r in test]
            for grp in (shuf_tr, shuf_te):
                for season in sorted({r["j1_season"] for r in grp}):
                    idx = [i for i, r in enumerate(grp) if r["j1_season"] == season]
                    perm = rng.permutation(idx)
                    vals = [{n: grp[j][n] for n in ir_names} for j in perm]
                    for i, v in zip(idx, vals):
                        grp[i].update(v)
            noise[s].append(fit_predict(shuf_tr, shuf_te, cands["J1"]))
        test_all.extend(test)
        print(f"fold {fold}: train {len(train)} rows, test {len(test)} JTWC rows", flush=True)
    P = {c: np.concatenate(v) for c, v in P.items()}
    y = np.array([r["ri_label_30kt"] for r in test_all], float)
    storms = np.array([r["storm_id"] for r in test_all])
    dev = {c: metrics(y, P[c]) for c in P}
    pr_a, pr_b = paired(y, P["A"], P["J1"], storms), paired(y, P["B"], P["J1"], storms)
    carried = bool(dev["J1"]["log_loss"] < dev["A"]["log_loss"] and dev["J1"]["brier"] < dev["A"]["brier"])
    noise_out = []
    for s in NOISE_SEEDS:
        pn = np.concatenate(noise[s])
        noise_out.append({"seed": s, **paired(y, P["B"], pn, storms)})
    by_basin, by_fold = {}, {}
    region = np.array(["SH" if r["basin"] in ("SI", "SP") else r["basin"] for r in test_all])
    for b in ("WP", "NI", "SH"):
        m = region == b
        if m.any():
            by_basin[b] = {c: metrics(y[m], P[c][m]) for c in P}
            by_basin[b]["J1_minus_A"] = paired(y[m], P["A"][m], P["J1"][m], storms[m])
    season = np.array([r["j1_season"] for r in test_all])
    for f in FOLDS:
        m = season == f
        by_fold[str(f)] = {c: metrics(y[m], P[c][m]) for c in P}
    for c in ("A", "B", "J1"):
        d = dev[c]
        print(f"DEV {c}: LL {d['log_loss']:.5f} Brier {d['brier']:.5f} AUC {d['auc']:.4f} (n {d['n']}, events {d['events']})")
    print(f"J1 - A: dLL {pr_a['d_ll']:+.5f} [{pr_a['d_ll_ci'][0]:+.5f}, {pr_a['d_ll_ci'][1]:+.5f}]  dBrier "
          f"{pr_a['d_brier']:+.5f} [{pr_a['d_brier_ci'][0]:+.5f}, {pr_a['d_brier_ci'][1]:+.5f}]; CARRIED J1: {carried}")
    print(f"J1 - B: dLL {pr_b['d_ll']:+.5f} [{pr_b['d_ll_ci'][0]:+.5f}, {pr_b['d_ll_ci'][1]:+.5f}]")
    for n in noise_out:
        print(f"noise control seed {n['seed']} (shuffled IR - B): dLL {n['d_ll']:+.5f} [{n['d_ll_ci'][0]:+.5f}, {n['d_ll_ci'][1]:+.5f}]")
    for b, d in by_basin.items():
        print(f"  {b}: A {d['A']['log_loss']:.4f} J1 {d['J1']['log_loss']:.4f} dLL {d['J1_minus_A']['d_ll']:+.4f} "
              f"[{d['J1_minus_A']['d_ll_ci'][0]:+.4f}, {d['J1_minus_A']['d_ll_ci'][1]:+.4f}] (events {d['A']['events']})")

    further = None
    t26 = [r for r in rows if r["j1_season"] == FURTHER and r["basin"] in JTWC]
    if t26:
        tr = [r for r in rows if FIRST <= r["j1_season"] <= LAST_DEV]
        p26 = {"A": np.array([r["v82_calibrated"] for r in t26]), "J1": fit_predict(tr, t26, cands["J1"])}
        y26, s26 = np.array([r["ri_label_30kt"] for r in t26], float), np.array([r["storm_id"] for r in t26])
        further = {"declared": "2026 JTWC basins to date: a further read, no claim",
                   "A": metrics(y26, p26["A"]), "J1": metrics(y26, p26["J1"]), "J1_minus_A": paired(y26, p26["A"], p26["J1"], s26)}
        print(f"2026 (further read): A LL {further['A']['log_loss']:.4f} J1 LL {further['J1']['log_loss']:.4f} "
              f"dLL {further['J1_minus_A']['d_ll']:+.4f} [{further['J1_minus_A']['d_ll_ci'][0]:+.4f}, "
              f"{further['J1_minus_A']['d_ll_ci'][1]:+.4f}] (n {len(t26)}, events {int(y26.sum())})")
    rows_sha = hashlib.sha256((_work() / ROWS_NAME).read_bytes()).hexdigest()
    OUT.write_text(json.dumps({
        "phase": "amendment 13 select", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 13)",
        "prereg_tag": "prereg-hurricane-ri-amend13",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows_sha256": rows_sha, "ir_tasks_sha256": hashlib.sha256(TASKS.read_bytes()).hexdigest(),
        "ir_collection_status": counts, "ir_coverage_rows": ir_cov, "candidates": cands,
        "development": dev, "J1_minus_A": pr_a, "J1_minus_B": pr_b, "carried": "J1" if carried else "v8.2",
        "noise_control": noise_out, "by_basin": by_basin, "by_fold": by_fold, "further_read_2026": further,
    }, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("rows", "collect", "verify", "select"))
    ap.add_argument("shard_dir", nargs="?")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int)
    ap.add_argument("--out", default="j1_out")
    ap.add_argument("--max-hours", type=int, default=0)
    a = ap.parse_args(argv)
    if a.phase == "rows":
        return rows_phase()
    if a.phase == "collect":
        return collect(a.shard, a.of, a.out, a.max_hours)
    if a.phase == "verify":
        _, counts = load_collection(a.shard_dir)
        print(f"collection complete: {sum(counts.values())} tasks, each once; status {counts}")
        return 0
    return select(a.shard_dir)


if __name__ == "__main__":
    raise SystemExit(main())
