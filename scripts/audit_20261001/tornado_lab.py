"""Tornado model laboratory over the v3 feature store.

    python tornado_lab.py assemble            # day files -> memory-mapped split arrays
    python tornado_lab.py run <exp.json>      # one experiment -> results/lab/<name>.json

Splits (fixed before any experiment; 2025 is the FINAL test and is only ever
read by `final`, once):
    train   2020-10-15 .. 2022-12-31
    val     2023-01-01 .. 2023-12-31   (every model choice: hyper-parameters,
                                         features, calibration, stacking)
    dev     2024-01-01 .. 2024-12-31   (development test)
    final   2025-01-01 .. 2025-12-31   (touched once, at the end)

Every population is EVERY storm observation of every day with ProbSevere data
(tornado days and quiet days); training alone subsamples negatives, with
weights that restore the population. All intervals resample whole UTC days.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import storm_features as sf

ROOT = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", str(ROOT / ".cache" / "feature_store_v3")))
LAB = Path(os.environ.get("HAZARDPULSE_LAB_DIR", str(STORE / "_lab")))
OUT = ROOT / "results" / "lab"
SPLITS = {
    "train": ("20201015", "20221231"),
    "val": ("20230101", "20231231"),
    "dev": ("20240101", "20241231"),
    "final": ("20250101", "20251231"),
}
NAMES = list(sf.FEATURE_NAMES)
FIDX = {n: i for i, n in enumerate(NAMES)}
LIDX = {n: i for i, n in enumerate(sf.LABEL_NAMES)}


# ---------------------------------------------------------------------------
# assemble: day files -> memmaps
# ---------------------------------------------------------------------------

def assemble(split: str) -> None:
    lo, hi = SPLITS[split]
    days = sorted(p for p in STORE.glob("*.npz") if lo <= p.stem <= hi)
    n = 0
    for p in days:
        with np.load(p) as z:
            n += z["Y"].shape[0]
    LAB.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(LAB / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(n, len(NAMES)))
    Y = np.lib.format.open_memmap(LAB / f"{split}_Y.npy", mode="w+", dtype=np.int8, shape=(n, len(sf.LABEL_NAMES)))
    meta = {k: [] for k in ("day", "t", "lat", "lon", "ef", "lead_min", "analysis")}
    i = 0
    for p in days:
        with np.load(p) as z:
            m = z["Y"].shape[0]
            X[i:i + m] = z["X"]
            Y[i:i + m] = z["Y"]
            meta["day"].append(np.full(m, int(p.stem), np.int64))
            for k in ("t", "lat", "lon", "ef", "lead_min", "analysis"):
                meta[k].append(z[k])
            i += m
    X.flush()
    Y.flush()
    np.savez(LAB / f"{split}_meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    print(f"{split}: {len(days)} days, {n} rows, positives storm_60={int(Y[:, LIDX['storm_60']].sum())}", flush=True)


def load(split: str):
    X = np.load(LAB / f"{split}_X.npy", mmap_mode="r")
    Y = np.load(LAB / f"{split}_Y.npy", mmap_mode="r")
    meta = dict(np.load(LAB / f"{split}_meta.npz"))
    return X, Y, meta


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(y, p, groups, n_boot=1000) -> dict:
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    ci = dm.cluster_bootstrap_auc_ci(y, p, groups, n_boot=n_boot)
    return {
        "auc": dm.compute_auc(y, p), "auc_ci": [ci["ci_lo"], ci["ci_hi"]],
        "pr_auc": dm.compute_pr_auc(y, p), "brier": dm.compute_brier(y, p), "bss": dm.compute_bss(y, p),
        "base_rate": float(y.mean()), "mean_forecast": float(p.mean()), "n": int(len(y)), "pos": int(y.sum()),
        "reliability": dm.compute_calibration_curve(y, p, n_bins=10),
    }


def cols_for(blocks: list[str], drop: list[str] | None = None) -> list[int]:
    idx: list[int] = []
    for b in blocks:
        if b in sf.BLOCKS:
            lo, hi = sf.BLOCKS[b]
            idx += list(range(lo, hi))
        elif b.endswith("*"):
            idx += [i for i, n in enumerate(NAMES) if n.startswith(b[:-1])]
        else:
            idx.append(FIDX[b])
    drop = set(drop or [])
    out = []
    for i in idx:
        if i not in out and not any(NAMES[i].startswith(d[:-1]) if d.endswith("*") else NAMES[i] == d for d in drop):
            out.append(i)
    return out


def train_sample(X, y, neg_per_pos: int, seed: int):
    """All positives + neg_per_pos negatives per positive, weighted back to the population."""
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = min(len(neg), neg_per_pos * len(pos))
    keep_neg = np.sort(rng.choice(neg, size=k, replace=False))
    rows = np.sort(np.concatenate([pos, keep_neg]))
    w = np.where(y[rows] == 1, 1.0, len(neg) / k)
    return rows, w


def predict_stream(model_fn, X, cols, chunk=200_000):
    out = np.empty(X.shape[0], np.float64)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = model_fn(np.asarray(X[s:s + chunk][:, cols], dtype=np.float32))
    return out


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def fit_model(kind: str, params: dict, Xtr, ytr, wtr, Xes, yes, wes, seed: int):
    """Returns score_fn(X) -> raw score (monotone in the model's belief)."""
    if kind == "lgbm":
        import lightgbm as lgb
        p = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=50,
                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, seed=seed, num_threads=4, deterministic=True, force_row_wise=True)
        p.update(params)
        n_rounds = p.pop("n_rounds", 4000)
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=True)
        dva = lgb.Dataset(Xes, yes, weight=wes, reference=dtr)
        bst = lgb.train(p, dtr, n_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        return (lambda X: bst.predict(X, raw_score=True)), {"best_iteration": bst.best_iteration}, bst
    if kind == "xgb":
        import xgboost as xgb
        p = dict(objective="binary:logistic", eta=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.7,
                 min_child_weight=5, reg_lambda=1.0, tree_method="hist", nthread=4, seed=seed,
                 eval_metric="logloss")
        p.update(params)
        n_rounds = p.pop("n_rounds", 4000)
        dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dva = xgb.DMatrix(Xes, label=yes, weight=wes)
        bst = xgb.train(p, dtr, n_rounds, evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
        return (lambda X: bst.predict(xgb.DMatrix(X), output_margin=True,
                                      iteration_range=(0, bst.best_iteration + 1))), \
            {"best_iteration": int(bst.best_iteration)}, bst
    if kind == "cat":
        from catboost import CatBoostClassifier
        p = dict(iterations=4000, learning_rate=0.05, depth=7, l2_leaf_reg=3.0, random_seed=seed,
                 thread_count=4, verbose=False, od_type="Iter", od_wait=200, task_type="CPU")
        p.update(params)
        m = CatBoostClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xes, yes), use_best_model=True)
        return (lambda X: m.predict(X, prediction_type="RawFormulaVal")), \
            {"best_iteration": int(m.get_best_iteration())}, m
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0) + 1e-9

        def prep(X):
            Z = (np.asarray(X, np.float64) - mu) / sd
            return np.nan_to_num(np.clip(Z, -8, 8))
        m = LogisticRegression(C=params.get("C", 1.0), max_iter=2000)
        m.fit(prep(Xtr), ytr, sample_weight=wtr)
        return (lambda X: m.decision_function(prep(X))), {}, m
    if kind == "column":  # a single published score used as-is (baselines)
        j = params["col"]
        return (lambda X: np.nan_to_num(np.asarray(X[:, j], np.float64), nan=-1.0)), {}, None
    raise ValueError(kind)


def fit_calibrator(kind: str, s_val, y_val):
    s_val = np.asarray(s_val, np.float64)
    if kind == "none":
        return lambda s: 1.0 / (1.0 + np.exp(-np.clip(s, -60, 60)))
    if kind == "platt":
        cal = dm.fit_platt(s_val, y_val)
        return lambda s: dm.apply_calibration(s, cal)
    if kind == "venn_abers":
        from hazardpulse.trust.venn_abers import VennAbersCalibrator
        va = VennAbersCalibrator(min_calibration=200, max_groups=512).fit(s_val, y_val)
        return lambda s: va.predict(np.asarray(s, np.float64))[0]
    raise ValueError(kind)


CAL_FOLDS = 5


def out_of_fold_calibrated(kind: str, s_val, y_val, days) -> np.ndarray:
    """Validation probabilities whose calibrator never saw their own day: whole UTC days are dealt
    to CAL_FOLDS folds by a fixed hash, each fold calibrated on the other four. Validation metrics
    use these, so a flexible calibrator (Venn-Abers) cannot flatter itself on the rows it fitted."""
    s_val = np.asarray(s_val, np.float64)
    fold = (np.asarray(days, np.int64) * 2654435761 % 2 ** 32) % CAL_FOLDS
    out = np.empty_like(s_val)
    for k in range(CAL_FOLDS):
        te = fold == k
        out[te] = fit_calibrator(kind, s_val[~te], y_val[~te])(s_val[te])
    return out


def run(exp: dict) -> dict:
    t0 = time.time()
    lab = LIDX[exp.get("label", "storm_60")]                  # the label the model is TRAINED on
    ev = LIDX[exp.get("eval_label", exp.get("label", "storm_60"))]   # the label it is calibrated and SCORED on
    cols = cols_for(exp["blocks"], exp.get("drop"))
    seed = int(exp.get("seed", 0))
    Xt, Yt, mt = load("train")
    y_tr_all = np.asarray(Yt[:, lab], np.int8)
    frac = float(exp.get("train_day_fraction", 1.0))       # learning curve: a fixed random subset of DAYS
    if frac < 1.0:
        ud = np.unique(mt["day"])
        keep_days = np.random.RandomState(seed + 7).choice(ud, size=max(1, int(round(frac * len(ud)))), replace=False)
        y_tr_all = np.where(np.isin(mt["day"], keep_days), y_tr_all, -1).astype(np.int8)
    rows, w = train_sample(None, y_tr_all, int(exp.get("neg_per_pos", 30)), seed)
    Xtr = np.asarray(Xt[rows][:, cols], np.float32)
    ytr = y_tr_all[rows]
    Xv, Yv, mv = load("val")
    yv_train = np.asarray(Yv[:, lab], np.int8)
    yv = np.asarray(Yv[:, ev], np.int8)
    # early-stopping set: a fixed weighted sample of VAL (model selection data), on the training label
    es_rows, es_w = train_sample(None, yv_train, 30, seed + 1)
    Xes = np.asarray(Xv[es_rows][:, cols], np.float32)
    score_fn, info, model = fit_model(exp["model"], dict(exp.get("params", {})), Xtr, ytr, w,
                                      Xes, yv_train[es_rows], es_w, seed)
    s_val = predict_stream(score_fn, Xv, cols)
    cal_kind = exp.get("calibration", "platt")
    cal = fit_calibrator(cal_kind, s_val, yv)                # all of val: what dev/final see
    p_val = out_of_fold_calibrated(cal_kind, s_val, yv, mv["day"])
    res = {"exp": exp, "n_features": len(cols), "train_rows": int(len(rows)),
           "train_pos": int(ytr.sum()), "fit": info,
           "val": metrics(yv, p_val, mv["day"], n_boot=int(exp.get("n_boot_val", 200)))}
    res["val"]["auc_raw_score"] = dm.compute_auc(yv.astype(np.float64), s_val)
    if exp.get("save_preds"):
        LAB.joinpath("preds").mkdir(exist_ok=True)
        np.save(LAB / "preds" / f"{exp['name']}_val.npy", p_val.astype(np.float32))
    if exp.get("eval_dev", True):
        Xd, Yd, md = load("dev")
        yd = np.asarray(Yd[:, ev], np.int8)
        p_dev = cal(predict_stream(score_fn, Xd, cols))
        res["dev"] = metrics(yd, p_dev, md["day"], n_boot=int(exp.get("n_boot_dev", 1000)))
        if exp.get("save_preds"):
            np.save(LAB / "preds" / f"{exp['name']}_dev.npy", p_dev.astype(np.float32))
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{exp['name']}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def compare(name_a: str, name_b: str, split: str, label: str = "storm_60", n_boot: int = 1000, seed: int = 42) -> dict:
    """B minus A on one split, paired, whole UTC days resampled: AUC and Brier deltas with 95% intervals.
    Both experiments must have been run with save_preds on the same split rows."""
    pa = np.load(LAB / "preds" / f"{name_a}_{split}.npy").astype(np.float64)
    pb = np.load(LAB / "preds" / f"{name_b}_{split}.npy").astype(np.float64)
    _, Y, meta = load(split)
    y = np.asarray(Y[:, LIDX[label]], np.float64)
    if not (len(pa) == len(pb) == len(y)):
        raise ValueError(f"row counts differ: {len(pa)} {len(pb)} {len(y)}")
    clusters = dm._cluster_index(meta["day"])
    rng = np.random.RandomState(seed)
    d_auc = np.empty(n_boot)
    d_bri = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([clusters[c] for c in rng.randint(0, len(clusters), size=len(clusters))])
        yy = y[idx]
        d_auc[i] = dm.compute_auc(yy, pb[idx]) - dm.compute_auc(yy, pa[idx])
        d_bri[i] = np.mean((pb[idx] - yy) ** 2) - np.mean((pa[idx] - yy) ** 2)
    out = {"a": name_a, "b": name_b, "split": split, "label": label, "n_days": len(clusters), "n_boot": n_boot,
           "delta_auc": dm.compute_auc(y, pb) - dm.compute_auc(y, pa),
           "delta_auc_ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
           "delta_brier": float(np.mean((pb - y) ** 2) - np.mean((pa - y) ** 2)),
           "delta_brier_ci": [float(np.percentile(d_bri, 2.5)), float(np.percentile(d_bri, 97.5))]}
    (OUT / f"compare_{name_b}_vs_{name_a}_{split}_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "compare":
        a, b, split = sys.argv[2], sys.argv[3], sys.argv[4]
        label = sys.argv[5] if len(sys.argv) > 5 else "storm_60"
        print(json.dumps(compare(a, b, split, label), indent=1))
        return 0
    if cmd == "assemble":
        for split in (sys.argv[2:] or ["train", "val", "dev"]):
            assemble(split)
        return 0
    if cmd == "run":
        exp = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")) if sys.argv[2].endswith(".json") \
            else json.loads(sys.argv[2])
        exps = exp if isinstance(exp, list) else [exp]
        for e in exps:
            r = run(e)
            v, d = r["val"], r.get("dev", {})
            print(f"{e['name']:40s} nfeat={r['n_features']:3d} val AUC {v['auc']:.4f} BSS {v['bss']:+.4f} | "
                  f"dev AUC {d.get('auc', float('nan')):.4f} {d.get('auc_ci')} BSS {d.get('bss', float('nan')):+.4f} "
                  f"({r['seconds']}s)", flush=True)
        return 0
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
