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
W_NAMES = ["w_tor_warning_active", "w_minutes_since_issue"]   # amendment 3
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
    W = np.lib.format.open_memmap(LAB / f"{split}_W.npy", mode="w+", dtype=np.float32, shape=(n, len(W_NAMES)))
    Y = np.lib.format.open_memmap(LAB / f"{split}_Y.npy", mode="w+", dtype=np.int8, shape=(n, len(sf.LABEL_NAMES)))
    meta = {k: [] for k in ("day", "t", "lat", "lon", "ef", "lead_min", "analysis", "v2")}
    i = 0
    for p in days:
        with np.load(p) as z:
            m = z["Y"].shape[0]
            X[i:i + m] = z["X"]
            Y[i:i + m] = z["Y"]
            meta["day"].append(np.full(m, int(p.stem), np.int64))
            for k in ("t", "lat", "lon", "ef", "lead_min", "analysis"):
                meta[k].append(z[k])
            # the served v2 model's probability (score_v2_on_store.py), NaN where absent
            v2 = STORE / "_v2" / f"{p.stem}.npy"
            meta["v2"].append(np.load(v2) if v2.exists() else np.full(m, np.nan, np.float32))
            # block W: NWS tornado-warning state (warning_state_on_store.py), NaN where absent
            nws = STORE / "_nws" / f"{p.stem}.npz"
            if nws.exists():
                with np.load(nws) as w:
                    W[i:i + m] = np.column_stack([w["active_now"].astype(np.float32), w["minutes_since_issue"]])
            else:
                W[i:i + m] = np.nan
            i += m
    W.flush()
    X.flush()
    Y.flush()
    np.savez(LAB / f"{split}_meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    print(f"{split}: {len(days)} days, {n} rows, positives storm_60={int(Y[:, LIDX['storm_60']].sum())}", flush=True)


def load_w(split: str):
    return np.load(LAB / f"{split}_W.npy", mmap_mode="r")


def matrix(X, W, rows, cols, use_w: bool) -> np.ndarray:
    """Model input for ``rows`` (index array or slice): the chosen store columns, then block W."""
    a = np.asarray(X[rows][:, cols], np.float32)
    return np.hstack([a, np.asarray(W[rows], np.float32)]) if use_w else a


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
        if b == "W":          # block W lives outside the store matrix (see matrix())
            continue
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


def balanced_weights(y_rows: np.ndarray) -> np.ndarray:
    """Class-balanced training weights (amendment 5): negatives 1, positives n_neg/n_pos, so both
    classes carry equal total weight and a positive's Hessian is not ~0.001 of a negative's.
    Population weights made every model early-stop at 22-45 trees; the calibrator restores
    the population afterwards."""
    y_rows = np.asarray(y_rows)
    n_pos, n_neg = int((y_rows == 1).sum()), int((y_rows == 0).sum())
    return np.where(y_rows == 1, n_neg / max(n_pos, 1), 1.0)


def predict_stream(model_fn, X, cols, chunk=200_000, W=None):
    out = np.empty(X.shape[0], np.float64)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = model_fn(matrix(X, W, slice(s, s + chunk), cols, W is not None))
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
                 metric="auc", first_metric_only=True,
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
                 eval_metric="auc")
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
                 thread_count=4, verbose=False, od_type="Iter", od_wait=200, task_type="CPU",
                 eval_metric="AUC")
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


EXPERIMENTS = Path(__file__).resolve().parent / "experiments"


def resolve_blocks(exp: dict) -> dict:
    """``blocks_from: "best"`` -> the block set chosen on validation, as committed in
    experiments/best_blocks.json (written once step 1 of the protocol has decided it)."""
    if exp.get("blocks_from") != "best":
        return exp
    best = json.loads((EXPERIMENTS / "best_blocks.json").read_text(encoding="utf-8"))
    return {**exp, "blocks": best["blocks"], "drop": best.get("drop", [])}


def run(exp: dict) -> dict:
    t0 = time.time()
    exp = resolve_blocks(exp)
    lab = LIDX[exp.get("label", "storm_60")]                  # the label the model is TRAINED on
    ev = LIDX[exp.get("eval_label", exp.get("label", "storm_60"))]   # the label it is calibrated and SCORED on
    cols = cols_for(exp["blocks"], exp.get("drop"))
    use_w = "W" in exp["blocks"]
    seed = int(exp.get("seed", 0))
    Xt, Yt, mt = load("train")
    y_tr_all = np.array(Yt[:, lab], np.int8)              # a copy: the memmap is read-only
    frac = float(exp.get("train_day_fraction", 1.0))       # learning curve: a fixed random subset of DAYS
    if frac < 1.0:
        ud = np.unique(mt["day"])
        keep_days = np.random.RandomState(seed + 7).choice(ud, size=max(1, int(round(frac * len(ud)))), replace=False)
        y_tr_all = np.where(np.isin(mt["day"], keep_days), y_tr_all, -1).astype(np.int8)
    if exp.get("shuffle_train_labels"):
        # the null: training labels permuted WITHIN each day (same daily base rates, no storm
        # information); a sound pipeline must then score AUC ~0.5 on the true held-out labels
        rng = np.random.RandomState(seed + 99)
        for idx in dm._cluster_index(mt["day"]):
            y_tr_all[idx] = y_tr_all[idx][rng.permutation(len(idx))]
    rows, _pop_w = train_sample(None, y_tr_all, int(exp.get("neg_per_pos", 30)), seed)
    Xtr = matrix(Xt, load_w("train"), rows, cols, use_w)
    ytr = y_tr_all[rows]
    w = balanced_weights(ytr)                                  # amendment 5
    Xv, Yv, mv = load("val")
    yv_train = np.asarray(Yv[:, lab], np.int8)
    yv = np.asarray(Yv[:, ev], np.int8)
    # early-stopping set: a fixed weighted sample of VAL (model selection data), on the training label
    es_rows, _ = train_sample(None, yv_train, 30, seed + 1)
    es_w = balanced_weights(yv_train[es_rows])                 # AUC is unchanged by class-constant weights
    Wv = load_w("val") if use_w else None
    Xes = matrix(Xv, Wv, es_rows, cols, use_w)
    score_fn, info, model = fit_model(exp["model"], dict(exp.get("params", {})), Xtr, ytr, w,
                                      Xes, yv_train[es_rows], es_w, seed)
    s_val = predict_stream(score_fn, Xv, cols, W=Wv)
    cal_kind = exp.get("calibration", "platt")
    cal = fit_calibrator(cal_kind, s_val, yv)                # all of val: what dev/final see
    p_val = out_of_fold_calibrated(cal_kind, s_val, yv, mv["day"])
    res = {"exp": exp, "n_features": len(cols) + (len(W_NAMES) if use_w else 0), "train_rows": int(len(rows)),
           "train_pos": int(ytr.sum()), "fit": info,
           "val": metrics(yv, p_val, mv["day"], n_boot=int(exp.get("n_boot_val", 200)))}
    res["val"]["auc_raw_score"] = dm.compute_auc(yv.astype(np.float64), s_val)
    if exp.get("save_preds"):
        LAB.joinpath("preds").mkdir(exist_ok=True)
        np.save(LAB / "preds" / f"{exp['name']}_val.npy", p_val.astype(np.float32))
    if exp.get("eval_dev", True):
        Xd, Yd, md = load("dev")
        yd = np.asarray(Yd[:, ev], np.int8)
        p_dev = cal(predict_stream(score_fn, Xd, cols, W=load_w("dev") if use_w else None))
        res["dev"] = metrics(yd, p_dev, md["day"], n_boot=int(exp.get("n_boot_dev", 1000)))
        if exp.get("save_preds"):
            np.save(LAB / "preds" / f"{exp['name']}_dev.npy", p_dev.astype(np.float32))
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{exp['name']}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


BASELINE_COLUMNS = {
    "probtor": ("p_ps_tor", 0.01),      # NOAA ProbTor, percent -> probability
    "probsevere": ("p_ps", 0.01),       # ProbSevere any-severe
    "stp80": ("h80_hrrr_stp", None),    # Significant Tornado Parameter, 80 km analysis
    "stp9": ("h9_stp_at", None),        # ... at the storm, 9 km
}


def baselines(splits=("val", "dev"), label: str = "storm_60") -> dict:
    """The published scores as prediction files, raw (a probability where the product states one)
    and Platt-calibrated on validation (fitted on val, out-of-fold on val itself), plus v2."""
    out: dict = {}
    val_X, val_Y, val_m = load("val")
    y_val = np.asarray(val_Y[:, LIDX[label]], np.int8)
    (LAB / "preds").mkdir(parents=True, exist_ok=True)
    cols = {name: (FIDX[c], scale) for name, (c, scale) in BASELINE_COLUMNS.items()}
    sources = {name: (lambda X, j=j: np.asarray(X[:, j], np.float64)) for name, (j, _) in cols.items()}
    for split in splits:
        X, Y, m = load(split)
        y = np.asarray(Y[:, LIDX[label]], np.int8)
        res = {}
        for name, (j, scale) in cols.items():
            raw = np.nan_to_num(np.asarray(X[:, j], np.float64), nan=-1.0)
            s_val = np.nan_to_num(sources[name](val_X), nan=-1.0)
            if scale is not None:
                p_raw = np.clip(raw * scale, 0.0, 1.0)
                np.save(LAB / "preds" / f"{name}_raw_{split}.npy", p_raw.astype(np.float32))
                res[f"{name}_raw"] = metrics(y, p_raw, m["day"], n_boot=200)
            # Platt on validation: a monotone recalibration, the fairest probability a raw index gets
            p_cal = (out_of_fold_calibrated("platt", s_val, y_val, val_m["day"]) if split == "val"
                     else fit_calibrator("platt", s_val, y_val)(raw))
            np.save(LAB / "preds" / f"{name}_platt_{split}.npy", p_cal.astype(np.float32))
            res[f"{name}_platt"] = metrics(y, p_cal, m["day"], n_boot=200)
        v2 = np.asarray(m["v2"], np.float64)
        np.save(LAB / "preds" / f"v2_{split}.npy", v2.astype(np.float32))
        ok = np.isfinite(v2)
        res["v2_served"] = {**metrics(y[ok], v2[ok], m["day"][ok], n_boot=200), "rows_unscored": int((~ok).sum())}
        out[split] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baselines_{label}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def stress_groups(X, meta) -> dict[str, np.ndarray]:
    """The protocol's stress strata, defined here before any result (boolean masks over rows).
    Positive-only strata (lead, EF2+) keep EVERY negative: the question is whether those events
    rank above the whole population."""
    lat, lon = np.asarray(meta["lat"], np.float64), np.asarray(meta["lon"], np.float64)
    month = (np.asarray(meta["day"], np.int64) // 100) % 100
    solar_h = ((np.asarray(meta["t"], np.int64) % 86400) / 3600.0 + lon / 15.0) % 24.0
    size = np.asarray(X[:, FIDX["p_size"]], np.float64)
    q1, q2 = np.nanquantile(size, [1 / 3, 2 / 3])
    an = np.asarray(meta["analysis"], np.int64)
    g = {
        "region_plains": (lon >= -105) & (lon < -94) & (lat >= 30),
        "region_midwest": (lon >= -94) & (lon < -80) & (lat >= 37),
        "region_southeast": (lon >= -94) & (lon < -75) & (lat < 37),
        "season_DJF": np.isin(month, (12, 1, 2)), "season_MAM": np.isin(month, (3, 4, 5)),
        "season_JJA": np.isin(month, (6, 7, 8)), "season_SON": np.isin(month, (9, 10, 11)),
        "local_night": (solar_h >= 20) | (solar_h < 6), "local_day": (solar_h >= 6) & (solar_h < 20),
        "size_small": size <= q1, "size_mid": (size > q1) & (size <= q2), "size_large": size > q2,
        "analysis_9km": (an >= 0) & (an < 100), "analysis_80km_only": an >= 100, "analysis_none": an < 0,
    }
    g["region_elsewhere"] = ~(g["region_plains"] | g["region_midwest"] | g["region_southeast"])
    return g


def stress(name: str, split: str, label: str = "storm_60", others=("v2", "probtor_platt"), n_boot: int = 300) -> dict:
    """AUC (day-clustered CI) of one experiment's saved predictions per stratum, beside the bars."""
    X, Y, meta = load(split)
    y = np.asarray(Y[:, LIDX[label]], np.int8)
    preds = {name: np.load(LAB / "preds" / f"{name}_{split}.npy").astype(np.float64)}
    for o in others:
        f = LAB / "preds" / f"{o}_{split}.npy"
        if f.exists():
            preds[o] = np.load(f).astype(np.float64)
    lead, ef = np.asarray(meta["lead_min"], np.float64), np.asarray(meta["ef"], np.float64)
    neg = y == 0
    strata = {k: v for k, v in stress_groups(X, meta).items()}
    strata.update({"lead_0_15": neg | ((y == 1) & (lead <= 15)), "lead_15_30": neg | ((y == 1) & (lead > 15) & (lead <= 30)),
                   "lead_30_60": neg | ((y == 1) & (lead > 30)), "ef2plus": neg | ((y == 1) & (ef >= 2)),
                   "ef0_1": neg | ((y == 1) & (ef >= 0) & (ef < 2))})
    out = {}
    for sname, m in strata.items():
        row = {"n": int(m.sum()), "pos": int(y[m].sum())}
        if row["pos"] >= 10 and (y[m] == 0).sum() >= 10:
            for pn, p in preds.items():
                ok = m & np.isfinite(p)
                ci = dm.cluster_bootstrap_auc_ci(y[ok].astype(np.float64), p[ok], meta["day"][ok], n_boot=n_boot)
                row[pn] = {"auc": dm.compute_auc(y[ok].astype(np.float64), p[ok]), "ci": [ci["ci_lo"], ci["ci_hi"]]}
        out[sname] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stress_{name}_{split}_{label}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def compare(name_a: str, name_b: str, split: str, label: str = "storm_60", n_boot: int = 1000, seed: int = 42) -> dict:
    """B minus A on one split, paired, whole UTC days resampled: AUC and Brier deltas with 95% intervals.
    Both experiments must have been run with save_preds on the same split rows."""
    pa = np.load(LAB / "preds" / f"{name_a}_{split}.npy").astype(np.float64)
    pb = np.load(LAB / "preds" / f"{name_b}_{split}.npy").astype(np.float64)
    _, Y, meta = load(split)
    y = np.asarray(Y[:, LIDX[label]], np.float64)
    if not (len(pa) == len(pb) == len(y)):
        raise ValueError(f"row counts differ: {len(pa)} {len(pb)} {len(y)}")
    ok = np.isfinite(pa) & np.isfinite(pb)          # e.g. v2 cannot score a row with no analysis
    n_masked = int((~ok).sum())
    pa, pb, y, days = pa[ok], pb[ok], y[ok], np.asarray(meta["day"])[ok]
    clusters = dm._cluster_index(days)
    rng = np.random.RandomState(seed)
    d_auc = np.empty(n_boot)
    d_bri = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([clusters[c] for c in rng.randint(0, len(clusters), size=len(clusters))])
        yy = y[idx]
        d_auc[i] = dm.compute_auc(yy, pb[idx]) - dm.compute_auc(yy, pa[idx])
        d_bri[i] = np.mean((pb[idx] - yy) ** 2) - np.mean((pa[idx] - yy) ** 2)
    out = {"a": name_a, "b": name_b, "split": split, "label": label, "n_days": len(clusters), "n_boot": n_boot,
           "n_rows": int(len(y)), "n_rows_masked_nonfinite": n_masked,
           "delta_auc": dm.compute_auc(y, pb) - dm.compute_auc(y, pa),
           "delta_auc_ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
           "delta_brier": float(np.mean((pb - y) ** 2) - np.mean((pa - y) ** 2)),
           "delta_brier_ci": [float(np.percentile(d_bri, 2.5)), float(np.percentile(d_bri, 97.5))]}
    (OUT / f"compare_{name_b}_vs_{name_a}_{split}_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "baselines":
        label = sys.argv[2] if len(sys.argv) > 2 else "storm_60"
        for split, res in baselines(label=label).items():
            for name, r in res.items():
                print(f"{split:4s} {name:22s} AUC {r['auc']:.4f} {[round(x, 4) for x in r['auc_ci']]} "
                      f"BSS {r['bss']:+.4f} pos {r['pos']}", flush=True)
        return 0
    if cmd == "stress":
        name, split = sys.argv[2], sys.argv[3]
        label = sys.argv[4] if len(sys.argv) > 4 else "storm_60"
        for s, r in stress(name, split, label).items():
            cells = "  ".join(f"{k} {v['auc']:.3f}" for k, v in r.items() if isinstance(v, dict))
            print(f"{s:20s} n={r['n']:8d} pos={r['pos']:5d}  {cells}", flush=True)
        return 0
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
        if "--no-dev" in sys.argv[3:]:      # validation only: every choice is made there
            exps = [{**e, "eval_dev": False} for e in exps]
        only = [a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--only=")]
        if only:
            exps = [e for e in exps if e["name"] in set(only[0].split(","))]
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
