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
OUT = Path(os.environ.get("HAZARDPULSE_LAB_OUT", str(ROOT / "results" / "lab")))
# amendment 8: "available" = the 80 km HRRR block recomputed from the analysis a forecast could have
# used (published before the observation; rebuild_h80_available.py), and v2 rescored by the same rule
H80_SOURCE = os.environ.get("HAZARDPULSE_H80_SOURCE", "store")
V2_DIR = os.environ.get("HAZARDPULSE_V2_DIR", "_v2")
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
DERIVED_LABELS = ("storm_60_ef2",)   # amendment 7: storm_60 AND the matched report is EF2+


def get_y(Y, meta, name: str) -> np.ndarray:
    """The label column NAME as an int8 copy; derived labels are built from stored columns."""
    if name == "storm_60_ef2":
        return ((np.asarray(Y[:, LIDX["storm_60"]]) == 1) & (np.asarray(meta["ef"]) >= 2)).astype(np.int8)
    return np.array(Y[:, LIDX[name]], np.int8)


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
            if H80_SOURCE == "available":
                f = STORE / "_h80avail" / f"{p.stem}.npz"
                if not f.exists():
                    raise SystemExit(f"{f} missing: run rebuild_h80_available.py first")
                lo, hi = sf.BLOCKS["H80"]
                with np.load(f) as a:
                    if a["H80"].shape != (m, hi - lo):
                        raise SystemExit(f"{f}: shape {a['H80'].shape} for {m} rows")
                    X[i:i + m, lo:hi] = a["H80"]
            elif H80_SOURCE != "store":
                raise SystemExit(f"unknown HAZARDPULSE_H80_SOURCE {H80_SOURCE!r}")
            Y[i:i + m] = z["Y"]
            meta["day"].append(np.full(m, int(p.stem), np.int64))
            for k in ("t", "lat", "lon", "ef", "lead_min", "analysis"):
                meta[k].append(z[k])
            # the served v2 model's probability (score_v2_on_store.py), NaN where absent
            v2 = STORE / V2_DIR / f"{p.stem}.npy"
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


def exp_cols(exp: dict) -> list[int]:
    """The store columns an experiment reads: its blocks minus its ``drop`` list. Every path that fits or
    scores an experiment uses this (amendment 10: the final pipeline and the exporter had read the blocks
    alone, so a ``drop`` honoured on validation was silently ignored in the final refit)."""
    return cols_for(exp["blocks"], exp.get("drop"))


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
                 eval_metric="AUC", allow_writing_files=False)
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
    if kind == "numpy_gbt":  # the in-repo pure-NumPy GBT (served v2's trainer); no NaN support
        med = np.nanmedian(np.asarray(Xtr, np.float64), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        def fill(X):
            A = np.asarray(X, np.float32).copy()
            bad = ~np.isfinite(A)
            A[bad] = np.broadcast_to(med.astype(np.float32), A.shape)[bad]
            return A
        norm = dm.FeatureNormalizer()
        norm.fit(fill(Xtr))
        gbt = dm.GradientBoostedTrees(**params)
        info = gbt.fit(norm.transform(fill(Xtr)), np.asarray(ytr, np.float64),
                       X_val=norm.transform(fill(Xes)), y_val=np.asarray(yes, np.float64))
        return (lambda X: gbt.decision_function(norm.transform(fill(X)))), \
            {"n_trees": len(gbt.trees) if hasattr(gbt, "trees") else None,
             "fit": {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str))}}, gbt
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
# the files that record CHOICES (best_blocks / best_config / chosen_on_validation); the corrected
# program (amendment 8) keeps its own in experiments/avail so the original record stays intact
CHOICES = Path(os.environ.get("HAZARDPULSE_CHOICES_DIR", str(EXPERIMENTS)))


def resolve_blocks(exp: dict) -> dict:
    """``blocks_from: "best"`` -> the block set chosen on validation, as committed in
    experiments/best_blocks.json (written once step 1 of the protocol has decided it)."""
    if exp.get("config_from") == "best":
        # model family + hyper-parameters chosen by steps 2-3 (experiments/best_config.json);
        # the experiment's own keys (calibration, neg_per_pos, label, seed, ...) still win
        cfg = json.loads((CHOICES / "best_config.json").read_text(encoding="utf-8"))
        chosen = {k: cfg[k] for k in ("calibration", "neg_per_pos", "label") if k in cfg}
        exp = {"model": cfg["model"], "params": cfg.get("params", {}), "blocks_from": "best", **chosen,
               **{k: v for k, v in exp.items() if k != "config_from"}}
    if exp.get("blocks_from") != "best":
        return exp
    best = json.loads((CHOICES / "best_blocks.json").read_text(encoding="utf-8"))
    return {**exp, "blocks": best["blocks"], "drop": best.get("drop", [])}


def run(exp: dict) -> dict:
    t0 = time.time()
    exp = resolve_blocks(exp)
    lab = exp.get("label", "storm_60")                        # the label the model is TRAINED on
    ev = exp.get("eval_label", lab)                           # the label it is calibrated and SCORED on
    cols = cols_for(exp["blocks"], exp.get("drop"))
    use_w = "W" in exp["blocks"]
    seed = int(exp.get("seed", 0))
    Xt, Yt, mt = load("train")
    y_tr_all = get_y(Yt, mt, lab)                         # a copy: the memmap is read-only
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
    yv_train = get_y(Yv, mv, lab)
    yv = get_y(Yv, mv, ev)
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
        yd = get_y(Yd, md, ev)
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
    y_val = get_y(val_Y, val_m, label)
    (LAB / "preds").mkdir(parents=True, exist_ok=True)
    cols = {name: (FIDX[c], scale) for name, (c, scale) in BASELINE_COLUMNS.items()}
    sources = {name: (lambda X, j=j: np.asarray(X[:, j], np.float64)) for name, (j, _) in cols.items()}
    for split in splits:
        X, Y, m = load(split)
        y = get_y(Y, m, label)
        res = {}
        for name, (j, scale) in cols.items():
            raw = np.nan_to_num(np.asarray(X[:, j], np.float64), nan=-1.0)
            s_val = np.nan_to_num(sources[name](val_X), nan=-1.0)
            if scale is not None:
                # a published probability is Platt-scaled on its LOG-ODDS: on the raw 0-100 percent
                # the Newton fit collapsed to a constant (AUC exactly 0.5000 on 2023)
                def lo(v, scale=scale):
                    q = np.clip(np.where(v < 0, 0.0, v) * scale, 0.0005, 0.9995)
                    return np.log(q / (1 - q))
                s_val_platt, raw_platt = lo(s_val), lo(raw)
            else:
                s_val_platt, raw_platt = s_val, raw
            if scale is not None:
                p_raw = np.clip(raw * scale, 0.0, 1.0)
                np.save(LAB / "preds" / f"{name}_raw_{split}.npy", p_raw.astype(np.float32))
                res[f"{name}_raw"] = metrics(y, p_raw, m["day"], n_boot=200)
            # Platt on validation: a monotone recalibration, the fairest probability a raw index gets
            p_cal = (out_of_fold_calibrated("platt", s_val_platt, y_val, val_m["day"]) if split == "val"
                     else fit_calibrator("platt", s_val_platt, y_val)(raw_platt))
            np.save(LAB / "preds" / f"{name}_platt_{split}.npy", p_cal.astype(np.float32))
            res[f"{name}_platt"] = metrics(y, p_cal, m["day"], n_boot=200)
        # ProbTor with its ties broken (amendment 8): published ProbTor is an integer percent and 0 for
        # ~93% of storms, so much of its AUC deficit is ties, not ranking. Ties are broken by ProbSevere's
        # own any-severe probability (+0.001 x ps): a discrimination comparison, not the product as issued.
        def tie_broken(Xs):
            tor = np.nan_to_num(np.asarray(Xs[:, FIDX["p_ps_tor"]], np.float64), nan=0.0)
            ps = np.nan_to_num(np.asarray(Xs[:, FIDX["p_ps"]], np.float64), nan=0.0)
            q = np.clip((tor + 0.001 * ps) / 100.0, 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))
        tb_val, tb = tie_broken(val_X), tie_broken(X)
        p_tb = (out_of_fold_calibrated("platt", tb_val, y_val, val_m["day"]) if split == "val"
                else fit_calibrator("platt", tb_val, y_val)(tb))
        np.save(LAB / "preds" / f"probtor_tiebroken_{split}.npy", p_tb.astype(np.float32))
        res["probtor_tiebroken"] = metrics(y, p_tb, m["day"], n_boot=200)
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
    y = get_y(Y, meta, label)
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


def ensemble(name: str, members: list[str], splits=("val", "dev"), label: str = "storm_60") -> dict:
    """Equal-weight mean of the members' calibrated LOG-ODDS (declared before any result: no
    weights are fitted, so validation is not spent on them), re-calibrated by Platt on
    validation (out-of-fold there), written as preds/<name>_<split>.npy like any experiment."""
    def logit(p):
        p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    out = {"name": name, "members": members}
    _, Yv, mv = load("val")
    yv = get_y(Yv, mv, label)
    s_val = np.mean([logit(np.load(LAB / "preds" / f"{m}_val.npy")) for m in members], axis=0)
    cal = fit_calibrator("platt", s_val, yv)
    for split in splits:
        files = [LAB / "preds" / f"{m}_{split}.npy" for m in members]
        if not all(f.exists() for f in files):
            continue
        s = np.mean([logit(np.load(f)) for f in files], axis=0)
        p = out_of_fold_calibrated("platt", s, yv, mv["day"]) if split == "val" else cal(s)
        np.save(LAB / "preds" / f"{name}_{split}.npy", p.astype(np.float32))
        _, Y, m = load(split)
        out[split] = metrics(get_y(Y, m, label), p, m["day"], n_boot=300)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


FINAL_SOURCES = ("train", "val", "dev")          # 2020-10-15 .. 2024-12-31, the refit population


def fit_fixed_rounds(exp: dict, X, y, w, n_rounds: int):
    """The chosen LightGBM configuration with a FIXED number of trees (no early stopping: in the
    final refit every season is training data, so there is nothing honest left to stop on)."""
    import lightgbm as lgb
    if exp["model"] != "lgbm":
        raise ValueError("the final pipeline is written for the chosen LightGBM configuration")
    p = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=50,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             verbose=-1, seed=int(exp.get("seed", 0)), num_threads=4, deterministic=True, force_row_wise=True)
    p.update(exp.get("params", {}))
    p.pop("n_rounds", None)
    return lgb.train(p, lgb.Dataset(X, y, weight=w, free_raw_data=True), int(n_rounds))


def _rows_for(parts: dict, years_in: set[int], neg_per_pos: int, seed: int, cols, use_w, label: str):
    """All positives + neg_per_pos negatives per positive from the given calendar years of every
    source split; returns (X, y, balanced weights). 2020 (Oct-Dec) counts as part of 2021."""
    Xs, ys = [], []
    for k, (X, Y, meta, W) in parts.items():
        yrs = np.maximum(np.asarray(meta["day"], np.int64) // 10000, 2021)
        y_all = get_y(Y, meta, label)
        y_all[~np.isin(yrs, list(years_in))] = -1
        if not (y_all == 1).any() or not (y_all == 0).any():
            continue        # e.g. the held-out year is this whole split (val = 2023): nothing to draw
        # a fixed offset per source split (str hash() is salted per process -- not reproducible)
        rows, _ = train_sample(None, y_all, neg_per_pos, seed + 101 * FINAL_SOURCES.index(k))
        Xs.append(matrix(X, W, rows, cols, use_w))
        ys.append(y_all[rows])
    Xc, yc = np.vstack(Xs), np.concatenate(ys)
    return Xc, yc, balanced_weights(yc)


LOYO_YEARS = (2021, 2022, 2023, 2024)


def _loyo_scores(exp: dict, rounds: int, parts: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-year-out raw scores for every row of 2020-10..2024 (each year scored by a model
    that never saw it), with their labels and days. Never touches 2025."""
    cols = exp_cols(exp)
    use_w = "W" in exp["blocks"]
    oof_s, oof_y, oof_day = [], [], []
    for yr in LOYO_YEARS:
        Xc, yc, wc = _rows_for(parts, set(LOYO_YEARS) - {yr}, exp["neg_per_pos"], exp["seed"], cols, use_w,
                               exp["label"])
        bst = fit_fixed_rounds(exp, Xc, yc, wc, rounds)
        for k, (X, Y, meta, W) in parts.items():
            yrs = np.maximum(np.asarray(meta["day"], np.int64) // 10000, 2021)
            idx = np.flatnonzero(yrs == yr)
            if len(idx) == 0:
                continue
            s = np.concatenate([bst.predict(matrix(X, W, idx[i:i + 200_000], cols, use_w), raw_score=True)
                                for i in range(0, len(idx), 200_000)])
            oof_s.append(s)
            oof_y.append(get_y(Y, meta, exp["label"])[idx])
            oof_day.append(np.asarray(meta["day"])[idx])
        print(f"  LOYO {yr}: fitted on {len(yc)} rows ({int(yc.sum())} positives)", flush=True)
    return np.concatenate(oof_s), np.concatenate(oof_y), np.concatenate(oof_day)


def _save_oof(name: str, s, y, d) -> None:
    LAB.joinpath("oof").mkdir(parents=True, exist_ok=True)
    np.savez(LAB / "oof" / f"{name}.npz", score=np.asarray(s, np.float64), y=np.asarray(y, np.int8),
             day=np.asarray(d, np.int64))


def _final_core(exp: dict, rounds: int) -> dict:
    """LOYO Platt, refit on every season, score 2025 ONCE. Refuses a second run."""
    out_path = OUT / f"final_{exp['name']}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: 2025 is read once")
    cols = exp_cols(exp)
    use_w = "W" in exp["blocks"]
    parts = {k: (*load(k), load_w(k)) for k in FINAL_SOURCES}
    s_oof, y_oof, d_oof = _loyo_scores(exp, rounds, parts)
    _save_oof(exp["name"], s_oof, y_oof, d_oof)
    cal = dm.fit_platt(s_oof, y_oof)
    # the served model: every season, the same rounds
    Xc, yc, wc = _rows_for(parts, set(LOYO_YEARS), exp["neg_per_pos"], exp["seed"], cols, use_w, exp["label"])
    bst = fit_fixed_rounds(exp, Xc, yc, wc, rounds)
    LAB.joinpath("models").mkdir(parents=True, exist_ok=True)
    model_path = LAB / "models" / f"{exp['name']}.lgbm.txt"
    bst.save_model(str(model_path))
    # 2025, once
    if not (LAB / "final_X.npy").exists():
        assemble("final")
    Xf, Yf, mf = load("final")
    Wf = load_w("final") if use_w else None
    sf_ = predict_stream(lambda A: bst.predict(A, raw_score=True), Xf, cols, W=Wf)
    pf = dm.apply_calibration(sf_, cal)
    np.save(LAB / "preds" / f"{exp['name']}_final.npy", pf.astype(np.float32))
    yf = get_y(Yf, mf, exp["label"])
    res = {"exp": exp, "rounds": rounds, "calibration": cal, "model_file": str(model_path),
           "refit_rows": int(len(yc)), "refit_positives": int(yc.sum()),
           "oof_2021_2024": metrics(y_oof, dm.apply_calibration(s_oof, cal), d_oof, n_boot=300),
           "final_2025": metrics(yf, pf, mf["day"], n_boot=2000)}
    out_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def _chosen_exp(which: str) -> dict:
    chosen = json.loads((CHOICES / "chosen_on_validation.json").read_text(encoding="utf-8"))
    exp = dict(chosen["primary"])
    if which == "plus_W":
        exp.update(chosen["secondary_with_warnings"]["same_as_primary_except"])
        exp["name"] = chosen["secondary_with_warnings"]["name"]
    return exp


def final_pipeline(which: str = "primary") -> dict:
    """Protocol 'Final pipeline': refit on 2020-10..2024 with the validation-chosen rounds,
    Platt-calibrate on leave-one-year-out scores, score 2025 ONCE. Refuses a second run."""
    rounds = int(json.loads((OUT / "c_platt.json").read_text(encoding="utf-8"))["fit"]["best_iteration"])
    return _final_core(_chosen_exp(which), rounds)


def final_product(product: str) -> dict:
    """Amendment 7: the served (+W) configuration on another label; its rounds are the ones its
    own dev run chose by early stopping on validation (results/lab/p_<product>.json)."""
    dev = json.loads((OUT / f"p_{product}.json").read_text(encoding="utf-8"))
    exp = {**_chosen_exp("plus_W"), "name": f"v3_plus_W_{product}", "label": dev["exp"]["label"]}
    return _final_core(exp, int(dev["fit"]["best_iteration"]))


def oof_only(which: str = "plus_W") -> dict:
    """Recompute an already-final model's LOYO scores (no 2025) for its Venn-Abers interval, and
    check they reproduce the final run's Platt calibration exactly (same seeds, same rows)."""
    exp = _chosen_exp(which)
    final = json.loads((OUT / f"final_{exp['name']}.json").read_text(encoding="utf-8"))
    parts = {k: (*load(k), load_w(k)) for k in FINAL_SOURCES}
    s, y, d = _loyo_scores(exp, int(final["rounds"]), parts)
    cal = dm.fit_platt(s, y)
    for k in ("a", "b"):
        if abs(float(cal[k]) - float(final["calibration"][k])) > 1e-9:
            raise SystemExit(f"LOYO scores do not reproduce the final calibration ({k}: {cal[k]} vs "
                             f"{final['calibration'][k]})")
    _save_oof(exp["name"], s, y, d)
    return {"name": exp["name"], "rows": int(len(s)), "positives": int(y.sum()), "calibration": cal}


def event_clusters(days, pos) -> np.ndarray:
    """Bootstrap unit for outbreak-driven statistics: consecutive calendar days that each hold a
    positive are ONE event (a multi-day outbreak is not several independent draws); a day without
    a positive is its own cluster. Returns a cluster index per row."""
    days = np.asarray(days, np.int64)
    uniq = np.unique(days)
    has_pos = set(np.unique(days[np.asarray(pos, bool)]).tolist())
    ords = np.array([dt_ordinal(int(d)) for d in uniq])
    cid = np.empty(len(uniq), np.int64)
    c = -1
    for i, d in enumerate(uniq):
        if i > 0 and int(d) in has_pos and int(uniq[i - 1]) in has_pos and ords[i] - ords[i - 1] == 1:
            cid[i] = c
        else:
            c += 1
            cid[i] = c
    return cid[np.searchsorted(uniq, days)]


def dt_ordinal(yyyymmdd: int) -> int:
    import datetime as _dt
    return _dt.date(yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100).toordinal()


def nws_bar(name: str, split: str, label: str = "storm_60", n_boot: int = 2000, seed: int = 42) -> dict:
    """NWS tornado warnings as a bar: their POD at their own false-alarm rate (POFD = share of
    negative storm observations inside an active warning), against the model's POD at the SAME
    POFD (threshold fixed on the whole split). Paired whole-day bootstrap of POD(model) - POD(NWS)."""
    _, Y, meta = load(split)
    y = get_y(Y, meta, label)
    W = load_w(split)
    warned = np.asarray(W[:, 0], np.float64)
    p = np.load(LAB / "preds" / f"{name}_{split}.npy").astype(np.float64)
    ok = np.isfinite(warned) & np.isfinite(p)
    y, warned, p, days = y[ok], warned[ok] > 0.5, p[ok], np.asarray(meta["day"])[ok]
    neg, pos = y == 0, y == 1
    pofd = float(warned[neg].mean())
    thr = float(np.quantile(p[neg], 1.0 - pofd))          # the model alarms on the same share of negatives
    alarm = p > thr
    uniq, di = np.unique(days, return_inverse=True)
    D = len(uniq)
    hit_m = np.bincount(di, weights=(alarm & pos), minlength=D)
    hit_w = np.bincount(di, weights=(warned & pos), minlength=D)
    npos = np.bincount(di, weights=pos, minlength=D)
    M = day_bootstrap_counts(D, n_boot, seed)
    d_day = (M @ hit_m - M @ hit_w) / np.maximum(M @ npos, 1)
    # amendment 8 (adversary): resample multi-day EVENTS, and re-match the model's threshold to the
    # warnings' false-alarm rate INSIDE every replicate (a threshold fixed on the whole split is a
    # choice made with the very sample being resampled)
    ev = event_clusters(days, pos)
    E = int(ev.max()) + 1
    grid = np.unique(np.quantile(p[neg], 1.0 - np.clip(pofd * np.geomspace(0.25, 4.0, 400), 1e-7, 0.5)))
    neg_above = np.zeros((E, len(grid)))
    pos_above = np.zeros((E, len(grid)))
    for k, t in enumerate(grid):
        a = p > t
        neg_above[:, k] = np.bincount(ev, weights=a & neg, minlength=E)
        pos_above[:, k] = np.bincount(ev, weights=a & pos, minlength=E)
    e_neg = np.bincount(ev, weights=neg, minlength=E)
    e_pos = np.bincount(ev, weights=pos, minlength=E)
    e_wneg = np.bincount(ev, weights=warned & neg, minlength=E)
    e_wpos = np.bincount(ev, weights=warned & pos, minlength=E)
    Me = day_bootstrap_counts(E, n_boot, seed + 1)
    nws_pofd_r = (Me @ e_wneg) / np.maximum(Me @ e_neg, 1)
    model_pofd_r = (Me @ neg_above) / np.maximum(Me @ e_neg, 1)[:, None]
    k_r = np.argmin(np.abs(model_pofd_r - nws_pofd_r[:, None]), axis=1)
    pod_m_r = (Me @ pos_above)[np.arange(n_boot), k_r] / np.maximum(Me @ e_pos, 1)
    pod_w_r = (Me @ e_wpos) / np.maximum(Me @ e_pos, 1)
    d_ev = pod_m_r - pod_w_r
    out = {"model": name, "split": split, "label": label, "n_rows": int(len(y)), "positives": int(pos.sum()),
           "nws_pofd": pofd, "model_pofd": float(alarm[neg].mean()), "threshold": thr,
           "nws_pod": float(warned[pos].mean()), "model_pod_at_nws_pofd": float(alarm[pos].mean()),
           "delta_pod": float(alarm[pos].mean() - warned[pos].mean()),
           "delta_pod_ci": [float(np.percentile(d_ev, 2.5)), float(np.percentile(d_ev, 97.5))],
           "delta_pod_ci_method": f"{E} multi-day event clusters, threshold re-matched per replicate",
           "p_delta_le_0": float(np.mean(d_ev <= 0)),
           "delta_pod_ci_day_fixed_threshold": [float(np.percentile(d_day, 2.5)), float(np.percentile(d_day, 97.5))],
           "rows_without_warning_state": int((~ok).sum())}
    (OUT / f"nws_bar_{name}_{split}_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def day_pair_matrix(y, s, day_idx: np.ndarray, n_days: int):
    """``U[d, e]`` = Mann-Whitney pair count between the positives of day d and the negatives of
    day e (ties 1/2), plus per-day positive / negative counts. A day bootstrap with day
    multiplicities m then has AUC = m.U.m / ((m.P)(m.N)) exactly -- every replicate costs a
    D x D quadratic form instead of a sort of the whole split."""
    y = np.asarray(y)
    s = np.asarray(s, np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    C = np.zeros((len(pos), n_days))
    sp = s[pos]
    for e in range(n_days):
        ne = np.sort(s[neg[day_idx[neg] == e]])
        if len(ne):
            lo = np.searchsorted(ne, sp, side="left")
            hi = np.searchsorted(ne, sp, side="right")
            C[:, e] = lo + 0.5 * (hi - lo)
    U = np.zeros((n_days, n_days))
    np.add.at(U, day_idx[pos], C)
    P = np.bincount(day_idx[pos], minlength=n_days).astype(np.float64)
    N = np.bincount(day_idx[neg], minlength=n_days).astype(np.float64)
    return U, P, N


def day_bootstrap_counts(n_days: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_days) day multiplicities of a whole-day bootstrap."""
    rng = np.random.RandomState(seed)
    return np.stack([np.bincount(rng.randint(0, n_days, size=n_days), minlength=n_days) for _ in range(n_boot)]
                    ).astype(np.float64)


def compare(name_a: str, name_b: str, split: str, label: str = "storm_60", n_boot: int = 2000, seed: int = 42) -> dict:
    """B minus A on one split, paired, whole UTC days resampled: AUC and Brier deltas with 95% intervals.
    Both experiments must have been run with save_preds on the same split rows."""
    pa = np.load(LAB / "preds" / f"{name_a}_{split}.npy").astype(np.float64)
    pb = np.load(LAB / "preds" / f"{name_b}_{split}.npy").astype(np.float64)
    _, Y, meta = load(split)
    y = get_y(Y, meta, label).astype(np.float64)
    if not (len(pa) == len(pb) == len(y)):
        raise ValueError(f"row counts differ: {len(pa)} {len(pb)} {len(y)}")
    ok = np.isfinite(pa) & np.isfinite(pb)          # e.g. v2 cannot score a row with no analysis
    n_masked = int((~ok).sum())
    pa, pb, y, days = pa[ok], pb[ok], y[ok], np.asarray(meta["day"])[ok]
    uniq, day_idx = np.unique(days, return_inverse=True)
    D = len(uniq)
    M = day_bootstrap_counts(D, n_boot, seed)                    # the SAME draws for both forecasts
    Ua, P, N = day_pair_matrix(y, pa, day_idx, D)
    Ub, _, _ = day_pair_matrix(y, pb, day_idx, D)
    denom = (M @ P) * (M @ N)
    d_auc = (np.einsum("bi,ij,bj->b", M, Ub, M) - np.einsum("bi,ij,bj->b", M, Ua, M)) / denom
    sse_a = np.bincount(day_idx, weights=(pa - y) ** 2, minlength=D)
    sse_b = np.bincount(day_idx, weights=(pb - y) ** 2, minlength=D)
    n_d = np.bincount(day_idx, minlength=D).astype(np.float64)
    d_bri = (M @ sse_b - M @ sse_a) / (M @ n_d)
    clusters = uniq
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
    if cmd == "nws_bar":
        r = nws_bar(sys.argv[2], sys.argv[3])
        print(f"{r['split']} NWS POD {r['nws_pod']:.3f} at POFD {r['nws_pofd']:.4f} | {r['model']} POD "
              f"{r['model_pod_at_nws_pofd']:.3f} | delta {r['delta_pod']:+.3f} "
              f"[{r['delta_pod_ci'][0]:+.3f}, {r['delta_pod_ci'][1]:+.3f}]", flush=True)
        return 0
    if cmd == "oof_only":
        print(json.dumps(oof_only(sys.argv[2] if len(sys.argv) > 2 else "plus_W"), indent=1, default=float))
        return 0
    if cmd in ("final", "final_product"):
        which = sys.argv[2] if len(sys.argv) > 2 else "primary"
        r = final_pipeline(which) if cmd == "final" else final_product(which)
        f = r["final_2025"]
        print(f"FINAL 2025 {r['exp']['name']}: AUC {f['auc']:.4f} {[round(x, 4) for x in f['auc_ci']]} "
              f"BSS {f['bss']:+.4f} n={f['n']} pos={f['pos']}", flush=True)
        return 0
    if cmd == "ensemble":
        r = ensemble(sys.argv[2], sys.argv[3].split(","))
        for split in ("val", "dev"):
            if split in r:
                print(f"{sys.argv[2]} {split} AUC {r[split]['auc']:.4f} {[round(x, 4) for x in r[split]['auc_ci']]} "
                      f"BSS {r[split]['bss']:+.4f}", flush=True)
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
