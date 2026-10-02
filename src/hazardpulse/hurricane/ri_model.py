"""Frozen, versioned serving artifacts for the operational hurricane RI models.

Why this module exists
----------------------
The daily hurricane workflow used to *retrain* all four v8.1 members on the 133,882-case
history inside the GitHub Actions job whenever any storm was on the board. The trainer's
split search was a pure-Python loop (measured on a 24-core dev box: 798 s for the 200-tree
depth-3 member alone), so from 2026-05-27 -- the first day of the season with a storm
present at every run -- every job was cancelled by its 10-minute budget and the live
hurricane product stopped updating. A model is a pure function of (training-data bytes,
config, trainer code), so it is trained once, offline, and served pinned.

Two further defects are fixed here (2026-10-02):

* **Calibration.** v8.1's Platt step ran 500 fixed gradient steps from (a=1, b=0) on its own
  training predictions. It never converged, so every served probability sat in
  [17.86%, 33.86%] whatever the storm (base rate 1.1%). An iteration count reports success
  by construction. :func:`fit_logit_calibrator` is a Newton/IRLS logistic fit on the
  ensemble's log-odds that runs to a gradient tolerance and RAISES if it does not get
  there; it is fitted on held-out seasons the members never saw.
* **Training clock (v8.2).** v8.1's training set stepped through 3-hourly IBTrACS rows as if
  they were 6-hourly (its "24 h" label spans 12 h). The v8.2 recipe trains on the true-clock
  set built by scripts/build_hurricane_training_data.py.

Recipes (``RECIPES``) -- all train the same four members (v8.1 hyper-parameters):

* ``hurricane_ri_v8_1``   -- the original pin, reproduced bit for bit for comparison only:
  members on all v8.1 rows, legacy in-sample Platt. Never served again.
* ``hurricane_ri_v8_1_1`` -- v8.1 rows (storm seasons <= 2021) + held-out converged
  calibration on true-clock 2022-2024 rows, as v8.1 is actually served live.
* ``hurricane_ri_v8_2``   -- true-clock rows (<= 2021) + the same held-out calibration.

Which one the scorer serves is decided by scripts/evaluate_hurricane_ri.py at the
2018/2021 rolling origin, by a rule written down before it ran.

Persistence is JSON with ``float.__repr__`` (lossless for float64) and a provenance block
binding the artifact to the SHA-256 of every data file it was fitted on (line endings
normalised) and to the exact config; :func:`load_model` refuses an artifact whose binding
does not match.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from hazardpulse.hurricane import operational_ri as ori

SCHEMA = "hazardpulse.hurricane_ri_model/2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "results"
DATASETS = {
    "v8.1": RESULTS / "hurricane_operational_ri_2000_2024_al_sst.jsonl",
    "v8.2": RESULTS / "hurricane_operational_ri_v8_2_2000_2024.jsonl",
}
DEFAULT_TRAINING_DATA = DATASETS["v8.1"]  # the v8.1 pin's data (kept for compatibility)

LIVE_SPEED_AGE_REASON = (
    "the v8.1 training set steps 3-hourly IBTrACS rows as 6-hourly, so its "
    "translation_speed_kmh is half the true speed and storm_age_h twice the true age; a "
    "true live value would sit on a different scale from what the model learned, so these "
    "two are median-imputed at serving time"
)

RECIPES: dict[str, dict[str, Any]] = {
    "hurricane_ri_v8_1": {
        "dataset": "v8.1",
        "members_years": (2000, 2024),
        "calibration": {"method": "platt_prob_gd500_insample"},
        "impute_live": ["translation_speed_kmh", "storm_age_h"],
        "served": False,
    },
    "hurricane_ri_v8_1_1": {
        "dataset": "v8.1",
        "members_years": (2000, 2021),
        "calibration": {"method": "logistic_on_logit_newton", "dataset": "v8.2", "years": (2022, 2024)},
        "impute_live": ["translation_speed_kmh", "storm_age_h"],
        "served": True,
    },
    "hurricane_ri_v8_2": {
        "dataset": "v8.2",
        "members_years": (2000, 2021),
        "calibration": {"method": "logistic_on_logit_newton", "dataset": "v8.2", "years": (2022, 2024)},
        "impute_live": [],
        "served": True,
    },
}
KNOWN_VERSIONS = frozenset(RECIPES)
ARTIFACTS = {v: RESULTS / "models" / f"{v}.json" for v in RECIPES}
DEFAULT_ARTIFACT = ARTIFACTS["hurricane_ri_v8_1"]

# Everything that decides the fitted member numbers. A change to any of these sources means
# the committed artifacts no longer describe what the code would train (a test enforces it).
TRAINER_FUNCTIONS = (
    "build_feature_matrix",
    "train_median_impute",
    "standardize",
    "select_features_by_logistic",
    "train_logistic",
    "train_bagged_logistic",
    "_pow2_like_scalar",
    "_best_split_for_feature",
    "_build_tree_recursive",
    "_predict_tree",
    "train_histogram_gbt",
    "predict_histogram_gbt",
    "predict_logistic",
    "predict_bagged",
    "platt_calibrate",
    "sigmoid",
)

LOGIT_EPS = 1e-6


class ModelArtifactError(RuntimeError):
    """The serving artifact is missing, malformed, or not bound to the current inputs."""


class CalibrationError(ModelArtifactError):
    """A calibration fit did not converge (or the data cannot identify one): never served."""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a text file's content with CRLF line endings normalised to LF.

    git checks text files out with CRLF on Windows (core.autocrlf) and LF on Linux. The
    training rows are identical either way (one JSON object per line), so the provenance
    binding must be too: a raw-byte digest taken on the Windows box (where the model is
    trained) never matched the Linux CI checkout of the same commit (measured 2026-10-02:
    feace170... vs 98c2b6...). Normalised, the digest equals ``sha256(git cat-file -p
    HEAD:<path>)`` on every platform.
    """
    digest = hashlib.sha256()
    carry_cr = False
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            if carry_cr:
                block = b"\r" + block
            carry_cr = block.endswith(b"\r")   # may be the first half of a CRLF split by the read
            if carry_cr:
                block = block[:-1]
            digest.update(block.replace(b"\r\n", b"\n"))
    if carry_cr:
        digest.update(b"\r")
    return digest.hexdigest()


CALIBRATION_FUNCTIONS = ("newton_logistic", "fit_logit_calibrator", "_logit", "apply_calibration")


def trainer_fingerprint() -> str:
    """SHA-256 over the source of every function that decides the fitted numbers.

    Members (operational_ri) and the calibration fit (this module): a change to either
    means a committed artifact no longer describes what the code would train.
    """
    import sys

    here = sys.modules[__name__]
    digest = hashlib.sha256()
    for owner, names in ((ori, TRAINER_FUNCTIONS), (here, CALIBRATION_FUNCTIONS)):
        for name in names:
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(inspect.getsource(getattr(owner, name)).encode("utf-8") + b"\0")
    return digest.hexdigest()


def config_dict(config: ori.OperationalRIConfig | None = None) -> dict[str, Any]:
    return dataclasses.asdict(config or ori.best_known_operational_ri_config())


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Data selection (by the storm's first season, so a storm never straddles a split)
# ---------------------------------------------------------------------------

def storm_first_year(cases: list[dict]) -> dict[str, int]:
    first: dict[str, int] = {}
    for c in cases:
        sid, year = c["storm_id"], int(c["season_year"])
        if sid not in first or year < first[sid]:
            first[sid] = year
    return first


def select_years(cases: list[dict], years: tuple[int, int]) -> list[dict]:
    lo, hi = int(years[0]), int(years[1])
    first = storm_first_year(cases)
    return [c for c in cases if lo <= first[c["storm_id"]] <= hi]


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------

def _tree_to_json(node: dict) -> dict:
    if "leaf" in node:
        return {"leaf": float(node["leaf"])}
    return {
        "feature": int(node["feature"]),
        "threshold": float(node["threshold"]),
        "left": _tree_to_json(node["left"]),
        "right": _tree_to_json(node["right"]),
    }


def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def fit_members(
    historical_cases: list[dict],
    config: ori.OperationalRIConfig | None = None,
    *,
    log=print,
    return_training_ensemble: bool = False,
):
    """Train the four v8.1 members exactly as the scorer's former in-job training did.

    The call sequence and arguments are the former ``train_and_score`` body verbatim (the
    member half); every fitted parameter is captured. With ``return_training_ensemble`` the
    in-sample ensemble probabilities and labels are returned too (the legacy Platt fit).
    """
    if not historical_cases:
        raise ModelArtifactError("no historical cases to train on")
    config = config or ori.best_known_operational_ri_config()

    X_hist, y_hist, _years, feature_names = ori.build_feature_matrix(historical_cases)
    log(f"  Training on {len(X_hist)} cases, {len(feature_names)} raw features")

    medians = []
    for j in range(X_hist.shape[1]):
        col = X_hist[:, j]
        valid = col[~np.isnan(col)]
        medians.append(float(np.median(valid)) if len(valid) > 0 else 0.0)
    X_hist_imp, _ = ori.train_median_impute(X_hist, X_hist[:1])
    X_hist_std, _ = ori.standardize(X_hist_imp, X_hist_imp[:1])

    pos = int(np.sum(y_hist == 1))
    neg = int(np.sum(y_hist == 0))
    cw_ratio = min(neg / max(pos, 1), 20.0)
    scale_pw = cw_ratio

    sel_idx, sel_names, _ = ori.select_features_by_logistic(
        X_hist_std, y_hist, feature_names,
        n_select=config.n_features_select,
        lam=config.feature_select_lam,
        n_epochs=config.feature_select_epochs,
        cw_ratio=cw_ratio,
    )
    log(f"  Selected {len(sel_idx)} features")

    X_hist_sel = X_hist_imp[:, sel_idx]
    mu = np.nanmean(X_hist_sel, axis=0)
    sd = np.nanstd(X_hist_sel, axis=0) + 1e-10
    X_hist_s, _ = ori.standardize(X_hist_sel, X_hist_sel[:1])

    log("  Training histogram GBT depth-3...")
    gbt3_trees, gbt3_init, gbt3_lr, _, _ = ori.train_histogram_gbt(
        X_hist_sel, y_hist,
        n_trees=config.hgbt_d3_n_trees, max_depth=config.hgbt_d3_max_depth,
        learning_rate=config.hgbt_d3_lr, lambda_reg=config.hgbt_d3_lambda_reg,
        scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )
    log("  Training histogram GBT depth-4...")
    gbt4_trees, gbt4_init, gbt4_lr, _, _ = ori.train_histogram_gbt(
        X_hist_sel, y_hist,
        n_trees=config.hgbt_d4_n_trees, max_depth=config.hgbt_d4_max_depth,
        learning_rate=config.hgbt_d4_lr, lambda_reg=config.hgbt_d4_lambda_reg,
        min_samples_leaf=config.hgbt_d4_min_samples_leaf,
        min_child_weight=config.hgbt_d4_min_child_weight,
        colsample=config.hgbt_d4_colsample, subsample=config.hgbt_d4_subsample,
        gamma=config.hgbt_d4_gamma, scale_pos_weight=scale_pw, n_bins=config.n_bins,
    )
    log("  Training logistic regression...")
    w_lr, b_lr = ori.train_logistic(
        X_hist_s, y_hist,
        lam=config.logistic_lam, lr=config.logistic_lr,
        n_epochs=config.logistic_epochs_final, cw_ratio=cw_ratio,
    )
    log(f"  Training bagged logistic ({config.bag_n_bags_final} bags)...")
    bags = ori.train_bagged_logistic(
        X_hist_s, y_hist,
        n_bags=config.bag_n_bags_final, feat_fraction=config.bag_feat_fraction,
        lam=config.bag_lam, lr=config.bag_lr,
        n_epochs=config.bag_epochs_final, cw_ratio=cw_ratio,
    )

    members = {
        "feature_names": list(feature_names),
        "impute_medians": medians,
        "selected_idx": [int(i) for i in sel_idx],
        "selected_features": list(sel_names),
        "standardize_mean": _floats(mu),
        "standardize_sd": _floats(sd),
        "class_weight_ratio": float(cw_ratio),
        "gbt_d3": {
            "init_score": float(gbt3_init),
            "learning_rate": float(gbt3_lr),
            "trees": [_tree_to_json(t) for t in gbt3_trees],
        },
        "gbt_d4": {
            "init_score": float(gbt4_init),
            "learning_rate": float(gbt4_lr),
            "trees": [_tree_to_json(t) for t in gbt4_trees],
        },
        "logistic": {"w": _floats(w_lr), "b": float(b_lr)},
        "bagged": [
            {"feat_idx": [int(i) for i in fi], "w": _floats(w), "b": float(b)}
            for fi, w, b in bags
        ],
        "training_summary": {"n_cases": int(len(y_hist)), "n_positive": pos, "n_negative": neg},
    }
    if not return_training_ensemble:
        return members
    p_hist_ens = (
        ori.predict_histogram_gbt(X_hist_sel, gbt3_trees, gbt3_init, gbt3_lr)
        + ori.predict_histogram_gbt(X_hist_sel, gbt4_trees, gbt4_init, gbt4_lr)
        + ori.predict_logistic(X_hist_s, w_lr, b_lr)
        + ori.predict_bagged(X_hist_s, bags)
    ) / 4.0
    return members, p_hist_ens, y_hist


def fit_operational_ri_model(
    historical_cases: list[dict],
    config: ori.OperationalRIConfig | None = None,
    *,
    log=print,
) -> dict[str, Any]:
    """The ORIGINAL v8.1 recipe (members + legacy in-sample Platt), bit for bit.

    Kept to reproduce the v8.1 pin for comparison and to back ``train_and_score``'s
    offline path. Its calibration is the defective one (see module docstring).
    """
    members, p_hist_ens, y_hist = fit_members(
        historical_cases, config, log=log, return_training_ensemble=True,
    )
    _, platt_a, platt_b = ori.platt_calibrate(y_hist, p_hist_ens, p_hist_ens[:1])
    train_auc, _, _ = ori.compute_roc_auc(y_hist, p_hist_ens)
    model = {
        "schema": SCHEMA,
        "model_version": "hurricane_ri_v8_1",
        **members,
        "calibration": {"method": "platt_prob_gd500_insample", "a": float(platt_a), "b": float(platt_b)},
        "serving": {"impute_live": [f for f in RECIPES["hurricane_ri_v8_1"]["impute_live"]
                                    if f in members["feature_names"]],
                    "reason": LIVE_SPEED_AGE_REASON},
    }
    model["training_summary"]["train_auc_ensemble_uncalibrated"] = float(train_auc)
    return model


# ---------------------------------------------------------------------------
# Calibration: Newton/IRLS to a tolerance, or refuse
# ---------------------------------------------------------------------------

def newton_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol_grad: float = 1e-10,
    max_iter: int = 50,
    max_abs_coef: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximum-likelihood logistic regression by damped Newton (IRLS) -- to convergence.

    Converged means the mean log-likelihood gradient's sup-norm is <= ``tol_grad``. Raises
    :class:`CalibrationError` if that is not reached in ``max_iter`` Newton steps, if the
    Hessian is singular, or if a coefficient runs away (``|coef| > max_abs_coef``: the data
    are (quasi-)separable and the MLE does not exist). Never returns an unconverged fit.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n == 0 or not np.all(np.isfinite(X)) or not np.all((y == 0) | (y == 1)):
        raise CalibrationError("logistic fit needs finite inputs and 0/1 labels")

    def loss(theta: np.ndarray) -> float:
        z = X @ theta
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def separable(theta: np.ndarray) -> bool:
        # Complete separation: the likelihood keeps rising as |theta| grows, so no MLE exists,
        # yet the gradient decays exponentially and would pass any tolerance. Refuse it.
        z = X @ theta
        pos, neg = z[y == 1], z[y == 0]
        return bool(len(pos) and len(neg) and (pos.min() > neg.max() or pos.max() < neg.min()))

    theta = np.zeros(k)
    current = loss(theta)
    for it in range(1, max_iter + 1):
        z = X @ theta
        p = ori.sigmoid(z)
        grad = X.T @ (p - y) / n
        g_inf = float(np.max(np.abs(grad)))
        if g_inf <= tol_grad:
            if separable(theta):
                raise CalibrationError("data are completely separable: no maximum-likelihood fit exists")
            return theta, {"iterations": it - 1, "grad_inf": g_inf, "mean_log_loss": current}
        w = p * (1.0 - p)
        hess = (X * w[:, None]).T @ X / n
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError(f"singular Hessian at iteration {it}: {exc}") from exc
        t = 1.0
        while True:  # Armijo backtracking: Newton is only guaranteed to descend near the optimum
            candidate = theta - t * step
            new = loss(candidate)
            if new <= current - 1e-4 * t * float(grad @ step) or t < 1e-12:
                break
            t *= 0.5
        theta, current = candidate, new
        if np.max(np.abs(theta)) > max_abs_coef:
            raise CalibrationError(
                f"coefficients diverging ({theta.tolist()}) at iteration {it}: separable data, no MLE"
            )
    z = X @ theta
    g_inf = float(np.max(np.abs(X.T @ (ori.sigmoid(z) - y) / n)))
    if g_inf <= tol_grad and not separable(theta):
        return theta, {"iterations": max_iter, "grad_inf": g_inf, "mean_log_loss": current}
    raise CalibrationError(
        f"logistic fit did not converge: |grad|_inf={g_inf:.3e} > {tol_grad:.1e} after {max_iter} Newton steps"
    )


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(q / (1.0 - q))


def fit_logit_calibrator(
    p_ensemble: np.ndarray,
    y: np.ndarray,
    *,
    min_events: int = 20,
    tol_grad: float = 1e-10,
    max_iter: int = 50,
) -> dict[str, Any]:
    """P(RI) = sigmoid(a * logit(p_ens) + b), fitted by :func:`newton_logistic` on HELD-OUT rows.

    The members are class-weighted (positives up to x20), which inflates their odds by a
    roughly constant factor -- a shift in log-odds space, which this two-parameter map
    removes. Refuses (raises) with fewer than ``min_events`` events or non-events, on
    non-convergence, or if the fitted map is not increasing (a <= 0).
    """
    y = np.asarray(y, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos < min_events or n_neg < min_events:
        raise CalibrationError(f"too few calibration events ({n_pos} RI, {n_neg} non-RI; need {min_events})")
    X = np.column_stack([_logit(p_ensemble), np.ones(len(y))])
    theta, diag = newton_logistic(X, y, tol_grad=tol_grad, max_iter=max_iter)
    a, b = float(theta[0]), float(theta[1])
    if not a > 0:
        raise CalibrationError(f"calibration map is not increasing (a={a}); the ensemble has no usable ranking")
    return {
        "method": "logistic_on_logit_newton",
        "a": a,
        "b": b,
        "logit_eps": LOGIT_EPS,
        "n": int(len(y)),
        "n_events": n_pos,
        "event_rate": float(np.mean(y)),
        "iterations": int(diag["iterations"]),
        "grad_inf": float(diag["grad_inf"]),
        "tol_grad": float(tol_grad),
        "mean_log_loss": float(diag["mean_log_loss"]),
    }


def apply_calibration(calibration: dict[str, Any], p_ensemble: np.ndarray) -> np.ndarray:
    method = calibration.get("method")
    if method == "logistic_on_logit_newton":
        return ori.sigmoid(calibration["a"] * _logit(p_ensemble) + calibration["b"])
    if method == "platt_prob_gd500_insample":  # legacy v8.1, bit-compatible
        return ori.sigmoid(calibration["a"] * p_ensemble + calibration["b"])
    raise ModelArtifactError(f"unknown calibration method {method!r}")


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def member_probabilities(model: dict[str, Any], cases: list[dict]) -> dict[str, np.ndarray]:
    """Member and uncalibrated-ensemble probabilities, under the model's serving contract.

    Features in ``model["serving"]["impute_live"]`` are replaced by the training median
    exactly as a live case lacking them would be, so offline evaluation scores the model
    as it is served.
    """
    names = list(model["feature_names"])
    X, _, _, _ = ori.build_feature_matrix(cases, feature_names=names)
    X = X.copy()
    for feat in model.get("serving", {}).get("impute_live", []):
        if feat in names:
            X[:, names.index(feat)] = np.nan
    for j, median_val in enumerate(model["impute_medians"]):
        nans = np.isnan(X[:, j])
        X[nans, j] = median_val
    sel = np.asarray(model["selected_idx"], dtype=np.int64)
    X_sel = X[:, sel]
    mu = np.asarray(model["standardize_mean"], dtype=float)
    sd = np.asarray(model["standardize_sd"], dtype=float)
    X_s = (X_sel - mu) / sd

    def gbt(member: dict) -> np.ndarray:
        return ori.predict_histogram_gbt(
            X_sel, member["trees"], member["init_score"], member["learning_rate"],
        )

    p_gbt3 = gbt(model["gbt_d3"])
    p_gbt4 = gbt(model["gbt_d4"])
    p_lr = ori.predict_logistic(
        X_s, np.asarray(model["logistic"]["w"], dtype=float), model["logistic"]["b"],
    )
    bags = [
        (np.asarray(b["feat_idx"], dtype=np.int64), np.asarray(b["w"], dtype=float), b["b"])
        for b in model["bagged"]
    ]
    p_bag = ori.predict_bagged(X_s, bags)
    p_ensemble = (p_gbt3 + p_gbt4 + p_lr + p_bag) / 4.0
    return {"gbt_d3": p_gbt3, "gbt_d4": p_gbt4, "logistic": p_lr, "bagged": p_bag,
            "ensemble": p_ensemble}


def score_cases(model: dict[str, Any], cases: list[dict]) -> dict[str, np.ndarray]:
    """Member, ensemble and calibrated probabilities for ``cases`` (one entry per case)."""
    out = member_probabilities(model, cases)
    out["calibrated"] = apply_calibration(model["calibration"], out["ensemble"])
    return out


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> list[dict]:
    return ori.load_operational_ri_cases(DATASETS[name])


def fit_recipe(
    version: str,
    *,
    datasets: dict[str, list[dict]] | None = None,
    config: ori.OperationalRIConfig | None = None,
    log=print,
) -> dict[str, Any]:
    """Fit a recipe from ``RECIPES`` end to end (members, then held-out calibration)."""
    recipe = RECIPES[version]
    datasets = dict(datasets or {})

    def rows(name: str) -> list[dict]:
        if name not in datasets:
            datasets[name] = load_dataset(name)
        return datasets[name]

    if recipe["calibration"]["method"] == "platt_prob_gd500_insample":
        model = fit_operational_ri_model(select_years(rows(recipe["dataset"]), recipe["members_years"]),
                                         config, log=log)
        model["model_version"] = version
        return model

    members = fit_members(select_years(rows(recipe["dataset"]), recipe["members_years"]), config, log=log)
    model = {"schema": SCHEMA, "model_version": version, **members}
    model["serving"] = {"impute_live": [f for f in recipe["impute_live"] if f in members["feature_names"]],
                        "reason": LIVE_SPEED_AGE_REASON if recipe["impute_live"] else
                        "every feature is computed live with the training set's own definition"}
    cal_spec = recipe["calibration"]
    cal_rows = select_years(rows(cal_spec["dataset"]), cal_spec["years"])
    p_cal = member_probabilities(model, cal_rows)["ensemble"]
    y_cal = np.array([float(c["ri_label_30kt"]) for c in cal_rows])
    calibration = fit_logit_calibrator(p_cal, y_cal)
    calibration["fitted_on"] = {"dataset": cal_spec["dataset"], "storm_years": list(cal_spec["years"])}
    model["calibration"] = calibration
    log(f"  Calibrated on {calibration['n']} held-out rows ({calibration['n_events']} RI): "
        f"a={calibration['a']:.4f} b={calibration['b']:.4f} in {calibration['iterations']} Newton steps, "
        f"|grad|={calibration['grad_inf']:.1e}")
    return model


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------

def attach_provenance(
    model: dict[str, Any],
    *,
    data: dict[str, dict[str, Any]] | None = None,
    training_data: str | Path | None = None,
    config: ori.OperationalRIConfig | None = None,
    train_seconds: float | None = None,
) -> dict[str, Any]:
    """Bind the artifact to its data files (by content hash), config and trainer source.

    ``data`` maps a role ("members", "calibration") to ``{"path": ..., "storm_years": ...}``;
    ``training_data`` is shorthand for a members-only binding.
    """
    data = dict(data or {})
    if training_data is not None:
        data.setdefault("members", {"path": training_data})
    bound = {}
    for role, spec in data.items():
        path = Path(spec["path"])
        bound[role] = {"path": _rel(path), "sha256": sha256_file(path),
                       **({"storm_years": list(spec["storm_years"])} if spec.get("storm_years") else {})}
    model["provenance"] = {
        "data": bound,
        "config": config_dict(config),
        "trainer_fingerprint": trainer_fingerprint(),
        "trained_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_seconds": None if train_seconds is None else round(float(train_seconds), 1),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    return model


def recipe_data_binding(version: str) -> dict[str, dict[str, Any]]:
    recipe = RECIPES[version]
    binding = {"members": {"path": DATASETS[recipe["dataset"]], "storm_years": recipe["members_years"]}}
    cal = recipe["calibration"]
    if "dataset" in cal:
        binding["calibration"] = {"path": DATASETS[cal["dataset"]], "storm_years": cal["years"]}
    return binding


def save_model(model: dict[str, Any], path: str | Path = DEFAULT_ARTIFACT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN weight is a training failure, never something to serve.
    text = json.dumps(model, indent=1, sort_keys=False, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as fh:  # same bytes on every OS
        fh.write(text)
    return path


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ModelArtifactError(message)


def validate_structure(model: dict[str, Any]) -> None:
    """Internal consistency: shapes, indices, calibration and tree references are in range."""
    _require(model.get("schema") == SCHEMA, f"unknown schema {model.get('schema')!r}")
    _require(model.get("model_version") in KNOWN_VERSIONS,
             f"unknown model_version {model.get('model_version')!r}")
    names = model.get("feature_names") or []
    d = len(names)
    _require(d > 0 and len(set(names)) == d, "feature_names empty or duplicated")
    _require(len(model.get("impute_medians", [])) == d, "impute_medians length != n_features")
    sel = model.get("selected_idx") or []
    _require(len(sel) > 0 and all(0 <= int(i) < d for i in sel), "selected_idx out of range")
    k = len(sel)
    _require(len(model.get("standardize_mean", [])) == k, "standardize_mean length != n_selected")
    _require(len(model.get("standardize_sd", [])) == k, "standardize_sd length != n_selected")
    _require(all(float(s) > 0 for s in model["standardize_sd"]), "non-positive standardize_sd")
    _require(len(model["logistic"]["w"]) == k, "logistic weight length != n_selected")
    for feat in model.get("serving", {}).get("impute_live", []):
        _require(feat in names, f"serving.impute_live names unknown feature {feat!r}")

    cal = model.get("calibration") or {}
    _require(cal.get("method") in ("logistic_on_logit_newton", "platt_prob_gd500_insample"),
             f"unknown calibration method {cal.get('method')!r}")
    _require(np.isfinite(cal.get("a", np.nan)) and np.isfinite(cal.get("b", np.nan)),
             "non-finite calibration coefficients")
    if cal["method"] == "logistic_on_logit_newton":
        _require(cal["a"] > 0, "calibration map not increasing")
        _require(cal.get("grad_inf", np.inf) <= cal.get("tol_grad", 0.0),
                 "calibration was not fitted to its convergence tolerance")

    def check_tree(node: dict, depth: int, max_depth: int) -> None:
        if "leaf" in node:
            _require(np.isfinite(node["leaf"]), "non-finite leaf")
            return
        _require(depth < max_depth, "tree deeper than its declared max_depth")
        _require(0 <= int(node["feature"]) < k, "tree split feature out of range")
        _require(np.isfinite(node["threshold"]), "non-finite threshold")
        check_tree(node["left"], depth + 1, max_depth)
        check_tree(node["right"], depth + 1, max_depth)

    cfg = model.get("provenance", {}).get("config", {})
    for member, depth_key in (("gbt_d3", "hgbt_d3_max_depth"), ("gbt_d4", "hgbt_d4_max_depth")):
        trees = model[member]["trees"]
        _require(len(trees) > 0, f"{member} has no trees")
        for tree in trees:
            check_tree(tree, 0, int(cfg.get(depth_key, 64)))
    _require(len(model["bagged"]) > 0, "no bagged members")
    for bag in model["bagged"]:
        _require(len(bag["feat_idx"]) == len(bag["w"]), "bag feat_idx/w length mismatch")
        _require(all(0 <= int(i) < k for i in bag["feat_idx"]), "bag feature out of range")


def load_model(
    path: str | Path = DEFAULT_ARTIFACT,
    *,
    verify_data: bool = True,
    config: ori.OperationalRIConfig | None = None,
) -> dict[str, Any]:
    """Load an artifact and refuse it unless it is bound to the current inputs.

    Checks: schema and structure (including that a Newton calibration reached its
    tolerance); the config equals the one it was trained with; with ``verify_data`` every
    data file it was fitted on exists and has the content hash recorded at training time.
    """
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(
            f"hurricane RI model artifact missing at {path}. Train it with "
            "`python scripts/train_hurricane_ri.py --recipe <version>` and commit it with `git add -f`."
        )
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"unreadable model artifact {path}: {exc}") from exc
    validate_structure(model)
    prov = model.get("provenance") or {}
    _require(prov.get("config") == config_dict(config),
             "model artifact was trained with a different config; retrain "
             "(python scripts/train_hurricane_ri.py)")
    if verify_data:
        bound = prov.get("data") or {}
        _require("members" in bound, "model artifact records no members training data")
        for role, spec in bound.items():
            data_path = _resolve(spec["path"])
            _require(data_path.exists(),
                     f"{role} data {data_path} missing; cannot verify the artifact's provenance")
            have = sha256_file(data_path)
            _require(spec.get("sha256") == have,
                     f"model artifact is stale: its {role} data was sha256 {spec.get('sha256')}, "
                     f"{data_path.name} is now {have}; retrain (python scripts/train_hurricane_ri.py)")
    return model


def model_parameters(model: dict[str, Any]) -> dict[str, Any]:
    """The fitted parameters alone (no provenance) -- what a reproducibility check compares."""
    return {k: v for k, v in model.items() if k != "provenance"}
