#!/usr/bin/env python3
"""Earthquake ablation: baseline (Block S + C) vs augmented (S + C + W).

Trains two GBT models on the same M6+ mainshock corpus and compares
out-of-sample AUC, Brier, and BSS. Block W (12 space-weather features)
is the only difference.

Pipeline:
  1. Load full USGS catalog from local .cache/earthquake/.
  2. Decluster (Gardner-Knopoff) and identify M6+ targets.
  3. Build positive/negative samples.
  4. Extract Block S, Block C, Block W per sample.
  5. Temporal split (train: 2005-2017, val: 2018-2019, test: 2020-2024).
  6. Train two GBT models with identical hyperparameters.
  7. Score test set, write per-tier AUC/Brier/BSS to results/.

Run:
    python scripts/train_earthquake_with_spaceweather.py
        [--max-samples N] [--output PATH] [--quick]

--quick uses a 250-sample subset and 50 trees (smoke test, ~5 min).
The full run takes 30-60 min depending on machine.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.earthquake import load_usgs_catalog  # noqa: E402
from hazardpulse.earthquake.definitive_model import (  # noqa: E402
    ALL_FEATURE_NAMES_ENHANCED,
    BLOCK_C_NAMES,
    BLOCK_S_NAMES,
    CatalogArrays,
    FeatureNormalizer,
    GradientBoostedTrees,
    compute_auc,
    compute_block_c,
    compute_block_s,
    compute_brier,
    compute_bss,
    bootstrap_auc_ci,
    build_samples,
    decluster_gardner_knopoff,
)
from hazardpulse.earthquake.space_weather_block import (  # noqa: E402
    BLOCK_W_NAMES,
    N_FEAT_W,
    compute_block_w_for_event,
)


def _epoch_to_dt(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(tzinfo=None)


def build_full_feature_matrix(
    samples: list[dict],
    full_catalog: list[dict],
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Block S, Block C, Block W per sample.

    Returns (X_baseline, X_w, y, years, ref_epochs):
      - X_baseline: (n, 73) float32  — Block S + Block C (the plus_cft variant)
      - X_w:        (n, 85) float32  — concat with Block W
      - y:          (n,) float32     — labels
      - years:      (n,) int32       — sample year (for temporal split)
      - ref_epochs: (n,) float64
    """
    if verbose:
        print(f"  Building feature matrix for {len(samples)} samples...")
        sys.stdout.flush()

    cat = CatalogArrays(full_catalog, verbose=verbose)
    n = len(samples)
    n_s = len(BLOCK_S_NAMES)
    n_c = len(BLOCK_C_NAMES)
    n_baseline = n_s + n_c

    X_baseline = np.full((n, n_baseline), np.nan, dtype=np.float32)
    X_w = np.full((n, n_baseline + N_FEAT_W), np.nan, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    years = np.zeros(n, dtype=np.int32)
    ref_epochs = np.zeros(n, dtype=np.float64)

    t_block = 0.0
    valid_mask = np.zeros(n, dtype=bool)

    for i, s in enumerate(samples):
        if verbose and i and i % 100 == 0:
            elapsed = time.time() - t_block if t_block else 0
            print(f"    sample {i}/{n} ({elapsed:.0f}s)...")
            sys.stdout.flush()
        if i == 0:
            t_block = time.time()

        ref_epoch = float(s["ref_epoch"])
        lat = float(s["latitude"])
        lon = float(s["longitude"])

        block_s = compute_block_s(lat, lon, ref_epoch, cat)
        if block_s is None:
            continue
        block_c = compute_block_c(full_catalog, lat, lon, ref_epoch)
        block_w = compute_block_w_for_event(_epoch_to_dt(ref_epoch))

        X_baseline[i, :n_s] = block_s
        X_baseline[i, n_s:] = block_c
        X_w[i, :n_s] = block_s
        X_w[i, n_s:n_baseline] = block_c
        X_w[i, n_baseline:] = block_w
        y[i] = float(s["label"])
        years[i] = int(s.get("year", 0))
        ref_epochs[i] = ref_epoch
        valid_mask[i] = True

    if verbose:
        print(f"  Valid samples: {int(valid_mask.sum())}/{n}")
        sys.stdout.flush()

    return (
        X_baseline[valid_mask],
        X_w[valid_mask],
        y[valid_mask],
        years[valid_mask],
        ref_epochs[valid_mask],
    )


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    *,
    train_end: int = 2017,
    val_end: int = 2019,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    train_mask = years <= train_end
    val_mask = (years > train_end) & (years <= val_end)
    test_mask = years > val_end
    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


def train_and_score(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    n_trees: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    label: str = "model",
    verbose: bool = True,
) -> dict:
    """Train one GBT, return test metrics."""
    norm = FeatureNormalizer()
    norm.fit(X_train)
    Xtr_n = norm.transform(X_train)
    Xv_n = norm.transform(X_val) if len(X_val) else None
    Xte_n = norm.transform(X_test)

    gbt = GradientBoostedTrees(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_reg=1.0,
        subsample=0.8,
        min_samples_leaf=10,
    )
    if verbose:
        print(f"  [{label}] Training GBT (n_trees={n_trees}, max_depth={max_depth})...")
        sys.stdout.flush()
    t0 = time.time()
    gbt.fit(Xtr_n, y_train, X_val=Xv_n, y_val=y_val if len(y_val) else None,
            verbose=verbose)
    train_time = time.time() - t0

    p_test = gbt.predict_proba(Xte_n)
    p_train = gbt.predict_proba(Xtr_n)

    auc_test = compute_auc(y_test, p_test)
    brier_test = compute_brier(y_test, p_test)
    bss_test = compute_bss(y_test, p_test)
    auc_train = compute_auc(y_train, p_train)

    ci = bootstrap_auc_ci(y_test, p_test, n_boot=200, seed=42)
    auc_lo, auc_hi = ci["ci_lo"], ci["ci_hi"]

    return {
        "label": label,
        "n_features": int(X_train.shape[1]),
        "n_trees": n_trees,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_seconds": round(train_time, 1),
        "auc_train": round(float(auc_train), 4),
        "auc_test": round(float(auc_test), 4),
        "auc_test_ci_lo": round(float(auc_lo), 4),
        "auc_test_ci_hi": round(float(auc_hi), 4),
        "brier_test": round(float(brier_test), 4),
        "bss_test": round(float(bss_test), 4),
        "test_pos_rate": round(float(y_test.mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 250 samples, 50 trees")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total sample count (debugging)")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "space_weather"
                                / "earthquake_block_w_ablation.json")
    parser.add_argument("--n-trees", type=int, default=None)
    args = parser.parse_args(argv)

    print("Earthquake space-weather ablation")
    print(f"  baseline features: {len(BLOCK_S_NAMES) + len(BLOCK_C_NAMES)} (S + C)")
    print(f"  augmented features: {len(BLOCK_S_NAMES) + len(BLOCK_C_NAMES) + N_FEAT_W} (S + C + W)")
    print(f"  Block W:  {BLOCK_W_NAMES}")
    print()

    n_trees = args.n_trees if args.n_trees is not None else (50 if args.quick else 200)
    max_samples = args.max_samples or (250 if args.quick else None)

    # Step 1: Load catalog
    print("Step 1: Loading USGS catalog...")
    catalog = load_usgs_catalog(min_year=2005, max_year=2025, min_mag=2.5)
    print(f"  Loaded {len(catalog)} M2.5+ events")
    if len(catalog) < 1000:
        print("  ERROR: Catalog too sparse — run scripts/download_earthquake_data.py first.")
        return 1

    # Step 2: Decluster + identify mainshocks
    print()
    print("Step 2: Declustering...")
    mainshocks, _aftershocks = decluster_gardner_knopoff(catalog)
    # Restrict to M6+ for mainshock-target ID
    mainshocks_m6 = [m for m in mainshocks if (m.get("mag") or 0) >= 6.0]
    print(f"  Mainshocks (declustered): {len(mainshocks)}, of which M6+: {len(mainshocks_m6)}")

    # Step 3: Build samples
    print()
    print("Step 3: Building positive/negative samples...")
    samples = build_samples(mainshocks_m6, mainshocks, verbose=True)
    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(samples)}")

    # Step 4: Extract features
    print()
    print("Step 4: Extracting Block S + C + W per sample...")
    X_b, X_w, y, years, _ = build_full_feature_matrix(samples, catalog, verbose=True)
    print(f"  Final corpus: {len(y)} samples, baseline {X_b.shape[1]} features, augmented {X_w.shape[1]}")
    print(f"  Class balance: {int(y.sum())} positive ({y.mean():.1%})")

    # Step 5: Temporal split
    print()
    print("Step 5: Temporal split (train<=2017, val 2018-2019, test 2020+)...")
    Xtr_b, ytr, Xv_b, yv, Xte_b, yte = temporal_split(X_b, y, years)
    Xtr_w, _, Xv_w, _, Xte_w, _ = temporal_split(X_w, y, years)
    print(f"  train={len(ytr)}  val={len(yv)}  test={len(yte)}")
    if len(yte) < 30:
        print("  WARNING: Test set is small — bootstrap CI will be wide.")

    # Step 6: Train both
    print()
    print("Step 6: Training baseline (S+C, 73 features)...")
    res_baseline = train_and_score(
        Xtr_b, ytr, Xv_b, yv, Xte_b, yte,
        n_trees=n_trees, label="baseline_s_c", verbose=True,
    )

    print()
    print("Step 6b: Training augmented (S+C+W, 85 features)...")
    res_w = train_and_score(
        Xtr_w, ytr, Xv_w, yv, Xte_w, yte,
        n_trees=n_trees, label="augmented_s_c_w", verbose=True,
    )

    # Step 7: Report
    print()
    print("=" * 72)
    print("ABLATION RESULTS")
    print("=" * 72)
    for r in (res_baseline, res_w):
        print(f"  {r['label']:25s}  AUC={r['auc_test']}  "
              f"[{r['auc_test_ci_lo']}, {r['auc_test_ci_hi']}]  "
              f"Brier={r['brier_test']}  BSS={r['bss_test']}")
    delta_auc = res_w["auc_test"] - res_baseline["auc_test"]
    delta_brier = res_w["brier_test"] - res_baseline["brier_test"]
    delta_bss = res_w["bss_test"] - res_baseline["bss_test"]
    print()
    print(f"  Delta AUC (W - baseline):  {delta_auc:+.4f}")
    print(f"  Delta Brier (W - base):    {delta_brier:+.4f}  (lower is better)")
    print(f"  Delta BSS (W - baseline):  {delta_bss:+.4f}")
    if delta_auc > 0.01:
        print("  -> Block W shows meaningful AUC lift; consider promoting.")
    elif delta_auc < -0.01:
        print("  -> Block W HURTS AUC; do NOT promote (likely noise overfitting).")
    else:
        print("  -> Inconclusive within bootstrap CI; keep heuristic baseline.")

    # Step 8: Persist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "n_total_samples": int(len(y)),
        "block_w_features": BLOCK_W_NAMES,
        "results": [res_baseline, res_w],
        "delta": {
            "auc": delta_auc,
            "brier": delta_brier,
            "bss": delta_bss,
        },
        "promote_w": delta_auc > 0.01,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
