#!/usr/bin/env python3
"""Compute same-location AUC to isolate TEMPORAL prediction skill.

The global AUC includes both spatial and temporal discrimination.
This script groups test samples by their source location (lat/lon),
computes AUC within each group, and macro-averages.

Within a single location group, all samples share the same (lat, lon),
so spatial features like plate_boundary_dist are constant.
Any discrimination must come from TEMPORAL features.

This is the definitive test of whether the model predicts WHEN
earthquakes happen, not just WHERE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via trapezoidal integration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    auc = 0.0
    tpr_prev = fpr_prev = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr
    return float(auc)


def main():
    print("=" * 70)
    print("SAME-LOCATION AUC: Isolating Temporal Prediction Skill")
    print("=" * 70)
    print()

    # Load the honest results to get model predictions
    # We need to re-run predictions OR load from checkpoints + retrain
    # Actually, the simplest approach: load the checkpointed features,
    # train the model, predict on test, and group by location.

    # But we don't have location metadata in the checkpoints (only X, y).
    # We need to rebuild samples to get lat/lon per sample.

    # Let's use the full model pipeline but add location tracking.

    from hazardpulse.earthquake.v4_regional import (
        REGIONS,
        HONEST_CONTROL_RATIO,
        HONEST_FORWARD_WINDOW_DAYS,
        HONEST_LABEL_RADIUS_KM,
        HONEST_MAGNITUDE_THRESHOLD,
        CatalogArrays,
        GNSSIndex,
        GradientBoostedTrees,
        FeatureNormalizer,
        RegionalCatalogSlice,
        build_samples_for_region,
        compute_auc as model_auc,
        decluster_gardner_knopoff,
        extract_all_features,
        haversine_km,
    )
    from hazardpulse.data.earthquake import load_usgs_catalog

    # Set honest mode
    import hazardpulse.earthquake.v4_regional as v4
    v4._HONEST_MODE = True

    TRAIN_END = 2017
    TEST_START = 2020
    TEST_END = 2024

    # Load data
    print("[1] Loading USGS catalog...")
    full_catalog = load_usgs_catalog(min_year=2000, max_year=2024, min_mag=2.5)
    print(f"    Loaded {len(full_catalog)} events")

    print("[2] Aftershock declustering...")
    mainshocks, _ = decluster_gardner_knopoff(full_catalog)
    print(f"    {len(mainshocks)} mainshocks")

    print("[3] Building catalog arrays...")
    global_cat = CatalogArrays(full_catalog)

    print("[4] Loading GNSS...")
    gnss = GNSSIndex()

    # For rest_of_world (the dominant region with adequate test data),
    # build samples and track locations using the same public code path as the
    # honest evaluation.
    print()
    print("[5] Building rest_of_world samples with location tracking...")
    samples, params = build_samples_for_region(mainshocks, "rest_of_world", verbose=True)

    # Split into test only
    test_samples = [s for s in samples if TEST_START <= s["year"] <= TEST_END]
    print(f"    Test samples: {len(test_samples)}")

    # Extract features and track locations
    print()
    print("[6] Extracting features for test samples...")
    regional_cat = global_cat

    N_FEAT = 55
    X_test = np.zeros((len(test_samples), N_FEAT), dtype=np.float32)
    y_test = np.zeros(len(test_samples), dtype=np.float32)
    locs = []  # (lat, lon) for each sample
    valid = np.ones(len(test_samples), dtype=bool)

    for idx, sample in enumerate(test_samples):
        if (idx + 1) % 100 == 0:
            print(f"      {idx + 1}/{len(test_samples)}")
            sys.stdout.flush()

        feats = extract_all_features(
            sample["latitude"],
            sample["longitude"],
            sample["ref_epoch"],
            "rest_of_world",
            regional_cat,
            global_cat,
            gnss,
        )

        if feats is None:
            valid[idx] = False
        else:
            X_test[idx] = feats

        y_test[idx] = sample["label"]
        locs.append((round(sample["latitude"], 4), round(sample["longitude"], 4)))

    X_test = X_test[valid]
    y_test = y_test[valid]
    locs = [locs[i] for i in range(len(valid)) if valid[i]]

    print(f"    Valid test: {len(y_test)} ({int(y_test.sum())} pos, {int(len(y_test) - y_test.sum())} neg)")

    # Also need training data to train the model
    print()
    print("[7] Loading training data from checkpoint...")
    ckpt_file = Path("results/earthquake_definitive/_checkpoints_honest/rest_of_world_features.npz")
    if not ckpt_file.exists():
        print("ERROR: No checkpoint found. Run --honest first.")
        return

    ckpt = np.load(ckpt_file)
    X_train = ckpt["X_train"]
    y_train = ckpt["y_train"]
    X_val = ckpt["X_val"]
    y_val = ckpt["y_val"]

    # Impute NaN
    train_mean = np.nanmean(X_train, axis=0)
    train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)
    for X in [X_train, X_val, X_test]:
        nan_mask = np.isnan(X)
        for col in range(X.shape[1]):
            col_nans = nan_mask[:, col]
            if col_nans.any():
                X[col_nans, col] = train_mean[col]

    # Normalize
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)
    X_train_n = normalizer.transform(X_train)
    X_val_n = normalizer.transform(X_val)
    X_test_n = normalizer.transform(X_test)

    # Train GBT
    print()
    print("[8] Training GBT with the same hyperparameters as the honest pipeline...")
    gbt = GradientBoostedTrees()
    gbt.fit(X_train_n, y_train, X_val=X_val_n, y_val=y_val, verbose=True)

    # Predict on test
    p_test = gbt.predict_proba(X_test_n)

    # Global AUC (sanity check)
    global_auc = compute_auc(y_test, p_test)
    print(f"\n    Global AUC (rest_of_world test): {global_auc:.4f}")

    # === SAME-LOCATION AUC ===
    print()
    print("=" * 70)
    print("SAME-LOCATION AUC (Temporal Skill Only)")
    print("=" * 70)

    # Group by location
    from collections import defaultdict
    loc_groups = defaultdict(list)
    for i, (lat, lon) in enumerate(locs):
        loc_groups[(lat, lon)].append(i)

    print(f"  Unique locations: {len(loc_groups)}")

    # Compute AUC per location (only where there's at least 1 positive and 1 negative)
    loc_aucs = []
    n_skipped = 0
    n_pos_only = 0
    n_neg_only = 0

    for (lat, lon), indices in sorted(loc_groups.items()):
        y_loc = y_test[indices]
        p_loc = p_test[indices]
        n_pos = int(y_loc.sum())
        n_neg = len(y_loc) - n_pos

        if n_pos == 0:
            n_neg_only += 1
            continue
        if n_neg == 0:
            n_pos_only += 1
            continue

        auc_loc = compute_auc(y_loc, p_loc)
        if not np.isnan(auc_loc):
            loc_aucs.append({
                "lat": lat, "lon": lon,
                "auc": auc_loc,
                "n_pos": n_pos, "n_neg": n_neg,
                "n_total": len(indices),
            })
        else:
            n_skipped += 1

    print(f"  Locations with both pos+neg: {len(loc_aucs)}")
    print(f"  Locations positive-only: {n_pos_only}")
    print(f"  Locations negative-only: {n_neg_only}")
    print(f"  Locations skipped (NaN): {n_skipped}")

    if loc_aucs:
        # Macro-average (unweighted)
        macro_auc = np.mean([l["auc"] for l in loc_aucs])

        # Weighted average (by number of samples)
        weights = np.array([l["n_total"] for l in loc_aucs], dtype=np.float64)
        weighted_auc = np.average([l["auc"] for l in loc_aucs], weights=weights)

        # Median
        median_auc = np.median([l["auc"] for l in loc_aucs])

        print()
        print(f"  +-------------------------------------------+")
        print(f"  | SAME-LOCATION AUC (macro-average): {macro_auc:.4f} |")
        print(f"  | SAME-LOCATION AUC (weighted):      {weighted_auc:.4f} |")
        print(f"  | SAME-LOCATION AUC (median):         {median_auc:.4f} |")
        print(f"  | Global AUC (for comparison):        {global_auc:.4f} |")
        print(f"  +-------------------------------------------+")

        print()
        print("  Interpretation:")
        if macro_auc > 0.80:
            print("  >>> STRONG temporal prediction skill confirmed.")
            print("  >>> The model predicts WHEN earthquakes happen, not just WHERE.")
        elif macro_auc > 0.70:
            print("  >>> MODERATE temporal prediction skill.")
            print("  >>> Some temporal signal exists beyond spatial knowledge.")
        elif macro_auc > 0.55:
            print("  >>> WEAK temporal prediction skill.")
            print("  >>> Most of the global AUC was from spatial discrimination.")
        else:
            print("  >>> NO temporal prediction skill.")
            print("  >>> The global AUC was entirely from knowing WHERE, not WHEN.")

        # Distribution
        print()
        print("  Per-location AUC distribution:")
        aucs_arr = np.array([l["auc"] for l in loc_aucs])
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {np.percentile(aucs_arr, pct):.4f}")

        # Top and bottom locations
        sorted_locs = sorted(loc_aucs, key=lambda x: x["auc"], reverse=True)
        print()
        print("  Top 5 locations (best temporal prediction):")
        for l in sorted_locs[:5]:
            print(f"    ({l['lat']:.2f}, {l['lon']:.2f}): AUC={l['auc']:.4f} ({l['n_pos']}pos/{l['n_neg']}neg)")

        print()
        print("  Bottom 5 locations (worst temporal prediction):")
        for l in sorted_locs[-5:]:
            print(f"    ({l['lat']:.2f}, {l['lon']:.2f}): AUC={l['auc']:.4f} ({l['n_pos']}pos/{l['n_neg']}neg)")

        # Save results
        results = {
            "test": "same_location_auc",
            "description": "AUC computed within same-location groups to isolate temporal skill",
            "global_auc": global_auc,
            "same_location_macro_auc": macro_auc,
            "same_location_weighted_auc": weighted_auc,
            "same_location_median_auc": median_auc,
            "n_locations_evaluated": len(loc_aucs),
            "n_locations_pos_only": n_pos_only,
            "n_locations_neg_only": n_neg_only,
            "distribution": {
                "p10": float(np.percentile(aucs_arr, 10)),
                "p25": float(np.percentile(aucs_arr, 25)),
                "p50": float(np.percentile(aucs_arr, 50)),
                "p75": float(np.percentile(aucs_arr, 75)),
                "p90": float(np.percentile(aucs_arr, 90)),
            },
            "per_location": sorted_locs,
        }

        out_path = Path("results/earthquake_honest/same_location_auc.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to: {out_path}")
    else:
        print("  ERROR: No locations with both positive and negative samples.")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
