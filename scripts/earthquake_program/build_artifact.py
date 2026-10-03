#!/usr/bin/env python3
"""Program step 5: build the served artifact (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 8).

The candidate recorded in ``results/earthquake_program/choose.json`` with its FIT
parameters -- the object the evaluation scored, no refit -- plus the frozen M>=5 catalog
1973-01-01 .. cutoff. For a boosted-tree choice (C0) the artifact also carries A's long-term
parameters, B's rate model (its maps are tree inputs) and the trees as a LightGBM payload
(``hazardpulse.tornado.lgbm_payload``, scored in NumPy, LightGBM's split semantics).

Parity, recorded in the artifact and enforced before it is accepted:

1. rates   -- at three issue times (one per CHOOSE/DEV/FINAL) the frozen-catalog rate model
              reproduces the evaluation's own float64 grid bit for bit (SHA-256);
2. trees   -- for sampled cells, the stored float32 feature rows give back the evaluated
              probabilities through the NumPy trees (max abs difference recorded; CI re-checks);
3. live    -- (build time only; needs the program cache) the full live code path
              ``forecast_from_artifact`` -- frozen catalog + a simulated live fetch -- gives
              back the evaluated grid at the same issue times.

The file is removed and the build fails if any check fails.

    python scripts/earthquake_program/build_artifact.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from hazardpulse.earthquake import operational_features as feat  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

OUT = C.REPO / "results" / "models" / "earthquake_operational_v1.json"
CUTOFF = "2026-10-01T00:00:00Z"
LIVE_SPAN_DAYS = 1827.0
TREE_TOL = 1e-12          # NumPy trees vs LightGBM's C++ on the same float32 rows


def _sha(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype=np.float64).tobytes()).hexdigest()


def _stamp(t: float) -> str:
    return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _simulated_live(raw25: dict, t: float) -> list[dict]:
    """What the live scorer's fetch would hand over at t: every M2.5+ event of the
    LIVE_SPAN_DAYS before t, with depth (from the audited M2.5 files)."""
    lo = np.searchsorted(raw25["t"], t - LIVE_SPAN_DAYS * C.SEC_DAY, side="left")
    hi = np.searchsorted(raw25["t"], t, side="left")
    return [{"time": float(raw25["t"][i]), "latitude": float(raw25["lat"][i]), "longitude": float(raw25["lon"][i]),
             "mag": float(raw25["mag"][i]), "depth": float(raw25["depth"][i])} for i in range(lo, hi)]


def main() -> int:
    choose = json.loads((C.RESULTS / "choose.json").read_text(encoding="utf-8"))
    chosen = choose["decision"]["chosen"]
    fit = json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))
    if chosen in ("A", "B"):
        spec = of.ModelSpec.from_dict(fit[f"{chosen}_chosen"])
        gbt = None
    elif chosen == "C0":
        import lightgbm as lgb
        from hazardpulse.tornado.lgbm_payload import export_booster

        spec = of.ModelSpec.from_dict(fit["B_chosen"])
        booster = lgb.Booster(model_file=str(C.PROGRAM_CACHE / "C0_lightgbm.txt"))
        fit_c = json.loads((C.RESULTS / "fit_c.json").read_text(encoding="utf-8"))
        payload = export_booster(
            booster, list(feat.CORE_FEATURES),
            calibration={"a": 1.0, "b": 0.0,
                         "note": "identity: P = 1/(1+exp(-raw)), LightGBM's binary transform; no recalibration"},
            provenance={"trainer": "scripts/earthquake_program/fit_c.py", "variant": "C0",
                        "params": fit_c["params"], "rounds": fit_c["variants"]["C0"]["best_iteration"]})
        if payload["n_trees"] != fit_c["variants"]["C0"]["best_iteration"]:
            raise RuntimeError(f"payload has {payload['n_trees']} trees, the evaluated model "
                               f"{fit_c['variants']['C0']['best_iteration']}")
        gbt = {"name": "C0", "long_A": fit["A_chosen"]["long"], "feature_names": list(feat.CORE_FEATURES),
               "feature_dtype": "float32", "live_span_days": LIVE_SPAN_DAYS, "payload": payload}
    else:
        raise SystemExit(f"chosen candidate {chosen} has no serving path")
    cutoff = of._parse_time(CUTOFF)
    events = C.load_events(4.5)
    frozen = events.select(events.mag >= of.INPUT_MIN_MAG).before(cutoff)

    rate_name = "B" if gbt is not None else chosen
    picks = []
    for split, pos in (("choose", 0), ("dev", None), ("final", -1)):
        issue = C.split_issue_times(split)
        picks.append((split, issue.size // 2 if pos is None else pos, issue))

    parity = []
    tree_rows = []
    if gbt is not None:
        raw45 = C.load_raw("program_catalog.npz")
        raw25 = C.load_raw("program_catalog_m25.npz")
        c45 = feat.CellCatalog(raw45["t"], raw45["lat"], raw45["lon"], raw45["mag"], raw45["depth"])
        c25 = feat.CellCatalog(raw25["t"], raw25["lat"], raw25["lon"], raw25["mag"])
        all_issue = C.all_issue_times()
        lam_a = np.load(C.PROGRAM_CACHE / "maps_lambda_A.npy")
        lam_s = np.load(C.PROGRAM_CACHE / "maps_lambda_short_B.npy")
    for split, i, issue in picks:
        P_rate = C.load_pred(rate_name, split)[i]
        rec = {"split": split, "issue_time": _stamp(issue[i]), "rate_model": rate_name,
               "rate_sha256_float64": _sha(P_rate)}
        if gbt is not None:
            P = C.load_pred("C0", split)[i]
            k = int(np.nonzero(all_issue == issue[i])[0][0])
            lam_b = -np.log1p(-P_rate)
            X = feat.core_features(issue[i], c45, c25, lam_a[k].astype(np.float64), lam_b,
                                   lam_s[k].astype(np.float64)).astype(np.float32)
            rng = np.random.default_rng(int(issue[i]))
            cells = np.unique(np.concatenate([np.argsort(-P, kind="stable")[:24],
                                              np.nonzero(X[:, feat.CORE_FEATURES.index("active")] == 1)[0][:20],
                                              rng.choice(of.N_CELLS, 20, replace=False)]))
            got = of.gbt_probability(gbt["payload"], X[cells])
            tree_rows.append(float(np.max(np.abs(got - P[cells]))))
            rec["sha256_float64"] = _sha(P)
            rec["tree_rows"] = {"cells": [int(c) for c in cells],
                                # null = missing (NaN), which the trees route by their missing type
                                "X_float32": [[None if np.isnan(v) else float(v) for v in row] for row in X[cells]],
                                "probability": [float(v) for v in P[cells]]}
        else:
            rec["sha256_float64"] = rec["rate_sha256_float64"]
            top = np.argsort(-P_rate, kind="stable")[:3]
            rec["samples"] = {str(int(c)): float(P_rate[c]) for c in list(top) + [0, 5850]}
        parity.append(rec)
    if tree_rows and max(tree_rows) > TREE_TOL:
        raise RuntimeError(f"NumPy trees differ from the evaluated LightGBM output by {max(tree_rows):.3e}")

    final = json.loads((C.RESULTS / "final.json").read_text(encoding="utf-8"))
    audit = json.loads((C.RESULTS / "catalog_audit.json").read_text(encoding="utf-8"))
    provenance = {
        "protocol": "docs/EARTHQUAKE_FORECAST_PROGRAM.md",
        "fit_split": "issue times 2005-01-03 .. 2017-12 (windows ending by 2018-01-01), weekly",
        "chosen_on": "CHOOSE (issue times 2018-2020) by the pre-registered rule",
        "decision": choose["decision"],
        "fit_results": ["results/earthquake_program/fit_ab.json"] + (["results/earthquake_program/fit_c.json"] if gbt else []),
        "final_results": "results/earthquake_program/final.json",
        "final_metrics": {k: final["candidates"][chosen][k] for k in
                          ("ig_per_target", "auc", "brier", "bss", "calib_ratio", "auc_active_cells")},
        "catalog": {"source": "ComCat via hazardpulse.data.usgs_fdsn.fetch_window, yearly, bisect-on-limit",
                    "n_m45": audit["m45"]["n_events"], "months_without_m5": audit["m45"]["months_without_m5"],
                    "max_rows_per_response": audit["m45"]["max_rows_per_response"]},
        "parity": parity,
        "parity_tree_max_abs_diff": max(tree_rows) if tree_rows else None,
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    version = of.write_artifact(OUT, spec, frozen, cutoff=CUTOFF, model_name=f"eq_operational_{chosen}_v1",
                                provenance=provenance, gbt=gbt)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes), model_version {version}, frozen events {len(frozen):,}")
    if tree_rows:
        print(f"  trees: NumPy vs evaluated LightGBM on {sum(len(r['tree_rows']['cells']) for r in parity)} "
              f"sampled rows, max |dp| = {max(tree_rows):.3e}")

    art = of.load_artifact(OUT)
    try:
        engine = of.RateEngine(art.frozen, art.spec)
        for rec in parity:
            t = of._parse_time(rec["issue_time"])
            got = _sha(engine.rates(t)["probability"])
            if got != rec["rate_sha256_float64"]:
                raise RuntimeError(f"rate model from the frozen catalog != evaluation at {rec['issue_time']}")
            if gbt is not None:
                live = _simulated_live(raw25, t)
                out = of.forecast_from_artifact(art, live, t)
                ref = C.load_pred("C0", rec["split"])[list(C.split_issue_times(rec["split"])).index(t)]
                diff = float(np.max(np.abs(out["probability"] - ref)))
                rec["live_path_max_abs_diff"] = diff
                exact = _sha(out["probability"]) == rec["sha256_float64"]
                print(f"  parity {rec['split']:6s} {rec['issue_time']}: rates bit-exact; live path vs evaluated "
                      f"grid max |dp| = {diff:.3e} ({'bit-exact' if exact else 'not bit-exact'})")
                if diff > TREE_TOL:
                    raise RuntimeError(f"live path differs from the evaluation by {diff:.3e} at {rec['issue_time']}")
            else:
                out = of.forecast_from_artifact(art, [], t)
                if _sha(out["probability"]) != rec["sha256_float64"]:
                    raise RuntimeError(f"live path != evaluation at {rec['issue_time']}")
                print(f"  parity {rec['split']:6s} {rec['issue_time']}: live path reproduces the evaluated grid bit for bit")
    except Exception:
        OUT.unlink(missing_ok=True)
        raise
    report = {"artifact": str(OUT.relative_to(C.REPO)).replace("\\", "/"), "model_version": version,
              "sha256_lf": art.sha256, "size_bytes": OUT.stat().st_size, "frozen_events": len(frozen),
              "parity": [{k: v for k, v in r.items() if k != "tree_rows"} for r in parity],
              "parity_tree_max_abs_diff": provenance["parity_tree_max_abs_diff"]}
    C.write_json("artifact.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
