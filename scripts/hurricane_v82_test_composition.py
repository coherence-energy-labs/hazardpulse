#!/usr/bin/env python3
"""What the published v8.2 test figure is a test OF (site audit 2026-10-05, finding 7).

    PYTHONPATH=src python scripts/hurricane_v82_test_composition.py

The site said v8.2's held-out AUC (``results/calibration/hurricane_ri_evaluation.json``, candidate C)
was measured on "held-out Atlantic and East Pacific" cases and that the model "has not been tested in
the basins where it publishes". Neither is what the evaluation did. This script measures, from the
evaluation's own data file and the served artifact, and writes
``results/calibration/hurricane_ri_v8_2_test_composition.json``:

* the test set's composition: the evaluation's selection (storms first seen 2022-2024,
  ``ri_model.select_years``) of the v8.2 dataset -- every IBTrACS basin, with best-track inputs
  (``scripts/build_hurricane_training_data.py``) -- by basin, with events;
* what the SERVED artifact is: the same recipe rolled forward (members 2000-2021), its calibration
  fitted on 2022-2024 -- the test set itself;
* how the served model scores the West Pacific test cycles with the inputs a live West Pacific
  forecast has. Live, a JTWC storm is scored from ONE warning; which of the model's inputs that
  leaves missing is derived here by running the live case builder on a real warning
  (``tests/fixtures/jtwc/wp2626web_20261002.txt``), not typed. Those inputs are median-imputed, as
  live. Descriptive: the served calibration saw these rows, so neither number is out of sample.

Nothing is trained or refitted; every number is a scoring of the frozen served artifact.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.hurricane import ri_model  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_v8_2_test_composition.json"
EVALUATION = ROOT / "results" / "calibration" / "hurricane_ri_evaluation.json"
CANDIDATE = "C_v8_2_heldout_newton"
JTWC_FIXTURE = ROOT / "tests" / "fixtures" / "jtwc" / "wp2626web_20261002.txt"
EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def jtwc_missing_inputs(model: dict) -> list[str]:
    """The served model's selected inputs a live single-warning JTWC case leaves missing, found by
    building that case with the live code (fetch_and_score)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fas_v82_comp", ROOT / "scripts" / "fetch_and_score.py")
    fas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fas)
    text = JTWC_FIXTURE.read_text(encoding="utf-8")
    sid, recs = fas._parse_jtwc_warning_to_atcf(text, product_id="wp2626", now=dt.datetime(2026, 10, 2, 12))
    case = fas.build_live_case(sid, recs, full_history=False)
    names = list(model["feature_names"])
    return sorted(names[j] for j in model["selected_idx"] if case.get(names[j]) is None)


def main() -> int:
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    path = ri_model.DATASETS["v8.2"]
    sha = ri_model.sha256_file(path)          # line endings normalised, as the artifact binds its data
    if sha != evaluation["data_sha256"]["v8.2"]:
        raise SystemExit(f"{path} is not the data the evaluation scored ({sha[:12]} vs "
                         f"{evaluation['data_sha256']['v8.2'][:12]})")
    years = tuple(evaluation["origin"]["test"])
    test = ri_model.select_years(ri_model.load_dataset("v8.2"), years)
    res = evaluation["results"][CANDIDATE]
    if len(test) != int(res["n"]):
        raise SystemExit(f"selected {len(test)} test rows, the evaluation scored {res['n']}")
    y = np.array([float(c["ri_label_30kt"]) for c in test])
    by_basin: dict[str, dict] = {}
    for c, yy in zip(test, y):
        b = by_basin.setdefault(str(c.get("basin")), {"n": 0, "events": 0})
        b["n"] += 1
        b["events"] += int(yy)

    model = ri_model.load_model(ri_model.ARTIFACTS["hurricane_ri_v8_2"])
    cal = model["calibration"]
    fitted = (cal.get("fitted_on") or {}).get("storm_years")
    served = {"model_version": model["model_version"],
              "members_storm_years": model["provenance"]["data"]["members"]["storm_years"],
              "calibration_storm_years": fitted, "calibration_n": int(cal["n"]),
              "calibration_fitted_on_the_test_cases": bool(list(fitted or []) == list(years)
                                                           and int(cal["n"]) == len(test))}

    missing = jtwc_missing_inputs(model)
    wp = [c for c in test if c.get("basin") == "WP"]
    y_wp = np.array([float(c["ri_label_30kt"]) for c in wp])
    p_full = ri_model.score_cases(model, wp)["calibrated"]
    stripped = [{**c, **{k: None for k in missing}} for c in wp]
    p_jtwc = ri_model.score_cases(model, stripped)["calibrated"]
    rate = float(y_wp.mean())
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "script": "scripts/hurricane_v82_test_composition.py",
        "evaluation": {"file": "results/calibration/hurricane_ri_evaluation.json", "candidate": CANDIDATE,
                       "n": int(res["n"]), "events": int(res["events"]), "auc": res["auc"],
                       "members_storm_years": evaluation["origin"]["members"],
                       "calibration_storm_years": evaluation["origin"]["calibration"]},
        "dataset": {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha,
                    "inputs": "IBTrACS best track, every basin (scripts/build_hurricane_training_data.py)"},
        "test_storm_years": list(years),
        "selection": "storms first seen in the test years (ri_model.select_years), as the evaluation selected them",
        "n": len(test), "events": int(y.sum()),
        "by_basin": dict(sorted(by_basin.items(), key=lambda kv: -kv[1]["n"])),
        "served_artifact": served,
        "single_jtwc_warning": {
            "fixture": str(JTWC_FIXTURE.relative_to(ROOT)).replace("\\", "/"),
            "n_selected_inputs": len(model["selected_idx"]),
            "inputs_missing_live": missing,
            "west_pacific_test_cycles": len(wp), "west_pacific_events": int(y_wp.sum()),
            "log_loss_full_inputs": log_loss(y_wp, p_full),
            "log_loss_single_warning_inputs": log_loss(y_wp, p_jtwc),
            "log_loss_climatology": log_loss(y_wp, np.full(len(wp), rate)),
            "climatology": "the West Pacific test cycles' own event rate (a constant fitted in sample)",
            "note": "descriptive: the served artifact's calibration was fitted on these cycles",
        },
    }
    for k in ("log_loss_full_inputs", "log_loss_single_warning_inputs", "log_loss_climatology"):
        if not math.isfinite(out["single_jtwc_warning"][k]):
            raise SystemExit(f"{k} is not finite")
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    s = out["single_jtwc_warning"]
    print(f"v8.2 test: {out['n']} cycles, {out['events']} events, basins "
          + ", ".join(f"{b} {v['n']}" for b, v in out["by_basin"].items()))
    print(f"served calibration fitted on the test cases: {served['calibration_fitted_on_the_test_cases']}")
    print(f"single JTWC warning leaves {len(missing)} of {s['n_selected_inputs']} inputs missing: {missing}")
    print(f"West Pacific ({s['west_pacific_test_cycles']} cycles): LL full {s['log_loss_full_inputs']:.4f}, "
          f"single warning {s['log_loss_single_warning_inputs']:.4f}, climatology {s['log_loss_climatology']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
