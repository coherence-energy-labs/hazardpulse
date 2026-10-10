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
  forecast has. Until 2026-10-05 a JTWC storm was scored from ONE warning; since then it carries its
  best-track history from RAL's real-time b-deck, and when the warning's own cycle has no fix there yet
  the warning is the analysis. Which inputs each case leaves missing is derived here by running the
  live code (``jtwc_live_case``, ``unavailable_inputs``) on a real warning
  (``tests/fixtures/jtwc/wp2626web_20261002.txt``) and the real b-deck
  (``tests/fixtures/ral/bwp262026_20261005.dat``), not typed. Those inputs are median-imputed, as
  live. Descriptive: the served calibration saw these rows, so no number is out of sample; and the
  test's history is the post-season best track, which revises the working one a live case reads.

Nothing is trained or refitted; every number is a scoring of the frozen served artifact.

    PYTHONPATH=src python scripts/hurricane_v82_test_composition.py --model hurricane_ri_v8_3

does the same for v8.3 (docs/HURRICANE_RI_V9_PROGRAM.md amendment 15), on ITS registered test: the storms of first
season 2025 and 2026 to date in the JTWC basins, from J1's rows file, bound to
``results/calibration/hurricane_ri_v8_3.json`` (the rows' sha256, the cycles, the events and v8.3's AUC must be
that file's) and written to ``results/calibration/hurricane_ri_v8_3_test_composition.json``. That test is out of
sample: v8.3's members saw storms of 2000-2021 and its calibration 2022-2024.
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
RAL_FIXTURE = ROOT / "tests" / "fixtures" / "ral" / "bwp262026_20261005.dat"
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


def late_fix_missing_inputs(model: dict) -> list[str]:
    """The inputs a live JTWC case leaves missing when RAL's best track reaches the fix before the warning's
    cycle but not the warning's own (the warning is then the analysis), from the live code on the real
    warning and the real b-deck cut just before that cycle."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fas_v82_late", ROOT / "scripts" / "fetch_and_score.py")
    fas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fas)
    sid, recs = fas._parse_jtwc_warning_to_atcf(JTWC_FIXTURE.read_text(encoding="utf-8"), product_id="wp2626",
                                                now=dt.datetime(2026, 10, 2, 12))
    cycle = max(r.cycle for r in recs if r.tau_hours == 0)
    lines = RAL_FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)
    before = "".join(ln for ln in lines if dt.datetime.strptime(ln.split(",")[2].strip(), "%Y%m%d%H") < cycle)
    case = fas.jtwc_live_case(sid, recs, fetch=lambda url: before)
    if case is None or case["track_source"] != "ral_bdeck" or case["analysis_model"] != "JTWC":
        raise SystemExit(f"the late-fix case did not take RAL's history with the warning as its analysis: {case}")
    names = list(model["feature_names"])
    X, _, _, _ = fas.build_feature_matrix([case], feature_names=names)
    missing, before_first_fix = fas.unavailable_inputs(case, names, X[0])
    if before_first_fix:
        raise SystemExit(f"the fixture storm is too young to measure the late-fix pattern: {before_first_fix}")
    return sorted(missing)


def single_warning_blocks(model: dict, wp: list[dict]) -> tuple[dict, dict]:
    """The served model on West Pacific test cycles scored with the inputs a live JTWC case has: from one warning,
    and with RAL's history but the warning's own fix not yet published (the inputs found by the live code)."""
    y_wp = np.array([float(c["ri_label_30kt"]) for c in wp])
    missing = jtwc_missing_inputs(model)
    p_full = ri_model.score_cases(model, wp)["calibrated"]
    p_jtwc = ri_model.score_cases(model, [{**c, **{k: None for k in missing}} for c in wp])["calibrated"]
    late = late_fix_missing_inputs(model)
    p_late = ri_model.score_cases(model, [{**c, **{k: None for k in late}} for c in wp])["calibrated"]
    rate = float(y_wp.mean())
    single = {
        "fixture": str(JTWC_FIXTURE.relative_to(ROOT)).replace("\\", "/"),
        "n_selected_inputs": len(model["selected_idx"]),
        "inputs_missing_live": missing,
        "west_pacific_test_cycles": len(wp), "west_pacific_events": int(y_wp.sum()),
        "log_loss_full_inputs": log_loss(y_wp, p_full),
        "log_loss_single_warning_inputs": log_loss(y_wp, p_jtwc),
        "log_loss_climatology": log_loss(y_wp, np.full(len(wp), rate)),
        "climatology": "the West Pacific test cycles' own event rate (a constant fitted in sample)",
    }
    late_block = {
        "fixtures": [str(JTWC_FIXTURE.relative_to(ROOT)).replace("\\", "/"),
                     str(RAL_FIXTURE.relative_to(ROOT)).replace("\\", "/")],
        "what": "since 2026-10-05: RAL's best track to the fix before the warning's cycle, the warning as "
                "the analysis (its cycle's fix not yet published)",
        "inputs_missing_live": late,
        "west_pacific_test_cycles": len(wp),
        "log_loss": log_loss(y_wp, p_late),
    }
    for k in ("log_loss_full_inputs", "log_loss_single_warning_inputs", "log_loss_climatology"):
        if not math.isfinite(single[k]):
            raise SystemExit(f"{k} is not finite")
    if not math.isfinite(late_block["log_loss"]):
        raise SystemExit("late_best_track_fix log_loss is not finite")
    return single, late_block


V83 = "hurricane_ri_v8_3"
V83_EVALUATION = ROOT / "results" / "calibration" / "hurricane_ri_v8_3.json"
V83_OUT = ROOT / "results" / "calibration" / "hurricane_ri_v8_3_test_composition.json"


def main_v83(rows_path: Path | None = None) -> int:
    """v8.3's registered test (amendment 15), read from J1's rows file and bound to the amendment's results."""
    import gzip
    import hashlib

    import hurricane_ri_v8_3 as a15
    evaluation = json.loads(V83_EVALUATION.read_text(encoding="utf-8"))
    test_spec = evaluation["test"]
    rows_path = rows_path or a15._rows_path()
    sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if sha != test_spec["rows_sha256"]:
        raise SystemExit(f"{rows_path} is not the rows amendment 15 scored ({sha[:12]} vs {test_spec['rows_sha256'][:12]})")
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    seasons, basins = tuple(test_spec["storm_first_seasons"]), tuple(test_spec["decision_basins"])
    test = [r for r in rows if r["j1_season"] in seasons and r["basin"] in basins]
    y = np.array([float(c["ri_label_30kt"]) for c in test])
    reg = evaluation["registered"]
    if (len(test), int(y.sum())) != (int(test_spec["n"]), int(test_spec["events"])):
        raise SystemExit(f"selected {len(test)} test rows ({int(y.sum())} RI); amendment 15 scored "
                         f"{test_spec['n']} ({test_spec['events']} RI)")
    model = ri_model.load_model(ri_model.ARTIFACTS[V83])
    if evaluation["candidate"]["artifact_sha256"] != hashlib.sha256(
            ri_model.ARTIFACTS[V83].read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest():
        raise SystemExit(f"{ri_model.ARTIFACTS[V83].name} is not the artifact amendment 15 scored")
    auc = a15._j1().metrics(y, ri_model.score_cases(model, test)["calibrated"])["auc"]
    if auc != reg["v8_3"]["auc"]:
        raise SystemExit(f"v8.3's AUC on the selected rows is {auc!r}, amendment 15 recorded {reg['v8_3']['auc']!r}")
    by_basin: dict[str, dict] = {}
    for c, yy in zip(test, y):
        b = by_basin.setdefault(str(c.get("basin")), {"n": 0, "events": 0})
        b["n"] += 1
        b["events"] += int(yy)
    cal = model["calibration"]
    fitted = list((cal.get("fitted_on") or {}).get("storm_years") or [])
    served = {"model_version": model["model_version"],
              "members_storm_years": model["provenance"]["data"]["members"]["storm_years"],
              "calibration_storm_years": fitted, "calibration_n": int(cal["n"]),
              "calibration_fitted_on_the_test_cases": bool(fitted == [min(seasons), max(seasons)]
                                                           and int(cal["n"]) == len(test))}
    single, late = single_warning_blocks(model, [c for c in test if c.get("basin") == "WP"])
    single["note"] = "descriptive: v8.3 never saw these cycles (members 2000-2021, calibration 2022-2024)"
    late["note"] = "descriptive, as above; the test's history is the post-season best track"
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "script": "scripts/hurricane_v82_test_composition.py --model hurricane_ri_v8_3",
        "evaluation": {"file": "results/calibration/hurricane_ri_v8_3.json", "candidate": V83,
                       "n": int(test_spec["n"]), "events": int(test_spec["events"]), "auc": reg["v8_3"]["auc"],
                       "members_storm_years": list(evaluation["candidate"]["recipe"]["members_years"]),
                       "calibration_storm_years": list(evaluation["candidate"]["recipe"]["calibration"]["years"])},
        "dataset": {"path": test_spec["rows"], "sha256": sha,
                    "inputs": "IBTrACS best track (release of 2026-10-08), the JTWC basins "
                              "(scripts/build_hurricane_training_data.py; de-duplicated, amendment 13a)"},
        "test_storm_years": list(seasons),
        "selection": "storms of first season 2025 and 2026 to date in the JTWC basins (WP NI SI SP), as "
                     "amendment 15 selected them",
        "n": len(test), "events": int(y.sum()),
        "by_basin": dict(sorted(by_basin.items(), key=lambda kv: -kv[1]["n"])),
        "served_artifact": served,
        "single_jtwc_warning": single,
        "late_best_track_fix": late,
    }
    with V83_OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(f"v8.3 test: {out['n']} cycles, {out['events']} events, basins "
          + ", ".join(f"{b} {v['n']}" for b, v in out["by_basin"].items()))
    print(f"served calibration fitted on the test cases: {served['calibration_fitted_on_the_test_cases']}")
    print(f"West Pacific ({single['west_pacific_test_cycles']} cycles): LL full {single['log_loss_full_inputs']:.4f}, "
          f"single warning {single['log_loss_single_warning_inputs']:.4f}, "
          f"climatology {single['log_loss_climatology']:.4f}; late fix {late['log_loss']:.4f}")
    print(f"wrote {V83_OUT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("hurricane_ri_v8_2", V83), default="hurricane_ri_v8_2")
    ap.add_argument("--rows", type=Path, default=None, help="v8.3 only: J1's rows file (default: its work directory)")
    args = ap.parse_args(argv)
    if args.model == V83:
        return main_v83(args.rows)
    return main_v82()


def main_v82() -> int:
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

    single, late_block = single_warning_blocks(model, [c for c in test if c.get("basin") == "WP"])
    missing = single["inputs_missing_live"]
    single["note"] = "descriptive: the served artifact's calibration was fitted on these cycles"
    late_block["note"] = "descriptive, as above; the test's history is the post-season best track"
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
        "single_jtwc_warning": single,
        "late_best_track_fix": late_block,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    s = out["single_jtwc_warning"]
    print(f"v8.2 test: {out['n']} cycles, {out['events']} events, basins "
          + ", ".join(f"{b} {v['n']}" for b, v in out["by_basin"].items()))
    print(f"served calibration fitted on the test cases: {served['calibration_fitted_on_the_test_cases']}")
    print(f"single JTWC warning leaves {len(missing)} of {s['n_selected_inputs']} inputs missing: {missing}")
    print(f"West Pacific ({s['west_pacific_test_cycles']} cycles): LL full {s['log_loss_full_inputs']:.4f}, "
          f"single warning {s['log_loss_single_warning_inputs']:.4f}, climatology {s['log_loss_climatology']:.4f}")
    lf = out["late_best_track_fix"]
    print(f"late best-track fix leaves {len(lf['inputs_missing_live'])} missing {lf['inputs_missing_live']}: "
          f"LL {lf['log_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
