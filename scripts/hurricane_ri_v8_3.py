"""Hurricane RI amendment 15 (docs/HURRICANE_RI_V9_PROGRAM.md): v8.3 = v8.2 retrained on de-duplicated data.

    PYTHONPATH=src python scripts/hurricane_ri_v8_3.py data          # the de-duplicated rows (control b)
    PYTHONPATH=src python scripts/train_hurricane_ri.py --recipe hurricane_ri_v8_3
    git add -f results/models/hurricane_ri_v8_3.json
    PYTHONPATH=src python scripts/hurricane_ri_v8_3.py evaluate      # controls a-c, then the registered test

Data: the frozen v8.2 file with its byte-identical (storm, issue time) copies removed by amendment 13a's rule
(``hurricane_ri_j1.dedupe``). IBTrACS's per-basin files each carry the whole track of a storm that enters the basin
and the builder concatenates them, so 178 basin-crossing storms are counted once per basin they touch. Every kept
line is byte-identical to its line in the frozen file; the output is LF bytes in the frozen order (the builder's
conventions).

Controls, each stops the run before any v8.3 number is computed:
  (a) ``hurricane_ri_v8_2`` refitted on the frozen (non-de-duplicated) file equals the committed v8.2 artifact bit
      for bit (parameters, data hashes, config, trainer fingerprint), and v8.2 recomputed by ``score_cases`` gives
      its recorded calibration log loss 0.17576412086244972 on its 8,317 calibration cases to 1e-12;
  (b) the dedupe removes exactly 5,950 rows (69,722 -> 63,772) and the v8.3 file holds exactly the kept lines;
  (c) the test rows are J1's registered rows (sha256 in results/calibration/hurricane_ri_j1.json), with 1,837 (68 RI)
      2025 and 1,130 (90 RI) 2026 JTWC cycles, and v8.2 recomputed reproduces every row's ``v82_calibrated`` to 1e-12.
The committed v8.3 artifact must also load bound to the v8.3 file (``ri_model.load_model``) and a refit of its
recipe must reproduce it bit for bit, so the number scored is the number the recipe makes.

Test (untouched): storms of first season 2025 and 2026 to date, JTWC basins (WP NI SI SP). Rule: v8.3 replaces v8.2
iff the upper end of the 95 % storm-bootstrap interval of dLL (v8.3 - v8.2, 30/24 log loss) is <= +0.002 nats
(``hurricane_ri_j1.paired``: storms resampled with replacement, 2,000 replicates, default_rng(0)).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.hurricane import ri_model as rm  # noqa: E402

PROGRAM = "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 15)"
PREREG_TAG = "prereg-hurricane-ri-amend15"
OUT = ROOT / "results" / "calibration" / "hurricane_ri_v8_3.json"
J1_RESULTS = ROOT / "results" / "calibration" / "hurricane_ri_j1.json"
V82, V83 = "hurricane_ri_v8_2", "hurricane_ri_v8_3"

FROZEN_ROWS = 69_722
EXPECTED_REMOVED = 5_950                      # amendment 13a
V82_CAL_LOG_LOSS = 0.17576412086244972        # the v8.2 artifact's own calibration number
V82_CAL_N = 8_317
TOL = 1e-12
MEMBER_YEARS, CAL_YEARS = (2000, 2021), (2022, 2024)

TEST_SEASONS = (2025, 2026)
JTWC = ("WP", "NI", "SI", "SP")
ALL_BASINS = ("WP", "NI", "SI", "SP", "NA", "EP")
EXPECTED_TEST = {2025: (1_837, 68), 2026: (1_130, 90)}   # JTWC (cycles, RI events), amendment 13a
MARGIN = 0.002                                # nats: the registered non-inferiority margin on the upper bound


def _j1():
    import hurricane_ri_j1 as j1
    return j1


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(lf_bytes(path)).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# data: amendment 13a's dedupe on the frozen file (control b)
# ---------------------------------------------------------------------------------------------------------------

def dedupe_frozen() -> dict:
    """The frozen v8.2 file's lines, de-duplicated by ``hurricane_ri_j1.dedupe`` (the registered rule), with every
    removal checked to be a BYTE-identical copy of the line kept for its key. Returns the kept bytes and the
    accounting; refuses (SystemExit) unless each line is the builder's own ``json.dumps`` of its row."""
    j1 = _j1()
    raw = lf_bytes(rm.DATASETS["v8.2"])
    if not raw.endswith(b"\n"):
        raise SystemExit(f"{rm.DATASETS['v8.2']} does not end with a newline")
    lines = raw.split(b"\n")[:-1]
    rows = [json.loads(line) for line in lines]
    for i, (line, row) in enumerate(zip(lines, rows)):
        if json.dumps(row).encode("utf-8") != line:
            raise SystemExit(f"line {i + 1} of the frozen file is not the builder's json.dumps of its row")
    kept_rows, removed = j1.dedupe(rows)          # the registered rule (raises on two different rows with one key)

    first: dict[str, int] = rm.storm_first_year(rows)
    seen: dict[str, int] = {}
    keep_idx: list[int] = []
    dup_idx: list[int] = []
    for i, row in enumerate(rows):
        k = j1.row_key(row)
        if k in seen:
            if lines[seen[k]] != lines[i]:
                raise SystemExit(f"{k}: a removed copy is not byte-identical to the kept line")
            dup_idx.append(i)
        else:
            seen[k] = i
            keep_idx.append(i)
    if len(dup_idx) != removed or [rows[i] for i in keep_idx] != kept_rows:
        raise SystemExit("the byte-level accounting disagrees with hurricane_ri_j1.dedupe")

    def role(year: int) -> str:
        if MEMBER_YEARS[0] <= year <= MEMBER_YEARS[1]:
            return "members_2000_2021"
        if CAL_YEARS[0] <= year <= CAL_YEARS[1]:
            return "calibration_2022_2024"
        return "other"

    by_role: dict[str, dict[str, int]] = {}
    for idx, key in ((range(len(rows)), "frozen"), (keep_idx, "kept"), (dup_idx, "removed")):
        for i in idx:
            r = by_role.setdefault(role(first[rows[i]["storm_id"]]), {})
            r[key] = r.get(key, 0) + 1
            if key == "removed":
                r["removed_ri"] = r.get("removed_ri", 0) + int(rows[i]["ri_label_30kt"])
    return {
        "kept_bytes": b"".join(lines[i] + b"\n" for i in keep_idx),
        "frozen_rows": len(rows), "kept_rows": len(keep_idx), "removed": removed,
        "removed_storms": len({rows[i]["storm_id"] for i in dup_idx}),
        "removed_ri": sum(int(rows[i]["ri_label_30kt"]) for i in dup_idx),
        "by_role": dict(sorted(by_role.items())),
    }


def control_b(d: dict) -> None:
    if d["frozen_rows"] != FROZEN_ROWS:
        raise SystemExit(f"control (b) failed: the frozen file has {d['frozen_rows']} rows, not {FROZEN_ROWS}")
    if d["removed"] != EXPECTED_REMOVED or d["kept_rows"] != FROZEN_ROWS - EXPECTED_REMOVED:
        raise SystemExit(f"control (b) failed: the dedupe removed {d['removed']} rows, not {EXPECTED_REMOVED}")


def data_phase() -> int:
    d = dedupe_frozen()
    control_b(d)
    out = rm.DATASETS["v8.3"]
    out.write_bytes(d["kept_bytes"])            # LF bytes on every OS, as the builder writes them
    print(f"control (b): {d['removed']} byte-identical copies removed from {d['frozen_rows']} rows "
          f"({d['removed_storms']} storms, {d['removed_ri']} RI-positive); by storm first season {d['by_role']}")
    print(f"wrote {out} ({d['kept_rows']} rows, sha256 {lf_sha256(out)})")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# refits (controls a and the v8.3 reproduction), one process each
# ---------------------------------------------------------------------------------------------------------------

def refit_job(version: str) -> dict:
    """Refit ``version`` from its recipe and compare with the committed artifact, as
    ``train_hurricane_ri.py --verify`` does (parameters, then provenance data / config / trainer fingerprint)."""
    import train_hurricane_ri as thr
    model = rm.fit_recipe(version, log=lambda m: None)
    rm.attach_provenance(model, data=rm.recipe_data_binding(version))
    rm.validate_structure(model)
    committed = json.loads(rm.ARTIFACTS[version].read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(model, allow_nan=False))
    diff = thr._first_difference(rm.model_parameters(committed), rm.model_parameters(fresh))
    for key in ("data", "config", "trainer_fingerprint"):
        if committed.get("provenance", {}).get(key) != fresh["provenance"].get(key):
            diff = diff or f"provenance.{key} differs"
    return {"version": version, "bit_identical": diff is None, "first_difference": diff,
            "trees": [len(fresh["gbt_d3"]["trees"]), len(fresh["gbt_d4"]["trees"])], "bags": len(fresh["bagged"]),
            "calibration_mean_log_loss": fresh["calibration"]["mean_log_loss"]}


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------------------------

def _rows_path() -> Path:
    j1 = _j1()
    return j1._work() / j1.ROWS_NAME


def evaluate_phase(rows_path: Path) -> int:
    j1 = _j1()
    # (b) the v8.3 file is exactly the registered dedupe of the frozen file
    d = dedupe_frozen()
    control_b(d)
    if lf_bytes(rm.DATASETS["v8.3"]) != d["kept_bytes"]:
        raise SystemExit(f"control (b) failed: {rm.DATASETS['v8.3']} is not the frozen file's de-duplicated lines")
    print(f"control (b): {d['removed']} rows removed; {rm.DATASETS['v8.3'].name} holds exactly the kept lines", flush=True)

    # recipes: v8.3 is v8.2's recipe with only the dataset changed
    a, b = dict(rm.RECIPES[V82]), dict(rm.RECIPES[V83])
    if (a.pop("dataset"), b.pop("dataset")) != ("v8.2", "v8.3") or \
            (a["calibration"].get("dataset"), b["calibration"].get("dataset")) != ("v8.2", "v8.3") or \
            {**a, "calibration": {k: v for k, v in a["calibration"].items() if k != "dataset"}} != \
            {**b, "calibration": {k: v for k, v in b["calibration"].items() if k != "dataset"}}:
        raise SystemExit("the v8.3 recipe is not v8.2's recipe with only the dataset changed")

    # (a) and the v8.3 reproduction: both refits, in parallel
    print("refitting hurricane_ri_v8_2 (frozen file) and hurricane_ri_v8_3 (de-duplicated file)...", flush=True)
    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {v: pool.submit(refit_job, v) for v in (V82, V83)}
        refits = {v: f.result() for v, f in jobs.items()}
    v82 = rm.load_model(rm.ARTIFACTS[V82])
    frozen = rm.load_dataset("v8.2")
    cal82 = rm.select_years(frozen, tuple(v82["calibration"]["fitted_on"]["storm_years"]))
    y_cal82 = np.array([c["ri_label_30kt"] for c in cal82], float)
    ll82 = log_loss(y_cal82, rm.score_cases(v82, cal82)["calibrated"])
    control_a = {"refit": refits[V82], "calibration_n": len(cal82), "calibration_log_loss": ll82,
                 "registered_log_loss": V82_CAL_LOG_LOSS, "abs_diff": abs(ll82 - V82_CAL_LOG_LOSS)}
    if not refits[V82]["bit_identical"]:
        raise SystemExit(f"control (a) failed: the v8.2 refit differs: {refits[V82]['first_difference']}")
    if len(cal82) != V82_CAL_N or abs(ll82 - V82_CAL_LOG_LOSS) > TOL:
        raise SystemExit(f"control (a) failed: v8.2 log loss {ll82!r} on {len(cal82)} cases")
    print(f"control (a): v8.2 refit bit-identical; calibration log loss {ll82!r} on {len(cal82)} cases", flush=True)

    v83 = rm.load_model(rm.ARTIFACTS[V83])       # refuses unless bound to the v8.3 file's sha256
    if not refits[V83]["bit_identical"]:
        raise SystemExit(f"the committed v8.3 artifact is not what its recipe makes: {refits[V83]['first_difference']}")
    print("v8.3: artifact bound to the de-duplicated file; refit bit-identical", flush=True)

    # (c) the test rows
    rows_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    registered_sha = json.loads(J1_RESULTS.read_text(encoding="utf-8"))["rows_sha256"]
    if rows_sha != registered_sha:
        raise SystemExit(f"control (c) failed: {rows_path} sha256 {rows_sha} != J1's registered {registered_sha}")
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    test = [r for r in rows if r["j1_season"] in TEST_SEASONS]
    for season, (n_want, e_want) in EXPECTED_TEST.items():
        s = [r for r in test if r["j1_season"] == season and r["basin"] in JTWC]
        n, e = len(s), sum(int(r["ri_label_30kt"]) for r in s)
        if (n, e) != (n_want, e_want):
            raise SystemExit(f"control (c) failed: {season} JTWC rows {n} ({e} RI), registered {n_want} ({e_want} RI)")
    sc82 = rm.score_cases(v82, test)
    p82 = np.asarray(sc82["calibrated"], float)
    stored = np.array([r["v82_calibrated"] for r in test], float)
    worst = float(np.max(np.abs(p82 - stored)))
    if worst > TOL:
        raise SystemExit(f"control (c) failed: v8.2 recomputed differs from the rows' v82_calibrated by {worst!r}")
    control_c = {"rows_sha256": rows_sha, "registered_rows_sha256": registered_sha,
                 "jtwc_counts": {str(k): {"n": v[0], "events": v[1]} for k, v in EXPECTED_TEST.items()},
                 "v82_recomputed_max_abs_diff": worst, "n_rows_checked": len(test)}
    print(f"control (c): rows are J1's ({rows_sha[:8]}...); counts as registered; v8.2 reproduces "
          f"{len(test)} rows' v82_calibrated (worst {worst:.1e})", flush=True)

    # ---- every control passed: the v8.3 numbers ----
    p83 = np.asarray(rm.score_cases(v83, test)["calibrated"], float)
    y = np.array([r["ri_label_30kt"] for r in test], float)
    storms = np.array([r["storm_id"] for r in test])
    basin = np.array([r["basin"] for r in test])
    season = np.array([r["j1_season"] for r in test])

    def finite(m: dict) -> dict:   # AUC is undefined (NaN) without an event; JSON carries it as null
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()}

    def block(mask: np.ndarray) -> dict:
        ym, a_, b_ = y[mask], p82[mask], p83[mask]
        return {"n": int(mask.sum()), "events": int(ym.sum()), "storms": int(len(set(storms[mask]))),
                "v8_2": finite(j1.metrics(ym, a_)), "v8_3": finite(j1.metrics(ym, b_)),
                "v8_3_minus_v8_2": j1.paired(ym, a_, b_, storms[mask])}

    jtwc = np.isin(basin, JTWC)
    registered = block(jtwc)
    upper = registered["v8_3_minus_v8_2"]["d_ll_ci"][1]
    met = bool(upper <= MARGIN)
    ties = {"v8_2": int(jtwc.sum() - len(np.unique(p82[jtwc]))), "v8_3": int(jtwc.sum() - len(np.unique(p83[jtwc])))}

    res = {
        "phase": "amendment 15 evaluate", "program": PROGRAM, "prereg_tag": PREREG_TAG,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate": {"version": V83, "artifact": rm._rel(rm.ARTIFACTS[V83]),
                      "artifact_sha256": lf_sha256(rm.ARTIFACTS[V83]),
                      "recipe": {**rm.RECIPES[V83], "members_years": list(rm.RECIPES[V83]["members_years"]),
                                 "calibration": {**rm.RECIPES[V83]["calibration"],
                                                 "years": list(rm.RECIPES[V83]["calibration"]["years"])}},
                      "training_summary": v83["training_summary"],
                      "calibration": {k: v83["calibration"][k] for k in
                                      ("a", "b", "n", "n_events", "iterations", "grad_inf", "mean_log_loss")}},
        "comparator": {"version": V82, "artifact": rm._rel(rm.ARTIFACTS[V82]),
                       "artifact_sha256": lf_sha256(rm.ARTIFACTS[V82]),
                       "training_summary": v82["training_summary"],
                       "calibration": {k: v82["calibration"][k] for k in
                                       ("a", "b", "n", "n_events", "iterations", "grad_inf", "mean_log_loss")}},
        "data": {"frozen": {"path": rm._rel(rm.DATASETS["v8.2"]), "sha256": rm.sha256_file(rm.DATASETS["v8.2"]),
                            "rows": d["frozen_rows"]},
                 "deduplicated": {"path": rm._rel(rm.DATASETS["v8.3"]), "sha256": rm.sha256_file(rm.DATASETS["v8.3"]),
                                  "rows": d["kept_rows"]},
                 "rule": "hurricane_ri_j1.dedupe: one row per (storm_id, issue_time), the first kept; only "
                         "byte-identical copies removed",
                 "removed": d["removed"], "removed_storms": d["removed_storms"], "removed_ri": d["removed_ri"],
                 "by_storm_first_season": d["by_role"]},
        "controls": {"a_v82_reproduces": control_a, "b_dedupe": {"removed": d["removed"],
                                                                 "registered": EXPECTED_REMOVED},
                     "c_test_rows": control_c, "v83_refit": refits[V83], "all_passed": True},
        "test": {"rows": ".cache/hurricane_ri_v9/j1/rows.jsonl.gz (hurricane_ri_j1 rows phase, amendment 13a)",
                 "rows_sha256": rows_sha, "storm_first_seasons": list(TEST_SEASONS), "decision_basins": list(JTWC),
                 "n": int(jtwc.sum()), "events": int(y[jtwc].sum()), "storms": int(len(set(storms[jtwc])))},
        "bootstrap": {"function": "hurricane_ri_j1.paired", "unit": "storm (resampled with replacement; each drawn "
                      "storm keeps all its cycles; the mean is a ratio of sums)", "replicates": j1.N_BOOT,
                      "rng": "numpy default_rng(0)", "interval": "percentile (2.5, 97.5)",
                      "clip": [1e-12, 1 - 1e-12]},
        "registered": registered,
        "ties_in_jtwc_probabilities": ties,
        "rule": {"statement": "v8.3 replaces v8.2 iff the upper end of the 95 % storm-bootstrap interval of the "
                              "30/24 log loss difference (v8.3 - v8.2) on the 2025 + 2026 JTWC rows is <= +0.002 nats",
                 "margin": MARGIN, "upper": upper, "met": met},
        "decision": {"adopt": met, "serve": V83 if met else V82},
        "by_region": {g: block(basin == g) for g in JTWC},
        "by_season": {str(s): block(jtwc & (season == s)) for s in TEST_SEASONS},
        "all_basins": block(np.isin(basin, ALL_BASINS)),
        "nhc_basins": {g: block(basin == g) for g in ("NA", "EP")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    r = registered
    print(f"registered (JTWC 2025 + 2026: {r['n']} cycles, {r['events']} RI, {r['storms']} storms):")
    print(f"  v8.2 LL {r['v8_2']['log_loss']:.5f} Brier {r['v8_2']['brier']:.5f} AUC {r['v8_2']['auc']:.4f}")
    print(f"  v8.3 LL {r['v8_3']['log_loss']:.5f} Brier {r['v8_3']['brier']:.5f} AUC {r['v8_3']['auc']:.4f}")
    dd = r["v8_3_minus_v8_2"]
    print(f"  dLL {dd['d_ll']:+.5f} [{dd['d_ll_ci'][0]:+.5f}, {dd['d_ll_ci'][1]:+.5f}]  "
          f"dBrier {dd['d_brier']:+.5f} [{dd['d_brier_ci'][0]:+.5f}, {dd['d_brier_ci'][1]:+.5f}]")
    print(f"rule: upper {upper:+.5f} <= +{MARGIN} -> {'MET: adopt v8.3' if met else 'NOT met: v8.2 stays'}")
    print(f"wrote {OUT}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("data", "evaluate"))
    ap.add_argument("--rows", type=Path, default=None, help="J1's rows file (default: the J1 work directory's)")
    a = ap.parse_args(argv)
    if a.phase == "data":
        return data_phase()
    return evaluate_phase(a.rows or _rows_path())


if __name__ == "__main__":
    raise SystemExit(main())
