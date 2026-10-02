#!/usr/bin/env python3
"""Hurricane RI: the pre-registered NOAA-aid stack program (docs/HURRICANE_RI_PROGRAM.md).

Two phases, run in order, each once; the second can read the 2025 season only once.

    python scripts/hurricane_ri_stack.py --phase select --cache-root <...>/.cache
    python scripts/hurricane_ri_stack.py --phase final  --cache-root <...>/.cache

SELECT (development seasons 2020-2024 only; no 2025 deck and no 2025 wind is read)
    * cases: every AL/EP/CP synoptic cycle whose e-deck holds a 30 kt / 24 h RIOD record
      (benchmark_hurricane_vs_ships.index_ri), truth = IBTrACS USA_WIND at t and t+24 h on the
      storm (SID) whose fix at t carries that ATCF id; v8.2 (the benchmark's model C) on the
      live builder's CARQ inputs from the archived a-deck truncated at t;
    * candidates A-F (hazardpulse.hurricane.ri_stack.CANDIDATES), forward chaining: for
      y = 2022, 2023, 2024 fit on 2020..y-1, score y; D/E/F per availability pattern;
    * pooled log loss decides under the parsimony rule (paired storm bootstrap, 2,000 draws);
    * writes results/calibration/hurricane_ri_stack_selection.json.

FINAL (refuses without the selection file; refuses if the final file exists)
    * downloads the 2025 e- and a-decks, builds the 2025 cases the same way;
    * refits the chosen candidate on 2020-2024 and scores 2025 ONCE against A, B, C and v8.2;
    * exports results/models/hurricane_ri_stack_v1.json (identity hurricane_ri_stack_v1-<sha12>)
      and writes results/calibration/hurricane_ri_stack_final.json.

--rehearse runs the FINAL code path on development data only (fit 2020-2023, score 2024) into
--out-dir, so the one real final run cannot be the first time that code executes.

Amendment 1 of the program fixes what the protocol left undefined (missing raw aids, pattern
fits, the parsimony order, the log-loss clip, BSS reference, 2025 isolation).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_hurricane_vs_ships as bvs  # noqa: E402  (e-decks, IBTrACS, model C, scoring)
from hazardpulse.hurricane import ri_evaluation as ev  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402
from hazardpulse.hurricane import ri_stack as rs  # noqa: E402

SELECTION_PATH = ROOT / "results" / "calibration" / "hurricane_ri_stack_selection.json"
FINAL_PATH = ROOT / "results" / "calibration" / "hurricane_ri_stack_final.json"
PROGRAM_PATH = ROOT / "docs" / "HURRICANE_RI_PROGRAM.md"
PROGRAM_COMMIT = "e351265df"  # the protocol, committed before any 2025 number

DEV_SEASONS = (2020, 2024)
FINAL_SEASON = 2025
FOLDS = (2022, 2023, 2024)
BOOTSTRAP_REPS = 2000
SEED = 20261002
NHC_IBTRACS_BASINS = ("NA", "EP")   # IBTrACS basin files that hold every AL/EP/CP storm's full track
COMPLEXITY_ORDER = ("D", "E", "F")  # challengers, by parameter count (Amendment 1, item 3)
POSITION_MATCH_KM = 100.0           # Amendment 2: basin-crossers renumbered operationally
ZERO_PARAM = ("A", "B", "C")


def log(*a: Any) -> None:
    print(*a, flush=True)


def sha256_lf(path: Path) -> str:
    return ri_model.sha256_file(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Truth: IBTrACS USA_WIND at t and t + 24 h
# ---------------------------------------------------------------------------

def nhc_ibtracs_paths(cache_root: Path) -> dict[str, Path]:
    paths = bvs.ibtracs_paths(cache_root)
    out = {b: paths[b] for b in NHC_IBTRACS_BASINS}
    for b, p in out.items():
        if not p.exists():
            raise SystemExit(f"IBTrACS {b} cache missing at {p}")
    return out


def scan_sids(path: Path, seasons: set[int]) -> set[str]:
    """SIDs whose SEASON is in ``seasons`` (reads two columns; no intensity is touched)."""
    wanted = {str(s) for s in seasons}
    out: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h.strip(): i for i, h in enumerate(header)}
        next(reader, None)
        i_sid, i_season = col["SID"], col["SEASON"]
        for row in reader:
            if len(row) > i_season and row[i_season].strip() in wanted:
                out.add(row[i_sid].strip())
    return out


def load_truth_fixes(cache_root: Path, seasons: Iterable[int], last_time: dt.datetime):
    """``(fixes {SID: {t: IbFix}}, index {(ATCF id, t): {SID}}, meta)`` for storms of ``seasons``.

    Only those storms' rows are parsed (``benchmark.parse_ibtracs_fixes``), and fixes after
    ``last_time`` are discarded, so the select phase never holds a 2025 wind.
    """
    seasons = set(seasons)
    paths = nhc_ibtracs_paths(cache_root)
    sids: set[str] = set()
    for p in paths.values():
        sids |= scan_sids(p, seasons)
    fixes: dict[str, dict[dt.datetime, bvs.IbFix]] = {}
    for p in paths.values():
        with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for sid, fx in bvs.parse_ibtracs_fixes(fh, sids).items():
                slot = fixes.setdefault(sid, {})
                for t, f in fx.items():
                    if t <= last_time:
                        slot.setdefault(t, f)
    index: dict[tuple[str, dt.datetime], set[str]] = collections.defaultdict(set)
    for sid, fx in fixes.items():
        for t, f in fx.items():
            if f.atcf_id:
                index[(f.atcf_id, t)].add(sid)
    meta = {"files": {b: {"path": str(p), "sha256": sha256_lf(p), "bytes": p.stat().st_size} for b, p in paths.items()},
            "storms": len(fixes), "seasons": sorted(seasons), "last_time": last_time.isoformat()}
    return fixes, dict(index), meta


def coverage_only(cache_root: Path, season: int) -> dict[str, Any]:
    """Counts of a season's storms/rows and of rows whose USA_WIND is NON-EMPTY -- the value is
    never parsed. This is how the select phase verifies the cache covers the final season."""
    out: dict[str, Any] = {}
    for basin, p in nhc_ibtracs_paths(cache_root).items():
        storms, atcf = set(), set()
        rows = with_wind = 0
        agency: collections.Counter = collections.Counter()
        with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            col = {h.strip(): i for i, h in enumerate(header)}
            next(reader, None)
            for row in reader:
                if len(row) <= col["USA_WIND"] or row[col["SEASON"]].strip() != str(season):
                    continue
                rows += 1
                storms.add(row[col["SID"]].strip())
                if row[col["USA_ATCF_ID"]].strip():
                    atcf.add(row[col["USA_ATCF_ID"]].strip().upper())
                with_wind += bool(row[col["USA_WIND"]].strip())
                agency[row[col["USA_AGENCY"]].strip() or "(blank: interpolated row)"] += 1
        out[basin] = {"storms": len(storms), "rows": rows, "rows_with_nonempty_usa_wind": with_wind,
                      "atcf_ids": sorted(atcf), "usa_agency_rows": dict(agency)}
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def build_cases(deck_cache: Path, cache_root: Path, seasons: tuple[int, int], last_time: dt.datetime
                ) -> tuple[list[dict], dict[str, Any]]:
    """Every NHC-basin synoptic cycle of ``seasons`` with a 30 kt / 24 h RIOD record and truth."""
    years = range(seasons[0], seasons[1] + 1)
    records, per_file = bvs.load_edecks(deck_cache, years)
    table, index_stats = bvs.index_ri(records)
    del records
    fixes, index, ib_meta = load_truth_fixes(cache_root, years, last_time)
    at_time: dict[dt.datetime, list[tuple[str, bvs.IbFix]]] = collections.defaultdict(list)
    for sid_, fx in fixes.items():
        for t_, f_ in fx.items():
            at_time[t_].append((sid_, f_))
    reasons: dict[int, collections.Counter] = {y: collections.Counter() for y in years}
    km: list[float] = []
    cases: list[dict] = []
    for (atcf_id, dtg), techs in sorted(table.items()):
        season = int(atcf_id[-4:])
        if season not in reasons:
            continue
        why = reasons[season]
        if "RIOD" not in techs:
            why["cycle_without_riod"] += 1
            continue
        t = dt.datetime.strptime(dtg, "%Y%m%d%H")
        if t.hour % 6:
            why["non_synoptic"] += 1
            continue
        sids = index.get((atcf_id, t), set())
        if len(sids) > 1:
            why["ambiguous_sid"] += 1
            continue
        by_position = False
        if not sids:
            # Amendment 2: a basin-crosser renumbered operationally (EP042022 = Bonnie's EP
            # stage, AL022022 in IBTrACS) -- the unique storm with a fix at t near the e-deck's
            # own position, or nothing.
            r = techs["RIOD"]
            near = {s for s, f in at_time.get(t, [])
                    if bvs.great_circle_km(r.lat, r.lon, f.lat, f.lon) <= POSITION_MATCH_KM}
            if len(near) != 1:
                why["no_ibtracs_fix_with_this_atcf_id" if not near else "ambiguous_by_position"] += 1
                continue
            sids, by_position = near, True
        sid = next(iter(sids))
        f0 = fixes[sid][t]
        f24 = fixes[sid].get(t + dt.timedelta(hours=24))
        if f24 is None:
            why["no_fix_at_t_plus_24h"] += 1
            continue
        if f0.usa_wind is None or f24.usa_wind is None or f0.usa_wind <= 0 or f24.usa_wind <= 0:
            why["usa_wind_missing"] += 1
            continue
        why["case"] += 1
        if by_position:
            why["case_matched_by_position"] += 1
        riod = techs["RIOD"]
        d_km = bvs.great_circle_km(riod.lat, riod.lon, f0.lat, f0.lon)
        if np.isfinite(d_km):
            km.append(d_km)
        cases.append({
            "atcf_id": atcf_id, "dtg": dtg, "season": season, "sid": sid, "basin": atcf_id[:2],
            "mapped_by": "position" if by_position else "atcf_id",
            "issue_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "aids": {tech: int(techs[tech].prob_pct) for tech in rs.ALL_TECHS if tech in techs},
            "v_t": float(f0.usa_wind), "v_t24": float(f24.usa_wind),
            "y": int(f24.usa_wind - f0.usa_wind >= 30),
            "edeck_fix_km": None if not np.isfinite(d_km) else round(float(d_km), 3),
            "v82": None, "v82_analysis_model": None,
        })
    kma = np.array(km)
    meta = {
        "edeck_files": len(per_file), "edeck_index": index_stats,
        "edeck_files_without_ri": sorted(s for s, v in per_file.items() if v.get("records", 0) == 0),
        "ibtracs": ib_meta,
        "mapping_by_season": {str(y): dict(c) for y, c in reasons.items()},
        "edeck_position_vs_ibtracs_fix_km": {
            "n": int(len(kma)), "median": float(np.median(kma)) if len(kma) else None,
            "p99": float(np.quantile(kma, 0.99)) if len(kma) else None,
            "max": float(kma.max()) if len(kma) else None, "n_over_150_km": int(np.sum(kma > 150))},
    }
    return cases, meta


def attach_v82(cases: list[dict], deck_cache: Path, model: dict) -> dict[str, int]:
    """v8.2 (model C) on the live builder's inputs from the archived a-deck truncated at t."""
    wanted: dict[str, list[str]] = collections.defaultdict(list)
    for c in cases:
        wanted[c["atcf_id"]].append(c["dtg"])
    op, why = bvs.operational_cases(deck_cache, wanted)
    keys = [(c["atcf_id"], c["dtg"]) for c in cases if (c["atcf_id"], c["dtg"]) in op]
    if keys:
        p = ri_model.score_cases(model, [op[k] for k in keys])["calibrated"]
        by_key = dict(zip(keys, p))
        for c in cases:
            k = (c["atcf_id"], c["dtg"])
            if k in by_key:
                c["v82"] = float(by_key[k])
                c["v82_analysis_model"] = str(op[k]["analysis_model"])
    return dict(why)


def case_table_bytes(cases: list[dict]) -> bytes:
    rows = sorted(cases, key=lambda c: (c["atcf_id"], c["dtg"]))
    return "".join(json.dumps(c, sort_keys=True, separators=(",", ":")) + "\n" for c in rows).encode("utf-8")


def write_case_table(cases: list[dict], path: Path) -> str:
    data = case_table_bytes(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(data)
    part.replace(path)
    return hashlib.sha256(data).hexdigest()


def pattern_counts(cases: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in cases:
        pat = "+".join(rs.available_inputs(c["aids"], c["v82"], rs.INPUT_ORDER))
        out[str(c["season"])][pat] += 1
        out["all"][pat] += 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Model C (v8.2 for candidate E and the v8.2 comparator)
# ---------------------------------------------------------------------------

def load_model_c(cache_root: Path) -> tuple[dict, dict]:
    """The benchmark's model C, refused unless it reproduces the evaluation's candidate C."""
    import evaluate_hurricane_ri as evr

    model, meta = bvs.model_c(cache_root, False, log=log)
    v82 = ri_model.load_dataset("v8.2")
    test_rows = ri_model.select_years(v82, evr.ORIGIN["test"])
    p = ri_model.score_cases(model, test_rows)["calibrated"]
    y = np.array([float(c["ri_label_30kt"]) for c in test_rows])
    ref = json.loads(bvs.EVALUATION_PATH.read_text(encoding="utf-8"))["results"]["C_v8_2_heldout_newton"]
    repro = {"auc": ev.auc(y, p), "brier": ev.brier(y, p), "evaluation_auc": ref["auc"], "evaluation_brier": ref["brier"]}
    repro["reproduces"] = bool(abs(repro["auc"] - ref["auc"]) <= 1e-12 and abs(repro["brier"] - ref["brier"]) <= 1e-12)
    if not repro["reproduces"]:
        raise SystemExit(f"model C does not reproduce the evaluation's candidate C: {repro}")
    labels: dict[tuple[str, str], int] = {}
    conflicts = 0
    for r in v82:
        k = (r["storm_id"], r["issue_time"])
        lab = int(r["ri_label_30kt"])
        if k in labels and labels[k] != lab:
            conflicts += 1
        labels.setdefault(k, lab)
    del v82, test_rows
    gc.collect()
    return model, {**meta, "reproduces_evaluation_candidate_C": repro, "v82_labels": labels,
                   "v82_label_key_conflicts": conflicts}


def label_agreement(cases: list[dict], labels: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Our truth vs the v8.2 evaluation's label on the same (SID, time)."""
    out: dict[str, Any] = {}
    for season in sorted({c["season"] for c in cases}):
        cs = [c for c in cases if c["season"] == season]
        both = [(c, labels[(c["sid"], c["issue_time"])]) for c in cs if (c["sid"], c["issue_time"]) in labels]
        dis = [{"atcf_id": c["atcf_id"], "dtg": c["dtg"], "ours": c["y"], "v8_2": lab, "v_t": c["v_t"], "v_t24": c["v_t24"]}
               for c, lab in both if c["y"] != lab]
        out[str(season)] = {"cases": len(cs), "overlap_with_v8_2_rows": len(both),
                            "agree": len(both) - len(dis), "disagree": len(dis),
                            "events_ours_on_overlap": int(sum(c["y"] for c, _ in both)),
                            "events_v8_2_on_overlap": int(sum(lab for _, lab in both)),
                            "disagreements": dis[:25]}
    return out


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _has(row: dict, name: str) -> bool:
    return row["v82"] is not None if name == rs.V82 else row["aids"].get(name) is not None


class PoolFitter:
    """Per-pattern logistic pools of one candidate on one set of training rows (Amendment 1, item 2)."""

    def __init__(self, cand: str, train: list[dict]):
        self.cand = cand
        self.spec = rs.CANDIDATES[cand]
        self.train = train
        self.fits: dict[tuple[str, ...], dict | None] = {}
        self.refusals: dict[str, str] = {}

    def fit(self, pattern: tuple[str, ...]) -> dict | None:
        if pattern in self.fits:
            return self.fits[pattern]
        rows = [r for r in self.train if all(_has(r, n) for n in pattern)]
        y = np.array([r["y"] for r in rows], dtype=float)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        fit: dict | None = None
        if n_pos < rs.MIN_EVENTS_PER_FIT or n_neg < rs.MIN_EVENTS_PER_FIT:
            self.refusals["+".join(pattern)] = f"{n_pos} events / {n_neg} non-events < {rs.MIN_EVENTS_PER_FIT}"
        else:
            cols = rs.design_columns(pattern, self.spec["basin"], rows)
            try:
                theta, diag = bvs.fit_stack(cols, y)
            except ri_model.CalibrationError as exc:
                self.refusals["+".join(pattern)] = f"Newton refused: {exc}"
            else:
                fit = {"pattern": list(pattern), "columns": rs.column_names(pattern, self.spec["basin"]) + ["intercept"],
                       "coef": [float(v) for v in theta], "n": len(rows), "events": n_pos,
                       "storms": len({r["sid"] for r in rows}), "iterations": int(diag["iterations"]),
                       "grad_inf": float(diag["grad_inf"]), "mean_log_loss": float(diag["mean_log_loss"])}
        self.fits[pattern] = fit
        return fit

    def resolve(self, own: tuple[str, ...]) -> tuple[str, ...] | None:
        for p in rs.sub_patterns(own):
            if self.fit(p) is not None:
                return p
        return None

    def predict(self, rows: list[dict]) -> tuple[np.ndarray, dict[str, Any]]:
        own = [rs.available_inputs(r["aids"], r["v82"], self.spec["inputs"]) for r in rows]
        resolved = [self.resolve(o) for o in own]
        if any(p is None for p in resolved):
            raise SystemExit(f"candidate {self.cand}: no fit even for RIOD alone -- fold fails")
        p = np.empty(len(rows))
        for pattern in sorted(set(resolved)):
            idx = [i for i, rp in enumerate(resolved) if rp == pattern]
            cols = rs.design_columns(pattern, self.spec["basin"], [rows[i] for i in idx])
            p[idx] = bvs.apply_stack(np.asarray(self.fits[pattern]["coef"]), cols)
        info = {"own_patterns": dict(collections.Counter("+".join(o) for o in own)),
                "used_patterns": dict(collections.Counter("+".join(r) for r in resolved)),
                "fallback_cycles": int(sum(tuple(o) != r for o, r in zip(own, resolved))),
                "refused_patterns": dict(self.refusals)}
        return p, info

    def fitted(self) -> list[dict]:
        return [f for _, f in sorted(self.fits.items()) if f is not None]


def raw_forecast(cand: str, rows: list[dict]) -> tuple[np.ndarray, dict[str, Any]]:
    """A/B/C through the served path's own function (ri_stack.predict on a raw-aid spec)."""
    spec = rs.CANDIDATES[cand]
    payload = {"kind": "raw_aid", "aid": spec["aid"], "fallback": spec["fallback"]}
    p, used = [], collections.Counter()
    for r in rows:
        prob, info = rs.predict(payload, r["aids"], r["basin"])
        p.append(prob)
        used[info["used"]] += 1
    return np.array(p), {"used": dict(used), "fallback_cycles": int(sum(v for k, v in used.items() if k != spec["aid"]))}


def forecast(cand: str, train: list[dict], test: list[dict]) -> tuple[np.ndarray, bool, dict[str, Any], PoolFitter | None]:
    """(probabilities, quantised?, info, fitter)."""
    if rs.CANDIDATES[cand]["kind"] == "raw_aid":
        p, info = raw_forecast(cand, test)
        return p, True, info, None
    fitter = PoolFitter(cand, train)
    p, info = fitter.predict(test)
    return p, False, info, fitter


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray, quantised: bool, groups: np.ndarray, clim_rows: np.ndarray,
            reps: int) -> dict[str, Any]:
    """Pooled metrics with storm-bootstrap intervals (benchmark.score_forecast) plus log loss.

    BSS is against ``clim_rows`` = the event rate of each case's own fit seasons (Amendment 1,
    item 7); score_forecast calls that column "basin climatology", renamed here.
    """
    sf = bvs.score_forecast(y, p, groups, float(np.mean(clim_rows)), clim_rows, reps, quantised)
    pc = bvs._clip(p, quantised)
    ll_ci = ev.storm_bootstrap(groups, lambda idx: {"log_loss": ev.log_loss(y[idx], pc[idx])}, reps=reps)["log_loss"]
    ci = sf["ci95"]
    return {
        "n": sf["n"], "events": sf["events"], "storms": int(len(set(groups))),
        "observed_rate": sf["observed_rate"], "mean_forecast": sf["mean_forecast"],
        "log_loss": sf["log_loss"], "log_loss_ci95": ll_ci,
        "auc": sf["auc"], "auc_ci95": ci["auc"],
        "brier": sf["brier"], "brier_ci95": ci["brier"],
        "bss": sf["bss_vs_basin_climatology"], "bss_ci95": ci["bss_vs_basin_climatology"],
        "bss_reference": "event rate of the case's fit seasons", "brier_reference": sf["brier_climatology_basin"],
        "calibration_intercept": sf["calibration_intercept"], "calibration_slope": sf["calibration_slope"],
        "calibration_slope_ci95": ci["calibration_slope"], "calibration_intercept_ci95": ci["calibration_intercept"],
        "calibrated": sf["calibrated"], "reliability": sf["reliability"], "prob_range": sf["prob_range"],
        "logit_clip_for_log_loss": sf["logit_clip"],
    }


def paired_all(y: np.ndarray, pa: np.ndarray, qa: bool, pb: np.ndarray, qb: bool, groups: np.ndarray,
               reps: int) -> dict[str, Any]:
    """a minus b: AUC and Brier by benchmark.paired, log loss by the same storm draws (same seed)."""
    base = bvs.paired(y, pa, pb, groups, reps)
    la, lb = bvs._clip(pa, qa), bvs._clip(pb, qb)
    ll = ev.storm_bootstrap(groups, lambda idx: {"delta_log_loss": ev.log_loss(y[idx], la[idx]) - ev.log_loss(y[idx], lb[idx])},
                            reps=reps)
    return {"delta_log_loss": ev.log_loss(y, la) - ev.log_loss(y, lb), "delta_brier": base["delta_brier"],
            "delta_auc": base["delta_auc"], "ci95": {**base["ci95"], **ll}}


def parsimony(pooled: dict[str, dict], y: np.ndarray, preds: dict[str, tuple[np.ndarray, bool]],
              groups: np.ndarray, reps: int) -> dict[str, Any]:
    """Amendment 1, item 3: best zero-parameter candidate, then D, E, F must each beat the
    incumbent with the paired interval of the log-loss difference wholly below 0."""
    zero = sorted(ZERO_PARAM, key=lambda c: (pooled[c]["log_loss"], c))
    incumbent = zero[0]
    steps: list[dict[str, Any]] = [{"step": "zero-parameter candidates by pooled log loss",
                                    "order": [{"candidate": c, "log_loss": pooled[c]["log_loss"]} for c in zero],
                                    "incumbent": incumbent}]
    for ch in COMPLEXITY_ORDER:
        d = paired_all(y, preds[ch][0], preds[ch][1], preds[incumbent][0], preds[incumbent][1], groups, reps)
        hi = d["ci95"]["delta_log_loss"][1]
        replaces = bool(hi < 0)
        steps.append({"challenger": ch, "incumbent": incumbent, "params": [rs.CANDIDATES[ch]["n_params"],
                      rs.CANDIDATES[incumbent]["n_params"]], "paired_challenger_minus_incumbent": d,
                      "interval_excludes_zero_favourably": replaces, "replaces_incumbent": replaces})
        if replaces:
            incumbent = ch
    return {"rule": ("best zero-parameter candidate (A, B, C) by pooled log loss is the incumbent; D, E, F in "
                     "parameter order replace it only if the 95% paired storm-bootstrap interval of "
                     "LL(challenger) - LL(incumbent) lies wholly below 0; otherwise the simpler stays"),
            "steps": steps, "choice": incumbent}


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _download(deck_cache: Path, kind: str, years: Iterable[int]) -> dict[int, int]:
    log(f"download {kind}-decks {list(years)} (sequential, cached)")
    return bvs.download_decks(deck_cache, kind, list(years))


def dev_cases(args, model: dict) -> tuple[list[dict], dict[str, Any]]:
    deck_cache = args.cache_root / "atcf_adecks"
    last = dt.datetime(DEV_SEASONS[1], 12, 31, 23, 59, 59)
    cases, meta = build_cases(deck_cache, args.cache_root, DEV_SEASONS, last)
    meta["v82_operational_build"] = attach_v82(cases, deck_cache, model)
    return cases, meta


def phase_select(args) -> int:
    if SELECTION_PATH.exists() and not args.force:
        raise SystemExit(f"{SELECTION_PATH} exists: the selection has been run (refusing to re-run)")
    t0 = time.perf_counter()
    deck_cache = args.cache_root / "atcf_adecks"
    downloads = {}
    if not args.no_download:
        downloads["e"] = _download(deck_cache, "e", range(DEV_SEASONS[0], DEV_SEASONS[1] + 1))
        downloads["a"] = _download(deck_cache, "a", range(DEV_SEASONS[0], DEV_SEASONS[1] + 1))
    model, model_meta = load_model_c(args.cache_root)
    labels = model_meta.pop("v82_labels")
    cases, meta = dev_cases(args, model)
    agreement = label_agreement(cases, labels)
    del labels
    gc.collect()
    table_path = args.cache_root / "hurricane_ri_stack" / "cases_2020_2024.jsonl"
    table_sha = write_case_table(cases, table_path)
    log(f"dev cases: {len(cases)} ({sum(c['y'] for c in cases)} RI), table sha256 {table_sha[:16]}")
    coverage = coverage_only(args.cache_root, FINAL_SEASON)

    folds, pooled_rows, preds = [], [], {c: ([], None) for c in rs.CANDIDATES}
    infos: dict[str, dict[str, Any]] = {c: {} for c in rs.CANDIDATES}
    fits: dict[str, dict[str, Any]] = {c: {} for c in rs.CANDIDATES}
    clim: list[float] = []
    for y_test in FOLDS:
        train = [c for c in cases if DEV_SEASONS[0] <= c["season"] < y_test]
        test = [c for c in cases if c["season"] == y_test]
        rate = float(np.mean([c["y"] for c in train]))
        folds.append({"test_season": y_test, "fit_seasons": [DEV_SEASONS[0], y_test - 1], "n_fit": len(train),
                      "events_fit": int(sum(c["y"] for c in train)), "n_test": len(test),
                      "events_test": int(sum(c["y"] for c in test)), "storms_test": len({c["sid"] for c in test}),
                      "fit_seasons_event_rate": rate})
        pooled_rows += test
        clim += [rate] * len(test)
        for cand in rs.CANDIDATES:
            p, q, info, fitter = forecast(cand, train, test)
            preds[cand][0].append(p)
            preds[cand] = (preds[cand][0], q)
            infos[cand][str(y_test)] = info
            if fitter is not None:
                fits[cand][str(y_test)] = fitter.fitted()
    y = np.array([c["y"] for c in pooled_rows], dtype=float)
    groups = np.array([c["sid"] for c in pooled_rows])
    clim_rows = np.array(clim)
    pred = {c: (np.concatenate(v[0]), v[1]) for c, v in preds.items()}

    log("scoring pooled forward-chained predictions (storm bootstrap, 2,000 draws)...")
    pooled = {c: metrics(y, p, q, groups, clim_rows, args.reps) for c, (p, q) in pred.items()}
    per_season: dict[str, dict[str, Any]] = {}
    seasons_arr = np.array([c["season"] for c in pooled_rows])
    for y_test in FOLDS:
        m = seasons_arr == y_test
        per_season[str(y_test)] = {c: metrics(y[m], p[m], q, groups[m], clim_rows[m], args.reps)
                                   for c, (p, q) in pred.items()}
    decision = parsimony(pooled, y, pred, groups, args.reps)
    choice = decision["choice"]

    report = {
        "phase": "select", "generated_at": utc_now(),
        "program": {"path": "docs/HURRICANE_RI_PROGRAM.md", "sha256": sha256_lf(PROGRAM_PATH),
                    "protocol_commit": PROGRAM_COMMIT,
                    "amendments": ["Amendment 1 (2026-10-02)", "Amendment 2 (2026-10-02)"]},
        "script": {"path": "scripts/hurricane_ri_stack.py", "sha256": sha256_lf(Path(__file__)),
                   "ri_stack_sha256": sha256_lf(Path(rs.__file__))},
        "case_table": {"path": str(table_path), "sha256": table_sha, "n": len(cases),
                       "events": int(sum(c["y"] for c in cases)), "storms": len({c["sid"] for c in cases}),
                       "by_season": {str(s): {"n": sum(c["season"] == s for c in cases),
                                              "events": sum(c["y"] for c in cases if c["season"] == s),
                                              "storms": len({c["sid"] for c in cases if c["season"] == s})}
                                     for s in range(DEV_SEASONS[0], DEV_SEASONS[1] + 1)}},
        "truth": {"definition": "IBTrACS USA_WIND(t+24 h) - USA_WIND(t) >= 30 kt on the SID whose fix at t carries the ATCF id",
                  "mapping": meta["mapping_by_season"],
                  "agreement_with_v8_2_evaluation_labels": agreement,
                  "edeck_position_vs_ibtracs_fix_km": meta["edeck_position_vs_ibtracs_fix_km"]},
        "data": {"edeck": {k: meta[k] for k in ("edeck_files", "edeck_index", "edeck_files_without_ri")},
                 "ibtracs": meta["ibtracs"], "downloads": downloads,
                 "v82_operational_build": meta["v82_operational_build"],
                 "final_season_ibtracs_coverage_counts_only": coverage},
        "model_c": model_meta,
        "availability_patterns": pattern_counts(cases),
        "folds": folds,
        "candidates": {c: {**{k: v for k, v in rs.CANDIDATES[c].items() if k != "inputs"},
                           "inputs": list(rs.CANDIDATES[c].get("inputs", ())),
                           "pooled": pooled[c], "per_season": {s: per_season[s][c] for s in per_season},
                           "patterns_by_season": infos[c], "fits_by_fold": fits[c]}
                       for c in rs.CANDIDATES},
        "parsimony": decision,
        "choice": choice,
        "bootstrap": {"unit": "IBTrACS SID", "reps": args.reps, "seed": SEED, "interval": "percentile 95%"},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(SELECTION_PATH, report)
    print_selection(report)
    log(f"Wrote {SELECTION_PATH}")
    return 0


# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------

def artifact_payload(choice: str, train: list[dict], selection_sha: str, table_sha: str) -> tuple[dict, PoolFitter | None]:
    spec = rs.CANDIDATES[choice]
    seasons = sorted({c["season"] for c in train})
    payload: dict[str, Any] = {
        "schema": rs.SCHEMA, "name": rs.MODEL_NAME, "candidate": choice, "kind": spec["kind"],
        "description": spec["description"],
        "event": "V(t+24 h) - V(t) >= 30 kt (IBTrACS USA_WIND / HURDAT2)",
        "representation": ("ATCF e-deck whole percent; live SHIPS text: SHIPS-RII from SHIPS's own whole-percent "
                           "line, every other row rounded half-up (hazardpulse.hurricane.ships_text)"),
        "logit_clip": [rs.PCT_CLIP, 1.0 - rs.PCT_CLIP],
        "ships_text_rows": dict(rs.SHIPS_TEXT_ROW),
        "training": {"seasons": [seasons[0], seasons[-1]], "n": len(train), "events": int(sum(c["y"] for c in train)),
                     "storms": len({c["sid"] for c in train}), "case_table_sha256": table_sha},
        "selection_sha256": selection_sha,
        "program": "docs/HURRICANE_RI_PROGRAM.md",
    }
    if spec["kind"] == "raw_aid":
        payload.update({"aid": spec["aid"], "fallback": spec["fallback"]})
        return payload, None
    fitter = PoolFitter(choice, train)
    for pattern in rs.all_patterns(spec["inputs"]):
        fitter.fit(pattern)
    payload.update({"inputs": list(spec["inputs"]), "basin": bool(spec["basin"]),
                    "min_events_per_fit": rs.MIN_EVENTS_PER_FIT,
                    "fallback_rule": "largest fitted sub-pattern; ties keep inputs earliest in " + ",".join(rs.INPUT_ORDER),
                    "patterns": fitter.fitted(), "refused_patterns": dict(fitter.refusals)})
    if rs.V82 in spec["inputs"]:
        payload["v82_model"] = {"description": "benchmark model C (members <= 2018, calibration 2019-2021) "
                                               "on fetch_and_score.build_live_case inputs",
                                "path": "results/models/hurricane_ri_stack_v1_v82c.json"}
    return payload, fitter


def phase_final(args) -> int:
    rehearse = bool(args.rehearse)
    if not SELECTION_PATH.exists():
        raise SystemExit(f"refusing: {SELECTION_PATH} does not exist -- run --phase select first")
    selection_sha = sha256_lf(SELECTION_PATH)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("phase") != "select" or selection.get("choice") not in rs.CANDIDATES:
        raise SystemExit("refusing: the selection file holds no valid choice")
    if rehearse:
        if args.out_dir is None:
            raise SystemExit("--rehearse needs --out-dir (it never writes into results/)")
        final_path = args.out_dir / "rehearsal_final.json"
        artifact_path = args.out_dir / "rehearsal_artifact.json"
        fit_seasons, final_season = (DEV_SEASONS[0], DEV_SEASONS[1] - 1), DEV_SEASONS[1]
    else:
        final_path, artifact_path = FINAL_PATH, rs.ARTIFACT_PATH
        fit_seasons, final_season = DEV_SEASONS, FINAL_SEASON
    if final_path.exists() and not args.force:
        raise SystemExit(f"refusing: {final_path} exists -- 2025 is read once")
    choice = selection["choice"]
    t0 = time.perf_counter()
    deck_cache = args.cache_root / "atcf_adecks"
    downloads = {}
    if not rehearse and not args.no_download:
        downloads["e"] = _download(deck_cache, "e", [FINAL_SEASON])
        downloads["a"] = _download(deck_cache, "a", [FINAL_SEASON])

    model, model_meta = load_model_c(args.cache_root)
    model_meta.pop("v82_labels")
    gc.collect()
    dev, dev_meta = dev_cases(args, model)
    table_sha = hashlib.sha256(case_table_bytes(dev)).hexdigest()
    if table_sha != selection["case_table"]["sha256"]:
        raise SystemExit(f"refusing: the 2020-2024 case table is {table_sha}, the selection was made on "
                         f"{selection['case_table']['sha256']} -- the data moved between the phases")
    if rehearse:
        train = [c for c in dev if fit_seasons[0] <= c["season"] <= fit_seasons[1]]
        final = [c for c in dev if c["season"] == final_season]
        final_meta = {"mapping_by_season": {str(final_season): dev_meta["mapping_by_season"][str(final_season)]},
                      "edeck_position_vs_ibtracs_fix_km": None, "v82_operational_build": None}
        final_table_sha = hashlib.sha256(case_table_bytes(final)).hexdigest()
        final_table_path = None
    else:
        train = dev
        last = dt.datetime(FINAL_SEASON, 12, 31, 23, 59, 59) + dt.timedelta(days=2)
        final, final_meta = build_cases(deck_cache, args.cache_root, (FINAL_SEASON, FINAL_SEASON), last)
        final_meta["v82_operational_build"] = attach_v82(final, deck_cache, model)
        final_table_path = args.cache_root / "hurricane_ri_stack" / f"cases_{FINAL_SEASON}.jsonl"
        final_table_sha = write_case_table(final, final_table_path)

    payload, fitter = artifact_payload(choice, train, selection_sha, hashlib.sha256(case_table_bytes(train)).hexdigest())
    y = np.array([c["y"] for c in final], dtype=float)
    groups = np.array([c["sid"] for c in final])
    clim_rate = float(np.mean([c["y"] for c in train]))
    clim_rows = np.full(len(final), clim_rate)

    preds: dict[str, tuple[np.ndarray, bool]] = {}
    infos: dict[str, Any] = {}
    for cand in sorted({choice, "A", "B", "C"}):
        if cand == choice and fitter is not None:
            p, info = fitter.predict(final)
            preds[cand], infos[cand] = (p, False), info
        else:
            p, q, info, _ = forecast(cand, train, final)
            preds[cand], infos[cand] = (p, q), info
    has_v82 = np.array([c["v82"] is not None for c in final])
    v82_all = np.array([np.nan if c["v82"] is None else c["v82"] for c in final])

    def table(mask: np.ndarray, with_v82: bool) -> dict[str, Any]:
        fc = {c: (p[mask], q) for c, (p, q) in preds.items()}
        if with_v82:
            fc["v8_2"] = (v82_all[mask], False)
        res = {"n": int(mask.sum()), "events": int(y[mask].sum()), "storms": int(len(set(groups[mask]))),
               "forecasts": {k: metrics(y[mask], p, q, groups[mask], clim_rows[mask], args.reps) for k, (p, q) in fc.items()},
               "paired_chosen_minus": {}}
        for other in fc:
            if other == choice:
                continue
            res["paired_chosen_minus"][other] = paired_all(y[mask], fc[choice][0], fc[choice][1], fc[other][0],
                                                           fc[other][1], groups[mask], args.reps)
        return res

    all_mask = np.ones(len(final), dtype=bool)
    results = {"all_cases": table(all_mask, False), "v82_subset": table(has_v82, True)}

    def better(res: dict, other: str) -> dict[str, Any]:
        d = res["paired_chosen_minus"].get(other)
        if d is None:
            return {"claim": False, "why": "the chosen candidate IS this comparator"}
        ll_hi, br_hi = d["ci95"]["delta_log_loss"][1], d["ci95"]["delta_brier"][1]
        ll_lo, br_lo = d["ci95"]["delta_log_loss"][0], d["ci95"]["delta_brier"][0]
        return {"claim": bool(ll_hi < 0 or br_hi < 0), "worse": bool(ll_lo > 0 or br_lo > 0),
                "delta_log_loss_ci95": d["ci95"]["delta_log_loss"], "delta_brier_ci95": d["ci95"]["delta_brier"]}

    claims = {"better_than_DTOPS_A": better(results["all_cases"], "A"),
              "better_than_SHIPS_RII_B": better(results["all_cases"], "B"),
              "better_than_RIOC_C": better(results["all_cases"], "C"),
              "better_than_v8_2": better(results["v82_subset"], "v8_2"),
              "rule": "claimed only if the paired 95% interval on Brier or log loss lies wholly below 0"}

    # served artifact: write, reload, and require the served path to reproduce the fitter
    model_version = rs.save_artifact(payload, artifact_path)
    loaded, loaded_version = rs.load_artifact(artifact_path)
    served = np.array([rs.predict(loaded, c["aids"], c["basin"], c["v82"])[0] for c in final])
    gap = float(np.max(np.abs(served - preds[choice][0]))) if len(final) else 0.0
    if loaded_version != model_version or gap > 1e-12:
        raise SystemExit(f"served artifact does not reproduce the fitted model: max |diff| {gap}")
    v82c = None
    if rs.V82 in rs.CANDIDATES[choice].get("inputs", ()):
        v82c_path = artifact_path.with_name(artifact_path.stem + "_v82c.json")
        model_c_art = ri_model.attach_provenance(dict(model), data={
            "members": {"path": ri_model.DATASETS["v8.2"], "storm_years": (2000, 2018)},
            "calibration": {"path": ri_model.DATASETS["v8.2"], "storm_years": (2019, 2021)}})
        ri_model.save_model(model_c_art, v82c_path)
        v82c = {"path": str(v82c_path), "sha256": sha256_lf(v82c_path)}

    report = {
        "phase": "final" if not rehearse else "rehearsal (fit 2020-2023, score 2024; no 2025 data)",
        "generated_at": utc_now(),
        "selection": {"path": str(SELECTION_PATH.relative_to(ROOT)), "sha256": selection_sha, "choice": choice,
                      "choice_pooled_dev": selection["candidates"][choice]["pooled"]},
        "script": {"sha256": sha256_lf(Path(__file__)), "sha256_at_selection": selection["script"]["sha256"],
                   "changed_since_selection": sha256_lf(Path(__file__)) != selection["script"]["sha256"],
                   "ri_stack_sha256": sha256_lf(Path(rs.__file__))},
        "program": {"sha256": sha256_lf(PROGRAM_PATH), "protocol_commit": PROGRAM_COMMIT},
        "fit": {"seasons": list(fit_seasons), "n": len(train), "events": int(sum(c["y"] for c in train)),
                "case_table_sha256_dev": table_sha, "event_rate": clim_rate},
        "final_season": final_season,
        "final_cases": {"n": len(final), "events": int(y.sum()), "storms": int(len(set(groups))),
                        "table_path": None if final_table_path is None else str(final_table_path),
                        "table_sha256": final_table_sha,
                        "truth_mapping": final_meta["mapping_by_season"],
                        "edeck_position_vs_ibtracs_fix_km": final_meta["edeck_position_vs_ibtracs_fix_km"],
                        "v82_operational_build": final_meta["v82_operational_build"],
                        "v82_available": int(has_v82.sum())},
        "availability_patterns": pattern_counts(final),
        "patterns_used": infos,
        "downloads": downloads,
        "model_c": model_meta,
        "results": results,
        "claims": claims,
        "artifact": {"path": str(artifact_path), "sha256": sha256_lf(artifact_path), "model_version": model_version,
                     "served_path_max_abs_diff": gap, "v82_component": v82c,
                     "patterns": len(payload.get("patterns", [])) or None},
        "bootstrap": {"unit": "IBTrACS SID", "reps": args.reps, "seed": SEED, "interval": "percentile 95%"},
        "adversary": "pending: the protocol requires an independent attack before the result is reported",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(final_path, report)
    print_final(report)
    log(f"Wrote {final_path} and {artifact_path} ({model_version})")
    return 0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=1, allow_nan=True, default=_default) + "\n")
    part.replace(path)


def _row(name: str, m: dict) -> str:
    return (f"  {name:8s} n={m['n']:5d} ev={m['events']:4d}  LL {m['log_loss']:.4f} "
            f"[{m['log_loss_ci95'][0]:.4f},{m['log_loss_ci95'][1]:.4f}]  AUC {m['auc']:.3f} "
            f"[{m['auc_ci95'][0]:.3f},{m['auc_ci95'][1]:.3f}]  Brier {m['brier']:.5f} "
            f"[{m['brier_ci95'][0]:.5f},{m['brier_ci95'][1]:.5f}]  BSS {m['bss']:+.3f} "
            f"[{m['bss_ci95'][0]:+.3f},{m['bss_ci95'][1]:+.3f}]  slope {m['calibration_slope']:.2f}")


def _pair(name: str, d: dict) -> str:
    ci = d["ci95"]
    return (f"  {name:24s} dLL {d['delta_log_loss']:+.4f} [{ci['delta_log_loss'][0]:+.4f},{ci['delta_log_loss'][1]:+.4f}]"
            f"  dBrier {d['delta_brier']:+.5f} [{ci['delta_brier'][0]:+.5f},{ci['delta_brier'][1]:+.5f}]"
            f"  dAUC {d['delta_auc']:+.4f} [{ci['delta_auc'][0]:+.4f},{ci['delta_auc'][1]:+.4f}]")


def print_selection(rep: dict) -> None:
    log("\nSELECTION (pooled forward-chained 2022-2024)")
    for c, v in rep["candidates"].items():
        log(_row(c, v["pooled"]))
    for s in rep["parsimony"]["steps"][1:]:
        log(_pair(f"{s['challenger']} - {s['incumbent']}", s["paired_challenger_minus_incumbent"])
            + f"  -> {'REPLACES' if s['replaces_incumbent'] else 'stays'}")
    log(f"choice: {rep['choice']}")


def print_final(rep: dict) -> None:
    for key in ("all_cases", "v82_subset"):
        r = rep["results"][key]
        log(f"\n{rep['final_season']} {key}: n={r['n']} events={r['events']} storms={r['storms']}")
        for name, m in r["forecasts"].items():
            log(_row(name, m))
        for name, d in r["paired_chosen_minus"].items():
            log(_pair(f"{rep['selection']['choice']} - {name}", d))
    log(json.dumps(rep["claims"], indent=1, default=_default))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("select", "final"), required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".cache",
                        help="holds ibtracs/, atcf_adecks/ and hurricane_vs_ships/ (model C members)")
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--no-download", action="store_true", help="use the deck cache as it is")
    parser.add_argument("--rehearse", action="store_true",
                        help="final phase on development data only (fit 2020-2023, score 2024) into --out-dir")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing phase output (never for the real final)")
    args = parser.parse_args(argv)
    if args.reps != BOOTSTRAP_REPS and not args.rehearse:
        parser.error(f"the protocol fixes {BOOTSTRAP_REPS} bootstrap draws")
    if args.rehearse and args.phase != "final":
        parser.error("--rehearse is a final-phase option")
    return phase_select(args) if args.phase == "select" else phase_final(args)


if __name__ == "__main__":
    raise SystemExit(main())
