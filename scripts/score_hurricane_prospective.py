#!/usr/bin/env python3
"""Score the PUBLISHED hurricane numbers against the realized intensity change.

Reads the forecast files ``dist/data/replay/hu_fcst_*.json`` and writes

- results/hurricane_prospective/prospective_summary.json
- results/hurricane_prospective/per_forecast_scores.jsonl

The unit is the storm-cycle (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 7, rules 1 and 4 --
``hazardpulse.hurricane.cycle_records``):

* every published storm number is the forecast of ONE storm at ONE synoptic time ``t`` (its
  ``issue_time``), and its outcome is the best-track change from ``t`` to ``t + 24 h``. The time the
  scorer ran never enters: aligning the truth to the run time scored storm-cycles against the wrong
  24 hours (a run at 04:54 for the 00Z cycle verified 04:54 -> 04:54 the next day);
* a storm-cycle is scored ONCE, from the first record made at or after ``t + 3 h 30 min``. Repeat runs
  of one cycle (Nolo 2026-10-03 12Z was recorded six times) are duplicates; records made earlier read
  preliminary inputs. Both are listed, with the reason, never scored.

Outcomes are three-valued. A storm whose best track has no fix at ``t`` or at ``t + 24 h`` is
UNVERIFIABLE -- counted, never scored as "no RI". Forecast files with no active storms carry no
prediction and are counted separately. Metrics are POOLED over storm-cycles with their counts; a
ranking score (AUC) is reported only when both outcomes occurred and the cycles span at least
``MIN_AUC_STORMS`` storms (an AUC inside one forecast of two or three storms is noise).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.core.metrics import roc_auc  # noqa: E402
from hazardpulse.data.http import fetch_bytes  # noqa: E402
from hazardpulse.hurricane import cycle_records as cr  # noqa: E402
from hazardpulse.hurricane.atcf import (  # noqa: E402
    ATCF_ROOT,
    ATCFRecord,
    parse_atcf_deck,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hurricane_prospective"

# RI definition: >= 30 kt increase in 24 hours (NHC standard)
RI_THRESHOLD_KT = cr.RI_THRESHOLD_KT
RI_WINDOW_HOURS = 24
# A ranking score needs both outcomes and more than a handful of storms.
MIN_AUC_STORMS = 5

# Best-track sources, tried in order. NHC's btk folder carries only its own basins
# (al/ep/cp) -- every West Pacific / Indian Ocean / Southern Hemisphere storm 404s
# there. UCAR RAL's open b-deck repository mirrors the operational working best
# tracks of every basin (JTWC-sourced outside NHC's area), keyed by season year.
NHC_BDECK_URL = "{root}/btk/b{storm_id}.dat"
RAL_BDECK_URL = "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{storm_id}.dat"
NHC_BASINS = ("al", "ep", "cp")
# Kept for callers that formatted the NHC URL themselves.
BDECK_URL = NHC_BDECK_URL


def parse_utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)


def format_utc_z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def best_track_urls(storm_id: str) -> list[str]:
    """Candidate b-deck URLs for an ATCF storm id such as ``WP042026``."""
    sid = storm_id.strip().lower()
    urls: list[str] = []
    if sid[:2] in NHC_BASINS:
        urls.append(NHC_BDECK_URL.format(root=ATCF_ROOT, storm_id=sid))
    if len(sid) == 8 and sid[-4:].isdigit():
        urls.append(RAL_BDECK_URL.format(year=sid[-4:], storm_id=sid))
    return urls


def fetch_best_track_with_source(storm_id: str) -> tuple[list[ATCFRecord], str | None]:
    """Fetch a storm's best track; returns (records, source_url) or ([], None).

    Never served from the on-disk HTTP cache: a b-deck grows while the storm is
    alive, so a copy cached before the verification window closed would make the
    outcome look unverifiable (or wrong) forever.
    """
    last_exc: Exception | None = None
    for url in best_track_urls(storm_id):
        for candidate, gz in ((url, False), (url + ".gz", True)):
            try:
                data = fetch_bytes(candidate, namespace="atcf_bdeck", timeout=30,
                                   use_cache=False)
                if gz:
                    data = gzip.decompress(data)
                records = parse_atcf_deck(data.decode("utf-8", errors="replace"))
            except Exception as exc:  # 404, network, bad gzip -> try the next source
                last_exc = exc
                continue
            if records:
                return records, url
    print(f"  Warning: no best track found for {storm_id} "
          f"(tried {len(best_track_urls(storm_id))} source(s); last error: {last_exc})")
    return [], None


def fetch_best_track(storm_id: str) -> list[ATCFRecord]:
    """Fetch best-track b-deck records for a storm (see fetch_best_track_with_source)."""
    return fetch_best_track_with_source(storm_id)[0]


def check_ri_occurred(
    records: list[ATCFRecord],
    cycle: dt.datetime,
) -> tuple[bool | None, float | None, float | None]:
    """Did the storm intensify by >= 30 kt from its synoptic time ``cycle`` to ``cycle + 24 h``?

    ``(ri_occurred, best-track wind at cycle, at cycle + 24 h)`` from the BEST fixes at exactly those
    two times (amendment 7, rule 4). ``ri_occurred`` is None -- UNVERIFIABLE -- when either fix is
    missing: absence of a track is not evidence that the storm did not intensify.
    """
    o = cr.outcome(cr.best_track_intensity(records), cycle)
    return o.ri, o.v_t, o.v_t24


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tie-aware ROC-AUC; invariant to row order (see hazardpulse.core.metrics)."""
    return roc_auc(y_true, y_score)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))


def load_replay_artifacts(replay_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    for path in sorted(replay_dir.glob("hu_fcst_*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    artifacts.sort(key=lambda a: a.get("issued_at", ""))
    return artifacts


def matured_artifacts(
    artifacts: list[dict], score_as_of: dt.datetime
) -> list[dict]:
    matured: list[dict] = []
    for artifact in artifacts:
        issued_at = parse_utc(artifact["issued_at"])
        # Hurricane forecasts use 24-hour RI window + 24h buffer for best-track
        mature_time = issued_at + dt.timedelta(hours=48)
        if mature_time <= score_as_of:
            matured.append(artifact)
    return matured


def _storm_ref(artifact: dict, index: int) -> tuple[str, int]:
    return str(artifact.get("forecast_id")), index


def select_published(artifacts: list[dict]) -> tuple[cr.Selection, dict[tuple[str, int], str]]:
    """Amendment 7 rule 1 over every PUBLISHED storm number (``storms`` of each forecast file; a
    catch-up record is a shadow, never a published number): ``(selection, {storm ref: why it is
    not scored})`` -- storms that name no synoptic time are unverifiable, not selectable."""
    cands, no_cycle = [], {}
    for art in artifacts:
        made = cr.record_time(art)
        for i, storm in enumerate(art.get("storms") or []):
            sc = cr.storm_cycle(storm)
            if sc is None:
                no_cycle[_storm_ref(art, i)] = "the record names no synoptic time"
                continue
            cands.append(cr.Candidate(sc[0], sc[1], made, catch_up=bool(storm.get("catch_up")),
                                      rebuilt=bool(storm.get("rebuilt")), ref=_storm_ref(art, i)))
    sel = cr.select(cands)
    why = dict(no_cycle)
    for c, reason in sel.excluded:
        why[c.ref] = reason
    return sel, why


# Forecasts made before model identities were recorded per storm.
LEGACY_HURRICANE_MODEL = "hurricane_ri_v8_1"
# Reserved key in the calibration accumulator: the model version being pooled.
_CALIB_VERSION_KEY = "__model_version__"


def newest_model_version(artifacts: list[dict]) -> str:
    """The model a freshly fitted calibrator is for: one of the models behind the newest
    matured forecast that scored a storm.

    Since 2026-10 two models serve at once (NOAA's aids in the NHC basins, v8.2 elsewhere), so
    the newest forecast can name both. Taking its first storm's model made the pool follow
    whichever storm ranked first that day; instead, among the newest forecast's models, the
    one with the most matured storm forecasts overall is pooled (ties: by name). A calibrator
    stays one model's curve either way -- score_single_forecast pools only that model's storms.
    """
    counts: dict[str, int] = {}
    newest: set[str] | None = None
    for artifact in sorted(artifacts, key=lambda a: str(a.get("issued_at", "")), reverse=True):
        versions = [str(storm.get("model_version") or artifact.get("model_version"))
                    for storm in artifact.get("storms") or []
                    if storm.get("model_version") or artifact.get("model_version")]
        for v in versions:
            counts[v] = counts.get(v, 0) + 1
        if newest is None and versions:
            newest = set(versions)
    if not newest:
        return LEGACY_HURRICANE_MODEL
    return min(newest, key=lambda v: (-counts[v], v))


def calibrator_input(storm: dict) -> tuple[float, str]:
    """``(value, key)`` the trust-layer calibrator would be applied to for this storm.

    The live scorer hands the trust layer ``ri_probability`` (fetch_and_score.py, ``enrich_cells(...,
    prob_key="ri_probability")``); when it runs it replaces ``ri_probability`` and keeps the number it
    was given as ``raw_probability``. So the pool takes ``raw_probability`` when the trust layer ran and
    ``ri_probability`` otherwise. ``ri_probability_raw`` is NOT that number: it is v8.2's ensemble
    before v8.2's own calibration, which the trust layer never sees -- pooling it would fit a calibrator
    on one quantity and apply it to another."""
    if storm.get("raw_probability") is not None:
        return float(storm["raw_probability"]), "raw_probability"
    return float(storm.get("ri_probability", storm.get("result_probability", 0.0))), "ri_probability"


def _accumulate_calibration(calib_acc: dict, scores: np.ndarray, y_true: np.ndarray) -> None:
    """Pool (storm RI probability -> #positive, #total) into a histogram."""
    rscore = np.round(scores, 6)
    uniq, inv = np.unique(rscore, return_inverse=True)
    tot = np.bincount(inv)
    pos = np.bincount(inv, weights=y_true).astype(np.int64)
    for u, t, p in zip(uniq, tot, pos):
        key = float(u)
        slot = calib_acc.get(key)
        if slot is None:
            calib_acc[key] = [int(t), int(p)]
        else:
            slot[0] += int(t)
            slot[1] += int(p)


def write_calibration_dataset(output_dir: Path, calib_acc: dict, hazard: str = "hurricane") -> Path:
    keys = sorted(k for k in calib_acc.keys() if k != _CALIB_VERSION_KEY)
    total = [int(calib_acc[k][0]) for k in keys]
    pos = [int(calib_acc[k][1]) for k in keys]
    n = int(sum(total))
    payload = {
        "hazard": hazard,
        "model_version": calib_acc.get(_CALIB_VERSION_KEY),
        "unit": "storm-cycle (docs/HURRICANE_RI_V9_PROGRAM.md amendment 7, rule 1)",
        "n": n,
        "n_groups": len(keys),
        "base_rate": (sum(pos) / n) if n else 0.0,
        "scores": [round(float(k), 6) for k in keys],
        "pos": pos,
        "total": total,
    }
    path = output_dir / "calibration_dataset.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def score_single_forecast(artifact: dict, calib_acc: dict | None = None,
                          best_track_fetcher=None, set_aside: dict[tuple[str, int], str] | None = None,
                          tracks: dict | None = None) -> dict:
    """Score the storms of one forecast file, each against ITS OWN cycle (rule 4).

    ``set_aside`` (from select_published) names the storms of this file that are not the storm-cycle's
    scored record -- duplicates, preliminary records, storms with no cycle; they are listed with the
    reason and never scored. Without it every storm of the file is scored. Only storms with a
    VERIFIABLE outcome enter the Brier score and the calibration pool. A forecast with no storms has
    nothing to verify: its Brier is None (not a perfect 0.0) and it is counted as a null forecast.
    """
    storms = artifact.get("storms", [])
    forecast_id = artifact["forecast_id"]
    made_at = cr.record_time(artifact)

    if not storms:
        # No active TCs: no probability was issued for anything, so there is
        # nothing to verify. Scoring it as Brier 0.0 used to dilute the mean.
        return {
            "forecast_id": forecast_id,
            "issued_at": artifact["issued_at"],
            "n_storms": 0,
            "n_verified": 0,
            "n_unverifiable": 0,
            "n_ri_events": 0,
            "predictions": [],
            "set_aside": [],
            "brier": None,
            "null_forecast": True,
        }

    fetcher = best_track_fetcher or fetch_best_track_with_source
    tracks = {} if tracks is None else tracks
    set_aside = set_aside or {}
    predictions: list[dict] = []
    aside: list[dict] = []

    for i, storm in enumerate(storms):
        storm_id = str(storm.get("storm_id", "")).upper()
        ref = _storm_ref(artifact, i)
        sc = cr.storm_cycle(storm)
        if ref in set_aside or sc is None:
            aside.append({"storm_id": storm_id, "issue_time": storm.get("issue_time"),
                          "why": set_aside.get(ref, "the record names no synoptic time")})
            continue
        _, cycle = sc
        pred_prob = float(storm.get("ri_probability", storm.get("result_probability", 0.0)))
        cal_value, cal_key = calibrator_input(storm)

        if storm_id not in tracks:      # one fetch per storm per run
            tracks[storm_id] = fetcher(storm_id)
        records, source = tracks[storm_id]
        ri_occurred, vmax_t, vmax_t24 = check_ri_occurred(records, cycle)

        predictions.append({
            "storm_id": storm_id,
            "cycle": format_utc_z(cycle),
            "record_made_at": format_utc_z(made_at) if made_at else None,
            "lag_hours": cr.lag_hours(made_at, cycle) if made_at else None,
            "model_version": str(storm.get("model_version") or artifact.get("model_version")
                                 or LEGACY_HURRICANE_MODEL),
            # which family produced the number (NOAA's aids or v8.2); absent before 2026-10
            "ri_source": storm.get("ri_source", "v8.2"),
            "storm_name": storm.get("storm_name", storm_id),
            "predicted_ri_probability": round(pred_prob, 4),
            "calibrator_input_probability": round(cal_value, 4),
            "calibrator_input_key": cal_key,
            "ri_occurred": ri_occurred,
            "verifiable": ri_occurred is not None,
            "outcome_source": source,
            # the best-track wind at the storm's own synoptic time t and at t + 24 h
            "vmax_at_issue": vmax_t,
            "vmax_at_end": vmax_t24,
            "intensity_change_kt": (
                round(vmax_t24 - vmax_t, 1)
                if vmax_t is not None and vmax_t24 is not None
                else None
            ),
        })

    verified = [p for p in predictions if p["verifiable"]]
    y_true = np.array([1.0 if p["ri_occurred"] else 0.0 for p in verified])
    y_score = np.array([p["predicted_ri_probability"] for p in verified])

    if calib_acc is not None and verified:
        # Pool only storms of the model the calibrator is for: a calibrator is
        # one model's curve (v8.1 and v8.2 forecasts must never share one).
        want = calib_acc.get(_CALIB_VERSION_KEY)
        keep = [want is None or p["model_version"] == want for p in verified]
        raw = np.array([p["calibrator_input_probability"] for p, k in zip(verified, keep) if k],
                       dtype=np.float64)
        if raw.size:
            _accumulate_calibration(calib_acc, raw, y_true[np.array(keep, dtype=bool)])

    return {
        "forecast_id": forecast_id,
        "issued_at": artifact["issued_at"],
        "n_storms": len(storms),
        "n_verified": len(verified),
        "n_unverifiable": len(predictions) - len(verified),
        "n_ri_events": int(np.sum(y_true)) if verified else 0,
        "predictions": predictions,
        "set_aside": aside,
        # within one forecast file: a Brier over its few storms, kept for the record; the summary
        # pools storm-cycles instead of averaging these
        "brier": brier_score(y_true, y_score) if verified else None,
        "null_forecast": False,
    }


def pooled_metrics(predictions: list[dict]) -> dict:
    """Pooled over storm-cycles, with the counts that make them readable."""
    verified = [p for p in predictions if p.get("ri_occurred") is not None]
    y = np.array([1.0 if p["ri_occurred"] else 0.0 for p in verified])
    s = np.array([p["predicted_ri_probability"] for p in verified], dtype=np.float64)
    storms = {p["storm_id"] for p in verified}
    n_ev = int(y.sum()) if verified else 0
    out = {"n_storm_cycles": len(verified), "n_events": n_ev, "n_storms": len(storms),
           "brier": round(brier_score(y, s), 6) if verified else None,
           "mean_forecast": round(float(s.mean()), 6) if verified else None,
           "observed_rate": round(n_ev / len(verified), 6) if verified else None,
           "auc": None, "auc_withheld": None}
    if not verified:
        out["auc_withheld"] = "nothing verified"
    elif n_ev == 0 or n_ev == len(verified):
        out["auc_withheld"] = "only one outcome occurred: a ranking score is undefined"
    elif len(storms) < MIN_AUC_STORMS:
        out["auc_withheld"] = f"the cycles span {len(storms)} storms (< {MIN_AUC_STORMS})"
    else:
        out["auc"] = round(compute_auc(y, s), 6)
    return out


def summarize_results(per_forecast_results: list[dict], selection: cr.Selection | None = None) -> dict:
    """Pool the scored storm-cycles. Null forecasts (no storms) and unverifiable storms are counted,
    never averaged; storms set aside by rule 1 are counted by reason."""
    n_null = sum(1 for r in per_forecast_results if r.get("null_forecast"))
    n_with_storms = len(per_forecast_results) - n_null
    predictions = [p for r in per_forecast_results for p in r.get("predictions", [])]
    verified = [p for p in predictions if p.get("ri_occurred") is not None]
    unverifiable_by_basin: dict[str, int] = {}
    for p in predictions:
        if p.get("ri_occurred") is None:
            basin = str(p.get("storm_id", ""))[:2].lower() or "unknown"
            unverifiable_by_basin[basin] = unverifiable_by_basin.get(basin, 0) + 1
    set_aside: dict[str, int] = {}
    for r in per_forecast_results:
        for a in r.get("set_aside") or []:
            set_aside[a["why"]] = set_aside.get(a["why"], 0) + 1
    pooled = pooled_metrics(predictions)
    by_version: dict[str, list[dict]] = {}
    for p in predictions:
        by_version.setdefault(str(p.get("model_version")), []).append(p)
    total_ri = pooled["n_events"]
    return {
        "unit": "storm-cycle: one scored record per storm and synoptic time t, the first made at or after "
                "t + 3 h 30 min (docs/HURRICANE_RI_V9_PROGRAM.md amendment 7, rule 1)",
        "outcome": "best-track (BEST) wind at t + 24 h minus at t, both fixes present; >= 30 kt is RI (rule 4)",
        "n_null_forecasts": n_null,
        "n_forecasts_with_storms": n_with_storms,
        "total_storms_scored": len(verified),
        "total_storm_predictions": len(predictions),
        "n_unverifiable_predictions": len(predictions) - len(verified),
        "unverifiable_by_basin": unverifiable_by_basin,
        "total_ri_events": total_ri,
        "ri_rate": round(total_ri / len(verified), 4) if verified else None,
        "pooled": pooled,
        "pooled_brier_verified_predictions": (round(pooled["brier"], 4) if pooled["brier"] is not None else None),
        "by_model_version": {v: pooled_metrics(ps) for v, ps in sorted(by_version.items())},
        "set_aside_records": set_aside,
        "cycles_without_a_qualifying_record": (
            [f"{sid} {format_utc_z(t)}" for sid, t in selection.cycles_without_a_record()] if selection else []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score matured hurricane replay artifacts against best-track.",
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--score-as-of", default=None,
        help="UTC timestamp for determining maturity. Defaults to now.",
    )
    parser.add_argument(
        "--emit-calibration", action="store_true",
        help="Pool per-storm (probability -> outcome) into calibration_dataset.json "
        "for the calibrator fitter (scripts/fit_calibration.py).",
    )
    args = parser.parse_args(argv)

    score_as_of = (
        parse_utc(args.score_as_of)
        if args.score_as_of
        else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    artifacts = load_replay_artifacts(args.replay_dir)
    matured = matured_artifacts(artifacts, score_as_of)
    # rule 1 over EVERY record (a cycle's first qualifying record may not have matured yet; then the
    # cycle waits -- a later duplicate never stands in for it)
    selection, set_aside = select_published(artifacts)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scored_as_of": format_utc_z(score_as_of),
        "replay_dir": str(args.replay_dir.resolve()),
        "n_replay_artifacts": len(artifacts),
        "n_matured_forecasts": len(matured),
        "ri_threshold_kt": RI_THRESHOLD_KT,
        "ri_window_hours": RI_WINDOW_HOURS,
        "outcome_source": "ATCF working best track (NHC btk for al/ep/cp; UCAR RAL open b-decks for all basins)",
        "status": "ok",
    }

    calib_acc: dict | None = None
    if args.emit_calibration:
        calib_acc = {_CALIB_VERSION_KEY: newest_model_version(matured)}

    per_forecast_path = output_dir / "per_forecast_scores.jsonl"
    per_forecast_results: list[dict] = []

    if matured:
        tracks: dict = {}
        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                result = score_single_forecast(artifact, calib_acc=calib_acc, set_aside=set_aside, tracks=tracks)
                per_forecast_results.append(result)
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="hurricane")
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_n"] = int(sum(
                slot[0] for k, slot in calib_acc.items() if k != _CALIB_VERSION_KEY
            ))
            summary["calibration_model_version"] = calib_acc.get(_CALIB_VERSION_KEY)

        summary.update(summarize_results(per_forecast_results, selection))
    else:
        summary["status"] = "waiting_for_matured_forecasts"
        summary["message"] = (
            "No hurricane replay artifacts have fully matured yet."
        )
        per_forecast_path.write_text("", encoding="utf-8")

    summary_path = output_dir / "prospective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("Hurricane prospective scoring")
    print(f"  Replay dir:          {args.replay_dir.resolve()}")
    print(f"  Matured forecasts:   {len(matured)} / {len(artifacts)}")
    print(f"  Summary:             {summary_path}")
    if matured:
        pooled = summary.get("pooled") or {}
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Null forecasts:      {summary.get('n_null_forecasts')}")
        print(f"  Storm-cycles scored: {pooled.get('n_storm_cycles')} ({pooled.get('n_events')} RI, "
              f"{pooled.get('n_storms')} storms); set aside {summary.get('set_aside_records')}")
        print(f"  Pooled Brier:        {pooled.get('brier')}; AUC {pooled.get('auc')}"
              + (f" (withheld: {pooled.get('auc_withheld')})" if pooled.get("auc_withheld") else ""))
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
