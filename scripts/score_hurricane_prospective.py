#!/usr/bin/env python3
"""Score frozen hurricane replay artifacts against realized intensity changes.

Reads replay artifacts from dist/data/replay/hu_fcst_*.json, evaluates
only forecasts whose windows have fully matured, fetches NHC ATCF
best-track data to determine if rapid intensification actually occurred,
and writes:

- results/hurricane_prospective/prospective_summary.json
- results/hurricane_prospective/per_forecast_scores.jsonl

Outcomes are three-valued. A storm whose best track cannot be found, or whose
track does not bracket [issue, issue + 24 h], is UNVERIFIABLE -- it is counted and
reported, never scored as "no RI" (that used to turn every 404 -- e.g. all West
Pacific storms, which NHC's btk folder does not carry -- into a confident
negative). Forecasts with no active storms carry no prediction to verify and are
counted separately, not averaged into the Brier score as perfect 0.0 entries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.core.metrics import roc_auc  # noqa: E402
from hazardpulse.data.http import fetch_bytes  # noqa: E402
from hazardpulse.hurricane.atcf import (  # noqa: E402
    ATCF_ROOT,
    ATCFRecord,
    DEFAULT_ANALYSIS_PRIORITY,
    parse_atcf_deck,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hurricane_prospective"

# RI definition: ≥30 kt increase in 24 hours (NHC standard)
RI_THRESHOLD_KT = 30.0
RI_WINDOW_HOURS = 24
# A best-track fix further than this from the issue / window-end time cannot
# stand in for the intensity at that time.
MAX_FIX_OFFSET_HOURS = 12.0

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
    issue_time: dt.datetime,
    window_hours: int = RI_WINDOW_HOURS,
    threshold_kt: float = RI_THRESHOLD_KT,
) -> tuple[bool | None, float | None, float | None]:
    """Did rapid intensification occur in [issue_time, issue_time + window]?

    Returns (ri_occurred, vmax_at_issue, vmax_at_window_end). ``ri_occurred`` is
    None -- UNVERIFIABLE -- when the best track has no fix within
    MAX_FIX_OFFSET_HOURS of either end of the window: absence of a track is not
    evidence that the storm did not intensify.
    """
    window_end = issue_time + dt.timedelta(hours=window_hours)

    # Find analysis records (BEST track, tau=0)
    best_records = [
        r for r in records
        if r.model in ("BEST", "CARQ", "OFCL")
        and r.tau_hours == 0
        and r.vmax_kt is not None
    ]
    if not best_records:
        return None, None, None

    # Find vmax closest to issue time
    best_at_issue = min(
        best_records,
        key=lambda r: abs((r.cycle - issue_time).total_seconds()),
    )
    dt_issue = abs((best_at_issue.cycle - issue_time).total_seconds()) / 3600.0
    if dt_issue > MAX_FIX_OFFSET_HOURS:  # too far from issue time
        return None, None, None

    vmax_issue = best_at_issue.vmax_kt

    # Find vmax closest to issue + 24h
    best_at_end = min(
        best_records,
        key=lambda r: abs((r.cycle - window_end).total_seconds()),
    )
    dt_end = abs((best_at_end.cycle - window_end).total_seconds()) / 3600.0
    if dt_end > MAX_FIX_OFFSET_HOURS:
        return None, vmax_issue, None

    vmax_end = best_at_end.vmax_kt

    dv = vmax_end - vmax_issue
    return dv >= threshold_kt, vmax_issue, vmax_end


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
                          best_track_fetcher=None) -> dict:
    """Score a single hurricane forecast.

    For forecasts with storms: fetch the best track and check whether RI occurred.
    Only storms with a VERIFIABLE outcome enter the Brier score, the AUC and the
    calibration pool. A forecast with no storms has nothing to verify: its Brier
    is None (not a perfect 0.0) and it is counted as a null forecast.
    """
    issued_at = parse_utc(artifact["issued_at"])
    storms = artifact.get("storms", [])
    forecast_id = artifact["forecast_id"]

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
            "auc": float("nan"),
            "brier": None,
            "null_forecast": True,
        }

    fetcher = best_track_fetcher or fetch_best_track_with_source
    predictions: list[dict] = []
    tracks: dict[str, tuple[list[ATCFRecord], str | None]] = {}

    for storm in storms:
        storm_id = storm.get("storm_id", "")
        pred_prob = float(storm.get("ri_probability", storm.get("result_probability", 0.0)))

        if storm_id not in tracks:      # a storm can be listed twice in one artifact
            tracks[storm_id] = fetcher(storm_id)
        records, source = tracks[storm_id]
        ri_occurred, vmax_issue, vmax_end = check_ri_occurred(records, issued_at)

        predictions.append({
            "storm_id": storm_id,
            "model_version": str(storm.get("model_version") or artifact.get("model_version")
                                 or LEGACY_HURRICANE_MODEL),
            # which family produced the number (NOAA's aids or v8.2); absent before 2026-10
            "ri_source": storm.get("ri_source", "v8.2"),
            "storm_name": storm.get("storm_name", storm_id),
            "predicted_ri_probability": round(pred_prob, 4),
            "raw_ri_probability": round(float(storm.get("raw_probability", pred_prob)), 4),
            "ri_occurred": ri_occurred,
            "verifiable": ri_occurred is not None,
            "outcome_source": source,
            "vmax_at_issue": vmax_issue,
            "vmax_at_end": vmax_end,
            "intensity_change_kt": (
                round(vmax_end - vmax_issue, 1)
                if vmax_issue is not None and vmax_end is not None
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
        raw = np.array([p["raw_ri_probability"] for p, k in zip(verified, keep) if k],
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
        "auc": compute_auc(y_true, y_score) if verified else float("nan"),
        "brier": brier_score(y_true, y_score) if verified else None,
        "null_forecast": False,
    }


def summarize_results(per_forecast_results: list[dict]) -> dict:
    """Aggregate per-forecast scores. Only VERIFIED storm predictions are scored:
    null forecasts (no storms) and unverifiable storms are counted, never averaged."""
    n_null = sum(1 for r in per_forecast_results if r.get("null_forecast"))
    n_with_storms = len(per_forecast_results) - n_null
    aucs = [r["auc"] for r in per_forecast_results
            if r.get("auc") is not None and math.isfinite(r["auc"])]
    briers = [r["brier"] for r in per_forecast_results
              if r.get("brier") is not None and math.isfinite(r["brier"])]
    predictions = [p for r in per_forecast_results for p in r.get("predictions", [])]
    verified = [p for p in predictions if p.get("ri_occurred") is not None]
    unverifiable_by_basin: dict[str, int] = {}
    for p in predictions:
        if p.get("ri_occurred") is None:
            basin = str(p.get("storm_id", ""))[:2].lower() or "unknown"
            unverifiable_by_basin[basin] = unverifiable_by_basin.get(basin, 0) + 1
    total_ri = sum(1 for p in verified if p["ri_occurred"])
    pooled_brier = (
        float(np.mean([(p["predicted_ri_probability"] - (1.0 if p["ri_occurred"] else 0.0)) ** 2
                       for p in verified]))
        if verified else None
    )
    return {
        "n_null_forecasts": n_null,
        "n_forecasts_with_storms": n_with_storms,
        "n_forecasts_with_verified_storms": len(briers),
        "total_storms_scored": len(verified),
        "total_storm_predictions": len(predictions),
        "n_unverifiable_predictions": len(predictions) - len(verified),
        "unverifiable_by_basin": unverifiable_by_basin,
        "total_ri_events": total_ri,
        "ri_rate": round(total_ri / len(verified), 4) if verified else None,
        "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "median_auc": round(float(np.median(aucs)), 4) if aucs else None,
        # Mean over forecasts that verified at least one storm (null and fully
        # unverifiable forecasts excluded), plus the per-prediction pooled Brier.
        "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
        "pooled_brier_verified_predictions": (
            round(pooled_brier, 4) if pooled_brier is not None else None
        ),
        "n_forecasts_with_valid_auc": len(aucs),
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
        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                result = score_single_forecast(artifact, calib_acc=calib_acc)
                per_forecast_results.append(result)
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="hurricane")
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_n"] = int(sum(
                slot[0] for k, slot in calib_acc.items() if k != _CALIB_VERSION_KEY
            ))
            summary["calibration_model_version"] = calib_acc.get(_CALIB_VERSION_KEY)

        summary.update(summarize_results(per_forecast_results))
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
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Null forecasts:      {summary.get('n_null_forecasts')}")
        print(f"  With storms:         {summary.get('n_forecasts_with_storms')}")
        print(f"  Mean AUC:            {summary.get('mean_auc')}")
        print(f"  Mean Brier:          {summary.get('mean_brier')}")
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
