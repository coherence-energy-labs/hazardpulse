from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PRIMARY_DOMAIN = "https://hazardpulse.com"
CONTACT_EMAIL = "josh@coherenceenergylabs.com"

LIVE_PULSE_PATH = DIST / "data" / "live-pulse.json"
LIVE_TORNADOES_PATH = DIST / "data" / "live-tornadoes.json"
LIVE_STORMS_PATH = DIST / "data" / "live-storms.json"
EQ_LEDGER_PATH = DIST / "data" / "earthquake-ledger.jsonl"
TO_LEDGER_PATH = DIST / "data" / "tornado-ledger.jsonl"
HU_LEDGER_PATH = DIST / "data" / "hurricane-ledger.jsonl"
REPLAY_DIR = DIST / "data" / "replay"
VERIFICATION_SUMMARY_PATH = DIST / "data" / "verification-summary.json"
VERIFICATION_DATA_DIR = DIST / "data" / "verification"
REPLAY_INDEX_PATH = DIST / "data" / "evidence" / "replay-index.json"
PREDICTION_LEDGER_PATH = DIST / "data" / "evidence" / "prediction-ledger.json"
PROVENANCE_PATH = DIST / "data" / "evidence" / "provenance-envelopes.json"
GATE_DECISIONS_PATH = DIST / "data" / "evidence" / "gate-decisions.json"
RESULTS_VERIFICATION_DIR = ROOT / "results" / "verification"
EQ_PROSPECTIVE_DIR = ROOT / "results" / "earthquake_prospective"
TO_PROSPECTIVE_DIR = ROOT / "results" / "tornado_prospective"
HU_PROSPECTIVE_DIR = ROOT / "results" / "hurricane_prospective"
# The verification workflow is scheduled every 4 h; over its last 60 runs (to 2026-10-01)
# GitHub's scheduler delivered a median gap of 5.7 h and a maximum of 10.9 h. A summary
# older than a full day means scoring has actually stopped, not that cron jittered.
PROSPECTIVE_STALE_AFTER = dt.timedelta(hours=24)
# A live skill score is quoted only once this many events have been observed for the model version: with
# fewer, a model that always says "no" scores as well as a skilful one (the summary quoted "live BSS 1.00"
# for 83 storm forecasts with zero tornadoes). The same floor as hazardpulse.site.pages.record.LIVE_MIN_EVENTS.
LIVE_MIN_EVENTS = 10
# Statuses that assert "no evaluator exists". Once a prospective scorer has written
# scored forecasts for a hazard, a rollup carrying one of these is a contradiction.
NO_EVALUATOR_STATUSES = frozenset({"matured_unscored_no_evaluator", "logging_live_no_evaluator"})
VERIFICATION_STATUS_BADGES = {
    "prospective_scored": "Scored",
    "prospective_stale": "Stale",
    "matured_unscored": "Backlog",
    "logging_waiting_maturity": "Waiting",
    "no_live_artifacts": "Missing",
    "inconsistent_with_prospective_ledger": "Inconsistent",
}
EQ_HONEST_RESULTS_PATH = ROOT / "results" / "earthquake_honest" / "v4_regional_honest_results.json"
EQ_SAME_LOCATION_PATH = ROOT / "results" / "earthquake_honest" / "same_location_auc.json"
TO_RETRO_RESULTS_PATH = ROOT / "results" / "definitive" / "definitive_results.json"

HAZARD_LABELS = {
    "eq": "Earthquake",
    "earthquake": "Earthquake",
    "hu": "Hurricane",
    "hurricane": "Hurricane",
    "to": "Tornado",
    "tornado": "Tornado",
}

HURRICANE_RETRO_FALLBACK = {
    "availability": "exact_model_benchmark",
    "label": "Retrospective benchmark available for the current live model version.",
    "model_version": "hurricane_ri_v8_1",
    "source_updated_at": "2026-03-13T03:00:00Z",
    "auc": 0.938,
    "brier": 0.034,
    "brier_skill_score": 0.290,
    "reliability_slope": 0.976,
    "n_cases": 9714,
}

# Temporal hold-out metrics for the served hurricane recipe, written by
# scripts/evaluate_hurricane_ri.py (members <= 2018, calibration 2019-2021, test 2022-2024,
# true 24-h RI). The v8.1 figures above were measured on v8.1's own 12-hour label.
HURRICANE_EVALUATION_PATH = ROOT / "results" / "calibration" / "hurricane_ri_evaluation.json"
_HURRICANE_EVAL_CANDIDATE = {
    "hurricane_ri_v8_2": "C_v8_2_heldout_newton",
    "hurricane_ri_v8_1_1": "B_v8_1_1_heldout_newton",
}


# The NOAA-aid model the NHC basins are served (scripts/hurricane_ri_stack.py): its identity is
# bound to the artifact's bytes, and its exact-model benchmark is the pre-registered program's
# read-once 2025 score -- shown only when the final report names this very artifact.
HURRICANE_STACK_PREFIX = "hurricane_ri_stack_v1-"
HURRICANE_STACK_FINAL_PATH = ROOT / "results" / "calibration" / "hurricane_ri_stack_final.json"


def _hurricane_stack_benchmark(model_version: str) -> dict | None:
    final = _read_json(HURRICANE_STACK_FINAL_PATH, {})
    if (final.get("phase") != "final"
            or (final.get("artifact") or {}).get("model_version") != model_version):
        return None
    choice = (final.get("selection") or {}).get("choice")
    result = (((final.get("results") or {}).get("all_cases") or {}).get("forecasts") or {}).get(choice)
    if not result:
        return None
    season = final.get("final_season")
    fit = (final.get("fit") or {}).get("seasons") or ["?", "?"]
    return {
        "availability": "exact_model_benchmark",
        "label": (
            f"Pre-registered program: candidate {choice} chosen on {fit[0]}-{fit[1]} by forward chaining, "
            f"then scored once on the {season} season ({result['n']} NHC-basin cycles, {result['events']} "
            "rapid-intensification events, intervals by storm)."
        ),
        "model_version": model_version,
        "source_updated_at": final.get("generated_at"),
        "auc": round(float(result["auc"]), 4),
        "auc_ci95": [round(float(x), 4) for x in result.get("auc_ci95", [])],
        "brier": round(float(result["brier"]), 5),
        "brier_skill_score": round(float(result["bss"]), 4),
        "reliability_slope": round(float(result["calibration_slope"]), 3),
        "n_cases": int(result["n"]),
    }


def _served_benchmark(hazard: str, model_version: str) -> dict | None:
    """The pre-registered, read-once final test of EXACTLY the live model version, from
    hazardpulse.verification.served_evidence (bound to the served artifact's bytes); None for any
    other version -- the homepage never shows one model's score beside another's forecasts."""
    from hazardpulse.verification import served_evidence as se

    try:
        ev = {"earthquake": se.earthquake_evidence, "tornado": se.tornado_evidence}[hazard]()
    except se.EvidenceError as exc:
        print(f"  Warning: {hazard} evidence not bound: {exc}")
        return None
    if not ev or not model_version:
        return None
    candidates = [ev] + ([ev["fallback"]] if hazard == "tornado" and ev.get("fallback") else [])
    match = next((c for c in candidates if c.get("model_version") == model_version), None)
    if match is None:
        return None
    t = match["test"]
    if hazard == "earthquake":
        ig = t["ig_per_target"]["value"]
        return {
            "availability": "exact_model_benchmark",
            "label": (f"Pre-registered programme ({ev['program']}): scored once on {t['when']} "
                      f"({t['n_issue_times']} issue times, {t['n_positive']} M6+ cell-windows); "
                      f"{ig:+.2f} nats of information per quake over a uniform map."),
            "model_version": model_version,
            "auc": round(float(t["auc"]["value"]), 4),
            "auc_ci95": [round(float(x), 4) for x in (t["auc"]["ci"] or [])],
            "brier_skill_score": round(float(t["bss"]["value"]), 4),
            "information_gain_per_event": round(float(ig), 4),
            "n_cases": int(t["n_cell_times"]),
        }
    return {
        "availability": "exact_model_benchmark",
        "label": (f"Pre-registered programme ({ev['program']}): every choice made on 2023, development test "
                  f"2024, scored once on every 2025 storm observation ({t['n']:,}; {t['pos']:,} tornadic)."),
        "model_version": model_version,
        "auc": round(float(t["auc"]), 4),
        "auc_ci95": [round(float(x), 4) for x in (t["auc_ci"] or [])],
        "brier_skill_score": round(float(t["bss"]), 4),
        "n_cases": int(t["n"]),
    }


def _hurricane_heldout_benchmark(model_version: str) -> dict | None:
    if model_version.startswith(HURRICANE_STACK_PREFIX):
        return _hurricane_stack_benchmark(model_version)
    report = _read_json(HURRICANE_EVALUATION_PATH, {})
    key = _HURRICANE_EVAL_CANDIDATE.get(model_version)
    result = (report.get("results") or {}).get(key) if key else None
    if not result:
        return None
    origin = report.get("origin", {})
    ci = result.get("ci95", {})

    def span(name: str) -> str:
        years = origin.get(name) or ["?", "?"]
        return f"{years[0]}-{years[1]}"

    return {
        "availability": "exact_model_benchmark",
        "label": (
            f"Temporal hold-out of this recipe: members trained on storms first seen "
            f"{span('members')}, calibrated on {span('calibration')}, scored once on "
            f"{span('test')} against the true 24-hour RI outcome."
        ),
        "model_version": model_version,
        "source_updated_at": report.get("generated_at"),
        "auc": round(float(result["auc"]), 4),
        "auc_ci95": [round(float(x), 4) for x in ci.get("auc", [])],
        "brier": round(float(result["brier"]), 5),
        "brier_skill_score": round(float(result["bss_vs_climatology"]), 4),
        "reliability_slope": round(float(result["calibration_slope"]), 3),
        "n_cases": int(result["n"]),
    }


def _read_json(path: Path, default: dict | list | None = None):
    if default is None:
        default = {}
    if not path.exists():
        return default.copy() if isinstance(default, dict) else list(default)
    try:
        return json.loads(path.read_text(encoding="utf-8").replace("\ufeff", ""))
    except json.JSONDecodeError:
        return default.copy() if isinstance(default, dict) else list(default)


def _write_json(path: Path, payload: dict | list) -> None:
    """LF bytes on every platform, and an unchanged file is not rewritten. Replay and evidence
    files are hashed and signed (.gitattributes marks them -text); write_text's newline
    translation made a Windows build rewrite two frozen replays to CRLF (2026-10-03) although
    their content had not changed."""
    data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# Per-build read cache for replay artifacts. Provenance and gate evaluation each
# read every forecast's replay artifact; caching halves that disk I/O. Read-only
# consumers only (the verification summary keeps its own reads because it mutates
# the artifact dicts). Cleared at the start of each build_site_artifacts() run.
_REPLAY_READ_CACHE: dict[str, dict] = {}


def _read_replay_cached(path: Path) -> dict:
    key = str(path)
    cached = _REPLAY_READ_CACHE.get(key)
    if cached is None:
        cached = _read_json(path, {})
        _REPLAY_READ_CACHE[key] = cached
    return cached


def _parse_utc(value: object) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith(" UTC"):
            text = text[:-4] + "Z"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _format_utc_z(value: dt.datetime | None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _forecast_id(prefix: str, issued_at: dt.datetime | None) -> str:
    issue = issued_at or dt.datetime.now(dt.timezone.utc)
    if issue.tzinfo is None:
        issue = issue.replace(tzinfo=dt.timezone.utc)
    issue = issue.astimezone(dt.timezone.utc)
    return f"{prefix}_fcst_{issue.strftime('%Y%m%d_%H%M')}"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _asset_ref(path: Path) -> str:
    return "/" + path.relative_to(DIST).as_posix()


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _hazard_label(value: object) -> str:
    return HAZARD_LABELS.get(str(value), str(value).replace("_", " ").title())


def _load_replay_index() -> dict:
    """The replay index, completed from the replay directory itself (the record of truth).

    It used to be only upserted with the current forecasts, so every replay written any other way (a
    backfill, a run whose index write was lost) was missing from it: 2,422 listed against 2,543 files on
    2026-10-04. Every file in the directory is listed now. The index is append-only (CI's
    check_append_only_monotonic), so an entry is never dropped, even if its file is absent from this
    checkout."""
    payload = _read_json(REPLAY_INDEX_PATH, {"generated_at": None, "items": []})
    listed = {str(i.get("forecast_id")): i for i in payload.get("items", []) if i.get("forecast_id")}
    on_disk = {p.stem: p for p in REPLAY_DIR.glob("*_fcst_*.json")} if REPLAY_DIR.exists() else {}
    items = list(listed.values()) + [{"forecast_id": fid, "replay_artifact": _asset_ref(path)}
                                     for fid, path in on_disk.items() if fid not in listed]
    items.sort(key=lambda item: item.get("forecast_id", ""))
    if [i.get("forecast_id") for i in items] != sorted(listed):
        payload["generated_at"] = _format_utc_z(dt.datetime.now(dt.timezone.utc))
    payload["items"] = items
    return payload


def _upsert_replay_index_item(index: dict, forecast_id: str, replay_path: Path) -> None:
    items = [item for item in index.get("items", []) if item.get("forecast_id") != forecast_id]
    items.append({"forecast_id": forecast_id, "replay_artifact": _asset_ref(replay_path)})
    items.sort(key=lambda item: item.get("forecast_id", ""))
    index["generated_at"] = _format_utc_z(dt.datetime.now(dt.timezone.utc))
    index["items"] = items


def _ensure_live_publish_artifacts() -> tuple[dict, dict]:
    # FAIL CLOSED: live-pulse.json is a tracked, pipeline-accumulated artifact. If it is
    # missing from disk the checkout is partial (e.g. sparse) -- fabricating an empty pulse
    # here would silently clobber the deployed live surfaces on the next commit, which is
    # exactly what happened in 71d56d53. Refuse instead of inventing state.
    if not LIVE_PULSE_PATH.exists():
        raise SystemExit(
            f"{LIVE_PULSE_PATH} is missing. This tracked live artifact must exist before the "
            "site can be rebuilt -- run from a full checkout (git sparse-checkout disable). "
            "Refusing to fabricate an empty live pulse."
        )
    pulse = _read_json(LIVE_PULSE_PATH, {"updated_at": None, "hazards": []})
    replay_index = _load_replay_index()

    def update_hazard(key: str, forecast_id: str | None) -> None:
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == key:
                hazard["forecast_id"] = forecast_id
                break

    def stamp_hazard(key: str, produced_at: dt.datetime | None) -> None:
        # Per-hazard freshness, taken from the hazard's OWN artifact. The pulse-level
        # updated_at is rewritten by every scorer, so a liveness check reading it saw the
        # hurricane product as fresh for four months after its last run (2026-05-26).
        if produced_at is None:
            return
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == key:
                hazard["updated_at"] = _format_utc_z(produced_at)
                break

    storms = _read_json(LIVE_STORMS_PATH, {})
    storms_updated = _parse_utc(storms.get("updated_at"))
    if storms_updated is not None:
        forecast_id = storms.get("forecast_id") or _forecast_id("hu", storms_updated)
        storms["forecast_id"] = forecast_id
        top_probability = 0.0
        if storms.get("storms"):
            top_probability = max(float(item.get("ri_probability", 0) or 0) for item in storms["storms"])
        artifact = {
            "forecast_id": forecast_id,
            "hazard": "hurricane",
            "issued_at": _format_utc_z(storms_updated),
            "model_version": storms.get("model_version", "hurricane_ri_v8_1"),
            "forecast_horizon_hours": 24,
            "n_active_storms": int(storms.get("n_active_storms", 0) or 0),
            "top_probability": round(top_probability, 4),
            "source_artifacts": ["/data/live-storms.json"],
            "storms": storms.get("storms", []),
        }
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        _write_json(replay_path, artifact)
        _upsert_replay_index_item(replay_index, forecast_id, replay_path)
        update_hazard("hu", forecast_id)
        stamp_hazard("hu", storms_updated)
        _write_json(LIVE_STORMS_PATH, storms)

    tornadoes = _read_json(LIVE_TORNADOES_PATH, {})
    tornadoes_updated = _parse_utc(tornadoes.get("updated_at"))
    if tornadoes_updated is not None:
        forecast_id = tornadoes.get("forecast_id") or _forecast_id("to", tornadoes_updated)
        tornadoes["forecast_id"] = forecast_id
        top_probability = 0.0
        if tornadoes.get("storms"):
            top_probability = max(
                float(item.get("tornado_probability", 0) or 0)
                for item in tornadoes["storms"]
            )
        artifact = {
            "forecast_id": forecast_id,
            "hazard": "tornado",
            "issued_at": _format_utc_z(tornadoes_updated),
            "model_version": tornadoes.get("model_version", "tornado_storm_v1_0"),
            "forecast_horizon_hours": 24,
            "scoring_tier": tornadoes.get("scoring_tier"),
            "scoring_tier_label": tornadoes.get("scoring_tier_label"),
            "coherence_source": tornadoes.get("coherence_source"),
            "n_active_storms": int(tornadoes.get("n_active_storms", 0) or 0),
            "top_probability": round(top_probability, 4),
            "source_artifacts": ["/data/live-tornadoes.json", "/data/tornado-storms.geojson"],
            "storms": tornadoes.get("storms", []),
        }
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        if replay_path.exists():
            # issued, so frozen: never rewritten (its hashes are in the provenance envelopes)
            _write_json(LIVE_TORNADOES_PATH, tornadoes)
        else:
            # ONE object for the record and the live file, so git stores the forecast once. Written as
            # two different documents, every tornado run committed the same ~1.2 MB of storms twice:
            # 74 KB compressed of a ~170 KB commit (measured 2026-10-05). The record takes the live
            # file's three extra fields; the live file (and /api/v1/live/tornado) gains the record's.
            for key in ("disclaimer", "updated_at", "recent_predictions"):
                if key in tornadoes:
                    artifact[key] = tornadoes[key]
            _write_json(replay_path, artifact)
            _write_json(LIVE_TORNADOES_PATH, artifact)
        _upsert_replay_index_item(replay_index, forecast_id, replay_path)
        update_hazard("to", forecast_id)
        stamp_hazard("to", tornadoes_updated)

    eq_hazard = next((item for item in pulse.get("hazards", []) if item.get("key") == "eq"), {})
    eq_forecast_id = eq_hazard.get("forecast_id")
    if eq_forecast_id:
        replay_path = REPLAY_DIR / f"{eq_forecast_id}.json"
        if replay_path.exists():
            _upsert_replay_index_item(replay_index, eq_forecast_id, replay_path)
            # The earthquake scorer's own artifact is the replay its forecast_id names.
            stamp_hazard("eq", _parse_utc(_read_json(replay_path, {}).get("issued_at")))

    _write_json(REPLAY_INDEX_PATH, replay_index)
    _write_json(LIVE_PULSE_PATH, pulse)
    return pulse, replay_index


def _collect_prediction_entries(pulse: dict, replay_index: dict) -> list[dict]:
    entries: list[dict] = []

    current_eq = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "eq"),
        None,
    )
    current_hu = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "hu"),
        None,
    )
    current_to = next(
        (hazard.get("forecast_id") for hazard in pulse.get("hazards", []) if hazard.get("key") == "to"),
        None,
    )

    for row in _read_jsonl(EQ_LEDGER_PATH):
        issued_at = _parse_utc(row.get("timestamp"))
        forecast_id = row.get("forecast_id") or _forecast_id("eq", issued_at)
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "earthquake",
                "issued_at": _format_utc_z(issued_at),
                "hash": f"sha256:{row.get('hash', '')}",
                "prev_hash": f"sha256:{row.get('prev_hash', '')}",
                "model_version": row.get("model_version"),
                "probability": row.get("top_probability", 0.0),
                "replay_artifact": _asset_ref(replay_path) if replay_path.exists() else None,
                "current": forecast_id == current_eq,
            }
        )

    for row in _read_jsonl(TO_LEDGER_PATH):
        issued_at = _parse_utc(row.get("timestamp"))
        forecast_id = row.get("forecast_id") or _forecast_id("to", issued_at)
        replay_path = REPLAY_DIR / f"{forecast_id}.json"
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "tornado",
                "issued_at": _format_utc_z(issued_at),
                "hash": f"sha256:{row.get('hash', '')}",
                "prev_hash": f"sha256:{row.get('prev_hash', '')}",
                "model_version": row.get("model_version"),
                "probability": row.get("top_probability", 0.0),
                "replay_artifact": _asset_ref(replay_path) if replay_path.exists() else None,
                "current": forecast_id == current_to,
            }
        )

    seen_hurricane: set[str] = set()
    hu_chain = {row.get("forecast_id"): row for row in _read_jsonl(HU_LEDGER_PATH)}
    for item in replay_index.get("items", []):
        forecast_id = item.get("forecast_id", "")
        if not forecast_id.startswith("hu_fcst_") or forecast_id in seen_hurricane:
            continue
        replay_artifact = item.get("replay_artifact")
        replay_path = DIST / replay_artifact.lstrip("/")
        if not replay_path.exists():
            continue
        artifact = _read_replay_cached(replay_path)
        chained = hu_chain.get(forecast_id)       # forecasts since the hurricane ledger began are chained
        entry_hash = f"sha256:{chained['hash']}" if chained else f"sha256:{_canonical_hash(artifact)}"
        seen_hurricane.add(forecast_id)
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "hurricane",
                "issued_at": artifact.get("issued_at", _format_utc_z(_parse_utc(artifact.get("issued_at")))),
                "hash": entry_hash,
                "prev_hash": f"sha256:{chained['prev_hash']}" if chained else None,
                "model_version": artifact.get("model_version"),
                "probability": artifact.get("top_probability", 0.0),
                "replay_artifact": replay_artifact,
                "current": forecast_id == current_hu,
            }
        )

    entries.sort(key=lambda item: item.get("issued_at", ""), reverse=True)
    return entries


def _build_provenance_envelopes(entries: list[dict]) -> list[dict]:
    envelopes: list[dict] = []
    for entry in entries:
        replay_artifact = entry.get("replay_artifact")
        if not replay_artifact:
            continue
        replay_path = DIST / replay_artifact.lstrip("/")
        if not replay_path.exists():
            continue
        artifact = _read_replay_cached(replay_path)
        hazard = entry.get("hazard")
        if hazard == "earthquake":
            input_manifest = {
                "source_catalog": artifact.get("source_catalog"),
                "forecast_domain": artifact.get("forecast_domain"),
                "feature_history_days": artifact.get("feature_history_days"),
                "recent_activity_days": artifact.get("recent_activity_days"),
            }
            sources = [artifact.get("source_catalog", {}).get("provider", "USGS FDSNWS")]
            transforms = [
                "catalog_ingest",
                "grid_binning",
                "coherence_feature_extraction",
                "singularity_scoring",
                "publish_artifact",
            ]
        elif hazard == "hurricane":
            input_manifest = {
                "source_artifacts": artifact.get("source_artifacts"),
                "n_active_storms": artifact.get("n_active_storms"),
                "storm_ids": [storm.get("storm_id") for storm in artifact.get("storms", [])],
            }
            sources = ["ATCF advisories", "NOAA tropical cyclone feed", "published live storm snapshot"]
            transforms = [
                "advisory_ingest",
                "feature_build",
                "ri_scoring",
                "calibration",
                "publish_artifact",
            ]
        else:
            input_manifest = {
                "source_artifacts": artifact.get("source_artifacts"),
                "scoring_tier": artifact.get("scoring_tier"),
                "coherence_source": artifact.get("coherence_source"),
                "n_active_storms": artifact.get("n_active_storms"),
                "storm_ids": [storm.get("storm_id") for storm in artifact.get("storms", [])],
            }
            sources = ["ProbSevere storm objects", "published live tornado snapshot"]
            if artifact.get("coherence_source") == "hrrr":
                sources.append("HRRR analysis")
            transforms = [
                "probsevere_ingest",
                "coherence_scoring",
                "storm_ranking",
                "publish_artifact",
            ]

        envelopes.append(
            {
                "provenance_id": f"prov_{entry['forecast_id']}",
                "forecast_id": entry["forecast_id"],
                "hazard": hazard,
                "model_version": artifact.get("model_version", entry.get("model_version")),
                "input_hash": f"sha256:{_canonical_hash(input_manifest)}",
                "output_hash": f"sha256:{_canonical_hash(artifact)}",
                "signed_at": artifact.get("issued_at", entry.get("issued_at")),
                "sources": sources,
                "transforms": transforms,
                "replay_artifact": replay_artifact,
            }
        )
    return envelopes


_GATE_CELL_DEG = {"earthquake": 2.0, "tornado": 2.0, "hurricane": None}


def _load_calibration_metrics() -> dict:
    """Per-hazard OUT-OF-SAMPLE calibration of the deployed calibrator.

    Reads ``metrics_after_heldout`` (fit_calibration's cross-fit). ``metrics_after`` is
    measured on the histogram the calibrator was fit to (ECE ~0 by construction), so
    a hazard without a held-out measurement counts as calibration-not-yet-measured.
    """
    from hazardpulse.trust.scoring import calibrator_admissible

    metrics: dict[str, dict] = {}
    for name in ("earthquake", "tornado", "hurricane"):
        rec = _read_json(ROOT / "results" / "calibration" / f"{name}_calibration.json", {})
        heldout = rec.get("metrics_after_heldout") if isinstance(rec, dict) else None
        # only a calibrator that is actually applied describes the published numbers: the tornado one
        # fitted on zero tornadoes passed G4 on ECE 0.037 while it was inflating them 567-1,822x
        if heldout and calibrator_admissible(rec)[0]:
            metrics[name] = heldout
    return metrics


def _live_data_age_seconds(entries: list[dict], now: dt.datetime) -> dict[str, float]:
    """How old each forecast's inputs had become while it was the public forecast.

    A forecast stays live until the next forecast of its hazard is issued (or until
    ``now`` for the current one); its inputs are at least as old as its issue time,
    so (live_until - issued_at) is a lower bound on the source-data age the public
    saw. This replaced a hard-coded 0.0 under which G1_SOURCE_FRESHNESS could not fail
    -- not even for a hurricane forecast left live for four months.
    """
    by_hazard: dict[str, list[tuple[dt.datetime, str]]] = {}
    for entry in entries:
        issued = _parse_utc(entry.get("issued_at"))
        if issued is not None:
            by_hazard.setdefault(str(entry.get("hazard")), []).append((issued, entry["forecast_id"]))
    ages: dict[str, float] = {}
    for items in by_hazard.values():
        items.sort()
        for i, (issued, fid) in enumerate(items):
            live_until = items[i + 1][0] if i + 1 < len(items) else now
            ages[fid] = max(0.0, (live_until - issued).total_seconds())
    return ages


def _gate_top_object(artifact: dict) -> dict:
    cells = artifact.get("active_cells")
    if isinstance(cells, list) and cells:
        return cells[0]
    storms = artifact.get("storms")
    if isinstance(storms, list) and storms:
        return storms[0]
    return {}


def _build_gate_decisions(entries: list[dict], pulse: dict,
                          now: dt.datetime | None = None) -> list[dict]:
    """Evaluate the real publish-gate spine per forecast (was: hardcoded 'pass').

    Each forecast is gated on its own replay artifact's trust fields — calibrated
    probability, [conf_lo, conf_hi] band, signed-receipt provenance — plus the
    deployed model's measured calibration. Forecasts without the trust layer yet
    DEGRADE (honest) instead of silently passing.
    """
    from hazardpulse.gates import GateContext, GateEngine

    engine = GateEngine()
    calib = _load_calibration_metrics()
    data_ages = _live_data_age_seconds(entries, now or dt.datetime.now(dt.timezone.utc))
    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    key_for_name = {"earthquake": "eq", "hurricane": "hu", "tornado": "to"}
    decisions: list[dict] = []
    for entry in entries:
        replay_ref = entry.get("replay_artifact")
        if not replay_ref:
            continue
        hazard_name = str(entry.get("hazard"))
        artifact = _read_replay_cached(DIST / replay_ref.lstrip("/"))
        top = _gate_top_object(artifact)
        receipt = top.get("receipt") if isinstance(top.get("receipt"), dict) else {}
        prob = top.get("probability")
        if prob is None:
            prob = top.get("tornado_probability", top.get("ri_probability"))
        if prob is None:
            prob = artifact.get("top_probability", 0.0)
        m = calib.get(hazard_name)
        ctx = GateContext(
            hazard=hazard_name,
            forecast_id=entry["forecast_id"],
            model_version=artifact.get("model_version") or entry.get("model_version"),
            model_sha256=receipt.get("model_sha256"),
            input_sha256=receipt.get("input_sha256"),
            receipt_sha256=top.get("receipt_sha256") or receipt.get("receipt_sha256"),
            replay_artifact=replay_ref,
            probability=prob,
            confidence_lo=top.get("confidence_lo"),
            confidence_hi=top.get("confidence_hi"),
            abstained=bool(top.get("abstained", False)),
            uncertainty_class=top.get("uncertainty_class"),
            lat=top.get("lat"),
            lon=top.get("lon"),
            cell_size_deg=_GATE_CELL_DEG.get(hazard_name),
            data_age_seconds=data_ages.get(entry["forecast_id"]),
            ece=(m or {}).get("ece"),
            brier_skill_score=(m or {}).get("brier_skill_score"),
            calibration_known=m is not None,
            risk_label=top.get("risk_band"),
        )
        decision = engine.evaluate(ctx, emitted_at=entry.get("issued_at"))
        payload = decision.as_dict()
        payload["issued_at"] = entry.get("issued_at")
        # Informational pulse-level context (kept from the prior stamping).
        hz = hazard_map.get(key_for_name.get(hazard_name, hazard_name), {})
        if hazard_name == "hurricane" and int(hz.get("n_active_storms", 0) or 0) == 0:
            payload["warnings"].append("no_active_tropical_cyclones_in_feed")
        if hazard_name == "tornado" and hz.get("coherence_source") == "probsevere":
            payload["warnings"].append("probsevere_coherence_fallback_active")
        decisions.append(payload)
    return decisions


def _count_link_mismatches(path: Path) -> tuple[int, int]:
    rows = _read_jsonl(path)
    mismatches = 0
    previous_hash = "0" * 64
    for index, row in enumerate(rows):
        expected = previous_hash if index else "0" * 64
        if str(row.get("prev_hash", "")) != expected:
            mismatches += 1
        previous_hash = str(row.get("hash", previous_hash))
    return len(rows), mismatches


def _artifact_hazard_key(artifact: dict) -> str | None:
    return {
        "earthquake": "eq",
        "hurricane": "hu",
        "tornado": "to",
        "eq": "eq",
        "hu": "hu",
        "to": "to",
    }.get(str(artifact.get("hazard", "")).strip())


def _artifact_mature_at(artifact: dict, default_horizon_hours: int | None = None) -> dt.datetime | None:
    """Maturity time of a replay artifact.

    ``default_horizon_hours`` applies to artifacts frozen before the horizon field
    existed; it must be the same default the hazard's prospective scorer uses, or
    the rollup's matured count and the scorer's scored count describe different sets.
    """
    issued_at = _parse_utc(artifact.get("issued_at"))
    if issued_at is None:
        return None
    if artifact.get("forecast_horizon_days") is not None:
        return issued_at + dt.timedelta(days=int(artifact.get("forecast_horizon_days", 0) or 0))
    if artifact.get("forecast_horizon_hours") is not None:
        return issued_at + dt.timedelta(hours=int(artifact.get("forecast_horizon_hours", 0) or 0))
    if default_horizon_hours is not None:
        return issued_at + dt.timedelta(hours=default_horizon_hours)
    return None


def _is_matured(artifact: dict, as_of: dt.datetime, default_horizon_hours: int | None = None) -> bool:
    mature_at = _artifact_mature_at(artifact, default_horizon_hours)
    return mature_at is not None and mature_at <= as_of


def _format_horizon(artifact: dict) -> str:
    if artifact.get("forecast_horizon_days") is not None:
        return f"{int(artifact.get('forecast_horizon_days', 0) or 0)} days"
    if artifact.get("forecast_horizon_hours") is not None:
        return f"{int(artifact.get('forecast_horizon_hours', 0) or 0)} hours"
    return "Unknown"


def _load_replay_artifacts_by_hazard() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"eq": [], "hu": [], "to": []}
    if not REPLAY_DIR.exists():
        return grouped
    for path in sorted(REPLAY_DIR.glob("*.json")):
        artifact = _read_json(path, {})
        if not artifact:
            continue
        hazard_key = _artifact_hazard_key(artifact)
        if hazard_key is None:
            continue
        artifact["_path"] = _asset_ref(path)
        mature_at = _artifact_mature_at(artifact)
        artifact["_mature_at"] = _format_utc_z(mature_at) if mature_at is not None else None
        grouped[hazard_key].append(artifact)
    for key in grouped:
        grouped[key].sort(key=lambda item: item.get("issued_at", ""))
    return grouped


def _legacy_verification_item(legacy_summary: dict, hazard_key: str, model_version: str | None) -> dict | None:
    for item in legacy_summary.get("hazards", []):
        item_key = str(item.get("key") or item.get("hazard") or "")
        normalized = {
            "earthquake": "eq",
            "hurricane": "hu",
            "tornado": "to",
            "eq": "eq",
            "hu": "hu",
            "to": "to",
        }.get(item_key)
        if normalized != hazard_key:
            continue
        if model_version and item.get("model_version") and item.get("model_version") != model_version:
            continue
        exact = item.get("exact_model_benchmark")
        if isinstance(exact, dict):
            merged = dict(exact)
            merged.setdefault("model_version", item.get("model_version"))
            merged.setdefault("auc", item.get("auc"))
            merged.setdefault("brier", item.get("brier"))
            merged.setdefault("brier_skill_score", item.get("brier_skill_score"))
            return merged
        return item
    return None


def _earthquake_related_benchmark() -> dict | None:
    honest = _read_json(EQ_HONEST_RESULTS_PATH, {})
    same_location = _read_json(EQ_SAME_LOCATION_PATH, {})
    global_metrics = honest.get("global_combined", {}).get("global_baseline", {})
    same_location_auc = (
        same_location.get("same_location_weighted_auc")
        or same_location.get("same_location_macro_auc")
    )
    if not global_metrics and not same_location_auc:
        return None
    return {
        "availability": "related_research_benchmark",
        "label": "Related research benchmark exists, but it is not yet bound to the current live model version.",
        "model_version": "earthquake_honest_regional_suite",
        "source_updated_at": honest.get("timestamp"),
        "global_auc": global_metrics.get("auc"),
        "global_brier": global_metrics.get("brier"),
        "same_location_auc": same_location_auc,
        "source_files": [
            "results/earthquake_honest/v4_regional_honest_results.json",
            "results/earthquake_honest/same_location_auc.json",
        ],
    }


def _tornado_related_benchmark() -> dict | None:
    payload = _read_json(TO_RETRO_RESULTS_PATH, {})
    full = payload.get("full", {})
    if not full:
        return None
    label = "A historical 2024 holdout benchmark exists for a related tornado GBT family, but not yet as an exact score for the live storm-object model."
    evaluated_base_rate = full.get("base_rate")
    if evaluated_base_rate is not None:
        # The holdout was downsampled (5 negatives per positive); Brier and BSS
        # scale with the base rate, so they do not transfer to the live stream.
        label += (
            f" Its Brier/BSS were measured at a downsampled base rate of {_fmt_float(evaluated_base_rate)}"
            " and are not comparable to live skill at the natural base rate."
        )
    return {
        "availability": "related_research_benchmark",
        "label": label,
        "model_version": payload.get("model"),
        "source_updated_at": payload.get("timestamp"),
        "evaluated_base_rate": evaluated_base_rate,
        "auc": full.get("auc"),
        "brier": full.get("brier"),
        "brier_skill_score": full.get("bss"),
        "source_files": [
            "results/definitive/definitive_results.json",
        ],
    }


def _prospective_scored_count(summary: dict) -> int:
    """Forecasts a prospective scorer actually scored (0 when it has not run)."""
    if not summary:
        return 0
    value = summary.get("n_scored_forecasts")
    if value is None:
        # Summaries written before n_scored_forecasts existed scored every matured forecast.
        value = summary.get("n_matured_forecasts", 0) if summary.get("status") == "ok" else 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _prospective_binding(
    summary: dict,
    *,
    hazard: str,
    summary_rel_path: str,
    n_matured_local: int,
    has_artifacts: bool,
    score_as_of: dt.datetime,
) -> dict:
    """Bind a hazard's rollup to the summary its prospective scorer wrote.

    The rollup must report what the scorer scored -- never a hardcoded 0 and never
    "no evaluator" once an evaluator has produced scores. Staleness is explicit.
    """
    n_scored = _prospective_scored_count(summary)
    scored_as_of = _parse_utc(summary.get("scored_as_of")) if summary else None
    age = (score_as_of - scored_as_of) if scored_as_of is not None else None
    stale = age is not None and age > PROSPECTIVE_STALE_AFTER
    if n_scored > 0 and not stale:
        status = "prospective_scored"
        label = f"Prospective live scoring is active for matured {hazard} forecasts."
    elif n_scored > 0:
        status = "prospective_stale"
        label = (
            f"Matured {hazard} forecasts were scored, but the latest scoring run "
            f"({_format_utc_z(scored_as_of)}) is older than {int(PROSPECTIVE_STALE_AFTER.total_seconds() // 3600)} h."
        )
    elif n_matured_local > 0:
        status = "matured_unscored"
        label = (
            f"Matured {hazard} forecasts exist, but the prospective scorer has not scored them yet"
            + (f" (scorer status: {summary.get('status')})." if summary else " (no scorer summary found).")
        )
    elif has_artifacts:
        status = "logging_waiting_maturity"
        label = f"{hazard.capitalize()} forecasts are being frozen; no window has matured yet."
    else:
        status = "no_live_artifacts"
        label = f"No live {hazard} replay artifacts are present."
    return {
        "status": status,
        "label": label,
        "n_scored": n_scored,
        "n_backlog": max(0, n_matured_local - n_scored),
        "prospective": {
            "summary_path": summary_rel_path if summary else None,
            "status": summary.get("status", "not_run") if summary else "not_run",
            "scored_as_of": summary.get("scored_as_of") if summary else None,
            "n_scored_forecasts": n_scored,
            "stale": bool(stale),
            "message": summary.get("message") if summary else None,
        },
    }


def _verification_rollup_violations(
    hazards: list[dict], summaries: dict[str, dict], *, allow_lag: bool = False
) -> list[str]:
    """Contradictions between rollups and the prospective summaries they were built from.

    Returns one message per violation; empty means consistent. A rollup that says
    0 scored (or "no evaluator") while its scorer's summary holds scored forecasts
    is exactly the defect this guards: it must surface, never pass silently.

    ``allow_lag`` is for comparing COMMITTED files written by different workflow
    runs: a rollup built from an older summary may legitimately differ in count
    (e.g. a later run excluded an SPC-outage day), but it may never read 0.
    """
    violations: list[str] = []
    for item in hazards:
        key = item.get("key")
        summary = summaries.get(key) or {}
        n_ledger = _prospective_scored_count(summary)
        if n_ledger <= 0:
            continue
        n_roll = int((item.get("forecast_storage") or {}).get("n_scored_forecasts", 0) or 0)
        label = _hazard_label(key)
        if n_roll == 0:
            violations.append(
                f"{label}: rollup reports 0 scored forecasts but the prospective summary scored {n_ledger}."
            )
        elif n_roll > n_ledger and not allow_lag:
            violations.append(
                f"{label}: rollup reports {n_roll} scored forecasts, more than the {n_ledger} the summary scored."
            )
        if item.get("verification_status") in NO_EVALUATOR_STATUSES or (
            (item.get("prospective") or {}).get("status") == "evaluator_missing"
        ):
            violations.append(
                f"{label}: rollup claims no evaluator although the prospective summary scored {n_ledger} forecasts."
            )
    return violations


def _build_verification_summary(pulse: dict) -> dict:
    score_as_of = _parse_utc(pulse.get("updated_at")) or dt.datetime.now(dt.timezone.utc)
    legacy_summary = _read_json(VERIFICATION_SUMMARY_PATH, {"hazards": []})
    replay_groups = _load_replay_artifacts_by_hazard()
    eq_rows, eq_mismatches = _count_link_mismatches(EQ_LEDGER_PATH)
    to_rows, to_mismatches = _count_link_mismatches(TO_LEDGER_PATH)
    hu_rows, hu_mismatches = _count_link_mismatches(HU_LEDGER_PATH)
    live_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    eq_related = _earthquake_related_benchmark()
    to_related = _tornado_related_benchmark()
    eq_prospective_summary = _read_json(EQ_PROSPECTIVE_DIR / "prospective_summary.json", {})
    to_prospective_summary = _read_json(TO_PROSPECTIVE_DIR / "prospective_summary.json", {})
    hu_prospective_summary = _read_json(HU_PROSPECTIVE_DIR / "prospective_summary.json", {})

    def next_mature_at(artifacts: list[dict]) -> str | None:
        future = []
        for artifact in artifacts:
            mature_at = _artifact_mature_at(artifact)
            if mature_at is not None and mature_at > score_as_of:
                future.append(mature_at)
        return _format_utc_z(min(future)) if future else None

    hazards: list[dict] = []

    eq_hazard = live_map.get("eq", {})
    eq_artifacts = replay_groups["eq"]
    eq_matured = [
        artifact
        for artifact in eq_artifacts
        if _artifact_mature_at(artifact) is not None and _artifact_mature_at(artifact) <= score_as_of
    ]
    eq_scored = int(eq_prospective_summary.get("n_matured_forecasts", 0) or 0)
    eq_backlog = max(0, len(eq_matured) - eq_scored)
    if eq_scored > 0:
        eq_status = "prospective_scored"
        eq_status_label = "Prospective live scoring is active for matured earthquake windows."
    elif eq_backlog > 0:
        eq_status = "matured_unscored"
        eq_status_label = "Matured earthquake windows exist, but they have not been scored yet."
    elif eq_artifacts:
        eq_status = "logging_waiting_maturity"
        eq_status_label = "Prospective earthquake logging is live; the 30-day windows have not matured yet."
    else:
        eq_status = "no_live_artifacts"
        eq_status_label = "No live earthquake replay artifacts are present."

    eq_latest = eq_artifacts[-1] if eq_artifacts else {}
    # The live record is THIS version's only: the pooled means mix every version that ever served
    # (2026-10: the replaced model's 600 windows, mean AUC 0.697, shown beside the new model).
    eq_version = str(eq_hazard.get("model_version") or "")
    eq_exact = _served_benchmark("earthquake", eq_version)
    eq_live = ((eq_prospective_summary.get("by_model_version") or {}).get(eq_version) or {}) if eq_scored > 0 else {}
    eq_live_n = int(eq_live.get("n_matured_forecasts", 0) or 0)
    eq_live_text = (
        f"Live record of this version: {eq_live_n} matured 30-day windows scored"
        + (f", mean AUC {_fmt_float(eq_live.get('mean_auc'))}." if eq_live.get("mean_auc") is not None else ".")
        if eq_live_n else "This version's live record starts when its first 30-day windows mature."
    )
    hazards.append(
        {
            "key": "eq",
            "hazard": "earthquake",
            "model_version": eq_hazard.get("model_version"),
            "verification_status": eq_status,
            "status_badge": {
                "prospective_scored": "Scored",
                "matured_unscored": "Backlog",
                "logging_waiting_maturity": "Waiting",
                "no_live_artifacts": "Missing",
            }.get(eq_status, "Status"),
            "verification_status_label": eq_status_label,
            "metric_source": (
                "retrospective_holdout_exact_model" if eq_exact
                else ("prospective_live" if eq_live_n else "no_exact_model_benchmark")
            ),
            "metric_source_label": (
                (eq_exact["label"] + " " + eq_live_text) if eq_exact
                else (eq_live_text if eq_live_n
                      else "The current live earthquake model does not yet have an exact benchmark in this repo.")
            ),
            "auc": eq_exact["auc"] if eq_exact else (eq_live.get("mean_auc") if eq_live_n else None),
            "brier": None if eq_exact else (eq_live.get("mean_brier") if eq_live_n else None),
            "brier_skill_score": eq_exact.get("brier_skill_score") if eq_exact else None,
            "homepage_line": (
                f"AUC {_fmt_float(eq_exact['auc'])} pre-registered final test"
                if eq_exact
                else (f"{eq_live_n} matured windows of this model scored" if eq_live_n
                      else f"{len(eq_artifacts)} frozen forecasts · {eq_backlog} matured backlog")
            ),
            "forecast_storage": {
                "n_replay_artifacts": len(eq_artifacts),
                "n_matured_forecasts": len(eq_matured),
                "n_scored_forecasts": eq_scored,
                "n_backlog": eq_backlog,
                "first_issued_at": eq_artifacts[0].get("issued_at") if eq_artifacts else None,
                "last_issued_at": eq_latest.get("issued_at"),
                "last_forecast_id": eq_latest.get("forecast_id"),
                "latest_replay_artifact": eq_latest.get("_path"),
                "forecast_horizon": _format_horizon(eq_latest) if eq_latest else "30 days",
                "next_mature_at": next_mature_at(eq_artifacts),
            },
            "ledger": {
                "supported": True,
                "path": "/data/earthquake-ledger.jsonl",
                "n_rows": eq_rows,
                "prev_hash_mismatches": eq_mismatches,
            },
            "prospective": {
                "summary_path": "results/earthquake_prospective/prospective_summary.json",
                "status": eq_prospective_summary.get("status", "not_run"),
                "scored_as_of": eq_prospective_summary.get("scored_as_of"),
                "message": eq_prospective_summary.get("message"),
                "top_5_hit_rate": eq_prospective_summary.get("top_5_hit_rate"),
                "this_model_version": eq_live or None,
            },
            "exact_model_benchmark": eq_exact,
            "related_benchmark": eq_related,
            "recommended_action": (
                "Keep freezing every earthquake forecast. Once the first 30-day windows mature, run the prospective scorer and use those scores to tune thresholds and calibration."
            ),
        }
    )

    hu_hazard = live_map.get("hu", {})
    hu_artifacts = replay_groups["hu"]
    # The hurricane and tornado scorers treat a horizon-less artifact as 24 h.
    hu_matured = [artifact for artifact in hu_artifacts if _is_matured(artifact, score_as_of, 24)]
    hu_binding = _prospective_binding(
        hu_prospective_summary,
        hazard="hurricane",
        summary_rel_path="results/hurricane_prospective/prospective_summary.json",
        n_matured_local=len(hu_matured),
        has_artifacts=bool(hu_artifacts),
        score_as_of=score_as_of,
    )
    hu_status = hu_binding["status"]
    hu_status_label = hu_binding["label"]
    hu_scored = hu_binding["n_scored"]
    hu_backlog = hu_binding["n_backlog"]
    hu_exact = _legacy_verification_item(
        legacy_summary,
        "hu",
        str(hu_hazard.get("model_version") or ""),
    )
    hu_heldout = _hurricane_heldout_benchmark(str(hu_hazard.get("model_version") or ""))
    if hu_heldout is not None:
        hu_exact = hu_heldout
    elif str(hu_hazard.get("model_version") or "") == HURRICANE_RETRO_FALLBACK["model_version"]:
        merged_exact = dict(HURRICANE_RETRO_FALLBACK)
        if isinstance(hu_exact, dict):
            merged_exact.update({key: value for key, value in hu_exact.items() if value is not None})
        hu_exact = merged_exact
    if hu_scored > 0:
        hu_status_label += (
            f" {int(hu_prospective_summary.get('total_storms_scored', 0) or 0)} storm forecasts scored against NHC best track;"
            f" {int(hu_prospective_summary.get('total_ri_events', 0) or 0)} rapid-intensification events observed."
        )

    hu_latest = hu_artifacts[-1] if hu_artifacts else {}
    hazards.append(
        {
            "key": "hu",
            "hazard": "hurricane",
            "model_version": hu_hazard.get("model_version"),
            "verification_status": hu_status,
            "status_badge": VERIFICATION_STATUS_BADGES.get(hu_status, "Status"),
            "verification_status_label": hu_status_label,
            "metric_source": "retrospective_holdout_exact_model" if hu_exact else "unverified_live_model",
            "metric_source_label": (
                "Exact-model retrospective benchmark is available."
                if hu_exact
                else "No exact-model benchmark is available in this repo."
            ),
            "auc": hu_exact.get("auc") if hu_exact else None,
            "brier": hu_exact.get("brier") if hu_exact else None,
            "brier_skill_score": hu_exact.get("brier_skill_score") if hu_exact else None,
            "homepage_line": (
                f"AUC {_fmt_float(hu_exact.get('auc'))} retrospective holdout"
                if hu_exact and hu_exact.get("auc") is not None
                else (
                    f"{hu_scored} matured forecasts scored"
                    if hu_scored > 0
                    else f"{len(hu_artifacts)} frozen forecasts · {hu_backlog} matured backlog"
                )
            ),
            "forecast_storage": {
                "n_replay_artifacts": len(hu_artifacts),
                "n_matured_forecasts": len(hu_matured),
                "n_scored_forecasts": hu_scored,
                "n_backlog": hu_backlog,
                "first_issued_at": hu_artifacts[0].get("issued_at") if hu_artifacts else None,
                "last_issued_at": hu_latest.get("issued_at"),
                "last_forecast_id": hu_latest.get("forecast_id"),
                "latest_replay_artifact": hu_latest.get("_path"),
                "forecast_horizon": _format_horizon(hu_latest) if hu_latest else "24 hours",
                "next_mature_at": next_mature_at(hu_artifacts),
            },
            "ledger": {
                "supported": hu_rows > 0,
                "path": "/data/hurricane-ledger.jsonl" if hu_rows > 0 else None,
                "n_rows": hu_rows,
                "prev_hash_mismatches": hu_mismatches,
                "since": "2026-10-03 (earlier hurricane forecasts are kept as replay files, unchained)",
            },
            "prospective": {
                **hu_binding["prospective"],
                "mean_brier": hu_prospective_summary.get("mean_brier") if hu_scored > 0 else None,
                "mean_auc": hu_prospective_summary.get("mean_auc") if hu_scored > 0 else None,
                "total_storms_scored": hu_prospective_summary.get("total_storms_scored") if hu_scored > 0 else None,
                "total_ri_events": hu_prospective_summary.get("total_ri_events") if hu_scored > 0 else None,
            },
            "exact_model_benchmark": (
                {
                    "availability": "exact_model_benchmark",
                    "label": hu_exact.get("label")
                    or "Retrospective benchmark available for the current live model version.",
                    "model_version": hu_exact.get("model_version"),
                    "source_updated_at": hu_exact.get("source_updated_at"),
                    "auc": hu_exact.get("auc"),
                    "auc_ci95": hu_exact.get("auc_ci95"),
                    "brier": hu_exact.get("brier"),
                    "brier_skill_score": hu_exact.get("brier_skill_score"),
                    "reliability_slope": hu_exact.get("reliability_slope"),
                    "n_cases": hu_exact.get("n_cases"),
                }
                if hu_exact
                else None
            ),
            "related_benchmark": None,
            "recommended_action": (
                "Keep scoring matured hurricane forecasts against NHC best track; live AUC stays undefined until at least one rapid-intensification event is observed, so promotion decisions still rest on the retrospective benchmark."
                if hu_scored > 0
                else "Run scripts/score_hurricane_prospective.py on the matured hurricane forecasts before using the model for calibration or promotion decisions."
            ),
        }
    )

    to_hazard = live_map.get("to", {})
    to_artifacts = replay_groups["to"]
    to_matured = [artifact for artifact in to_artifacts if _is_matured(artifact, score_as_of, 24)]
    to_binding = _prospective_binding(
        to_prospective_summary,
        hazard="tornado",
        summary_rel_path="results/tornado_prospective/prospective_summary.json",
        n_matured_local=len(to_matured),
        has_artifacts=bool(to_artifacts),
        score_as_of=score_as_of,
    )
    to_status = to_binding["status"]
    to_status_label = to_binding["label"]
    to_scored = to_binding["n_scored"]
    to_backlog = to_binding["n_backlog"]
    # Headline skill is the scorer's POOLED storm-level score with a named, causal
    # reference. Summaries written before pooled scoring existed only carry
    # per-forecast means; those are shown as AUC/Brier with no BSS rather than
    # promoting a mean of per-forecast in-sample skill scores to a headline.
    # ...and only the live model version's own storms: the summary's "pooled" mixes every version
    # that ever served (a summary without the per-version split shows no live skill at all).
    to_version = str(to_hazard.get("model_version") or "")
    to_exact = _served_benchmark("tornado", to_version)
    to_pooled = (((to_prospective_summary.get("pooled_by_model_version") or {}).get(to_version) or {})
                 if to_scored > 0 else {})
    if not to_pooled.get("n_storm_forecasts"):
        to_pooled = {}
    # (the raw-vs-calibrated split is not separated by version either, so it is not quoted)
    to_served: dict = {}
    if to_pooled:
        to_auc = to_pooled.get("auc")
        to_brier = to_pooled.get("brier")
        to_bss = to_pooled.get("bss_vs_causal_climatology")
        to_bss_reference = "causal_climatology"
        to_live_events = int(to_pooled.get("n_positive", 0) or 0)
        if to_live_events >= LIVE_MIN_EVENTS:
            to_metric_label = (
                f"Live record of this version: {int(to_pooled.get('n_storm_forecasts', 0) or 0):,} storm forecasts "
                f"from {int(to_pooled.get('n_forecasts', 0) or 0):,} closed live forecasts, {to_live_events:,} followed "
                f"by a tornado (SPC reports); Brier skill {_fmt_float(to_bss, 2)} against the base rate of outcomes "
                "that had closed before each forecast was issued."
            )
        else:
            to_metric_label = (
                f"Live record of this version: {int(to_pooled.get('n_storm_forecasts', 0) or 0):,} storm forecasts "
                f"closed, {to_live_events:,} followed by a tornado; too few tornadoes to score yet (a live score is "
                f"quoted from {LIVE_MIN_EVENTS})."
            )
            to_auc = to_brier = to_bss = None
    else:
        to_auc = to_brier = to_bss = to_bss_reference = None
        to_metric_label = ("No storm forecast of this model version has been scored yet."
                           if to_scored > 0 else "No matured tornado forecast has been scored yet.")
    if to_exact:
        to_metric_label = to_exact["label"] + " " + to_metric_label
    to_served_lines = []
    for mode_key, mode_label in (("raw_model_probability", "raw model"), ("calibrated_probability", "calibrated")):
        mode = to_served.get(mode_key) or {}
        if mode.get("n_storm_forecasts"):
            to_served_lines.append(
                f"{mode_label}: BSS {_fmt_float(mode.get('bss_vs_causal_climatology'), 3)} over {int(mode['n_storm_forecasts'])} storm forecasts"
            )
    if to_served_lines:
        to_metric_label += " Served-probability split: " + "; ".join(to_served_lines) + "."

    to_latest = to_artifacts[-1] if to_artifacts else {}
    hazards.append(
        {
            "key": "to",
            "hazard": "tornado",
            "model_version": to_hazard.get("model_version"),
            "verification_status": to_status,
            "status_badge": VERIFICATION_STATUS_BADGES.get(to_status, "Status"),
            "verification_status_label": to_status_label,
            "metric_source": (
                "retrospective_holdout_exact_model" if to_exact
                else ("prospective_live" if to_pooled else "no_exact_model_benchmark")
            ),
            "metric_source_label": to_metric_label,
            "auc": to_exact["auc"] if to_exact else to_auc,
            "brier": None if to_exact else to_brier,
            "brier_skill_score": to_exact["brier_skill_score"] if to_exact else to_bss,
            "brier_skill_score_reference": "test_year_base_rate" if to_exact else to_bss_reference,
            "live_this_version": (
                {"auc": to_auc, "brier": to_brier, "brier_skill_score": to_bss,
                 "n_storm_forecasts": int(to_pooled.get("n_storm_forecasts", 0) or 0),
                 "n_events": int(to_pooled.get("n_positive", 0) or 0)} if to_pooled else None
            ),
            "homepage_line": (
                f"AUC {_fmt_float(to_exact['auc'])} pre-registered 2025 test"
                + (f" · live BSS {_fmt_float(to_bss, 2)}" if to_bss is not None else "")
                if to_exact
                else (
                    f"{int(to_pooled.get('n_forecasts', 0) or 0)} matured forecasts of this model scored"
                    + (f" · BSS {_fmt_float(to_bss, 2)} vs climatology" if to_bss is not None else "")
                    if to_pooled
                    else f"{len(to_artifacts)} frozen forecasts · {to_backlog} matured backlog"
                )
            ),
            "forecast_storage": {
                "n_replay_artifacts": len(to_artifacts),
                "n_matured_forecasts": len(to_matured),
                "n_scored_forecasts": to_scored,
                "n_backlog": to_backlog,
                "first_issued_at": to_artifacts[0].get("issued_at") if to_artifacts else None,
                "last_issued_at": to_latest.get("issued_at"),
                "last_forecast_id": to_latest.get("forecast_id"),
                "latest_replay_artifact": to_latest.get("_path"),
                "forecast_horizon": _format_horizon(to_latest) if to_latest else "24 hours",
                "next_mature_at": next_mature_at(to_artifacts),
            },
            "ledger": {
                "supported": True,
                "path": "/data/tornado-ledger.jsonl",
                "n_rows": to_rows,
                "prev_hash_mismatches": to_mismatches,
            },
            "prospective": {
                **to_binding["prospective"],
                "outcome_time_convention": to_prospective_summary.get("outcome_time_convention") if to_scored > 0 else None,
                "pooled": (
                    {key: value for key, value in to_pooled.items() if key != "reliability"} if to_pooled else None
                ),
                "pooled_by_served_mode": (
                    {
                        mode_key: {key: value for key, value in mode.items() if key != "reliability"}
                        for mode_key, mode in to_served.items()
                    }
                    if to_served
                    else None
                ),
                "skill_reference": to_prospective_summary.get("skill_reference") if to_scored > 0 else None,
                "pooled_model_version": to_version if to_pooled else None,
            },
            "exact_model_benchmark": to_exact,
            "related_benchmark": to_related,
            "recommended_action": (
                "Judge the live tornado model on the pooled BSS against causal climatology (and its served-probability "
                "split), not on per-forecast means or the downsampled research holdout."
                if to_scored > 0
                else "Run scripts/score_tornado_prospective.py on the matured tornado forecasts before using the live model for calibration or threshold changes."
            ),
        }
    )

    # Guard: a rollup that contradicts the scorer summary it was built from is never
    # published as if it were true. The offending hazard is relabelled explicitly and
    # the violation is surfaced in the alerts (and fails `--verification-only`).
    violations = _verification_rollup_violations(
        hazards,
        {"eq": eq_prospective_summary, "hu": hu_prospective_summary, "to": to_prospective_summary},
    )
    for item in hazards:
        if any(message.startswith(_hazard_label(item["key"]) + ":") for message in violations):
            item["verification_status"] = "inconsistent_with_prospective_ledger"
            item["status_badge"] = VERIFICATION_STATUS_BADGES["inconsistent_with_prospective_ledger"]
            item["verification_status_label"] = (
                "This rollup contradicts the prospective scorer's summary; its counts and metrics are not trustworthy."
            )

    total_replays = sum(item["forecast_storage"]["n_replay_artifacts"] for item in hazards)
    total_matured = sum(item["forecast_storage"]["n_matured_forecasts"] for item in hazards)
    total_scored = sum(item["forecast_storage"]["n_scored_forecasts"] for item in hazards)
    total_backlog = sum(item["forecast_storage"]["n_backlog"] for item in hazards)
    total_chain_mismatches = eq_mismatches + to_mismatches
    alerts: list[str] = [f"Verification rollup inconsistent: {message}" for message in violations]
    if total_backlog:
        alerts.append(f"{total_backlog} matured forecast windows are waiting for scoring.")
    if total_chain_mismatches:
        alerts.append(f"{total_chain_mismatches} hash-chain mismatches were detected in raw ledgers.")
    for item in hazards:
        if item.get("exact_model_benchmark") is None and item.get("auc") is None:
            alerts.append(
                f"{_hazard_label(item['key'])}: no exact benchmark is attached to the current live model version."
            )

    summary = {
        "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
        "score_as_of": _format_utc_z(score_as_of),
        "system": {
            "frozen_forecasts": total_replays,
            "matured_forecasts": total_matured,
            "scored_forecasts": total_scored,
            "matured_unscored_backlog": total_backlog,
            "raw_chain_rows": eq_rows + to_rows,
            "hash_chain_mismatches": total_chain_mismatches,
            "exact_model_benchmarks": sum(1 for item in hazards if item.get("exact_model_benchmark")),
            "alerts": alerts,
            "rollup_violations": violations,
        },
        "hazards": hazards,
    }

    _write_json(VERIFICATION_SUMMARY_PATH, summary)
    _write_json(VERIFICATION_DATA_DIR / "ops-summary.json", summary)
    _write_json(RESULTS_VERIFICATION_DIR / "system" / "summary.json", summary)
    for item in hazards:
        _write_json(VERIFICATION_DATA_DIR / f"{item['key']}.json", item)
        _write_json(RESULTS_VERIFICATION_DIR / _hazard_label(item["key"]).lower() / "live_rollup.json", item)
    return summary


def _withhold_pulse_bands_excluding_probability(pulse: dict) -> bool:
    """A pulse band ``[conf_lo, conf_hi]`` is published only when it contains the pulse's own probability;
    otherwise both become null. The tornado scorer's 02:05Z run of 2026-10-05 put 0.06% beside a band of
    0.07%-0.07% (a Venn-Abers pair beside a Platt probability), and every deploy gates on
    tests/test_site_integrity.py, which asserts containment. The scorers withhold such bands themselves; this
    keeps a pulse written before that rule from being republished. Returns True when the pulse changed."""
    changed = False
    for hazard in pulse.get("hazards", []):
        lo, hi, p = hazard.get("conf_lo"), hazard.get("conf_hi"), hazard.get("probability")
        if lo is None or hi is None:
            continue
        try:
            ok = p is not None and 0.0 <= float(lo) <= float(p) <= float(hi) <= 1.0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            hazard["conf_lo"] = hazard["conf_hi"] = None
            changed = True
    return changed


def _stamp_pulse_from_records(pulse: dict, gate_decisions: list[dict]) -> bool:
    """Write each current forecast's REAL quality-check outcome into the live pulse.

    The scorers stamped ``gate_status = "pass"`` on every forecast before the checks had run, so the
    API said "pass" for forecasts the gate log recorded as "degrade" (found 2026-10-04). The checks run
    here, after the scorers; the pulse now carries their outcome, or "unknown" when no decision exists.
    Returns True when the pulse changed."""
    by_id = {d.get("forecast_id"): d for d in gate_decisions}
    changed = _withhold_pulse_bands_excluding_probability(pulse)
    for hazard in pulse.get("hazards", []):
        decision = by_id.get(hazard.get("forecast_id"))
        status = str(decision.get("decision")) if decision else "unknown"
        if hazard.get("gate_status") != status:
            hazard["gate_status"] = status
            changed = True
    # ...and its "delta" by the pages' one definition: the headline chance minus the same headline in the
    # previous frozen forecast of that hazard (the scorers' own deltas disagreed with the ledger)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from hazardpulse.site.data import SiteData

    heads = SiteData(root=ROOT, dist=DIST).headlines
    for hazard in pulse.get("hazards", []):
        head = heads.get(hazard.get("key"))
        if head is None or head.forecast_id != hazard.get("forecast_id"):
            continue
        delta = (round(head.probability - head.previous, 4)
                 if head.probability is not None and head.previous is not None else None)
        if hazard.get("delta") != delta:
            hazard["delta"] = delta
            changed = True
    return changed


def _render_site() -> list[str]:
    """Every page, from the artifacts just written (hazardpulse.site.build: one renderer for the whole
    site, so whichever scorer ran, every page agrees with the same files). DIST and ROOT are read at call
    time, so a test that points this module at a scratch tree renders into that tree."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from hazardpulse.site import build as site_build

    changed = site_build.build_site(DIST, ROOT)
    if changed:
        print(f"  Site: {len(changed)} files re-rendered")
    return changed


def _publish_signing_key() -> None:
    """Publish the Ed25519 public key so anyone can verify forecast receipts
    independently (scripts/verify_forecast.py). No-op if no key is configured."""
    try:
        from hazardpulse.trust.scoring import load_signer, publish_public_key
        signer = load_signer()
        if signer is not None:
            publish_public_key(signer, DIST / "data" / "evidence" / "public-key.json")
    except Exception:
        pass


def build_site_artifacts() -> dict:
    _REPLAY_READ_CACHE.clear()
    pulse, replay_index = _ensure_live_publish_artifacts()
    entries = _collect_prediction_entries(pulse, replay_index)
    envelopes = _build_provenance_envelopes(entries)
    gate_decisions = _build_gate_decisions(entries, pulse)
    verification_summary = _build_verification_summary(pulse)
    _publish_signing_key()

    _write_json(
        PREDICTION_LEDGER_PATH,
        {
            "mode": "append_only",
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "entries": entries,
        },
    )
    _write_json(
        PROVENANCE_PATH,
        {
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "envelopes": envelopes,
        },
    )
    _write_json(
        GATE_DECISIONS_PATH,
        {
            "gate_set_version": "2026.04",
            "generated_at": _format_utc_z(dt.datetime.now(dt.timezone.utc)),
            "decisions": gate_decisions,
        },
    )
    if _stamp_pulse_from_records(pulse, gate_decisions):
        _write_json(LIVE_PULSE_PATH, pulse)
    _render_site()

    return {
        "pulse": pulse,
        "verification_summary": verification_summary,
        "entries": entries,
        "envelopes": envelopes,
        "gate_decisions": gate_decisions,
        "replay_index": replay_index,
    }


def build_verification_rollups() -> dict:
    """Rebuild ONLY the verification rollups (+ the /verification page) from the
    committed live pulse, replay artifacts and prospective scorer summaries.

    This is what the verification-scoring workflow runs right after the scorers,
    so results/verification/** is regenerated in the same commit as the
    summaries it is derived from.
    """
    if not LIVE_PULSE_PATH.exists():
        raise SystemExit(
            f"{LIVE_PULSE_PATH} is missing; refusing to build verification rollups without the live pulse."
        )
    _REPLAY_READ_CACHE.clear()
    pulse = _read_json(LIVE_PULSE_PATH, {"updated_at": None, "hazards": []})
    summary = _build_verification_summary(pulse)
    # the prospective summaries this workflow just rewrote feed the track record and the model-evidence
    # blocks (e.g. a challenger's error budget and matured count); re-render the site in the same commit,
    # or main stays inconsistent with its own results until some other scorer runs a full build
    _render_site()
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build HazardPulse site artifacts.")
    parser.add_argument(
        "--verification-only",
        action="store_true",
        help="Rebuild only the verification rollups and page; exit 1 if a rollup contradicts "
        "its prospective scorer summary.",
    )
    args = parser.parse_args(argv)

    if args.verification_only:
        summary = build_verification_rollups()
    else:
        summary = build_site_artifacts()["verification_summary"]
        print("Built HazardPulse evidence, replay index, verification summary and every page.")
    for item in summary.get("hazards", []):
        storage = item.get("forecast_storage", {})
        print(
            f"  {_hazard_label(item.get('key'))}: {item.get('verification_status')} -- "
            f"{storage.get('n_scored_forecasts', 0)} scored / {storage.get('n_matured_forecasts', 0)} matured"
        )
    violations = (summary.get("system") or {}).get("rollup_violations") or []
    if violations:
        for message in violations:
            print(f"VERIFICATION ROLLUP INCONSISTENT: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
