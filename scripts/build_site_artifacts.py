from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
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
REPLAY_DIR = DIST / "data" / "replay"
VERIFICATION_SUMMARY_PATH = DIST / "data" / "verification-summary.json"
VERIFICATION_DATA_DIR = DIST / "data" / "verification"
REPLAY_INDEX_PATH = DIST / "data" / "evidence" / "replay-index.json"
PREDICTION_LEDGER_PATH = DIST / "data" / "evidence" / "prediction-ledger.json"
PROVENANCE_PATH = DIST / "data" / "evidence" / "provenance-envelopes.json"
GATE_DECISIONS_PATH = DIST / "data" / "evidence" / "gate-decisions.json"
EVIDENCE_PAGE_PATH = DIST / "evidence" / "index.html"
VERIFICATION_PAGE_PATH = DIST / "verification" / "index.html"
SITEMAP_PATH = DIST / "sitemap.xml"
FEED_PATH = DIST / "feed.xml"
RESULTS_VERIFICATION_DIR = ROOT / "results" / "verification"
EQ_PROSPECTIVE_DIR = ROOT / "results" / "earthquake_prospective"
EQ_HONEST_RESULTS_PATH = ROOT / "results" / "earthquake_honest" / "v4_regional_honest_results.json"
EQ_SAME_LOCATION_PATH = ROOT / "results" / "earthquake_honest" / "same_location_auc.json"
TO_RETRO_RESULTS_PATH = ROOT / "results" / "definitive" / "definitive_results.json"

ROUTES = [
    ("/", "daily", "1.0"),
    ("/live/", "hourly", "1.0"),
    ("/live/earthquake/", "hourly", "0.9"),
    ("/live/hurricane/", "hourly", "0.9"),
    ("/live/tornado/", "hourly", "0.9"),
    ("/verification/", "daily", "0.8"),
    ("/evidence/", "daily", "0.8"),
    ("/methods/", "weekly", "0.7"),
    ("/registry/", "daily", "0.8"),
    ("/api/", "weekly", "0.7"),
    ("/ops/status/", "hourly", "0.7"),
    ("/legal/disclaimer/", "monthly", "0.4"),
]

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _format_http_date(value: dt.datetime | None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


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


def _short_hash(value: str | None) -> str:
    if not value:
        return "--"
    clean = value.replace("sha256:", "")
    if len(clean) <= 16:
        return clean
    return f"{clean[:8]}..{clean[-8:]}"


def _hazard_label(value: object) -> str:
    return HAZARD_LABELS.get(str(value), str(value).replace("_", " ").title())


def _load_replay_index() -> dict:
    payload = _read_json(REPLAY_INDEX_PATH, {"generated_at": None, "items": []})
    payload.setdefault("items", [])
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
        _write_json(replay_path, artifact)
        _upsert_replay_index_item(replay_index, forecast_id, replay_path)
        update_hazard("to", forecast_id)
        _write_json(LIVE_TORNADOES_PATH, tornadoes)

    eq_hazard = next((item for item in pulse.get("hazards", []) if item.get("key") == "eq"), {})
    eq_forecast_id = eq_hazard.get("forecast_id")
    if eq_forecast_id:
        replay_path = REPLAY_DIR / f"{eq_forecast_id}.json"
        if replay_path.exists():
            _upsert_replay_index_item(replay_index, eq_forecast_id, replay_path)

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
    for item in replay_index.get("items", []):
        forecast_id = item.get("forecast_id", "")
        if not forecast_id.startswith("hu_fcst_") or forecast_id in seen_hurricane:
            continue
        replay_artifact = item.get("replay_artifact")
        replay_path = DIST / replay_artifact.lstrip("/")
        if not replay_path.exists():
            continue
        artifact = _read_replay_cached(replay_path)
        entry_hash = f"sha256:{_canonical_hash(artifact)}"
        seen_hurricane.add(forecast_id)
        entries.append(
            {
                "forecast_id": forecast_id,
                "hazard": "hurricane",
                "issued_at": artifact.get("issued_at", _format_utc_z(_parse_utc(artifact.get("issued_at")))),
                "hash": entry_hash,
                "prev_hash": None,
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
    """Per-hazard deployed-model calibration (results/models/<hazard>_calibration.json)."""
    metrics: dict[str, dict] = {}
    for name in ("earthquake", "tornado", "hurricane"):
        rec = _read_json(ROOT / "results" / "calibration" / f"{name}_calibration.json", {})
        after = rec.get("metrics_after") if isinstance(rec, dict) else None
        if after:
            metrics[name] = after
    return metrics


def _gate_top_object(artifact: dict) -> dict:
    cells = artifact.get("active_cells")
    if isinstance(cells, list) and cells:
        return cells[0]
    storms = artifact.get("storms")
    if isinstance(storms, list) and storms:
        return storms[0]
    return {}


def _build_gate_decisions(entries: list[dict], pulse: dict) -> list[dict]:
    """Evaluate the real publish-gate spine per forecast (was: hardcoded 'pass').

    Each forecast is gated on its own replay artifact's trust fields — calibrated
    probability, [conf_lo, conf_hi] band, signed-receipt provenance — plus the
    deployed model's measured calibration. Forecasts without the trust layer yet
    DEGRADE (honest) instead of silently passing.
    """
    from hazardpulse.gates import GateContext, GateEngine

    engine = GateEngine()
    calib = _load_calibration_metrics()
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
            data_age_seconds=0.0,  # source data was fresh at issue time
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
    """LINKAGE ONLY: does each row's stored prev_hash equal the previous row's stored hash?

    This is NOT an integrity check and must never be published as one. It compares two stored
    values to each other and never recomputes a digest from row content, so it returns 0 for a
    rewritten entry, a backdated timestamp, a wholly re-chained history and a truncated tail
    alike. `_ledger_integrity()` below is the real check; this one is kept because the number it
    produces ("do the pointers agree?") is still worth reporting SEPARATELY, and because the
    negative tests use it to prove each forgery fixture is one the linkage check cannot see.
    """
    rows = _read_jsonl(path)
    mismatches = 0
    previous_hash = "0" * 64
    for index, row in enumerate(rows):
        expected = previous_hash if index else "0" * 64
        if str(row.get("prev_hash", "")) != expected:
            mismatches += 1
        previous_hash = str(row.get("hash", previous_hash))
    return len(rows), mismatches


def _load_ledger_verifier():
    import importlib.util
    import sys

    name = "hazardpulse_ledger_verifier"
    if name in sys.modules:
        return sys.modules[name]
    module_path = Path(__file__).resolve().parent / "verify_ledger_chain.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    # MUST be registered BEFORE exec_module: @dataclass resolves cls.__module__ through
    # sys.modules, and raises AttributeError on None if the module is not there yet.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ledger_integrity(path: Path) -> dict:
    """RECOMPUTE every digest, verify the linkage and the anchor. This is what gets published."""
    verifier = _load_ledger_verifier()
    rel = path.relative_to(ROOT).as_posix()
    spec = next((s for s in verifier.LEDGERS if s.path == rel), None)
    if spec is None:
        return {"n_rows": 0, "link_mismatches": 0, "violations": ["no digest spec registered for "
                                                                 f"{rel}"], "verified": False}
    text, _origin = verifier.read_source(rel, None)
    n_rows, link_mismatches = _count_link_mismatches(path)
    if text is None:
        return {"n_rows": n_rows, "link_mismatches": link_mismatches, "verified": False,
                "violations": [f"{rel} could not be read"]}
    report = verifier.Report(path=rel)
    rows = verifier.parse_rows(text, report)
    verifier.verify_rows(rows, spec, report)
    verifier.verify_anchor(report, verifier.load_anchors(None).get(rel))
    return {
        "n_rows": report.n_rows,
        "link_mismatches": link_mismatches,
        "violations": report.violations,
        "verified": report.ok,
        "head_hash": report.head_hash,
        "chain_commitment": report.chain_commitment,
    }


def _artifact_hazard_key(artifact: dict) -> str | None:
    return {
        "earthquake": "eq",
        "hurricane": "hu",
        "tornado": "to",
        "eq": "eq",
        "hu": "hu",
        "to": "to",
    }.get(str(artifact.get("hazard", "")).strip())


def _artifact_mature_at(artifact: dict) -> dt.datetime | None:
    issued_at = _parse_utc(artifact.get("issued_at"))
    if issued_at is None:
        return None
    if artifact.get("forecast_horizon_days") is not None:
        return issued_at + dt.timedelta(days=int(artifact.get("forecast_horizon_days", 0) or 0))
    if artifact.get("forecast_horizon_hours") is not None:
        return issued_at + dt.timedelta(hours=int(artifact.get("forecast_horizon_hours", 0) or 0))
    return None


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
    return {
        "availability": "related_research_benchmark",
        "label": "A historical 2024 holdout benchmark exists for a related tornado GBT family, but not yet as an exact score for the live storm-object model.",
        "model_version": payload.get("model"),
        "source_updated_at": payload.get("timestamp"),
        "auc": full.get("auc"),
        "brier": full.get("brier"),
        "brier_skill_score": full.get("bss"),
        "source_files": [
            "results/definitive/definitive_results.json",
        ],
    }


def _status_chip_class(status: str) -> str:
    if status in {"prospective_scored"}:
        return "good"
    if status in {"matured_unscored", "matured_unscored_no_evaluator"}:
        return "bad"
    return "warn"


def _build_verification_summary(pulse: dict) -> dict:
    score_as_of = _parse_utc(pulse.get("updated_at")) or dt.datetime.now(dt.timezone.utc)
    legacy_summary = _read_json(VERIFICATION_SUMMARY_PATH, {"hazards": []})
    replay_groups = _load_replay_artifacts_by_hazard()
    eq_integrity = _ledger_integrity(EQ_LEDGER_PATH)
    to_integrity = _ledger_integrity(TO_LEDGER_PATH)
    eq_rows, eq_mismatches = eq_integrity["n_rows"], len(eq_integrity["violations"])
    to_rows, to_mismatches = to_integrity["n_rows"], len(to_integrity["violations"])
    live_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    eq_related = _earthquake_related_benchmark()
    to_related = _tornado_related_benchmark()
    eq_prospective_summary = _read_json(EQ_PROSPECTIVE_DIR / "prospective_summary.json", {})

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
            "metric_source": "prospective_live" if eq_scored > 0 else "no_exact_model_benchmark",
            "metric_source_label": (
                "Computed from matured live forecasts."
                if eq_scored > 0
                else "The current live earthquake model does not yet have an exact benchmark in this repo."
            ),
            "auc": eq_prospective_summary.get("mean_auc") if eq_scored > 0 else None,
            "brier": eq_prospective_summary.get("mean_brier") if eq_scored > 0 else None,
            "homepage_line": (
                f"{eq_scored} matured windows scored"
                if eq_scored > 0
                else f"{len(eq_artifacts)} frozen forecasts · {eq_backlog} matured backlog"
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
                # Every digest RECOMPUTED from row content, not just pointers compared.
                "verified": eq_integrity["verified"],
                "integrity_violations": eq_mismatches,
                "prev_hash_mismatches": eq_integrity["link_mismatches"],
                "head_hash": eq_integrity.get("head_hash"),
                "verifier": "scripts/verify_ledger_chain.py",
            },
            "prospective": {
                "summary_path": "results/earthquake_prospective/prospective_summary.json",
                "status": eq_prospective_summary.get("status", "not_run"),
                "scored_as_of": eq_prospective_summary.get("scored_as_of"),
                "message": eq_prospective_summary.get("message"),
                "top_5_hit_rate": eq_prospective_summary.get("top_5_hit_rate"),
            },
            "exact_model_benchmark": None,
            "related_benchmark": eq_related,
            "recommended_action": (
                "Keep freezing every earthquake forecast. Once the first 30-day windows mature, run the prospective scorer and use those scores to tune thresholds and calibration."
            ),
        }
    )

    hu_hazard = live_map.get("hu", {})
    hu_artifacts = replay_groups["hu"]
    hu_matured = [
        artifact
        for artifact in hu_artifacts
        if _artifact_mature_at(artifact) is not None and _artifact_mature_at(artifact) <= score_as_of
    ]
    hu_backlog = len(hu_matured)
    hu_exact = _legacy_verification_item(
        legacy_summary,
        "hu",
        str(hu_hazard.get("model_version") or ""),
    )
    if str(hu_hazard.get("model_version") or "") == HURRICANE_RETRO_FALLBACK["model_version"]:
        merged_exact = dict(HURRICANE_RETRO_FALLBACK)
        if isinstance(hu_exact, dict):
            merged_exact.update({key: value for key, value in hu_exact.items() if value is not None})
        hu_exact = merged_exact
    if hu_backlog > 0:
        hu_status = "matured_unscored_no_evaluator"
        hu_status_label = "Matured hurricane forecasts exist, but no live advisory-to-outcome scorer is wired yet."
    elif hu_artifacts:
        hu_status = "logging_live_no_evaluator"
        hu_status_label = "Hurricane forecasts are being frozen, but live outcome scoring is not wired yet."
    else:
        hu_status = "no_live_artifacts"
        hu_status_label = "No live hurricane replay artifacts are present."

    hu_latest = hu_artifacts[-1] if hu_artifacts else {}
    hazards.append(
        {
            "key": "hu",
            "hazard": "hurricane",
            "model_version": hu_hazard.get("model_version"),
            "verification_status": hu_status,
            "status_badge": {
                "matured_unscored_no_evaluator": "Backlog",
                "logging_live_no_evaluator": "Logging",
                "no_live_artifacts": "Missing",
            }.get(hu_status, "Status"),
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
                else f"{len(hu_artifacts)} frozen forecasts · scorer pending"
            ),
            "forecast_storage": {
                "n_replay_artifacts": len(hu_artifacts),
                "n_matured_forecasts": len(hu_matured),
                "n_scored_forecasts": 0,
                "n_backlog": hu_backlog,
                "first_issued_at": hu_artifacts[0].get("issued_at") if hu_artifacts else None,
                "last_issued_at": hu_latest.get("issued_at"),
                "last_forecast_id": hu_latest.get("forecast_id"),
                "latest_replay_artifact": hu_latest.get("_path"),
                "forecast_horizon": _format_horizon(hu_latest) if hu_latest else "24 hours",
                "next_mature_at": next_mature_at(hu_artifacts),
            },
            "ledger": {
                "supported": False,
                "path": None,
                "n_rows": 0,
                "verified": False,
                "integrity_violations": 0,
                "prev_hash_mismatches": 0,
            },
            "prospective": {
                "summary_path": None,
                "status": "evaluator_missing",
                "scored_as_of": None,
                "message": "Live hurricane forecasts are stored, but the repo does not yet score them against realized 24-hour intensity change.",
            },
            "exact_model_benchmark": (
                {
                    "availability": "exact_model_benchmark",
                    "label": "Retrospective benchmark available for the current live model version.",
                    "model_version": hu_exact.get("model_version"),
                    "source_updated_at": hu_exact.get("source_updated_at"),
                    "auc": hu_exact.get("auc"),
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
                "Implement an advisory-to-outcome scorer that joins frozen hurricane forecasts to realized 24-hour intensity change before using the model for calibration or promotion decisions."
            ),
        }
    )

    to_hazard = live_map.get("to", {})
    to_artifacts = replay_groups["to"]
    to_matured = [
        artifact
        for artifact in to_artifacts
        if _artifact_mature_at(artifact) is not None and _artifact_mature_at(artifact) <= score_as_of
    ]
    to_backlog = len(to_matured)
    if to_backlog > 0:
        to_status = "matured_unscored_no_evaluator"
        to_status_label = "Matured tornado storm-object forecasts exist, but no live outcome scorer is wired yet."
    elif to_artifacts:
        to_status = "logging_live_no_evaluator"
        to_status_label = "Tornado storm-object forecasts are being frozen, but live outcome scoring is not wired yet."
    else:
        to_status = "no_live_artifacts"
        to_status_label = "No live tornado replay artifacts are present."

    to_latest = to_artifacts[-1] if to_artifacts else {}
    hazards.append(
        {
            "key": "to",
            "hazard": "tornado",
            "model_version": to_hazard.get("model_version"),
            "verification_status": to_status,
            "status_badge": {
                "matured_unscored_no_evaluator": "Backlog",
                "logging_live_no_evaluator": "Logging",
                "no_live_artifacts": "Missing",
            }.get(to_status, "Status"),
            "verification_status_label": to_status_label,
            "metric_source": "no_exact_model_benchmark",
            "metric_source_label": "No exact benchmark is currently bound to the live tornado storm-object model version in this repo.",
            "auc": None,
            "brier": None,
            "homepage_line": f"{len(to_artifacts)} frozen forecasts · {to_backlog} matured backlog",
            "forecast_storage": {
                "n_replay_artifacts": len(to_artifacts),
                "n_matured_forecasts": len(to_matured),
                "n_scored_forecasts": 0,
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
                # Every digest RECOMPUTED from row content, not just pointers compared.
                "verified": to_integrity["verified"],
                "integrity_violations": to_mismatches,
                "prev_hash_mismatches": to_integrity["link_mismatches"],
                "head_hash": to_integrity.get("head_hash"),
                "verifier": "scripts/verify_ledger_chain.py",
            },
            "prospective": {
                "summary_path": None,
                "status": "evaluator_missing",
                "scored_as_of": None,
                "message": "Live tornado storm-object forecasts are stored, but the repo does not yet score them against matched outcomes.",
            },
            "exact_model_benchmark": None,
            "related_benchmark": to_related,
            "recommended_action": (
                "Bind each frozen tornado storm-object forecast to a matched outcome definition and write a 24-hour scorer before using the live model for calibration or threshold changes."
            ),
        }
    )

    total_replays = sum(item["forecast_storage"]["n_replay_artifacts"] for item in hazards)
    total_matured = sum(item["forecast_storage"]["n_matured_forecasts"] for item in hazards)
    total_scored = sum(item["forecast_storage"]["n_scored_forecasts"] for item in hazards)
    total_backlog = sum(item["forecast_storage"]["n_backlog"] for item in hazards)
    total_chain_mismatches = eq_mismatches + to_mismatches
    alerts: list[str] = []
    if total_backlog:
        alerts.append(f"{total_backlog} matured forecast windows are waiting for scoring.")
    if total_chain_mismatches:
        alerts.append(
            f"{total_chain_mismatches} hash-chain INTEGRITY VIOLATIONS were detected by "
            f"recomputing every entry's digest from its own content. Do not trust these ledgers "
            f"until they are resolved.")
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


_SCOREBOARD_PREFIX = {"earthquake": "eq", "tornado": "to", "hurricane": "hu"}


def _latest_forecast_scores(hazard: str) -> list[float]:
    """Raw probabilities of the most recent forecast (for distribution-drift PSI)."""
    prefix = _SCOREBOARD_PREFIX.get(hazard)
    if not prefix or not REPLAY_DIR.exists():
        return []
    paths = sorted(REPLAY_DIR.glob(f"{prefix}_fcst_*.json"))
    if not paths:
        return []
    art = _read_replay_cached(paths[-1])
    scores: list[float] = []
    for item in (art.get("active_cells") or art.get("storms") or []):
        if not isinstance(item, dict):
            continue
        v = item.get("raw_probability")
        if v is None:
            v = item.get("probability",
                         item.get("tornado_probability", item.get("ri_probability")))
        if v is not None:
            try:
                scores.append(float(v))
            except (TypeError, ValueError):
                continue
    return scores


def _render_calibration_scoreboard() -> str:
    """Reliability diagrams + before/after calibration metrics per hazard, drawn
    from the platform's OWN matured forecasts. Empty until a calibrator exists,
    so the section simply does not appear until real calibration data flows."""
    try:
        from hazardpulse.trust.calibration import reliability_curve_from_counts
        from hazardpulse.trust.venn_abers import VennAbersCalibrator
        from hazardpulse.trust.viz import reliability_diagram_svg
    except Exception:
        return ""

    accents = {"earthquake": "#e65100", "tornado": "#6a1b9a", "hurricane": "#1976d2"}

    def _m(d: dict, key: str) -> str:
        v = (d or {}).get(key)
        if v is None:
            return "--"
        return f"{v:.3f}" if isinstance(v, (int, float)) else _esc(v)

    cards: list[str] = []
    for name in ("earthquake", "tornado", "hurricane"):
        rec = _read_json(ROOT / "results" / "calibration" / f"{name}_calibration.json", {})
        ds = _read_json(ROOT / "results" / f"{name}_prospective" / "calibration_dataset.json", {})
        after = rec.get("metrics_after") if isinstance(rec, dict) else None
        if not after or not ds.get("scores"):
            continue
        try:
            cal = VennAbersCalibrator.from_dict(rec["calibrator"])
            cal_scores, _, _ = cal.predict(ds["scores"])
            curve = reliability_curve_from_counts(cal_scores, ds["pos"], ds["total"])
            svg = reliability_diagram_svg(
                curve, title=f"{name.title()} (deployed)", accent=accents.get(name, "#1976d2"))
        except Exception:
            continue
        before = rec.get("metrics_before", {})
        drift = {"status": "insufficient_data", "psi": 0.0}
        try:
            from hazardpulse.trust.monitor import expand_histogram, forecast_drift_status
            cur_scores = _latest_forecast_scores(name)
            if cur_scores:
                drift = forecast_drift_status(
                    expand_histogram(ds["scores"], ds["total"]), cur_scores)
        except Exception:
            pass
        cards.append(
            '<div class="card col-4">'
            f'<h3 style="margin-top:0;">{_esc(_hazard_label(name))}</h3>'
            f'{svg}'
            f'<div class="kv"><span>ECE (raw &rarr; calibrated)</span>'
            f'<strong>{_m(before, "ece")} &rarr; {_m(after, "ece")}</strong></div>'
            f'<div class="kv"><span>Brier skill</span>'
            f'<strong>{_m(before, "brier_skill_score")} &rarr; {_m(after, "brier_skill_score")}</strong></div>'
            f'<div class="kv"><span>Distribution drift</span>'
            f'<strong>{_esc(str(drift["status"]).replace("_", " "))} (PSI {drift["psi"]:.3f})</strong></div>'
            f'<div class="kv"><span>Calibration sample</span>'
            f'<strong>{int(rec.get("n_calibration", 0) or 0):,}</strong></div>'
            "</div>"
        )
    if not cards:
        return ""
    return (
        '<section class="section" aria-labelledby="calib-heading">'
        '<h2 id="calib-heading">Calibration scoreboard</h2>'
        '<p class="muted" style="margin-top:-8px;margin-bottom:16px;">'
        "Reliability of the deployed (calibrated) forecasts, measured against the "
        "platform's own realized outcomes. A perfectly calibrated forecaster sits on "
        "the dashed diagonal; ECE and Brier-skill are shown raw &rarr; calibrated.</p>"
        f'<div class="grid">{"".join(cards)}</div>'
        "</section>"
    )


def _render_verification_page(summary: dict) -> None:
    system = summary.get("system", {})
    hazards = list(summary.get("hazards", []))
    calibration_section = _render_calibration_scoreboard()

    system_cards = [
        (
            str(system.get("frozen_forecasts", 0)),
            "Frozen forecasts",
            "Replay artifacts preserved across all live hazards.",
        ),
        (
            str(system.get("matured_unscored_backlog", 0)),
            "Scoring backlog",
            "Matured windows waiting for an evaluator or scoring run.",
        ),
        (
            str(system.get("hash_chain_mismatches", 0)),
            "Hash mismatches",
            "Prev-hash continuity failures in raw append-only ledgers.",
        ),
        (
            str(system.get("exact_model_benchmarks", 0)),
            "Exact benchmarks",
            "Live model versions with an attached exact benchmark.",
        ),
    ]

    system_cards_html = "".join(
        '<div class="card col-3">'
        f'<div class="metric mono">{_esc(value)}</div>'
        f'<div class="metric-label">{_esc(label)}</div>'
        f'<p class="muted" style="font-size:12px;margin-top:8px;">{_esc(note)}</p>'
        "</div>"
        for value, label, note in system_cards
    )

    alert_items = system.get("alerts", [])
    alert_html = (
        "<ul>"
        + "".join(f"<li>{_esc(item)}</li>" for item in alert_items)
        + "</ul>"
        if alert_items
        else '<p class="muted" style="margin:0;">No current verification-control alerts.</p>'
    )

    hazard_cards: list[str] = []
    benchmark_rows: list[str] = []
    for item in hazards:
        storage = item.get("forecast_storage", {})
        ledger = item.get("ledger", {})
        exact_benchmark = item.get("exact_model_benchmark")
        related_benchmark = item.get("related_benchmark")
        metric_html = ""
        if item.get("auc") is not None or item.get("brier") is not None:
            metric_html = (
                f'<div class="kv"><span>Primary metric</span><strong>AUC {_fmt_float(item.get("auc"))} &middot; '
                f'Brier {_fmt_float(item.get("brier"))}</strong></div>'
            )
        if exact_benchmark:
            benchmark_html = (
                f'<div class="kv"><span>Exact benchmark</span><strong>AUC {_fmt_float(exact_benchmark.get("auc"))} &middot; '
                f'Brier {_fmt_float(exact_benchmark.get("brier"))}</strong></div>'
            )
        elif related_benchmark:
            related_bits = []
            if related_benchmark.get("same_location_auc") is not None:
                related_bits.append(f"same-location AUC {_fmt_float(related_benchmark.get('same_location_auc'))}")
            if related_benchmark.get("global_auc") is not None:
                related_bits.append(f"global AUC {_fmt_float(related_benchmark.get('global_auc'))}")
            if related_benchmark.get("auc") is not None:
                related_bits.append(f"AUC {_fmt_float(related_benchmark.get('auc'))}")
            benchmark_html = (
                f'<div class="kv"><span>Related benchmark</span><strong>{_esc(" | ".join(related_bits) or "Available")}</strong></div>'
            )
        else:
            benchmark_html = (
                '<div class="kv"><span>Benchmark</span><strong>No benchmark artifact attached to this live model yet</strong></div>'
            )
        ledger_html = (
            f'<div class="kv"><span>Raw ledger</span><strong>{int(ledger.get("n_rows", 0) or 0)} rows &middot; '
            f'{int(ledger.get("integrity_violations", 0) or 0)} integrity violations '
            f'({"every digest recomputed" if ledger.get("verified") else "NOT VERIFIED"})</strong></div>'
            if ledger.get("supported")
            else '<div class="kv"><span>Raw ledger</span><strong>Not implemented for this hazard yet</strong></div>'
        )
        hazard_cards.append(
            f'<div class="card col-4 hazard-{_esc(item["key"])}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;"><h2 style="margin:0;">{_esc(_hazard_label(item["key"]))}</h2>'
            f'<span class="chip {_status_chip_class(str(item.get("verification_status", "")))}" style="margin-left:auto;">{_esc(item.get("status_badge", "Status"))}</span></div>'
            f'<p style="margin:0 0 12px;line-height:1.6;">{_esc(item.get("verification_status_label"))}</p>'
            f'<div class="kv"><span>Live model</span><strong>{_esc(item.get("model_version") or "--")}</strong></div>'
            f'<div class="kv"><span>Latest forecast</span><strong>{_esc(storage.get("last_forecast_id") or "--")}</strong></div>'
            f'<div class="kv"><span>Storage</span><strong>{int(storage.get("n_replay_artifacts", 0) or 0)} replays &middot; horizon {_esc(storage.get("forecast_horizon") or "--")}</strong></div>'
            f'<div class="kv"><span>Maturity</span><strong>{int(storage.get("n_matured_forecasts", 0) or 0)} matured &middot; {int(storage.get("n_scored_forecasts", 0) or 0)} scored</strong></div>'
            f'{metric_html}'
            f'{benchmark_html}'
            f'{ledger_html}'
            f'<p class="muted" style="margin:12px 0 0;">{_esc(item.get("recommended_action"))}</p>'
            "</div>"
        )

        source = exact_benchmark or related_benchmark or {}
        metric_bits = []
        if exact_benchmark and exact_benchmark.get("auc") is not None:
            metric_bits.append(f"AUC {_fmt_float(exact_benchmark.get('auc'))}")
        if exact_benchmark and exact_benchmark.get("brier") is not None:
            metric_bits.append(f"Brier {_fmt_float(exact_benchmark.get('brier'))}")
        if related_benchmark and related_benchmark.get("same_location_auc") is not None:
            metric_bits.append(f"same-location AUC {_fmt_float(related_benchmark.get('same_location_auc'))}")
        if related_benchmark and related_benchmark.get("global_auc") is not None:
            metric_bits.append(f"global AUC {_fmt_float(related_benchmark.get('global_auc'))}")
        if related_benchmark and related_benchmark.get("auc") is not None:
            metric_bits.append(f"AUC {_fmt_float(related_benchmark.get('auc'))}")
        benchmark_rows.append(
            "<tr>"
            f"<td>{_esc(_hazard_label(item['key']))}</td>"
            f"<td>{_esc(source.get('availability', 'unavailable').replace('_', ' ').title())}</td>"
            f"<td>{_esc(source.get('model_version') or item.get('model_version') or '--')}</td>"
            f"<td>{_esc(' | '.join(metric_bits) or 'None attached')}</td>"
            f"<td>{_esc(source.get('source_updated_at') or '--')}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verification - HazardPulse</title>
  <meta name="description" content="Forecast storage, scoring readiness, and benchmark status for live HazardPulse models. This page distinguishes exact live scoring, pending maturity, and scorer backlogs.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/verification/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Verification - HazardPulse">
  <meta property="og:description" content="Forecast storage, scoring readiness, and benchmark status for live HazardPulse models.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/verification/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Verification - HazardPulse">
  <meta name="twitter:description" content="Forecast storage, scoring readiness, and benchmark status for live HazardPulse models.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Verification Status",
    "description": "Storage, scoring readiness, benchmark provenance, and live verification status for HazardPulse forecast models.",
    "url": "{PRIMARY_DOMAIN}/verification/",
    "creator": {{ "@type": "Organization", "name": "HazardPulse", "url": "{PRIMARY_DOMAIN}/" }}
  }}
  </script>
  <script type="speculationrules">
  {{ "prefetch": [{{ "source": "list", "urls": ["/", "/live/", "/evidence/", "/api/"] }}] }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/" aria-current="page">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Verification</div>
      <h1>Verification now reflects what we can actually prove.</h1>
      <p class="subtitle">
        HazardPulse freezes every live forecast into replay artifacts, tracks raw append-only ledgers, and now
        separates exact model benchmarks, pending maturity windows, and scoring backlogs. If a live model is not
        scored yet, this page says so directly.
      </p>
      <p class="muted">Built {_esc(summary.get("generated_at"))} &middot; Score as of {_esc(summary.get("score_as_of"))}</p>
    </section>
    <section class="section">
      <div class="grid">
        {system_cards_html}
      </div>
    </section>
    <section class="section">
      <h2>Control alerts</h2>
      <div class="card">
        {alert_html}
      </div>
    </section>
    <section class="section">
      <h2>Hazard by hazard</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">These cards tell you whether each live model has exact scores, only related research benchmarks, or just frozen forecasts waiting for scoring.</p>
      <div class="grid">
        {''.join(hazard_cards)}
      </div>
    </section>
    {calibration_section}
    <section class="section">
      <h2>Benchmark provenance</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Exact benchmarks are safe to cite for the current live model version. Related benchmarks are useful for research context, but not as proof of live performance.</p>
      <div class="card">
        <table>
          <thead><tr><th>Hazard</th><th>Type</th><th>Model</th><th>Metrics</th><th>Source updated</th></tr></thead>
          <tbody>
            {''.join(benchmark_rows)}
          </tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <h2>Storage and audit surfaces</h2>
      <div class="grid">
        <div class="card col-6">
          <div class="kv"><span>Verification summary</span><strong><a href="/data/verification-summary.json">/data/verification-summary.json</a></strong></div>
          <div class="kv"><span>Replay index</span><strong><a href="/data/evidence/replay-index.json">/data/evidence/replay-index.json</a></strong></div>
          <div class="kv"><span>Prediction ledger</span><strong><a href="/data/evidence/prediction-ledger.json">/data/evidence/prediction-ledger.json</a></strong></div>
          <div class="kv"><span>Earthquake raw chain</span><strong><a href="/data/earthquake-ledger.jsonl">/data/earthquake-ledger.jsonl</a></strong></div>
          <div class="kv"><span>Tornado raw chain</span><strong><a href="/data/tornado-ledger.jsonl">/data/tornado-ledger.jsonl</a></strong></div>
        </div>
        <div class="card col-6">
          <p style="margin:0 0 12px;line-height:1.6;">
            This surface is intentionally stricter than marketing copy. A billion-dollar company needs a page that tells operators
            what is frozen, what is scored, what is only a research benchmark, and what still needs engineering work before it
            can influence model adjustment or promotion.
          </p>
          <p class="muted" style="margin:0;">Use the evidence ledger for artifact-level traceability and this page for scoring readiness and benchmark discipline.</p>
        </div>
      </div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>About</h4>
        <a href="mailto:{CONTACT_EMAIL}">Contact</a>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the USGS, NHC, NWS, SPC, JMA, and IMD.
      </p>
      <p class="footer-build">Static-first HTML &middot; Evidence-linked data &middot; Verification state generated from live artifacts</p>
    </div>
  </footer>
</body>
</html>
"""
    VERIFICATION_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION_PAGE_PATH.write_text(page, encoding="utf-8")


def _render_evidence_page(
    pulse: dict,
    entries: list[dict],
    envelopes: list[dict],
    gate_decisions: list[dict],
    replay_index: dict,
) -> None:
    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    eq_integrity = _ledger_integrity(EQ_LEDGER_PATH)
    to_integrity = _ledger_integrity(TO_LEDGER_PATH)
    eq_rows, eq_mismatches = eq_integrity["n_rows"], len(eq_integrity["violations"])
    to_rows, to_mismatches = to_integrity["n_rows"], len(to_integrity["violations"])
    replayable_entries = [entry for entry in entries if entry.get("replay_artifact")]

    evidence_cards: list[str] = []
    for key in ("eq", "hu", "to"):
        hazard = hazard_map.get(key, {})
        label = _hazard_label(key)
        forecast_id = hazard.get("forecast_id") or "No published artifact id"
        gate_text = str(hazard.get("gate_status", "pass")).replace("_", " ")
        replay_ready = "Yes" if any(item.get("forecast_id") == hazard.get("forecast_id") for item in replay_index.get("items", [])) else "Pending"
        evidence_cards.append(
            f'<div class="card col-4 hazard-{key}">'
            f"<h3>{_esc(label)}</h3>"
            f'<div class="metric">{_pct(hazard.get("probability", 0))}</div>'
            f'<div class="metric-label">Current published state</div>'
            f'<div class="kv"><span>Forecast ID</span><strong><code>{_esc(forecast_id)}</code></strong></div>'
            f'<div class="kv"><span>Gate</span><strong>{_esc(gate_text.title())}</strong></div>'
            f'<div class="kv"><span>Replay ready</span><strong>{replay_ready}</strong></div>'
            f"</div>"
        )

    ledger_rows = []
    for entry in entries[:12]:
        replay_link = entry.get("replay_artifact")
        ledger_rows.append(
            "<tr>"
            f"<td><code>{_esc(entry.get('forecast_id'))}</code></td>"
            f"<td>{_esc(_hazard_label(entry.get('hazard')))}</td>"
            f"<td>{_esc(entry.get('issued_at'))}</td>"
            f"<td>{_pct(entry.get('probability', 0))}</td>"
            f"<td><code>{_esc(_short_hash(entry.get('hash')))}</code></td>"
            f"<td>{'<a href=\"' + _esc(replay_link) + '\">Replay</a>' if replay_link else 'Archive pending'}</td>"
            "</tr>"
        )

    provenance_cards = []
    for envelope in envelopes[:6]:
        provenance_cards.append(
            '<div class="card">'
            f"<h3><code>{_esc(envelope['provenance_id'])}</code></h3>"
            f'<div class="kv"><span>Forecast</span><strong><code>{_esc(envelope["forecast_id"])}</code></strong></div>'
            f'<div class="kv"><span>Input hash</span><strong><code>{_esc(_short_hash(envelope["input_hash"]))}</code></strong></div>'
            f'<div class="kv"><span>Output hash</span><strong><code>{_esc(_short_hash(envelope["output_hash"]))}</code></strong></div>'
            f'<div class="kv"><span>Signed</span><strong>{_esc(envelope["signed_at"])}</strong></div>'
            f'<p class="muted" style="margin:12px 0 0;">Sources: {_esc(", ".join(envelope["sources"]))}</p>'
            "</div>"
        )

    gate_rows = []
    for decision in gate_decisions[:10]:
        warnings = ", ".join(decision.get("warnings", [])) or "None"
        gate_rows.append(
            "<tr>"
            f"<td><code>{_esc(decision['gate_decision_id'])}</code></td>"
            f"<td>{_esc(_hazard_label(decision['hazard']))}</td>"
            f"<td>{_esc(decision['decision'])}</td>"
            f"<td>{_esc(warnings)}</td>"
            f"<td>{_esc(decision.get('issued_at'))}</td>"
            "</tr>"
        )

    replay_rows = []
    for item in replay_index.get("items", []):
        replay_rows.append(
            "<tr>"
            f"<td><code>{_esc(item.get('forecast_id'))}</code></td>"
            f"<td><a href=\"{_esc(item.get('replay_artifact'))}\">{_esc(item.get('replay_artifact'))}</a></td>"
            "</tr>"
        )

    coverage = f"{len(envelopes)}/{max(1, len(replayable_entries))}"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Ledger - HazardPulse</title>
  <meta name="description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/evidence/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Evidence Ledger - HazardPulse">
  <meta property="og:description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/evidence/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Evidence Ledger - HazardPulse">
  <meta name="twitter:description" content="Live evidence ledger built from real forecast archives, current publish artifacts, provenance hashes, and gate decisions.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "DataCatalog",
    "name": "HazardPulse Evidence Ledger",
    "description": "Real forecast archives, provenance hashes, gate decisions, and replay artifacts for published HazardPulse forecasts.",
    "url": "{PRIMARY_DOMAIN}/evidence/",
    "creator": {{ "@type": "Organization", "name": "HazardPulse", "url": "{PRIMARY_DOMAIN}/" }}
  }}
  </script>
  <script type="speculationrules">
  {{ "prefetch": [{{ "source": "list", "urls": ["/", "/live/", "/verification/", "/api/"] }}] }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/" aria-current="page">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Evidence</div>
      <h1>Every published state has a real artifact.</h1>
      <p class="subtitle">
        This surface is generated from the live publish artifacts in <code>/data</code>, the replay archive,
        and the raw earthquake and tornado ledgers. No sample hashes, no synthetic gate records, no fabricated provenance.
      </p>
      <p class="muted">Updated {_esc(pulse.get("updated_at"))} &middot; Replayable artifacts: {len(replay_index.get("items", []))}</p>
    </section>
    <section class="section">
      <div class="grid">
        {"".join(evidence_cards)}
      </div>
    </section>
    <section class="section">
      <div class="grid">
        <div class="card col-3"><div class="metric mono">{eq_rows + to_rows}</div><div class="metric-label">Raw chain rows audited</div></div>
        <div class="card col-3"><div class="metric mono">{eq_mismatches + to_mismatches}</div><div class="metric-label">Integrity violations (digests recomputed)</div></div>
        <div class="card col-3"><div class="metric mono">{coverage}</div><div class="metric-label">Provenance coverage</div></div>
        <div class="card col-3"><div class="metric mono">{len(replay_index.get("items", []))}</div><div class="metric-label">Replay artifacts</div></div>
      </div>
    </section>
    <section class="section" id="ledger">
      <h2>Prediction ledger</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Recent append-only and archive-backed forecast records, newest first.</p>
      <div class="card">
        <table>
          <thead><tr><th>Forecast ID</th><th>Hazard</th><th>Issued</th><th>Probability</th><th>Hash</th><th>Artifact</th></tr></thead>
          <tbody>
            {"".join(ledger_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/prediction-ledger.json" class="btn btn-secondary">Download prediction ledger</a></div>
    </section>
    <section class="section">
      <h2>Provenance envelopes</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Hashes are computed from the actual published replay artifacts and their source manifests.</p>
      <div class="grid">
        {"".join(provenance_cards) or '<div class="card"><p class="muted" style="margin:0;">No replay artifacts are available yet.</p></div>'}
      </div>
      <div class="cta-row"><a href="/data/evidence/provenance-envelopes.json" class="btn btn-secondary">Download provenance envelopes</a></div>
    </section>
    <section class="section" id="gates">
      <h2>Gate decisions</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Current warnings are derived from the live publish state, not hardcoded examples.</p>
      <div class="card">
        <table>
          <thead><tr><th>Decision ID</th><th>Hazard</th><th>Decision</th><th>Warnings</th><th>Issued</th></tr></thead>
          <tbody>
            {"".join(gate_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/gate-decisions.json" class="btn btn-secondary">Download gate decisions</a></div>
    </section>
    <section class="section" id="replay">
      <h2>Replay index</h2>
      <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Frozen publish artifacts that can be fetched directly through the public API.</p>
      <div class="card">
        <table>
          <thead><tr><th>Forecast ID</th><th>Artifact</th></tr></thead>
          <tbody>
            {"".join(replay_rows)}
          </tbody>
        </table>
      </div>
      <div class="cta-row"><a href="/data/evidence/replay-index.json" class="btn btn-secondary">Download replay index</a></div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>About</h4>
        <a href="mailto:{CONTACT_EMAIL}">Contact</a>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the USGS, NHC, NWS, SPC, JMA, and IMD.
      </p>
      <p class="footer-build">Static-first HTML &middot; Evidence-linked data &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    EVIDENCE_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PAGE_PATH.write_text(page, encoding="utf-8")


def _render_live_hurricane_page() -> None:
    storms = _read_json(LIVE_STORMS_PATH, {})
    updated_at = _parse_utc(storms.get("updated_at"))
    storms_list = list(storms.get("storms", []))

    if storms_list:
        rows = []
        sorted_storms = sorted(
            storms_list,
            key=lambda item: float(item.get("ri_probability", 0) or 0),
            reverse=True,
        )
        for storm in sorted_storms[:8]:
            rows.append(
                f"<tr>"
                f"<td>{_esc(storm.get('storm_name', storm.get('storm_id', 'Storm')))}</td>"
                f"<td>{_esc(storm.get('category', '--'))}</td>"
                f"<td>{_esc(storm.get('lat', '--'))}, {_esc(storm.get('lon', '--'))}</td>"
                f"<td>{_pct(storm.get('ri_probability', 0))}</td>"
                f"<td>{_esc(storm.get('vmax_kt', '--'))} kt</td>"
                f"</tr>"
            )
        storms_html = (
            "<table><thead><tr><th>Storm</th><th>Status</th><th>Location</th><th>RI 24h</th><th>Wind</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        top = sorted_storms[0]
        summary_html = (
            '<div class="card hazard-hu">'
            '<h2 style="margin-top:0;">Top storm</h2>'
            f'<div class="metric">{_pct(top.get("ri_probability", 0))}</div>'
            '<div class="metric-label">Rapid intensification in 24h</div>'
            f'<div class="kv"><span>Name</span><strong>{_esc(top.get("storm_name", top.get("storm_id", "Storm")))}</strong></div>'
            f'<div class="kv"><span>Status</span><strong>{_esc(top.get("category", "--"))} &middot; {_esc(top.get("vmax_kt", "--"))} kt</strong></div>'
            "</div>"
        )
    else:
        storms_html = (
            '<div class="card"><p class="muted" style="margin:0;">'
            "No active tropical cyclones are present in the current feed."
            "</p></div>"
        )
        summary_html = (
            '<div class="card hazard-hu"><h2 style="margin-top:0;">Current state</h2>'
            '<div class="metric">0.0%</div>'
            '<div class="metric-label">Rapid intensification in 24h</div>'
            '<p class="muted">No active tropical cyclones are present in the current feed.</p>'
            "</div>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Hurricane Forecasts - HazardPulse</title>
  <meta name="description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/hurricane/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=9">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta property="og:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/hurricane/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta name="twitter:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Live Hurricane Forecasts",
    "url": "{PRIMARY_DOMAIN}/live/hurricane/",
    "description": "Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output."
  }}
  </script>
  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/", "/live/", "/live/earthquake/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Hurricane live page</div>
      <h1>Current tropical cyclone view</h1>
      <p class="subtitle">
        This page is rendered from the current tropical cyclone feed. When there are no active storms,
        it says so plainly instead of showing synthetic examples.
      </p>
      <p class="muted">Updated {_esc(_format_utc_z(updated_at))} &middot; Model: hurricane_ri_v8_1 &middot; Independent hazard intelligence platform</p>
    </section>
    <section class="section">
      <div class="grid">
        <div class="col-4">
          {summary_html}
        </div>
        <div class="card col-8">
          <h2 style="margin-top:0;">Active tropical systems</h2>
          {storms_html}
          <p class="muted" style="margin-top:12px;">Source feed: <a href="/data/live-storms.json">/data/live-storms.json</a>. Always follow official advisories from the National Hurricane Center and local authorities.</p>
        </div>
      </div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the NHC, JTWC, WMO RSMCs, and local emergency authorities.
      </p>
      <p class="footer-build">Static-first HTML &middot; Live data under <code>/data</code> &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
</body>
</html>
"""
    live_hurricane_path = DIST / "live" / "hurricane" / "index.html"
    live_hurricane_path.parent.mkdir(parents=True, exist_ok=True)
    live_hurricane_path.write_text(page, encoding="utf-8")


def _write_sitemap_and_feed(pulse: dict) -> None:
    updated_at = _parse_utc(pulse.get("updated_at")) or dt.datetime.now(dt.timezone.utc)
    lastmod = updated_at.strftime("%Y-%m-%d")

    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for route, changefreq, priority in ROUTES:
        sitemap_lines.append(
            f"  <url><loc>{PRIMARY_DOMAIN}{route}</loc><lastmod>{lastmod}</lastmod><changefreq>{changefreq}</changefreq><priority>{priority}</priority></url>"
        )
    sitemap_lines.append("</urlset>")
    SITEMAP_PATH.write_text("\n".join(sitemap_lines) + "\n", encoding="utf-8")

    hazard_map = {hazard.get("key"): hazard for hazard in pulse.get("hazards", [])}
    summary = (
        f"Earthquake {_pct(hazard_map.get('eq', {}).get('probability', 0))}, "
        f"Hurricane {_pct(hazard_map.get('hu', {}).get('probability', 0))}, "
        f"Tornado {_pct(hazard_map.get('to', {}).get('probability', 0))}."
    )
    feed_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>HazardPulse Updates</title>
    <link>{PRIMARY_DOMAIN}/</link>
    <description>Current publish-cycle updates from the live HazardPulse surface</description>
    <language>en-us</language>
    <item>
      <title>HazardPulse publish cycle {updated_at.strftime('%Y-%m-%d %H:%M UTC')}</title>
      <link>{PRIMARY_DOMAIN}/</link>
      <guid>hazardpulse-publish-{updated_at.strftime('%Y%m%d%H%M')}</guid>
      <pubDate>{_format_http_date(updated_at)}</pubDate>
      <description>{_esc(summary)}</description>
    </item>
  </channel>
</rss>
"""
    FEED_PATH.write_text(feed_xml, encoding="utf-8")


def _normalize_html_accessibility_labels() -> None:
    for html_path in DIST.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        normalized = text.replace(
            'aria-label="Switch to light mode"',
            'aria-label="Switch to dark mode"',
        )
        if normalized != text:
            html_path.write_text(normalized, encoding="utf-8")


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
    _render_live_hurricane_page()
    _render_evidence_page(pulse, entries, envelopes, gate_decisions, replay_index)
    _render_verification_page(verification_summary)
    _write_sitemap_and_feed(pulse)
    _normalize_html_accessibility_labels()

    return {
        "pulse": pulse,
        "verification_summary": verification_summary,
        "entries": entries,
        "envelopes": envelopes,
        "gate_decisions": gate_decisions,
        "replay_index": replay_index,
    }


def sync_homepage_forecast_refs() -> bool:
    """Rewrite the homepage's earthquake forecast references from live-pulse.json.

    The homepage is fully rendered only by the hurricane scoring cycle, so an
    earthquake cycle used to advance live-pulse.json (and the replay ledger)
    while dist/index.html kept pointing at the previous forecast id. Because
    every scoring workflow runs this builder before committing dist/, doing the
    sync here makes cross-file forecast identity consistent in EVERY commit,
    whichever hazard's cycle produced it (guarded by test_site_integrity).
    """
    import re

    index_path = DIST / "index.html"
    if not (index_path.exists() and LIVE_PULSE_PATH.exists()):
        return False
    pulse = json.loads(LIVE_PULSE_PATH.read_text(encoding="utf-8"))
    eq = next((h for h in pulse.get("hazards", []) if h.get("key") == "eq"), None)
    forecast_id = (eq or {}).get("forecast_id")
    if not forecast_id or not re.fullmatch(r"eq_fcst_\d{8}_\d{4}", forecast_id):
        return False
    page = index_path.read_text(encoding="utf-8")
    synced = re.sub(r"eq_fcst_\d{8}_\d{4}", forecast_id, page)
    if synced != page:
        index_path.write_text(synced, encoding="utf-8")
        return True
    return False


def main() -> None:
    build_site_artifacts()
    if sync_homepage_forecast_refs():
        print("Synced homepage earthquake forecast references to live-pulse.json.")
    print("Built HazardPulse evidence, replay, sitemap, and feed artifacts.")


if __name__ == "__main__":
    main()
