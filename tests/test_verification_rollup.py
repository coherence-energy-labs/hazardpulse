"""The verification rollup must report what the prospective scorers scored.

Regression for audit finding #4 (2026-10-01): from 2026-04-12, when
score_tornado_prospective.py was wired into the verification workflow, until
2026-10-01 the tornado (and hurricane) rollup in build_site_artifacts.py
hardcoded ``n_scored_forecasts: 0`` and ``status: evaluator_missing`` and never
read the scorer's summary -- so /verification said "0 scored, no evaluator"
while 1,737 matured tornado forecasts had been scored.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BSA_PATH = REPO / "scripts" / "build_site_artifacts.py"

NOW = dt.datetime(2026, 10, 1, 22, 0, tzinfo=dt.timezone.utc)


def _z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("bsa_rollup_under_test", BSA_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    root = tmp_path
    dist = root / "dist"
    m.ROOT = root
    m.DIST = dist
    m.LIVE_PULSE_PATH = dist / "data" / "live-pulse.json"
    m.EQ_LEDGER_PATH = dist / "data" / "earthquake-ledger.jsonl"
    m.TO_LEDGER_PATH = dist / "data" / "tornado-ledger.jsonl"
    m.REPLAY_DIR = dist / "data" / "replay"
    m.VERIFICATION_SUMMARY_PATH = dist / "data" / "verification-summary.json"
    m.VERIFICATION_DATA_DIR = dist / "data" / "verification"
    m.RESULTS_VERIFICATION_DIR = root / "results" / "verification"
    m.EQ_PROSPECTIVE_DIR = root / "results" / "earthquake_prospective"
    m.TO_PROSPECTIVE_DIR = root / "results" / "tornado_prospective"
    m.HU_PROSPECTIVE_DIR = root / "results" / "hurricane_prospective"
    m.EQ_HONEST_RESULTS_PATH = root / "results" / "earthquake_honest" / "v4.json"
    m.EQ_SAME_LOCATION_PATH = root / "results" / "earthquake_honest" / "same.json"
    m.TO_RETRO_RESULTS_PATH = root / "results" / "definitive" / "definitive_results.json"
    m.REPLAY_DIR.mkdir(parents=True)
    m.LIVE_PULSE_PATH.write_text(json.dumps({
        "updated_at": _z(NOW),
        "hazards": [
            {"key": "eq", "model_version": "eq_coherence_v1_0"},
            {"key": "hu", "model_version": "hurricane_ri_v8_1"},
            {"key": "to", "model_version": "tornado_storm_v1_0"},
        ],
    }), encoding="utf-8")
    return m


def _write_replays(m, prefix: str, hazard: str, n: int, *, start_hours_ago: int = 72) -> None:
    for i in range(n):
        issued = NOW - dt.timedelta(hours=start_hours_ago - i)
        fid = f"{prefix}_fcst_{issued.strftime('%Y%m%d_%H%M')}"
        (m.REPLAY_DIR / f"{fid}.json").write_text(json.dumps({
            "forecast_id": fid,
            "hazard": hazard,
            "issued_at": _z(issued),
            "forecast_horizon_hours": 24,
            "storms": [],
        }), encoding="utf-8")


def _write_summary(m, directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prospective_summary.json").write_text(json.dumps(payload), encoding="utf-8")


POOLED = {
    "n_storm_forecasts": 900,
    "n_positive": 12,
    "base_rate": 0.0033,
    "mean_forecast_probability": 0.05,
    "brier": 0.0123,
    "auc": 0.77,
    "bss_vs_causal_climatology": -2.5,
    "bss_vs_sample_climatology": -2.6,
    "reliability": [{"bin": "[0, 1]", "n": 900}],
}


def _tornado_summary(n: int, *, scored_as_of: dt.datetime = NOW - dt.timedelta(hours=1), pooled: bool = True,
                     version: str = "tornado_storm_v1_0") -> dict:
    payload = {
        "scored_as_of": _z(scored_as_of),
        "n_matured_forecasts": n,
        "n_scored_forecasts": n,
        "status": "ok",
        "mean_auc": 0.71,
        "mean_brier": 0.031,
        "outcome_time_convention": "convective day",
        "skill_reference": "causal",
    }
    if pooled:
        payload["pooled"] = dict(POOLED)
        # the live record per model version (score_tornado_prospective.summarize): the homepage
        # quotes ONLY the live version's own entry, never the pool across versions
        payload["pooled_by_model_version"] = {version: {**POOLED, "n_forecasts": n}}
        payload["pooled_by_served_mode"] = {
            "raw_model_probability": {**POOLED, "n_storm_forecasts": 400, "bss_vs_causal_climatology": -9.0},
            "calibrated_probability": {**POOLED, "n_storm_forecasts": 500, "bss_vs_causal_climatology": 0.002},
        }
    return payload


def _hazard(summary: dict, key: str) -> dict:
    return next(item for item in summary["hazards"] if item["key"] == key)


def test_tornado_rollup_reports_the_scorers_count_and_pooled_skill(tmp_path):
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)  # issued 72..68 h ago: all matured
    _write_summary(m, m.TO_PROSPECTIVE_DIR, _tornado_summary(5))

    summary = m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text()))
    item = _hazard(summary, "to")

    assert item["forecast_storage"]["n_matured_forecasts"] == 5
    assert item["forecast_storage"]["n_scored_forecasts"] == 5          # was hardcoded 0
    assert item["forecast_storage"]["n_backlog"] == 0
    assert item["verification_status"] == "prospective_scored"           # was matured_unscored_no_evaluator
    assert item["prospective"]["status"] == "ok"                         # was evaluator_missing
    assert item["prospective"]["summary_path"] == "results/tornado_prospective/prospective_summary.json"
    assert item["auc"] == POOLED["auc"] and item["brier"] == POOLED["brier"]
    assert item["brier_skill_score"] == POOLED["bss_vs_causal_climatology"]
    assert item["brier_skill_score_reference"] == "causal_climatology"
    assert "reliability" not in item["prospective"]["pooled"]
    assert summary["system"]["scored_forecasts"] >= 5
    assert summary["system"]["rollup_violations"] == []
    on_disk = json.loads((m.RESULTS_VERIFICATION_DIR / "tornado" / "live_rollup.json").read_text())
    assert on_disk["forecast_storage"]["n_scored_forecasts"] == 5
    served = json.loads((m.VERIFICATION_DATA_DIR / "to.json").read_text())
    assert served["forecast_storage"]["n_scored_forecasts"] == 5


def test_a_live_score_needs_events_before_it_is_quoted(tmp_path):
    """With fewer than LIVE_MIN_EVENTS tornadoes, a model that always says "no" scores as well as a skilful
    one: the summary quoted "live BSS 1.00" on 83 storm forecasts and zero tornadoes (2026-10-04)."""
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    summary = _tornado_summary(5)
    for pooled in (summary.get("pooled_by_model_version") or {}).values():
        pooled["n_positive"] = m.LIVE_MIN_EVENTS - 1
    _write_summary(m, m.TO_PROSPECTIVE_DIR, summary)
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "to")
    assert item["auc"] is None and item["brier"] is None and item["brier_skill_score"] is None
    assert "too few tornadoes to score yet" in item["metric_source_label"]
    assert item["forecast_storage"]["n_scored_forecasts"] == 5          # the count is still reported


def test_a_live_record_of_another_model_version_is_never_shown_as_the_live_models(tmp_path):
    # 2026-10: the pooled live record mixed every version that ever served; with v3 live it would
    # have shown the replaced model's skill beside v3's forecasts
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    _write_summary(m, m.TO_PROSPECTIVE_DIR, _tornado_summary(5, version="tornado_gbt_v2-48637c637e01"))
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "to")
    assert item["forecast_storage"]["n_scored_forecasts"] == 5        # the count stays the scorer's
    assert item["auc"] is None and item["brier"] is None and item["brier_skill_score"] is None
    assert item["prospective"]["pooled"] is None
    assert "No storm forecast of this model version has been scored yet" in item["metric_source_label"]


def test_hurricane_rollup_is_bound_to_its_scorer_too(tmp_path):
    m = _load(tmp_path)
    _write_replays(m, "hu", "hurricane", 4)
    pooled = {"n_storm_cycles": 3, "n_events": 0, "n_storms": 2, "brier": 0.0014, "auc": None,
              "auc_withheld": "only one outcome occurred: a ranking score is undefined"}
    _write_summary(m, m.HU_PROSPECTIVE_DIR, {
        "scored_as_of": _z(NOW - dt.timedelta(hours=2)), "n_matured_forecasts": 4, "status": "ok",
        "total_storms_scored": 3, "total_ri_events": 0, "pooled": pooled,
        "mean_auc": 0.9,                       # a stale per-forecast mean in an old summary: never carried
    })
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "hu")
    assert item["forecast_storage"]["n_scored_forecasts"] == 4
    assert item["verification_status"] == "prospective_scored"
    assert item["prospective"]["status"] == "ok"
    assert item["prospective"]["total_ri_events"] == 0
    assert item["prospective"]["pooled"] == pooled and "mean_auc" not in item["prospective"]


def test_a_stale_summary_is_labelled_stale_not_zero(tmp_path):
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    _write_summary(m, m.TO_PROSPECTIVE_DIR, _tornado_summary(3, scored_as_of=NOW - dt.timedelta(days=2)))
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "to")
    assert item["verification_status"] == "prospective_stale"
    assert item["prospective"]["stale"] is True
    assert item["forecast_storage"]["n_scored_forecasts"] == 3
    assert item["forecast_storage"]["n_backlog"] == 2


def test_without_a_summary_the_backlog_is_explicit_and_no_evaluator_is_not_claimed(tmp_path):
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "to")
    assert item["verification_status"] == "matured_unscored"
    assert item["forecast_storage"]["n_scored_forecasts"] == 0
    assert item["forecast_storage"]["n_backlog"] == 5
    assert item["prospective"]["status"] == "not_run"


def test_a_pre_pooled_summary_never_promotes_a_mean_of_per_forecast_bss(tmp_path):
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    legacy = _tornado_summary(5, pooled=False)
    legacy.pop("n_scored_forecasts")
    legacy["mean_brier_skill_score"] = -0.3673  # the misleading statistic from the audit
    _write_summary(m, m.TO_PROSPECTIVE_DIR, legacy)
    item = _hazard(m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text())), "to")
    assert item["forecast_storage"]["n_scored_forecasts"] == 5
    # a summary that predates both pooling and the per-version split cannot say which model its
    # means belong to: no live skill at all (before 2026-10 its per-forecast mean AUC was shown)
    assert item["auc"] is None and item["brier"] is None
    assert item["brier_skill_score"] is None
    assert item["brier_skill_score_reference"] is None


def test_the_violation_detector_can_fail_and_pass(tmp_path):
    m = _load(tmp_path)
    summaries = {"to": _tornado_summary(5)}
    old_style = {  # exactly what the producer emitted before the fix
        "key": "to",
        "verification_status": "matured_unscored_no_evaluator",
        "forecast_storage": {"n_scored_forecasts": 0},
        "prospective": {"status": "evaluator_missing"},
    }
    violations = m._verification_rollup_violations([old_style], summaries)
    assert len(violations) == 2 and all(v.startswith("Tornado:") for v in violations)
    bound = {
        "key": "to",
        "verification_status": "prospective_scored",
        "forecast_storage": {"n_scored_forecasts": 5},
        "prospective": {"status": "ok"},
    }
    assert m._verification_rollup_violations([bound], summaries) == []
    over = dict(bound, forecast_storage={"n_scored_forecasts": 6})
    assert m._verification_rollup_violations([over], summaries)
    # No scored ledger -> nothing to contradict.
    assert m._verification_rollup_violations([old_style], {"to": {}}) == []


def test_verification_only_refuses_a_contradictory_rollup(tmp_path, monkeypatch, capsys):
    m = _load(tmp_path)
    _write_replays(m, "to", "tornado", 5)
    _write_summary(m, m.TO_PROSPECTIVE_DIR, _tornado_summary(5))
    assert m.main(["--verification-only"]) == 0

    real_binding = m._prospective_binding

    def regressed(summary, **kwargs):  # a producer that ignores its scorer again
        out = real_binding(summary, **kwargs)
        out["n_scored"] = 0
        return out

    monkeypatch.setattr(m, "_prospective_binding", regressed)
    assert m.main(["--verification-only"]) == 1
    assert "VERIFICATION ROLLUP INCONSISTENT: Tornado" in capsys.readouterr().err
    item = json.loads((m.RESULTS_VERIFICATION_DIR / "tornado" / "live_rollup.json").read_text())
    assert item["verification_status"] == "inconsistent_with_prospective_ledger"


def test_related_benchmark_states_its_downsampled_base_rate(tmp_path):
    m = _load(tmp_path)
    m.TO_RETRO_RESULTS_PATH.parent.mkdir(parents=True)
    m.TO_RETRO_RESULTS_PATH.write_text(json.dumps({
        "model": "hazardpulse_tornado_definitive_v1",
        "full": {"auc": 0.894, "brier": 0.114, "bss": 0.176, "base_rate": 0.16666666666666666},
    }))
    related = m._tornado_related_benchmark()
    assert related["evaluated_base_rate"] == pytest.approx(1 / 6)
    assert "downsampled base rate of 0.167" in related["label"]
