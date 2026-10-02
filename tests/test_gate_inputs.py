"""The publish gates must be fed inputs that can make them fail.

Regression for the 2026-10 finding: build_site_artifacts hard-coded
data_age_seconds=0.0 (G1_SOURCE_FRESHNESS could not fail -- the hurricane forecast
live since 2026-05-26 passed it) and fed G4_CALIBRATION_FLOOR the calibrator's
in-sample metrics (ECE ~0 by construction)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc


def _script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def site(tmp_path, monkeypatch):
    bsa = _script("hp_bsa_gate_inputs_test", "scripts/build_site_artifacts.py")
    dist = tmp_path / "dist"
    (dist / "data" / "replay").mkdir(parents=True)
    (tmp_path / "results" / "calibration").mkdir(parents=True)
    monkeypatch.setattr(bsa, "ROOT", tmp_path)
    monkeypatch.setattr(bsa, "DIST", dist)
    bsa._REPLAY_READ_CACHE.clear()
    return bsa, tmp_path


def _entry(bsa, root: Path, fid: str, hazard: str, issued: str) -> dict:
    path = root / "dist" / "data" / "replay" / f"{fid}.json"
    path.write_text(json.dumps({"forecast_id": fid, "hazard": hazard, "issued_at": issued,
                                "model_version": "m1", "storms": []}), encoding="utf-8")
    return {"forecast_id": fid, "hazard": hazard, "issued_at": issued, "model_version": "m1",
            "replay_artifact": "/data/replay/" + path.name}


def _gate(decision: dict, gate_id: str) -> dict:
    return next(g for g in decision["gates"] if g["gate_id"] == gate_id)


def test_a_forecast_left_live_for_months_fails_source_freshness(site):
    bsa, root = site
    entries = [_entry(bsa, root, "hu_fcst_20260526_1250", "hurricane", "2026-05-26T12:50:43Z")]
    now = dt.datetime(2026, 10, 1, 22, 30, tzinfo=UTC)
    (decision,) = bsa._build_gate_decisions(entries, {"hazards": []}, now=now)
    g1 = _gate(decision, "G1_SOURCE_FRESHNESS")
    assert g1["outcome"] == "block" and "stale" in g1["reason"]


def test_freshness_is_measured_until_the_next_forecast_replaced_it(site):
    bsa, root = site
    entries = [
        _entry(bsa, root, "eq_fcst_20261001_0000", "earthquake", "2026-10-01T00:00:00Z"),
        _entry(bsa, root, "eq_fcst_20261002_1600", "earthquake", "2026-10-02T16:00:00Z"),  # 40 h gap
        _entry(bsa, root, "eq_fcst_20261002_2200", "earthquake", "2026-10-02T22:00:00Z"),
    ]
    now = dt.datetime(2026, 10, 2, 23, 0, tzinfo=UTC)
    ages = bsa._live_data_age_seconds(entries, now)
    assert ages == {"eq_fcst_20261001_0000": 40 * 3600.0,
                    "eq_fcst_20261002_1600": 6 * 3600.0,
                    "eq_fcst_20261002_2200": 3600.0}
    outcomes = [_gate(d, "G1_SOURCE_FRESHNESS")["outcome"]
                for d in bsa._build_gate_decisions(entries, {"hazards": []}, now=now)]
    assert outcomes == ["block", "pass", "pass"]          # 40 h > the 36 h earthquake hard limit


def test_calibration_gate_ignores_in_sample_metrics(site):
    bsa, root = site
    rec = {"metrics_after": {"ece": 0.0, "brier_skill_score": 0.5}}       # in-sample only
    (root / "results" / "calibration" / "earthquake_calibration.json").write_text(json.dumps(rec))
    entries = [_entry(bsa, root, "eq_fcst_20261001_2200", "earthquake", "2026-10-01T22:00:00Z")]
    now = dt.datetime(2026, 10, 1, 22, 30, tzinfo=UTC)
    (d,) = bsa._build_gate_decisions(entries, {"hazards": []}, now=now)
    assert _gate(d, "G4_CALIBRATION_FLOOR")["outcome"] == "degrade"

    rec["metrics_after_heldout"] = {"ece": 0.3, "brier_skill_score": 0.1}  # out-of-sample, bad ECE
    (root / "results" / "calibration" / "earthquake_calibration.json").write_text(json.dumps(rec))
    (d,) = bsa._build_gate_decisions(entries, {"hazards": []}, now=now)
    assert _gate(d, "G4_CALIBRATION_FLOOR")["outcome"] == "block"


def test_heldout_calibration_metrics_are_not_in_sample():
    fc = _script("hp_fitcal_heldout_test", "scripts/fit_calibration.py")
    rng = np.random.default_rng(11)
    scores = np.linspace(0.0, 1.0, 400)
    total = np.full(scores.size, 4.0)
    pos = rng.binomial(4, 0.2, scores.size).astype(float)     # labels independent of the score
    from hazardpulse.trust.venn_abers import VennAbersCalibrator

    cal = VennAbersCalibrator(min_calibration=200, max_groups=512).fit_grouped(scores, pos, total)
    in_sample = fc._metrics(cal.predict(scores)[0], pos, total)
    held = fc.heldout_metrics(scores, pos, total)
    assert held is not None
    assert held["brier"] > in_sample["brier"]                 # in-sample fit flatters itself
    assert held["brier_skill_score"] < in_sample["brier_skill_score"]
