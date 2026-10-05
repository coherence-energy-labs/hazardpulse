"""A live calibrator may replace a model's published probabilities only when it has earned it.

Witness (2026-10-04 22:47Z): a Venn-Abers calibrator fitted to 794 tornado storm-forecasts containing zero
tornadoes was applied to every live storm. It mapped storms the model put at 0.03% to 16.7-33.3% (567-1,822x)
while its own held-out Brier was 467x worse than the model's. These tests pin the rule both ways.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hazardpulse.trust.scoring import (MIN_CALIBRATION_EVENTS, calibration_events, calibrator_admissible,
                                       load_forecaster)

ROOT = Path(__file__).resolve().parents[1]

ZERO_EVENT_TORNADO = {   # the record that inflated the site, as committed on 2026-10-04 (metrics verbatim)
    "model_version": "tornado_v3-05a06c843c87", "n_calibration": 794, "inflated": False,
    "metrics_before": {"ece": 1.9e-05, "brier": 1e-08, "brier_skill_score": None, "base_rate": 0.0},
    "metrics_after_heldout": {"ece": 0.037492, "brier": 0.00467138, "brier_skill_score": None, "base_rate": 0.0},
}


def test_a_calibrator_fitted_on_zero_events_is_refused():
    ok, why = calibrator_admissible(ZERO_EVENT_TORNADO)
    assert not ok and "0 events" in why


def test_a_calibrator_worse_than_the_model_on_held_out_cells_is_refused_however_many_events():
    rec = {**ZERO_EVENT_TORNADO, "n_positive": 500}
    ok, why = calibrator_admissible(rec)
    assert not ok and "does not beat" in why


def test_a_calibrator_with_enough_events_that_beats_the_model_is_admitted():
    rec = {"inflated": False, "n_calibration": 100_000, "n_positive": MIN_CALIBRATION_EVENTS,
           "metrics_before": {"brier": 0.0029, "base_rate": 0.0003},
           "metrics_after_heldout": {"brier": 0.0008}}
    assert calibrator_admissible(rec)[0]
    assert not calibrator_admissible({**rec, "n_positive": MIN_CALIBRATION_EVENTS - 1})[0]
    assert not calibrator_admissible({**rec, "metrics_after_heldout": None})[0]     # no comparison, no curve
    assert not calibrator_admissible({**rec, "inflated": True})[0]


def test_events_are_counted_from_the_base_rate_when_a_record_predates_n_positive():
    assert calibration_events({"n_calibration": 7_172_100, "metrics_before": {"base_rate": 0.00082695}}) == 5931
    assert calibration_events({"n_calibration": 794, "metrics_before": {"base_rate": 0.0}}) == 0


def test_the_scorers_loader_refuses_the_zero_event_record(tmp_path):
    """load_forecaster is the one door every scorer uses; it must not open for that record."""
    from hazardpulse.trust.venn_abers import VennAbersCalibrator
    cal = VennAbersCalibrator(min_calibration=200, max_groups=512)
    scores = np.linspace(0.0001, 0.01, 50)
    cal.fit_grouped(scores, np.zeros(50), np.full(50, 16.0))
    (tmp_path / "tornado_calibration.json").write_text(
        json.dumps({**ZERO_EVENT_TORNADO, "calibrator": cal.to_dict()}), encoding="utf-8")
    assert load_forecaster("tornado", models_dir=tmp_path) is None


def test_the_committed_calibration_records_carry_no_admissible_curve_without_events():
    """Whatever is committed now: any record the scorers would apply has the events and the held-out win."""
    for p in sorted((ROOT / "results" / "calibration").glob("*_calibration.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "calibrator" not in rec:
            continue
        ok, _ = calibrator_admissible(rec)
        if ok:
            assert calibration_events(rec) >= MIN_CALIBRATION_EVENTS, p.name
            assert rec["metrics_after_heldout"]["brier"] < rec["metrics_before"]["brier"], p.name
