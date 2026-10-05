"""A published risk band must describe the published probability.

Regression for the 2026-10 finding: the earthquake scorer banded each cell from its
RAW score, then the trust layer replaced ``probability`` with the calibrated value,
so 29,263 of 32,516 calibrated earthquake cells (and the live pulse: 4.54% labelled
"critical") carried a band contradicting their own probability."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.trust.scoring import (
    band_contradictions,
    enrich_cells,
    load_forecaster,
    rederive_band,
)
from hazardpulse.trust.venn_abers import VennAbersCalibrator

ROOT = Path(__file__).resolve().parents[1]


def _eq_module():
    name = "hp_fse_band_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "fetch_and_score_earthquake.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _shrinking_forecaster(tmp_path):
    """A calibrator that maps high raw scores to low probabilities (rare events) --
    the live earthquake situation (raw 0.996 -> published 0.0454)."""
    rng = np.random.RandomState(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < 0.05 * p).astype(float)
    cal = VennAbersCalibrator().fit(p, y)
    # with the evidence that makes the scorers apply it (calibrator_admissible): ~500 events, and a
    # held-out Brier that beats the raw scores (which overstate the rate 20x)
    rec = {"hazard": "earthquake", "model_version": "eq_coherence_v1_0", "inflated": False,
           "n_calibration": int(p.size), "n_positive": int(y.sum()),
           "metrics_before": {"brier": float(np.mean((p - y) ** 2))},
           "metrics_after_heldout": {"brier": float(np.mean((0.05 * p - y) ** 2))},
           "calibrator": cal.to_dict()}
    (tmp_path / "earthquake_calibration.json").write_text(json.dumps(rec), encoding="utf-8")
    return load_forecaster("earthquake", models_dir=tmp_path)


def test_rederive_band_keeps_the_raw_band_for_audit():
    cell = {"probability": 0.0454, "risk_band": "critical"}
    bands = lambda p: "critical" if p >= 0.5 else "low"   # noqa: E731
    rederive_band(cell, bands)
    assert cell["risk_band"] == "low" and cell["raw_risk_band"] == "critical"


def test_calibration_without_band_fn_leaves_contradictions(tmp_path):
    """Documents the hazard the band_fn hook exists for."""
    eq = _eq_module()
    tf = _shrinking_forecaster(tmp_path)
    cells = [{"probability": p, "risk_band": eq._risk_band(p)} for p in (0.996, 0.9, 0.6, 0.35)]
    enrich_cells(cells, tf, issued_at="2026-10-01T22:00:00Z")
    assert band_contradictions(cells, eq._risk_band)        # the pre-fix behaviour


def test_enrich_cells_with_band_fn_never_contradicts(tmp_path):
    eq = _eq_module()
    tf = _shrinking_forecaster(tmp_path)
    cells = [{"probability": p, "risk_band": eq._risk_band(p)} for p in (0.996, 0.9, 0.6, 0.35, 0.01)]
    enrich_cells(cells, tf, issued_at="2026-10-01T22:00:00Z", band_fn=eq._risk_band)
    assert band_contradictions(cells, eq._risk_band) == []
    assert all("raw_risk_band" in c for c in cells)


def test_earthquake_trust_step_publishes_consistent_bands(tmp_path):
    """The calibrated path maps the WHOLE scored grid (calibrate_scored_grid) and the listed cells are
    banded from what they publish -- the calibrated value -- so no band describes the raw score."""
    import datetime as dt

    from hazardpulse.earthquake import operational_forecast as of

    eq = _eq_module()
    tf = _shrinking_forecaster(tmp_path)
    grid = np.full(of.N_CELLS, 1e-4)
    for flat, p in ((40 * of.N_LON + 160, 0.996), (41 * of.N_LON + 160, 0.98), (10 * of.N_LON + 20, 0.7),
                    (11 * of.N_LON + 21, 0.2)):
        grid[flat] = p
    operational = eq.calibrate_scored_grid(
        {"probability": grid, "lambda_long": grid * 0.5, "lambda_short": grid * 0.5}, tf)
    cells = eq.score_grid_cells([], candidate_events=[], now=dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc),
                                operational=operational)
    eq.check_published_bands(cells)                       # raises on any contradiction
    assert band_contradictions(cells, eq._risk_band) == []
    top = next(c for c in cells if (c["row"], c["col"]) == (40, 160))
    assert top["raw_probability"] == pytest.approx(0.996) and eq._risk_band(top["raw_probability"]) == "critical"
    assert top["probability"] < 0.5 and top["risk_band"] == eq._risk_band(top["probability"])
    assert eq.listed_cells_off_grid(cells, operational["probability"]) == []


def test_earthquake_trust_step_refuses_a_contradiction():
    """A band that contradicts the published probability fails the run loudly instead of publishing."""
    eq = _eq_module()
    with pytest.raises(eq.RiskBandContradiction):
        eq.check_published_bands([{"probability": 0.0454, "risk_band": "critical"}])
    eq.check_published_bands([{"probability": 0.0454, "risk_band": eq._risk_band(0.0454)}])   # can pass
