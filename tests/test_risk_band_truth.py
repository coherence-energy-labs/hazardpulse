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
    rec = {"hazard": "earthquake", "model_version": "eq_coherence_v1_0", "calibrator": cal.to_dict()}
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
    eq = _eq_module()
    tf = _shrinking_forecaster(tmp_path)
    cells = [{"probability": p, "risk_band": eq._risk_band(p), "conditions_met": 0}
             for p in (0.996, 0.98, 0.7, 0.2)]
    eq.apply_trust_layer(cells, tf, issued_at="2026-10-01T22:00:00Z")
    assert band_contradictions(cells, eq._risk_band) == []
    top = cells[0]
    assert top["raw_probability"] >= 0.5 and top["raw_risk_band"] == "critical"
    assert top["risk_band"] == eq._risk_band(top["probability"])


def test_earthquake_trust_step_refuses_a_contradiction(tmp_path, monkeypatch):
    """If enrichment ever stops re-deriving the band, the run fails loudly instead of
    publishing (the RiskBandContradiction is not swallowed by the trust-layer guard)."""
    eq = _eq_module()
    tf = _shrinking_forecaster(tmp_path)
    import hazardpulse.trust.scoring as scoring

    real = scoring.enrich_cells

    def no_band(cells, forecaster, **kw):
        kw.pop("band_fn", None)
        return real(cells, forecaster, **kw)

    monkeypatch.setattr(scoring, "enrich_cells", no_band)
    cells = [{"probability": 0.996, "risk_band": "critical"}]
    with pytest.raises(eq.RiskBandContradiction):
        eq.apply_trust_layer(cells, tf, issued_at="2026-10-01T22:00:00Z")
