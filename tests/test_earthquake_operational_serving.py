"""The live earthquake scorer serves the operational forecast (docs/EARTHQUAKE_FORECAST_PROGRAM.md).

* the primary probability of every listed cell is the operational grid's value, and the
  replay artifact carries the whole grid, which the prospective verifier then scores
  instead of a default 0;
* a calibrator fitted to another model is never applied, and the calibration pool only
  mixes forecasts of one model;
* operational forecasts carry receipts that verify, with no invented interval.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.earthquake import operational_forecast as of
from hazardpulse.earthquake.coherence_engine import grid_cell_to_latlon
from hazardpulse.trust.forecast import verify_forecast_receipt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fse():
    return _load("hazardpulse_fse_operational_test", "scripts/fetch_and_score_earthquake.py")


@pytest.fixture(scope="module")
def verifier():
    return _load("hazardpulse_score_eq_prospective_operational_test", "scripts/score_earthquake_prospective.py")


class _Forecaster:
    def __init__(self, version):
        self.model_version = version


def test_a_calibrator_of_another_model_is_never_applied(fse):
    assert fse.trust_layer_applies(_Forecaster("eq_coherence_v1_0"), "eq_operational_B_v1-0123456789ab") is False
    assert fse.trust_layer_applies(None, "eq_operational_B_v1-0123456789ab") is False
    assert fse.trust_layer_applies(_Forecaster("eq_operational_B_v1-0123456789ab"),
                                   "eq_operational_B_v1-0123456789ab") is True


def test_operational_receipts_verify_and_claim_no_interval(fse):
    cells = [{"row": 40, "col": 160, "probability": 0.0123}, {"row": 3, "col": 7, "probability": 2.5e-05}]
    fse.attach_operational_receipts(cells, model_version="eq_operational_B_v1-0123456789ab",
                                    model_sha256="ab" * 32, input_sha256="cd" * 32,
                                    issued_at="2026-10-02T12:00:00Z", signer=None)
    seen = set()
    for cell in cells:
        r = cell["receipt"]
        assert verify_forecast_receipt(r)
        assert r["probability"] == r["raw_probability"] == cell["probability"]
        assert r["confidence_lo"] is None and r["confidence_hi"] is None
        assert r["uncertainty_class"] == "no_interval"
        seen.add(r["input_sha256"])
        tampered = dict(r, probability=0.5)
        assert not verify_forecast_receipt(tampered)
    assert len(seen) == 2          # each cell's receipt binds its own cell


def test_score_grid_cells_publishes_the_operational_grid(fse):
    now = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    lat, lon = grid_cell_to_latlon(40, 160)
    history = [{"time": (now - dt.timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "latitude": lat + 0.1 * (d % 5), "longitude": lon - 0.1 * (d % 3), "mag": 3.0 + 0.1 * (d % 10),
                "depth": 10.0} for d in range(1, 25)]
    grid = np.full(of.N_CELLS, 1e-5)
    grid[40 * of.N_LON + 160] = 0.031
    grid[10 * of.N_LON + 20] = 0.42          # a quiet cell with a high forecast must still be listed
    operational = {"probability": grid, "lambda_long": grid * 0.5, "lambda_short": grid * 0.5}
    cells = fse.score_grid_cells(history, candidate_events=history, now=now, operational=operational)
    by_key = {(c["row"], c["col"]): c for c in cells}
    assert by_key[(10, 20)]["probability"] == pytest.approx(0.42)
    assert by_key[(40, 160)]["probability"] == pytest.approx(0.031)
    assert all(c["scoring_tier"] == "tier1_operational" for c in cells)
    assert all(c["probability"] == pytest.approx(grid[c["row"] * of.N_LON + c["col"]], rel=1e-5) for c in cells)
    assert len(cells) >= fse.TOP_PROBABILITY_CELLS
    assert (cells[0]["row"], cells[0]["col"]) == (10, 20)       # sorted by probability
    assert all(c["model_id"] == fse.MODEL_VERSION for c in cells)


def test_replay_carries_the_full_grid_and_the_verifier_scores_it(fse, verifier, tmp_path):
    now = dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc)
    grid = np.full(of.N_CELLS, 1e-4)
    hot = 30 * of.N_LON + 100
    grid[hot] = 0.2
    path = fse.write_replay_artifact([], now, forecast_id="eq_fcst_20260201_0000", n_history_events=1,
                                     n_recent_events=0, replay_dir=tmp_path, update_index=False,
                                     probability_grid=grid, operational_meta={"model_version": "x"})
    artifact = json.loads(path.read_text(encoding="utf-8"))
    values = [float(v) for v in artifact["probability_grid"].split(",")]
    assert len(values) == of.N_CELLS
    assert values[hot] == 0.2 and values[0] == 1e-4
    np.testing.assert_array_equal(verifier.forecast_grid(artifact), np.asarray(values))
    lat, lon = grid_cell_to_latlon(30, 100)
    observed = [{"time": "2026-02-10T00:00:00Z", "latitude": lat, "longitude": lon, "depth": 10.0,
                 "mag": 6.3, "id": "e1"}]
    res = verifier.score_single_forecast(artifact, observed, tmp_path)
    # the hit cell holds the single highest forecast: AUC 1, not the 0.5 of an all-default grid
    assert res["auc"] == pytest.approx(1.0)
    assert res["n_positive_cells"] == 1
    no_grid = dict(artifact)
    no_grid.pop("probability_grid")
    assert verifier.score_single_forecast(no_grid, observed, tmp_path)["auc"] == pytest.approx(0.5)


def test_calibration_pool_holds_one_model(verifier, tmp_path):
    acc = {}
    verifier._accumulate_calibration(acc, np.array([0.1, 0.2]), np.array([1.0, 0.0]))
    path = verifier.write_calibration_dataset(tmp_path, acc, model_version="eq_operational_B_v1-0123456789ab")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model_version"] == "eq_operational_B_v1-0123456789ab"
    assert data["n"] == 2
