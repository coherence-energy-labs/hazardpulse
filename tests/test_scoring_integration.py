"""Tests for the scorer<->trust glue (hazardpulse.trust.scoring)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hazardpulse.trust import verify_forecast_receipt
from hazardpulse.trust.scoring import enrich_cells, load_forecaster, load_signer
from hazardpulse.trust.venn_abers import VennAbersCalibrator


def _write_calibration(models_dir, hazard="earthquake"):
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 8000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    cal = VennAbersCalibrator().fit(p, y)
    # the evidence a fitted record carries (scripts/fit_calibration.py); without it the scorers refuse
    # the curve (hazardpulse.trust.scoring.calibrator_admissible)
    raw_brier = float(np.mean((p - y) ** 2))
    record = {"hazard": hazard, "model_version": "eq_coherence_v1_0", "inflated": False,
              "n_calibration": int(p.size), "n_positive": int(y.sum()),
              "metrics_before": {"brier": raw_brier}, "metrics_after_heldout": {"brier": raw_brier * 0.99},
              "calibrator": cal.to_dict()}
    path = models_dir / f"{hazard}_calibration.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_load_forecaster_none_when_absent(tmp_path):
    assert load_forecaster("earthquake", models_dir=tmp_path) is None


def test_enrich_cells_adds_calibration_and_intervals(tmp_path):
    _write_calibration(tmp_path)
    tf = load_forecaster("earthquake", models_dir=tmp_path)
    assert tf is not None and tf.fitted
    cells = [{"probability": v} for v in (0.02, 0.2, 0.5, 0.8)]
    out = enrich_cells(cells, tf, issued_at="2026-06-25T03:00:00Z")
    for c in out:
        assert c["confidence_lo"] is not None and c["confidence_hi"] is not None
        assert c["confidence_lo"] <= c["probability"] <= c["confidence_hi"] + 1e-6
        assert c["calibrated"] is True
        assert c["receipt_sha256"] and c["receipt"]["spec"].startswith("hazardpulse/forecast/")


def test_enrich_cells_noop_without_forecaster():
    cells = [{"probability": 0.3}]
    out = enrich_cells(cells, None)
    assert out[0] == {"probability": 0.3}          # untouched, honest no-op


def test_signed_receipts_when_signer_configured(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from hazardpulse.trust import load_ed25519_pubkey

    key = Ed25519PrivateKey.generate()
    monkeypatch.setenv("HAZARDPULSE_SIGNING_KEY", key.private_bytes_raw().hex())
    signer = load_signer()
    assert signer is not None

    _write_calibration(tmp_path)
    tf = load_forecaster("earthquake", models_dir=tmp_path, signer=signer)
    cells = enrich_cells([{"probability": 0.4}], tf, issued_at="2026-06-25T03:00:00Z")
    pubkey = load_ed25519_pubkey(key.public_key().public_bytes_raw())
    assert verify_forecast_receipt(cells[0]["receipt"], pubkey=pubkey) is True
