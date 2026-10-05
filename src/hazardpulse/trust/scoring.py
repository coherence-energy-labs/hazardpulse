"""Glue between the live scorers and the trust layer.

A scorer produces a list of cell/storm dicts each carrying a raw ``probability``.
These helpers load the fitted calibrator (if one has been produced from matured
forecasts yet), load the Ed25519 signing key (if configured), and enrich each
cell in place with a calibrated probability, an honest [confidence_lo,
confidence_hi] band, an abstention decision, and a signed re-runnable receipt.

Designed to fail safe: if no calibration record exists yet, ``load_forecaster``
returns None and the scorer keeps emitting raw (uncalibrated) forecasts — honest
degradation, never a crash. Once the prospective scorer + ``fit_calibration``
have run, the calibrator appears and forecasts start being calibrated
automatically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from .forecast import TrustedForecaster

__all__ = ["load_signer", "load_forecaster", "enrich_cell", "enrich_cells", "publish_public_key",
           "rederive_band", "band_contradictions"]

BandFn = Callable[[float], str]

_SIGNING_KEY_ENV = "HAZARDPULSE_SIGNING_KEY"   # 32-byte Ed25519 seed, hex-encoded


def load_signer(env_var: str = _SIGNING_KEY_ENV):
    """Return an Ed25519 private key from a hex env var, or None (unsigned receipts)."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw))
    except Exception:
        return None


def publish_public_key(signer, out_path: Path) -> dict | None:
    """Write the signer's public key (hex) so third parties can verify receipts."""
    if signer is None:
        return None
    try:
        pub = signer.public_key().public_bytes_raw().hex()
    except Exception:
        return None
    payload = {"alg": "ed25519", "public_key_hex": pub,
               "verify": "verify_forecast_receipt(receipt, pubkey=load_ed25519_pubkey(bytes.fromhex(public_key_hex)))"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


MIN_CALIBRATION_EVENTS = 30


def calibration_events(record: dict) -> int:
    """Positive outcomes in the data a calibration record was fitted on."""
    if record.get("n_positive") is not None:
        return int(record["n_positive"])
    before = record.get("metrics_before") or {}
    return int(round(float(before.get("base_rate") or 0.0) * float(record.get("n_calibration") or 0)))


def calibrator_admissible(record: dict) -> tuple[bool, str]:
    """Whether a fitted calibrator may REPLACE the model's own published probabilities.

    All of:
    * it is not inflated (fitted from too few groups to be a curve at all);
    * its data holds at least MIN_CALIBRATION_EVENTS positive outcomes;
    * on cells it never saw (``metrics_after_heldout``) its Brier score beats the model's own
      probabilities on the same cells (``metrics_before``).

    Why: on 2026-10-04 at 22:47Z a Venn-Abers calibrator was fitted to 794 tornado storm-forecasts with
    ZERO tornadoes. With no positives it maps every storm to 1/(group size + 2), so storms the model put at
    0.03% were published at 16.7-33.3% -- inflation of 567-1,822x on every published tornado number --
    although its own held-out Brier was 4.67e-3 against the model's 1e-8, 467x worse. Nothing checked
    either. The earthquake record was set to do the same from a single matured window (2026-11-02).
    """
    if record.get("inflated"):
        return False, "inflated (too few groups to fit a curve)"
    events = calibration_events(record)
    if events < MIN_CALIBRATION_EVENTS:
        return False, f"{events} events in its data (needs {MIN_CALIBRATION_EVENTS})"
    before = (record.get("metrics_before") or {}).get("brier")
    held = (record.get("metrics_after_heldout") or {}).get("brier")
    if before is None or held is None:
        return False, "no held-out comparison with the model's own probabilities"
    if not float(held) < float(before):
        return False, f"held-out Brier {held:.3g} does not beat the model's own {before:.3g}"
    return True, f"{events} events; held-out Brier {held:.3g} beats the model's own {before:.3g}"


def load_forecaster(hazard: str, *, models_dir: Path | None = None, signer=None,
                    alpha: float = 0.1) -> TrustedForecaster | None:
    """Build a TrustedForecaster from results/calibration/<hazard>_calibration.json.

    Returns None -- the scorer then publishes the model's own probabilities -- if no record exists or the
    record is not admissible (``calibrator_admissible``).
    """
    models_dir = models_dir or (Path(__file__).resolve().parents[3] / "results" / "calibration")
    path = models_dir / f"{hazard}_calibration.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "calibrator" not in record:
            return None
        ok, why = calibrator_admissible(record)
        if not ok:
            print(f"  Trust layer: the {hazard} calibrator is not applied -- {why}; "
                  "publishing the model's own probabilities.")
            return None
        return TrustedForecaster.from_calibration_dict(record, signer=signer, alpha=alpha)
    except Exception:
        return None


def rederive_band(cell: dict, band_fn: BandFn, *, prob_key: str = "probability",
                  band_key: str = "risk_band") -> dict:
    """Re-derive a probability-derived label from the probability the cell PUBLISHES.

    A scorer bands a cell from its raw score; when the trust layer then replaces the
    probability with a calibrated one, that band describes a number nobody sees
    (e.g. an earthquake cell published at 4.54% still labelled "critical"). The
    pre-calibration band is kept under ``raw_<band_key>`` for audit. Hazard-agnostic:
    pass the hazard's own probability->band mapping.
    """
    prob = cell.get(prob_key)
    if prob is None:
        return cell
    if band_key in cell and f"raw_{band_key}" not in cell:
        cell[f"raw_{band_key}"] = cell[band_key]
    cell[band_key] = band_fn(float(prob))
    return cell


def band_contradictions(cells: list[dict], band_fn: BandFn, *, prob_key: str = "probability",
                        band_key: str = "risk_band") -> list[int]:
    """Indices of cells whose label disagrees with band_fn(published probability)."""
    return [i for i, c in enumerate(cells)
            if c.get(prob_key) is not None and band_key in c
            and c[band_key] != band_fn(float(c[prob_key]))]


def enrich_cell(cell: dict, forecaster: TrustedForecaster, *, prob_key: str = "probability",
                feature_key: str | None = None, data_health_key: str = "data_health_ok",
                issued_at: str | None = None, band_fn: BandFn | None = None,
                band_key: str = "risk_band") -> dict:
    """Enrich one cell/storm dict in place with calibrated probability + interval +
    abstention + signed receipt. The raw probability is preserved under
    ``raw_probability`` for audit. If ``band_fn`` is given, ``cell[band_key]`` is
    re-derived from the probability actually published (see rederive_band)."""
    raw = cell.get(prob_key)
    if raw is None:
        return cell
    features = cell.get(feature_key) if feature_key else None
    health = bool(cell.get(data_health_key, True))
    res = forecaster.forecast_one(float(raw), features, data_health_ok=health,
                                  issued_at=issued_at)
    cell["raw_probability"] = res.raw_probability
    # Calibrated probability replaces the displayed/ranked probability when available.
    if res.probability is not None:
        cell[prob_key] = round(float(res.probability), 4)
    cell["confidence_lo"] = None if res.confidence_lo is None else round(res.confidence_lo, 4)
    cell["confidence_hi"] = None if res.confidence_hi is None else round(res.confidence_hi, 4)
    cell["uncertainty_class"] = res.uncertainty_class
    cell["abstained"] = res.abstained
    cell["abstain_reason"] = res.abstain_reason
    cell["gateway_mode"] = res.gateway_mode
    cell["calibrated"] = True
    cell["receipt"] = res.receipt
    cell["receipt_sha256"] = res.receipt.get("receipt_sha256")
    if band_fn is not None:
        rederive_band(cell, band_fn, prob_key=prob_key, band_key=band_key)
    return cell


def enrich_cells(cells: list[dict], forecaster: TrustedForecaster | None, *,
                 prob_key: str = "probability", feature_key: str | None = None,
                 data_health_key: str = "data_health_ok", issued_at: str | None = None,
                 resort: bool = True, band_fn: BandFn | None = None,
                 band_key: str = "risk_band") -> list[dict]:
    """Enrich a list of cells. If ``forecaster`` is None, returns cells untouched
    (honest no-op until a calibrator has been produced). Pass the hazard's
    probability->band mapping as ``band_fn`` whenever the cells carry a band, so the
    band always describes the probability that is published."""
    if forecaster is None:
        return cells
    for cell in cells:
        enrich_cell(cell, forecaster, prob_key=prob_key, feature_key=feature_key,
                    data_health_key=data_health_key, issued_at=issued_at,
                    band_fn=band_fn, band_key=band_key)
    if resort:
        cells.sort(key=lambda c: (c.get("abstained", False),
                                  -(c.get(prob_key) if c.get(prob_key) is not None else -1.0)))
    return cells
