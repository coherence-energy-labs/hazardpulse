"""v10.1 rapid-intensification forecast (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 2).

The whole 24-h exceedance curve P(dV >= k), k = 15..45 kt, from one threshold-stacked model (five
seed members, each a NumPy LightGBM payload, averaged) wherever the cycle has the early guidance it
was trained on (DSHP, IVCN and NNIC); otherwise each threshold is NOAA DTOPS's own published value
(SHIPS-RII where DTOPS is missing; none where NOAA publishes no 24-h value at that threshold).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hazardpulse.hurricane import ri_model
from hazardpulse.hurricane import ri_v9_features as fx
from hazardpulse.tornado import lgbm_payload as lp

SCHEMA = "hazardpulse.hurricane_ri_v10/1"
MODEL_PATH = ri_model.RESULTS / "models" / "hurricane_ri_v10.json"
# amendment 3's carried challenger (V5: v10.1's inputs, monotone in the guidance and NOAA's
# probabilities); same schema and inputs, served in shadow beside v10.1 under its own label
V10_2_PATH = ri_model.RESULTS / "models" / "hurricane_ri_v10_2.json"
# amendment 5's carried challenger (V8: V5 + the 14 IR structure features from GMGSI)
V10_3_PATH = ri_model.RESULTS / "models" / "hurricane_ri_v10_3.json"
GATE_AIDS = ("DSHP", "IVCN", "NNIC")


def allowed_feature_sets() -> dict[str, list[str]]:
    """The input sets a v10-schema artifact may declare (the order is the model's)."""
    from hazardpulse.hurricane import ir_features
    base = list(fx.names_for("ONH"))
    return {"ONH": base, "ONH+IR": base + list(ir_features.IR_NAMES)}


def needs_ir(art: Mapping) -> bool:
    return art["feature_names"] == allowed_feature_sets()["ONH+IR"]
NOAA_24H = (25, 30, 35, 40)          # the 24-h thresholds NOAA's aids publish


def canonical_bytes(art: Mapping) -> bytes:
    return json.dumps(art, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load(path: str | Path = MODEL_PATH) -> tuple[dict, str]:
    art = json.loads(Path(path).read_bytes().decode("utf-8"))
    if art.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {art.get('schema')!r}, expected {SCHEMA!r}")
    names = list(art["feature_names"])
    if names not in allowed_feature_sets().values():
        raise ValueError(f"{path}: inputs are not a v10 feature set (ONH, or ONH + IR)")
    for m in art["members"]:
        if m["feature_names"] != names + ["threshold_kt"]:
            raise ValueError(f"{path}: a member's inputs are not the feature set + threshold")
    digest = hashlib.sha256(canonical_bytes(art)).hexdigest()
    return art, f"{art['model_name']}-{digest[:12]}"


def recompute(art: Mapping, inputs: Mapping[str, float | None]) -> dict[str, float]:
    """The model's exceedance curve from a forecast record's stored inputs alone (None = missing),
    for auditing a past forecast against the artifact its record names."""
    row = np.array([[np.nan if inputs.get(n) is None else float(inputs[n]) for n in art["feature_names"]]])
    return {str(int(k)): float(predict_matrix(art, row, k)[0]) for k in art["thresholds_kt"]}


def predict_matrix(art: Mapping, X: np.ndarray, k: float) -> np.ndarray:
    Xk = np.hstack([np.asarray(X, np.float64), np.full((len(X), 1), float(k))])
    return np.mean([lp.predict_proba(m, Xk) for m in art["members"]], axis=0)


def predict(art: Mapping, version: str, records: Iterable, cycle, basin: str,
            ri_pcts: Mapping[tuple[str, str], float | None], extra: Mapping[str, float] | None = None) -> dict:
    """The exceedance curve for one cycle, with what produced it -- labelled by the artifact's own
    ``label`` (v10.1's artifact has none and is "v10.1"). ``extra`` carries inputs read outside the
    decks (the IR features of an ONH+IR artifact); an input it lacks is NaN, as in training."""
    label = str(art.get("label") or "v10.1")
    records = list(records)
    f = fx.adeck_features(fx.cycle_table(records, cycle), basin)
    f.update(fx.ri_features(ri_pcts))
    if extra:
        f.update(extra)
    gate_ok = all(math.isfinite(f[f"dv24_{a}"]) for a in GATE_AIDS)
    X = fx.vector(f, art["feature_names"])[None, :]
    model = {int(k): float(predict_matrix(art, X, k)[0]) for k in art["thresholds_kt"]}
    noaa = {}
    for k in NOAA_24H:
        d, s = ri_pcts.get(("DTOP", f"{k}/24")), ri_pcts.get(("RIOD", f"{k}/24"))
        noaa[k] = (d if d is not None else s) / 100.0 if (d is not None or s is not None) else None
    if gate_ok:
        probs, source = model, label
    else:
        probs = {k: noaa.get(k) for k in art["thresholds_kt"]}
        source = f"DTOPS ({label} gate: early guidance missing)"
    return {"probability": None if probs.get(30) is None else round(probs[30], 4),
            "probabilities": {str(k): (None if v is None else round(v, 4)) for k, v in probs.items()},
            "model_probabilities": {str(k): round(v, 4) for k, v in model.items()},
            "noaa_24h": {str(k): (None if v is None else round(v, 4)) for k, v in noaa.items()},
            "source": source, "gate_ok": gate_ok,
            "gate_missing": [a for a in GATE_AIDS if not math.isfinite(f[f"dv24_{a}"])],
            "dtops_pct": ri_pcts.get(("DTOP", "30/24")), "riod_pct": ri_pcts.get(("RIOD", "30/24")),
            "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"), "model_version": version,
            "inputs": fx.record_inputs(f, art["feature_names"]),
            **({"ir_inputs": {n: (round(float(f[n]), 4) if math.isfinite(float(f.get(n, math.nan))) else None)
                              for n in art["feature_names"] if n.startswith(("ir_", "d_ir_"))}}
               if needs_ir(art) else {})}
