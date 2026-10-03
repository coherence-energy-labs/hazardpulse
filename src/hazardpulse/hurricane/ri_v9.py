"""v9.1 rapid-intensification forecast (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 1) -- SHADOW.

The frozen D_gbt (``results/models/hurricane_ri_v9.json``, scored in NumPy) where the cycle has the
early aids it was trained on (DSHP, IVCN and NNIC all with a 24-h forecast); otherwise NOAA's DTOPS
30/24 (SHIPS-RII where DTOPS is missing). Until the prospective test meets its claim rule this is
computed and recorded beside the published number, never published as the forecast.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hazardpulse.hurricane import ri_model
from hazardpulse.hurricane import ri_v9_features as fx
from hazardpulse.tornado import lgbm_payload as lp

MODEL_PATH = ri_model.RESULTS / "models" / "hurricane_ri_v9.json"
GATE_AIDS = ("DSHP", "IVCN", "NNIC")
PROSPECTIVE_START = "2026-10-04T00:00:00Z"   # amendment 1: the first cycle the test can score


def load(path: str | Path = MODEL_PATH) -> tuple[dict, str]:
    payload = lp.load(path)
    names = list(payload["feature_names"])
    if names != list(fx.names_for("ONH")):
        raise ValueError(f"{path}: inputs {names[:3]}... are not the v9 feature set")
    return payload, lp.model_version(payload, prefix="hurricane_ri_v9")


def predict(payload: Mapping, version: str, records: Iterable, cycle, basin: str,
            ri_pcts: Mapping[tuple[str, str], float | None]) -> dict:
    """The v9.1 probability for one cycle, with what produced it."""
    table = fx.cycle_table(records, cycle)
    f = fx.adeck_features(table, basin)
    f.update(fx.ri_features(ri_pcts))
    gate_ok = all(math.isfinite(f[f"dv24_{a}"]) for a in GATE_AIDS)
    p_model = float(lp.predict_proba(payload, fx.vector(f, payload["feature_names"])[None, :])[0])
    dtops, riod = ri_pcts.get(("DTOP", "30/24")), ri_pcts.get(("RIOD", "30/24"))
    if gate_ok:
        p, source = p_model, "v9.1"
    elif dtops is not None:
        p, source = dtops / 100.0, "DTOPS (v9.1 gate: early aids missing)"
    elif riod is not None:
        p, source = riod / 100.0, "SHIPS-RII (v9.1 gate: early aids missing)"
    else:
        p, source = None, "none: no early aids and no NOAA RI guidance"
    missing_gate = [a for a in GATE_AIDS if not math.isfinite(f[f"dv24_{a}"])]
    return {"probability": None if p is None else round(p, 4), "source": source,
            "model_probability": round(p_model, 4), "gate_ok": gate_ok, "gate_missing": missing_gate,
            "dtops_pct": dtops, "riod_pct": riod, "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "model_version": version,
            "inputs_present": int(sum(np.isfinite(fx.vector(f, payload["feature_names"]))))}
