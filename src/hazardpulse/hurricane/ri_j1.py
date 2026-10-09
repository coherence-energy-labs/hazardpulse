"""J1: the JTWC-basin RI model with satellite infrared (hurricane RI amendments 13 and 13b).

P(RI: V(t + 24 h) - V(t) >= 30 kt) for West Pacific, North Indian and Southern Hemisphere storms. It is LightGBM
(five seeds, averaged) on v8.2's own score (as a logit), v8.2's 17 inputs, amendment 5's 14 GMGSI IR features at
t + 2 h and t - 4 h, and basin indicators. It was carried against the published v8.2 on the 2024-2025 JTWC
cycles (amendment 13 outcome), and it runs live in SHADOW beside v8.2: recorded, never the published number,
until the rule registered in amendment 13b says otherwise.

One module for the export, the live scorer and the record audit, so the inputs a shadow records are exactly the
row it was scored from, and an audit can recompute it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from hazardpulse.hurricane import ir_features
from hazardpulse.tornado import lgbm_payload as lp

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "results" / "models" / "hurricane_ri_j1.json"
SCHEMA = "hazardpulse.hurricane_ri_j1.v1"
LABEL = "J1"
SHADOW_KEY = "ri_j1_shadow"
FIX_MODELS = ("BEST", "JTWC")          # where a JTWC storm's IR centre comes from: its best track, then the warning
LIVE_JTWC_BASINS = ("WP", "IO", "SH")  # the live ATCF basin codes J1 scores
BASIN_FLAGS = ("is_wp", "is_ni", "is_sh", "is_nhc")


def basin_flags(basin: str) -> dict[str, float]:
    """The basin indicators, from either the training (IBTrACS: WP NI SI SP NA EP) or the live (ATCF: WP IO SH
    AL EP CP) basin code."""
    b = str(basin).upper()
    return {"is_wp": float(b == "WP"), "is_ni": float(b in ("NI", "IO")), "is_sh": float(b in ("SI", "SP", "SH")),
            "is_nhc": float(b in ("NA", "AL", "EP", "CP"))}


def logit(p: float) -> float:
    q = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def canonical_bytes(art: Mapping) -> bytes:
    return json.dumps(art, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def version_of(art: Mapping) -> str:
    return f"hurricane_ri_j1-{hashlib.sha256(canonical_bytes(art)).hexdigest()[:12]}"


def load(path: str | Path = MODEL_PATH) -> tuple[dict, str]:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    if art.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {art.get('schema')!r}, expected {SCHEMA!r}")
    if not art.get("members") or any(m["feature_names"] != art["feature_names"] for m in art["members"]):
        raise ValueError(f"{path}: members do not read the artifact's feature names")
    return art, version_of(art)


def predict_matrix(art: Mapping, X: np.ndarray) -> np.ndarray:
    """The seeds' probabilities, averaged (as the program's learner averages them)."""
    return np.mean([lp.predict_proba(m, X) for m in art["members"]], axis=0)


def row(art: Mapping, inputs: Mapping[str, float | None]) -> np.ndarray:
    return np.array([[np.nan if inputs.get(n) is None else float(inputs[n]) for n in art["feature_names"]]], float)


def recompute(art: Mapping, inputs: Mapping[str, float | None]) -> float:
    return float(predict_matrix(art, row(art, inputs))[0])


def live_inputs(art: Mapping, case: Mapping, v82_ensemble: float, ir: Mapping[str, float]) -> dict[str, float | None]:
    """The row J1 scores, by name: v8.2's ensemble as a logit, v8.2's inputs from the live case (the same keys the
    v8.2 scorer reads), the IR features and the basin flags. A missing value is None, never imputed (the trees'
    own missing-value branches handle it, as in training)."""
    src: dict[str, float | None] = {"v82_logit": logit(v82_ensemble), **basin_flags(str(case.get("basin", "")))}
    for n in ir_features.IR_NAMES:
        v = ir.get(n)
        src[n] = float(v) if v is not None and math.isfinite(float(v)) else None
    out = {}
    for n in art["feature_names"]:
        if n in src:
            out[n] = src[n]
        else:
            v = case.get(n)
            out[n] = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
    return out


def ir_centres(positions: Mapping | None, cycle: dt.datetime) -> dict[str, tuple[dt.datetime, tuple[float, float]]] | None:
    """``{tag: (image hour, centre)}`` from a case's recorded fix positions, by the same extrapolation as training
    (``ir_source.extrapolate``); None without a position at t."""
    from hazardpulse.hurricane import ir_source
    if not positions or positions.get("t") is None:
        return None
    p0 = tuple(float(x) for x in positions["t"])
    pm6 = positions.get("t_minus_6h")
    pm6 = tuple(float(x) for x in pm6) if pm6 is not None else None
    return {tag: (cycle + dt.timedelta(hours=h), ir_source.extrapolate(p0, pm6, h))
            for tag, h in ir_source.OFFSETS.items()}
