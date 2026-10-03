"""Features of the v9 rapid-intensification model (docs/HURRICANE_RI_V9_PROGRAM.md).

ONE builder for every path -- training (archived a- and e-decks), the 2026 final and live serving
(``atcf/aid_public`` a-decks + the SHIPS text) -- so a feature means the same thing everywhere.

Groups (the protocol's names):

* O (ours), from the a-deck at cycle t: CARQ intensity and its 12/24-h tendency, MSLP, |latitude|,
  an Atlantic indicator, and the 24-h intensity CHANGE forecast by the EARLY aids (NOAA's
  statistical models, NHC's consensus aids incl. the neural-network NNIC, the GEFS mean, the
  regional hurricane models and the global models as group means), with their spread and the
  fraction forecasting >= 30 kt;
* N (NOAA RI aids): logits of SHIPS-RII, Logistic, Bayesian, Consensus and DTOPS at six thresholds;
* H (human): the NHC official forecast's 12- and 24-h change.

A missing input is NaN, never a value that looks like data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

THRESHOLDS = ("20/12", "25/24", "30/24", "35/24", "40/24", "45/36")
RI_TECHS = ("RIOD", "RIOL", "RIOB", "RIOC", "DTOP")
INDIVIDUAL = ("DSHP", "LGEM", "IVCN", "HCCA", "NNIC", "AEMI")
REGIONAL = ("HWFI", "HMNI", "HFAI", "HFBI", "CTCI")
GLOBAL = ("AVNI", "EMXI", "EGRI", "NVGI")
SPREAD_AIDS = INDIVIDUAL + REGIONAL + GLOBAL
PCT_CLIP = 0.005
RI_KT = 30.0

O_NAMES = (
    "v0", "dv_past12", "dv_past24", "mslp0", "abs_lat", "is_atlantic",
    *(f"dv24_{a}" for a in INDIVIDUAL),
    "dv24_regional_mean", "dv24_regional_max", "dv24_global_mean", "dv24_spread", "frac_ge30",
)
N_NAMES = tuple(f"ri_{tech}_{th.replace('/', '_')}" for tech in RI_TECHS for th in THRESHOLDS)
H_NAMES = ("ofcl_dv24", "ofcl_dv12")
# v10 (amendment 2): how the guidance has been doing on THIS storm -- observed minus forecast
ERROR_AIDS = ("DSHP", "LGEM", "IVCN", "HCCA", "NNIC", "OFCL")
E_NAMES = (*(f"err12_{a}" for a in ERROR_AIDS), *(f"err24_{a}" for a in ERROR_AIDS),
           "err12_mean", "err24_mean", "dv_past6")
GROUPS = {"O": O_NAMES, "N": N_NAMES, "H": H_NAMES, "E": E_NAMES}
# what a feed without the early guidance loses (the masking copies of amendment 2)
GUIDANCE_NAMES = tuple(n for n in O_NAMES if n.startswith("dv24_") or n == "frac_ge30") + E_NAMES


def names_for(groups: str) -> tuple[str, ...]:
    """``"ON"`` -> O_NAMES + N_NAMES, in the fixed order (O, N, H, E)."""
    out: list[str] = []
    for g in "ONHE":
        if g in groups:
            out.extend(GROUPS[g])
    return tuple(out)


def error_features(records: Iterable, cycle, v0: float | None) -> dict[str, float]:
    """Group E: each aid's error on THIS storm -- V_CARQ(t) minus the aid's forecast for t made
    12 h and 24 h earlier -- their means, and the CARQ 6-h tendency. NaN where absent."""
    import datetime as _dt
    f: dict[str, float] = {k: float("nan") for k in E_NAMES}
    if v0 is None or not math.isfinite(v0):
        return f
    t0 = cycle_table(records, cycle)
    past6 = t0.get(("CARQ", -6))
    if past6 is not None and past6.vmax is not None:
        f["dv_past6"] = float(v0) - float(past6.vmax)
    for lead in (12, 24):
        prev = cycle_table(records, cycle - _dt.timedelta(hours=lead))
        errs = []
        for a in ERROR_AIDS:
            fx_ = prev.get((a, lead))
            if fx_ is not None and fx_.vmax is not None:
                e = float(v0) - float(fx_.vmax)
                f[f"err{lead}_{a}"] = e
                errs.append(e)
        if errs:
            f[f"err{lead}_mean"] = float(np.mean(errs))
    return f


def threshold_key(th: str) -> tuple[int, int]:
    """``"30/24"`` -> (dV kt, window h)."""
    kt, hr = th.split("/")
    return int(kt), int(hr)


@dataclass(frozen=True)
class Fix:
    vmax: float | None
    mslp: float | None
    lat: float | None
    lon: float | None


def cycle_table(records: Iterable, cycle) -> dict[tuple[str, int], Fix]:
    """``{(tech, tau): Fix}`` for one cycle of one storm's a-deck (``atcf.ATCFRecord`` items).
    A-decks repeat a (tech, tau) once per wind-radii row; the first is kept."""
    out: dict[tuple[str, int], Fix] = {}
    for r in records:
        if r.cycle != cycle:
            continue
        key = (r.model, int(r.tau_hours))
        if key not in out:
            out[key] = Fix(r.vmax_kt, r.mslp_hpa, r.lat, r.lon)
    return out


def _nan(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def adeck_features(table: Mapping[tuple[str, int], Fix], basin: str) -> dict[str, float]:
    """Groups O and H from one cycle's a-deck table."""
    f: dict[str, float] = {k: float("nan") for k in O_NAMES + H_NAMES}
    f["is_atlantic"] = 1.0 if basin.upper() == "AL" else 0.0
    carq0 = table.get(("CARQ", 0))
    v0 = carq0.vmax if carq0 else None
    if carq0 is not None:
        f["mslp0"] = _nan(carq0.mslp)
        f["abs_lat"] = abs(carq0.lat) if carq0.lat is not None else float("nan")
    if v0 is None:
        return f                                        # no analysis intensity: no change is defined
    f["v0"] = float(v0)
    for tau, name in ((-12, "dv_past12"), (-24, "dv_past24")):
        past = table.get(("CARQ", tau))
        if past is not None and past.vmax is not None:
            f[name] = float(v0) - float(past.vmax)

    def dv(tech: str, tau: int = 24) -> float:
        fx = table.get((tech, tau))
        return float(fx.vmax) - float(v0) if fx is not None and fx.vmax is not None else float("nan")

    for a in INDIVIDUAL:
        f[f"dv24_{a}"] = dv(a)
    reg = np.array([dv(a) for a in REGIONAL])
    glo = np.array([dv(a) for a in GLOBAL])
    if np.isfinite(reg).any():
        f["dv24_regional_mean"] = float(np.nanmean(reg))
        f["dv24_regional_max"] = float(np.nanmax(reg))
    if np.isfinite(glo).any():
        f["dv24_global_mean"] = float(np.nanmean(glo))
    allv = np.array([dv(a) for a in SPREAD_AIDS])
    allv = allv[np.isfinite(allv)]
    if len(allv) >= 3:
        f["dv24_spread"] = float(np.std(allv))
    if len(allv):
        f["frac_ge30"] = float(np.mean(allv >= RI_KT))
    f["ofcl_dv24"] = dv("OFCL", 24)
    f["ofcl_dv12"] = dv("OFCL", 12)
    return f


def aid_logit(pct: float | None) -> float:
    if pct is None or not math.isfinite(float(pct)):
        return float("nan")
    q = min(max(float(pct) / 100.0, PCT_CLIP), 1.0 - PCT_CLIP)
    return math.log(q / (1.0 - q))


def ri_features(pcts: Mapping[tuple[str, str], float | None]) -> dict[str, float]:
    """Group N from ``{(tech, threshold): whole percent}``."""
    return {f"ri_{tech}_{th.replace('/', '_')}": aid_logit(pcts.get((tech, th)))
            for tech in RI_TECHS for th in THRESHOLDS}


def vector(features: Mapping[str, float], names: Iterable[str]) -> np.ndarray:
    return np.asarray([float(features.get(n, float("nan"))) for n in names], dtype=np.float64)


def record_inputs(features: Mapping[str, float], names: Iterable[str]) -> dict[str, float | None]:
    """The exact input vector a forecast used, for its record: full float precision (a rounded
    input can cross a tree split and change the output), None for a missing input. ``vector`` of
    this mapping gives back the model's input row bit for bit, so the record alone recomputes
    the forecast."""
    return {n: (v if math.isfinite(v) else None) for n, v in zip(names, vector(features, names).tolist())}
