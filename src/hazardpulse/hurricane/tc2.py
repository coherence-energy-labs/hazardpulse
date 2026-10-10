"""TC2: TC1's intensity forecast held to the median our RI model's 24-h curve implies (docs/
HURRICANE_TRACK_INTENSITY_PROGRAM.md, amendment 2).

NHC scores intensity by mean absolute error, which the median of the predictive distribution minimizes. The RI
model's curve P(V(t+24 h) - V(t) >= k), k = 15..45 kt, brackets that median with nothing fitted: P(>= k) >= 0.5
puts the median at or above k, P(>= k) < 0.5 below it. TC2 moves TC1's 24-h change to the nearest edge of the
bracket when it lies outside, ramps the move in over the first day, and lets it decay beyond 24 h as TC1 decays
the storm toward Kaplan and DeMaria's (1995) inland background of 26.7 kt. One module for the backtest and any
live use.
"""
from __future__ import annotations

import math
from typing import Mapping

THRESHOLDS_KT = (15, 20, 25, 30, 35, 40, 45)
BACKGROUND_KT = 26.7          # Kaplan and DeMaria (1995): the inland decay model's background intensity
HALF = 0.5


def monotone(curve: Mapping[int, float]) -> dict[int, float]:
    """The exceedance curve made non-increasing in k (a running minimum), on the thresholds it has."""
    out, run = {}, math.inf
    for k in THRESHOLDS_KT:
        p = curve.get(k, curve.get(str(k)))
        if p is None or not math.isfinite(float(p)):
            continue
        run = min(run, float(p))
        out[k] = run
    return out


def bracket(curve: Mapping[int, float]) -> tuple[float, float]:
    """``(L, U)``: the median of the 24-h change is at least L and below U. L is the largest threshold with
    P(>= k) >= 0.5 (-inf if none), U the smallest with P(>= k) < 0.5 (+inf if none)."""
    c = monotone(curve)
    lo = max((k for k, p in c.items() if p >= HALF), default=-math.inf)
    hi = min((k for k, p in c.items() if p < HALF), default=math.inf)
    return float(lo), float(hi)


def shift_24h(dv_tc1: float, lo: float, hi: float) -> float:
    """How far TC1's 24-h change must move to lie in [L, U]: 0 when it already does."""
    return min(max(dv_tc1, lo), hi) - dv_tc1


TC2B_TAPER_END_H = 72.0      # amendment 3: the shift is gone by 72 h (chosen from amendment 2's DEV per-lead pattern)


def lead_factor(lead: int, v24: float, v_lead: float, taper_end: float | None = None) -> float:
    """The share of the 24-h shift applied at ``lead``: linear to 24 h; beyond, the fraction of the 24-h excess over
    the background TC1 still holds at that lead (capped at 1, floored at 0) -- and, with ``taper_end`` (TC2b),
    times a linear taper from 1 at 24 h to 0 at ``taper_end``."""
    if lead <= 24:
        return lead / 24.0
    den = v24 - BACKGROUND_KT
    if den <= 0:
        return 0.0
    f = min(1.0, max(0.0, (v_lead - BACKGROUND_KT) / den))
    if taper_end is not None:
        f *= max(0.0, (taper_end - lead) / (taper_end - 24.0))
    return f


def project(tc1: Mapping[int, float], v0: float | None, curve: Mapping[int, float] | None,
            taper_end: float | None = None) -> tuple[dict[int, float], dict[str, object]]:
    """``(TC2 intensity by lead, what was done)``. TC2 is TC1 wherever there is no curve, no analysis intensity or
    no 24-h TC1 forecast -- and wherever TC1's 24-h change already lies inside the bracket."""
    out = {int(k): float(v) for k, v in tc1.items()}
    v24 = out.get(24)
    if curve is None or v0 is None or v24 is None or not math.isfinite(v0) or not math.isfinite(v24):
        return out, {"applied": False, "why": "no curve" if curve is None else "no analysis or 24-h forecast"}
    lo, hi = bracket(curve)
    dv = v24 - v0
    s = shift_24h(dv, lo, hi)
    if s == 0.0:
        return out, {"applied": False, "why": "TC1 inside the bracket", "bracket": [lo, hi], "dv_tc1": dv}
    for lead, v in out.items():
        out[lead] = v + s * lead_factor(lead, v24, v, taper_end)
    return out, {"applied": True, "shift_24h": s, "bracket": [lo, hi], "dv_tc1": dv}
