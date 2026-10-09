"""A storm's coherence state (hurricane RI program amendment 10): how organized it has been, not only how it looks now.

The coherence framework's concept, fitted to a forecaster's needs. Organization is CREATED by a source, DECAYS
unless it is fed, and PERSISTS only while the source keeps paying for it; it is strongest when its parts are
aligned. The models so far see a storm at one cycle and its 6-hour change. The state below carries each
organizing signal's memory over the storm's own past cycles, with exponential forgetting -- the time-domain
operator ``dS/dt = s(t) - S/tau``, discretized over cycles and normalized so it reads in the signal's units.

Sources (all inputs the models already compute at every cycle, read at the storm's earlier cycles):
- ``conv``: deep convection over the core, the fraction of very cold cloud within 100 km (``ir_vcold_0_100``);
- ``sym``: symmetry of the 50-200 km ring, minus the octant asymmetry (``-ir_asym_50_200``) -- the alignment of
  the convection around the vortex;
- ``hold``: how much of that heating the vortex holds (``h8_core_retention``, the balanced response);
- ``spin``: the intensity tendency, kt per 6 h, from the storm's own intensity at its cycles.

Every constant is declared here and fixed before any coherence feature met an outcome.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, Mapping

TAU_SHORT_H = 12.0          # a half-day memory
TAU_LONG_H = 36.0           # a day-and-a-half memory: the RI forecast's own horizon plus its lead-in
WINDOW_H = 36.0             # cycles older than this are not in the state at all
DEEP = 0.5                  # "deep convection dominates the core": at least half the 0-100 km pixels very cold
COH_NAMES = ("coh_conv_12", "coh_conv_36", "coh_sym_36", "coh_hold_36", "coh_conv_rise", "coh_persist",
             "coh_spin_36", "coh_n")


def signals(f: Mapping[str, float]) -> dict[str, float]:
    """One cycle's organizing signals from its feature row (NaN where the row has none)."""
    def g(k):
        try:
            v = float(f.get(k, math.nan))
        except (TypeError, ValueError):
            return math.nan
        return v
    asym = g("ir_asym_50_200")
    return {"conv": g("ir_vcold_0_100"), "sym": -asym if math.isfinite(asym) else math.nan,
            "hold": g("h8_core_retention"), "v": g("v0")}


def _memory(points: list[tuple[float, float]], tau_h: float) -> float:
    """Exponentially forgotten mean of (age in hours, value) pairs over the finite values; NaN with none."""
    num = den = 0.0
    for age, x in points:
        if math.isfinite(x):
            w = math.exp(-age / tau_h)
            num += w * x
            den += w
    return num / den if den > 0 else math.nan


def state(history: Iterable[tuple[dt.datetime, Mapping[str, float]]], t: dt.datetime) -> dict[str, float]:
    """The coherence state at cycle ``t`` from the storm's cycles ``(time, signals)`` at or before ``t``, within
    ``WINDOW_H``. Signals are ``signals()`` dicts. A cycle after ``t`` is never read."""
    pts = sorted((u, s) for u, s in history if u <= t and (t - u).total_seconds() <= WINDOW_H * 3600.0)
    out = {k: math.nan for k in COH_NAMES}
    out["coh_n"] = float(len(pts))
    if not pts:
        return out
    age = [((t - u).total_seconds() / 3600.0, s) for u, s in pts]
    conv = [(a, s["conv"]) for a, s in age]
    out["coh_conv_12"] = _memory(conv, TAU_SHORT_H)
    out["coh_conv_36"] = _memory(conv, TAU_LONG_H)
    out["coh_sym_36"] = _memory([(a, s["sym"]) for a, s in age], TAU_LONG_H)
    out["coh_hold_36"] = _memory([(a, s["hold"]) for a, s in age], TAU_LONG_H)
    now = next((s["conv"] for a, s in age if a == 0.0), math.nan)
    if math.isfinite(now) and math.isfinite(out["coh_conv_36"]):
        out["coh_conv_rise"] = now - out["coh_conv_36"]              # the source above what the storm has held
    fin = [x for _, x in conv if math.isfinite(x)]
    if fin:
        out["coh_persist"] = sum(1.0 for x in fin if x >= DEEP) / len(fin)
    # intensity tendency between consecutive cycles, per 6 h, aged at the later cycle
    spin = []
    for (u0, s0), (u1, s1) in zip(pts, pts[1:]):
        gap = (u1 - u0).total_seconds() / 3600.0
        if gap > 0 and math.isfinite(s0["v"]) and math.isfinite(s1["v"]):
            spin.append(((t - u1).total_seconds() / 3600.0, (s1["v"] - s0["v"]) * 6.0 / gap))
    out["coh_spin_36"] = _memory(spin, TAU_LONG_H)
    return out


def add_states(rows: list[dict]) -> int:
    """Write the coherence state into each row's features (``rows``: the case table, ``atcf_id``, ``dtg``,
    ``f``), from the same storm's rows at or before each cycle. Returns how many rows have a convection memory."""
    by: dict[str, list[tuple[dt.datetime, dict]]] = {}
    for r in rows:
        by.setdefault(r["atcf_id"], []).append((dt.datetime.strptime(r["dtg"], "%Y%m%d%H"), signals(r["f"])))
    n = 0
    for r in rows:
        t = dt.datetime.strptime(r["dtg"], "%Y%m%d%H")
        st = state(by[r["atcf_id"]], t)
        r["f"].update(st)
        n += math.isfinite(st["coh_conv_36"])
    return n
