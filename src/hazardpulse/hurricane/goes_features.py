"""Program G1's features: the inner core from GOES 2 km band-13 polar images, every hour from t-12 h to t+2 h.

The physics of rapid intensification as it looks from above: before RI the inner core organizes. Deep convection
wraps into a ring, the ring closes and becomes symmetric, an eye clears and the eyewall contracts, and the deep
convection persists rather than pulsing. At 2 km and hourly cadence these are measurable. The coherence concept
enters where it has the resolution to work: how long the core convection has been sustained, and how fast the
storm is symmetrizing. Every constant is fixed here before any feature meets an outcome (program amendment 12).

Per hour, from a storm-centred polar image (``goes_abi``):
- ``core_cold``: the fraction of samples within 50 km at or below DEEP_K (deep convection);
- ``ring_cold``: the same fraction at 50-150 km;
- ``eye_contrast``: the mean within 5 km minus the coldest azimuthal mean within 100 km;
- ``eye_radius``: when an eye was found, the smallest radius where the azimuthal mean falls halfway from the eye
  to that coldest ring -- the eyewall's inner edge, measured from the image (NaN without an eye);
- ``sym``: 1 - (wavenumber-1 amplitude / mean) of the 20-100 km cloud-top depression (300 K - BT) by azimuth;
- ``cold_min``: the coldest cloud top within 100 km.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from hazardpulse.hurricane import goes_abi as g

DEEP_K = 208.0             # about -65 C: deep convection
REF_K = 300.0              # the depression reference
MIN_VALID = 0.5            # a region with fewer valid samples than this is NaN
SUSTAIN = 0.5              # "deep convection dominates the core": at least half the 0-50 km samples
MIN_TREND_POINTS = 4
G1_NAMES = ("g_core_cold", "g_ring_cold", "g_eye_contrast", "g_eye_radius", "g_sym", "g_cold_min",
            "g_core_run", "g_sym_trend", "g_eye_trend", "g_core_trend", "g_eye_frac", "g_n")


def _frac_cold(bt: np.ndarray) -> float:
    fin = np.isfinite(bt)
    if fin.mean() < MIN_VALID:
        return math.nan
    return float(np.mean(bt[fin] <= DEEP_K))


def hour_stats(img: np.ndarray, eye: bool) -> dict[str, float]:
    """One hour's inner-core quantities from a uint8 polar image."""
    bt = g.decode(img)
    r = g.RADII_KM
    out = {"core_cold": _frac_cold(bt[r <= 50.0]), "ring_cold": _frac_cold(bt[(r > 50.0) & (r <= 150.0)]),
           "eye_contrast": math.nan, "eye_radius": math.nan, "sym": math.nan, "cold_min": math.nan, "eye": bool(eye)}
    inner = bt[r <= 100.0]
    if np.isfinite(inner).mean() < MIN_VALID:
        return out
    with np.errstate(all="ignore"):
        am = np.nanmean(inner, axis=1)                                       # azimuthal mean by radius
    if not np.isfinite(am).any():
        return out
    ring_min = float(np.nanmin(am))
    centre = bt[r <= 5.0]
    eye_t = float(np.nanmean(centre)) if np.isfinite(centre).any() else math.nan
    out["eye_contrast"] = eye_t - ring_min
    out["cold_min"] = float(np.nanmin(inner))
    if eye and math.isfinite(eye_t):
        half = (eye_t + ring_min) / 2.0
        below = np.nonzero(am <= half)[0]
        if below.size:
            out["eye_radius"] = float(r[below[0]])
    band = bt[(r >= 20.0) & (r <= 100.0)]
    if np.isfinite(band).mean() >= MIN_VALID:
        with np.errstate(all="ignore"):
            dep = np.nanmean(REF_K - band, axis=0)                           # by azimuth
        if np.isfinite(dep).all() and dep.mean() > 0:
            a1 = 2.0 * abs(np.fft.rfft(dep)[1]) / dep.size
            out["sym"] = float(1.0 - a1 / dep.mean())
    return out


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
    if len(pts) < MIN_TREND_POINTS:
        return math.nan
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if np.ptp(x) == 0:
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def features(hours: Sequence[tuple[int, dict | None]]) -> dict[str, float]:
    """G1's features for one cycle from ``(hour offset from t, hour_stats or None)`` for offsets -12 .. +2.
    The latest available hour gives the static features; the window gives the dynamics."""
    avail = sorted((k, s) for k, s in hours if s is not None)
    out = {n: math.nan for n in G1_NAMES}
    out["g_n"] = float(len(avail))
    if not avail:
        return out
    k_last, last = avail[-1]
    for n in ("core_cold", "ring_cold", "eye_contrast", "eye_radius", "sym", "cold_min"):
        out[f"g_{n}"] = last[n]
    run = 0
    for k, s in reversed(avail):
        if math.isfinite(s["core_cold"]) and s["core_cold"] >= SUSTAIN:
            run += 1
        else:
            break
    out["g_core_run"] = float(run)
    ks = [k for k, _ in avail]
    out["g_sym_trend"] = _slope(ks, [s["sym"] for _, s in avail])
    out["g_eye_trend"] = _slope(ks, [s["eye_contrast"] for _, s in avail])
    out["g_core_trend"] = _slope(ks, [s["core_cold"] for _, s in avail])
    out["g_eye_frac"] = float(np.mean([s["eye"] for _, s in avail]))
    return out
