"""Storm-centred infrared structure from NOAA GMGSI longwave counts (hurricane RI amendment 5).

One code path for training and live: both hand this module the same kind of crop (8-bit GMGSI
counts on the product's own latitude/longitude axes, +-4 degrees around the storm) and get the same
features back.

GMGSI longwave is an 8-bit product ("0-255 Brightness Temperature"): a HIGHER count is a COLDER
cloud top; 255 is used here for missing. Every feature is defined in counts, so no count-to-kelvin
curve is assumed. (Under the standard McIDAS IR curve, Tb = 418 - count above count 176, the
thresholds 195 and 215 are about -50 C and -70 C -- an interpretation only, not used.)
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

EARTH_RADIUS_KM = 6371.0
MISSING = 255
COLD = 195          # "cold cloud" count threshold
VERY_COLD = 215     # "very cold cloud" count threshold
MIN_VALID = 0.5     # a region with fewer valid pixels than this fraction gives NaN

STATIC = ("ir_mean_0_50", "ir_mean_50_200", "ir_mean_200_300", "ir_std_50_200", "ir_cold_50_200",
          "ir_vcold_0_100", "ir_cold_0_300", "ir_asym_50_200", "ir_eye", "ir_max_0_50")
TREND = ("ir_mean_0_50", "ir_mean_50_200", "ir_cold_50_200", "ir_asym_50_200")
IR_NAMES = STATIC + tuple(f"d_{n}" for n in TREND)


def distances_km(lat: np.ndarray, lon: np.ndarray, centre: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance (km) and bearing (radians, clockwise from north) of every pixel of a
    (lat rows x lon columns) crop from ``centre``."""
    la = np.radians(np.asarray(lat, np.float64))[:, None]
    lo = np.radians(np.asarray(lon, np.float64))[None, :]
    la0, lo0 = math.radians(centre[0]), math.radians(centre[1])
    dlo = lo - lo0
    cosc = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlo)
    dist = EARTH_RADIUS_KM * np.arccos(np.clip(cosc, -1.0, 1.0))
    bearing = np.arctan2(np.sin(dlo) * np.cos(la), np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlo))
    return dist, np.mod(bearing, 2 * np.pi)


def _region(counts, valid, dist, lo, hi):
    ring = (dist >= lo) & (dist < hi)
    n = int(ring.sum())
    if n == 0 or (valid & ring).sum() < MIN_VALID * n:
        return None
    return counts[valid & ring].astype(np.float64)


def static_features(counts: np.ndarray, lat: np.ndarray, lon: np.ndarray, centre: tuple[float, float]) -> dict[str, float]:
    counts = np.asarray(counts)
    valid = counts != MISSING
    dist, bearing = distances_km(lat, lon, centre)
    nan = float("nan")
    out = {n: nan for n in STATIC}
    r0_50, r50_200 = _region(counts, valid, dist, 0, 50), _region(counts, valid, dist, 50, 200)
    r200_300, r0_100 = _region(counts, valid, dist, 200, 300), _region(counts, valid, dist, 0, 100)
    r0_300 = _region(counts, valid, dist, 0, 300)
    if r0_50 is not None:
        out["ir_mean_0_50"], out["ir_max_0_50"] = float(r0_50.mean()), float(r0_50.max())
    if r50_200 is not None:
        out["ir_mean_50_200"], out["ir_std_50_200"] = float(r50_200.mean()), float(r50_200.std())
        out["ir_cold_50_200"] = float(np.mean(r50_200 >= COLD))
        octant = np.floor(bearing / (np.pi / 4)).astype(int) % 8
        ring = (dist >= 50) & (dist < 200) & valid
        means = [counts[ring & (octant == k)].astype(np.float64).mean() for k in range(8) if (ring & (octant == k)).any()]
        if len(means) == 8:
            out["ir_asym_50_200"] = float(np.std(means))
    if r200_300 is not None:
        out["ir_mean_200_300"] = float(r200_300.mean())
    if r0_100 is not None:
        out["ir_vcold_0_100"] = float(np.mean(r0_100 >= VERY_COLD))
    if r0_300 is not None:
        out["ir_cold_0_300"] = float(np.mean(r0_300 >= COLD))
    core, wall = _region(counts, valid, dist, 0, 25), _region(counts, valid, dist, 25, 75)
    if core is not None and wall is not None:
        out["ir_eye"] = float(wall.mean() - core.min())       # a warm eye inside a cold wall is large
    return out


def features(now: Mapping | None, before: Mapping | None) -> dict[str, float]:
    """All IR features from the image at t + 2 h (``now``) and t - 4 h (``before``); each is a
    mapping with ``counts``, ``lat``, ``lon``, ``centre``. A missing image gives NaNs."""
    nan = float("nan")
    s_now = static_features(now["counts"], now["lat"], now["lon"], tuple(now["centre"])) if now is not None \
        else {n: nan for n in STATIC}
    s_bef = static_features(before["counts"], before["lat"], before["lon"], tuple(before["centre"])) \
        if before is not None else {n: nan for n in STATIC}
    out = dict(s_now)
    for n in TREND:
        out[f"d_{n}"] = s_now[n] - s_bef[n]
    return out
