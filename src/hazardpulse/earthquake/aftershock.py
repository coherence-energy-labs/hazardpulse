"""AS, the aftershock layer of amendment E5 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 13.2).

The chance of at least one M6+ epicentre in a forecast cell within the 30 days from issue time t, from the shallow
M6+ earthquakes of the 30 days before t:

    P_AS(c, t) = 1 - exp(-Lambda),
    Lambda = sum_k K 10^(alpha (M_k - 7)) [Omega(t - t_k + 30 d) - Omega(t - t_k)] S_k(c)

Omega is the Omori-Utsu integral with (p, c_t). S_k(c) is the fraction of an isotropic kernel centred on event k
falling in cell c; its density is proportional to (1 + r^2 / d_k^2)^-1.5, and d_k = d0 10^(0.5 (M_k - 7)). Times are
in days, distances in km.

The cell fractions are found by Monte Carlo: one fixed set of unit-kernel draws per event (common random numbers),
scaled to each width of a fixed grid and interpolated in log width, so a fit can move d0 smoothly. The cells are the
operational grid's (``operational_forecast.cell_index``).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Iterable

import numpy as np

from hazardpulse.earthquake import operational_forecast as of

EARTH_KM = 6371.0
WINDOW_D = 30.0
MIN_MAG = 6.0
MAX_DEPTH_KM = 70.0
D_GRID_KM = np.geomspace(2.0, 2000.0, 31)
N_DRAWS = 20_000
SEED = 20261011


@dataclasses.dataclass(frozen=True)
class Params:
    K: float
    alpha: float
    p: float
    c_t: float          # days
    d0: float           # km, the kernel width of an M7

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def omori_integral(tau: np.ndarray, p: float, c_t: float) -> np.ndarray:
    """int_0^tau (s + c_t)^-p ds (tau in days, >= 0)."""
    tau = np.maximum(np.asarray(tau, float), 0.0)
    if abs(p - 1.0) < 1e-9:
        return np.log1p(tau / c_t)
    return ((tau + c_t) ** (1.0 - p) - c_t ** (1.0 - p)) / (1.0 - p)


def unit_draws(seed: int = SEED, n: int = N_DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """(r / d, azimuth) for n draws from the unit kernel, density (1 + r^2)^-1.5 r dr dtheta / (2 pi) normalised:
    the radial CDF is 1 - (1 + r^2)^-1/2, so r = sqrt((1 - U)^-2 - 1)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    r = np.sqrt((1.0 - u) ** -2 - 1.0)
    return r, rng.random(n) * 2.0 * math.pi


_UNIT = unit_draws()


def kernel_fractions(lat: float, lon: float, d_grid: np.ndarray = D_GRID_KM) -> list[dict[int, float]]:
    """For each width in ``d_grid``, {cell: fraction of the kernel centred at (lat, lon) that falls in it}. Draws past
    a quarter of the Earth's circumference fall nowhere."""
    r1, az = _UNIT
    p1, l1 = math.radians(lat), math.radians(lon)
    out = []
    for d in d_grid:
        delta = np.minimum(r1 * d, 0.5 * math.pi * EARTH_KM) / EARTH_KM
        lat2 = np.arcsin(np.sin(p1) * np.cos(delta) + np.cos(p1) * np.sin(delta) * np.cos(az))
        lon2 = l1 + np.arctan2(np.sin(az) * np.sin(delta) * np.cos(p1), np.cos(delta) - np.sin(p1) * np.sin(lat2))
        keep = r1 * d < 0.5 * math.pi * EARTH_KM
        cells = of.cell_index(np.degrees(lat2[keep]), (np.degrees(lon2[keep]) + 180.0) % 360.0 - 180.0)
        u, cnt = np.unique(cells, return_counts=True)
        out.append({int(c): float(k) / len(r1) for c, k in zip(u, cnt)})
    return out


def fraction_at(table: list[dict[int, float]], cell: int, d: float, d_grid: np.ndarray = D_GRID_KM) -> float:
    """The kernel fraction in ``cell`` at width d, linear in log width between the grid's widths."""
    x = math.log(min(max(d, d_grid[0]), d_grid[-1]))
    lg = np.log(d_grid)
    j = int(np.clip(np.searchsorted(lg, x) - 1, 0, len(d_grid) - 2))
    w = (x - lg[j]) / (lg[j + 1] - lg[j])
    return (1.0 - w) * table[j].get(cell, 0.0) + w * table[j + 1].get(cell, 0.0)


@dataclasses.dataclass
class Trigger:
    t: float            # days
    lat: float
    lon: float
    mag: float
    table: list[dict[int, float]]


class AftershockLayer:
    """P_AS for any (cell, issue time), given the shallow M6+ events known at that time."""

    def __init__(self, params: Params):
        self.params = params
        self._tables: dict[tuple[float, float], list[dict[int, float]]] = {}

    def trigger(self, t: float, lat: float, lon: float, mag: float) -> Trigger:
        key = (round(lat, 4), round(lon, 4))
        if key not in self._tables:
            self._tables[key] = kernel_fractions(lat, lon)
        return Trigger(t, lat, lon, mag, self._tables[key])

    def rate(self, cell: int, t: float, triggers: Iterable[Trigger]) -> float:
        P = self.params
        lam = 0.0
        for k in triggers:
            dt_ = t - k.t
            if not 0.0 <= dt_ < WINDOW_D or k.mag < MIN_MAG:
                continue
            prod = P.K * 10.0 ** (P.alpha * (k.mag - 7.0))
            dom = float(omori_integral(dt_ + WINDOW_D, P.p, P.c_t) - omori_integral(dt_, P.p, P.c_t))
            lam += prod * dom * fraction_at(k.table, cell, P.d0 * 10.0 ** (0.5 * (k.mag - 7.0)))
        return lam

    def probability(self, cell: int, t: float, triggers: Iterable[Trigger]) -> float:
        return 1.0 - math.exp(-self.rate(cell, t, triggers))

    def increment(self, t: float, triggers: list[Trigger], base: np.ndarray, floor: float = 1e-6) -> dict[int, float]:
        """Amendment E7 (section 15): the served forecast ``base`` (one value per cell) with the aftershock rate added
        as an increment, P = 1 - (1 - base) exp(-Lambda), for every cell where Lambda exceeds ``floor``."""
        out: dict[int, float] = {}
        cells = set()
        for k in triggers:
            if 0.0 <= t - k.t < WINDOW_D and k.mag >= MIN_MAG:
                for table in k.table:
                    cells.update(table.keys())
        for c in sorted(cells):
            lam = self.rate(c, t, triggers)
            if lam > floor:
                out[c] = 1.0 - (1.0 - float(base[c])) * math.exp(-lam)
        return out

    def grid(self, t: float, triggers: list[Trigger]) -> np.ndarray:
        """P_AS for every cell of the grid at t (zero where no recent shallow M6+ reaches)."""
        out = np.zeros(of.N_CELLS)
        cells = set()
        for k in triggers:
            if 0.0 <= t - k.t < WINDOW_D and k.mag >= MIN_MAG:
                d = self.params.d0 * 10.0 ** (0.5 * (k.mag - 7.0))
                j = int(np.clip(np.searchsorted(np.log(D_GRID_KM), math.log(max(d, D_GRID_KM[0]))) , 0, len(D_GRID_KM) - 1))
                cells.update(k.table[j].keys())
                cells.update(k.table[max(0, j - 1)].keys())
        for c in cells:
            out[c] = self.probability(c, t, triggers)
        return out
