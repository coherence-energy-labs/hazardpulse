"""Our own track and intensity forecast: a causal, online-weighted consensus of the early aids (program TC1,
docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md).

What it is an instance of: prediction with expert advice whose experts change. The aid suite is not stationary --
HWRF and HMON gave way to HAFS in 2023, Google DeepMind's GDMI arrived in 2025, consensus aids are renamed -- so a
combination fitted once on past seasons (HCCA's design) lags every change by a season or more. Here the weights
are re-estimated at every cycle from what has *already verified*: each aid's errors at each lead, measured against
the operational analysis (CARQ) of the verifying time, in the basin and in this storm, with exponential
forgetting. A new aid enters on its first verified forecast, shrunk toward the pool until its record speaks.

The combination is Bates and Granger's minimum-variance weights ``w = S^-1 1 / 1' S^-1 1`` on the error covariance
of the members present, shrunk toward its diagonal (Ledoit-Wolf style) and constrained to be non-negative. Track
errors are 2-D (east, north km), so a covariance entry is the mean dot product of two aids' error vectors.
Optionally each aid's recent mean error (its bias) is removed first.

Causality is the contract: a forecast issued at cycle t uses only forecasts initialised at t0 <= t - tau that
verified at t0 + tau <= t, against the CARQ position and intensity at that time -- the analysis that existed then.
The final best track never enters the weights; it only scores.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from hazardpulse.hurricane.atcf import _parse_lat, _parse_lon

LEADS = (12, 24, 36, 48, 60, 72, 96, 120)
# early (interpolated) aids that are public in real time; ECMWF-derived aids (EMXI, EEMN, FSSE, ...) are absent from
# the public 2026 decks, so a forecast that used them could not run live
TRACK_MEMBERS = ("AVNI", "AEMI", "HFAI", "HFBI", "HWFI", "HMNI", "CTCI", "UKXI", "CMCI", "CEMI", "NVGI", "GDMI",
                 "TVCN", "HCCA")
INTENSITY_MEMBERS = ("DSHP", "LGEM", "NNIC", "NNIB", "HFAI", "HFBI", "HWFI", "HMNI", "CTCI", "GDMI", "IVCN", "HCCA")
OFFICIAL = "OFCL"
MIN_MEMBERS = 2                  # TVCN's own rule: a consensus needs at least two members
EARTH_KM = 6371.0
KM_PER_NM = 1.852


@dataclass(frozen=True)
class Config:
    """One member of the declared candidate grid. ``half_life_days=None`` is the equal-weight reference."""
    half_life_days: float | None = 60.0
    shrink: float = 0.5          # 1.0 = diagonal (inverse-MSE weights); 0.0 = full covariance
    debias: bool = False         # remove each aid's recent mean error before combining
    storm_boost: float = 0.0     # extra weight on this storm's own verifications (0 = none)
    prior_n: float = 5.0         # pseudo-verifications pulling a new aid toward the pool
    include_official: bool = False

    def members(self, kind: str) -> tuple[str, ...]:
        base = TRACK_MEMBERS if kind == "track" else INTENSITY_MEMBERS
        return base + ((OFFICIAL,) if self.include_official else ())


# ---------------------------------------------------------------------------------------------------------------
# reading a deck
# ---------------------------------------------------------------------------------------------------------------

_NAN3 = (math.nan, math.nan, math.nan)


@dataclass
class StormDeck:
    """One storm's early forecasts and CARQ analyses as arrays (a dict per forecast would cost ~250 MB for a
    six-season backtest): ``fc[cycle row, tech, lead] = (lat, lon, vmax)`` and ``carq[cycle row]``, NaN where
    absent or blank."""
    storm: str
    basin: str
    techs: tuple[str, ...]
    cycles: list[dt.datetime]
    fc: np.ndarray                   # (n_cycles, n_techs, len(LEADS), 3)
    carq: np.ndarray                 # (n_cycles, 3)
    _row: dict = field(default_factory=dict, repr=False)
    _tech: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._row = {c: i for i, c in enumerate(self.cycles)}
        self._tech = {t: i for i, t in enumerate(self.techs)}

    @classmethod
    def from_dicts(cls, storm: str, basin: str, fc: Mapping, carq: Mapping) -> "StormDeck":
        """From ``fc[(cycle, tech)][lead] = (lat, lon, vmax)`` and ``carq[cycle] = (lat, lon, vmax)``; None is
        absent."""
        techs = tuple(sorted({t for _, t in fc}))
        cycles = sorted({c for c, _ in fc} | set(carq))
        row, col = {c: i for i, c in enumerate(cycles)}, {t: i for i, t in enumerate(techs)}
        F = np.full((len(cycles), len(techs), len(LEADS), 3), np.nan)
        C = np.full((len(cycles), 3), np.nan)
        lead_ix = {ld: i for i, ld in enumerate(LEADS)}
        for (c, t), by_lead in fc.items():
            for ld, val in by_lead.items():
                if ld in lead_ix:
                    F[row[c], col[t], lead_ix[ld]] = [math.nan if x is None else x for x in val]
        for c, val in carq.items():
            C[row[c]] = [math.nan if x is None else x for x in val]
        return cls(storm, basin, techs, cycles, F, C)

    def get(self, t: dt.datetime, tech: str, lead: int) -> tuple[float, float, float] | None:
        i, j = self._row.get(t), self._tech.get(tech)
        if i is None or j is None:
            return None
        v = self.fc[i, j, _LEAD_IX[lead]]
        return None if np.isnan(v).all() else (float(v[0]), float(v[1]), float(v[2]))

    def analysis(self, t: dt.datetime) -> tuple[float, float, float] | None:
        i = self._row.get(t)
        if i is None or np.isnan(self.carq[i]).all():
            return None
        v = self.carq[i]
        return float(v[0]), float(v[1]), float(v[2])

    @property
    def analysis_times(self) -> list[dt.datetime]:
        return [c for c, v in zip(self.cycles, self.carq) if not np.isnan(v).all()]


_LEAD_IX = {ld: i for i, ld in enumerate(LEADS)}


def parse_deck(storm: str, text: str, techs: Iterable[str]) -> StormDeck:
    """The wanted techs' forecasts at the program's leads, and CARQ at tau 0. The first line of each
    (cycle, tech, tau) is kept -- the wind-radii repeats carry the same position and intensity."""
    want = frozenset(techs)
    fc: dict = {}
    carq: dict = {}
    for line in text.splitlines():
        f = line.split(",", 9)
        if len(f) < 9:
            continue
        tech = f[4].strip()
        if tech != "CARQ" and tech not in want:
            continue
        try:
            tau = int(f[5])
            cyc = dt.datetime.strptime(f[2].strip(), "%Y%m%d%H")
        except ValueError:
            continue
        lat, lon = _parse_lat(f[6]), _parse_lon(f[7])
        try:
            v = float(f[8])
        except ValueError:
            v = None
        v = v if v is not None and v > 0 else None
        if tech == "CARQ":
            if tau == 0:
                carq.setdefault(cyc, (lat, lon, v))
            continue
        if tau in _LEAD_IX:
            fc.setdefault((cyc, tech), {}).setdefault(tau, (lat, lon, v))
    return StormDeck.from_dicts(storm, storm[:2].upper(), fc, carq)


# ---------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------

def _wrap(dlon: float) -> float:
    return (dlon + 180.0) % 360.0 - 180.0


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(_wrap(lon2 - lon1)) / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def error_vector_km(f_lat: float, f_lon: float, t_lat: float, t_lon: float) -> np.ndarray:
    """Forecast minus truth, (east, north) km, on the local tangent plane at the truth."""
    k = math.pi * EARTH_KM / 180.0
    return np.array([_wrap(f_lon - t_lon) * k * math.cos(math.radians(t_lat)), (f_lat - t_lat) * k])


def shift_km(lat: float, lon: float, east_km: float, north_km: float) -> tuple[float, float]:
    k = math.pi * EARTH_KM / 180.0
    nl = lat + north_km / k
    return nl, _wrap(lon + east_km / (k * max(math.cos(math.radians(lat)), 1e-6)))


# ---------------------------------------------------------------------------------------------------------------
# the online state
# ---------------------------------------------------------------------------------------------------------------

class _Errors:
    """Exponentially forgotten sums of joint error products for one (scope, kind, lead)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.t: dt.datetime | None = None
        self.S: dict[tuple[str, str], float] = {}     # sum of weight * e_m . e_n over joint verifications
        self.W: dict[tuple[str, str], float] = {}     # sum of weights of those verifications
        self.B: dict[str, np.ndarray] = {}            # sum of weight * e_m

    def decay_to(self, t: dt.datetime, half_life_days: float | None) -> None:
        if self.t is not None and half_life_days is not None and t > self.t:
            g = 0.5 ** ((t - self.t).total_seconds() / 86400.0 / half_life_days)
            for d in (self.S, self.W):
                for k in d:
                    d[k] *= g
            for k in self.B:
                self.B[k] = self.B[k] * g
        if self.t is None or t > self.t:
            self.t = t

    def add(self, errors: Mapping[str, np.ndarray]) -> None:
        ms = sorted(errors)
        for i, m in enumerate(ms):
            em = errors[m]
            self.B[m] = self.B.get(m, np.zeros(self.dim)) + em
            for n in ms[i:]:
                v = float(np.dot(em, errors[n]))
                self.S[(m, n)] = self.S.get((m, n), 0.0) + v
                self.W[(m, n)] = self.W.get((m, n), 0.0) + 1.0


def _nonneg_min_variance(C: np.ndarray) -> np.ndarray:
    """argmin w'Cw subject to sum(w) = 1, w >= 0: the unconstrained solution on an active set, dropping the most
    negative weight until none is negative (deterministic; exact for the small sets used here)."""
    idx = list(range(len(C)))
    while True:
        sub = C[np.ix_(idx, idx)]
        try:
            x = np.linalg.solve(sub, np.ones(len(idx)))
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(sub, np.ones(len(idx)), rcond=None)[0]
        w = x / x.sum() if x.sum() != 0 else np.full(len(idx), 1.0 / len(idx))
        if (w >= 0).all() or len(idx) == 1:
            out = np.zeros(len(C))
            out[idx] = np.clip(w, 0.0, None)
            return out / out.sum()
        del idx[int(np.argmin(w))]


class OnlineConsensus:
    """The causal consensus for one config. Feed it cycles in time order (``run`` does)."""

    def __init__(self, config: Config, kind: str):
        if kind not in ("track", "intensity"):
            raise ValueError(kind)
        self.cfg, self.kind = config, kind
        self.members = config.members(kind)
        self.dim = 2 if kind == "track" else 1
        self.state: dict[tuple, _Errors] = {}

    def _get(self, key) -> _Errors:
        if key not in self.state:
            self.state[key] = _Errors(self.dim)
        return self.state[key]

    def usable(self, fc) -> bool:
        """Does a (lat, lon, vmax) carry what this kind needs (None and NaN are absent)?"""
        vals = (fc[0], fc[1]) if self.kind == "track" else (fc[2],)
        return all(x is not None and math.isfinite(x) for x in vals)

    def _error(self, fc, truth) -> np.ndarray | None:
        if not (self.usable(fc) and self.usable(truth)):
            return None
        if self.kind == "track":
            return error_vector_km(fc[0], fc[1], truth[0], truth[1])
        return np.array([fc[2] - truth[2]])

    def verify(self, t: dt.datetime, basin: str, storm: str, lead: int, errors: Mapping[str, np.ndarray]) -> None:
        if not errors:
            return
        for key in ((basin, lead), (basin, lead, storm)):
            st = self._get(key)
            st.decay_to(t, self.cfg.half_life_days)
            st.add(errors)

    def weights(self, t: dt.datetime, basin: str, storm: str, lead: int, present: list[str]) -> tuple[np.ndarray, dict]:
        """Weights over ``present`` (sorted) and each aid's bias vector, from what verified by ``t``."""
        n = len(present)
        if self.cfg.half_life_days is None:
            return np.full(n, 1.0 / n), {}
        pool = self._get((basin, lead))
        pool.decay_to(t, self.cfg.half_life_days)
        own = self.state.get((basin, lead, storm))
        kappa = self.cfg.storm_boost if own is not None else 0.0
        if kappa:
            own.decay_to(t, self.cfg.half_life_days)
        zero = np.zeros(self.dim)

        def S(m, k):                       # this storm's verifications count (1 + kappa) times
            key = (m, k)
            return pool.S.get(key, 0.0) + (kappa * own.S.get(key, 0.0) if kappa else 0.0)

        def W(m, k):
            key = (m, k)
            return pool.W.get(key, 0.0) + (kappa * own.W.get(key, 0.0) if kappa else 0.0)

        n0 = self.cfg.prior_n
        seen = [S(m, m) / W(m, m) for m in present if W(m, m) > 0]
        mse_pool = float(np.mean(seen)) if seen else 1.0
        bias, moment, var = {}, {}, np.empty(n)
        for i, m in enumerate(present):
            w_mm = W(m, m)
            B = pool.B.get(m, zero) + (kappa * own.B.get(m, zero) if kappa else zero)
            bias[m] = B / (w_mm + n0) if self.cfg.debias else zero
            moment[m] = S(m, m) / w_mm if w_mm > 0 else None
            # shrunk toward the pool by n0 pseudo-verifications; >= n0 mse_pool / (n0 + w_mm) > 0 by Cauchy-Schwarz
            var[i] = (n0 * mse_pool + S(m, m)) / (n0 + w_mm) - float(np.dot(bias[m], bias[m]))
        C = np.diag(var)
        if self.cfg.shrink < 1.0:
            for i, m in enumerate(present):
                for j in range(i + 1, n):
                    k = present[j]
                    w = W(m, k)
                    if w <= 0 or moment[m] is None or moment[k] is None:
                        continue
                    cm = moment[m] - float(np.dot(bias[m], bias[m]))
                    ck = moment[k] - float(np.dot(bias[k], bias[k]))
                    cov = S(m, k) / w - float(np.dot(bias[m], bias[k]))
                    r = cov / math.sqrt(max(cm * ck, 1e-12))
                    r = max(-1.0, min(1.0, r)) * w / (w + n0)       # a correlation from few joint cases shrinks to 0
                    C[i, j] = C[j, i] = (1.0 - self.cfg.shrink) * r * math.sqrt(var[i] * var[j])
            # a pairwise estimate need not be positive definite: lift the spectrum to the diagonal's floor
            ev = np.linalg.eigvalsh(C)
            if ev[0] <= 1e-9 * float(np.mean(var)):
                C = C + (1e-9 * float(np.mean(var)) - ev[0]) * np.eye(n)
        return _nonneg_min_variance(C), bias

    def combine(self, t: dt.datetime, basin: str, storm: str, lead: int, fcs: Mapping[str, tuple]):
        """The consensus at one (cycle, lead) from the members' forecasts there; None below MIN_MEMBERS."""
        present = sorted(m for m, f in fcs.items() if f is not None and self.usable(f))
        if len(present) < MIN_MEMBERS:
            return None, {}
        w, bias = self.weights(t, basin, storm, lead, present)
        if self.kind == "track":
            ref_lat, ref_lon = fcs[present[0]][0], fcs[present[0]][1]
            east = north = 0.0
            for wi, m in zip(w, present):
                e = error_vector_km(fcs[m][0], fcs[m][1], ref_lat, ref_lon) - bias.get(m, 0.0)
                east += wi * e[0]
                north += wi * e[1]
            return shift_km(ref_lat, ref_lon, east, north), dict(zip(present, map(float, w)))
        v = sum(wi * (fcs[m][2] - float(bias.get(m, np.zeros(1))[0])) for wi, m in zip(w, present))
        return float(v), dict(zip(present, map(float, w)))


def run(decks: Iterable[StormDeck], config: Config, kind: str, cycles: Iterable[tuple[str, dt.datetime]] | None = None,
        keep_weights: bool = True):
    """Causal pass over every storm's cycles in time order. Returns ``{(storm, cycle, lead): (forecast, weights)}``
    for the requested ``cycles`` (every analysis cycle when None; weights None unless ``keep_weights``). At each
    time t, everything that verifies at t is learned before anything is issued at t: CARQ at t is in the deck
    when the forecast is made (t + 3 h 30)."""
    model = OnlineConsensus(config, kind)
    decks = {d.storm: d for d in decks}
    times: dict[dt.datetime, list[str]] = {}
    for d in decks.values():
        for t in d.analysis_times:
            times.setdefault(t, []).append(d.storm)
    wanted = None if cycles is None else set(cycles)
    out = {}
    for t in sorted(times):
        storms = sorted(times[t])
        for s in storms:                                   # learn what verifies at t
            d = decks[s]
            truth = d.analysis(t)
            for lead in LEADS:
                t0 = t - dt.timedelta(hours=lead)
                errs = {}
                for m in model.members:
                    fc = d.get(t0, m, lead)
                    if fc is None:
                        continue
                    e = model._error(fc, truth)
                    if e is not None:
                        errs[m] = e
                model.verify(t, d.basin, s, lead, errs)
        for s in storms:                                   # then issue at t
            if wanted is not None and (s, t) not in wanted:
                continue
            d = decks[s]
            for lead in LEADS:
                fcs = {m: f for m in model.members if (f := d.get(t, m, lead)) is not None}
                f, w = model.combine(t, d.basin, s, lead, fcs)
                if f is not None:
                    out[(s, t, lead)] = (f, w if keep_weights else None)
    return out
