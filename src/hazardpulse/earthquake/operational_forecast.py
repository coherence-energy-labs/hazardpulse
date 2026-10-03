"""Operational 30-day M6+ earthquake forecast on the site's global 2-degree grid.

The contract (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 1): for every cell of the
65 x 180 grid HazardPulse publishes (``coherence_engine``: lat -60..70, lon -180..180,
2 degrees; an event is assigned to a cell by ``latlon_to_grid_cell``, edge cells absorb
what lies beyond the band), issued at time ``t``,

    P( at least one ComCat event with M >= 6.0 has its epicentre in the cell
       during [t, t + 30 days) ),

computed from catalog events with origin time strictly before ``t`` -- nothing else.

This module is the ONE implementation of the rate models the program compares and the
site serves. The evaluation scripts (``scripts/earthquake_program``) and the live scorer
(``scripts/fetch_and_score_earthquake.py``) both call :class:`RateEngine`, so the served
probability is, bit for bit, the evaluated one (tests/test_earthquake_operational_forecast.py).

Models
------
Long-term (candidate A, the standard reference forecast): every M >= 5.0 epicentre since
1973 is spread over the grid by a normalised isotropic power-law kernel
``(r^2 + d^2)^-1.5`` (integrated over 16 sub-points per cell, truncated at the support
radius); the normalised map ``s_t`` is mixed with an area-uniform floor:

    lambda_A(c, t) = mu * ((1 - eps) * s_t(c) + eps * area(c) / total_area)

Short-term clustering (candidate B): an ETAS-style sum over every earlier M >= 5.0 event,

    lambda_B = lambda_long + sum_i K * 10^(alpha (M_i - 5)) * Omega(t - t_i) * g_i(c),
    Omega(D) = integral_D^{D+30} (s + c)^-p ds        (days),

with ``g_i`` the same kernel at a magnitude-scaled width ``d0 * 10^(0.5 (M_i - 5))``.
``P = 1 - exp(-lambda)``: the parameters are fitted by the Bernoulli likelihood of THAT
probability, so clustering of several M6+ in one cell-window is absorbed by the fit
rather than double counted.

Everything is a deterministic function of (event arrays, parameters, issue time).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from hazardpulse.earthquake.coherence_engine import (
    GRID_DLAT,
    GRID_DLON,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    N_LAT,
    N_LON,
)

__all__ = [
    "ARTIFACT_SCHEMA",
    "CATALOG_START",
    "EARTH_RADIUS_KM",
    "EventSet",
    "HORIZON_DAYS",
    "INPUT_MIN_MAG",
    "LongTermParams",
    "ModelSpec",
    "N_CELLS",
    "LoadedStack",
    "OperationalArtifactError",
    "RateEngine",
    "STACK_SCHEMA",
    "ShortTermParams",
    "TARGET_MAG",
    "apply_stack",
    "artifact_model_version",
    "causal_aftershock_flags",
    "cell_areas",
    "cell_centres",
    "cell_index",
    "forecast_from_artifact",
    "forecast_with_stack",
    "kernel_matrix",
    "load_artifact",
    "load_stack",
    "omori_window_integral",
    "sha256_text_file",
    "stack_model_version",
    "target_matrix",
    "write_artifact",
    "write_stack",
]

HORIZON_DAYS = 30.0
TARGET_MAG = 6.0
INPUT_MIN_MAG = 5.0
CATALOG_START = "1973-01-01T00:00:00Z"
EARTH_RADIUS_KM = 6371.0
SEC_DAY = 86400.0
N_CELLS = N_LAT * N_LON
ARTIFACT_SCHEMA = "hazardpulse/earthquake-operational-forecast/v1"

# Kernel geometry (part of the model definition; changing any of these is a new model).
N_SUB = 4                     # 4 x 4 sub-points per cell
SUPPORT_FACTOR = 20.0         # support radius = clip(20 * width, 300, 1500) km
SUPPORT_MIN_KM = 300.0
SUPPORT_MAX_KM = 1500.0
_HALF_DIAG_KM = 160.0         # > half-diagonal of a 2-degree cell; centre prefilter margin
MAG_WIDTH_EXPONENT = 0.5      # short-term kernel width d0 * 10^(0.5 (M - 5))

_DEG = math.pi / 180.0


class OperationalArtifactError(RuntimeError):
    """The served artifact is missing, malformed, or cannot cover the issue time."""


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def cell_index(lat, lon) -> np.ndarray:
    """Flat cell index (row * N_LON + col), exactly ``coherence_engine.latlon_to_grid_cell``.

    ``int()`` truncates toward zero; ``np.trunc`` does the same, so an event at lat -61
    lands in row 0 exactly as the site's verifier puts it there.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    row = np.trunc((lat - LAT_MIN) / GRID_DLAT).astype(np.int64)
    col = np.trunc((lon - LON_MIN) / GRID_DLON).astype(np.int64)
    row = np.clip(row, 0, N_LAT - 1)
    col = np.clip(col, 0, N_LON - 1)
    return row * N_LON + col


def cell_centres() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.divmod(np.arange(N_CELLS), N_LON)
    return LAT_MIN + (rows + 0.5) * GRID_DLAT, LON_MIN + (cols + 0.5) * GRID_DLON


def cell_areas() -> np.ndarray:
    """Cell areas in steradians (row-major, length N_CELLS)."""
    rows = np.arange(N_CELLS) // N_LON
    lat0 = (LAT_MIN + rows * GRID_DLAT) * _DEG
    lat1 = (LAT_MIN + (rows + 1) * GRID_DLAT) * _DEG
    return (np.sin(lat1) - np.sin(lat0)) * (GRID_DLON * _DEG)


def _unit(lat_deg, lon_deg) -> np.ndarray:
    la = np.asarray(lat_deg, dtype=np.float64) * _DEG
    lo = np.asarray(lon_deg, dtype=np.float64) * _DEG
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


_GEOM_CACHE: dict = {}


def _geometry():
    if not _GEOM_CACHE:
        clat, clon = cell_centres()
        off = (np.arange(N_SUB) + 0.5) / N_SUB
        rows, cols = np.divmod(np.arange(N_CELLS), N_LON)
        slat = (LAT_MIN + (rows[:, None, None] + off[None, :, None]) * GRID_DLAT)
        slon = (LON_MIN + (cols[:, None, None] + off[None, None, :]) * GRID_DLON)
        slat, slon = np.broadcast_arrays(slat, slon)
        slat = slat.reshape(N_CELLS, N_SUB * N_SUB)
        slon = slon.reshape(N_CELLS, N_SUB * N_SUB)
        _GEOM_CACHE["centre_unit"] = _unit(clat, clon)
        _GEOM_CACHE["sub_unit"] = _unit(slat, slon)                 # (N_CELLS, 16, 3)
        _GEOM_CACHE["sub_w"] = np.cos(slat * _DEG)                  # area weight per sub-point
    return _GEOM_CACHE


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _canon(t, lat, lon, mag):
    """Canonical precision: ms times, 1e-4 degree positions, 0.01 magnitudes.

    The frozen catalog in the served artifact stores exactly these values, and the
    evaluation and the live path canonicalise the same way, so both build identical
    float arrays from the same events.
    """
    return (np.round(np.asarray(t, np.float64), 3), np.round(np.asarray(lat, np.float64), 4),
            np.round(np.asarray(lon, np.float64), 4), np.round(np.asarray(mag, np.float64), 2))


def _parse_time(text: str) -> float:
    parsed = dt.datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


@dataclasses.dataclass(frozen=True)
class EventSet:
    """Time-sorted catalog arrays (epoch seconds, degrees, magnitude), canonical precision."""

    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    mag: np.ndarray

    @classmethod
    def from_arrays(cls, t, lat, lon, mag, *, min_mag: float | None = None) -> "EventSet":
        t, lat, lon, mag = _canon(t, lat, lon, mag)
        keep = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(mag)
        if min_mag is not None:
            keep &= mag >= min_mag
        t, lat, lon, mag = t[keep], lat[keep], lon[keep], mag[keep]
        order = np.lexsort((mag, lon, lat, t))      # total order: ties broken deterministically
        return cls(t[order], lat[order], lon[order], mag[order])

    @classmethod
    def from_dicts(cls, events: Iterable[dict], *, min_mag: float | None = None) -> "EventSet":
        t, la, lo, m = [], [], [], []
        for e in events:
            try:
                mag = float(e["mag"])
                lat = float(e["latitude"])
                lon = float(e["longitude"])
                tt = _parse_time(e["time"]) if isinstance(e["time"], str) else float(e["time"])
            except (KeyError, TypeError, ValueError):
                continue
            t.append(tt); la.append(lat); lo.append(lon); m.append(mag)
        return cls.from_arrays(t, la, lo, m, min_mag=min_mag)

    def __len__(self) -> int:
        return int(self.t.size)

    def select(self, mask) -> "EventSet":
        mask = np.asarray(mask)
        return EventSet(self.t[mask], self.lat[mask], self.lon[mask], self.mag[mask])

    def before(self, t: float) -> "EventSet":
        return self.select(self.t < t)

    def concat(self, other: "EventSet") -> "EventSet":
        return EventSet.from_arrays(np.concatenate([self.t, other.t]), np.concatenate([self.lat, other.lat]),
                                    np.concatenate([self.lon, other.lon]), np.concatenate([self.mag, other.mag]))

    def n_before(self, t: float) -> int:
        return int(np.searchsorted(self.t, t, side="left"))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def support_radius_km(width_km) -> np.ndarray:
    return np.clip(SUPPORT_FACTOR * np.asarray(width_km, np.float64), SUPPORT_MIN_KM, SUPPORT_MAX_KM)


def kernel_matrix(lat, lon, width_km, *, chunk: int = 256) -> sparse.csr_matrix:
    """Row i = event i's normalised spatial kernel over the grid (rows sum to 1, or 0).

    Weight of cell c = sum over its 16 sub-points s of cos(lat_s) * (r_is^2 + d_i^2)^-1.5
    for r_is <= R_i (great-circle km), normalised over the grid. An event whose support
    holds no sub-point of the domain gets an empty row (it adds nothing anywhere).
    """
    g = _geometry()
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    width = np.broadcast_to(np.asarray(width_km, np.float64), lat.shape).astype(np.float64)
    radius = support_radius_km(width)
    n = lat.size
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    centre = g["centre_unit"]
    sub = g["sub_unit"]
    sub_w = g["sub_w"]
    for s in range(0, n, chunk):
        e = slice(s, min(n, s + chunk))
        u = _unit(lat[e], lon[e])                                   # (m, 3)
        cosang = np.clip(u @ centre.T, -1.0, 1.0)                   # (m, N_CELLS)
        dist_c = EARTH_RADIUS_KM * np.arccos(cosang)
        cand = dist_c <= (radius[e][:, None] + _HALF_DIAG_KM)
        ev, cell = np.nonzero(cand)                                 # row-major: ev ascending, cell ascending
        if ev.size:
            diff = u[ev][:, None, :] - sub[cell]                    # (P, 16, 3)
            chord = np.sqrt(np.einsum("psk,psk->ps", diff, diff))
            r = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(chord / 2.0, 1.0))
            d = width[e][ev][:, None]
            f = sub_w[cell] * (r * r + d * d) ** -1.5
            f = np.where(r <= radius[e][ev][:, None], f, 0.0)
            w = f.sum(axis=1)
            tot = np.bincount(ev, weights=w, minlength=e.stop - e.start)
            nz = w > 0.0
            ev, cell, w = ev[nz], cell[nz], w[nz]
            w = w / tot[ev]
            counts = np.bincount(ev, minlength=e.stop - e.start)
        else:
            counts = np.zeros(e.stop - e.start, dtype=np.int64)
            cell = np.zeros(0, np.int64)
            w = np.zeros(0, np.float64)
        indptr[e.start + 1:e.stop + 1] = counts
        idx_parts.append(cell.astype(np.int32))
        val_parts.append(w.astype(np.float64))
    indptr = np.cumsum(indptr)
    data = np.concatenate(val_parts) if val_parts else np.zeros(0)
    indices = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=(n, N_CELLS))


# ---------------------------------------------------------------------------
# Causal declustering (Gardner-Knopoff windows, aftershock rule only)
# ---------------------------------------------------------------------------

def gk_window(mag) -> tuple[np.ndarray, np.ndarray]:
    """Gardner & Knopoff (1974) window: (distance km, time days), as definitive_model uses."""
    mag = np.asarray(mag, np.float64)
    d_km = 10.0 ** (0.1238 * mag + 0.983)
    t_days = np.where(mag >= 6.5, 10.0 ** (0.032 * mag + 2.7389), 10.0 ** (0.5409 * mag - 0.547))
    return d_km, t_days


def causal_aftershock_flags(events: EventSet) -> np.ndarray:
    """True for an event inside the GK window of an EARLIER event at least as large.

    Only earlier events are consulted (no foreshock rule), so an event's flag is fixed
    the moment it happens and no later event can ever change it.
    """
    n = len(events)
    flags = np.zeros(n, dtype=bool)
    d_km, t_days = gk_window(events.mag)
    u = _unit(events.lat, events.lon)
    ends = np.searchsorted(events.t, events.t + t_days * SEC_DAY, side="right")
    for j in range(n):
        lo, hi = j + 1, int(ends[j])
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        cand = (events.mag[sl] <= events.mag[j]) & (events.t[sl] > events.t[j])
        if not cand.any():
            continue
        cosang = np.clip(u[sl] @ u[j], -1.0, 1.0)
        near = EARTH_RADIUS_KM * np.arccos(cosang) <= d_km[j]
        hit = np.nonzero(cand & near)[0]
        if hit.size:
            flags[lo + hit] = True
    return flags


# ---------------------------------------------------------------------------
# Omori-Utsu window integral
# ---------------------------------------------------------------------------

def omori_window_integral(delta_days, c_days: float, p: float, horizon_days: float = HORIZON_DAYS) -> np.ndarray:
    """Integral of (s + c)^-p over s in [delta, delta + horizon] (days)."""
    a = np.asarray(delta_days, np.float64) + c_days
    b = a + horizon_days
    if abs(p - 1.0) < 1e-9:
        return np.log(b / a)
    q = 1.0 - p
    return (a ** q - b ** q) / (p - 1.0)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def target_matrix(events: EventSet, issue_times: Sequence[float], *, min_mag: float = TARGET_MAG,
                  horizon_days: float = HORIZON_DAYS) -> sparse.csr_matrix:
    """(n_issue x N_CELLS) count of M>=min_mag epicentres in [t, t + horizon) per cell."""
    sel = events.mag >= min_mag
    t = events.t[sel]
    cells = cell_index(events.lat[sel], events.lon[sel])
    issue = np.asarray(issue_times, np.float64)
    lo = np.searchsorted(t, issue, side="left")
    hi = np.searchsorted(t, issue + horizon_days * SEC_DAY, side="left")
    rows, cols = [], []
    for k in range(issue.size):
        if hi[k] > lo[k]:
            rows.append(np.full(hi[k] - lo[k], k))
            cols.append(cells[lo[k]:hi[k]])
    if not rows:
        return sparse.csr_matrix((issue.size, N_CELLS))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(r.size), (r, c)), shape=(issue.size, N_CELLS))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LongTermParams:
    kernel_km: float
    declustered: bool
    mu: float
    eps: float


@dataclasses.dataclass(frozen=True)
class ShortTermParams:
    d0_km: float
    K: float
    alpha: float
    c_days: float
    p: float


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """A = long-term only; B = long-term + short-term (``short`` set)."""

    name: str
    long: LongTermParams
    short: ShortTermParams | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "long": dataclasses.asdict(self.long),
                "short": dataclasses.asdict(self.short) if self.short else None}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(name=str(d["name"]), long=LongTermParams(**d["long"]),
                   short=ShortTermParams(**d["short"]) if d.get("short") else None)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RateEngine:
    """Rate maps of one model over one catalog; every map uses events strictly before t.

    The long-term kernel ``G_long`` and the short-term kernel ``G_short`` are built once
    for the whole catalog (row i depends on event i alone), so the maps at time t read
    rows ``[0, n_before(t))`` only: an event at or after t has no path into them.
    """

    def __init__(self, events: EventSet, spec: ModelSpec | None = None, *,
                 kernel_km: float | None = None, declustered: bool | None = None,
                 d0_km: float | None = None) -> None:
        self.spec = spec
        ev = events.select(events.mag >= INPUT_MIN_MAG)
        self.events = ev
        k_km = spec.long.kernel_km if spec is not None else kernel_km
        decl = spec.long.declustered if spec is not None else bool(declustered)
        self.flags = causal_aftershock_flags(ev) if decl else np.zeros(len(ev), dtype=bool)
        self.long_rows = np.nonzero(~self.flags)[0]
        if k_km is None:
            raise ValueError("kernel_km is required")
        G = kernel_matrix(ev.lat[self.long_rows], ev.lon[self.long_rows], k_km)
        self.G_long = G
        self.long_t = ev.t[self.long_rows]
        d0 = spec.short.d0_km if (spec is not None and spec.short is not None) else d0_km
        self.G_short = None
        if d0 is not None:
            width = d0 * 10.0 ** (MAG_WIDTH_EXPONENT * (ev.mag - 5.0))
            self.G_short = kernel_matrix(ev.lat, ev.lon, width)
            self.short_mass = np.asarray(self.G_short.sum(axis=1)).ravel()
        self.area_pdf = cell_areas() / cell_areas().sum()

    # -- long-term --------------------------------------------------------
    def long_term_density(self, t: float) -> np.ndarray:
        """Normalised smoothed density s_t (sums to 1) of the long-term input before t.

        Rows ``[0, n)`` of the CSR kernel are the contiguous buffer prefix
        ``data[:indptr[n]]``; ``bincount`` sums it in buffer order, so the same events give
        the same bits whatever else the catalog holds after t.
        """
        n = int(np.searchsorted(self.long_t, t, side="left"))
        if n == 0:
            return self.area_pdf.copy()
        end = int(self.G_long.indptr[n])
        S = np.bincount(self.G_long.indices[:end], weights=self.G_long.data[:end], minlength=N_CELLS)
        tot = S.sum()
        return S / tot if tot > 0 else self.area_pdf.copy()

    def long_term_rate(self, t: float, mu: float, eps: float, density: np.ndarray | None = None) -> np.ndarray:
        s = self.long_term_density(t) if density is None else density
        return mu * ((1.0 - eps) * s + eps * self.area_pdf)

    # -- short-term -------------------------------------------------------
    def short_term_weights(self, t: float, sp: ShortTermParams) -> np.ndarray:
        n = self.events.n_before(t)
        delta = (t - self.events.t[:n]) / SEC_DAY
        return sp.K * 10.0 ** (sp.alpha * (self.events.mag[:n] - 5.0)) * omori_window_integral(delta, sp.c_days, sp.p)

    def short_term_rate(self, t: float, sp: ShortTermParams) -> np.ndarray:
        if self.G_short is None:
            raise ValueError("engine built without a short-term kernel")
        w = self.short_term_weights(t, sp)
        n = w.size
        if n == 0:
            return np.zeros(N_CELLS)
        G = self.G_short
        end = int(G.indptr[n])
        per_nz = np.repeat(w, np.diff(G.indptr[:n + 1]))
        return np.bincount(G.indices[:end], weights=G.data[:end] * per_nz, minlength=N_CELLS)

    # -- full model ---------------------------------------------------------
    def rates(self, t: float, spec: ModelSpec | None = None) -> dict[str, np.ndarray]:
        spec = spec or self.spec
        if spec is None:
            raise ValueError("no model spec")
        lam_long = self.long_term_rate(t, spec.long.mu, spec.long.eps)
        lam_short = self.short_term_rate(t, spec.short) if spec.short is not None else np.zeros(N_CELLS)
        lam = lam_long + lam_short
        return {"lambda_long": lam_long, "lambda_short": lam_short, "lambda": lam,
                "probability": -np.expm1(-lam)}

    def probability(self, t: float, spec: ModelSpec | None = None) -> np.ndarray:
        return self.rates(t, spec)["probability"]


# ---------------------------------------------------------------------------
# Artifact (served model): parameters + the frozen M>=5 catalog, content-bound identity
# ---------------------------------------------------------------------------

def sha256_text_file(path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF -- the same digest on a Windows
    (core.autocrlf) and a Linux checkout; equal to ``hazardpulse.hurricane.ri_model.sha256_file``
    (asserted by the tests)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def artifact_model_version(path) -> str:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    return f"{meta['model_name']}-{sha256_text_file(path)[:12]}"


def write_artifact(path, spec: ModelSpec, frozen: EventSet, *, cutoff: str, model_name: str,
                   provenance: dict, gbt: dict | None = None) -> str:
    """Write the served artifact (LF line endings) and return its model_version.

    ``gbt`` (candidate C): ``{"name", "long_A", "feature_names", "feature_dtype",
    "live_span_days", "payload"}`` -- ``spec`` is then the rate model whose maps (with A's
    long-term rate, ``long_A``) feed the trees, and the trees set the probability.
    """
    body = {
        "schema": ARTIFACT_SCHEMA,
        "model_name": model_name,
        "contract": {
            "event": f"at least one ComCat event with M >= {TARGET_MAG} whose epicentre maps to the cell "
                     f"(coherence_engine.latlon_to_grid_cell) during [issue, issue + {HORIZON_DAYS:g} days)",
            "grid": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX,
                     "dlat": GRID_DLAT, "dlon": GRID_DLON, "n_lat": N_LAT, "n_lon": N_LON},
            "inputs": f"ComCat events with M >= {INPUT_MIN_MAG} and origin time strictly before the issue time",
            "horizon_days": HORIZON_DAYS,
            "target_magnitude_min": TARGET_MAG,
        },
        "kernel_geometry": {"n_sub": N_SUB, "support_factor": SUPPORT_FACTOR, "support_min_km": SUPPORT_MIN_KM,
                            "support_max_km": SUPPORT_MAX_KM, "mag_width_exponent": MAG_WIDTH_EXPONENT},
        "spec": spec.to_dict(),
        "frozen_catalog": {
            "description": f"ComCat M>={INPUT_MIN_MAG} events from {CATALOG_START} to the cutoff "
                           "(exclusive); live events from the cutoff on come from the scorer's own fetch",
            "start": CATALOG_START,
            "cutoff": cutoff,
            "n_events": len(frozen),
            "t": [float(x) for x in frozen.t],
            "lat": [float(x) for x in frozen.lat],
            "lon": [float(x) for x in frozen.lon],
            "mag": [float(x) for x in frozen.mag],
        },
        "provenance": provenance,
    }
    if gbt is not None:
        body["gbt"] = gbt
    text = json.dumps(body, separators=(",", ":"), sort_keys=False, allow_nan=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return artifact_model_version(path)


@dataclasses.dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    model_name: str
    model_version: str
    sha256: str
    spec: ModelSpec
    frozen: EventSet
    cutoff: float
    meta: dict
    gbt: dict | None = None


GBT_SECTION_KEYS = {"name", "long_A", "feature_names", "feature_dtype", "live_span_days", "payload"}


def load_artifact(path) -> LoadedArtifact:
    path = Path(path)
    if not path.is_file():
        raise OperationalArtifactError(f"served earthquake artifact missing: {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise OperationalArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if meta.get("schema") != ARTIFACT_SCHEMA:
        raise OperationalArtifactError(f"{path}: schema {meta.get('schema')!r} != {ARTIFACT_SCHEMA!r}")
    geom = meta.get("kernel_geometry", {})
    expect = {"n_sub": N_SUB, "support_factor": SUPPORT_FACTOR, "support_min_km": SUPPORT_MIN_KM,
              "support_max_km": SUPPORT_MAX_KM, "mag_width_exponent": MAG_WIDTH_EXPONENT}
    if geom != expect:
        raise OperationalArtifactError(f"{path}: kernel geometry {geom} differs from this code {expect}")
    fc = meta["frozen_catalog"]
    frozen = EventSet.from_arrays(fc["t"], fc["lat"], fc["lon"], fc["mag"])
    if len(frozen) != int(fc["n_events"]):
        raise OperationalArtifactError(f"{path}: frozen catalog has {len(frozen)} events, header says {fc['n_events']}")
    gbt = meta.get("gbt")
    if gbt is not None:
        from hazardpulse.earthquake.operational_features import CORE_FEATURES

        missing = GBT_SECTION_KEYS - set(gbt)
        if missing:
            raise OperationalArtifactError(f"{path}: gbt section lacks {sorted(missing)}")
        if list(gbt["feature_names"]) != list(CORE_FEATURES) or list(gbt["payload"]["feature_names"]) != list(CORE_FEATURES):
            raise OperationalArtifactError(f"{path}: gbt feature names differ from operational_features.CORE_FEATURES")
        if gbt["feature_dtype"] != "float32":
            raise OperationalArtifactError(f"{path}: unsupported feature dtype {gbt['feature_dtype']!r}")
        if meta["spec"].get("short") is None:
            raise OperationalArtifactError(f"{path}: a gbt artifact needs the short-term rate model")
    sha = sha256_text_file(path)
    return LoadedArtifact(path=path, model_name=meta["model_name"], model_version=f"{meta['model_name']}-{sha[:12]}",
                          sha256=sha, spec=ModelSpec.from_dict(meta["spec"]), frozen=frozen,
                          cutoff=_parse_time(fc["cutoff"]), meta=meta, gbt=gbt)


def serving_catalog(art: LoadedArtifact, live_events: Iterable[dict], issue_time: float) -> EventSet:
    """Frozen events before the cutoff + live events from the cutoff to the issue time.

    Refuses when the live fetch does not reach back to the cutoff (a hole would silently
    drop events), unless the issue time is itself before the cutoff.
    """
    live = EventSet.from_dicts(live_events, min_mag=INPUT_MIN_MAG)
    if issue_time <= art.cutoff:
        return art.frozen.before(issue_time)
    if len(live) == 0 or live.t[0] > art.cutoff + 2 * SEC_DAY:
        start = "none" if len(live) == 0 else dt.datetime.fromtimestamp(live.t[0], dt.timezone.utc).isoformat()
        raise OperationalArtifactError(
            f"live catalog starts at {start}, after the artifact cutoff "
            f"{dt.datetime.fromtimestamp(art.cutoff, dt.timezone.utc).isoformat()}: events in between would be lost")
    tail = live.select((live.t >= art.cutoff) & (live.t < issue_time))
    return art.frozen.concat(tail)


def forecast_from_artifact(art: LoadedArtifact, live_events: Iterable[dict], issue_time: dt.datetime | float) -> dict:
    """Full-grid forecast at ``issue_time`` from the served artifact + the live catalog."""
    t = issue_time.timestamp() if isinstance(issue_time, dt.datetime) else float(issue_time)
    live_events = list(live_events)
    catalog = serving_catalog(art, live_events, t)
    engine = RateEngine(catalog, art.spec)
    out = engine.rates(t)
    n = int(engine.events.n_before(t))
    digest = input_digest(engine.events, n)
    if art.gbt is not None:
        rate_inputs = gbt_rate_inputs(engine, t, art.spec, LongTermParams(**art.gbt["long_A"]))
        c45, c25 = gbt_feature_catalogs(catalog, live_events, t, float(art.gbt["live_span_days"]))
        from hazardpulse.earthquake.operational_features import core_features

        X = core_features(t, c45, c25, *rate_inputs).astype(np.float32)
        out["probability_rate_model"] = out["probability"]
        out["probability"] = gbt_probability(art.gbt["payload"], X)
        h = hashlib.sha256(digest.encode("ascii"))
        for arr in (c25.t, c25.mag, c45.t, c45.mag):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        digest = h.hexdigest()
    out["model_version"] = art.model_version
    out["n_input_events"] = n
    out["input_sha256"] = digest
    return out


# ---------------------------------------------------------------------------
# Stack on the served artifact (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10, amendment E1)
# ---------------------------------------------------------------------------

STACK_SCHEMA = "hazardpulse/earthquake-operational-stack/v1"
STACK_CLIP = 1e-12


@dataclasses.dataclass(frozen=True)
class LoadedStack:
    path: Path
    model_name: str
    model_version: str
    base_model_version: str
    a: float
    c: float
    b: float
    g_log10: np.ndarray
    meta: dict


def stack_model_version(path) -> str:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    return f"{meta['model_name']}-{sha256_text_file(path)[:12]}"


def write_stack(path, *, model_name: str, base_model_version: str, a: float, c: float, b: float,
                g_log10: np.ndarray, provenance: dict) -> str:
    """``logit p = a + c logit(p_base) + b g_log10[cell]`` on top of the artifact named
    ``base_model_version``; returns the stack's model_version."""
    g = np.asarray(g_log10, np.float64)
    if g.shape != (N_CELLS,) or not np.all(np.isfinite(g)):
        raise OperationalArtifactError(f"stack map must be {N_CELLS} finite values")
    body = {"schema": STACK_SCHEMA, "model_name": model_name, "base_model_version": base_model_version,
            "formula": "logit p = a + c logit(p_base) + b g_log10[cell]",
            "coefficients": {"a": float(a), "c": float(c), "b": float(b)},
            "g_log10": [float(x) for x in g], "provenance": provenance}
    text = json.dumps(body, separators=(",", ":"), allow_nan=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return stack_model_version(path)


def load_stack(path, base: LoadedArtifact) -> LoadedStack:
    """Refused unless it names exactly the loaded base artifact's model_version."""
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != STACK_SCHEMA:
        raise OperationalArtifactError(f"{path}: schema {meta.get('schema')!r} != {STACK_SCHEMA!r}")
    if meta.get("base_model_version") != base.model_version:
        raise OperationalArtifactError(f"{path} stacks on {meta.get('base_model_version')!r}, "
                                       f"the served base is {base.model_version!r}")
    g = np.asarray(meta["g_log10"], np.float64)
    if g.shape != (N_CELLS,) or not np.all(np.isfinite(g)):
        raise OperationalArtifactError(f"{path}: map is not {N_CELLS} finite values")
    co = meta["coefficients"]
    return LoadedStack(path=path, model_name=meta["model_name"], model_version=stack_model_version(path),
                       base_model_version=base.model_version, a=float(co["a"]), c=float(co["c"]),
                       b=float(co["b"]), g_log10=g, meta=meta)


def apply_stack(stack: LoadedStack, p_base: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_base, np.float64), STACK_CLIP, 1 - STACK_CLIP)
    eta = stack.a + stack.c * (np.log(p) - np.log1p(-p)) + stack.b * stack.g_log10
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


def forecast_with_stack(art: LoadedArtifact, stack: LoadedStack, live_events: Iterable[dict],
                        issue_time: dt.datetime | float) -> dict:
    """The base artifact's forecast with the stack applied; the base probability is kept."""
    out = forecast_from_artifact(art, live_events, issue_time)
    out["probability_base"] = out["probability"]
    out["probability"] = apply_stack(stack, out["probability_base"])
    out["base_model_version"] = art.model_version
    out["model_version"] = stack.model_version
    return out


def gbt_rate_inputs(engine: RateEngine, t: float, spec_b: ModelSpec, long_a: LongTermParams):
    """(lambda_A, lambda_B, lambda_short) exactly as the evaluation fed them to the trees:
    lambda_A and lambda_short stored as float32 maps and read back as float64, lambda_B
    recovered from B's probability (scripts/earthquake_program/fit_ab.py, fit_c.py)."""
    r = engine.rates(t, spec_b)
    s = engine.long_term_density(t)
    lam_a = long_a.mu * ((1 - long_a.eps) * s + long_a.eps * engine.area_pdf)
    lam_a = lam_a.astype(np.float32).astype(np.float64)
    lam_s = r["lambda_short"].astype(np.float32).astype(np.float64)
    lam_b = -np.log1p(-r["probability"])
    return lam_a, lam_b, lam_s


def _live_arrays(live_events: list[dict]) -> dict[str, np.ndarray]:
    t, la, lo, m, d = [], [], [], [], []
    for e in live_events:
        try:
            tt = _parse_time(e["time"]) if isinstance(e["time"], str) else float(e["time"])
            row = (tt, float(e["latitude"]), float(e["longitude"]), float(e["mag"]))
        except (KeyError, TypeError, ValueError):
            continue
        dep = e.get("depth")
        try:
            dep = float(dep) if dep is not None else float("nan")
        except (TypeError, ValueError):
            dep = float("nan")
        t.append(row[0]); la.append(row[1]); lo.append(row[2]); m.append(row[3]); d.append(dep)
    ct, cla, clo, cm = _canon(t, la, lo, m)
    return {"t": ct, "lat": cla, "lon": clo, "mag": cm, "depth": np.asarray(d, np.float64)}


def gbt_feature_catalogs(catalog: EventSet, live_events: list[dict], t: float, span_days: float):
    """The M4.5+ (with depth) and M2.5+ catalogs candidate C reads at t.

    Inside the live span (the scorer's fetch, ``span_days`` before t) every event comes from
    the live fetch; before it, only the M>=5 rows of the rate catalog (frozen + live tail),
    which is all the long-memory features read (M>=5 b-value over 10 y, time since the last
    M>=6 / M>=7). Refuses a live fetch that does not reach back to the span start.
    """
    from hazardpulse.earthquake.operational_features import cell_catalog_from_arrays

    span_start = t - span_days * SEC_DAY
    live = _live_arrays(live_events)
    inside = (live["t"] >= span_start) & (live["t"] < t)
    if not inside.any() or live["t"][inside].min() > span_start + SEC_DAY:
        raise OperationalArtifactError(
            f"live catalog does not cover the {span_days:g} days before the issue time "
            "that the boosted-tree features read")
    old = catalog.t < span_start
    l45 = inside & (live["mag"] >= 4.5)
    c45 = cell_catalog_from_arrays(
        np.concatenate([catalog.t[old], live["t"][l45]]), np.concatenate([catalog.lat[old], live["lat"][l45]]),
        np.concatenate([catalog.lon[old], live["lon"][l45]]), np.concatenate([catalog.mag[old], live["mag"][l45]]),
        np.concatenate([np.full(int(old.sum()), np.nan), live["depth"][l45]]))
    c25 = cell_catalog_from_arrays(live["t"][inside], live["lat"][inside], live["lon"][inside], live["mag"][inside])
    return c45, c25


def gbt_probability(payload: dict, X: np.ndarray) -> np.ndarray:
    """Trees walked by the repository's LightGBM payload scorer (LightGBM's split semantics),
    then LightGBM's binary transform 1 / (1 + exp(-raw))."""
    from hazardpulse.tornado.lgbm_payload import predict_raw

    raw = predict_raw(payload, np.asarray(X, np.float32))
    return 1.0 / (1.0 + np.exp(-raw))


def input_digest(events: EventSet, n: int) -> str:
    """SHA-256 of the exact model input: the first n canonical (t, lat, lon, mag) rows."""
    h = hashlib.sha256()
    for arr in (events.t[:n], events.lat[:n], events.lon[:n], events.mag[:n]):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()
