"""Storm-object feature and label engine (v3) -- ONE implementation for training and live.

Every candidate input of the tornado model, computed per ProbSevere storm
observation, plus the labels it is judged against. The training feature store
(scripts/audit_20261001/build_feature_store.py) and the live scorer call the
same functions, so a feature can never mean one thing in training and another
in production.

Blocks
------
P   every numeric ProbSevere attribute, NOAA's ProbTor/Hail/Wind included
E   storm track history (<= STORM_HISTORY_LOOKBACK steps): levels, deltas,
    maxima, means and least-squares slopes of the rotation / severity series
H80 the 80 km HRRR environment (definitive_model.extract_block_h)
C80 the 80 km coherence field (definitive_model.extract_block_c)
H9  the STORM-SCALE HRRR environment on native 9 km Lambert blocks: value at the
    storm's block, 27 km window max/mean, 63 km window max, derived STP, and
    the storm-relative shear geometry
C9  the coherence framework at storm scale: the variable-coefficient Helmholtz
    field D lap(tau) - Gamma tau + S = 0 solved (certified PCG) at three
    coherence lengths (9, 27, 81 km), its gradient, alignment and tilting
    torsion -- and, as the CONTROL that decides whether the PDE adds anything
    beyond smoothing, the same diagnostics from a Gaussian smoothing of S/Gamma
    at the same three scales.

Labels (absolute UTC instants; SPC reports via definitive_model)
------
storm_H  a report within H min whose start lies within R of the storm's
         centroid ADVECTED by its motion to the report time,
         R = clip(sqrt(size / pi) + 5 km, 8, 25) km -- "this storm produces a
         tornado", the target NOAA's ProbTor predicts
nbhd_H   a report within H min and 40 km of the current centroid (the v2 label)
for H in 30 / 60 / 90 min; plus the EF of the strongest storm_60 match and the
lead time to the first storm_90 match.
"""
from __future__ import annotations

import math

import numpy as np

from hazardpulse.coherence.tau_c_solver import solve_helmholtz_2d_certified
from hazardpulse.data import hrrr as H
from hazardpulse.tornado import definitive_model as dm

# ---------------------------------------------------------------------------
# Block P -- ProbSevere attributes
# ---------------------------------------------------------------------------
PS_ATTRS: tuple[str, ...] = (
    "ps", "ps_tor", "ps_hail", "ps_wind", "ps_severe",
    "mucape", "mlcape", "mlcin", "ebshear", "srh01", "mesh", "vil_density",
    "flash_rate", "flash_density", "maxllaz", "p98llaz", "p98mlaz", "lja", "size",
    "motion_east", "motion_south", "pwat", "cape_m10m30", "meanwind_1_3kmagl",
    "wetbulb_0c_hgt", "avg_beam_hgt", "maxrc_emiss", "maxrc_icecf",
)
MAX_MOTION_MS = 40.0   # ProbSevere motion has garbage outliers (to -240 m/s)


def block_p(storm: dict) -> np.ndarray:
    out = np.full(len(PS_ATTRS), np.nan, dtype=np.float32)
    for i, k in enumerate(PS_ATTRS):
        v = storm.get(k)
        if v is not None:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# Block E -- track history
# ---------------------------------------------------------------------------
TRACK_SERIES: tuple[str, ...] = ("maxllaz", "p98llaz", "ps_tor", "ps", "mesh", "flash_rate", "size")
TRACK_STATS: tuple[str, ...] = ("delta", "max", "mean", "slope")
TRACK_NAMES: tuple[str, ...] = (
    ("age_min", "n_steps", "rot_sustained_min", "motion_speed", "motion_dir_deg")
    + tuple(f"{s}_{t}" for s in TRACK_SERIES for t in TRACK_STATS)
)


def storm_motion(storm: dict) -> tuple[float, float]:
    """(east, north) storm motion in m/s, clipped to MAX_MOTION_MS."""
    ue = float(storm.get("motion_east") or 0.0)
    vn = -float(storm.get("motion_south") or 0.0)
    sp = math.hypot(ue, vn)
    if not math.isfinite(sp) or sp == 0.0:
        return 0.0, 0.0
    if sp > MAX_MOTION_MS:
        ue, vn = ue * MAX_MOTION_MS / sp, vn * MAX_MOTION_MS / sp
    return ue, vn


def block_e(storm: dict, history: list[dict], step_minutes: float) -> np.ndarray:
    n = len(history)
    out = [max(0, n - 1) * step_minutes, float(n)]
    out.append(sum(1 for s in history if float(s.get("maxllaz") or 0.0) > dm.ROTATION_THRESHOLD) * step_minutes)
    ue, vn = storm_motion(storm)
    out += [math.hypot(ue, vn), (math.degrees(math.atan2(ue, vn)) + 360.0) % 360.0]
    x = np.arange(n, dtype=np.float64)
    for key in TRACK_SERIES:
        v = np.array([float(s.get(key) or 0.0) for s in history], dtype=np.float64)
        if n == 0:
            out += [np.nan] * 4
            continue
        delta = v[-1] - v[-2] if n >= 2 else 0.0
        slope = float(np.polyfit(x, v, 1)[0]) if n >= 3 else delta
        out += [delta, float(v.max()), float(v.mean()), slope]
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Storm-scale analysis (9 km native blocks) -- computed once per analysis
# ---------------------------------------------------------------------------
LCC_K = 3
C9_SCALES_KM: tuple[float, ...] = (9.0, 27.0, 81.0)
C9_TOL = 1e-6
H9_VARS: tuple[str, ...] = (
    "mlcape", "mucape", "mlcin", "srh_01", "srh_03", "refc", "pwat",
    "td_dep", "shear_06", "shear_01", "storm_speed", "stp",
)
H9_STATS: tuple[str, ...] = ("at", "max27", "mean27", "max63")
H9_NAMES: tuple[str, ...] = (
    tuple(f"h9_{v}_{s}" for v in H9_VARS for s in H9_STATS)
    + ("h9_shear01_streamwise", "h9_shear06_storm_angle")
)
C9_DIAG: tuple[str, ...] = ("tau", "grad", "align", "torsion")
C9_NAMES: tuple[str, ...] = (
    tuple(f"c9_{kind}_{int(s)}km_{d}" for kind in ("pde", "gauss") for s in C9_SCALES_KM for d in C9_DIAG)
    + ("c9_S_over_Gamma", "c9_tau27_max27")
)


def _window_stat(field: np.ndarray, r: int, how: str) -> np.ndarray:
    """(2r+1)^2 window max or mean, edge-padded; separable (max/mean are)."""
    w = 2 * r + 1
    a = np.pad(np.asarray(field, dtype=np.float64), r, mode="edge")
    for axis in (0, 1):
        n = a.shape[axis] - w + 1
        sl = [slice(None), slice(None)]
        acc = None
        for k in range(w):
            sl[axis] = slice(k, k + n)
            part = a[tuple(sl)]
            if acc is None:
                acc = part.copy()
            elif how == "max":
                np.maximum(acc, part, out=acc)
            else:
                acc += part
        a = acc if how == "max" else acc / w
    return a


def _coarsen(a: np.ndarray, f: int) -> np.ndarray:
    ny, nx = a.shape
    py, px = (-ny) % f, (-nx) % f
    a = np.pad(a, ((0, py), (0, px)), mode="edge")
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def _refine(a: np.ndarray, f: int, shape: tuple[int, int]) -> np.ndarray:
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)[: shape[0], : shape[1]]


def _gaussian_smooth(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Separable Gaussian smoothing, edge-normalised (pure NumPy)."""
    rad = max(1, int(math.ceil(3.0 * sigma_cells)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma_cells) ** 2)
    k /= k.sum()

    def conv(a: np.ndarray, axis: int) -> np.ndarray:
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, a)

    ones = np.ones_like(field, dtype=np.float64)
    num = conv(conv(field.astype(np.float64), 0), 1)
    den = conv(conv(ones, 0), 1)
    return num / den


def _block_theta() -> np.ndarray:
    """Grid rotation angle (rad) at each 9 km block centre: grid +x is -theta from east."""
    lat, lon = H.native_latlon()
    k = LCC_K
    ny, nx = (H.NATIVE_NY // k) * k, (H.NATIVE_NX // k) * k
    lonc = lon[k // 2:ny:k, k // 2:nx:k]
    return H._LCC_N * np.radians(lonc - math.degrees(H._LCC_LON0))


_THETA9: np.ndarray | None = None


def _earth_to_grid(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of hrrr.rotate_grid_winds_to_earth on the 9 km block grid."""
    global _THETA9
    if _THETA9 is None:
        _THETA9 = _block_theta()
    c, s = np.cos(_THETA9), np.sin(_THETA9)
    return c * u - s * v, s * u + c * v


def _diagnostics(tau: np.ndarray, us_g: np.ndarray, vs_g: np.ndarray) -> dict[str, np.ndarray]:
    gy, gx = np.gradient(tau)
    g = np.hypot(gx, gy)
    gs = np.maximum(g, 1e-9)
    return {
        "tau": tau,
        "grad": g,
        "align": (us_g * gx + vs_g * gy) / gs,
        "torsion": (us_g * gy - vs_g * gx) / 25.0,
    }


def analysis_fields(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """All storm-scale fields of one analysis (9 km native blocks) -- H9 and C9 inputs."""
    f = {k: np.asarray(v, dtype=np.float64) for k, v in blocks.items()}
    f["td_dep"] = f["t2m"] - f["td2m"]
    f["shear_06"] = np.hypot(f["ushear_06"], f["vshear_06"])
    f["shear_01"] = np.hypot(f["ushear_01"], f["vshear_01"])
    f["storm_speed"] = np.hypot(f["ustorm"], f["vstorm"])
    lcl = 125.0 * f["td_dep"]
    cape = np.maximum(f["mlcape"], 0.0)
    lcl_t = np.clip((2000.0 - lcl) / 1000.0, 0.0, 1.0)
    sh_t = np.where(f["shear_06"] < 12.5, 0.0, np.clip(f["shear_06"] / 20.0, 0.0, 1.5))
    cin_t = np.clip((200.0 + f["mlcin"]) / 150.0, 0.0, 1.0)
    f["stp"] = (cape / 1500.0) * lcl_t * (np.maximum(f["srh_01"], 0.0) / 150.0) * sh_t * cin_t

    out: dict[str, np.ndarray] = {}
    for v in H9_VARS:
        out[f"{v}_at"] = f[v]
        out[f"{v}_max27"] = _window_stat(f[v], 1, "max")
        out[f"{v}_mean27"] = _window_stat(f[v], 1, "mean")
        out[f"{v}_max63"] = _window_stat(f[v], 3, "max")

    # Coherence framework at storm scale (same S, Gamma, D as the 80 km engine).
    S = (np.maximum(f["mlcape"], 0) / 2000.0 + 0.3 * np.abs(f["srh_01"]) / 200.0
         + 0.2 * f["shear_06"] / 25.0 + 0.1 * np.minimum(np.maximum(f["refc"], 0) / 40.0, 2.0))
    G = np.abs(f["mlcin"]) / 200.0 + 0.1 + 0.2 * np.maximum(f["td_dep"], 0) / 15.0 + 0.05
    D = 1.0 + 0.3 * f["storm_speed"] / 20.0
    out["S_over_Gamma"] = S / np.maximum(G, 0.01)
    us_g, vs_g = _earth_to_grid(f["ushear_01"], f["vshear_01"])
    med = float(np.median(G / D))
    for scale in C9_SCALES_KM:
        cells = scale / (3.0 * LCC_K)  # 9 km per cell
        # A field whose coherence length spans >= 9 cells is smooth at that
        # scale: solve it on a grid f x coarser (f = cells // 3) and refine --
        # PCG work drops ~f^3 (fewer unknowns AND fewer iterations), the field
        # it represents does not change. Diagnostics stay in per-9-km-cell units.
        f_c = max(1, int(cells) // 3)
        if f_c > 1:
            Sc, Gc, Dc = _coarsen(S, f_c), _coarsen(G, f_c), _coarsen(D, f_c)
            usc, vsc = _coarsen(us_g, f_c), _coarsen(vs_g, f_c)
            cc = cells / f_c
            tau_c, _info = solve_helmholtz_2d_certified(Sc, Gc, 1.0, D=cc * cc * med * Dc, tol=C9_TOL)
            diag = {k: _refine(v, f_c, S.shape) for k, v in _diagnostics(tau_c, usc, vsc).items()}
            # gradient and torsion scale with 1/spacing; alignment is the shear
            # projected on the gradient DIRECTION (m/s) and does not
            for k in ("grad", "torsion"):
                diag[k] = diag[k] / f_c
        else:
            alpha = cells * cells * med     # sets the median coherence length sqrt(alpha D / Gamma)
            tau, _info = solve_helmholtz_2d_certified(S, G, 1.0, D=alpha * D, tol=C9_TOL)
            diag = _diagnostics(tau, us_g, vs_g)
        for name, arr in diag.items():
            out[f"pde_{int(scale)}_{name}"] = arr
        taug = _gaussian_smooth(S / np.maximum(G, 0.01), cells)
        for name, arr in _diagnostics(taug, us_g, vs_g).items():
            out[f"gauss_{int(scale)}_{name}"] = arr
    out["pde27_max27"] = _window_stat(out["pde_27_tau"], 1, "max")
    # storm-relative shear geometry needs the storm's own motion: kept as vectors
    for k in ("ushear_01", "vshear_01", "ushear_06", "vshear_06"):
        out[k] = f[k]
    return {k: v.astype(np.float32) for k, v in out.items()}


def block_h9_c9(fields: dict[str, np.ndarray], storm: dict) -> np.ndarray:
    """H9 + C9 features of one storm from its analysis' storm-scale fields."""
    r, c = H.lcc_block_of_latlon(float(storm.get("lat", 0.0)), float(storm.get("lon", 0.0)), LCC_K)
    ny, nx = fields["mlcape_at"].shape
    r, c = int(min(max(r, 0), ny - 1)), int(min(max(c, 0), nx - 1))
    vals = [float(fields[f"{v}_{s}"][r, c]) for v in H9_VARS for s in H9_STATS]
    ue, vn = storm_motion(storm)
    sp = math.hypot(ue, vn)
    u1, v1 = float(fields["ushear_01"][r, c]), float(fields["vshear_01"][r, c])
    u6, v6 = float(fields["ushear_06"][r, c]), float(fields["vshear_06"][r, c])
    if sp > 0.5:
        streamwise = (u1 * ue + v1 * vn) / sp
        ang6 = math.degrees(math.acos(max(-1.0, min(1.0, (u6 * ue + v6 * vn) / (sp * max(math.hypot(u6, v6), 1e-6))))))
    else:
        streamwise, ang6 = np.nan, np.nan
    vals += [streamwise, ang6]
    for kind in ("pde", "gauss"):
        for s in C9_SCALES_KM:
            vals += [float(fields[f"{kind}_{int(s)}_{d}"][r, c]) for d in C9_DIAG]
    vals += [float(fields["S_over_Gamma"][r, c]), float(fields["pde27_max27"][r, c])]
    return np.asarray(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"p_{k}" for k in PS_ATTRS)
    + tuple(f"e_{k}" for k in TRACK_NAMES)
    + tuple(f"h80_{k}" for k in dm.BLOCK_H_NAMES)
    + tuple(f"c80_{k}" for k in dm.BLOCK_C_NAMES)
    + H9_NAMES
    + C9_NAMES
)
BLOCKS: dict[str, tuple[int, int]] = {}
_off = 0
for _name, _n in (("P", len(PS_ATTRS)), ("E", len(TRACK_NAMES)), ("H80", len(dm.BLOCK_H_NAMES)),
                  ("C80", len(dm.BLOCK_C_NAMES)), ("H9", len(H9_NAMES)), ("C9", len(C9_NAMES))):
    BLOCKS[_name] = (_off, _off + _n)
    _off += _n
N_FEATURES = _off
assert N_FEATURES == len(FEATURE_NAMES)


def feature_vector(storm, history, step_minutes, h80=None, c80=None, fields9=None) -> np.ndarray:
    """One storm observation's full feature vector (NaN where an input is missing)."""
    parts = [block_p(storm), block_e(storm, history, step_minutes)]
    if h80 is not None:
        grids, derived = h80
        parts.append(dm.extract_block_h(storm, grids, derived).astype(np.float32))
    else:
        parts.append(np.full(len(dm.BLOCK_H_NAMES), np.nan, np.float32))
    if c80 is not None and h80 is not None:
        parts.append(dm.extract_block_c(storm, c80, h80[0]).astype(np.float32))
    else:
        parts.append(np.full(len(dm.BLOCK_C_NAMES), np.nan, np.float32))
    if fields9 is not None:
        parts.append(block_h9_c9(fields9, storm))
    else:
        parts.append(np.full(len(H9_NAMES) + len(C9_NAMES), np.nan, np.float32))
    v = np.concatenate(parts)
    assert v.shape[0] == N_FEATURES
    return v


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
HORIZONS_MIN: tuple[int, ...] = (30, 60, 90)
# Every candidate attribution of docs/TORNADO_MODEL_PROGRAM.md amendments 1-2, kept in the store so
# the choice can be audited; "storm_*" aliases the family chosen by the declared rule.
LABEL_FAMILIES: tuple[str, ...] = ("centroid", "poly5", "poly10", "track5", "track10", "nbhd")
# Chosen 2026-10-02 by the amendment rule on 2021 (results/lab/label_attribution_2021-01-01_2021-12-31.json):
# report coverage 0.848 at ambiguity 1.21 vs centroid 0.629, advected polygon 10 km 0.797, nbhd 0.903 at 2.43.
PRIMARY_FAMILY = "track10"
LABEL_NAMES: tuple[str, ...] = (
    tuple(f"storm_{h}" for h in HORIZONS_MIN)
    + tuple(f"{fam}_{h}" for fam in LABEL_FAMILIES for h in HORIZONS_MIN)
)
NBHD_RADIUS_KM = dm.LABEL_RADIUS_KM
POLY_BUFFERS_KM = {"poly5": 5.0, "poly10": 10.0}
TRACK_BUFFERS_KM = {"track5": 5.0, "track10": 10.0}
TRACK_MATCH_S = 900.0
LABEL_REACH_KM = 150.0   # centroid-to-report distance (plus advection) beyond which no family can match


def storm_radius_km(storm: dict) -> float:
    size = float(storm.get("size") or 0.0)
    return float(min(max(math.sqrt(max(size, 0.0) / math.pi) + 5.0, 8.0), 25.0))


def polygon_ring_km(storm: dict) -> np.ndarray | None:
    """The object's outer ring as (n, 2) east/north km from its centroid (local
    equirectangular), closed; None when the observation has no usable polygon."""
    geom = storm.get("geometry") or {}
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    if not coords:
        return None
    ring = coords[0] if geom.get("type") == "Polygon" else (coords[0][0] if geom.get("type") == "MultiPolygon" else None)
    if ring is None or len(ring) < 3:
        return None
    a = np.asarray(ring, dtype=np.float64)
    lat0, lon0 = float(storm.get("lat", 0.0)), float(storm.get("lon", 0.0))
    x = (a[:, 0] - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (a[:, 1] - lat0) * 111.32
    xy = np.column_stack([x, y])
    if not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    return xy


def point_polygon_distance_km(ring: np.ndarray, px: float, py: float) -> float:
    """0 inside the closed ring, else the distance to its nearest edge (same km frame)."""
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    crosses = ((y0 > py) != (y1 > py)) & (px < (x1 - x0) * (py - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    if np.count_nonzero(crosses) % 2 == 1:
        return 0.0
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / np.where(L2 == 0, 1.0, L2), 0.0, 1.0)
    return float(np.sqrt(np.min((x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2)))


def advected_polygon_distance_km(storm: dict, ring: np.ndarray, rep_lat: float, rep_lon: float,
                                 dt_s: float) -> float:
    """Distance from a report to the storm's polygon advected by its own motion for dt_s seconds."""
    ue, vn = storm_motion(storm)
    lat0, lon0 = float(storm.get("lat", 0.0)), float(storm.get("lon", 0.0))
    px = (rep_lon - lon0) * 111.32 * math.cos(math.radians(lat0)) - ue * dt_s / 1000.0
    py = (rep_lat - lat0) * 111.32 - vn * dt_s / 1000.0
    return point_polygon_distance_km(ring, px, py)


def tracked_distance_km(track: list[tuple[float, dict]], rep_lat: float, rep_lon: float, t_rep: float) -> float:
    """Distance from a report to the polygon of the storm's OWN track at the report time: the
    slot of the same id nearest t_rep (within TRACK_MATCH_S), else its last slot before t_rep,
    advected by the residual time. ``track`` = [(valid time, storm)] of one id, any order."""
    if not track:
        return float("inf")
    near = min(track, key=lambda e: abs(e[0] - t_rep))
    if abs(near[0] - t_rep) > TRACK_MATCH_S:
        before = [e for e in track if e[0] <= t_rep]
        if not before:
            return float("inf")
        near = max(before, key=lambda e: e[0])
    ring = polygon_ring_km(near[1])
    if ring is None:
        return float("inf")
    return advected_polygon_distance_km(near[1], ring, rep_lat, rep_lon, t_rep - near[0])


def labels(storm: dict, t_storm: float, reports: list[dict],
           track: list[tuple[float, dict]] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All label families for one storm observation.

    Returns (labels[LABEL_NAMES], ef[LABEL_FAMILIES], lead[LABEL_FAMILIES]): per family, the max EF
    of the reports it matches within 60 min (-1 none) and the minutes to its first match within the
    longest horizon (nan none). ``track``: the same storm id's [(valid time, storm)] (labels may use
    the future); None means only this observation is known, i.e. the track-ended fallback.
    """
    nf, nh = len(LABEL_FAMILIES), len(HORIZONS_MIN)
    fam_lab = np.zeros((nf, nh), dtype=np.int8)
    ef = np.full(nf, -1.0)
    lead = np.full(nf, np.nan)
    lat0, lon0 = float(storm.get("lat", 0.0)), float(storm.get("lon", 0.0))
    ue, vn = storm_motion(storm)
    speed = math.hypot(ue, vn)
    R = storm_radius_km(storm)
    ring = None
    trk = track if track is not None else [(t_storm, storm)]
    for rep in reports:
        tr = rep.get("time_utc")
        if tr is None:
            continue
        dt_s = float(tr) - t_storm
        if not (0.0 <= dt_s <= 60.0 * HORIZONS_MIN[-1]):
            continue
        d_nb = dm.haversine_km(lat0, lon0, rep["slat"], rep["slon"])
        if d_nb > LABEL_REACH_KM + speed * dt_s / 1000.0:
            continue
        dmin = dt_s / 60.0
        # centroid advected by the storm's motion to the report time
        dn_km, de_km = vn * dt_s / 1000.0, ue * dt_s / 1000.0
        lat_a = lat0 + dn_km / 111.32
        lon_a = lon0 + de_km / (111.32 * max(math.cos(math.radians(lat0)), 1e-3))
        hit = {"centroid": dm.haversine_km(lat_a, lon_a, rep["slat"], rep["slon"]) <= R,
               "nbhd": d_nb <= NBHD_RADIUS_KM}
        if ring is None:
            ring = polygon_ring_km(storm)
            ring = False if ring is None else ring
        d_poly = (advected_polygon_distance_km(storm, ring, rep["slat"], rep["slon"], dt_s)
                  if ring is not False else float("inf"))
        for fam, buf in POLY_BUFFERS_KM.items():
            hit[fam] = d_poly <= buf
        d_trk = tracked_distance_km(trk, rep["slat"], rep["slon"], float(tr))
        for fam, buf in TRACK_BUFFERS_KM.items():
            hit[fam] = d_trk <= buf
        for i, fam in enumerate(LABEL_FAMILIES):
            if not hit[fam]:
                continue
            for j, h in enumerate(HORIZONS_MIN):
                if dmin <= h:
                    fam_lab[i, j] = 1
            if dmin <= 60:
                ef[i] = max(ef[i], float(rep.get("mag", -1)))
            lead[i] = dmin if not np.isfinite(lead[i]) else min(lead[i], dmin)
    primary = fam_lab[LABEL_FAMILIES.index(PRIMARY_FAMILY)]
    return np.concatenate([primary, fam_lab.reshape(-1)]), ef, lead
