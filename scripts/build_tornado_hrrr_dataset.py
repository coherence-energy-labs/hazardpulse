#!/usr/bin/env python3
"""Self-contained HRRR-environment tornado dataset (NO ProbSevere dependency).

The question this benchmark asks (default ``--negatives hard``):

    Among STORM cells at the HRRR analysis time -- a convective core in a
    supercell-capable environment (tornado.hrrr_env.storm_population_mask) --
    which ones are tornadic?

  population : every 80 km cell on the day that passes the ONE eligibility gate
               (refc >= 40 dBZ, 0-6 km shear >= 12.5 m/s, CAPE >= 100 J/kg). The
               same gate is applied to positives AND negatives (and to serving), so
               no class can be identified by a field the other class was filtered on.
  positives  : eligible cells holding a tornado report whose start time is within
               LABEL_WINDOW_H of the analysis valid time (absolute UTC instants), the
               report position carried back to the analysis time along the HRRR
               0-6 km storm-motion vector at the report's cell.
  negatives  : ALL other eligible cells that day, minus a guard band: any cell within
               GUARD_RADIUS_BLOCKS (Chebyshev) of any tornado report (raw or carried
               back position) within +-GUARD_WINDOW_H of the analysis, and of any
               report whose clock is unknown filed on that day.
  geolocation: hrrr geometry g3 grids (each cell pooled from the native points inside
               it), reports placed with tornado.hrrr_env.locate_cell (= the cell
               hrrr.latlon_to_hrrr_cell names; off-grid reports refused, not clamped).
               The builder refuses to run on any other grid geometry.
  features   : the 26-feature vector (17 raw + 9 derived) on SANITIZED grids
               (feature spec v2).

``--negatives legacy-easy`` reproduces the historical dataset exactly (the one behind
the published 0.811 / 0.850 AUC, commit c89fb866): 40 random cells per day with
mlcape >= 250 (no storm required), positives from every report on the CST calendar
day regardless of hour, unsanitized grids, read from the pre-2026-10-01 cache files
(``{date}_{hh}z.npz``: pooled in native index space and read as lat/lon, so every
cell sits 185-375 km from its label over tornado country). It measures mostly
storm-vs-no-storm, selection asymmetry and mislocation; it exists ONLY to reproduce
that number and must never be the default.

    python scripts/build_tornado_hrrr_dataset.py --start 20220401 --end 20240831 \
        --hour 20 --cached-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Peak-pool the 80 km cells (capture the convective maximum) BEFORE importing hrrr.
os.environ.setdefault("HAZARDPULSE_HRRR_POOL", "max")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data.hrrr import (  # noqa: E402
    CACHE_ROOT,
    HRRR_VARS,
    HRRR_N_LAT,
    HRRR_N_LON,
    fetch_hrrr_grid,
    load_cached_hrrr,
    latlon_to_hrrr_cell,
)
from hazardpulse.tornado import hrrr_env  # noqa: E402
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr  # noqa: E402
from hazardpulse.tornado.definitive_model import load_spc_tornado_reports  # noqa: E402

# Raw HRRR vars + the canonical derived tornado discriminators (bulk shear, LCL
# height, 0-500 m SRH, Significant Tornado Parameter, streamwise vorticity). STP/SRH/
# shear are what actually separate tornadic from non-tornadic storm environments.
_VAR_NAMES = list(HRRR_VARS)
_DERIVED_NAMES = [
    "shear_01", "shear_06", "storm_speed", "td_depression", "lcl_est",
    "srh_05_est", "stp_eff", "rfd_warmth", "streamwise_vort",
]
_FEATURE_NAMES = _VAR_NAMES + _DERIVED_NAMES
assert _FEATURE_NAMES == hrrr_env.FEATURE_NAMES, "builder and serving feature order diverged"
_SPC_CSV = REPO / ".cache" / "spc" / "1950-2024_actual_tornadoes.csv"
_OUT = REPO / ".cache" / "tornado" / "hrrr_env_dataset.npz"

NEGATIVE_MODES = ("hard", "legacy-easy")
DEFAULT_NEGATIVES = "hard"

# A report labels the analysis when its start time is within [-2 h, +2 h) of the valid
# time. MEASURED (259 days 2022-2024, geometry g3 grids, carried back along the storm
# motion): the carried-back cell holds >= 40 dBZ for 91 / 98 / 95 / 80 % of reports in
# the hourly bins -2..-1, -1..0, 0..1, 1..2 h, falling to 63 % at 2..3 h and 42 % at
# 3..4 h: past +2 h the parent storm is increasingly not yet in the analysis. Chosen on
# that visibility measurement, not on any model score.
LABEL_WINDOW_H = (-2.0, 2.0)
# Negatives keep clear of every report within +-12 h (the surrounding convective day)
# by one block in every direction: a storm 80 km from a tornado, or that produces one
# 8 h later, is not a clean null.
GUARD_WINDOW_H = 12.0
GUARD_RADIUS_BLOCKS = 1

# Legacy-easy constants, frozen at their historical values.
LEGACY_NEG_PER_DAY = 40
LEGACY_NEG_MIN_CAPE = 250.0


SUPPORTED_GEOMETRY = "g3"


def _hrrr_geometry() -> str:
    """hrrr's grid geometry version; the hard benchmark refuses anything but g3 (cells
    that ARE the lat/lon cells hrrr_env.locate_cell names)."""
    from hazardpulse.data import hrrr as _h
    version = getattr(_h, "GEOMETRY_VERSION", None)
    if version != SUPPORTED_GEOMETRY:
        raise SystemExit(f"refusing: hazardpulse.data.hrrr geometry {version!r} is not "
                         f"{SUPPORTED_GEOMETRY!r}; its cells are not where locate_cell puts them")
    return version


def old_geometry_cache_path(date: str, hour: int, cache_dir: Path | None = None) -> Path:
    """A pre-2026-10-01 cache file: pooled in native INDEX space (geometry g1).

    Looked up in ``cache_dir``, else the active HRRR cache root, else the repo's own
    ``.cache/hrrr`` (where every g1 file was written).
    """
    name = f"{date}_{hour:02d}z.npz"
    for root in ([Path(cache_dir)] if cache_dir else [Path(CACHE_ROOT), REPO / ".cache" / "hrrr"]):
        if (root / name).exists():
            return root / name
    return Path(cache_dir or CACHE_ROOT) / name


def load_old_geometry_cache(date: str, hour: int, cache_dir: Path | None = None) -> dict | None:
    """Read a pre-2026-10-01 cache file as-is (historical reproduction / baseline only).

    hazardpulse.data.hrrr no longer reads these: cell (i, j) holds native rows
    31i..31i+30, cols 28j..28j+27, which is NOT the lat/lon cell (i, j).
    """
    path = old_geometry_cache_path(date, hour, cache_dir)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _date_range(start: str, end: str) -> list[str]:
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


def _feature_vector(grids: dict, derived: dict, i: int, j: int) -> np.ndarray:
    raw = [float(grids[v][i, j]) for v in _VAR_NAMES]
    der = [float(derived[v][i, j]) for v in _DERIVED_NAMES]
    return np.array(raw + der, dtype=np.float32)


# ---------------------------------------------------------------------------
# legacy-easy (historical reproduction ONLY)
# ---------------------------------------------------------------------------

def _cells_for(grids: dict, reports: list[dict], neg_per_day: int,
               neg_min_cape: float, rng: np.random.RandomState):
    """LEGACY-EASY day labeling (historical reproduction only -- see module doc).

    Positives: the lat/lon-box cell of every report given (no time filter). Negatives:
    ``neg_per_day`` random non-positive cells with cape >= ``neg_min_cape`` -- no storm
    required, and positives are not held to the cape floor (selection asymmetry).
    """
    try:
        derived = compute_derived_hrrr(grids)
    except Exception:
        # a missing raw var would break derived; skip the day rather than emit garbage
        return [], []
    pos_cells = set()
    for r in reports:
        try:
            i, j = latlon_to_hrrr_cell(float(r["slat"]), float(r["slon"]))
            pos_cells.add((i, j))
        except Exception:
            continue
    cape = grids.get("mlcape")
    if cape is None:
        cape = grids.get("cape")
    rows, labels = [], []
    for (i, j) in pos_cells:
        rows.append(_feature_vector(grids, derived, i, j))
        labels.append(1)
    cand = [
        (i, j)
        for i in range(HRRR_N_LAT) for j in range(HRRR_N_LON)
        if (i, j) not in pos_cells
        and np.isfinite(cape[i, j]) and cape[i, j] >= neg_min_cape
    ]
    if cand:
        k = min(neg_per_day, len(cand))
        for idx in rng.choice(len(cand), k, replace=False):
            i, j = cand[idx]
            rows.append(_feature_vector(grids, derived, i, j))
            labels.append(0)
    return rows, labels


def reports_by_local_date(reports_by_utc: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Re-key the loader's UTC-date index by the SPC row's own CST date.

    The historical dataset keyed reports by the CST calendar date (the loader did,
    before it moved to UTC dates); legacy-easy needs that keying to reproduce it.
    """
    out: dict[str, list[dict]] = {}
    for reps in reports_by_utc.values():
        for r in reps:
            out.setdefault(str(r.get("local_date")), []).append(r)
    return out


# ---------------------------------------------------------------------------
# hard (default): tornadic vs non-tornadic STORM
# ---------------------------------------------------------------------------

def analysis_epoch(date: str, hour: int) -> float:
    """POSIX seconds of the HRRR analysis valid time ``date`` at ``hour`` UTC."""
    t = dt.datetime.strptime(date, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return (t + dt.timedelta(hours=int(hour))).timestamp()


def reports_near_analysis(reports_by_utc: dict[str, list[dict]], date: str, hour: int,
                          max_abs_h: float) -> tuple[list[tuple[dict, float]], list[dict]]:
    """Reports within ``max_abs_h`` of the analysis, as ``(report, offset_h)``.

    Compared on absolute UTC instants (``time_utc``), never on calendar dates. Also
    returns the reports with an unknown clock filed under ``date`` (guard-only).
    """
    t0 = analysis_epoch(date, hour)
    day = dt.datetime.strptime(date, "%Y%m%d")
    keys = [(day + dt.timedelta(days=k)).strftime("%Y%m%d") for k in (-1, 0, 1)]
    timed, unknown = [], []
    for k in keys:
        for r in reports_by_utc.get(k, []):
            tu = r.get("time_utc")
            if tu is None:
                if k == date or r.get("local_date") == date:
                    unknown.append(r)
                continue
            off = (float(tu) - t0) / 3600.0
            if abs(off) <= max_abs_h:
                timed.append((r, off))
    return timed, unknown


def report_blocks(report: dict, offset_h: float, grids: dict, *,
                  locate=None, carry_back: bool = True):
    """``(raw_cell, carried_cell)`` of a report at the analysis time.

    raw_cell: the grid cell holding the report's start point. carried_cell: that point
    moved back by ``offset_h`` hours along the HRRR 0-6 km storm motion read at
    raw_cell (where the parent storm was at the analysis time); equals raw_cell when
    the storm motion is missing or carries off the grid. ``(None, None)`` off the grid.

    ``locate`` defaults to hrrr_env.locate_cell (the hrrr grid's own locator); another
    locator and ``carry_back=False`` exist only for the benchmark audit.
    """
    locate = locate or hrrr_env.locate_cell
    lat, lon = float(report["slat"]), float(report["slon"])
    raw = locate(lat, lon)
    if raw is None:
        return None, None
    if not carry_back:
        return raw, raw
    us = float(grids["ustorm"][raw])
    vs = float(grids["vstorm"][raw])
    if not (np.isfinite(us) and np.isfinite(vs)):
        return raw, raw
    moved = hrrr_env.carry_latlon(lat, lon, us, vs, offset_h)
    carried = locate(*moved) if moved is not None else None
    return raw, (carried if carried is not None else raw)


def _dilate(blocks, radius: int) -> set:
    out = set()
    for (i, j) in blocks:
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < HRRR_N_LAT and 0 <= jj < HRRR_N_LON:
                    out.add((ii, jj))
    return out


class DayLabels:
    """Rows, labels, (i, j) cells and bookkeeping counts for one analysis.

    A plain class, not a dataclass: this script is also loaded with
    importlib.util.spec_from_file_location (tests, the audit), which does not register
    the module in sys.modules -- and dataclass processing requires that.
    """

    def __init__(self) -> None:
        self.rows: list = []
        self.labels: list = []
        self.cells: list = []
        self.counts: dict = {}


def label_day_hard(grids_raw: dict, reports_by_utc: dict[str, list[dict]], date: str, hour: int,
                   *, label_window_h: tuple[float, float] = LABEL_WINDOW_H,
                   guard_window_h: float = GUARD_WINDOW_H,
                   guard_radius: int = GUARD_RADIUS_BLOCKS,
                   locate=None, carry_back: bool = True,
                   population_kw: dict | None = None) -> DayLabels:
    """Hard-benchmark rows for one analysis: eligible storm cells, tornadic or not.

    ``locate`` (a locator for a non-g3 grid), ``carry_back`` and ``population_kw``
    (threshold overrides for hrrr_env.storm_population_mask) are audit and sensitivity
    knobs; the defaults are the benchmark.
    """
    locate = locate or hrrr_env.locate_cell
    out = DayLabels()
    grids = hrrr_env.sanitize_grids(grids_raw)
    try:
        derived = compute_derived_hrrr(grids)
    except Exception:
        out.counts = {"skipped": 1}
        return out
    eligible = hrrr_env.storm_population_mask(grids, derived, **(population_kw or {}))
    lo, hi = label_window_h
    timed, unknown = reports_near_analysis(reports_by_utc, date, hour,
                                           max(guard_window_h, abs(lo), abs(hi)))
    pos_blocks, guard_seed = set(), set()
    n_window = n_outside_domain = 0
    for r, off in timed:
        raw, carried = report_blocks(r, off, grids, locate=locate, carry_back=carry_back)
        if raw is None:
            n_outside_domain += 1
            continue
        if abs(off) <= guard_window_h:
            guard_seed.update((raw, carried))
        if lo <= off < hi:
            n_window += 1
            pos_blocks.add(carried)
    for r in unknown:
        b = locate(float(r["slat"]), float(r["slon"]))
        if b is not None:
            guard_seed.add(b)
    guard = _dilate(guard_seed, guard_radius)

    pos = sorted(b for b in pos_blocks if eligible[b])
    neg = sorted((i, j) for i, j in zip(*np.nonzero(eligible))
                 if (i, j) not in guard and (i, j) not in pos_blocks)
    for (i, j) in pos:
        out.rows.append(hrrr_env.cell_features(grids, derived, i, j))
        out.labels.append(1)
        out.cells.append((i, j))
    for (i, j) in neg:
        out.rows.append(hrrr_env.cell_features(grids, derived, int(i), int(j)))
        out.labels.append(0)
        out.cells.append((int(i), int(j)))
    out.counts = {
        "reports_in_window": n_window,
        "reports_outside_domain": n_outside_domain,
        "reports_unknown_time": len(unknown),
        "positive_blocks": len(pos_blocks),
        "positive_blocks_eligible": len(pos),
        "positive_blocks_outside_population": len(pos_blocks) - len(pos),
        "eligible_blocks": int(eligible.sum()),
        "eligible_blocks_guarded": int(sum(1 for b in zip(*np.nonzero(eligible))
                                           if tuple(int(v) for v in b) in guard)),
        "negatives": len(neg),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hour", type=int, default=20, help="HRRR analysis hour (Z)")
    ap.add_argument("--negatives", choices=NEGATIVE_MODES, default=DEFAULT_NEGATIVES,
                    help="hard (default): eligible storm cells, tornadic vs not. "
                         "legacy-easy: reproduce the historical easy-negative dataset ONLY")
    ap.add_argument("--neg-per-day", type=int, default=LEGACY_NEG_PER_DAY,
                    help="legacy-easy only")
    ap.add_argument("--neg-min-cape", type=float, default=LEGACY_NEG_MIN_CAPE,
                    help="legacy-easy only (J/kg)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel HRRR fetches (the public archive drops chunks under load)")
    ap.add_argument("--max-dates", type=int, default=0, help="0 = no cap")
    ap.add_argument("--tornado-days-only", action="store_true",
                    help="only fetch days with >=1 tornado (cheaper, balanced)")
    ap.add_argument("--cached-only", action="store_true",
                    help="use only already-cached HRRR dates (no network) -- instant rebuild")
    ap.add_argument("--dates-file", default="",
                    help="restrict to the YYYYMMDD dates listed (one per line) within the range")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args(argv)

    reports_utc = load_spc_tornado_reports(_SPC_CSV)
    legacy = args.negatives == "legacy-easy"
    # legacy-easy reproduces a number that was built from the pre-g2 files; nothing else
    # may read them (their cells are not where hrrr.latlon_to_hrrr_cell says)
    old_files = legacy
    geometry = "g1-index-pooled(legacy)" if legacy else f"hrrr-{_hrrr_geometry()}"
    if legacy and (args.neg_per_day, args.neg_min_cape) != (LEGACY_NEG_PER_DAY, LEGACY_NEG_MIN_CAPE):
        print("  note: legacy-easy with non-historical knobs does not reproduce the published set")
    if not legacy and (args.neg_per_day, args.neg_min_cape) != (LEGACY_NEG_PER_DAY, LEGACY_NEG_MIN_CAPE):
        ap.error("--neg-per-day / --neg-min-cape apply only to --negatives legacy-easy")
    day_reports = reports_by_local_date(reports_utc) if legacy else reports_utc

    dates = _date_range(args.start, args.end)
    if args.dates_file:
        keep = {ln.strip() for ln in Path(args.dates_file).read_text(encoding="utf-8").splitlines()
                if ln.strip()}
        dates = [d for d in dates if d in keep]
    if args.tornado_days_only:
        dates = [d for d in dates if day_reports.get(d)]
    if args.max_dates:
        dates = dates[: args.max_dates]
    print(f"{len(dates)} candidate dates ({args.start}..{args.end}), "
          f"{sum(1 for d in dates if day_reports.get(d))} with tornadoes  "
          f"[negatives={args.negatives}]")

    # Gather HRRR grids: cached-only (instant, no network) or parallel fetch.
    fetched: dict[str, dict] = {}
    if old_files:
        for d in dates:
            g = load_old_geometry_cache(d, args.hour)
            if g is not None:
                fetched[d] = g
        print(f"HRRR old-geometry cache files for {len(fetched)}/{len(dates)} dates "
              f"(no network; geometry={geometry})")
    elif args.cached_only:
        for d in dates:
            g = load_cached_hrrr(d, args.hour)
            if g is not None:
                fetched[d] = g
        print(f"HRRR cached for {len(fetched)}/{len(dates)} dates (no network; geometry={geometry})")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_hrrr_grid, d, args.hour): d for d in dates}
            for fut in as_completed(futs):
                d = futs[fut]
                done += 1
                try:
                    g = fut.result()
                except Exception as exc:
                    print(f"  {d}: fetch error {exc}")
                    g = None
                if g is not None:
                    fetched[d] = g
                if done % 25 == 0:
                    print(f"  fetched {done}/{len(dates)} ({len(fetched)} ok)")
        print(f"HRRR available for {len(fetched)}/{len(dates)} dates")

    X_rows, y_rows, date_rows, cell_rows = [], [], [], []
    totals: dict[str, int] = {}
    for d in sorted(fetched):
        if legacy:
            rng = np.random.RandomState(int(d))   # deterministic per-day negatives
            rows, labels = _cells_for(fetched[d], day_reports.get(d, []),
                                      args.neg_per_day, args.neg_min_cape, rng)
            cells = [(-1, -1)] * len(rows)
        else:
            lab = label_day_hard(fetched[d], reports_utc, d, args.hour)
            rows, labels, cells = lab.rows, lab.labels, lab.cells
            for k, v in lab.counts.items():
                totals[k] = totals.get(k, 0) + int(v)
        X_rows.extend(rows)
        y_rows.extend(labels)
        date_rows.extend([int(d)] * len(rows))
        cell_rows.extend(cells)

    X = np.asarray(X_rows, dtype=np.float32).reshape(-1, len(_FEATURE_NAMES))
    y = np.asarray(y_rows, dtype=np.int8)
    dates_arr = np.asarray(date_rows, dtype=np.int64)
    meta = {
        "negatives": args.negatives,
        "hour": args.hour,
        "feature_spec": hrrr_env.FEATURE_SPEC_V1 if legacy else hrrr_env.FEATURE_SPEC_V2,
        "geometry": geometry,
        "registration": ("latlon_to_hrrr_cell on index-pooled grids(legacy, mislocated)"
                         if legacy else "hrrr_env.locate_cell"),
        "report_keying": "cst_calendar_day(legacy)" if legacy else "utc_instant_window",
    }
    if not legacy:
        meta.update({
            "label_window_h": list(LABEL_WINDOW_H), "guard_window_h": GUARD_WINDOW_H,
            "guard_radius_blocks": GUARD_RADIUS_BLOCKS,
            "refc_min_dbz": hrrr_env.REFC_MIN_DBZ, "shear06_min_ms": hrrr_env.SHEAR06_MIN_MS,
            "cape_min_jkg": hrrr_env.CAPE_MIN_JKG, "counts": totals,
        })
    else:
        meta.update({"neg_per_day": args.neg_per_day, "neg_min_cape": args.neg_min_cape})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, dates=dates_arr,
                        cells=np.asarray(cell_rows, dtype=np.int16).reshape(-1, 2),
                        feature_names=np.array(_FEATURE_NAMES),
                        feature_spec=np.array(meta["feature_spec"]),
                        meta=np.array(json.dumps(meta, sort_keys=True)))
    n_pos = int(y.sum())
    kind = "convective-null" if legacy else "eligible non-tornadic storm"
    print(f"dataset: {len(y)} cells  ({n_pos} tornado, {len(y) - n_pos} {kind})  "
          f"features={X.shape[1]}  dates={len(set(date_rows))}")
    if totals:
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
