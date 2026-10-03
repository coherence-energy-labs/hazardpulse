"""The HRRR tornado benchmark asks tornadic-vs-non-tornadic STORM, by default.

Pins the roots found in the 2026-10-01 audit of the published 0.81-0.85 AUC:

* negatives were random CAPE>=250 cells (median refc -10 dBZ) while positives were
  not held to any floor -> storm-vs-no-storm plus a selection asymmetry;
* every report of the CST calendar day labelled a single 20z analysis (offsets
  -14 h .. +10 h), so 29-34 % of "tornadic" cells had no echo at all;
* the grid was pooled in native index space and read as lat/lon (fixed in
  hazardpulse.data.hrrr geometry g2/g3) -- reports must be located with the grid cell
  locator and never clamped into an edge cell.

Every test here can fail: each asserts a property the legacy configuration violates.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data import hrrr as H  # noqa: E402
from hazardpulse.tornado import hrrr_env as E  # noqa: E402

DATE, HOUR = "20240501", 20
T0 = dt.datetime(2024, 5, 1, HOUR, tzinfo=dt.timezone.utc).timestamp()
OKC = (35.5, -97.5)
OKC_CELL = (14, 29)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def builder():
    return _load("thd_hard", "build_tornado_hrrr_dataset.py")


def _centre(i, j):
    return float(H.GRID_LATS[i]), float(H.GRID_LONS[j])


def _report(lat, lon, offset_h, t0=T0):
    tu = t0 + offset_h * 3600.0
    t = dt.datetime.fromtimestamp(tu, tz=dt.timezone.utc)
    return {"slat": lat, "slon": lon, "time_utc": tu, "hour": t.hour + t.minute / 60.0,
            "mag": 1, "local_date": (t - dt.timedelta(hours=6)).strftime("%Y%m%d")}


def _by_utc_date(reports):
    out = {}
    for r in reports:
        k = dt.datetime.fromtimestamp(r["time_utc"], tz=dt.timezone.utc).strftime("%Y%m%d")
        out.setdefault(k, []).append(r)
    return out


def _grid():
    """Calm CONUS + three regions: an eligible storm band, a weak-shear storm, and a
    high-CAPE clear-air patch (eligible for the legacy negative rule, not a storm)."""
    ny, nx = H.HRRR_N_LAT, H.HRRR_N_LON
    g = {v: np.zeros((ny, nx), np.float32) for v in H.HRRR_VARS}
    g["refc"][:] = -10.0
    g["t2m"][:] = 300.0
    g["td2m"][:] = 293.0
    g["mlcin"][:] = -25.0
    g["cin"][:] = -25.0
    g["pwat"][:] = 30.0
    band = (slice(10, 17), slice(28, 37))          # contains OKC's cell (14, 29)
    for k, v in (("refc", 50.0), ("ushear_06", 20.0), ("ushear_01", 8.0), ("cape", 2500.0),
                 ("mlcape", 2000.0), ("mucape", 2600.0), ("srh_01", 200.0), ("srh_03", 300.0)):
        g[k][band] = v
    weak = (slice(20, 23), slice(10, 15))           # a storm, but 5 m/s of shear
    for k, v in (("refc", 50.0), ("ushear_06", 5.0), ("cape", 2000.0), ("mlcape", 1500.0),
                 ("mucape", 2000.0)):
        g[k][weak] = v
    clear = (slice(5, 8), slice(45, 51))            # unstable, sheared, NO echo
    for k, v in (("ushear_06", 20.0), ("cape", 3500.0), ("mlcape", 3000.0), ("mucape", 3500.0)):
        g[k][clear] = v
    return g


def _feature(rows, name):
    return np.array([r[E.FEATURE_NAMES.index(name)] for r in rows], float)


# ---------------------------------------------------------------------------
# locating a report on the grid
# ---------------------------------------------------------------------------

def test_native_projection_matches_an_independent_implementation():
    pyproj = pytest.importorskip("pyproj")
    P = pyproj.Proj("+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
                    "+R=6371229 +units=m +no_defs")
    rng = np.random.RandomState(0)
    lat, lon = rng.uniform(22, 50, 300), rng.uniform(-125, -66, 300)
    px, py = P(lon, lat)
    x0, y0 = P(-122.719528, 21.138123)
    row, col = H.native_index_of_latlon(lat, lon)
    assert np.max(np.abs(col - (px - x0) / 3000.0)) < 1e-6
    assert np.max(np.abs(row - (py - y0) / 3000.0)) < 1e-6


def test_report_lands_in_its_own_latlon_cell_and_off_grid_is_refused(builder):
    assert E.locate_cell(*OKC) == H.latlon_to_hrrr_cell(*OKC) == OKC_CELL
    raw, carried = builder.report_blocks({"slat": OKC[0], "slon": OKC[1]}, 0.0, _grid())
    assert raw == carried == OKC_CELL
    # off the grid: latlon_to_hrrr_cell clamps into an edge cell, a label must not
    assert H.latlon_to_hrrr_cell(52.0, -97.5)[0] == H.HRRR_N_LAT - 1
    assert E.locate_cell(52.0, -97.5) is None and E.locate_cell(24.0, -97.5) is None
    assert builder.report_blocks({"slat": 52.0, "slon": -97.5}, 0.0, _grid()) == (None, None)


def test_cell_centres_round_trip():
    dom = H.domain_mask()
    for (i, j) in [(0, 10), OKC_CELL, (33, 40), (20, 5), (5, 50), (16, 36)]:
        if dom[i, j]:
            assert E.locate_cell(*_centre(i, j)) == (i, j)


# ---------------------------------------------------------------------------
# sanitization
# ---------------------------------------------------------------------------

def test_sanitize_removes_fill_and_contaminated_means_only():
    g = _grid()
    g["t2m"][0, 0] = -10000.0           # archive fill
    g["t2m"][0, 1] = -3000.0            # mean-pooled cell straddling a missing chunk
    g["srh_01"][3, 3] = -10000.0
    s = E.sanitize_grids(g)
    assert np.isnan(s["t2m"][0, 0]) and np.isnan(s["t2m"][0, 1]) and np.isnan(s["srh_01"][3, 3])
    assert s["t2m"][5, 5] == 300.0 and s["refc"][OKC_CELL] == 50.0 and s["refc"][0, 0] == -10.0
    assert np.isfinite(g["t2m"][0, 0]), "sanitize must not mutate its input"


# ---------------------------------------------------------------------------
# the hard benchmark
# ---------------------------------------------------------------------------

def test_hard_negatives_satisfy_the_storm_population_gate(builder):
    reports = _by_utc_date([_report(*OKC, offset_h=0.5)])
    lab = builder.label_day_hard(_grid(), reports, DATE, HOUR)
    y = np.array(lab.labels)
    assert y.sum() == 1 and lab.cells[int(np.argmax(y))] == OKC_CELL
    neg = [r for r, lbl in zip(lab.rows, lab.labels) if lbl == 0]
    assert len(neg) > 0
    assert np.all(_feature(neg, "refc") >= E.REFC_MIN_DBZ)
    assert np.all(_feature(neg, "shear_06") >= E.SHEAR06_MIN_MS)
    cape_max = np.max(np.stack([_feature(neg, k) for k in ("cape", "mlcape", "mucape")]), axis=0)
    assert np.all(cape_max >= E.CAPE_MIN_JKG)
    # the clear-air patch and the weak-shear storm are never negatives
    cells = {c for c, lbl in zip(lab.cells, lab.labels) if lbl == 0}
    assert not any(5 <= i < 8 and 45 <= j < 51 for i, j in cells)
    assert not any(20 <= i < 23 and 10 <= j < 15 for i, j in cells)
    # guard band: no negative touches the tornado's cell
    oi, oj = OKC_CELL
    assert all(max(abs(i - oi), abs(j - oj)) > builder.GUARD_RADIUS_BLOCKS for i, j in cells)
    # every eligible, unguarded band cell is a negative (ALL of them, no per-day cap)
    band = {(i, j) for i in range(10, 17) for j in range(28, 37)}
    assert cells == {c for c in band if max(abs(c[0] - oi), abs(c[1] - oj)) > 1}


def test_positive_outside_the_population_is_dropped_not_kept(builder):
    # a tornado under the clear-air patch at the analysis time: not a storm cell
    lab = builder.label_day_hard(_grid(), _by_utc_date([_report(*_centre(6, 47), 0.0)]), DATE, HOUR)
    assert sum(lab.labels) == 0
    assert lab.counts["positive_blocks_outside_population"] == 1


def test_easy_negative_configuration_is_not_the_default(builder, tmp_path):
    assert builder.DEFAULT_NEGATIVES == "hard" and "legacy-easy" in builder.NEGATIVE_MODES
    # the CLI default really builds the hard benchmark (no data needed: empty range)
    out = tmp_path / "ds.npz"
    assert builder.main(["--start", "20990101", "--end", "20990101", "--cached-only",
                         "--out", str(out)]) == 0
    meta = json.loads(str(np.load(out, allow_pickle=True)["meta"]))
    assert meta["negatives"] == "hard" and meta["registration"] == "hrrr_env.locate_cell"
    assert meta["geometry"] == "hrrr-g3" and meta["feature_spec"] == E.FEATURE_SPEC_V2
    # and the legacy rule on the same day DOES emit no-storm negatives -- the thing the
    # hard mode exists to exclude
    rows, labels = builder._cells_for(_grid(), [], 40, 250.0, np.random.RandomState(0))
    neg_refc = _feature([r for r, lbl in zip(rows, labels) if lbl == 0], "refc")
    assert np.any(neg_refc < E.REFC_MIN_DBZ)


def test_hard_mode_refuses_a_grid_geometry_it_cannot_locate_in(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(H, "GEOMETRY_VERSION", "g1")
    with pytest.raises(SystemExit):
        builder.main(["--start", "20990101", "--end", "20990101", "--cached-only",
                      "--out", str(tmp_path / "never_written.npz")])
    assert not (tmp_path / "never_written.npz").exists()


def test_label_window_is_absolute_utc_time(builder):
    reps = _by_utc_date([_report(*OKC, 1.5),
                         _report(*_centre(10, 29), 3.0),     # outside the window, guarded
                         _report(*_centre(16, 36), 13.0)])   # outside window AND guard
    lab = builder.label_day_hard(_grid(), reps, DATE, HOUR)
    pos = {c for c, lbl in zip(lab.cells, lab.labels) if lbl == 1}
    neg = {c for c, lbl in zip(lab.cells, lab.labels) if lbl == 0}
    assert pos == {OKC_CELL}
    assert (10, 29) not in pos and (10, 29) not in neg and (11, 30) not in neg
    assert (16, 36) in neg
    # a report in the window but filed under the NEXT UTC day (analysis at 23z)
    t23 = dt.datetime(2024, 5, 1, 23, tzinfo=dt.timezone.utc).timestamp()
    reps = _by_utc_date([_report(*OKC, 1.5, t0=t23)])
    assert "20240502" in reps and "20240501" not in reps
    lab = builder.label_day_hard(_grid(), reps, DATE, 23)
    assert sum(lab.labels) == 1


def test_report_is_carried_back_along_storm_motion(builder):
    g = _grid()
    g["ustorm"][:] = 20.0                           # storms moving east at 20 m/s
    lat, lon = _centre(14, 31)
    raw, carried = builder.report_blocks({"slat": lat, "slon": lon}, 1.9, E.sanitize_grids(g))
    assert raw == (14, 31)
    assert carried[0] == 14 and carried[1] in (29, 30)   # ~137 km west = 1.6 cells
    raw, same = builder.report_blocks({"slat": lat, "slon": lon}, 1.9, g, carry_back=False)
    assert same == raw


# ---------------------------------------------------------------------------
# trainer statistics
# ---------------------------------------------------------------------------

def test_rank_auc_matches_reference_and_bootstrap_brackets_it():
    try:
        tr = _load("ttr_hard", "train_tornado_hrrr.py")
    except Exception as exc:  # pragma: no cover - optional training deps
        pytest.skip(f"trainer not importable: {exc}")
    rng = np.random.RandomState(4)
    y = (rng.rand(600) < 0.1).astype(int)
    s = np.round(rng.randn(600) + 1.2 * y, 1)        # rounded -> many ties
    # ground truth: P(score_pos > score_neg) + 0.5 P(tie), over all pairs
    sp, sn = s[y == 1][:, None], s[y == 0][None, :]
    brute = float(np.mean((sp > sn) + 0.5 * (sp == sn)))
    assert abs(tr.auc_rank(y, s) - brute) < 1e-12
    # the shared stepwise roc_auc agrees when there are no ties; with ties it depends on
    # the order argsort leaves tied scores in (it does not count a tie as 1/2)
    s_cont = s + rng.rand(600) * 1e-6
    assert abs(tr.auc_rank(y, s_cont) - tr.roc_auc(y, s_cont)) < 1e-12
    dates = np.repeat(np.arange(60), 10)
    b = tr.day_bootstrap(y, {"m": s}, dates, reps=300, seed=1)
    lo, hi = b["m"]["ci95"]
    assert lo < b["m"]["auc"] < hi and hi - lo < 0.25
    assert tr.day_bootstrap(y, {"m": s}, dates, reps=300, seed=1) == b       # deterministic


# ---------------------------------------------------------------------------
# real data anchor (skips without a local g3 cache)
# ---------------------------------------------------------------------------

def test_cached_reports_near_20z_sit_under_echo_in_their_located_cell(builder):
    if not builder._SPC_CSV.exists():
        pytest.skip("no local SPC report file")
    from hazardpulse.tornado.definitive_model import load_spc_tornado_reports
    reports = load_spc_tornado_reports(builder._SPC_CSV)
    if not any("time_utc" in r for rs in list(reports.values())[:50] for r in rs):
        pytest.skip("SPC loader without UTC instants")
    days = sorted(d for d in reports if "20220101" <= d <= "20241231")
    by, bx = H.NATIVE_NY // H.HRRR_N_LAT, H.NATIVE_NX // H.HRRR_N_LON
    hit_cell, hit_index_block = [], []
    for d in days:
        g = H.load_cached_hrrr(d, 20)                # the cache helper: current-geometry files only
        if g is None:
            continue
        timed, _ = builder.reports_near_analysis(reports, d, 20, 1.0)
        for r, _off in timed:
            cell = E.locate_cell(r["slat"], r["slon"])
            if cell is None:
                continue
            hit_cell.append(g["refc"][cell] >= 35.0)
            # control: the pre-g2 index-block reading of the same report
            row, col = H.native_index_of_latlon(r["slat"], r["slon"])
            blk = (min(int(round(float(row))) // by, H.HRRR_N_LAT - 1),
                   min(int(round(float(col))) // bx, H.HRRR_N_LON - 1))
            hit_index_block.append(g["refc"][blk] >= 35.0)
    if len(hit_cell) < 50:
        pytest.skip(f"only {len(hit_cell)} reports near 20z in the g3 cache")
    # MEASURED 2026-10-01 on 289 reports: 0.948 on g3 grids (pre-g2 files: 0.927 with their
    # own index-block locator, 0.436 read through the lat/lon box)
    assert np.mean(hit_cell) >= 0.85
    assert np.mean(hit_index_block) <= 0.65
