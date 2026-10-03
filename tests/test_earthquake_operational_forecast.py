"""The served operational earthquake forecast (src/hazardpulse/earthquake/operational_forecast.py).

Contract: docs/EARTHQUAKE_FORECAST_PROGRAM.md section 1. These tests pin the properties the
program's comparison relies on:

* causality -- an event at or after the issue time can never change a forecast, and the
  instrument that checks it can see a change (the same event one second earlier does);
* rate-model sanity -- probabilities in (0, 1), the long-term rate integrates to mu, the
  Omori window integral is the integral it claims to be;
* serving parity -- the frozen-catalog + live-tail path the scorer uses gives the same
  bits as the engine the evaluation used, and the committed artifact reproduces the
  forecasts the evaluation scored;
* identity -- the model_version is bound to the artifact's content, line endings aside.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.earthquake import operational_forecast as of
from hazardpulse.earthquake.coherence_engine import N_LON, latlon_to_grid_cell

REPO = Path(__file__).resolve().parents[1]
SERVED = REPO / "results" / "models" / "earthquake_operational_v1.json"
DAY = 86400.0

SPEC_B = of.ModelSpec(
    "B",
    of.LongTermParams(kernel_km=20.0, declustered=True, mu=10.0, eps=0.01),
    of.ShortTermParams(d0_km=10.0, K=0.02, alpha=1.0, c_days=0.01, p=1.1),
)
SPEC_A = of.ModelSpec("A", of.LongTermParams(kernel_km=35.0, declustered=False, mu=10.0, eps=0.02))


def _epoch(y, m=1, d=1, h=0):
    return dt.datetime(y, m, d, h, tzinfo=dt.timezone.utc).timestamp()


def _catalog(n=300, seed=7) -> of.EventSet:
    """Clustered synthetic M>=5 catalog with Gutenberg-Richter magnitudes, plus edge cases."""
    rng = np.random.default_rng(seed)
    centres = np.array([[38.0, 142.0], [-33.0, -72.0], [-20.0, -175.0], [36.0, 70.0]])
    which = rng.integers(0, len(centres), n)
    lat = centres[which, 0] + rng.normal(0, 1.5, n)
    lon = centres[which, 1] + rng.normal(0, 1.5, n)
    t = rng.uniform(_epoch(1990), _epoch(2020), n)
    mag = np.minimum(5.0 + rng.exponential(1.0 / np.log(10.0), n), 8.8)
    extra_t = [_epoch(2000, 6), _epoch(2001, 3), _epoch(2002, 2)]
    extra_lat = [75.0, -62.0, 0.0]          # north of the band, south of the band, equator
    extra_lon = [10.0, -60.0, 179.99]
    return of.EventSet.from_arrays(np.r_[t, extra_t], np.r_[lat, extra_lat], np.r_[lon, extra_lon],
                                   np.r_[mag, [6.1, 6.4, 5.5]])


def _dicts(ev: of.EventSet) -> list[dict]:
    out = []
    for t, la, lo, m in zip(ev.t, ev.lat, ev.lon, ev.mag):
        stamp = dt.datetime.fromtimestamp(float(t), dt.timezone.utc)
        out.append({"time": stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z",
                    "latitude": float(la), "longitude": float(lo), "mag": float(m)})
    return out


def _with(ev: of.EventSet, t, lat, lon, mag) -> of.EventSet:
    return ev.concat(of.EventSet.from_arrays([t], [lat], [lon], [mag]))


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def test_cell_index_is_the_site_function():
    rng = np.random.default_rng(0)
    lat = np.r_[rng.uniform(-90, 90, 5000), [-61.0, -60.0, -59.999, 70.0, 69.999, 90.0, -90.0]]
    lon = np.r_[rng.uniform(-180, 180, 5000), [180.0, -180.0, 0.0, 179.999, -0.0, 45.0, -45.0]]
    ref = np.array([r * N_LON + c for r, c in (latlon_to_grid_cell(a, b) for a, b in zip(lat, lon))])
    np.testing.assert_array_equal(of.cell_index(lat, lon), ref)


def test_cell_areas_cover_the_band():
    band = 2 * np.pi * (np.sin(np.radians(70.0)) - np.sin(np.radians(-60.0)))
    assert of.cell_areas().sum() == pytest.approx(band, rel=1e-12)
    assert of.cell_areas().size == of.N_CELLS == 11700


# ---------------------------------------------------------------------------
# Kernel, Omori, targets
# ---------------------------------------------------------------------------

def test_kernel_rows_are_normalised_and_local():
    lat = np.array([38.3, -20.1, 0.5, 75.0, 89.0])
    lon = np.array([142.4, -175.2, 179.9, 10.0, 0.0])
    G = of.kernel_matrix(lat, lon, np.array([10.0, 10.0, 10.0, 10.0, 10.0]))
    sums = np.asarray(G.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums[:3], 1.0, rtol=1e-12)
    assert sums[4] == 0.0                     # 89N: no domain sub-point within 300 km
    assert sums[3] in (0.0, pytest.approx(1.0))
    assert (G.data >= 0).all()
    for i in range(3):
        row = G.getrow(i).toarray().ravel()
        assert int(np.argmax(row)) == int(of.cell_index(lat[i], lon[i]))


@pytest.mark.parametrize("p", [0.8, 1.0, 1.1, 1.7])
def test_omori_window_integral_is_the_integral(p):
    c = 0.05
    for delta in (0.0, 0.3, 12.0, 4000.0):
        s = np.linspace(delta, delta + of.HORIZON_DAYS, 200001)
        y = (s + c) ** (-p)
        quad = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(s)))
        assert of.omori_window_integral(np.array([delta]), c, p)[0] == pytest.approx(quad, rel=1e-6)
    near = of.omori_window_integral(np.array([2.0]), c, 1.0 + 1e-7)[0]
    assert near == pytest.approx(of.omori_window_integral(np.array([2.0]), c, 1.0)[0], rel=1e-6)


def test_target_window_is_half_open():
    t0 = _epoch(2020, 5, 1)
    ev = of.EventSet.from_arrays([t0 - 1, t0, t0 + 29.99 * DAY, t0 + 30 * DAY, t0 + 1],
                                 [10.0, 10.0, 12.5, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0, 20.0],
                                 [7.0, 6.0, 6.5, 6.2, 5.9])
    Y = of.target_matrix(ev, [t0]).toarray()[0]
    c1 = int(of.cell_index(10.0, 20.0))
    c2 = int(of.cell_index(12.5, 20.0))
    assert Y[c1] == 1          # the event AT t counts; the one before t, the one at t+30d and the M5.9 do not
    assert Y[c2] == 1
    assert Y.sum() == 2


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def _rates(ev, t, spec=SPEC_B):
    return of.RateEngine(ev, spec).rates(t, spec)


@pytest.mark.parametrize("offset_s", [0.0, 1.0, 10 * DAY])
def test_an_event_at_or_after_the_issue_time_never_changes_the_forecast(offset_s):
    ev = _catalog()
    t = _epoch(2010, 7, 1)
    base = _rates(ev, t)
    later = _with(ev, t + offset_s, 38.3, 142.4, 9.1)
    after = _rates(later, t)
    for key in ("lambda_long", "lambda_short", "lambda", "probability"):
        np.testing.assert_array_equal(after[key], base[key])
    # the long-term input's declustering flags of past events are fixed too
    e0, e1 = of.RateEngine(ev, SPEC_B), of.RateEngine(later, SPEC_B)
    n = e0.events.n_before(t)
    np.testing.assert_array_equal(e0.flags[:n], e1.flags[:n])


def test_the_causality_instrument_can_see_a_change():
    """The same M9.1 one second BEFORE the issue time must move the forecast -- otherwise
    the test above would pass for a model that ignores its inputs."""
    ev = _catalog()
    t = _epoch(2010, 7, 1)
    base = _rates(ev, t)
    before = _rates(_with(ev, t - 1.0, 38.3, 142.4, 9.1), t)
    cell = int(of.cell_index(38.3, 142.4))
    assert before["lambda_short"][cell] > 100 * base["lambda_short"][cell]
    assert before["probability"][cell] > base["probability"][cell] + 0.5


# ---------------------------------------------------------------------------
# Rate-model sanity
# ---------------------------------------------------------------------------

def test_rate_model_sanity():
    ev = _catalog()
    eng = of.RateEngine(ev, SPEC_B)
    t = _epoch(2015, 1, 1)
    r = eng.rates(t)
    assert np.all(r["probability"] > 0) and np.all(r["probability"] < 1)
    assert r["lambda_long"].sum() == pytest.approx(SPEC_B.long.mu, rel=1e-12)
    assert np.all(r["lambda_short"] >= 0)
    np.testing.assert_allclose(r["probability"], -np.expm1(-(r["lambda_long"] + r["lambda_short"])), rtol=1e-15)
    # the short-term total equals the sum of event weights times their in-domain kernel mass
    w = eng.short_term_weights(t, SPEC_B.short)
    assert r["lambda_short"].sum() == pytest.approx(float(np.sum(w * eng.short_mass[:w.size])), rel=1e-10)
    # A has no short-term part; its probability is 1 - exp(-lambda_long)
    ra = of.RateEngine(ev, SPEC_A).rates(t, SPEC_A)
    assert np.all(ra["lambda_short"] == 0)


def test_short_term_rate_decays_after_a_large_event():
    ev = _with(_catalog(), _epoch(2012, 1, 1), -20.1, -175.2, 8.0)
    eng = of.RateEngine(ev, SPEC_B)
    cell = int(of.cell_index(-20.1, -175.2))
    ts = [_epoch(2012, 1, 2), _epoch(2012, 2, 1), _epoch(2012, 7, 1), _epoch(2014, 1, 1)]
    vals = [eng.short_term_rate(t, SPEC_B.short)[cell] for t in ts]
    assert all(a > b for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# Artifact, identity, serving parity
# ---------------------------------------------------------------------------

def _write(tmp_path, ev, cutoff, spec=SPEC_B, name="eq_operational_B_test"):
    frozen = ev.before(cutoff)
    path = tmp_path / "art.json"
    stamp = dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = of.write_artifact(path, spec, frozen, cutoff=stamp, model_name=name, provenance={"test": True})
    return path, version


def test_serving_path_equals_the_evaluated_engine_bit_for_bit(tmp_path):
    ev = _catalog()
    cutoff = _epoch(2012, 1, 1)
    path, _ = _write(tmp_path, ev, cutoff)
    art = of.load_artifact(path)
    live = _dicts(ev.select(ev.t >= cutoff - 400 * DAY))       # overlaps the frozen part; must be ignored there
    for t in (_epoch(2011, 6, 1), _epoch(2013, 3, 4), _epoch(2019, 12, 30)):
        served = of.forecast_from_artifact(art, live, t)
        evaluated = of.RateEngine(ev, SPEC_B).rates(t, SPEC_B)
        for key in ("lambda_long", "lambda_short", "probability"):
            np.testing.assert_array_equal(served[key], evaluated[key])
        assert served["model_version"] == art.model_version


def test_serving_refuses_a_hole_between_the_cutoff_and_the_live_fetch(tmp_path):
    ev = _catalog()
    cutoff = _epoch(2012, 1, 1)
    path, _ = _write(tmp_path, ev, cutoff)
    art = of.load_artifact(path)
    live = _dicts(ev.select(ev.t >= cutoff + 60 * DAY))
    with pytest.raises(of.OperationalArtifactError):
        of.forecast_from_artifact(art, live, _epoch(2015, 1, 1))


def test_model_version_is_bound_to_content_not_line_endings(tmp_path):
    ev = _catalog()
    path, version = _write(tmp_path, ev, _epoch(2012, 1, 1))
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert of.artifact_model_version(crlf) == version
    assert version.endswith(hashlib.sha256(raw).hexdigest()[:12])
    from hazardpulse.hurricane.ri_model import sha256_file
    assert of.sha256_text_file(crlf) == sha256_file(crlf) == of.sha256_text_file(path)
    other, v2 = _write(tmp_path / "x", ev, _epoch(2012, 1, 1),
                       spec=of.ModelSpec("B", SPEC_B.long, of.ShortTermParams(10.0, 0.03, 1.0, 0.01, 1.1)))
    assert v2 != version


def test_load_refuses_a_foreign_kernel_geometry(tmp_path):
    ev = _catalog()
    path, _ = _write(tmp_path, ev, _epoch(2012, 1, 1))
    body = json.loads(path.read_text(encoding="utf-8"))
    body["kernel_geometry"]["n_sub"] = 2
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(of.OperationalArtifactError):
        of.load_artifact(path)
    with pytest.raises(of.OperationalArtifactError):
        of.load_artifact(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Candidate C (boosted trees over the rate maps and recent counts)
# ---------------------------------------------------------------------------

def _one_split_payload(feature: int, threshold: float, low: float, high: float) -> dict:
    """A one-tree LightGBM payload: raw = low if x[feature] <= threshold else high."""
    from hazardpulse.earthquake.operational_features import CORE_FEATURES
    tree = {"feature": [feature, -1, -1], "threshold": [threshold, 0.0, 0.0], "default_left": [True, False, False],
            "missing_type": [0, 0, 0], "left": [1, -1, -1], "right": [2, -1, -1], "value": [0.0, low, high],
            "node_value": [0.0, low, high]}
    return {"schema": "hazardpulse_lgbm_payload/1", "feature_names": list(CORE_FEATURES), "n_trees": 1,
            "trees": [tree], "calibration": {"a": 1.0, "b": 0.0}, "provenance": {}}


def _live_dicts_with_depth(ev: of.EventSet, seed=3) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = _dicts(ev)
    for d in out:
        d["depth"] = float(rng.uniform(5, 300))
    return out


def _gbt_artifact(tmp_path, ev, cutoff, payload):
    from hazardpulse.earthquake.operational_features import CORE_FEATURES
    gbt = {"name": "C0", "long_A": {"kernel_km": 20.0, "declustered": True, "mu": 10.0, "eps": 0.01},
           "feature_names": list(CORE_FEATURES), "feature_dtype": "float32", "live_span_days": 1827.0,
           "payload": payload}
    path = tmp_path / "gbt.json"
    stamp = dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    of.write_artifact(path, SPEC_B, ev.before(cutoff), cutoff=stamp, model_name="eq_operational_C0_test",
                      provenance={}, gbt=gbt)
    return of.load_artifact(path)


def _dense_live(t_end, n=4000, seed=11):
    """A synthetic M2.5+ live fetch: events every ~10 h over 1,830 days, some M4.5+."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(t_end - 1830 * DAY, t_end + 40 * DAY, n))
    lat = 38.0 + rng.normal(0, 2.0, n)
    lon = 142.0 + rng.normal(0, 2.0, n)
    mag = np.minimum(2.5 + rng.exponential(1.0 / np.log(10.0), n), 7.9)
    return of.EventSet.from_arrays(t, lat, lon, mag)


def test_boosted_tree_forecast_is_causal_and_its_check_is_live(tmp_path):
    from hazardpulse.earthquake.operational_features import CORE_FEATURES
    ev = _catalog()
    cutoff = _epoch(2012, 1, 1)
    t = _epoch(2014, 6, 2)
    live_ev = _dense_live(t)
    # split on the short-term rate so a nearby event before t must move the probability
    art = _gbt_artifact(tmp_path, ev, cutoff,
                        _one_split_payload(CORE_FEATURES.index("log10_lambda_short"), -6.0, -6.0, -1.0))
    live = _live_dicts_with_depth(live_ev.concat(ev.select(ev.t >= cutoff)))
    base = of.forecast_from_artifact(art, [d for d in live if of._parse_time(d["time"]) < t], t)
    with_future = of.forecast_from_artifact(art, live + [{"time": t + 1.0, "latitude": 38.3, "longitude": 142.4,
                                                          "mag": 9.1, "depth": 20.0}], t)
    np.testing.assert_array_equal(with_future["probability"], base["probability"])
    assert base["input_sha256"] == with_future["input_sha256"]
    # a quiet South-Atlantic cell sits on the low branch; an M8.5 there one second before t
    # must lift its short-term rate over the split and move the probability
    cell = int(of.cell_index(-45.1, 0.3))
    assert base["probability"][cell] == pytest.approx(1 / (1 + np.exp(6.0)), rel=1e-12)
    before = of.forecast_from_artifact(art, live + [{"time": t - 1.0, "latitude": -45.1, "longitude": 0.3,
                                                     "mag": 8.5, "depth": 20.0}], t)
    assert before["probability"][cell] == pytest.approx(1 / (1 + np.exp(1.0)), rel=1e-12)


def test_boosted_tree_forecast_refuses_a_live_fetch_shorter_than_its_features(tmp_path):
    ev = _catalog()
    cutoff = _epoch(2012, 1, 1)
    t = _epoch(2014, 6, 2)
    art = _gbt_artifact(tmp_path, ev, cutoff, _one_split_payload(0, -2.0, -6.0, -1.0))
    live = _dense_live(t)
    short = _live_dicts_with_depth(live.select(live.t >= t - 400 * DAY))
    with pytest.raises(of.OperationalArtifactError):
        of.forecast_from_artifact(art, short + _dicts(ev.select((ev.t >= cutoff) & (ev.t < t))), t)


def test_boosted_tree_features_never_read_the_future():
    from hazardpulse.earthquake import operational_features as feat
    live = _dense_live(_epoch(2014, 6, 2))
    t = _epoch(2014, 6, 2)
    rng = np.random.default_rng(5)
    depth = rng.uniform(5, 300, len(live))
    lam = np.full(of.N_CELLS, 1e-4)

    def feats(extra_t):
        tt = np.r_[live.t, extra_t]
        c45 = feat.cell_catalog_from_arrays(tt, np.r_[live.lat, [38.3]], np.r_[live.lon, [142.4]],
                                            np.r_[live.mag, [8.0]], np.r_[depth, [10.0]])
        c25 = feat.cell_catalog_from_arrays(tt, np.r_[live.lat, [38.3]], np.r_[live.lon, [142.4]],
                                            np.r_[live.mag, [8.0]])
        return feat.core_features(t, c45, c25, lam, lam, lam)

    np.testing.assert_array_equal(feats(t), feats(t + 86400.0 * 3))      # at t and after t: identical
    assert not np.array_equal(feats(t - 1.0), feats(t))                  # one second before t: seen


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------

def _sha64(p) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype=np.float64).tobytes()).hexdigest()


@pytest.mark.skipif(not SERVED.exists(), reason="served artifact not built")
def test_the_served_artifact_reproduces_the_forecasts_the_program_scored():
    """The committed artifact gives back what the evaluation scored at its recorded issue
    times: the rate model's float64 grid bit for bit from the frozen catalog, and -- for a
    boosted-tree artifact -- the evaluated probabilities from the stored feature rows through
    the NumPy trees. (The full live path, which also needs five years of M2.5+ events, is
    checked at build time by scripts/earthquake_program/build_artifact.py.)"""
    art = of.load_artifact(SERVED)
    checks = art.meta["provenance"]["parity"]
    assert len(checks) >= 3
    engine = of.RateEngine(art.frozen, art.spec)
    for rec in checks:
        t = of._parse_time(rec["issue_time"])
        p = engine.rates(t)["probability"]
        assert _sha64(p) == rec["rate_sha256_float64"]
        if art.gbt is not None:
            rows = rec["tree_rows"]
            X = np.asarray([[np.nan if v is None else v for v in row] for row in rows["X_float32"]], dtype=np.float32)
            assert np.isnan(X).any() and (X[:, -1] == 1).any()      # rows exercise missing values and active cells
            got = of.gbt_probability(art.gbt["payload"], X)
            np.testing.assert_allclose(got, rows["probability"], rtol=0, atol=1e-12)
        else:
            assert _sha64(p) == rec["sha256_float64"]
            for cell, val in rec["samples"].items():
                assert p[int(cell)] == val
