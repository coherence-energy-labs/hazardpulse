"""Tornado program amendment 10 (NOAA's 2025 ProbSevere format change): the decision script's load-bearing facts.

The served payloads read `p_ps` and `p_vil_density`, which the new file no longer carries and the parser reads
as 0.0. Candidate (b), "the trees' missing-value branch", was registered as a repair; on these payloads it is
the served model itself, by construction -- pinned here so a re-export that changes the rule cannot hide.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.tornado import lgbm_payload as lp

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"


@pytest.fixture(scope="module")
def t2():
    spec = importlib.util.spec_from_file_location("t2_decide_t", ROOT / "scripts" / "tornado_program" / "t2_decide.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    P = rng.gamma(2.0, 20.0, size=(n, 28)).astype(np.float32)
    P[:, 0] = 0.0          # p_ps, as the parser reads it since 2025-08-05
    P[:, 11] = 0.0         # p_vil_density
    active = (rng.random(n) < 0.1).astype(np.float32)
    return {"P": P, "y": np.zeros(n, np.int8), "w_active": active,
            "w_minutes": np.where(active > 0, rng.uniform(0, 45, n), np.nan).astype(np.float32)}


def test_the_missing_value_branch_is_the_served_model_on_these_payloads(t2):
    """Every split on the two inputs has missing type None, where a NaN reads as 0.0."""
    rows = _rows(4000, 1)
    for fname in ("tornado_v3_w.json", "tornado_v3.json"):
        payload = json.loads((MODELS / fname).read_text(encoding="utf-8"))
        a = t2.predict(payload, t2.columns(payload, rows, False))
        b = t2.predict(payload, t2.columns(payload, rows, True))
        assert np.array_equal(a, b), fname
        for feat in t2.ABSENT:
            j = payload["feature_names"].index(feat)
            kinds = {mt for tr in payload["trees"] for fe, mt in zip(tr["feature"], tr["missing_type"]) if fe == j}
            assert kinds == {0}, (fname, feat, kinds)


def test_the_nan_column_really_is_nan_and_the_warning_state_is_block_w(t2):
    rows = _rows(50, 2)
    payload = json.loads((MODELS / "tornado_v3_w.json").read_text(encoding="utf-8"))
    names = payload["feature_names"]
    a, b = t2.columns(payload, rows, False), t2.columns(payload, rows, True)
    for feat in t2.ABSENT:
        assert (a[:, names.index(feat)] == 0).all() and np.isnan(b[:, names.index(feat)]).all()
    assert np.array_equal(a[:, names.index("w_tor_warning_active")], rows["w_active"])
    np.testing.assert_array_equal(a[:, names.index("w_minutes_since_issue")], rows["w_minutes"])
    j = names.index("p_ps_tor")
    assert np.array_equal(a[:, j], rows["P"][:, t2.P_NAMES.index("p_ps_tor")].astype(np.float64))


@pytest.mark.skipif(not (MODELS / "candidates" / "tornado_v3_drop2_w.json").exists(), reason="candidate not exported")
def test_candidate_d_never_reads_the_inputs_noaa_removed(t2):
    for fname in ("candidates/tornado_v3_drop2_w.json", "candidates/tornado_v3_drop2.json"):
        payload = json.loads((MODELS / fname).read_text(encoding="utf-8"))
        assert not set(t2.ABSENT) & set(payload["feature_names"]), fname
        assert payload["provenance"]["final_run"].startswith("v3_drop2_")


def test_preliminary_reports_are_utc_instants_whatever_the_machines_clock(t2, monkeypatch):
    """score_tornado_prospective.parse_utc returns NAIVE UTC; .timestamp() of a naive time reads LOCAL time."""
    import score_tornado_prospective as stp
    fake = [{"time": "2026-04-27T21:05:00Z", "lat": 35.0, "lon": -97.0, "mag": 1}]
    monkeypatch.setattr(stp, "fetch_spc_reports_range", lambda a, b: (fake, ["2026-04-20"]))
    old = os.environ.get("TZ")
    try:
        if hasattr(time, "tzset"):
            os.environ["TZ"] = "America/Chicago"
            time.tzset()
        by_date, failed = t2.preliminary_reports(dt.date(2026, 4, 27), dt.date(2026, 4, 27))
    finally:
        if hasattr(time, "tzset"):
            if old is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old
            time.tzset()
    (r,) = by_date["20260427"]
    assert r["time_utc"] == dt.datetime(2026, 4, 27, 21, 5, tzinfo=dt.timezone.utc).timestamp()
    assert (r["slat"], r["slon"]) == (35.0, -97.0) and failed == ["2026-04-20"]


def test_a_day_whose_reports_could_not_be_read_is_never_scored(t2, tmp_path, monkeypatch):
    monkeypatch.setattr(t2, "ROWS", tmp_path)
    for ds in ("20260426", "20260427", "20260428", "20260429"):
        np.savez(tmp_path / f"{ds}.npz", P=np.zeros((2, 28), np.float32), y=np.zeros(2, np.int8),
                 w_active=np.zeros(2, np.float32), w_minutes=np.full(2, np.nan, np.float32))
    # convective day 04-27 holds UTC 04-27 12Z .. 04-28 12Z: labels of 04-26 .. 04-28 may need it
    rows, dropped = t2.load_rows("20260426", "20260429", ["2026-04-27"])
    assert dropped == ["20260426", "20260427", "20260428"]
    assert set(np.unique(rows["day"])) == {20260429}
