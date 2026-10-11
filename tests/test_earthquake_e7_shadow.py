"""Amendment E7's live shadow (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 15): recorded beside every earthquake
forecast after a shallow M6+, never published, judged at the looks."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.earthquake import aftershock as A
from hazardpulse.earthquake import operational_forecast as of

ROOT = Path(__file__).resolve().parents[1]
NOW = dt.datetime(2026, 10, 9, 18, 13, tzinfo=dt.timezone.utc)
PARAMS = {"K": 0.0794, "alpha": 0.484, "p": 0.770, "c_t": 0.0001, "d0": 37.8}


@pytest.fixture(scope="module")
def fse():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fse_e7_t", ROOT / "scripts" / "fetch_and_score_earthquake.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fit(tmp_path: Path) -> Path:
    p = tmp_path / "aftershock_e7_fit.json"
    p.write_text(json.dumps({"params": PARAMS}), encoding="utf-8")
    return p


PANAMA = {"time": "2026-10-09T17:56:06Z", "latitude": 7.587, "longitude": -80.769, "depth": 12.6, "mag": 7.7, "id": "us6000u18k"}


def test_nothing_is_recorded_without_the_fit_or_without_a_recent_large_shallow_earthquake(fse, tmp_path):
    base = np.full(of.N_CELLS, 1e-4)
    assert fse.aftershock_e7_shadow([PANAMA], NOW, base, fit_path=tmp_path / "absent.json") is None
    deep = dict(PANAMA, depth=300.0)
    small = dict(PANAMA, mag=5.9)
    old = dict(PANAMA, time="2026-08-01T00:00:00Z")
    assert fse.aftershock_e7_shadow([deep, small, old], NOW, base, fit_path=_fit(tmp_path)) is None


def test_the_shadow_is_the_served_forecast_plus_the_aftershock_increment(fse, tmp_path):
    base = np.full(of.N_CELLS, 1e-4)
    home = int(of.cell_index(7.587, -80.769))
    base[home] = 0.0597                                         # what was published at 18:00
    rec = fse.aftershock_e7_shadow([PANAMA], NOW, base, fit_path=_fit(tmp_path))
    assert rec["triggers"][0]["id"] == "us6000u18k" and rec["params"] == PARAMS
    cells = {int(k): v for k, v in rec["cells"].items()}
    assert cells[home] == max(cells.values()) and cells[home] > base[home]
    layer = A.AftershockLayer(A.Params(**PARAMS))
    t = NOW.timestamp() / 86400.0
    trig = [layer.trigger(dt.datetime(2026, 10, 9, 17, 56, 6, tzinfo=dt.timezone.utc).timestamp() / 86400.0, 7.587, -80.769, 7.7)]
    want = 1.0 - (1.0 - 0.0597) * math.exp(-layer.rate(home, t, trig))
    assert cells[home] == pytest.approx(want, rel=1e-5)
    assert all(v >= base[k] - 1e-12 for k, v in cells.items())   # an increment never lowers the served number
    json.dumps(rec, allow_nan=False)                              # strict JSON, as every published record
