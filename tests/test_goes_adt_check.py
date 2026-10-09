"""The ADT input check: every ADT row format parses (a first parser silently lost 16 % of rows -- all pinhole, large
and integer-wind rows, and the lat/lon of every eye row), an unknown format raises, and every gate can fail."""
from __future__ import annotations

import datetime as dt
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# real rows, one per format (CIMSS ADT archives: AL142024, EP082020, AL012022, AL022024, AL052022, AL142024)
ROWS = """\
2024OCT05 154020  2.0 1005.4  30.0  2.0 2.0 2.0  NO LIMIT  OFF  OFF  OFF  OFF -34.06 -45.55  CRVBND   N/A    N/A   22.14   95.08  FCST    GOES16 34.3
2020JUL22 125032  3.9  991.0  63.0  3.9 5.0 5.3  1.7T/6hr  OFF  OFF  OFF  OFF -17.94 -59.23  EYE    -99 IR  -0.0   11.90  128.80  FCST    GOES17 17.0
2022JUN05 072021  3.0   990   45  3.0 3.0 3.0  NO LIMIT  OFF  OFF  OFF  OFF  19.73  15.49  SHEAR    N/A   -3.7   29.01   77.18  FCST    GOES16 33.9 ETadj CI=3.0
2024JUL03 024021  6.0  946.5 115.0  6.0 6.3 6.3  NO LIMIT   ON  OFF  OFF  OFF -23.82 -71.20  EYE/P  -99 IR  48.9   16.16   72.54  ARCHER  GOES16 19.2
2022SEP04 045023  2.4  1003   34  2.4 2.7 4.6  0.5T/hour  ON  OFF  OFF  OFF   2.56 -51.78  EYE/L   48 IR   5.1   38.13   45.17  FCST    GOES16 54.1
2024OCT10 014020  0.0    0.0   0.0  0.0 0.0 0.0            N/A  N/A  N/A  OFF  99.50  99.50  LAND     N/A    N/A   27.83   82.31  ARCHER  GOES16 33.4
"""
HEADER = "ADT91 LIST 14L.ODT CKZ=YES\n=====    ADT-Version 9.1 =====\n   Date    (UTC)   CI  (CKZ)/(kts)\n"


def _mod():
    spec = importlib.util.spec_from_file_location("goes_adt_check_t", ROOT / "scripts" / "goes_adt_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_adt_row_format_parses_with_its_position_and_scene():
    ac = _mod()
    recs = ac.parse_adt(HEADER + ROWS)
    assert [r["scene"] for r in recs] == ["CRVBND", "EYE", "SHEAR", "EYE", "EYE", "LAND"]
    assert recs[0]["time"] == dt.datetime(2024, 10, 5, 15, 40, 20) and recs[0]["vmax"] == 30.0
    assert (recs[1]["lat"], recs[1]["lon"]) == (11.90, -128.80)          # the two-token RMW field ("-99 IR")
    assert recs[1]["eye_c"] == -17.94 and recs[2]["vmax"] == 45.0         # integer wind
    assert (recs[3]["lat"], recs[3]["lon"]) == (16.16, -72.54)
    assert (recs[0]["lat"], recs[0]["lon"]) == (22.14, -95.08)


def test_an_unknown_row_format_raises_rather_than_dropping_the_row():
    ac = _mod()
    with pytest.raises(ValueError, match="unparsed ADT row"):
        ac.parse_adt(HEADER + "2024OCT05 154020  garbled row\n")


def _m(n=30000, eye_rate=0.15, agree=0.95, weak_eye=0.0, dist=9.0, teye_noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    adt = rng.random(n) < eye_rate
    eye = np.where(rng.random(n) < agree, adt, ~adt)
    vmax = np.where(adt, 100.0, rng.uniform(25, 90, n))
    weak = vmax < 50
    eye[weak] = rng.random(weak.sum()) < weak_eye
    adt_c = np.where(adt, rng.uniform(-30, 20, n), 99.5)
    return {"year": np.where(rng.random(n) < 0.5, 2022.0, 2025.0), "vmax": vmax, "adt_eye": adt, "adt_eye_c": adt_c,
            "eye": eye, "teye": adt_c + 273.15 + rng.normal(0, teye_noise, n), "dist_km": np.full(n, dist)}


def test_the_gates_pass_a_good_collection_and_each_one_can_fail():
    ac = _mod()
    assert ac.check(_m())["all_passed"]
    assert not ac.check(_m(agree=0.80))["passed"]["hss_2024_2026"]
    assert not ac.check(_m(weak_eye=0.2))["passed"]["eye_frac_below_50kt"]
    assert not ac.check(_m(dist=30.0))["passed"]["centre_km_median"]
    assert not ac.check(_m(teye_noise=40.0))["passed"]["teye_spearman"]
    assert not ac.check(_m(n=5000))["passed"]["matched"]


def test_nothing_to_measure_fails_every_gate():
    ac = _mod()
    empty = {k: np.zeros(0, bool if k in ("adt_eye", "eye") else float)
             for k in ("year", "vmax", "adt_eye", "adt_eye_c", "eye", "teye", "dist_km")}
    res = ac.check(empty)
    assert not any(res["passed"].values()) and math.isnan(res["measured"]["hss_2024_2026"])
