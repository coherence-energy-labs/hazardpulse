"""TC2b live (TC1 program amendment 4): the record is the backtest's function on the storm's own v10.4 curve for
TC1's cycle, it never changes TC1's or NHC's numbers, and every reason it cannot move TC1 is written down."""
from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

from hazardpulse.hurricane import tc2

ROOT = Path(__file__).resolve().parents[1]


def _fs():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_and_score_tc2_t", ROOT / "scripts" / "fetch_and_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Deck:
    def __init__(self, v0):
        self.v0 = v0

    def analysis(self, t):
        return (24.8, -89.0, self.v0)


CYCLE = "2026-10-07T06:00:00Z"
CURVE = {"15": 0.9, "20": 0.8, "25": 0.7, "30": 0.66, "35": 0.4, "40": 0.2, "45": 0.1}   # Isaias 7 Oct 06Z-like


def _rec():
    lead = {"12": 42.0, "24": 50.0, "36": 55.0, "48": 58.0, "72": 60.0, "96": 55.0, "120": 45.0}
    return {"cycle": CYCLE, "TC1": {k: {"lat": 25.0, "lon": -89.0, "vmax_kt": v} for k, v in lead.items()},
            "OFCL": {"24": {"lat": 25.1, "lon": -89.1, "vmax_kt": 60.0}}}


def _storm(**shadow):
    sh = {"status": "ok", "cycle": CYCLE, "gate_ok": True, "model_probabilities": CURVE,
          "model_version": "hurricane_ri_v10_4-x"}
    sh.update(shadow)
    return {"storm_id": "AL092026", "ri_v10_4_shadow": sh}


def test_tc2b_live_is_the_backtest_function_on_the_storms_own_curve_and_leaves_tc1_alone():
    fs = _fs()
    rec = _rec()
    before = copy.deepcopy(rec)
    out = fs.tc2b_record(_storm(), rec, _Deck(35.0))
    assert rec == before                                              # TC1's and NHC's numbers untouched
    want, info = tc2.project({int(k): v["vmax_kt"] for k, v in rec["TC1"].items()}, 35.0,
                             {int(k): v for k, v in CURVE.items()}, tc2.TC2B_TAPER_END_H)
    assert out["moved_tc1"] and out["shift_24h_kt"] == pytest.approx(15.0)
    assert out["intensity"] == {str(k): round(v, 1) for k, v in sorted(want.items())}
    assert out["intensity"]["24"] == 65.0 and out["intensity"]["72"] == 60.0     # raised at 24 h; TC1 from 72 h


@pytest.mark.parametrize("shadow, why", [
    ({"status": "error: x"}, "no v10.4 shadow"),
    ({"cycle": "2026-10-07T00:00:00Z"}, "not TC1's cycle"),
    ({"gate_ok": False}, "gate failed"),
])
def test_without_a_gated_curve_for_tc1s_cycle_tc2b_is_tc1_and_says_why(shadow, why):
    fs = _fs()
    rec = _rec()
    out = fs.tc2b_record(_storm(**shadow), rec, _Deck(35.0))
    assert not out["moved_tc1"] and why in out["why_unmoved"]
    assert out["intensity"] == {k: v["vmax_kt"] for k, v in sorted(rec["TC1"].items(), key=lambda kv: int(kv[0]))}


def test_no_analysis_intensity_means_no_move():
    fs = _fs()
    out = fs.tc2b_record(_storm(), _rec(), None)
    assert not out["moved_tc1"] and out["v0_kt"] is None
    assert dt.datetime.fromisoformat(CYCLE.replace("Z", "")).hour == 6


def test_an_open_bracket_is_strict_json_and_the_writer_never_publishes_a_non_finite_number():
    """2026-10-10 03:39Z: TC2b recorded its open lower edge as -inf, Python wrote -Infinity, and the Worker could not
    parse live-storms.json -- /api/v1/live/hurricane answered 404 until the next run."""
    import json
    fs = _fs()
    steady = {"15": 0.3, "20": 0.2, "25": 0.1, "30": 0.05, "35": 0.03, "40": 0.02, "45": 0.01}   # no lower edge
    out = fs.tc2b_record(_storm(model_probabilities=steady), _rec(), _Deck(35.0))
    assert out["bracket_kt"] == [None, 15.0]
    json.dumps(out, allow_nan=False)                                  # raises on any non-finite number
    clean, bad = fs.strict_json({"a": [1.0, float("-inf")], "b": {"c": float("nan"), "d": 2}})
    assert clean == {"a": [1.0, None], "b": {"c": None, "d": 2}} and bad == ["/a/1", "/b/c"]


def test_tc2b_plus_o_is_the_same_rule_on_tc1_plus_os_own_winds():
    fs = _fs()
    rec = _rec()
    rec["TC1+O"] = {k: dict(v, vmax_kt=v["vmax_kt"] + 4.0) for k, v in rec["TC1"].items()}   # TC1+O: 4 kt higher
    out = fs.tc2b_record(_storm(), rec, _Deck(35.0), base="TC1+O", label="TC2b+O")
    want, _ = tc2.project({int(k): v["vmax_kt"] for k, v in rec["TC1+O"].items()}, 35.0,
                          {int(k): v for k, v in CURVE.items()}, tc2.TC2B_TAPER_END_H)
    assert out["label"] == "TC2b+O" and out["base"] == "TC1+O"
    assert out["intensity"] == {str(k): round(v, 1) for k, v in sorted(want.items())}
    assert out["shift_24h_kt"] == pytest.approx(11.0)                   # TC1+O +19 kt; the bracket starts at +30
