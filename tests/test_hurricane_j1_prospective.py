"""J1's prospective test (amendment 13b): amendment 7's record selection, the full-precision comparator (it must
round to what was published), the outcome rule, and v9.1's claim rule -- each able to pass and to fail."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("j1p_t", ROOT / "scripts" / "score_hurricane_j1_prospective.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _file(tmp, made: dt.datetime, storms):
    art = {"forecast_id": f"hu_fcst_{made:%Y%m%d_%H%M%S}", "generated_at": made.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "issued_at": made.strftime("%Y-%m-%dT%H:%M:%SZ"), "storms": storms}
    (tmp / f"{art['forecast_id']}.json").write_text(json.dumps(art), encoding="utf-8")


def _storm(sid, t, pub, j1p, full=None):
    return {"storm_id": sid, "issue_time": t.isoformat(), "ri_source": "v8.2", "ri_probability": pub,
            "ri_j1_shadow": {"status": "ok", "model_probability": j1p, "probability": round(j1p, 4),
                             "model_version": "hurricane_ri_j1-x", "v8_2_model_probability": pub if full is None else full}}


def test_one_record_per_cycle_full_precision_comparator_and_rounding_guard(tmp_path):
    j = _mod()
    t = dt.datetime(2026, 10, 12, 0)
    _file(tmp_path, t + dt.timedelta(hours=2), [_storm("WP282026", t, 0.10, 0.30)])                 # preliminary
    _file(tmp_path, t + dt.timedelta(hours=4), [_storm("WP282026", t, 0.0000, 0.40, full=0.000041),    # the test record
                                                _storm("SH022027", t, 0.05, 0.06, full=0.0612)])       # rounds wrongly
    _file(tmp_path, t + dt.timedelta(hours=5), [_storm("WP282026", t, 0.2, 0.5)])                     # a later duplicate
    recs = j.collect(tmp_path)
    assert [(r["storm_id"], r["a"], r["p"]) for r in recs] == [("WP282026", 0.000041, 0.40)]


def _calibration(version):
    return json.loads((ROOT / "results" / "models" / f"{version}.json").read_text(encoding="utf-8"))["calibration"]


def _input_for(p, version="hurricane_ri_v8_2"):
    """The ``v82_logit`` (an ensemble logit) that ``version``'s calibration maps to ``p``."""
    cal = _calibration(version)
    return (float(np.log(p / (1 - p))) - cal["b"]) / cal["a"]


def _storm_v83(sid, t, pub83, j1p, v82_full, beside=None, shadow_base="hurricane_ri_v8_2", x=None):
    """A record made after amendment 15: v8.3 published, v8.2 (J1's comparator) recorded beside it and in the
    shadow at full precision, J1's first input the v8.2 ensemble behind it."""
    s = {"storm_id": sid, "issue_time": t.isoformat(), "ri_source": "v8.3", "ri_probability": pub83,
         "model_version": "hurricane_ri_v8_3",
         "ri_j1_shadow": {"status": "ok", "model_probability": j1p, "probability": round(j1p, 4),
                          "model_version": "hurricane_ri_j1-x", "v8_2_model_probability": v82_full,
                          "v8_2_model_version": shadow_base,
                          "inputs": {"v82_logit": _input_for(v82_full) if x is None else x}}}
    if beside is not False:
        s["v8_2"] = beside or {"ri_probability": round(v82_full, 4), "model_version": "hurricane_ri_v8_2"}
    return s


def test_after_amendment_15_the_comparator_is_v82_never_the_published_v83(tmp_path):
    """Amendment 15: v8.3 is published and J1's comparator stays v8.2. A record is scored only when its comparator
    is v8.2's own number -- named so in the shadow and matching v8.2's number recorded beside the published one;
    the published v8.3 never stands in for it."""
    j = _mod()
    t = dt.datetime(2026, 10, 20, 0)
    _file(tmp_path, t + dt.timedelta(hours=4), [
        _storm_v83("WP302026", t, 0.0812, 0.30, v82_full=0.064412),                     # good: scored against 0.064412
        _storm_v83("WP312026", t, 0.0812, 0.30, v82_full=0.0812, beside=False),         # no v8.2 beside: refused
        _storm_v83("WP322026", t, 0.0812, 0.30, v82_full=0.0812,                        # v8.3 posing as comparator
                   beside={"ri_probability": 0.0812, "model_version": "hurricane_ri_v8_3"},
                   shadow_base="hurricane_ri_v8_3"),
        _storm_v83("WP332026", t, 0.0812, 0.30, v82_full=0.064412,                      # beside disagrees: refused
                   beside={"ri_probability": 0.0700, "model_version": "hurricane_ri_v8_2"}),
        _storm_v83("WP342026", t, 0.0812, 0.30, v82_full=0.064412, shadow_base=None),   # shadow names no base
        # every label says v8.2, but J1's input is the ensemble v8.3's calibration maps to the comparator: the
        # scorer fed J1 the served model's numbers and called them v8.2's -- refused by the input check
        _storm_v83("WP372026", t, 0.0812, 0.30, v82_full=0.0812, x=_input_for(0.0812, "hurricane_ri_v8_3")),
    ])
    recs = j.collect(tmp_path)
    assert [(r["storm_id"], r["a"]) for r in recs] == [("WP302026", 0.064412)]
    assert _calibration("hurricane_ri_v8_2")["a"] != _calibration("hurricane_ri_v8_3")["a"]  # the check can see it
    # the rule that admitted the good record is the one that refuses the others, record by record
    good = _storm_v83("WP302026", t, 0.0812, 0.30, v82_full=0.064412)
    assert j.comparator_checks(good, good["ri_j1_shadow"])
    assert not j.comparator_checks({**good, "v8_2": {"ri_probability": 0.0812, "model_version": "hurricane_ri_v8_3"}},
                                   good["ri_j1_shadow"])
    # records made before amendment 15 are read as before: v8.2 published, the comparator rounds to it
    legacy = _storm("WP352026", t, 0.0644, 0.30, full=0.064412)
    assert j.comparator_checks(legacy, legacy["ri_j1_shadow"])
    assert not j.comparator_checks(_storm("WP362026", t, 0.0812, 0.30, full=0.064412),
                                   _storm("WP362026", t, 0.0812, 0.30, full=0.064412)["ri_j1_shadow"])


def _scored(n_storms, better, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_storms):
        for c in range(8):
            y = int(rng.random() < 0.12)
            a = 0.1
            p = (0.6 if y else 0.03) if better else a
            out.append({"storm_id": f"WP{s:02d}2026", "cycle": f"2026-10-{1 + c:02d}T00:00:00Z", "p": p, "a": a, "y": y})
    return out


def test_the_claim_rule_passes_for_a_clearly_better_model_and_fails_for_an_equal_one():
    j = _mod()
    assert j.evaluate(_scored(40, True))["claim"] is True
    assert j.evaluate(_scored(40, False))["claim"] is False
    assert j.evaluate([]) == {"n": 0}


def test_a_cycle_is_scored_only_once_both_fixes_exist(tmp_path):
    j = _mod()
    t = dt.datetime(2026, 10, 12, 0)
    recs = [{"storm_id": "WP282026", "cycle": "2026-10-12T00:00:00Z", "p": 0.4, "a": 0.1}]
    track = {t: 50.0, t + dt.timedelta(hours=24): 85.0}
    out = j.score(recs, t + dt.timedelta(hours=30), best_track=lambda sid: track)
    assert out and out[0]["y"] == 1 and out[0]["dv"] == 35.0
    assert j.score(recs, t + dt.timedelta(hours=20), best_track=lambda sid: track) == []      # not yet matured
    assert j.score(recs, t + dt.timedelta(hours=30), best_track=lambda sid: {t: 50.0}) == []  # no t + 24 h fix
