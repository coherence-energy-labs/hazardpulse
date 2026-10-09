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
