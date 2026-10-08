"""TC1 live: the forecast is the backtest's pass over the season's decks, issued only after t + 3 h 30, from saved
models that must match the selection on disk."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from hazardpulse.hurricane import consensus as cs
from hazardpulse.hurricane import tc1_live as live

T0 = dt.datetime(2026, 9, 1, 0)


def _decks(seed=3):
    """Two 2026 storms with five members and OFCL; every member is truth plus noise."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(2):
        start = T0 + dt.timedelta(days=2 * s)
        truth = {start + dt.timedelta(hours=6 * c): (15.0 + 0.2 * c, -45.0 - 0.4 * c, 50.0 + 1.5 * c) for c in range(60)}
        fc, carq = {}, {}
        for c in range(36):
            t = start + dt.timedelta(hours=6 * c)
            carq[t] = truth[t]
            for m, sd in (("AVNI", 80.0), ("GDMI", 30.0), ("HCCA", 50.0), ("DSHP", 60.0), ("LGEM", 40.0), ("OFCL", 35.0)):
                by = {}
                for lead in cs.LEADS:
                    la, lo, v = truth[t + dt.timedelta(hours=lead)]
                    e, n = rng.normal(0, sd * lead / 48, 2)
                    by[lead] = (*cs.shift_km(la, lo, e, n), v + rng.normal(0, sd / 8))
                fc[(t, m)] = by
        out.append(cs.StormDeck.from_dicts(f"al{s + 1:02d}2026", "AL", fc, carq))
    return out


def _state(selection_path, cfg_track, cfg_int):
    sel = {"track": {"chosen": {"config": json.loads(json.dumps(cs.OnlineConsensus(cfg_track, "track").to_dict()["config"]))}},
           "intensity": {"chosen": {"config": cs.OnlineConsensus(cfg_int, "intensity").to_dict()["config"]}}}
    selection_path.write_text(json.dumps(sel), encoding="utf-8")
    models = {}
    for kind, cfg in (("track", cfg_track), ("intensity", cfg_int)):
        for product, off in (("TC1", False), ("TC1+O", True)):
            c = cs.Config(**{**cs.OnlineConsensus(cfg, kind).to_dict()["config"], "include_official": off})
            models[f"{product}/{kind}"] = cs.OnlineConsensus(c, kind).to_dict()
    return {"season_end": 2025, "selection_sha256": live.sha256(selection_path), "models": models}


def test_the_live_forecast_is_the_backtests_pass_and_waits_for_t_plus_3h30(tmp_path):
    cfg = cs.Config(half_life_days=60.0, shrink=0.5)
    state = _state(tmp_path / "sel.json", cfg, cfg)
    decks = _decks()
    now = T0 + dt.timedelta(days=5, hours=3, minutes=29)        # 5 d 00Z is not yet issuable; 4 d 18Z is
    got = live.forecast(state, decks, ["al012026", "al022026"], now)
    t = T0 + dt.timedelta(days=4, hours=18)
    assert got["al012026"]["cycle"] == t.strftime("%Y-%m-%dT%H:00:00Z")
    whole = cs.run(decks, cfg, "track")
    lat, lon = whole[("al012026", t, 72)][0]
    assert got["al012026"]["TC1"]["72"]["lat"] == round(lat, 2) and got["al012026"]["TC1"]["72"]["lon"] == round(lon, 2)
    v = cs.run(decks, cfg, "intensity")[("al012026", t, 72)][0]
    assert got["al012026"]["TC1"]["72"]["vmax_kt"] == round(v, 1)
    for lead, rec in got["al012026"]["TC1"].items():                          # TC1 never reads the official forecast
        assert "OFCL" not in (rec["track_weights"] or {}) and "OFCL" not in (rec["intensity_weights"] or {})
    plus = got["al012026"]["TC1+O"]["72"]
    assert (plus["lat"], plus["lon"]) != (got["al012026"]["TC1"]["72"]["lat"], got["al012026"]["TC1"]["72"]["lon"])
    assert got["al012026"]["OFCL"]["72"]["lat"] is not None                  # NHC's forecast beside ours
    later = live.forecast(state, decks, ["al012026"], now + dt.timedelta(minutes=2))
    assert later["al012026"]["cycle"] == (t + dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00Z")


def test_saved_models_from_another_selection_or_season_are_refused(tmp_path):
    cfg = cs.Config(half_life_days=60.0, shrink=0.5)
    state = _state(tmp_path / "sel.json", cfg, cfg)
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    assert live.load_state(p, tmp_path / "sel.json")["season_end"] == 2025
    sel = tmp_path / "sel.json"
    sel.write_bytes(sel.read_bytes().replace(b",", b",\r\n  ") + b"\r\n")    # a Windows checkout's bytes: same content
    assert live.load_state(p, sel)["season_end"] == 2025
    doc = json.loads(sel.read_text(encoding="utf-8"))
    doc["track"]["chosen"]["config"]["half_life_days"] = 20.0
    sel.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(live.StateError):
        live.load_state(p, sel)                                                 # another selection
    with pytest.raises(live.StateError):
        live.forecast(dict(state, season_end=2024), _decks(), ["al012026"], T0 + dt.timedelta(days=3))
    with pytest.raises(live.StateError):
        live.load_state(tmp_path / "absent.json", tmp_path / "sel.json")


def test_the_committed_state_is_the_committed_selection():
    """The file the scorer loads was made from the selection that is committed (and refuses otherwise)."""
    if not live.STATE.exists():
        pytest.skip("no saved TC1 models")
    doc = live.load_state()
    assert set(doc["models"]) == {f"{p}/{k}" for p in live.PRODUCTS for k in ("track", "intensity")}
    assert all(len(k) == 2 for m in doc["models"].values() for k, _ in m["state"])     # no past storm kept


def test_issue_until_is_the_last_synoptic_time_three_and_a_half_hours_old():
    assert live.issue_until(dt.datetime(2026, 10, 8, 15, 29)) == dt.datetime(2026, 10, 8, 6)
    assert live.issue_until(dt.datetime(2026, 10, 8, 15, 30)) == dt.datetime(2026, 10, 8, 12)
    assert live.issue_until(dt.datetime(2026, 10, 9, 2, 0)) == dt.datetime(2026, 10, 8, 18)
