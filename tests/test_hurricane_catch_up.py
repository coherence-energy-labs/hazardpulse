"""Amendment 7 in the live scorer (docs/HURRICANE_RI_V9_PROGRAM.md): a cycle no run recorded after its
advisory is forecast in shadow by the next run (rule 2), by the same computation a live run makes; and a
shadow made without an IR image never says its IR inputs were "ok".

The decks are synthetic: the real AL01 2026 fixture deck moved in time so its cycles fall after the
test's start (2026-10-04 00Z), read by runs at times that are NOT on the hour.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane import atcf, cycle_records as cr, ri_v10

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "hurricane_v9"
pytestmark = pytest.mark.skipif(not ri_v10.MODEL_PATH.exists(), reason="v10 artifact not built")

SID = "AL012026"
SHIFT = dt.datetime(2026, 10, 5, 0) - dt.datetime(2026, 6, 16, 18)     # fixture 2026061618 -> 2026-10-05 00Z
T00, T06 = dt.datetime(2026, 10, 5, 0), dt.datetime(2026, 10, 5, 6)


def _load(name, rel):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fs():
    return _load("fetch_and_score_catch_up_t", "scripts/fetch_and_score.py")


@pytest.fixture(scope="module")
def models():
    from hazardpulse.hurricane import ri_v9
    v9 = dict(zip(("payload", "model_version"), ri_v9.load()))
    v10 = dict(zip(("artifact", "model_version"), ri_v10.load()))
    ch = {"ri_v10_2_shadow": dict(zip(("artifact", "model_version"), ri_v10.load(ri_v10.V10_2_PATH)))}
    if ri_v10.V10_3_PATH.exists():
        ch["ri_v10_3_shadow"] = dict(zip(("artifact", "model_version"), ri_v10.load(ri_v10.V10_3_PATH)))
    return v9, v10, ch


def _deck(last: dt.datetime) -> list:
    """The fixture deck moved by SHIFT, as it stands once cycle ``last`` is in it."""
    recs = atcf.parse_atcf_deck(gzip.decompress((FIX / "aal012026.dat.gz").read_bytes()).decode("utf-8", "replace"))
    return [dataclasses.replace(r, cycle=r.cycle + SHIFT) for r in recs if r.cycle + SHIFT <= last]


def _orig(cycle: dt.datetime) -> str:
    return (cycle - SHIFT).strftime("%Y%m%d%H")


def _ships(sid, cycle):
    name = f"{_orig(cycle)[2:]}AL0126_ships.txt"
    p = FIX / name
    return (p.read_text(encoding="utf-8"), name) if p.exists() else (None, "absent (HTTPError)")


def _ir(sid, cycle, records):
    """The fixture's real crops for the moved cycle (both images read)."""
    from hazardpulse.hurricane import ir_features

    def crop(tag):
        p = FIX / "ir" / f"AL012026_{_orig(cycle)}_{tag}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return {"counts": z["counts"], "lat": z["lat"], "lon": z["lon"], "centre": tuple(z["centre"])}
    return ir_features.features(crop("p2"), crop("m4"))


def test_a_missed_cycle_is_due_only_between_t_plus_3_h_30_and_t_plus_12_h():
    cycles = [T00 - dt.timedelta(hours=6), T00, T06]
    now = dt.datetime(2026, 10, 5, 6, 47, 13)                               # a run at 06Z + 47 min
    assert cr.due_catch_up_cycles(cycles, now) == [T00]                     # 18Z is 12 h 47 old; 06Z preliminary
    assert cr.due_catch_up_cycles(cycles, now, have={T00}) == []           # it has a test record already
    t18 = T00 - dt.timedelta(hours=6)
    assert cr.due_catch_up_cycles(cycles, T00 + dt.timedelta(hours=3, minutes=29, seconds=59)) == [t18]
    assert cr.due_catch_up_cycles(cycles, T00 + dt.timedelta(hours=3, minutes=30)) == [t18, T00]
    assert cr.due_catch_up_cycles(cycles, T00 + dt.timedelta(hours=12)) == [T06]          # 00Z: t + 12 h passed
    assert cr.due_catch_up_cycles([dt.datetime(2026, 10, 3, 18)], dt.datetime(2026, 10, 4, 1)) == []   # before start


def test_the_next_run_forecasts_the_missed_cycle_exactly_as_a_live_run_would(fs, models):
    """EP15 and EP18 at 10-04 00Z: the scheduler dropped the run, and the next one read only the latest
    cycle. Now a run at 06Z + 47 min (its own cycle still preliminary) also forecasts 00Z in shadow."""
    v9, v10, ch = models
    now = dt.datetime(2026, 10, 5, 6, 47, 13)
    deck = _deck(T06)
    live = fs.build_live_case(SID, deck)
    assert live["issue_time"] == T06.isoformat()
    cases = fs.catch_up_cases([live], {SID: deck}, now, have=set())
    assert [c["issue_time"] for c in cases] == [T00.isoformat()]
    reads = {"ships": [], "adeck": 0}

    def ships(sid, cyc):
        reads["ships"].append(cyc)
        return _ships(sid, cyc)

    def adeck(sid):
        reads["adeck"] += 1
        return deck
    (rec,) = fs.catch_up_forecasts(cases, now, v9, v10, ch, ships_raw_fetcher=ships, adeck_fetcher=adeck, ir_fetcher=_ir)
    assert rec["catch_up"] is True and rec["published"] is False
    assert rec["lag_hours"] == pytest.approx(6.7869, abs=1e-4) and rec["issue_time"] == T00.isoformat()
    assert reads == {"ships": [T00], "adeck": 1}                              # one read of each, as live

    # a live run at 00Z + 3 h 30 would have seen the deck only up to 00Z: the same forecast, to the bit
    deck00 = _deck(T00)
    live00 = fs.build_live_case(SID, deck00)
    want = fs.shadow_forecasts(live00, v9, v10, _ships, lambda s: deck00, ch, _ir)
    keys = fs._shadow_keys(v9, v10, ch)
    assert keys and all(rec[k] == want[k] for k in keys)
    exp = next(c for c in json.loads((FIX / "expected.json").read_text(encoding="utf-8"))["cases"]
               if c["dtg"] == _orig(T00))
    for k, p in exp["v10"].items():
        assert rec["ri_v10_shadow"]["probabilities"][k] == pytest.approx(p, abs=5e-5)    # == the lab's curve


def test_a_catch_up_without_its_ships_text_follows_the_live_fallback(fs, models):
    v9, v10, ch = models
    now = dt.datetime(2026, 10, 5, 6, 47, 13)
    deck = _deck(T06)
    cases = fs.catch_up_cases([fs.build_live_case(SID, deck)], {SID: deck}, now, have=set())
    none = lambda sid, cyc: (None, "absent (HTTPError)")                    # noqa: E731
    (rec,) = fs.catch_up_forecasts(cases, now, v9, v10, ch, ships_raw_fetcher=none, adeck_fetcher=lambda s: deck,
                                   ir_fetcher=_ir)
    live = fs.shadow_forecasts(cases[0], v9, v10, none, lambda s: deck, ch, _ir)
    for k in fs._shadow_keys(v9, v10, ch):
        assert rec[k] == live[k] and rec[k]["ships_text"] == "ships_text_absent (HTTPError)"
        assert rec[k]["dtops_pct"] is None


def test_catch_up_records_are_kept_hashed_audited_and_then_count_as_the_test_record(fs, models, tmp_path, monkeypatch):
    v9, v10, ch = models
    monkeypatch.setattr(fs, "DIST", tmp_path)
    (tmp_path / "data").mkdir()
    now = dt.datetime(2026, 10, 5, 6, 47, 13)
    deck = _deck(T06)
    live = fs.build_live_case(SID, deck)
    assert fs.recorded_test_cycles(now) == set()
    cases = fs.catch_up_cases([live], {SID: deck}, now, fs.recorded_test_cycles(now))
    catch_up = fs.catch_up_forecasts(cases, now, v9, v10, ch, _ships, lambda s: deck, _ir)
    storm = {"storm_id": SID, "issue_time": live["issue_time"], "ri_probability": 0.1, "model_version": "m",
             **fs.shadow_forecasts(live, v9, v10, _ships, lambda s: deck, ch, _ir)}
    fs.write_outputs([storm], now, "m", note="n", catch_up=catch_up)
    fid = "hu_fcst_20261005_0647"
    art = json.loads((tmp_path / "data" / "replay" / f"{fid}.json").read_text(encoding="utf-8"))
    assert [s["issue_time"] for s in art["storms"]] == [T06.isoformat()]                # never published
    assert [s["issue_time"] for s in art[cr.CATCH_UP_KEY]] == [T00.isoformat()]
    live_file = json.loads((tmp_path / "data" / "live-storms.json").read_text(encoding="utf-8"))
    assert live_file[cr.CATCH_UP_KEY] == art[cr.CATCH_UP_KEY]
    row = json.loads((tmp_path / "data" / "hurricane-ledger.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["catch_up"][0]["issue_time"] == T00.isoformat()
    audit = _load("audit_catch_up_t", "scripts/audit_hurricane_records.py")
    root = tmp_path / "root"
    (root / "dist" / "data" / "replay").mkdir(parents=True)
    (root / "dist" / "data" / "replay" / f"{fid}.json").write_text(json.dumps(art), encoding="utf-8")
    (root / "dist" / "data" / "hurricane-ledger.jsonl").write_text(
        (tmp_path / "data" / "hurricane-ledger.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (root / "results" / "models").mkdir(parents=True)
    for p in (ri_v10.MODEL_PATH, ri_v10.V10_2_PATH):
        (root / "results" / "models" / p.name).write_bytes(p.read_bytes())
    res = audit.audit(root)
    assert res["ok"] and res["ledger"]["content_matches"] == 1
    assert res["recompute"]["matched"] >= 4                                 # live + catch-up v10.1 and v10.2
    art[cr.CATCH_UP_KEY][0]["ri_v10_shadow"]["probability"] = 0.99          # an edit to a catch-up record shows
    (root / "dist" / "data" / "replay" / f"{fid}.json").write_text(json.dumps(art), encoding="utf-8")
    assert audit.audit(root)["ledger"]["content_mismatches"] == [fid]

    # the next run (12Z + 52 min) finds 00Z recorded: no second catch-up of it
    later = dt.datetime(2026, 10, 5, 12, 52, 40)
    assert (SID, T00) in fs.recorded_test_cycles(later)
    deck12 = _deck(dt.datetime(2026, 10, 5, 12))
    nxt = fs.catch_up_cases([fs.build_live_case(SID, deck12)], {SID: deck12}, later, fs.recorded_test_cycles(later))
    assert [c["issue_time"] for c in nxt] == [T06.isoformat()]             # 06Z: its only record was preliminary

    # and the prospective test takes it as the cycle's test record
    v9p = _load("v9p_catch_up_t", "scripts/score_hurricane_v9_prospective.py")
    (rec,) = [r for r in v9p.collect(tmp_path / "data" / "replay") if r["cycle"] == "2026-10-05T00:00:00Z"]
    assert rec["catch_up"] is True and rec["lag_hours"] == pytest.approx(6.7869, abs=1e-4)


def test_the_site_build_keeps_catch_up_records_in_the_forecast_file(tmp_path, monkeypatch):
    bsa = _load("bsa_catch_up_t", "scripts/build_site_artifacts.py")
    dist = tmp_path / "dist"
    (dist / "data" / "replay").mkdir(parents=True)
    (dist / "data" / "evidence").mkdir(parents=True)
    for name in ("LIVE_STORMS_PATH", "LIVE_PULSE_PATH", "REPLAY_DIR", "REPLAY_INDEX_PATH", "LIVE_TORNADOES_PATH"):
        monkeypatch.setattr(bsa, name, {"LIVE_STORMS_PATH": dist / "data" / "live-storms.json",
                                        "LIVE_PULSE_PATH": dist / "data" / "live-pulse.json",
                                        "REPLAY_DIR": dist / "data" / "replay",
                                        "REPLAY_INDEX_PATH": dist / "data" / "evidence" / "replay-index.json",
                                        "LIVE_TORNADOES_PATH": dist / "data" / "live-tornadoes.json"}[name])
    monkeypatch.setattr(bsa, "DIST", dist)
    cu = [{"storm_id": SID, "issue_time": T00.isoformat(), "catch_up": True}]
    (dist / "data" / "live-storms.json").write_text(json.dumps({
        "updated_at": "2026-10-05T06:47:13Z", "forecast_id": "hu_fcst_20261005_0647", "n_active_storms": 0,
        "storms": [], cr.CATCH_UP_KEY: cu}), encoding="utf-8")
    (dist / "data" / "live-pulse.json").write_text(json.dumps({"hazards": [{"key": "hu"}]}), encoding="utf-8")
    bsa._ensure_live_publish_artifacts()
    art = json.loads((dist / "data" / "replay" / "hu_fcst_20261005_0647.json").read_text(encoding="utf-8"))
    assert art[cr.CATCH_UP_KEY] == cu and art["storms"] == []


# ------------------------------------------------------------------------------------------------
# IR status (finding: Rachel 2026-10-03 18Z, 14 of 14 IR features NaN, recorded "ir": "ok")
# ------------------------------------------------------------------------------------------------

def test_a_forecast_made_before_the_image_exists_says_its_ir_is_missing(fs, models, monkeypatch):
    from hazardpulse.hurricane import ir_source
    v9, v10, ch = models
    if "ri_v10_3_shadow" not in ch:
        pytest.skip("v10.3 artifact not built")
    deck = _deck(T00)
    lat, lon = np.linspace(72.7, -72.7, 2001), np.linspace(-180.0, 179.9, 5000)
    counts = np.random.default_rng(3).integers(60, 230, size=(lat.size, lon.size)).astype(np.uint8)
    future = T00 + dt.timedelta(hours=2)

    def fetch(hour):                      # run at t + 46 min: the t + 2 h image does not exist yet
        return (None, None, None, None) if hour == future else ("key-" + hour.strftime("%H"), counts, lat, lon)
    monkeypatch.setattr(ir_source, "fetch_image", fetch)
    fs._IR_IMAGES.clear()
    case = fs.build_live_case(SID, deck)
    out = fs.shadow_forecasts(case, None, None, _ships, lambda s: deck, {"ri_v10_3_shadow": ch["ri_v10_3_shadow"]})
    sh = out["ri_v10_3_shadow"]
    assert all(v is None for v in sh["ir_inputs"].values())                 # every IR feature NaN ...
    assert sh["ir"].startswith("missing: 0 of 14 IR features") and "t + 2 h (2026-10-05T02:00Z)" in sh["ir"]
    assert sh["ir_images"]["p2"]["key"] is None and sh["ir_images"]["m4"]["key"] == "key-20"
    fs._IR_IMAGES.clear()
    monkeypatch.setattr(ir_source, "fetch_image", lambda hour: ("k", counts, lat, lon))
    ok = fs.shadow_forecasts(case, None, None, _ships, lambda s: deck, {"ri_v10_3_shadow": ch["ri_v10_3_shadow"]})
    assert ok["ri_v10_3_shadow"]["ir"] == "ok"                               # both images: every feature
    fs._IR_IMAGES.clear()


def test_the_ir_status_counts_what_was_computed():
    fs = _load("fetch_and_score_ir_status_t", "scripts/fetch_and_score.py")
    from hazardpulse.hurricane import ir_features
    full = {k: 1.0 for k in ir_features.IR_NAMES}
    nan = {k: math.nan for k in ir_features.IR_NAMES}
    trend_missing = {k: (math.nan if k.startswith("d_") else 1.0) for k in ir_features.IR_NAMES}
    read = {"p2": {"hour": "2026-10-03T20:00Z", "key": None}, "m4": {"hour": "2026-10-03T14:00Z", "key": None}}
    assert fs.ir_status(full) == "ok"
    assert fs.ir_status(nan, read) == ("missing: 0 of 14 IR features; no image read: t + 2 h (2026-10-03T20:00Z), "
                                       "t - 4 h (2026-10-03T14:00Z)")
    assert fs.ir_status(nan, {"centre": "no CARQ position at t"}).endswith("no CARQ position at t")
    m4_absent = {"p2": {"hour": "h2", "key": "k"}, "m4": {"hour": "h4", "key": None}}
    assert fs.ir_status(trend_missing, m4_absent) == "partial: 10 of 14 IR features; no image at t - 4 h (h4)"
