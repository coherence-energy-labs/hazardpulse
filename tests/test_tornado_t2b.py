"""Tornado program amendment 11 (T2b): the recalibration shadow, its audit, the fit's checks and the prospective rule.

Every check here is shown to FAIL on a broken input as well as to pass on a good one: a shadow whose margin, served or
recalibrated value was altered; a record whose published fields would change; a fit on rows that are not the new
format; a rule fed an equal calibration or too few events.
"""
from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import t2b_shadow as tb
from hazardpulse.tornado import v3_serving as vs

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
pytestmark = pytest.mark.skipif(not all((MODELS / f).exists() for f in tb.CANDIDATES.values()),
                                reason="v3 payloads not present")


def _load(name: str, path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _load("t2b_live_tornado", ROOT / "scripts" / "fetch_and_score_tornado.py")


@pytest.fixture(scope="module")
def suite():
    return vs.V3Suite.load(MODELS)


@pytest.fixture(scope="module")
def prospective():
    return _load("score_tornado_t2b_prospective", ROOT / "scripts" / "score_tornado_t2b_prospective.py")


@pytest.fixture(scope="module")
def verifier():
    return _load("t2b_score_tornado_prospective", ROOT / "scripts" / "score_tornado_prospective.py")


def _artifact(suite, shift=(1.05, 0.2), payload_sha=None) -> dict:
    """A test artifact bound to the served payloads, each map moved off the served one."""
    payloads = {"p60_w": suite.main, "p60": suite.fallback, **suite.products}
    cands = {}
    for k in tb.CANDIDATES:
        cal = payloads[k]["calibration"]
        cands[k] = {"a": float(cal["a"]) * shift[0], "b": float(cal["b"]) + shift[1], "method": "platt",
                    "label": tb.LABELS[k], "payload": tb.CANDIDATES[k],
                    "payload_sha256": payload_sha or vs.model_digest(payloads[k])}
    return tb.validate({"schema": tb.SCHEMA, "program": tb.PROGRAM, "candidates": cands})


def _write(art: dict, path: Path) -> Path:
    path.write_bytes(tb.canonical_bytes(art))
    return path


def _steps():
    """Six storms over three slots, varied enough to give spread margins and some clipped products."""
    t0 = dt.datetime(2026, 4, 27, 18, 0, 39, tzinfo=dt.timezone.utc)
    steps = []
    for k in range(3):
        t = t0 + dt.timedelta(minutes=30 * k)
        storms = []
        for j in range(6):
            storms.append({"id": str(100 + j), "lat": 33.0 + j + 0.05 * k, "lon": -97.0 + 0.1 * k,
                           "ps": 0, "ps_tor": 5 * j + 10 * k, "ps_severe": 10 * j, "maxllaz": 0.002 * j + 0.001 * k,
                           "p98llaz": 0.0015 * j, "mlcape": 400 * j, "mucape": 500 * j, "ebshear": 8 * j,
                           "srh01": 60 * j, "mesh": 0.3 * j, "flash_rate": 4 * j, "size": 100 + 60 * j,
                           "motion_east": 12, "motion_south": -8})
        steps.append({"valid_time": t.strftime("%Y%m%d_%H%M%S UTC"), "storms": storms})
    return steps


class _Inputs:
    def __init__(self, m):
        self._m = m

    def matrix(self):
        return self._m


@pytest.fixture()
def scored(live, suite, monkeypatch):
    """Storm forecasts as the live scorer writes them: +W for five storms, the products clipped where they must be."""
    def fake(lats, lons, times, **kw):
        m = np.array([[1.0, 5.0 + 3 * i] if i % 2 else [0.0, np.nan] for i in range(len(lats))], np.float32)
        return _Inputs(m)
    monkeypatch.setattr(live.nws_live, "live_tor_warning_inputs", fake)
    out = live.score_storms(_steps(), None, None, None, dt.datetime(2026, 4, 27, 19, 10),
                            scoring_tier="tier1_v3", v3_suite=suite)
    live.round_published(out)                     # main()'s order: every published field final, receipts last
    live.refresh_risk_bands(out)
    live.withhold_bands_excluding_probability(out)
    live.stamp_receipts(out, "2026-04-27T19:12:00Z")
    return out


# ---------------------------------------------------------------------------------------------- the shadow
def test_the_margin_path_is_predict_proba_bit_for_bit(suite):
    rng = np.random.default_rng(3)
    for payload in (suite.main, suite.fallback, *suite.products.values()):
        for _ in range(25):
            X = rng.normal(size=(1, len(payload["feature_names"]))) * 30
            X[rng.random(X.shape) < 0.2] = np.nan
            want = lp.predict_proba(payload, X)[0]
            got = tb.proba_from_margin(payload["calibration"], lp.predict_raw(payload, X))[0]
            assert got == want                                   # the same IEEE operations, the same bits


def test_every_v3_storm_carries_the_shadow_and_its_served_value_is_the_published_one(scored, suite, tmp_path, live):
    art = _artifact(suite)
    desc = live.attach_t2b_shadow(scored, suite, _write(art, tmp_path / "a.json"))
    assert desc["status"] == "ok" and desc["artifact_sha256"] == tb.digest(art) and desc["published"] is False
    assert set(desc["candidates"]) == set(tb.CANDIDATES)
    for k, d in desc["candidates"].items():
        assert d["a"] == art["candidates"][k]["a"] and d["b"] == art["candidates"][k]["b"]
    assert all(tb.SHADOW_KEY in s for s in scored)
    clipped = 0
    for s in scored:
        sh, v3 = s[tb.SHADOW_KEY], s["v3"]
        served_by = tb.SERVED_BY[v3["model"]]
        assert sh[served_by]["served"] == v3["probability_60min"]           # exactly: the same computation
        assert sh["probability_60min"] == sh[served_by]["recal"] != v3["probability_60min"]
        assert "p60" in sh                                                  # the fallback, on every storm
        for k, field in tb.PRODUCT_FIELDS.items():
            if v3[field] is None:
                assert k not in sh
            elif k in v3["coherence_clipped"]:
                clipped += 1
                assert v3[field] == v3["probability_60min"] != sh[k]["served"]
            else:
                assert sh[k]["served"] == v3[field]
    assert sum(s["v3"]["model"] == "v3_w" for s in scored) >= 2


def test_published_fields_are_byte_identical_with_and_without_the_shadow(scored, suite, tmp_path, live):
    without = copy.deepcopy(scored)
    live.attach_t2b_shadow(scored, suite, _write(_artifact(suite), tmp_path / "a.json"))
    assert scored != without
    stripped = [{k: v for k, v in s.items() if k != tb.SHADOW_KEY} for s in scored]
    assert json.dumps(stripped, indent=2) == json.dumps(without, indent=2)


def test_the_written_forecast_differs_only_by_the_shadow(scored, suite, tmp_path, live, monkeypatch):
    monkeypatch.setattr(live, "DIST", tmp_path / "dist")
    monkeypatch.setattr(live, "LEDGER_PATH", tmp_path / "dist" / "data" / "ledger.jsonl")
    now = dt.datetime(2026, 4, 27, 19, 12)
    plain = copy.deepcopy(scored)
    live.write_outputs(plain, now, scoring_tier="tier1_v3")
    a = json.loads((tmp_path / "dist" / "data" / "live-tornadoes.json").read_text(encoding="utf-8"))
    desc = live.attach_t2b_shadow(scored, suite, _write(_artifact(suite), tmp_path / "a.json"))
    live.write_outputs(scored, now, scoring_tier="tier1_v3", t2b_shadow=desc)
    b = json.loads((tmp_path / "dist" / "data" / "live-tornadoes.json").read_text(encoding="utf-8"))
    assert b.pop(tb.SHADOW_KEY) == desc
    for s in b["storms"]:
        s.pop(tb.SHADOW_KEY)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_no_artifact_no_shadow_no_error(scored, suite, tmp_path, live):
    before = copy.deepcopy(scored)
    assert live.attach_t2b_shadow(scored, suite, tmp_path / "absent.json") is None
    assert scored == before


def test_a_malformed_artifact_never_breaks_the_forecast(scored, suite, tmp_path, live):
    before = copy.deepcopy(scored)
    bad = _artifact(suite)
    bad["candidates"]["p60_w"]["a"] = -1.0                       # not an increasing map: refused by validate
    (tmp_path / "bad.json").write_bytes(json.dumps(bad).encode())
    assert live.attach_t2b_shadow(scored, suite, tmp_path / "bad.json") is None
    assert scored == before
    with pytest.raises(tb.ArtifactError):
        tb.validate(bad)


def test_an_artifact_fitted_for_another_payload_is_refused(scored, suite, tmp_path, live):
    before = copy.deepcopy(scored)
    desc = live.attach_t2b_shadow(scored, suite, _write(_artifact(suite, payload_sha="0" * 64), tmp_path / "a.json"))
    assert desc["status"] == "refused" and set(desc["refused"]) == set(tb.CANDIDATES) and not desc["candidates"]
    assert scored == before


# ---------------------------------------------------------------------------------------------- the audit
def _record(scored, desc) -> dict:
    return json.loads(json.dumps({"storms": scored, tb.SHADOW_KEY: desc}))        # through the JSON record


def test_the_audit_recomputes_every_shadow_and_catches_each_alteration(scored, suite, tmp_path, live):
    art = _artifact(suite)
    desc = live.attach_t2b_shadow(scored, suite, _write(art, tmp_path / "a.json"))
    rec = _record(scored, desc)
    by_sha = {vs.model_digest(p): p for p in (suite.main, suite.fallback, *suite.products.values())}
    good = tb.recompute(art, by_sha, rec[tb.SHADOW_KEY], rec["storms"])
    assert good["mismatched"] == [] and good["storms"] == len(scored) and good["values_checked"] > 3 * len(scored)

    def broken(mutate):
        r = copy.deepcopy(rec)
        mutate(r)
        return tb.recompute(art, by_sha, r[tb.SHADOW_KEY], r["storms"])["mismatched"]

    s0 = next(i for i, s in enumerate(rec["storms"]) if s["v3"]["model"] == "v3_w")
    assert broken(lambda r: r["storms"][s0][tb.SHADOW_KEY]["p60_w"].update(
        margin=np.nextafter(r["storms"][s0][tb.SHADOW_KEY]["p60_w"]["margin"], 1.0)))          # one ulp of margin
    assert broken(lambda r: r["storms"][s0][tb.SHADOW_KEY]["p60_w"].update(recal=0.5))
    assert broken(lambda r: r["storms"][s0][tb.SHADOW_KEY]["p60"].update(served=0.5))
    assert broken(lambda r: r["storms"][s0][tb.SHADOW_KEY].update(probability_60min=0.123))
    assert broken(lambda r: r["storms"][s0]["v3"].update(probability_60min=0.5))
    assert broken(lambda r: r[tb.SHADOW_KEY]["candidates"]["p60_w"].update(a=1.5))
    assert broken(lambda r: r["storms"][s0].pop(tb.SHADOW_KEY))                     # a storm left without one
    assert broken(lambda r: r["storms"][s0][tb.SHADOW_KEY].pop("p60"))              # a bound candidate missing
    assert broken(lambda r: r["storms"][s0]["v3"]["inputs"].update(
        {n: 1e6 for n in ("p_maxllaz", "p_ps_tor", "p_mlcape")}))


def test_the_record_audit_runs_the_shadow_check(scored, suite, tmp_path, live, monkeypatch):
    audit = _load("t2b_audit_tornado_records", ROOT / "scripts" / "audit_tornado_records.py")
    root = tmp_path / "repo"
    (root / "dist" / "data" / "replay").mkdir(parents=True)
    models = root / "results" / "models"
    models.mkdir(parents=True)
    for f in tb.CANDIDATES.values():
        (models / f).write_bytes((MODELS / f).read_bytes())
    art = _artifact(suite)
    _write(art, models / tb.ARTIFACT_FILE)
    desc = live.attach_t2b_shadow(scored, suite, models / tb.ARTIFACT_FILE)
    rec = _record(scored, desc)
    path = root / "dist" / "data" / "replay" / "to_fcst_20260427_1912.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    res = audit.audit(root)
    assert res["ok"] and res["checked"] == len(scored) == res["matched"]
    assert res["t2b"]["storms"] == len(scored) and res["t2b"]["records"] == 1 and not res["t2b"]["mismatched"]
    rec["storms"][0][tb.SHADOW_KEY][tb.SERVED_BY[rec["storms"][0]["v3"]["model"]]]["recal"] += 1e-6
    path.write_text(json.dumps(rec), encoding="utf-8")
    res = audit.audit(root)
    assert not res["ok"] and res["t2b"]["mismatched"]


# ---------------------------------------------------------------------------------------------- the fit
@pytest.fixture()
def recal(monkeypatch, tmp_path):
    mod = _load("t2b_recalibrate_t", ROOT / "scripts" / "tornado_program" / "t2b_recalibrate.py")
    monkeypatch.setattr(mod, "T2B_DIR", tmp_path / "t2b")
    monkeypatch.setattr(mod, "ROWS", tmp_path / "t2b" / "rows")
    monkeypatch.setattr(mod, "ARTIFACT", tmp_path / "out" / tb.ARTIFACT_FILE)
    monkeypatch.setattr(mod, "FACTS", tmp_path / "out" / "t2b_fit.json")
    monkeypatch.setattr(mod, "MODELS", tmp_path / "out")
    monkeypatch.setattr(mod, "_payload", lambda k: lp.load(MODELS / tb.CANDIDATES[k]))
    return mod


def _fake_rows(recal, days=("20250805", "20250806", "20250807"), n=1500, seed=0, p_ps_after=0.0,
               old_rows=True) -> None:
    rng = np.random.default_rng(seed)
    recal.ROWS.mkdir(parents=True, exist_ok=True)
    main = lp.load(MODELS / vs.MAIN_FILE)
    for ds in days:
        d0 = dt.datetime.strptime(ds, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        t = (d0.timestamp() + rng.uniform(0, 86400, n)).astype(np.int64)
        if ds == "20250805" and not old_rows:
            t = np.maximum(t, int(recal.FIRST_NEW_FORMAT.timestamp()))
        P = rng.gamma(2.0, 20.0, size=(n, 28)).astype(np.float32)
        new = t >= int(recal.FIRST_NEW_FORMAT.timestamp())
        for name in recal.t2.ABSENT:
            P[new, recal.t2.P_NAMES.index(name)] = p_ps_after
        active = (rng.random(n) < 0.1).astype(np.float32)
        rows = {"P": P, "w_active": active, "y": np.zeros(n, np.int8),
                "w_minutes": np.where(active > 0, rng.uniform(0, 45, n), np.nan).astype(np.float32)}
        F = lp.predict_raw(main, recal.t2.columns(main, rows, absent_nan=False))
        p = 1.0 / (1.0 + np.exp(-(0.8 * (F - F.mean()) - 2.5)))
        y60 = (rng.random(n) < p).astype(np.int8)
        y30 = (y60 & (rng.random(n) < 0.6)).astype(np.int8)
        y90 = (y60 | (rng.random(n) < 0.02)).astype(np.int8)
        rows["y"] = y60
        np.savez_compressed(recal.ROWS / f"{ds}.npz", **rows, y3=np.stack([y30, y60, y90], 1), t=t,
                            sid=np.array([f"s{i}" for i in range(n)]))
    log = {"start": days[0], "end": days[-1], "spc_failed_convective_days": [],
           "days": {d: "built" for d in days}}
    (recal.T2B_DIR / "build_log_x.json").write_text(json.dumps(log), encoding="utf-8")


def test_the_fit_writes_an_artifact_the_shadow_accepts(recal, suite):
    _fake_rows(recal)
    assert recal.fit("20250805", "20250807") == 0
    art = tb.load(recal.ARTIFACT)
    assert set(art["candidates"]) == set(tb.CANDIDATES)
    for k, c in art["candidates"].items():
        assert c["a"] > 0 and c["payload_sha256"] == vs.model_digest(lp.load(MODELS / tb.CANDIDATES[k]))
        assert c["events"] > 0 and c["n"] > 0
    facts = json.loads(recal.FACTS.read_text(encoding="utf-8"))
    assert facts["artifact_sha256"] == tb.digest(art) and facts["window"]["first_row"] >= "2025-08-05T20:48:00Z"
    assert facts["rows_before_cut_on_2025_08_05"] > 0 and facts["old_format_rows_before_cut_with_p_ps"] > 0
    assert tb.Shadow.from_suite(suite, art).active
    # the same rows give the same digest; any row changed gives another
    rows, _ = recal.load_window("20250805", "20250807")
    d = recal.data_digest(rows)
    assert d == recal.data_digest({k: v.copy() for k, v in rows.items()})
    rows["y3"][0, 1] ^= 1
    assert recal.data_digest(rows) != d


def test_the_fit_stops_on_rows_that_are_not_the_new_format(recal):
    _fake_rows(recal, p_ps_after=3.0)
    with pytest.raises(SystemExit, match="nonzero"):
        recal.fit("20250805", "20250807")
    assert not recal.ARTIFACT.exists()


def test_the_fit_stops_when_the_cut_separates_nothing(recal):
    _fake_rows(recal, old_rows=False)
    with pytest.raises(SystemExit, match="does not separate"):
        recal.fit("20250805", "20250807")


def test_the_fit_stops_on_a_day_never_built(recal):
    _fake_rows(recal, days=("20250805", "20250807"))
    with pytest.raises(SystemExit, match="never built"):
        recal.fit("20250805", "20250807")


def test_control_loyo_reproduces_and_can_fail(recal, monkeypatch, tmp_path):
    from hazardpulse.tornado import definitive_model as dm
    rng = np.random.default_rng(5)
    oof = tmp_path / "oof"
    oof.mkdir()
    fits = {}
    for k, f in recal.LOYO_SETS.items():
        s = rng.normal(-6, 2, 20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-(s + 1)))).astype(np.int8)
        np.savez(oof / f, score=s, y=y, day=np.zeros(20000, np.int64))
        fits[k] = dm.fit_platt(s, y)
    monkeypatch.setattr(recal, "CONTROL_LOYO", tmp_path / "control.json")

    def payload_with(cal_of):
        def get(k):
            p = lp.load(MODELS / tb.CANDIDATES[k])
            p["calibration"] = dict(p["calibration"], **cal_of(k))
            return p
        return get
    monkeypatch.setattr(recal, "_payload", payload_with(lambda k: {"a": fits[k]["a"], "b": fits[k]["b"]}))
    assert recal.control_loyo(oof) == 0
    monkeypatch.setattr(recal, "_payload", payload_with(lambda k: {"a": fits[k]["a"] + 1e-6, "b": fits[k]["b"]}))
    assert recal.control_loyo(oof) == 1


# ---------------------------------------------------------------------------------------------- the rule
def _rows(n_days=60, per_day=400, seed=0, recal_from="truth", served_scale=0.3, start="2027-03-01"):
    """Live-record rows: y ~ Bernoulli(q); served is q scaled (miscalibrated), recal is ``recal_from``.
    Recal - served Brier is -(1 - scale)^2 E[q^2] = -1.6e-3 at scale 0.3, about 35 standard errors at 24,000 rows."""
    rng = np.random.default_rng(seed)
    d0 = dt.date.fromisoformat(start)
    out = []
    for d in range(n_days):
        day = d0 + dt.timedelta(days=d)
        q = rng.uniform(1e-4, 0.1, per_day)
        y = rng.random(per_day) < q
        served = q * served_scale
        recal = q if recal_from == "truth" else served
        for i in range(per_day):
            t = dt.datetime(day.year, day.month, day.day, 18) + dt.timedelta(minutes=(i * 7) % 600)
            out.append({"candidate": "p60_w", "storm_id": f"{d}-{i}", "valid_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "issued_at": (t + dt.timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "cday": (t - dt.timedelta(hours=12)).date().isoformat(), "y": int(y[i]),
                        "served": float(served[i]), "recal": float(recal[i])})
    return out


def test_the_rule_replaces_a_clearly_worse_calibration(prospective):
    rows = _rows()
    res = prospective.evaluate(rows, decide=True)
    assert res["events"] >= prospective.MIN_EVENTS and res["event_convective_days"] >= prospective.MIN_EVENT_DAYS
    assert res["d_brier_ci95"][1] < 0 and res["d_log_loss"] < 0
    assert res["replace"] is True


def test_the_rule_keeps_an_equal_calibration(prospective):
    res = prospective.evaluate(_rows(recal_from="served"), decide=True)
    assert res["d_brier"] == 0.0 and res["replace"] is False             # not strictly below: never a switch


def test_the_rule_keeps_the_served_calibration_when_it_is_the_better_one(prospective):
    rows = _rows(served_scale=1.0)
    for r in rows:
        r["served"], r["recal"] = r["served"], r["served"] * 2.5
    assert prospective.evaluate(rows, decide=True)["replace"] is False


def test_no_data_and_too_few_events_make_no_claim(prospective):
    assert prospective.evaluate([], decide=True)["claim"] is None
    few = _rows(n_days=5, per_day=200)
    res = prospective.evaluate(few, decide=True)
    assert res["claim"] is None and res["replace"] is False and "below the registered" in res["reason"]


def test_one_forecast_per_storm_per_valid_hour(prospective):
    rows = _rows(n_days=2, per_day=3)
    later = lambda s: (dt.datetime.fromisoformat(s[:-1]) + dt.timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    dup = [dict(r, issued_at=later(r["issued_at"]), served=0.9) for r in rows]   # a later run, same storm-hour
    kept = prospective.first_per_storm_hour(dup + rows)
    assert len(kept) == len(rows) and all(k["served"] != 0.9 for k in kept)   # the first issued wins


def test_no_claim_before_a_look_and_a_look_is_frozen(prospective, tmp_path):
    out = tmp_path / "t2b.json"
    rows = _rows(start="2027-03-01", n_days=100)
    res = prospective.update(rows, dt.datetime(2027, 6, 30), [], "a" * 64, out_path=out)
    assert res["looks"] == {"2027-01-01": res["looks"]["2027-01-01"]}        # only the descriptive read is due
    assert res["looks"]["2027-01-01"]["kind"] == "descriptive"
    assert "replace" not in res["running"]["p60_w"] and res["running"]["p60_w"]["claim"] == "none before a look"
    assert "replace" not in res["looks"]["2027-01-01"]["candidates"]["p60_w"]
    # 3 days after the look date the look is applied, once
    res = prospective.update(rows, dt.datetime(2027, 7, 4, 1), [], "a" * 64, out_path=out)
    look = res["looks"]["2027-07-01"]
    assert look["kind"] == "decision" and look["candidates"]["p60_w"]["replace"] is True
    assert look["candidates"]["p30"]["claim"] is None                          # no p30 rows: no claim
    # a later run with other rows never recomputes it
    later = [dict(r, recal=r["served"]) for r in rows]
    res2 = prospective.update(later, dt.datetime(2027, 8, 1), [], "a" * 64, out_path=out)
    assert res2["looks"]["2027-07-01"] == look
    assert prospective.check_frozen(res, res2) == []
    tampered = copy.deepcopy(res2)
    tampered["looks"]["2027-07-01"]["candidates"]["p60_w"]["replace"] = False
    assert prospective.check_frozen(res, tampered)
    # the 2028 look skips a candidate the 2027 look replaced
    res3 = prospective.update(later, dt.datetime(2028, 7, 5), [], "a" * 64, out_path=out)
    assert res3["looks"]["2028-07-01"]["candidates"]["p60_w"] == {"skipped": "replaced at an earlier look"}


def test_a_look_waits_for_unscorable_records_but_not_forever(prospective, tmp_path):
    out = tmp_path / "t2b.json"
    rows = _rows(start="2027-03-01", n_days=100)
    res = prospective.update(rows, dt.datetime(2027, 7, 5), ["2027-06-30T22:00:00Z"], "a" * 64, out_path=out)
    assert "2027-07-01" not in res["looks"] and "2027-07-01" in res["looks_pending"]
    res = prospective.update(rows, dt.datetime(2027, 7, 15, 1), ["2027-06-30T22:00:00Z"], "a" * 64, out_path=out)
    assert res["looks"]["2027-07-01"]["records_unscorable_excluded"] == 1


def _square(lat, lon, half=0.05):
    ring = [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half]]
    return {"type": "Polygon", "coordinates": [ring]}


def _moving_north(day):
    """An archived track that went north 0.18 deg per 30-min slot from 2025-05-06 18:00:39Z."""
    if day != "20250506":
        return [{"valid_time": "20250507_000039 UTC", "storms": []}]
    t0 = dt.datetime(2025, 5, 6, 18, 0, 39)
    return [{"valid_time": (t0 + dt.timedelta(minutes=30 * k)).strftime("%Y%m%d_%H%M%S UTC"),
             "storms": [{"id": "77", "lat": 35.0 + 0.18 * k, "lon": -97.0, "motion_east": 15.0, "motion_south": 0.0,
                         "geometry": _square(35.0 + 0.18 * k, -97.0)}]} for k in range(4)]


@pytest.mark.parametrize("report, want", [
    ({"lat": 35.24, "lon": -97.0, "time": "2025-05-06T18:41:00Z", "mag": -1}, (0.0, 1.0, 1.0)),   # at 40 min
    ({"lat": 35.54, "lon": -97.0, "time": "2025-05-06T19:30:00Z", "mag": -1}, (0.0, 0.0, 1.0)),   # at 89 min
    ({"lat": 35.16, "lon": -97.0, "time": "2025-05-06T18:20:00Z", "mag": -1}, (1.0, 1.0, 1.0)),   # at 19 min
])
def test_the_verifier_labels_the_products_horizons_from_the_same_call(verifier, report, want):
    art = {"forecast_id": "to_fcst_test", "issued_at": "2025-05-06T18:05:00Z", "storms": [
        {"storm_id": "77", "lat": 35.0, "lon": -97.0, "motion_east": 15.0, "motion_south": 0.0,
         "geometry": _square(35.0, -97.0), "valid_time": "20250506_180039 UTC", "tornado_probability": 0.3,
         "model_version": "tornado_v3-abc123def456"}]}
    lab = verifier.label_storms(art, [report], tracks=verifier.TrackSource(fetch=_moving_north))
    assert (lab["y_storm_30"][0], lab["y_true"][0], lab["y_storm_90"][0]) == want
    assert lab["storm_times"] == [dt.datetime(2025, 5, 6, 18, 0, 39)]
