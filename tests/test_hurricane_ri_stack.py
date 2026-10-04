"""The NOAA RI-aid stack: fitter == served path, the pre-registered program's guards, routing.

Pins:
* the served path's representation is the fitter's (logits, design columns, intercept last);
* a served artifact round-trips through its file to the fitter's probabilities, and its
  identity is bound to its content (a CRLF checkout is the same model);
* tampered artifacts are refused; the pattern fallback picks the documented sub-pattern;
* the parsimony rule replaces the incumbent only on an interval wholly below 0;
* --phase final reads 2025 once: it refuses without a selection, and refuses a second time;
* the committed selection, final report and artifact agree with each other;
* fetch_and_score routes NHC storms with SHIPS text to the stack, every other case to v8.2,
  and a trust calibrator touches only its own model's storms;
* the prospective scorer's calibration pool is deterministic when two models serve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hazardpulse.hurricane import ri_model  # noqa: E402
from hazardpulse.hurricane import ri_stack as rs  # noqa: E402
from hazardpulse.hurricane import ships_text as st  # noqa: E402

import benchmark_hurricane_vs_ships as bvs  # noqa: E402
import hurricane_ri_stack as hrs  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "ships_text"
NOLO = "26100212EP1526_ships.txt"   # SHIPS-RII 19.3% (integer line 19%), DTOPS 15.0%


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic development rows
# ---------------------------------------------------------------------------

def _rows(n: int = 1500, seed: int = 7, dtop_missing: float = 0.15) -> list[dict]:
    rng = np.random.default_rng(seed)
    z = rng.normal(-3.0, 1.4, n)                     # latent RI propensity
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)

    def aid(noise: float) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(z + rng.normal(0, noise, n))))
        return np.clip(np.round(p * 100), 0, 100).astype(int)

    riod, riol, riob, dtop = aid(1.0), aid(1.3), aid(1.6), aid(0.6)
    rows = []
    for i in range(n):
        aids = {"RIOD": int(riod[i]), "RIOL": int(riol[i]), "RIOB": int(riob[i]),
                "RIOC": int(round((riod[i] + riol[i] + riob[i]) / 3))}
        if rng.random() > dtop_missing:
            aids["DTOP"] = int(dtop[i])
        rows.append({"aids": aids, "v82": float(1 / (1 + np.exp(-(z[i] + rng.normal(0, 1.2))))),
                     "basin": "AL" if i % 2 else "EP", "y": int(y[i]), "sid": f"S{i // 25:04d}",
                     "season": 2020 + (i % 5)})
    return rows


# ---------------------------------------------------------------------------
# Representation: served path == fitter
# ---------------------------------------------------------------------------

def test_served_logits_are_the_benchmark_logits():
    pct = np.array([0, 1, 7, 50, 99, 100])
    assert np.array_equal(rs.aid_logit(pct), bvs.logit_of(pct / 100.0, True))
    p = np.array([1e-9, 0.01, 0.5, 0.999999999])
    assert np.array_equal(rs.v82_logit(p), bvs.logit_of(p, False))


def test_design_columns_follow_the_pattern_and_basin_rule():
    rows = _rows(4)
    assert rs.column_names(("RIOD", "DTOP"), True) == ["logit_RIOD", "logit_DTOP", "basin_al", "basin_al_x_logit_DTOP"]
    assert rs.column_names(("RIOD", "RIOL"), True) == ["logit_RIOD", "logit_RIOL", "basin_al"]
    full = [r for r in rows if "DTOP" in r["aids"]]
    cols = rs.design_columns(("RIOD", "DTOP"), True, full)
    al = np.array([1.0 if r["basin"] == "AL" else 0.0 for r in full])
    assert np.array_equal(cols[3], al * cols[1])


def test_sub_patterns_order_is_size_then_input_order():
    subs = rs.sub_patterns(("RIOD", "RIOL", "RIOB", "DTOP"))
    assert subs[0] == ("RIOD", "RIOL", "RIOB", "DTOP")
    assert subs[1:4] == [("RIOD", "RIOL", "RIOB"), ("RIOD", "RIOL", "DTOP"), ("RIOD", "RIOB", "DTOP")]
    assert subs[-1] == ("RIOD",) and len(subs) == 8
    assert rs.sub_patterns(("RIOL", "DTOP")) == []      # no RIOD, no fit


# ---------------------------------------------------------------------------
# Pattern fits, fallback, and the artifact round trip
# ---------------------------------------------------------------------------

def _pool_artifact(cand: str, train: list[dict], drop: tuple[str, ...] = ()) -> tuple[dict, hrs.PoolFitter]:
    fitter = hrs.PoolFitter(cand, train)
    for pattern in rs.all_patterns(rs.CANDIDATES[cand]["inputs"]):
        if pattern not in drop:
            fitter.fit(pattern)
    fitter.fits = {p: f for p, f in fitter.fits.items() if p not in drop}
    spec = rs.CANDIDATES[cand]
    payload = {"schema": rs.SCHEMA, "name": rs.MODEL_NAME, "candidate": cand, "kind": spec["kind"],
               "logit_clip": [rs.PCT_CLIP, 1 - rs.PCT_CLIP], "inputs": list(spec["inputs"]),
               "basin": spec["basin"], "patterns": fitter.fitted(), "training": {"seasons": [2020, 2024], "n": len(train)},
               "description": spec["description"]}
    return payload, fitter


@pytest.mark.parametrize("cand", ["D", "E", "F"])
def test_artifact_round_trip_reproduces_the_fitter(tmp_path, cand):
    rows = _rows()
    train, test = rows[:1000], rows[1000:]
    payload, fitter = _pool_artifact(cand, train)
    p_fit, info = fitter.predict(test)
    path = tmp_path / "stack.json"
    version = rs.save_artifact(payload, path)
    loaded, loaded_version = rs.load_artifact(path)
    assert loaded_version == version and version.startswith(rs.MODEL_NAME + "-")
    served = np.array([rs.predict(loaded, r["aids"], r["basin"], r["v82"])[0] for r in test])
    assert np.max(np.abs(served - p_fit)) < 1e-12
    assert info["own_patterns"] and info["fallback_cycles"] == 0


def test_model_version_is_bound_to_content_not_line_endings(tmp_path):
    payload, _ = _pool_artifact("D", _rows()[:800])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    version = rs.save_artifact(payload, a)
    b.write_bytes(a.read_bytes().replace(b"\n", b"\r\n"))
    assert rs.model_version_of(b) == version
    payload["patterns"][0]["coef"][0] += 1e-9
    assert rs.save_artifact(payload, b) != version


def test_the_pattern_fallback_uses_the_largest_fitted_sub_pattern(tmp_path):
    rows = _rows()
    full = ("RIOD", "RIOL", "RIOB", "DTOP")
    payload, fitter = _pool_artifact("D", rows[:1000], drop=(full,))
    path = tmp_path / "s.json"
    rs.save_artifact(payload, path)
    loaded, _ = rs.load_artifact(path)
    row = next(r for r in rows[1000:] if "DTOP" in r["aids"])
    prob, info = rs.predict(loaded, row["aids"], row["basin"])
    assert info["own_pattern"] == list(full) and info["fallback"] is True
    assert info["pattern"] == ["RIOD", "RIOL", "RIOB"]   # same size ties keep the earliest inputs
    fit = next(f for f in loaded["patterns"] if f["pattern"] == ["RIOD", "RIOL", "RIOB"])
    cols = rs.design_columns(("RIOD", "RIOL", "RIOB"), False, [row])
    assert prob == pytest.approx(float(bvs.apply_stack(np.asarray(fit["coef"]), cols)[0]), abs=1e-15)


def test_a_pattern_with_too_few_events_is_refused_and_falls_back():
    rows = _rows()
    for r in rows:   # DTOP only where nothing happened: a DTOP pattern cannot be fitted
        if r["y"] == 1:
            r["aids"].pop("DTOP", None)
    fitter = hrs.PoolFitter("D", rows[:1000])
    test = [r for r in rows[1000:] if "DTOP" in r["aids"]]
    p, info = fitter.predict(test)
    assert info["fallback_cycles"] == len(test) and "RIOD+RIOL+RIOB+DTOP" in info["refused_patterns"]
    assert set(info["used_patterns"]) == {"RIOD+RIOL+RIOB"}
    assert np.all((p > 0) & (p < 1))


def test_raw_candidates_fall_back_to_ships_rii():
    rows = [{"aids": {"RIOD": 30, "DTOP": 80, "RIOC": 40}, "basin": "AL", "v82": None},
            {"aids": {"RIOD": 30}, "basin": "EP", "v82": None}]
    pa, ia = hrs.raw_forecast("A", rows)
    pc, ic = hrs.raw_forecast("C", rows)
    assert list(pa) == [0.80, 0.30] and ia["fallback_cycles"] == 1
    assert list(pc) == [0.40, 0.30] and ic["fallback_cycles"] == 1
    payload = {"kind": "raw_aid", "aid": "DTOP", "fallback": "RIOD"}
    assert rs.predict(payload, {"RIOD": None, "DTOP": None}, "AL") is None


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.update(kind="raw_aid"), "candidate D is logit_pool"),
    (lambda p: p.update(logit_clip=[0.01, 0.99]), "logit_clip"),
    (lambda p: p["patterns"][0].update(columns=["x"]), "columns"),
    (lambda p: p["patterns"][0]["coef"].__setitem__(0, float("nan")), "non-finite"),
    (lambda p: p.update(patterns=[f for f in p["patterns"] if f["pattern"] != ["RIOD"]]), "minimal pattern"),
    (lambda p: p.update(inputs=["RIOD", "DTOP"]), "inputs differ"),
    (lambda p: p.update(schema="other/1"), "unknown schema"),
])
def test_tampered_artifacts_are_refused(tmp_path, mutate, match):
    payload, _ = _pool_artifact("D", _rows()[:800])
    bad = json.loads(json.dumps(payload))
    mutate(bad)
    with pytest.raises(rs.StackArtifactError, match=match):
        rs.validate_artifact(bad)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad, allow_nan=True), encoding="utf-8")
    with pytest.raises(rs.StackArtifactError):
        rs.load_artifact(path)


# ---------------------------------------------------------------------------
# The parsimony rule
# ---------------------------------------------------------------------------

def test_parsimony_replaces_only_on_an_interval_wholly_below_zero():
    rows = _rows(2000, seed=11)
    y = np.array([r["y"] for r in rows], dtype=float)
    groups = np.array([r["sid"] for r in rows])
    truth_p = np.clip(np.array([r["aids"]["RIOD"] for r in rows]) / 100.0, 0.005, 0.995)
    good = np.where(y == 1, 0.6, 0.02)          # far better than any aid: must replace
    preds = {c: (truth_p, True) for c in "ABC"}
    preds.update({"D": (good, False), "E": (good, False), "F": (good, False)})
    pooled = {c: {"log_loss": ev_ll(y, *preds[c])} for c in preds}
    out = hrs.parsimony(pooled, y, preds, groups, reps=200)
    assert out["choice"] == "D"          # D replaces A; E and F tie with D -> D stays
    assert [s["replaces_incumbent"] for s in out["steps"][1:]] == [True, False, False]
    same = {c: (truth_p, c in "ABC") for c in "ABCDEF"}
    out = hrs.parsimony({c: {"log_loss": ev_ll(y, *same[c])} for c in same}, y, same, groups, reps=200)
    assert out["choice"] == "A"          # an exact tie: the simpler (and earlier letter) stays


def ev_ll(y, p, q):
    from hazardpulse.hurricane import ri_evaluation as ev
    return ev.log_loss(y, bvs._clip(p, q))


# ---------------------------------------------------------------------------
# Read once
# ---------------------------------------------------------------------------

def _final_args(tmp_path, **kw):
    base = dict(phase="final", cache_root=tmp_path / "no-cache", reps=2000, no_download=True,
                rehearse=False, out_dir=None, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_final_refuses_without_a_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(hrs, "SELECTION_PATH", tmp_path / "missing_selection.json")
    monkeypatch.setattr(hrs, "FINAL_PATH", tmp_path / "final.json")
    with pytest.raises(SystemExit, match="does not exist"):
        hrs.phase_final(_final_args(tmp_path))


def test_final_refuses_a_second_reading(tmp_path, monkeypatch):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"phase": "select", "choice": "A"}), encoding="utf-8")
    final = tmp_path / "final.json"
    final.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hrs, "SELECTION_PATH", sel)
    monkeypatch.setattr(hrs, "FINAL_PATH", final)
    with pytest.raises(SystemExit, match="read once"):
        hrs.phase_final(_final_args(tmp_path))

    class ReachedTheData(Exception):
        pass

    def stop(*_a, **_k):
        raise ReachedTheData

    monkeypatch.setattr(hrs, "load_model_c", stop)
    final.unlink()   # the control: without the final file the guard lets it through, to the data step
    with pytest.raises(ReachedTheData):
        hrs.phase_final(_final_args(tmp_path))


def test_the_committed_final_refuses_to_run_again():
    assert hrs.FINAL_PATH.exists(), "the final report is the read-once record; it must be committed"
    with pytest.raises(SystemExit, match="read once"):
        hrs.main(["--phase", "final", "--no-download"])
    with pytest.raises(SystemExit, match="refusing to re-run"):
        hrs.main(["--phase", "select", "--no-download"])


def test_the_protocols_2000_draws_are_enforced():
    with pytest.raises(SystemExit):
        hrs.main(["--phase", "select", "--reps", "200"])
    with pytest.raises(SystemExit):
        hrs.main(["--phase", "final", "--reps", "200"])


# ---------------------------------------------------------------------------
# The committed records agree with each other
# ---------------------------------------------------------------------------

def test_committed_selection_final_and_artifact_agree():
    sel = json.loads(hrs.SELECTION_PATH.read_text(encoding="utf-8"))
    final = json.loads(hrs.FINAL_PATH.read_text(encoding="utf-8"))
    payload, version = rs.load_artifact(rs.ARTIFACT_PATH)
    assert final["phase"] == "final" and final["final_season"] == 2025
    assert final["selection"]["sha256"] == ri_model.sha256_file(hrs.SELECTION_PATH)
    assert payload["selection_sha256"] == final["selection"]["sha256"]
    assert sel["choice"] == final["selection"]["choice"] == payload["candidate"]
    assert final["artifact"]["model_version"] == version
    assert final["fit"]["case_table_sha256_dev"] == sel["case_table"]["sha256"] == payload["training"]["case_table_sha256"]
    # the recorded parsimony steps imply the recorded choice
    steps = sel["parsimony"]["steps"]
    incumbent = steps[0]["incumbent"]
    for s in steps[1:]:
        assert s["incumbent"] == incumbent
        hi = s["paired_challenger_minus_incumbent"]["ci95"]["delta_log_loss"][1]
        assert s["replaces_incumbent"] == (hi < 0)
        if s["replaces_incumbent"]:
            incumbent = s["challenger"]
    assert incumbent == sel["choice"]
    assert steps[0]["incumbent"] == min("ABC", key=lambda c: (sel["candidates"][c]["pooled"]["log_loss"], c))


def test_committed_artifact_serves_what_it_was_scored_as():
    payload, _ = rs.load_artifact(rs.ARTIFACT_PATH)
    assert payload["kind"] == "raw_aid" and (payload["aid"], payload["fallback"]) == ("DTOP", "RIOD")
    assert rs.predict(payload, {"RIOD": 19, "DTOP": 15}, "EP")[0] == 0.15
    assert rs.predict(payload, {"RIOD": 19, "DTOP": None}, "EP")[0] == 0.19
    assert rs.predict(payload, {"RIOD": None, "DTOP": None}, "EP") is None


# ---------------------------------------------------------------------------
# Serving: fetch_and_score routing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fas():
    return _load("fas_stack_test", "scripts/fetch_and_score.py")


def _case(fas, storm_id: str, cycle: dt.datetime, analysis_model: str = "CARQ") -> dict:
    rec = [fas.ATCFRecord(basin=storm_id[:2], storm_number=int(storm_id[2:4]), cycle=cycle - dt.timedelta(hours=h),
                          tau_hours=0, model=analysis_model, lat=15.0, lon=-110.0, vmax_kt=60.0 - h, mslp_hpa=990.0,
                          storm_name="TEST") for h in (0, 6, 12, 24)]
    case = fas.build_live_case(storm_id, rec)
    assert case is not None and case["issue_time"] == cycle.isoformat()
    return case


NOLO_CYCLE = dt.datetime(2026, 10, 2, 12)


def _nolo_fetcher(calls: list, text: str | None = None):
    def fetch(sid, cycle):
        calls.append((sid, cycle))
        t = text if text is not None else (FIXTURES / NOLO).read_text(encoding="utf-8")
        try:
            return st.parse_ships_text(t, filename=st.filename_for(sid, cycle)), "ok"
        except st.ShipsTextError as exc:
            return None, f"refused ({exc})"
    return fetch


def test_an_nhc_storm_with_ships_text_is_served_the_stack(fas):
    model, stack = fas.load_serving_model(), fas.load_stack_model()
    calls: list = []
    [s] = fas.score_live_cases(model, [_case(fas, "EP152026", NOLO_CYCLE)], stack=stack,
                               ships_fetcher=_nolo_fetcher(calls))
    assert calls == [("EP152026", NOLO_CYCLE)]
    assert s["ri_source"] == "noaa_aid_stack" and s["model_version"] == stack["model_version"]
    assert s["ri_probability"] == 0.15 and s["ri_inputs"]["used"] == "DTOP"
    assert s["ri_inputs"]["ships_text"]["whole_percent"]["DTOP"] == 15
    assert s["ri_inputs"]["ships_text"]["file"] == NOLO
    assert s["v8_2"]["model_version"] == "hurricane_ri_v8_2" and 0 <= s["v8_2"]["ri_probability"] <= 1
    assert s["ri_source_label"] == "NOAA DTOPS"


def test_an_nhc_storm_without_dtops_is_served_ships_rii(fas):
    text = "\n".join(line for line in (FIXTURES / NOLO).read_text(encoding="utf-8").splitlines()
                     if not line.strip().startswith("DTOPS:"))
    [s] = fas.score_live_cases(fas.load_serving_model(), [_case(fas, "EP152026", NOLO_CYCLE)],
                               stack=fas.load_stack_model(), ships_fetcher=_nolo_fetcher([], text))
    assert s["ri_source"] == "noaa_aid_stack" and s["ri_probability"] == 0.19 and s["ri_inputs"]["used"] == "RIOD"
    assert s["ri_source_label"] == "NOAA SHIPS-RII"


def test_an_nhc_storm_without_ships_text_is_served_v82(fas):
    def absent(sid, cycle):
        return None, "absent (HTTPError)"
    model = fas.load_serving_model()
    case = _case(fas, "EP152026", NOLO_CYCLE)
    [s] = fas.score_live_cases(model, [case], stack=fas.load_stack_model(), ships_fetcher=absent)
    [plain] = fas.score_live_cases(model, [case])
    assert s["ri_source"] == "v8.2" and s["model_version"] == "hurricane_ri_v8_2"
    assert s["ri_probability"] == plain["ri_probability"]
    assert s["ri_inputs"]["noaa_aid_stack"]["status"].startswith("ships_text_absent")
    assert set(s["model_scores"]) == {"gbt_d3", "gbt_d4", "logistic", "bagged"}


def test_ships_text_for_another_storm_is_not_used(fas):
    """The fetcher returns Rachel's file under Nolo's name: refused, so v8.2 serves."""
    text = (FIXTURES / "26100212EP1826_ships.txt").read_text(encoding="utf-8")
    [s] = fas.score_live_cases(fas.load_serving_model(), [_case(fas, "EP152026", NOLO_CYCLE)],
                               stack=fas.load_stack_model(), ships_fetcher=_nolo_fetcher([], text))
    assert s["ri_source"] == "v8.2" and "refused" in s["ri_inputs"]["noaa_aid_stack"]["status"]


def test_a_west_pacific_storm_is_served_v82_and_never_asks_for_ships_text(fas):
    calls: list = []
    [s] = fas.score_live_cases(fas.load_serving_model(), [_case(fas, "WP262026", NOLO_CYCLE, "JTWC")],
                               stack=fas.load_stack_model(), ships_fetcher=_nolo_fetcher(calls))
    assert calls == [] and s["ri_source"] == "v8.2" and s["model_version"] == "hurricane_ri_v8_2"
    assert s["ri_inputs"]["noaa_aid_stack"]["status"].startswith("jtwc_basin")


def test_main_routes_through_the_network_path(fas, monkeypatch):
    """End to end through main(): the SHIPS text is fetched from its stext URL by name."""
    import hazardpulse.trust.scoring as trust

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cycle = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=3)
    deck = [fas.ATCFRecord(basin="EP", storm_number=15, cycle=cycle - dt.timedelta(hours=h), tau_hours=0, model="CARQ",
                           lat=15.0, lon=-110.0, vmax_kt=60.0 - h, mslp_hpa=990.0, storm_name="NOLO") for h in (0, 6, 12, 24)]
    wp = [fas.ATCFRecord(basin="WP", storm_number=26, cycle=cycle, tau_hours=0, model="JTWC",
                         lat=15.0, lon=140.0, vmax_kt=70.0, mslp_hpa=None, storm_name="WPTEST")]
    nolo = (FIXTURES / NOLO).read_text(encoding="utf-8")
    stamp = f"{cycle:%m/%d/%y}  {cycle:%H} UTC"
    ships = nolo.replace("10/02/26  12 UTC", stamp)
    index = f'<a href="aep152026.dat.gz">aep152026.dat.gz</a>  {now:%Y-%m-%d %H:%M}  1.0M\n'
    urls: list[str] = []

    def fake_fetch_text(url, **_k):
        urls.append(url)
        if "aid_public" in url:
            return index
        if url == st.url_for("EP152026", cycle):
            return ships
        raise OSError("404")

    captured: dict = {}
    monkeypatch.setattr(fas, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(fas, "fetch_realtime_adeck", lambda sid: deck)
    monkeypatch.setattr(fas, "_discover_jtwc_storms", lambda: {"WP262026": wp})
    monkeypatch.setattr(fas, "write_outputs",
                        lambda scored, now, version, **k: captured.update(scored=scored, version=version, **k))
    monkeypatch.setattr(fas, "build_site_artifacts", lambda: None)
    monkeypatch.setattr(fas, "DIST", REPO / "nonexistent-dist-for-test")
    monkeypatch.setattr(trust, "load_forecaster", lambda *a, **k: None)
    fas.main()
    by_id = {s["storm_id"]: s for s in captured["scored"]}
    assert by_id["EP152026"]["ri_source"] == "noaa_aid_stack" and by_id["EP152026"]["ri_probability"] == 0.15
    assert by_id["WP262026"]["ri_source"] == "v8.2"
    assert st.url_for("EP152026", cycle) in urls and not any("WP26" in u for u in urls)
    top = max(captured["scored"], key=lambda s: s["ri_probability"])
    assert captured["version"] == top["model_version"]
    assert "DTOPS as issued" in captured["note"] and "HazardPulse v8.2" in captured["note"]


def test_a_trust_calibrator_touches_only_its_own_models_storms(fas, monkeypatch):
    import hazardpulse.trust.scoring as trust

    model, stack = fas.load_serving_model(), fas.load_stack_model()
    scored = fas.score_live_cases(model, [_case(fas, "EP152026", NOLO_CYCLE), _case(fas, "WP262026", NOLO_CYCLE, "JTWC")],
                                  stack=stack, ships_fetcher=_nolo_fetcher([]))
    assert {s["ri_source"] for s in scored} == {"noaa_aid_stack", "v8.2"}
    touched: list = []

    class Cal:
        model_version = "hurricane_ri_v8_2"

    monkeypatch.setattr(trust, "load_forecaster", lambda *a, **k: Cal())
    monkeypatch.setattr(trust, "enrich_cells", lambda cells, *a, **k: touched.extend(c["storm_id"] for c in cells))
    monkeypatch.setattr(fas, "score_live_cases", lambda *a, **k: [dict(s) for s in scored])
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cycle = now.replace(minute=0, second=0, microsecond=0)
    wp = [fas.ATCFRecord(basin="WP", storm_number=26, cycle=cycle, tau_hours=0, model="JTWC",
                         lat=15.0, lon=140.0, vmax_kt=70.0, mslp_hpa=None, storm_name="WPTEST")]
    monkeypatch.setattr(fas, "fetch_text", lambda url, **k: "")
    monkeypatch.setattr(fas, "_discover_jtwc_storms", lambda: {"WP262026": wp})
    monkeypatch.setattr(fas, "write_outputs", lambda *a, **k: None)
    monkeypatch.setattr(fas, "build_site_artifacts", lambda: None)
    monkeypatch.setattr(fas, "DIST", REPO / "nonexistent-dist-for-test")
    fas.main()
    assert touched == ["WP262026"], "a v8.2 calibrator must not re-map the NOAA-aid storm"


def test_write_outputs_names_every_model_and_the_headline(fas, tmp_path, monkeypatch):
    monkeypatch.setattr(fas, "DIST", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "live-pulse.json").write_text(json.dumps({"hazards": [{"key": "hu"}]}), encoding="utf-8")
    model, stack = fas.load_serving_model(), fas.load_stack_model()
    scored = fas.score_live_cases(model, [_case(fas, "EP152026", NOLO_CYCLE), _case(fas, "WP262026", NOLO_CYCLE, "JTWC")],
                                  stack=stack, ships_fetcher=_nolo_fetcher([]))
    version = fas.headline_model_version(scored, stack["model_version"])
    fas.write_outputs(scored, NOLO_CYCLE, version, note=fas.ri_sources_note(stack))
    out = json.loads((tmp_path / "data" / "live-storms.json").read_text(encoding="utf-8"))
    assert out["model_versions"] == sorted({stack["model_version"], "hurricane_ri_v8_2"})
    assert out["ri_sources"] == {"noaa_aid_stack": 1, "v8.2": 1}
    assert "SHIPS text" in out["ri_sources_note"]
    hu = json.loads((tmp_path / "data" / "live-pulse.json").read_text(encoding="utf-8"))["hazards"][0]
    top = max(scored, key=lambda s: s["ri_probability"])
    assert (hu["model_version"], hu["ri_source"]) == (top["model_version"], top["ri_source"])


# ---------------------------------------------------------------------------
# Downstream: prospective pooling, site benchmark, registry
# ---------------------------------------------------------------------------

def test_the_calibration_pool_is_deterministic_with_two_models():
    shp = _load("hp_shp_stack_test", "scripts/score_hurricane_prospective.py")
    stack_v = "hurricane_ri_stack_v1-0123456789ab"
    old = {"issued_at": "2026-10-01T00:00:00Z", "storms": [
        {"storm_id": "AL01", "model_version": stack_v}, {"storm_id": "AL02", "model_version": stack_v},
        {"storm_id": "WP01", "model_version": "hurricane_ri_v8_2"}]}
    new_a = {"issued_at": "2026-10-02T00:00:00Z", "storms": [
        {"storm_id": "WP01", "model_version": "hurricane_ri_v8_2"}, {"storm_id": "AL01", "model_version": stack_v}]}
    new_b = {"issued_at": "2026-10-02T00:00:00Z", "storms": list(reversed(new_a["storms"]))}
    assert shp.newest_model_version([old, new_a]) == shp.newest_model_version([old, new_b]) == stack_v
    assert shp.newest_model_version([new_a]) == stack_v   # 1-1 tie: by name ("..._stack_..." < "..._v8_2")
    acc = {shp._CALIB_VERSION_KEY: stack_v}
    res = shp.score_single_forecast(
        {"forecast_id": "x", "issued_at": "2026-10-02T00:00:00Z", "storms": [
            {"storm_id": "AL01", "ri_probability": 0.15, "model_version": stack_v, "ri_source": "noaa_aid_stack"},
            {"storm_id": "WP01", "ri_probability": 0.40, "model_version": "hurricane_ri_v8_2"}]},
        calib_acc=acc, best_track_fetcher=lambda sid: (_track(), "test"))
    pooled = {k: v for k, v in acc.items() if k != shp._CALIB_VERSION_KEY}
    assert pooled == {0.15: [1, 0]}, "only the stack's storm enters the stack's pool"
    assert [p["ri_source"] for p in res["predictions"]] == ["noaa_aid_stack", "v8.2"]


def _track():
    from hazardpulse.hurricane.atcf import ATCFRecord
    t0 = dt.datetime(2026, 10, 2, 0)
    return [ATCFRecord(basin="AL", storm_number=1, cycle=t0 + dt.timedelta(hours=h), tau_hours=0, model="BEST",
                       lat=20.0, lon=-60.0, vmax_kt=50.0, mslp_hpa=None) for h in (0, 6, 12, 18, 24)]


def test_the_site_shows_the_stacks_benchmark_only_for_the_scored_artifact():
    bsa = _load("bsa_stack_test", "scripts/build_site_artifacts.py")
    final = json.loads(hrs.FINAL_PATH.read_text(encoding="utf-8"))
    version = final["artifact"]["model_version"]
    got = bsa._hurricane_heldout_benchmark(version)
    want = final["results"]["all_cases"]["forecasts"][final["selection"]["choice"]]
    assert got["auc"] == round(want["auc"], 4) and got["n_cases"] == want["n"] and got["model_version"] == version
    assert bsa._hurricane_heldout_benchmark("hurricane_ri_stack_v1-000000000000") is None
    assert bsa._hurricane_heldout_benchmark("hurricane_ri_v8_2")["model_version"] == "hurricane_ri_v8_2"


def test_the_registry_entry_is_derived_from_the_artifact_and_its_final_report(tmp_path, monkeypatch):
    pmr = _load("pmr_stack_test", "scripts/publish_model_registry.py")
    entry = pmr.hurricane_stack_entry()
    final = json.loads(hrs.FINAL_PATH.read_text(encoding="utf-8"))
    assert entry["record_id"] == "weights_hazardpulse_" + final["artifact"]["model_version"]
    assert entry["benchmark"]["benchmark_bound_to_this_artifact"] is True
    choice = final["selection"]["choice"]
    assert entry["benchmark"]["test_auc"] == final["results"]["all_cases"]["forecasts"][choice]["auc"]
    # a modified artifact is a different model: its entry must not borrow the benchmark
    other = tmp_path / "hurricane_ri_stack_v1.json"
    payload = json.loads(rs.ARTIFACT_PATH.read_text(encoding="utf-8"))
    payload["description"] += " (edited)"
    rs.save_artifact(payload, other)
    monkeypatch.setattr(pmr, "PROJECT_ROOT", tmp_path)   # _make_entry records paths relative to the root
    unbound = pmr.hurricane_stack_entry(other)
    assert unbound["benchmark"]["benchmark_bound_to_this_artifact"] is False and unbound["benchmark"]["test_auc"] is None
