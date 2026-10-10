"""Earthquake pipeline integrity: the defects found by the 2026-10-05 audit stay fixed.

Each test below FAILS on the pre-fix code (checked by reverting the fix) and pins one finding:

1. overlapping 30-day windows are not independent evidence -- earthquakes are counted once (by USGS
   event id), windows by how many do not overlap, and no live score is quoted from one window;
2. the programme's information gain is per M6+ cell-window, never "per quake";
3. the status badge and the hit rate describe the SERVED model version, and S1's 2023-2025 numbers
   are a declared second read, never a "final test" scored once;
4. the homepage proof figure shows one decimal beside the reference it is compared with;
5. an empty or failed catalog pull publishes nothing (no replay, no pulse, no ledger row) and fails;
6. receipts bind the served S1 stack as well as the C0 artifact under it;
7. the live and the verification catalog pulls are audited for holes (an empty month answered 200/204);
8. the scorer defines each function once (the shadowed copies are gone);
9. the per-cell sentence states the rate model's split as exactly that;
10. a calibrator applies to the grid that is scored, or not at all.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from hazardpulse.data import usgs_fdsn
from hazardpulse.earthquake.served import SERVED_STACK_RELPATH
from hazardpulse.earthquake import operational_forecast as of
from hazardpulse.earthquake import prospective
from hazardpulse.earthquake.coherence_engine import N_LAT, N_LON, grid_cell_to_latlon
from hazardpulse.trust.forecast import verify_forecast_receipt

# what a holed catalog raises (prospective.CatalogIncompleteError is its subclass)
INCOMPLETE = usgs_fdsn.USGSCatalogIncompleteError
ROOT = Path(__file__).resolve().parents[1]


class _LazyLiveRecord:
    """hazardpulse.earthquake.live_record, imported at use (so every test of this file runs, and fails
    for its own reason, against code that predates the module)."""

    def __getattr__(self, name):
        from hazardpulse.earthquake import live_record as mod
        return getattr(mod, name)


live_record = _LazyLiveRecord()
UTC = dt.timezone.utc
DAY = 86400.0
SERVED_ART = ROOT / "results" / "models" / "earthquake_operational_v1.json"
SERVED_STACK = ROOT / SERVED_STACK_RELPATH          # the stack the scorer publishes (hazardpulse.earthquake.served)


def _script(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fse():
    return _script("hp_fse_integrity_test", "scripts/fetch_and_score_earthquake.py")


@pytest.fixture(scope="module")
def scorer():
    return _script("hp_eq_prospective_integrity_test", "scripts/score_earthquake_prospective.py")


def _z(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# A fake FDSN service: one event every `step_hours` (an M6.2 every 40th, M4.6 otherwise), stable ids
# ---------------------------------------------------------------------------

def _fake_usgs(holes=(), gap=None, step_hours=6.0):
    calls = []

    def fetch_text(url, namespace=None, use_cache=False, refresh=True):
        q = parse_qs(urlsplit(url).query)
        start = dt.datetime.fromisoformat(q["starttime"][0]).replace(tzinfo=UTC)
        end = dt.datetime.fromisoformat(q["endtime"][0]).replace(tzinfo=UTC)
        minmag = float(q["minmagnitude"][0])
        calls.append((start, end, minmag))
        if (start.year, start.month) in holes:
            return ""                                   # what FDSN sends for "no events" (200 / 204)
        step = step_hours * 3600.0
        lines = ["time,latitude,longitude,depth,mag,magType,place,type,id"]
        i = int(np.ceil(start.timestamp() / step))
        while i * step <= end.timestamp():
            t = dt.datetime.fromtimestamp(i * step, UTC)
            if not (gap and gap[0] <= t < gap[1]):
                mag = 6.2 if i % 40 == 0 else 4.6
                if mag >= minmag:
                    lines.append(f"{t:%Y-%m-%dT%H:%M:%S}.000Z,10.0,20.0,10.0,{mag},mb,Test,earthquake,ev{i}")
            i += 1
        return "\n".join(lines) + "\n"

    fetch_text.calls = calls
    return fetch_text


# ---------------------------------------------------------------------------
# 7. Catalog completeness
# ---------------------------------------------------------------------------

START, END = dt.datetime(2026, 1, 15, tzinfo=UTC), dt.datetime(2026, 4, 10, tzinfo=UTC)


def test_an_empty_month_answered_200_fails_the_pull(monkeypatch):
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs(holes={(2026, 2)}))
    with pytest.raises(INCOMPLETE, match="2026-02"):
        prospective.fetch_usgs_catalog_range(START, END, min_magnitude=2.5)


def test_a_multi_day_gap_inside_a_month_fails_the_pull(monkeypatch):
    gap = (dt.datetime(2026, 3, 5, tzinfo=UTC), dt.datetime(2026, 3, 10, tzinfo=UTC))
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs(gap=gap))
    with pytest.raises(INCOMPLETE, match="gap"):
        prospective.fetch_usgs_catalog_range(START, END, min_magnitude=2.5)


def test_a_complete_pull_passes_the_audit(monkeypatch):
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs())
    events = prospective.fetch_usgs_catalog_range(START, END, min_magnitude=2.5)
    assert len(events) == len({e["id"] for e in events}) > 300
    assert prospective.audit_catalog(events, START, END) == []
    assert issubclass(prospective.CatalogIncompleteError, INCOMPLETE)


def test_the_target_catalog_is_pulled_dense_audited_and_filtered(monkeypatch):
    fake = _fake_usgs()
    monkeypatch.setattr(prospective, "fetch_text", fake)
    targets = prospective.fetch_target_catalog(START, END, target_min_magnitude=6.0)
    assert targets and all(e["mag"] >= 6.0 for e in targets)
    assert {c[2] for c in fake.calls} == {prospective.AUDIT_MAX_MAGNITUDE}     # never pulled at M6
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs(holes={(2026, 3)}))
    with pytest.raises(prospective.CatalogIncompleteError):
        prospective.fetch_target_catalog(START, END, target_min_magnitude=6.0)
    with pytest.raises(ValueError, match="too sparse"):
        prospective.fetch_usgs_catalog_range(START, END, min_magnitude=6.0)


def test_the_live_pull_is_audited(monkeypatch, fse):
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs(holes={(2026, 2)}))
    with pytest.raises(INCOMPLETE):
        fse.fetch_usgs_catalog(days=80, end_time=dt.datetime(2026, 4, 1, tzinfo=UTC))


def test_the_verifier_refuses_a_truth_catalog_with_a_hole(monkeypatch, scorer, tmp_path):
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "eq_fcst_20260201_0000.json").write_text(json.dumps({
        "forecast_id": "eq_fcst_20260201_0000", "issued_at": "2026-02-01T00:00:00Z", "forecast_horizon_days": 30,
        "model_version": "eq_S1-test", "forecast_domain": {"n_lat": 2, "n_lon": 2, "default_probability": 0.0},
        "active_cells": []}), encoding="utf-8")
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs(holes={(2026, 2)}))
    out = tmp_path / "out"
    with pytest.raises(INCOMPLETE):
        scorer.main(["--replay-dir", str(replay), "--output-dir", str(out), "--score-as-of", "2026-04-01T00:00:00Z"])
    assert not (out / "prospective_summary.json").exists()


def test_the_verifier_counts_distinct_earthquakes_end_to_end(monkeypatch, scorer, tmp_path):
    """A complete truth pull passes the audit; two overlapping windows sharing the same M6+ earthquakes
    are recorded as those earthquakes once, in the summary AND in the calibration pool."""
    replay = tmp_path / "replay"
    replay.mkdir()
    for fid, issued in (("eq_fcst_20260201_0000", "2026-02-01T00:00:00Z"),
                        ("eq_fcst_20260201_0600", "2026-02-01T06:00:00Z")):
        (replay / f"{fid}.json").write_text(json.dumps({
            "forecast_id": fid, "issued_at": issued, "forecast_horizon_days": 30, "model_version": "eq_S1-test",
            "forecast_domain": {"n_lat": 2, "n_lon": 2, "default_probability": 0.0}, "active_cells": []}),
            encoding="utf-8")
    monkeypatch.setattr(prospective, "fetch_text", _fake_usgs())
    out = tmp_path / "out"
    assert scorer.main(["--replay-dir", str(replay), "--output-dir", str(out), "--score-as-of",
                        "2026-04-01T00:00:00Z", "--emit-calibration"]) == 0
    summary = json.loads((out / "prospective_summary.json").read_text(encoding="utf-8"))
    rec = summary["by_model_version"]["eq_S1-test"]
    assert rec["n_matured_forecasts"] == 2 and rec["n_independent_windows"] == 1
    assert rec["n_event_windows"] == 2 * rec["n_distinct_events"] and rec["n_distinct_events"] >= 2
    pool = json.loads((out / "calibration_dataset.json").read_text(encoding="utf-8"))
    assert pool["n_distinct_events"] == rec["n_distinct_events"] and pool["n_independent_windows"] == 1
    assert "audited" in summary["observed_catalog_window"]["completeness_audit"]


# ---------------------------------------------------------------------------
# 1. Overlapping windows are not independent evidence
# ---------------------------------------------------------------------------

def _window_artifact(issued: dt.datetime, version: str = "eq_S1-test") -> dict:
    return {"forecast_id": f"eq_fcst_{issued:%Y%m%d_%H}00", "issued_at": _z(issued), "forecast_horizon_days": 30,
            "model_version": version,
            "forecast_domain": {"n_lat": N_LAT, "n_lon": N_LON, "default_probability": 0.0},
            "probability_grid": ",".join(["0.001"] * (N_LAT * N_LON)), "active_cells": []}


def test_overlapping_windows_count_each_earthquake_once(scorer, tmp_path):
    t0 = dt.datetime(2026, 5, 1, tzinfo=UTC)
    lat, lon = grid_cell_to_latlon(30, 100)
    quakes = [{"time": _z(t0 + dt.timedelta(days=d)), "latitude": lat, "longitude": lon, "depth": 10.0,
               "mag": 6.1, "id": f"us{d}"} for d in (2, 3, 5, 8)]
    results = []
    for h in (0, 6, 12):                                  # three windows, 6 h apart: the same 4 earthquakes
        art = _window_artifact(t0 + dt.timedelta(hours=h))
        r = scorer.score_single_forecast(art, quakes, tmp_path)
        r["model_version"] = art["model_version"]
        results.append(r)
    rec = scorer.live_record_by_version(results)["eq_S1-test"]
    assert rec.get("n_distinct_events") == 4 and rec.get("n_event_windows") == 12
    assert rec.get("n_independent_windows") == 1
    assert "n_observed_events" not in rec                 # the summed count is never offered as "events"
    assert live_record.quotable(rec)[0] is False
    assert scorer.summarize_results(results)["n_distinct_observed_events"] == 4


def test_the_quote_rule_needs_distinct_earthquakes_and_two_independent_windows():
    assert live_record.quotable({"n_distinct_events": 64, "n_independent_windows": 6})[0] is True
    ok, why = live_record.quotable({"n_distinct_events": 12, "n_independent_windows": 1})
    assert not ok and "non-overlapping" in why
    ok, why = live_record.quotable({"n_distinct_events": 4, "n_independent_windows": 20})
    assert not ok and "4 distinct" in why
    assert live_record.quotable({"n_matured_forecasts": 613})[0] is False        # an old summary: never guessed
    w = [(dt.datetime(2026, 1, 1, tzinfo=UTC) + dt.timedelta(hours=6 * i),
          dt.datetime(2026, 1, 31, tzinfo=UTC) + dt.timedelta(hours=6 * i)) for i in range(250)]
    assert live_record.non_overlapping_windows(w) == 3         # 62 days of issues, 30-day windows


def test_the_stored_record_resummarizes_to_distinct_counts(scorer, tmp_path):
    rows = []
    for i, ids in enumerate((["a", "b"], ["b", "c"])):
        fid = f"eq_fcst_2026050{i + 1}_0000"
        rows.append({"forecast_id": fid, "issued_at": f"2026-05-0{i + 1}T00:00:00Z",
                     "window_end": f"2026-05-3{i}T00:00:00Z", "model_version": "v", "n_observed_events": 2,
                     "auc": 0.7, "pr_auc": 0.1, "auc_active_cells": 0.6, "n_events_in_active_cells": 1,
                     "brier": 0.001, "information_gain_per_event": 1.0, "poisson_log_likelihood": -5.0,
                     "uniform_log_likelihood": -6.0, "top_1_hit": False, "top_5_hit": True,
                     "top_10_hit": True, "top_20_hit": True})
        csv = tmp_path / "observed_events" / f"{fid}_observed.csv"
        csv.parent.mkdir(exist_ok=True)
        csv.write_text("time,latitude,longitude,depth,mag,id,row,col\n" + "".join(
            f"2026-05-1{k}T00:00:00Z,1.0,2.0,10.0,6.1,{eid},0,0\n" for k, eid in enumerate(ids)), encoding="utf-8")
    (tmp_path / "per_forecast_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "prospective_summary.json").write_text(json.dumps({"status": "ok", "total_observed_events": 4}),
                                                       encoding="utf-8")
    s = scorer.resummarize_from_stored(tmp_path)
    assert s["by_model_version"]["v"]["n_distinct_events"] == 3 and s["n_distinct_observed_events"] == 3
    assert s["total_event_windows"] == 4 and "total_observed_events" not in s
    rows[0]["n_observed_events"] = 3                      # a score that disagrees with its stored events
    rows[0].pop("observed_event_ids", None)
    (tmp_path / "per_forecast_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        scorer.resummarize_from_stored(tmp_path)


# --- the rollup (build_site_artifacts) -------------------------------------

def _bsa(tmp_path, served_version: str):
    m = _script(f"hp_bsa_integrity_{abs(hash(str(tmp_path)))}", "scripts/build_site_artifacts.py")
    dist = tmp_path / "dist"
    m.ROOT, m.DIST = tmp_path, dist
    m.LIVE_PULSE_PATH = dist / "data" / "live-pulse.json"
    m.EQ_LEDGER_PATH = dist / "data" / "earthquake-ledger.jsonl"
    m.TO_LEDGER_PATH = dist / "data" / "tornado-ledger.jsonl"
    m.REPLAY_DIR = dist / "data" / "replay"
    m.VERIFICATION_SUMMARY_PATH = dist / "data" / "verification-summary.json"
    m.VERIFICATION_DATA_DIR = dist / "data" / "verification"
    m.RESULTS_VERIFICATION_DIR = tmp_path / "results" / "verification"
    m.EQ_PROSPECTIVE_DIR = tmp_path / "results" / "earthquake_prospective"
    m.TO_PROSPECTIVE_DIR = tmp_path / "results" / "tornado_prospective"
    m.HU_PROSPECTIVE_DIR = tmp_path / "results" / "hurricane_prospective"
    m.EQ_HONEST_RESULTS_PATH = tmp_path / "results" / "earthquake_honest" / "v4.json"
    m.EQ_SAME_LOCATION_PATH = tmp_path / "results" / "earthquake_honest" / "same.json"
    m.TO_RETRO_RESULTS_PATH = tmp_path / "results" / "definitive" / "definitive_results.json"
    m.REPLAY_DIR.mkdir(parents=True)
    m.LIVE_PULSE_PATH.write_text(json.dumps({"updated_at": "2026-10-05T00:00:00Z", "hazards": [
        {"key": "eq", "model_version": served_version}, {"key": "hu", "model_version": "hurricane_ri_v8_1"},
        {"key": "to", "model_version": "tornado_storm_v1_0"}]}), encoding="utf-8")
    return m


def _eq_item(m) -> dict:
    summary = m._build_verification_summary(json.loads(m.LIVE_PULSE_PATH.read_text(encoding="utf-8")))
    return next(item for item in summary["hazards"] if item["key"] == "eq")


def _eq_summary(m, by_version: dict, **top) -> None:
    m.EQ_PROSPECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    n = sum(r["n_matured_forecasts"] for r in by_version.values())
    (m.EQ_PROSPECTIVE_DIR / "prospective_summary.json").write_text(json.dumps(
        {"status": "ok", "n_matured_forecasts": n, "by_model_version": by_version, **top}), encoding="utf-8")


def _eq_replays(m, version: str, issued: list[dt.datetime]) -> None:
    for t in issued:
        a = _window_artifact(t, version)
        a["hazard"] = "earthquake"
        a.pop("probability_grid")
        (m.REPLAY_DIR / f"{a['forecast_id']}.json").write_text(json.dumps(a), encoding="utf-8")


def test_a_single_window_mean_auc_is_never_printed_as_a_live_record(tmp_path):
    m = _bsa(tmp_path, "eq_S1-test")
    _eq_replays(m, "eq_S1-test", [dt.datetime(2026, 8, 1, tzinfo=UTC)])
    _eq_summary(m, {"eq_S1-test": {"n_matured_forecasts": 1, "mean_auc": 0.95, "mean_brier": 0.001,
                                   "n_distinct_events": 12, "n_independent_windows": 1, "n_event_windows": 12}})
    item = _eq_item(m)
    assert item["auc"] is None and item["brier"] is None
    assert "mean AUC" not in item["metric_source_label"] and "0.95" not in item["metric_source_label"]
    assert "non-overlapping" in item["metric_source_label"]
    # 120 overlapping windows holding the same 4 earthquakes (480 event-windows) are not 480 events
    _eq_summary(m, {"eq_S1-test": {"n_matured_forecasts": 120, "mean_auc": 0.95, "mean_brier": 0.001,
                                   "n_distinct_events": 4, "n_independent_windows": 1, "n_event_windows": 480}})
    assert _eq_item(m)["auc"] is None
    # quotable once it is: distinct earthquakes and two independent windows
    _eq_summary(m, {"eq_S1-test": {"n_matured_forecasts": 240, "mean_auc": 0.91, "mean_brier": 0.001,
                                   "n_distinct_events": 30, "n_independent_windows": 2, "n_event_windows": 900}})
    assert _eq_item(m)["auc"] == pytest.approx(0.91)


def test_the_status_badge_describes_the_served_version_only(tmp_path):
    m = _bsa(tmp_path, "eq_S1-test")
    _eq_replays(m, "eq_coherence_v1_0", [dt.datetime(2026, 4, 1, tzinfo=UTC) + dt.timedelta(hours=6 * i)
                                         for i in range(5)])
    _eq_replays(m, "eq_S1-test", [dt.datetime(2026, 10, 3, tzinfo=UTC), dt.datetime(2026, 10, 4, tzinfo=UTC)])
    _eq_summary(m, {"eq_coherence_v1_0": {"n_matured_forecasts": 5, "mean_auc": 0.66, "top_5_hit_rate": 0.02,
                                          "n_distinct_events": 64, "n_independent_windows": 6}},
                top_5_hit_rate=0.02)
    item = _eq_item(m)
    assert item["verification_status"] == "logging_waiting_maturity" and item["status_badge"] == "Waiting"
    assert item["prospective"]["top_5_hit_rate"] is None          # the retired model's rate is not S1's
    assert item["forecast_storage"]["n_scored_forecasts"] == 5    # the scorer's count, all versions
    assert item["auc"] is None


@pytest.mark.skipif(not SERVED_STACK.exists(), reason="served stack not present")
def test_a_second_read_is_never_called_a_final_test_scored_once(tmp_path):
    served = of.stack_model_version(SERVED_STACK)
    m = _bsa(tmp_path, served)
    bench = m._served_benchmark("earthquake", served)
    assert bench and bench.get("second_read") is True
    assert "second read" in bench["label"] and "scored once" not in bench["label"]
    assert "information_gain_per_event" not in bench and "per M6+ cell-window" in bench["label"]
    item = _eq_item(m)
    assert "final test" not in item["homepage_line"] and "second read" in item["homepage_line"]


# --- the track-record page ---------------------------------------------------

def _site(tmp_path, summary: dict, served: str):
    from hazardpulse.site.data import SiteData

    (tmp_path / "results" / "earthquake_prospective").mkdir(parents=True)
    (tmp_path / "results" / "earthquake_prospective" / "prospective_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    (tmp_path / "dist" / "data").mkdir(parents=True)
    (tmp_path / "dist" / "data" / "verification-summary.json").write_text(json.dumps(
        {"hazards": [{"key": "eq", "model_version": served}]}), encoding="utf-8")
    return SiteData(root=tmp_path, dist=tmp_path / "dist")


def test_the_track_record_counts_distinct_earthquakes_not_event_windows(tmp_path):
    from hazardpulse.site.pages import record

    rec = {"n_matured_forecasts": 613, "n_distinct_events": 64, "n_event_windows": 6955,
           "n_independent_windows": 6, "mean_auc": 0.6584, "event_weighted_information_gain_per_event": -15.46}
    d = _site(tmp_path, {"by_model_version": {"eq_coherence_v1_0": rec}}, "eq_S1-test")
    row = next(r for r in record._live_rows(d) if "eq_coherence_v1_0" in r[1])
    assert row[2] == "retired" and row[3] == "613 (6 non-overlapping)" and row[4] == "64"
    assert "6,955" not in " ".join(row) and "per quake" not in row[5]
    assert "ranking 65.8%" in row[5]
    # a record of one window -- or an old summary without distinct counts -- shows no score
    one = dict(rec, n_matured_forecasts=1, n_independent_windows=1)
    assert "no score yet" in record._eq_live_row("v", one)[2]
    old = {"n_matured_forecasts": 613, "n_observed_events": 6955, "mean_auc": 0.66}
    closed, events, score = record._eq_live_row("v", old)
    assert events == "not counted" and "no score yet" in score and "6,955" not in closed + events + score


# ---------------------------------------------------------------------------
# 2. Information gain is per M6+ cell-window
# ---------------------------------------------------------------------------

_PER_QUAKE = re.compile(r"per (quake|earthquake)\b|nats/quake|quakes than", re.I)


def test_information_gain_is_labelled_per_cell_window_never_per_quake():
    from hazardpulse.verification import evidence_pages as ep
    from hazardpulse.verification import served_evidence as se

    eq = se.earthquake_evidence()
    if eq is None:
        pytest.skip("no earthquake evidence bound in this checkout")
    ev = {"earthquake": eq, "hurricane": None, "tornado": None}
    for name, html in (("methods_simple", ep.methods_simple(ev)), ("methods_earthquake", ep.methods_earthquake(ev)),
                       ("registry_active", ep.registry_active(ev)), ("registry_history", ep.registry_history(ev))):
        assert not _PER_QUAKE.search(html), (name, _PER_QUAKE.search(html).group(0))
        assert "per M6+ cell-window" in html, name
    for page in ("methods/index.html", "registry/index.html", "verification/index.html", "index.html"):
        text = (ROOT / "dist" / page).read_text(encoding="utf-8")
        hit = _PER_QUAKE.search(text)
        assert hit is None, (page, text[max(0, hit.start() - 80):hit.end() + 20])


# ---------------------------------------------------------------------------
# 4. The homepage proof figure
# ---------------------------------------------------------------------------

def test_the_homepage_proof_shows_one_decimal_and_the_real_gap(tmp_path):
    from hazardpulse.site.data import SiteData
    from hazardpulse.site.pages import overview
    from hazardpulse.verification import served_evidence as se

    eq = se.earthquake_evidence()
    if eq is None or not (eq.get("vs") or {}).get("A"):
        pytest.skip("no earthquake evidence with a reference comparison bound in this checkout")
    d = SiteData(root=tmp_path, dist=tmp_path / "dist")
    d.__dict__["evidence"] = {"earthquake": eq, "hurricane": None, "tornado": None, "_errors": []}
    html = overview._proofs(d)
    auc, ref = eq["test"]["auc"]["value"], eq["vs"]["A"]["auc"]
    assert f'<p class="proof-figure">{100 * auc:.1f}%</p>' in html
    assert f'<p class="proof-figure">{100 * auc:.0f}%</p>' not in html
    assert f"difference +{100 * ref['diff']:.2f} points" in html
    assert f"{100 * (auc - ref['diff']):.1f}%" in html


# ---------------------------------------------------------------------------
# 5, 6, 10. The live pipeline: fail closed, receipts, calibration
# ---------------------------------------------------------------------------

SPEC_B = of.ModelSpec("B", of.LongTermParams(kernel_km=20.0, declustered=True, mu=10.0, eps=0.01),
                      of.ShortTermParams(d0_km=10.0, K=0.02, alpha=1.0, c_days=0.01, p=1.1))
CUTOFF = dt.datetime(2026, 9, 1, tzinfo=UTC)
ISSUE = dt.datetime(2026, 9, 11, 6, tzinfo=UTC)


def _stamp(t: float) -> str:
    s = dt.datetime.fromtimestamp(float(t), UTC)
    return s.strftime("%Y-%m-%dT%H:%M:%S.") + f"{s.microsecond // 1000:03d}Z"


def _served_pair(tmp_path):
    """A small rate-model artifact and an S1-style stack bound to it."""
    rng = np.random.default_rng(7)
    n = 300
    centres = np.array([[38.0, 142.0], [-33.0, -72.0], [-20.0, -175.0]])
    k = rng.integers(0, len(centres), n)
    ev = of.EventSet.from_arrays(rng.uniform(dt.datetime(1995, 1, 1, tzinfo=UTC).timestamp(),
                                             CUTOFF.timestamp() - 30 * DAY, n),
                                 centres[k, 0] + rng.normal(0, 1.5, n), centres[k, 1] + rng.normal(0, 1.5, n),
                                 np.minimum(5.0 + rng.exponential(1 / np.log(10.0), n), 8.5))
    art_path = tmp_path / "art.json"
    of.write_artifact(art_path, SPEC_B, ev, cutoff=_z(CUTOFF), model_name="eq_operational_B_test", provenance={})
    art = of.load_artifact(art_path)
    stack_path = tmp_path / "stack.json"
    version = of.write_stack(stack_path, model_name="eq_S1_test", base_model_version=art.model_version,
                             a=0.1, c=1.0, b=0.2, g_log10=np.linspace(-1.0, 1.0, of.N_CELLS), provenance={})
    return art_path, stack_path, version


def _live_events() -> list[dict]:
    """M2.5+ events every 3 h from 30 days before the cutoff to the issue time, clustered off Japan so
    a few cells are active, plus an M5.2 one day before the cutoff (the live tail must reach it)."""
    rng = np.random.default_rng(3)
    t = np.arange(CUTOFF.timestamp() - 30 * DAY, ISSUE.timestamp(), 3 * 3600.0)
    out = [{"time": _stamp(x), "latitude": 38.0 + rng.normal(0, 0.6), "longitude": 142.0 + rng.normal(0, 0.6),
            "depth": 10.0, "mag": round(2.5 + rng.exponential(0.4), 1), "id": f"live{i}"} for i, x in enumerate(t)]
    out.append({"time": _stamp(CUTOFF.timestamp() - DAY), "latitude": 38.2, "longitude": 142.3, "depth": 20.0,
                "mag": 5.2, "id": "m5"})
    return sorted(out, key=lambda e: e["time"])


class _HalfCalibrator:
    def predict(self, scores):
        s = np.asarray(scores, dtype=np.float64).ravel()
        return 0.5 * s, 0.4 * s, 0.6 * s


class _BoundForecaster:
    def __init__(self, version):
        self.model_version = version
        self.model_sha256_ = "c" * 64
        self.calibrator = _HalfCalibrator()


def _wire(monkeypatch, fse, tmp_path, *, events, forecaster=None, pool=None):
    """Point the live scorer at a synthetic served pair and catalog. ``pool`` (n_distinct_events,
    n_independent_windows) is the calibration pool's independent evidence, written for the served
    version."""
    import hazardpulse.trust.scoring as scoring

    art_path, stack_path, version = _served_pair(tmp_path)
    pool_path = tmp_path / "calibration_dataset.json"
    if pool is not None:
        pool_path.write_text(json.dumps({"model_version": version, "pos": [5931], "total": [7172100],
                                         "n_distinct_events": pool[0], "n_independent_windows": pool[1]}),
                             encoding="utf-8")
    monkeypatch.setattr(fse, "CALIBRATION_DATASET_PATH", pool_path, raising=False)
    monkeypatch.setattr(fse, "OPERATIONAL_ARTIFACT_PATH", art_path)
    monkeypatch.setattr(fse, "STACK_PATH", stack_path)
    monkeypatch.setattr(fse, "MODEL_VERSION", version)
    monkeypatch.setattr(fse, "DIST", tmp_path / "dist")
    monkeypatch.setattr(fse, "fetch_usgs_catalog", lambda days, end_time: list(events))
    monkeypatch.setattr(fse, "load_deep_eq_shortterm_model", lambda: None)
    monkeypatch.setattr(fse, "load_deep_eq_operational_model", lambda: None)

    def no_field(*_a, **_k):
        raise RuntimeError("field skipped in this test")

    monkeypatch.setattr(fse, "compute_seismic_coherence_field", no_field)
    monkeypatch.setattr(scoring, "load_signer", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "load_forecaster", lambda *a, **k: forecaster(version) if forecaster else None)
    return art_path, stack_path, version


def _run(fse, tmp_path):
    return fse.run_pipeline(issue_time=ISSUE, replay_dir=tmp_path / "replay", ledger_path=tmp_path / "ledger.jsonl",
                            skip_site=True, skip_live_pulse=True, skip_replay_index=True)


def _no_forecast(match: str = ""):
    """fse.NoForecastError (a RuntimeError), matched by message so the check reads the same on code
    that predates the class."""
    return pytest.raises(RuntimeError, match=match or "no scored cell|no probability grid|returned no events")


def test_an_empty_catalog_publishes_nothing_and_fails_the_run(monkeypatch, fse, tmp_path):
    _wire(monkeypatch, fse, tmp_path, events=[])
    with _no_forecast("returned no events") as err:
        _run(fse, tmp_path)
    assert type(err.value).__name__ == "NoForecastError"
    assert not (tmp_path / "replay").exists() or not any((tmp_path / "replay").iterdir())
    assert not (tmp_path / "ledger.jsonl").exists()


def test_a_named_stack_that_is_missing_publishes_nothing(monkeypatch, fse, tmp_path):
    """The scorer used to fall back to C0 alone when the stack file was absent: a pointer moved ahead of its file
    would have silently changed the published model. A named stack must exist; only None means C0 is served."""
    art_path, stack_path, _ = _wire(monkeypatch, fse, tmp_path, events=_live_events())
    monkeypatch.setattr(fse, "STACK_PATH", tmp_path / "earthquake_gear1_stack_v9.json")
    with pytest.raises(FileNotFoundError):
        _run(fse, tmp_path)
    assert not (tmp_path / "ledger.jsonl").exists()
    monkeypatch.setattr(fse, "STACK_PATH", None)                                   # C0 served, by name
    monkeypatch.setattr(fse, "MODEL_VERSION", of.load_artifact(art_path).model_version)
    out = _run(fse, tmp_path)
    replay = json.loads(Path(out["replay_path"]).read_text(encoding="utf-8"))
    assert replay["operational_model"]["model_version"] == of.load_artifact(art_path).model_version


def test_a_failed_catalog_pull_publishes_nothing(monkeypatch, fse, tmp_path):
    _wire(monkeypatch, fse, tmp_path, events=[])

    def holed(days, end_time):
        raise INCOMPLETE("2026-08: no events")

    monkeypatch.setattr(fse, "fetch_usgs_catalog", holed)
    with pytest.raises(INCOMPLETE):
        _run(fse, tmp_path)
    assert not (tmp_path / "ledger.jsonl").exists()


def test_no_writer_publishes_an_empty_forecast(fse, tmp_path):
    pulse = tmp_path / "live-pulse.json"
    pulse.write_text(json.dumps({"hazards": [{"key": "eq", "probability": 0.21, "forecast_id": "eq_prev"}]}),
                     encoding="utf-8")
    now = dt.datetime(2026, 10, 5, tzinfo=UTC)
    with _no_forecast("no scored cell"):
        fse.write_outputs([], now, forecast_id="eq_fcst_20261005_0000", pulse_path=pulse)
    assert json.loads(pulse.read_text(encoding="utf-8"))["hazards"][0]["forecast_id"] == "eq_prev"
    with _no_forecast("no scored cell"):
        fse.append_ledger([], now, forecast_id="eq_fcst_20261005_0000", ledger_path=tmp_path / "l.jsonl")
    with _no_forecast("no probability grid"):
        fse.write_replay_artifact([], now, forecast_id="eq_fcst_20261005_0000", n_history_events=0,
                                  n_recent_events=0, replay_dir=tmp_path / "r", update_index=False)
    assert not (tmp_path / "l.jsonl").exists() and not (tmp_path / "r").exists()


def _expected_served_digest(art_path, stack_path) -> str:
    """The digest a receipt must carry for the S1 pair, computed here from the two files (the recipe
    the replay states), not by the code under test."""
    text = (f"hazardpulse/earthquake-served-digest/v1\nbase {of.sha256_text_file(art_path)}\n"
            f"stack {of.sha256_text_file(stack_path)}\n")
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def test_the_pipeline_receipts_bind_the_served_stack_and_its_base(monkeypatch, fse, tmp_path):
    art_path, stack_path, version = _wire(monkeypatch, fse, tmp_path, events=_live_events())
    out = _run(fse, tmp_path)
    replay = json.loads(Path(out["replay_path"]).read_text(encoding="utf-8"))
    served = _expected_served_digest(art_path, stack_path)
    cells = replay["active_cells"]
    assert cells and all(c["receipt"]["model_sha256"] == served for c in cells)       # was C0's digest alone
    assert served != of.sha256_text_file(art_path)
    meta = replay["operational_model"]
    assert meta.get("stack_sha256_lf") == of.sha256_text_file(stack_path)
    assert meta.get("served_sha256") == served
    assert all(verify_forecast_receipt(c["receipt"]) for c in cells)
    assert all(c["receipt"]["model_version"] == version for c in cells)
    grid = np.array([float(v) for v in replay["probability_grid"].split(",")])
    assert all(c["probability"] == float(f"{grid[c['row'] * N_LON + c['col']]:.6g}") for c in cells)


def test_a_calibrator_applies_to_the_scored_grid_or_not_at_all(monkeypatch, fse, tmp_path):
    art_path, stack_path, version = _wire(monkeypatch, fse, tmp_path, events=_live_events(),
                                          forecaster=_BoundForecaster, pool=(40, 3))
    out = _run(fse, tmp_path)
    replay = json.loads(Path(out["replay_path"]).read_text(encoding="utf-8"))
    art = of.load_artifact(art_path)
    stack = of.load_stack(stack_path, art)
    raw = of.forecast_with_stack(art, stack, _live_events(), ISSUE)["probability"]
    grid = np.array([float(v) for v in replay["probability_grid"].split(",")])
    np.testing.assert_array_equal(grid, [float(f"{0.5 * p:.6g}") for p in raw])     # the SCORED grid is calibrated
    cells = replay["active_cells"]
    assert cells and all(c["probability"] == grid[c["row"] * N_LON + c["col"]] for c in cells)
    served = _expected_served_digest(art_path, stack_path)
    bound = hashlib.sha256(f"{served}\ncalibrator {'c' * 64}\n".encode("ascii")).hexdigest()
    for c in cells:
        flat = c["row"] * N_LON + c["col"]
        assert c.get("calibrated") is True and c["raw_probability"] == float(f"{raw[flat]:.6g}")
        assert c["receipt"]["raw_probability"] == c["raw_probability"]
        assert c["receipt"]["model_sha256"] == bound                  # served pair + calibrator fingerprint
        assert c["confidence_lo"] < c["probability"] < c["confidence_hi"]
    assert replay["operational_model"]["calibrator_model_version"] == version


@pytest.mark.parametrize("pool", [(4, 6), (64, 1), None])
def test_a_calibrator_on_overlapping_evidence_is_not_applied(monkeypatch, fse, tmp_path, pool):
    """5,931 positive cell-windows from 4 earthquakes (or from one 30-day stretch, or a pool written
    before distinct counting) are not 30 events: the grid stays as the model computed it."""
    art_path, stack_path, _ = _wire(monkeypatch, fse, tmp_path, events=_live_events(),
                                    forecaster=_BoundForecaster, pool=pool)
    out = _run(fse, tmp_path)
    replay = json.loads(Path(out["replay_path"]).read_text(encoding="utf-8"))
    art = of.load_artifact(art_path)
    raw = of.forecast_with_stack(art, of.load_stack(stack_path, art), _live_events(), ISSUE)["probability"]
    grid = np.array([float(v) for v in replay["probability_grid"].split(",")])
    np.testing.assert_array_equal(grid, [float(f"{p:.6g}") for p in raw])
    assert not any(c.get("calibrated") for c in replay["active_cells"])
    assert "calibrator_model_version" not in replay["operational_model"]


def test_a_listed_cell_off_the_scored_grid_is_never_written(fse, tmp_path):
    grid = np.full(N_LAT * N_LON, 1e-4)
    grid[40 * N_LON + 160] = 0.996
    calibrated_listed_only = [{"row": 40, "col": 160, "probability": 0.0454, "risk_band": "low"}]
    with pytest.raises(RuntimeError, match="scored grid does not hold"):
        fse.write_replay_artifact(calibrated_listed_only, ISSUE, forecast_id="eq_fcst_20260911_0600",
                                  n_history_events=1, n_recent_events=1, replay_dir=tmp_path,
                                  update_index=False, probability_grid=grid)
    ok = [{"row": 40, "col": 160, "probability": 0.996, "risk_band": "critical"}]
    fse.write_replay_artifact(ok, ISSUE, forecast_id="eq_fcst_20260911_0600", n_history_events=1, n_recent_events=1,
                              replay_dir=tmp_path, update_index=False, probability_grid=grid)


# ---------------------------------------------------------------------------
# 6. The served digest
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (SERVED_ART.exists() and SERVED_STACK.exists()), reason="served pair not present")
def test_the_served_digest_binds_the_base_artifact_and_the_stack():
    art = of.load_artifact(SERVED_ART)
    stack = of.load_stack(SERVED_STACK, art)
    assert stack.sha256 == of.sha256_text_file(SERVED_STACK)
    assert stack.model_version == of.stack_model_version(SERVED_STACK)
    text = f"{of.SERVED_DIGEST_SCHEMA}\nbase {art.sha256}\nstack {stack.sha256}\n"
    served = of.served_sha256(art, stack)
    assert served == hashlib.sha256(text.encode("ascii")).hexdigest()
    assert served not in (art.sha256, stack.sha256)
    assert of.served_sha256(art, None) == art.sha256


# ---------------------------------------------------------------------------
# 8. No shadowed copies
# ---------------------------------------------------------------------------

def test_the_scorer_defines_each_function_once():
    tree = ast.parse((ROOT / "scripts" / "fetch_and_score_earthquake.py").read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert dupes == [], f"shadowed definitions: {dupes}"


# ---------------------------------------------------------------------------
# 9. The per-cell sentence
# ---------------------------------------------------------------------------

def test_the_cell_sentence_states_the_rate_model_split_and_nothing_more():
    from hazardpulse.site.pages import earthquake

    quiet = {"lambda_long": 0.02, "lambda_short": 0.12, "n_events": 2}     # 86% "short-term", 2 recent events
    text = earthquake.driver_sentence(quiet)
    assert "86% of this cell&rsquo;s 30-day rate comes from its short-term term" in text
    assert "not a breakdown of the chance" in text and "old earthquakes count" in text
    for claim in ("Recent earthquakes nearby", "aftershock-style clustering", "account for"):
        assert claim not in text, claim
