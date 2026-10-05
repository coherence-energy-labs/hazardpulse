"""Tornado record integrity (tornado audit, 2026-10-05): true receipts (item 4), a ledger checked row by row
(item 5), freshness limits for a 60-minute product (item 7), expiry counted from the data's valid time on every
page that shows the forecast (item 8), the products' clip rate recorded (item 9), and the input-format gap
recorded and shown (item 3)."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.gates import GateContext, GateEngine
from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado import v3_serving as vs

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc


def _script(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _script("integrity_live_tornado", "scripts/fetch_and_score_tornado.py")


def _payload(names, splits, b=-3.0):
    """A hand-built payload (no LightGBM needed): one stump per (input, threshold, left, right)."""
    trees = []
    for name, thr, lv, rv in splits:
        j = names.index(name)
        trees.append({"feature": [j, -1, -1], "threshold": [thr, 0.0, 0.0], "default_left": [True, False, False],
                      "missing_type": [0, 0, 0], "left": [1, -1, -1], "right": [2, -1, -1],
                      "value": [0.0, lv, rv], "node_value": [(lv + rv) / 2, lv, rv]})
    return {"schema": lp.SCHEMA, "feature_names": list(names), "n_trees": len(trees), "trees": trees,
            "calibration": {"a": 1.0, "b": b}, "provenance": {}}


@pytest.fixture(scope="module")
def suite():
    base = list(sf.FEATURE_NAMES[sf.BLOCKS["P"][0]:sf.BLOCKS["P"][1]])
    stumps = [("p_maxllaz", 0.0045, -1.0, 1.0), ("p_ps_tor", 25.0, -0.5, 0.5)]
    w = base + list(vs.W_NAMES)
    return vs.V3Suite(main=_payload(w, stumps + [("w_tor_warning_active", 0.5, -0.3, 1.5)]),
                      fallback=_payload(base, stumps),
                      # a 90-min model that reads LOWER than the 60-min one: every storm's p90 is clipped up
                      products={"p90": _payload(w, stumps, b=-3.5)})


def _steps():
    t0 = dt.datetime(2025, 5, 6, 18, 0, 39, tzinfo=UTC)
    steps = []
    for k in range(3):
        t = t0 + dt.timedelta(minutes=30 * k)
        steps.append({"valid_time": t.strftime("%Y%m%d_%H%M%S UTC"), "storms": [
            {"id": "11", "lat": 35.0, "lon": -97.5, "ps": 60, "ps_tor": 20 + 10 * k, "maxllaz": 0.004 + 0.001 * k,
             "size": 300, "motion_east": 12, "motion_south": -8},
            {"id": "22", "lat": 33.0, "lon": -90.0, "ps": 10, "ps_tor": 1, "maxllaz": 0.001, "size": 80,
             "motion_east": 10, "motion_south": -2},
            {"id": "33", "lat": 31.0, "lon": -88.0, "ps": 10, "ps_tor": 1, "maxllaz": 0.0011, "size": 81,
             "motion_east": 10, "motion_south": -2}]})
    return steps


class _Inputs:
    def __init__(self, m):
        self._m = m

    def matrix(self):
        return self._m


@pytest.fixture
def scored(live, suite, monkeypatch):
    monkeypatch.setattr(live.nws_live, "live_tor_warning_inputs",
                        lambda la, lo, t, **k: _Inputs(np.array([[1.0, 12.0], [0.0, np.nan], [0.0, np.nan]], np.float32)))
    return live.score_storms(_steps(), None, None, None, dt.datetime(2025, 5, 6, 19, 10),
                             scoring_tier="tier1_v3", v3_suite=suite)


def _publish(live, scored, forecaster=None):
    issued = "2025-05-06T19:10:00Z"
    if forecaster is not None:
        live.apply_trust_layer(scored, forecaster, issued)
    live.round_published(scored)
    live.refresh_risk_bands(scored)
    live.withhold_bands_excluding_probability(scored)
    live.stamp_receipts(scored, issued)
    return scored


# ------------------------------------------------------------------------------------------------
# item 4: the receipt's hashes are of the inputs and of the model
# ------------------------------------------------------------------------------------------------

def test_the_receipt_hashes_the_input_vector_and_the_served_payload(live, suite, scored):
    from hazardpulse.trust.forecast import verify_forecast_receipt
    _publish(live, scored)
    payload = {lp.model_version(p): p for p in (suite.main, suite.fallback)}
    hashes = set()
    for s in scored:
        v3, r = s["v3"], s["receipt"]
        p = payload[s["model_version"]]
        assert r["input_sha256"] == v3["input_sha256"] == vs.input_digest(p, v3["inputs"])
        assert r["model_sha256"] == v3["model_sha256"] == hashlib.sha256(lp.canonical_bytes(p)).hexdigest()
        assert s["model_version"].endswith(r["model_sha256"][:12])        # the version names these bytes
        # the old "input" hash was of the published number
        assert r["input_sha256"] != hashlib.sha256(np.asarray([s["tornado_probability"]], np.float64).tobytes()).hexdigest()
        assert verify_forecast_receipt(r) and r["raw_probability"] == v3["probability_60min"]
        assert r["probability"] == s["tornado_probability"] and s["receipt_sha256"] == r["receipt_sha256"]
        hashes.add(r["input_sha256"])
    assert len(hashes) == len(scored)            # each storm's own inputs, even where the rounded chance ties
    # one input changed -> another input hash
    p = payload[scored[0]["model_version"]]
    changed = dict(scored[0]["v3"]["inputs"], p_maxllaz=(scored[0]["v3"]["inputs"]["p_maxllaz"] or 0.0) + 1e-6)
    assert vs.input_digest(p, changed) != scored[0]["v3"]["input_sha256"]


def test_with_a_calibrator_the_receipt_still_hashes_the_inputs_and_binds_model_and_calibrator(live, suite, scored):
    from hazardpulse.trust.forecast import TrustedForecaster, verify_forecast_receipt
    rng = np.random.RandomState(0)
    raw = rng.uniform(0, 0.6, 2000)
    tf = TrustedForecaster(model_version=lp.model_version(suite.main), ood_reject_quantile=None)
    tf.fit(raw, (rng.uniform(size=2000) < raw).astype(int))
    _publish(live, scored, forecaster=tf)
    calibrated = [s for s in scored if s.get("calibrated")]
    assert calibrated
    for s in calibrated:
        r = s["receipt"]
        assert r["input_sha256"] == vs.input_digest(suite.main, s["v3"]["inputs"])
        assert s["calibrator_sha256"] == tf.model_sha256_
        assert r["model_sha256"] == hashlib.sha256(json.dumps(
            {"model_sha256": s["v3"]["model_sha256"], "calibrator_sha256": tf.model_sha256_},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert verify_forecast_receipt(r)


def test_the_record_audit_rejects_a_record_whose_hashes_are_not_true(live, suite, scored):
    audit = _script("integrity_audit_records", "scripts/audit_tornado_records.py")
    _publish(live, scored)
    res = {"hashes_checked": 0, "hashes_matched": 0, "hash_mismatched": []}
    s = scored[0]
    p = suite.main if s["model_version"] == lp.model_version(suite.main) else suite.fallback
    audit._check_hashes(res, "ok", s, p)
    assert res["hashes_matched"] == 1 and not res["hash_mismatched"]
    tampered = json.loads(json.dumps(s))
    tampered["v3"]["inputs"]["p_maxllaz"] = 0.5            # an input edited after publication
    audit._check_hashes(res, "tampered", tampered, p)
    assert res["hash_mismatched"] and "input_sha256" in res["hash_mismatched"][0]


# ------------------------------------------------------------------------------------------------
# item 9: the clip rate of the nested products is recorded
# ------------------------------------------------------------------------------------------------

def test_the_products_clip_rate_is_recorded_per_run(live, scored, tmp_path, monkeypatch):
    rec = live.product_coherence(scored)
    n_clipped = sum(1 for s in scored if "p90" in (s["v3"].get("coherence_clipped") or []))
    assert n_clipped == len(scored) == 3                  # the fixture's 90-min model reads below its 60-min one
    assert rec["n_storms_with_products"] == 3 and rec["clipped"] == {"p30": 0, "p90": 3, "p_ef2": 0}
    assert rec["clipped_rate"]["p90"] == 1.0
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    monkeypatch.setattr(live, "DIST", dist)
    monkeypatch.setattr(live, "LEDGER_PATH", dist / "data" / "tornado-ledger.jsonl")
    live.write_outputs(_publish(live, scored), dt.datetime(2025, 5, 6, 19, 10), "tier1_v3",
                       data_valid_time="2025-05-06T19:00:30Z", product_coherence=rec,
                       input_gaps={"absent": [{"input": "p_ps"}], "partial": [], "changed_format": []})
    out = json.loads((dist / "data" / "live-tornadoes.json").read_text(encoding="utf-8"))
    assert out["product_coherence"] == rec and out["data_valid_time"] == "2025-05-06T19:00:30Z"
    assert out["input_age_at_issue_min"] == 9.5 and out["input_gaps"]["absent"][0]["input"] == "p_ps"


def test_the_record_and_its_provenance_keep_the_runs_diagnostics(tmp_path, monkeypatch):
    bsa = _script("integrity_bsa_record", "scripts/build_site_artifacts.py")
    dist = tmp_path / "dist"
    (dist / "data" / "replay").mkdir(parents=True)
    for name, value in (("ROOT", tmp_path), ("DIST", dist), ("LIVE_PULSE_PATH", dist / "data" / "live-pulse.json"),
                        ("LIVE_TORNADOES_PATH", dist / "data" / "live-tornadoes.json"),
                        ("LIVE_STORMS_PATH", dist / "data" / "live-storms.json"),
                        ("REPLAY_DIR", dist / "data" / "replay"),
                        ("REPLAY_INDEX_PATH", dist / "data" / "evidence" / "replay-index.json")):
        monkeypatch.setattr(bsa, name, value)
    (dist / "data" / "live-pulse.json").write_text(json.dumps({"hazards": [{"key": "to"}]}), encoding="utf-8")
    extra = {"data_valid_time": "2026-10-05T00:24:42Z", "input_age_at_issue_min": 0.7,
             "input_gaps": {"absent": [{"input": "p_ps", "fed": "zero"}]},
             "product_coherence": {"clipped": {"p90": 3}}}
    (dist / "data" / "live-tornadoes.json").write_text(json.dumps({
        "updated_at": "2026-10-05T00:25:23Z", "model_version": "tornado_v3-x", "storms": [], **extra}),
        encoding="utf-8")
    bsa._ensure_live_publish_artifacts()
    record = json.loads((dist / "data" / "replay" / "to_fcst_20261005_0025.json").read_text(encoding="utf-8"))
    assert {k: record.get(k) for k in extra} == extra
    (env,) = bsa._build_provenance_envelopes([{"forecast_id": "to_fcst_20261005_0025", "hazard": "tornado",
                                               "replay_artifact": "/data/replay/to_fcst_20261005_0025.json"}])
    manifest = {"source_artifacts": record["source_artifacts"], "scoring_tier": None, "coherence_source": None,
                "n_active_storms": 0, "storm_ids": []}
    assert env["input_hash"] == "sha256:" + bsa._canonical_hash(
        {**manifest, **{k: extra[k] for k in ("data_valid_time", "input_gaps", "product_coherence")}})
    # an envelope of a record without them hashes exactly what it hashed before
    del record["input_gaps"], record["product_coherence"], record["data_valid_time"]
    (dist / "data" / "replay" / "to_fcst_20261005_0025.json").write_text(json.dumps(record), encoding="utf-8")
    bsa._REPLAY_READ_CACHE.clear()
    (env,) = bsa._build_provenance_envelopes([{"forecast_id": "to_fcst_20261005_0025", "hazard": "tornado",
                                               "replay_artifact": "/data/replay/to_fcst_20261005_0025.json"}])
    assert env["input_hash"] == "sha256:" + bsa._canonical_hash(manifest)


# ------------------------------------------------------------------------------------------------
# item 5: every ledger row is checked against its own hash
# ------------------------------------------------------------------------------------------------

def _row(prev: str, ts: str, **extra) -> dict:
    row = {"timestamp": ts, "model_version": "m", "n_storms": 1, "top_probability": 0.01, "prev_hash": prev,
           "storms": [{"id": 1, "prob": 0.01, "risk": "minimal"}], **extra}
    row["hash"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return row


def _ledger(tmp_path, tamper: bool) -> Path:
    rows, prev = [], "0" * 64
    for i in range(4):
        r = _row(prev, f"2026-03-2{i}T00:00:00Z")
        if i < 2:
            r["forecast_id"] = f"to_fcst_2026032{i}_0000"       # added AFTER hashing: the 170 legacy rows
        rows.append(r)
        prev = r["hash"]
    if tamper:
        rows[3]["top_probability"] = 0.9                         # an edit to history keeps its old hash
    path = tmp_path / "tornado-ledger.jsonl"
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
    return path


def test_each_ledger_row_is_verified_and_the_legacy_rows_are_named_not_hidden(tmp_path):
    bsa = _script("integrity_bsa_ledger", "scripts/build_site_artifacts.py")
    good = bsa._verify_tornado_ledger(_ledger(tmp_path, tamper=False))
    assert good["n_rows"] == 4 and good["prev_hash_mismatches"] == 0
    assert good["content_hash_verified"] == 2 and good["content_hash_verified_without_forecast_id"] == 2
    assert good["content_hash_mismatches"] == 0
    assert "hashed before forecast_id was added" in good["content_hash_legacy_note"]
    # the old check (prev_hash links only) passes the tampered ledger; the row check does not
    bad_path = _ledger(tmp_path, tamper=True)
    assert bsa._count_link_mismatches(bad_path) == (4, 0)
    bad = bsa._verify_tornado_ledger(bad_path)
    assert bad["content_hash_mismatches"] == 1 and bad["content_hash_mismatched_rows"][0]["row"] == 3


def test_the_committed_tornado_ledger_verifies_row_by_row():
    bsa = _script("integrity_bsa_ledger_live", "scripts/build_site_artifacts.py")
    res = bsa._verify_tornado_ledger(ROOT / "dist" / "data" / "tornado-ledger.jsonl")
    assert res["n_rows"] > 1800 and res["content_hash_mismatches"] == 0 and res["prev_hash_mismatches"] == 0
    assert res["content_hash_verified_without_forecast_id"] == 170          # 2026-03-22..2026-04-13, measured


# ------------------------------------------------------------------------------------------------
# item 7: freshness limits for a 60-minute product, counted from its data's valid time
# ------------------------------------------------------------------------------------------------

def _g1(age_h: float) -> tuple[str, str | None]:
    d = GateEngine().evaluate(GateContext(hazard="tornado", forecast_id="to_x", model_version="m",
                                          probability=0.01, replay_artifact="/r", data_age_seconds=age_h * 3600))
    g = next(r for r in d.gate_results if r.gate_id == "G1_SOURCE_FRESHNESS")
    return g.outcome, g.reason


def test_tornado_freshness_degrades_past_two_and_a_half_hours_and_blocks_past_six():
    assert _g1(2.0)[0] == "pass"                 # the quiet cadence is 2 h
    outcome, reason = _g1(3.0)
    assert outcome == "degrade" and "> 2.5h" in reason
    outcome, reason = _g1(6.5)
    assert outcome == "block" and "> 6h" in reason


def test_a_tornado_forecasts_age_counts_from_its_datas_valid_time(tmp_path, monkeypatch):
    bsa = _script("integrity_bsa_gates", "scripts/build_site_artifacts.py")
    dist = tmp_path / "dist"
    (dist / "data" / "replay").mkdir(parents=True)
    (tmp_path / "results" / "calibration").mkdir(parents=True)
    monkeypatch.setattr(bsa, "ROOT", tmp_path)
    monkeypatch.setattr(bsa, "DIST", dist)
    bsa._REPLAY_READ_CACHE.clear()
    path = dist / "data" / "replay" / "to_fcst_20261005_0025.json"
    path.write_text(json.dumps({"forecast_id": "to_fcst_20261005_0025", "hazard": "tornado",
                                "issued_at": "2026-10-05T00:25:23Z", "model_version": "m",
                                "storms": [{"valid_time": "20261005_000038 UTC", "lat": 35, "lon": -97,
                                            "tornado_probability": 0.01}]}), encoding="utf-8")
    entry = {"forecast_id": "to_fcst_20261005_0025", "hazard": "tornado", "issued_at": "2026-10-05T00:25:23Z",
             "model_version": "m", "replay_artifact": "/data/replay/" + path.name}
    # 2 h 15 min after issue: under 2.5 h from the issue time, over it from the data (00:00:38)
    (d,) = bsa._build_gate_decisions([entry], {"hazards": []}, now=dt.datetime(2026, 10, 5, 2, 40, tzinfo=UTC))
    g1 = next(g for g in d["gates"] if g["gate_id"] == "G1_SOURCE_FRESHNESS")
    assert g1["outcome"] == "degrade" and "2.7h" in g1["reason"]


# ------------------------------------------------------------------------------------------------
# items 3 and 8 on the pages: the input gap in words, the expiry counted from the data
# ------------------------------------------------------------------------------------------------

def _to_record(**extra):
    return {"updated_at": "2026-10-05T00:25:23Z", "forecast_id": "to_fcst_20261005_0025", "storms": [
        {"storm_id": "1", "lat": 35.0, "lon": -97.0, "valid_time": "20261005_000038 UTC",
         "tornado_probability": 0.0004}], **extra}


def test_the_tornado_page_counts_its_window_from_the_data_and_names_the_input_gap():
    from hazardpulse.site.pages import tornado as page
    rec = _to_record()
    assert page.data_valid_time(rec) == "2026-10-05T00:00:38Z"          # older records: the storms' valid time
    assert page.data_valid_time(dict(rec, data_valid_time="2026-10-05T00:24:42Z")) == "2026-10-05T00:24:42Z"
    gaps = {"census": "newest file", "absent": [
        {"input": "p_ps", "label": "ProbSevere severe probability", "fed": "zero", "model_splits": 196},
        {"input": "p_vil_density", "label": "VIL density", "fed": "zero", "model_splits": 475},
        {"input": "p_maxrc_icecf", "label": "satellite glaciation rate", "fed": "missing", "model_splits": 0}],
        "partial": [], "changed_format": []}
    html = page.input_gap_notice(_to_record(input_gaps=gaps))
    assert "Known input gap" in html and "ProbSevere severe probability and VIL density" in html
    assert "receives 0 for them" in html and "1 more input the model does not use" in html
    assert "style=" not in html
    assert page.input_gap_notice(_to_record()) == ""
    assert page.input_gap_notice(_to_record(input_gaps={"census": "unavailable", "absent": [], "partial": [],
                                                        "changed_format": []})) == ""


def test_the_expiry_notices_carry_the_data_valid_time_and_the_cards_carry_one():
    from hazardpulse.site.pages import common
    full = common.forecast_age("2026-10-05T00:25:23Z", 60, "every 2 hours", valid_at="2026-10-05T00:00:38Z")
    assert 'data-valid="2026-10-05T00:00:38Z"' in full and 'data-issued="2026-10-05T00:25:23Z"' in full
    short = common.forecast_age("2026-10-05T00:25:23Z", 60, "every 2 hours", valid_at="2026-10-05T00:00:38Z",
                                short=True)
    assert short.startswith('<span class="forecast-age" data-format="short"') and "style=" not in short
    # the pages: home and /live/ carry the short note on the tornado card, /live/tornado/ the full one
    for rel in ("index.html", "live/index.html"):
        text = (ROOT / "dist" / rel).read_text(encoding="utf-8")
        card = text[text.index('hazard-card hz-to'):]
        card = card[:card.index("</a>")]
        assert 'class="forecast-age" data-format="short"' in card and "data-valid=" in card, rel
    text = (ROOT / "dist" / "live" / "tornado" / "index.html").read_text(encoding="utf-8")
    assert 'class="notice notice-warn forecast-age"' in text and "data-valid=" in text
