"""Truth tests for scripts/score_tornado_prospective.py (audit finding #4, 2026-10-01).

1. SPC daily report files are CONVECTIVE days (12 UTC -> 12 UTC) timed in UTC, so a
   row timed 0241 in file 240426 happened at 2024-04-27T02:41Z. The scorer used to
   place it on 2024-04-26, 24 h early, and the +-4 h match could never see it.
2. A failed SPC fetch is "outcome unavailable", never "no tornado".
3. Skill is pooled against a causal climatology; a per-forecast BSS against the
   forecast's own all-zero outcomes is undefined, not 0.

The scorer is executed from a copy inside tmp_path, so everything it writes
(including dist/data/tornado-recovery.json, resolved from its own __file__) lands
in the sandbox, never in the repository.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import json
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "scripts" / "score_tornado_prospective.py"

HEADER = "Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
# Real rows from https://www.spc.noaa.gov/climo/reports/240426_rpts_filtered_torn.csv
SPC_240426 = (
    HEADER
    + "1716,UNK,2 ESE Ravenna,Buffalo,NE,41.02,-98.87,This tornado touched down at 1216 PM CDT (GID)\n"
    + "0241,UNK,Monroe,Jasper,IA,41.52,-93.1,*** 1 INJ *** Reports of damage in Monroe. (DMX)\n"
)


def _load_sandboxed(tmp_path: Path, source: Path = SCORER):
    (tmp_path / "scripts").mkdir(exist_ok=True)
    target = tmp_path / "scripts" / "score_tornado_prospective.py"
    shutil.copyfile(source, target)
    spec = importlib.util.spec_from_file_location("stp_under_test", target)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve_spc(monkeypatch, files: dict[str, str], failing: set[str] = frozenset()):
    """Serve SPC CSVs by YYMMDD; unknown days are real no-tornado days (header only)."""

    def fake_urlopen(req, timeout=None, context=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        key = url.rsplit("/", 1)[-1][:6]
        if key in failing:
            raise urllib.error.URLError("simulated outage")
        return _Resp(files.get(key, HEADER).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _write_forecast(replay: Path, issued: str, storms: list[dict], tier: str = "tier1_ml") -> str:
    t = dt.datetime.fromisoformat(issued.replace("Z", "+00:00"))
    fid = f"to_fcst_{t.strftime('%Y%m%d_%H%M')}"
    replay.mkdir(parents=True, exist_ok=True)
    (replay / f"{fid}.json").write_text(json.dumps({
        "forecast_id": fid,
        "hazard": "tornado",
        "issued_at": issued,
        "forecast_horizon_hours": 24,
        "scoring_tier": tier,
        "storms": storms,
    }), encoding="utf-8")
    return fid


def _storm(lat, lon, valid, prob):
    return {"lat": lat, "lon": lon, "valid_time": valid, "tornado_probability": prob}


def _run(m, tmp_path, *extra):
    out = tmp_path / "out"
    rc = m.main(["--replay-dir", str(tmp_path / "replay"), "--output-dir", str(out),
                 "--score-as-of", "2024-05-01T00:00:00Z", *extra])
    assert rc == 0
    summary = json.loads((out / "prospective_summary.json").read_text())
    rows = [json.loads(line) for line in (out / "per_forecast_scores.jsonl").read_text().splitlines() if line]
    return summary, rows


def test_overnight_spc_report_is_matched_on_its_utc_date(tmp_path, monkeypatch):
    m = _load_sandboxed(tmp_path)
    _serve_spc(monkeypatch, {"240426": SPC_240426})
    # Monroe, IA storm observed 02:30Z on 27 April; the report 0241 in file 240426 is 02:41Z on 27 April.
    _write_forecast(tmp_path / "replay", "2024-04-27T02:00:00Z", [
        _storm(41.52, -93.10, "20240427_023000 UTC", 0.30),
        _storm(30.00, -85.00, "20240427_023000 UTC", 0.20),
    ])
    summary, rows = _run(m, tmp_path)
    assert rows[0]["n_matched_storms"] == 1          # was 0: report placed on 26 April
    assert summary["total_matched_storms"] == 1
    assert (tmp_path / "dist" / "data" / "tornado-recovery.json").exists()


def test_spc_csv_rows_before_1200_belong_to_the_next_utc_date(tmp_path):
    m = _load_sandboxed(tmp_path)
    reports = m.parse_spc_reports_csv(SPC_240426, dt.date(2024, 4, 26))
    assert [r["time"] for r in reports] == ["2024-04-26T17:16:00Z", "2024-04-27T02:41:00Z"]
    lo, hi = m.convective_day_span(dt.date(2024, 4, 26))
    assert (lo, hi) == (dt.datetime(2024, 4, 26, 12), dt.datetime(2024, 4, 27, 12))
    with pytest.raises(ValueError):
        m.spc_report_time_utc(dt.date(2024, 4, 26), "2460")


def test_a_failed_spc_fetch_is_outcome_unavailable_not_no_tornado(tmp_path, monkeypatch):
    m = _load_sandboxed(tmp_path)
    _serve_spc(monkeypatch, {"240426": SPC_240426}, failing={"240426"})
    _write_forecast(tmp_path / "replay", "2024-04-27T02:00:00Z", [_storm(41.52, -93.10, "20240427_023000 UTC", 0.30)])
    _write_forecast(tmp_path / "replay", "2024-04-29T13:00:00Z", [_storm(35.0, -97.0, "20240429_130000 UTC", 0.10)])
    summary, rows = _run(m, tmp_path)
    assert summary["observed_window"]["spc_days_unavailable"] == ["2024-04-26"]
    assert summary["n_forecasts_outcome_unavailable"] == 1
    assert summary["n_scored_forecasts"] == 1
    assert [r["forecast_id"] for r in rows] == ["to_fcst_20240429_1300"]


def test_per_forecast_bss_is_undefined_without_an_observed_tornado(tmp_path):
    m = _load_sandboxed(tmp_path)
    assert math.isnan(m.brier_skill_score(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])))
    assert m.brier_skill_score(np.array([0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_auc_counts_ties_as_half_whatever_the_order(tmp_path):
    m = _load_sandboxed(tmp_path)
    assert m.compute_auc(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert m.compute_auc(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    y = np.array([0, 0, 1, 1, 0, 1.0])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.4, 0.4])
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    expected = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    assert m.compute_auc(y, s) == pytest.approx(expected)


def test_causal_climatology_uses_only_outcomes_matured_before_issue(tmp_path):
    m = _load_sandboxed(tmp_path)
    t0 = dt.datetime(2024, 5, 1)
    issued = [t0, t0 + dt.timedelta(hours=12), t0 + dt.timedelta(hours=30), t0 + dt.timedelta(hours=60)]
    ends = [t + dt.timedelta(hours=24) for t in issued]
    n_storms = np.array([10.0, 10.0, 20.0, 5.0])
    n_pos = np.array([1.0, 3.0, 0.0, 5.0])
    ref = m.causal_climatology(issued, ends, n_storms, n_pos)
    assert math.isnan(ref[0]) and math.isnan(ref[1])      # nothing had matured yet
    assert ref[2] == pytest.approx(1 / 10)                  # only forecast 0 matured by t0+30h
    assert ref[3] == pytest.approx((1 + 3 + 0) / 40)        # 0,1,2 matured; its own outcome excluded

    y = np.array([0, 1, 0, 0.0])
    p = np.array([0.1, 0.1, 0.2, 0.0])
    refs = np.array([np.nan, 0.25, 0.25, 0.25])
    out = m.pooled_metrics(y, p, refs)
    bs_model = np.mean((p[1:] - y[1:]) ** 2)
    bs_ref = np.mean((0.25 - y[1:]) ** 2)
    assert out["bss_vs_causal_climatology"] == pytest.approx(1 - bs_model / bs_ref, abs=1e-4)
    assert out["n_with_causal_reference"] == 3
    assert out["bss_vs_sample_climatology"] == pytest.approx(1 - np.mean((p - y) ** 2) / (0.25 * 0.75), abs=1e-4)


def test_summary_headline_is_pooled_and_recovery_is_anchored_on_maturity(tmp_path, monkeypatch):
    m = _load_sandboxed(tmp_path)
    _serve_spc(monkeypatch, {"240426": SPC_240426})
    _write_forecast(tmp_path / "replay", "2024-04-27T02:00:00Z", [
        _storm(41.52, -93.10, "20240427_023000 UTC", 0.30),
        _storm(30.00, -85.00, "20240427_023000 UTC", 0.20),
    ])
    _write_forecast(tmp_path / "replay", "2024-04-29T12:00:00Z", [_storm(35.0, -97.0, "20240429_120000 UTC", 0.05)])
    summary, _ = _run(m, tmp_path)
    pooled = summary["pooled"]
    assert pooled["n_storm_forecasts"] == 3 and pooled["n_positive"] == 1
    assert pooled["brier"] == pytest.approx(((0.3 - 1) ** 2 + 0.2 ** 2 + 0.05 ** 2) / 3, abs=1e-8)
    assert "mean_brier_skill_score" not in summary
    assert summary["per_forecast_bss_own_climatology"]["n_undefined_no_observed_tornado"] == 1
    # Forecast 2 matured 2024-04-30T12:00Z, inside the 24 h before score_as_of.
    assert summary["by_tier_recovery"]["tier1_ml"]["last_24h"]["n_forecasts"] == 1
