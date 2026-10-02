import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from hazardpulse.data import earthquake as earthquake_data
from hazardpulse.earthquake.coherence_engine import grid_cell_to_latlon
from hazardpulse.earthquake.prospective import (
    forecast_id_for_time,
    format_utc_z,
    iter_monthly_windows,
    parse_utc_datetime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_prospective_time_helpers_are_stable():
    issued = parse_utc_datetime("2026-04-02T12:34:56Z")
    assert issued.tzinfo == dt.timezone.utc
    assert format_utc_z(issued) == "2026-04-02T12:34:56Z"

    rounded = dt.datetime(2026, 4, 2, 12, 0, tzinfo=dt.timezone.utc)
    assert forecast_id_for_time(rounded) == "eq_fcst_20260402_1200"

    windows = list(
        iter_monthly_windows(
            dt.datetime(2026, 1, 15, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc),
        )
    )
    assert windows == [
        (
            dt.datetime(2026, 1, 15, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc),
        ),
    ]


def test_load_usgs_catalog_bootstraps_missing_years(monkeypatch, tmp_path):
    monkeypatch.setattr(earthquake_data, "USGS_DIR", tmp_path)

    class DummyDownloader:
        def download_usgs_year(self, year: int, **_kw):
            # A complete-looking year (the loader now audits completeness): one
            # sub-threshold event every day plus the M4.2 event under test.
            day = dt.date(year, 1, 1)
            lines = ["time,latitude,longitude,depth,mag,magType,place,type,id"]
            while day.year == year:
                lines.append(f"{day.isoformat()}T06:00:00.000Z,10.0,20.0,5.0,2.0,mb,Test,event,bg-{day}")
                day += dt.timedelta(days=1)
            lines.append(f"{year}-01-15T00:00:00.000Z,10.0,20.0,5.0,4.2,mb,Test,event,evt-{year}")
            (tmp_path / f"usgs_catalog_{year}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(earthquake_data, "_load_download_module", lambda: DummyDownloader())

    events = earthquake_data.load_usgs_catalog(min_year=2024, max_year=2024, min_mag=2.5)
    assert len(events) == 1
    assert events[0]["id"] == "evt-2024"
    assert events[0]["mag"] == 4.2


def test_write_replay_artifact_serializes_nan_values(tmp_path, monkeypatch):
    fetch_module = _load_script_module(
        "hazardpulse_fetch_and_score_earthquake_test",
        "scripts/fetch_and_score_earthquake.py",
    )
    monkeypatch.setattr(fetch_module, "DIST", tmp_path)

    issued = dt.datetime(2026, 4, 2, 0, 0, tzinfo=dt.timezone.utc)
    replay_path = fetch_module.write_replay_artifact(
        [
            {
                "row": np.int64(1),
                "col": np.int64(2),
                "probability": np.float64(0.25),
                "risk_band": "low",
                "b_trend": float("nan"),
                "conditions_met": np.int64(3),
                "diagnostic_ok": np.bool_(True),
            }
        ],
        issued,
        forecast_id="eq_fcst_20260402_0000",
        n_history_events=123,
        n_recent_events=7,
        replay_dir=tmp_path / "data" / "replay",
        update_index=False,
    )

    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["forecast_id"] == "eq_fcst_20260402_0000"
    assert payload["active_cells"][0]["b_trend"] is None
    assert payload["active_cells"][0]["row"] == 1
    assert payload["active_cells"][0]["diagnostic_ok"] is True


def test_append_ledger_skips_duplicate_forecasts(tmp_path):
    fetch_module = _load_script_module(
        "hazardpulse_fetch_and_score_earthquake_ledger_test",
        "scripts/fetch_and_score_earthquake.py",
    )

    ledger_path = tmp_path / "earthquake-ledger.jsonl"
    replay_path = tmp_path / "eq_fcst_20260402_0000.json"
    replay_path.write_text("{}", encoding="utf-8")
    issued = dt.datetime(2026, 4, 2, 0, 0, tzinfo=dt.timezone.utc)
    scored = [
        {
            "lat": 10.0,
            "lon": 20.0,
            "probability": 0.4,
            "conditions_met": 3,
            "max_mag": 5.8,
        }
    ]

    fetch_module.append_ledger(
        scored,
        issued,
        forecast_id="eq_fcst_20260402_0000",
        ledger_path=ledger_path,
        replay_path=replay_path,
    )
    fetch_module.append_ledger(
        scored,
        issued,
        forecast_id="eq_fcst_20260402_0000",
        ledger_path=ledger_path,
        replay_path=replay_path,
    )
    fetch_module.append_ledger(
        scored,
        issued + dt.timedelta(hours=6),
        forecast_id="eq_fcst_20260402_0600",
        ledger_path=ledger_path,
        replay_path=replay_path,
    )

    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["forecast_id"] == "eq_fcst_20260402_0000"
    assert lines[1]["forecast_id"] == "eq_fcst_20260402_0600"
    assert lines[1]["prev_hash"] == lines[0]["hash"]


def test_score_module_filters_matured_forecasts_and_scores_hits(tmp_path):
    score_module = _load_script_module(
        "hazardpulse_score_earthquake_prospective_test",
        "scripts/score_earthquake_prospective.py",
    )

    score_as_of = dt.datetime(2026, 4, 2, 0, 0, tzinfo=dt.timezone.utc)
    artifacts = [
        {
            "forecast_id": "eq_fcst_20260201_0000",
            "issued_at": "2026-02-01T00:00:00Z",
            "forecast_horizon_days": 30,
        },
        {
            "forecast_id": "eq_fcst_20260310_0000",
            "issued_at": "2026-03-10T00:00:00Z",
            "forecast_horizon_days": 30,
        },
    ]
    matured = score_module.matured_artifacts(artifacts, score_as_of)
    assert [artifact["forecast_id"] for artifact in matured] == ["eq_fcst_20260201_0000"]

    lat, lon = grid_cell_to_latlon(0, 0)
    artifact = {
        "forecast_id": "eq_fcst_20260201_0000",
        "issued_at": "2026-02-01T00:00:00Z",
        "forecast_horizon_days": 30,
        "forecast_domain": {
            "n_lat": 2,
            "n_lon": 2,
            "default_probability": 0.0,
        },
        "active_cells": [
            {"row": 0, "col": 0, "probability": 0.8},
            {"row": 1, "col": 1, "probability": 0.1},
        ],
    }
    observed = [
        {
            "time": "2026-02-10T00:00:00Z",
            "latitude": lat,
            "longitude": lon,
            "depth": 10.0,
            "mag": 6.2,
            "id": "evt-1",
        }
    ]

    result = score_module.score_single_forecast(artifact, observed, tmp_path)
    assert result["forecast_id"] == "eq_fcst_20260201_0000"
    assert result["n_observed_events"] == 1
    assert result["n_positive_cells"] == 1
    assert result["top_1_hit"] is True
    assert result["top_5_hit"] is True
    assert 0.0 <= result["auc"] <= 1.0
