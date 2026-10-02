"""USGS catalog completeness: the limit-aware fetcher, the completeness audit, the
downloader and the loader. Regression for the 2026-10 finding that one
``limit=20000`` request per year silently kept only each year's first 20,000 events
(2000-2025: 24.1% of M2.5+, 22.0% of M6+ missing)."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import io
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from hazardpulse.data import earthquake as earthquake_data
from hazardpulse.data import usgs_fdsn

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
FIELDS = ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "type", "id"]


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _synthetic_year(year: int, per_day: int) -> list[dict]:
    """A global-catalog-like year: ``per_day`` events every day, evenly spaced."""
    rows = []
    t = dt.datetime(year, 1, 1, tzinfo=UTC)
    end = dt.datetime(year + 1, 1, 1, tzinfo=UTC)
    step = dt.timedelta(seconds=86400 / per_day)
    i = 0
    while t < end:
        rows.append({"time": _iso(t), "latitude": "10.0", "longitude": "20.0", "depth": "10",
                     "mag": "3.1" if i % 50 else "6.2", "magType": "mb", "place": "Somewhere, X",
                     "type": "earthquake", "id": f"ev{year}{i:07d}"})
        t += step
        i += 1
    return rows


class FakeFDSN:
    """Serves FDSN CSV like the real endpoint: inclusive endtime, time-asc, and the
    rows SILENTLY cut at ``limit`` (the behaviour that truncated the cache)."""

    def __init__(self, rows: list[dict], fail_on: str | None = None):
        self.rows = rows
        self.times = [usgs_fdsn.parse_event_time(r["time"]) for r in rows]
        self.calls: list[str] = []
        self.fail_on = fail_on

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        if self.fail_on and self.fail_on in q["starttime"]:
            raise OSError("simulated network failure")
        start = usgs_fdsn.parse_event_time(q["starttime"])
        end = usgs_fdsn.parse_event_time(q["endtime"])
        limit = int(q["limit"])
        sel = [r for r, t in zip(self.rows, self.times) if start <= t <= end][:limit]
        if not sel:
            return ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sel)
        return buf.getvalue()


def _old_single_year_request(fake: FakeFDSN, year: int) -> list[dict]:
    """What the pre-fix downloader did: one request per year, limit 20000."""
    url = usgs_fdsn.query_url(dt.datetime(year, 1, 1, tzinfo=UTC),
                              dt.datetime(year + 1, 1, 1, tzinfo=UTC), min_magnitude=2.5)
    return list(csv.DictReader(io.StringIO(fake(url))))


def test_old_yearly_request_truncates_and_the_audit_catches_it():
    rows = _synthetic_year(2018, per_day=100)                 # 36,500 events > 20,000
    old = _old_single_year_request(FakeFDSN(rows), 2018)
    assert len(old) == 20000 and old[-1]["time"].startswith("2018-07")   # cut mid-year
    problems = usgs_fdsn.audit_year_catalog([r["time"] for r in old], 2018,
                                            now=dt.datetime(2026, 10, 1, tzinfo=UTC))
    text = " ".join(problems)
    assert "truncation signature" in text and "month(s) with no events" in text
    assert "before the period end" in text


def test_fetch_window_bisects_until_every_response_is_under_the_limit():
    rows = _synthetic_year(2018, per_day=100)
    fake = FakeFDSN(rows)
    stats: dict = {}
    start, end = dt.datetime(2018, 1, 1, tzinfo=UTC), dt.datetime(2019, 1, 1, tzinfo=UTC)
    _, got = usgs_fdsn.fetch_window(start, end, min_magnitude=2.5, fetch_text=fake,
                                    limit=5000, stats=stats)
    assert len(got) == len(rows)
    assert {r["id"] for r in got} == {r["id"] for r in rows}
    assert stats["splits"] > 0 and stats["max_rows_per_response"] == 5000


def test_fetch_window_refuses_when_a_window_cannot_be_split():
    t = dt.datetime(2020, 5, 1, tzinfo=UTC)
    rows = [{"time": _iso(t), "latitude": "0", "longitude": "0", "depth": "0", "mag": "3",
             "magType": "mb", "place": "p", "type": "earthquake", "id": f"same{i}"} for i in range(30)]
    with pytest.raises(usgs_fdsn.USGSTruncationError):
        usgs_fdsn.fetch_window(t, t + dt.timedelta(seconds=10), min_magnitude=2.5,
                               fetch_text=FakeFDSN(rows), limit=10)


def test_month_boundaries_are_half_open_and_deduplicated():
    rows = _synthetic_year(2019, per_day=24)                  # one event exactly at 00:00 each day
    fake = FakeFDSN(rows)
    _, got = usgs_fdsn.fetch_catalog(dt.datetime(2019, 1, 1, tzinfo=UTC),
                                     dt.datetime(2020, 1, 1, tzinfo=UTC),
                                     min_magnitude=2.5, fetch_text=fake)
    assert len(got) == len(rows) == len({r["id"] for r in got})


def test_non_csv_response_is_an_error_not_an_empty_month():
    with pytest.raises(usgs_fdsn.USGSResponseError):
        usgs_fdsn.fetch_window(dt.datetime(2020, 1, 1, tzinfo=UTC), dt.datetime(2020, 2, 1, tzinfo=UTC),
                               min_magnitude=2.5, fetch_text=lambda url: "<html>Service busy</html>")


def test_audit_flags_missing_months_like_the_m2_full_cache_gap():
    rows = [r for r in _synthetic_year(2005, per_day=10) if r["time"] < "2005-10"]
    problems = usgs_fdsn.audit_year_catalog([r["time"] for r in rows], 2005,
                                            now=dt.datetime(2026, 10, 1, tzinfo=UTC))
    assert any("3 month(s) with no events: 2005-10, 2005-11, 2005-12" in p for p in problems)


def test_audit_passes_a_complete_year_and_a_partial_running_year():
    now = dt.datetime(2026, 10, 1, 12, tzinfo=UTC)
    full = _synthetic_year(2024, per_day=10)
    assert usgs_fdsn.audit_year_catalog([r["time"] for r in full], 2024, now=now) == []
    running = [r for r in _synthetic_year(2026, per_day=10) if r["time"] < "2026-10-01T12"]
    assert usgs_fdsn.audit_year_catalog([r["time"] for r in running], 2026, now=now) == []


def _downloader():
    spec = importlib.util.spec_from_file_location("hp_dl_test", ROOT / "scripts" / "download_earthquake_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hp_dl_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_downloader_writes_the_complete_year_with_a_manifest(tmp_path):
    dl = _downloader()
    rows = _synthetic_year(2018, per_day=100)
    now = dt.datetime(2026, 10, 1, tzinfo=UTC)
    path = dl.download_usgs_year(2018, usgs_dir=tmp_path, fetch_text=FakeFDSN(rows),
                                 now=now, pause_seconds=0.0)
    with path.open(encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == len(rows)
    manifest = usgs_fdsn.read_manifest(path)
    assert manifest["n_events"] == len(rows) and manifest["max_rows_per_response"] < 20000
    assert dl.audit_usgs_year_file(path, 2018, now=now) == []


def test_downloader_raises_and_writes_nothing_when_a_month_fails(tmp_path):
    dl = _downloader()
    fake = FakeFDSN(_synthetic_year(2018, per_day=10), fail_on="2018-06-01")
    with pytest.raises(OSError):
        dl.download_usgs_year(2018, usgs_dir=tmp_path, fetch_text=fake,
                              now=dt.datetime(2026, 10, 1, tzinfo=UTC), pause_seconds=0.0)
    assert not (tmp_path / "usgs_catalog_2018.csv").exists()


def test_downloader_replaces_a_legacy_truncated_cache_file(tmp_path):
    dl = _downloader()
    rows = _synthetic_year(2018, per_day=100)
    legacy = _old_single_year_request(FakeFDSN(rows), 2018)
    with (tmp_path / "usgs_catalog_2018.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(legacy)
    path = dl.download_usgs_year(2018, usgs_dir=tmp_path, fetch_text=FakeFDSN(rows),
                                 now=dt.datetime(2026, 10, 1, tzinfo=UTC), pause_seconds=0.0)
    with path.open(encoding="utf-8") as fh:
        assert sum(1 for _ in csv.DictReader(fh)) == len(rows)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_loader_refuses_a_truncated_year_it_cannot_repair(monkeypatch, tmp_path):
    rows = _synthetic_year(2018, per_day=100)
    _write_rows(tmp_path / "usgs_catalog_2018.csv", _old_single_year_request(FakeFDSN(rows), 2018))
    monkeypatch.setattr(earthquake_data, "USGS_DIR", tmp_path)

    class Offline:
        def download_usgs_year(self, year, **kw):
            raise OSError("offline")

    monkeypatch.setattr(earthquake_data, "_load_download_module", lambda: Offline())
    with pytest.raises(usgs_fdsn.USGSCatalogIncompleteError, match="truncation signature"):
        earthquake_data.load_usgs_catalog(min_year=2018, max_year=2018, min_mag=2.5)


def test_loader_repairs_a_truncated_year_through_the_downloader(monkeypatch, tmp_path):
    rows = _synthetic_year(2018, per_day=100)
    _write_rows(tmp_path / "usgs_catalog_2018.csv", _old_single_year_request(FakeFDSN(rows), 2018))
    monkeypatch.setattr(earthquake_data, "USGS_DIR", tmp_path)
    dl = _downloader()

    class Online:
        def download_usgs_year(self, year, usgs_dir=None, **kw):
            return dl.download_usgs_year(year, usgs_dir=usgs_dir, fetch_text=FakeFDSN(rows),
                                         pause_seconds=0.0)

    monkeypatch.setattr(earthquake_data, "_load_download_module", lambda: Online())
    events = earthquake_data.load_usgs_catalog(min_year=2018, max_year=2018, min_mag=6.0)
    assert len(events) == sum(1 for r in rows if float(r["mag"]) >= 6.0)


def test_loader_refuses_a_missing_year(monkeypatch, tmp_path):
    monkeypatch.setattr(earthquake_data, "USGS_DIR", tmp_path)
    monkeypatch.setattr(earthquake_data, "_load_download_module", lambda: None)
    with pytest.raises(usgs_fdsn.USGSCatalogIncompleteError, match="missing"):
        earthquake_data.load_usgs_catalog(min_year=2019, max_year=2019)
