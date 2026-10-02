"""ProbSevere archive gaps cost nothing and never poison the cache.

Measured 2026-10-01: for days the archive does not hold (20210515, 20210516 --
the S3 listing answers OK with zero keys), the old fetcher probed 76 guessed
slot names, each 404 retried with back-off, for ~35 minutes per day, then
cached the empty result -- which ``load_cached_probsevere`` returns as ``[]``
(not ``None``) forever after.
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from hazardpulse.data import http as hp_http
from hazardpulse.data import probsevere as ps


def test_authoritative_empty_listing_does_not_probe_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_list_s3_files", lambda d: ([], True))

    def boom(*a, **k):
        raise AssertionError("probed a day the listing says does not exist")

    monkeypatch.setattr(ps, "_fetch_single_timestep", boom)
    assert ps.fetch_probsevere_day("20210515", cache_dir=tmp_path) == []
    assert ps.load_cached_probsevere("20210515", cache_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_failed_listing_still_probes_but_never_caches_an_empty_day(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_list_s3_files", lambda d: ([], False))
    calls = []
    monkeypatch.setattr(ps, "_fetch_single_timestep", lambda *a: calls.append(a) or None)
    assert ps.fetch_probsevere_day("20210515", cache_dir=tmp_path) == []
    assert len(calls) == len(ps.CONVECTIVE_HOURS) * len(ps.SCAN_MINUTES)
    assert ps.load_cached_probsevere("20210515", cache_dir=tmp_path) is None


def test_a_non_empty_day_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_list_s3_files", lambda d: ([], False))
    step = {"valid_time": "20210515_200000 UTC", "storms": [{"id": 1}]}
    monkeypatch.setattr(ps, "_fetch_single_timestep", lambda *a: step)
    out = ps.fetch_probsevere_day("20210515", cache_dir=tmp_path)
    assert len(out) == len(ps.CONVECTIVE_HOURS) * len(ps.SCAN_MINUTES)
    assert ps.load_cached_probsevere("20210515", cache_dir=tmp_path) == out


def _raiser(code):
    def _open(*a, **k):
        _open.n += 1
        raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))
    _open.n = 0
    return _open


def test_http_404_is_not_retried(monkeypatch):
    opener = _raiser(404)
    monkeypatch.setattr(hp_http.urllib.request, "urlopen", opener)
    monkeypatch.setattr(hp_http.time, "sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        hp_http.fetch_bytes("https://example.invalid/x", use_cache=False)
    assert opener.n == 1


def test_http_503_is_retried(monkeypatch):
    opener = _raiser(503)
    monkeypatch.setattr(hp_http.urllib.request, "urlopen", opener)
    monkeypatch.setattr(hp_http.time, "sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        hp_http.fetch_bytes("https://example.invalid/x", use_cache=False)
    assert opener.n == 3
