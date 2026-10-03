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


def test_noaa_hazard_probabilities_are_read_from_the_models_object():
    """ProbSevere v3 puts ProbTor/Hail/Wind in feature["models"], not in
    properties; the old parser stored 0.0 for every storm (MEASURED on the
    2024 test year: 1,004,531 storms, ProbTor max 0)."""
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-97.0, 35.0], [-96.9, 35.0], [-96.9, 35.1]]]},
        "properties": {"ID": "7", "PS": "16", "MAXLLAZ": "0.012"},
        "models": {"probsevere": {"PROB": "61"}, "probtor": {"PROB": "88"},
                   "probhail": {"PROB": "40"}, "probwind": {"PROB": "12"}},
    }
    (s,) = ps._parse_storms({"features": [feature]})
    assert (s["ps_tor"], s["ps_hail"], s["ps_wind"], s["ps_severe"]) == (88.0, 40.0, 12.0, 61.0)
    assert s["ps"] == 16.0
    # pre-v3 files that carried the value in properties still parse
    legacy = dict(feature, models=None, properties={"ID": "8", "PROBTOR": "33"})
    (s2,) = ps._parse_storms({"features": [legacy]})
    assert s2["ps_tor"] == 33.0


def _raiser(code):
    def _open(*a, **k):
        _open.n += 1
        raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))
    _open.n = 0
    return _open


def test_string_valued_noaa_attributes_are_parsed_not_dropped():
    """MAXRC_EMISS / MAXRC_ICECF / AVG_BEAM_HGT arrive as strings with units (verbatim from
    MRMS_PROBSEVERE_20230415_000039.json); a float() filter left them NaN in every row."""
    from hazardpulse.data import probsevere as ps
    data = {"validTime": "20230415_000039 UTC", "features": [{
        "geometry": {"type": "Polygon", "coordinates": [[[-97.0, 35.0], [-96.9, 35.0], [-96.9, 35.1], [-97.0, 35.0]]]},
        "properties": {"ID": "1", "PS": "40", "MAXRC_EMISS": "2251Z 1.5%/min (weak)",
                       "MAXRC_ICECF": "2241Z 0.01/min (strong)", "AVG_BEAM_HGT": "4.09 kft / 1.25 km"}}]}
    s = ps._parse_storms(data)[0]
    assert s["maxrc_emiss"] == 1.5 and s["maxrc_emiss_cat"] == 1.0
    assert s["maxrc_emiss_age_min"] == 69.0          # 22:51 -> 00:00, across midnight
    assert s["maxrc_icecf"] == 0.01 and s["maxrc_icecf_cat"] == 3.0 and s["maxrc_icecf_age_min"] == 79.0
    assert s["avg_beam_hgt"] == 1.25                 # km, not the kft figure
    blank = ps._parse_storms({**data, "features": [{**data["features"][0], "properties": {
        "ID": "2", "MAXRC_EMISS": "N/A", "MAXRC_ICECF": "", "AVG_BEAM_HGT": "N/A"}}]})[0]
    assert not any(k.startswith(("maxrc_", "avg_beam")) for k in blank)   # missing stays missing


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
