"""Live verification scores v3 storms against the event v3 forecasts, not the old 40 km / 4 h one."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ver():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("score_tornado_prospective_v3", ROOT / "scripts" / "score_tornado_prospective.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _square(lat, lon, half=0.05):
    ring = [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half]]
    return {"type": "Polygon", "coordinates": [ring]}


T0 = dt.datetime(2025, 5, 6, 18, 0, 39)


def _artifact(model_version):
    return {"forecast_id": "to_fcst_test", "issued_at": "2025-05-06T18:05:00Z", "storms": [
        {"storm_id": "77", "lat": 35.0, "lon": -97.0, "motion_east": 15.0, "motion_south": 0.0,
         "geometry": _square(35.0, -97.0), "valid_time": "20250506_180039 UTC",
         "tornado_probability": 0.3, "model_version": model_version}]}


def _fetch_moving_north(day):
    # archived track: the storm went NORTH (its motion vector said east)
    if day != "20250506":
        return [{"valid_time": "20250507_000039 UTC", "storms": []}]
    out = []
    for k in range(4):
        t = T0 + dt.timedelta(minutes=30 * k)
        lat = 35.0 + 0.18 * k
        out.append({"valid_time": t.strftime("%Y%m%d_%H%M%S UTC"),
                    "storms": [{"id": "77", "lat": lat, "lon": -97.0, "motion_east": 15.0, "motion_south": 0.0,
                                "geometry": _square(lat, -97.0)}]})
    return out


def test_a_v3_storm_is_labelled_by_its_tracked_polygon(ver):
    # a report 40 min later where the storm actually went (~0.24 deg north): a v3 hit
    reports = [{"lat": 35.24, "lon": -97.0, "time": "2025-05-06T18:41:00Z", "mag": 1}]
    tracks = ver.TrackSource(fetch=_fetch_moving_north)
    lab = ver.label_storms(_artifact("tornado_v3-abc123def456"), reports, tracks=tracks)
    assert lab["y_true"].tolist() == [1.0]
    # the same storm under the old definition (40 km, +-4 h) is ALSO positive here -- now a report
    # 30 km away 3 h later: old says yes, v3 says no (not this storm, not within 60 min)
    far_late = [{"lat": 35.27, "lon": -97.0, "time": "2025-05-06T21:00:00Z", "mag": 1}]
    v3 = ver.label_storms(_artifact("tornado_v3-abc123def456"), far_late, tracks=tracks)
    old = ver.label_storms(_artifact("tornado_gbt_v2-48637c637e01"), far_late, tracks=tracks)
    assert v3["y_true"].tolist() == [0.0] and old["y_true"].tolist() == [1.0]


def test_an_unreadable_archive_leaves_a_v3_forecast_unscored(ver):
    def broken(day):
        raise OSError("S3 timed out")
    with pytest.raises(ver.TrackUnavailable):
        ver.label_storms(_artifact("tornado_v3-abc123def456"), [], tracks=ver.TrackSource(fetch=broken))
