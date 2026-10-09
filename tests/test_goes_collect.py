"""The G1 collector: a crop that fails mid-hour is recorded missing under ITS key, every other image keeps its own
key, a run that reads nothing fails, and a misaligned record is refused (run 2's shard 8 filed 2,187 images
under their neighbours' keys after one crop raised)."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from hazardpulse.hurricane import goes_abi as g

ROOT = Path(__file__).resolve().parents[1]


def _collector():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("goes_collect_t", ROOT / "scripts" / "goes_collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tasks(tmp_path, n_hours=1, storms=("AL01", "AL02", "AL03")):
    p = tmp_path / "tasks.jsonl.gz"
    hours = [dt.datetime(2024, 9, 1, 0) + dt.timedelta(hours=h * 20) for h in range(n_hours)]   # 20 h apart: one shard
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for h in hours:
            for i, s in enumerate(storms):
                fh.write(json.dumps({"storm": f"{s}2024", "hour": h.strftime("%Y%m%d%H"), "kind": "interp",
                                     "cycle": "", "lat": 20.0 + i, "lon": -60.0}) + "\n")
    return p, hours


def _fake_io(monkeypatch, fail_on=None):
    calls = []

    def polar(disk, lat, lon):
        calls.append(lat)
        if fail_on is not None and abs(lat - fail_on) < 1e-9:
            raise RuntimeError("read failed")
        img = np.full((g.N_R, g.N_AZ), int(lat), np.uint8)                     # the image encodes its storm
        return g.EyeCrop(img + 100, img, lat, lon, False, 200.0 + lat, 190.0 + lat)
    monkeypatch.setattr(g, "list_keys", lambda sat, hour, fetch: [f"k_s{hour:%Y%j%H%M%S}0"])
    monkeypatch.setattr(g, "first_scan_at_or_after", lambda keys, hour: keys[0])
    monkeypatch.setattr(g, "open_full_disk", lambda sat, key: (None, None))
    monkeypatch.setattr(g, "FullDisk", lambda h5: object())
    monkeypatch.setattr(g, "polar_eye", polar)
    return calls


def _run(gc, tmp_path, tasks, hours):
    gc.TASKS = tasks
    shard = gc.shard_of(hours[0].strftime("%Y%m%d%H"), 20)
    rc = gc.main(["--shard", str(shard), "--of", "20", "--out", str(tmp_path / "out")])
    return rc, np.load(tmp_path / "out" / f"goes_shard_{shard:02d}_of_20.npz")


def test_a_crop_that_fails_mid_hour_is_missing_under_its_own_key_and_the_rest_keep_theirs(tmp_path, monkeypatch):
    gc = _collector()
    tasks, hours = _tasks(tmp_path)
    _fake_io(monkeypatch, fail_on=21.0)                        # the second storm's crop raises
    rc, z = _run(gc, tmp_path, tasks, hours)
    assert rc == 0
    assert len({len(z[k]) for k in ("keys", "images", "analysed", "status", "lat", "eye", "teye", "tcw", "scan")}) == 1
    by_key = {str(k): (int(img[0, 0]), int(a[0, 0]), float(te), str(st))
              for k, img, a, te, st in zip(z["keys"], z["images"], z["analysed"], z["teye"], z["status"])}
    assert by_key["AL012024_2024090100_interp"] == (20, 120, 220.0, "ok")
    assert by_key["AL022024_2024090100_interp"][3].startswith("missing: RuntimeError")
    assert by_key["AL022024_2024090100_interp"][1] == g.MISSING and math.isnan(by_key["AL022024_2024090100_interp"][2])
    assert by_key["AL032024_2024090100_interp"] == (22, 122, 222.0, "ok")      # its own images, not its neighbour's


def test_a_run_that_reads_nothing_fails(tmp_path, monkeypatch):
    gc = _collector()
    tasks, hours = _tasks(tmp_path, n_hours=1)
    _fake_io(monkeypatch, fail_on=None)
    monkeypatch.setattr(g, "polar_eye", lambda disk, lat, lon: (_ for _ in ()).throw(TypeError("0-d")))
    rc, z = _run(gc, tmp_path, tasks, hours)
    assert rc == 1 and not any(str(s) == "ok" for s in z["status"])


def test_shards_are_fixed_by_utc_not_by_the_machines_time_zone():
    """Run 2's shards 8 and 16, re-run on this machine (EDT), collected other shards' hours. Fails on the naive
    .timestamp() wherever the local zone is not UTC (here; and on Linux, where TZ is switched below)."""
    import os
    import time as _time
    gc = _collector()
    want = int(dt.datetime(2024, 9, 1, 5, tzinfo=dt.timezone.utc).timestamp() // 3600) % 20
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    if hasattr(_time, "tzset"):
        _time.tzset()
    try:
        assert gc.shard_of("2024090105", 20) == want
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        if hasattr(_time, "tzset"):
            _time.tzset()


def _g1():
    sys.path.insert(0, str(ROOT / "scripts"))
    import hurricane_ri_g1
    return hurricane_ri_g1


def test_the_stats_phase_refuses_old_detector_shards_duplicates_and_gaps(tmp_path, monkeypatch):
    """Amendment 12b: stats are computed only from shards carrying the closed-ring detector's ADT temperatures, and
    only when every task key is present exactly once."""
    import pytest
    gc = _collector()
    g1 = _g1()
    tasks, hours = _tasks(tmp_path)
    _fake_io(monkeypatch)
    rc, z = _run(gc, tmp_path, tasks, hours)
    monkeypatch.setattr(g1, "STATS", tmp_path / "stats.json")
    assert g1.stats(str(tmp_path / "out"), tasks) == 0                       # complete and current: accepted
    assert gc.main(["--verify", str(tmp_path / "out")]) == 0                 # the runners' verify job, same gate
    doc = json.loads((tmp_path / "stats.json").read_text())
    assert doc["ok"] == 3 and doc["stats"]["AL012024_2024090100_interp"]["eye_contrast"] == 10.0
    old = tmp_path / "old"
    old.mkdir()
    np.savez(old / "goes_shard_00_of_20.npz", **{k: z[k] for k in ("keys", "images", "lat", "lon", "eye", "scan", "status")})
    with pytest.raises(SystemExit, match="before amendment 12b"):
        g1.stats(str(old), tasks)
    dup = tmp_path / "dup"
    dup.mkdir()
    for i in (0, 1):
        np.savez(dup / f"goes_shard_0{i}_of_20.npz", **{k: z[k] for k in z.files})
    with pytest.raises(SystemExit, match="collected twice"):
        g1.stats(str(dup), tasks)
    gap = tmp_path / "gap"
    gap.mkdir()
    np.savez(gap / "goes_shard_00_of_20.npz", **{k: z[k][:2] for k in z.files})
    with pytest.raises(SystemExit, match="1 missing"):
        g1.stats(str(gap), tasks)
    with pytest.raises(SystemExit, match="1 missing"):
        gc.main(["--verify", str(gap)])
    late = tmp_path / "late"                                                   # a shard that ran out of time
    late.mkdir()
    st = z["status"].astype("<U40")                                          # "ok"-only arrays are <U2
    st[1] = "not_attempted"
    np.savez(late / "goes_shard_00_of_20.npz", **{k: (st if k == "status" else z[k]) for k in z.files})
    with pytest.raises(SystemExit, match="1 tasks were not attempted"):
        gc.main(["--verify", str(late)])
