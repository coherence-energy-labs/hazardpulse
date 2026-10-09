"""The G1 collector: a crop that fails mid-hour is recorded missing under ITS key, every other image keeps its own
key, a run that reads nothing fails, and a misaligned record is refused (run 2's shard 8 filed 2,187 images
under their neighbours' keys after one crop raised)."""
from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import json
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
        return np.full((g.N_R, g.N_AZ), int(lat), np.uint8), lat, lon, False   # the image encodes its storm
    monkeypatch.setattr(g, "list_keys", lambda sat, hour, fetch: [f"k_s{hour:%Y%j%H%M%S}0"])
    monkeypatch.setattr(g, "first_scan_at_or_after", lambda keys, hour: keys[0])
    monkeypatch.setattr(g, "open_full_disk", lambda sat, key: (None, None))
    monkeypatch.setattr(g, "FullDisk", lambda h5: object())
    monkeypatch.setattr(g, "polar_recentred", polar)
    return calls


def _run(gc, tmp_path, tasks, hours):
    gc.TASKS = tasks
    shard = int(hours[0].timestamp() // 3600) % 20
    rc = gc.main(["--shard", str(shard), "--of", "20", "--out", str(tmp_path / "out")])
    return rc, np.load(tmp_path / "out" / f"goes_shard_{shard:02d}_of_20.npz")


def test_a_crop_that_fails_mid_hour_is_missing_under_its_own_key_and_the_rest_keep_theirs(tmp_path, monkeypatch):
    gc = _collector()
    tasks, hours = _tasks(tmp_path)
    _fake_io(monkeypatch, fail_on=21.0)                        # the second storm's crop raises
    rc, z = _run(gc, tmp_path, tasks, hours)
    assert rc == 0
    assert len({len(z[k]) for k in ("keys", "images", "status", "lat", "eye", "scan")}) == 1
    by_key = {str(k): (int(img[0, 0]), str(st)) for k, img, st in zip(z["keys"], z["images"], z["status"])}
    assert by_key["AL012024_2024090100_interp"] == (20, "ok")
    assert by_key["AL022024_2024090100_interp"][1].startswith("missing: RuntimeError")
    assert by_key["AL032024_2024090100_interp"] == (22, "ok")      # its own image, not its neighbour's


def test_a_run_that_reads_nothing_fails(tmp_path, monkeypatch):
    gc = _collector()
    tasks, hours = _tasks(tmp_path, n_hours=1)
    _fake_io(monkeypatch, fail_on=None)
    monkeypatch.setattr(g, "polar_recentred", lambda disk, lat, lon: (_ for _ in ()).throw(TypeError("0-d")))
    rc, z = _run(gc, tmp_path, tasks, hours)
    assert rc == 1 and not any(str(s) == "ok" for s in z["status"])
