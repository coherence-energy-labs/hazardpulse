"""Amendment 13 (J1) machinery: the stored-statics arithmetic IS ir_features.features, the collection gate refuses
duplicates and gaps, a collection that reads nothing fails, and the paired interval behaves."""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.hurricane import ir_features as irf
from hazardpulse.hurricane import ir_source as irs

ROOT = Path(__file__).resolve().parents[1]


def _j1():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("hurricane_ri_j1_t", ROOT / "scripts" / "hurricane_ri_j1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _crop(centre, cold_core=True, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.arange(centre[0] - 4, centre[0] + 4.01, 0.072)
    lon = np.arange(centre[1] - 4, centre[1] + 4.01, 0.072)
    d, _ = irf.distances_km(lat, lon, centre)
    counts = np.clip(150 + (60 if cold_core else 10) * np.exp(-(d / 120.0) ** 2) + rng.normal(0, 3, d.shape), 0, 254)
    counts = counts.astype(np.uint8)
    counts[0, :5] = irf.MISSING
    return {"counts": counts, "lat": lat, "lon": lon, "centre": np.array(centre)}


def test_the_stored_statics_arithmetic_is_ir_features_features():
    j1 = _j1()
    now, bef = _crop((15.0, 140.0), True, 1), _crop((14.8, 140.3), False, 2)
    want = irf.features(now, bef)
    s_now = irf.static_features(now["counts"], now["lat"], now["lon"], tuple(now["centre"]))
    s_bef = irf.static_features(bef["counts"], bef["lat"], bef["lon"], tuple(bef["centre"]))
    stats = {"S|2024-09-01 00:00:00|p2": [s_now[n] for n in irf.STATIC],
             "S|2024-09-01 00:00:00|m4": [s_bef[n] for n in irf.STATIC]}
    got = j1.ir_row_features(stats, "S|2024-09-01 00:00:00")
    assert set(got) == set(irf.IR_NAMES)
    for n in irf.IR_NAMES:
        assert got[n] == pytest.approx(want[n], nan_ok=True, abs=0, rel=0) or (math.isnan(got[n]) and math.isnan(want[n]))
    # a missing image gives NaN, never a number
    miss = j1.ir_row_features({}, "S|2024-09-01 00:00:00")
    assert all(math.isnan(v) for v in miss.values())


def _tasks(tmp_path, j1, n_hours=3):
    tasks = []
    for h in range(n_hours):
        for s in ("A", "B"):
            tasks.append({"row": f"{s}|2024-09-01 {h:02d}:00:00", "tag": "p2", "hour": f"20240901{h + 2:02d}",
                          "lat": 15.0, "lon": 140.0})
    p = tmp_path / "ir_tasks.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    j1.TASKS = p
    return tasks


def test_collect_records_every_task_and_the_gate_refuses_duplicates_and_gaps(tmp_path, monkeypatch):
    j1 = _j1()
    tasks = _tasks(tmp_path, j1)
    img = _crop((15.0, 140.0))
    monkeypatch.setattr(irs, "fetch_image", lambda hour: ("k", img["counts"], img["lat"], img["lon"]))
    out = tmp_path / "out"
    for shard in range(2):
        assert j1.collect(shard, 2, str(out)) == 0
    got, counts = j1.load_collection(str(out))
    assert len(got) == len(tasks) and counts == {"ok": len(tasks)}
    z = dict(np.load(next(out.glob("j1_shard_00_*.npz"))))
    dup = tmp_path / "dup"
    dup.mkdir()
    np.savez(dup / "j1_shard_00_of_02.npz", **z)
    np.savez(dup / "j1_shard_01_of_02.npz", **z)
    with pytest.raises(SystemExit, match="collected twice"):
        j1.load_collection(str(dup))
    gap = tmp_path / "gap"
    gap.mkdir()
    np.savez(gap / "j1_shard_00_of_02.npz", **z)
    with pytest.raises(SystemExit, match="missing"):
        j1.load_collection(str(gap))


def test_a_collection_that_reads_nothing_fails(tmp_path, monkeypatch):
    j1 = _j1()
    _tasks(tmp_path, j1, n_hours=12)
    monkeypatch.setattr(irs, "fetch_image", lambda hour: (_ for _ in ()).throw(TypeError("0-d")))
    assert j1.collect(0, 1, str(tmp_path / "out")) == 1


def test_the_paired_interval_covers_zero_for_identical_arms_and_excludes_it_for_a_better_one():
    j1 = _j1()
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(float)
    storms = np.repeat(np.arange(300), 10)
    pa = np.full(3000, 0.1)
    same = j1.paired(y, pa, pa.copy(), storms, n_boot=300)
    assert same["d_ll"] == 0 and same["d_ll_ci"][0] <= 0 <= same["d_ll_ci"][1]
    better = np.where(y == 1, 0.5, 0.05)
    res = j1.paired(y, pa, better, storms, n_boot=300)
    assert res["d_ll"] < 0 and res["d_ll_ci"][1] < 0 and res["d_brier_ci"][1] < 0


def test_dedupe_removes_identical_basin_file_copies_and_refuses_conflicting_ones():
    """IBTrACS lists a basin-crossing storm in every basin file it enters; the v8.2 builder concatenates the files
    (5,950 identical copies in the frozen 2000-2024 file). Identical copies go; two different rows for one key stop."""
    j1 = _j1()
    a = {"storm_id": "S1", "issue_time": "2024-09-01 00:00:00", "basin": "WP", "ri_label_30kt": 1, "v": 1.0}
    b = {"storm_id": "S1", "issue_time": "2024-09-01 06:00:00", "basin": "WP", "ri_label_30kt": 0, "v": 2.0}
    rows, removed = j1.dedupe([a, b, dict(a), dict(b)])
    assert removed == 2 and rows == [a, b]
    with pytest.raises(SystemExit, match="two different rows"):
        j1.dedupe([a, dict(a, v=9.0)])
