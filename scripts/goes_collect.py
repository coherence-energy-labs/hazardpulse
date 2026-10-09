"""Collect program G1's GOES polar crops for one shard of the task list (made for GitHub runners).

    python scripts/goes_collect.py --shard 3 --of 20 --out goes_out [--max-hours N] [--budget-min 330]

Tasks come from results/goes/tasks.jsonl.gz. A shard holds every task whose hour index mod ``of`` equals ``shard``,
so all storms at one hour share one file opening. Each hour's operational GOES (East or West, by longitude) is read
at the first band-13 full-disk scan at or after the hour, by HTTP byte ranges. Each task gets
``goes_abi.polar_recentred``, re-centred on a clear eye.

Output: ``<out>/goes_shard_<i>_of_<n>.npz`` holding keys, uint8 polar images, centres, the eye flag and the scan read.
A task whose hour has no scan or no readable file is recorded with status "missing", never dropped silently. When
the time budget runs out, the done part is saved and the rest listed as "not_attempted", so a rerun can finish it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import signal
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import goes_abi as g  # noqa: E402

TASKS = ROOT / "results" / "goes" / "tasks.jsonl.gz"
TASK_BUDGET_S = 180        # one crop (network reads included); a read with no deadline hung shard 16 for 5 h 50 min
EARLY_HOURS = 10           # no successful crop in the first this-many hours: abort the shard
MIN_OK_FRACTION = 0.5      # fewer successful crops than this fraction of those attempted: the shard fails


def fetch_text(url: str, tries: int = 4) -> str:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hazardpulse-g1"}),
                                        timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 -- retried, then raised
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


class Deadline(Exception):
    pass


class _deadline:
    """Bound a block's wall time with SIGALRM where the platform has it (the Linux runners); elsewhere a no-op, and
    the per-request HTTP timeout in ``goes_abi`` still bounds every read."""

    def __init__(self, seconds: int):
        self.seconds = int(seconds)

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            def _raise(signum, frame):
                raise Deadline(f"crop exceeded {self.seconds} s")
            self.prev = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.prev)
        return False


def task_key(t: dict) -> str:
    return f"{t['storm']}_{t['hour']}_{t['kind']}{('_' + t['cycle']) if t['cycle'] else ''}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--out", default="goes_out")
    ap.add_argument("--max-hours", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=330.0)
    a = ap.parse_args(argv)
    t_start = time.time()
    with gzip.open(TASKS, "rt", encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh]
    by_hour: dict[str, list[dict]] = {}
    for t in tasks:
        if int(dt.datetime.strptime(t["hour"], "%Y%m%d%H").timestamp() // 3600) % a.of == a.shard:
            by_hour.setdefault(t["hour"], []).append(t)
    hours = sorted(by_hour)
    if a.max_hours:
        hours = hours[:a.max_hours]
    keys, imgs, lats, lons, eyes, scans, status = [], [], [], [], [], [], []
    done_hours = 0
    for hour_s in hours:
        if (time.time() - t_start) / 60.0 > a.budget_min:
            for t in by_hour[hour_s]:
                keys.append(task_key(t)), imgs.append(np.full((g.N_R, g.N_AZ), g.MISSING, np.uint8))
                lats.append(np.nan), lons.append(np.nan), eyes.append(False), scans.append(""), status.append("not_attempted")
            continue
        hour = dt.datetime.strptime(hour_s, "%Y%m%d%H")
        by_sat: dict[str, list[dict]] = {}
        for t in by_hour[hour_s]:
            try:
                by_sat.setdefault(g.satellite_for(hour, t["lon"]), []).append(t)
            except ValueError:
                by_sat.setdefault("", []).append(t)
        def record(t, img=None, la=np.nan, lo=np.nan, eye=False, scan="", st="ok"):
            # one task's fields are appended together, or not at all: run 2 appended a key before its crop raised,
            # and every later image in shard 8 was filed under its neighbour's key
            keys.append(task_key(t))
            imgs.append(img if img is not None else np.full((g.N_R, g.N_AZ), g.MISSING, np.uint8))
            lats.append(la), lons.append(lo), eyes.append(eye), scans.append(scan), status.append(st)

        for sat, group in by_sat.items():
            key, fh5, h5, disk, opened = None, None, None, None, "missing: no scan"
            try:
                if sat:
                    key = g.first_scan_at_or_after(g.list_keys(sat, hour, fetch_text), hour)
                if key is not None:
                    fh5, h5 = g.open_full_disk(sat, key)
                    disk = g.FullDisk(h5)
            except Exception as exc:  # noqa: BLE001 -- the whole group is recorded missing, the run continues
                disk, opened = None, f"missing: {type(exc).__name__}: {str(exc)[:120]}"
            try:
                for t in group:
                    if disk is None:
                        record(t, st=opened)
                        continue
                    try:
                        with _deadline(TASK_BUDGET_S):
                            img, la, lo, eye = g.polar_recentred(disk, t["lat"], t["lon"])
                    except Exception as exc:  # noqa: BLE001 -- this task only
                        record(t, st=f"missing: {type(exc).__name__}: {str(exc)[:120]}")
                        continue
                    record(t, img, la, lo, eye, f"{sat}/{key}", "ok")
            finally:
                if h5 is not None:
                    h5.close()
                if fh5 is not None:
                    fh5.close()
        done_hours += 1
        if done_hours == EARLY_HOURS and not any(s == "ok" for s in status):
            # the first collection "succeeded" with 0 of 43,429 crops: a run that reads nothing must fail, loudly
            print(f"ABORT: no crop succeeded in the first {EARLY_HOURS} hours; first failure: "
                  f"{next((s for s in status if s.startswith('missing: ') and s != 'missing: no scan'), status[:1])}",
                  flush=True)
            return 1
        if done_hours % 50 == 0:
            el = time.time() - t_start
            print(f"{done_hours}/{len(hours)} hours, {len(keys)} tasks, {el / 60:.1f} min "
                  f"({el / done_hours:.1f} s/hour)", flush=True)
    lengths = {len(x) for x in (keys, imgs, lats, lons, eyes, scans, status)}
    if len(lengths) != 1:
        raise SystemExit(f"misaligned record arrays {lengths}: refusing to save images under the wrong keys")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"goes_shard_{a.shard:02d}_of_{a.of:02d}.npz"
    np.savez_compressed(path, keys=np.array(keys), images=np.stack(imgs) if imgs else np.zeros((0, g.N_R, g.N_AZ), np.uint8),
                        lat=np.array(lats), lon=np.array(lons), eye=np.array(eyes), scan=np.array(scans),
                        status=np.array(status))
    n_ok = sum(s == "ok" for s in status)
    attempted = sum(1 for s in status if s not in ("missing: no scan", "not_attempted"))
    print(f"shard {a.shard}/{a.of}: {len(keys)} tasks, {n_ok} ok, {len(keys) - n_ok} not ok "
          f"({attempted} attempted); {(time.time() - t_start) / 60:.1f} min; wrote {path}", flush=True)
    if attempted and n_ok < MIN_OK_FRACTION * attempted:
        print(f"FAIL: only {n_ok} of {attempted} attempted crops succeeded", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
