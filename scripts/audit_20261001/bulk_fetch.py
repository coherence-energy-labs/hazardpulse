"""Bulk, resumable re-fetch of ProbSevere + HRRR-18z for every 2021-2024 tornado date.

The date set is the UNION of SPC tornado dates read as CST (the old, buggy keying)
and the same reports converted to UTC (+6 h), so the BEFORE and AFTER label runs
see the identical storm population. Run from the worktree with PYTHONPATH=src.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def tornado_dates(csv_path: Path) -> list[str]:
    dates: set[str] = set()
    with open(csv_path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if not (2021 <= int(r["yr"]) <= 2024):
                continue
            h, m = (int(x) for x in r["time"].split(":")[:2])
            local = dt.datetime(int(r["yr"]), int(r["mo"]), int(r["dy"]), h, m)
            dates.add(local.strftime("%Y%m%d"))
            dates.add((local + dt.timedelta(hours=6)).strftime("%Y%m%d"))
    return sorted(d for d in dates if "20210101" <= d <= "20241231")


def work(date_str: str) -> tuple[str, str]:
    from hazardpulse.data.hrrr import fetch_hrrr_grid, load_cached_hrrr
    from hazardpulse.data.probsevere import fetch_probsevere_day, load_cached_probsevere

    status = []
    try:
        if load_cached_probsevere(date_str) is None:
            ts = fetch_probsevere_day(date_str)
            status.append(f"ps={len(ts)}")
        else:
            status.append("ps=cached")
    except Exception as exc:  # report, never hide
        status.append(f"ps=ERR:{type(exc).__name__}:{exc}")
    try:
        if load_cached_hrrr(date_str, hour=18) is None:
            g = fetch_hrrr_grid(date_str, hour=18)
            status.append(f"hrrr={'ok' if g else 'NONE'}")
        else:
            status.append("hrrr=cached")
    except Exception as exc:
        status.append(f"hrrr=ERR:{type(exc).__name__}:{exc}")
    return date_str, " ".join(status)


def main() -> int:
    dates = tornado_dates(Path(".cache/spc/1950-2024_actual_tornadoes.csv"))
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"{len(dates)} dates, {n_workers} workers", flush=True)
    t0 = time.time()
    with Pool(n_workers) as pool:
        for i, (d, s) in enumerate(pool.imap_unordered(work, dates), 1):
            el = time.time() - t0
            print(f"[{i}/{len(dates)} {el:7.0f}s eta {el / i * (len(dates) - i):6.0f}s] {d} {s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
