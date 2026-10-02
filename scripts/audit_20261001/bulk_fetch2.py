"""Bulk re-fetch driver: one fresh subprocess per date, hard wall-clock timeout, retries.

The first driver (bulk_fetch.py, a multiprocessing Pool) stalled after 58 dates:
fsspec/aiohttp opens a client session per HRRR fetch and never closes it, and
each long-lived worker accumulated 84-222 CLOSE_WAIT sockets until every read
blocked. A process per date cannot accumulate anything.
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_fetch import tornado_dates  # noqa: E402

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 900


def run_one(d: str) -> tuple[str, str]:
    try:
        cp = subprocess.run(
            [sys.executable, str(HERE / "fetch_one.py"), d],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        (HERE / "fetch_stalls").mkdir(exist_ok=True)
        (HERE / "fetch_stalls" / f"{d}.txt").write_text(err, encoding="utf-8")
        return d, "TIMEOUT"
    lines = [l for l in cp.stdout.splitlines() if l.startswith("RESULT ")]
    if cp.returncode != 0 or not lines:
        tail = (cp.stderr or "").strip().splitlines()[-1:] or ["?"]
        return d, f"ERR rc={cp.returncode} {tail[0][:160]}"
    return d, lines[-1][7:]


def main() -> int:
    dates = tornado_dates(Path(".cache/spc/1950-2024_actual_tornadoes.csv"))
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if len(sys.argv) > 2:  # optional upper bound on dates (completion passes)
        dates = [d for d in dates if d <= sys.argv[2]]
    todo = list(dates)
    t0 = time.time()
    for rnd in range(3):
        if not todo:
            break
        print(f"round {rnd}: {len(todo)} dates, {n_workers} workers", flush=True)
        failed: list[str] = []
        with ThreadPoolExecutor(n_workers) as ex:
            futs = {ex.submit(run_one, d): d for d in todo}
            for i, f in enumerate(as_completed(futs), 1):
                d, s = f.result()
                el = time.time() - t0
                print(f"[r{rnd} {i}/{len(todo)} {el:7.0f}s] {d} {s}", flush=True)
                if s.startswith(("TIMEOUT", "ERR")) or "NONE" in s:
                    failed.append(d)
        todo = sorted(failed)
    print(f"DONE unresolved={len(todo)} {' '.join(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
