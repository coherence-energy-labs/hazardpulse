"""Data pass v2 driver: every UTC day 2020-10-15 .. 2025-12-31, one subprocess per date.

    PYTHONPATH=src python scripts/audit_20261001/data_pass_v2.py [workers] [start] [end]

Resumable (cached products are skipped), retries failures twice, never runs a
date in a long-lived process (a long-lived fsspec/aiohttp process leaked
sockets and stalled in the first pass).
"""
import datetime as dt
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
start = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2020, 10, 15)
end = dt.date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else dt.date(2025, 12, 31)
dates = [(start + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)]


def run(d):
    try:
        cp = subprocess.run([sys.executable, str(HERE / "data_pass_one.py"), d],
                            capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return d, "TIMEOUT"
    lines = [l for l in cp.stdout.splitlines() if l.startswith("RESULT ")]
    if cp.returncode != 0 or not lines:
        tail = (cp.stderr or "").strip().splitlines()[-1:] or ["?"]
        return d, f"ERR rc={cp.returncode} {tail[0][:200]}"
    return d, lines[-1][7:]


t0 = time.time()
todo = dates
for rnd in range(3):
    if not todo:
        break
    print(f"round {rnd}: {len(todo)} dates, {workers} workers", flush=True)
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, d) for d in todo]
        for i, f in enumerate(as_completed(futs), 1):
            d, s = f.result()
            print(f"[r{rnd} {i}/{len(todo)} {time.time() - t0:7.0f}s] {d} {s}", flush=True)
            if s.startswith(("TIMEOUT", "ERR")) or "NONE" in s:
                failed.append(d)
    todo = sorted(failed)
print(f"DONE unresolved={len(todo)} {' '.join(todo)}", flush=True)
