"""Old-geometry 18Z HRRR grids for ladder arms A/B, fetched with the PRISTINE code.

Run from hp_base (origin/main aedf08cd1) with PYTHONPATH=src. Each date runs in
its own subprocess (socket hygiene) with a wall-clock timeout; the old code
writes {date}_18z.npz (index-space pooling, stride mode) exactly as the
pre-fix pipeline did, fill-value contamination included.
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_fetch import tornado_dates  # noqa: E402

CODE = (
    "import sys; from hazardpulse.data.hrrr import fetch_hrrr_grid, load_cached_hrrr; "
    "d=sys.argv[1]; c=load_cached_hrrr(d, hour=18); "
    "g = c if c is not None else fetch_hrrr_grid(d, hour=18); "
    "print('RESULT', 'cached' if c is not None else ('ok' if g else 'NONE'))"
)


def run(d: str) -> tuple[str, str]:
    try:
        cp = subprocess.run([sys.executable, "-c", CODE, d], capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return d, "TIMEOUT"
    lines = [l for l in cp.stdout.splitlines() if l.startswith("RESULT")]
    return d, (lines[-1][7:] if lines else f"ERR rc={cp.returncode}")


def main() -> int:
    dates = tornado_dates(Path(".cache/spc/1950-2024_actual_tornadoes.csv"))
    t0 = time.time()
    todo = dates
    for rnd in range(2):
        failed = []
        with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
            futs = [ex.submit(run, d) for d in todo]
            for i, f in enumerate(as_completed(futs), 1):
                d, s = f.result()
                print(f"[r{rnd} {i}/{len(todo)} {time.time() - t0:6.0f}s] {d} {s}", flush=True)
                if s not in ("ok", "cached"):
                    failed.append(d)
        todo = failed
        if not todo:
            break
    print(f"DONE unresolved={len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
