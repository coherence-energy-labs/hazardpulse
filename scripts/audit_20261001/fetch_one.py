"""Fetch ProbSevere + the 3-hourly HRRR analyses for ONE UTC date, in a fresh process.

Dumps every thread's stack to stderr after 400 s so a stall names its own line.
"""
import faulthandler
import sys
import time

faulthandler.dump_traceback_later(400, exit=False)

from hazardpulse.data.hrrr import fetch_hrrr_grid, load_cached_hrrr
from hazardpulse.data.probsevere import fetch_probsevere_day, load_cached_probsevere

HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

d = sys.argv[1]
out = []
t0 = time.time()
cached = load_cached_probsevere(d)
if cached is None:
    ts = fetch_probsevere_day(d)
    n_ps = len(ts)
    out.append(f"ps={n_ps}({time.time() - t0:.0f}s)")
else:
    n_ps = len(cached)
    out.append("ps=cached")
print(f"PHASE ps done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
if n_ps == 0:
    out.append("hrrr=skipped(no-probsevere)")
else:
    got, fetched, missing = 0, 0, []
    t1 = time.time()
    for h in HOURS:
        if load_cached_hrrr(d, hour=h) is not None:
            got += 1
            continue
        g = fetch_hrrr_grid(d, hour=h)
        if g:
            got += 1
            fetched += 1
        else:
            missing.append(f"{h:02d}")
    out.append(f"hrrr={got}/{len(HOURS)} new={fetched}({time.time() - t1:.0f}s)"
               + (f" NONE:{','.join(missing)}" if missing else ""))
faulthandler.cancel_dump_traceback_later()
print("RESULT " + " ".join(out), flush=True)
