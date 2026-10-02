"""Data pass v2 for ONE UTC date, in a fresh process (see data_pass_v2.py).

ProbSevere with the fixed parser (NOAA ProbTor/Hail/Wind + every numeric
attribute) into .cache/probsevere_v2, then each 3-hourly HRRR analysis read
ONCE natively and written twice: the 80 km g3 lat/lon grid (.cache/hrrr) and
the 9 km native-block grid (.cache/hrrr_lcc). Prints one RESULT line.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.dump_traceback_later(600, exit=False)

import numpy as np  # noqa: E402

from hazardpulse.data import hrrr as H  # noqa: E402
from hazardpulse.data.probsevere import fetch_probsevere_day, load_cached_probsevere  # noqa: E402

HOURS = (0, 3, 6, 9, 12, 15, 18, 21)
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", str(H.PROJECT_ROOT / ".cache" / "probsevere_v2")))

d = sys.argv[1]
t0 = time.time()
out = []
ps = load_cached_probsevere(d, cache_dir=PS_V2)
if ps is None:
    ps = fetch_probsevere_day(d, cache_dir=PS_V2)
    out.append(f"ps={len(ps)}/{sum(len(s['storms']) for s in ps)}({time.time() - t0:.0f}s)")
else:
    out.append("ps=cached")
if not ps:
    out.append("hrrr=skipped(no-probsevere)")
else:
    t1, new, have, missing = time.time(), 0, 0, []
    for h in HOURS:
        need80 = H.load_cached_hrrr(d, hour=h) is None
        need9 = not H._lcc_path(d, h, 3).exists()
        if not (need80 or need9):
            have += 1
            continue
        nat = H.fetch_hrrr_natives(d, h)
        if nat is None:
            missing.append(f"{h:02d}")
            continue
        if need80:
            grids = {k: H._subsample_to_grid(v.ravel(), k) for k, v in nat.items()}
            p = H._npz_path(d, h)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), **grids)
            os.replace(tmp, p)
        if need9:
            H.save_lcc_blocks(nat, d, h, 3)
        new += 1
    out.append(f"hrrr={have + new}/{len(HOURS)} new={new}({time.time() - t1:.0f}s)"
               + (f" NONE:{','.join(missing)}" if missing else ""))
faulthandler.cancel_dump_traceback_later()
print("RESULT " + " ".join(out), flush=True)
