"""NWS tornado-warning state for every row of the v3 feature store (block W, and the NWS bar).

    PYTHONPATH=src python scripts/audit_20261001/warning_state_on_store.py 2020-10-15 2025-12-31

Per storm observation: active_now (inside a tornado warning polygon in force at the
observation time; only warnings issued at or before it) and minutes_since_issue (NaN when not
warned). From the IEM storm-based warning archive via hazardpulse.verification.nws_warnings.
Writes STORE/_nws/<day>.npz; a day is redone when its store file is newer.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

from hazardpulse.verification import nws_warnings as nw

ROOT = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", str(ROOT / ".cache" / "feature_store_v3")))
OUT = STORE / "_nws"


def main() -> int:
    start, end = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
    warnings = nw.load_tor_warnings(list(range(start.year, end.year + 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    d = start
    while d <= end:
        ds = d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)
        src = STORE / f"{ds}.npz"
        out = OUT / f"{ds}.npz"
        if not src.exists():
            continue
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            continue
        with np.load(src) as z:
            lat, lon, t = z["lat"].astype(np.float64), z["lon"].astype(np.float64), z["t"].astype(np.float64)
        active, _, since = nw.tor_warning_state(lat, lon, t, warnings=warnings)
        tmp = OUT / f"{ds}.tmp.npz"
        np.savez(tmp, active_now=np.asarray(active, np.int8),
                 minutes_since_issue=np.where(np.asarray(active, bool), np.asarray(since, np.float32), np.nan).astype(np.float32))
        os.replace(tmp, out)
        print(ds, len(lat), int(np.asarray(active).sum()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
