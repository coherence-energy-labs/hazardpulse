#!/usr/bin/env python3
"""Program step 2a: Block S (definitive_model.compute_block_s, 61 features) for every
active cell at every issue time, as the live scorer would compute it: on the M2.5+
catalog of the 1,827 days strictly before the issue time (FEATURE_HISTORY_DAYS), at the
cell centre. Cached for candidate C1.

    python scripts/earthquake_program/blocks_c1.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

HISTORY_DAYS = 1827
OUT = C.PROGRAM_CACHE / "block_s_active.npz"

_RAW = None


class _Cat:
    pass


def _init():
    global _RAW
    _RAW = C.load_raw("program_catalog_m25.npz")


def _one(k_t):
    from hazardpulse.earthquake.definitive_model import compute_block_s
    k, t = k_t
    r = _RAW
    lo = int(np.searchsorted(r["t"], t - HISTORY_DAYS * C.SEC_DAY, side="left"))
    hi = int(np.searchsorted(r["t"], t, side="left"))
    cat = _Cat()
    cat.times = r["t"][lo:hi]
    cat.lats = r["lat"][lo:hi]
    cat.lons = r["lon"][lo:hi]
    cat.mags = r["mag"][lo:hi]
    d = r["depth"][lo:hi]
    cat.depths = np.where(np.isfinite(d), d, 10.0)
    m6 = cat.mags >= 6.0
    cat.m6_times, cat.m6_lats, cat.m6_lons = cat.times[m6], cat.lats[m6], cat.lons[m6]
    cells = np.nonzero(np.bincount(of.cell_index(cat.lats[cat.times >= t - 30 * C.SEC_DAY],
                                                 cat.lons[cat.times >= t - 30 * C.SEC_DAY]),
                                   minlength=of.N_CELLS) >= 5)[0]
    clat, clon = of.cell_centres()
    out_cells, out_f = [], []
    for c in cells:
        b = compute_block_s(float(clat[c]), float(clon[c]), float(t), cat)
        out_cells.append(c)
        out_f.append(np.full(61, np.nan, np.float32) if b is None else np.asarray(b, np.float32))
    return k, np.array(out_cells, np.int32), (np.stack(out_f) if out_f else np.zeros((0, 61), np.float32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    issue = C.all_issue_times()
    t0 = time.time()
    ks, cs, fs = [], [], []
    with Pool(args.workers, initializer=_init) as pool:
        for i, (k, cells, feats) in enumerate(pool.imap_unordered(_one, list(enumerate(issue)), chunksize=4)):
            ks.append(np.full(cells.size, k, np.int32))
            cs.append(cells)
            fs.append(feats)
            if i % 100 == 0:
                print(f"  {i}/{issue.size} issue times, {time.time() - t0:.0f}s", flush=True)
    k = np.concatenate(ks)
    c = np.concatenate(cs)
    f = np.concatenate(fs)
    order = np.lexsort((c, k))
    np.savez_compressed(OUT, issue_idx=k[order], cell=c[order], feats=f[order], issue_times=issue)
    print(f"wrote {OUT}: {k.size} active cell-times, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
