"""Storm-centred infrared crops from NOAA GMGSI (docs/MODEL_IMPROVEMENT_LEDGER.md, H1) -- data only.

    PYTHONPATH=src python scripts/hurricane_ir_crops.py [--workers 12] [--limit-hours N]

For every RI development cycle from GMGSI's first archived day (2021-07-12) and every 2026 final
cycle, two images are cropped around the storm: hour t + 2 h (the newest image certain to exist when
the live forecast runs at t + 3 h 30; GMGSI lands ~35-40 min after its hour) and hour t - 4 h (six
hours earlier, for trends). The centre and the crop are made by ``hazardpulse.hurricane.ir_source``,
the code the live scorer also uses. Each image is downloaded once for every storm active at that
hour and discarded: only the crops (~12 kB each) are kept, under the shared cache. No outcome is
read. Parallel downloads matter: one connection to the bucket gives ~0.9 MB/s, eight give ~10.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import atcf, ir_source  # noqa: E402

CROPS = v9.CACHE / "hurricane_ir" / "crops"
FIRST_HOUR = dt.datetime(2021, 7, 12)          # the archive's first day (listed 2026-10-03)


def cases() -> list[dict]:
    """(atcf id, dtg, image hour and centre per tag) for every development cycle from 2021-07-12
    and every 2026 cycle."""
    dev = [json.loads(l) for l in (v9.WORK / "dev_cases_v10.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    y26 = [json.loads(l) for l in (v9.Y26 / "final_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, decks = [], {}
    for c, path_of in [(c, v9.adeck_path) for c in dev] + [(c, lambda a: v9.Y26 / "adeck" / f"a{a[:2].lower()}{a[2:]}.dat.gz") for c in y26]:
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        if cyc + dt.timedelta(hours=ir_source.OFFSETS["m4"]) < FIRST_HOUR:
            continue
        aid = c["atcf_id"]
        if aid not in decks:
            p = path_of(aid)
            decks[aid] = atcf.parse_atcf_deck(gzip.decompress(p.read_bytes()).decode("utf-8", "replace")) if p.exists() else []
        cen = ir_source.centres(decks[aid], cyc)
        if cen is None:
            continue
        out.append({"atcf_id": aid, "dtg": c["dtg"], "centres": {t: v[1] for t, v in cen.items()},
                    "hours": {t: v[0] for t, v in cen.items()}})
    return out


def crop_path(aid: str, dtg: str, tag: str) -> Path:
    return CROPS / dtg[:4] / f"{aid}_{dtg}_{tag}.npz"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-hours", type=int, default=None)
    args = ap.parse_args(argv)
    todo: dict[dt.datetime, list[tuple[str, str, str, tuple]]] = defaultdict(list)
    cs = cases()
    for c in cs:
        for tag, hour in c["hours"].items():
            if not crop_path(c["atcf_id"], c["dtg"], tag).exists():
                todo[hour].append((c["atcf_id"], c["dtg"], tag, c["centres"][tag]))
    hours = sorted(todo)[: args.limit_hours]
    print(f"{len(cs)} cycles; {sum(len(v) for v in todo.values())} crops to make from {len(todo)} image hours"
          f" (doing {len(hours)})", flush=True)
    missing, done, t0 = [], 0, time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ir_source.fetch_image, h): h for h in hours}
        for fut in cf.as_completed(futs):
            h = futs[fut]
            try:
                key, counts, lat, lon = fut.result()
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{h:%Y%m%d%H}: {type(exc).__name__}")
                continue
            if key is None:
                missing.append(f"{h:%Y%m%d%H}: no image")
                continue
            for aid, dtg, tag, centre in todo[h]:
                p = crop_path(aid, dtg, tag)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + ".part.npz")             # written whole, then renamed:
                np.savez_compressed(tmp, **ir_source.crop(counts, lat, lon, centre), key=np.array(key),
                                    hour=np.array(f"{h:%Y%m%d%H}"))
                tmp.replace(p)                                        # a killed run leaves no half crop
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(hours)} hours, {time.time() - t0:.0f} s", flush=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    (CROPS.parent / "missing_hours.json").write_text(json.dumps(sorted(missing), indent=1), encoding="utf-8")
    print(f"done: {done} hours cropped, {len(missing)} missing ({missing[:5]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
