"""Storm-centred infrared crops from NOAA GMGSI (docs/MODEL_IMPROVEMENT_LEDGER.md, H1) -- data only.

    PYTHONPATH=src python scripts/hurricane_ir_crops.py [--workers 4] [--limit-hours N]

For every RI development cycle from GMGSI's first archived month (2021-07) and every 2026 final
cycle, two images are cropped around the storm: hour t + 2 h (the newest image certain to exist when
the live forecast runs at t + 3 h 30; GMGSI lands ~35-40 min after its hour) and hour t - 4 h (six
hours earlier, for trends). The centre is the CARQ position at t extrapolated along the t - 6 h -> t
motion. Each image is downloaded once for every storm active at that hour, cropped to +-4 degrees,
and discarded: only the crops (~12 kB each) are kept, under the shared cache. No outcome is read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import io
import json
import math
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import atcf  # noqa: E402

BUCKET = "https://noaa-gmgsi-pds.s3.amazonaws.com"
CROPS = v9.CACHE / "hurricane_ir" / "crops"
FIRST_HOUR = dt.datetime(2021, 7, 12)          # the archive's first day (listed 2026-10-03)
HALF_DEG = 4.0
OFFSETS = {"p2": 2, "m4": -4}          # image hour relative to the cycle t


def carq_positions(records, cyc: dt.datetime) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """CARQ tau-0 (lat, lon) at t and at t - 6 h from the storm's a-deck."""
    at = {}
    for r in records:
        if r.model == "CARQ" and r.tau_hours == 0 and r.lat is not None and r.lon is not None:
            at.setdefault(r.cycle, (float(r.lat), float(r.lon)))
    return at.get(cyc), at.get(cyc - dt.timedelta(hours=6))


def extrapolate(p0, pm6, hours: float) -> tuple[float, float]:
    """Linear in lat/lon along the last 6 h of motion (no motion known -> stay)."""
    if pm6 is None:
        return p0
    dlon = ((p0[1] - pm6[1] + 180.0) % 360.0) - 180.0
    lon = ((p0[1] + dlon * hours / 6.0 + 180.0) % 360.0) - 180.0
    return p0[0] + (p0[0] - pm6[0]) * hours / 6.0, lon


def cases() -> list[dict]:
    """(atcf id, dtg, centre per image tag) for every development cycle from 2021-07 and every 2026 cycle."""
    dev = [json.loads(l) for l in (v9.WORK / "dev_cases_v10.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    y26 = [json.loads(l) for l in (v9.Y26 / "final_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, decks = [], {}
    for c, path_of in [(c, v9.adeck_path) for c in dev] + [(c, lambda a: v9.Y26 / "adeck" / f"a{a[:2].lower()}{a[2:]}.dat.gz") for c in y26]:
        cyc = dt.datetime.strptime(c["dtg"], "%Y%m%d%H")
        if cyc + dt.timedelta(hours=OFFSETS["m4"]) < FIRST_HOUR:
            continue
        aid = c["atcf_id"]
        if aid not in decks:
            p = path_of(aid)
            decks[aid] = atcf.parse_atcf_deck(gzip.decompress(p.read_bytes()).decode("utf-8", "replace")) if p.exists() else []
        p0, pm6 = carq_positions(decks[aid], cyc)
        if p0 is None:
            continue
        out.append({"atcf_id": aid, "dtg": c["dtg"],
                    "centres": {tag: extrapolate(p0, pm6, h) for tag, h in OFFSETS.items()},
                    "hours": {tag: cyc + dt.timedelta(hours=h) for tag, h in OFFSETS.items()}})
    return out


def crop_path(aid: str, dtg: str, tag: str) -> Path:
    return CROPS / dtg[:4] / f"{aid}_{dtg}_{tag}.npz"


def image_key(hour: dt.datetime) -> str | None:
    prefix = f"GMGSI_LW/{hour:%Y/%m/%d/%H}/"
    with urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=60) as r:
        keys = re.findall(r"<Key>([^<]+)</Key>", r.read().decode("utf-8"))
    return sorted(keys)[0] if keys else None


def fetch_image(hour: dt.datetime):
    """``(key, uint8 counts (rows, cols), lat per row, lon per column)`` or ``(None, ...)``."""
    import h5py
    key = image_key(hour)
    if key is None:
        return None, None, None, None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r:
                blob = r.read()
            break
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(5)
    with h5py.File(io.BytesIO(blob), "r") as f:
        d = f["data"][0]
        lat, lon = f["lat"][:, 0].astype(np.float64), f["lon"][0, :].astype(np.float64)
    counts = np.where(d < 0, 255, np.clip(d, 0, 254)).astype(np.uint8)      # 255 = missing
    counts[d >= 255] = 254
    return key, counts, lat, lon


def crop(counts, lat, lon, centre) -> dict:
    """+-HALF_DEG around ``centre``; columns wrap at the dateline."""
    rows = np.nonzero(np.abs(lat - centre[0]) <= HALF_DEG)[0]
    dlon = ((lon - centre[1] + 180.0) % 360.0) - 180.0
    cols = np.nonzero(np.abs(dlon) <= HALF_DEG)[0]
    cols = cols[np.argsort(dlon[cols])]
    return {"counts": counts[np.ix_(rows, cols)], "lat": lat[rows].astype(np.float32),
            "lon": (centre[1] + dlon[cols]).astype(np.float32), "centre": np.array(centre, np.float64)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
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
        futs = {ex.submit(fetch_image, h): h for h in hours}
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
                np.savez_compressed(p, **crop(counts, lat, lon, centre), key=np.array(key), hour=np.array(f"{h:%Y%m%d%H}"))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(hours)} hours, {time.time() - t0:.0f} s", flush=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    (CROPS.parent / "missing_hours.json").write_text(json.dumps(sorted(missing), indent=1), encoding="utf-8")
    print(f"done: {done} hours cropped, {len(missing)} missing ({missing[:5]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
