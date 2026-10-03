"""Where the IR crops come from (hurricane RI amendment 5): NOAA GMGSI longwave on AWS.

Shared by the training collector (``scripts/hurricane_ir_crops.py``) and the live scorer, so a
crop is made the same way in both: the image at a given hour, cropped to +-4 degrees around the
CARQ position at t extrapolated along the t - 6 h -> t motion to that hour.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import time
import urllib.request
from typing import Iterable

import numpy as np

BUCKET = "https://noaa-gmgsi-pds.s3.amazonaws.com"
HALF_DEG = 4.0
OFFSETS = {"p2": 2, "m4": -4}          # image hour relative to the cycle t
UA = {"User-Agent": "HazardPulse/1.0 (hurricane RI IR features)"}


def carq_positions(records: Iterable, cyc: dt.datetime) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
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


def image_key(hour: dt.datetime) -> str | None:
    prefix = f"GMGSI_LW/{hour:%Y/%m/%d/%H}/"
    req = urllib.request.Request(f"{BUCKET}/?list-type=2&prefix={prefix}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        keys = re.findall(r"<Key>([^<]+)</Key>", r.read().decode("utf-8"))
    return sorted(keys)[0] if keys else None


def fetch_image(hour: dt.datetime):
    """``(key, uint8 counts (rows, cols), lat per row, lon per column)``, or Nones when the hour
    has no image. 255 marks a missing pixel."""
    import h5py
    key = image_key(hour)
    if key is None:
        return None, None, None, None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{BUCKET}/{key}", headers=UA), timeout=120) as r:
                blob = r.read()
            break
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(5)
    with h5py.File(io.BytesIO(blob), "r") as f:
        d = f["data"][0]
        lat, lon = f["lat"][:, 0].astype(np.float64), f["lon"][0, :].astype(np.float64)
    counts = np.where(d < 0, 255, np.clip(d, 0, 254)).astype(np.uint8)
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


def centres(records: Iterable, cyc: dt.datetime) -> dict[str, tuple[dt.datetime, tuple[float, float]]] | None:
    """``{tag: (image hour, centre)}`` for a cycle, or None without a CARQ fix at t."""
    p0, pm6 = carq_positions(records, cyc)
    if p0 is None:
        return None
    return {tag: (cyc + dt.timedelta(hours=h), extrapolate(p0, pm6, h)) for tag, h in OFFSETS.items()}
