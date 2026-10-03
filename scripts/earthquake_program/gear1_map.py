#!/usr/bin/env python3
"""Amendment E1 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10): GEAR1 on the site's grid.

    python scripts/earthquake_program/gear1_map.py

Streams GEAR1.dat (Zenodo record 7086053, CC-BY-4.0; Bird et al. 2015) once, hashing it as it
goes. Its columns are CUMULATIVE rate densities (earthquakes with M >= the column's magnitude per
m^2 per year -- pyCSEP ``read_GEAR1_format``), so a 0.1-degree cell's shallow M >= 5.95 rate is its
first column times its exact spherical area. Those rates are summed into the 11,700 cells of the
contract grid by the verifier's own assignment (``cell_index``). Nothing but the 11,700 sums is
kept: the 4.7 GB source is never written to disk.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

URL = "https://zenodo.org/api/records/7086053/files/GEAR1.dat/content"
OUT = C.RESULTS / "gear1_cells.json"
N_BINS = 31
CHUNK = 64 << 20
EARTH_RADIUS_M = 6371000.0
HALF_CELL = np.radians(0.05)


def cell_area_m2(lat_centre_deg) -> np.ndarray:
    """Exact spherical area of a 0.1 x 0.1 degree cell centred at ``lat_centre_deg``."""
    la = np.radians(np.asarray(lat_centre_deg, np.float64))
    return EARTH_RADIUS_M ** 2 * np.radians(0.1) * (np.sin(np.minimum(la + HALF_CELL, np.pi / 2))
                                                    - np.sin(np.maximum(la - HALF_CELL, -np.pi / 2)))


def aggregate(stream, n_cols: int = 2 + N_BINS, log=print) -> dict:
    """Sum the rate columns of a GEAR1-format CSV stream into grid cells. Every data line must have
    exactly ``n_cols`` numbers, or the run stops (a short line is a corrupt download, not data)."""
    h = hashlib.sha256()
    cells = np.zeros(of.N_CELLS, dtype=np.float64)
    header, tail, n_lines, n_bytes, total = None, b"", 0, 0, 0.0
    lat_rng, lon_rng = [np.inf, -np.inf], [np.inf, -np.inf]
    def take(body: bytes) -> None:
        nonlocal total, n_lines, lat_rng, lon_rng
        text = body.replace(b"\r", b"").strip(b"\n").replace(b"\n", b",").decode("ascii")
        if not text:
            return
        vals = np.array(text.split(","), dtype=np.float64)
        if vals.size % n_cols:
            raise SystemExit(f"a line does not have {n_cols} numbers near byte {n_bytes}")
        rows = vals.reshape(-1, n_cols)
        # the columns are CUMULATIVE rate densities (M >= bin, per m^2 per year; pyCSEP
        # read_GEAR1_format): the M >= 5.95 rate of a cell is its first column times its area
        rate = rows[:, 2] * cell_area_m2(rows[:, 1])
        np.add.at(cells, of.cell_index(rows[:, 1], rows[:, 0]), rate)
        total += float(rate.sum())
        n_lines += len(rows)
        lat_rng = [min(lat_rng[0], rows[:, 1].min()), max(lat_rng[1], rows[:, 1].max())]
        lon_rng = [min(lon_rng[0], rows[:, 0].min()), max(lon_rng[1], rows[:, 0].max())]

    while True:
        block = stream.read(CHUNK)
        if not block:
            break
        h.update(block)
        n_bytes += len(block)
        data = tail + block
        if header is None:                      # the first line names the columns
            first = data.find(b"\n")
            if first < 0:
                tail = data
                continue
            header, data = data[:first].decode("ascii", "replace").strip(), data[first + 1:]
        cut = data.rfind(b"\n")
        if cut < 0:
            tail = data
            continue
        take(data[:cut])
        tail = data[cut + 1:]
        log(f"  {n_bytes / 1e9:.2f} GB, {n_lines:,} cells, running total {total:.2f}")
    take(tail)                                  # a last line without its newline is still a line
    return {"sha256": h.hexdigest(), "bytes": n_bytes, "lines": n_lines, "header": header,
            "lat_range": lat_rng, "lon_range": lon_rng, "global_total": total, "cells": cells}


def main() -> int:
    req = urllib.request.Request(URL, headers={"User-Agent": "HazardPulse-research/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = aggregate(r)
    expected_lines = 3600 * 1800
    if res["lines"] != expected_lines:
        raise SystemExit(f"{res['lines']:,} 0.1-degree cells, expected {expected_lines:,}")
    cells = res.pop("cells")
    # the unit check the registration requires: GEAR1's global shallow M >= 5.95 rate is of order
    # 10^2 per year; a total far outside [50, 500] means the file was read in the wrong unit
    if not 50.0 <= res["global_total"] <= 500.0:
        raise SystemExit(f"global total {res['global_total']:.4g} per year is not a shallow M>=5.95 rate: unit misread")
    C.write_json("gear1_cells.json", {
        "program": "docs/EARTHQUAKE_FORECAST_PROGRAM.md section 10 (amendment E1)",
        "source": URL, "citation": "Bird, Jackson, Kagan, Kreemer & Stein (2015) BSSA doi:10.1785/0120150058",
        "license": "CC-BY-4.0", "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **res, "unit": "shallow M>=5.95 earthquakes per year per grid cell (first, cumulative column x 0.1-deg cell area)",
        "cells_per_year": [float(x) for x in cells]})
    print(f"wrote {OUT}: sha256 {res['sha256'][:16]}..., {res['lines']:,} cells, global total {res['global_total']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
