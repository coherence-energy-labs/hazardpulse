"""ATCF (Automated Tropical Cyclone Forecasting) deck parser.

Parses NHC/JTWC a-deck and b-deck text into structured records for the
operational hurricane scoring pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# NHC real-time ATCF root
ATCF_ROOT = "https://ftp.nhc.noaa.gov/atcf"

# Aid models to extract forecast features from
DEFAULT_AID_MODELS = (
    "AVNO", "HWRF", "HMON", "DSHP", "LGEM", "CTCX",
    "EMX0", "AEMN", "OFCL", "JTWC",
)

# Priority order for picking the analysis (tau=0) record. CARQ precedes OFCL: an NHC OFCL
# tau-0 line carries MSLP 0 (151 of 151 in Ian 2022, Otis 2023, Milton 2024; CARQ 291 of
# 291 carry one), so with OFCL first 88% of live cases lost the analysis pressure and all
# three pressure tendencies -- live v8.2 AUC 0.824 and mean forecast 0.042 against 0.832 and
# 0.060 from CARQ on the same 2,220 held-out cases (observed 0.067; results/calibration/
# hurricane_vs_ships.json "operational_inputs"). CARQ is also what SHIPS-RII starts from.
DEFAULT_ANALYSIS_PRIORITY = (
    "BEST", "CARQ", "OFCL", "JTWC", "WRNG",
    "AVNO", "HWRF", "HMON",
)

# Forecast lead times (hours) to extract
DEFAULT_LEAD_TIMES = (12, 24, 36, 48, 72)


@dataclass
class ATCFRecord:
    """Single parsed ATCF a-deck or b-deck record."""

    basin: str
    storm_number: int
    cycle: dt.datetime
    tau_hours: int
    model: str
    lat: float | None
    lon: float | None
    vmax_kt: float | None
    mslp_hpa: float | None
    storm_name: str | None = None


def _safe_float(val: str) -> float | None:
    """Parse a numeric string, returning None for missing/bad values."""
    val = val.strip()
    if not val or val in ("", " "):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_lat(val: str) -> float | None:
    """Parse ATCF latitude like '281N' -> 28.1."""
    val = val.strip()
    if not val:
        return None
    hemisphere = val[-1].upper() if val[-1].isalpha() else ""
    num_str = val[:-1] if hemisphere else val
    num = _safe_float(num_str)
    if num is None:
        return None
    lat = num / 10.0
    if hemisphere == "S":
        lat = -lat
    return lat


def _parse_lon(val: str) -> float | None:
    """Parse ATCF longitude like '0945W' -> -94.5."""
    val = val.strip()
    if not val:
        return None
    hemisphere = val[-1].upper() if val[-1].isalpha() else ""
    num_str = val[:-1] if hemisphere else val
    num = _safe_float(num_str)
    if num is None:
        return None
    lon = num / 10.0
    if hemisphere == "W":
        lon = -lon
    return lon


def parse_atcf_deck(text: str) -> list[ATCFRecord]:
    """Parse ATCF a-deck or b-deck text into ATCFRecord list.

    Format reference: https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt

    Each line is comma-separated with at minimum:
      BASIN, CY, YYYYMMDDHH, TECHNUM/MIN, TECH, TAU, LAT, LON, VMAX, MSLP, ...
    """
    records: list[ATCFRecord] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue

        try:
            basin = parts[0].strip()
            storm_number = int(parts[1].strip())

            # Parse cycle datetime (YYYYMMDDHH)
            dtstr = parts[2].strip()
            if len(dtstr) < 10:
                continue
            cycle = dt.datetime(
                int(dtstr[:4]), int(dtstr[4:6]), int(dtstr[6:8]),
                int(dtstr[8:10]),
            )

            # parts[3] = TECHNUM/MIN (skip)
            model = parts[4].strip().upper()
            tau_hours = int(parts[5].strip())

            lat = _parse_lat(parts[6])
            lon = _parse_lon(parts[7])

            vmax_kt = _safe_float(parts[8])
            if vmax_kt is not None and vmax_kt <= 0:
                vmax_kt = None

            mslp_hpa = _safe_float(parts[9])
            if mslp_hpa is not None and mslp_hpa <= 0:
                mslp_hpa = None

            # Storm name is sometimes in column 27 (index 27) if present
            storm_name = None
            if len(parts) > 27:
                name_candidate = parts[27].strip()
                if name_candidate and name_candidate not in ("", " "):
                    storm_name = name_candidate

            records.append(ATCFRecord(
                basin=basin,
                storm_number=storm_number,
                cycle=cycle,
                tau_hours=tau_hours,
                model=model,
                lat=lat,
                lon=lon,
                vmax_kt=vmax_kt,
                mslp_hpa=mslp_hpa,
                storm_name=storm_name,
            ))
        except (ValueError, IndexError):
            continue

    return records


def build_operational_ri_dataset(*args, **kwargs):
    """Placeholder for CLI — build advisory-time RI cases from ATCF decks."""
    raise NotImplementedError("build_operational_ri_dataset not yet implemented")


def summarize_operational_ri_cases(cases: list[dict]) -> dict:
    """Return summary statistics for a set of RI cases."""
    n = len(cases)
    if n == 0:
        return {"n_cases": 0}
    n_pos = sum(1 for c in cases if c.get("ri_label_30kt", 0) == 1)
    return {
        "n_cases": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "ri_rate": n_pos / n if n > 0 else 0.0,
    }


def write_operational_ri_dataset(cases: list[dict], path) -> None:
    """Write cases to JSONL file."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")
