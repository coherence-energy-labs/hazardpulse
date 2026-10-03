#!/usr/bin/env python3
"""HazardPulse hurricane RI v8.2 against NOAA's operational RI guidance, on the same held-out cases.

WHAT IS COMPARED
    v8.2 (C)   the model scripts/evaluate_hurricane_ri.py evaluates as candidate C: v8.2 members
               fitted on storms first seen 2000-2018, Newton calibration on 2019-2021, never
               touched by the 2022-2024 test storms. Its full-test AUC/Brier must reproduce
               results/calibration/hurricane_ri_evaluation.json exactly or this script stops.
    v8.2 served the pinned artifact results/models/hurricane_ri_v8_2.json (members <= 2021,
               calibration FITTED ON 2022-2024 = the test seasons): its AUC is honest (members
               never saw a test storm; calibration is monotone), its Brier/reliability are not.
    SHIPS-RII  NOAA's operational Rapid Intensification Index for 30 kt / 24 h, read from the
               NHC ATCF archive e-decks (probability decks; the a-decks carry no RI
               probabilities -- their RI25/RI30/... aids are deterministic intensity tracks).

THE ATCF NAMES (each identity carries its evidence class; see RI_TECHS)
    e-deck record "RI": BASIN, CY, YYYYMMDDHH, RI, TECH, TAU, LAT, LON, PROB(%), dV(kt),
    V_final(kt), initials, RIstartTAU, RIstopTAU  (NRL ATCF probability format, 11/2020).
    The 30 kt / 24 h probability is the record with TAU=24, dV=30, start=0, stop=24.

ALIGNMENT (exact; tested in tests/test_hurricane_vs_ships_alignment.py)
    evaluation case = (IBTrACS SID, synoptic issue_time) of the v8.2 test rows
    -> IBTrACS USA_ATCF_ID at that very fix (per fix: a storm that crosses AL -> EP changes id)
    -> e-deck DTG = issue_time as YYYYMMDDHH, same storm id. No nearest-time matching.
    Truth is the evaluation's own label (best-track wind(t+24h) - wind(t) >= 30 kt), which is
    also SHIPS-RII's definition (>= 30 kt in 24 h), so the two are scored on one event.
    The v8.2 file holds 673 test rows (637 calibration rows) that are exact duplicates: storms
    present in two IBTrACS basin files are emitted twice by the builder. A case is a (storm,
    time): the primary analysis counts each once; the evaluation's row weighting is reported
    as a sensitivity.

ADDED (beyond the head-to-head)
    * stacking v8.2 + SHIPS-RII by logistic regression on the CALIBRATION seasons only (the
      e-decks carry RI records from 2020, so in practice storms first seen 2020-2021), with the
      controls that separate "RII adds information" from "an AL/EP recalibration helps":
      v8.2 recalibrated alone on the same rows, and RII recalibrated alone;
    * v8.2 scored on OPERATIONAL inputs (the live case builder of scripts/fetch_and_score.py
      run on the archived a-deck truncated at the forecast time) -- a measured answer to "v8.2
      reads best track, RII reads real-time analyses". The live builder takes OFCL tau 0 as the
      analysis, and OFCL lines carry no pressure, so a second variant routes it to CARQ (what
      SHIPS-RII starts from) to separate that serving defect from the input-source effect;
    * the RII developmental-sample definition (storm tropical/subtropical at t and t+24 h and
      over water at every IBTrACS fix in between) as a second case definition.

Intervals: 95% percentile bootstrap over STORMS (2,000 replicates), as the evaluation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.hurricane import operational_ri as ori  # noqa: E402
from hazardpulse.hurricane import ri_evaluation as ev  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402
from hazardpulse.hurricane.atcf import ATCFRecord, parse_atcf_deck  # noqa: E402

REPORT_PATH = ROOT / "results" / "calibration" / "hurricane_vs_ships.json"
EVALUATION_PATH = ROOT / "results" / "calibration" / "hurricane_ri_evaluation.json"
BOOTSTRAP_REPS = 2000

NHC_ARCHIVE = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/"
USER_AGENT = "HazardPulse-research/1.0 (SHIPS-RII benchmark; sequential, cached)"
NHC_PREFIXES = ("al", "ep", "cp")
NHC_BASINS = frozenset({"AL", "EP", "CP"})

# The 30 kt / 24 h event: the evaluation's truth and SHIPS-RII's own threshold.
RI_DEFINITION = {"tau": 24, "dv_kt": 30, "start_tau": 0, "stop_tau": 24}
PRIMARY_TECH = "RIOD"
# Whole-percent forecasts: logit-based quantities (calibration slope, stacking) use the
# probability clipped to half the reporting resolution. Brier/AUC/reliability use it raw.
PCT_CLIP = 0.005

RI_TECHS: dict[str, dict[str, str]] = {
    "RIOD": {
        "product": "SHIPS-RII (the Rapid Intensification Index; linear-discriminant version)",
        "identity_class": "MEASURED",
        "evidence": (
            "NHC Ian (AL092022) discussions quote SHIPS-RII values that equal RIOD and no other "
            "e-deck tech: #6 (12Z 24 Sep) 67% for 65 kt/72 h (RIOD 67; RIOB 60, RIOL 82, RIOC 70), "
            "#7 (18Z 24 Sep) 66% 65 kt/72 h (RIOD 66; 39/82/62), #14 (12Z 26 Sep) 73% 35 kt/24 h "
            "and 79% 45 kt/36 h (RIOD 73 and 79; RIOB 59/17, RIOL 64/42, RIOC 65/46)"),
    },
    "RIOL": {
        "product": "SHIPS-RII logistic-regression version",
        "identity_class": "HYPOTHESIS",
        "evidence": "tech letter (L) and membership of the RIOC consensus; no quoted value checked",
    },
    "RIOB": {
        "product": "SHIPS-RII Bayesian version",
        "identity_class": "HYPOTHESIS",
        "evidence": "tech letter (B) and membership of the RIOC consensus; no quoted value checked",
    },
    "RIOC": {
        "product": "SHIPS RI consensus = mean(RIOD, RIOL, RIOB)",
        "identity_class": "MEASURED",
        "evidence": ("RIOC equals the mean of RIOB/RIOD/RIOL within 1 point in 45,579 of 45,580 "
                     "2020-2024 records with all four present (one outlier); SHIPS text output "
                     "labels the same average 'Consensus' of SHIPS-RII/Logistic/Bayesian"),
    },
    "DTOP": {
        "product": "DTOPS (Deterministic to Probabilistic Statistical RI model)",
        "identity_class": "MEASURED",
        "evidence": ("NHC Helene (AL092024) discussion #10 (18Z 25 Sep): DTOPS 'at least a 90 percent "
                     "chance of a 35-kt increase over the next 24 hours' = DTOP 90 (every other tech "
                     "<= 62; DTPE does not exist in 2024); Ian 2022 #8-#10 DTOPS '>90%' at 48/72 h "
                     "consistent with DTOP 91-95"),
    },
    "SDCN": {
        "product": "SDCON-like consensus of RIOC and DTOP (2024 e-decks only)",
        "identity_class": "HYPOTHESIS",
        "evidence": "equals mean(RIOC, DTOP) within 1 point in 5,524 of 5,928 2024 records (93.2%)",
    },
    "EIOD": {"product": "E-prefixed twin of RIOD (unverified; possibly the ECMWF-driven SHIPS run)",
             "identity_class": "HYPOTHESIS", "evidence": "name pattern only"},
    "EIOC": {"product": "E-prefixed twin of RIOC = mean(EIOB, EIOD, EIOL)",
             "identity_class": "HYPOTHESIS", "evidence": "consensus identity measured; source unverified"},
}
PRIMARY_NOAA = ("RIOD", "RIOC", "DTOP")           # the head-to-head table and paired deltas
ALL_AIDS_STACK = ("RIOD", "RIOL", "RIOB", "DTOP")  # the "all NOAA aids" stack

TROPICAL_STATUS = frozenset({"TD", "TS", "HU", "SD", "SS"})

# v8.2's inputs, grouped by what the operational source changes together (MPI features are
# derived from the analysis wind and latitude, so they travel with them).
FEATURE_GROUPS = {
    "vmax_and_mpi": ("analysis_vmax_kt", "mpi_deficit", "intensity_frac_mpi"),
    "mslp": ("analysis_mslp_hpa",),
    "dv_6_12_24h": ("analysis_dv_6h", "analysis_dv_12h", "analysis_dv_24h"),
    "dp_6_12_24h": ("analysis_dp_6h", "analysis_dp_12h", "analysis_dp_24h"),
    "position": ("analysis_lat", "analysis_lon", "abs_lat"),
    "translation_speed": ("translation_speed_kmh",),
    "storm_age": ("storm_age_h",),
}


# ---------------------------------------------------------------------------
# Download (polite: sequential, cached, integrity-checked, atomic)
# ---------------------------------------------------------------------------

def _get(url: str, timeout: float = 180.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_decks(cache: Path, kind: str, years: Iterable[int], pause_s: float = 0.5) -> dict[int, int]:
    """Mirror ``{kind}{al,ep,cp}NNYYYY.dat.gz`` from the NHC archive into ``cache/<year>/``."""
    counts: dict[int, int] = {}
    for year in years:
        listing = _get(NHC_ARCHIVE.format(year=year)).decode("utf-8", "replace")
        names = sorted(set(re.findall(rf'href="({kind}(?:al|ep|cp)\d{{2}}{year}\.dat\.gz)"', listing)))
        target = cache / str(year)
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            path = target / name
            if path.exists():
                try:
                    gzip.decompress(path.read_bytes())
                    continue
                except (OSError, EOFError):
                    pass
            data = _get(NHC_ARCHIVE.format(year=year) + name)
            gzip.decompress(data)  # refuse a truncated download before it lands
            part = path.with_suffix(".part")
            part.write_bytes(data)
            part.replace(path)
            time.sleep(pause_s)
        counts[year] = len(names)
        time.sleep(1.0)
    return counts


# ---------------------------------------------------------------------------
# e-deck RI records
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EdeckRI:
    atcf_id: str        # e.g. AL092022
    dtg: str            # YYYYMMDDHH
    tech: str
    tau: int
    dv_kt: int
    v_final_kt: int | None
    start_tau: int
    stop_tau: int
    prob_pct: int
    lat: float | None
    lon: float | None


def _lat(s: str) -> float | None:
    s = s.strip()
    if len(s) < 2 or s[-1] not in "NS":
        return None
    try:
        v = int(s[:-1]) / 10.0
    except ValueError:
        return None
    return -v if s[-1] == "S" else v


def _lon(s: str) -> float | None:
    s = s.strip()
    if len(s) < 2 or s[-1] not in "EW":
        return None
    try:
        v = int(s[:-1]) / 10.0
    except ValueError:
        return None
    return -v if s[-1] == "W" else v


def storm_id_from_filename(name: str) -> str | None:
    """``eal092022.dat.gz`` -> ``AL092022`` (None for anything else)."""
    m = re.fullmatch(r"[abe]([a-z]{2})(\d{2})(\d{4})\.dat(?:\.gz)?", name.lower())
    return f"{m.group(1).upper()}{m.group(2)}{m.group(3)}" if m else None


def parse_edeck_ri(text: str, file_storm_id: str) -> tuple[list[EdeckRI], dict[str, int]]:
    """Every well-formed ``RI`` record of one e-deck. The storm id is the line's BASIN+CY with
    the file's season year; a line whose BASIN+CY differs from the file's is counted (and kept
    under the line's own id: that is the storm it describes)."""
    out: list[EdeckRI] = []
    stats = collections.Counter()
    year = file_storm_id[-4:]
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[3] != "RI":
            continue
        stats["ri_lines"] += 1
        if len(parts) < 14:
            stats["malformed"] += 1
            continue
        try:
            basin, cy, dtg = parts[0].upper(), int(parts[1]), parts[2]
            tau, prob, dv = int(parts[5]), int(parts[8]), int(parts[9])
            start, stop = int(parts[12]), int(parts[13])
        except ValueError:
            stats["malformed"] += 1
            continue
        if not re.fullmatch(r"\d{10}", dtg) or not 0 <= prob <= 100:
            stats["malformed"] += 1
            continue
        v_final = int(parts[10]) if parts[10].lstrip("-").isdigit() else None
        atcf_id = f"{basin}{cy:02d}{year}"
        if atcf_id != file_storm_id:
            stats["line_id_differs_from_file"] += 1
        out.append(EdeckRI(atcf_id, dtg, parts[4].upper(), tau, dv, v_final, start, stop, prob,
                           _lat(parts[6]), _lon(parts[7])))
    stats["records"] = len(out)
    return out, dict(stats)


def index_ri(
    records: Iterable[EdeckRI],
    *,
    tau: int = RI_DEFINITION["tau"],
    dv_kt: int = RI_DEFINITION["dv_kt"],
    start_tau: int = RI_DEFINITION["start_tau"],
    stop_tau: int = RI_DEFINITION["stop_tau"],
) -> tuple[dict[tuple[str, str], dict[str, EdeckRI]], dict[str, int]]:
    """``{(atcf_id, dtg): {tech: record}}`` for ONE event definition.

    A record matches only if all four of (tau, dV, start, stop) match: the 30 kt / 24 h
    probability, never the 25 or 35 kt one, never 30 kt over another window. Repeats of a key
    with the same probability are kept once; a key whose repeats DISAGREE is dropped (no
    guessing which run was operational) and counted.
    """
    table: dict[tuple[str, str], dict[str, EdeckRI]] = {}
    conflicted: set[tuple[str, str, str]] = set()
    stats = collections.Counter()
    for r in records:
        if (r.tau, r.dv_kt, r.start_tau, r.stop_tau) != (tau, dv_kt, start_tau, stop_tau):
            continue
        key = (r.atcf_id, r.dtg)
        slot = table.setdefault(key, {})
        if (r.atcf_id, r.dtg, r.tech) in conflicted:
            continue
        prev = slot.get(r.tech)
        if prev is None:
            slot[r.tech] = r
        elif prev.prob_pct == r.prob_pct:
            stats["identical_repeats"] += 1
        else:
            stats["conflicting_keys_dropped"] += 1
            conflicted.add((r.atcf_id, r.dtg, r.tech))
            del slot[r.tech]
    for key in [k for k, v in table.items() if not v]:
        del table[key]
    stats["cycles"] = len(table)
    return table, dict(stats)


def load_edecks(cache: Path, years: Iterable[int]) -> tuple[list[EdeckRI], dict[str, dict[str, int]]]:
    records: list[EdeckRI] = []
    per_file: dict[str, dict[str, int]] = {}
    for year in years:
        for path in sorted((cache / str(year)).glob("e*.dat.gz")):
            sid = storm_id_from_filename(path.name)
            if sid is None or sid[:2] not in NHC_BASINS:
                continue
            recs, stats = parse_edeck_ri(gzip.decompress(path.read_bytes()).decode("utf-8", "replace"), sid)
            records.extend(recs)
            per_file[sid] = stats
    return records, per_file


# ---------------------------------------------------------------------------
# IBTrACS: the ATCF id, status and land distance at every fix of the evaluation's storms
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IbFix:
    time: dt.datetime
    atcf_id: str
    usa_wind: float | None
    usa_status: str
    dist2land_km: float | None
    lat: float | None
    lon: float | None


def _num(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s) if s else None
    except ValueError:
        return None


def parse_ibtracs_fixes(lines: Iterable[str], wanted_sids: set[str]) -> dict[str, dict[dt.datetime, IbFix]]:
    """``{SID: {time: fix}}`` for the wanted storms. A storm present in two basin files has the
    same rows in both; the first copy of each (SID, time) is kept."""
    reader = csv.reader(lines)
    header = next(reader)
    col = {h.strip(): i for i, h in enumerate(header)}
    next(reader, None)  # units row
    out: dict[str, dict[dt.datetime, IbFix]] = {}
    for row in reader:
        if len(row) <= col["USA_STATUS"]:
            continue
        sid = row[col["SID"]].strip()
        if sid not in wanted_sids:
            continue
        try:
            t = dt.datetime.strptime(row[col["ISO_TIME"]].strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        fixes = out.setdefault(sid, {})
        if t in fixes:
            continue
        fixes[t] = IbFix(t, row[col["USA_ATCF_ID"]].strip().upper(), _num(row[col["USA_WIND"]]),
                         row[col["USA_STATUS"]].strip().upper(), _num(row[col["DIST2LAND"]]),
                         _num(row[col["LAT"]]), _num(row[col["LON"]]))
    return out


def ibtracs_paths(cache_root: Path) -> dict[str, Path]:
    import build_hurricane_training_data as bh  # the builder's own URL + cache naming

    return {b: cache_root / "ibtracs" / f"{hashlib.md5(bh.IBTRACS_URL.format(basin=b).encode()).hexdigest()}.csv"
            for b in bh.BASINS}


def load_ibtracs(cache_root: Path, wanted_sids: set[str]) -> dict[str, dict[dt.datetime, IbFix]]:
    merged: dict[str, dict[dt.datetime, IbFix]] = {}
    for basin, path in ibtracs_paths(cache_root).items():
        if not path.exists():
            raise SystemExit(f"IBTrACS {basin} cache missing at {path} (the file the v8.2 set was built from)")
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for sid, fixes in parse_ibtracs_fixes(fh, wanted_sids).items():
                slot = merged.setdefault(sid, {})
                for t, f in fixes.items():
                    slot.setdefault(t, f)
    return merged


def rii_sample_mask(fixes: dict[dt.datetime, IbFix], t: dt.datetime) -> bool | None:
    """SHIPS-RII developmental-sample definition, implemented: tropical/subtropical status at
    t and t+24 h and over water (DIST2LAND > 0) at every IBTrACS fix in [t, t+24 h]."""
    t24 = t + dt.timedelta(hours=24)
    a, b = fixes.get(t), fixes.get(t24)
    if a is None or b is None:
        return None
    window = [f for tt, f in fixes.items() if t <= tt <= t24]
    if any(f.dist2land_km is None for f in window):
        return None
    return bool(a.usa_status in TROPICAL_STATUS and b.usa_status in TROPICAL_STATUS
                and all(f.dist2land_km > 0 for f in window))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

REASONS = ("matched", "non_synoptic", "no_ibtracs_fix", "no_usa_atcf_id", "jtwc_basin_no_public_ri",
           "no_edeck_file", "edeck_has_no_ri_records", "no_ri_at_cycle", "primary_tech_missing")


def case_key(case: dict) -> tuple[str, str]:
    return case["storm_id"], case["issue_time"]


def unique_cases(cases: list[dict]) -> tuple[list[dict], dict[tuple[str, str], int]]:
    """One row per (storm, time), first occurrence kept; plus the multiplicity of each key.
    Refuses if two rows of one key disagree (they would not be duplicates)."""
    seen: dict[tuple[str, str], dict] = {}
    mult: dict[tuple[str, str], int] = collections.Counter()
    for c in cases:
        k = case_key(c)
        mult[k] += 1
        if k in seen:
            if json.dumps(seen[k], sort_keys=True) != json.dumps(c, sort_keys=True):
                raise ValueError(f"rows of {k} differ: not a duplicate")
            continue
        seen[k] = c
    return list(seen.values()), dict(mult)


def align_case(
    case: dict,
    ibtracs: dict[str, dict[dt.datetime, IbFix]],
    ri_table: dict[tuple[str, str], dict[str, EdeckRI]],
    edeck_files: dict[str, int],
    primary_tech: str = PRIMARY_TECH,
) -> dict[str, Any]:
    """Map one evaluation case to its e-deck cycle, exactly, or say why it cannot be.

    ``edeck_files``: ``{atcf_id: number of RI records in its e-deck}`` for every e-deck file
    present. Returns ``{"reason", "atcf_id", "dtg", "techs": {tech: EdeckRI}, "fix"}``.
    """
    out: dict[str, Any] = {"reason": None, "atcf_id": None, "dtg": None, "techs": {}, "fix": None}
    t = dt.datetime.strptime(case["issue_time"], "%Y-%m-%d %H:%M:%S")
    if t.minute or t.second or t.hour % 6:
        out["reason"] = "non_synoptic"
        return out
    fix = ibtracs.get(case["storm_id"], {}).get(t)
    if fix is None:
        out["reason"] = "no_ibtracs_fix"
        return out
    out["fix"] = fix
    if not re.fullmatch(r"[A-Z]{2}\d{2}\d{4}", fix.atcf_id or ""):
        out["reason"] = "no_usa_atcf_id"
        return out
    out["atcf_id"] = fix.atcf_id
    out["dtg"] = t.strftime("%Y%m%d%H")
    if fix.atcf_id[:2] not in NHC_BASINS:
        out["reason"] = "jtwc_basin_no_public_ri"
        return out
    if fix.atcf_id not in edeck_files:
        out["reason"] = "no_edeck_file"
        return out
    if edeck_files[fix.atcf_id] == 0:
        out["reason"] = "edeck_has_no_ri_records"
        return out
    techs = ri_table.get((fix.atcf_id, out["dtg"]))
    if not techs:
        out["reason"] = "no_ri_at_cycle"
        return out
    out["techs"] = dict(techs)
    out["reason"] = "matched" if primary_tech in techs else "primary_tech_missing"
    return out


# ---------------------------------------------------------------------------
# Operational-input v8.2: the live case builder on the archived a-deck, truncated at t
# ---------------------------------------------------------------------------

def adeck_tau0_records(path: Path) -> list[ATCFRecord]:
    """The tau-0 records of an archived a-deck (all the live builder reads for v8.2's features:
    analysis wind/pressure/position now and 6/12/24 h earlier, first analysis time)."""
    keep = []
    for line in gzip.decompress(path.read_bytes()).decode("utf-8", "replace").splitlines():
        parts = line.split(",", 6)
        if len(parts) > 6 and parts[5].strip() == "0":
            keep.append(line)
    return parse_atcf_deck("\n".join(keep))


def operational_cases(
    adeck_cache: Path, wanted: dict[str, list[str]], exclude_models: frozenset[str] = frozenset()
) -> tuple[dict[tuple[str, str], dict], collections.Counter]:
    """``{(atcf_id, dtg): live case}`` built exactly as scripts/fetch_and_score.py does live,
    from the archived a-deck with every record after the forecast time removed.

    ``exclude_models`` drops aids before the builder sees them: excluding OFCL makes the
    builder fall through to CARQ -- the real-time analysis SHIPS-RII itself starts from, and
    the only tau-0 record in an NHC a-deck that carries a pressure (OFCL tau-0 lines carry
    MSLP 0: 151 of 151 in Ian 2022, Otis 2023 and Milton 2024; CARQ 291 of 291).
    """
    import fetch_and_score as live  # the canonical live builder

    out: dict[tuple[str, str], dict] = {}
    why = collections.Counter()
    for atcf_id, dtgs in sorted(wanted.items()):
        path = adeck_cache / atcf_id[-4:] / f"a{atcf_id.lower()}.dat.gz"
        if not path.exists():
            why["no_adeck_file"] += len(dtgs)
            continue
        recs = [r for r in adeck_tau0_records(path) if r.model not in exclude_models]
        for dtg in dtgs:
            t = dt.datetime.strptime(dtg, "%Y%m%d%H")
            case = live.build_live_case(atcf_id, [r for r in recs if r.cycle <= t])
            if case is None or case["issue_time"] != t.isoformat():
                why["no_analysis_at_cycle"] += 1
                continue
            why["built_" + str(case["analysis_model"])] += 1
            out[(atcf_id, dtg)] = case
    return out, why


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _labels(rows: list[dict]) -> np.ndarray:
    return np.array([float(c["ri_label_30kt"]) for c in rows])


def great_circle_km(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return float(6371.0 * np.arccos(np.clip(c, -1.0, 1.0)))


def model_c(cache_root: Path, smoke: bool, log=print) -> tuple[dict, dict]:
    """Candidate C of the evaluation, rebuilt by the evaluation's own code (members cached)."""
    import evaluate_hurricane_ri as evr

    data_sha = ri_model.sha256_file(ri_model.DATASETS["v8.2"])
    key = hashlib.sha256(json.dumps({
        "data": data_sha, "trainer": ri_model.trainer_fingerprint(),
        "config": ri_model.config_dict(evr.SMOKE_CONFIG if smoke else None),
        "years": evr.ORIGIN["members"],
    }, sort_keys=True).encode()).hexdigest()
    path = cache_root / "hurricane_vs_ships" / f"model_c_members_{key[:16]}{'_smoke' if smoke else ''}.json"
    if path.exists():
        members = json.loads(path.read_text(encoding="utf-8"))
        log(f"model C members from cache {path.name}")
    else:
        members = evr._fit_members_job("v8.2", evr.ORIGIN["members"], False, smoke)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(members), encoding="utf-8")
    v82 = ri_model.load_dataset("v8.2")
    cal = ri_model.select_years(v82, evr.ORIGIN["calibration"])
    model = {"schema": ri_model.SCHEMA, "model_version": "hurricane_ri_v8_2", **members,
             "serving": {"impute_live": []}, "calibration": None}
    model["calibration"] = ri_model.fit_logit_calibrator(
        ri_model.member_probabilities(model, cal)["ensemble"], _labels(cal))
    return model, {"members_cache": str(path), "members_key": key}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _clip(p: np.ndarray, quantised: bool) -> np.ndarray:
    return np.clip(p, PCT_CLIP, 1 - PCT_CLIP) if quantised else p


def score_forecast(y, p, groups, clim_global: float, clim_rows: np.ndarray, reps: int,
                   quantised: bool) -> dict[str, Any]:
    """Point metrics, storm-bootstrap intervals, reliability; the evaluation's 'calibrated' test."""
    import evaluate_hurricane_ri as evr

    point = ev.point_metrics(y, p, clim_global)
    pc = _clip(p, quantised)
    try:
        point["calibration_intercept"], point["calibration_slope"] = ev.calibration_intercept_slope(y, pc)
    except ri_model.CalibrationError:
        point["calibration_intercept"] = point["calibration_slope"] = float("nan")
    point["log_loss"] = ev.log_loss(y, pc)
    point["brier_climatology_basin"] = ev.brier(y, clim_rows)
    point["bss_vs_basin_climatology"] = 1.0 - point["brier"] / point["brier_climatology_basin"]

    def stat(idx):
        yy, pp, pcc = y[idx], p[idx], pc[idx]
        b = ev.brier(yy, pp)
        out = {"auc": ev.auc(yy, pp), "brier": b,
               "bss_vs_climatology": 1.0 - b / ev.brier(yy, np.full(len(yy), clim_global)),
               "bss_vs_basin_climatology": 1.0 - b / ev.brier(yy, clim_rows[idx]),
               "mean_minus_observed": float(pp.mean() - yy.mean())}
        try:
            out["calibration_intercept"], out["calibration_slope"] = ev.calibration_intercept_slope(yy, pcc)
        except ri_model.CalibrationError:
            out["calibration_intercept"] = out["calibration_slope"] = float("nan")
        return out

    ci = ev.storm_bootstrap(groups, stat, reps=reps)
    return {**point, "ci95": ci, "reliability": ev.reliability(y, p), "calibrated": evr.calibrated(ci),
            "logit_clip": [PCT_CLIP, 1 - PCT_CLIP] if quantised else None}


def paired(y, pa, pb, groups, reps: int) -> dict[str, Any]:
    point = {"delta_auc": ev.auc(y, pa) - ev.auc(y, pb), "delta_brier": ev.brier(y, pa) - ev.brier(y, pb)}
    ci = ev.storm_bootstrap(groups, lambda idx: {
        "delta_auc": ev.auc(y[idx], pa[idx]) - ev.auc(y[idx], pb[idx]),
        "delta_brier": ev.brier(y[idx], pa[idx]) - ev.brier(y[idx], pb[idx]),
    }, reps=reps)
    return {**point, "ci95": ci}


def fit_stack(cols: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, dict]:
    X = np.column_stack(cols + [np.ones(len(y))])
    return ri_model.newton_logistic(X, y)


def apply_stack(theta: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    return ori.sigmoid(np.column_stack(cols + [np.ones(len(cols[0]))]) @ theta)


def logit_of(p: np.ndarray, quantised: bool) -> np.ndarray:
    return ri_model._logit(_clip(p, quantised))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".cache",
                        help="holds ibtracs/ (the v8.2 build cache) and atcf_adecks/ (NHC decks)")
    parser.add_argument("--download", action="store_true", help="mirror missing NHC e-/a-decks first")
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-operational", action="store_true", help="skip the a-deck operational-input variant")
    parser.add_argument("--smoke", action="store_true", help="tiny members, few replicates: wiring only")
    args = parser.parse_args(argv)
    if args.smoke and args.out == REPORT_PATH:
        parser.error("--smoke must not overwrite the real report; pass --out")
    import evaluate_hurricane_ri as evr

    t0 = time.perf_counter()
    deck_cache = args.cache_root / "atcf_adecks"
    cal_years, test_years = evr.ORIGIN["calibration"], evr.ORIGIN["test"]
    edeck_years = range(cal_years[0], test_years[1] + 1)
    adeck_years = range(test_years[0], test_years[1] + 1)
    if args.download:
        print("e-decks:", download_decks(deck_cache, "e", edeck_years), flush=True)
        if not args.no_operational:
            print("a-decks:", download_decks(deck_cache, "a", adeck_years), flush=True)

    # --- data ------------------------------------------------------------------------------
    v82 = ri_model.load_dataset("v8.2")
    rows = {"calibration": ri_model.select_years(v82, cal_years), "test": ri_model.select_years(v82, test_years)}
    members_rows = ri_model.select_years(v82, evr.ORIGIN["members"])
    uniq, mult = {}, {}
    for split, rr in rows.items():
        uniq[split], mult[split] = unique_cases(rr)

    records, per_file = load_edecks(deck_cache, edeck_years)
    edeck_files = {sid: s.get("records", 0) for sid, s in per_file.items()}
    ri_table, index_stats = index_ri(records)
    tech_counts = collections.Counter(t for v in ri_table.values() for t in v)
    print(f"e-decks: {len(per_file)} files, {len(records)} RI records, {len(ri_table)} cycles with a "
          f"30kt/24h record; techs {dict(tech_counts)}", flush=True)

    wanted = {c["storm_id"] for split in uniq.values() for c in split}
    ibtracs = load_ibtracs(args.cache_root, wanted)

    # IBTrACS must be the file the v8.2 rows were built from: its wind at each case's fix
    wind_check = collections.Counter()
    for c in uniq["test"] + uniq["calibration"]:
        fix = ibtracs.get(c["storm_id"], {}).get(dt.datetime.strptime(c["issue_time"], "%Y-%m-%d %H:%M:%S"))
        if fix is None:
            wind_check["no_fix"] += 1
        elif fix.usa_wind is None or fix.usa_wind <= 0:
            wind_check["no_usa_wind"] += 1
        else:
            wind_check["equal" if fix.usa_wind == c["analysis_vmax_kt"] else "differs"] += 1

    align = {split: [align_case(c, ibtracs, ri_table, edeck_files) for c in uniq[split]] for split in uniq}

    def by_basin(split: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c, a in zip(uniq[split], align[split]):
            for label in (f"dataset:{c['basin']}", f"atcf:{(a['atcf_id'] or '??')[:2]}"):
                b = out.setdefault(label, {"cases": 0, "events": 0, "matched": 0, "matched_events": 0,
                                           "storms": set(), "matched_storms": set(), "reasons": collections.Counter()})
                b["cases"] += 1
                b["events"] += int(c["ri_label_30kt"])
                b["storms"].add(c["storm_id"])
                b["reasons"][a["reason"]] += 1
                if a["reason"] == "matched":
                    b["matched"] += 1
                    b["matched_events"] += int(c["ri_label_30kt"])
                    b["matched_storms"].add(c["storm_id"])
        for b in out.values():
            b["storms"], b["matched_storms"] = len(b["storms"]), len(b["matched_storms"])
            b["reasons"] = dict(b["reasons"])
        return dict(sorted(out.items()))

    match_tables = {split: by_basin(split) for split in uniq}

    # --- models ----------------------------------------------------------------------------
    model, model_meta = model_c(args.cache_root, args.smoke)
    p_c = {split: ri_model.score_cases(model, uniq[split])["calibrated"] for split in uniq}
    # reproduction of the evaluation (its rows, duplicates included)
    p_full = ri_model.score_cases(model, rows["test"])["calibrated"]
    y_full = _labels(rows["test"])
    repro = {"auc": ev.auc(y_full, p_full), "brier": ev.brier(y_full, p_full)}
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    ref = evaluation["results"]["C_v8_2_heldout_newton"]
    repro.update({"evaluation_auc": ref["auc"], "evaluation_brier": ref["brier"],
                  "abs_diff_auc": abs(repro["auc"] - ref["auc"]), "abs_diff_brier": abs(repro["brier"] - ref["brier"])})
    repro["reproduces"] = bool(repro["abs_diff_auc"] <= 1e-12 and repro["abs_diff_brier"] <= 1e-12)
    print(f"model C reproduction: AUC {repro['auc']:.10f} vs {ref['auc']:.10f}; Brier {repro['brier']:.10f} vs "
          f"{ref['brier']:.10f} -> {repro['reproduces']}", flush=True)
    if not repro["reproduces"] and not args.smoke:
        raise SystemExit("model C does not reproduce the evaluation's candidate C; refusing to benchmark a different model")

    served = ri_model.load_model(ri_model.ARTIFACTS["hurricane_ri_v8_2"])
    p_served_test = ri_model.score_cases(served, uniq["test"])["calibrated"]

    y_members = _labels(members_rows)
    clim_global = float(y_members.mean())
    clim_by_basin = {b: float(np.mean([c["ri_label_30kt"] for c in members_rows if c["basin"] == b]))
                     for b in sorted({c["basin"] for c in members_rows})}
    dv_med = float(np.nanmedian(np.array([c.get("analysis_dv_24h") for c in members_rows], dtype=float)))

    def dv24(rr):
        x = np.array([c.get("analysis_dv_24h") for c in rr], dtype=float)
        return np.where(np.isnan(x), dv_med, x)

    theta_p, _ = ri_model.newton_logistic(np.column_stack([dv24(members_rows), np.ones(len(members_rows))]), y_members)

    # --- the intersection ------------------------------------------------------------------
    def subset(split: str, need: tuple[str, ...]) -> np.ndarray:
        return np.array([a["reason"] == "matched" and all(t in a["techs"] for t in need) for a in align[split]])

    def noaa(split: str, idx: np.ndarray, tech: str) -> np.ndarray:
        return np.array([align[split][i]["techs"][tech].prob_pct / 100.0 for i in np.flatnonzero(idx)])

    def block(split: str, idx: np.ndarray, forecasts: dict[str, tuple[np.ndarray, bool]],
              pairs: list[tuple[str, str]], reps: int) -> dict[str, Any]:
        cs = [uniq[split][i] for i in np.flatnonzero(idx)]
        y = _labels(cs)
        g = np.array([c["storm_id"] for c in cs])
        clim_rows = np.array([clim_by_basin[c["basin"]] for c in cs])
        res = {"n": int(len(cs)), "events": int(y.sum()), "storms": int(len(set(g))),
               "observed_rate": float(y.mean()),
               "by_dataset_basin": {b: {"n": int(sum(c["basin"] == b for c in cs)),
                                        "events": int(sum(c["ri_label_30kt"] for c in cs if c["basin"] == b))}
                                    for b in sorted({c["basin"] for c in cs})},
               "forecasts": {}, "paired": {}}
        for name, (p, q) in forecasts.items():
            res["forecasts"][name] = score_forecast(y, p, g, clim_global, clim_rows, reps, q)
        for a_name, b_name in pairs:
            res["paired"][f"{a_name}_minus_{b_name}"] = paired(y, forecasts[a_name][0], forecasts[b_name][0], g, reps)
        return res

    test_idx = subset("test", (PRIMARY_TECH,))
    test_cases = [uniq["test"][i] for i in np.flatnonzero(test_idx)]
    persistence = ori.sigmoid(theta_p[0] * dv24(test_cases) + theta_p[1])
    forecasts = {
        "v8_2_C": (p_c["test"][test_idx], False),
        "v8_2_served_auc_only": (p_served_test[test_idx], False),
        "SHIPS_RII_RIOD": (noaa("test", test_idx, "RIOD"), True),
        "persistence_dv24": (persistence, False),
    }
    # (the other NOAA aids are scored together on their own all-present intersection below)
    print(f"primary intersection: {int(test_idx.sum())} test cases with SHIPS-RII", flush=True)
    primary = block("test", test_idx, forecasts,
                    [("v8_2_C", "SHIPS_RII_RIOD"), ("v8_2_C", "persistence_dv24"),
                     ("SHIPS_RII_RIOD", "persistence_dv24")], args.reps)
    atcf_basin = np.array([(a["atcf_id"] or "??")[:2] for a in align["test"]])
    per_basin = {}
    for label, members in (("AL", ("AL",)), ("EP_CP", ("EP", "CP"))):
        b_idx = test_idx & np.isin(atcf_basin, members)
        per_basin[label] = block("test", b_idx, {
            "v8_2_C": (p_c["test"][b_idx], False),
            "v8_2_served_auc_only": (p_served_test[b_idx], False),
            "SHIPS_RII_RIOD": (noaa("test", b_idx, "RIOD"), True),
        }, [("v8_2_C", "SHIPS_RII_RIOD")], args.reps)
    primary["per_atcf_basin"] = per_basin

    # every NOAA aid on the cases where all of them exist (one case set for the whole table)
    aid_list = ("RIOD", "RIOL", "RIOB", "RIOC", "DTOP")
    all_idx = subset("test", aid_list)
    aid_forecasts = {"v8_2_C": (p_c["test"][all_idx], False)}
    for tech in aid_list:
        aid_forecasts[tech] = (noaa("test", all_idx, tech), True)
    all_aids = block("test", all_idx, aid_forecasts, [("v8_2_C", t) for t in aid_list], args.reps)
    extras = {}
    for tech in ("SDCN", "EIOD", "EIOC"):
        idx = subset("test", (tech,))
        if idx.sum() >= 50:
            extras[tech] = block("test", idx, {"v8_2_C": (p_c["test"][idx], False), tech: (noaa("test", idx, tech), True)},
                                 [("v8_2_C", tech)], args.reps)

    # --- RII developmental-sample definition --------------------------------------------------
    def kaplan(split: str) -> np.ndarray:
        out = []
        for c, a in zip(uniq[split], align[split]):
            t = dt.datetime.strptime(c["issue_time"], "%Y-%m-%d %H:%M:%S")
            out.append(bool(rii_sample_mask(ibtracs.get(c["storm_id"], {}), t)))
        return np.array(out)

    k_idx = test_idx & kaplan("test")
    rii_sample = block("test", k_idx, {"v8_2_C": (p_c["test"][k_idx], False),
                                       "SHIPS_RII_RIOD": (noaa("test", k_idx, "RIOD"), True)},
                       [("v8_2_C", "SHIPS_RII_RIOD")], args.reps)

    # --- evaluation row weighting (duplicates counted as the evaluation counts them) ----------
    w_idx = np.flatnonzero(test_idx)
    rep = np.concatenate([np.full(mult["test"][case_key(uniq["test"][i])], i) for i in w_idx])
    y_w = np.array([uniq["test"][i]["ri_label_30kt"] for i in rep], dtype=float)
    g_w = np.array([uniq["test"][i]["storm_id"] for i in rep])
    pc_w = p_c["test"][rep]
    rii_w = np.array([align["test"][i]["techs"]["RIOD"].prob_pct / 100.0 for i in rep])
    weighted = {"n_rows": int(len(rep)), "events": int(y_w.sum()),
                "v8_2_C": {"auc": ev.auc(y_w, pc_w), "brier": ev.brier(y_w, pc_w)},
                "SHIPS_RII_RIOD": {"auc": ev.auc(y_w, rii_w), "brier": ev.brier(y_w, rii_w)},
                "paired_v8_2_C_minus_SHIPS_RII_RIOD": paired(y_w, pc_w, rii_w, g_w, args.reps)}

    # --- who is left out: v8.2 on the NHC-basin cases without a SHIPS-RII value ------------------
    nhc_idx = np.array([(a["atcf_id"] or "")[:2] in NHC_BASINS for a in align["test"]])
    unmatched_idx = nhc_idx & ~test_idx

    def quick(idx: np.ndarray) -> dict[str, Any]:
        cs = [uniq["test"][i] for i in np.flatnonzero(idx)]
        y = _labels(cs)
        p = p_c["test"][idx]
        return {"n": int(len(cs)), "events": int(y.sum()), "storms": len({c["storm_id"] for c in cs}),
                "v8_2_C_auc": ev.auc(y, p) if 0 < y.sum() < len(y) else float("nan"),
                "v8_2_C_brier": ev.brier(y, p) if len(y) else float("nan")}

    coverage = {"nhc_basin_all": quick(nhc_idx), "matched": quick(test_idx), "nhc_basin_unmatched": quick(unmatched_idx)}
    coverage["nhc_basin_unmatched"]["best_track_status"] = dict(collections.Counter(
        align["test"][i]["fix"].usa_status for i in np.flatnonzero(unmatched_idx)))
    coverage["matched"]["best_track_status"] = dict(collections.Counter(
        align["test"][i]["fix"].usa_status for i in np.flatnonzero(test_idx)))

    # --- stacking, fitted on the calibration seasons only ----------------------------------------
    cal_idx = subset("calibration", (PRIMARY_TECH,))
    cal_cases = [uniq["calibration"][i] for i in np.flatnonzero(cal_idx)]
    y_cal = _labels(cal_cases)
    zc_cal = logit_of(p_c["calibration"][cal_idx], False)
    zr_cal = logit_of(noaa("calibration", cal_idx, "RIOD"), True)
    zc_test = logit_of(p_c["test"][test_idx], False)
    zr_test = logit_of(noaa("test", test_idx, "RIOD"), True)
    stacks: dict[str, Any] = {
        "fit_rows": {"n": int(len(cal_cases)), "events": int(y_cal.sum()),
                     "storms": len({c["storm_id"] for c in cal_cases}),
                     "storm_first_years": sorted({ri_model.storm_first_year(rows["calibration"])[c["storm_id"]]
                                                  for c in cal_cases})},
    }
    th_stack, d_stack = fit_stack([zc_cal, zr_cal], y_cal)
    th_crec, d_crec = fit_stack([zc_cal], y_cal)
    th_rrec, d_rrec = fit_stack([zr_cal], y_cal)
    stack_fc = {
        "stack_v8_2_C_plus_RII": (apply_stack(th_stack, [zc_test, zr_test]), False),
        "v8_2_C_recalibrated_AL_EP": (apply_stack(th_crec, [zc_test]), False),
        "SHIPS_RII_recalibrated": (apply_stack(th_rrec, [zr_test]), False),
        "v8_2_C": (p_c["test"][test_idx], False),
        "SHIPS_RII_RIOD": (noaa("test", test_idx, "RIOD"), True),
    }
    stacks["coefficients"] = {
        "stack_v8_2_C_plus_RII": {"logit_v8_2_C": float(th_stack[0]), "logit_RII": float(th_stack[1]),
                                  "intercept": float(th_stack[2]), **d_stack},
        "v8_2_C_recalibrated_AL_EP": {"logit_v8_2_C": float(th_crec[0]), "intercept": float(th_crec[1]), **d_crec},
        "SHIPS_RII_recalibrated": {"logit_RII": float(th_rrec[0]), "intercept": float(th_rrec[1]), **d_rrec},
    }
    stacks["test"] = block("test", test_idx, stack_fc, [
        ("stack_v8_2_C_plus_RII", "v8_2_C"), ("stack_v8_2_C_plus_RII", "SHIPS_RII_RIOD"),
        ("stack_v8_2_C_plus_RII", "v8_2_C_recalibrated_AL_EP"),
        ("stack_v8_2_C_plus_RII", "SHIPS_RII_recalibrated"),
        ("v8_2_C_recalibrated_AL_EP", "v8_2_C")], args.reps)
    # all component aids + v8.2, on the all-aids intersections
    cal_all = subset("calibration", ALL_AIDS_STACK)
    test_all = subset("test", ALL_AIDS_STACK)
    cols_cal = [logit_of(p_c["calibration"][cal_all], False)] + [logit_of(noaa("calibration", cal_all, t), True)
                                                                for t in ALL_AIDS_STACK]
    cols_test = [logit_of(p_c["test"][test_all], False)] + [logit_of(noaa("test", test_all, t), True)
                                                           for t in ALL_AIDS_STACK]
    y_cal_all = _labels([uniq["calibration"][i] for i in np.flatnonzero(cal_all)])
    th_all, d_all = fit_stack(cols_cal, y_cal_all)
    th_nonly, d_nonly = fit_stack(cols_cal[1:], y_cal_all)
    th_conly, _ = fit_stack(cols_cal[:1], y_cal_all)
    stacks["all_aids"] = {
        "fit_rows": {"n": int(cal_all.sum()), "events": int(y_cal_all.sum())},
        "coefficients": {"stack_v8_2_C_plus_all": dict(zip(["logit_v8_2_C", *ALL_AIDS_STACK, "intercept"],
                                                           map(float, th_all)), **d_all),
                         "stack_noaa_only": dict(zip([*ALL_AIDS_STACK, "intercept"], map(float, th_nonly)), **d_nonly)},
        "test": block("test", test_all, {
            "stack_v8_2_C_plus_all": (apply_stack(th_all, cols_test), False),
            "stack_noaa_only": (apply_stack(th_nonly, cols_test[1:]), False),
            "v8_2_C_recalibrated_AL_EP": (apply_stack(th_conly, cols_test[:1]), False),
            "v8_2_C": (p_c["test"][test_all], False),
        }, [("stack_v8_2_C_plus_all", "v8_2_C_recalibrated_AL_EP"), ("stack_v8_2_C_plus_all", "stack_noaa_only"),
            ("v8_2_C_recalibrated_AL_EP", "stack_noaa_only")], args.reps),
    }

    # --- input-source bias: what SHIPS saw vs what v8.2 read ----------------------------------------
    m_cases = [uniq["test"][i] for i in np.flatnonzero(test_idx)]
    m_align = [align["test"][i] for i in np.flatnonzero(test_idx)]
    v0_ops = np.array([(a["techs"]["RIOD"].v_final_kt - a["techs"]["RIOD"].dv_kt)
                       if a["techs"]["RIOD"].v_final_kt is not None else np.nan for a in m_align], dtype=float)
    v0_bt = np.array([c["analysis_vmax_kt"] for c in m_cases], dtype=float)
    d0 = v0_ops - v0_bt
    ok = np.isfinite(d0)
    y_m = _labels(m_cases)

    def dist_stats(d: np.ndarray) -> dict[str, float]:
        return {"n": int(len(d)), "mean": float(np.mean(d)), "median": float(np.median(d)),
                "mae": float(np.mean(np.abs(d))), "p05": float(np.quantile(d, 0.05)),
                "p95": float(np.quantile(d, 0.95)), "frac_exact": float(np.mean(d == 0)),
                "frac_abs_ge_10kt": float(np.mean(np.abs(d) >= 10))}

    # alignment check independent of the key: the e-deck's own position vs the IBTrACS fix
    km = np.array([great_circle_km(a["techs"]["RIOD"].lat, a["techs"]["RIOD"].lon, a["fix"].lat, a["fix"].lon)
                   for a in m_align], dtype=float)
    km_ok = km[np.isfinite(km)]
    position_check = {"n": int(len(km_ok)), "median_km": float(np.median(km_ok)),
                      "p99_km": float(np.quantile(km_ok, 0.99)), "max_km": float(np.max(km_ok)),
                      "n_over_150_km": int(np.sum(km_ok > 150.0))}
    bias = {"ships_initial_minus_besttrack_vmax_kt": dist_stats(d0[ok]),
            "same_on_RI_events": dist_stats(d0[ok & (y_m == 1)]),
            "same_on_non_events": dist_stats(d0[ok & (y_m == 0)]),
            "edeck_position_vs_ibtracs_fix": position_check,
            "note": ("SHIPS-RII initial intensity = e-deck V_final - dV of its 30kt/24h record; the truth is "
                     "best-track wind(t+24) - best-track wind(t), so a real-time V(t) that is too low/high "
                     "shifts RII's baseline while v8.2 is scored on the best-track baseline itself")}

    operational: dict[str, Any] = {"skipped": True}
    if not args.no_operational:
        need: dict[str, list[str]] = collections.defaultdict(list)
        for a in m_align:
            need[a["atcf_id"]].append(a["dtg"])
        op_cases, op_why = operational_cases(deck_cache, need)
        carq_cases, carq_why = operational_cases(deck_cache, need, frozenset({"OFCL"}))
        has = np.array([(a["atcf_id"], a["dtg"]) in op_cases and (a["atcf_id"], a["dtg"]) in carq_cases
                        for a in m_align])
        op_list = [op_cases[(a["atcf_id"], a["dtg"])] for a, h in zip(m_align, has) if h]
        carq_list = [carq_cases[(a["atcf_id"], a["dtg"])] for a, h in zip(m_align, has) if h]
        p_op = ri_model.score_cases(model, op_list)["calibrated"]
        p_carq = ri_model.score_cases(model, carq_list)["calibrated"]
        op_idx = test_idx.copy()
        op_idx[np.flatnonzero(test_idx)[~has]] = False
        p_bt = p_c["test"][op_idx]
        v_op = np.array([c["analysis_vmax_kt"] for c in op_list], dtype=float)
        v_bt = np.array([uniq["test"][i]["analysis_vmax_kt"] for i in np.flatnonzero(op_idx)], dtype=float)
        dv_op = np.array([np.nan if c.get("analysis_dv_24h") is None else c["analysis_dv_24h"] for c in op_list], dtype=float)
        dv_bt = np.array([np.nan if uniq["test"][i].get("analysis_dv_24h") is None else uniq["test"][i]["analysis_dv_24h"]
                          for i in np.flatnonzero(op_idx)], dtype=float)
        both = np.isfinite(dv_op) & np.isfinite(dv_bt)
        # attribution: swap ONE feature group from best-track to operational, keep the rest
        bt_list = [uniq["test"][i] for i in np.flatnonzero(op_idx)]
        y_op = _labels(bt_list)
        swap: dict[str, Any] = {"none (best track)": {"mean_forecast": float(p_bt.mean()), "auc": ev.auc(y_op, p_bt)}}
        for group, feats in FEATURE_GROUPS.items():
            mixed = [dict(b, **{f: o.get(f) for f in feats}) for b, o in zip(bt_list, op_list)]
            pm = ri_model.score_cases(model, mixed)["calibrated"]
            swap[group] = {"mean_forecast": float(pm.mean()), "auc": ev.auc(y_op, pm),
                           "delta_mean_vs_besttrack": float(pm.mean() - p_bt.mean()),
                           "delta_auc_vs_besttrack": ev.auc(y_op, pm) - ev.auc(y_op, p_bt)}
        swap["all (operational)"] = {"mean_forecast": float(p_op.mean()), "auc": ev.auc(y_op, p_op)}
        missing = {f: {"besttrack": float(np.mean([b.get(f) is None for b in bt_list])),
                       "operational_as_live": float(np.mean([o.get(f) is None for o in op_list])),
                       "operational_carq": float(np.mean([o.get(f) is None for o in carq_list]))}
                   for feats in FEATURE_GROUPS.values() for f in feats}
        age = np.array([o["storm_age_h"] - b["storm_age_h"] for b, o in zip(bt_list, op_list)
                        if o.get("storm_age_h") is not None and b.get("storm_age_h") is not None], dtype=float)
        v_carq = np.array([c["analysis_vmax_kt"] for c in carq_list], dtype=float)
        operational = {
            "skipped": False,
            "variants": {
                "v8_2_C_operational_inputs": "scripts/fetch_and_score.py build_live_case on the archived a-deck "
                                             "truncated at t: the analysis is OFCL tau 0 (as live), which "
                                             "carries no pressure",
                "v8_2_C_operational_carq_inputs": "the same builder with OFCL removed, so the analysis is CARQ "
                                                  "tau 0 -- SHIPS-RII's own real-time initial conditions, with "
                                                  "pressure: the input-source comparison without the live defect",
            },
            "built": {"as_live": dict(op_why), "carq": dict(carq_why)},
            "input_differences": {
                "vmax_operational_minus_besttrack_kt": dist_stats(v_op - v_bt),
                "vmax_carq_minus_besttrack_kt": dist_stats(v_carq - v_bt),
                "dv24_operational_minus_besttrack_kt": dist_stats((dv_op - dv_bt)[both]),
                "storm_age_operational_minus_besttrack_h": dist_stats(age),
                "missing_fraction": missing,
            },
            "one_group_swaps_as_live": swap,
            "test": block("test", op_idx, {
                "v8_2_C_besttrack_inputs": (p_bt, False),
                "v8_2_C_operational_inputs": (p_op, False),
                "v8_2_C_operational_carq_inputs": (p_carq, False),
                "SHIPS_RII_RIOD": (noaa("test", op_idx, "RIOD"), True),
            }, [("v8_2_C_besttrack_inputs", "v8_2_C_operational_inputs"),
                ("v8_2_C_besttrack_inputs", "v8_2_C_operational_carq_inputs"),
                ("v8_2_C_operational_carq_inputs", "v8_2_C_operational_inputs"),
                ("v8_2_C_operational_inputs", "SHIPS_RII_RIOD"),
                ("v8_2_C_operational_carq_inputs", "SHIPS_RII_RIOD"),
                ("v8_2_C_besttrack_inputs", "SHIPS_RII_RIOD")], args.reps),
        }

    # --- report --------------------------------------------------------------------------------
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "served-recipe hurricane RI v8.2 vs NOAA operational RI guidance on the same held-out cases",
        "origin": {k: list(v) for k, v in evr.ORIGIN.items()},
        "truth": ("evaluation label: best-track wind(t+24h) - wind(t) >= 30 kt at synoptic t; SHIPS-RII's own "
                  "threshold is the same (>= 30 kt / 24 h), so no second threshold is needed"),
        "ri_definition_selected": RI_DEFINITION,
        "ri_techs": RI_TECHS,
        "data": {
            "v8_2_sha256": ri_model.sha256_file(ri_model.DATASETS["v8.2"]),
            "edeck_source": NHC_ARCHIVE.format(year="{year}") + "e{al,ep,cp}NNYYYY.dat.gz",
            "edeck_files": len(per_file), "edeck_ri_records": len(records),
            "edeck_files_without_ri": sorted(s for s, n in edeck_files.items() if n == 0),
            "edeck_index": index_stats,
            "edeck_line_id_differs_from_file": int(sum(s.get("line_id_differs_from_file", 0) for s in per_file.values())),
            "techs_at_30kt_24h_cycles": dict(tech_counts),
            "ibtracs_wind_vs_case_wind": dict(wind_check),
            "duplicate_rows": {split: int(sum(v - 1 for v in m.values())) for split, m in mult.items()},
            "unique_cases": {split: len(u) for split, u in uniq.items()},
        },
        "model": {"v8_2_C": {**model_meta, "calibration": model["calibration"]},
                  "reproduces_evaluation": repro,
                  "served_artifact_note": ("calibration fitted on 2022-2024 = the test seasons: AUC honest "
                                           "(members <= 2021), Brier/reliability in-sample for 2 parameters")},
        "climatology": {"members_years_global": clim_global, "members_years_by_dataset_basin": clim_by_basin},
        "matching": match_tables,
        "coverage": coverage,
        "primary_intersection": primary,
        "all_noaa_aids_intersection": all_aids,
        "other_techs": extras,
        "rii_developmental_sample": rii_sample,
        "evaluation_row_weighting": weighted,
        "stacking": stacks,
        "input_source_bias": bias,
        "operational_inputs": operational,
        "bootstrap": {"unit": "storm (IBTrACS SID)", "reps": args.reps, "interval": "percentile 95%",
                      "seed": 20261002},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, EdeckRI):
            return dataclasses.asdict(o)
        raise TypeError(type(o))

    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, indent=1, allow_nan=True, default=default) + "\n")
    _print_summary(report)
    print(f"Wrote {args.out} in {report['seconds']}s")
    return 0


def _fmt(r: dict) -> str:
    ci = r["ci95"]
    return (f"AUC {r['auc']:.3f} [{ci['auc'][0]:.3f},{ci['auc'][1]:.3f}]  Brier {r['brier']:.5f} "
            f"[{ci['brier'][0]:.5f},{ci['brier'][1]:.5f}]  BSS {r['bss_vs_climatology']:+.3f} "
            f"(basin {r['bss_vs_basin_climatology']:+.3f})  mean {r['mean_forecast']:.4f} obs {r['observed_rate']:.4f}  "
            f"slope {r['calibration_slope']:.2f} [{ci['calibration_slope'][0]:.2f},{ci['calibration_slope'][1]:.2f}]  "
            f"cal {r['calibrated']}")


def _print_block(title: str, b: dict) -> None:
    print(f"\n== {title}: n={b['n']} events={b['events']} storms={b['storms']} rate={b['observed_rate']:.4f}")
    for name, r in b["forecasts"].items():
        print(f"  {name:30s} {_fmt(r)}")
    for name, d in b["paired"].items():
        ci = d["ci95"]
        print(f"  {name:52s} dAUC {d['delta_auc']:+.4f} [{ci['delta_auc'][0]:+.4f},{ci['delta_auc'][1]:+.4f}]  "
              f"dBrier {d['delta_brier']:+.5f} [{ci['delta_brier'][0]:+.5f},{ci['delta_brier'][1]:+.5f}]")


def _print_summary(rep: dict) -> None:
    print("\nmatching (test, unique cases):")
    for k, v in rep["matching"]["test"].items():
        print(f"  {k:12s} cases {v['cases']:5d} events {v['events']:4d} matched {v['matched']:5d} "
              f"(events {v['matched_events']}) storms {v['matched_storms']}/{v['storms']}  {v['reasons']}")
    _print_block("PRIMARY (v8.2 vs SHIPS-RII)", rep["primary_intersection"])
    for k, b in rep["primary_intersection"]["per_atcf_basin"].items():
        _print_block(f"PRIMARY {k}", b)
    _print_block("ALL NOAA AIDS", rep["all_noaa_aids_intersection"])
    for k, b in rep["other_techs"].items():
        _print_block(f"OTHER {k}", b)
    _print_block("RII DEVELOPMENTAL SAMPLE", rep["rii_developmental_sample"])
    _print_block("STACKING (fit on calibration seasons)", rep["stacking"]["test"])
    _print_block("STACKING ALL AIDS", rep["stacking"]["all_aids"]["test"])
    if not rep["operational_inputs"]["skipped"]:
        _print_block("OPERATIONAL INPUTS", rep["operational_inputs"]["test"])
        for g, s in rep["operational_inputs"]["one_group_swaps_as_live"].items():
            print(f"  swap {g:22s} mean {s['mean_forecast']:.4f}  AUC {s['auc']:.4f}")
    print("\ninput bias:", json.dumps(rep["input_source_bias"]["ships_initial_minus_besttrack_vmax_kt"]))
    print("weighted:", json.dumps({k: v for k, v in rep["evaluation_row_weighting"].items() if not k.startswith("paired")}))
    print("coverage:", json.dumps(rep["coverage"]))


if __name__ == "__main__":
    raise SystemExit(main())
