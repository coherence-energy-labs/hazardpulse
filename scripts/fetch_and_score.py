#!/usr/bin/env python3
"""Fetch active tropical cyclones from NHC ATCF + JTWC, score P(RI in 24 h), output JSON.

Designed to run once daily via GitHub Actions cron. Outputs:
  - dist/data/live-storms.json    (scored active storms)
  - dist/data/live-pulse.json     (updated hurricane entry)

If no active storms exist, outputs empty arrays with honest timestamps.

The model is NOT trained here. It is served from the pinned artifact named by
SERVED_MODEL_VERSION (results/models/<version>.json, scripts/train_hurricane_ri.py), bound
to its data files' SHA-256 and config, with a calibration fitted to convergence on held-out
seasons. Training in this job is what kept every run since 2026-05-27 from finishing inside
the workflow's 10-minute budget; which version is served is decided by
scripts/evaluate_hurricane_ri.py.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_site_artifacts import build_site_artifacts

from hazardpulse.data.http import fetch_bytes, fetch_text  # noqa: E402
from hazardpulse.hurricane.atcf import (  # noqa: E402
    ATCF_ROOT,
    ATCFRecord,
    DEFAULT_AID_MODELS,
    DEFAULT_ANALYSIS_PRIORITY,
    DEFAULT_LEAD_TIMES,
    parse_atcf_deck,
)
from hazardpulse.hurricane.operational_ri import (  # noqa: E402
    climatological_mpi_features,
    translation_speed_kmh,
)
from hazardpulse.hurricane import ri_model  # noqa: E402

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
HISTORICAL_DATA = RESULTS / "hurricane_operational_ri_2000_2024_al_sst.jsonl"
# The served model, as chosen by scripts/evaluate_hurricane_ri.py's pre-registered rule
# (results/calibration/hurricane_ri_evaluation.json; a test keeps the two in agreement).
SERVED_MODEL_VERSION = "hurricane_ri_v8_2"
MODEL_ARTIFACT = ri_model.ARTIFACTS[SERVED_MODEL_VERSION]
PRIMARY_DOMAIN = "https://hazardpulse.com"

# NHC real-time ATCF data URLs
REALTIME_ADECK_INDEX = f"{ATCF_ROOT}/aid_public/"
REALTIME_ADECK_FILE_RE = re.compile(
    r'href="(a[a-z]{2}\d{2}\d{4}\.dat\.gz)"', re.IGNORECASE
)
# Same listing, with the server's "last modified" column (UTC) when it is present.
REALTIME_ADECK_ROW_RE = re.compile(
    r'href="(a[a-z]{2}\d{2}\d{4}\.dat\.gz)"[^\n]*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})',
    re.IGNORECASE,
)

# aid_public keeps EVERY a-deck of the season -- 53 files on 2026-10-02, of which three
# had an analysis in the last 12 h; AL01 was last analysed 106 days earlier. A storm is
# active only if its latest analysis cycle is this recent. NHC cycles every 6 h, so 12 h
# tolerates one late or missed cycle; two missed cycles means NHC is no longer tracking it.
ACTIVE_MAX_AGE_HOURS = 12.0

# All global basins — NHC ATCF hosts a-decks for NHC basins; JTWC storms
# are discovered from the JTWC RSS feed and parsed from warning text.
ALL_BASINS = ("AL", "EP", "CP", "WP", "IO", "SH")
# Basins NHC/CPHC warn on: their a-decks are authoritative, and JTWC's warnings for them
# are re-issues that would only duplicate the storm (under the wrong basin, before 2026-10).
NHC_BASINS = frozenset({"AL", "EP", "CP"})

# JTWC public RSS feed listing active tropical cyclone warnings
JTWC_RSS_URL = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"
# JTWC warning text products (public, no auth required)
JTWC_WARNING_URL = "https://www.metoc.navy.mil/jtwc/products/{product_id}web.txt"
# Regex to extract storm product IDs from JTWC RSS
JTWC_PRODUCT_RE = re.compile(r"products/(\w+)web\.txt", re.IGNORECASE)
# Warning products are "<basin><NN><YY>": wp2626, ep1526, sh0126, io0226. Anything else on
# the feed (abpw, abio: area advisories) is not a single-storm warning.
JTWC_WARNING_PRODUCT_RE = re.compile(r"^(wp|io|sh|ep|cp)(\d{2})(\d{2})$", re.IGNORECASE)
# The storm's identity comes from the SUBJ header ONLY. The body routinely names OTHER
# storms ("REFER TO TROPICAL STORM 15E (NOLO) WARNINGS"), and the old whole-text search
# published Hurricane Rachel's 100 kt / 19.4N 109.4W under Nolo's name (2026-10-02).
JTWC_SUBJ_RE = re.compile(
    r"^SUBJ/(?P<kind>[A-Z][A-Z -]*?)\s+(?P<num>\d{2}[A-Z])\s+\((?P<name>[^)]+)\)\s+WARNING",
    re.IGNORECASE | re.MULTILINE,
)
# A final warning on a system that is no longer a tropical cyclone.
JTWC_NOT_A_TC_RE = re.compile(r"REMNANTS|POST-TROPICAL|EXTRATROPICAL|DISSIPATED", re.IGNORECASE)
JTWC_SUFFIX_BASIN = {"W": "WP", "A": "IO", "B": "IO", "S": "SH", "P": "SH", "E": "EP", "C": "CP"}

# Every model feature is built live by build_live_case() with the v8.2 training set's own
# definitions (true 6/12/24-h deltas, displacement over the real 6 h, hours since the first
# analysis). A model whose training set cannot be matched live declares the features to
# median-impute in its artifact (``serving.impute_live``: v8.1's 3-hourly-stepped speed and
# age), and ri_model applies that contract at scoring time -- offline evaluation included.
# Features a SOURCE cannot provide (a single JTWC warning has no history: no lagged deltas,
# no motion, no age) arrive as None and are imputed the same way; a test checks that an
# a-deck case leaves nothing undeclared.

# Category thresholds (Saffir-Simpson)
CATEGORY_THRESHOLDS = [
    (137, "Category 5"),
    (113, "Category 4"),
    (96, "Category 3"),
    (83, "Category 2"),
    (64, "Category 1"),
    (34, "Tropical Storm"),
    (0, "Tropical Depression"),
]


def classify_storm(vmax_kt: float | None) -> str:
    if vmax_kt is None:
        return "Unknown"
    for threshold, label in CATEGORY_THRESHOLDS:
        if vmax_kt >= threshold:
            return label
    return "Tropical Disturbance"


def _discover_jtwc_storms() -> dict[str, list[ATCFRecord]]:
    """Discover active JTWC storms from RSS feed and parse warning text.

    Returns dict mapping storm_id to list of ATCFRecords (analysis + forecasts),
    ready for build_live_case() and the RI scoring pipeline.
    """
    try:
        rss = fetch_text(JTWC_RSS_URL, namespace="jtwc_rss", use_cache=False)
    except Exception as e:
        print(f"  Warning: Could not fetch JTWC RSS: {e}")
        return {}

    # Structural validation — distinguish "no active storms" from "RSS format changed"
    if "<rss" not in rss.lower() and "<feed" not in rss.lower():
        print(
            "  WARNING: JTWC RSS returned non-RSS response "
            f"(first 200 bytes: {rss[:200]!r}). Format may have changed."
        )
        return {}
    if "<item" not in rss.lower() and "<entry" not in rss.lower():
        print(
            "  JTWC RSS reachable but contains no <item>/<entry> tags — "
            "treating as no-active-storms state."
        )
        return {}

    product_ids = JTWC_PRODUCT_RE.findall(rss)
    if not product_ids:
        # Items exist but our regex matched nothing — likely a schema change.
        print(
            "  WARNING: JTWC RSS contains items but product regex "
            f"{JTWC_PRODUCT_RE.pattern!r} matched zero IDs. Regex may need update."
        )
        return {}

    result: dict[str, list[ATCFRecord]] = {}
    for pid in product_ids:
        pm = JTWC_WARNING_PRODUCT_RE.match(pid)
        if pm is None:
            continue  # area advisories (abpw, abio), not single-storm warnings
        if pm.group(1).upper() in NHC_BASINS:
            # JTWC re-issues NHC/CPHC warnings for EP/CP storms; the NHC a-deck is the
            # authority and already carries the storm.
            continue
        url = JTWC_WARNING_URL.format(product_id=pid)
        try:
            text = fetch_text(url, namespace="jtwc_warning", use_cache=False)
        except Exception:
            continue

        storm_id, records = _parse_jtwc_warning_to_atcf(text, product_id=pid)
        if storm_id and records:
            if storm_id in result:
                print(f"  WARNING: two JTWC warnings resolve to {storm_id}; keeping the first")
                continue
            result[storm_id] = records

    return result


def _jtwc_day_to_datetime(ddhhmm: str, now: dt.datetime) -> dt.datetime | None:
    """Resolve a JTWC ``DDHHMM`` group to the most recent such time not after ``now`` (+1 day).

    A warning carries only day-of-month: at a month boundary (warning 302100Z read on the
    1st) the naive ``now.month`` reading lands 29 days in the future.
    """
    day, hour, minute = int(ddhhmm[:2]), int(ddhhmm[2:4]), int(ddhhmm[4:6])
    year, month = now.year, now.month
    for _ in range(3):
        try:
            candidate = dt.datetime(year, month, day, hour, minute)
        except ValueError:
            candidate = None
        if candidate is not None and candidate <= now + dt.timedelta(days=1):
            return candidate
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return None


def _parse_jtwc_warning_to_atcf(
    text: str,
    product_id: str | None = None,
    now: dt.datetime | None = None,
) -> tuple[str, list[ATCFRecord]]:
    """Parse a JTWC warning text product into ATCFRecords.

    Extracts analysis position + all forecast lead times (12h through 120h)
    and returns them as ATCFRecord objects compatible with build_live_case().

    Identity (number, name) is read from the ``SUBJ/`` header only; the basin from the
    product id (``wp2626`` -> WP) when given, else from the number's suffix letter. Final
    warnings on systems that are no longer tropical cyclones ("REMNANTS OF ...") and
    warnings whose basin cannot be determined yield no storm.
    """
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    lines = text.strip().split("\n")

    subj = JTWC_SUBJ_RE.search(text)
    if subj is None:
        return "", []
    if JTWC_NOT_A_TC_RE.search(subj.group("kind")):
        return "", []
    storm_number = subj.group("num").upper()
    storm_name = subj.group("name").strip().title()

    basin = None
    if product_id:
        pm = JTWC_WARNING_PRODUCT_RE.match(product_id)
        if pm is not None:
            basin = pm.group(1).upper()
    if basin is None:
        basin = JTWC_SUFFIX_BASIN.get(storm_number[-1])
    if basin is None:
        return "", []
    year = now.year
    storm_id = f"{basin}{storm_number[:2]}{year}"

    # Parse analysis position: "DDHHMM Z --- NEAR NN.NN NS.S EEE.EE EW"
    cycle_time = None
    analysis_lat = analysis_lon = analysis_vmax = None

    for i, line in enumerate(lines):
        m = re.search(r"(\d{6})Z\s*---\s*NEAR\s+([\d.]+)([NS])\s+([\d.]+)([EW])", line)
        if m:
            ddhhmm = m.group(1)
            cycle_time = _jtwc_day_to_datetime(ddhhmm, now)
            if cycle_time is None:
                continue
            analysis_lat = float(m.group(2)) * (-1 if m.group(3) == "S" else 1)
            analysis_lon = float(m.group(4)) * (-1 if m.group(5) == "W" else 1)
            break

    if cycle_time is None or analysis_lat is None:
        return "", []

    # Get analysis max wind (first MAX SUSTAINED WINDS)
    for line in lines:
        m = re.search(r"MAX SUSTAINED WINDS\s*-\s*(\d+)\s*KT", line)
        if m:
            analysis_vmax = float(m.group(1))
            break

    records: list[ATCFRecord] = []

    # Analysis record (tau=0)
    records.append(ATCFRecord(
        basin=basin, storm_number=int(storm_number[:2]), cycle=cycle_time,
        tau_hours=0, model="JTWC", lat=analysis_lat, lon=analysis_lon,
        vmax_kt=analysis_vmax, mslp_hpa=None, storm_name=storm_name,
    ))

    # Parse forecast positions: "NN HRS, VALID AT:" followed by position and winds
    forecast_re = re.compile(r"(\d+)\s+HRS?,\s*VALID AT:", re.IGNORECASE)
    pos_re = re.compile(r"(\d{6})Z\s*---\s*([\d.]+)([NS])\s+([\d.]+)([EW])")
    wind_re = re.compile(r"MAX SUSTAINED WINDS\s*-\s*(\d+)\s*KT")

    i = 0
    while i < len(lines):
        fm = forecast_re.search(lines[i])
        if fm:
            tau = int(fm.group(1))
            # Next lines: position, then winds
            fc_lat = fc_lon = fc_vmax = None
            for j in range(i + 1, min(i + 5, len(lines))):
                pm = pos_re.search(lines[j])
                if pm:
                    fc_lat = float(pm.group(2)) * (-1 if pm.group(3) == "S" else 1)
                    fc_lon = float(pm.group(4)) * (-1 if pm.group(5) == "W" else 1)
                wm = wind_re.search(lines[j])
                if wm:
                    fc_vmax = float(wm.group(1))

            if fc_lat is not None:
                records.append(ATCFRecord(
                    basin=basin, storm_number=int(storm_number[:2]),
                    cycle=cycle_time, tau_hours=tau, model="JTWC",
                    lat=fc_lat, lon=fc_lon, vmax_kt=fc_vmax, mslp_hpa=None,
                    storm_name=storm_name,
                ))
        i += 1

    return storm_id, records


def fetch_adeck_index(
    basin_prefixes: tuple[str, ...] = ALL_BASINS,
) -> dict[str, dt.datetime | None]:
    """Every a-deck in the NHC ATCF real-time index, with its last-modified time.

    The index holds the WHOLE season's a-decks, dissipated storms and old invests
    included; whether a storm is active is decided from its deck (is_active()).
    """
    try:
        html = fetch_text(
            REALTIME_ADECK_INDEX,
            namespace="atcf_realtime",
            use_cache=False,
        )
    except Exception as e:
        print(f"  Warning: Could not fetch ATCF real-time index: {e}")
        return {}
    return parse_adeck_index(html, basin_prefixes)


def list_realtime_storms(
    basin_prefixes: tuple[str, ...] = ALL_BASINS,
) -> list[str]:
    """Storm IDs of every a-deck in the NHC ATCF real-time index (active OR NOT).

    The NHC ATCF server at ftp.nhc.noaa.gov hosts a-deck files for all global
    basins including JTWC-tracked Western Pacific (WP), Indian Ocean (IO),
    and Southern Hemisphere (SH) storms. Filter with is_active() before scoring.
    """
    return sorted(fetch_adeck_index(basin_prefixes))


def parse_adeck_index(
    html: str,
    basin_prefixes: tuple[str, ...] = ALL_BASINS,
) -> dict[str, dt.datetime | None]:
    """``{storm_id: last-modified (UTC, naive) or None}`` for every a-deck in the listing."""
    wanted = {p.upper() for p in basin_prefixes}
    out: dict[str, dt.datetime | None] = {}
    for filename in REALTIME_ADECK_FILE_RE.findall(html):
        storm_id = filename[1:9].upper()
        if storm_id[:2] in wanted:
            out.setdefault(storm_id, None)
    for filename, stamp in REALTIME_ADECK_ROW_RE.findall(html):
        storm_id = filename[1:9].upper()
        if storm_id in out:
            try:
                out[storm_id] = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M")
            except ValueError:
                pass
    return out


def latest_analysis_cycle(
    records: list[ATCFRecord],
    analysis_priority: tuple[str, ...] = DEFAULT_ANALYSIS_PRIORITY,
) -> dt.datetime | None:
    """The most recent cycle with an analysis build_live_case() would use (tau 0, a wind)."""
    wanted = set(analysis_priority)
    cycles = [
        r.cycle for r in records
        if r.tau_hours == 0 and r.vmax_kt is not None and r.model in wanted
    ]
    return max(cycles) if cycles else None


def is_active(
    records: list[ATCFRecord],
    now: dt.datetime,
    max_age_hours: float = ACTIVE_MAX_AGE_HOURS,
) -> tuple[bool, float | None]:
    """(active?, age of the latest analysis in hours). No analysis at all is inactive."""
    latest = latest_analysis_cycle(records)
    if latest is None:
        return False, None
    age_h = (now - latest).total_seconds() / 3600.0
    return age_h <= max_age_hours, age_h


def fetch_realtime_adeck(storm_id: str) -> list[ATCFRecord]:
    """Fetch real-time a-deck data for an active storm (all basins via NHC ATCF)."""

    filename = f"a{storm_id.lower()}.dat.gz"
    url = f"{REALTIME_ADECK_INDEX}{filename}"
    try:
        data = fetch_bytes(url, namespace="atcf_realtime", use_cache=False)
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        return parse_atcf_deck(text)
    except Exception as e:
        print(f"  Warning: Could not fetch a-deck for {storm_id}: {e}")
        return []


def _select_analysis_record(
    by_model_tau: dict[tuple[str, int], ATCFRecord],
    analysis_priority: tuple[str, ...] = DEFAULT_ANALYSIS_PRIORITY,
) -> ATCFRecord | None:
    for model in analysis_priority:
        record = by_model_tau.get((model, 0))
        if record is not None and record.vmax_kt is not None:
            return record
    return None


def build_live_case(
    storm_id: str,
    adeck_records: list[ATCFRecord],
    *,
    aid_models: tuple[str, ...] = DEFAULT_AID_MODELS,
    lead_times: tuple[int, ...] = DEFAULT_LEAD_TIMES,
    full_history: bool = True,
) -> dict[str, object] | None:
    """Build a single feature case from the LATEST advisory cycle of an active storm.

    Unlike build_operational_ri_cases, this doesn't need b-deck/best-track
    because we're scoring a live storm, not evaluating historical accuracy.

    ``full_history``: the records hold the storm's whole track (an a-deck). A JTWC warning
    holds one cycle, so its storm age is unknown (None, not 0).
    """

    if not adeck_records:
        return None

    # Find the latest cycle with analysis data
    cycles = sorted(
        {r.cycle for r in adeck_records if r.tau_hours == 0},
        reverse=True,
    )

    for cycle in cycles:
        by_model_tau = {
            (r.model, r.tau_hours): r
            for r in adeck_records
            if r.cycle == cycle
        }
        analysis = _select_analysis_record(by_model_tau)
        if analysis is None or analysis.vmax_kt is None:
            continue

        current_vmax = analysis.vmax_kt
        current_mslp = analysis.mslp_hpa

        # Build the case dict — same structure as historical cases
        row: dict[str, object] = {
            "storm_id": storm_id.upper(),
            "season_year": int(storm_id[-4:]),
            "issue_time": cycle.isoformat(),
            "basin": analysis.basin,
            "storm_number": analysis.storm_number,
            "storm_name": analysis.storm_name or storm_id,
            "analysis_model": analysis.model,
            "analysis_lat": analysis.lat,
            "analysis_lon": analysis.lon,
            "analysis_vmax_kt": current_vmax,
            "analysis_mslp_hpa": current_mslp,
            # No ri_label — this is a live case, truth not yet known
            "ri_label_30kt": 0,  # placeholder, not used for scoring
        }

        # Previous cycles for delta features
        for lag_hours, field_suffix in ((6, "6h"), (12, "12h"), (24, "24h")):
            lag_cycle = cycle - dt.timedelta(hours=lag_hours)
            lag_map = {
                (r.model, r.tau_hours): r
                for r in adeck_records
                if r.cycle == lag_cycle
            }
            prev = _select_analysis_record(lag_map)
            if prev is not None and prev.vmax_kt is not None:
                row[f"analysis_dv_{field_suffix}"] = current_vmax - prev.vmax_kt
            else:
                row[f"analysis_dv_{field_suffix}"] = None
            if (
                prev is not None
                and current_mslp is not None
                and prev.mslp_hpa is not None
            ):
                row[f"analysis_dp_{field_suffix}"] = current_mslp - prev.mslp_hpa
            else:
                row[f"analysis_dp_{field_suffix}"] = None

        # Aid model features
        available_aids = 0
        dv_by_lead: dict[int, list[float]] = {lt: [] for lt in lead_times}

        for aid in aid_models:
            for lead in lead_times:
                key_root = f"aid_{aid.lower()}_{lead}h"
                aid_record = by_model_tau.get((aid, lead))
                if aid_record is None or aid_record.vmax_kt is None:
                    row[f"{key_root}_available"] = 0
                    row[f"{key_root}_vmax_kt"] = None
                    row[f"{key_root}_dv_kt"] = None
                    row[f"{key_root}_mslp_hpa"] = None
                    continue
                available_aids += 1
                dv = float(aid_record.vmax_kt - current_vmax)
                row[f"{key_root}_available"] = 1
                row[f"{key_root}_vmax_kt"] = aid_record.vmax_kt
                row[f"{key_root}_dv_kt"] = dv
                row[f"{key_root}_mslp_hpa"] = aid_record.mslp_hpa
                dv_by_lead[lead].append(dv)

        row["available_aid_forecasts"] = available_aids

        # Consensus features per lead time
        for lead in lead_times:
            prefix = f"aid_consensus_{lead}h"
            dvs = dv_by_lead[lead]
            if dvs:
                row[f"{prefix}_dv_mean"] = float(np.mean(dvs))
                row[f"{prefix}_dv_min"] = float(np.min(dvs))
                row[f"{prefix}_dv_max"] = float(np.max(dvs))
                row[f"{prefix}_dv_range"] = float(np.max(dvs) - np.min(dvs))
            else:
                row[f"{prefix}_dv_mean"] = None
                row[f"{prefix}_dv_min"] = None
                row[f"{prefix}_dv_max"] = None
                row[f"{prefix}_dv_range"] = None

        # OFCL-specific features
        ofcl_24 = by_model_tau.get(("OFCL", 24))
        if ofcl_24 is not None and ofcl_24.vmax_kt is not None:
            row["aid_ofcl_24h_dv_kt"] = float(ofcl_24.vmax_kt - current_vmax)
        consensus_24_mean = row.get("aid_consensus_24h_dv_mean")
        ofcl_24_dv = row.get("aid_ofcl_24h_dv_kt")
        if consensus_24_mean is not None and ofcl_24_dv is not None:
            row["aid_ofcl_minus_consensus_24h"] = float(ofcl_24_dv) - float(
                consensus_24_mean
            )
        else:
            row["aid_ofcl_minus_consensus_24h"] = None

        # Month features
        month = cycle.month
        row["issue_month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        row["issue_month_cos"] = float(np.cos(2 * np.pi * month / 12.0))

        # Location / potential-intensity features, by the training set's own definition
        # (until 2026-10 these were never set live and were median-imputed).
        row.update(climatological_mpi_features(analysis.lat, current_vmax, month))

        # Motion and age on the REAL clock, as the v8.2 training set defines them.
        prev6 = _select_analysis_record({
            (r.model, r.tau_hours): r for r in adeck_records
            if r.cycle == cycle - dt.timedelta(hours=6)
        })
        row["translation_speed_kmh"] = (
            translation_speed_kmh(prev6.lat, prev6.lon, analysis.lat, analysis.lon, 6.0)
            if prev6 is not None else None
        )
        if full_history:
            first_cycle = min(r.cycle for r in adeck_records if r.tau_hours == 0)
            row["storm_age_h"] = (cycle - first_cycle).total_seconds() / 3600.0
        else:
            row["storm_age_h"] = None

        return row

    return None


def score_live_cases(
    model: dict[str, object],
    live_cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Score live cases with the pinned artifact. No training happens here.

    ``ri_probability`` is the artifact's calibrated probability; every downstream band
    (pulse risk_band, page) is derived from it, never from a raw member score.
    """

    if not live_cases:
        return []
    p = ri_model.score_cases(model, live_cases)
    n_features = len(model["selected_idx"])

    scored: list[dict[str, object]] = []
    for i, case in enumerate(live_cases):
        scored.append({
            "storm_id": case["storm_id"],
            "storm_name": case.get("storm_name", case["storm_id"]),
            "basin": case.get("basin", ""),
            "lat": case.get("analysis_lat"),
            "lon": case.get("analysis_lon"),
            "vmax_kt": case.get("analysis_vmax_kt"),
            "mslp_hpa": case.get("analysis_mslp_hpa"),
            "category": classify_storm(case.get("analysis_vmax_kt")),
            "issue_time": case.get("issue_time"),
            "ri_probability": round(float(p["calibrated"][i]), 4),
            "ri_probability_raw": round(float(p["ensemble"][i]), 4),
            "model_scores": {
                "gbt_d3": round(float(p["gbt_d3"][i]), 4),
                "gbt_d4": round(float(p["gbt_d4"][i]), 4),
                "logistic": round(float(p["logistic"][i]), 4),
                "bagged": round(float(p["bagged"][i]), 4),
            },
            "n_features": n_features,
            "model_version": model["model_version"],
            "calibration": model["calibration"]["method"],
        })

    return scored


def _render_hurricane_geojson(storms: list[dict[str, object]]) -> str:
    """Render scored hurricanes as GeoJSON FeatureCollection for MapLibre."""
    features = []
    for s in storms:
        lon = s.get("lon")
        lat = s.get("lat")
        if lon is None or lat is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "storm_id": s.get("storm_id", ""),
                "storm_name": s.get("storm_name", ""),
                "ri_probability": s.get("ri_probability", 0),
                "category": s.get("category", ""),
                "vmax_kt": s.get("vmax_kt"),
                "mslp_hpa": s.get("mslp_hpa"),
                "basin": s.get("basin", ""),
            },
        })
    return json.dumps({"type": "FeatureCollection", "features": features})


def write_outputs(
    scored_storms: list[dict[str, object]],
    now: dt.datetime,
    model_version: str = SERVED_MODEL_VERSION,
) -> None:
    """Write scored results to dist/data/."""
    forecast_id = f"hu_fcst_{now.strftime('%Y%m%d_%H%M')}"

    # Write live-storms.json
    output = {
        "updated_at": now.isoformat() + "Z",
        "forecast_id": forecast_id,
        "model_version": model_version,
        "n_active_storms": len(scored_storms),
        "storms": scored_storms,
    }
    storms_path = DIST / "data" / "live-storms.json"
    storms_path.parent.mkdir(parents=True, exist_ok=True)
    storms_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {storms_path} ({len(scored_storms)} storms)")

    # Write GeoJSON for MapLibre hurricane map
    geojson_path = DIST / "data" / "hurricane-storms.geojson"
    geojson_path.write_text(
        _render_hurricane_geojson(scored_storms) + "\n", encoding="utf-8",
    )
    print(f"  Wrote {geojson_path} ({len(scored_storms)} features)")

    # Write replay artifact (required by site integrity tests)
    replay_path = DIST / "data" / "replay" / f"{forecast_id}.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_payload = {
        "forecast_id": forecast_id,
        "issued_at": now.isoformat() + "Z",
        "model_version": model_version,
        "hazard": "hurricane",
        "n_active_storms": len(scored_storms),
        "storms": scored_storms,
    }
    replay_path.write_text(json.dumps(replay_payload, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {replay_path}")

    # Update live-pulse.json hurricane entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "hu":
                # Benchmark fields left in the pulse by earlier model versions (v8.1's
                # "model_auc": 0.938 under its 12-h label, "n_features": 65) would ride along
                # beside a different model. The served model's held-out metrics live on the
                # verification surface (results/calibration/hurricane_ri_evaluation.json).
                hazard.pop("model_auc", None)
                hazard.pop("n_features", None)
                if scored_storms:
                    # Use highest RI probability storm
                    top = max(scored_storms, key=lambda s: s["ri_probability"])
                    hazard["probability"] = top["ri_probability"]
                    # Real uncertainty band from the calibrator (None until one exists).
                    hazard["conf_lo"] = top.get("confidence_lo")
                    hazard["conf_hi"] = top.get("confidence_hi")
                    hazard["uncertainty_class"] = top.get("uncertainty_class")
                    hazard["abstained"] = top.get("abstained", False)
                    hazard["receipt_sha256"] = top.get("receipt_sha256")
                    hazard["risk_band"] = _risk_band(top["ri_probability"])
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = model_version
                    hazard["forecast_id"] = forecast_id
                    hazard["n_active_storms"] = len(scored_storms)
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "none"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = model_version
                    hazard["forecast_id"] = forecast_id
                    hazard["n_active_storms"] = 0
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(json.dumps(pulse, indent=2) + "\n", encoding="utf-8")
        print(f"  Updated {pulse_path}")


def _risk_band(prob: float) -> str:
    if prob >= 0.50:
        return "critical"
    if prob >= 0.30:
        return "elevated"
    if prob >= 0.15:
        return "watch"
    if prob >= 0.05:
        return "guarded"
    return "none"


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _band_text(lo: object, hi: object) -> str:
    """Calibrated 90% band, e.g. ' (90% band: 8.0%-18.0%)'; empty when unavailable."""
    if lo is None or hi is None:
        return ""
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return ""
    if lo_f != lo_f or hi_f != hi_f:   # NaN
        return ""
    return f" (90% band: {lo_f * 100:.1f}%-{hi_f * 100:.1f}%)"


def _format_time(ts: dt.datetime) -> str:
    return ts.strftime("%a, %d %b %Y %H:%M:%S UTC")


def render_hurricane_page(
    scored_storms: list[dict[str, object]],
    now: dt.datetime,
    model_version: str = SERVED_MODEL_VERSION,
) -> None:
    """Render an honest hurricane live page from the current saved feed."""
    page_path = DIST / "live" / "hurricane" / "index.html"
    page_path.parent.mkdir(parents=True, exist_ok=True)

    if scored_storms:
        rows = []
        for storm in sorted(
            scored_storms,
            key=lambda item: float(item.get("ri_probability", 0) or 0),
            reverse=True,
        )[:8]:
            rows.append(
                f"<tr>"
                f"<td>{_esc(storm.get('storm_name', storm.get('storm_id', 'Storm')))}</td>"
                f"<td>{_esc(storm.get('category', '--'))}</td>"
                f"<td>{storm.get('lat', '--')}, {storm.get('lon', '--')}</td>"
                f"<td>{float(storm.get('ri_probability', 0) or 0) * 100:.1f}%</td>"
                f"<td>{storm.get('vmax_kt', '--')} kt</td>"
                f"</tr>"
            )
        storms_html = (
            "<table><thead><tr><th>Storm</th><th>Status</th><th>Location</th><th>RI 24h</th><th>Wind</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        summary_html = (
            f"<div class=\"card hazard-hu\">"
            f"<h2 style=\"margin-top:0;\">Top storm</h2>"
            f"<div class=\"metric\">{float(scored_storms[0].get('ri_probability', 0) or 0) * 100:.1f}%</div>"
            f"<div class=\"metric-label\">Rapid intensification in 24h"
            f"{_band_text(scored_storms[0].get('confidence_lo'), scored_storms[0].get('confidence_hi'))}</div>"
            f"<div class=\"kv\"><span>Name</span><strong>{_esc(scored_storms[0].get('storm_name', scored_storms[0].get('storm_id', 'Storm')))}</strong></div>"
            f"<div class=\"kv\"><span>Status</span><strong>{_esc(scored_storms[0].get('category', '--'))} &middot; {scored_storms[0].get('vmax_kt', '--')} kt</strong></div>"
            f"</div>"
        )
    else:
        storms_html = (
            '<div class="card"><p class="muted" style="margin:0;">'
            "No active tropical cyclones are present in the current feed."
            "</p></div>"
        )
        summary_html = (
            '<div class="card hazard-hu"><h2 style="margin-top:0;">Current state</h2>'
            '<div class="metric">0.0%</div>'
            '<div class="metric-label">Rapid intensification in 24h</div>'
            '<p class="muted">No active tropical cyclones are present in the current feed.</p>'
            "</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Hurricane Forecasts - HazardPulse</title>
  <meta name="description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="{PRIMARY_DOMAIN}/live/hurricane/">
  <script src="/assets/site-shell.js?v=2"></script>
  <link rel="stylesheet" href="/assets/styles.css?v=10">
  <link rel="stylesheet" href="/assets/vendor/maplibre-gl.css">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta property="og:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <meta property="og:url" content="{PRIMARY_DOMAIN}/live/hurricane/">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Live Hurricane Forecasts - HazardPulse">
  <meta name="twitter:description" content="Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output.">
  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Live Hurricane Forecasts",
    "url": "{PRIMARY_DOMAIN}/live/hurricane/",
    "description": "Static live hurricane page built from the current tropical cyclone feed and HazardPulse rapid-intensification model output."
  }}
  </script>
  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/", "/live/", "/live/earthquake/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>
  <div class="live-bar"></div>
  <div class="emergency-banner" role="alert" aria-live="assertive"></div>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>
  <main id="main" class="container">
    <section class="hero">
      <div class="eyebrow">Hurricane live page</div>
      <h1>Current tropical cyclone view</h1>
      <p class="subtitle">
        This page is rendered from the current tropical cyclone feed. When there are no active storms,
        it says so plainly instead of showing synthetic examples.
      </p>
      <p class="muted">Updated {_esc(_format_time(now))} &middot; Model: {_esc(model_version)} &middot; Independent hazard intelligence platform</p>
    </section>

    <section class="section">
      <div class="card" style="margin-bottom:var(--space-lg);">
        <h2 style="margin-top:0;">Global tropical cyclone map</h2>
        <div id="hurricane-map"
             data-geojson-src="/data/hurricane-storms.geojson"
             style="width:100%;height:420px;border-radius:var(--radius);overflow:hidden;"
             role="img"
             aria-label="Interactive map showing active tropical cyclones across all ocean basins">
        </div>
        <p class="muted" style="margin-top:8px;">Coverage: Atlantic, East Pacific, West Pacific, Indian Ocean, Southern Hemisphere. Data: NHC ATCF + JTWC. Markers sized by wind speed, colored by Saffir-Simpson category.</p>
      </div>
      <div class="grid">
        <div class="col-4">
          {summary_html}
        </div>
        <div class="card col-8">
          <h2 style="margin-top:0;">Active tropical systems</h2>
          {storms_html}
          <p class="muted" style="margin-top:12px;">Source feed: <a href="/data/live-storms.json">/data/live-storms.json</a> &middot; GeoJSON: <a href="/data/hurricane-storms.geojson">/data/hurricane-storms.geojson</a>. Always follow official advisories from the NHC, JTWC, and local authorities.</p>
        </div>
      </div>
    </section>
  </main>
  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="/legal/privacy/">Privacy</a>
        <a href="/legal/terms/">Terms</a>
        <a href="/COMMERCIAL_LICENSE.md">Commercial License</a>
      </div>
      <p class="footer-disclaimer">
        Independent hazard intelligence platform. Always follow official guidance from the NHC, JTWC, WMO RSMCs, and local emergency authorities.
      </p>
      <p class="footer-build">Static-first HTML &middot; Live data under <code>/data</code> &middot; Edge geolocation by Cloudflare</p>
    </div>
  </footer>
  <script src="/assets/vendor/maplibre-gl.js"></script>
  <script src="/assets/hurricane-monitor.js?v=1" defer></script>
</body>
</html>
"""
    page_path.write_text(html, encoding="utf-8")
    print(f"  Wrote {page_path} ({len(scored_storms)} storms, honest state)")


def load_serving_model() -> dict[str, object]:
    """The pinned served artifact, verified against its data files, config and calibration.

    Loaded on EVERY run, storms or not, so a missing or stale artifact fails the job the
    same day instead of hiding until the first storm of the season. Refuses anything but a
    held-out, converged calibration: the legacy in-sample Platt is never served again.
    """
    model = ri_model.load_model(MODEL_ARTIFACT)
    if model["model_version"] != SERVED_MODEL_VERSION:
        raise ri_model.ModelArtifactError(
            f"{MODEL_ARTIFACT} holds {model['model_version']}, expected {SERVED_MODEL_VERSION}")
    if model["calibration"]["method"] != "logistic_on_logit_newton":
        raise ri_model.ModelArtifactError(
            f"refusing to serve calibration {model['calibration']['method']!r}: only a held-out "
            "converged calibration (logistic_on_logit_newton) may be published")
    prov = model.get("provenance", {})
    cal = model["calibration"]
    data = prov.get("data", {})
    print(
        f"  Model {model['model_version']}: {len(model['selected_idx'])} features, "
        f"{len(model['gbt_d3']['trees'])}+{len(model['gbt_d4']['trees'])} trees, "
        f"{len(model['bagged'])} bags; members {data.get('members', {}).get('storm_years')}, "
        f"calibrated on {cal.get('fitted_on', {}).get('storm_years')} "
        f"({cal['n']} rows, a={cal['a']:.3f}, b={cal['b']:.3f}); trained {prov.get('trained_at', '?')}"
    )
    return model


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    print(f"HazardPulse hurricane RI scoring pipeline ({SERVED_MODEL_VERSION}) — {now.isoformat()}Z")
    print()

    print("Step 0: Loading the pinned model artifact...")
    model = load_serving_model()
    print()

    # Step 1: Check NHC ATCF for active storms
    print("Step 1: Checking NHC ATCF for active tropical cyclones...")
    adeck_index = fetch_adeck_index()
    active_ids = sorted(adeck_index)
    if active_ids:
        print(f"  {len(active_ids)} a-decks in the ATCF index (the whole season's, active or not)")
    else:
        # In peak season (Jun-Nov Atlantic, year-round WP), zero storms is
        # unusual and worth flagging loudly in case the feed rotted.
        month = now.month
        is_peak_atlantic = 6 <= month <= 11
        if is_peak_atlantic:
            print(
                "  WARNING: NHC ATCF returned zero storms during peak Atlantic "
                f"season (month={month}). Verify ftp.nhc.noaa.gov/atcf/aid_public/ "
                "is populated — an empty index in August could mean a feed outage."
            )
        else:
            print(
                f"  No storms in NHC ATCF index (month={month}, "
                "off-peak for most NHC basins)."
            )

    # Step 1b: Check JTWC for additional storms (WP, IO, SH)
    print()
    print("Step 1b: Checking JTWC RSS for Western Pacific / Indian Ocean / SH storms...")
    jtwc_storm_records = _discover_jtwc_storms()
    if jtwc_storm_records:
        for sid, recs in jtwc_storm_records.items():
            analysis = next((r for r in recs if r.tau_hours == 0), None)
            if analysis:
                cat = classify_storm(analysis.vmax_kt)
                print(f"  JTWC: {analysis.storm_name} ({sid}) — {cat}, {analysis.vmax_kt} kt at {analysis.lat}N {analysis.lon}E")
                print(f"    {len(recs)} ATCF records (analysis + {len(recs) - 1} forecasts)")
    else:
        print("  No active JTWC storms.")

    # Step 2: Build live cases from all sources (NHC ATCF + JTWC warnings)
    print()
    print("Step 2: Building feature cases from ATCF + JTWC data...")
    live_cases: list[dict[str, object]] = []

    # NHC-tracked storms from ATCF a-deck
    # A file not modified since well before the activity window cannot hold a recent
    # analysis; skip downloading it (the deck's own cycles stay the authority below).
    stale_before = now - dt.timedelta(hours=ACTIVE_MAX_AGE_HOURS + 6)
    n_skipped_unmodified = 0
    for sid in active_ids:
        if sid in jtwc_storm_records:
            continue  # Will use JTWC data instead
        modified = adeck_index.get(sid)
        if modified is not None and modified < stale_before:
            n_skipped_unmodified += 1
            continue
        print(f"  Fetching ATCF a-deck for {sid}...")
        records = fetch_realtime_adeck(sid)
        if not records:
            print(f"    No records for {sid}, skipping")
            continue
        active, age_h = is_active(records, now)
        if not active:
            age_txt = "no analysis" if age_h is None else f"latest analysis {age_h:.0f} h old"
            print(f"    {sid}: inactive ({age_txt} > {ACTIVE_MAX_AGE_HOURS:.0f} h), skipping")
            continue
        case = build_live_case(sid, records)
        if case is not None:
            name = case.get("storm_name", sid)
            vmax = case.get("analysis_vmax_kt", "?")
            cat = classify_storm(vmax if isinstance(vmax, (int, float)) else None)
            print(f"    {name}: {cat} ({vmax} kt)")
            live_cases.append(case)

    if n_skipped_unmodified:
        print(
            f"  {n_skipped_unmodified} a-decks not modified in the last "
            f"{ACTIVE_MAX_AGE_HOURS + 6:.0f} h: inactive, not downloaded"
        )

    # JTWC-tracked storms from warning text (parsed into ATCFRecords)
    for sid, records in jtwc_storm_records.items():
        active, age_h = is_active(records, now)
        if not active:
            age_txt = "no analysis" if age_h is None else f"warning analysis {age_h:.0f} h old"
            print(f"    {sid} (JTWC): inactive ({age_txt}), skipping")
            continue
        case = build_live_case(sid, records, full_history=False)
        if case is not None:
            name = case.get("storm_name", sid)
            vmax = case.get("analysis_vmax_kt", "?")
            cat = classify_storm(vmax if isinstance(vmax, (int, float)) else None)
            print(f"    {name} (JTWC): {cat} ({vmax} kt)")
            live_cases.append(case)

    if not live_cases:
        print()
        print("  No active tropical cyclones in any basin.")
        write_outputs([], now, model["model_version"])
        render_hurricane_page([], now, model["model_version"])
        build_site_artifacts()
        print("Done. No storms to score.")
        return

    # Steps 3-4: score all storms (NHC + JTWC unified) with the pinned model. The model was
    # loaded and provenance-checked in step 0; nothing is trained in this job.
    print()
    print(f"Step 3-4: Scoring {len(live_cases)} active storms with the pinned {model['model_version']} model...")
    scored = score_live_cases(model, live_cases)

    for s in scored:
        ri = s.get("ri_probability", 0) or 0
        print(f"  {s.get('storm_name', s['storm_id'])}: P(RI) = {ri:.1%} "
              f"({s['category']}, {s['vmax_kt']} kt)")

    # Trust layer: recalibrate RI probabilities on live outcomes + attach honest
    # [conf_lo, conf_hi] bands + Ed25519-signed receipts. Fail-safe: the model's
    # own held-out calibrated ri_probability stands until a live calibrator exists.
    try:
        from hazardpulse.trust.scoring import enrich_cells, load_forecaster, load_signer

        _signer = load_signer()
        _forecaster = load_forecaster("hurricane", signer=_signer)
        trust_version = getattr(_forecaster, "model_version", None)
        if _forecaster is not None and trust_version != model["model_version"]:
            # A calibrator fitted on another model's forecasts must not re-map, band or SIGN
            # this model's probabilities (the receipt would name the wrong model).
            print(
                f"  Trust layer: skipped -- its calibrator was fitted for {trust_version}, "
                f"the served model is {model['model_version']}; emitting model-calibrated forecasts."
            )
        elif _forecaster is not None and scored:
            enrich_cells(scored, _forecaster, prob_key="ri_probability",
                         issued_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(
                f"  Trust layer: calibrated {len(scored)} storms "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
        elif scored:
            print(
                "  Trust layer: no calibrator yet "
                "(results/calibration/hurricane_calibration.json); emitting model-calibrated forecasts."
            )
    except Exception as exc:  # never let the trust layer break a live forecast
        print(f"  Trust layer: skipped ({exc})")

    # Step 5: Write outputs
    print()
    print("Step 5: Writing outputs...")
    write_outputs(scored, now, model["model_version"])
    render_hurricane_page(scored, now, model["model_version"])
    build_site_artifacts()

    # ---- Alert manager evaluation ----
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        try:
            from hazardpulse.alerts import build_default_manager
            mgr = build_default_manager(
                audit_path=RESULTS / "alerts" / "audit.ndjson",
                recent_path=DIST / "data" / "alerts-recent.json",
            )
            pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
            fired = mgr.evaluate(pulse)
            for a in fired:
                if a.severity != "suppressed":
                    print(f"  ALERT [{a.severity}] {a.rule_name}: {a.message}")
        except Exception as exc:
            print(f"  Warning: alert evaluation skipped: {exc}")

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
