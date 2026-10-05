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

Atlantic / East / Central Pacific storms are served NOAA's own RI guidance instead whenever the
cycle's SHIPS text carries it (results/models/hurricane_ri_stack_v1.json, chosen by the
pre-registered program docs/HURRICANE_RI_PROGRAM.md); every storm says which (``ri_source``).
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
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
from hazardpulse.site.places import haversine_km  # noqa: E402
from hazardpulse.hurricane.operational_ri import (  # noqa: E402
    build_feature_matrix,
    climatological_mpi_features,
    translation_speed_kmh,
)
from hazardpulse.hurricane import cycle_records as cr  # noqa: E402
from hazardpulse.hurricane import ri_model  # noqa: E402
from hazardpulse.hurricane import ri_stack, ships_text  # noqa: E402
from hazardpulse.hurricane import ri_v9  # noqa: E402
from hazardpulse.hurricane import ri_v10  # noqa: E402
from hazardpulse.hurricane import ri_v9_features as v9fx  # noqa: E402

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
HISTORICAL_DATA = RESULTS / "hurricane_operational_ri_2000_2024_al_sst.jsonl"
# The served model, as chosen by scripts/evaluate_hurricane_ri.py's pre-registered rule
# (results/calibration/hurricane_ri_evaluation.json; a test keeps the two in agreement).
SERVED_MODEL_VERSION = "hurricane_ri_v8_2"
MODEL_ARTIFACT = ri_model.ARTIFACTS[SERVED_MODEL_VERSION]
PRIMARY_DOMAIN = "https://hazardpulse.com"

# NHC basins (AL/EP/CP) are served NOAA's own RI guidance, as the pre-registered program
# chose it (docs/HURRICANE_RI_PROGRAM.md, scripts/hurricane_ri_stack.py; the artifact names the
# candidate -- since 2026-10-02 A: DTOPS as issued, SHIPS-RII where DTOPS is missing), read from
# the SAME cycle's SHIPS text. v8.2 serves everything else: the JTWC basins (no public RI
# guidance) and NHC cycles whose SHIPS text is absent, unreadable, or lacks the needed aids.
# Every scored storm says which (``ri_source``) and carries the inputs it used.
STACK_ARTIFACT = ri_stack.ARTIFACT_PATH
RI_SOURCE_STACK = "noaa_aid_stack"
RI_SOURCE_V82 = "v8.2"
# What each pre-registered candidate serves, in words for the site (ri_stack.CANDIDATES).
STACK_PUBLIC = {
    "A": "DTOPS as issued, and SHIPS-RII where DTOPS is missing",
    "B": "SHIPS-RII as issued",
    "C": "NOAA's SHIPS RI consensus as issued (SHIPS-RII where it is missing)",
    "D": "a logistic pool of SHIPS-RII, its logistic and Bayesian versions, and DTOPS",
    "E": "a logistic pool of SHIPS-RII, its logistic and Bayesian versions, DTOPS and HazardPulse v8.2",
    "F": "a logistic pool of SHIPS-RII, its logistic and Bayesian versions, and DTOPS, with an Atlantic term",
}


def ri_sources_note(stack: dict[str, object] | None) -> str:
    v82_text = ("West Pacific, Indian Ocean, Southern Hemisphere (no public RI guidance) and any NHC cycle "
                "without a usable SHIPS text: HazardPulse v8.2.")
    if stack is None:
        return "Every basin: HazardPulse v8.2."
    return ("Atlantic, East and Central Pacific: NOAA's rapid-intensification guidance read from NHC's SHIPS "
            f"text for the same cycle -- {STACK_PUBLIC[stack['payload']['candidate']]} -- chosen by a "
            "pre-registered comparison on 2020-2024 and scored once on 2025 (docs/HURRICANE_RI_PROGRAM.md). "
            + v82_text)

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


# The storm's track history for the JTWC basins. A JTWC warning holds ONE cycle, so v8.2 -- trained on the
# best track -- had 9 of its 17 inputs (the 6/12/24-h intensity and pressure changes, the pressure, motion,
# age) median-imputed on every West Pacific storm: scored that way its 2022-2024 West Pacific log loss was
# 0.2589 against climatology's 0.2649 (0.2057 with every input), and Choi-Wan intensified 45, 45 and 40 kt in
# 24 h while v8.2 published 5.7%, 5.4% and 3.2% (2026-10-05 audit). UCAR RAL publishes the working best track
# of every active storm in real time; its fixes reached the file 1 h 47 min (00Z) and 3 h 17 min (12Z) after
# synoptic time on 2026-10-05, inside this scorer's t + 3 h 30 min.
RAL_BDECK_URL = "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{basin}{num}{year}.dat"
# A best track is this storm's only if it reaches the warning: its last fix at most 12 h before the warning's
# cycle (RAL's lag was 3 h 17 min at worst on 2026-10-05) and no farther from the warning's position than a storm moving
# 60 km/h could be, plus 200 km for the two agencies' fixes of one storm to differ.
HISTORY_MAX_GAP = dt.timedelta(hours=12)
HISTORY_KM_AT_SAME_TIME, HISTORY_KM_PER_HOUR = 200.0, 60.0


def atcf_season_year(basin: str, cycle: dt.datetime) -> int:
    """The year in an ATCF storm id. The southern hemisphere's season runs July to June and takes the year
    it ends in (RAL: bsh012026 began 2025-07-16, bsh012025 2024-10-01); every other basin's is the calendar
    year. A storm keeps the year it formed in, so one warned across a season's turn has the year before
    (``with_track_history`` tries it)."""
    return cycle.year + 1 if basin.upper() == "SH" and cycle.month >= 7 else cycle.year


def _previous_season(storm_id: str) -> str:
    return f"{storm_id[:4]}{int(storm_id[4:8]) - 1}"


def history_mismatch(history: list[ATCFRecord], analysis: ATCFRecord) -> str | None:
    """Why ``history`` is not the track of the storm ``analysis`` warns on, or None when it is."""
    last = max(history, key=lambda r: r.cycle)
    gap = analysis.cycle - last.cycle
    if gap > HISTORY_MAX_GAP:
        return f"its last fix ({last.cycle:%Y-%m-%d %HZ}) is {gap.total_seconds() / 3600:.0f} h before the warning"
    km = haversine_km(last.lat, last.lon, analysis.lat, analysis.lon)
    limit = HISTORY_KM_AT_SAME_TIME + HISTORY_KM_PER_HOUR * gap.total_seconds() / 3600
    if km > limit:
        return f"its last fix is {km:.0f} km from the warning's position (at most {limit:.0f})"
    return None


def ral_track_history(storm_id: str, current: dt.datetime, fetch=None) -> list[ATCFRecord]:
    """The storm's best-track analyses (``BEST``, tau 0) at or before ``current`` -- never a later fix --
    from RAL's real-time b-deck. [] when the file cannot be read: the warning alone is then scored, and the
    record says so."""
    sid = storm_id.strip().lower()
    if len(sid) != 8:
        return []
    url = RAL_BDECK_URL.format(year=sid[4:8], basin=sid[:2], num=sid[2:4])
    try:
        text = (fetch or (lambda u: fetch_text(u, namespace="ral_bdeck", use_cache=False)))(url)
    except Exception as exc:  # noqa: BLE001 -- a missing history must never stop the forecast
        print(f"    {storm_id}: RAL b-deck unavailable ({exc}); scoring from the warning alone")
        return []
    return [r for r in parse_atcf_deck(text or "") if r.model == "BEST" and r.tau_hours == 0 and r.cycle <= current]


def with_track_history(storm_id: str, warning_records: list[ATCFRecord], fetch=None
                       ) -> tuple[list[ATCFRecord], str, str]:
    """``(records, track_source, storm_id)``: the warning's records plus the storm's best-track history up to
    the warning's own cycle, and the storm's ATCF id -- the one the warning implies, or the season before's
    when only that track reaches the warning. A track that does not reach it (``history_mismatch``) is
    never merged: it would be another storm's. The analysis priority prefers ``BEST`` over ``JTWC`` for any
    cycle both hold, as v8.2's training set did; the warning stays the analysis while its cycle's fix is not
    yet published."""
    analyses = [r for r in warning_records if r.tau_hours == 0 and r.lat is not None and r.lon is not None]
    if not analyses or len(storm_id) != 8 or not storm_id[4:].isdigit():
        return warning_records, "jtwc_warning", storm_id
    analysis = max(analyses, key=lambda r: r.cycle)
    for sid in (storm_id, _previous_season(storm_id)):
        history = ral_track_history(sid, analysis.cycle, fetch=fetch)
        if not history:
            continue
        why = history_mismatch(history, analysis)
        if why is None:
            return warning_records + history, "ral_bdeck", sid
        print(f"    {sid}: RAL best track is not this storm's, {why}")
    return warning_records, "jtwc_warning", storm_id


def jtwc_live_case(storm_id: str, warning_records: list[ATCFRecord], fetch=None) -> dict[str, object] | None:
    """A JTWC storm's live case: its warning plus its best-track history, with ``track_source`` saying
    which the case was built from (the record keeps it, so a forecast from the warning alone is known)."""
    merged, track_source, sid = with_track_history(storm_id, warning_records, fetch=fetch)
    case = build_live_case(sid, merged, full_history=track_source == "ral_bdeck")
    if case is not None:
        case["track_source"] = track_source
    return case


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
    # the warning's own cycle, not the clock: SH02 warned 2026-10-05 is SH022027, and a warning read just
    # after midnight on 1 January belongs to the year before
    storm_id = f"{basin}{storm_number[:2]}{atcf_season_year(basin, cycle_time)}"

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
    cycle: dt.datetime | None = None,
) -> dict[str, object] | None:
    """Build a single feature case from the LATEST advisory cycle of an active storm.

    Unlike build_operational_ri_cases, this doesn't need b-deck/best-track
    because we're scoring a live storm, not evaluating historical accuracy.

    ``full_history``: the records hold the storm's whole track (an a-deck). A JTWC warning
    holds one cycle, so its storm age is unknown (None, not 0).

    ``cycle``: build that cycle instead of the latest (a catch-up forecast, amendment 7 rule 2);
    None when the deck has no analysis for it.
    """

    if not adeck_records:
        return None

    # Find the latest cycle with analysis data
    cycles = sorted(
        {r.cycle for r in adeck_records if r.tau_hours == 0},
        reverse=True,
    )
    if cycle is not None:
        cycles = [c for c in cycles if c == cycle]

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


def load_stack_model() -> dict[str, object]:
    """The served NOAA-aid stack artifact, refused unless it validates (ri_stack.load_artifact).

    Loaded on every run, storms or not, like the v8.2 artifact: a missing or corrupt file fails
    the job the day it happens, not at the first Atlantic storm.
    """
    payload, version = ri_stack.load_artifact(STACK_ARTIFACT)
    if ri_stack.V82 in (payload.get("inputs") or []):
        # The fitted pool would need v8.2 from the benchmark's model C on live inputs, a model
        # this scorer does not load: refuse rather than feed it a different v8.2.
        raise ri_stack.StackArtifactError(
            f"{STACK_ARTIFACT} pools v8.2 (candidate {payload['candidate']}); this scorer serves no such stack")
    print(f"  RI stack {version}: candidate {payload['candidate']} -- {payload['description']}; "
          f"fitted on {payload['training']['seasons']} ({payload['training']['n']} cycles)")
    return {"payload": payload, "model_version": version}


def fetch_ships_ri(storm_id: str, cycle: dt.datetime) -> tuple[ships_text.ShipsRI | None, str]:
    """``(parsed SHIPS RI matrix, "ok")`` for the storm's cycle, or ``(None, why)``."""
    name = ships_text.filename_for(storm_id, cycle)
    try:
        raw = fetch_text(ships_text.STEXT_ROOT + name, namespace="ships_text", use_cache=False)
    except Exception as exc:  # 404 (not yet published), network
        return None, f"absent ({type(exc).__name__})"
    try:
        return ships_text.parse_ships_text(raw, filename=name), "ok"
    except ships_text.ShipsTextError as exc:
        return None, f"refused ({exc})"


def stack_forecast(
    case: dict[str, object],
    stack: dict[str, object],
    ships_fetcher=None,
) -> tuple[float | None, dict[str, object]]:
    """``(P(RI), inputs)`` from NOAA's aids at the case's own cycle, or ``(None, {"status": why})``.

    Only an NHC a-deck case qualifies: the JTWC basins publish no RI guidance, and a JTWC
    warning is never the source of an NHC-basin storm (see _discover_jtwc_storms).
    """
    sid = str(case["storm_id"]).upper()
    if sid[:2] not in NHC_BASINS:
        return None, {"status": "jtwc_basin: NOAA publishes no RI guidance"}
    if case.get("analysis_model") == "JTWC":
        return None, {"status": "jtwc_warning_case"}
    cycle = dt.datetime.fromisoformat(str(case["issue_time"]))
    fetcher = ships_fetcher or fetch_ships_ri
    ri, status = fetcher(sid, cycle)
    if ri is None:
        return None, {"status": f"ships_text_{status}", "ships_text": ships_text.filename_for(sid, cycle)}
    out = ri_stack.predict(stack["payload"], ri.whole_percent, sid[:2], None)
    if out is None:
        return None, {"status": "ships_text_lacks_the_needed_aids", "ships_text": ships_text.summary(ri)}
    prob, info = out
    return prob, {"status": "ok", "candidate": stack["payload"]["candidate"],
                  "ships_text": ships_text.summary(ri), "url": ships_text.url_for(sid, cycle), **info}


def load_v9_model() -> dict[str, object] | None:
    """The frozen v9.1 model for SHADOW scoring (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 1),
    or None when its artifact is absent. Never the published forecast until its claim is met."""
    if not ri_v9.MODEL_PATH.exists():
        print("  v9.1 shadow: no artifact; not scored")
        return None
    payload, version = ri_v9.load()
    print(f"  v9.1 shadow {version}: {payload['n_trees']} trees, {len(payload['feature_names'])} inputs "
          "(recorded beside the published number, not published)")
    return {"payload": payload, "model_version": version}


def fetch_ships_raw(storm_id: str, cycle: dt.datetime) -> tuple[str | None, str]:
    """``(SHIPS text, file name)`` for the storm's cycle, or ``(None, why)``."""
    name = ships_text.filename_for(storm_id, cycle)
    try:
        return fetch_text(ships_text.STEXT_ROOT + name, namespace="ships_text", use_cache=False), name
    except Exception as exc:  # 404 (not yet published), network
        return None, f"absent ({type(exc).__name__})"


def load_v10_model(path: Path | None = None, label: str = "v10.1") -> dict[str, object] | None:
    """A frozen v10-schema exceedance-curve model, SHADOW like v9.1: v10.1 (amendment 2) by default,
    or a challenger such as v10.2 (amendment 3) from its own artifact."""
    path = path or ri_v10.MODEL_PATH
    if not path.exists():
        print(f"  {label} shadow: no artifact; not scored")
        return None
    art, version = ri_v10.load(path)
    print(f"  {label} shadow {version}: {len(art['members'])} members, thresholds {art['thresholds_kt']} kt "
          "(recorded beside the published number, not published)")
    return {"artifact": art, "model_version": version}


def load_challengers() -> dict[str, dict[str, object]]:
    """``{shadow key: loaded model}`` for every challenger whose artifact is present."""
    out = {}
    for key, path, label in (("ri_v10_2_shadow", ri_v10.V10_2_PATH, "v10.2"),
                             ("ri_v10_3_shadow", ri_v10.V10_3_PATH, "v10.3")):
        m = load_v10_model(path, label)
        if m is None:
            continue
        if ri_v10.needs_ir(m["artifact"]) and not ir_reader_available():
            # a missing image is NaN, as in training; a missing reader would record every forecast
            # without IR under this model's name -- so the model is not scored at all
            print(f"  {label} shadow: NOT scored -- its IR reader (h5py) is not installed")
            continue
        out[key] = m
    return out


def ir_reader_available() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


_IR_IMAGES: dict = {}            # one download per image hour per run, shared by every storm


IR_IMAGE_LABELS = {"p2": "t + 2 h", "m4": "t - 4 h"}


def live_ir_read(sid: str, cycle: dt.datetime, records) -> tuple[dict[str, float], dict[str, object]]:
    """``(IR features, what was read)`` for a cycle, made by the training code path (``ir_source``
    crops, ``ir_features``): GMGSI at t + 2 h and t - 4 h around the extrapolated CARQ centre. A
    missing image gives NaNs, as it did in training; ``what was read`` names each image's key, or
    None with its hour when there was no image (a run before t + 2 h 40 min finds none at t + 2 h)."""
    from hazardpulse.hurricane import ir_features, ir_source
    cen = ir_source.centres(records, cycle)
    if cen is None:
        return ir_features.features(None, None), {"centre": "no CARQ position at t"}
    crops, read = {}, {}
    for tag, (hour, centre) in cen.items():
        if hour not in _IR_IMAGES:
            _IR_IMAGES[hour] = ir_source.fetch_image(hour)
        key, counts, lat, lon = _IR_IMAGES[hour]
        crops[tag] = None if key is None else ir_source.crop(counts, lat, lon, centre)
        read[tag] = {"hour": hour.strftime("%Y-%m-%dT%H:00Z"), "key": key}
    return ir_features.features(crops["p2"], crops["m4"]), read


def live_ir_features(sid: str, cycle: dt.datetime, records) -> dict[str, float]:
    """The amendment-5 IR features for a cycle (``live_ir_read`` without its account of the reads)."""
    return live_ir_read(sid, cycle, records)[0]


def ir_status(features: dict[str, float], read: dict[str, object] | None = None) -> str:
    """What the IR inputs of a forecast actually were: ``ok`` only when every IR feature was computed;
    ``missing`` when none was (Rachel 2026-10-03 18Z, run at t + 46 min: no image at t + 2 h yet, all 14
    features NaN, and the record said "ok"); ``partial`` in between. Names the images that were not read."""
    from hazardpulse.hurricane import ir_features
    n = sum(1 for k in ir_features.IR_NAMES if math.isfinite(float(features.get(k, math.nan))))
    total = len(ir_features.IR_NAMES)
    why = []
    if read:
        if "centre" in read:
            why.append(str(read["centre"]))
        not_read = [f"{IR_IMAGE_LABELS[t]} ({v['hour']})" for t, v in read.items()
                    if t in IR_IMAGE_LABELS and isinstance(v, dict) and v.get("key") is None]
        if not_read and len(not_read) == sum(t in IR_IMAGE_LABELS for t in read):
            why.append("no image read: " + ", ".join(not_read))
        elif not_read:
            why.append("no image at " + ", ".join(not_read))
    detail = ("; " + "; ".join(why)) if why else ""
    if n == total:
        return "ok"
    if n == 0:
        return f"missing: 0 of {total} IR features" + detail
    return f"partial: {n} of {total} IR features" + detail


def _shadow_inputs(case: dict[str, object], ships_raw_fetcher=None, adeck_fetcher=None):
    """``(sid, cycle, {(tech, threshold): pct}, a-deck records, SHIPS file note)`` for an NHC a-deck
    case, or None: every RI threshold of the cycle's SHIPS text and the storm's a-deck."""
    sid = str(case["storm_id"]).upper()
    if sid[:2] not in NHC_BASINS or case.get("analysis_model") == "JTWC":
        return None
    cycle = dt.datetime.fromisoformat(str(case["issue_time"]))
    raw, name = (ships_raw_fetcher or fetch_ships_raw)(sid, cycle)
    pcts: dict[tuple[str, str], float] = {}
    if raw is not None:
        for th in v9fx.THRESHOLDS:
            try:
                ri = ships_text.parse_ships_text(raw, filename=name, threshold=th)
            except ships_text.ShipsTextError:
                continue
            for tech in v9fx.RI_TECHS:
                v = ri.whole_percent.get(tech)
                if v is not None:
                    pcts[(tech, th)] = v
    records = (adeck_fetcher or fetch_realtime_adeck)(sid)
    return sid, cycle, pcts, records, (name if raw is not None else f"ships_text_{name}")


def v9_shadow(case: dict[str, object], v9: dict[str, object], ships_raw_fetcher=None,
              adeck_fetcher=None) -> dict[str, object]:
    """The v9.1 forecast for an NHC a-deck case, through the same feature builder as training."""
    got = _shadow_inputs(case, ships_raw_fetcher, adeck_fetcher)
    if got is None:
        return {"status": "not an NHC a-deck case"}
    sid, cycle, pcts, records, note = got
    out = ri_v9.predict(v9["payload"], str(v9["model_version"]), records, cycle, sid[:2], pcts)
    return {"status": "ok", "ships_text": note, **out}


def _shadow_keys(v9, v10, challengers) -> list[str]:
    return ([k for k, m in (("ri_v9_shadow", v9), ("ri_v10_shadow", v10)) if m]
            + [k for k, m in (challengers or {}).items() if m])


def shadow_forecasts(case: dict[str, object], v9: dict[str, object] | None, v10: dict[str, object] | None,
                     ships_raw_fetcher=None, adeck_fetcher=None,
                     challengers: dict[str, dict[str, object]] | None = None,
                     ir_fetcher=None) -> dict[str, dict[str, object]]:
    """Every shadow (v9.1, v10.1 and each challenger) from ONE read of the cycle's SHIPS text and
    the storm's a-deck; an IR model also gets the cycle's IR features, read once.

    ``ir_fetcher(sid, cycle, records)`` returns the IR features, or ``(features, what was read)`` as
    ``live_ir_read`` does; the shadow's ``ir`` says what the inputs were (``ir_status``), never "ok"
    for a forecast made without an image."""
    got = _shadow_inputs(case, ships_raw_fetcher, adeck_fetcher)
    if got is None:
        return {k: {"status": "not an NHC a-deck case"} for k in _shadow_keys(v9, v10, challengers)}
    sid, cycle, pcts, records, note = got
    out: dict[str, dict[str, object]] = {}
    if v9 is not None:
        out["ri_v9_shadow"] = {"status": "ok", "ships_text": note,
                               **ri_v9.predict(v9["payload"], str(v9["model_version"]), records, cycle, sid[:2], pcts)}
    ir, ir_note, ir_read = None, None, None
    for key, m in (("ri_v10_shadow", v10), *(challengers or {}).items()):
        if m is None:
            continue
        extra = None
        if ri_v10.needs_ir(m["artifact"]):
            if ir is None:
                try:
                    got_ir = (ir_fetcher or live_ir_read)(sid, cycle, records)
                    ir, ir_read = got_ir if isinstance(got_ir, tuple) else (got_ir, None)
                    ir_note = ir_status(ir, ir_read)
                except Exception as exc:  # noqa: BLE001 -- no IR: NaN inputs, as a missing image was in training
                    ir, ir_note = {}, f"error: {type(exc).__name__}: {exc}"
            extra = ir
        out[key] = {"status": "ok", "ships_text": note,
                    **({"ir": ir_note} if extra is not None else {}),
                    **({"ir_images": ir_read} if extra is not None and ir_read else {}),
                    **ri_v10.predict(m["artifact"], str(m["model_version"]), records, cycle, sid[:2], pcts, extra)}
    return out


def analysis_cycles(records: list[ATCFRecord]) -> list[dt.datetime]:
    """Every cycle of a deck that build_live_case can build (an analysis with a wind)."""
    groups: dict[dt.datetime, dict[tuple[str, int], ATCFRecord]] = {}
    for r in records:
        groups.setdefault(r.cycle, {})[(r.model, r.tau_hours)] = r       # as build_live_case: last line wins
    return sorted(c for c, m in groups.items() if _select_analysis_record(m) is not None)


def recorded_test_cycles(now: dt.datetime, replay_dir: Path | None = None) -> set[tuple[str, dt.datetime]]:
    """Storm-cycles that already have a test record (amendment 7 rules 1-2), read from the forecast
    files that can hold one for a cycle still inside the catch-up window (made in the last 13 h)."""
    replay_dir = DIST / "data" / "replay" if replay_dir is None else replay_dir
    oldest = now - cr.CATCH_UP_LIMIT - dt.timedelta(hours=1)
    cands: list[cr.Candidate] = []
    for path in sorted(replay_dir.glob("hu_fcst_*.json")):
        stamp = cr.record_time({"forecast_id": path.stem})
        if stamp is not None and stamp < oldest:
            continue
        try:
            art = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cands.extend(cr.file_candidates(art))
    return set(cr.select(cands).chosen)


def catch_up_cases(live_cases: list[dict[str, object]], adeck_by_storm: dict[str, list[ATCFRecord]],
                   now: dt.datetime, have: set[tuple[str, dt.datetime]]) -> list[dict[str, object]]:
    """Amendment 7 rule 2: for every active NHC-basin numbered storm, a case for each cycle of its deck
    with ``t >= 2026-10-04 00Z``, ``t + 3 h 30 min`` passed, ``t + 12 h`` not passed and no test record
    -- built from the deck already read this run (no second read). The cycle this run forecasts live
    is not one: its live record is made now."""
    out = []
    for case in live_cases:
        sid = str(case["storm_id"]).upper()
        if case.get("analysis_model") == "JTWC" or not cr.is_nhc_numbered(sid):
            continue
        records = adeck_by_storm.get(sid) or []
        done = {t for s, t in have if s == sid} | {dt.datetime.fromisoformat(str(case["issue_time"]))}
        for t in cr.due_catch_up_cycles(analysis_cycles(records), now, done):
            c = build_live_case(sid, records, cycle=t)
            if c is not None:
                out.append(c)
    return out


def catch_up_forecasts(cases: list[dict[str, object]], now: dt.datetime, v9, v10,
                       challengers: dict[str, dict[str, object]] | None = None, ships_raw_fetcher=None,
                       adeck_fetcher=None, ir_fetcher=None) -> list[dict[str, object]]:
    """The shadow forecasts of the catch-up cycles: the SAME computation a live run makes for a cycle
    (``shadow_forecasts`` on build_live_case's case: one read of the cycle's SHIPS text, the storm's
    deck and the IR images), recorded with ``catch_up: true`` and its lag (record time minus t).
    Never a published number; a failure is written down and touches nothing else."""
    keys = _shadow_keys(v9, v10, challengers)
    out = []
    for case in cases:
        t = dt.datetime.fromisoformat(str(case["issue_time"]))
        rec: dict[str, object] = {
            "storm_id": case["storm_id"], "storm_name": case.get("storm_name", case["storm_id"]),
            "basin": case.get("basin", ""), "issue_time": case["issue_time"],
            "lat": case.get("analysis_lat"), "lon": case.get("analysis_lon"),
            "vmax_kt": case.get("analysis_vmax_kt"), "mslp_hpa": case.get("analysis_mslp_hpa"),
            "catch_up": True, "lag_hours": cr.lag_hours(now, t), "published": False,
        }
        try:
            rec.update(shadow_forecasts(case, v9, v10, ships_raw_fetcher, adeck_fetcher, challengers, ir_fetcher))
        except Exception as exc:  # noqa: BLE001
            err = {"status": f"error: {type(exc).__name__}: {exc}"}
            rec.update({k: dict(err) for k in keys})
        out.append(rec)
    return out


HU_LEDGER_PATH = DIST / "data" / "hurricane-ledger.jsonl"


def _canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def forecast_content(forecast_id: str, storms: list[dict], catch_up: list[dict] | None = None) -> dict:
    """What a forecast file said: its id, every storm record and -- when it has any -- its catch-up
    records (amendment 7 rule 2). A file without catch-up records hashes exactly as before."""
    body: dict[str, object] = {"forecast_id": forecast_id, "storms": storms}
    if catch_up:
        body[cr.CATCH_UP_KEY] = catch_up
    return body


def forecast_content_sha256(forecast_id: str, storms: list[dict], catch_up: list[dict] | None = None) -> str:
    """SHA-256 of what a forecast said -- its id and every storm record (published number, every
    shadow, every stored input), and its catch-up records. The site build rewrites the replay's
    envelope, never these, so this is recomputable from the replay file at any later time."""
    return _canonical_sha256(forecast_content(forecast_id, storms, catch_up))


def _ledger_shadows(s: dict) -> dict:
    return {k: {"model_version": v.get("model_version"), "probability": v.get("probability")}
            for k, v in s.items() if k.endswith("_shadow") and isinstance(v, dict) and v.get("status") == "ok"}


def append_hurricane_ledger(forecast_id: str, now: dt.datetime, model_version: str, storms: list[dict],
                            path: Path | None = None, catch_up: list[dict] | None = None) -> dict:
    """Append one hash-chained entry per forecast (the earthquake and tornado ledgers' scheme:
    ``hash`` = SHA-256 of the entry without ``hash``, ``prev_hash`` = the previous entry's hash).
    The entry carries the forecast's content hash, so editing any stored storm record, shadow or
    input afterwards breaks the chain's agreement with the replay file. One writer at a time: every
    scoring workflow shares the ``hazardpulse-scoring`` concurrency group."""
    # resolved from DIST at call time, not import time: a run (or a test) that points DIST elsewhere
    # must write its ledger there too, never into the real one
    path = DIST / "data" / HU_LEDGER_PATH.name if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = "0" * 64
    if path.exists():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            prev_hash = json.loads(lines[-1])["hash"]
    entry = {
        "timestamp": now.isoformat() + "Z",
        "forecast_id": forecast_id,
        "model_version": model_version,
        "content_sha256": forecast_content_sha256(forecast_id, storms, catch_up),
        "storms": [{
            "id": s.get("storm_id"), "issue_time": s.get("issue_time"),
            "published": s.get("ri_probability"), "source": s.get("ri_source_label") or s.get("ri_source"),
            "published_model_version": s.get("model_version"),
            "shadows": _ledger_shadows(s),
        } for s in storms],
        "prev_hash": prev_hash,
    }
    if catch_up:
        entry["catch_up"] = [{"id": s.get("storm_id"), "issue_time": s.get("issue_time"),
                              "lag_hours": s.get("lag_hours"), "shadows": _ledger_shadows(s)} for s in catch_up]
    entry["hash"] = _canonical_sha256(entry)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


def ri_source_label(storm: dict[str, object]) -> str:
    """What produced a storm's number, for people: 'NOAA DTOPS', 'NOAA SHIPS-RII', 'HazardPulse v8.2'."""
    if storm.get("ri_source") != RI_SOURCE_STACK:
        return "HazardPulse v8.2"
    used = (storm.get("ri_inputs") or {}).get("used")
    names = {"DTOP": "DTOPS", "RIOD": "SHIPS-RII", "RIOC": "SHIPS RI consensus"}
    return f"NOAA {names.get(str(used), 'RI guidance (SHIPS-RII/DTOPS)')}"


_CHANGE_SPAN_RE = re.compile(r"^analysis_d[vp]_(\d+)h$")


def unavailable_inputs(case: dict[str, object], names: list[str], row: np.ndarray) -> tuple[list[str], list[str]]:
    """``(missing, before_first_fix)``: the model's inputs this case could not provide (non-finite in the
    matrix it is scored from). A change over a span longer than the storm has existed is missing in the
    training set as well (build_hurricane_training_data: None when that fix is absent) and imputed alike, so
    it is ``before_first_fix``, not a deficit; with no known age every gap is a deficit."""
    age = case.get("storm_age_h")
    missing: list[str] = []
    before: list[str] = []
    for j, name in enumerate(names):
        if np.isfinite(row[j]):
            continue
        span = _CHANGE_SPAN_RE.match(name)
        if span is not None and isinstance(age, (int, float)) and int(span.group(1)) > age:
            before.append(name)
        else:
            missing.append(name)
    return missing, before


def score_live_cases(
    model: dict[str, object],
    live_cases: list[dict[str, object]],
    stack: dict[str, object] | None = None,
    ships_fetcher=None,
    v9: dict[str, object] | None = None,
    ships_raw_fetcher=None,
    v10: dict[str, object] | None = None,
    adeck_fetcher=None,
    challengers: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Score live cases with the pinned artifacts. No training happens here.

    v8.2 scores every case. With ``stack`` (load_stack_model), an NHC-basin a-deck case whose
    cycle's SHIPS text carries the needed aids is served NOAA's guidance instead, and keeps
    v8.2's number beside it (``v8_2``) for comparison. ``ri_probability`` is what is published;
    every downstream band (pulse risk_band, page) is derived from it.
    """

    if not live_cases:
        return []
    p = ri_model.score_cases(model, live_cases)
    n_features = len(model["selected_idx"])
    # which of v8.2's inputs each case could not provide (imputed at scoring), from the very matrix it scores
    names = list(model["feature_names"])
    X_inputs, _, _, _ = build_feature_matrix(live_cases, feature_names=names)
    unavailable = [unavailable_inputs(case, names, X_inputs[i]) for i, case in enumerate(live_cases)]

    scored: list[dict[str, object]] = []
    for i, case in enumerate(live_cases):
        v82 = {
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
        }
        storm: dict[str, object] = {
            "storm_id": case["storm_id"],
            "storm_name": case.get("storm_name", case["storm_id"]),
            "basin": case.get("basin", ""),
            "lat": case.get("analysis_lat"),
            "lon": case.get("analysis_lon"),
            "vmax_kt": case.get("analysis_vmax_kt"),
            "mslp_hpa": case.get("analysis_mslp_hpa"),
            "category": classify_storm(case.get("analysis_vmax_kt")),
            "issue_time": case.get("issue_time"),
            **v82,
            "ri_source": RI_SOURCE_V82,
            "ri_inputs": {"analysis_model": case.get("analysis_model"),
                          "track_source": case.get("track_source"),
                          "n_inputs": len(names),
                          "inputs_missing": unavailable[i][0],
                          "inputs_before_first_fix": unavailable[i][1]},
        }
        if stack is not None:
            prob, inputs = stack_forecast(case, stack, ships_fetcher)
            if prob is None:
                storm["ri_inputs"]["noaa_aid_stack"] = inputs
            else:
                payload = stack["payload"]
                storm.update({
                    "ri_probability": round(float(prob), 4),
                    "ri_probability_raw": round(float(prob), 4),
                    "model_scores": {RI_SOURCE_STACK: round(float(prob), 4)},
                    "n_features": len(inputs.get("inputs") or {}),
                    "model_version": stack["model_version"],
                    "calibration": ("none: NOAA's probability as issued" if payload["kind"] == "raw_aid"
                                    else "logistic pool of NOAA aids fitted 2020-2024"),
                    "ri_source": RI_SOURCE_STACK,
                    "ri_inputs": inputs,
                    "v8_2": {k: v82[k] for k in ("ri_probability", "model_version")},
                })
        if _shadow_keys(v9, v10, challengers):
            # shadows: recorded for the prospective test, never the published number; a failure here
            # is written down and must not touch the forecast that is published
            try:
                storm.update(shadow_forecasts(case, v9, v10, ships_raw_fetcher, adeck_fetcher, challengers))
            except Exception as exc:  # noqa: BLE001
                err = {"status": f"error: {type(exc).__name__}: {exc}"}
                storm.update({k: dict(err) for k in _shadow_keys(v9, v10, challengers)})
        storm["ri_source_label"] = ri_source_label(storm)
        scored.append(storm)

    return scored


def headline_model_version(scored: list[dict[str, object]], default: str) -> str:
    """The model behind the number the pulse shows (the top storm's), else ``default``."""
    if not scored:
        return default
    return str(max(scored, key=lambda s: s["ri_probability"])["model_version"])


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
                "ri_source": s.get("ri_source", RI_SOURCE_V82),
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
    note: str | None = None,
    catch_up: list[dict[str, object]] | None = None,
) -> None:
    """Write scored results to dist/data/.

    ``model_version`` is the model behind the headline (top-storm) number; each storm carries
    its own ``model_version`` and ``ri_source``, and ``model_versions`` lists every model that
    produced a number in this run. ``catch_up``: the run's shadow-only records of missed cycles
    (amendment 7 rule 2), kept beside the storms -- never among them, so nothing publishes them.
    """
    forecast_id = f"hu_fcst_{now.strftime('%Y%m%d_%H%M')}"
    sources: dict[str, int] = {}
    for s in scored_storms:
        key = str(s.get("ri_source", RI_SOURCE_V82))
        sources[key] = sources.get(key, 0) + 1

    # Write live-storms.json
    output = {
        "updated_at": now.isoformat() + "Z",
        "forecast_id": forecast_id,
        "model_version": model_version,
        "model_versions": sorted({str(s.get("model_version")) for s in scored_storms if s.get("model_version")}),
        "ri_sources": sources,
        "ri_sources_note": note if note is not None else ri_sources_note(None),
        "n_active_storms": len(scored_storms),
        "storms": scored_storms,
    }
    if catch_up:
        output[cr.CATCH_UP_KEY] = catch_up
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
    if catch_up:
        replay_payload[cr.CATCH_UP_KEY] = catch_up
    replay_path.write_text(json.dumps(replay_payload, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {replay_path}" + (f" (+{len(catch_up)} catch-up shadow records)" if catch_up else ""))
    entry = append_hurricane_ledger(forecast_id, now, model_version, scored_storms, catch_up=catch_up)
    print(f"  Ledger: {HU_LEDGER_PATH.name} +1 (hash {entry['hash'][:12]}, prev {entry['prev_hash'][:12]})")

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
                    hazard["model_version"] = top.get("model_version", model_version)
                    hazard["ri_source"] = top.get("ri_source", RI_SOURCE_V82)
                    hazard["ri_source_label"] = top.get("ri_source_label") or ri_source_label(top)
                    hazard["forecast_id"] = forecast_id
                    hazard["n_active_storms"] = len(scored_storms)
                else:
                    hazard["probability"] = 0.0
                    hazard["conf_lo"] = None
                    hazard["conf_hi"] = None
                    hazard["risk_band"] = "none"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = model_version
                    hazard.pop("ri_source", None)
                    hazard.pop("ri_source_label", None)
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


def _format_time(ts: dt.datetime) -> str:
    return ts.strftime("%a, %d %b %Y %H:%M:%S UTC")


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

    print("Step 0: Loading the pinned model artifacts...")
    model = load_serving_model()
    stack = load_stack_model()
    v9 = load_v9_model()
    v10 = load_v10_model()
    challengers = load_challengers()
    note = ri_sources_note(stack)
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
    # each storm's a-deck as fetched here, reused by the v9.1 shadow (one download per storm per run)
    adeck_by_storm: dict[str, list[ATCFRecord]] = {}

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
        adeck_by_storm[str(sid).upper()] = records
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
        case = jtwc_live_case(sid, records)
        if case is not None:
            name = case.get("storm_name", sid)
            vmax = case.get("analysis_vmax_kt", "?")
            cat = classify_storm(vmax if isinstance(vmax, (int, float)) else None)
            print(f"    {name} (JTWC): {cat} ({vmax} kt); track history: {case['track_source']}")
            live_cases.append(case)

    if not live_cases:
        print()
        print("  No active tropical cyclones in any basin.")
        # No number is published; the page names the model that serves the NHC basins.
        write_outputs([], now, stack["model_version"], note=note)
        build_site_artifacts()
        print("Done. No storms to score.")
        return

    # Steps 3-4: score all storms (NHC + JTWC unified) with the pinned artifacts. Both were
    # loaded and checked in step 0; nothing is trained in this job.
    print()
    print(f"Step 3-4: Scoring {len(live_cases)} active storms: NOAA aids ({stack['model_version']}) where the "
          f"cycle's SHIPS text has them, else {model['model_version']}...")
    scored = score_live_cases(model, live_cases, stack=stack, v9=v9, v10=v10,
                              adeck_fetcher=lambda s: adeck_by_storm.get(str(s).upper(), []),
                              challengers=challengers)

    for s in scored:
        ri = s.get("ri_probability", 0) or 0
        why = "" if s["ri_source"] == RI_SOURCE_STACK else f" [{s['ri_inputs'].get('noaa_aid_stack', {}).get('status', '')}]"
        print(f"  {s.get('storm_name', s['storm_id'])}: P(RI) = {ri:.1%} from {s['ri_source_label']}"
              f"{why} ({s['category']}, {s['vmax_kt']} kt)")

    # Catch-up (docs/HURRICANE_RI_V9_PROGRAM.md amendment 7 rule 2): every cycle of an active NHC storm
    # that no run recorded after its advisory, still within t + 12 h, forecast in shadow from the same
    # deck read above. A failure here must not touch the published forecast.
    catch_up: list[dict[str, object]] = []
    if _shadow_keys(v9, v10, challengers):
        try:
            due = catch_up_cases(live_cases, adeck_by_storm, now, recorded_test_cycles(now))
            catch_up = catch_up_forecasts(due, now, v9, v10, challengers,
                                          adeck_fetcher=lambda s: adeck_by_storm.get(str(s).upper(), []))
        except Exception as exc:  # noqa: BLE001
            print(f"  Catch-up: skipped ({type(exc).__name__}: {exc})")
        for c in catch_up:
            print(f"  Catch-up (shadow only): {c['storm_id']} {c['issue_time']}, {c['lag_hours']:.1f} h after t")

    # Trust layer: recalibrate RI probabilities on live outcomes + attach honest
    # [conf_lo, conf_hi] bands + Ed25519-signed receipts. Fail-safe: the model's
    # own held-out calibrated ri_probability stands until a live calibrator exists.
    try:
        from hazardpulse.trust.scoring import enrich_cells, load_forecaster, load_signer

        _signer = load_signer()
        _forecaster = load_forecaster("hurricane", signer=_signer)
        trust_version = getattr(_forecaster, "model_version", None)
        # A calibrator is one model's curve: it may re-map, band and SIGN only the storms that
        # model scored (the receipt would otherwise name the wrong model). With two models
        # serving (NOAA aids in NHC basins, v8.2 elsewhere) that is a per-storm decision.
        own = [s for s in scored if s.get("model_version") == trust_version]
        foreign = len(scored) - len(own)
        if _forecaster is not None and own:
            enrich_cells(own, _forecaster, prob_key="ri_probability",
                         issued_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            scored.sort(key=lambda c: (c.get("abstained", False), -float(c.get("ri_probability") or 0.0)))
            print(
                f"  Trust layer: calibrated {len(own)} storms "
                f"(model {_forecaster.model_version}, signed={_signer is not None})"
            )
        if _forecaster is not None and foreign:
            print(
                f"  Trust layer: skipped {foreign} storms -- its calibrator was fitted for {trust_version}, "
                "they were scored by another model; emitting their model-calibrated forecasts."
            )
        elif _forecaster is None and scored:
            print(
                "  Trust layer: no calibrator yet "
                "(results/calibration/hurricane_calibration.json); emitting model-calibrated forecasts."
            )
    except Exception as exc:  # never let the trust layer break a live forecast
        print(f"  Trust layer: skipped ({exc})")

    # Step 5: Write outputs
    print()
    print("Step 5: Writing outputs...")
    headline_version = headline_model_version(scored, stack["model_version"])
    write_outputs(scored, now, headline_version, note=note, catch_up=catch_up)
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
