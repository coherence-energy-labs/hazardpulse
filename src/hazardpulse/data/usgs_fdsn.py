"""Limit-aware access to the USGS FDSN event service, plus a catalog completeness audit.

Why this module exists
----------------------
The FDSN endpoint answers ``limit=20000`` queries by silently returning the FIRST
20,000 rows -- no error, no flag. The earthquake downloader asked for one calendar
year at a time with ``orderby=time-asc&limit=20000``, so every year with more than
20,000 M2.5+ events was cut off mid-year (2018 ended on 2018-06-30, 2024 on
2024-10-20) and 2000-2025 lost 24.1% of M2.5+, 23.4% of M5+ and 22.0% of M6+
events. Every consumer of ``load_usgs_catalog`` trained and backtested on that.

Two rules make that impossible to repeat silently:

1. :func:`fetch_window` treats a response that reaches the limit as INCOMPLETE and
   bisects the window until every response is under the limit; a window that
   cannot be split further raises :class:`USGSTruncationError`.
2. :func:`audit_event_times` checks a stored catalog for the signatures a truncated
   or partially-failed pull leaves behind (exactly ``limit`` rows, calendar months
   with no events, a tail that stops well before the period ends), so files written
   by older code are detected instead of trusted.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import time as _time
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import urlencode

__all__ = [
    "audit_year_catalog",
    "manifest_path_for",
    "read_manifest",
    "USGS_FDSN_EVENT_URL",
    "USGS_FDSN_ROW_LIMIT",
    "USGSResponseError",
    "USGSTruncationError",
    "USGSCatalogIncompleteError",
    "iter_monthly_windows",
    "parse_event_time",
    "query_url",
    "fetch_window",
    "fetch_catalog",
    "audit_event_times",
]

USGS_FDSN_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_FDSN_ROW_LIMIT = 20000

FetchText = Callable[[str], str]


class USGSResponseError(RuntimeError):
    """The service answered with something that is not an FDSN event CSV."""


class USGSTruncationError(RuntimeError):
    """A window still hit the row limit after it could no longer be split."""


class USGSCatalogIncompleteError(RuntimeError):
    """A stored catalog shows the signature of a truncated or partial pull."""


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _fmt(value: dt.datetime) -> str:
    """FDSN time parameter with millisecond precision (bisection midpoints need it)."""
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%S.") + f"{_utc(value).microsecond // 1000:03d}"


def iter_monthly_windows(
    start: dt.datetime,
    end: dt.datetime,
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """Yield contiguous month-aligned UTC windows spanning [start, end)."""
    start, end = _utc(start), _utc(end)
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            nxt = dt.datetime(cursor.year + 1, 1, 1, tzinfo=dt.timezone.utc)
        else:
            nxt = dt.datetime(cursor.year, cursor.month + 1, 1, tzinfo=dt.timezone.utc)
        yield cursor, min(nxt, end)
        cursor = nxt


def parse_event_time(text: str) -> dt.datetime:
    """Parse an FDSN CSV ``time`` value (e.g. ``2018-06-30T02:06:32.200Z``) to UTC."""
    parsed = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return _utc(parsed)


def query_url(
    start: dt.datetime,
    end: dt.datetime,
    *,
    min_magnitude: float,
    limit: int = USGS_FDSN_ROW_LIMIT,
    base_url: str = USGS_FDSN_EVENT_URL,
) -> str:
    params = urlencode(
        {
            "format": "csv",
            "starttime": _fmt(start),
            "endtime": _fmt(end),
            "minmagnitude": f"{min_magnitude:.1f}",
            "orderby": "time-asc",
            "limit": int(limit),
        }
    )
    return f"{base_url}?{params}"


def _parse_csv(text: str, url: str) -> tuple[list[str], list[dict]]:
    if not text.strip():
        return [], []          # FDSN answers "no events" with 204 / an empty body
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    if "time" not in fieldnames or "id" not in fieldnames:
        head = text[:120].replace("\n", " ")
        raise USGSResponseError(f"not an FDSN event CSV from {url}: {head!r}")
    return fieldnames, list(reader)


def fetch_window(
    start: dt.datetime,
    end: dt.datetime,
    *,
    min_magnitude: float,
    fetch_text: FetchText,
    limit: int = USGS_FDSN_ROW_LIMIT,
    min_window_seconds: float = 1.0,
    pause_seconds: float = 0.0,
    stats: dict | None = None,
    base_url: str = USGS_FDSN_EVENT_URL,
) -> tuple[list[str], list[dict]]:
    """Every event with ``start <= time < end``, however many there are.

    A response with ``>= limit`` rows is incomplete by definition (the service cut
    it), so the window is bisected and both halves are fetched instead. FDSN's
    ``endtime`` is inclusive; rows are clipped to the half-open window and
    de-duplicated by id so adjacent windows never double count.
    """
    start, end = _utc(start), _utc(end)
    if stats is not None:
        stats.setdefault("requests", 0)
        stats.setdefault("splits", 0)
        stats.setdefault("max_rows_per_response", 0)
    url = query_url(start, end, min_magnitude=min_magnitude, limit=limit, base_url=base_url)
    fieldnames, rows = _parse_csv(fetch_text(url), url)
    if stats is not None:
        stats["requests"] += 1
        stats["max_rows_per_response"] = max(stats["max_rows_per_response"], len(rows))
    if pause_seconds > 0:
        _time.sleep(pause_seconds)

    if len(rows) >= limit:
        if (end - start).total_seconds() <= min_window_seconds:
            raise USGSTruncationError(
                f"{len(rows)} events between {_fmt(start)} and {_fmt(end)} "
                f"(M{min_magnitude:.1f}+) reach the {limit}-row limit and the window "
                f"cannot be split further"
            )
        if stats is not None:
            stats["splits"] += 1
        mid = start + (end - start) / 2
        kw = dict(min_magnitude=min_magnitude, fetch_text=fetch_text, limit=limit,
                  min_window_seconds=min_window_seconds, pause_seconds=pause_seconds,
                  stats=stats, base_url=base_url)
        f1, r1 = fetch_window(start, mid, **kw)
        f2, r2 = fetch_window(mid, end, **kw)
        return (f1 or f2 or fieldnames), _dedupe(r1 + r2)

    kept = []
    for row in rows:
        try:
            t = parse_event_time(row.get("time", ""))
        except ValueError:
            if stats is not None:
                stats["unparsable_time"] = stats.get("unparsable_time", 0) + 1
            continue
        if start <= t < end:
            kept.append(row)
    return fieldnames, _dedupe(kept)


def _dedupe(rows: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = row.get("id") or f"{row.get('time')}|{row.get('latitude')}|{row.get('longitude')}|{row.get('mag')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def fetch_catalog(
    start: dt.datetime,
    end: dt.datetime,
    *,
    min_magnitude: float,
    fetch_text: FetchText,
    limit: int = USGS_FDSN_ROW_LIMIT,
    pause_seconds: float = 0.0,
    stats: dict | None = None,
    base_url: str = USGS_FDSN_EVENT_URL,
) -> tuple[list[str], list[dict]]:
    """[start, end) pulled month by month (each month bisected if it hits the limit).

    Raises on any failed request -- a partial catalog is never returned.
    """
    fieldnames: list[str] = []
    rows: list[dict] = []
    for w_start, w_end in iter_monthly_windows(start, end):
        f, r = fetch_window(w_start, w_end, min_magnitude=min_magnitude, fetch_text=fetch_text,
                            limit=limit, pause_seconds=pause_seconds, stats=stats,
                            base_url=base_url)
        fieldnames = fieldnames or f
        rows.extend(r)
    rows = _dedupe(rows)
    rows.sort(key=lambda row: row.get("time", ""))
    return fieldnames, rows


def audit_event_times(
    times: Iterable[str],
    *,
    period_start: dt.datetime,
    period_end: dt.datetime | None,
    n_rows: int | None = None,
    limit: int = USGS_FDSN_ROW_LIMIT,
    max_edge_gap_days: float = 7.0,
    proven_unclipped: bool = False,
) -> list[str]:
    """Problems that mark a stored global catalog as truncated or partially pulled.

    ``period_end`` is where the catalog is supposed to stop; pass None when that is
    unknown (a current-year file of unknown fetch time) to check only the interior.
    Assumes a GLOBAL catalog at M<=2.5, which has events every day -- a calendar month
    with none, or a week-long gap at either edge, is a missing chunk, not quiet Earth.
    ``proven_unclipped`` (a fetch manifest recorded every response under the limit)
    waives only the exactly-``limit``-rows signature.
    """
    period_start = _utc(period_start)
    stamps = []
    for text in times:
        try:
            stamps.append(parse_event_time(text))
        except (ValueError, AttributeError):
            continue
    n = len(stamps) if n_rows is None else int(n_rows)
    problems: list[str] = []
    if not stamps:
        return ["no events"]
    if n == limit and not proven_unclipped:
        problems.append(f"exactly {limit} rows: the FDSN row-limit truncation signature")
    first, last = min(stamps), max(stamps)
    end = _utc(period_end) if period_end is not None else last
    gap = dt.timedelta(days=max_edge_gap_days)
    if first - period_start > gap:
        problems.append(f"first event {first:%Y-%m-%d} is {(first - period_start).days} d after the period start")
    if period_end is not None and end - last > gap:
        problems.append(f"last event {last:%Y-%m-%d} is {(end - last).days} d before the period end {end:%Y-%m-%d}")
    present = {(t.year, t.month) for t in stamps}
    expected = [(w.year, w.month) for w, _ in iter_monthly_windows(period_start, end)]
    missing = [f"{y}-{m:02d}" for y, m in expected if (y, m) not in present]
    if missing:
        problems.append(f"{len(missing)} month(s) with no events: {', '.join(missing[:12])}")
    return problems


def manifest_path_for(csv_path) -> Path:
    """Fetch-manifest sidecar of a cached catalog file (``x.csv`` -> ``x.manifest.json``)."""
    return Path(csv_path).with_suffix(".manifest.json")


def read_manifest(csv_path) -> dict:
    """The fetch manifest written next to a catalog file, or {} for a legacy file."""
    path = manifest_path_for(csv_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def audit_year_catalog(
    times: list[str],
    year: int,
    *,
    manifest: dict | None = None,
    now: dt.datetime | None = None,
) -> list[str]:
    """Completeness problems of one calendar year of a global catalog ([] = usable).

    The year is expected to run to Jan 1 of the next year, or -- for the running
    year -- to the manifest's ``coverage_end``. A running-year file with no manifest
    (fetch time unknown) is checked on its interior only.
    """
    now = _utc(now or dt.datetime.now(dt.timezone.utc))
    manifest = manifest or {}
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end: dt.datetime | None = min(dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc), now)
    if manifest.get("coverage_end"):
        end = min(end, parse_event_time(manifest["coverage_end"]))
    elif end > now - dt.timedelta(days=1):
        end = None
    limit = int(manifest.get("row_limit", USGS_FDSN_ROW_LIMIT))
    unclipped = bool(manifest) and int(manifest.get("max_rows_per_response", limit)) < limit
    problems = audit_event_times(times, period_start=start, period_end=end,
                                 limit=limit, proven_unclipped=unclipped)
    if manifest and int(manifest.get("n_events", -1)) != len(times):
        problems.append(f"manifest records {manifest.get('n_events')} events, file has {len(times)}")
    return problems
