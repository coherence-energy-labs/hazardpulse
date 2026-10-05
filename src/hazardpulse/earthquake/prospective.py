"""Shared helpers for prospective earthquake forecast logging and scoring."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Iterator

from hazardpulse.data import usgs_fdsn
from hazardpulse.data.http import fetch_text


USGS_API = usgs_fdsn.USGS_FDSN_EVENT_URL

# Completeness audit of a pulled catalog (the live forecast's input, the verifier's truth).
#
# FDSN answers "no events" with HTTP 204 or an empty 200 body, and ``usgs_fdsn._parse_csv`` must
# accept that as an empty month -- so a month the service failed to fill looks exactly like a quiet
# month. A GLOBAL catalog at M4.5 or below is never quiet for long: in the ComCat M4.5+ year files
# 1990-2026 the largest gap between consecutive events is 1.13 days (1996) and the fewest events in
# a calendar month 245 (1996-05); M2.5+ is a superset, so its gaps are no longer. A gap of
# CATALOG_MAX_GAP_DAYS anywhere -- at either edge or inside the period -- is therefore a missing
# chunk, not quiet Earth, and the pull is refused. At M6 the same audit has no power (2018-06 had
# ONE M6+ event), so a sparse target catalog is pulled at AUDIT_MAX_MAGNITUDE, audited, then
# filtered (``fetch_target_catalog``).
CATALOG_MAX_GAP_DAYS = 3.0
AUDIT_MAX_MAGNITUDE = 4.5


class CatalogIncompleteError(usgs_fdsn.USGSCatalogIncompleteError):
    """A pulled catalog has a hole: an empty month, a stale tail, or a multi-day gap."""


def audit_catalog(
    events: Iterable[dict],
    start: dt.datetime,
    end: dt.datetime,
    *,
    max_gap_days: float = CATALOG_MAX_GAP_DAYS,
) -> list[str]:
    """Problems that mark a pulled global catalog (M <= AUDIT_MAX_MAGNITUDE) as incomplete.

    ``usgs_fdsn.audit_event_times`` (empty calendar months, a late first event, an early last
    event) with the edge tolerance tightened to ``max_gap_days``, plus the same bound on every
    gap between consecutive events inside the period, which the month check cannot see.
    ``[]`` means usable. The row-limit signature is waived: ``fetch_window`` bisects every
    response that reaches the limit, so no response here was clipped.
    """
    times = [str(event.get("time", "")) for event in events]
    problems = usgs_fdsn.audit_event_times(
        times,
        period_start=start,
        period_end=end,
        max_edge_gap_days=max_gap_days,
        proven_unclipped=True,
    )
    stamps = []
    for text in times:
        try:
            stamps.append(usgs_fdsn.parse_event_time(text))
        except (ValueError, AttributeError):
            continue
    stamps.sort()
    limit = dt.timedelta(days=max_gap_days)
    gaps = [(b - a, a, b) for a, b in zip(stamps, stamps[1:]) if b - a > limit]
    if gaps:
        width, a, b = max(gaps)
        problems.append(
            f"{len(gaps)} gap(s) longer than {max_gap_days:g} d between events; the longest is "
            f"{width.total_seconds() / 86400:.1f} d ({format_utc_z(a)} to {format_utc_z(b)})"
        )
    return problems


def parse_utc_datetime(text: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp and return a UTC-aware datetime."""
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def format_utc_z(value: dt.datetime) -> str:
    """Format a datetime as UTC with a trailing Z."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def forecast_id_for_time(
    when: dt.datetime,
    *,
    prefix: str = "eq_fcst",
) -> str:
    """Return a stable forecast identifier for a UTC issue time."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    when = when.astimezone(dt.timezone.utc)
    return f"{prefix}_{when.strftime('%Y%m%d_%H00')}"


def iter_monthly_windows(
    start: dt.datetime,
    end: dt.datetime,
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """Yield contiguous month-aligned windows spanning [start, end)."""
    yield from usgs_fdsn.iter_monthly_windows(start, end)


def fetch_usgs_catalog_range(
    start: dt.datetime,
    end: dt.datetime,
    *,
    min_magnitude: float = 2.5,
    namespace: str = "usgs_prospective",
    verbose: bool = False,
    audit: bool = True,
) -> list[dict]:
    """Fetch a USGS earthquake catalog over a time range.

    The USGS CSV endpoint silently truncates a query at its row limit, so this
    pulls month by month and bisects any month whose response reaches the limit
    (hazardpulse.data.usgs_fdsn.fetch_window); deduplicates by event id.

    With ``audit`` (the default) the result is checked by ``audit_catalog`` and a catalog
    with a hole raises :class:`CatalogIncompleteError` instead of being returned: an empty
    month answered 200/204 is otherwise indistinguishable from "no events". The audit needs a
    dense catalog, so ``audit`` refuses ``min_magnitude`` above AUDIT_MAX_MAGNITUDE -- pull a
    sparse target set with ``fetch_target_catalog``.
    """
    if audit and min_magnitude > AUDIT_MAX_MAGNITUDE:
        raise ValueError(
            f"an M{min_magnitude:.1f}+ global catalog is too sparse to audit for holes (one M6+ event "
            f"in 2018-06); fetch at M{AUDIT_MAX_MAGNITUDE:.1f}+ and filter (fetch_target_catalog)"
        )
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    start = start.astimezone(dt.timezone.utc)
    end = end.astimezone(dt.timezone.utc)

    events_by_id: dict[str, dict] = {}

    def _get(url: str) -> str:
        return fetch_text(url, namespace=namespace, use_cache=False, refresh=True)

    for window_start, window_end in iter_monthly_windows(start, end):
        if verbose:
            print(
                "  USGS fetch",
                format_utc_z(window_start),
                "to",
                format_utc_z(window_end),
            )
        _, rows = usgs_fdsn.fetch_window(
            window_start,
            window_end,
            min_magnitude=min_magnitude,
            fetch_text=_get,
        )
        for row in rows:
            try:
                event_id = row.get("id", "")
                event = {
                    "time": row.get("time", ""),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "depth": float(row.get("depth", 0) or 0),
                    "mag": float(row["mag"]),
                    "magType": row.get("magType", ""),
                    "place": row.get("place", ""),
                    "id": event_id,
                }
            except (KeyError, ValueError):
                continue

            key = event_id or (
                f"{event['time']}|{event['latitude']:.4f}|"
                f"{event['longitude']:.4f}|{event['mag']:.2f}"
            )
            events_by_id[key] = event

    events = sorted(events_by_id.values(), key=lambda event: event.get("time", ""))
    if audit:
        problems = audit_catalog(events, start, end)
        if problems:
            raise CatalogIncompleteError(
                f"USGS M{min_magnitude:.1f}+ catalog {format_utc_z(start)} to {format_utc_z(end)} is "
                f"incomplete ({len(events)} events): " + "; ".join(problems)
            )
    return events


def fetch_target_catalog(
    start: dt.datetime,
    end: dt.datetime,
    *,
    target_min_magnitude: float = 6.0,
    namespace: str = "usgs_prospective_targets",
    verbose: bool = False,
) -> list[dict]:
    """The events with ``mag >= target_min_magnitude`` in [start, end), from an AUDITED pull.

    The pull is made at AUDIT_MAX_MAGNITUDE, where a hole is detectable (``audit_catalog``),
    and filtered here on the same ``mag`` column FDSN's ``minmagnitude`` reads; a pull with a
    hole raises instead of silently scoring a missed M6+ as a correct "no".
    """
    if target_min_magnitude < AUDIT_MAX_MAGNITUDE:
        raise ValueError(f"target magnitude {target_min_magnitude} is below the audited pull "
                         f"(M{AUDIT_MAX_MAGNITUDE:.1f}+)")
    events = fetch_usgs_catalog_range(
        start,
        end,
        min_magnitude=AUDIT_MAX_MAGNITUDE,
        namespace=namespace,
        verbose=verbose,
        audit=True,
    )
    return [event for event in events if float(event["mag"]) >= target_min_magnitude]
