"""Shared helpers for prospective earthquake forecast logging and scoring."""

from __future__ import annotations

import datetime as dt
from typing import Iterator

from hazardpulse.data import usgs_fdsn
from hazardpulse.data.http import fetch_text


USGS_API = usgs_fdsn.USGS_FDSN_EVENT_URL


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
) -> list[dict]:
    """Fetch a USGS earthquake catalog over a time range.

    The USGS CSV endpoint silently truncates a query at its row limit, so this
    pulls month by month and bisects any month whose response reaches the limit
    (hazardpulse.data.usgs_fdsn.fetch_window); deduplicates by event id.
    """
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

    return sorted(events_by_id.values(), key=lambda event: event.get("time", ""))
