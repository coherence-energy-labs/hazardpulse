"""The live record of an earthquake model version, counted so that overlap is not evidence.

A 30-day forecast is issued several times a day, so consecutive windows overlap almost
entirely and one M6+ earthquake falls inside ~100 of them. Summing events over windows counts
that earthquake ~100 times: on 2026-10-04 the 613 scored windows of ``eq_coherence_v1_0``
held 64 distinct M6+ earthquakes (USGS event ids) and the record said "6,955 events
observed", and the site's 10-event floor for quoting a live score would have passed after
about two windows. Everything here counts what is independent:

* ``n_distinct_events``: earthquakes, each counted once by its USGS event id;
* ``n_independent_windows``: the most windows that do not overlap (greedy by issue time,
  optimal for equal-length intervals);
* ``quotable``: a live score is stated only with at least LIVE_MIN_DISTINCT_EVENTS distinct
  earthquakes AND at least LIVE_MIN_INDEPENDENT_WINDOWS non-overlapping windows, so a mean
  over one window (or over copies of one window) is never printed as a record.

The scorer (``scripts/score_earthquake_prospective.py``) writes these counts; the rollup
(``scripts/build_site_artifacts.py``) and the track-record page read them through
``quotable`` and never fall back to the summed count.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable

# The same floor the site applies to every hazard (hazardpulse.site.pages.record.LIVE_MIN_EVENTS),
# here counted in distinct earthquakes.
LIVE_MIN_DISTINCT_EVENTS = 10
LIVE_MIN_INDEPENDENT_WINDOWS = 2


def event_key(event: dict) -> str:
    """One earthquake's identity: its USGS event id, or (when a row has none) the same
    time|lat|lon|mag key ``fetch_usgs_catalog_range`` de-duplicates on."""
    eid = str(event.get("id") or "").strip()
    if eid:
        return eid
    return (f"{event.get('time', '')}|{float(event['latitude']):.4f}|{float(event['longitude']):.4f}|"
            f"{float(event['mag']):.2f}")


def _utc(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        out = value
    else:
        out = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def non_overlapping_windows(windows: Iterable[tuple]) -> int:
    """The largest number of the given [issued_at, window_end) windows that are pairwise
    disjoint. Greedy by earliest end is optimal for intervals (and for equal-length windows
    earliest end is earliest start)."""
    spans = sorted((_utc(a), _utc(b)) for a, b in windows)
    spans.sort(key=lambda ab: ab[1])
    count, free_from = 0, None
    for start, end in spans:
        if free_from is None or start >= free_from:
            count += 1
            free_from = end
    return count


def version_counts(results: list[dict]) -> dict:
    """The independent counts of one model version's scored windows.

    Each result needs ``issued_at``, ``window_end`` and ``observed_event_ids`` (the scorer
    writes them); a result without ``observed_event_ids`` makes the distinct count unknown
    (None), never a guess.
    """
    ids: set[str] = set()
    known = True
    for r in results:
        if r.get("observed_event_ids") is None:
            known = False
            continue
        ids.update(str(i) for i in r["observed_event_ids"])
    return {
        "n_distinct_events": len(ids) if known else None,
        "n_event_windows": int(sum(int(r.get("n_observed_events") or 0) for r in results)),
        "n_independent_windows": non_overlapping_windows(
            (r["issued_at"], r["window_end"]) for r in results if r.get("issued_at") and r.get("window_end")),
    }


def quotable(record: dict | None) -> tuple[bool, str]:
    """Whether a live score of this record may be stated, and if not, why (in words)."""
    if not record:
        return False, "no window of this model version has closed yet"
    n_ev = record.get("n_distinct_events")
    n_ind = record.get("n_independent_windows")
    if n_ev is None or n_ind is None:
        return False, "the scorer's summary predates distinct-event counting"
    if int(n_ev) < LIVE_MIN_DISTINCT_EVENTS:
        return False, (f"{int(n_ev)} distinct M6+ earthquake{'s' if int(n_ev) != 1 else ''} so far "
                       f"(a score is quoted from {LIVE_MIN_DISTINCT_EVENTS})")
    if int(n_ind) < LIVE_MIN_INDEPENDENT_WINDOWS:
        return False, (f"{int(n_ind)} non-overlapping 30-day window{'s' if int(n_ind) != 1 else ''} so far "
                       f"(a score is quoted from {LIVE_MIN_INDEPENDENT_WINDOWS})")
    return True, ""
