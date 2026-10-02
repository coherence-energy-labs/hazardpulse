"""Which HRRR analysis a forecast issued at time t could actually have used -- ONE rule for
training, evaluation and the live scorer.

An analysis valid at hour h is not usable at h: the HRRR-Zarr archive publishes it about
1 h 40 min later. Measured 2026-10-02 on the eight most recent 00/06/12/18Z sfc analyses
(first object LastModified - valid time): 100.6-100.7 min every time. The v3 store had given
each storm observation the latest 3-hourly analysis VALID at or before it, so about 56% of rows
read an analysis that had not been published (found by the independent adversary pass,
2026-10-02). The rule here: an analysis is usable at t iff valid_time + LATENCY <= t, and the
newest usable one is taken if it is at most MAX_AGE old -- the previous UTC day's 21Z serves
the first hours of a day.

Training uses the 3-hourly analyses (00, 03, ..., 21Z) it cached; the live scorer is held to
the same cadence so that a live forecast sees exactly what a training row saw.
"""

from __future__ import annotations

import datetime as dt

HRRR_PUBLICATION_LATENCY_MIN = 100.0
ANALYSIS_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)
MAX_AGE_H = 3.0 + HRRR_PUBLICATION_LATENCY_MIN / 60.0     # one cadence step past availability


def usable_analysis(t: dt.datetime, available: list[tuple[str, int]] | None = None,
                    latency_min: float = HRRR_PUBLICATION_LATENCY_MIN,
                    max_age_h: float = MAX_AGE_H) -> tuple[str, int] | None:
    """The (YYYYMMDD, hour) of the newest analysis a forecast at ``t`` (UTC) could have used.

    ``available``: the (date, hour) analyses that exist (e.g. cached); None = every 3-hourly
    analysis is assumed to exist. Returns None if none is usable within ``max_age_h``.
    """
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    have = None if available is None else {(d, int(h)) for d, h in available}
    for back in range(0, int(max_age_h * 60 // 180) + 3):
        base = t - dt.timedelta(minutes=latency_min) - dt.timedelta(hours=3 * back)
        cand = base.replace(hour=(base.hour // 3) * 3, minute=0, second=0, microsecond=0)
        if (t - cand).total_seconds() / 3600.0 > max_age_h:
            return None
        if cand + dt.timedelta(minutes=latency_min) > t:
            continue
        key = (cand.strftime("%Y%m%d"), cand.hour)
        if have is None or key in have:
            return key
    return None


def live_candidates(now: dt.datetime) -> list[tuple[str, int]]:
    """The analyses a live run may try, newest first: every 3-hourly analysis already published
    and at most MAX_AGE_H old -- the order ``usable_analysis`` falls back through when one is missing."""
    if now.tzinfo is not None:
        now = now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    base = now - dt.timedelta(minutes=HRRR_PUBLICATION_LATENCY_MIN)
    cand = base.replace(hour=(base.hour // 3) * 3, minute=0, second=0, microsecond=0)
    out = []
    while (now - cand).total_seconds() / 3600.0 <= MAX_AGE_H:
        out.append((cand.strftime("%Y%m%d"), cand.hour))
        cand -= dt.timedelta(hours=3)
    return out
