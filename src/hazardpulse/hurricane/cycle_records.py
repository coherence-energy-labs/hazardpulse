"""What counts as the forecast of a storm-cycle, and what its outcome is
(docs/HURRICANE_RI_V9_PROGRAM.md, amendment 7).

ONE implementation for the three places that need it, so they cannot disagree:

* the live scorer (``scripts/fetch_and_score.py``), which forecasts in shadow every cycle that has no
  test record yet (rule 2, catch-up);
* the prospective test (``scripts/score_hurricane_v9_prospective.py``), rules 1-5;
* the verifier of the published numbers (``scripts/score_hurricane_prospective.py``), rules 1 and 4.

Rule 1. The unit is the storm-cycle: storm ``sid`` at synoptic time ``t``. Its record is the first one
made at or after ``t + 3 h 30 min``, judged by the time the record itself carries (the forecast file's
``issued_at``). Earlier records read preliminary inputs and never count; later ones are duplicates.
Rule 2. A catch-up record counts only if it was made before ``t + 12 h``.
Rule 4. The outcome is the best-track intensity change from ``t`` to ``t + 24 h`` for the storm's own
``t``; the time a forecast was made never enters.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

ADVISORY_DELAY = dt.timedelta(hours=3, minutes=30)      # rule 1: NHC's advisory for t is out at t + 3 h
CATCH_UP_LIMIT = dt.timedelta(hours=12)                  # rule 2: within one cycle of operational timing
CATCH_UP_START = dt.datetime(2026, 10, 4, 0)             # rule 2 (= the prospective test's first cycle)
OUTCOME_WINDOW = dt.timedelta(hours=24)                  # rule 4
RI_THRESHOLD_KT = 30.0
NHC_BASINS = frozenset({"AL", "EP", "CP"})
CATCH_UP_KEY = "shadow_catch_up"                         # where a forecast file keeps its catch-up records
REBUILT_KEY = "rebuilt_records"                          # where a rebuild file keeps its rule 3 records

EARLY = "made before t + 3 h 30 min (preliminary inputs)"
LATE_CATCH_UP = "catch-up made at or after t + 12 h"
DUPLICATE = "a later record of a cycle that already has one"
NO_TIME = "no record time"

_FID_TIME = re.compile(r"_(\d{8})_(\d{4})$")


def parse_utc(text: object) -> dt.datetime | None:
    """A naive-UTC datetime from an ISO string with or without ``Z``/offset; None if absent or bad."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t


def format_utc(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_time(artifact: Mapping[str, Any]) -> dt.datetime | None:
    """When a forecast file was made: its own ``issued_at`` (else the minute in its forecast id)."""
    t = parse_utc(artifact.get("issued_at"))
    if t is not None:
        return t
    m = _FID_TIME.search(str(artifact.get("forecast_id") or ""))
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M") if m else None


def storm_cycle(storm: Mapping[str, Any]) -> tuple[str, dt.datetime] | None:
    """``(storm id, synoptic time t)`` of a storm record, or None when it names no cycle."""
    sid = str(storm.get("storm_id") or "").strip().upper()
    t = parse_utc(storm.get("issue_time"))
    if not sid or t is None:
        return None
    return sid, t


def is_nhc_numbered(storm_id: str) -> bool:
    """An NHC-basin numbered storm (AL/EP/CP 01-49): the prospective test's population."""
    sid = str(storm_id).upper()
    return sid[:2] in NHC_BASINS and sid[2:4].isdigit() and 1 <= int(sid[2:4]) <= 49


def lag_hours(made_at: dt.datetime, t: dt.datetime) -> float:
    return round((made_at - t).total_seconds() / 3600.0, 4)


@dataclass
class Candidate:
    """One record of a storm-cycle: where it lives (``ref``) and when it was made."""

    storm_id: str
    cycle: dt.datetime
    made_at: dt.datetime | None
    catch_up: bool = False
    rebuilt: bool = False
    ref: Any = None
    order: int = 0

    @property
    def key(self) -> tuple[str, dt.datetime]:
        return self.storm_id, self.cycle


@dataclass
class Selection:
    """The test record of every storm-cycle (rules 1-2) and why each other record was set aside."""

    chosen: dict[tuple[str, dt.datetime], Candidate] = field(default_factory=dict)
    excluded: list[tuple[Candidate, str]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, why in self.excluded:
            out[why] = out.get(why, 0) + 1
        return out

    def cycles_without_a_record(self) -> list[tuple[str, dt.datetime]]:
        """Storm-cycles that had records, none of which qualifies (e.g. only preliminary ones)."""
        seen = {c.key for c, _ in self.excluded}
        return sorted(k for k in seen if k not in self.chosen)


def select(candidates: Iterable[Candidate]) -> Selection:
    """Rules 1 and 2: per storm-cycle, the first record made at or after ``t + 3 h 30 min`` (a
    catch-up record only if made before ``t + 12 h``); ties in time keep the order given."""
    groups: dict[tuple[str, dt.datetime], list[Candidate]] = {}
    for i, c in enumerate(candidates):
        c.order = i
        groups.setdefault(c.key, []).append(c)
    sel = Selection()
    for key, items in groups.items():
        items.sort(key=lambda c: (c.made_at is None, c.made_at or dt.datetime.max, c.order))
        for c in items:
            if c.made_at is None:
                why = NO_TIME
            elif c.made_at < c.cycle + ADVISORY_DELAY:
                why = EARLY
            elif c.catch_up and c.made_at >= c.cycle + CATCH_UP_LIMIT:
                why = LATE_CATCH_UP
            elif key in sel.chosen:
                why = DUPLICATE
            else:
                sel.chosen[key] = c
                continue
            sel.excluded.append((c, why))
    return sel


def has_shadow(entry: Mapping[str, Any]) -> bool:
    """A record of the prospective test carries at least one shadow forecast that was computed."""
    return any(str(k).endswith("_shadow") and isinstance(v, Mapping) and v.get("status") == "ok"
               for k, v in entry.items())


def file_candidates(artifact: Mapping[str, Any]) -> list[Candidate]:
    """Every shadow-carrying record in one forecast file -- its published storms and its catch-up
    records (or a rebuild file's rule 3 records) -- keyed by the record's own cycle (``issue_time``; the
    shadows' ``cycle`` for a record that has none), made at the file's own time. ``ref`` is
    ``(forecast id, list name, record)``."""
    made = record_time(artifact)
    out = []
    for kind in ("storms", CATCH_UP_KEY, REBUILT_KEY):
        for s in artifact.get(kind) or []:
            if not isinstance(s, Mapping) or not has_shadow(s):
                continue
            sc = storm_cycle(s)
            if sc is None:
                sh = next(v for k, v in s.items() if str(k).endswith("_shadow") and isinstance(v, Mapping)
                          and v.get("status") == "ok")
                t = parse_utc(sh.get("cycle"))
                sid = str(s.get("storm_id") or "").strip().upper()
                sc = (sid, t) if sid and t is not None else None
            if sc is None:
                continue
            out.append(Candidate(sc[0], sc[1], made, catch_up=bool(s.get("catch_up")),
                                 rebuilt=bool(s.get("rebuilt")), ref=(artifact.get("forecast_id"), kind, s)))
    return out


def due_catch_up_cycles(cycles: Iterable[dt.datetime], now: dt.datetime,
                        have: set[dt.datetime] | frozenset = frozenset(),
                        start: dt.datetime = CATCH_UP_START) -> list[dt.datetime]:
    """Rule 2: the cycles a run at ``now`` forecasts in shadow -- ``t >= start``, ``t + 3 h 30 min``
    passed, ``t + 12 h`` not yet passed, and no test record (``have``)."""
    return sorted({t for t in cycles
                   if t >= start and t + ADVISORY_DELAY <= now < t + CATCH_UP_LIMIT and t not in have})


# ------------------------------------------------------------------------------------------------
# Rule 4: the outcome
# ------------------------------------------------------------------------------------------------

def best_track_intensity(records: Iterable) -> dict[dt.datetime, float]:
    """``{time: max wind (kt)}`` from a b-deck's BEST tau-0 lines (the first line per time; a b-deck
    repeats each time once per wind-radii row). Zero or missing winds are not intensities."""
    out: dict[dt.datetime, float] = {}
    for r in records:
        if getattr(r, "model", None) != "BEST" or int(getattr(r, "tau_hours", -1)) != 0:
            continue
        if r.cycle in out or r.vmax_kt is None or float(r.vmax_kt) <= 0:
            continue
        out[r.cycle] = float(r.vmax_kt)
    return out


@dataclass(frozen=True)
class Outcome:
    v_t: float | None
    v_t24: float | None

    @property
    def decided(self) -> bool:
        return self.v_t is not None and self.v_t24 is not None

    @property
    def dv(self) -> float | None:
        return self.v_t24 - self.v_t if self.decided else None

    @property
    def ri(self) -> bool | None:
        return None if not self.decided else self.dv >= RI_THRESHOLD_KT


def outcome(best: Mapping[dt.datetime, float], t: dt.datetime) -> Outcome:
    """The best-track intensity at ``t`` and at ``t + 24 h`` exactly (rule 4). A missing fix leaves the
    outcome undecided -- never "no RI"."""
    return Outcome(best.get(t), best.get(t + OUTCOME_WINDOW))
