"""The scorer scheduler (scripts/ci/schedule.py): each hazard is dispatched exactly when it is due.

GitHub's cron dropped or delayed most scheduled scoring runs (tornado: a 2-hour cron ran every 5.5 h at
the median, 2026-10-01..05), so a scheduler that runs four times an hour decides from the run history.
These tests pin the decision: due when it should be, not due when it should not -- each rule both ways.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hp_ci_schedule", ROOT / "scripts" / "ci" / "schedule.py")
schedule = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = schedule            # a dataclass looks its module up while it is being defined
_spec.loader.exec_module(schedule)

UTC = dt.timezone.utc


def t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def due(hazard, now, last, in_flight=False, elevated=False):
    return schedule.decide(hazard, t(now), t(last) if last else None, in_flight, elevated=elevated).due


def test_a_run_already_queued_or_running_is_never_doubled():
    for hazard in schedule.WORKFLOWS:
        assert not due(hazard, "2026-10-05 12:00", "2026-10-01 00:00", in_flight=True)


def test_a_scorer_with_no_successful_run_is_due():
    for hazard in schedule.WORKFLOWS:
        assert due(hazard, "2026-10-05 12:00", None)


def test_tornado_every_two_hours_when_quiet_and_every_half_hour_when_elevated():
    assert not due("tornado", "2026-10-05 12:00", "2026-10-05 10:30")             # 90 min, quiet
    assert due("tornado", "2026-10-05 12:00", "2026-10-05 10:00")                 # 2 h
    assert due("tornado", "2026-10-05 12:00", "2026-10-05 10:04")                 # within the tolerance
    assert not due("tornado", "2026-10-05 12:00", "2026-10-05 11:40", elevated=True)
    assert due("tornado", "2026-10-05 12:00", "2026-10-05 11:30", elevated=True)
    assert due("tornado", "2026-10-05 12:00", "2026-10-05 10:30", elevated=True)  # the case quiet would wait on


def test_hurricane_once_per_nhc_cycle_from_three_and_a_half_hours_after_it():
    # 12Z cycle serves from 15:30Z
    assert not due("hurricane", "2026-10-05 15:20", "2026-10-05 09:35")   # 12Z not yet servable; 06Z served
    assert due("hurricane", "2026-10-05 15:40", "2026-10-05 09:35")       # 12Z servable, not served
    assert not due("hurricane", "2026-10-05 16:10", "2026-10-05 15:40")   # 12Z served by the 15:40 run
    assert not due("hurricane", "2026-10-05 21:20", "2026-10-05 15:40")   # 18Z not servable until 21:30
    assert due("hurricane", "2026-10-06 04:00", "2026-10-05 15:40")       # 18Z and 00Z both missed: serve now
    # a cycle served late is still served once
    assert not due("hurricane", "2026-10-05 20:00", "2026-10-05 19:55")


def test_earthquake_once_per_six_hour_slot():
    assert due("earthquake", "2026-10-05 06:20", "2026-10-05 00:40")
    assert not due("earthquake", "2026-10-05 07:00", "2026-10-05 06:20")
    assert not due("earthquake", "2026-10-05 11:59", "2026-10-05 06:20")
    assert due("earthquake", "2026-10-05 12:01", "2026-10-05 06:20")


def test_verification_every_four_hours():
    assert not due("verification", "2026-10-05 12:00", "2026-10-05 09:00")
    assert due("verification", "2026-10-05 13:00", "2026-10-05 09:00")


def test_tornado_risk_is_elevated_by_the_forecast_or_a_warning_and_unknown_is_not_evidence():
    pulse = lambda p: {"hazards": [{"key": "eq", "probability": 0.9}, {"key": "to", "probability": p}]}
    assert schedule.tornado_elevated(pulse(0.1667), 0)[0]
    assert not schedule.tornado_elevated(pulse(0.02), 0)[0]
    assert schedule.tornado_elevated(pulse(0.02), 3)[0]
    assert not schedule.tornado_elevated(pulse(0.02), None)[0]          # NWS unreachable: not elevated...
    assert "NWS unreachable" in schedule.tornado_elevated(pulse(0.02), None)[1]   # ...and said so
    assert not schedule.tornado_elevated(None, 0)[0]
    assert schedule.tornado_elevated(None, 1)[0]
    # another hazard's number never counts
    assert not schedule.tornado_elevated({"hazards": [{"key": "eq", "probability": 0.9}]}, 0)[0]


def test_every_scheduled_workflow_exists_and_has_no_cron_of_its_own():
    """The scheduler is the only clock: a scorer with its own cron as well would run twice."""
    for workflow in schedule.WORKFLOWS.values():
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "workflow_dispatch" in text, workflow
        assert "cron:" not in text, workflow
    sched = (ROOT / ".github" / "workflows" / "scheduler.yml").read_text(encoding="utf-8")
    assert "scripts/ci/schedule.py" in sched and "actions: write" in sched
