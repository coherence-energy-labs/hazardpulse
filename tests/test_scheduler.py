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


def test_main_runs_are_filtered_by_their_own_branch_never_by_githubs_query():
    """`gh run list --branch main` returned nothing after 2026-09-23 while today's runs all said main."""
    runs = [{"databaseId": 1, "headBranch": "main"}, {"databaseId": 2, "headBranch": "audit/x"},
            {"databaseId": 3, "headBranch": None}]
    assert [r["databaseId"] for r in schedule.main_runs(runs)] == [1]
    src = (ROOT / "scripts" / "ci" / "schedule.py").read_text(encoding="utf-8")
    assert '"--branch"' not in src, "GitHub's branch filter served a stale index; filter on headBranch"


def test_liveness_is_checked_every_four_hours_on_the_same_clock():
    assert not due("liveness", "2026-10-05 12:00", "2026-10-05 09:00")
    assert due("liveness", "2026-10-05 13:00", "2026-10-05 09:00")
    assert "liveness" in schedule.MONITORS and "liveness" not in schedule.WORKFLOWS   # not a gated scorer
    text = (ROOT / ".github" / "workflows" / schedule.MONITORS["liveness"]).read_text(encoding="utf-8")
    assert "workflow_dispatch" in text and "scripts/check_liveness.py" in text


def test_one_missed_run_shows_on_the_status_page_and_in_the_liveness_check():
    """The limits are the slot plus slack, small enough that ONE missed run is overdue: a missed hurricane
    cycle never reached the old 14 h limit, because the next cycle's run landed first."""
    from hazardpulse.site.hazards import MAX_AGE_HOURS
    assert MAX_AGE_HOURS["to"] < 2 * schedule.TORNADO_QUIET.total_seconds() / 3600
    assert MAX_AGE_HOURS["hu"] < 12 and MAX_AGE_HOURS["eq"] < 12          # two 6-hour slots
    assert MAX_AGE_HOURS["hu"] > 6 + 1 and MAX_AGE_HOURS["eq"] > 6 + 1    # but not one on time


def test_the_shared_queue_is_never_overfilled():
    """GitHub's concurrency group holds one running and ONE pending job; a job joining while one is pending
    cancels it (2026-10-05 17:31Z: four dispatched at once, hurricane and earthquake cancelled within 5 s)."""
    assert schedule.queue_room(0, 0) == 2          # one runs, one waits
    assert schedule.queue_room(1, 0) == 1          # it waits behind the running one
    assert schedule.queue_room(0, 1) == 0 and schedule.queue_room(1, 1) == 0
    every = list(schedule.WORKFLOWS)
    assert schedule.by_priority(list(reversed(every)), False) == ["hurricane", "tornado", "earthquake", "verification"]
    assert schedule.by_priority(every, True)[0] == "tornado"
    assert sorted(schedule.PRIORITY) == sorted(schedule.WORKFLOWS)


def _run_main(monkeypatch, queue, argv=()):
    """main() with every scorer and the monitor due, the queue as given; returns the workflows dispatched."""
    dispatched = []
    monkeypatch.setattr(schedule, "nws_tornado_warnings", lambda: 0)
    monkeypatch.setattr(schedule, "run_state", lambda wf, exclude_run_id=None: (None, False))
    monkeypatch.setattr(schedule, "monitor_state", lambda wf: (None, False))
    monkeypatch.setattr(schedule, "queue_state", lambda exclude_run_id=None: queue)
    monkeypatch.setattr(schedule.subprocess, "run", lambda cmd, check=False: dispatched.append(cmd[3]))
    schedule.main(list(argv))
    return dispatched


def test_the_scheduler_dispatches_only_what_the_queue_can_hold(monkeypatch):
    assert _run_main(monkeypatch, (0, 0)) == ["liveness-check.yml", "hurricane-score.yml", "tornado-score.yml"]
    assert _run_main(monkeypatch, (1, 0)) == ["liveness-check.yml", "hurricane-score.yml"]
    assert _run_main(monkeypatch, (1, 1)) == ["liveness-check.yml"]         # the monitor is not in the queue
    assert _run_main(monkeypatch, (0, 0), ["--dry-run"]) == []


def test_a_scorers_own_gate_does_not_join_a_full_queue(monkeypatch):
    monkeypatch.setattr(schedule, "run_state", lambda wf, exclude_run_id=None: (None, False))
    now = t("2026-10-05 18:00")
    for queue, joins in (((0, 0), True), ((1, 0), True), ((1, 1), False), ((0, 1), False)):
        monkeypatch.setattr(schedule, "queue_state", lambda exclude_run_id=None, q=queue: q)
        assert schedule.gate("hurricane", now, False) is joins, queue


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


def test_every_scorer_gates_its_own_cron_and_keeps_the_queue_for_scoring_jobs():
    """Two independent clocks, one rule: each scorer's off-peak cron starts a `due` gate that lets the scoring
    job run only when due (GitHub left the new scheduler's own cron unfired for its first 3 slots), and the
    shared queue holds scoring jobs only. Plain text, not a YAML parser (CI installs no PyYAML)."""
    for hazard, workflow in schedule.WORKFLOWS.items():
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "workflow_dispatch" in text and "cron:" in text, workflow
        cron = text.split("cron:")[1].split("\n")[0]
        assert not cron.strip().strip('"').startswith("0 "), (workflow, "the top of the hour is when GitHub drops runs")
        assert f"python scripts/ci/schedule.py --hazard {hazard}" in text, workflow
        assert "needs: due" in text and "needs.due.outputs.run == 'true'" in text, workflow
        top, jobs = text.split("\njobs:\n", 1)
        assert "concurrency:" not in top, (workflow, "a workflow-level queue would hold the gates too")
        assert "group: hazardpulse-scoring" in jobs.split("\n  score:\n", 1)[1], workflow
    sched = (ROOT / ".github" / "workflows" / "scheduler.yml").read_text(encoding="utf-8")
    assert "scripts/ci/schedule.py" in sched and "actions: write" in sched


def test_a_gate_does_not_see_itself_and_a_gate_that_skipped_is_not_a_scoring_run():
    def run(i, status, conclusion=None, at="2026-10-05T10:00:00Z", scored=True):
        return {"databaseId": i, "status": status, "conclusion": conclusion, "createdAt": at, "startedAt": at,
                "scored": scored}
    runs = [run(9, "in_progress", at="2026-10-05T12:13:00Z"),                       # the asking gate itself
            run(8, "completed", "success", "2026-10-05T11:43:00Z", scored=False),    # a gate that skipped
            run(7, "completed", "success", "2026-10-05T10:05:00Z")]                  # the last real run
    last, in_flight = schedule.summarize_runs(runs, exclude_run_id=9)
    assert in_flight is False and last == t("2026-10-05 10:05")
    last, in_flight = schedule.summarize_runs(runs)                                  # seen from outside: busy
    assert in_flight is True
