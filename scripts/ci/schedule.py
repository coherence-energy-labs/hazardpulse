#!/usr/bin/env python3
"""Decide which scorers are due and dispatch them (run by .github/workflows/scheduler.yml every 15 minutes).

GitHub's own cron is best-effort, and for this repository it was far from it. Measured 2026-10-01..05
from the run history, scheduled runs against their crons:

    tornado       cron every 2 h      median gap 5.5 h, worst 8.6 h   (a 60-minute forecast)
    hurricane     cron 4 a day        8 runs in 4 days, worst gap 23.3 h, with three active storms
    verification  cron every 4 h      median gap 5.9 h, worst 10.9 h
    earthquake    cron every 6 h      median gap 7.1 h, worst 9.5 h

Most missing runs were never created at all (GitHub drops scheduled runs under load, worst at the top of
the hour); a few were cancelled by the shared scoring queue. So no scorer depends on hitting a cron slot.
Two independent paths decide, by the same rule:

* each scorer's own frequent off-peak cron starts a seconds-long `due` job (``--hazard``) that lets the
  scoring job run only if it is due and no other run of it is queued or running;
* scheduler.yml runs this script at four other off-peak minutes an hour and dispatches what is due.

Neither can double a run, and a dropped attempt costs minutes, not a cycle. (The first version had the
scheduler alone: GitHub then did not fire its own new cron for its first 3 slots, 2026-10-05.)

Neither may overfill the shared scoring queue either. Every scoring job is in one GitHub concurrency group,
which holds one running job and ONE pending; a job that joins while another is pending cancels that one.
On 2026-10-05 at 17:31Z the scheduler dispatched four scorers within 15 s: tornado ran, verification waited,
and hurricane and earthquake were cancelled within 5 s of being queued. They stayed due, so the next tick
would have cancelled them the same way. Now each joins only while the queue has room (``queue_room``), in
priority order (``by_priority``); whatever waits is still due at the next tick.

When each is due:

* tornado: every 2 hours; every 30 minutes while tornado risk is elevated -- the latest forecast has a
  storm at 5% or more, or the National Weather Service has a tornado warning in effect anywhere. The
  forecast covers the next 60 minutes, so its freshness should follow the hazard; a quiet forecast two
  hours old is still nearly right, and the repository grows ~170 KB per tornado run (measured), which a
  fixed 30-minute cadence would turn into ~3 GB a year.
* hurricane: once per NHC cycle (00/06/12/18 UTC), from 3 h 30 min after it -- after the cycle's advisory
  and, for over 90% of storms, its SHIPS text (the lag measured in hurricane-score.yml).
* earthquake: once per 6-hour slot (the forecast is stamped at 00/06/12/18 UTC).
* verification: every 4 hours.

Standard library only; GitHub access is through the `gh` CLI with the workflow's token.

    python scripts/ci/schedule.py              # decide and dispatch
    python scripts/ci/schedule.py --dry-run    # decide and print only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UTC = dt.timezone.utc
TOLERANCE = dt.timedelta(minutes=5)          # a run that started a little early still counts

TORNADO_QUIET = dt.timedelta(hours=2)
TORNADO_ELEVATED = dt.timedelta(minutes=30)
TORNADO_ELEVATED_PROBABILITY = 0.05
HURRICANE_DELAY = dt.timedelta(hours=3, minutes=30)
EARTHQUAKE_SLOT = dt.timedelta(hours=6)
VERIFICATION_EVERY = dt.timedelta(hours=4)
NWS_TORNADO_WARNINGS = "https://api.weather.gov/alerts/active?event=Tornado%20Warning"
USER_AGENT = "HazardPulse scheduler (+https://hazardpulse.com; josh@coherenceenergylabs.com)"

WORKFLOWS = {
    "tornado": "tornado-score.yml",
    "hurricane": "hurricane-score.yml",
    "earthquake": "earthquake-score.yml",
    "verification": "verification-score.yml",
}
# monitors: dispatched on the same clock, every LIVENESS_EVERY, counted by their last COMPLETED run (a failed
# liveness check is the signal, not a reason to run it again at once -- each failure is an email)
MONITORS = {"liveness": "liveness-check.yml"}
LIVENESS_EVERY = dt.timedelta(hours=4)
# who joins the shared scoring queue first when several are due: a hurricane cycle has a deadline (its catch-up
# window closes at t + 12 h), the tornado forecast covers only the next hour, the earthquake slot is 6 hours,
# and verification has no deadline. Elevated tornado risk goes first.
PRIORITY = ("hurricane", "tornado", "earthquake", "verification")


@dataclass(frozen=True)
class Decision:
    hazard: str
    due: bool
    reason: str


def _floor(t: dt.datetime, step: dt.timedelta) -> dt.datetime:
    epoch = dt.datetime(1970, 1, 1, tzinfo=UTC)
    return epoch + ((t - epoch) // step) * step


def decide(hazard: str, now: dt.datetime, last_success: dt.datetime | None, in_flight: bool,
           elevated: bool = False) -> Decision:
    """Whether `hazard`'s scorer should be dispatched now. Pure: every input is an argument.

    last_success is when the scorer's last successful run STARTED (None: never, or not in the history).
    """
    if in_flight:
        return Decision(hazard, False, "a run is already queued or running")
    if last_success is None:
        return Decision(hazard, True, "no successful run on record")
    age = now - last_success
    if hazard == "tornado":
        every = TORNADO_ELEVATED if elevated else TORNADO_QUIET
        mode = "elevated risk" if elevated else "quiet"
        if age + TOLERANCE >= every:
            return Decision(hazard, True, f"{mode}: last run {_minutes(age)} ago, due every {_minutes(every)}")
        return Decision(hazard, False, f"{mode}: last run {_minutes(age)} ago, due every {_minutes(every)}")
    if hazard == "hurricane":
        # the most recent cycle whose serving time has come
        cycle = _floor(now - HURRICANE_DELAY, dt.timedelta(hours=6))
        serve_from = cycle + HURRICANE_DELAY
        if last_success + TOLERANCE < serve_from:
            return Decision(hazard, True, f"cycle {cycle:%d %H}Z not yet served (serving from {serve_from:%H:%M}Z)")
        return Decision(hazard, False, f"cycle {cycle:%d %H}Z already served")
    if hazard == "earthquake":
        slot = _floor(now, EARTHQUAKE_SLOT)
        if last_success + TOLERANCE < slot:
            return Decision(hazard, True, f"slot {slot:%d %H}Z not yet run")
        return Decision(hazard, False, f"slot {slot:%d %H}Z already run")
    if hazard == "verification":
        if age + TOLERANCE >= VERIFICATION_EVERY:
            return Decision(hazard, True, f"last run {_minutes(age)} ago, due every 4 h")
        return Decision(hazard, False, f"last run {_minutes(age)} ago, due every 4 h")
    if hazard == "liveness":
        if age + TOLERANCE >= LIVENESS_EVERY:
            return Decision(hazard, True, f"last check {_minutes(age)} ago, due every 4 h")
        return Decision(hazard, False, f"last check {_minutes(age)} ago, due every 4 h")
    raise ValueError(f"unknown hazard {hazard!r}")


def tornado_elevated(pulse: dict | None, nws_warnings: int | None) -> tuple[bool, str]:
    """Elevated if the latest forecast has a storm at TORNADO_ELEVATED_PROBABILITY or more, or the NWS has a
    tornado warning in effect. An input that could not be read is not evidence of either."""
    top = None
    for h in (pulse or {}).get("hazards") or []:
        if h.get("key") == "to" and isinstance(h.get("probability"), (int, float)):
            top = float(h["probability"])
    elevated = (top is not None and top >= TORNADO_ELEVATED_PROBABILITY) or bool(nws_warnings)
    why = [f"top storm {top:.1%}" if top is not None else "top storm unknown",
           f"{nws_warnings} NWS tornado warning(s)" if nws_warnings is not None else "NWS unreachable"]
    return elevated, ", ".join(why)


def _minutes(td: dt.timedelta) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _gh_json(args: list[str]):
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out or "null")


IN_FLIGHT = ("queued", "in_progress", "waiting", "pending", "requested")


def summarize_runs(runs: list[dict], exclude_run_id: int | None = None) -> tuple[dt.datetime | None, bool]:
    """(start of the last successful SCORING run, whether another run is queued or in progress).

    ``exclude_run_id`` is the run asking (a scorer's own gate must not see itself as "already running").
    A run counts as a scoring run only if its scoring job ran: a cron-triggered run whose gate said "not
    due" also ends in success, and counting it would push the next due time back forever.
    """
    others = [r for r in runs if exclude_run_id is None or int(r.get("databaseId") or 0) != int(exclude_run_id)]
    in_flight = any(r.get("status") in IN_FLIGHT for r in others)
    ok = [_parse(r.get("startedAt") or r["createdAt"]) for r in others
          if r.get("status") == "completed" and r.get("conclusion") == "success" and r.get("scored", True)]
    return (max(ok) if ok else None), in_flight


def _scored(run_id: int) -> bool:
    """Whether a completed run's scoring job actually ran (not skipped by its gate)."""
    jobs = (_gh_json(["run", "view", str(run_id), "--json", "jobs"]) or {}).get("jobs") or []
    return any(j.get("name") == "score" and j.get("conclusion") == "success" for j in jobs)


def main_runs(runs: list[dict]) -> list[dict]:
    """The runs of the main branch, filtered HERE, by each run's own ``headBranch``.

    Never ask GitHub to filter (``gh run list --branch main``): on 2026-10-05 that query returned nothing
    after 2026-09-23 while today's runs, listed unfiltered, all said ``headBranch: main``. Every scheduler
    decision had rested on runs two weeks old -- the cause of the "slot not yet run" dispatch at 03:32Z."""
    return [r for r in runs if r.get("headBranch") == "main"]


def _runs(workflow: str, limit: int) -> list[dict]:
    return main_runs(_gh_json(["run", "list", "--workflow", workflow, "--limit", str(limit), "--json",
                               "databaseId,status,conclusion,createdAt,startedAt,event,headBranch"]) or [])


def monitor_state(workflow: str) -> tuple[dt.datetime | None, bool]:
    """(start of the last COMPLETED run, whatever its conclusion; whether a run is queued or running)."""
    runs = _runs(workflow, 30)
    in_flight = any(r.get("status") in IN_FLIGHT for r in runs)
    done = [_parse(r.get("startedAt") or r["createdAt"]) for r in runs if r.get("status") == "completed"
            and r.get("conclusion") in ("success", "failure")]
    return (max(done) if done else None), in_flight


def run_state(workflow: str, exclude_run_id: int | None = None) -> tuple[dt.datetime | None, bool]:
    runs = _runs(workflow, 60)
    # a scheduled run may have been a gate that found nothing due: look at its jobs (newest first, and
    # only until the newest run that really scored -- older ones cannot change the answer)
    for r in sorted(runs, key=lambda r: r.get("createdAt") or "", reverse=True):
        if r.get("status") == "completed" and r.get("conclusion") == "success":
            r["scored"] = r.get("event") != "schedule" or _scored(int(r["databaseId"]))
            if r["scored"]:
                break
    return summarize_runs(runs, exclude_run_id)


def queue_room(running: int, waiting: int) -> int:
    """How many scoring jobs may join the shared queue without cancelling one: it holds one running and one
    pending, and a job joining while one is pending cancels it -- so none while anything waits."""
    if waiting:
        return 0
    return 1 if running else 2


def by_priority(hazards: list[str], elevated: bool) -> list[str]:
    order = (("tornado",) if elevated else ()) + PRIORITY
    return sorted(hazards, key=order.index)


def _score_job_status(run_id: int) -> str | None:
    jobs = (_gh_json(["run", "view", str(run_id), "--json", "jobs"]) or {}).get("jobs") or []
    return next((j.get("status") for j in jobs if j.get("name") == "score"), None)


def queue_state(exclude_run_id: int | None = None) -> tuple[int, int]:
    """``(running, waiting)``: scoring runs of main in flight, across every scorer. Running: its scoring job
    is in progress. Waiting: anything else in flight -- queued, its gate still deciding, or its scoring job
    pending in the group -- because joining could cancel it."""
    running = waiting = 0
    for workflow in WORKFLOWS.values():
        for r in _runs(workflow, 10):
            if r.get("status") not in IN_FLIGHT or (exclude_run_id and int(r["databaseId"]) == exclude_run_id):
                continue
            if _score_job_status(int(r["databaseId"])) == "in_progress":
                running += 1
            else:
                waiting += 1
    return running, waiting


def nws_tornado_warnings() -> int | None:
    req = urllib.request.Request(NWS_TORNADO_WARNINGS,
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return len(json.load(r).get("features") or [])
    except Exception as exc:                              # unreachable is not "none in effect"
        print(f"  NWS alerts unreachable: {exc}", file=sys.stderr)
        return None


def gate(hazard: str, now: dt.datetime, elevated: bool) -> bool:
    """A scorer's own gate (its cron-triggered `due` job): run only if due, never doubling a run already
    queued or running. If GitHub's run history cannot be read, run: a duplicate costs a commit, a skip
    costs a forecast."""
    import os
    me = os.environ.get("GITHUB_RUN_ID")
    try:
        last, in_flight = run_state(WORKFLOWS[hazard], exclude_run_id=int(me) if me else None)
    except Exception as exc:
        print(f"  {hazard}: run history unreadable ({exc}); running")
        return True
    d = decide(hazard, now, last, in_flight, elevated=elevated)
    if d.due:
        try:
            running, waiting = queue_state(exclude_run_id=int(me) if me else None)
        except Exception as exc:
            print(f"  {hazard}: scoring queue unreadable ({exc}); running")
            return True
        if queue_room(running, waiting) < 1:
            print(f"  {hazard:12s} skip {d.reason}, but the scoring queue is full ({running} running, {waiting} "
                  "waiting): joining would cancel the waiting run. Still due at the next tick.")
            return False
    print(f"  {hazard:12s} {'RUN ' if d.due else 'skip'} {d.reason}")
    return d.due


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="decide and print, dispatch nothing")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--hazard", choices=sorted(WORKFLOWS),
                    help="gate mode: decide for this scorer only and write run=true|false to $GITHUB_OUTPUT")
    args = ap.parse_args(argv)
    now = dt.datetime.now(UTC)
    pulse_path = ROOT / "dist" / "data" / "live-pulse.json"
    pulse = json.loads(pulse_path.read_text(encoding="utf-8")) if pulse_path.exists() else None
    needs_risk = args.hazard in (None, "tornado")
    elevated, why = tornado_elevated(pulse, nws_tornado_warnings()) if needs_risk else (False, "not needed")
    print(f"{now:%Y-%m-%d %H:%MZ}  tornado risk: {'ELEVATED' if elevated else 'quiet'} ({why})")
    if args.hazard:
        import os
        run = gate(args.hazard, now, elevated)
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"run={'true' if run else 'false'}\n")
        return 0
    failed = False
    due_scorers: list[str] = []
    for hazard, workflow in {**WORKFLOWS, **MONITORS}.items():
        try:
            last, in_flight = monitor_state(workflow) if hazard in MONITORS else run_state(workflow)
        except Exception as exc:
            print(f"  {hazard:12s} UNKNOWN  could not read the run history: {exc}")
            failed = True
            continue
        d = decide(hazard, now, last, in_flight, elevated=elevated)
        # what the decision rested on, so a wrong one can be traced (2026-10-05 03:32Z: earthquake was
        # dispatched as "slot not yet run" although a run had succeeded at 02:04Z; not reproducible locally)
        seen = f"last scoring run {last:%d %H:%MZ}" if last else "no scoring run on record"
        print(f"  {hazard:12s} {'due     ' if d.due else 'wait    '} {d.reason}  [{seen}; in flight: {in_flight}]")
        if not d.due:
            continue
        if hazard in MONITORS:                       # not in the scoring queue: nothing to cancel
            _dispatch(hazard, workflow, args)
        else:
            due_scorers.append(hazard)
    if due_scorers:
        try:
            running, waiting = queue_state()
            room = queue_room(running, waiting)
            state = f"{running} running, {waiting} waiting"
        except Exception as exc:                     # one dispatch can cancel at most one waiting run
            room, state = 1, f"unreadable ({exc})"
            failed = True
        print(f"  scoring queue: {state} -> room for {room}")
        for i, hazard in enumerate(by_priority(due_scorers, elevated)):
            if i < room:
                _dispatch(hazard, WORKFLOWS[hazard], args)
            else:
                print(f"  {hazard:12s} HOLD     due, but the scoring queue is full: at the next tick")
    return 1 if failed else 0


def _dispatch(hazard: str, workflow: str, args) -> None:
    print(f"  {hazard:12s} DISPATCH{' (dry run)' if args.dry_run else ''}")
    if not args.dry_run:
        subprocess.run(["gh", "workflow", "run", workflow, "--ref", args.ref], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
