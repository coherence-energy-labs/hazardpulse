#!/usr/bin/env python3
"""Decide which scorers are due and dispatch them (run by .github/workflows/scheduler.yml every 15 minutes).

GitHub's own cron is best-effort, and for this repository it was far from it. Measured 2026-10-01..05
from the run history, scheduled runs against their crons:

    tornado       cron every 2 h      median gap 5.5 h, worst 8.6 h   (a 60-minute forecast)
    hurricane     cron 4 a day        8 runs in 4 days, worst gap 23.3 h, with three active storms
    verification  cron every 4 h      median gap 5.9 h, worst 10.9 h
    earthquake    cron every 6 h      median gap 7.1 h, worst 9.5 h

Most missing runs were never created at all (GitHub drops scheduled runs under load, worst at the top of
the hour); a few were cancelled by the shared scoring queue. So the scorers no longer depend on hitting a
cron slot. This script runs at four off-peak minutes an hour, asks GitHub when each scorer last succeeded,
and dispatches the ones that are due and not already queued or running. A dropped attempt now costs 15
minutes, not a cycle.

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


def run_state(workflow: str) -> tuple[dt.datetime | None, bool]:
    """(start of the last successful run, whether a run is queued or in progress)."""
    runs = _gh_json(["run", "list", "--workflow", workflow, "--branch", "main", "--limit", "30",
                     "--json", "status,conclusion,createdAt,startedAt"]) or []
    in_flight = any(r.get("status") in ("queued", "in_progress", "waiting", "pending", "requested") for r in runs)
    ok = [_parse(r.get("startedAt") or r["createdAt"]) for r in runs
          if r.get("status") == "completed" and r.get("conclusion") == "success"]
    return (max(ok) if ok else None), in_flight


def nws_tornado_warnings() -> int | None:
    req = urllib.request.Request(NWS_TORNADO_WARNINGS,
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return len(json.load(r).get("features") or [])
    except Exception as exc:                              # unreachable is not "none in effect"
        print(f"  NWS alerts unreachable: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="decide and print, dispatch nothing")
    ap.add_argument("--ref", default="main")
    args = ap.parse_args(argv)
    now = dt.datetime.now(UTC)
    pulse_path = ROOT / "dist" / "data" / "live-pulse.json"
    pulse = json.loads(pulse_path.read_text(encoding="utf-8")) if pulse_path.exists() else None
    elevated, why = tornado_elevated(pulse, nws_tornado_warnings())
    print(f"{now:%Y-%m-%d %H:%MZ}  tornado risk: {'ELEVATED' if elevated else 'quiet'} ({why})")
    failed = False
    for hazard, workflow in WORKFLOWS.items():
        try:
            last, in_flight = run_state(workflow)
        except Exception as exc:
            print(f"  {hazard:12s} UNKNOWN  could not read the run history: {exc}")
            failed = True
            continue
        d = decide(hazard, now, last, in_flight, elevated=elevated)
        print(f"  {hazard:12s} {'DISPATCH' if d.due else 'wait    '} {d.reason}")
        if d.due and not args.dry_run:
            subprocess.run(["gh", "workflow", "run", workflow, "--ref", args.ref], check=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
