#!/usr/bin/env python3
"""Fail loudly if any hazard scorer has gone silent (run by .github/workflows/liveness-check.yml).

Freshness is read from each hazard's OWN ``updated_at`` in dist/data/live-pulse.json, which
build_site_artifacts.py stamps from that hazard's own artifact (live-storms.json,
live-tornadoes.json, the earthquake replay). There is deliberately no fallback to the
pulse-level ``updated_at``: every scorer rewrites that one, so the previous check -- which
fell back to it -- reported the hurricane product fresh on every run for four months after
its last successful cycle (2026-05-26). A hazard with no timestamp of its own FAILS.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PULSE_PATH = ROOT / "dist" / "data" / "live-pulse.json"
REPLAY_DIR = ROOT / "dist" / "data" / "replay"

# Max acceptable age per hazard (hours) -- the scheduled cadence plus slack.
MAX_AGE_HOURS = {
    "eq": 12,   # earthquake scorer runs every 6h
    "hu": 36,   # hurricane scorer runs daily
    "to": 6,    # tornado scorer runs every 2h
}


def _parse_utc(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def evaluate_freshness(
    pulse: dict,
    now: dt.datetime,
    max_age_hours: dict[str, float] = MAX_AGE_HOURS,
) -> tuple[list[str], list[str]]:
    """Return (report lines, failures) for every hazard in ``max_age_hours``.

    A hazard fails when it is absent from the pulse, has no parseable ``updated_at`` of its
    own, is older than its limit, or claims a time more than 1 h in the future.
    """
    lines: list[str] = []
    failures: list[str] = []
    hazards = {h.get("key"): h for h in pulse.get("hazards", []) if isinstance(h, dict)}
    for key, max_age in max_age_hours.items():
        hazard = hazards.get(key)
        if hazard is None:
            failures.append(f"{key}: hazard missing from live-pulse.json")
            continue
        raw = hazard.get("updated_at")
        stamp = _parse_utc(raw)
        if stamp is None:
            failures.append(
                f"{key}: no per-hazard updated_at (got {raw!r}); the pulse-level timestamp "
                "is refreshed by every scorer and cannot show that THIS one ran"
            )
            continue
        age_h = (now - stamp).total_seconds() / 3600.0
        status = "OK" if -1.0 <= age_h < max_age else "STALE" if age_h >= max_age else "FUTURE"
        lines.append(f"  {key}: last_update={raw}  age={age_h:.1f}h  limit={max_age}h  {status}")
        if age_h >= max_age:
            failures.append(f"{key} is stale: {age_h:.1f}h since last update (threshold: {max_age}h)")
        elif age_h < -1.0:
            failures.append(f"{key} claims a future update time {raw} ({-age_h:.1f}h ahead)")
    return lines, failures


def evaluate_tornado_tier(replay_dir: Path = REPLAY_DIR, n: int = 5) -> tuple[list[str], list[str]]:
    """Ensure tier1_ml is winning, not silently falling back, over the last ``n`` replays."""
    lines: list[str] = []
    failures: list[str] = []
    replays = sorted(replay_dir.glob("to_fcst_*.json"), reverse=True)[:n]
    tier_counts: dict[str, int] = {}
    for r in replays:
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = d.get("scoring_tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    if tier_counts:
        lines.append(f"  Tornado tier distribution (last {len(replays)} replays): {tier_counts}")
        if tier_counts.get("tier1_ml", 0) == 0 and len(replays) >= 3:
            failures.append(
                "Tornado tier1_ml has NOT triggered in the last "
                f"{len(replays)} runs (got {tier_counts}). HRRR or "
                "trained-model loading is likely broken."
            )
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    pulse = json.loads(PULSE_PATH.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc)
    lines, failures = evaluate_freshness(pulse, now)
    tier_lines, tier_failures = evaluate_tornado_tier()
    for line in lines:
        print(line)
    if tier_lines:
        print()
        for line in tier_lines:
            print(line)
    failures += tier_failures
    if failures:
        print()
        print("LIVENESS CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print()
    print("All three scorers are fresh; tier1_ml is firing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
