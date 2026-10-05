#!/usr/bin/env python3
"""Background USGS streaming → local cache ingestor.

Runs as a long-lived process (started by the streaming-ingest GitHub
Actions workflow, or by an operator on a worker VM). Polls USGS every
60 seconds, dedupes new events, appends them to:

  - .cache/earthquake/streamed_events.jsonl (append-only)
  - .cache/earthquake/usgs_catalog_<year>.csv (merged into the year's
    file the next time the batch scorer normalises the cache)

The live earthquake scorer continues to read its existing per-year
CSVs; the streaming ingestor just keeps the most-recent events warm
between scorer runs (so the gap from a 6-hour scoring cadence shrinks
to 60 seconds on the live-pulse endpoint).

If Signalbook is importable, every event is also forwarded to a
CrossScaleIngestRecord callback that the operator can wire to a
Signalbook publisher. The forwarder is a no-op if Signalbook is
absent — no hard dependency.

Usage:
    # one-shot poll (ops drill / smoke test)
    python scripts/stream_usgs_to_cache.py --once

    # daemon mode (CI runner / long-lived worker)
    python scripts/stream_usgs_to_cache.py --daemon --feed all_hour
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.usgs_stream import USGS_FEEDS, USGSStreamingClient  # noqa: E402

CACHE_DIR = PROJECT_ROOT / ".cache" / "earthquake"
STREAM_LOG = CACHE_DIR / "streamed_events.jsonl"


def _on_event_factory(stream_log: Path):
    """Return a callback that appends events to ``stream_log``."""
    stream_log.parent.mkdir(parents=True, exist_ok=True)

    def _on_event(event: dict) -> None:
        try:
            stream_log.open("a", encoding="utf-8").write(json.dumps(event) + "\n")
            print(
                f"[{dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat()}Z] "
                f"M{event.get('mag')} {event.get('place', '')[:60]}"
            )
        except Exception as exc:
            print(f"  Failed to log event: {exc}")
    return _on_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feed", default="all_hour",
                        choices=list(USGS_FEEDS.keys()),
                        help="USGS summary feed to poll (default all_hour)")
    parser.add_argument("--poll-interval-s", type=float, default=60.0,
                        help="Seconds between polls (default 60)")
    parser.add_argument("--once", action="store_true",
                        help="Poll once and exit (good for CI smoke).")
    parser.add_argument("--daemon", action="store_true",
                        help="Run forever (requires --feed). SIGINT-aware.")
    args = parser.parse_args(argv)

    print(f"USGS streamer -> {STREAM_LOG}")
    print(f"  Feed:           {args.feed}")
    print(f"  Poll interval:  {args.poll_interval_s}s")

    on_event = _on_event_factory(STREAM_LOG)
    client = USGSStreamingClient(
        on_event=on_event,
        feed=args.feed,
        poll_interval_s=args.poll_interval_s,
    )
    print(f"  Signalbook integration: "
          f"{'available' if client.signalbook_available else 'not installed'}")

    if args.once:
        emitted = client._poll_once()
        print(f"  One-shot poll emitted {emitted} new events")
        print(f"  Stream log: {STREAM_LOG} ({STREAM_LOG.stat().st_size if STREAM_LOG.exists() else 0} bytes)")
        return 0

    if args.daemon:
        print(f"  Starting daemon (Ctrl-C to stop)...")
        client.start()
        client.join()
        print(f"  Daemon stopped. Final metrics: {client.metrics}")
        return 0

    # Default: run for a single ~5-min cycle (CI-friendly)
    print(f"  Running 5-minute single-cycle (no --daemon, no --once)")
    client.start()
    import time as _time
    _time.sleep(300)
    client.stop()
    print(f"  Done. {client.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
