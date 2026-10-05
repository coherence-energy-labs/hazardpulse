"""USGS earthquake feed streaming client.

Polls USGS GeoJSON summary feeds at a fixed cadence, dedupes events by
``id``, and emits each new event to a callback. Architecture mirrors
Signalbook's ``signalbook.streaming.usgs_stream`` but is self-contained
so HazardPulse doesn't take Signalbook as a hard dependency.

If Signalbook IS importable, this module will also publish each new
event into Signalbook's ``CrossScaleIngestRecord`` schema via the
optional ``SignalbookForwarder``.

Usage:
    from hazardpulse.data.usgs_stream import USGSStreamingClient

    def on_event(event):
        print("got", event["id"], event["mag"])

    client = USGSStreamingClient(on_event=on_event)
    client.start()  # spawns background thread
    client.join()   # blocks; SIGINT-aware
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import signal
import ssl
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

_log = logging.getLogger("hazardpulse.usgs_stream")

# Standard USGS summary feeds
USGS_FEEDS = {
    "all_hour":           "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "2.5_day":            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
    "all_day":            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "significant_day":    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson",
    "4.5_week":           "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson",
    "significant_week":   "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson",
}


@dataclass
class StreamMetrics:
    polls_attempted: int = 0
    polls_succeeded: int = 0
    polls_failed: int = 0
    events_seen: int = 0
    events_emitted: int = 0
    last_poll_at: dt.datetime | None = None
    last_event_at: dt.datetime | None = None
    last_error: str | None = None


@dataclass
class USGSStreamingClient:
    """Polls a USGS feed, emits new (deduplicated) events to ``on_event``."""

    on_event: Callable[[dict], None]
    feed: str = "all_hour"
    poll_interval_s: float = 60.0
    dedup_size: int = 5000
    max_backoff_s: float = 600.0
    user_agent: str = "HazardPulse/1.0 (research)"
    on_signalbook_record: Callable[[object], None] | None = None  # optional Signalbook forwarder

    _seen_ids: deque = field(init=False, default_factory=lambda: deque(maxlen=5000))
    _seen_set: set = field(init=False, default_factory=set)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)
    metrics: StreamMetrics = field(init=False, default_factory=StreamMetrics)

    def __post_init__(self) -> None:
        self._seen_ids = deque(maxlen=self.dedup_size)
        self._signalbook_record_cls = self._maybe_load_signalbook_record_cls()

    @staticmethod
    def _maybe_load_signalbook_record_cls():
        try:
            from signalbook.connectors.external._base import CrossScaleIngestRecord  # type: ignore
            return CrossScaleIngestRecord
        except Exception:
            return None

    @property
    def signalbook_available(self) -> bool:
        return self._signalbook_record_cls is not None

    def _feed_url(self) -> str:
        if self.feed in USGS_FEEDS:
            return USGS_FEEDS[self.feed]
        if self.feed.startswith(("http://", "https://")):
            return self.feed
        raise ValueError(f"Unknown USGS feed: {self.feed!r}")

    def _fetch(self) -> dict | None:
        req = urllib.request.Request(
            self._feed_url(), headers={"User-Agent": self.user_agent}
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception as exc:
            self.metrics.polls_failed += 1
            self.metrics.last_error = str(exc)
            _log.warning("USGS feed fetch failed: %s", exc)
            return None

    def _to_event(self, feature: dict) -> dict | None:
        try:
            props = feature.get("properties", {}) or {}
            geom = feature.get("geometry", {}) or {}
            coords = geom.get("coordinates", [None, None, None])
            return {
                "id": feature.get("id") or props.get("ids", "").split(",")[0],
                "time": dt.datetime.fromtimestamp(props["time"] / 1000.0, dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
                "time_epoch_ms": int(props["time"]),
                "mag": props.get("mag"),
                "magType": props.get("magType"),
                "place": props.get("place"),
                "type": props.get("type", "earthquake"),
                "latitude": float(coords[1]) if coords and coords[1] is not None else None,
                "longitude": float(coords[0]) if coords and coords[0] is not None else None,
                "depth": float(coords[2]) if coords and len(coords) > 2 and coords[2] is not None else None,
                "url": props.get("url"),
                "source": "usgs",
            }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            _log.debug("Skipping feature: %s", exc)
            return None

    def _to_signalbook_record(self, event: dict):
        if self._signalbook_record_cls is None or event["latitude"] is None:
            return None
        try:
            phenomenon = event.get("type", "earthquake") or "earthquake"
            return self._signalbook_record_cls(
                record_id=f"usgs_{event['id']}",
                modality="seismic_event",
                observatory="usgs",
                phenomenon=phenomenon,
                time_utc_ns=event["time_epoch_ms"] * 1_000_000,
                lat_deg=event["latitude"],
                lon_deg=event["longitude"],
                payload={
                    "magnitude": event.get("mag"),
                    "depth_km": event.get("depth"),
                    "place": event.get("place"),
                    "url": event.get("url"),
                },
            )
        except Exception as exc:
            _log.debug("Signalbook record build failed: %s", exc)
            return None

    def _poll_once(self) -> int:
        """Poll the feed once. Returns number of new events emitted."""
        self.metrics.polls_attempted += 1
        self.metrics.last_poll_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        payload = self._fetch()
        if payload is None:
            return 0
        self.metrics.polls_succeeded += 1
        features = payload.get("features", []) or []
        emitted = 0
        for feat in features:
            event = self._to_event(feat)
            if event is None or not event.get("id"):
                continue
            self.metrics.events_seen += 1
            eid = event["id"]
            if eid in self._seen_set:
                continue
            self._seen_set.add(eid)
            self._seen_ids.append(eid)
            # Maintain bounded set
            while len(self._seen_ids) > self.dedup_size and self._seen_ids:
                old = self._seen_ids.popleft()
                self._seen_set.discard(old)
            try:
                self.on_event(event)
            except Exception as exc:
                _log.exception("on_event callback failed: %s", exc)
            sb_rec = self._to_signalbook_record(event)
            if sb_rec is not None and self.on_signalbook_record is not None:
                try:
                    self.on_signalbook_record(sb_rec)
                except Exception as exc:
                    _log.debug("Signalbook forwarder failed: %s", exc)
            self.metrics.events_emitted += 1
            self.metrics.last_event_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            emitted += 1
        return emitted

    def _run(self) -> None:
        backoff = self.poll_interval_s
        while not self._stop_event.is_set():
            try:
                emitted = self._poll_once()
                _log.debug("USGS poll: emitted %d new events", emitted)
                backoff = self.poll_interval_s
            except Exception as exc:
                _log.exception("USGS poll loop error: %s", exc)
                backoff = min(backoff * 2, self.max_backoff_s)
            if self._stop_event.wait(backoff):
                break
        _log.info("USGS streaming client stopped.")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="usgs-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def join(self) -> None:
        """Block until the stream stops (SIGINT/SIGTERM aware)."""
        def _handler(signum, frame):
            _log.info("USGS streaming: caught signal %d, stopping...", signum)
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (AttributeError, ValueError):
            pass
        if self._thread:
            while self._thread.is_alive():
                time.sleep(1)


__all__ = ["USGSStreamingClient", "StreamMetrics", "USGS_FEEDS"]
