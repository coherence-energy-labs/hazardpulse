"""HazardPulse alert manager: rules, sinks, persistence, rate limiting.

Designed for the operational scoring loop: every forecast run finishes
with a ``manager.evaluate(pulse_json)`` that walks each registered
``AlertRule`` against the new live state, fires matching alerts to the
configured sinks, and persists every fired (or suppressed) alert to an
NDJSON audit log.

Three sink types ship with this module:

  * ``FileSink``    — append a JSON line to a local file
  * ``WebhookSink`` — HTTP POST the alert to an arbitrary URL
  * ``SlackSink``   — POST to a Slack incoming-webhook URL with a
                     readable text + structured ``blocks`` payload

Rate limiting is per-rule (rolling 60-second window). Sink failures are
caught individually so one broken sink does not block the others.

If Signalbook is importable, alerts can also be forwarded into
Signalbook's ``cross_scale_alerts`` SQLite table via a thin adapter
(opt-in at construction time).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_log = logging.getLogger("hazardpulse.alerts")

VALID_SEVERITIES = ("info", "watch", "warning", "critical", "suppressed")


# ----------------------------------------------------------------------
# Sinks
# ----------------------------------------------------------------------

class AlertSink:
    """Base sink — subclass and implement ``send(alert: Alert)``."""

    name: str = "base"

    def send(self, alert: "Alert") -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class FileSink(AlertSink):
    """Append-only NDJSON file sink (one alert per line)."""

    path: Path | str
    name: str = "file"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: "Alert") -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), separators=(",", ":")) + "\n")


@dataclass
class WebhookSink(AlertSink):
    """Generic HTTP POST sink. Sends ``alert.to_dict()`` as JSON body."""

    url: str
    name: str = "webhook"
    timeout_s: float = 5.0
    headers: dict[str, str] = field(default_factory=dict)

    def send(self, alert: "Alert") -> None:
        body = json.dumps(alert.to_dict()).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=ctx) as resp:
            resp.read()


@dataclass
class SlackSink(AlertSink):
    """Slack incoming-webhook sink with text + blocks formatting."""

    webhook_url: str
    name: str = "slack"
    timeout_s: float = 5.0

    def send(self, alert: "Alert") -> None:
        emoji = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "watch": ":eyes:",
            "info": ":information_source:",
            "suppressed": ":mute:",
        }.get(alert.severity, ":bell:")
        text = f"{emoji} *{alert.rule_name}* — {alert.message}"
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"hazard: `{alert.hazard}`"},
                    {"type": "mrkdwn", "text": f"forecast_id: `{alert.forecast_id}`"},
                    {"type": "mrkdwn", "text": f"prob: `{alert.probability:.1%}`"},
                ],
            },
        ]
        body = json.dumps({"text": text, "blocks": blocks}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=ctx) as resp:
            resp.read()


# ----------------------------------------------------------------------
# Rule + Alert
# ----------------------------------------------------------------------

@dataclass
class AlertRule:
    """Rule definition.

    A rule fires when ``predicate(pulse_entry)`` returns True for one of
    the hazards in ``pulse_entry``. ``predicate`` receives the full
    hazard dict (probability, risk_band, model_version, forecast_id, ...).
    """

    name: str
    severity: str
    predicate: Callable[[dict], bool]
    message_template: str = "{rule_name}: {hazard} probability {probability:.1%}"
    hazard_filter: tuple[str, ...] = ("eq", "hu", "to")
    coincidence_check: Callable[[dict], dict | None] | None = None
    max_alerts_per_min: int = 5
    sinks: tuple[str, ...] = ()  # empty = use AlertManager.default_sinks

    def matches_hazard(self, hazard_key: str) -> bool:
        return hazard_key in self.hazard_filter


@dataclass
class Alert:
    rule_name: str
    severity: str
    hazard: str
    forecast_id: str
    probability: float
    issued_at: str
    triggered_at: str
    message: str
    coincidence: dict | None = None
    sinks_notified: list[str] = field(default_factory=list)
    sink_failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------

class AlertManager:
    """Per-process alert dispatch + audit log."""

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        sinks: dict[str, AlertSink] | None = None,
        default_sinks: tuple[str, ...] = (),
        audit_path: Path | str | None = None,
        signalbook_db_path: Path | str | None = None,
    ):
        self.rules = list(rules or [])
        self.sinks: dict[str, AlertSink] = dict(sinks or {})
        self.default_sinks = default_sinks
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._rate_windows: dict[str, deque] = {}  # rule_name -> deque of timestamps
        self._lock = threading.Lock()
        self._signalbook_forwarder = None
        if signalbook_db_path:
            self._signalbook_forwarder = self._load_signalbook_forwarder(signalbook_db_path)

    @staticmethod
    def _load_signalbook_forwarder(db_path: Path | str):
        try:
            import sqlite3
            from signalbook.alerts.manager import _ALERTS_SCHEMA  # type: ignore
            conn = sqlite3.connect(str(db_path))
            conn.executescript(_ALERTS_SCHEMA)
            return conn
        except Exception as exc:
            _log.warning("Signalbook forwarder unavailable: %s", exc)
            return None

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def add_sink(self, sink: AlertSink) -> None:
        self.sinks[sink.name] = sink

    def _is_rate_limited(self, rule: AlertRule) -> bool:
        now = time.time()
        window = self._rate_windows.setdefault(rule.name, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= rule.max_alerts_per_min:
            return True
        window.append(now)
        return False

    def _persist(self, alert: Alert) -> None:
        if self.audit_path is not None:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict(), separators=(",", ":")) + "\n")
        if self._signalbook_forwarder is not None:
            try:
                conn = self._signalbook_forwarder
                conn.execute(
                    "INSERT INTO cross_scale_alerts "
                    "(rule_name, severity, triggered_at_ns, trigger_record_id, "
                    " trigger_modality, payload_json, coincident_record_ids_json, "
                    " message, sinks_notified_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        alert.rule_name,
                        alert.severity,
                        int(time.time() * 1_000_000_000),
                        alert.forecast_id,
                        f"hazardpulse_{alert.hazard}",
                        json.dumps({"probability": alert.probability}),
                        json.dumps(alert.coincidence or {}),
                        alert.message,
                        json.dumps(alert.sinks_notified),
                    ),
                )
                conn.commit()
            except Exception as exc:
                _log.warning("Signalbook forwarder write failed: %s", exc)

    def _dispatch(self, alert: Alert, sink_names: tuple[str, ...]) -> None:
        for name in sink_names:
            sink = self.sinks.get(name)
            if sink is None:
                alert.sink_failures[name] = "sink_not_registered"
                continue
            try:
                sink.send(alert)
                alert.sinks_notified.append(name)
            except Exception as exc:
                alert.sink_failures[name] = str(exc)
                _log.warning("Sink %s failed for rule %s: %s",
                             name, alert.rule_name, exc)

    def evaluate(self, pulse_payload: dict) -> list[Alert]:
        """Evaluate every rule against the live pulse JSON. Returns fired alerts."""
        alerts_fired: list[Alert] = []
        hazards = pulse_payload.get("hazards", []) or []
        updated_at = pulse_payload.get("updated_at", dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z")
        with self._lock:
            for hazard in hazards:
                hkey = hazard.get("key", "")
                forecast_id = hazard.get("forecast_id", "")
                prob = float(hazard.get("probability") or 0.0)
                for rule in self.rules:
                    if not rule.matches_hazard(hkey):
                        continue
                    try:
                        if not rule.predicate(hazard):
                            continue
                    except Exception as exc:
                        _log.warning("Rule %s predicate raised: %s", rule.name, exc)
                        continue
                    coincidence = None
                    if rule.coincidence_check is not None:
                        try:
                            coincidence = rule.coincidence_check(hazard)
                        except Exception as exc:
                            _log.warning("Rule %s coincidence check raised: %s",
                                         rule.name, exc)
                            coincidence = None
                        if coincidence is None:
                            continue  # coincidence required but not met
                    severity = rule.severity
                    if self._is_rate_limited(rule):
                        severity = "suppressed"
                    fmt_args = dict(hazard)
                    fmt_args.setdefault("rule_name", rule.name)
                    fmt_args.setdefault("hazard", hkey)
                    fmt_args["probability"] = prob
                    fmt_args["forecast_id"] = forecast_id
                    try:
                        msg = rule.message_template.format(**fmt_args)
                    except KeyError as exc:
                        msg = (f"{rule.name}: {hkey} probability {prob:.1%} "
                               f"(template missing key: {exc})")
                    alert = Alert(
                        rule_name=rule.name,
                        severity=severity,
                        hazard=hkey,
                        forecast_id=forecast_id,
                        probability=prob,
                        issued_at=updated_at,
                        triggered_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
                        message=msg,
                        coincidence=coincidence,
                    )
                    if severity != "suppressed":
                        sinks_to_use = rule.sinks if rule.sinks else self.default_sinks
                        self._dispatch(alert, sinks_to_use)
                    self._persist(alert)
                    alerts_fired.append(alert)
        return alerts_fired


# ----------------------------------------------------------------------
# Convenience: pre-built rules used by the live scoring workflow
# ----------------------------------------------------------------------

def critical_eq_rule(threshold: float = 0.50) -> AlertRule:
    return AlertRule(
        name="eq_critical_probability",
        severity="critical",
        predicate=lambda h: h.get("key") == "eq" and float(h.get("probability") or 0) >= threshold,
        message_template="Earthquake critical: {probability:.1%} (forecast {forecast_id})",
        hazard_filter=("eq",),
        max_alerts_per_min=2,
    )


def severe_to_rule(threshold: float = 0.30) -> AlertRule:
    return AlertRule(
        name="to_severe_probability",
        severity="warning",
        predicate=lambda h: h.get("key") == "to" and float(h.get("probability") or 0) >= threshold,
        message_template="Tornado outlook elevated: {probability:.1%} (forecast {forecast_id})",
        hazard_filter=("to",),
        max_alerts_per_min=4,
    )


def hu_ri_rule(threshold: float = 0.30) -> AlertRule:
    return AlertRule(
        name="hu_rapid_intensification",
        severity="warning",
        predicate=lambda h: h.get("key") == "hu" and float(h.get("probability") or 0) >= threshold,
        message_template="Hurricane RI risk: {probability:.1%} (forecast {forecast_id})",
        hazard_filter=("hu",),
        max_alerts_per_min=2,
    )


def build_default_manager(
    *,
    audit_path: Path | str,
    recent_path: Path | str | None = None,
    slack_webhook_url: str | None = None,
    webhook_url: str | None = None,
):
    """Construct an AlertManager with the standard sinks + 3 prebuilt rules.

    Honors environment variables when explicit args are not given:
      HAZARDPULSE_SLACK_WEBHOOK_URL
      HAZARDPULSE_ALERT_WEBHOOK_URL

    The "recent" sink (when ``recent_path`` is set) keeps a rolling
    window of the last N alerts as a JSON file the worker can serve
    at /api/v1/alerts/recent.
    """
    import os
    sinks: dict[str, AlertSink] = {"file": FileSink(audit_path)}
    sink_names: list[str] = ["file"]

    slack_url = slack_webhook_url or os.environ.get("HAZARDPULSE_SLACK_WEBHOOK_URL", "")
    if slack_url:
        sinks["slack"] = SlackSink(slack_url)
        sink_names.append("slack")

    webhook = webhook_url or os.environ.get("HAZARDPULSE_ALERT_WEBHOOK_URL", "")
    if webhook:
        sinks["webhook"] = WebhookSink(webhook)
        sink_names.append("webhook")

    if recent_path is not None:
        sinks["recent"] = RollingRecentSink(recent_path)
        sink_names.append("recent")

    return AlertManager(
        rules=[critical_eq_rule(0.50), severe_to_rule(0.30), hu_ri_rule(0.30)],
        sinks=sinks,
        default_sinks=tuple(sink_names),
        audit_path=audit_path,
    )


@dataclass
class RollingRecentSink(AlertSink):
    """Keep the last N alerts in a JSON file (worker-served)."""

    path: Path | str
    name: str = "recent"
    max_alerts: int = 50

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: "Alert") -> None:
        existing: list[dict] = []
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                existing = payload.get("alerts", [])
            except Exception:
                existing = []
        existing.insert(0, alert.to_dict())
        existing = existing[: self.max_alerts]
        out = {
            "schema_version": 1,
            "updated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
            "n_alerts": len(existing),
            "alerts": existing,
        }
        self.path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSink",
    "FileSink",
    "WebhookSink",
    "SlackSink",
    "RollingRecentSink",
    "VALID_SEVERITIES",
    "critical_eq_rule",
    "severe_to_rule",
    "hu_ri_rule",
    "build_default_manager",
]
