"""Everything the pages show, read from the files the scorers published -- and nothing else.

A page is a pure function of ``SiteData``: the live pulse, the current forecast of each hazard (its
frozen replay record), the verification summary, the evidence indexes and the model evidence bound
to each served artifact. No page computes a forecast, and no number on a page is typed by hand.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from hazardpulse.site import fmt
from hazardpulse.site.hazards import HAZARDS


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8").lstrip("﻿"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@dataclass
class Headline:
    """The single number a hazard is summarised by: its highest current chance, and where."""
    key: str
    probability: float | None
    where: str                      # the cell, storm or place the number belongs to
    forecast_id: str | None
    issued_at: str | None
    gate: str                       # pass / degrade / block: the forecast's quality-check outcome
    gate_notes: list[str] = field(default_factory=list)
    previous: float | None = None   # the same headline in the previous forecast
    previous_issued_at: str | None = None
    source: str = ""                # whose number it is (hurricanes: NOAA DTOPS or HazardPulse v8.2)


@dataclass
class SiteData:
    root: Path
    dist: Path

    # -- raw artifacts --------------------------------------------------------------------------
    @cached_property
    def pulse(self) -> dict:
        return read_json(self.dist / "data" / "live-pulse.json", {"hazards": []}) or {"hazards": []}

    @cached_property
    def pulse_by_key(self) -> dict[str, dict]:
        return {h.get("key"): h for h in self.pulse.get("hazards", [])}

    def replay(self, forecast_id: str | None) -> dict:
        if not forecast_id:
            return {}
        return read_json(self.dist / "data" / "replay" / f"{forecast_id}.json", {}) or {}

    @cached_property
    def replay_ids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {k: [] for k in HAZARDS}
        for p in (self.dist / "data" / "replay").glob("*_fcst_*.json"):
            k = p.name.split("_", 1)[0]
            if k in out:
                out[k].append(p.stem)
        for k in out:
            out[k].sort()
        return out

    @cached_property
    def earthquake(self) -> dict:
        return self.replay(self.pulse_by_key.get("eq", {}).get("forecast_id"))

    @cached_property
    def hurricanes(self) -> dict:
        return read_json(self.dist / "data" / "live-storms.json", {}) or {}

    @cached_property
    def tornadoes(self) -> dict:
        return read_json(self.dist / "data" / "live-tornadoes.json", {}) or {}

    @cached_property
    def verification(self) -> dict:
        return read_json(self.dist / "data" / "verification-summary.json", {}) or {}

    @cached_property
    def verification_by_key(self) -> dict[str, dict]:
        return {h.get("key"): h for h in self.verification.get("hazards", [])}

    @cached_property
    def registry(self) -> dict:
        return read_json(self.dist / "data" / "model-registry.json", {}) or {}

    @cached_property
    def research(self) -> dict:
        return read_json(self.dist / "data" / "cross-modality-summary.json", {}) or {}

    @cached_property
    def tornado_recovery(self) -> dict:
        return read_json(self.dist / "data" / "tornado-recovery.json", {}) or {}

    @cached_property
    def gate_decisions(self) -> list[dict]:
        d = read_json(self.dist / "data" / "evidence" / "gate-decisions.json", {}) or {}
        return list(d.get("decisions") or [])

    @cached_property
    def gate_by_forecast(self) -> dict[str, dict]:
        return {d.get("forecast_id"): d for d in self.gate_decisions}

    @cached_property
    def ledger_entries(self) -> list[dict]:
        d = read_json(self.dist / "data" / "evidence" / "prediction-ledger.json", {}) or {}
        return list(d.get("entries") or [])

    @cached_property
    def envelopes(self) -> list[dict]:
        d = read_json(self.dist / "data" / "evidence" / "provenance-envelopes.json", {}) or {}
        return list(d.get("envelopes") or [])

    @cached_property
    def replay_index(self) -> list[dict]:
        d = read_json(self.dist / "data" / "evidence" / "replay-index.json", {}) or {}
        return list(d.get("items") or [])

    @cached_property
    def evidence(self) -> dict:
        """Model evidence bound to each served artifact (``served_evidence``); a hazard whose evidence
        cannot be bound is None and the pages say so in words."""
        from hazardpulse.verification import served_evidence
        errors: list[str] = []
        ev = served_evidence.all_evidence(self.root, errors=errors)
        ev["_errors"] = errors
        return ev

    # -- derived ------------------------------------------------------------------------------------
    def gate_of(self, forecast_id: str | None) -> tuple[str, list[str]]:
        d = self.gate_by_forecast.get(forecast_id or "")
        if not d:
            return "unknown", []
        notes = list(d.get("reasons") or []) + list(d.get("warnings") or [])
        return str(d.get("decision") or "unknown"), notes

    def _previous_replay(self, key: str, forecast_id: str | None) -> dict:
        ids = self.replay_ids.get(key) or []
        if forecast_id in ids:
            i = ids.index(forecast_id)
            return self.replay(ids[i - 1]) if i > 0 else {}
        return {}

    @cached_property
    def headlines(self) -> dict[str, Headline]:
        out: dict[str, Headline] = {}
        # earthquake: the highest cell
        eq = self.earthquake
        cells = sorted(eq.get("active_cells") or [], key=lambda c: -float(c.get("probability") or 0))
        fid = eq.get("forecast_id")
        gate, notes = self.gate_of(fid)
        prev = self._previous_replay("eq", fid)
        if cells:
            from hazardpulse.site.places import fe_region
            top = cells[0]
            out["eq"] = Headline("eq", float(top["probability"]), fe_region(top["lat"], top["lon"]), fid,
                                 eq.get("issued_at"), gate, notes, _num(prev.get("top_probability")),
                                 prev.get("issued_at"))
        else:
            out["eq"] = Headline("eq", None, "", fid, eq.get("issued_at"), gate, notes)
        # hurricane: the storm with the highest PUBLISHED chance (whoever's number it is)
        hu = self.hurricanes
        storms = sorted(hu.get("storms") or [], key=lambda s: -float(s.get("ri_probability") or 0))
        fid = hu.get("forecast_id")
        gate, notes = self.gate_of(fid)
        prev = self._previous_replay("hu", fid)
        if storms:
            top = storms[0]
            out["hu"] = Headline("hu", _num(top.get("ri_probability")), storm_name(top), fid,
                                 hu.get("updated_at"), gate, notes, _num(prev.get("top_probability")),
                                 prev.get("issued_at"), source=ri_source_label(top))
        else:
            out["hu"] = Headline("hu", None, "", fid, hu.get("updated_at"), gate, notes)
        # tornado: the storm with the highest chance
        to = self.tornadoes
        tstorms = sorted(to.get("storms") or [], key=lambda s: -float(s.get("tornado_probability") or 0))
        fid = to.get("forecast_id")
        gate, notes = self.gate_of(fid)
        prev = self._previous_replay("to", fid)
        if tstorms:
            from hazardpulse.site.places import describe_us
            top = tstorms[0]
            out["to"] = Headline("to", _num(top.get("tornado_probability")), describe_us(top["lat"], top["lon"]),
                                 fid, to.get("updated_at"), gate, notes, _num(prev.get("top_probability")),
                                 prev.get("issued_at"))
        else:
            out["to"] = Headline("to", None, "", fid, to.get("updated_at"), gate, notes)
        return out

    @cached_property
    def built_at(self) -> str:
        """The newest issue time among the current forecasts: the moment this build describes."""
        times = [h.issued_at for h in self.headlines.values() if h.issued_at]
        parsed = sorted((fmt.parse_time(t), t) for t in times if fmt.parse_time(t))
        return parsed[-1][1] if parsed else (self.pulse.get("updated_at") or "")


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def storm_name(storm: dict) -> str:
    name = str(storm.get("storm_name") or "").strip()
    sid = str(storm.get("storm_id") or "").strip()
    if not name or name.upper() in ("INVEST", "UNNAMED", "NONAME"):
        return sid or "Unnamed storm"
    return name[:1].upper() + name[1:].lower() if name.isupper() else name


def ri_source_label(storm: dict) -> str:
    """Whose rapid-intensification number this row publishes, in words."""
    label = str(storm.get("ri_source_label") or "").strip()
    if label:
        return label
    src = str(storm.get("ri_source") or "")
    if src == "noaa_aid_stack":
        return "NOAA"
    # the scorer writes the label of the HazardPulse model that made the number (fetch_and_score.model_label)
    if re.fullmatch(r"v\d+(?:\.\d+)+", src):
        return f"HazardPulse {src}"
    return src or "&mdash;"


BASIN_NAMES = {
    "AL": "Atlantic", "EP": "East Pacific", "CP": "Central Pacific", "WP": "West Pacific",
    "IO": "North Indian Ocean", "SH": "Southern Hemisphere", "SI": "South Indian Ocean", "SP": "South Pacific",
}


def basin_of(storm: dict) -> str:
    """The basin a storm is IN now. NHC keeps an East Pacific number (EP) for a storm that has crossed
    140&deg;W, where the Central Pacific Hurricane Center takes over; the Central Pacific runs to the dateline,
    where NOAA reports a position as +180 (Nolo, 2026-10-04). West of it the storm has left NOAA's area."""
    b = str(storm.get("basin") or "").upper()
    lon = _num(storm.get("lon"))
    if b in ("EP", "CP") and lon is not None:
        if 0.0 < lon < 180.0:
            return "West Pacific"
        return "Central Pacific" if lon < -140.0 or lon >= 180.0 else "East Pacific"
    return BASIN_NAMES.get(b, b or "&mdash;")


def official_center(storm: dict) -> tuple[str, str]:
    """(name, url) of the agency that issues the official forecast for the storm's position."""
    basin = basin_of(storm)
    if basin == "Atlantic" or basin == "East Pacific":
        return "National Hurricane Center", "https://www.nhc.noaa.gov/"
    if basin == "Central Pacific":
        return "Central Pacific Hurricane Center", "https://www.nhc.noaa.gov/?cpac"
    return "Joint Typhoon Warning Center", "https://www.metoc.navy.mil/jtwc/jtwc.html"
