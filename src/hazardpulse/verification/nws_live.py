"""LIVE NWS tornado-warning inputs (model block W) from api.weather.gov.

The tornado model v3+W takes two inputs per storm observation ``(lat, lon, t)``::

    w_tor_warning_active    1.0 if the storm centroid lies inside the polygon of a
                            tornado warning in force at t (issued at or before t)
    w_minutes_since_issue   minutes from that warning's ORIGINAL issuance to t;
                            NaN when not warned

In training both came from the IEM storm-based-warning archive through
:func:`hazardpulse.verification.nws_warnings.tor_warning_state` (written per
store row by ``scripts/audit_20261001/warning_state_on_store.py``).  This module
computes the same two numbers from the National Weather Service API, by the
same code: the API messages are turned into polygon states with IEM's
definitions, packed into a :class:`~nws_warnings.TorWarnings`, and queried by
:func:`~nws_warnings.query_tor_warnings` -- the training query itself, with
its point-in-polygon kernel, half-open validity rule and tie rule.  Only the
construction of the polygon states is new code here.

Training semantics (nws_warnings, read 2026-10-02)
==================================================
* One **polygon state** per product (NEW issuance, CON / EXP / COR statement);
  a CAN is not a state.  A state is in effect on ``[valid_from, valid_to)``
  with ``valid_from`` = the product's issuance minute and ``valid_to`` = the
  next product of the event (SVS update or CAN) or the event's VTEC end,
  whichever is first (late-EXP artefact clipped back to the VTEC end).
* ``active_now``: the point is inside a state in effect at ``t`` (exact
  even-odd test over every ring).  A product issued after ``t`` cannot affect
  ``t``: it only ends a state at its own (later) instant.
* ``minutes_since_issue`` = ``(t - issue_time) / 60`` where ``issue_time`` is
  the issuance of the **latest NEW product of the same event key at or before
  the state** -- the original issuance, not the SVS instant.  Several covering
  warnings: the **earliest-issued** wins (largest minutes).  ``t`` is floored
  to whole seconds; instants are minute-precision UTC.

The API (measured 2026-10-02 on all 119 TO.W messages api.weather.gov held, 34 events)
=======================================================================================
``GET /alerts/active?event=Tornado%20Warning`` -> GeoJSON FeatureCollection,
one Feature per CAP message *segment*.  The fields that matter, from the real
SVS ``KICT TO.W 0030`` (Lincoln/Russell KS, 2026-09-30)::

    id            urn:oid:2.49.0.1.840.0.27ba48ac...b974766.002.1   (.SEG.1: segments of
                  one product share the stem; this product's CAN is .001.1)
    sent          2026-09-30T18:46:00-05:00     product issuance (whole minute: 119/119)
    effective     = sent (119/119)
    onset         = sent (119/119) -- for a CON this is the SVS instant, NOT the issuance
    expires       2026-09-30T19:00:00-05:00     CAP display expiry -- NOT the warning's end:
                  EXP messages carry end + 1..11 min (17/17), CAN messages sent + 15.2..16.0 min (11/11)
    ends          2026-09-30T19:00:00-05:00     = VTEC end (119/119)
    messageType   Update                        unreliable: 5 of 17 EXP statements are "Alert"
    status        Actual
    parameters.VTEC  ["/O.CON.KICT.TO.W.0030.000000T0000Z-261001T0000Z/"]
                  (VTEC begin is 000000T0000Z on every CON/EXP/CAN: 85/85; on NEW it
                  equals sent: 34/34)
    parameters.WMOidentifier ["WWUS53 KICT 302346"], AWIPSidentifier ["SVSICT"]
    references    [{identifier, sent}, ...] of earlier messages of the chain; contains the
                  NEW in only 80 of 85 follow-ups -- the 5 EXP statements typed "Alert"
                  have references []; CAN segments are never referenced (0 of 11)
    geometry      Polygon [[lon, lat], ...] (119/119 non-null)

A **partial cancellation** is ONE SVS product with two segments sharing ``sent``
and the id stem: a CAN (the cancelled counties) and a CON (the rest).  Both carry
the *remaining* polygon (7/7 such pairs).  IEM keeps the CON state alive to the
VTEC end; so does this module: a CAN ends the event only if no non-CAN segment
of the same event shares its product instant.

Mapping to training
-------------------
* event key            VTEC office + ETN (VTEC ``O`` class, ``TO.W`` only)
* polygon state        every non-CAN message; ``valid_from`` = ``sent`` (floored to the
                       minute -- IEM's precision; 0 of 119 had seconds)
* state end            ``min(next product of the event (CAN included), VTEC end)``;
                       ``expires`` is never used
* original issuance    ``sent`` of the latest NEW of the same office+ETN at or before the
                       state (IEM's rule).  ``onset`` and ``references`` are NOT sufficient
                       (see above), so the snapshot always includes the API's message
                       history back to ``t - MAX_STATE_LIFETIME_S`` (``/alerts?start=``),
                       which holds every NEW of every warning that can be in force at t.
                       Fallbacks if a NEW is still absent (counted, ``issue_source``):
                       earliest ``references[].sent`` (code 4), then the state's own start
                       (code 3, IEM's last fallback too).

Why the history fetch is not optional: the scorer evaluates at the ProbSevere
valid time ``t``, minutes before the fetch.  ``/alerts/active`` alone drops (a)
superseded messages -- an SVS issued between t and the fetch hides the polygon
in force at t -- and (b) warnings that expired between t and the fetch.  The
history window has both.

Parity, and what the naive readings cost (MEASURED on the 2026-09-25..30 record
against the IEM record of the same 34 warnings, 200,000 points sampled around
the 108 polygon states, +-10 min): this module reproduces all 108 IEM states
(product, interval, issuance, vertices) and both inputs at every point, 0
mismatches.  Taking ``onset`` as the issuance makes minutes too small on 64 %
of warned points (median 15 min); ending at ``expires`` adds false
``active`` points equal to 7.4 % of the warned ones; the active feed alone,
fetched 2 / 5 / 10 min after t, gets ``active`` wrong on 6 / 18 / 43 % of them.

Null / unusable geometry
------------------------
Tornado warnings are storm-based; IEM's training rows are polygons only
(``limit1=yes``).  ``affectedZones`` lists *counties*, whose area is many times a
warning polygon's, so it is not used -- a county test is not the training
feature.  A message with null or unusable geometry is counted in the report and:

* a NEW (or a follow-up with no earlier polygon): contributes no polygon;
* a follow-up statement: **inherits the event's previous polygon**.  Statements
  reduce a warning's polygon; they are not exact subsets -- in the 74
  follow-ups of the 2026-09-25..30 record, 20 re-drew an edge slightly outside
  the previous polygon, by at most 2.6 % of their area (MEASURED, sampled
  point-in-polygon).  So the inherited polygon covers >= 97 % of the true one:
  the error is ``active = 1`` in the area the statement cut plus ``active = 0``
  on such slivers -- against ``active = 0`` over the entire current polygon,
  until the next statement, if the message were dropped (and the storm sits
  inside its current polygon).

``strict=True`` raises :class:`NwsLiveDataError` instead, for any message that
could not be represented exactly.  None was observed in the 119-message record.

Failure contract
----------------
Every network, HTTP, JSON or envelope failure raises :class:`NwsLiveFetchError`
(a :class:`NwsLiveError`); query instants the snapshot cannot answer raise
:class:`NwsLiveCoverageError`.  Nothing here ever returns zeros in place of an
unknown warning state; "no warning" is reported only from a successful fetch.
"""

from __future__ import annotations

import bisect
import http.client
import json
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from hazardpulse.data import http as hp_http
from hazardpulse.verification import nws_warnings as nw

__all__ = [
    "API_ROOT",
    "ACTIVE_URL",
    "USER_AGENT",
    "W_NAMES",
    "ISSUE_FROM_REFERENCES",
    "NwsLiveError",
    "NwsLiveFetchError",
    "NwsLiveDataError",
    "NwsLiveCoverageError",
    "AlertSnapshot",
    "LiveWarningInputs",
    "history_url",
    "fetch_alert_snapshot",
    "warnings_from_api_alerts",
    "inputs_from_snapshot",
    "live_tor_warning_inputs",
]

API_ROOT = "https://api.weather.gov"
ACTIVE_URL = API_ROOT + "/alerts/active?event=Tornado%20Warning"
#: api.weather.gov requires an identifying User-Agent with a contact.
USER_AGENT = "HazardPulse/1.0 (hazardpulse.com; josh@coherenceenergylabs.com)"
#: Model column names of block W, in the training order (tornado_lab.W_NAMES).
W_NAMES = ("w_tor_warning_active", "w_minutes_since_issue")

DEFAULT_TIMEOUT_S = 10.0
#: Instants after the API's last ingest cycle that are still answered.  A
#: product issued in that gap is not yet visible (the feed refreshes every
#: ~1.5-2 min, MEASURED 2026-10-02).
DEFAULT_MAX_FUTURE_S = 180
HISTORY_PAGE_LIMIT = 500
MAX_HISTORY_PAGES = 20

#: issue_source code for an issuance taken from the earliest ``references``
#: entry (codes 0-3 are nws_warnings' ISSUE_FROM_*).
ISSUE_FROM_REFERENCES = 4

_ACTION_ORDER = {"NEW": 0, "COR": 1, "EXT": 1, "EXA": 1, "EXB": 1, "CON": 2, "UPG": 2, "EXP": 3, "CAN": 4}
_OFFICE_TO_IEM_WFO = {"TJSJ": "SJU"}  # IEM's 3-char id where it is not the ICAO id minus its first letter


class NwsLiveError(RuntimeError):
    """Base: the live warning inputs could not be determined."""


class NwsLiveFetchError(NwsLiveError):
    """The API could not be reached, answered with an error, or returned an unreadable body."""


class NwsLiveDataError(NwsLiveError):
    """(strict mode) a tornado-warning message could not be represented exactly."""


class NwsLiveCoverageError(NwsLiveError):
    """A query instant lies outside the interval the snapshot can answer."""


# --------------------------------------------------------------------------
# Snapshot + fetch
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertSnapshot:
    """The API's TO.W messages, with the instants that bound what they can answer.

    Every message ``sent`` in ``[known_from, known_until]`` is present;
    ``known_until`` is the API's last ingest instant (the active feed's
    ``updated``) or the request instant, whichever is earlier.  A ``None``
    ``known_from`` marks an active-only snapshot, which answers only instants at
    or after ``known_until``.
    """

    features: tuple[dict, ...]
    known_from: int | None
    known_until: int
    sources: tuple[str, ...] = ()
    fetched_at: int | None = None

    def answerable(self, max_future_s: int = DEFAULT_MAX_FUTURE_S) -> tuple[int, int]:
        """``[lo, hi]`` (POSIX s) of the query instants this snapshot answers exactly."""

        lo = self.known_until if self.known_from is None else self.known_from + nw.MAX_STATE_LIFETIME_S
        return lo, self.known_until + int(max_future_s)


def history_url(start_utc: int, limit: int = HISTORY_PAGE_LIMIT) -> str:
    """``/alerts`` query for every TO.W message sent at or after ``start_utc`` (inclusive, MEASURED)."""

    start = datetime.fromtimestamp(int(start_utc), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = {"event": "Tornado Warning", "start": start, "limit": int(limit)}
    return f"{API_ROOT}/alerts?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote, safe=":")


def _get_json(url: str, timeout: float) -> dict:
    try:
        data = hp_http.fetch_bytes(url, timeout=timeout, use_cache=False, user_agent=USER_AGENT)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        raise NwsLiveFetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
    try:
        doc = json.loads(data)
    except ValueError as exc:
        raise NwsLiveFetchError(f"{url}: body is not JSON ({data[:120]!r})") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("features"), list):
        raise NwsLiveFetchError(f"{url}: not a GeoJSON FeatureCollection (keys {sorted(doc)[:8] if isinstance(doc, dict) else type(doc).__name__})")
    for f in doc["features"]:
        if not isinstance(f, dict) or not isinstance(f.get("properties"), dict):
            raise NwsLiveFetchError(f"{url}: a feature without a properties object")
    return doc


def _iso_s(value) -> int | None:
    """ISO-8601 with an explicit offset (or Z) -> POSIX seconds (floor); None if absent."""

    if value is None or value == "":
        return None
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive timestamp {value!r}")
    return int(np.floor(dt.timestamp()))


def _feature_id(f: dict) -> str:
    return str(f.get("id") or f["properties"].get("id") or f["properties"].get("@id") or "")


def _merge(*feature_lists: Iterable[dict]) -> tuple[dict, ...]:
    seen: dict[str, dict] = {}
    anon: list[dict] = []
    for lst in feature_lists:
        for f in lst:
            fid = _feature_id(f)
            if not fid:
                anon.append(f)
            elif fid not in seen:
                seen[fid] = f
    return tuple(seen.values()) + tuple(anon)


def fetch_alert_snapshot(
    t_min_utc,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_pages: int = MAX_HISTORY_PAGES,
) -> AlertSnapshot:
    """Fetch the active TO.W alerts plus every TO.W message since ``t_min - MAX_STATE_LIFETIME_S``.

    Raises :class:`NwsLiveFetchError` on any failure; never returns a partial snapshot.
    """

    t_min = int(nw.to_epoch_seconds([t_min_utc])[0])
    requested = int(time.time())
    active = _get_json(ACTIVE_URL, timeout)
    updated = None
    try:
        updated = _iso_s(active.get("updated"))
    except ValueError:
        updated = None
    start = (t_min - nw.MAX_STATE_LIFETIME_S) // 60 * 60
    url: str | None = history_url(start)
    first_url = url
    history: list[dict] = []
    pages = 0
    while url:
        doc = _get_json(url, timeout)
        pages += 1
        if not doc["features"]:
            break
        history.extend(doc["features"])
        if pages >= max_pages:
            raise NwsLiveFetchError(f"history needs more than {max_pages} pages from {first_url}; refusing a partial record")
        nxt = doc.get("pagination") or {}
        url = nxt.get("next") if isinstance(nxt, dict) else None
    known_until = requested if updated is None else min(requested, updated)
    return AlertSnapshot(
        features=_merge(active["features"], history),
        known_from=int(start),
        known_until=int(known_until),
        sources=(ACTIVE_URL, str(first_url)),
        fetched_at=requested,
    )


# --------------------------------------------------------------------------
# Messages -> polygon states (IEM definitions)
# --------------------------------------------------------------------------


@dataclass
class _Msg:
    fid: str
    office: str
    etn: int
    action: str
    sent: int
    vtec_begin: int | None
    vtec_end: int
    rings: list[np.ndarray] | None
    geometry_problem: str  # "" | "null" | "bad: ..."
    ref_min_sent: int | None
    product_id: str
    tornado_tag: str
    damage_tag: str
    hail_in: float


def _rings_of(geometry) -> list[np.ndarray]:
    """GeoJSON Polygon / MultiPolygon -> closed (n, 2) lon/lat rings; ValueError if unusable."""

    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        raise ValueError(f"geometry type {gtype!r}")
    rings: list[np.ndarray] = []
    for poly in polys or []:
        for rg in poly or []:
            a = np.asarray([p[:2] for p in rg], dtype=np.float64)
            if a.ndim != 2 or a.shape[1] != 2:
                raise ValueError("ring is not a list of [lon, lat]")
            if not np.all(np.isfinite(a)):
                raise ValueError("non-finite vertex")
            if np.any(np.abs(a[:, 1]) > 90) or np.any(np.abs(a[:, 0]) > 180):
                raise ValueError("vertex outside lon/lat range")
            if not np.array_equal(a[0], a[-1]):
                a = np.vstack([a, a[:1]])
            if len(a) < 4:
                raise ValueError(f"ring with {len(a)} vertices")
            rings.append(a)
    if not rings:
        raise ValueError("no rings")
    allv = np.vstack(rings)
    if allv[:, 0].max() - allv[:, 0].min() >= 180.0:
        raise ValueError("polygon spans >= 180 deg of longitude (antimeridian not supported)")
    return rings


def _first(params: dict, key: str, default: str = "") -> str:
    v = params.get(key)
    if isinstance(v, list):
        return str(v[0]) if v else default
    return default if v is None else str(v)


def _parse(f: dict, counts: dict) -> list[_Msg]:
    p = f["properties"]
    if p.get("event") != "Tornado Warning":
        counts["skipped_not_tornado_warning"] += 1
        return []
    if p.get("status") != "Actual":
        counts["skipped_not_actual"] += 1
        return []
    params = p.get("parameters") or {}
    vtecs = []
    for s in params.get("VTEC") or []:
        for m in nw._VTEC_RE.finditer(str(s)):
            if m.group(1) == "O" and m.group(4) == "TO" and m.group(5) == "W":
                vtecs.append(m)
    if not vtecs:
        counts["skipped_no_operational_tow_vtec"] += 1
        return []
    sent_raw = _iso_s(p.get("sent"))
    if sent_raw is None:
        counts["skipped_no_sent"] += 1
        return []
    if sent_raw % 60:
        counts["sent_not_whole_minute"] += 1
    sent = sent_raw // 60 * 60  # IEM's instants are minute precision (to_char ... HH24MI)
    ends = _iso_s(p.get("ends"))
    geom = f.get("geometry")
    rings: list[np.ndarray] | None = None
    problem = ""
    if not geom:
        problem = "null"
    else:
        try:
            rings = _rings_of(geom)
        except (ValueError, TypeError, IndexError) as exc:
            problem = f"bad: {exc}"
    refs = [r for r in (p.get("references") or []) if isinstance(r, dict)]
    ref_sent = [s for s in (_iso_s(r.get("sent")) for r in refs) if s is not None]
    wmo = _first(params, "WMOidentifier").split()
    awips = _first(params, "AWIPSidentifier")
    stamp = datetime.fromtimestamp(sent, tz=timezone.utc).strftime("%Y%m%d%H%M")
    product_id = f"{stamp}-{wmo[1]}-{wmo[0]}-{awips}" if len(wmo) >= 2 and awips else f"{stamp}-{_feature_id(f)}"
    hail = _first(params, "maxHailSize")
    try:
        hail_in = float(hail) if hail else float("nan")
    except ValueError:
        hail_in = float("nan")
    out = []
    for m in vtecs:
        end = nw._vtec_ts(m.group(8))
        if end is None:
            end = ends
        if end is None:
            counts["skipped_no_end"] += 1
            continue
        if ends is not None and ends != end:
            counts["ends_ne_vtec_end"] += 1
        out.append(
            _Msg(
                fid=_feature_id(f),
                office=m.group(3),
                etn=int(m.group(6)),
                action=m.group(2),
                sent=sent,
                vtec_begin=nw._vtec_ts(m.group(7)),
                vtec_end=int(end),
                rings=rings,
                geometry_problem=problem,
                ref_min_sent=min(ref_sent) // 60 * 60 if ref_sent else None,
                product_id=product_id,
                tornado_tag=_first(params, "tornadoDetection"),
                damage_tag=_first(params, "tornadoDamageThreat"),
                hail_in=hail_in,
            )
        )
    return out


def _same_rings(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def warnings_from_api_alerts(features: Sequence[dict], *, strict: bool = False) -> tuple[nw.TorWarnings, dict]:
    """API alert Features -> (:class:`~nws_warnings.TorWarnings` of polygon states, build report).

    Pure function of the message set: duplicates (same id) are ignored, the
    order of ``features`` does not matter, and messages sent after a query
    instant cannot change the answer at that instant.
    """

    counts: dict = {
        k: 0
        for k in (
            "features",
            "duplicate_ids",
            "skipped_not_tornado_warning",
            "skipped_not_actual",
            "skipped_no_operational_tow_vtec",
            "skipped_no_sent",
            "skipped_no_end",
            "sent_not_whole_minute",
            "ends_ne_vtec_end",
            "null_geometry",
            "bad_geometry",
            "geometry_inherited",
            "geometry_dropped",
            "end_disagreement_within_product",
        )
    }
    merged = _merge(features)
    counts["features"] = len(features)
    counts["duplicate_ids"] = len(features) - len(merged)
    msgs: list[_Msg] = []
    for f in merged:
        if not isinstance(f, dict) or not isinstance(f.get("properties"), dict):
            raise NwsLiveDataError("alert feature without a properties object")
        try:
            msgs.extend(_parse(f, counts))
        except (ValueError, TypeError) as exc:
            raise NwsLiveDataError(f"{_feature_id(f)}: unreadable alert ({exc})") from exc
    for m in msgs:
        if m.geometry_problem:
            counts["null_geometry" if m.geometry_problem == "null" else "bad_geometry"] += 1
            if strict:
                raise NwsLiveDataError(f"{m.fid} ({m.office} TO.W {m.etn:04d} {m.action}): geometry {m.geometry_problem}")
    action_counts: dict[str, int] = {}
    for m in msgs:
        action_counts[m.action] = action_counts.get(m.action, 0) + 1

    chains: dict[tuple[str, int], list[_Msg]] = {}
    for m in msgs:
        chains.setdefault((m.office, m.etn), []).append(m)

    rows: list[dict] = []
    events_without_new = 0
    for (office, etn), ms in sorted(chains.items()):
        ms.sort(key=lambda m: (m.sent, _ACTION_ORDER.get(m.action, 9), m.fid))
        news = sorted({m.sent for m in ms if m.action == "NEW"})
        new_begin = {m.sent: (m.vtec_begin if m.vtec_begin is not None else m.sent) for m in ms if m.action == "NEW"}
        new_end = {m.sent: m.vtec_end for m in ms if m.action == "NEW"}
        if not news:
            events_without_new += 1
        group_times = sorted({m.sent for m in ms})
        prev_rings: list[np.ndarray] | None = None
        for gi, gt in enumerate(group_times):
            grp = [m for m in ms if m.sent == gt]
            nxt = group_times[gi + 1] if gi + 1 < len(group_times) else None
            poly = [m for m in grp if m.action != "CAN"]
            if not poly:  # a product that only cancels: the event ends at its instant
                prev_rings = None
                continue
            if any(m.action == "NEW" for m in poly):
                prev_rings = None
            ends = {m.vtec_end for m in poly}
            if len(ends) > 1:
                counts["end_disagreement_within_product"] += 1
            end = max(ends)
            valid_to = end if nxt is None else min(nxt, end)
            k = bisect.bisect_right(news, gt) - 1
            if k >= 0:
                issue, source = news[k], nw.ISSUE_FROM_NEW_PRODUCT
            else:
                refs = [m.ref_min_sent for m in grp if m.ref_min_sent is not None]
                if refs:
                    issue, source = min(refs), ISSUE_FROM_REFERENCES
                else:
                    issue, source = gt, nw.ISSUE_FROM_STATE_START
            year = int(nw._years_of(np.array([new_begin[news[k]] if k >= 0 else issue]))[0])
            shapes: list[tuple[_Msg, list[np.ndarray]]] = []
            for m in poly:
                rings = m.rings
                if rings is None:
                    if prev_rings is not None and m.action != "NEW":
                        rings = prev_rings
                        counts["geometry_inherited"] += 1
                    else:
                        counts["geometry_dropped"] += 1
                        continue
                if not any(_same_rings(rings, r) for _, r in shapes):
                    shapes.append((m, rings))
            for m, rings in shapes:
                rows.append(
                    {
                        "wfo": _OFFICE_TO_IEM_WFO.get(office, office[1:] if len(office) == 4 else office),
                        "etn": etn,
                        "vtec_year": year,
                        "status": m.action,
                        "product_id": m.product_id,
                        "product_time": gt,
                        "valid_from": gt,
                        "valid_to": valid_to,
                        "issue_time": issue,
                        "issue_source": source,
                        "is_issuance": m.action == "NEW",
                        "expired": end,
                        "init_exp": new_end[news[k]] if k >= 0 else nw._NO_TIME,
                        "tornado_tag": m.tornado_tag,
                        "damage_tag": m.damage_tag,
                        "hail_tag_in": m.hail_in,
                        "is_emergency": m.damage_tag.upper() == "CATASTROPHIC",
                        "rings": rings,
                    }
                )
            if shapes:
                prev_rings = shapes[0][1]

    arrays = _pack(rows)
    live = arrays["valid_to"] > arrays["valid_from"]
    lifetime = (arrays["valid_to"] - arrays["issue_time"])[live]
    report = dict(
        counts,
        messages=len(msgs),
        action_counts=dict(sorted(action_counts.items())),
        events=len(chains),
        events_without_new=events_without_new,
        n_states=int(len(rows)),
        empty_states=int(np.sum(~live)),
        issue_source_counts={
            name: int(np.sum(arrays["issue_source"] == code))
            for name, code in (
                ("new_product", nw.ISSUE_FROM_NEW_PRODUCT),
                ("references", ISSUE_FROM_REFERENCES),
                ("state_start", nw.ISSUE_FROM_STATE_START),
            )
        },
        max_state_lifetime_s=int(lifetime.max()) if lifetime.size else 0,
    )
    if strict and (report["issue_source_counts"]["references"] or report["issue_source_counts"]["state_start"]):
        raise NwsLiveDataError(f"{events_without_new} event(s) without their NEW message: original issuance not exact")
    label = int(nw._years_of(np.array([int(arrays["valid_from"].max()) if len(rows) else int(time.time())]))[0])
    meta = dict(report, year=label, source="api.weather.gov", year_complete_at_fetch=False)
    W = nw._assemble([arrays], [label], {str(label): meta})
    return W, report


def _pack(rows: list[dict]) -> dict[str, np.ndarray]:
    """Rows -> the column + ragged-geometry arrays of a nws_warnings compact year."""

    R = len(rows)
    col = lambda k, dt: np.array([r[k] for r in rows], dtype=dt) if R else np.zeros(0, dtype=dt)  # noqa: E731
    arrays = {
        "wfo": col("wfo", "U4"),
        "etn": col("etn", np.int32),
        "vtec_year": col("vtec_year", np.int16),
        "status": col("status", "U3"),
        "product_id": col("product_id", "U96"),
        "product_time": col("product_time", np.int64),
        "valid_from": col("valid_from", np.int64),
        "valid_to": col("valid_to", np.int64),
        "issue_time": col("issue_time", np.int64),
        "issue_source": col("issue_source", np.int8),
        "is_issuance": col("is_issuance", bool),
        "expired": col("expired", np.int64),
        "init_exp": col("init_exp", np.int64),
        "tornado_tag": col("tornado_tag", "U16"),
        "damage_tag": col("damage_tag", "U16"),
        "hail_tag_in": col("hail_tag_in", np.float32),
        "is_emergency": col("is_emergency", bool),
        "area_km2": np.full(R, np.nan, dtype=np.float32),
    }
    ring_lens: list[int] = []
    poly_rings: list[int] = []
    vx: list[np.ndarray] = []
    vy: list[np.ndarray] = []
    bbox = np.zeros((R, 4))
    for i, r in enumerate(rows):
        poly_rings.append(len(r["rings"]))
        for rg in r["rings"]:
            ring_lens.append(len(rg))
            vx.append(rg[:, 0])
            vy.append(rg[:, 1])
        allv = np.vstack(r["rings"])
        bbox[i] = allv[:, 0].min(), allv[:, 0].max(), allv[:, 1].min(), allv[:, 1].max()
        arrays["area_km2"][i] = abs(sum(nw._ring_area_km2(rg[:, 0], rg[:, 1]) for rg in r["rings"]))
    arrays["minlon"], arrays["maxlon"], arrays["minlat"], arrays["maxlat"] = (bbox[:, j].copy() for j in range(4))
    arrays["ring_start"] = np.concatenate([[0], np.cumsum(ring_lens)]).astype(np.int64)
    arrays["poly_ring_start"] = np.concatenate([[0], np.cumsum(poly_rings)]).astype(np.int64)
    arrays["vx"] = np.concatenate(vx) if vx else np.zeros(0)
    arrays["vy"] = np.concatenate(vy) if vy else np.zeros(0)
    return arrays


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveWarningInputs:
    """Block W for N storm observations, in input order (training dtypes: float32)."""

    w_tor_warning_active: np.ndarray  # float32 1.0 / 0.0
    w_minutes_since_issue: np.ndarray  # float32, NaN unless warned
    event: np.ndarray  # str "ICT.TO.W.0030" (IEM WFO id) of the earliest-issued covering warning, "" if none
    report: dict = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        """``(N, 2)`` float32 in :data:`W_NAMES` order -- the columns appended to the model input."""

        return np.column_stack([self.w_tor_warning_active, self.w_minutes_since_issue]).astype(np.float32)


def inputs_from_snapshot(
    lats,
    lons,
    times_utc,
    snapshot: AlertSnapshot,
    *,
    strict: bool = False,
    require_coverage: bool = True,
    max_future_s: int = DEFAULT_MAX_FUTURE_S,
) -> LiveWarningInputs:
    """Block W for ``(lats[i], lons[i], times_utc[i])`` from an :class:`AlertSnapshot` (no network)."""

    t = nw.to_epoch_seconds(times_utc)
    if require_coverage and t.size:
        lo, hi = snapshot.answerable(max_future_s)
        if int(t.min()) < lo or int(t.max()) > hi:
            fmt = lambda s: datetime.fromtimestamp(int(s), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
            raise NwsLiveCoverageError(
                f"query instants {fmt(t.min())}..{fmt(t.max())} outside the snapshot's answerable "
                f"interval {fmt(lo)}..{fmt(hi)} (known_from={snapshot.known_from}, known_until={snapshot.known_until})"
            )
    W, report = warnings_from_api_alerts(snapshot.features, strict=strict)
    q = nw.query_tor_warnings(lats, lons, t, warnings=W, require_coverage=False)
    active = np.asarray(q.active_now, dtype=bool)
    minutes = np.where(active, np.asarray(q.minutes_since_issue, np.float32), np.nan).astype(np.float32)
    event = np.full(len(active), "", dtype="U20")
    rows = q.active_state[active]
    if rows.size:
        event[active] = [f"{w}.TO.W.{int(e):04d}" for w, e in zip(W.wfo[rows], W.etn[rows])]
    report = dict(
        report,
        n_points=int(len(active)),
        n_active=int(active.sum()),
        known_from=snapshot.known_from,
        known_until=snapshot.known_until,
        fetched_at=snapshot.fetched_at,
        sources=list(snapshot.sources),
    )
    return LiveWarningInputs(active.astype(np.float32), minutes, event, report)


def live_tor_warning_inputs(
    lats,
    lons,
    times_utc,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    strict: bool = False,
    max_future_s: int = DEFAULT_MAX_FUTURE_S,
) -> LiveWarningInputs:
    """Fetch api.weather.gov now and return block W for the storm observations.

    Raises :class:`NwsLiveError` (fetch, coverage, or -- with ``strict`` -- data)
    whenever the inputs cannot be stated with training semantics; the caller
    then serves the model without block W.  Never returns zeros for "unknown".
    """

    t = nw.to_epoch_seconds(times_utc)
    if t.size == 0:
        z = np.zeros(0, dtype=np.float32)
        return LiveWarningInputs(z, z.copy(), np.zeros(0, dtype="U20"), {"n_points": 0})
    snapshot = fetch_alert_snapshot(int(t.min()), timeout=timeout)
    return inputs_from_snapshot(lats, lons, t, snapshot, strict=strict, max_future_s=max_future_s)
