"""The proof pages: the track record (/verification/), the evidence (/evidence/) and system status
(/ops/status/). Built from the verification summary, the scorers' prospective summaries, the evidence
indexes and the model evidence bound to each served artifact."""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter

from hazardpulse.site import fmt
from hazardpulse.site.data import SiteData, read_json
from hazardpulse.site.hazards import HAZARDS
from hazardpulse.site.pages import common

esc = fmt.esc

# A live skill score is quoted only once this many events have been observed for the model version:
# with fewer, a model that always says "no" scores as well as a skilful one (the tornado record had a
# "live BSS 1.00" on 83 storm forecasts and zero tornadoes).
LIVE_MIN_EVENTS = 10

# What each quality check tests, in words (tests/test_site_pages.py fails if the engine gains a check
# this table does not describe).
GATE_TEXT = {
    "G0_SCHEMA_VALIDITY": ("Well-formed", "The probability, and any range, are finite numbers."),
    "G1_SOURCE_FRESHNESS": ("Fresh inputs", "The input data is recent enough for the hazard&rsquo;s schedule."),
    "G3_PROVENANCE_COMPLETE": ("Complete record", "The model version, the hashes of the model file, inputs and "
                                                  "receipt, and the replay file are all recorded."),
    "G4_CALIBRATION_FLOOR": ("Calibration", "The model&rsquo;s measured calibration on live forecasts is within "
                                            "bounds (a warning until enough forecasts with events have closed)."),
    "G5_SPATIOTEMPORAL_SANITY": ("Sane values", "Coordinates are valid, the probability is between 0 and 1, and the "
                                                "grid is no finer than the published resolution."),
    "G6_ALERT_HARM_GUARD": ("No confident alarm on weak evidence", "A high chance is never shown on inputs unlike "
                                                                   "the training data or with a very wide range."),
    "G8_REPLAYABILITY_REQUIRED": ("Reproducible", "A replay file exists, so the forecast can be re-run."),
    "G9_PUBLIC_PROJECTION_POLICY": ("No official terms", "Labels never use terms reserved for official warnings."),
}


def _hazard_name(key: str) -> str:
    return HAZARDS[key].name if key in HAZARDS else key


def _key_of(forecast_id: str) -> str:
    return str(forecast_id or "").split("_", 1)[0]


# ------------------------------------------------------------------------------------------------
# /verification/
# ------------------------------------------------------------------------------------------------

def _prospective(d: SiteData, name: str) -> dict:
    return read_json(d.root / "results" / f"{name}_prospective" / "prospective_summary.json", {}) or {}


def _first_issue_of(d: SiteData, key: str, version: str) -> str | None:
    times = [e.get("issued_at") for e in d.ledger_entries
             if _key_of(e.get("forecast_id")) == key and e.get("model_version") == version and e.get("issued_at")]
    return min(times) if times else None


def _live_rows(d: SiteData) -> list[list[str]]:
    """Every model version's live record, published or retired."""
    rows = []
    served = {k: (d.verification_by_key.get(k) or {}).get("model_version") for k in HAZARDS}
    eq = _prospective(d, "earthquake").get("by_model_version") or {}
    for version, r in sorted(eq.items()):
        n_ev = int(r.get("n_observed_events") or 0)
        rows.append(["eq", version, int(r.get("n_matured_forecasts") or 0), n_ev, r.get("mean_auc"),
                     r.get("event_weighted_information_gain_per_event"), "information gain per quake vs a uniform map"])
    to = _prospective(d, "tornado").get("pooled_by_model_version") or {}
    for version, r in sorted(to.items()):
        rows.append(["to", version, int(r.get("n_storm_forecasts") or 0), int(r.get("n_positive") or 0),
                     r.get("auc"), r.get("bss_vs_causal_climatology"), "Brier skill vs climatology"])
    hu = _prospective(d, "hurricane")
    if hu:
        rows.append(["hu", "all published hurricane numbers", int(hu.get("total_storm_predictions") or 0),
                     int(hu.get("total_ri_events") or 0), None, None, ""])
    out = []
    for key, version, n, n_ev, auc, skill, skill_name in rows:
        status = ("published" if version == served.get(key) else
                  ("&mdash;" if key == "hu" else "retired"))
        enough = n_ev >= LIVE_MIN_EVENTS
        if enough:
            score = f"ranking {fmt.pct(auc, 1)}" if auc is not None else "&mdash;"
            if skill is not None and skill_name:
                score += f"; {skill_name} {fmt.num(skill, 2)}"
        else:
            score = (f"too few events to score yet (need {LIVE_MIN_EVENTS})" if n else "nothing closed yet")
        out.append([common.hazard_label(key), f"<code>{esc(version)}</code>" if key != "hu" else esc(version),
                     status, f"{n:,}", f"{n_ev:,}", score])
    return out


def _hazard_record_card(d: SiteData, key: str) -> str:
    v = d.verification_by_key.get(key) or {}
    h = HAZARDS[key]
    ev = d.evidence.get({"eq": "earthquake", "hu": "hurricane", "to": "tornado"}[key])
    version = v.get("model_version") or ""
    rows = [("Published model", f"<code>{esc(version)}</code>")]
    if key == "hu":
        rows[0] = ("Published numbers", "NOAA DTOPS (SHIPS-RII as fallback) for the Atlantic, East and Central "
                                        "Pacific; HazardPulse v8.2 elsewhere, tested only on Atlantic and East "
                                        "Pacific cases")
    rows.append(("Window", h.window))
    # the test result, from the evidence bound to the served artifact
    if ev:
        t = ev["test"]
        if key == "eq":
            auc = (t.get("auc") or {})
            rows.append(((f"Second look at {fmt.years(t.get('when'))}" if t.get("second_read")
                          else f"Test, {fmt.years(t.get('when'))} (scored once)"),
                         f"ranking accuracy {fmt.pct(auc.get('value'), 1)}; information gain "
                         f"{fmt.num((t.get('ig_per_target') or {}).get('value'), 2)} nats per quake over a uniform map"))
        elif key == "hu":
            rows.append((f"Test, {esc(t.get('when'))} season (scored once)",
                         f"DTOPS ranking accuracy {fmt.pct(t.get('auc'), 1)}, Brier skill {fmt.num(t.get('bss'), 2)}; "
                         f"{t.get('n', 0):,} cycles, {t.get('events', 0):,} rapid intensifications"))
        else:
            rows.append((f"Test, {esc(t.get('when'))} (scored once)",
                         f"ranking accuracy {fmt.pct(t.get('auc'), 1)}, Brier skill {fmt.num(t.get('bss'), 2)}; "
                         f"{int(t.get('n') or 0):,} storm observations, {int(t.get('pos') or 0):,} tornadic"))
    else:
        rows.append(("Test", "No final test is bound to the published model in this build"))
    # the live record of THIS version
    live = _published_live(d, key, version)
    rows.append(("Live record of this version", live))
    storage = v.get("forecast_storage") or {}
    rows.append(("Forecasts recorded", f"{int(storage.get('n_replay_artifacts') or 0):,} "
                                       f"({int(storage.get('n_scored_forecasts') or 0):,} closed and scored, "
                                       f"{int(storage.get('n_backlog') or 0):,} waiting)"))
    ledger = v.get("ledger") or {}
    if ledger.get("supported"):
        rows.append(("Ledger", f'<a href="{esc(ledger.get("path"))}">{int(ledger.get("n_rows") or 0):,} chained rows</a>, '
                               f"{int(ledger.get('prev_hash_mismatches') or 0):,} breaks"))
    return (f'<article class="card record-card hz-{key}"><div class="record-card-head">{common.hazard_label(key)}</div>'
            f"{common.facts(rows)}</article>")


def _published_live(d: SiteData, key: str, version: str) -> str:
    if key == "eq":
        r = (_prospective(d, "earthquake").get("by_model_version") or {}).get(version)
        if not r:
            first = _first_issue_of(d, "eq", version)
            if first:
                t = fmt.parse_time(first) + dt.timedelta(days=30)
                return f"none yet: its first 30-day window closes on {fmt.date(t)}"
            return "none yet"
        n_ev = int(r.get("n_observed_events") or 0)
        return (f"{int(r.get('n_matured_forecasts') or 0):,} closed forecasts, {n_ev:,} M6+ events"
                + (f"; ranking accuracy {fmt.pct(r.get('mean_auc'), 1)}" if n_ev >= LIVE_MIN_EVENTS else ""))
    if key == "to":
        r = (_prospective(d, "tornado").get("pooled_by_model_version") or {}).get(version)
        if not r:
            return "none yet"
        n, pos = int(r.get("n_storm_forecasts") or 0), int(r.get("n_positive") or 0)
        if pos < LIVE_MIN_EVENTS:
            return (f"{n:,} storm forecasts closed, {pos:,} followed by a tornado: too few tornadoes to score yet "
                    f"(the record is quoted from {LIVE_MIN_EVENTS})")
        return f"{n:,} storm forecasts, {pos:,} tornadic; ranking accuracy {fmt.pct(r.get('auc'), 1)}"
    hu = _prospective(d, "hurricane")
    n, ev = int(hu.get("total_storm_predictions") or 0), int(hu.get("total_ri_events") or 0)
    if ev < LIVE_MIN_EVENTS:
        return (f"{n:,} storm forecasts closed against the best track, {ev:,} rapid intensifications: too few to "
                "score yet")
    return f"{n:,} storm forecasts, {ev:,} rapid intensifications"


def _reliability(d: SiteData) -> str:
    to = d.evidence.get("tornado")
    table = (to or {}).get("reliability")
    if not table or not table.get("bins"):
        return "<p>No calibration table is bound to the published tornado model in this build.</p>"
    rows = []
    for b in table["bins"]:
        if not b.get("n"):
            continue
        ci = b.get("observed_ci") or [None, None]
        rows.append([f"{fmt.pct(b['lo'], 1)}&ndash;{fmt.pct(b['hi'], 1)}", f"{int(b['n']):,}",
                     fmt.pct(b.get("mean_forecast"), 2), fmt.pct(b.get("observed"), 2),
                     f"{fmt.pct(ci[0], 2)}&ndash;{fmt.pct(ci[1], 2)}" if ci[0] is not None else "&mdash;"])
    return (common.table(["Forecast range", "Storm observations", "Average forecast", "Tornado followed",
                          "95% interval"], rows, num_cols=(1, 2, 3, 4),
                         caption="Tornado model, every storm of 2025 (scored once)")
            + '<p class="section-foot">In a well-calibrated row the two middle numbers agree. '
              '<a href="/verification/tornado/">Every breakdown of the tornado test</a></p>')


def verification(d: SiteData) -> str:
    sysm = d.verification.get("system") or {}
    hero = common.hero(
        "Track record", "How the forecasts have scored",
        "Two kinds of evidence, kept apart. <strong>Test results</strong>: how each published model scored on years "
        "held back from its development, in tests written down before the data was scored. <strong>Live "
        "record</strong>: how the forecasts we actually published scored once their windows closed. If a live model "
        "is not scored yet, this page says so directly. It also shows any scoring backlogs.",
        meta=[("Scores as of", fmt.time_tag(d.verification.get("score_as_of")))])
    tiles = [
        (f"{int(sysm.get('frozen_forecasts') or 0):,}", "forecasts recorded", "Every published forecast, frozen with its inputs."),
        (f"{int(sysm.get('scored_forecasts') or 0):,}", "closed and scored", "Windows that have ended and been scored against what happened."),
        (f"{int(sysm.get('matured_unscored_backlog') or 0):,}", "waiting to be scored", "Windows that have ended but are not scored yet."),
        (f"{int(sysm.get('hash_chain_mismatches') or 0):,}", "breaks in the ledgers", "Each entry carries the previous entry&rsquo;s hash, so an edit to history would show here."),
    ]
    tiles_html = '<div class="stats">' + "".join(
        f'<div class="stat"><p class="stat-figure">{a}</p><p class="stat-label">{b}</p><p class="stat-note">{c}</p></div>'
        for a, b, c in tiles) + "</div>"
    alerts = sysm.get("alerts") or []
    attention = ("" if not alerts else
                 '<div class="notice notice-warn"><p><strong>Needs attention</strong></p><ul>'
                 + "".join(f"<li>{esc(a)}</li>" for a in alerts) + "</ul></div>")
    cards = '<div class="cards cards-3">' + "".join(_hazard_record_card(d, k) for k in ("eq", "hu", "to")) + "</div>"
    live = common.table(["Hazard", "Model version", "Status", "Forecasts closed", "Events observed", "Score"],
                        _live_rows(d), caption="Live record by model version", num_cols=(3, 4))
    downloads = ('<ul class="link-list">'
                 '<li><a href="/data/verification-summary.json">Verification summary (JSON)</a></li>'
                 '<li><a href="/data/earthquake-ledger.jsonl">Earthquake ledger (JSONL)</a></li>'
                 '<li><a href="/data/hurricane-ledger.jsonl">Hurricane ledger (JSONL)</a></li>'
                 '<li><a href="/data/tornado-ledger.jsonl">Tornado ledger (JSONL)</a></li>'
                 '<li><a href="/data/evidence/prediction-ledger.json">Every forecast, all hazards (JSON)</a></li>'
                 '<li><a href="/api/v1/verification/summary">The same summary through the API</a></li></ul>')
    body = (common.section("summary", "In numbers", tiles_html + attention)
            + common.section("by-hazard", "By hazard", cards)
            + common.section("live", "Live record, by model version", live,
                             intro="Every model version that has published a number, including retired ones. A score "
                                   f"is shown only once {LIVE_MIN_EVENTS} or more events have been observed for that "
                                   "version: with fewer, a model that always says &ldquo;no&rdquo; would look as good "
                                   "as a skilful one.")
            + common.section("calibration", "Does a 10% forecast come true 1 time in 10?", _reliability(d),
                             intro="Calibration compares what was forecast with what happened. The published tornado "
                                   "model&rsquo;s 2025 test, split by how high the forecast was. Live calibration "
                                   "is measured once enough forecasts with events have closed.")
            + common.section("downloads", "Download the records", downloads))
    return f'<main id="main" class="page page-record">{hero}{body}</main>'


# ------------------------------------------------------------------------------------------------
# /evidence/
# ------------------------------------------------------------------------------------------------

def _short(h: str | None, n: int = 12) -> str:
    s = str(h or "")
    s = s.split(":", 1)[1] if s.startswith("sha256:") else s
    return f'<code title="{esc(s)}">{esc(s[:n])}&hellip;</code>' if s else "&mdash;"


def _current_rows(d: SiteData) -> list[list[str]]:
    env = {e.get("forecast_id"): e for e in d.envelopes}
    rows = []
    for key in ("eq", "hu", "to"):
        head = d.headlines[key]
        fid = head.forecast_id
        if not fid:
            continue
        e = env.get(fid) or {}
        rows.append([common.hazard_label(key),
                     f'<a href="/data/replay/{esc(fid)}.json"><code>{esc(fid)}</code></a>'
                     f'<span class="cell-sub"><code>{esc(e.get("model_version") or "")}</code></span>',
                     fmt.time_tag(head.issued_at),
                     f'<span class="cell-sub">in</span> {_short(e.get("input_hash"))}<br>'
                     f'<span class="cell-sub">out</span> {_short(e.get("output_hash"))}',
                     f'<a href="/api/v1/gates/gdec_{esc(fid)}">{common.gate_chip(head.gate)}</a>'])
    return rows


def _gate_summary(d: SiteData, days: int = 7) -> str:
    now = fmt.parse_time(d.built_at) or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(days=days)
    counts: dict[str, Counter] = {k: Counter() for k in HAZARDS}
    reasons: Counter = Counter()
    for g in d.gate_decisions:
        t = fmt.parse_time(g.get("emitted_at"))
        k = _key_of(g.get("forecast_id"))
        if t is None or t < since or k not in counts:
            continue
        counts[k][g.get("decision")] += 1
        for r in list(g.get("reasons") or []) + list(g.get("warnings") or []):
            reasons[common.plain_gate_note(r)] += 1
    rows = [[common.hazard_label(k), f"{c['pass']:,}", f"{c['degrade']:,}", f"{c['block']:,}"]
            for k, c in counts.items()]
    out = common.table(["Hazard", "Passed", "Published with warnings", "Blocked"], rows, num_cols=(1, 2, 3),
                       caption=f"Quality-check outcomes, last {days} days")
    if reasons:
        out += ('<p class="section-foot">Most frequent findings: '
                + "; ".join(f"{esc(r)} ({n:,})" for r, n in reasons.most_common(4)) + ".</p>")
    return out


def _gate_list() -> str:
    from hazardpulse.gates import engine
    items = []
    for fn in engine._GATES:
        gid = fn.__name__.split("_", 1)[0].upper()
        full = next((k for k in GATE_TEXT if k.startswith(gid + "_")), None)
        name, text = GATE_TEXT.get(full, (full or gid, ""))
        items.append(f"<li><strong>{name}.</strong> {text}</li>")
    return f'<ol class="gate-list">{"".join(items)}</ol>'


def evidence(d: SiteData) -> str:
    index = sorted(d.replay_index, key=lambda i: str(i.get("forecast_id")), reverse=True)
    hero = common.hero(
        "Evidence", "The record behind every forecast",
        "Each published forecast is saved with the data it used, the model version, cryptographic hashes of its inputs "
        "and output, and the result of the automatic quality checks. Download any forecast&rsquo;s replay file to "
        "check it or reproduce it.",
        meta=[("Replay files", f"{sum(len(v) for v in d.replay_ids.values()):,}"),
              ("Ledger entries", f"{len(d.ledger_entries):,}")])
    current = common.table(["Hazard", "Forecast and model", "Issued", "Hashes (SHA-256)", "Quality checks"],
                           _current_rows(d), caption="The forecasts on the site now")
    gates = (_gate_summary(d)
             + "<p>Every forecast goes through these checks when it is published. Each ends in one of three outcomes: "
               "<strong>passed</strong>; <strong>published with warnings</strong> &mdash; a non-blocking check flagged "
               "something, and the warning is shown with the forecast; or <strong>blocked</strong> &mdash; the "
               "forecast is withheld from every page.</p>" + _gate_list())
    led_rows = []
    for e in sorted(d.ledger_entries, key=lambda e: str(e.get("issued_at")), reverse=True)[:20]:
        k = _key_of(e.get("forecast_id"))
        led_rows.append([common.hazard_label(k) if k in HAZARDS else esc(k),
                         f'<a href="{esc(e.get("replay_artifact"))}"><code>{esc(e.get("forecast_id"))}</code></a>',
                         fmt.time_tag(e.get("issued_at")), fmt.pct(e.get("probability")), _short(e.get("hash"))])
    ledger = ("<p>Every forecast is appended to a ledger in which each entry includes the hash of the one before it. "
              "Changing any past forecast would change its hash and break every link after it, so the history cannot "
              "be edited without it showing.</p>"
              + common.table(["Hazard", "Forecast", "Issued", "Headline chance", "Entry hash"], led_rows,
                             caption="The 20 most recent ledger entries", num_cols=(3,))
              + '<p class="section-foot"><a href="/data/evidence/prediction-ledger.json">The full ledger (JSON)</a> '
                '&middot; <a href="/data/evidence/provenance-envelopes.json">Input records (JSON)</a> &middot; '
                '<a href="/data/evidence/gate-decisions.json">Every quality-check record (JSON)</a></p>')
    replays = ("<p>Every forecast&rsquo;s replay file holds the inputs, the model version and the output exactly as "
               "published; the ledger above links each one. All of them are listed in "
               f'<a href="/data/evidence/replay-index.json">the replay index ({len(index):,} files, JSON)</a> and served '
               "from <code>/data/replay/&lt;forecast id&gt;.json</code>.</p>")
    key = read_json(d.dist / "data" / "evidence" / "public-key.json", {}) or {}
    verify = ("<ol>"
              "<li>Download a forecast&rsquo;s replay file from the table above or from <code>/data/replay/</code>.</li>"
              "<li>Recompute its SHA-256 hash and compare it with the output hash in the input record "
              "(<code>/api/v1/evidence/prov_&lt;forecast id&gt;</code>).</li>"
              "<li>Check that the ledger entry for the forecast carries the same hash and that each entry&rsquo;s "
              "<code>prev_hash</code> matches the entry before it.</li>"
              + ("<li>Where a forecast carries a signed receipt (earthquake and tornado forecasts do), verify "
                 "its signature with our Ed25519 public key: "
                 f'<a href="/data/evidence/public-key.json"><code>{esc(str(key.get("public_key_hex", ""))[:16])}&hellip;</code></a>.</li>'
                 if key.get("public_key_hex") else "")
              + "</ol><p>The scripts that do this are in the "
              '<a href="https://github.com/coherence-energy-labs/hazardpulse" rel="noopener">source repository</a> '
              "(<code>scripts/verify_forecast.py</code>).</p>")
    body = (common.section("current", "The forecasts on the site now", current)
            + common.section("checks", "Quality checks", gates)
            + common.section("ledger", "Forecast ledger", ledger)
            + common.section("replays", "Replay files", replays)
            + common.section("verify", "Check a forecast yourself", verify))
    return f'<main id="main" class="page page-evidence">{hero}{body}</main>'


# ------------------------------------------------------------------------------------------------
# /ops/status/
# ------------------------------------------------------------------------------------------------

def status(d: SiteData) -> str:
    now = fmt.parse_time(d.built_at) or dt.datetime.now(dt.timezone.utc)
    rows, overdue, flagged = [], [], []
    for key in ("eq", "hu", "to"):
        h = HAZARDS[key]
        head = d.headlines[key]
        t = fmt.parse_time(head.issued_at)
        age_h = (now - t).total_seconds() / 3600 if t else None
        late = age_h is None or age_h > h.max_age_hours
        if late:
            overdue.append(h.name)
        if head.gate in ("degrade", "block"):
            flagged.append(h.name)
        state = ('<span class="chip bad">Overdue</span>' if late else '<span class="chip good">Current</span>')
        rows.append([common.hazard_label(key), fmt.time_tag(head.issued_at), h.schedule,
                     f"after {h.max_age_hours} hours", state, common.gate_chip(head.gate)])
    if overdue:
        banner = ('<div class="notice notice-bad"><p><strong>Delayed:</strong> '
                  + ", ".join(overdue) + " has not published within its expected window.</p></div>")
    elif flagged:
        banner = ('<div class="notice notice-warn"><p><strong>All forecasts are current.</strong> '
                  + ", ".join(flagged) + " published with quality-check warnings (listed below).</p></div>")
    else:
        banner = '<div class="notice notice-good"><p><strong>All forecasts are current and passed every check.</strong></p></div>'
    sysm = d.verification.get("system") or {}
    notes = []
    for key in ("eq", "hu", "to"):
        head = d.headlines[key]
        for n in head.gate_notes:
            notes.append(f"<li>{common.hazard_label(key)} {esc(common.plain_gate_note(n))}</li>")
    hero = common.hero(
        "Status", "System status",
        "Whether each forecast is current, and what the quality checks found. This page is rebuilt every time a "
        "forecast is published, and scheduled runs can start late, so a forecast is flagged only once it passes the "
        "limit shown. For the answer at this moment, ask the API: "
        '<a href="/api/v1/ops/status"><code>/api/v1/ops/status</code></a>.',
        meta=[("As of", fmt.time_tag(now.isoformat()))])
    body = (common.section("forecasts", "Forecasts", banner + common.table(
                ["Hazard", "Latest forecast", "Schedule", "Overdue", "Status", "Quality checks"], rows,
                caption="Latest forecast per hazard"))
            + common.section("findings", "Quality-check findings on the current forecasts",
                             (f'<ul class="finding-list">{"".join(notes)}</ul>' if notes else
                              "<p>No findings: every current forecast passed every check.</p>")
                             + '<p class="section-foot"><a href="/evidence/#checks">What each check tests</a></p>')
            + common.section("integrity", "Records", common.facts([
                ("Breaks in the forecast ledgers", f"{int(sysm.get('hash_chain_mismatches') or 0):,}"),
                ("Closed forecasts waiting to be scored", f"{int(sysm.get('matured_unscored_backlog') or 0):,}"),
                ("Forecasts recorded", f"{int(sysm.get('frozen_forecasts') or 0):,}"),
            ])))
    return f'<main id="main" class="page page-status">{hero}{body}</main>'


def to_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))
