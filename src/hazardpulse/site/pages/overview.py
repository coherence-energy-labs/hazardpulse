"""The home page and the live overview: every hazard's current headline, on one map."""
from __future__ import annotations

from hazardpulse.site import fmt, maps
from hazardpulse.site.data import SiteData, basin_of, ri_source_label, storm_name
from hazardpulse.site.hazards import EARTHQUAKE, HURRICANE, TORNADO
from hazardpulse.site.pages import common
from hazardpulse.site.places import coords, describe_far, describe_us, fe_region

esc = fmt.esc


# ------------------------------------------------------------------------------------------------
# the three hazard cards (home and /live/)
# ------------------------------------------------------------------------------------------------

def _card(key: str, d: SiteData, *, label: str, where: str, count: str, empty: str) -> str:
    h = {"eq": EARTHQUAKE, "hu": HURRICANE, "to": TORNADO}[key]
    head = d.headlines[key]
    gate = "" if head.gate in ("pass", "unknown") else common.gate_chip(head.gate)
    if head.probability is None:
        body = f'<p class="hazard-card-empty">{empty}</p>'
    else:
        change = common.since_previous(head)
        body = (f'<div class="hazard-card-figure">{fmt.chance(head.probability, big=True)}</div>'
                f'<p class="hazard-card-label">{label}</p>'
                f'<p class="hazard-card-where">{where}</p>'
                + (f'<p class="hazard-card-change">{change}</p>' if change else ""))
    issued = fmt.time_tag(head.issued_at) if head.issued_at else "&mdash;"
    fid = f' data-forecast="{esc(head.forecast_id)}"' if head.forecast_id else ""
    return (f'<a class="card card-link hazard-card hz-{key}" href="{h.path}"{fid}>'
            f'<div class="hazard-card-top">{common.hazard_label(key)}<span class="hazard-card-window">next '
            f"{h.window}</span></div>{body}"
            f'<p class="hazard-card-meta">{count} &middot; issued {issued} {gate}</p>'
            f'<span class="card-cta">{h.name} forecast</span></a>')


def hazard_cards(d: SiteData) -> str:
    eq = d.earthquake
    dom = eq.get("forecast_domain") or {}
    n_cells = int(dom.get("n_lat", 0)) * int(dom.get("n_lon", 0))
    eq_head = d.headlines["eq"]
    top_cell = max(eq.get("active_cells") or [{}], key=lambda c: float(c.get("probability") or 0))
    eq_card = _card(
        "eq", d,
        label="Highest chance of an M6+ earthquake in any 2&deg; grid cell",
        where=(f"{esc(eq_head.where)} <span class=\"coord\">{coords(top_cell['lat'], top_cell['lon'], 0)}</span>"
               if top_cell.get("lat") is not None else ""),
        count=f"{n_cells:,} cells forecast" if n_cells else "no forecast",
        empty="No earthquake forecast is published right now.")

    storms = d.hurricanes.get("storms") or []
    hu_head = d.headlines["hu"]
    top = max(storms, key=lambda s: float(s.get("ri_probability") or 0)) if storms else {}
    hu_card = _card(
        "hu", d,
        label=f"Highest chance of rapid intensification: {esc(hu_head.where)}",
        where=(f"{esc(basin_of(top))} &middot; source: {esc(ri_source_label(top))}" if top else ""),
        count=fmt.plural(len(storms), "active storm"),
        empty="No tropical cyclones are active anywhere right now.")

    tstorms = d.tornadoes.get("storms") or []
    to_card = _card(
        "to", d,
        label="Highest chance a tracked thunderstorm produces a tornado",
        where=esc(d.headlines["to"].where),
        count=fmt.plural(len(tstorms), "storm") + " tracked",
        empty="No thunderstorms are being tracked over the US right now.")
    return f'<div class="cards cards-3 hazard-cards">{eq_card}{hu_card}{to_card}</div>'


# ------------------------------------------------------------------------------------------------
# the combined map
# ------------------------------------------------------------------------------------------------

def world_map(d: SiteData, ident: str = "world-map") -> str:
    proj = maps.WORLD
    eq = d.earthquake
    marks = [maps.heat_grid(proj, eq)]
    cells = sorted(eq.get("active_cells") or [], key=lambda c: -float(c.get("probability") or 0))[:5]
    for i, c in enumerate(cells, 1):
        marks.append(maps.pin(proj, c["lat"], c["lon"], str(i), href=f"/live/earthquake/#cell-{i}",
                              title=f"Earthquake: {fe_region(c['lat'], c['lon'])}, {fmt.pct_plain(c['probability'])} "
                                    "chance of M6+ in 30 days", cls="pin-eq"))
    for s in d.hurricanes.get("storms") or []:
        marks.append(maps.storm(proj, s["lat"], s["lon"], storm_name(s), href=f"/live/hurricane/#storm-{esc(s.get('storm_id'))}",
                                title=f"{storm_name(s)}: {fmt.pct_plain(s.get('ri_probability'))} chance of rapid "
                                      "intensification in 24 hours", cls=fmt.level(s.get("ri_probability"))))
    tstorms = sorted(d.tornadoes.get("storms") or [], key=lambda s: -float(s.get("tornado_probability") or 0))
    for s in tstorms[:40]:
        marks.append(maps.dot(proj, s["lat"], s["lon"], cls=f"dot-to {fmt.level(s.get('tornado_probability'))}",
                              title=f"Thunderstorm {describe_us(s['lat'], s['lon'])}: "
                                    f"{fmt.pct_plain(s.get('tornado_probability'))} chance of a tornado within 60 minutes",
                              r=2.2))
    key = ('<ul class="map-key">'
           '<li><span class="key-swatch heat h3" aria-hidden="true"></span>Earthquake chance by cell (shaded)</li>'
           '<li><span class="key-pin pin-eq" aria-hidden="true">1</span>Highest earthquake cells</li>'
           '<li><span class="key-storm" aria-hidden="true"></span>Tropical cyclone</li>'
           '<li><span class="key-dot dot-to" aria-hidden="true"></span>Tracked thunderstorm</li></ul>')
    return maps.figure(
        proj, "".join(marks), ident=ident,
        title="Map of the current forecasts",
        desc="Shaded 2-degree cells show the 30-day chance of an M6+ earthquake; numbered pins mark the five "
             "highest cells; ringed markers are active tropical cyclones; small dots are thunderstorms tracked "
             "over the US. The tables on the forecast pages list the same information.",
        legend=key)


# ------------------------------------------------------------------------------------------------
# home
# ------------------------------------------------------------------------------------------------

def _proofs(d: SiteData) -> str:
    """How each served model tested, from the evidence bound to the served artifacts."""
    ev = d.evidence
    cards = []
    eq = ev.get("earthquake")
    if eq:
        t = eq["test"]
        auc = (t.get("auc") or {}).get("value")
        when = str(t.get("when") or t.get("period") or "").replace("-", "&ndash;")
        how = "a second look at those years" if t.get("second_read") else "scored once"
        ref = ((eq.get("vs") or {}).get("A") or {}).get("auc") or {}
        ref_auc = auc - ref["diff"] if auc is not None and ref.get("diff") is not None else None
        cards.append(("eq", fmt.pct(auc, 0) if auc is not None else "&mdash;",
                      "of the time, a cell that went on to have an M6+ earthquake was ranked above one that did not"
                      + (f" (a long-term seismicity map alone: {fmt.pct(ref_auc, 1)})" if ref_auc is not None else ""),
                      f"Every 2&deg; cell, {when} ({how})"))
    hu = ev.get("hurricane")
    if hu:
        t = hu["test"]
        beaten = [c["against"] for c in hu.get("claims", []) if c.get("better")]
        name = esc(hu["candidate_name"].split(" (")[0])
        cards.append(("hu", fmt.pct(t.get("auc"), 0) if t.get("auc") is not None else "&mdash;",
                      f"ranking accuracy of {name}, the probability we publish for the Atlantic, East and Central "
                      "Pacific" + (f"; in the same test it beat {fmt.join([esc(b) for b in beaten])}" if beaten else "")
                      + ". Elsewhere we publish our v8.2 model, which has no test in those basins",
                      f"The {esc(t.get('when'))} season, scored once, {t.get('n', 0):,} forecast cycles"))
    to = ev.get("tornado")
    if to:
        t = to["test"]
        pt = ((to.get("probtor_final") or {}).get("probtor_tiebroken") or {}).get("auc")
        cards.append(("to", fmt.pct(t.get("auc"), 0) if t.get("auc") is not None else "&mdash;",
                      "of the time, a storm that went on to produce a tornado was ranked above one that did not"
                      + (f" (NOAA&rsquo;s ProbTor on the same storms: {fmt.pct(pt, 0)}, its whole-percent ties broken)"
                         if pt is not None else ""),
                      "Every storm NOAA tracked in 2025, scored once"))
    if not cards:
        return '<p class="muted">No final test is bound to the served models in this build.</p>'
    items = "".join(
        f'<div class="proof hz-{k}">{common.hazard_label(k)}<p class="proof-figure">{fig}</p>'
        f'<p class="proof-label">{label}</p><p class="proof-source">{src}</p></div>'
        for k, fig, label, src in cards)
    return f'<div class="proof-strip">{items}</div>'


EMERGENCY_LINE = (
    '<p class="notice notice-official"><span class="notice-icon" aria-hidden="true"></span><span>In an emergency, '
    'follow official warnings: the <a href="https://www.weather.gov/" rel="noopener">National Weather Service</a>, '
    'the <a href="https://www.nhc.noaa.gov/" rel="noopener">National Hurricane Center</a>, the '
    '<a href="https://earthquake.usgs.gov/" rel="noopener">USGS</a>, your national meteorological and geological '
    "services, and your local authorities. HazardPulse publishes research forecasts, not warnings.</span></p>")

STEPS = (
    ("Public data in", "Earthquake catalogs from the USGS, tropical cyclone guidance from NOAA&rsquo;s National "
                       "Hurricane Center, and NOAA&rsquo;s live storm tracking and National Weather Service warnings."),
    ("A fixed, versioned model", "Each model was chosen by a test written down before the data was scored, and is "
                                 "recorded in the registry with the hash of its file. Changing it means a new "
                                 "version and a new test."),
    ("Checked before it is published", "Automatic quality checks run on every forecast: fresh inputs, a valid "
                                       "schema, sane values, a complete record. A failed blocking check stops it."),
    ("Frozen, then scored", "Every forecast is saved with its inputs and hashes, chained into a ledger that shows "
                            "any later edit, and scored against what actually happened."),
)


LIVE_STATS = (
    ("quakes_day", "earthquakes M2.5+ in the last 24 hours", "eq"),
    ("storms", "tropical cyclones active worldwide", "hu"),
    ("tornado_warnings", "tornado warnings in effect in the US", "to"),
    ("tracked_thunderstorms", "thunderstorms tracked in the latest tornado forecast", "to"),
)


def live_hero(d: SiteData, *, eyebrow: str, title: str, lede: str, ident: str = "hero", cta: str = "") -> str:
    """The live hero: a globe of what is happening now, the visitor's own area, and live counters.

    The edge worker fills ``#near-you``, the ``data-live`` counters and the visitor's position on the globe from
    the live feeds as the page is served; /assets/app.js keeps them current. Without either, the page still
    says what it is and shows the static map."""
    from hazardpulse.site.shell import asset
    stats = "".join(
        f'<li class="live-stat live-stat-{k}"><span class="live-stat-figure" data-live="{key}">&mdash;</span>'
        f'<span class="live-stat-label">{label}</span></li>' for key, label, k in LIVE_STATS)
    globe = (
        f'<div class="globe-wrap" id="{ident}-globe-wrap">'
        f'<div id="globe" class="globe" data-land="{asset("maps/land-2048.webp")}" data-globe="/data/globe.json" hidden>'
        '<canvas class="globe-gl" aria-hidden="true"></canvas>'
        '<canvas class="globe-overlay" role="img" aria-label="A globe showing the earthquakes of the last day, active '
        'tropical cyclones, tornado warnings and the 30-day earthquake forecast"></canvas>'
        '<div class="globe-tip" role="status" hidden></div>'
        '<p class="globe-hint">Drag to turn the globe &middot; select a marker for details</p></div>'
        f'<div class="globe-fallback">{world_map(d, ident + "-map")}</div>'
        '<ul class="globe-key" aria-label="Globe key">'
        '<li><span class="gk gk-quake" aria-hidden="true"></span>Earthquake, last 24 h</li>'
        '<li><span class="gk gk-storm" aria-hidden="true"></span>Tropical cyclone</li>'
        '<li><span class="gk gk-warn" aria-hidden="true"></span>Tornado warning</li>'
        '<li><span class="gk gk-heat" aria-hidden="true"></span>30-day earthquake forecast</li>'
        '<li><span class="gk gk-you" aria-hidden="true"></span>You</li></ul></div>')
    return (
        f'<section class="live-hero" aria-labelledby="{ident}-title"><div class="container live-hero-grid">'
        f'<div class="live-hero-copy"><p class="eyebrow eyebrow-live"><span class="live-dot" aria-hidden="true"></span>'
        f'{eyebrow}</p><h1 id="{ident}-title">{title}</h1><p class="lede">{lede}</p>'
        '<div id="near-you" class="near-you" aria-live="polite"></div>'
        f"{cta}"
        # the page speaks to the visitor's own location, so it says what it is beside that, not further down
        '<p class="live-hero-note">HazardPulse publishes research forecasts and relays official reports; it is not '
        'a warning service. In an emergency, follow your official warnings and local authorities. '
        '<a href="/legal/disclaimer/">Disclaimer</a></p></div>'
        f'<div class="live-hero-visual">{globe}</div></div>'
        f'<div class="container"><ul class="live-stats" aria-label="Live counts">{stats}</ul>'
        '<p class="live-updated">Live counts and events update every minute &middot; '
        '<span class="live-clock" data-live-clock>as of this page view</span></p></div></section>')


def live_feed(title: str = "Happening now", ident: str = "happening") -> str:
    return common.section(
        ident, title,
        '<ol id="live-feed" class="live-feed" aria-live="polite">'
        '<li class="feed-empty">The latest earthquakes, storm advisories and tornado warnings appear here.</li></ol>'
        '<p class="section-foot">Observations from the <a href="https://earthquake.usgs.gov/" rel="noopener">USGS</a>, '
        'the <a href="https://www.nhc.noaa.gov/" rel="noopener">National Hurricane Center</a> and the '
        '<a href="https://www.weather.gov/" rel="noopener">National Weather Service</a>, as they publish them. '
        "Storms outside NOAA&rsquo;s areas come from the HazardPulse forecast feed.</p>",
        intro="What the agencies that observe hazards have just reported, newest first.")


def home(d: SiteData) -> str:
    built = fmt.time_tag(d.built_at) if d.built_at else "&mdash;"
    hero = live_hero(
        d, eyebrow="Live &middot; updated every minute",
        title="Hazards near you and around the world, right now.",
        lede="Earthquakes as they happen, every active tropical cyclone and tornado warnings &mdash; beside our "
             "forecasts of what is likely next, each traceable to its data, its model and its track record.",
        cta=('<div class="cta-row"><a class="btn btn-primary btn-on-dark" href="/live/">Open the live map</a>'
             '<a class="btn btn-ghost" href="/verification/">How the forecasts score</a></div>'))
    feed = live_feed()
    cards = common.section(
        "next", "What is likely next", hazard_cards(d) + EMERGENCY_LINE,
        intro=f"Our forecasts, updated {built}. The three windows differ &mdash; 30 days, 24 hours, 60 minutes &mdash; "
              "so the three numbers are not comparable with each other.")
    proof = common.section(
        "proof", "How the forecasts have tested", _proofs(d) +
        '<p class="section-foot"><a href="/verification/">The full track record</a> &middot; '
        '<a href="/methods/">How each test was designed</a></p>',
        intro="Each figure is read from the results files of the model version that publishes the number, in "
              "tests written down before the data was scored; the earthquake figure is a later second look at "
              "its test years, and says so. Ranking accuracy (AUC) is 50% for a coin flip and 100% for a perfect "
              "ranking.")
    steps = "".join(f'<li class="step"><h3>{t}</h3><p>{b}</p></li>' for t, b in STEPS)
    how = common.section("how", "How a forecast is made", f'<ol class="steps">{steps}</ol>')
    build = common.section(
        "build", "Build on it",
        '<div class="cards cards-3">'
        '<a class="card card-link" href="/api/"><h3>Open API</h3><p>Every live forecast, its record and its track '
        'record as JSON, and the live feed itself. No key needed.</p><span class="card-cta">API reference</span></a>'
        f'<a class="card card-link" href="{d_release()}"><h3>Open data</h3><p>The research datasets behind every '
        'model, in versioned releases with a hash for every file.</p><span class="card-cta">Data releases</span></a>'
        '<a class="card card-link" href="/registry/"><h3>Model registry</h3><p>Every model version that has published '
        'a number, why it was promoted, and why it was retired.</p><span class="card-cta">See the models</span></a>'
        "</div>")
    return f'<main id="main" class="page page-home">{hero}{feed}{cards}{proof}{how}{build}</main>'


def d_release() -> str:
    from hazardpulse.site.shell import DATA_RELEASE_URL
    return DATA_RELEASE_URL


# ------------------------------------------------------------------------------------------------
# /live/
# ------------------------------------------------------------------------------------------------

def _eq_table(d: SiteData, n: int = 10) -> str:
    cells = sorted(d.earthquake.get("active_cells") or [], key=lambda c: -float(c.get("probability") or 0))[:n]
    rows = [[f'<a href="/live/earthquake/#cell-{i}">{esc(fe_region(c["lat"], c["lon"]))}</a>',
             coords(c["lat"], c["lon"], 0), fmt.chance(c.get("probability"))]
            for i, c in enumerate(cells, 1)]
    return common.table(["Region", "Cell centre", "Chance, 30 days"], rows, num_cols=(2,),
                        caption="Highest 30-day chances of an M6+ earthquake")


def _hu_table(d: SiteData) -> str:
    storms = sorted(d.hurricanes.get("storms") or [], key=lambda s: -float(s.get("ri_probability") or 0))
    if not storms:
        return '<p class="empty">No tropical cyclones are active anywhere right now.</p>'
    rows = [[f'<a href="/live/hurricane/#storm-{esc(s.get("storm_id"))}">{esc(storm_name(s))}</a>',
             f'{esc(s.get("category") or "&mdash;")}, {fmt.num(s.get("vmax_kt"))} kt',
             esc(describe_far(s["lat"], s["lon"])), fmt.chance(s.get("ri_probability")), esc(ri_source_label(s))]
            for s in storms]
    return common.table(["Storm", "Strength", "Position", "Chance, 24 h", "Source"], rows, num_cols=(3,),
                        caption="Active tropical cyclones")


def _to_table(d: SiteData, n: int = 10) -> str:
    storms = sorted(d.tornadoes.get("storms") or [], key=lambda s: -float(s.get("tornado_probability") or 0))
    if not storms:
        return '<p class="empty">No thunderstorms are being tracked over the US right now.</p>'
    rows = [[f'<a href="/live/tornado/#storm-{i}">{esc(describe_us(s["lat"], s["lon"]))}</a>',
             "Yes" if ((s.get("v3") or {}).get("nws_warning") or {}).get("active") else "No",
             fmt.chance(s.get("tornado_probability"))]
            for i, s in enumerate(storms[:n], 1)]
    return common.table(["Storm location", "NWS tornado warning", "Chance, 60 min"], rows, num_cols=(2,),
                        caption=f"Highest tornado chances: top {min(n, len(storms))} of {len(storms)} tracked storms")


def live(d: SiteData) -> str:
    hero = live_hero(
        d, ident="live", eyebrow="Live map &middot; updated every minute",
        title="Live hazards, worldwide",
        lede="Every earthquake of the last day, every active tropical cyclone and every US tornado warning, on one "
             "globe with our 30-day earthquake forecast. Turn it, and select any marker.")
    body = live_feed("Latest events", "latest")
    body += common.section("now", "What is likely next", hazard_cards(d),
                           intro="Our forecasts, from the latest published forecast records.")
    body += common.section(
        "earthquake", "Earthquake: highest chances", _eq_table(d) +
        '<p class="section-foot"><a href="/live/earthquake/">All cells and the full earthquake map</a></p>',
        intro="The chance of at least one magnitude 6+ earthquake in the 2&deg; cell over the next 30 days.")
    body += common.section(
        "hurricane", "Hurricanes: rapid intensification", _hu_table(d) +
        '<p class="section-foot"><a href="/live/hurricane/">Each storm in detail</a></p>',
        intro="The chance that the storm&rsquo;s maximum sustained winds rise by 30 knots or more in the next 24 hours.")
    body += common.section(
        "tornado", "Tornadoes: highest chances", _to_table(d) +
        '<p class="section-foot"><a href="/live/tornado/">Every tracked storm</a></p>',
        intro="The chance that a thunderstorm tracked by NOAA ProbSevere, over and near the contiguous US, "
              "produces a tornado within 60 minutes.")
    return f'<main id="main" class="page page-live">{hero}{body}</main>'
