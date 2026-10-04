"""/live/earthquake/: the 30-day M6+ forecast for every 2-degree cell."""
from __future__ import annotations

from hazardpulse.site import fmt, maps
from hazardpulse.site.data import SiteData
from hazardpulse.site.hazards import EARTHQUAKE
from hazardpulse.site.pages import common
from hazardpulse.site.places import coords, describe_far, fe_region

esc = fmt.esc
N_LISTED = 20


def _catalog_start_year() -> str:
    from hazardpulse.earthquake.operational_forecast import CATALOG_START
    return CATALOG_START[:4]


def driver_sentence(cell: dict) -> str:
    """What moves the cell's number: the split of the clustering-model rate the boosted trees read
    as their two strongest inputs (``lambda_short`` from recent nearby quakes, ``lambda_long`` from the
    long-term rate)."""
    lam_long = float(cell.get("lambda_long") or 0.0)
    lam_short = float(cell.get("lambda_short") or 0.0)
    total = lam_long + lam_short
    year = _catalog_start_year()
    if total <= 0:
        return "This cell&rsquo;s modelled rate is at the model&rsquo;s floor: no recorded M5+ earthquake nearby."
    share = lam_short / total
    if share >= 0.5:
        return (f"Recent earthquakes nearby &mdash; aftershock-style clustering &mdash; account for {share:.0%} of this "
                f"cell&rsquo;s modelled rate; the rest is its long-term rate of M5+ earthquakes since {year}.")
    if share >= 0.1:
        return (f"Mostly this cell&rsquo;s long-term rate of M5+ earthquakes since {year}, raised by recent "
                f"earthquakes nearby ({share:.0%} of the modelled rate).")
    return (f"Almost entirely this cell&rsquo;s long-term rate of M5+ earthquakes since {year}; recent activity "
            "adds little.")


def _research(cell: dict) -> str:
    """Seismicity statistics the forecast does not use, kept apart and labelled as such."""
    rows = []
    if cell.get("rate_acceleration") is not None:
        rows.append(("Recent rate vs. this cell&rsquo;s background", f"{fmt.num(cell['rate_acceleration'], 2)}&times;"))
    if cell.get("b_value") is not None:
        rows.append(("Gutenberg&ndash;Richter b-value", fmt.num(cell["b_value"], 2)))
    if cell.get("ell_km") is not None:
        rows.append(("Correlation length", f"{fmt.num(cell['ell_km'], 1)} km"))
    if cell.get("spatial_concentration") is not None:
        rows.append(("Spatial spread of recent events", f"{fmt.num(cell['spatial_concentration'], 1)} km"))
    if cell.get("depth_trend") is not None:
        rows.append(("Depth trend", f"{fmt.num(cell['depth_trend'], 1)} km"))
    if not rows:
        return ""
    return common.disclosure(
        "Research statistics (not used by the forecast)",
        '<p class="small muted">Standard seismicity statistics of the recent catalog in this cell, shown for '
        "research context only. They do not set the probability, and no test of them as forecasts is published "
        "on this site.</p>" + common.facts(rows, "facts-compact"),
        cls="disclosure-research")


def _row(i: int, cell: dict) -> str:
    lat, lon = float(cell["lat"]), float(cell["lon"])
    n = int(cell.get("n_events") or 0)
    mmax = cell.get("max_mag")
    sub = (f"{fmt.plural(n, 'M2.5+ earthquake')} in the last 30 days"
           + (f", largest M{fmt.num(mmax, 1)}" if n and mmax is not None else ""))
    near = describe_far(lat, lon, min_population=50_000)
    details = common.facts([
        ("Chance of an M6+ earthquake in this cell, next 30 days",
         fmt.pct(cell.get("probability")) + (f" ({fmt.interval(cell)})" if fmt.interval(cell) else "")),
        ("Cell", f"2&deg; &times; 2&deg; centred on {coords(lat, lon, 0)}, {esc(near)}"),
        ("Recent earthquakes in the cell", sub),
    ])
    return (f'<li id="cell-{i}"><details class="row"><summary>'
            f'<span class="row-rank">{i}</span>'
            f'<span class="row-main"><span class="row-title">{esc(fe_region(lat, lon))}</span>'
            f'<span class="row-sub">{coords(lat, lon, 0)} &middot; {sub}</span></span>'
            f'<span class="row-figure">{fmt.chance(cell.get("probability"))}</span></summary>'
            f'<div class="row-body"><p class="row-lead"><strong>Why this cell:</strong> {driver_sentence(cell)}</p>'
            f"{details}{_research(cell)}</div></details></li>")


def _model_box(d: SiteData) -> str:
    eq = d.evidence.get("earthquake")
    replay = d.earthquake
    version = esc(replay.get("model_version") or "")
    if not eq:
        return common.facts([("Model version", f"<code>{version}</code>"),
                             ("Test", "No final test is bound to the served model in this build.")])
    t = eq["test"]
    g1 = eq.get("gear1")
    auc = (t.get("auc") or {})
    what = ("Boosted trees over each cell&rsquo;s long-term M5+ rate and its aftershock-style clustering after recent "
            "earthquakes nearby" + (", combined with GEAR1, a published global model of long-term earthquake "
                                     "rates from crustal strain and past seismicity (Bird et al. 2015)" if g1 else ""))
    when = str(t.get("when") or "").replace("-", "&ndash;")
    how = "a second look at those years" if t.get("second_read") else "scored once"
    active = t.get("auc_active_cells") or {}
    rows = [
        ("What the model is", what),
        (f"How it tested, {when}", f"Across every cell, a cell that went on to have an M6+ earthquake was ranked above "
                                   f"one that did not {_pct_ci(auc)} of the time ({how})."),
    ]
    if active.get("value") is not None:
        rows.append(("The harder question", f"Among recently active cells only, where telling cells apart is hardest: "
                                            f"{_pct_ci(active)}."))
    cr = t.get("calib_ratio") or {}
    if cr.get("value") is not None:
        over = float(cr["value"]) - 1.0
        rows.append(("Calibration in the test",
                     f"It forecast {fmt.pct(abs(over), 0)} {'more' if over > 0 else 'fewer'} cells with an M6+ "
                     f"earthquake than there were (ratio {fmt.num(cr['value'], 2)}, 95% interval "
                     f"{fmt.num((cr.get('ci') or [None, None])[0], 2)}&ndash;{fmt.num((cr.get('ci') or [None, None])[1], 2)})."))
    rows += [("Model version", f"<code>{version}</code>"),
             ("Full method", '<a href="/methods/#earthquake">Methods: earthquake</a>')]
    return common.facts(rows)


def _pct_ci(m: dict) -> str:
    ci = m.get("ci") or [None, None]
    return (f"{fmt.pct(m.get('value'), 1)} (95% interval {fmt.pct(ci[0], 1)}&ndash;{fmt.pct(ci[1], 1)})"
            if ci[0] is not None else fmt.pct(m.get("value"), 1))


def page(d: SiteData) -> str:
    replay = d.earthquake
    head = d.headlines["eq"]
    dom = replay.get("forecast_domain") or {}
    n_cells = int(dom.get("n_lat", 0)) * int(dom.get("n_lon", 0))
    cat = replay.get("source_catalog") or {}
    cells = sorted(replay.get("active_cells") or [], key=lambda c: -float(c.get("probability") or 0))
    lat_lo, lat_hi = dom.get("lat_min"), dom.get("lat_max")
    hero = common.hero(
        "Earthquake &middot; next 30 days",
        "Where a large earthquake is more likely in the next 30 days",
        "Earthquakes cannot be predicted. This forecast gives the chance of at least one magnitude 6+ earthquake in "
        "each 2&deg; grid cell of the world over the next 30 days, so you can see where the odds are higher than usual.",
        meta=[("Issued", fmt.time_tag(replay.get("issued_at"))),
              ("Cells forecast", f"{n_cells:,}" + (f" ({coords(lat_lo, 0, 0).split()[0]} to {coords(lat_hi, 0, 0).split()[0]})"
                                                 if lat_lo is not None and lat_hi is not None else "")),
              ("Earthquakes recorded, last 30 days", f"{int(cat.get('n_recent_events') or 0):,} (M2.5+, "
                                                     "USGS catalog)"),
              ("Expected M6+ cells, next 30 days",
               fmt.num((replay.get("operational_model") or {}).get("expected_positive_cells"), 1))])
    notices = common.gate_notice(head) + common.official_notice(
        EARTHQUAKE, "For earthquake information, alerts and safety guidance, follow")
    proj = maps.WORLD
    pins = "".join(maps.pin(proj, c["lat"], c["lon"], str(i), href=f"#cell-{i}", cls="pin-eq",
                            title=f"{i}. {fe_region(c['lat'], c['lon'])}: {fmt.pct_plain(c.get('probability'))}")
                   for i, c in enumerate(cells[:10], 1))
    themap = maps.figure(
        proj, maps.heat_grid(proj, replay) + pins, ident="eq-map",
        title="Map: 30-day chance of an M6+ earthquake",
        desc="Each shaded 2-degree cell is coloured by its chance of an M6+ earthquake in the next 30 days. "
             "Numbered pins mark the ten highest cells, listed below in the same order.",
        legend=maps.heat_legend(),
        caption="Unshaded cells have a chance below 0.1%. Numbered pins match the list below.")
    rows = "".join(_row(i, c) for i, c in enumerate(cells[:N_LISTED], 1))
    listing = (f'<ol class="rows">{rows}</ol>'
               f'<p class="section-foot">Every cell&rsquo;s value is in the forecast record: '
               f'<a href="/data/replay/{esc(replay.get("forecast_id"))}.json">{esc(replay.get("forecast_id"))}.json</a> '
               "(the <code>probability_grid</code> field, row by row from the south-west corner).</p>")
    body = (common.section("map", "Map", notices + themap)
            + common.section("cells", "Cells with the highest chance", listing,
                             intro=f"The {min(N_LISTED, len(cells))} cells with the highest 30-day chance. Open a row to "
                                   "see what drives its number.")
            + common.section("model", "About this forecast", _model_box(d),
                             intro="A chance is not an alarm: a 20% chance still means that, of five such 30-day "
                                   "periods, four would pass without an M6+ earthquake in that cell. Compare a cell "
                                   "with others, and with what drives its number.")
            + common.section("check", "Check this forecast", common.check_this(
                replay.get("forecast_id"),
                scored_note="How this model scored on past years, and how its live forecasts score as their 30-day "
                            "windows close.")))
    return f'<main id="main" class="page page-eq">{hero}{body}</main>'
