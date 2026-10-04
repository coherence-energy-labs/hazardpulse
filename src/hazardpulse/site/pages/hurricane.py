"""/live/hurricane/: the chance each active tropical cyclone intensifies rapidly in the next 24 hours."""
from __future__ import annotations

from hazardpulse.site import fmt, maps
from hazardpulse.site.data import SiteData, basin_of, official_center, ri_source_label, storm_name
from hazardpulse.site.hazards import HURRICANE
from hazardpulse.site.pages import common
from hazardpulse.site.places import coords, describe_far

esc = fmt.esc

USED = {"DTOP": "DTOPS, as issued", "RII": "SHIPS-RII (DTOPS was missing for this cycle)"}


def _ours(storm: dict) -> dict | None:
    v = storm.get("ri_v10_shadow") or {}
    if v.get("status") != "ok" or v.get("probability") is None:
        return None
    return v


def _basis(storm: dict) -> str:
    """Why this storm's published number comes from where it does."""
    src = str(storm.get("ri_source") or "")
    if src == "noaa_aid_stack":
        used = str(((storm.get("ri_inputs") or {}).get("used")) or "")
        url = (storm.get("ri_inputs") or {}).get("url")
        text = f"NOAA&rsquo;s rapid-intensification guidance: {USED.get(used, esc(used) or 'as issued')}"
        if url:
            text += f', read from <a href="{esc(url)}" rel="noopener">NHC&rsquo;s SHIPS text for this cycle</a>'
        return text
    status = str((((storm.get("ri_inputs") or {}).get("noaa_aid_stack")) or {}).get("status") or "")
    if status.startswith("jtwc_basin") or str(storm.get("basin") or "").upper() not in ("AL", "EP", "CP"):
        return "HazardPulse v8.2: NOAA publishes no rapid-intensification guidance for this basin"
    return "HazardPulse v8.2: NOAA&rsquo;s guidance was not usable for this cycle"


def _storm_card(s: dict) -> str:
    sid = esc(s.get("storm_id"))
    name = esc(storm_name(s))
    lat, lon = float(s["lat"]), float(s["lon"])
    center, center_url = official_center(s)
    ours = _ours(s)
    rows = [
        ("Strength", f"{esc(s.get('category') or '&mdash;')}: {fmt.num(s.get('vmax_kt'))} kt maximum sustained "
                     "winds" + (f", {fmt.num(s.get('mslp_hpa'))} hPa" if s.get("mslp_hpa") else "")),
        ("Position", f"{coords(lat, lon, 1)}, {esc(describe_far(lat, lon))}"),
        ("Forecast cycle", fmt.time_tag(s.get("issue_time"))),
        ("Where the number comes from", _basis(s)),
    ]
    if ours:
        fallback = ("" if ours.get("gate_ok", True) else
                    " (the early intensity guidance was missing for this cycle, so it equals NOAA&rsquo;s DTOPS)")
        rows.append(("Our experimental model (v10.1)",
                     f"{fmt.pct(ours['probability'])}{fallback} &mdash; shown for comparison while it is tested on "
                     'new forecasts; <a href="#ours">what this is</a>'))
    rows.append(("Official forecast", f'<a href="{center_url}" rel="noopener">{center}</a>'))
    return (f'<article class="card storm-card" id="storm-{sid}">'
            f'<div class="storm-card-head"><div><h3>{name}</h3><p class="muted">{esc(basin_of(s))} &middot; '
            f'{sid}</p></div><div class="storm-card-figure">{fmt.chance(s.get("ri_probability"), big=True)}'
            f'<span>chance of rapid intensification, next 24 hours &middot; source: {esc(ri_source_label(s))}</span>'
            f"</div></div>{common.facts(rows)}</article>")


def _sources(d: SiteData) -> str:
    hu = d.evidence.get("hurricane")
    if not hu:
        return ("<p>For storms the National Hurricane Center covers we publish NOAA&rsquo;s own guidance; elsewhere "
                "our v8.2 model. No final test is bound to the served choice in this build.</p>")
    t = hu["test"]
    beaten = [c["against"] for c in hu.get("claims", []) if c.get("better")]
    seasons = hu.get("chosen_on_seasons") or []
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "earlier seasons"
    ob = (hu.get("other_basins") or {}).get("test") or {}
    nhc = (f"<p><strong>Atlantic, East and Central Pacific.</strong> NOAA&rsquo;s own probability, read from the "
           "National Hurricane Center&rsquo;s SHIPS text for the same forecast cycle: DTOPS as issued, or SHIPS-RII "
           f"when DTOPS is missing. We chose it in a comparison written down in advance and run on {span}, then "
           f"scored it once on the {esc(t.get('when'))} season: it ranked a cycle that went on to intensify rapidly "
           f"above one that did not {fmt.pct(t.get('auc'), 1)} of the time over {t.get('n', 0):,} forecast cycles "
           f"and {t.get('storms', 0):,} storms"
           + (f", and beat {_join([esc(b) for b in beaten])} on the same cycles" if beaten else "") + ".</p>")
    other = ("<p><strong>West Pacific, Indian Ocean and Southern Hemisphere</strong>, where no public "
             "rapid-intensification guidance exists, and any NHC cycle without usable SHIPS text: our v8.2 model"
             + (f". Its test, on held-out {fmt.years(ob.get('when', ''))} Atlantic and East Pacific cases, ranked "
                f"them correctly {fmt.pct(ob.get('auc'), 1)} of the time; it has not been tested in the basins where it "
                "now publishes" if ob.get("auc") is not None else "") + ".</p>")
    return nhc + other + '<p><a href="/methods/#hurricane">Read the full test</a></p>'


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _ours_section(d: SiteData) -> str:
    hu = d.evidence.get("hurricane") or {}
    ours = hu.get("ours")
    if not ours:
        return ""
    dev = ours.get("dev") or {}
    pr = ours.get("prospective") or {}
    looks = pr.get("look_dates") or []
    body = (
        "<p>HazardPulse v10.1 is our own rapid-intensification model, built from NOAA&rsquo;s intensity guidance, "
        "its rapid-intensification probabilities and the official forecast. Over the 2022&ndash;2025 seasons, each "
        "forecast only from earlier seasons, it scored better than DTOPS: log loss "
        f"{fmt.num(dev.get('log_loss'), 3)} against {fmt.num(dev.get('dtops_log_loss'), 3)} (lower is better).</p>"
        "<p>That is not yet proof. It is shown beside NOAA&rsquo;s number, never instead of it, while a test written "
        "down in advance scores it on new forecasts"
        + (f"; the verdicts are due on {_join([fmt.date(x) for x in looks])}" if looks else "")
        + f". Forecast cycles scored so far: {int(pr.get('matured_and_scored', 0) or 0):,}.</p>"
        '<p><a href="/methods/#hurricane-ours">The full comparison, including the variants under test</a></p>')
    return common.section("ours", "Our experimental model", body)


def page(d: SiteData) -> str:
    hu = d.hurricanes
    head = d.headlines["hu"]
    storms = sorted(hu.get("storms") or [], key=lambda s: -float(s.get("ri_probability") or 0))
    srcs = sorted({ri_source_label(s) for s in storms})
    hero = common.hero(
        "Hurricane &middot; next 24 hours",
        "Rapid intensification: the chance in the next 24 hours",
        "Rapid intensification means a tropical cyclone&rsquo;s maximum sustained winds rise by 30 knots or more within "
        "24 hours &mdash; the kind of change official forecasts find hardest to anticipate. For each active storm we publish the best "
        "available probability: NOAA&rsquo;s own guidance where it exists, our model elsewhere. Each storm names its "
        "source.",
        meta=[("Updated", fmt.time_tag(hu.get("updated_at"))),
              ("Active storms", f"{len(storms):,}"),
              ("Sources in use", esc(", ".join(srcs)) if srcs else "&mdash;")])
    notices = (common.forecast_age(hu.get("updated_at"), 24 * 60, HURRICANE.schedule) + common.gate_notice(head)
               + common.official_notice(HURRICANE, "For tracks, warnings and evacuation decisions, follow"))
    if storms:
        proj = maps.WORLD
        marks = "".join(maps.storm(proj, s["lat"], s["lon"], storm_name(s), href=f"#storm-{esc(s.get('storm_id'))}",
                                   title=f"{storm_name(s)}: {fmt.pct_plain(s.get('ri_probability'))}",
                                   cls=fmt.level(s.get("ri_probability"))) for s in storms)
        themap = maps.figure(proj, marks, ident="hu-map", title="Map of active tropical cyclones",
                             desc="Each ringed marker is an active tropical cyclone, labelled with its name and "
                                  "coloured by its chance of rapid intensification. The cards below give each storm.",
                             legend=fmt.legend("Chance of rapid intensification, next 24 hours"))
        cards = f'<div class="storm-cards">{"".join(_storm_card(s) for s in storms)}</div>'
        body = (common.section("storms", "Active storms", notices + themap + cards))
    else:
        body = common.section("storms", "Active storms", notices + '<p class="empty">No tropical cyclones are '
                                                                  "active anywhere right now. This page updates "
                                                                  "after every forecast cycle.</p>")
    body += common.section("sources", "Where each number comes from", _sources(d))
    body += _ours_section(d)
    body += common.section("check", "Check this forecast", common.check_this(
        hu.get("forecast_id"), scored_note="How the published numbers scored on past seasons, and how live forecasts "
                                           "score against the National Hurricane Center&rsquo;s best track."))
    return f'<main id="main" class="page page-hu">{hero}{body}</main>'
