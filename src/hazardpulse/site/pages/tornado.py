"""/live/tornado/: for every thunderstorm tracked over the US, the chance of a tornado within 60 minutes."""
from __future__ import annotations

from hazardpulse.site import fmt, maps
from hazardpulse.site.data import SiteData
from hazardpulse.site.hazards import TORNADO
from hazardpulse.site.pages import common
from hazardpulse.site.places import coords, describe_us

esc = fmt.esc
N_DETAILED = 20

# ProbSevere attributes shown per storm: (input, label, unit, decimals, scale). Units as NOAA publishes them;
# NOAA's model probabilities arrive in percent (0-100). (The legacy "PS" attribute is absent from ProbSevere v3,
# so it reads 0 for every storm and is not shown.)
PERCENT_INPUTS = {"p_ps_tor", "p_ps_hail", "p_ps_wind", "p_ps_severe"}
ATTRIBUTES = (
    ("p_ps_tor", "NOAA ProbTor", "%", 0, 1.0),
    ("p_ps_severe", "NOAA ProbSevere (any severe weather)", "%", 0, 1.0),
    ("p_maxllaz", "Low-level rotation (max azimuthal shear)", "&times;10&#8315;&sup3; s&#8315;&sup1;", 1, 1000.0),
    ("p_mucape", "Most-unstable CAPE", "J/kg", 0, 1.0),
    ("p_ebshear", "Effective bulk shear", "kt", 0, 1.0),
    ("p_srh01", "0&ndash;1 km storm-relative helicity", "m&sup2;/s&sup2;", 0, 1.0),
    ("p_mesh", "Maximum expected hail size", "in", 2, 1.0),
    ("p_flash_rate", "Lightning flash rate", "per min", 0, 1.0),
    ("p_size", "Storm area", "km&sup2;", 0, 1.0),
)


def _sorted(d: SiteData) -> list[dict]:
    return sorted(d.tornadoes.get("storms") or [], key=lambda s: -float(s.get("tornado_probability") or 0))


def _warning(s: dict) -> str:
    w = ((s.get("v3") or {}).get("nws_warning")) or {}
    if (s.get("v3") or {}).get("nws_feed_error"):
        return "unknown (the NWS warnings feed did not answer this cycle)"
    if w.get("active"):
        mins = w.get("minutes_since_issue")
        return "<strong>in effect</strong>" + (f", issued {fmt.num(mins)} min before this forecast" if mins is not None else "")
    return "none in effect for this storm"


def _chances(s: dict) -> str:
    v3 = s.get("v3") or {}
    cells = []
    for key, label in (("probability_30min", "within 30 minutes"), ("probability_60min", "within 60 minutes"),
                       ("probability_90min", "within 90 minutes"), ("probability_ef2plus_60min",
                                                                     "a strong (EF2+) tornado within 60 minutes")):
        if v3.get(key) is not None:
            cells.append(f'<div class="mini-stat"><span class="mini-stat-figure">{fmt.pct(v3[key])}</span>'
                         f'<span class="mini-stat-label">{label}</span></div>')
    return f'<div class="mini-stats">{"".join(cells)}</div>' if cells else ""


def _drivers(s: dict) -> str:
    drivers = (s.get("v3") or {}).get("drivers") or []
    if not drivers:
        return "<p>No input moved this storm&rsquo;s chance away from the background rate.</p>"
    rows = []
    for dv in drivers:
        lo = float(dv.get("log_odds") or 0.0)
        cls, verb = ("up", "raises") if lo > 0 else ("down", "lowers")
        rows.append(f'<li class="driver {cls}"><span class="driver-name">{esc(dv.get("label") or dv.get("input"))}'
                    f'</span><span class="driver-value">{_driver_value(dv)}</span>'
                    f'<span class="driver-effect">{verb} the chance <span class="mono">({lo:+.2f})</span>'
                    "</span></li>")
    return ('<ul class="drivers">' + "".join(rows) + "</ul>"
            '<p class="small muted">Effects are the model&rsquo;s own attribution, additive on the log-odds scale: '
            "+0.69 roughly doubles the odds, &minus;0.69 halves them.</p>")


def _driver_value(dv: dict) -> str:
    val = dv.get("value")
    if val is None:
        return "&mdash;"
    v = float(val)
    if dv.get("input") in PERCENT_INPUTS:
        return f"{fmt.num(v, 0)}%"
    return fmt.num(v, 3 if 0 < abs(v) < 0.1 else (0 if abs(v) >= 100 else 1))


def _track_record(d: SiteData, s: dict) -> str:
    from hazardpulse.verification import served_evidence
    v3 = s.get("v3") or {}
    table = served_evidence.reliability_for(d.evidence.get("tornado"), str(v3.get("model", "")))
    if table is None:
        return "<p>No 2025 test table is bound to the model that scored this storm.</p>"
    p = float(v3.get("probability_60min", s.get("tornado_probability") or 0.0))
    b = served_evidence.reliability_bin(table, p)
    if b is None:
        return "<p>No storm in the 2025 test was scored in this range, so no observed rate is shown.</p>"
    ci = b.get("observed_ci")
    ci_txt = f" (95% interval {fmt.pct(ci[0], 2)}&ndash;{fmt.pct(ci[1], 2)})" if ci else ""
    return (f"<p>Of the {int(b['n']):,} storm observations in 2025 that this model gave between {fmt.pct(b['lo'], 1)} "
            f"and {fmt.pct(b['hi'], 1)}, {fmt.pct(b['observed'], 2)}{ci_txt} were followed by a tornado from that "
            "storm within 60 minutes.</p>")


def _attributes(s: dict) -> str:
    inputs = ((s.get("v3") or {}).get("inputs")) or {}
    rows = []
    for key, label, unit, dec, scale in ATTRIBUTES:
        v = inputs.get(key)
        if v is None:
            val = "not reported"
        elif unit == "%":
            val = f"{fmt.num(v, 0)}%"
        else:
            val = f"{fmt.num(float(v) * scale, dec)} {unit}"
        rows.append((label, val))
    return common.facts(rows, "facts-compact")


def _row(i: int, s: dict, d: SiteData) -> str:
    lat, lon = float(s["lat"]), float(s["lon"])
    where = describe_us(lat, lon)
    warn = ((s.get("v3") or {}).get("nws_warning") or {}).get("active")
    badge = '<span class="chip bad">NWS tornado warning</span>' if warn else ""
    valid = fmt.utc(s.get("valid_time"), with_date=False)
    v3 = s.get("v3") or {}
    body = (
        _chances(s)
        + '<div class="row-cols"><div><h4>What moved this number</h4>' + _drivers(s) + "</div>"
        + '<div><h4>How storms scored like this one turned out</h4>' + _track_record(d, s)
        + f"<h4>NWS tornado warning</h4><p>{_warning(s)}</p></div></div>"
        + common.disclosure("Storm attributes from NOAA ProbSevere", _attributes(s) + common.facts([
            ("ProbSevere storm ID", f"<code>{esc(s.get('storm_id'))}</code>"),
            ("Position", coords(lat, lon, 2)),
            ("Observed at", fmt.time_tag(s.get("valid_time"))),
            ("Model", f"<code>{esc(v3.get('model_version') or s.get('model_version'))}</code>"),
        ], "facts-compact")))
    return (f'<li id="storm-{i}"><details class="row"><summary><span class="row-rank">{i}</span>'
            f'<span class="row-main"><span class="row-title">{esc(where[:1].upper() + where[1:])} {badge}</span>'
            f'<span class="row-sub">Observed {valid} &middot; storm {esc(s.get("storm_id"))}'
            f'{(" &middot; " + fmt.interval(s)) if fmt.interval(s) else ""}</span></span>'
            f'<span class="row-figure">{fmt.chance(s.get("tornado_probability"))}</span></summary>'
            f'<div class="row-body">{body}</div></details></li>')


def _all_storms(storms: list[dict]) -> str:
    rows = [[str(i), esc(describe_us(s["lat"], s["lon"])),
             "Yes" if ((s.get("v3") or {}).get("nws_warning") or {}).get("active") else "No",
             fmt.chance(s.get("tornado_probability"))] for i, s in enumerate(storms, 1)]
    return common.disclosure(
        f"All {len(storms):,} tracked storms",
        common.table(["#", "Location", "NWS tornado warning", "Chance, 60 min"], rows, num_cols=(0, 3),
                     caption="Every tracked storm, highest chance first"),
        cls="disclosure-table")


def page(d: SiteData) -> str:
    to = d.tornadoes
    head = d.headlines["to"]
    storms = _sorted(d)
    n_warned = sum(1 for s in storms if ((s.get("v3") or {}).get("nws_warning") or {}).get("active"))
    hero = common.hero(
        "Tornado &middot; contiguous US &middot; next 60 minutes",
        "The tornado chance of every tracked storm",
        "For every thunderstorm NOAA&rsquo;s ProbSevere system is tracking over the contiguous United States, the chance "
        "that it produces a tornado within the next 60 minutes. This is not a warning: if the National Weather Service "
        "issues a tornado warning for your area, take shelter.",
        meta=[("Updated", fmt.time_tag(to.get("updated_at"))),
              ("Storms tracked", f"{len(storms):,}"),
              ("Under an NWS tornado warning", f"{n_warned:,}")])
    notices = common.gate_notice(head) + common.official_notice(TORNADO, "For tornado warnings, follow")
    proj = maps.CONUS
    marks = [maps.dot(proj, s["lat"], s["lon"], cls=f"dot-to {fmt.level(s.get('tornado_probability'))}",
                      title=f"{describe_us(s['lat'], s['lon'])}: {fmt.pct_plain(s.get('tornado_probability'))}", r=4.5)
             for s in storms[N_DETAILED:]]
    marks += [maps.pin(proj, s["lat"], s["lon"], str(i), href=f"#storm-{i}",
                       cls=f"pin-to {fmt.level(s.get('tornado_probability'))}",
                       title=f"{i}. {describe_us(s['lat'], s['lon'])}: {fmt.pct_plain(s.get('tornado_probability'))}",
                       r=9) for i, s in reversed(list(enumerate(storms[:N_DETAILED], 1)))]
    themap = maps.figure(proj, "".join(marks), ident="to-map", title="Map of tracked thunderstorms",
                         desc="Each marker is a thunderstorm tracked by NOAA ProbSevere over the contiguous US, "
                              "coloured by its chance of producing a tornado within 60 minutes. Numbered markers "
                              "are the storms listed below.",
                         legend=fmt.legend("Chance of a tornado within 60 minutes"),
                         caption="Numbered markers match the list below. Storms over Mexico and the Gulf are "
                                 "tracked too when NOAA&rsquo;s radar sees them.")
    rows = "".join(_row(i, s, d) for i, s in enumerate(storms[:N_DETAILED], 1))
    listing = (f'<ol class="rows">{rows}</ol>' + _all_storms(storms)) if storms else (
        '<p class="empty">No thunderstorms are being tracked over the US right now.</p>')
    body = (common.section("map", "Map", notices + themap)
            + common.section("storms", "Storms with the highest chance", listing,
                             intro=f"The {min(N_DETAILED, len(storms))} storms with the highest chance, then every "
                                   "tracked storm. Open a row for what drives its number and how storms like it "
                                   "turned out.")
            + common.section("model", "About this forecast", _about(d))
            + common.section("check", "Check this forecast", common.check_this(
                to.get("forecast_id"), scored_note="How the model scored on every storm of 2025 against NOAA&rsquo;s "
                                                   "ProbTor and the Weather Service&rsquo;s warnings.")))
    return f'<main id="main" class="page page-to">{hero}{body}</main>'


def _about(d: SiteData) -> str:
    to = d.evidence.get("tornado")
    version = esc(d.tornadoes.get("model_version") or "")
    if not to:
        return common.facts([("Model version", f"<code>{version}</code>"),
                             ("Test", "No final test is bound to the served model in this build.")])
    t = to["test"]
    pt = ((to.get("probtor_final") or {}).get("probtor_raw") or {}).get("auc")
    nws = to.get("vs_nws_warnings") or {}
    rows = [
        ("What the model is", "Gradient-boosted trees on NOAA ProbSevere&rsquo;s storm attributes &mdash; radar, "
                              "lightning, satellite and the near-storm environment &mdash;"
                              + (" and the live National Weather Service tornado-warning state" if to.get("inputs_use_nws") else "")),
        ("How it tested", f"On every storm observation of 2025 ({int(t.get('n') or 0):,}, of which "
                          f"{int(t.get('pos') or 0):,} were followed by a tornado), scored once after every choice was "
                          f"fixed: a storm that went on to produce a tornado was ranked above one that did not "
                          f"{fmt.pct(t.get('auc'), 1)} of the time"
                          + (f"; NOAA&rsquo;s ProbTor, on the same storms: {fmt.pct(pt, 1)}" if pt is not None else "")),
        ("What counts as a tornado", esc(str(to.get("event") or "a tornado report near the storm within 60 minutes").replace("THIS", "this"))),
        ("Model version", f"<code>{version}</code>"),
        ("Full results", '<a href="/verification/tornado/">Tornado model test results</a>'),
    ]
    if nws.get("note"):
        rows.insert(3, ("Against NWS warnings", esc(nws["note"])))
    return common.facts(rows)
