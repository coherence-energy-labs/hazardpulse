"""/live/hurricane/: the chance each active tropical cyclone intensifies rapidly in the next 24 hours."""
from __future__ import annotations

import re

from hazardpulse.site import fmt, maps
from hazardpulse.site.data import SiteData, basin_of, official_center, ri_source_label, storm_name
from hazardpulse.site.hazards import HURRICANE
from hazardpulse.site.pages import common
from hazardpulse.site.places import coords, describe_far, haversine_km
from hazardpulse.verification.served_evidence import j1_shown_shadow

esc = fmt.esc
TC1_LEADS = ("12", "24", "36", "48", "72", "96", "120")
KM_PER_NM = 1.852

USED = {"DTOP": "DTOPS, as issued", "RII": "SHIPS-RII (DTOPS was missing for this cycle)"}


def _ours(storm: dict, shown: dict | None) -> dict | None:
    """The shown model's shadow forecast in this storm's record: its key and label come from the one pointer
    (``served_evidence.shown_model``), never from this page."""
    if not shown or not shown.get("shadow_key"):
        return None
    v = storm.get(shown["shadow_key"]) or {}
    if v.get("status") != "ok" or v.get("probability") is None:
        return None
    return v


def _span(ob: dict | None) -> str:
    """The years of the other-basins model's test, read from its evaluation."""
    when = ((ob or {}).get("test") or {}).get("when")
    return fmt.years(when) if when else "test"


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
    who = esc(ri_source_label(storm))            # the record's own label for the model that made the number
    if status.startswith("jtwc_basin") or str(storm.get("basin") or "").upper() not in ("AL", "EP", "CP"):
        return f"{who}: NOAA publishes no rapid-intensification guidance for this basin"
    return f"{who}: NOAA&rsquo;s guidance was not usable for this cycle"


def _single_warning(s: dict) -> bool:
    """The published v8.2 number was scored without inputs the model needs. Since 2026-10-05 a record says
    which (``ri_inputs.inputs_missing``): a West Pacific storm now carries its best-track history from RAL's
    b-deck, so the caution shows only when that history was not there. Older records: a JTWC analysis."""
    if str(s.get("ri_source") or "") == "noaa_aid_stack":
        return False
    inputs = s.get("ri_inputs") or {}
    if "inputs_missing" in inputs:
        return bool(inputs["inputs_missing"])
    return str(inputs.get("analysis_model") or "") == "JTWC"


def _single_warning_caveat(ob: dict | None, s: dict | None = None) -> str:
    """A plain warning for a number scored without some of its inputs, with the measured cost when the record's
    missing inputs are the pattern that was measured (served_evidence: the served model on its test years' West
    Pacific cycles scored from a single warning)."""
    composition = (ob or {}).get("composition")
    jt = (composition or {}).get("single_jtwc_warning") or {}
    inputs = (s or {}).get("ri_inputs") or {}
    missing = inputs.get("inputs_missing")
    measured_pattern = missing is None or (inputs.get("track_source") == "jtwc_warning"
                                           and len(missing) == jt.get("n_missing"))
    if missing is not None and not measured_pattern:
        return (f"{len(missing)} of the model&rsquo;s {inputs.get('n_inputs') or len(missing)} inputs were not "
                "available for this storm and are filled with typical values. Treat this number as rough.")
    if jt.get("n_missing") and jt.get("log_loss") is not None and jt.get("log_loss_climatology") is not None:
        return (f"Scored from a single JTWC warning, which does not carry the storm&rsquo;s recent track: "
                f"{jt['n_missing']} of the model&rsquo;s {jt['n_inputs']} inputs are unknown and filled with "
                f"typical values. Scored the same way, the {_span(ob)} West Pacific cycles were barely better "
                "than always forecasting the average (log loss "
                f"{fmt.num(jt['log_loss'], 3)} against {fmt.num(jt['log_loss_climatology'], 3)}; "
                f"{fmt.num(jt.get('log_loss_full_inputs'), 3)} with every input). Treat this number as rough.")
    return ("Scored from a single JTWC warning, which does not carry the storm&rsquo;s recent track, so several "
            "of the model&rsquo;s inputs are filled with typical values. Treat this number as rough.")


def _which_inputs(names: list[str]) -> str:
    if names and all(n == "analysis_mslp_hpa" or n.startswith("analysis_dp_") for n in names):
        return "the storm&rsquo;s central pressure and its changes"
    return f"{len(names)} of the model&rsquo;s inputs"


def _late_fix_note(ob: dict | None, s: dict) -> str | None:
    """The measured note for a record scored with RAL's history but the warning as its analysis (this cycle's
    fix not yet published) -- only when its missing inputs are exactly the measured pattern."""
    composition = (ob or {}).get("composition")
    lf = (composition or {}).get("late_best_track_fix") or {}
    jt = (composition or {}).get("single_jtwc_warning") or {}
    inputs = s.get("ri_inputs") or {}
    missing = sorted(inputs.get("inputs_missing") or [])
    if (inputs.get("track_source") != "ral_bdeck" or not missing or lf.get("log_loss") is None
            or missing != list(lf.get("inputs") or [])):
        return None
    return (f"This cycle&rsquo;s best-track fix was not yet published, so {_which_inputs(missing)} are filled with "
            f"typical values; the storm&rsquo;s track and winds are known. Scored the same way, the {_span(ob)} "
            f"West Pacific cycles had a log loss of {fmt.num(lf['log_loss'], 3)} "
            f"({fmt.num(jt.get('log_loss_full_inputs'), 3)} with every input, "
            f"{fmt.num(jt.get('log_loss_climatology'), 3)} for always forecasting the average).")


def _tc1_version(s: dict, tc1_ev: dict | None) -> str | None:
    """The TC1 version that made this storm's forecast: the saved models' version, named only when the record says
    it was made from them (the same selection, replayed from the same season's end)."""
    tc = s.get("tc1") or {}
    if (tc1_ev and tc1_ev.get("model_version") and tc.get("selection_sha256") == tc1_ev.get("selection_sha256")
            and str(tc.get("season_state_end")) == str(tc1_ev.get("season_end"))):
        return tc1_ev["model_version"]
    return None


def _tc1_table(s: dict, tc1_ev: dict | None = None) -> str:
    """Our track and intensity forecast (TC1) beside NHC's official one, lead by lead, for one storm."""
    tc = s.get("tc1") or {}
    ours, ofcl = tc.get("TC1") or {}, tc.get("OFCL") or {}
    if not ours:
        return ""

    def pos(p):
        return coords(p["lat"], p["lon"], 1) if p and p.get("lat") is not None and p.get("lon") is not None else "&mdash;"

    def kt(p):
        return f"{fmt.num(p['vmax_kt'])} kt" if p and p.get("vmax_kt") is not None else "&mdash;"

    def apart(a, b):
        if not (a and b) or None in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")):
            return "&mdash;"
        return f"{fmt.num(haversine_km(a['lat'], a['lon'], b['lat'], b['lon']) / KM_PER_NM)} n mi"

    rows = [[f"{lead} h", pos(ours.get(lead)), kt(ours.get(lead)), pos(ofcl.get(lead)), kt(ofcl.get(lead)),
             apart(ours.get(lead), ofcl.get(lead))]
            for lead in TC1_LEADS if ours.get(lead) or ofcl.get(lead)]
    caption = (f"Forecast from the {fmt.utc(tc.get('cycle'))} cycle: HazardPulse TC1 beside the National Hurricane "
               "Center&rsquo;s official forecast. Positions are the storm&rsquo;s centre; winds are maximum sustained "
               "(1-minute) winds.")
    body = common.table(["Hours ahead", "HazardPulse position", "HazardPulse winds", "NHC position", "NHC winds",
                         "Apart"], rows, cls="tc1-table", caption=caption, num_cols=(2, 4, 5))
    version = _tc1_version(s, tc1_ev)
    body += ('<p class="muted">' + (f"Model <code>{esc(version)}</code> &middot; " if version else "")
             + '<a href="#track">How this forecast is made and how it has scored</a></p>')
    return common.disclosure("Where it is going: our track and intensity forecast", body, cls="tc1")


def _level(x) -> str:
    """A confidence level read from a results file: 0.9875 -> 98.75%."""
    return "&mdash;" if x is None else f"{100 * float(x):.4f}".rstrip("0").rstrip(".") + "%"


def _tc1_section(d: SiteData) -> str:
    """How TC1 is made and how it scored -- every number, and every confidence level, from the results bound to the
    served models. The claim against NHC's official forecast is the test's; the comparison with HCCA is descriptive
    and says so (until 2026-10-09 it read "a gain beyond chance" beside the claim's tie)."""
    tc = (d.evidence.get("hurricane") or {}).get("tc1")
    if not tc:
        return ""
    tr, iv = tc["track"], tc["intensity"]
    seasons = tc.get("seasons") or []
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "the test seasons"

    def ci(x):
        return f"[{fmt.num(x['ci'][0], 1)}, {fmt.num(x['ci'][1], 1)}]" if x and x.get("ci") else ""

    track_claim = tr["vs_ofcl"]["claim"]
    hcca = tr.get("vs_hcca") or {}
    per = hcca.get("per_lead") or {}
    leads = sorted(per, key=int)
    ahead_from = next((lead for i, lead in enumerate(leads)
                       if all((per[x].get("ci") or [0, 0])[1] < 0 for x in leads[i:])), None)
    hcca_ahead = bool(hcca.get("ci") and hcca["ci"][1] < 0)
    s26, i26 = tr.get("season_2026_vs_ofcl"), iv.get("season_2026_vs_ofcl")
    o = tr.get("tc1o_vs_ofcl")
    body = (
        "<p>For every storm the National Hurricane Center follows, HazardPulse issues its own track and intensity "
        "forecast, TC1, out to five days. It combines the computer models NHC publishes in real time. Each model is "
        "weighted by how it has actually performed: its errors are re-measured every six hours against where the "
        "storm turned out to be, so a model that is doing well this season counts for more, and a new one earns its "
        "weight as it proves itself."
        + (f" Model <code>{esc(tc['model_version'])}</code>." if tc.get("model_version") else "") + "</p>"
        f"<p><strong>Track.</strong> Tested in advance on the {span} seasons, our track forecasts were off by "
        f"{fmt.num(tr['errors']['TC1'], 1)} nautical miles on average over one to five days. NHC&rsquo;s official "
        f"forecast was off by {fmt.num(tr['errors']['OFCL'], 1)}, and its best automatic guidance (HCCA) by "
        f"{fmt.num(tr['errors']['HCCA'], 1)}. "
        + ("That beats the official forecast at the confidence the test required "
           f"({_level(tr['vs_ofcl'].get('level'))}). "
           if track_claim else
           "Against the official forecast that is a tie at the confidence the test required "
           f"({_level(tr['vs_ofcl'].get('level'))}): the difference, {fmt.num(tr['vs_ofcl']['d'], 1)} n mi "
           f"{ci(tr['vs_ofcl'])}, could be chance. ")
        + (f"Against HCCA, in a comparison the test reported but did not make a claim of ({_level(hcca.get('level'))} "
           "interval), it is ahead" + (f" at every lead from {ahead_from} hours" if ahead_from else "")
           if hcca_ahead else
           f"Against HCCA, in a comparison the test reported but did not make a claim of ({_level(hcca.get('level'))} "
           "interval), it is a tie")
        + f": {fmt.num(hcca.get('d'), 1)} n mi {ci(hcca)}.</p>"
        f"<p><strong>Intensity.</strong> Our intensity forecasts were off by {fmt.num(iv['errors']['TC1'], 1)} knots "
        f"on average, NHC&rsquo;s official ones by {fmt.num(iv['errors']['OFCL'], 1)}: "
        + ("better than the official forecast at the confidence the test required."
           if iv["vs_ofcl"]["claim"] else f"a tie ({fmt.num(iv['vs_ofcl']['d'], 1)} kt {ci(iv['vs_ofcl'])}).")
        + " Intensity is where every forecast struggles most.</p>"
        + (f"<p><strong>2026 so far.</strong> Against NHC&rsquo;s official forecast on this season&rsquo;s storms, "
           f"scored against NHC&rsquo;s operational best tracks: track {fmt.num(s26['d'], 1)} n mi {ci(s26)}"
           + (f", intensity {fmt.num(i26['d'], 1)} kt {ci(i26)}" if i26 else "")
           + f" ({_level(s26.get('level'))} intervals). A season still in progress, so no claim is made from it.</p>"
           if s26 else "")
        + ("<p><strong>TC1+O</strong>, the same weighting with NHC&rsquo;s official forecast as one more member, "
           f"is recorded too: its track was off by {fmt.num(tr['errors']['TC1+O'], 1)} n mi on {span}, "
           f"{fmt.num(o['d'], 1)} n mi {ci(o)} against the official forecast, "
           + ("better at the confidence the test required" if o.get("claim") else
              f"a tie at the confidence the test required ({_level(o.get('level'))})")
           + ". It is not shown on the storm cards.</p>" if o else "")
        + "<p>Follow the National Hurricane Center for decisions. Ours is shown beside theirs so the two can be "
        'compared as the season goes on. <a href="/methods/#hurricane-track">The full test</a></p>')
    return common.section("track", "Our track and intensity forecast", body)


def _j1_row(s: dict, j1: dict | None) -> tuple[str, str] | None:
    """Our satellite model's forecast beside the published v8.2, for a storm in its scope; its label, the key of
    its live record and its scope come from its artifact and library (``served_evidence.j1_shown_shadow``), never
    from this page. The published number above is untouched."""
    v = j1_shown_shadow(s, j1)
    if v is None:
        return None
    m = re.fullmatch(r"\s*(\d+) of (\d+)\s*", str(v.get("ir_features_read") or ""))
    partial = (f" (only {m.group(1)} of its {m.group(2)} satellite inputs were available for this cycle)"
               if m and int(m.group(1)) < int(m.group(2)) else "")
    return (f"Our satellite model ({esc(j1['label'])})",
            f"{fmt.pct(v['probability'])}{partial} &mdash; shown for comparison while it is tested on new forecasts; "
            '<a href="/methods/#hurricane-j1">what this is</a>')


def _storm_card(s: dict, ob: dict | None = None, shown: dict | None = None, tc1_ev: dict | None = None,
                j1: dict | None = None) -> str:
    """One storm. ``ob`` is the other-basins model's evidence (its measured caveats), ``shown`` our shown RI
    model's (its label and the key of its shadow forecast), ``tc1_ev`` TC1's (its version), ``j1`` our satellite
    model's for the JTWC basins (its label, the key of its live record and its scope)."""
    sid = esc(s.get("storm_id"))
    name = esc(storm_name(s))
    lat, lon = float(s["lat"]), float(s["lon"])
    center, center_url = official_center(s)
    ours = _ours(s, shown)
    rows = [
        ("Strength", f"{esc(s.get('category') or '&mdash;')}: {fmt.num(s.get('vmax_kt'))} kt maximum sustained "
                     "winds" + (f", {fmt.num(s.get('mslp_hpa'))} hPa" if s.get("mslp_hpa") else "")),
        ("Position", f"{coords(lat, lon, 1)}, {esc(describe_far(lat, lon))}"),
        ("Forecast cycle", fmt.time_tag(s.get("issue_time"))),
        ("Where the number comes from", _basis(s)),
    ]
    if _single_warning(s):
        late = _late_fix_note(ob, s)
        rows.append(("Inputs", late) if late else ("Caution", _single_warning_caveat(ob, s)))
    if ours:
        fallback = ("" if ours.get("gate_ok", True) else
                    " (the early intensity guidance was missing for this cycle, so it equals NOAA&rsquo;s DTOPS)")
        rows.append((f"Our experimental model ({esc(shown['label'])})",
                     f"{fmt.pct(ours['probability'])}{fallback} &mdash; shown for comparison while it is tested on "
                     'new forecasts; <a href="#ours">what this is</a>'))
    j1_row = _j1_row(s, j1)
    if j1_row:
        rows.append(j1_row)
    rows.append(("Official forecast", f'<a href="{center_url}" rel="noopener">{center}</a>'))
    return (f'<article class="card storm-card" id="storm-{sid}">'
            f'<div class="storm-card-head"><div><h3>{name}</h3><p class="muted">{esc(basin_of(s))} &middot; '
            f'{sid}</p></div><div class="storm-card-figure">{fmt.chance(s.get("ri_probability"), big=True)}'
            f'<span>chance of rapid intensification, next 24 hours &middot; source: {esc(ri_source_label(s))}</span>'
            f"</div></div>{common.facts(rows)}{_tc1_table(s, tc1_ev)}</article>")


def _sources(d: SiteData) -> str:
    hu = d.evidence.get("hurricane")
    if not hu:
        return ("<p>For storms the National Hurricane Center covers we publish NOAA&rsquo;s own guidance; elsewhere "
                "our own model. No final test is bound to the served choice in this build.</p>")
    t = hu["test"]
    beaten = [c["against"] for c in hu.get("claims", []) if c.get("better")]
    seasons = hu.get("chosen_on_seasons") or []
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "earlier seasons"
    nhc = (f"<p><strong>Atlantic, East and Central Pacific.</strong> NOAA&rsquo;s own probability, read from the "
           "National Hurricane Center&rsquo;s SHIPS text for the same forecast cycle: DTOPS as issued, or SHIPS-RII "
           f"when DTOPS is missing. We chose it in a comparison written down in advance and run on {span}, then "
           f"scored it once on the {esc(t.get('when'))} season: it ranked a cycle that went on to intensify rapidly "
           f"above one that did not {fmt.pct(t.get('auc'), 1)} of the time over {t.get('n', 0):,} forecast cycles "
           f"and {t.get('storms', 0):,} storms"
           + (f", and beat {_join([esc(b) for b in beaten])} on the same cycles" if beaten else "") + ".</p>")
    label = esc((hu.get("other_basins") or {}).get("label") or "")
    other = ("<p><strong>West Pacific, Indian Ocean and Southern Hemisphere</strong>, where no public "
             "rapid-intensification guidance exists, and any NHC cycle without usable SHIPS text: "
             + (f"our {label} model." if label else "our model.")
             + v82_test_text(hu.get("other_basins")) + j1_sources_text(hu.get("j1"), label) + "</p>")
    return nhc + other + '<p><a href="/methods/#hurricane">Read the full test</a></p>'


def j1_sources_text(j1: dict | None, published: str) -> str:
    """One sentence on our satellite model beside the published number: where it is shown (its scope, from its
    library) and that it is a model in test, not the published number."""
    regions = [esc(r["name"]) for r in (j1 or {}).get("regions") or [] if r.get("in_scope")]
    if not j1 or not regions or not published:
        return ""
    return (f" On {_join(regions)} storms, our satellite model {esc(j1['label'])} is shown beside it for comparison "
            "while it is tested on new forecasts; it would replace "
            f"{published} only by meeting that test&rsquo;s rule, written down in advance "
            '(<a href="/methods/#hurricane-j1">its results and rule</a>).')


def v82_short(other_basins: dict | None) -> str:
    """One clause on what the other-basins model's test is, for the home page's proof strip: its held-out cycles
    from every basin (it said by hand, until 2026-10-09, that the model had no test in those basins)."""
    ob = (other_basins or {}).get("test") or {}
    label = esc((other_basins or {}).get("label") or "")
    if not label:
        return ""
    comp = (other_basins or {}).get("composition") or {}
    wp = next((b for b in comp.get("by_basin") or [] if b.get("basin") == "WP"), None)
    if ob.get("auc") is None or not comp.get("by_basin"):
        return f"Elsewhere we publish our {label} model"
    return (f"Elsewhere we publish our {label} model; its method was tested on {int(ob.get('n') or 0):,} held-out "
            f"{fmt.years(ob.get('when', ''))} cycles from every basin"
            + (f", {int(wp['n']):,} of them West Pacific" if wp else "")
            + (", with the published calibration fitted on those same cycles"
               if comp.get("served_calibration_fitted_on_test_cases") else ""))


def v82_test_text(other_basins: dict | None) -> str:
    """What v8.2's test figure is, stated as it was measured (site audit 2026-10-05): a test of its METHOD on
    held-out cycles from every basin with best-track inputs -- not of the published model, whose calibration
    was fitted on those same cycles, and not with the inputs a live West Pacific forecast has."""
    ob = (other_basins or {}).get("test") or {}
    if ob.get("auc") is None:
        return ""
    comp = (other_basins or {}).get("composition") or {}
    when = fmt.years(ob.get("when", ""))
    basins = ", ".join(f"{esc(b['name'])} {int(b['n']):,}" for b in comp.get("by_basin") or [])
    where = f" in every basin ({basins})" if basins else ""
    text = (f" Its method was tested on {int(ob.get('n') or 0):,} forecast cycles from {when}{where}, none of them "
            "used to fit it: it ranked a cycle that went on to intensify rapidly above one that did not "
            f"{fmt.pct(ob.get('auc'), 1)} of the time.")
    if comp.get("inputs") == "best track":
        text += " That test read each storm&rsquo;s inputs from the best track"
        text += (", and the model now published is the same method refitted with its calibration fitted on those "
                 "same cycles, so the figure is not a test of the published model on unseen data."
                 if comp.get("served_calibration_fitted_on_test_cases") else ".")
    jt = comp.get("single_jtwc_warning") or {}
    if jt.get("log_loss") is not None and jt.get("log_loss_climatology") is not None:
        text += (" In the West Pacific, Indian Ocean and southern hemisphere a storm&rsquo;s recent track comes from "
                 "the working best track (UCAR RAL&rsquo;s real-time copy), the kind of track the model was trained "
                 "on before its post-season revision. Scored from a single JTWC warning instead &mdash; which "
                 f"leaves {jt['n_missing']} of its {jt['n_inputs']} inputs unknown, as it did until 2026-10-05 and "
                 f"still does if that file cannot be read &mdash; the {when} West Pacific cycles had a log loss of "
                 f"{fmt.num(jt['log_loss'], 3)}, against {fmt.num(jt['log_loss_climatology'], 3)} for always "
                 f"forecasting the average ({fmt.num(jt.get('log_loss_full_inputs'), 3)} with every input).")
        lf = comp.get("late_best_track_fix") or {}
        if lf.get("log_loss") is not None:
            text += (" When a cycle&rsquo;s best-track fix is not yet published, the warning is the analysis and "
                     f"{_which_inputs(lf.get('inputs') or [])} are filled with typical values: scored that way, "
                     f"{fmt.num(lf['log_loss'], 3)}.")
    return text


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
    span = fmt.years(ours["dev_period"]) if ours.get("dev_period") else None
    beat = (dev.get("log_loss") is not None and dev.get("dtops_log_loss") is not None
            and dev["log_loss"] < dev["dtops_log_loss"])
    body = (
        f"<p>HazardPulse {esc(ours['label'])} is our own rapid-intensification model, built from NOAA&rsquo;s "
        "intensity guidance, its rapid-intensification probabilities and the official forecast"
        + (f", {ours['adds']}" if ours.get("adds") else "") + "."
        + (f" Over the {span} seasons, each forecast only from earlier seasons, it scored "
           + ("better than" if beat else "against") + " DTOPS: log loss "
           f"{fmt.num(dev.get('log_loss'), 3)} against {fmt.num(dev.get('dtops_log_loss'), 3)} (lower is better)."
           if span and dev.get("dtops_log_loss") is not None else "")
        + "</p>"
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
        ev = d.evidence.get("hurricane") or {}
        cards = ('<div class="storm-cards">'
                 + "".join(_storm_card(s, ev.get("other_basins"), ev.get("ours"), ev.get("tc1"), ev.get("j1"))
                           for s in storms)
                 + "</div>")
        body = (common.section("storms", "Active storms", notices + themap + cards))
    else:
        body = common.section("storms", "Active storms", notices + '<p class="empty">No tropical cyclones are '
                                                                  "active anywhere right now. This page updates "
                                                                  "after every forecast cycle.</p>")
    body += common.section("sources", "Where each number comes from", _sources(d))
    body += _tc1_section(d)
    body += _ours_section(d)
    body += common.section("check", "Check this forecast", common.check_this(
        hu.get("forecast_id"), scored_note="How the published numbers scored on past seasons, and how live forecasts "
                                           "score against the National Hurricane Center&rsquo;s best track."))
    return f'<main id="main" class="page page-hu">{hero}{body}</main>'
