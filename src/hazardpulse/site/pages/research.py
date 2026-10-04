"""/verification/cross-modality/: population tests of proposed precursor signals, published whatever they find."""
from __future__ import annotations

import math

from hazardpulse.site import fmt
from hazardpulse.site.data import SiteData
from hazardpulse.site.pages import common

esc = fmt.esc

# The hypotheses the suite tests, in words (analysis_id -> (hazard key, question, source)).
HYPOTHESES = {
    "geomagnetic_precursor_kp": ("eq", "Is geomagnetic activity (the Kp index, highest in the 72 hours before) higher "
                                       "before M6+ earthquakes?", "Sobolev; Hayakawa"),
    "solar_flare_precursor_xray": ("eq", "Are there more M- and X-class solar flares in the 72 hours before M6+ "
                                         "earthquakes?", "Freund; Pulinets"),
    "imf_bz_precursor": ("eq", "Is the interplanetary magnetic field more strongly southward (most negative Bz, "
                               "72 hours) before M6+ earthquakes?", ""),
    "hurricane_lightning_correlation": ("hu", "Is there more lightning in a hurricane&rsquo;s core before it "
                                              "intensifies rapidly?", "Fierro et al. 2014"),
    "tornado_lightning_leadup": ("to", "Does a storm&rsquo;s lightning rate jump in the 15 minutes before a "
                                       "tornado?", "Steiger et al. 2007; Schultz et al. 2011"),
    "cme_hurricane_intensification": ("hu", "Do solar storms (coronal mass ejections) arrive more often in the 72 "
                                            "hours before rapid intensification?", "Thakur et al."),
}


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _verdict(a: dict) -> tuple[str, str]:
    """(chip class, words) for one analysis, read from its interval -- never from its p-value alone."""
    if a.get("status") == "needs_curated_input":
        return "neutral", "Not run yet: needs a curated list of events"
    n = int(a.get("n_targets") or a.get("n_ri_events") or a.get("n_tornado_events") or 0)
    lo = _finite(a.get("bootstrap_ci_delta_lo", a.get("bootstrap_ci_lo")))
    hi = _finite(a.get("bootstrap_ci_delta_hi", a.get("bootstrap_ci_hi")))
    if n == 0 or lo is None or hi is None:
        return "neutral", "Not run: no data for these events in this window"
    if lo > 0:
        return "warn", "Higher before events in this sample"
    if hi < 0:
        return "warn", "Lower before events in this sample"
    return "good", "No difference found"


def _row(a: dict) -> list[str]:
    aid = str(a.get("analysis_id") or a.get("name") or "")
    key, question, source = HYPOTHESES.get(aid, (a.get("hazard", "")[:2], esc(a.get("name") or aid), ""))
    cls, words = _verdict(a)
    n_t = int(a.get("n_targets") or a.get("n_ri_events") or 0)
    n_c = int(a.get("n_controls") or a.get("n_no_ri_events") or 0)
    delta = _finite(a.get("delta_mean"))
    lo = _finite(a.get("bootstrap_ci_delta_lo", a.get("bootstrap_ci_lo")))
    hi = _finite(a.get("bootstrap_ci_delta_hi", a.get("bootstrap_ci_hi")))
    diff = (f"{fmt.num(delta, 2)} (95% interval {fmt.num(lo, 2)} to {fmt.num(hi, 2)})"
            if delta is not None and lo is not None and hi is not None else "&mdash;")
    return [f"{question}" + (f' <span class="muted">({esc(source)})</span>' if source else ""),
            f"{n_t:,} events, {n_c:,} matched times" if n_t else "&mdash;", diff,
            f'<span class="chip {cls}">{words}</span>']


def page(d: SiteData) -> str:
    r = d.research
    analyses = list(r.get("analyses") or [])
    seen = {str(a.get("analysis_id")) for a in analyses}
    for aid in HYPOTHESES:
        if aid not in seen:
            analyses.append({"analysis_id": aid, "status": "needs_curated_input"})
    hero = common.hero(
        "Research", "Do proposed precursor signals show up before hazards?",
        "Published studies have proposed that space weather, geomagnetic activity or lightning change before "
        "earthquakes, rapid hurricane intensification or tornadoes. These tests check those claims against the "
        "event record. They are research, separate from the published forecasts, and the results are published "
        "whatever they find.",
        meta=[("Last run", fmt.time_tag(r.get("generated_at")) if r.get("generated_at") else "not yet run"),
              ("Earthquakes used", f"{int(r.get('n_eq_events_used') or 0):,} M6+")])
    table = common.table(["Question", "Sample", "Difference before events", "Result"],
                         [_row(a) for a in analyses], caption="Precursor tests")
    how = ("<p>Each test compares a measurement in the hours before events with the same measurement at matched "
           "times without an event (the same calendar dates a year earlier and later). An interval that excludes "
           "zero means a difference was found in this sample &mdash; it does not show cause, and with several tests "
           "run together one may cross that line by chance. An interval that contains zero means no difference was "
           "found at this sample size.</p>"
           "<p>None of these signals is used by the published earthquake or hurricane forecasts. NOAA&rsquo;s storm "
           "lightning measurements are among the inputs of the tornado model, which uses them as NOAA reports them.</p>")
    body = (common.section("results", "Results", table + '<p class="section-foot"><a href="/api/v1/laic/summary">'
                                                         "The results as JSON</a></p>")
            + common.section("how", "How to read the results", how))
    return f'<main id="main" class="page page-research">{hero}{body}</main>'
