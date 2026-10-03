"""Render the site's model-evidence blocks from ``served_evidence`` -- never from typed numbers.

The static pages (methods, registry, tornado verification) carry marked blocks::

    <!-- hp-evidence:NAME -->  ...rendered here...  <!-- /hp-evidence:NAME -->

``render_pages`` re-renders every block from the results files; ``check_pages`` reports any page
whose blocks differ from what the files say (the site-integrity test runs it, so a results change
that is not re-rendered fails CI instead of shipping a stale number). A block whose evidence is not
bound to the served artifact says so in words; it never falls back to an older model's number.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Callable

from hazardpulse.verification import served_evidence as se

REPO = "https://github.com/coherence-energy-labs/hazardpulse"


class PageBlockError(ValueError):
    """A page is missing a block it must carry, or carries one twice."""


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _f(x: float | None, d: int = 3) -> str:
    return "--" if x is None else f"{x:.{d}f}"


def _signed(x: float | None, d: int = 3) -> str:
    if x is None:
        return "--"
    s = f"{abs(x):.{d}f}"
    return ("+" if x >= 0 else "&minus;") + s


def _ci(ci: list[float] | None, d: int = 3, signed: bool = False) -> str:
    if not ci:
        return ""
    fmt = (lambda v: _signed(v, d)) if signed else (lambda v: _f(v, d))
    return f" [{fmt(ci[0])}, {fmt(ci[1])}]"


def _pct(x: float | None, d: int = 1) -> str:
    return "--" if x is None else f"{100.0 * x:.{d}f}%"


def _pct_fine(x: float | None) -> str:
    """A rate readable at any scale (0.010%, 0.13%, 2.3%, 17%); a fixed decimal prints 0.01% as 0.0%."""
    if x is None:
        return "--"
    v = 100.0 * float(x)
    if v == 0.0:
        return "0%"
    if v >= 10.0:
        return f"{v:.0f}%"
    if v >= 1.0:
        return f"{v:.1f}%"
    if v >= 0.1:
        return f"{v:.2f}%"
    return f"{v:.3f}%"


def _n(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "--"


def _kv(label: str, value: str) -> str:
    return f'<div class="kv"><span>{label}</span><strong>{value}</strong></div>'


def _doc(path: str) -> str:
    return f'<a href="{REPO}/blob/main/{_e(path)}" rel="noopener">{_e(path)}</a>'


def _years(period: str) -> str:
    return period


def _when(t: dict) -> str:
    """A test's short period label ("2025", "2023-2025") with an en dash."""
    return re.sub(r"(\d)-(\d)", r"\1&ndash;\2", _e(t.get("when") or t.get("period", "")))


def _families(fam: dict[str, int]) -> str:
    return ", ".join(f"{v} {_e(k)}" for k, v in fam.items())


_PRODUCT_NAMES = {"p30": "Tornado within 30 min", "p90": "Tornado within 90 min",
                  "p_ef2": "EF2+ tornado within 60 min"}


# ---------------------------------------------------------------------------
# methods page
# ---------------------------------------------------------------------------

def methods_simple(ev: dict) -> str:
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    out = []
    if eq:
        t = eq["test"]
        replaced = eq.get("replaced_ig_per_target") or {}
        out.append(
            '<div class="card col-4 hazard-eq">\n  <h3>Earthquake</h3>\n  <p class="muted">'
            f"The chance of a magnitude {eq['target_magnitude_min']:.0f}+ earthquake in each 2&deg; cell of the globe "
            f"over the next {eq['horizon_days']:.0f} days, from where quakes happen in the long run, how they cluster "
            f"after recent ones, and boosted trees on both. Tested once on {_when(t)}: "
            f"{_signed(t['ig_per_target']['value'], 2)} nats of information per quake over a uniform map"
            + (f" (the model it replaced: {_signed(replaced.get('value'), 2)})" if replaced.get("value") is not None else "")
            + ". Earthquakes cannot be predicted; this ranks where the odds are higher.</p>\n</div>")
    else:
        out.append('<div class="card col-4 hazard-eq">\n  <h3>Earthquake</h3>\n  <p class="muted">'
                   "No final test is bound to the served earthquake model in this repository.</p>\n</div>")
    if hu:
        t = hu["test"]
        ob = (hu.get("other_basins") or {}).get("test") or {}
        same = hu.get("v8_2_same_cases") or {}
        out.append(
            '<div class="card col-4 hazard-hu">\n  <h3>Hurricane</h3>\n  <p class="muted">'
            "Chance of rapid intensification (a 30+ knot wind increase in 24 hours). For storms the National "
            f"Hurricane Center tracks we serve {_e(hu['candidate_name'].split(' (')[0])}, which beat every alternative "
            f"in a pre-registered test on {_when(t)}: AUC {_f(t['auc'])}"
            + (f" vs our own model&rsquo;s {_f(same.get('auc'))} on the same cycles" if same.get("auc") is not None else "")
            + "."
            + (f" Everywhere else, our v8.2 model (AUC {_f(ob.get('auc'))} on held-out {_when(ob)} cases)."
               if ob.get("auc") is not None else "")
            + " Each storm says which.</p>\n</div>")
    else:
        out.append('<div class="card col-4 hazard-hu">\n  <h3>Hurricane</h3>\n  <p class="muted">'
                   "No final test is bound to the served hurricane model in this repository.</p>\n</div>")
    if to:
        t = to["test"]
        pt = ((to.get("probtor_final") or {}).get("probtor_raw") or {}).get("auc")
        inputs = "NOAA&rsquo;s own storm attributes (radar, lightning, satellite, environment)"
        if to.get("inputs_use_nws"):
            inputs += " and the live NWS tornado-warning state"
        if to.get("inputs_use_hrrr"):
            inputs += " and the HRRR environment"
        out.append(
            '<div class="card col-4 hazard-to">\n  <h3>Tornado</h3>\n  <p class="muted">'
            "Scores every thunderstorm NOAA&rsquo;s ProbSevere tracks: the chance that this storm produces a tornado "
            f"within the next hour, from {inputs}. On every storm observation of 2025, scored once after training: "
            f"AUC {_f(t['auc'])}" + (f" (NOAA ProbTor {_f(pt)})" if pt is not None else "") + ".</p>\n</div>")
    else:
        out.append('<div class="card col-4 hazard-to">\n  <h3>Tornado</h3>\n  <p class="muted">'
                   "No final test is bound to the served tornado model in this repository.</p>\n</div>")
    return "\n".join(out)


def methods_earthquake(ev: dict) -> str:
    eq = ev.get("earthquake")
    if not eq:
        return ('<div class="card col-4 hazard-eq">\n  <h3>Earthquake</h3>\n'
                '  <p class="muted">No final test is bound to the served earthquake model.</p>\n</div>')
    t = eq["test"]
    rows = [
        _kv("Target", f"P(an M{eq['target_magnitude_min']:.1f}+ epicentre in the 2&deg; cell within "
                      f"{eq['horizon_days']:.0f} days), every one of {_n(t['n_cells'])} cells, from events before the "
                      "issue time only"),
        _kv("Architecture", f"{_e(eq['candidate_name'][0].upper() + eq['candidate_name'][1:])}"
                            f" ({_n(eq.get('n_trees'))} trees, {_n(eq.get('n_inputs'))} inputs, scored in NumPy)"),
        _kv("How it was chosen", "Pre-registered comparison of smoothed seismicity, aftershock clustering, two "
                                 f"boosted models and the previous model ({_doc(eq['program'])})"),
        _kv(f"Final test {_when(t)}",
            f"{_signed(t['ig_per_target']['value'], 2)}{_ci(t['ig_per_target']['ci'], 2)} nats per quake over a "
            f"uniform map; AUC {_f(t['auc']['value'])}{_ci(t['auc']['ci'])} over all cells, "
            f"{_f(t['auc_active_cells']['value'])}{_ci(t['auc_active_cells']['ci'])} among recently active cells; "
            f"Brier skill {_signed(t['bss']['value'])}"),
    ]
    a = (eq.get("vs") or {}).get("A")
    if a and a.get("ig_per_target"):
        rows.append(_kv("Against the standard reference",
                        f"vs {_e(a['name'])}: {_signed(a['ig_per_target']['diff'])}"
                        f"{_ci(a['ig_per_target']['ci'], signed=True)} nats per quake, AUC "
                        f"{_signed(a['auc']['diff'], 4)}{_ci(a['auc']['ci'], 4, signed=True)} (paired, by month)"))
    replaced = eq.get("replaced_ig_per_target")
    if replaced and replaced.get("value") is not None:
        share = t.get("share_outside_active_cells")
        rows.append(_kv("Replaced", f"The previous model scored {_signed(replaced['value'], 2)} nats per quake on "
                                    "the same test"
                        + (f"; {_pct(share, 0)} of the test&rsquo;s M6+ cell-windows fell outside recently active "
                           "cells" if share is not None else "")))
    cr = t.get("calib_ratio") or {}
    if cr.get("value") is not None:
        over = cr["value"] - 1.0
        rows.append(_kv("Limits", f"Forecasts {_pct(abs(over), 0)} {'more' if over > 0 else 'fewer'} quakes than "
                                  f"occurred in the test period (ratio {_f(cr['value'], 2)}{_ci(cr.get('ci'), 2)}); "
                                  "live catalogs are preliminary in the first days after a large quake"))
    body = "\n  ".join(rows)
    return (f'<div class="card col-4 hazard-eq">\n  <h3>Earthquake ({_e(eq["model_version"])})</h3>\n  {body}\n</div>')


def methods_hurricane(ev: dict) -> str:
    hu = ev.get("hurricane")
    if not hu:
        return ('<div class="card col-4 hazard-hu">\n  <h3>Hurricane RI</h3>\n'
                '  <p class="muted">No final test is bound to the served hurricane model.</p>\n</div>')
    t = hu["test"]
    seasons = hu.get("chosen_on_seasons") or ["?", "?"]
    better = [c["against"] for c in hu.get("claims", []) if c["better"]]
    rows = [
        _kv("What is served", f"{_e(hu['candidate_name'])} as issued (SHIPS-RII where it is missing: "
                              f"{_n(hu.get('fallback_cycles'))} of {_n(t['n'])} cycles in the test), read from "
                              "NHC&rsquo;s SHIPS text for the same cycle"),
        _kv("How it was chosen", f"Pre-registered comparison of NOAA&rsquo;s aids and pools of them on "
                                 f"{seasons[0]}&ndash;{seasons[-1]} by forward chaining ({_doc(hu['program'])})"),
        _kv(f"Final test ({_when(t)}, read once)",
            f"AUC {_f(t['auc'])}{_ci(t['auc_ci'])}, Brier skill {_signed(t['bss'])}{_ci(t['bss_ci'], signed=True)}; "
            f"{_n(t['n'])} cycles, {_n(t['events'])} RI events, {_n(t['storms'])} storms (intervals by storm)"),
    ]
    if better:
        rows.append(_kv("Beats", ", ".join(_e(b) for b in better)
                        + " (paired 95% intervals on Brier and log loss below zero)"))
    adv = hu.get("adversary") or {}
    if adv.get("verdict"):
        rows.append(_kv("Independent attack", _e(adv["verdict"])))
    out = [f'<div class="card col-4 hazard-hu">\n  <h3>Hurricane RI, NHC basins ({_e(hu["model_version"])})</h3>\n  '
           + "\n  ".join(rows) + "\n</div>"]
    ob = hu.get("other_basins")
    if ob:
        o = ob["test"]
        out.append(
            '<div class="card col-4 hazard-hu">\n  <h3>Hurricane RI, other basins (hurricane_ri_v8_2)</h3>\n  '
            + "\n  ".join([
                _kv("Architecture", "Histogram-GBT (depth 3 + 4) + L2 logistic + bagged logistic ensemble, "
                                    "Newton-calibrated"),
                _kv("Key inputs", "Analysis intensity and pressure (CARQ) and their 6&ndash;24 h tendencies, "
                                  "aid-model intensity forecasts, climatological potential intensity, motion, storm age"),
                _kv("AUC", f"{_f(o['auc'])}{_ci(o['auc_ci'])} on {_n(o['n'])} held-out "
                           f"{_when(o)} NHC cases"
                    + (f"; {_f((hu.get('v8_2_same_cases') or {}).get('auc'))} on the 2025 NHC cycles above"
                       if (hu.get("v8_2_same_cases") or {}).get("auc") is not None else "")),
                _kv("Used for", "West Pacific, Indian Ocean and Southern Hemisphere storms (no public RI guidance), "
                                "and NHC cycles without SHIPS text"),
            ]) + "\n</div>")
    return "\n\n".join(out)


def _tornado_inputs_text(to: dict) -> str:
    return _families(to.get("input_families") or {})


_PROBTOR_LABEL = {"probtor_raw": "as issued", "probtor_tiebroken": "with its integer ties broken"}


def _probtor_sentence(to: dict) -> str | None:
    """ProbTor's AUC as issued and the paired gap to the comparison least favourable to us."""
    vs = to.get("vs_probtor")
    if not vs:
        return None
    pf = to.get("probtor_final") or {}
    issued = (pf.get("probtor_raw") or {}).get("auc")
    against = (pf.get(vs["against"]) or {}).get("auc")
    s = f"NOAA ProbTor scores {_f(issued)} as issued"
    if vs["against"] != "probtor_raw" and against is not None:
        s += f" and {_f(against)} {_PROBTOR_LABEL.get(vs['against'], vs['against'])}"
    s += (f"; the model is {_signed(vs['delta_auc'])}{_ci(vs['delta_auc_ci'], signed=True)} AUC above the "
          f"{'tie-broken' if vs['against'] == 'probtor_tiebroken' else 'issued'} version, paired by day")
    return s


def _nws_sentence(to: dict) -> str | None:
    """At the warnings' own false-alarm rate -- with AND without the warning state as an input,
    because a model that reads the warnings only measures what it adds on top of them."""
    nws = to.get("vs_nws_warnings")
    if not nws or nws.get("model_pod") is None:
        return None
    s = (f"at the NWS warnings&rsquo; own false-alarm rate it catches {_pct(nws['model_pod'])} of tornadic storm "
         f"observations vs the warnings&rsquo; {_pct(nws['nws_pod'])} ("
         f"{_signed(100 * nws['delta_pod'], 1)}"
         + (_ci([100 * x for x in nws["delta_pod_ci"]], 1, signed=True) if nws.get("delta_pod_ci") else "")
         + " points, using the warning state as an input)")
    alone = (to.get("fallback") or {}).get("vs_nws_warnings")
    if alone and alone.get("model_pod") is not None:
        s += (f"; without that input it catches {_pct(alone['model_pod'])} ("
              f"{_signed(100 * alone['delta_pod'], 1)}"
              + (_ci([100 * x for x in alone["delta_pod_ci"]], 1, signed=True) if alone.get("delta_pod_ci") else "")
              + " points)")
    return s


def methods_tornado(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return ('<div class="card col-4 hazard-to">\n  <h3>Tornado storm-object</h3>\n'
                '  <p class="muted">No final test is bound to the served tornado model.</p>\n</div>')
    t = to["test"]
    products = [_PRODUCT_NAMES[k] for k in ("p30", "p90", "p_ef2") if k in (to.get("products") or {})]
    rows = [
        _kv("Target", _e(to["event"][0].upper() + to["event"][1:])
            + (f" (also served: {', '.join(_e(p.lower()) for p in products)})" if products else "")),
        _kv("Architecture", f"LightGBM ({_n(to['n_trees'])} trees), exported to a JSON payload scored in pure NumPy; "
                            "Platt calibration on leave-one-year-out scores"
                            + ("; Venn&ndash;Abers band per storm" if to.get("has_band") else "")),
        _kv(f"Inputs ({to['n_inputs']})", _tornado_inputs_text(to)),
        _kv("How it was chosen", f"Pre-registered programme: every choice on 2023; development test 2024; refit "
                                 f"2020&ndash;2024; final test 2025 read once ({_doc(to['program'])})"),
        _kv("Final test 2025", f"AUC {_f(t['auc'])}{_ci(t['auc_ci'])} on all {_n(t['n'])} storm observations "
                               f"({_n(t['pos'])} tornadic); Brier skill {_signed(t['bss'])}"
                               + (f". 2024: {_f(to['dev']['auc'])}" if to.get("dev") else "")),
    ]
    against = [x for x in (_probtor_sentence(to), _nws_sentence(to)) if x]
    if against:
        rows.append(_kv("Against NOAA", "; ".join(against)))
    if to.get("results_bound") is False:
        rows.append(_kv("Comparisons", "The lab results for this payload are not in the repository, so no "
                                       "comparison is quoted"))
    for nu in to.get("tested_not_served") or []:
        if "coherence" in nu["what"] and "PDE" not in nu["what"]:
            rows.append(_kv("Coherence field", f"Tested on top of the HRRR fields: {_signed(nu['delta_auc'], 4)}"
                                               f"{_ci(nu['delta_auc_ci'], 4, signed=True)} validation AUC (paired); "
                                               "no lift, so not served"))
    rows.append(_kv("Status", "Live"))
    return (f'<div class="card col-4 hazard-to">\n  <h3>Tornado storm-object ({_e(to["model_version"])})</h3>\n  '
            + "\n  ".join(rows) + "\n</div>")


# ---------------------------------------------------------------------------
# registry page
# ---------------------------------------------------------------------------

def registry_simple(ev: dict) -> str:
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    cards = []
    if eq:
        t = eq["test"]
        cards.append(('hazard-eq', "Earthquake model",
                      f"Combines where large quakes happen in the long run with how they cluster after recent ones. "
                      f"On {_when(t)}, scored once, it ranked a cell that went on to have an M6+ quake above one "
                      f"that did not {_pct(t['auc']['value'], 0)} of the time across the globe, and "
                      f"{_pct(t['auc_active_cells']['value'], 0)} among cells with recent quakes, the harder question."))
    if hu:
        t = hu["test"]
        cards.append(('hazard-hu', "Hurricane model",
                      f"For storms the National Hurricane Center tracks, we serve NOAA&rsquo;s own DTOPS guidance: in a "
                      f"pre-registered test on the {_when(t)} season it beat every alternative, including our own "
                      f"model, ranking a forecast cycle that went on to intensify rapidly above one that did not "
                      f"{_pct(t['auc'], 0)} of the time."))
    if to:
        t = to["test"]
        cards.append(('hazard-to', "Tornado model (storm-level)",
                      f"Scores every thunderstorm NOAA tracks for the chance it produces a tornado in the next hour. "
                      f"On every storm of 2025, scored once, it ranked a storm that went on to produce a tornado above "
                      f"one that did not {_pct(t['auc'], 0)} of the time."))
    out = []
    for cls, title, text in cards:
        out.append(f'<div class="card col-6 {cls}">\n  <h3>{title}</h3>\n  <p class="muted">{text}</p>\n</div>')
    return "\n".join(out)


def _registry_row(model: str, hazard: str, status: str, test: str, auc: str, skill: str, arch: str) -> str:
    chip = {"active": "good", "fallback": "neutral"}.get(status, "neutral")
    return (f"<tr><td><strong>{_e(model)}</strong></td><td>{hazard}</td>"
            f'<td><span class="chip {chip}">{status}</span></td><td>{test}</td>'
            f'<td class="mono">{auc}</td><td class="mono">{skill}</td><td>{arch}</td></tr>')


def registry_active(ev: dict) -> str:
    rows = []
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    if eq:
        t = eq["test"]
        rows.append(_registry_row(eq["model_version"], "Earthquake (30 d, M6+)", "active",
                                  _when(t), f"{_f(t['auc']['value'])}{_ci(t['auc']['ci'])}",
                                  f"IG {_signed(t['ig_per_target']['value'], 2)} nats/quake",
                                  "Smoothed seismicity + ETAS-style clustering + boosted trees"))
    if hu:
        t = hu["test"]
        rows.append(_registry_row(hu["model_version"], "Hurricane RI (NHC basins)", "active",
                                  _when(t), f"{_f(t['auc'])}{_ci(t['auc_ci'])}",
                                  f"BSS {_signed(t['bss'])}", "NOAA DTOPS as issued (SHIPS-RII fallback)"))
        ob = hu.get("other_basins")
        if ob:
            o = ob["test"]
            rows.append(_registry_row(ob["model"], "Hurricane RI (other basins)", "active",
                                      _when(o), f"{_f(o['auc'])}{_ci(o['auc_ci'])}",
                                      f"BSS {_signed(o['bss'])}", "GBT + logistic ensemble, Newton-calibrated"))
    if to:
        t = to["test"]
        rows.append(_registry_row(to["model_version"], "Tornado (storm, 60 min)", "active", "2025",
                                  f"{_f(t['auc'])}{_ci(t['auc_ci'])}", f"BSS {_signed(t['bss'])}",
                                  f"LightGBM, {_n(to['n_trees'])} trees, {to['n_inputs']} inputs"))
        fb = to.get("fallback")
        if fb:
            ft = fb["test"]
            rows.append(_registry_row(fb["model_version"], "Tornado (no NWS feed)", "fallback", "2025",
                                      f"{_f(ft['auc'])}{_ci(ft['auc_ci'])}", f"BSS {_signed(ft['bss'])}",
                                      f"LightGBM, {_n(fb['n_trees'])} trees, {fb['n_inputs']} inputs"))
        for key, p in (to.get("products") or {}).items():
            pt = p["test"]
            rows.append(_registry_row(p["model_version"], _e(_PRODUCT_NAMES[key]), "active", "2025",
                                      f"{_f(pt['auc'])}{_ci(pt['auc_ci'])}", f"BSS {_signed(pt['bss'])}",
                                      f"LightGBM, {_n(p['n_trees'])} trees"))
    return "\n".join(rows)


def registry_history(ev: dict) -> str:
    rows = []
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")

    def row(action: str, model: str, reason: str) -> str:
        chip = {"promote": "good", "retire": "neutral"}[action]
        return (f'<tr><td>2026-10</td><td><span class="chip {chip}">{action}</span></td>'
                f"<td>{_e(model)}</td><td>{reason}</td></tr>")
    if to:
        t = to["test"]
        vs = to.get("vs_probtor")
        rows.append(row("promote", to["model_version"],
                        f"Pre-registered programme, 2025 read once: AUC {_f(t['auc'])}{_ci(t['auc_ci'])}"
                        + (f"; vs NOAA ProbTor {_signed(vs['delta_auc'])}{_ci(vs['delta_auc_ci'], signed=True)} "
                           "(paired by day)" if vs else "")))
        v2 = to.get("vs_v2")
        rows.append(row("retire", "tornado_gbt_v2",
                        "Superseded by v3" + (f": on 2025, v3 &minus; v2 {_signed(v2['delta_auc'])}"
                                               f"{_ci(v2['delta_auc_ci'], signed=True)} AUC on the same storms and event"
                                               if v2 else "")))
        rows.append(row("retire", "hp-tornado-coherence-v1",
                        "Its published 0.894 was measured on a 5:1-downsampled split with labels on the wrong clock "
                        "(docs/AUDIT_2026-10-01.md)"))
    if eq:
        t = eq["test"]
        rows.append(row("promote", eq["model_version"],
                        f"Pre-registered, {_when(t)} read once: "
                        f"{_signed(t['ig_per_target']['value'], 2)} nats per quake, AUC {_f(t['auc']['value'])}"))
        replaced = eq.get("replaced_ig_per_target")
        if replaced and replaced.get("value") is not None:
            rows.append(row("retire", "eq_coherence_v1_0",
                            f"{_signed(replaced['value'], 2)} nats per quake on the same test (worse than a uniform map)"))
    if hu:
        t = hu["test"]
        rows.append(row("promote", hu["model_version"],
                        f"NOAA DTOPS served for NHC basins: {_when(t)} AUC {_f(t['auc'])}; beats "
                        + ", ".join(_e(c["against"]) for c in hu.get("claims", []) if c["better"])))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# tornado verification page
# ---------------------------------------------------------------------------

def tornado_hero(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return ('<p class="subtitle">No final test is bound to the served tornado model in this repository.</p>')
    return ('<p class="subtitle">\n  Out-of-sample results for the served HazardPulse tornado model, '
            f"<span class=\"mono\">{_e(to['model_version'])}</span>. Every number on this page is read from the "
            "results files of a pre-registered programme whose final test (every storm of 2025) was run once, "
            "after every choice was frozen.\n</p>")


def _metric_row(name: str, m: dict, best: bool = False) -> str:
    cls = ' class="best-row"' if best else ""
    return (f"<tr{cls}><td>{name}</td><td class=\"mono\">{_f(m.get('auc'))}</td>"
            f"<td class=\"mono\">{_ci(m.get('auc_ci')).strip() or '--'}</td>"
            f"<td class=\"mono\">{_f(m.get('pr_auc'))}</td><td class=\"mono\">{_signed(m.get('bss')) if m.get('bss') is not None else '--'}</td>"
            f"<td class=\"mono\">{_n(m.get('n'))} / {_n(m.get('pos'))}</td></tr>")


def tornado_body(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return ('<section class="section"><div class="card"><p class="muted">No final test is bound to the served '
                "tornado model in this repository, so no result is shown.</p></div></section>")
    t = to["test"]
    fam = _tornado_inputs_text(to)
    parts = []
    parts.append(
        '<section class="section" aria-labelledby="overview-heading">\n'
        '  <h2 id="overview-heading">The served model</h2>\n  <div class="card">\n    <div class="overview-grid">\n'
        "      <div>\n        "
        + "\n        ".join([
            _kv("Model", f"<span class=\"mono\">{_e(to['model_version'])}</span>"),
            _kv("Forecasts", _e(to["forecasts"])),
            _kv("Event", _e(to["event"])),
            _kv(f"Inputs ({to['n_inputs']})", fam),
        ])
        + "\n      </div>\n      <div>\n        "
        + "\n        ".join([
            _kv("Model form", f"LightGBM, {_n(to['n_trees'])} trees, scored in NumPy from a JSON payload"),
            _kv("Calibration", "Platt on leave-one-year-out scores"
                + ("; Venn&ndash;Abers band per storm" if to.get("has_band") else "")),
            _kv("Training", _e(to["trained"])),
            _kv("Final test", f"2025, read once: {_n(t['n'])} storm observations, {_n(t['pos'])} tornadic"),
        ])
        + "\n      </div>\n    </div>\n"
        f'    <p class="muted" style="margin-top:12px;font-size:12px;">Protocol: {_doc(to["program"])}. Every choice '
        "(inputs, model family, hyperparameters, calibration, label) was made on 2023; 2024 was a development test; "
        "the model was then refitted on 2020&ndash;2024 and scored once on 2025. A forecast is scored against SPC "
        "storm reports matched to the storm&rsquo;s own tracked radar polygon.</p>\n  </div>\n</section>")

    rows = [_metric_row("Served: with the NWS warning state", t, best=True)]
    fb = to.get("fallback")
    if fb:
        rows.append(_metric_row("Fallback: no NWS input (feed down)", fb["test"]))
    for key, p in (to.get("products") or {}).items():
        rows.append(_metric_row(_e(_PRODUCT_NAMES[key]), p["test"]))
    for key, name in (("probtor_raw", "NOAA ProbTor, as issued"), ("probtor_tiebroken", "NOAA ProbTor, ties broken")):
        m = (to.get("probtor_final") or {}).get(key)
        if m:
            rows.append(f"<tr><td>{name}</td><td class=\"mono\">{_f(m.get('auc'))}</td>"
                        f"<td class=\"mono\">{_ci(m.get('auc_ci')).strip() or '--'}</td>"
                        "<td class=\"mono\">--</td><td class=\"mono\">--</td><td class=\"mono\">same storms</td></tr>")
    dev = to.get("dev")
    parts.append(
        '<section class="section" aria-labelledby="results-heading">\n'
        '  <h2 id="results-heading">Final test: every storm of 2025</h2>\n  <div class="card">\n'
        '    <table class="results-table">\n      <thead><tr><th>Forecast</th><th>AUC</th><th>95% CI</th>'
        "<th>PR-AUC</th><th>Brier skill</th><th>Storm obs / tornadic</th></tr></thead>\n      <tbody>\n        "
        + "\n        ".join(rows) + "\n      </tbody>\n    </table>\n"
        '    <p class="muted" style="margin-top:12px;font-size:12px;">Intervals: bootstrap over days. Brier skill '
        "is against the test year&rsquo;s base rate."
        + (f" Development year {_when(dev)}: AUC {_f(dev['auc'])}{_ci(dev['auc_ci'])}, Brier "
           f"skill {_signed(dev['bss'])}." if dev else "")
        + "</p>\n  </div>\n</section>")

    comp = []
    vs = to.get("vs_probtor")
    s = _probtor_sentence(to)
    if s:
        comp.append(f"<li><strong>NOAA ProbTor</strong> {s.replace('NOAA ProbTor ', '', 1)} over "
                    f"{_n(vs.get('n_days'))} days; Brier "
                    f"{_signed(vs['delta_brier'], 6)}{_ci(vs['delta_brier_ci'], 6, signed=True)}. The issued product "
                    "is an integer percent and 0 for most storms, so its ties are broken by ProbSevere&rsquo;s own "
                    "severe probability for a fair ranking comparison.</li>")
    s = _nws_sentence(to)
    nws = to.get("vs_nws_warnings")
    if s:
        comp.append(f"<li><strong>NWS tornado warnings:</strong> {s[0].upper() + s[1:]}. The false-alarm rate is "
                    f"the warnings&rsquo; own ({_pct(nws['nws_pofd'], 3)} of non-tornadic storm observations). A model "
                    "that reads the warning state measures what it adds on top of the warnings; the figure without "
                    "that input is the contest on its own.</li>")
    v2 = to.get("vs_v2")
    if v2:
        comp.append(f"<li><strong>Our previous model (v2):</strong> {_signed(v2['delta_auc'])}"
                    f"{_ci(v2['delta_auc_ci'], signed=True)} AUC on the same storms and the same event.</li>")
    if comp:
        parts.append('<section class="section" aria-labelledby="comparison-heading">\n'
                     '  <h2 id="comparison-heading">Against NOAA and the NWS, on the same storms</h2>\n'
                     '  <div class="card">\n    <ul class="theory-list">\n      ' + "\n      ".join(comp)
                     + "\n    </ul>\n  </div>\n</section>")

    st = to.get("stress")
    if st:
        srows = []
        for r in st["strata"]:
            srows.append(f"<tr><td>{_e(r['stratum'].replace('_', ' '))}</td><td class=\"mono\">{_n(r['n'])} / "
                         f"{_n(r['pos'])}</td><td class=\"mono\">{_f(r['auc'])}{_ci(r['auc_ci'])}</td>"
                         f"<td class=\"mono\">{_f(r['probtor_auc'])}{_ci(r['probtor_ci'])}</td></tr>")
        parts.append(
            '<section class="section" aria-labelledby="stress-heading">\n'
            '  <h2 id="stress-heading">Every stratum</h2>\n  <div class="card">\n'
            f'    <p class="muted" style="margin-bottom:12px;">The protocol&rsquo;s strata, defined before any result. '
            f"Lowest AUC in any stratum: {_f(st['min_auc'])}; in {st['n_clear_of_probtor']} of {st['n_strata']} the "
            "model&rsquo;s interval lies wholly above calibrated ProbTor&rsquo;s.</p>\n"
            '    <table class="results-table">\n      <thead><tr><th>Stratum</th><th>Storm obs / tornadic</th>'
            "<th>Model AUC</th><th>ProbTor (calibrated) AUC</th></tr></thead>\n      <tbody>\n        "
            + "\n        ".join(srows) + "\n      </tbody>\n    </table>\n  </div>\n</section>")

    rel = to.get("reliability")
    if rel:
        rrows = []
        for b in rel["bins"]:
            if not b.get("n"):
                continue
            ci = b.get("observed_ci")
            rrows.append(f"<tr><td class=\"mono\">{_pct_fine(b['lo'])} &ndash; {_pct_fine(b['hi'])}</td>"
                         f"<td class=\"mono\">{_n(b['n'])}</td><td class=\"mono\">{_pct_fine(b['mean_forecast'])}</td>"
                         f"<td class=\"mono\">{_pct_fine(b['observed'])}"
                         f"{(' [' + _pct_fine(ci[0]) + ', ' + _pct_fine(ci[1]) + ']') if ci else ''}</td></tr>")
        parts.append(
            '<section class="section" aria-labelledby="calibration-heading">\n'
            '  <h2 id="calibration-heading">Does 10% mean 10%?</h2>\n  <div class="card">\n'
            '    <p class="muted" style="margin-bottom:12px;">2025 storm observations grouped by the probability the '
            "served model gave them, and how often a tornado followed (95% Jeffreys interval).</p>\n"
            '    <table class="results-table">\n      <thead><tr><th>Forecast range</th><th>Storm obs</th>'
            "<th>Mean forecast</th><th>Observed</th></tr></thead>\n      <tbody>\n        "
            + "\n        ".join(rrows) + "\n      </tbody>\n    </table>\n  </div>\n</section>")

    nu = to.get("tested_not_served") or []
    if nu:
        items = []
        for x in nu:
            how = "paired by day" if x.get("paired") else "difference of two validation runs"
            items.append(f"<li><strong>{_e(x['what'][0].upper() + x['what'][1:])}:</strong> "
                         f"{_signed(x['delta_auc'], 4)}{_ci(x.get('delta_auc_ci'), 4, signed=True)} validation AUC "
                         f"({how}{'; ' + _e(x['note']) if x.get('note') else ''}).</li>")
        parts.append('<section class="section" aria-labelledby="theory-heading">\n'
                     '  <h2 id="theory-heading">Tested and not served</h2>\n  <div class="card">\n'
                     '    <p class="muted" style="margin-bottom:12px;">Each was measured on 2023 under the same '
                     "protocol. None improved the forecast, so none is in the served model.</p>\n"
                     '    <ul class="theory-list">\n      ' + "\n      ".join(items) + "\n    </ul>\n  </div>\n</section>")
    return "\n\n".join(parts)


def tornado_different(ev: dict) -> str:
    to = ev.get("tornado")
    items = [
        ("Pre-registered and read once",
         "Every modelling choice was fixed on 2023 and checked on 2024 before 2025 was scored, once. The protocol and "
         "its dated amendments are public, including the ones that corrected our own earlier numbers."),
        ("Scored against NOAA on the same storms",
         "The comparison with NOAA ProbTor and the NWS warnings uses the same storm observations, the same event and "
         "intervals paired by day."),
        ("Open and reproducible",
         "The served model is a JSON file in the repository, scored in pure NumPy; its identity is the hash of its "
         "bytes, and every live forecast is appended to a hash-chained ledger."),
    ]
    if to and to.get("tested_not_served"):
        items.append(("We publish what did not work",
                      "The HRRR environment and the coherence field were tested and did not improve the forecast; "
                      "the measurements are above."))
    lis = "\n    ".join(f"<li>\n      <strong>{t}</strong>\n      <p>{d}</p>\n    </li>" for t, d in items)
    links = [
        (f"{REPO}", "GitHub repository"),
        (f"{REPO}/blob/main/{se.TORNADO_PROGRAM}", "Model programme"),
        ("/data/tornado-ledger.jsonl", "Prediction ledger (JSONL)"),
        ("/verification/", "All verification scores"),
    ]
    noopener = ' rel="noopener"'
    link_html = "\n  ".join(
        f'<a href="{_e(h)}" class="btn btn-secondary" style="padding:8px 16px;font-size:13px;"'
        f'{noopener if h.startswith("http") else ""}>{t}</a>' for h, t in links)
    return (f'<div class="card">\n  <ul class="diff-list">\n    {lis}\n  </ul>\n</div>\n\n'
            f'<div class="link-row">\n  {link_html}\n</div>')


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

BLOCKS: dict[str, dict[str, Callable[[dict], str]]] = {
    "methods/index.html": {
        "methods-simple": methods_simple,
        "methods-earthquake": methods_earthquake,
        "methods-hurricane": methods_hurricane,
        "methods-tornado": methods_tornado,
    },
    "registry/index.html": {
        "registry-simple": registry_simple,
        "registry-active": registry_active,
        "registry-history": registry_history,
    },
    "verification/tornado/index.html": {
        "tornado-hero": tornado_hero,
        "tornado-body": tornado_body,
        "tornado-different": tornado_different,
    },
}


def _marker_re(name: str, prefix: str) -> re.Pattern:
    return re.compile(r"(<!-- " + prefix + ":" + re.escape(name) + r" -->)(.*?)(<!-- /" + prefix + ":"
                      + re.escape(name) + r" -->)", re.DOTALL)


def apply_block(page: str, name: str, content: str, prefix: str = "hp-evidence") -> str:
    """Replace what lies between a block's markers. A page without the block -- or with it twice --
    is an error, never a silent no-op (a regex that stopped matching froze the public tornado
    ledger from 2026-03-31 while every run printed "ledger baked in")."""
    rx = _marker_re(name, prefix)
    found = rx.findall(page)
    if len(found) != 1:
        raise PageBlockError(f"block {prefix}:{name}: {len(found)} marker pairs (need exactly 1)")
    return rx.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), page)


def render_page(rel: str, page: str, ev: dict) -> str:
    for name, fn in BLOCKS[rel].items():
        page = apply_block(page, name, fn(ev))
    return page


def render_pages(dist: Path, root: Path = se.ROOT, ev: dict | None = None) -> list[str]:
    """Re-render every evidence block; returns the pages that changed."""
    ev = se.all_evidence(root) if ev is None else ev
    changed = []
    for rel in BLOCKS:
        path = dist / rel
        old = path.read_text(encoding="utf-8")
        new = render_page(rel, old, ev)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


def check_pages(dist: Path, root: Path = se.ROOT, ev: dict | None = None) -> list[str]:
    """Pages whose evidence blocks differ from what the results files say (empty = all current)."""
    ev = se.all_evidence(root) if ev is None else ev
    stale = []
    for rel in BLOCKS:
        old = (dist / rel).read_text(encoding="utf-8")
        if render_page(rel, old, ev) != old:
            stale.append(rel)
    return stale
