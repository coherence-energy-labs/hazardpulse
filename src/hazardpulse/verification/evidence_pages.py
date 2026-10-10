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
import math
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


def _pct_sig(x: float | None, sig: int = 2) -> str:
    """A small probability as a percentage to ``sig`` significant figures: 6.9e-4 -> 0.069%, 1.3e-5 -> 0.0013%."""
    if x is None or x <= 0:
        return "--"
    v = 100.0 * float(x)
    d = max(0, sig - 1 - math.floor(math.log10(v)))
    return f"{v:.{d}f}%"


def _n(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "--"


def _kv(label: str, value: str) -> str:
    """One label/value row; a card's rows go inside ``_facts``."""
    return f"<div><dt>{label}</dt><dd>{value}</dd></div>"


def _facts(rows: list[str]) -> str:
    return '<dl class="facts">\n    ' + "\n    ".join(rows) + "\n  </dl>"


def _card(cls: str, title: str, body: str, ident: str = "") -> str:
    idattr = f' id="{ident}"' if ident else ""
    return f'<article class="card {cls}"{idattr}>\n  <h3>{title}</h3>\n  {body}\n</article>'


def _table(head: list[str], rows: list[str], caption: str = "") -> str:
    ths = "".join(f'<th scope="col">{h}</th>' for h in head)
    cap = f"<caption>{caption}</caption>" if caption else ""
    label = html.unescape(re.sub("<[^>]+>", "", caption or head[0]))     # an entity is spelled once, not twice
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{_e(label)}">'
            f"<table>{cap}<thead><tr>{ths}</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table></div>")


def _doc(path: str) -> str:
    return f'<a href="{REPO}/blob/main/{_e(path)}" rel="noopener">{_e(path)}</a>'


def _years(period: Any) -> str:
    """A span read from an artifact ("2022-2025") with an en dash; "--" when the artifact gives none."""
    return re.sub(r"(\d)-(\d)", r"\1&ndash;\2", _e(period)) if period else "--"


def _level(x: float | None) -> str:
    """A confidence level read from a results file: 0.9875 -> "98.75%", 0.95 -> "95%"."""
    return "--" if x is None else (f"{100 * x:.4f}".rstrip("0").rstrip(".") + "%")


def _when(t: dict) -> str:
    """A test's short period label ("2025", "2023-2025") with an en dash."""
    return re.sub(r"(\d)-(\d)", r"\1&ndash;\2", _e(t.get("when") or t.get("period", "")))


def _replacement_test_row(ob: dict, basins: str) -> str:
    """The methods card's test row for a model that replaced v8.2 by a registered rule: the published artifact on
    unseen cycles where it publishes, against the model it replaced, with the rule and where it is written."""
    o, rep, d, rule = ob.get("test") or {}, ob.get("replaced") or {}, ob.get("vs_replaced") or {}, ob.get("rule") or {}
    lab = _e(rep.get("label") or "")
    return (f"AUC {_f(o.get('auc'))} on {_n(o.get('n'))} {_when(o)} cycles ({_n(o.get('events'))} RI events, "
            f"{_n(o.get('storms'))} storms) in the basins where it publishes" + (f" ({basins})" if basins else "")
            + ", none used to fit it"
            + (f"; log loss {_f(o.get('log_loss'), 5)} against {lab}&rsquo;s {_f(rep.get('log_loss'), 5)} on the "
               f"same cycles (difference {_signed(d.get('d_ll'), 5)}{_ci(d.get('d_ll_ci'), 5, signed=True)}, a 95% "
               f"interval by storm), Brier {_f(o.get('brier'), 5)} against {_f(rep.get('brier'), 5)}" if lab else "")
            + (f". It replaced {lab} by a rule written down before the test (amendment {_e(ob.get('amendment'))}, "
               f"{_doc(ob.get('program') or '')}): the upper end of that interval at most "
               f"{_signed(rule.get('margin'), 3)}; it was {_signed(rule.get('upper'), 5)}"
               if lab and ob.get("amendment") and rule.get("margin") is not None else ""))


def _other_basins_test(ob: dict) -> str:
    """What the other-basins model's bound test was, in one clause (served_evidence.other_basins_evidence): v8.2's
    METHOD on held-out cycles from every basin, or a replacement's own registered test -- the published artifact
    on unseen cycles in the basins where it publishes, against the model it replaced."""
    o = ob.get("test") or {}
    if ob.get("subject") == "model":
        rep = ob.get("replaced") or {}
        return (f"tested itself on {_n(o.get('n'))} {_when(o)} cycles in the basins where it publishes, none used to "
                f"fit it, with best-track inputs: AUC {_f(o.get('auc'))}, log loss {_f(o.get('log_loss'), 4)}"
                + (f" against {_e(rep['label'])}&rsquo;s {_f(rep.get('log_loss'), 4)}; it replaced {_e(rep['label'])} "
                   "by a rule written down before the test" if rep.get("label") else ""))
    return (f"its method: AUC {_f(o.get('auc'))} on {_n(o.get('n'))} held-out {_when(o)} cycles from every basin, with "
            "best-track inputs"
            + ("; the published model&rsquo;s calibration was fitted on those same cycles"
               if (ob.get("composition") or {}).get("served_calibration_fitted_on_test_cases") else ""))


def _families(fam: dict[str, int]) -> str:
    return ", ".join(f"{v} {_e(k)}" for k, v in fam.items())


_PRODUCT_NAMES = {"p30": "Tornado within 30 min", "p90": "Tornado within 90 min",
                  "p_ef2": "EF2+ tornado within 60 min"}

# The earthquake programme's information gain is (LL - LL_uniform) / (cell-windows that held an M6+)
# (scripts/earthquake_program/common.py: ig_per_target), NOT per earthquake. On the 2023-2025 test
# there are 1,384 such cell-windows, 1,637 (earthquake, window) pairs (final.json n_target_events:
# the weekly issue times' 30-day windows overlap ~4x) and 390 distinct M6+ earthquakes, so no
# "per quake" figure follows from a division. Until 2026-10 nine places on the site said "per quake".
EQ_IG_UNIT = "nats per M6+ cell-window"
EQ_IG_UNIT_EXPLAINED = ("nats per M6+ cell-window (a 2&deg; cell over a 30-day window in which an M6+ "
                        "earthquake occurred)")


# ---------------------------------------------------------------------------
# methods page
# ---------------------------------------------------------------------------

def methods_simple(ev: dict) -> str:
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    out = []
    if eq:
        t = eq["test"]
        replaced = eq.get("replaced_ig_per_target") or {}
        g1 = eq.get("gear1")
        out.append(_card(
            "hz-eq", "Earthquake",
            f"<p>The chance of a magnitude {eq['target_magnitude_min']:.0f}+ earthquake in each 2&deg; cell of the "
            f"globe over the next {eq['horizon_days']:.0f} days, from where earthquakes happen in the long run, how "
            "they cluster after recent ones, and boosted trees on both"
            + (", plus GEAR1, a published global model of where the crust is straining" if g1 else "")
            + (", weighted most where recent earthquakes are few" if g1 and g1.get("activity") else "")
            + (f". Scored on {_when(t)} (a second look at those years): " if t.get("second_read")
               else f". Tested once on {_when(t)}: ")
            + f"{_signed(t['ig_per_target']['value'], 2)} {EQ_IG_UNIT_EXPLAINED} of information over a uniform map"
            + (f" ({_e(eq.get('replaced_name', 'the model it replaced'))}: {_signed(replaced.get('value'), 2)})"
               if replaced.get("value") is not None else "")
            + ". Earthquakes cannot be predicted; this ranks where the odds are higher.</p>"))
    else:
        out.append(_card("hz-eq", "Earthquake",
                         "<p>No final test is bound to the served earthquake model in this repository.</p>"))
    if hu:
        t = hu["test"]
        ob = (hu.get("other_basins") or {}).get("test") or {}
        v82 = _e((hu.get("other_basins") or {}).get("label") or "")
        same = hu.get("v8_2_same_cases") or {}
        beaten = [_e(c["against"]) for c in hu.get("claims", []) if c["better"]]
        ours = hu.get("ours") or {}
        od = ours.get("dev") or {}
        out.append(_card(
            "hz-hu", "Hurricane",
            "<p>The chance of rapid intensification: maximum sustained winds rising 30 knots or more in 24 hours. For "
            f"storms the National Hurricane Center covers we publish {_e(hu['candidate_name'].split(' (')[0])}, which "
            + (f"beat {_join(beaten)} " if beaten else "was chosen ")
            + f"in a test written down in advance and run once on {_when(t)}: AUC {_f(t['auc'])}"
            + (f", against {_f(same.get('auc'))} for our {_e(same.get('label') or v82)} model on the same cycles"
               if same.get("auc") is not None and (same.get("label") or v82) else "")
            + "."
            + (f" Everywhere else we publish {v82} ({_other_basins_test(hu.get('other_basins') or {})})."
               if ob.get("auc") is not None and v82 else "")
            + " Each storm says which."
            + (f" Our newer model, {_e(ours['label'])}, scored better than DTOPS over {_years(ours['dev_period'])}, each "
               f"season forecast only from earlier ones (log loss {_f(od['log_loss'])} against "
               f"{_f(od['dtops_log_loss'])}); it is shown beside DTOPS, labelled experimental, while a "
               "test written down in advance scores it on new forecasts."
               if ours and od.get("log_loss") is not None and od.get("dtops_log_loss") is not None
               and ours.get("dev_period") else
               (f" Our model {_e(ours['label'])} is shown beside DTOPS, labelled experimental, while a test written "
                "down in advance scores it on new forecasts." if ours else ""))
            + "</p>"))
    else:
        out.append(_card("hz-hu", "Hurricane",
                         "<p>No final test is bound to the served hurricane model in this repository.</p>"))
    if to:
        t = to["test"]
        pt = ((to.get("probtor_final") or {}).get("probtor_raw") or {}).get("auc")
        inputs = "NOAA&rsquo;s own storm attributes (radar, lightning, satellite, environment)"
        if to.get("inputs_use_nws"):
            inputs += " and the live NWS tornado-warning state"
        if to.get("inputs_use_hrrr"):
            inputs += " and the HRRR environment"
        out.append(_card(
            "hz-to", "Tornado",
            "<p>For every thunderstorm NOAA&rsquo;s ProbSevere tracks over the contiguous US, the chance that it "
            f"produces a tornado within the next hour, from {inputs}. On every storm observation of 2025, scored once "
            f"after every choice was fixed: AUC {_f(t['auc'])}" + (f" (NOAA ProbTor: {_f(pt)})" if pt is not None else "")
            + ".</p>"))
    else:
        out.append(_card("hz-to", "Tornado",
                         "<p>No final test is bound to the served tornado model in this repository.</p>"))
    return "\n".join(out)


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _gear1_sentence(g1: dict | None) -> str:
    """How GEAR1 earned its place, read from the bound evaluation."""
    if not g1 or not g1.get("dev_vs_recalibrated"):
        return ""
    d = g1["dev_vs_recalibrated"]["ig_per_target"]
    act = g1.get("activity") or {}
    s1 = (act.get("dev_vs_S1") or {}).get("ig_per_target") or {}
    n = {3: "three", 4: "four"}.get(g1.get("n_weights") or 3, str(g1.get("n_weights")))
    return ("; GEAR1 (Bird et al. 2015) was added by a later pre-registered test: " + n + " weights "
            + (f"fitted on {_years(g1['fitted_on'])}, " if g1.get("fitted_on") else "")
            + f"decided on {_years(g1['decided_on'])} against C0 recalibrated on the "
            f"same years, {_signed(d['diff'])}{_ci(d['ci'], signed=True)} {EQ_IG_UNIT}"
            + ("; a further pre-registered test let GEAR1&rsquo;s weight follow how active the cell is "
               f"({_f(act.get('weight_typical_cell'), 2)} in a typical cell, {_f(act.get('weight_active_cell'), 2)} "
               f"where M6+ earthquakes happen), decided against GEAR1 with one weight, {_signed(s1['diff'])}"
               f"{_ci(s1['ci'], signed=True)} {EQ_IG_UNIT}" if s1.get("diff") is not None else ""))


def _cell_region(grid: dict | None, row: int, col: int) -> str | None:
    """The Flinn-Engdahl region of a forecast cell's centre, from the served grid's own geometry."""
    try:
        lat = float(grid["lat_min"]) + (row + 0.5) * float(grid["dlat"])
        lon = float(grid["lon_min"]) + (col + 0.5) * float(grid["dlon"])
    except (TypeError, KeyError, ValueError):
        return None
    from hazardpulse.site.places import fe_region
    return fe_region(lat, lon)


def earthquake_inputs_sentence(eq: dict | None) -> str:
    """What the served map's inputs are by ComCat event type, and what the registered test of removing the
    non-earthquakes decided (amendment E3), read from its run."""
    it = (eq or {}).get("input_types")
    if not it or not it.get("non_earthquakes"):
        return ""
    top = it.get("top_cell") or {}
    region = _cell_region(eq.get("grid"), top["row"], top["col"]) if top else None
    ci = it.get("dev_d_ig_ci") or [None, None]
    s = (f"Its inputs are ComCat&rsquo;s M5+ events of every type, not only earthquakes: {_n(it['non_earthquakes'])} "
         f"of the {_n(it['frozen_events'])} in its frozen catalog are other events, nuclear tests among them")
    if top.get("with") and top.get("without"):
        s += (f". The cell with the most of them" + (f" ({_e(region)}, {_n(top['removed'])} events)" if region else "")
              + f" averaged a {_pct_sig(top['with'])} chance over "
              + (f"{_years(it['dev_years'])}" if it.get("dev_years") else "the decision years")
              + f", against {_pct_sig(top['without'])} with earthquakes only, {top['with'] / top['without']:.0f} "
                "times lower")
    if it.get("margin") is not None and ci[0] is not None:
        s += (f". A test written down in advance was to remove them all unless the 95% interval of the change in "
              f"information reached below {_signed(it['margin'], 3)} {EQ_IG_UNIT}; it measured "
              f"{_signed(it['dev_d_ig'], 4)}{_ci(ci, 4, signed=True)}, so "
              + ("the earthquakes-only inputs were adopted" if it.get("adopted") else
                 "the published map keeps them. Removing only the man-made events is an open question")
              + f" ({_doc(EQ_PROGRAM_DOC)}, {_e(it['program_section'])}; {_doc(it['file'])})")
    return s


EQ_PROGRAM_DOC = se.EARTHQUAKE_PROGRAM


def methods_earthquake(ev: dict) -> str:
    eq = ev.get("earthquake")
    if not eq:
        return _card("hz-eq", "Earthquake", "<p>No final test is bound to the served earthquake model.</p>")
    t = eq["test"]
    rows = [
        _kv("Target", f"P(an M{eq['target_magnitude_min']:.1f}+ epicentre in the 2&deg; cell within "
                      f"{eq['horizon_days']:.0f} days), every one of {_n(t['n_cells'])} cells, from events before the "
                      "issue time only"),
        _kv("Architecture", f"{_e(eq['candidate_name'][0].upper() + eq['candidate_name'][1:])}"
                            f" ({_n(eq.get('n_trees'))} trees, {_n(eq.get('n_inputs'))} inputs, scored in NumPy)"),
        _kv("How it was chosen", "Pre-registered comparison of smoothed seismicity, aftershock clustering, two "
                                 f"boosted models and the previous model ({_doc(eq['program'])})"
            + _gear1_sentence(eq.get("gear1"))),
        _kv(f"{'Test' if t.get('second_read') else 'Final test'} {_when(t)}"
            + (" (a second read)" if t.get("second_read") else ""),
            f"{_signed(t['ig_per_target']['value'], 2)}{_ci(t['ig_per_target']['ci'], 2)} {EQ_IG_UNIT_EXPLAINED} over "
            f"a uniform map ({_n(t.get('n_positive'))} such cell-windows); AUC {_f(t['auc']['value'])}{_ci(t['auc']['ci'])} over all cells, "
            f"{_f(t['auc_active_cells']['value'])}{_ci(t['auc_active_cells']['ci'])} among recently active cells; "
            f"Brier skill {_signed(t['bss']['value'])}"),
    ]
    a = (eq.get("vs") or {}).get("A")
    if a and a.get("ig_per_target"):
        rows.append(_kv("Against the standard reference",
                        f"vs {_e(a['name'])}: {_signed(a['ig_per_target']['diff'])}"
                        f"{_ci(a['ig_per_target']['ci'], signed=True)} {EQ_IG_UNIT}, AUC "
                        f"{_signed(a['auc']['diff'], 4)}{_ci(a['auc']['ci'], 4, signed=True)} (paired, by month)"))
    replaced = eq.get("replaced_ig_per_target")
    if replaced and replaced.get("value") is not None:
        share = t.get("share_outside_active_cells")
        rows.append(_kv("Replaced", f"{_e(eq.get('replaced_name', 'The previous model'))[:1].upper()}"
                                    f"{_e(eq.get('replaced_name', 'The previous model'))[1:]} scored "
                                    f"{_signed(replaced['value'], 2)} {EQ_IG_UNIT} on the same test"
                        + (f"; {_pct(share, 0)} of the test&rsquo;s M6+ cell-windows fell outside recently active "
                           "cells" if share is not None else "")))
    cr = t.get("calib_ratio") or {}
    if cr.get("value") is not None:
        over = cr["value"] - 1.0
        rows.append(_kv("Limits", f"Forecasts {_pct(abs(over), 0)} {'more' if over > 0 else 'fewer'} M6+ cell-windows than "
                                  f"occurred in the test period (ratio {_f(cr['value'], 2)}{_ci(cr.get('ci'), 2)}); "
                                  "live catalogs are preliminary in the first days after a large quake"))
    inputs = earthquake_inputs_sentence(eq)
    if inputs:
        rows.append(_kv("Inputs that are not earthquakes", inputs))
    return _card("hz-eq", f'Earthquake <code>{_e(eq["model_version"])}</code>', _facts(rows), ident="earthquake-model")


def methods_hurricane(ev: dict) -> str:
    hu = ev.get("hurricane")
    if not hu:
        return _card("hz-hu", "Hurricane rapid intensification",
                     "<p>No final test is bound to the served hurricane model.</p>")
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
        verdict = str(adv["verdict"])
        rows.append(_kv("Independent review",
                        ("An adversarial re-check, by a reviewer that did not build the test, confirmed the result"
                         if verdict.upper().startswith("CONFIRMED") else _e(verdict[:1].upper() + verdict[1:].lower()))
                        + (f" ({_e(verdict.split(',', 1)[1].strip())})" if "," in verdict else "")))
    out = [_card("hz-hu", f'Hurricane, NHC areas: NOAA DTOPS <code>{_e(hu["model_version"])}</code>', _facts(rows),
                 ident="hurricane-model")]
    ours = hu.get("ours")
    if ours:
        out.append(_ours_hurricane_card(ours))
    if hu.get("v9"):
        out.append(_v9_card(hu["v9"], hu["candidate_name"].split(" (")[0]))
    ob = hu.get("other_basins")
    if ob:
        o = ob["test"]
        comp = ob.get("composition") or {}
        basins = ", ".join(f"{_e(b['name'])} {_n(b['n'])}" for b in comp.get("by_basin") or [])
        jt = comp.get("single_jtwc_warning") or {}
        limits = []
        if comp.get("inputs") == "best track":
            limits.append("the test read every input from the best track")
        if comp.get("served_calibration_fitted_on_test_cases"):
            limits.append("the published model is the method refitted, with its calibration fitted on the test "
                          "cycles themselves, so the AUC is not a test of it on unseen data")
        if jt.get("log_loss") is not None:
            lf = comp.get("late_best_track_fix") or {}
            limits.append(f"until 2026-10-05 live West Pacific storms were scored from one JTWC warning, leaving "
                          f"{jt['n_missing']} of {jt['n_inputs']} inputs unknown (and still are when RAL&rsquo;s "
                          f"real-time best track cannot be read); scored that way the {_when(o)} West Pacific cycles "
                          f"had log loss {_f(jt['log_loss'], 4)} against {_f(jt.get('log_loss_climatology'), 4)} for "
                          f"climatology ({_f(jt.get('log_loss_full_inputs'), 4)} with every input"
                          + (f"; {_f(lf['log_loss'], 4)} with the history but this cycle&rsquo;s fix not yet "
                             f"published, {lf['n_missing']} inputs filled" if lf.get("log_loss") is not None else "")
                          + ")")
            limits.append("the live history is the working best track, which the post-season one the test read revises")
        out.append(_card(
            "hz-hu", f"Hurricane, other basins: HazardPulse {_e(ob.get('label'))} <code>{_e(ob['model'])}</code>", _facts([
                _kv("Architecture", "Histogram-GBT (depth 3 + 4) + L2 logistic + bagged logistic ensemble, "
                                    "Newton-calibrated"),
                _kv("Key inputs", "Analysis intensity and pressure and their 6&ndash;24 h tendencies, position, time of "
                                  "year, climatological potential intensity, motion, storm age (no forecast aids)"),
                (_kv("Its registered test", _replacement_test_row(ob, basins)) if ob.get("subject") == "model" else
                 _kv("AUC (its method)", f"{_f(o['auc'])}{_ci(o['auc_ci'])} on {_n(o['n'])} held-out "
                                         f"{_when(o)} cycles from every basin" + (f" ({basins})" if basins else "")
                     + (f"; {_f((hu.get('v8_2_same_cases') or {}).get('auc'))} on the 2025 NHC cycles above"
                        if (hu.get("v8_2_same_cases") or {}).get("auc") is not None
                        and (hu.get("v8_2_same_cases") or {}).get("label", ob.get("label")) == ob.get("label")
                        else ""))),
                _kv("Used for", "West Pacific, Indian Ocean and Southern Hemisphere storms (no public RI guidance), "
                                "and NHC cycles without SHIPS text"),
                *([_kv("Limits of that test", "; ".join(limits)[:1].upper() + "; ".join(limits)[1:] + ".")]
                  if limits else []),
            ])))
    if hu.get("j1"):
        out.append(_j1_card(hu["j1"]))
    if hu.get("tc1"):
        out.append(_tc1_card(hu["tc1"]))
    return "\n\n".join(out)


def _v9_card(v9: dict, published: str) -> str:
    """v9.1: the entrant whose prospective claim would switch the published NHC-area number to it."""
    f = v9["final_2026"]
    run = v9.get("running") or {}
    rows = [
        _kv("What it is", f"Gradient-boosted trees ({_n(v9.get('trees'))} trees, trained on {_years(v9.get('trained'))}) on "
                          "the early intensity guidance, NOAA&rsquo;s RI probabilities and the official forecast, "
                          f"wherever {_join([_e(a) for a in v9.get('gate_aids') or []])} have a 24-hour forecast; "
                          f"{_e(published)} where they do not"),
        _kv("2026, read once", f"log loss {_f(f['log_loss'], 4)} against {_e(published)}&rsquo;s "
                               f"{_f(f['dtops_log_loss'], 4)}, Brier {_f(f['brier'], 4)} against {_f(f['dtops_brier'], 4)}, "
                               f"AUC {_f(f['auc'])} against {_f(f['dtops_auc'])} ({_n(f.get('n'))} cycles, "
                               f"{_n(f.get('events'))} RI events)"
            + (": better on all three" if f.get("better_on_every_point") else "")
            + ("; the claim was met" if f.get("claim") else
               f"; the claim rule, an interval wholly below zero, was not met, so {_e(published)} stays published")),
        _kv("Why it still matters", f"It is the one entrant whose claim would change what is published: if its "
                                    f"prospective test meets its rule at a look ({_level(v9.get('level'))} intervals; "
                                    + " and ".join(_e(x) for x in v9.get("look_dates") or [])
                                    + f"), it replaces {_e(published)} as the published number in the NHC areas"),
        _kv("Live so far (descriptive, no claim before a look)",
            (f"{_n(run['n'])} cycles from {_n(run.get('storms'))} storms ({_n(run.get('events'))} RI events): log loss "
             f"{_f(run['log_loss'], 3)} against {_e(published)}&rsquo;s {_f(run['dtops_log_loss'], 3)}, difference "
             f"{_signed(run.get('d_log_loss'), 3)}{_ci(run.get('d_log_loss_ci'), 3, signed=True)} at "
             f"{_level(run.get('level'))}"
             if run else "no cycle has matured yet")),
        _kv("Status", ("claim met at a look" if v9.get("claimed") else "in shadow: recorded on every live NHC cycle, "
                       "never shown as the forecast")),
    ]
    return _card("hz-hu", f'Hurricane, NHC areas, the entrant that could replace {_e(published)}: '
                          f'{_e(v9["label"])} <code>{_e(v9["model_version"])}</code>', _facts(rows), ident="hurricane-v9")


def _tc1_leads_ahead(diff: dict | None) -> list[str]:
    """The leads (h) from which every later lead's interval lies wholly below zero."""
    per = (diff or {}).get("per_lead") or {}
    leads = sorted(per, key=int)
    out = []
    for i, lead in enumerate(leads):
        if all((per[x].get("ci") or [0, 0])[1] < 0 for x in leads[i:]):
            out = leads[i:]
            break
    return out


def tc1_lines(tc: dict) -> list[tuple[str, str]]:
    """(label, value) lines describing TC1, every number from its bound DEV and 2026 results."""
    tr, iv = tc["track"], tc["intensity"]
    span = _years(f"{tc['seasons'][0]}-{tc['seasons'][-1]}") if tc.get("seasons") else "--"
    choose = tc.get("choose_seasons") or []
    ld = tc.get("leads") or []
    leads = f"the {len(ld)} leads from {ld[0]} to {ld[-1]} h" if ld else "its leads"

    def diff(x: dict | None, d: int = 2) -> str:
        return f"{_signed((x or {}).get('d'), d)}{_ci((x or {}).get('ci'), d, signed=True)}" if x else "--"

    def vs_ofcl(k: dict, unit: str) -> str:
        v = k["vs_ofcl"]
        return (f"{diff(v)} {unit} at {_level(v.get('level'))}: "
                + ("met" if v["claim"] else "not met; the interval includes zero, so this is a tie"))

    hcca = tr.get("vs_hcca") or {}
    ahead = _tc1_leads_ahead(hcca)
    lines = [
        ("What it is", "Our own track and intensity forecast to five days for every storm the National Hurricane Center "
                       "follows: the real-time computer guidance NHC publishes, each model weighted by how its recent "
                       "forecasts verified, re-measured every six hours; a new model earns weight as it verifies"),
        ("How it was chosen", "Pre-registered: settings chosen on "
                              + (_years(f"{choose[0]}-{choose[-1]}") if choose else "earlier seasons")
                              + f", tested on {span} against NHC&rsquo;s official forecast ({_doc(tc['program'])})"),
        (f"Track, {span}", f"mean error {_f(tr['errors']['TC1'], 2)} n mi over {leads}; NHC official "
                           f"{_f(tr['errors']['OFCL'], 2)}; NHC&rsquo;s corrected consensus HCCA "
                           f"{_f(tr['errors']['HCCA'], 2)}"),
        ("Track against the official forecast (the claim)", vs_ofcl(tr, "n mi")),
        ("Track against HCCA (descriptive, not a claim)",
         f"{diff(hcca)} n mi at {_level(hcca.get('level'))}"
         + (f"; ahead at every lead from {ahead[0]} h" if ahead else "")),
        (f"Intensity, {span}", f"mean error {_f(iv['errors']['TC1'], 2)} kt against NHC official "
                               f"{_f(iv['errors']['OFCL'], 2)}; against the official forecast {vs_ofcl(iv, 'kt')}"),
    ]
    o = tr.get("tc1o_vs_ofcl")
    if o:
        lines.append(("TC1+O, also recorded",
                      "The same weighting with NHC&rsquo;s official forecast as one more member, issued after the "
                      f"advisory: track {_f(tr['errors']['TC1+O'], 2)} n mi, against the official forecast {diff(o)} at "
                      f"{_level(o.get('level'))} (" + ("met" if o.get("claim") else "not met") + "); intensity "
                      f"{_f(iv['errors']['TC1+O'], 2)} kt, {diff(iv.get('tc1o_vs_ofcl'))} ("
                      + ("met" if (iv.get("tc1o_vs_ofcl") or {}).get("claim") else "not met") + ")"))
    s26, i26 = tr.get("season_2026_vs_ofcl"), iv.get("season_2026_vs_ofcl")
    if s26:
        lines.append(("2026 so far (no claim)",
                      f"against NHC&rsquo;s official forecast, on operational best tracks"
                      + (f" ({_n(tc.get('season_2026_storms'))} storms, read {_e(str(tc.get('season_2026_at') or '')[:10])})"
                         if tc.get("season_2026_storms") else "")
                      + f": track {diff(s26)} n mi"
                      + (f", intensity {diff(i26)} kt" if i26 else "") + f" ({_level(s26.get('level'))} intervals)"))
    lines.append(("Status", "Shown beside NHC&rsquo;s official forecast on every NHC storm, never instead of it; NHC is "
                            "the authority. Its live record is scored at each look against NHC&rsquo;s best tracks"))
    return lines


def _tc1_card(tc: dict) -> str:
    return _card("hz-hu", f'Hurricane track and intensity, NHC areas: HazardPulse TC1 <code>{_e(tc["model_version"])}</code>',
                 _facts([_kv(k, v) for k, v in tc1_lines(tc)]), ident="hurricane-track")


def _ours_vs_all_lines(va: dict | None) -> list[tuple[str, str]]:
    """Every public RI probability, then the intensity guidance as yes/no RI calls, each verdict
    read from its interval: a comparison whose interval straddles zero is reported as a tie."""
    if not va:
        return []
    seasons = va.get("seasons") or []
    span = f"{min(seasons)}&ndash;{max(seasons)}" if seasons else "the development seasons"
    out = []
    aids = [a for a in va.get("aids") or [] if a["ours_log_loss"] is not None and a["d_log_loss_ci"]]
    if aids:
        parts = " &middot; ".join(f"{_e(a['label'])} {_f(a['aid_log_loss'])} (ours {_f(a['ours_log_loss'])})"
                                  for a in aids)
        better = [a for a in aids if a["d_log_loss_ci"][1] < 0]
        verdict = ("lower than every one, each paired by storm with its 95% interval below zero"
                   if len(better) == len(aids) else
                   "lower at 95% than " + (", ".join(_e(a["label"]) for a in better) or "none of them"))
        out.append(("Against every RI probability NOAA publishes",
                    f"{span}, as served, each season predicted from earlier ones; 30-kt log loss {parts}: "
                    f"ours is {verdict}"))
    calls = [c for c in va.get("calls") or [] if c["d_pod_ci"] and c["d_pod"] is not None]
    if calls:
        def pts(c):
            return f"{_signed(100 * c['d_pod'], 0)} points{_ci([100 * x for x in c['d_pod_ci']], 0, signed=True)}"
        more = [c for c in calls if c["d_pod_ci"][0] > 0]
        fewer = [c for c in calls if c["d_pod_ci"][1] < 0]
        tied = [c for c in calls if c not in more and c not in fewer]
        bits = []
        if more:
            bits.append("catches more RI events than " + ", ".join(f"{_e(c['label'])} ({pts(c)})" for c in more))
        if fewer:
            bits.append("catches fewer than " + ", ".join(f"{_e(c['label'])} ({pts(c)})" for c in fewer))
        if tied:
            gaps = [100 * c["d_pod"] for c in tied]
            bits.append("is within noise of " + " &middot; ".join(_e(c["label"]) for c in tied)
                        + f" (gaps {_signed(min(gaps), 0)} to {_signed(max(gaps), 0)} points)")
        out.append(("Against intensity forecasts as yes/no RI calls",
                    "at each call&rsquo;s own false-alarm rate (a forecast 24-h rise of 30 kt or more), ours "
                    + "; and ".join(bits)))
    return out


def _ours_challenger_lines(ch: dict | None) -> list[tuple[str, str]]:
    """The next model, running in shadow beside the shown one: its development numbers against the
    shown model's, whether its interval yet excludes zero, and how it can replace it."""
    if not ch:
        return []
    d = ch["dev"]
    ci = d.get("d_log_loss_ci")
    settled = bool(ci and ci[1] < 0)
    vs = ch.get("versus_champion") or {}
    live = (f"; live so far, {_n(vs['n'])} cycles both scored, four-threshold Brier difference "
            f"{_signed(vs['d_brier4'])}{_ci(vs.get('d_brier4_ci'), signed=True)}" if vs.get("n") else "")
    level = (" at " + f"{100 * ch['level']:.4f}".rstrip("0").rstrip(".") + "%") if ch.get("level") else ""
    against = _e(ch.get("against") or "the shown model")
    s26 = ch.get("season_2026") or {}
    y26 = (f"; 2026 (a later look at a season that informed the design, not a proof): {_f(s26['log_loss'], 4)} vs "
           f"{_f(s26['champion_log_loss'], 4)}{_ci(s26.get('d_log_loss_ci'), 4, signed=True)}"
           if s26.get("log_loss") is not None and s26.get("champion_log_loss") is not None else "")
    what = ch.get("what") or ("the same inputs, never lowering the odds when the guidance or "
                              "NOAA&rsquo;s probability rises")
    level_txt = (f" (confidence level {level.replace(' at ', '')})" if level else "")
    return [(f"Variant under test ({_e(ch.get('label') or '')})".replace(" ()", ""),
             f"<code>{_e(ch['model_version'])}</code>: {what}. {_years(ch.get('dev_period'))} against {against}: log loss "
             f"{_f(d['log_loss'], 4)} vs {_f(d['champion_log_loss'], 4)}{_ci(ci, 4, signed=True)}, four-threshold "
             f"Brier {_f(d['brier4'], 4)} vs {_f(d['champion_brier4'], 4)}; "
             + ("better at 95%" if settled else "a gain whose interval still includes zero")
             + f"{y26}. Tested live under its own rule, written before it scored a forecast{level_txt}{live}; it "
             "replaces the shown model only if it meets that rule")]


def ours_hurricane_lines(ours: dict) -> list[tuple[str, str]]:
    """(label, value) lines describing our hurricane RI model, from its bound evidence."""
    d, s26, pr = ours["dev"], ours["season_2026"], ours["prospective"]
    mt = d.get("multi_threshold") or {}
    lines = [
        ("What it is", "Our own model: one exceedance curve P(a rise of k kt in 24 h), k = 15&ndash;45, from the "
                       "early intensity guidance (NOAA&rsquo;s statistical models, the hurricane and global models, "
                       "NHC&rsquo;s consensus aids), NOAA&rsquo;s RI probabilities at six thresholds and the official "
                       "forecast" + (f"; {ours['adds']}" if ours.get("adds") else "")
                       + "; NOAA&rsquo;s DTOPS wherever that guidance is missing"),
        (f"{_years(ours.get('dev_period'))}, each season forecast only from earlier ones",
         f"log loss {_f(d['log_loss'])}"
         + (f" vs DTOPS {_f(d['dtops_log_loss'])} (paired by storm{_ci(d.get('d_log_loss_ci'), signed=True)})"
            if d.get("dtops_log_loss") is not None else "")
         + (f"; over DTOPS&rsquo;s four 24-h thresholds, Brier {_f(mt.get('ours'))} vs "
            f"{_f(mt.get('dtops'))}{_ci(mt.get('d_ci'), signed=True)}" if mt else "")),
        *_ours_vs_all_lines(ours.get("vs_all")),
        ("2026 so far", f"log loss {_f(s26['log_loss'])}"
                        + (f" vs DTOPS {_f(s26['dtops_log_loss'])}" if s26.get("dtops_log_loss") is not None else "")
                        + (f", AUC {_f(s26['auc'])}" if s26.get("auc") is not None else "")
                        + " (a later look at a season that informed its design, so not a proof)"),
        *[line for ch in (ours.get("challengers") or ([ours["challenger"]] if ours.get("challenger") else []))
          for line in _ours_challenger_lines(ch)],
        ("Status", f"{_e(ours['status'])}: shown beside NOAA&rsquo;s number, not instead of it, until the "
                   "pre-registered prospective test on cycles from "
                   + _e(str(pr.get("start") or "")[:10] or "its start")
                   + (f" meets its rule at {_level(ours.get('level'))} (verdicts " if ours.get("level") else
                      " meets its rule (verdicts ")
                   + " and ".join(_e(x) for x in pr.get("look_dates") or [])
                   + f"); cycles scored so far: {_n(pr.get('matured_and_scored', 0))}"
                   + (f". Which of our models is shown is named in one file, {_doc(ours['pointer'])}"
                      if ours.get("pointer") else "")),
    ]
    return lines


def _ours_hurricane_card(ours: dict) -> str:
    rows = [_kv(k, v) for k, v in ours_hurricane_lines(ours)]
    return _card("hz-hu", f'Hurricane, our model (experimental): {_e(ours["label"])} '
                          f'<code>{_e(ours["model_version"])}</code>', _facts(rows), ident="hurricane-ours")


def _md(text: str) -> str:
    """A passage quoted from a program document: escaped, with its `code` and **bold** kept as markup."""
    out = _e(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)


def _d(p: dict, d: int = 4, what: str = "log loss") -> str:
    """A paired difference and its interval: ``log loss -0.0076 [-0.0147, -0.0003]``. An interval of zero width
    (one storm: every resample is the same) informs nothing and is left out."""
    key = "d_ll" if what == "log loss" else "d_brier"
    ci = p.get(key + "_ci")
    return f"{what} {_signed(p.get(key), d)}{_ci(ci, d, signed=True) if ci and ci[0] != ci[1] else ''}"


def _events(n: Any, cycles: Any = None) -> str:
    """``1 RI event in 391 cycles``, ``0 RI events``."""
    word = "RI event" if _n(n) == "1" else "RI events"
    return f"{_n(n)} {word}" + (f" in {_n(cycles)} cycles" if cycles is not None else "")


def _worse_than(r: dict, who: str, v82: str, period: str) -> str:
    """How a region's log loss compared, said as strongly as its interval allows."""
    ci = r.get("d_ll_ci")
    if r.get("d_ll") is None:
        return f"{who} was not scored against {v82}"
    if r["d_ll"] > 0:
        return (f"{who} tested worse than {v82} on {period}, beyond its interval" if ci and ci[0] > 0 else
                f"{who}&rsquo;s log loss was above {v82}&rsquo;s on {period}, with an interval that includes zero")
    return f"{who} was not worse than {v82} on {period}"


def j1_lines(j1: dict) -> list[tuple[str, str]]:
    """(label, value) lines describing our satellite model for the JTWC basins, every number read from its bound
    results (served_evidence.j1_hurricane): the registered test, what the satellite adds, how narrow the pooled
    interval is (quoted from the program), each region apart, the further read, where it is shown and why, and the
    rule that would make it the published number."""
    lab, v82 = _e(j1["label"]), _e(j1.get("against") or "")
    # the number published beside J1 (v8.3 since amendment 15); J1's comparator stays v82
    pub = _e(j1.get("published") or j1.get("against") or "")
    beside = (f"the published {pub}, never instead of it; it is built on {v82} and tested against it"
              if pub != v82 else f"{v82}, never instead of it")
    inp, dev, period = j1["inputs"], j1["dev"], _years(j1.get("dev_period"))
    level = _level(j1.get("level"))
    model, bar, control = dev["model"], dev["bar"], dev["control"]
    regions = j1.get("regions") or []
    names = {r["region"]: _e(r["name"]) for r in regions}
    in_scope = [names[g] for g in j1.get("scope") or [] if g in names]
    out_scope = [r for r in regions if not r["in_scope"]]
    lines = [
        ("What it is",
         f"Our own rapid-intensification model for the {_join([_e(x) for x in j1['live_basin_names']])} basins, where "
         f"NOAA publishes no such guidance: gradient-boosted trees ({_n(j1.get('seeds'))} seeds, averaged) on {v82}&rsquo;s "
         f"own probability, its {_n(inp['against'])} inputs, {_n(inp['ir'])} measures of the storm&rsquo;s cloud tops "
         f"from NOAA&rsquo;s GMGSI satellite infrared ({_join([_n(h) for h in inp['ir_hours_after']])} hours after the "
         f"cycle and {_join([_n(h) for h in inp['ir_hours_before']])} hours before) and the basin; trained on "
         f"{_years(j1.get('trained'))}. It is recorded beside {beside}"),
    ]
    rows = [f'<tr><td>{name}</td><td class="num">{_n(m["n"])}</td><td class="num">{_n(m["events"])}</td>'
            f'<td class="num">{_f(m["log_loss"], 5)}</td><td class="num">{_f(m["brier"], 5)}</td>'
            f'<td class="num">{_f(m["auc"])}</td></tr>'
            for name, m in ((f"{v82} as published (the bar)", bar),
                            (f"{lab} without the satellite inputs (a control)", control),
                            (f"<strong>{lab}</strong>", model))]
    vb = j1["vs_bar"]
    both = vb.get("d_ll") is not None and vb["d_ll"] < 0 and vb.get("d_brier") is not None and vb["d_brier"] < 0
    lines.append((
        f"Registered test, {period} (a hindcast)",
        f"Every cycle of the {_join([names[r['region']] for r in regions]) or 'JTWC'} basins, pooled, each season "
        "forecast from earlier ones only, with the satellite crops centred on post-season best-track positions."
        + _table(["", "Cycles", "RI events", "Log loss", "Brier", "AUC"], rows,
                 caption=f"{lab} against {v82}, {period} (lower log loss and Brier are better)")
        + f"{lab} minus {v82}: {_d(vb, 5)}, {_d(vb, 5, 'Brier')} ({level} intervals by storm; below zero is better)"
        + (f". Both point estimates are below {v82}&rsquo;s, the rule written before the result, so {lab} was "
           "carried on this pooled hindcast; the claim is not made basin by basin, nor for live forecasts"
           if both else "")))
    vc, noise = j1["vs_control"], [x["d_ll"] for x in j1.get("noise") or [] if x.get("d_ll") is not None]
    relearn = (control["log_loss"] is not None and bar["log_loss"] is not None)
    lines.append((
        "What the satellite adds",
        f"{lab} minus the same model without the satellite inputs: {_d(vc, 5)}"
        + (f". The control alone scored {_f(control['log_loss'], 5)} against {v82}&rsquo;s {_f(bar['log_loss'], 5)}: "
           + (f"re-learning {v82}&rsquo;s inputs does not help, the gain is the satellite&rsquo;s"
              if control["log_loss"] >= bar["log_loss"] else "re-learning its inputs helps too")
           if relearn else "")
        + (f". With the {_n(inp['ir'])} satellite columns shuffled within each season (a noise control, "
           f"{_n(len(noise))} seeds), the same comparison gave {_signed(min(noise), 5)} to {_signed(max(noise), 5)}"
           if noise else "")))
    fr = j1.get("fragility") or {}
    if fr.get("text") or fr.get("bullets"):
        lines.append((
            "How narrow the pooled interval is",
            f"From the program{(' (amendment ' + _e(fr['amendment']) + ')') if fr.get('amendment') else ''}, after "
            f"an independent pass that did not build {lab}: &ldquo;{_md(fr.get('text') or '')}&rdquo;"
            + ("<ul>" + "".join(f"<li>{_md(b)}</li>" for b in fr.get("bullets") or []) + "</ul>"
               if fr.get("bullets") else "")
            + f" ({_doc(fr['doc'])})"))
    if regions:
        def diff(r: dict, key: str) -> str:
            ci = r.get(key + "_ci")
            return f"{_signed(r.get(key), 4)}{_ci(ci, 4, signed=True) if ci and ci[0] != ci[1] else ''}"
        codes = {r["region"] for r in regions}
        outcome = list(j1.get("outcome_basins") or [])
        merged = [b for b in outcome if b not in codes]
        split = [_e(r["name"]) for r in regions if r["region"] not in outcome]
        lines.append((
            f"Each region, {period}",
            (f"The registered outcome&rsquo;s table merged the {_join(split)} into one "
             f"{_join([_e(se.HURRICANE_BASIN_NAMES.get(b, b)) for b in merged])} row; here each region is apart."
             if merged and split else "")
            + _table(["Region", "Cycles", "RI events", "Log loss", "Brier", "Shown"],
                     [f'<tr><td>{_e(r["name"])}</td><td class="num">{_n(r["n"])}</td>'
                      f'<td class="num">{_n(r["events"])}</td><td class="num">{diff(r, "d_ll")}</td>'
                      f'<td class="num">{diff(r, "d_brier")}</td><td>{"yes" if r["in_scope"] else "no"}</td></tr>'
                      for r in regions],
                     caption=f"{lab} minus {v82} by region, {period} ({level} intervals by storm; below zero is "
                             "better)")))
    fu = j1.get("further")
    if fu:
        fm, fb = fu["model"], fu["bar"]
        lines.append((
            f"{_e(fu['season'])} so far (a further read, no claim)",
            f"{_n(fm['n'])} cycles, {_n(fm['events'])} RI events, every JTWC region pooled: log loss "
            f"{_f(fm['log_loss'], 4)} against {v82}&rsquo;s {_f(fb['log_loss'], 4)} (difference "
            f"{_signed(fu.get('d_ll'), 4)}{_ci(fu.get('d_ll_ci'), 4, signed=True)}), "
            f"Brier {_f(fm['brier'], 4)} against {_f(fb['brier'], 4)}, AUC {_f(fm['auc'])} against {_f(fb['auc'])}"))
    var = j1.get("variant") or {}
    scope_why = []
    variant_name = (f"{_e(var['label'])} (a variant of {lab} registered in amendment "
                    f"{_e(_amendment_of(var.get('program')))}, {'carried' if var.get('carried') else 'not carried'})"
                    if var else "")
    for r in out_scope:
        v26 = next((x for x in var.get("season_2026") or [] if x["region"] == r["region"]), None)
        text = (f"<strong>{_e(r['name'])}</strong>: {_worse_than(r, lab, v82, period)} "
                f"({_events(r['events'], r['n'])}), so it is not shown there")
        if v26 and v26.get("d_ll") is not None:
            text += (f"; on {_e(var.get('season'))}, {variant_name} scored {_d(v26)} against {v82} there "
                     f"({_events(v26['events'], v26['n'])})"
                     + (": the rule excludes it although that season points the other way" if v26["d_ll"] <= 0
                        else ", the same direction"))
        scope_why.append(text + ".")
    lines.append((
        "Where it is shown",
        (f"{_join(in_scope)} storms only" if in_scope else "No storm")
        + f", by the rule written before the result (amendment {_e(j1.get('scope_amendment') or '')}): the regions "
        f"where its {period} log loss against {v82} is at or below zero."
        + "".join(f" {t}" for t in scope_why)
        + f" {lab} is still recorded on every {_join([_e(x) for x in j1['live_basin_names']])} storm, so the record "
        f"stays complete, but it is shown, and its claim is judged, only in its scope ({_doc(j1['scope_file'])})"
        if j1.get("scope_file") else "No storm: its scope file is not in this build"))
    for r in regions:
        if r["events"] == 0:
            higher = (r.get("mean_forecast") is not None and r.get("bar_mean_forecast") is not None
                      and r["mean_forecast"] > r["bar_mean_forecast"])
            lines.append((
                f"Untested: {_e(r['name'])}",
                f"The {_e(r['name'])} basin had no RI event in the test seasons ({_n(r['n'])} cycles), so {lab} is untested "
                "there on RI events" + (f"; where nothing happened it forecast higher than {v82} ({_d(r, 4, 'Brier')})"
                                        if higher else "")))
    pr, rule = j1["prospective"], j1.get("rule") or {}
    run = pr.get("running")
    lines.append((
        "Status",
        ("Claim met at a look" if pr.get("claimed") else f"In test beside {pub}")
        + (f": never the published number. Its test against {v82} as published was written before its first live "
           if pub == v82 else
           f": never the published number. Its test against {v82} (published until {pub} replaced it; its "
           "comparator is unchanged) was written before its first live ")
        + f"cycle ({_doc(rule.get('scorer') or '')}): at each look ("
        + " and ".join(_e(x) for x in rule.get("looks") or []) + "), on live cycles in its scope only, the claim is "
        f"met if {lab} minus {v82} has a {_level(rule.get('level'))} storm-bootstrap interval wholly below zero on log "
        "loss or on Brier, with both point estimates at or below zero; a met claim makes "
        f"{lab} the published number. The hindcast above is not evidence for that. "
        + (f"Live so far, in its scope (descriptive, no claim before a look): {_n(run['n'])} cycles from "
           f"{_n(run.get('storms'))} storms ({_n(run.get('events'))} RI events), log loss {_f(run['log_loss'], 3)} "
           f"against {_f(run['bar_log_loss'], 3)}, difference {_signed(run.get('d_ll'), 3)}"
           f"{_ci(run.get('d_ll_ci'), 3, signed=True)} at {_level(run.get('level'))}"
           if run else
           (f"{_n(pr.get('records'))} live records carry it; none in its scope has matured yet"
            if pr.get("exists") else "The live record starts with its first matured cycle"))))
    return lines


def _amendment_of(program: Any) -> str:
    m = re.search(r"amendments? ([\w, ]+)\)", str(program or ""))
    return m.group(1) if m else "--"


def _j1_card(j1: dict) -> str:
    return _card("hz-hu", f'Hurricane, other basins, our satellite model (in test): {_e(j1["label"])} '
                          f'<code>{_e(j1["model_version"])}</code>',
                 _facts([_kv(k, v) for k, v in j1_lines(j1)]), ident="hurricane-j1")


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


def format_change_date(to: dict | None) -> str | None:
    """The day NOAA switched the ProbSevere format, as the input guard records it: ``5 August 2025``."""
    d = ((to or {}).get("format_change") or {}).get("date")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(d or ""))
    if not m:
        return None
    months = ("January February March April May June July August September October November December").split()
    return f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)}"


def format_change_sentence(to: dict | None) -> str:
    """What NOAA's 2025 ProbSevere format change cost the published model, as the registered test measured it
    (tornado program amendment 10, ``results/tornado_program/t2_2026.json``)."""
    fc = (to or {}).get("format_change")
    if not fc or fc.get("auc") is None:
        return ""
    period = fc.get("period") or []
    s = ("A test written down in advance scored the published model on every storm observation "
         + (f"from {_e(period[0])} to {_e(period[1])} " if len(period) == 2 else "")
         + f"({_n(fc['n'])} of them, {_n(fc['pos'])} followed by a tornado), all in the new format: AUC "
         f"{_f(fc['auc'], 4)}, "
         f"against {_f(fc['auc_2025'], 4)} on its 2025 test, so no loss of ranking is visible (2026 is scored on "
         "SPC&rsquo;s preliminary reports)")
    if fc.get("without_inputs_d_auc") is not None:
        s += (f". A model built without the inputs the feed lost did not rank better "
              f"({_signed(fc['without_inputs_d_auc'], 4)}{_ci(fc.get('without_inputs_d_auc_ci'), 4, signed=True)} AUC, "
              "paired by day)")
    s += ", so the published model stays" if fc.get("served_stays") else ", so it replaced the published model"
    if fc.get("mean_over_base") is not None:
        s += (f". It may cost calibration: over that period the model&rsquo;s average forecast was "
              f"{_f(fc['mean_over_base'], 2)} times the rate at which tornadoes followed. A recalibration for the new "
              "format is an open question, to be judged on forecasts made after it is registered")
    return s + f" ({_doc(fc['program'])}, amendment {_e(fc.get('amendment'))}; {_doc(fc['file'])})."


def methods_tornado(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return _card("hz-to", "Tornado", "<p>No final test is bound to the served tornado model.</p>")
    t = to["test"]
    products = [_PRODUCT_NAMES[k] for k in ("p30", "p90", "p_ef2") if k in (to.get("products") or {})]
    event = _e(to["event"]).replace("THIS", "this")
    rows = [
        _kv("Target", event[0].upper() + event[1:]
            + (f" (also served: {', '.join(_e(p.lower()) for p in products)})" if products else "")),
        _kv("Architecture", f"LightGBM ({_n(to['n_trees'])} trees), exported to a JSON payload scored in pure NumPy; "
                            "Platt calibration on leave-one-year-out scores"
                            + ("; Venn&ndash;Abers band per storm" if to.get("has_band") else "")),
        _kv(f"Inputs ({to['n_inputs']})", _tornado_inputs_text(to)),
        _kv("How it was chosen", f"Pre-registered programme: every choice on 2023; development test 2024; refit "
                                 f"{_years(to.get('trained_years'))}; final test 2025 read once ({_doc(to['program'])})"),
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
    fc = format_change_sentence(to)
    if fc:
        rows.append(_kv("NOAA&rsquo;s 2025 format change", fc))
    for nu in to.get("tested_not_served") or []:
        if "coherence" in nu["what"] and "PDE" not in nu["what"]:
            rows.append(_kv("Coherence field", f"Tested on top of the HRRR fields: {_signed(nu['delta_auc'], 4)}"
                                               f"{_ci(nu['delta_auc_ci'], 4, signed=True)} validation AUC (paired); "
                                               "no lift, so not served"))
    rows.append(_kv("Status", "Published"))
    return _card("hz-to", f'Tornado, storm by storm <code>{_e(to["model_version"])}</code>', _facts(rows),
                 ident="tornado-model")


def methods_data_hurricane(ev: dict) -> str:
    """The Data card's row for our hurricane models: which of them read satellite infrared, from each artifact's
    own inputs (it said "the v10.3 variant" by hand while v10.4 read the same images)."""
    hu = ev.get("hurricane") or {}
    ir = [_e(x) for x in hu.get("ir_models") or []]
    j1 = hu.get("j1") or {}
    v82 = _e(j1.get("against") or "")
    return ("<div><dt>Our models</dt><dd>The early intensity guidance in NHC&rsquo;s forecast files"
            + (f"; for {_join(ir)}, also NOAA&rsquo;s GMGSI satellite infrared" if ir else "")
            + (f"; for {_e(j1['label'])}, in the other basins, {v82}&rsquo;s own probability and inputs with the same "
               "GMGSI infrared" if j1 and (j1.get("inputs") or {}).get("ir") and v82 else "")
            + ("; for TC1, the track and intensity guidance in NHC&rsquo;s real-time forecast files" if hu.get("tc1")
               else "") + "</dd></div>")


def _lat(x: Any) -> str:
    v = float(x)
    return f"{abs(v):g}&deg;{'S' if v < 0 else 'N'}"


def methods_limits(ev: dict) -> str:
    """The Limits list's items that state what a model was tested on or fed, each read from its results: coverage
    (until 2026-10-09 a hand-typed sentence here said the other-basins model had no test in the basins where it
    publishes, while its test held cycles from every basin -- CONTRADICTED below), and what the inputs carry that
    the models were not built for."""
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    grid = (eq or {}).get("grid") or {}
    ob = (hu or {}).get("other_basins") or {}
    v82 = _e(ob.get("label") or "")
    parts = []
    if grid.get("lat_min") is not None and grid.get("lat_max") is not None and grid.get("dlat"):
        parts.append(f"Earthquake: the globe between {_lat(grid['lat_min'])} and {_lat(grid['lat_max'])} on a "
                     f"{float(grid['dlat']):g}&deg; grid.")
    if hu:
        o = ob.get("test") or {}
        comp = ob.get("composition") or {}
        basins = ", ".join(f"{_e(b['name'])} {_n(b['n'])}" for b in comp.get("by_basin") or [])
        parts.append("Hurricane: every active tropical cyclone, with NOAA&rsquo;s guidance in the Atlantic, East and "
                     "Central Pacific" + (f" and our {v82} model elsewhere" if v82 else "") + "."
                     + ((f" {v82} was tested on {_n(o.get('n'))} {_when(o)} cycles in the basins where it publishes"
                         + (f" ({basins})" if basins else "") + ", none used to fit it, with best-track inputs."
                         if ob.get("subject") == "model" else
                         f" {v82}&rsquo;s method was tested on {_n(o.get('n'))} held-out {_when(o)} cycles from every "
                         f"basin" + (f" ({basins})" if basins else "") + ", with best-track inputs"
                         + ("; the published model&rsquo;s calibration was fitted on those same cycles, so that is not "
                            "a test of it on unseen data" if comp.get("served_calibration_fitted_on_test_cases") else "")
                         + ".") if v82 and o.get("auc") is not None else ""))
    parts.append("Tornado: every storm NOAA&rsquo;s radar tracks over and near the contiguous US, including northern "
                 "Mexico and the Gulf; tornado reports, and so the tests, cover the US only.")
    items = ["<li><strong>Coverage.</strong> " + " ".join(parts) + "</li>"]
    eqs = earthquake_inputs_sentence(eq)
    if eqs:
        items.append(f"<li><strong>Earthquake inputs.</strong> {eqs}.</li>")
    day = format_change_date(to)
    fcs = format_change_sentence(to)
    if day and fcs:
        items.append(f"<li><strong>Tornado inputs.</strong> NOAA changed the format of its ProbSevere feed on {day}, "
                     f"and some inputs the model learned from are no longer in it. {fcs}</li>")
    return "\n".join(items)


_RESEARCH_WHAT = {
    "h8": "The coherence equation solved for the convective heating the vortex can hold inside its local Rossby "
          "radius, from the satellite infrared and NHC&rsquo;s radius of maximum wind",
    "h9": "A memory of each storm&rsquo;s recent convection, symmetry, heating retention and intensity trend "
          "(its &ldquo;coherence state&rdquo;)",
    "fusion": "Every RI probability, ours and NOAA&rsquo;s, weighted by how each has verified, with old verifications "
              "forgotten",
    "g1": "The inner core seen hourly at 2 km by GOES: core and ring convection, the eye, the eyewall&rsquo;s edge, "
          "symmetry, persistence and their trends",
    "j1": "Satellite infrared (the storm&rsquo;s cloud tops from NOAA GMGSI) for the West Pacific, North Indian and "
          "Southern Hemisphere, where NOAA publishes no RI guidance and we publish our own model",
    "j2": "Separate South Indian and South Pacific basins in that satellite model, after an independent pass found it "
          "worse than the published model in the South Pacific",
}


def _research_rows(ev: dict) -> list[str]:
    rows = []
    for r in ev.get("research") or []:
        src = _doc(r["file"])
        amend = (f" ({'amendments' if ',' in str(r['amendment']) else 'amendment'} {_e(r['amendment'])})"
                 if r.get("amendment") else "")
        if r["kind"] == "candidate":
            nf = f"{_n(r['n_features'])} features: " if r.get("n_features") else ""
            res = (f"against {_e(r.get('control_label') or r.get('control'))} on {_years(r.get('dev_period'))} "
                   f"({_n(r.get('n'))} cycles, {_n(r.get('events'))} RI events): log loss {_signed(r['d_ll'], 5)}"
                   f"{_ci(r['d_ll_ci'], 5, signed=True)}, four-threshold Brier {_signed(r['d_brier4'], 5)}"
                   f"{_ci(r['d_brier4_ci'], 5, signed=True)} (below zero is better)")
            if r.get("noise"):
                res += (f"; the same columns shuffled within each season (a noise control) gave log loss "
                        f"{_signed(min(r['noise']), 5)} to {_signed(max(r['noise']), 5)}")
            if r["carried"]:
                dec = ("Carried" + (f": it is {_e(r['became'])}, in shadow" if r.get("became") else "")
                       + " (both point estimates below the model it was tested against)")
            else:
                worse = [name for name, d, ci in (("log loss", r["d_ll"], r.get("d_ll_ci")),
                                                  ("Brier", r["d_brier4"], r.get("d_brier4_ci")))
                         if d is not None and d > 0]
                beyond = [name for name, ci in (("log loss", r.get("d_ll_ci")), ("Brier", r.get("d_brier4_ci")))
                          if ci and ci[0] > 0]
                def which(names: list[str]) -> str:
                    return "both measures" if len(names) == 2 else names[0]
                dec = ("Not carried: " + (f"worse on {which(worse)}" if worse else "not better on both measures")
                       + (f", beyond its interval on {which(beyond)}" if beyond else ""))
            rows.append(f"<tr><td>{_RESEARCH_WHAT.get(r['key'], _e(r['key']))}{amend}</td><td>{nf}{res}</td>"
                        f"<td>{dec}</td><td>{src}</td></tr>")
        elif r["kind"] == "fusion":
            res = (f"{_n(r.get('n_sources'))} sources; the fusion minus the best single source on the choosing season "
                   f"({_e(r.get('best_single_label') or r.get('best_single'))}) on {_years(r.get('dev_period'))}, "
                   f"{_n(r.get('n'))} cycles: log loss {_signed(r['d_ll'], 4)}{_ci(r['d_ll_ci'], 4, signed=True)}")
            dec = "Claim met" if r.get("claim") else "Claim not met: the interval includes zero"
            rows.append(f"<tr><td>{_RESEARCH_WHAT['fusion']}{amend}</td><td>{res}</td><td>{dec}</td><td>{src}</td></tr>")
        elif r["kind"] == "input_check":
            res = (f"the 2 km eye detector against CIMSS&rsquo;s Advanced Dvorak Technique on {_n(r.get('matched'))} "
                   f"matched images: skill (HSS) {_f(r.get('hss'))} on the held-out {_years(r.get('held_out'))} images "
                   f"(required {_f(r.get('hss_gate'), 2)}); an eye called in {_pct(r.get('weak_eye'))} of images "
                   f"below {_n(r.get('weak_kt'))} kt (allowed {_pct(r.get('weak_eye_gate'))}); centre "
                   f"{_f(r.get('centre_km'), 1)} km from ADT&rsquo;s (median); eye temperature rank correlation "
                   f"{_f(r.get('teye_spearman'))}")
            dec = ("Passed every gate, so the satellite input was sound before any outcome was read"
                   if r.get("passed") else "Failed a gate")
            rows.append(f"<tr><td>Input check for the 2 km GOES test: is the eye where ADT says it is?</td>"
                        f"<td>{res}</td><td>{dec}</td><td>{src}</td></tr>")
        elif r["kind"] == "vs_published":
            ctl = _e(r.get("control_label") or "the published model")
            nf = f"{_n(r['n_features'])} features: " if r.get("n_features") else ""
            res = (f"{nf}against {ctl} as published on {_years(r.get('dev_period'))}, every JTWC cycle pooled "
                   f"({_n(r.get('n'))} cycles, {_n(r.get('events'))} RI events; satellite crops on best-track "
                   f"positions): {_d(r, 5)}, {_d(r, 5, 'Brier')} (below zero is better); what the "
                   f"satellite adds, against the same model without it: {_d(r.get('ir') or {}, 5)}")
            if r.get("noise"):
                res += (f"; the satellite columns shuffled within each season (a noise control) gave log loss "
                        f"{_signed(min(r['noise']), 5)} to {_signed(max(r['noise']), 5)}")
            if r["carried"]:
                dec = ("Carried on the pooled hindcast" + (f": it is {_e(r['became'])}, in test beside {ctl}"
                                                           if r.get("became") else "")
                       + (f", shown only for {_join([_e(x) for x in r['scope']])} storms" if r.get("scope") else "")
                       + " (both point estimates below the model it was tested against)")
            else:
                dec = "Not carried: not better on both measures"
            rows.append(f"<tr><td>{_RESEARCH_WHAT['j1']}{amend}</td><td>{res}</td><td>{dec}</td><td>{src}</td></tr>")
        elif r["kind"] == "variant":
            inc = _e(r.get("incumbent") or "")
            res = (f"{_e(r.get('candidate'))} minus {inc} on {_e(r.get('season'))} ({_n(r.get('n'))} cycles, "
                   f"{_n(r.get('events'))} RI events): {_d(r, 5)}, {_d(r, 5, 'Brier')} (below zero is better)")
            why = (f"its log loss is not below {inc}&rsquo;s" if r.get("d_ll") is not None and r["d_ll"] >= 0 else
                   f"its Brier score is not below {inc}&rsquo;s" if r.get("d_brier") is not None and r["d_brier"] >= 0
                   else f"it did not meet its rule against {inc}")
            dec = (f"Carried: it replaces {inc}" if r["carried"] else
                   f"Not carried: {why}"
                   + (f"; {_e(r['live_entrant'])} stays the live entrant" if r.get("live_entrant") else "")
                   + (f", in its scope ({_join([_e(x) for x in r['scope']])})" if r.get("scope") else ""))
            rows.append(f"<tr><td>{_RESEARCH_WHAT['j2']}{amend}</td><td>{res}</td><td>{dec}</td><td>{src}</td></tr>")
    to_fc = (ev.get("tornado") or {}).get("format_change") or {}
    if to_fc.get("without_inputs_d_auc") is not None:
        period = to_fc.get("period") or []
        rows.append("<tr><td>Tornado: a model without the inputs NOAA&rsquo;s 2025 format change removed "
                    f"(amendment {_e(to_fc.get('amendment'))})</td><td>AUC {_signed(to_fc['without_inputs_d_auc'], 4)}"
                    f"{_ci(to_fc.get('without_inputs_d_auc_ci'), 4, signed=True)} against the published model, "
                    + (f"{_e(period[0])} to {_e(period[1])}" if len(period) == 2 else "") + ", paired by day</td><td>"
                    + ("Not carried: the published model stays" if to_fc.get("served_stays") else "Carried")
                    + f"</td><td>{_doc(to_fc['file'])}</td></tr>")
    it = (ev.get("earthquake") or {}).get("input_types") or {}
    if it.get("dev_d_ig") is not None:
        rows.append("<tr><td>Earthquake: only earthquakes in the inputs (" + _e(it.get("program_section")) + ")</td>"
                    f"<td>information {_signed(it['dev_d_ig'], 4)}{_ci(it.get('dev_d_ig_ci'), 4, signed=True)} "
                    f"{EQ_IG_UNIT}" + (f" on {_years(it['dev_years'])}" if it.get("dev_years") else "")
                    + f"; adopted unless the interval reached below {_signed(it.get('margin'), 3)}</td><td>"
                    + ("Adopted" if it.get("adopted") else "Not adopted: the interval reached below the margin")
                    + f"</td><td>{_doc(it['file'])}</td></tr>")
    return rows


def methods_research(ev: dict) -> str:
    """Every registered test since the published models were chosen, the ones that changed nothing included,
    each row read from the file its run wrote."""
    rows = _research_rows(ev)
    if not rows:
        return "<p>No research results are bound in this build.</p>"
    return _table(["What was tested", "Result", "Decision", "Results file"], rows,
                  caption="Registered tests, decided by rules written before their results existed")


# ---------------------------------------------------------------------------
# registry page
# ---------------------------------------------------------------------------

def registry_simple(ev: dict) -> str:
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    cards = []
    if eq:
        t = eq["test"]
        how = "a second look at those years" if t.get("second_read") else "scored once"
        act = (eq.get("gear1") or {}).get("activity")
        cards.append(('hz-eq', (f"Earthquake: {_e(eq['candidate'])} (C0 + GEAR1"
                                + (", weighted by activity)" if act else ")")) if eq.get("gear1") else "Earthquake",
                      "Combines where large earthquakes happen in the long run with how they cluster after recent "
                      + ("ones, plus GEAR1&rsquo;s long-term rate from crustal strain"
                         + (", counted most where recent earthquakes are few" if act else "") + ". "
                         if eq.get("gear1") else "ones. ")
                      + f"On {_when(t)} ({how}), it ranked a cell that went on to have an M6+ earthquake above one "
                      f"that did not {_pct(t['auc']['value'], 0)} of the time across the globe, and "
                      f"{_pct(t['auc_active_cells']['value'], 0)} among cells with recent earthquakes, the harder question."))
    if hu:
        t = hu["test"]
        beaten = [_e(c["against"]) for c in hu.get("claims", []) if c["better"]]
        v82 = _e((hu.get("other_basins") or {}).get("label") or "")
        ours, tc1 = hu.get("ours"), hu.get("tc1")
        cards.append(('hz-hu', "Hurricane: NOAA DTOPS" + (f", and {v82} elsewhere" if v82 else ""),
                      "For storms the National Hurricane Center covers, we publish NOAA&rsquo;s own DTOPS guidance: in a "
                      f"test written down in advance and run once on the {_when(t)} season it "
                      + (f"beat {_join(beaten)}, " if beaten else "")
                      + "ranking a forecast cycle that went on to intensify rapidly above one that did not "
                      f"{_pct(t['auc'], 0)} of the time."
                      + (f" Elsewhere we publish our {v82} model." if v82 else "")
                      + (f" Beside it, our satellite model {_e(hu['j1']['label'])} is recorded and, for "
                         + _join([_e(r["name"]) for r in hu["j1"].get("regions") or [] if r.get("in_scope")])
                         + " storms, shown while it is tested on new forecasts."
                         if v82 and hu.get("j1") and any(r.get("in_scope") for r in hu["j1"].get("regions") or [])
                         else "")
                      + (f" Our {_e(ours['label'])} model is shown beside DTOPS, labelled experimental, while it is "
                         "tested on new forecasts." if ours else "")
                      + (" Our track and intensity forecast, TC1, is shown beside NHC&rsquo;s official forecast."
                         if tc1 else "")))
    if to:
        t = to["test"]
        cards.append(('hz-to', "Tornado: v3",
                      "Scores every thunderstorm NOAA tracks over the contiguous US for the chance it produces a tornado "
                      "in the next hour. On every storm of 2025, scored once, it ranked a storm that went on to produce "
                      f"a tornado above one that did not {_pct(t['auc'], 0)} of the time."))
    return "\n".join(_card(cls, title, f"<p>{text}</p>") for cls, title, text in cards)


_STATUS_CHIP = {"published": "good", "experimental": "warn", "fallback": "neutral", "shadow": "neutral"}


def _registry_row(model: str, hazard: str, status: str, test: str, auc: str, skill: str, arch: str) -> str:
    return (f"<tr><td><code>{_e(model)}</code></td><td>{hazard}</td>"
            f'<td><span class="chip {_STATUS_CHIP.get(status, "neutral")}">{status}</span></td><td>{test}</td>'
            f'<td class="num">{auc}</td><td class="num">{skill}</td><td>{arch}</td></tr>')


def registry_active(ev: dict) -> str:
    rows = []
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    if eq:
        t = eq["test"]
        rows.append(_registry_row(eq["model_version"], "Earthquake (30 days, M6+)", "published",
                                  _when(t), f"{_f(t['auc']['value'])}{_ci(t['auc']['ci'])}",
                                  f"IG {_signed(t['ig_per_target']['value'], 2)} {EQ_IG_UNIT}",
                                  "Boosted trees on long-term seismicity and aftershock-style clustering"
                                  + (", plus GEAR1&rsquo;s strain-rate model" if eq.get("gear1") else "")))
    if hu:
        t = hu["test"]
        rows.append(_registry_row(hu["model_version"], "Hurricane RI (NHC areas)", "published",
                                  _when(t), f"{_f(t['auc'])}{_ci(t['auc_ci'])}",
                                  f"BSS {_signed(t['bss'])}", "NOAA DTOPS as issued (SHIPS-RII fallback)"))
        ours = hu.get("ours")
        if ours:
            d = ours["dev"]
            rows.append(_registry_row(ours["model_version"], f"Hurricane RI (NHC areas, our model {_e(ours['label'])})",
                                      "experimental",
                                      f"{_years(ours.get('dev_period'))} (each season out of sample)", f"{_f(d['auc'])}",
                                      f"LL {_f(d['log_loss'])}" + (f" vs DTOPS {_f(d['dtops_log_loss'])}"
                                                                   if d.get("dtops_log_loss") is not None else ""),
                                      "LightGBM exceedance curve; in prospective verification, shown beside DTOPS"))
        v9 = hu.get("v9")
        if v9:
            f = v9["final_2026"]
            rows.append(_registry_row(v9["model_version"], f"Hurricane RI (NHC areas, {_e(v9['label'])})", "shadow",
                                      "2026 (read once)", f"{_f(f['auc'])}",
                                      f"LL {_f(f['log_loss'])} vs DTOPS {_f(f['dtops_log_loss'])}",
                                      "LightGBM on the early guidance, gated; "
                                      + ("claim met on 2026" if f.get("claim") else "claim not met on 2026")
                                      + ". Its prospective test decides whether it replaces DTOPS as the published "
                                        "number"))
        ob = hu.get("other_basins")
        if ob:
            o = ob["test"]
            if ob.get("subject") == "model":
                rep = ob.get("replaced") or {}
                rows.append(_registry_row(ob["model"], "Hurricane RI (other basins)", "published",
                                          f"{_when(o)} (registered test; the basins where it publishes, unseen cycles, "
                                          "best-track inputs)", f"{_f(o['auc'])}",
                                          f"LL {_f(o.get('log_loss'), 4)} vs {_e(rep.get('label') or '')} "
                                          f"{_f(rep.get('log_loss'), 4)}",
                                          f"GBT + logistic ensemble, Newton-calibrated; {_e(rep.get('label') or '')}"
                                          f"&rsquo;s recipe on de-duplicated rows, adopted by amendment "
                                          f"{_e(ob.get('amendment') or '')}&rsquo;s rule"))
            else:
                rows.append(_registry_row(ob["model"], "Hurricane RI (other basins)", "published",
                                          f"{_when(o)} (its method; every basin, best-track inputs)",
                                          f"{_f(o['auc'])}{_ci(o['auc_ci'])}",
                                          f"BSS {_signed(o['bss'])}", "GBT + logistic ensemble, Newton-calibrated"))
        j1 = hu.get("j1")
        if j1:
            m, a, v82 = j1["dev"]["model"], j1["dev"]["bar"], _e(j1.get("against") or "")
            pub = _e(j1.get("published") or j1.get("against") or "")
            scope = [_e(r["name"]) for r in j1.get("regions") or [] if r.get("in_scope")]
            rows.append(_registry_row(j1["model_version"],
                                      f"Hurricane RI (other basins, our satellite model {_e(j1['label'])})", "shadow",
                                      f"{_years(j1.get('dev_period'))} (registered hindcast, every JTWC cycle pooled)",
                                      f"{_f(m['auc'])}", f"LL {_f(m['log_loss'], 4)} vs {v82} {_f(a['log_loss'], 4)}",
                                      f"LightGBM on {v82}&rsquo;s probability and inputs plus GMGSI satellite infrared; "
                                      f"recorded beside {'the published ' + pub if pub != v82 else v82}"
                                      + (f", shown for {_join(scope)} storms" if scope else ", shown nowhere")
                                      + (f". Its prospective test against {v82} decides whether it replaces {pub} as "
                                         "the published number" if pub != v82 else
                                         f". Its prospective test decides whether it replaces {v82} as the published "
                                         "number")))
        tc = hu.get("tc1")
        if tc:
            tr = tc["track"]
            span = _years(f"{tc['seasons'][0]}-{tc['seasons'][-1]}") if tc.get("seasons") else "--"
            rows.append(_registry_row(tc["model_version"], "Hurricane track and intensity, to 5 days (NHC areas)",
                                      "published", f"{span} (pre-registered)", "&mdash;",
                                      f"track {_f(tr['errors']['TC1'], 1)} n mi vs NHC {_f(tr['errors']['OFCL'], 1)}",
                                      "Online-weighted consensus of NHC&rsquo;s real-time guidance; shown beside "
                                      "NHC&rsquo;s official forecast"))
    if to:
        t = to["test"]
        rows.append(_registry_row(to["model_version"], "Tornado within 60 min", "published", "2025",
                                  f"{_f(t['auc'])}{_ci(t['auc_ci'])}", f"BSS {_signed(t['bss'])}",
                                  f"LightGBM, {_n(to['n_trees'])} trees, {to['n_inputs']} inputs"))
        fb = to.get("fallback")
        if fb:
            ft = fb["test"]
            rows.append(_registry_row(fb["model_version"], "Tornado within 60 min, when the NWS feed is down", "fallback", "2025",
                                      f"{_f(ft['auc'])}{_ci(ft['auc_ci'])}", f"BSS {_signed(ft['bss'])}",
                                      f"LightGBM, {_n(fb['n_trees'])} trees, {fb['n_inputs']} inputs"))
        for key, p in (to.get("products") or {}).items():
            pt = p["test"]
            rows.append(_registry_row(p["model_version"], _e(_PRODUCT_NAMES[key]), "published", "2025",
                                      f"{_f(pt['auc'])}{_ci(pt['auc_ci'])}", f"BSS {_signed(pt['bss'])}",
                                      f"LightGBM, {_n(p['n_trees'])} trees"))
    return _table(["Model version", "Forecasts", "Status", "Test", "AUC [95% CI]", "Skill", "What it is"], rows,
                  caption="Every model version on the site now, and the entrants whose tests could change what is "
                          "published")


def registry_history(ev: dict) -> str:
    """Promotions and retirements, newest first, each reason read from the evidence of the model that
    replaced it. Only changes with a record in this repository are listed."""
    rows = []
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")

    def row(action: str, model: str, reason: str) -> str:
        chip = {"promote": "good", "retire": "neutral", "supersede": "neutral"}[action]
        word = {"promote": "promoted", "retire": "retired", "supersede": "superseded"}[action]
        return (f'<tr><td>Oct 2026</td><td><span class="chip {chip}">{word}</span></td>'
                f"<td><code>{_e(model)}</code></td><td>{reason}</td></tr>")
    if to:
        t = to["test"]
        vs = to.get("vs_probtor")
        rows.append(row("promote", to["model_version"],
                        f"Pre-registered programme, 2025 scored once: AUC {_f(t['auc'])}{_ci(t['auc_ci'])}"
                        + (f"; against NOAA ProbTor {_signed(vs['delta_auc'])}{_ci(vs['delta_auc_ci'], signed=True)} "
                           "(paired by day)" if vs else "")))
        rows.append(row("retire", "tornado_storm_v1_0",
                        "Published from March 2026 until v3 replaced it on 3 Oct 2026; its live record is on the "
                        "track record page"))
        v2 = to.get("vs_v2")
        rows.append(row("supersede", "tornado_gbt_v2",
                        "Overtaken by v3 before it published" + (f": on 2025, v3 &minus; v2 {_signed(v2['delta_auc'])}"
                                                                 f"{_ci(v2['delta_auc_ci'], signed=True)} AUC on the "
                                                                 "same storms and event" if v2 else "")))
        rows.append(row("retire", "hp-tornado-coherence-v1",
                        "Its published AUC of 0.894 was measured on a 5:1-downsampled split with labels on the wrong "
                        "clock (docs/AUDIT_2026-10-01.md)"))
    if eq:
        t = eq["test"]
        act = (eq.get("gear1") or {}).get("activity") or {}
        g1 = act.get("dev_vs_S1") or (eq.get("gear1") or {}).get("dev_vs_recalibrated") or {}
        how = "a second look at" if t.get("second_read") else "scored once on"
        rows.append(row("promote", eq["model_version"],
                        "Adds GEAR1&rsquo;s long-term rate to C0"
                        + (", weighted by how active each cell is" if act else "")
                        + (f", decided by a test written down in advance on {_e(eq['gear1'].get('decided_on', '')).replace('-', '&ndash;')}: "
                           f"{_signed(g1['ig_per_target']['diff'])}{_ci(g1['ig_per_target']['ci'], signed=True)} "
                           f"{EQ_IG_UNIT}" + (" over GEAR1 with one weight" if act else "")
                           if g1.get("ig_per_target") else "")
                        + f"; on {_when(t)} ({how} those years) {_signed(t['ig_per_target']['value'], 2)} "
                          f"{EQ_IG_UNIT}, AUC {_f(t['auc']['value'])}"))
        replaced = eq.get("replaced_ig_per_target")
        if act and replaced and replaced.get("value") is not None and eq.get("replaced_model_version"):
            rows.append(row("supersede", eq["replaced_model_version"],
                            f"GEAR1 with one weight; replaced by the activity-weighted stack. It scored "
                            f"{_signed(replaced['value'], 2)} {EQ_IG_UNIT} on the same test"))
        base = eq.get("base_ig_per_target") or replaced
        if base and base.get("value") is not None and eq.get("base_model_version"):
            rows.append(row("supersede", eq["base_model_version"],
                            f"Now the base of the published model rather than published itself; it scored "
                            f"{_signed(base['value'], 2)} {EQ_IG_UNIT} on the same test"))
        rows.append(row("retire", "eq_coherence_v1_0",
                        "Replaced by the pre-registered operational forecast (docs/EARTHQUAKE_FORECAST_PROGRAM.md); "
                        "its live record is on the track record page"))
    if hu:
        t = hu["test"]
        beaten = [_e(c["against"]) for c in hu.get("claims", []) if c["better"]]
        rows.append(row("promote", hu["model_version"],
                        f"NOAA DTOPS published for the NHC areas: {_when(t)} season scored once, AUC {_f(t['auc'])}"
                        + (f"; beat {_join(beaten)}" if beaten else "")))
    return _table(["Date", "Change", "Model version", "Reason"], rows, caption="Promotions and retirements")


# ---------------------------------------------------------------------------
# tornado verification page
# ---------------------------------------------------------------------------

def tornado_hero(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return '<p class="lede">No final test is bound to the published tornado model in this repository.</p>'
    t = to["test"]
    return ('<p class="lede">How the published tornado model scored on every thunderstorm NOAA tracked over the US in '
            f"2025: {_n(t['n'])} storm observations, {_n(t['pos'])} of them followed by a tornado. Every modelling "
            "choice was fixed before 2025 was scored, and it was scored once. Every number below is read from the "
            "programme&rsquo;s results files.</p>")


# the protocol's strata (scripts/audit_20261001/tornado_lab.stress_groups), in words
_STRATA = {
    "region_plains": "Plains (105&deg;W to 94&deg;W, north of 30&deg;N)",
    "region_midwest": "Midwest (94&deg;W to 80&deg;W, north of 37&deg;N)",
    "region_southeast": "Southeast (94&deg;W to 75&deg;W, south of 37&deg;N)",
    "region_elsewhere": "Outside these three regions",
    "season_DJF": "Winter (Dec&ndash;Feb)", "season_MAM": "Spring (Mar&ndash;May)",
    "season_JJA": "Summer (Jun&ndash;Aug)", "season_SON": "Autumn (Sep&ndash;Nov)",
    "local_night": "Night (20:00&ndash;06:00 local solar time)", "local_day": "Day (06:00&ndash;20:00 local solar time)",
    "size_small": "Smallest third of storms", "size_mid": "Middle third of storms", "size_large": "Largest third of storms",
    "analysis_9km": "Fine-grid (9 km) environment analysis available",
    "lead_0_15": "Tornado 0&ndash;15 minutes ahead", "lead_15_30": "Tornado 15&ndash;30 minutes ahead",
    "lead_30_60": "Tornado 30&ndash;60 minutes ahead",
    "ef2plus": "Strong tornadoes (EF2 and above)", "ef0_1": "Weak tornadoes (EF0&ndash;EF1)",
}


def _metric_row(name: str, m: dict, best: bool = False) -> str:
    cls = ' class="row-highlight"' if best else ""
    bss = _signed(m.get("bss")) if m.get("bss") is not None else "&mdash;"
    return (f"<tr{cls}><td>{name}</td><td class=\"num\">{_f(m.get('auc'))}</td>"
            f"<td class=\"num\">{_ci(m.get('auc_ci')).strip() or '&mdash;'}</td>"
            f"<td class=\"num\">{_f(m.get('pr_auc'))}</td><td class=\"num\">{bss}</td>"
            f"<td class=\"num\">{_n(m.get('n'))} / {_n(m.get('pos'))}</td></tr>")


def _section(ident: str, title: str, body: str, intro: str = "") -> str:
    head = f'<div class="section-head"><div><h2 id="{ident}">{title}</h2>' + (f"<p>{intro}</p>" if intro else "") + "</div></div>"
    return (f'<section class="section" aria-labelledby="{ident}"><div class="container">{head}{body}</div></section>')


def tornado_body(ev: dict) -> str:
    to = ev.get("tornado")
    if not to:
        return _section("results", "Results", "<p>No final test is bound to the published tornado model in this "
                                               "repository, so no result is shown.</p>")
    t = to["test"]
    event = _e(to["event"]).replace("THIS", "this")
    parts = []
    parts.append(_section("model", "The model", _facts([
        _kv("Model version", f"<code>{_e(to['model_version'])}</code>"),
        _kv("What it forecasts", "The chance that this storm produces a tornado within 60 minutes"
            + (" (its inputs include the live NWS tornado-warning state)" if to.get("inputs_use_nws") else "")),
        _kv("What counts as a tornado", event[0].upper() + event[1:]),
        _kv(f"Inputs ({to['n_inputs']})", _tornado_inputs_text(to)),
        _kv("Model form", f"Gradient-boosted trees (LightGBM, {_n(to['n_trees'])} trees), scored in NumPy from a JSON "
                          "file in the repository"),
        _kv("Calibration", "Platt scaling fitted on leave-one-year-out scores"
            + ("; a Venn&ndash;Abers band per storm" if to.get("has_band") else "")),
        _kv("Training", _e(to["trained"]).replace("..", "&ndash;")),
    ]) + (f'<p class="section-foot">How the test was run: every choice &mdash; inputs, model type, settings, '
          "calibration and the event definition &mdash; was made on 2023; 2024 was a development check; the model "
          f"was then refitted on {_years(to.get('trained_years'))} and scored once on 2025, against SPC storm reports "
          f"matched to each storm&rsquo;s own tracked radar outline. Protocol: {_doc(to['program'])}.</p>")))

    rows = [_metric_row("Published model (with the NWS warning state)", t, best=True)]
    fb = to.get("fallback")
    if fb:
        rows.append(_metric_row("Fallback, when the NWS warnings feed is down", fb["test"]))
    for key, pr in (to.get("products") or {}).items():
        rows.append(_metric_row(_e(_PRODUCT_NAMES[key]), pr["test"]))
    for key, name in (("probtor_raw", "NOAA ProbTor, as issued"), ("probtor_tiebroken", "NOAA ProbTor, ties broken")):
        m = (to.get("probtor_final") or {}).get(key)
        if m:
            rows.append(f"<tr><td>{name}</td><td class=\"num\">{_f(m.get('auc'))}</td>"
                        f"<td class=\"num\">{_ci(m.get('auc_ci')).strip() or '&mdash;'}</td>"
                        "<td class=\"num\">&mdash;</td><td class=\"num\">&mdash;</td>"
                        "<td class=\"num\">same storms</td></tr>")
    dev = to.get("dev")
    parts.append(_section(
        "results", "Final test: every storm of 2025",
        _table(["Forecast", "AUC", "95% interval", "PR-AUC", "Brier skill", "Storm observations / tornadic"], rows,
               caption="Scored once, after every choice was fixed")
        + '<p class="section-foot"><strong>AUC</strong>: how often a storm that produced a tornado is ranked above '
          "one that did not (0.5 is chance, 1 is perfect). <strong>PR-AUC</strong>: the same idea, focused on the "
          "rare tornadic storms. <strong>Brier skill</strong>: improvement over always forecasting 2025&rsquo;s "
          "average rate. Intervals: 95%, bootstrapped over days. NOAA ProbTor is issued in whole percents and is 0 for "
          "most storms, so its ties are broken with ProbSevere&rsquo;s own severe probability for a fair ranking."
        + (f" Development year {_when(dev)}: AUC {_f(dev['auc'])}{_ci(dev['auc_ci'])}, Brier skill "
           f"{_signed(dev['bss'])}." if dev else "") + "</p>"))

    comp = []
    vs = to.get("vs_probtor")
    s1 = _probtor_sentence(to)
    if s1:
        comp.append(f"<li><strong>NOAA ProbTor.</strong> {s1[0].upper() + s1[1:]}, over {_n(vs.get('n_days'))} days; "
                    f"Brier {_signed(vs['delta_brier'], 6)}{_ci(vs['delta_brier_ci'], 6, signed=True)}.</li>")
    s2 = _nws_sentence(to)
    nws = to.get("vs_nws_warnings")
    if s2:
        comp.append(f"<li><strong>NWS tornado warnings.</strong> {s2[0].upper() + s2[1:]}. The false-alarm rate is "
                    f"the warnings&rsquo; own ({_pct(nws['nws_pofd'], 3)} of storm observations without a tornado). A "
                    "model that reads the warning state measures what it adds on top of the warnings; the figure "
                    "without that input is the contest on its own. HazardPulse does not issue warnings; NWS warnings "
                    "remain the authority for protective action.</li>")
    v2 = to.get("vs_v2")
    if v2:
        comp.append(f"<li><strong>Our previous model (v2).</strong> {_signed(v2['delta_auc'])}"
                    f"{_ci(v2['delta_auc_ci'], signed=True)} AUC on the same storms and the same event.</li>")
    if comp:
        parts.append(_section("against", "Against NOAA and the NWS, on the same storms",
                              '<ul class="compare-list">' + "".join(comp) + "</ul>"))

    st = to.get("stress")
    if st:
        order = {k: i for i, k in enumerate(_STRATA)}
        strata = sorted(st["strata"], key=lambda r: order.get(r["stratum"], len(order)))
        srows = [f"<tr><td>{_STRATA.get(r['stratum'], _e(r['stratum'].replace('_', ' ')))}</td>"
                 f"<td class=\"num\">{_n(r['n'])} / {_n(r['pos'])}</td>"
                 f"<td class=\"num\">{_f(r['auc'])}{_ci(r['auc_ci'])}</td>"
                 f"<td class=\"num\">{_f(r['probtor_auc'])}{_ci(r['probtor_ci'])}</td></tr>" for r in strata]
        parts.append(_section(
            "strata", "By region, season, time of day, storm size and lead time",
            _table(["Group", "Storm observations / tornadic", "Model AUC", "ProbTor (calibrated) AUC"], srows,
                   caption="The groups were defined before any result"),
            intro=f"The lowest AUC in any group is {_f(st['min_auc'])}; in {st['n_clear_of_probtor']} of "
                  f"{st['n_strata']} groups the model&rsquo;s whole interval lies above calibrated ProbTor&rsquo;s."))

    rel = to.get("reliability")
    if rel:
        rrows = []
        for b in rel["bins"]:
            if not b.get("n"):
                continue
            ci = b.get("observed_ci")
            rrows.append(f"<tr><td>{_pct_fine(b['lo'])} &ndash; {_pct_fine(b['hi'])}</td>"
                         f"<td class=\"num\">{_n(b['n'])}</td><td class=\"num\">{_pct_fine(b['mean_forecast'])}</td>"
                         f"<td class=\"num\">{_pct_fine(b['observed'])}"
                         f"{(' [' + _pct_fine(ci[0]) + ', ' + _pct_fine(ci[1]) + ']') if ci else ''}</td></tr>")
        parts.append(_section(
            "calibration", "Does 10% mean 10%?",
            _table(["Forecast range", "Storm observations", "Average forecast", "Tornado followed [95% interval]"], rrows,
                   caption="2025 storm observations grouped by the chance the model gave them"),
            intro="In a well-calibrated row the average forecast and the rate at which a tornado followed agree. "
                  "Intervals are 95% Jeffreys intervals."))

    nu = to.get("tested_not_served") or []
    if nu:
        items = []
        for x in nu:
            how = "paired by day" if x.get("paired") else "difference of two validation runs"
            items.append(f"<li><strong>{_e(x['what'][0].upper() + x['what'][1:])}:</strong> "
                         f"{_signed(x['delta_auc'], 4)}{_ci(x.get('delta_auc_ci'), 4, signed=True)} validation AUC "
                         f"({how}{'; ' + _e(x['note']) if x.get('note') else ''}).</li>")
        parts.append(_section(
            "not-served", "Tested and not used",
            '<ul class="compare-list">' + "".join(items) + "</ul>",
            intro="Each was measured on 2023 under the same protocol. None improved the forecast, so none is in the "
                  "published model."))
    return "\n\n".join(parts)


def tornado_different(ev: dict) -> str:
    to = ev.get("tornado")
    items = [
        ("Fixed in advance, scored once",
         "Every modelling choice was fixed on 2023 and checked on 2024 before 2025 was scored, once. The protocol and "
         "its dated amendments are public, including the ones that corrected our own earlier numbers."),
        ("Scored against NOAA on the same storms",
         "The comparison with NOAA ProbTor and the NWS warnings uses the same storm observations, the same event and "
         "intervals paired by day."),
        ("Open and reproducible",
         "The model is a JSON file in the repository, scored in pure NumPy; its identity is the hash of its bytes, "
         "and every live forecast is appended to a hash-chained ledger."),
    ]
    if to and to.get("tested_not_served"):
        items.append(("What did not work is published too",
                      "The HRRR environment and the coherence field were tested and did not improve the forecast; "
                      "the measurements are above."))
    cards = "".join(f'<div class="card"><h3>{t}</h3><p>{d}</p></div>' for t, d in items)
    links = [
        (f"{REPO}/blob/main/{se.TORNADO_PROGRAM}", "The model programme"),
        ("/data/tornado-ledger.jsonl", "Every live forecast (ledger, JSONL)"),
        ("/verification/", "The track record for every hazard"),
        (REPO, "Source code"),
    ]
    link_html = "".join(f'<li><a href="{_e(h)}"{" rel=" + chr(34) + "noopener" + chr(34) if h.startswith("http") else ""}>'
                        f"{t}</a></li>" for h, t in links)
    return f'<div class="cards">{cards}</div><ul class="link-list section-foot">{link_html}</ul>'


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

BLOCKS: dict[str, dict[str, Callable[[dict], str]]] = {
    "methods/index.html": {
        "methods-data-hurricane": methods_data_hurricane,
        "methods-simple": methods_simple,
        "methods-earthquake": methods_earthquake,
        "methods-hurricane": methods_hurricane,
        "methods-tornado": methods_tornado,
        "methods-research": methods_research,
        "methods-limits": methods_limits,
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
    """Pages whose evidence blocks differ from what the results files say, or that carry -- anywhere, inside a
    block or in the hand-kept prose around one -- a sentence the results contradict (empty = all current)."""
    ev = se.all_evidence(root) if ev is None else ev
    stale = []
    for rel in BLOCKS:
        old = (dist / rel).read_text(encoding="utf-8")
        if render_page(rel, old, ev) != old:
            stale.append(rel)
        stale += [f"{rel} (contradicted: {c})" for c in contradictions(old)]
    return stale


# ---------------------------------------------------------------------------
# sentences the results contradict
# ---------------------------------------------------------------------------

_V82_EVERY_BASIN = ("the other-basins model's held-out test holds cycles from every basin, West Pacific among them "
                    f"({se.HURRICANE_V82_COMPOSITION}, by_basin)")
_T2_DECIDED = (f"NOAA's 2025 ProbSevere format change was decided by tornado amendment 10 ({se.TORNADO_T2}): the "
               "switch was 2025-08-05, ranking was unaffected, calibration (T2b) is the open question")
# Sentences the site once carried in prose that its own artifacts contradict. ``check_pages`` used to compare only
# the text BETWEEN evidence markers, so a sentence typed around a block -- or in a generated page's code -- could
# contradict the block beside it and the build stayed green; each of these did until 2026-10-09. A page that
# carries one, anywhere, now fails ``check_pages`` and ``hazardpulse.site.build.check_site``.
CONTRADICTED: tuple[tuple[str, str], ...] = (
    ("which has no test in those basins", _V82_EVERY_BASIN),
    ("not in the basins where it now publishes", _V82_EVERY_BASIN),
    ("tested on Atlantic and East Pacific cases", _V82_EVERY_BASIN),
    ("an open question we are testing", _T2_DECIDED),
    ("6 August 2025", _T2_DECIDED),
    ("a gain beyond chance", "the TC1 comparison with HCCA is descriptive at 95%, not one of the test's claims "
                             f"(results/hurricane_tc1/dev.json, reported TC1-HCCA level)"),
)


def page_text(page: str) -> str:
    """A page's words as a reader sees them: tags dropped, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page))).strip()


def contradictions(page: str) -> list[str]:
    """Each CONTRADICTED sentence the page carries, with the result that contradicts it."""
    text = page_text(page).lower()
    return [f"{phrase!r} -- {why}" for phrase, why in CONTRADICTED if phrase.lower() in text]
