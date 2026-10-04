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
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{_e(re.sub("<[^>]+>", "", caption or head[0]))}">'
            f"<table>{cap}<thead><tr>{ths}</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table></div>")


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
        g1 = eq.get("gear1")
        out.append(_card(
            "hz-eq", "Earthquake",
            f"<p>The chance of a magnitude {eq['target_magnitude_min']:.0f}+ earthquake in each 2&deg; cell of the "
            f"globe over the next {eq['horizon_days']:.0f} days, from where earthquakes happen in the long run, how "
            "they cluster after recent ones, and boosted trees on both"
            + (", plus GEAR1, a published global model of where the crust is straining" if g1 else "")
            + (f". Scored on {_when(t)} (a second look at those years): " if t.get("second_read")
               else f". Tested once on {_when(t)}: ")
            + f"{_signed(t['ig_per_target']['value'], 2)} nats of information per earthquake over a uniform map"
            + (f" ({_e(eq.get('replaced_name', 'the model it replaced'))}: {_signed(replaced.get('value'), 2)})"
               if replaced.get("value") is not None else "")
            + ". Earthquakes cannot be predicted; this ranks where the odds are higher.</p>"))
    else:
        out.append(_card("hz-eq", "Earthquake",
                         "<p>No final test is bound to the served earthquake model in this repository.</p>"))
    if hu:
        t = hu["test"]
        ob = (hu.get("other_basins") or {}).get("test") or {}
        same = hu.get("v8_2_same_cases") or {}
        beaten = [_e(c["against"]) for c in hu.get("claims", []) if c["better"]]
        out.append(_card(
            "hz-hu", "Hurricane",
            "<p>The chance of rapid intensification: maximum sustained winds rising 30 knots or more in 24 hours. For "
            f"storms the National Hurricane Center covers we publish {_e(hu['candidate_name'].split(' (')[0])}, which "
            + (f"beat {_join(beaten)} " if beaten else "was chosen ")
            + f"in a test written down in advance and run once on {_when(t)}: AUC {_f(t['auc'])}"
            + (f", against {_f(same.get('auc'))} for our v8.2 model on the same cycles" if same.get("auc") is not None else "")
            + "."
            + (f" Everywhere else we publish v8.2 (AUC {_f(ob.get('auc'))} on held-out {_when(ob)} Atlantic and "
               "East Pacific cases; it has not been tested in the basins where it publishes)."
               if ob.get("auc") is not None else "")
            + " Each storm says which."
            + (f" Our newer model, v10.1, scored better than DTOPS over 2022&ndash;2025, each season forecast only "
               f"from earlier ones (log loss {_f(hu['ours']['dev']['log_loss'])} against "
               f"{_f(hu['ours']['dev']['dtops_log_loss'])}); it is shown beside DTOPS, labelled experimental, while a "
               "test written down in advance scores it on new forecasts." if hu.get("ours") else "")
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
    return ("; GEAR1 (Bird et al. 2015) was added by a later pre-registered test: three weights fitted on "
            f"2018&ndash;2020, decided on {_e(g1['decided_on']).replace('-', '&ndash;')} against C0 recalibrated on the "
            f"same years, {_signed(d['diff'])}{_ci(d['ci'], signed=True)} nats per quake")


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
        rows.append(_kv("Replaced", f"{_e(eq.get('replaced_name', 'The previous model'))[:1].upper()}"
                                    f"{_e(eq.get('replaced_name', 'The previous model'))[1:]} scored "
                                    f"{_signed(replaced['value'], 2)} nats per quake on the same test"
                        + (f"; {_pct(share, 0)} of the test&rsquo;s M6+ cell-windows fell outside recently active "
                           "cells" if share is not None else "")))
    cr = t.get("calib_ratio") or {}
    if cr.get("value") is not None:
        over = cr["value"] - 1.0
        rows.append(_kv("Limits", f"Forecasts {_pct(abs(over), 0)} {'more' if over > 0 else 'fewer'} quakes than "
                                  f"occurred in the test period (ratio {_f(cr['value'], 2)}{_ci(cr.get('ci'), 2)}); "
                                  "live catalogs are preliminary in the first days after a large quake"))
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
    ob = hu.get("other_basins")
    if ob:
        o = ob["test"]
        out.append(_card(
            "hz-hu", "Hurricane, other basins: HazardPulse v8.2 <code>hurricane_ri_v8_2</code>", _facts([
                _kv("Architecture", "Histogram-GBT (depth 3 + 4) + L2 logistic + bagged logistic ensemble, "
                                    "Newton-calibrated"),
                _kv("Key inputs", "Analysis intensity and pressure (CARQ) and their 6&ndash;24 h tendencies, "
                                  "aid-model intensity forecasts, climatological potential intensity, motion, storm age"),
                _kv("AUC", f"{_f(o['auc'])}{_ci(o['auc_ci'])} on {_n(o['n'])} held-out "
                           f"{_when(o)} NHC cases (Atlantic and East Pacific)"
                    + (f"; {_f((hu.get('v8_2_same_cases') or {}).get('auc'))} on the 2025 NHC cycles above"
                       if (hu.get("v8_2_same_cases") or {}).get("auc") is not None else "")),
                _kv("Used for", "West Pacific, Indian Ocean and Southern Hemisphere storms (no public RI guidance), "
                                "and NHC cycles without SHIPS text"),
                _kv("Not yet tested", "In the basins where it publishes: its test cases are NHC&rsquo;s"),
            ])))
    return "\n\n".join(out)


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
             f"<code>{_e(ch['model_version'])}</code>: {what}. 2022&ndash;2025 against {against}: log loss "
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
                       "forecast; NOAA&rsquo;s DTOPS wherever that guidance is missing"),
        ("2022&ndash;2025, each season forecast only from earlier ones", f"log loss {_f(d['log_loss'])} vs DTOPS "
                                       f"{_f(d['dtops_log_loss'])} (paired by storm{_ci(d.get('d_log_loss_ci'), signed=True)})"
                                       + (f"; over DTOPS&rsquo;s four 24-h thresholds, Brier {_f(mt.get('ours'))} vs "
                                          f"{_f(mt.get('dtops'))}{_ci(mt.get('d_ci'), signed=True)}" if mt else "")),
        *_ours_vs_all_lines(ours.get("vs_all")),
        ("2026 so far", f"log loss {_f(s26['log_loss'])} vs DTOPS {_f(s26['dtops_log_loss'])}, AUC {_f(s26['auc'])} "
                        "(a second look at a season that informed its design, so not a proof)"),
        *[line for ch in (ours.get("challengers") or ([ours["challenger"]] if ours.get("challenger") else []))
          for line in _ours_challenger_lines(ch)],
        ("Status", f"{_e(ours['status'])}: shown beside NOAA&rsquo;s number, not instead of it, until the "
                   "pre-registered prospective test on cycles from 2026-10-04 meets its rule (verdicts "
                   + " and ".join(_e(x) for x in pr.get("look_dates") or [])
                   + f"); cycles scored so far: {_n(pr.get('matured_and_scored', 0))}"),
    ]
    return lines


def _ours_hurricane_card(ours: dict) -> str:
    rows = [_kv(k, v) for k, v in ours_hurricane_lines(ours)]
    return _card("hz-hu", f'Hurricane, our model (experimental): v10.1 <code>{_e(ours["model_version"])}</code>',
                 _facts(rows), ident="hurricane-ours")


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
    rows.append(_kv("Status", "Published"))
    return _card("hz-to", f'Tornado, storm by storm <code>{_e(to["model_version"])}</code>', _facts(rows),
                 ident="tornado-model")


# ---------------------------------------------------------------------------
# registry page
# ---------------------------------------------------------------------------

def registry_simple(ev: dict) -> str:
    eq, hu, to = ev.get("earthquake"), ev.get("hurricane"), ev.get("tornado")
    cards = []
    if eq:
        t = eq["test"]
        how = "a second look at those years" if t.get("second_read") else "scored once"
        cards.append(('hz-eq', "Earthquake: S1 (C0 + GEAR1)" if eq.get("gear1") else "Earthquake",
                      "Combines where large earthquakes happen in the long run with how they cluster after recent "
                      + ("ones, plus GEAR1&rsquo;s long-term rate from crustal strain. " if eq.get("gear1") else "ones. ")
                      + f"On {_when(t)} ({how}), it ranked a cell that went on to have an M6+ earthquake above one "
                      f"that did not {_pct(t['auc']['value'], 0)} of the time across the globe, and "
                      f"{_pct(t['auc_active_cells']['value'], 0)} among cells with recent earthquakes, the harder question."))
    if hu:
        t = hu["test"]
        beaten = [_e(c["against"]) for c in hu.get("claims", []) if c["better"]]
        cards.append(('hz-hu', "Hurricane: NOAA DTOPS, and v8.2 elsewhere",
                      "For storms the National Hurricane Center covers, we publish NOAA&rsquo;s own DTOPS guidance: in a "
                      f"test written down in advance and run once on the {_when(t)} season it "
                      + (f"beat {_join(beaten)}, " if beaten else "")
                      + "ranking a forecast cycle that went on to intensify rapidly above one that did not "
                      f"{_pct(t['auc'], 0)} of the time. Elsewhere we publish our v8.2 model. Our v10.1 model is shown "
                      "beside DTOPS, labelled experimental, while it is tested on new forecasts."))
    if to:
        t = to["test"]
        cards.append(('hz-to', "Tornado: v3",
                      "Scores every thunderstorm NOAA tracks over the contiguous US for the chance it produces a tornado "
                      "in the next hour. On every storm of 2025, scored once, it ranked a storm that went on to produce "
                      f"a tornado above one that did not {_pct(t['auc'], 0)} of the time."))
    return "\n".join(_card(cls, title, f"<p>{text}</p>") for cls, title, text in cards)


_STATUS_CHIP = {"published": "good", "experimental": "warn", "fallback": "neutral"}


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
                                  f"IG {_signed(t['ig_per_target']['value'], 2)} nats/quake",
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
            rows.append(_registry_row(ours["model_version"], "Hurricane RI (NHC areas, our model v10.1)", "experimental",
                                      "2022&ndash;2025 (each season out of sample)", f"{_f(d['auc'])}",
                                      f"LL {_f(d['log_loss'])} vs DTOPS {_f(d['dtops_log_loss'])}",
                                      "LightGBM exceedance curve; in prospective verification, shown beside DTOPS"))
        ob = hu.get("other_basins")
        if ob:
            o = ob["test"]
            rows.append(_registry_row(ob["model"], "Hurricane RI (other basins)", "published",
                                      f"{_when(o)} (NHC cases only)", f"{_f(o['auc'])}{_ci(o['auc_ci'])}",
                                      f"BSS {_signed(o['bss'])}", "GBT + logistic ensemble, Newton-calibrated"))
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
                  caption="Every model version that publishes a number now")


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
        g1 = (eq.get("gear1") or {}).get("dev_vs_recalibrated") or {}
        how = "a second look at" if t.get("second_read") else "scored once on"
        rows.append(row("promote", eq["model_version"],
                        "Adds GEAR1&rsquo;s long-term rate to C0"
                        + (f", decided by a test written down in advance on {_e(eq['gear1'].get('decided_on', '')).replace('-', '&ndash;')}: "
                           f"{_signed(g1['ig_per_target']['diff'])}{_ci(g1['ig_per_target']['ci'], signed=True)} nats per "
                           "earthquake" if g1.get("ig_per_target") else "")
                        + f"; on {_when(t)} ({how} those years) {_signed(t['ig_per_target']['value'], 2)} nats per "
                          f"earthquake, AUC {_f(t['auc']['value'])}"))
        replaced = eq.get("replaced_ig_per_target")
        if replaced and replaced.get("value") is not None and eq.get("base_model_version"):
            rows.append(row("supersede", eq["base_model_version"],
                            f"Now the base of the published model rather than published itself; it scored "
                            f"{_signed(replaced['value'], 2)} nats per earthquake on the same test"))
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
          "was then refitted on 2020&ndash;2024 and scored once on 2025, against SPC storm reports matched to each "
          f"storm&rsquo;s own tracked radar outline. Protocol: {_doc(to['program'])}.</p>")))

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
