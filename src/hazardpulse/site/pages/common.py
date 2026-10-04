"""Building blocks shared by the generated pages. Every visual state is a class (the CSP refuses
inline styles); every number arrives formatted by ``fmt``."""
from __future__ import annotations

from hazardpulse.site import fmt
from hazardpulse.site.data import Headline
from hazardpulse.site.hazards import HAZARDS, Hazard, official_links

esc = fmt.esc


def hero(eyebrow: str, title: str, lede: str, *, meta: list[tuple[str, str]] | None = None,
         extra: str = "", cls: str = "") -> str:
    """The top of a page: a small label, the one ``<h1>``, a plain-language summary, then facts."""
    facts = ""
    if meta:
        facts = '<dl class="hero-meta">' + "".join(
            f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in meta) + "</dl>"
    return (f'<header class="page-hero {cls}"><div class="container">'
            f'<p class="eyebrow">{eyebrow}</p><h1>{title}</h1><p class="lede">{lede}</p>{facts}{extra}'
            "</div></header>")


def section(ident: str, title: str, body: str, *, intro: str = "", cls: str = "", aside: str = "") -> str:
    head = f'<div class="section-head"><div><h2 id="{ident}">{title}</h2>'
    head += f"<p>{intro}</p>" if intro else ""
    head += f"</div>{aside}</div>"
    return (f'<section class="section {cls}" aria-labelledby="{ident}"><div class="container">'
            f"{head}{body}</div></section>")


def facts(rows: list[tuple[str, str]], cls: str = "") -> str:
    """A definition list of label/value rows."""
    items = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in rows if v is not None)
    return f'<dl class="facts {cls}">{items}</dl>'


def official_notice(h: Hazard, lead: str = "For warnings and safety decisions, follow") -> str:
    return (f'<p class="notice notice-official"><span class="notice-icon" aria-hidden="true"></span>'
            f"<span>{lead} the {official_links(h)}. HazardPulse publishes research forecasts, not warnings.</span></p>")


GATE_WORDS = {
    "pass": ("Passed", "good"),
    "degrade": ("Published with warnings", "warn"),
    "block": ("Blocked", "bad"),
    "unknown": ("Not recorded", "neutral"),
}


def gate_chip(gate: str) -> str:
    word, cls = GATE_WORDS.get(gate, GATE_WORDS["unknown"])
    return f'<span class="chip {cls}">{word}</span>'


def plain_gate_note(note: str) -> str:
    """The quality checks' reasons, in words a reader can act on."""
    n = str(note)
    low = n.lower()
    if low.startswith("cryptographic provenance incomplete"):
        missing = n.split(":", 1)[1].strip() if ":" in n else ""
        names = {"model_sha256": "model file", "input_sha256": "input data", "receipt_sha256": "receipt"}
        parts = [names.get(x.strip(), x.strip()) for x in missing.split(",") if x.strip()]
        return "Hashes missing for: " + ", ".join(parts) if parts else "Some hashes are missing"
    if low.startswith("model calibration not yet measured"):
        return "Calibration on live forecasts not yet measured (too few closed forecasts with events)"
    if low.startswith("source data is stale"):
        return "Input data older than the limit " + n[len("source data is stale"):].strip()
    return n[:1].upper() + n[1:]


def gate_notice(head: Headline) -> str:
    """Shown on a live page when its forecast passed with warnings or was blocked."""
    if head.gate in ("pass", "unknown"):
        return ""
    word, cls = GATE_WORDS.get(head.gate, GATE_WORDS["unknown"])
    notes = "".join(f"<li>{esc(plain_gate_note(n))}</li>" for n in head.gate_notes)
    fid = esc(head.forecast_id or "")
    return (f'<div class="notice notice-{cls}"><p><strong>Quality checks: {word}.</strong> '
            "The forecast is published; a non-blocking check flagged the following. "
            f'<a href="/api/v1/gates/gdec_{fid}">Full check record</a></p>'
            + (f"<ul>{notes}</ul>" if notes else "") + "</div>")


def check_this(forecast_id: str | None, *, scored_note: str) -> str:
    """The three ways to check a forecast, at the foot of each live page."""
    fid = esc(forecast_id or "")
    return ('<div class="cards cards-3">'
            '<a class="card card-link" href="/evidence/#current"><h3>See what produced it</h3>'
            "<p>The forecast&rsquo;s record: its data sources, model version, and hashes of its inputs and output.</p>"
            '<span class="card-cta">Open the evidence</span></a>'
            f'<a class="card card-link" href="/verification/"><h3>See the track record</h3><p>{scored_note}</p>'
            '<span class="card-cta">Open the track record</span></a>'
            f'<a class="card card-link" href="/data/replay/{fid}.json"><h3>Re-run it yourself</h3>'
            "<p>Download this forecast&rsquo;s frozen record, with everything needed to reproduce the published "
            "numbers.</p>"
            f'<span class="card-cta">Download {fid}.json</span></a></div>')


def since_previous(head: Headline) -> str:
    """``+1.9 pts since the previous forecast (4 Oct, 00:00 UTC)``, or nothing when there is none."""
    if head.probability is None or head.previous is None:
        return ""
    when = f" ({fmt.utc(head.previous_issued_at)})" if head.previous_issued_at else ""
    return f"{fmt.pts(head.probability - head.previous)} since the previous forecast{when}"


def hazard_label(key: str) -> str:
    h = HAZARDS[key]
    return f'<span class="hazard-tag {key}"><span class="hazard-dot {key}" aria-hidden="true"></span>{h.name}</span>'


def table(head: list[str], rows: list[list[str]], *, cls: str = "", caption: str = "",
          num_cols: tuple[int, ...] = ()) -> str:
    ths = "".join(f'<th scope="col"{" class=" + chr(34) + "num" + chr(34) if i in num_cols else ""}>{h}</th>'
                  for i, h in enumerate(head))
    trs = "".join("<tr>" + "".join(
        f'<td{" class=" + chr(34) + "num" + chr(34) if i in num_cols else ""}>{c}</td>' for i, c in enumerate(r))
        + "</tr>" for r in rows)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<div class="table-wrap {cls}" tabindex="0" role="region" aria-label="{esc(fmt.esc(caption) or head[0])}">'
            f"<table>{cap}<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>")


def disclosure(title: str, body: str, *, cls: str = "", open_: bool = False, ident: str = "") -> str:
    idattr = f' id="{ident}"' if ident else ""
    return (f'<details class="disclosure {cls}"{idattr}{" open" if open_ else ""}><summary>{title}</summary>'
            f'<div class="disclosure-body">{body}</div></details>')
