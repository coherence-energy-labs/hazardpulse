"""How numbers, times and places read on every page -- one definition, so no two pages disagree.

* A chance is ``21.1%``; below a tenth of a percent it is ``<0.1%``, never ``0.0%``.
* A change is in percentage points with its baseline named by the caller: ``+1.9 pts``.
* A time is ``4 Oct 2026, 09:59 UTC`` (the API keeps ISO 8601).
* A chance is coloured on ONE scale shared by every hazard (``LEVELS``); the pages print its legend.
  The scorers' own band words differ by hazard (one of them is the National Weather Service's "watch"),
  so no page uses them.
"""
from __future__ import annotations

import datetime as dt
import html
import math

MINUS = "−"


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def num(value, decimals: int = 0) -> str:
    if value is None:
        return "&mdash;"
    v = float(value)
    s = f"{abs(v):,.{decimals}f}"
    return (MINUS if v < 0 else "") + s


def years(span) -> str:
    """``2022-2025`` as ``2022&ndash;2025`` (escaped first, so the dash entity survives)."""
    return esc(span).replace("-", "&ndash;")


def join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def pct(p, decimals: int = 1) -> str:
    """A probability as a percentage: ``21.1%``, ``<0.1%``; ``&mdash;`` when absent."""
    if p is None:
        return "&mdash;"
    p = float(p)
    if not math.isfinite(p):
        return "&mdash;"
    if p <= 0:
        return "0%"
    floor = 10 ** -(decimals + 2)
    if p < floor:
        return f"&lt;{100 * floor:.{decimals}f}%"
    if p >= 1:
        return "100%"
    return f"{100 * p:.{decimals}f}%"


def pct_plain(p, decimals: int = 1) -> str:
    """``pct`` for attribute values and plain text (no entities)."""
    return html.unescape(pct(p, decimals))


def pts(delta) -> str:
    """A change between two probabilities, in percentage points."""
    if delta is None:
        return "&mdash;"
    d = 100 * float(delta)
    if abs(d) < 0.05:
        return "no change"
    return f"{'+' if d > 0 else MINUS}{abs(d):.1f} pts"


def parse_time(value) -> dt.datetime | None:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        t = value
    else:
        s = str(value).strip().replace(" UTC", "Z")
        if len(s) >= 15 and s[8] == "_":            # ProbSevere's 20261004_073035
            s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"
        try:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc)


def utc(value, *, with_date: bool = True) -> str:
    """``4 Oct 2026, 09:59 UTC``; ``&mdash;`` when absent or unparseable."""
    t = parse_time(value)
    if t is None:
        return "&mdash;"
    if not with_date:
        return t.strftime("%H:%M UTC")
    return f"{t.day} {t.strftime('%b %Y, %H:%M')} UTC"


def date(value) -> str:
    t = parse_time(value)
    return "&mdash;" if t is None else f"{t.day} {t.strftime('%b %Y')}"


def iso(value) -> str:
    t = parse_time(value)
    return "" if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def time_tag(value, *, with_date: bool = True) -> str:
    """A ``<time>`` element: machine-readable ISO, human-readable text."""
    stamp = iso(value)
    text = utc(value, with_date=with_date)
    return f'<time datetime="{stamp}">{text}</time>' if stamp else text


# One colour scale for every chance on the site. (lower bound, css class, legend text)
LEVELS: tuple[tuple[float, str, str], ...] = (
    (0.50, "p6", "50% or more"),
    (0.30, "p5", "30&ndash;50%"),
    (0.15, "p4", "15&ndash;30%"),
    (0.05, "p3", "5&ndash;15%"),
    (0.01, "p2", "1&ndash;5%"),
    (0.0, "p1", "under 1%"),
)


def level(p) -> str:
    """The colour class of a chance on the shared scale."""
    v = float(p or 0.0)
    for lo, cls, _ in LEVELS:
        if v >= lo:
            return cls
    return "p1"


def legend(label: str = "Chance") -> str:
    items = "".join(f'<li><span class="swatch {cls}" aria-hidden="true"></span>{text}</li>'
                    for _, cls, text in reversed(LEVELS))
    return f'<div class="legend" role="group" aria-label="{esc(label)} colour scale"><span class="legend-title">{esc(label)}</span><ul>{items}</ul></div>'


_PROBABILITY_KEYS = ("probability", "tornado_probability", "ri_probability")


def interval(item: dict) -> str:
    """``Calibration range 8.0%&ndash;18.0%`` for a forecast whose calibrator produced an informative band
    that contains the probability shown beside it; nothing otherwise.

    The band is a Venn-Abers PAIR -- the two calibrated probabilities the calibration data supports -- not an
    interval with a stated coverage, so it is never labelled "90%" (the receipt's ``coverage_target`` is the
    trust layer's nominal target, not a property of the pair). Not shown when it is too wide to inform (the
    trust layer's ``uncertainty_class`` "wide": [0%, 25%] around a 0.2% chance tells a reader nothing), or when
    it does not contain the published probability (measured on the v3 tornado records to 2026-10-05 02:05Z:
    the model's 60-minute probability lay outside its own pair for 971 of 1,602 storm forecasts; a range that
    excludes the number it qualifies reads like an error, because it is one)."""
    lo, hi = item.get("confidence_lo"), item.get("confidence_hi")
    cls = item.get("uncertainty_class")
    if lo is None or hi is None or cls not in ("tight", "moderate"):
        return ""
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return ""
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        return ""
    p = next((item[k] for k in _PROBABILITY_KEYS if item.get(k) is not None), None)
    try:
        if p is not None and not lo <= float(p) <= hi:
            return ""
    except (TypeError, ValueError):
        return ""
    return f"Calibration range {pct(lo)}&ndash;{pct(hi)}"


def chance(p, *, big: bool = False) -> str:
    """A chance as a coloured figure."""
    cls = "chance chance-big" if big else "chance"
    return f'<span class="{cls} {level(p)}">{pct(p)}</span>'
