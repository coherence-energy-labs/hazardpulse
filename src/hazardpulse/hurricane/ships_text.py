"""NHC's live SHIPS text: the RI probabilities of one storm and cycle, in the e-deck representation.

NHC publishes every SHIPS run at ``https://ftp.nhc.noaa.gov/atcf/stext/`` as
``{YYMMDDHH}{BB}{NN}{YY}_ships.txt`` (``26100212EP1826_ships.txt`` = EP18 2026, 12 UTC 2 Oct
2026). Near the end it prints::

    Matrix of RI probabilities
    ------------------------------------------------------------------------------
      RI (kt / h)  | 20/12 | 25/24 | 30/24 | 35/24 | 40/24 | 45/36 | 55/48  |65/72
    ------------------------------------------------------------------------------
       SHIPS-RII:     4.1%    8.8%    6.7%    6.7%    5.2%    6.8%    0.0%    0.0%
        Logistic:     0.1%    0.2%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%
        ...

The training data are the ATCF e-deck RI records, which carry WHOLE percent. So the parser:

* finds the threshold column BY ITS HEADER ("30/24"), never by position, and refuses a header
  that does not name it or a row whose value count differs from the header's;
* reads each row BY NAME (SHIPS-RII -> RIOD, Logistic -> RIOL, Bayesian -> RIOB,
  Consensus -> RIOC, DTOPS -> DTOP, SDCON -> SDCN); a row that is absent, or printed as
  SHIPS's missing sentinel ``999.0%``, is ``None``;
* returns WHOLE percent, the training representation. For SHIPS-RII the file prints SHIPS's own
  integer ("SHIPS Prob RI for 30kt/ 24hr RI threshold=    7%"), and that is used when it is
  consistent with the matrix value (within the 0.05 the one-decimal print can hide, plus 0.5).
  MEASURED on 203 live 2026 files (1,589 SHIPS-RII values): SHIPS's integer is round-to-nearest
  of the unprinted value -- all 1,513 values not printed as x.5 match, truncation matches 1,282
  -- but at the 76 printed x.5 values it goes either way (half-up agrees 40 times), because the
  hidden second decimal decides. Every other row has no integer line and is rounded HALF-UP in
  decimal arithmetic (6.5 -> 7, 0.4 -> 0, 99.5 -> 100), so at a printed x.5 it can differ by
  1 point from what SHIPS itself would print. Whether the e-deck rounds the same way is
  unverified -- no season has both files public (docs/HURRICANE_RI_PROGRAM.md);
* refuses a file whose storm id or time (the banner, and the RI-index header when present)
  disagrees with its file name: a mislabelled file must never feed another storm's forecast.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

STEXT_ROOT = "https://ftp.nhc.noaa.gov/atcf/stext/"
DEFAULT_THRESHOLD = "30/24"

# Row label in the text -> ATCF e-deck tech.
ROW_TECH = {
    "SHIPS-RII": "RIOD",
    "Logistic": "RIOL",
    "Bayesian": "RIOB",
    "Consensus": "RIOC",
    "DTOPS": "DTOP",
    "SDCON": "SDCN",
}
TECHS = tuple(ROW_TECH.values())

FILENAME_RE = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<hh>\d{2})"
                         r"(?P<basin>[A-Z]{2})(?P<nn>\d{2})(?P<sy>\d{2})_ships\.txt$")
# Banner: "*  RACHEL      EP182026  10/02/26  12 UTC        *"
BANNER_RE = re.compile(r"^\s*\*\s+(?P<name>\S(?:.*?\S)?)\s+(?P<sid>[A-Z]{2}\d{6})\s+"
                       r"(?P<mm>\d{2})/(?P<dd>\d{2})/(?P<yy>\d{2})\s+(?P<hh>\d{2})\s+UTC\s*\*\s*$")
# RI index header: "**2026 E. Pacific RI INDEX EP182026 RACHEL     10/02/26  12 UTC **"
RI_INDEX_RE = re.compile(r"RI INDEX\s+(?P<sid>[A-Z]{2}\d{6})\s+(?P<name>.*?)\s+"
                         r"(?P<mm>\d{2})/(?P<dd>\d{2})/(?P<yy>\d{2,4})\s+(?P<hh>\d{2})\s+UTC")
MATRIX_TITLE = "Matrix of RI probabilities"
# SHIPS-RII's own whole percent: "SHIPS Prob RI for 30kt/ 24hr RI threshold=    7% is ..."
RII_INTEGER_RE = re.compile(r"SHIPS Prob RI for\s+(?P<kt>\d+)kt/\s*(?P<hr>\d+)hr RI threshold=\s*(?P<pct>\d+)%")
ROW_RE = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z0-9\- ]*?)\s*:\s*(?P<values>\S.*?)\s*$")
VALUE_RE = re.compile(r"^(?P<num>\d{1,3}(?:\.\d+)?)%$")
# SHIPS prints an aid that did not run as 999.0% (e.g. 26071618EP0526: SHIPS-RII, Logistic,
# Bayesian, Consensus and SDCON all 999.0% while DTOPS ran). Missing, not a refusal; any other
# value above 100% is still refused.
MISSING_SENTINEL = Decimal("999")


class ShipsTextError(ValueError):
    """A SHIPS text that cannot be trusted for this storm and cycle."""


@dataclasses.dataclass(frozen=True)
class ShipsRI:
    storm_id: str                         # ATCF id, e.g. EP182026
    cycle: dt.datetime                    # synoptic time (naive UTC)
    storm_name: str | None
    threshold: str                        # the matrix column read, e.g. "30/24"
    percent: dict[str, str | None]        # tech -> the printed percent text ("6.7"), None if absent
    whole_percent: dict[str, int | None]  # tech -> whole percent (see the module docstring)
    rounding: dict[str, str | None]       # tech -> "ships_integer_line" | "half_up" | None
    columns: tuple[str, ...]              # the matrix header as printed
    unknown_rows: tuple[str, ...]         # row labels this parser does not map (kept for audit)

    def aids(self) -> dict[str, int | None]:
        return dict(self.whole_percent)


def round_half_up(text: str | Decimal) -> int:
    """Whole percent of a printed percent, half-up in DECIMAL (6.5 -> 7; binary 6.49999... never
    enters: the printed digits are the value)."""
    try:
        value = Decimal(str(text))
    except InvalidOperation as exc:
        raise ShipsTextError(f"not a number: {text!r}") from exc
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def filename_for(storm_id: str, cycle: dt.datetime) -> str:
    """``EP182026`` at 2026-10-02 12 UTC -> ``26100212EP1826_ships.txt``."""
    sid = storm_id.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}\d{6}", sid):
        raise ValueError(f"not an ATCF storm id: {storm_id!r}")
    return f"{cycle:%y%m%d%H}{sid[:2]}{sid[2:4]}{sid[-2:]}_ships.txt"


def url_for(storm_id: str, cycle: dt.datetime) -> str:
    return STEXT_ROOT + filename_for(storm_id, cycle)


def parse_filename(name: str) -> tuple[str, dt.datetime]:
    """``26100212EP1826_ships.txt`` -> (``EP182026``, 2026-10-02 12:00)."""
    m = FILENAME_RE.match(name.rsplit("/", 1)[-1])
    if m is None:
        raise ShipsTextError(f"not a SHIPS text file name: {name!r}")
    try:
        cycle = dt.datetime(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]), int(m["hh"]))
    except ValueError as exc:
        raise ShipsTextError(f"impossible date in file name {name!r}") from exc
    return f"{m['basin']}{m['nn']}20{m['sy']}", cycle


def _time(mm: str, dd: str, yy: str, hh: str, where: str) -> dt.datetime:
    year = int(yy) if len(yy) == 4 else 2000 + int(yy)
    try:
        return dt.datetime(year, int(mm), int(dd), int(hh))
    except ValueError as exc:
        raise ShipsTextError(f"impossible date in the {where}: {mm}/{dd}/{yy} {hh} UTC") from exc


def parse_header(text: str) -> tuple[str, dt.datetime, str | None]:
    """(storm id, cycle, name) from the banner; the RI-index header, when present, must agree."""
    banner = None
    for line in text.splitlines()[:40]:
        m = BANNER_RE.match(line)
        if m:
            banner = m
            break
    if banner is None:
        raise ShipsTextError("no storm banner ('*  NAME  BBNNYYYY  MM/DD/YY  HH UTC  *') in the first 40 lines")
    sid = banner["sid"]
    cycle = _time(banner["mm"], banner["dd"], banner["yy"], banner["hh"], "banner")
    for m in RI_INDEX_RE.finditer(text):
        t = _time(m["mm"], m["dd"], m["yy"], m["hh"], "RI-index header")
        if m["sid"] != sid or t != cycle:
            raise ShipsTextError(f"RI-index header says {m['sid']} {t:%Y-%m-%d %H}Z, banner says {sid} {cycle:%Y-%m-%d %H}Z")
    return sid, cycle, banner["name"].strip() or None


def _matrix(text: str) -> tuple[tuple[str, ...], dict[str, list[str]], list[str]]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == MATRIX_TITLE]
    if not starts:
        raise ShipsTextError(f"no '{MATRIX_TITLE}' block")
    if len(starts) > 1:
        raise ShipsTextError(f"{len(starts)} '{MATRIX_TITLE}' blocks: which one is this cycle's?")
    i = starts[0] + 1
    header: tuple[str, ...] | None = None
    rows: dict[str, list[str]] = {}
    order: list[str] = []
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if set(stripped) <= {"-"}:
            continue
        if header is None:
            if "|" not in line:
                raise ShipsTextError(f"expected the matrix header with '|' columns, got {stripped!r}")
            header = tuple(part.strip() for part in line.split("|")[1:])
            if not header or any(not re.fullmatch(r"\d{2}/\d{2}", c) for c in header):
                raise ShipsTextError(f"unreadable matrix header {stripped!r}")
            if len(set(header)) != len(header):
                raise ShipsTextError(f"matrix header repeats a column: {header}")
            continue
        m = ROW_RE.match(line)
        if m is None:
            break  # the block has ended (next section)
        label = m["label"].strip()
        values = m["values"].split()
        if len(values) != len(header):
            raise ShipsTextError(f"row {label!r} has {len(values)} values for {len(header)} columns")
        if label in rows:
            raise ShipsTextError(f"row {label!r} appears twice")
        rows[label] = values
        order.append(label)
    if header is None:
        raise ShipsTextError("matrix block has no header")
    if not rows:
        raise ShipsTextError("matrix block has no rows")
    return header, rows, order


def parse_ships_text(
    text: str,
    *,
    filename: str | None = None,
    expected_storm_id: str | None = None,
    expected_cycle: dt.datetime | None = None,
    threshold: str = DEFAULT_THRESHOLD,
) -> ShipsRI:
    """The ``threshold`` column of the RI matrix, by row name, as whole percent.

    ``filename`` (and/or ``expected_storm_id``/``expected_cycle``) is the identity the caller
    asked for; the text's own banner and RI-index header must match it or this refuses.
    """
    sid, cycle, name = parse_header(text)
    expected: list[tuple[str, dt.datetime]] = []
    if filename is not None:
        expected.append(parse_filename(filename))
    if expected_storm_id is not None or expected_cycle is not None:
        expected.append((expected_storm_id or sid, expected_cycle or cycle))
    for want_sid, want_cycle in expected:
        if want_sid.upper() != sid or want_cycle != cycle:
            raise ShipsTextError(
                f"file is {sid} {cycle:%Y-%m-%d %H}Z but was requested as {want_sid.upper()} {want_cycle:%Y-%m-%d %H}Z")
    header, rows, order = _matrix(text)
    if threshold not in header:
        raise ShipsTextError(f"matrix header {header} has no {threshold} column")
    col = header.index(threshold)
    percent: dict[str, str | None] = {tech: None for tech in TECHS}
    whole: dict[str, int | None] = {tech: None for tech in TECHS}
    rounding: dict[str, str | None] = {tech: None for tech in TECHS}
    kt, hr = (int(x) for x in threshold.split("/"))
    rii_integer = [int(m["pct"]) for m in RII_INTEGER_RE.finditer(text)
                   if int(m["kt"]) == kt and int(m["hr"]) == hr]
    if len(rii_integer) > 1:
        raise ShipsTextError(f"{len(rii_integer)} 'SHIPS Prob RI for {kt}kt/ {hr}hr' lines")
    for label, values in rows.items():
        tech = ROW_TECH.get(label)
        if tech is None:
            continue
        token = values[col]
        m = VALUE_RE.match(token)
        if m is None:
            if token.upper() in ("N/A", "NA", "XX.X%", "XXX.X%", "-"):
                continue  # printed as missing: the aid did not run
            raise ShipsTextError(f"row {label!r}, column {threshold}: unreadable value {token!r}")
        value = Decimal(m["num"])
        if value == MISSING_SENTINEL:
            continue  # 999.0%: SHIPS's "this aid did not run" (29 of 203 live 2026 files carry it)
        if value > 100:
            raise ShipsTextError(f"row {label!r}, column {threshold}: {value}% is not a probability")
        percent[tech] = m["num"]
        whole[tech] = round_half_up(value)
        rounding[tech] = "half_up"
        if tech == "RIOD" and rii_integer:
            k = rii_integer[0]
            # the matrix prints one decimal of a value SHIPS rounded itself: they must agree
            # to within half a point plus the 0.05 the print can hide
            if abs(Decimal(k) - value) > Decimal("0.55"):
                raise ShipsTextError(f"SHIPS-RII {threshold}: the file's own integer {k}% contradicts "
                                     f"the matrix's {value}%")
            whole[tech] = k
            rounding[tech] = "ships_integer_line"
    unknown = tuple(label for label in order if label not in ROW_TECH)
    return ShipsRI(sid, cycle, name, threshold, percent, whole, rounding, header, unknown)


def summary(ri: ShipsRI) -> dict[str, Any]:
    """JSON-ready record of what was read (what a served forecast carries as its inputs)."""
    return {
        "storm_id": ri.storm_id,
        "cycle": ri.cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "threshold": ri.threshold,
        "file": filename_for(ri.storm_id, ri.cycle),
        "whole_percent": dict(ri.whole_percent),
        "printed_percent": dict(ri.percent),
        "rounding": dict(ri.rounding),
    }
