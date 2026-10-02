"""NHC SHIPS text parser: real 2026 files, then adversarial edits of them.

Fixtures are real files from https://ftp.nhc.noaa.gov/atcf/stext/ (fetched 2026-10-02):
26100212EP1826 (Rachel), 26100212EP1526 (Nolo), 26093012AL0826 (Hanna).
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.hurricane import ships_text as st  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "ships_text"
RACHEL = "26100212EP1826_ships.txt"
NOLO = "26100212EP1526_ships.txt"
HANNA = "26093012AL0826_ships.txt"


def text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------

EXPECTED = {
    # printed 30/24 values and the whole percent they must become
    RACHEL: ("EP182026", dt.datetime(2026, 10, 2, 12), "RACHEL",
             {"RIOD": ("6.7", 7), "RIOL": ("0.0", 0), "RIOB": ("0.0", 0), "RIOC": ("2.3", 2),
              "DTOP": ("0.0", 0), "SDCN": ("1.1", 1)}),
    NOLO: ("EP152026", dt.datetime(2026, 10, 2, 12), "NOLO",
           {"RIOD": ("19.3", 19), "RIOL": ("0.3", 0), "RIOB": ("0.0", 0), "RIOC": ("6.5", 7),
            "DTOP": ("15.0", 15), "SDCN": ("10.7", 11)}),
    HANNA: ("AL082026", dt.datetime(2026, 9, 30, 12), "HANNA",
            {"RIOD": ("0.0", 0), "RIOL": ("1.6", 2), "RIOB": ("0.0", 0), "RIOC": ("0.5", 1),
             "DTOP": ("0.0", 0), "SDCN": ("0.2", 0)}),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_files_parse_to_the_printed_values(name):
    sid, cycle, storm, values = EXPECTED[name]
    ri = st.parse_ships_text(text(name), filename=name)
    assert (ri.storm_id, ri.cycle, ri.storm_name, ri.threshold) == (sid, cycle, storm, "30/24")
    assert ri.columns == ("20/12", "25/24", "30/24", "35/24", "40/24", "45/36", "55/48", "65/72")
    for tech, (printed, whole) in values.items():
        assert ri.percent[tech] == printed, tech
        assert ri.whole_percent[tech] == whole, tech
    assert ri.rounding["RIOD"] == "ships_integer_line"
    assert ri.rounding["DTOP"] == "half_up"
    assert ri.unknown_rows == ()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_ships_rii_equals_the_files_own_whole_percent_at_every_threshold(name):
    """An oracle inside the file: SHIPS prints SHIPS-RII as an integer for each threshold."""
    t = text(name)
    ints = {f"{int(kt)}/{int(hr)}": int(k)
            for kt, hr, k in re.findall(r"SHIPS Prob RI for\s+(\d+)kt/\s*(\d+)hr RI threshold=\s*(\d+)%", t)}
    assert len(ints) == 8
    for thr, k in ints.items():
        assert st.parse_ships_text(t, filename=name, threshold=thr).whole_percent["RIOD"] == k, thr


def test_another_threshold_is_read_by_its_header():
    ri = st.parse_ships_text(text(NOLO), filename=NOLO, threshold="45/36")
    assert ri.percent["DTOP"] == "36.0" and ri.percent["RIOD"] == "21.0"


def test_filename_round_trip():
    assert st.filename_for("EP182026", dt.datetime(2026, 10, 2, 12)) == RACHEL
    assert st.parse_filename(RACHEL) == ("EP182026", dt.datetime(2026, 10, 2, 12))
    assert st.url_for("AL082026", dt.datetime(2026, 9, 30, 12)) == "https://ftp.nhc.noaa.gov/atcf/stext/" + HANNA
    with pytest.raises(st.ShipsTextError):
        st.parse_filename("26100212EP1826_ships.dat")
    with pytest.raises(st.ShipsTextError):
        st.parse_filename("26133112EP1826_ships.txt")   # month 13


# ---------------------------------------------------------------------------
# Adversarial edits of a real file
# ---------------------------------------------------------------------------

def _matrix_lines(t: str) -> tuple[list[str], int, int]:
    lines = t.splitlines()
    head = next(i for i, line in enumerate(lines) if "RI (kt / h)" in line)
    end = next(i for i in range(head + 2, len(lines)) if not lines[i].strip())
    return lines, head, end


def _permute_columns(t: str, order: list[int], values_too: bool = True) -> str:
    lines, head, end = _matrix_lines(t)
    label, *cols = lines[head].split("|")
    lines[head] = label + "|" + "|".join(f" {cols[i].strip()} " for i in order)
    if values_too:
        for i in range(head + 2, end):
            name, vals = lines[i].split(":", 1)
            v = vals.split()
            lines[i] = name + ":  " + "  ".join(v[j] for j in order)
    return "\n".join(lines) + "\n"


def test_column_order_changed_is_still_read_by_header():
    order = [7, 6, 5, 4, 3, 2, 1, 0]   # 30/24 moves from position 2 to 5
    ri = st.parse_ships_text(_permute_columns(text(NOLO), order), filename=NOLO)
    assert ri.columns[5] == "30/24"
    assert (ri.percent["RIOD"], ri.percent["DTOP"], ri.percent["RIOL"]) == ("19.3", "15.0", "0.3")


def test_a_header_that_moves_without_its_values_changes_what_is_read():
    """The control for the test above: reading by header is what makes the difference --
    and SHIPS's own integer line catches the misread when it is present."""
    order = [7, 6, 5, 4, 3, 2, 1, 0]
    shuffled = _permute_columns(text(NOLO), order, values_too=False)
    with pytest.raises(st.ShipsTextError, match="contradicts"):
        st.parse_ships_text(shuffled, filename=NOLO)   # SHIPS-RII 21.0 under '30/24' vs its 19%
    no_line = "\n".join(line for line in shuffled.splitlines() if "SHIPS Prob RI for 30kt" not in line)
    ri = st.parse_ships_text(no_line, filename=NOLO)
    assert ri.percent["DTOP"] == "36.0"   # the column now headed 30/24 holds 45/36's numbers


def test_a_missing_row_is_none_and_the_rest_survive():
    t = "\n".join(line for line in text(NOLO).splitlines() if not line.strip().startswith("DTOPS:")) + "\n"
    ri = st.parse_ships_text(t, filename=NOLO)
    assert ri.whole_percent["DTOP"] is None and ri.percent["DTOP"] is None and ri.rounding["DTOP"] is None
    assert ri.whole_percent["RIOD"] == 19 and ri.whole_percent["SDCN"] == 11


def test_the_999_sentinel_is_missing_not_a_value_and_not_a_refusal():
    t = text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%", "       DTOPS:   999.0%  999.0%  999.0%")
    assert "999.0%" in t
    ri = st.parse_ships_text(t, filename=NOLO)
    assert ri.whole_percent["DTOP"] is None and ri.whole_percent["RIOD"] == 19
    bad = text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%", "       DTOPS:     3.0%   23.0%  150.0%")
    with pytest.raises(st.ShipsTextError, match="not a probability"):
        st.parse_ships_text(bad, filename=NOLO)


def test_zero_and_hundred_percent():
    t = (text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%", "       DTOPS:     3.0%   23.0%  100.0%")
         .replace("    Logistic:     0.2%    0.6%    0.3%", "    Logistic:     0.2%    0.6%    0.0%"))
    ri = st.parse_ships_text(t, filename=NOLO)
    assert ri.whole_percent["DTOP"] == 100 and ri.whole_percent["RIOL"] == 0


@pytest.mark.parametrize("printed,whole", [("0.4", 0), ("0.5", 1), ("2.5", 3), ("6.5", 7), ("99.5", 100), ("99.4", 99),
                                           ("12.49", 12)])
def test_rounding_is_half_up_in_decimal(printed, whole):
    assert st.round_half_up(printed) == whole
    t = text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%",
                           f"       DTOPS:     3.0%   23.0%   {printed}%")
    assert st.parse_ships_text(t, filename=NOLO).whole_percent["DTOP"] == whole


def test_round_half_up_is_not_python_round():
    assert round(2.5) == 2 and st.round_half_up("2.5") == 3   # banker's rounding would differ
    assert st.round_half_up(Decimal("0.5")) == 1


def test_the_ships_integer_line_decides_a_printed_half_and_must_be_consistent():
    # SHIPS-RII printed 19.3 with integer 19: move it to an x.5 the hidden decimal rounded down
    t = text(NOLO).replace("   SHIPS-RII:    12.2%   21.1%   19.3%", "   SHIPS-RII:    12.2%   21.1%   18.5%")
    t = t.replace("SHIPS Prob RI for 30kt/ 24hr RI threshold=   19%", "SHIPS Prob RI for 30kt/ 24hr RI threshold=   18%")
    ri = st.parse_ships_text(t, filename=NOLO)
    assert ri.whole_percent["RIOD"] == 18 and ri.rounding["RIOD"] == "ships_integer_line"   # half-up says 19
    contradiction = text(NOLO).replace("SHIPS Prob RI for 30kt/ 24hr RI threshold=   19%",
                                       "SHIPS Prob RI for 30kt/ 24hr RI threshold=   23%")
    with pytest.raises(st.ShipsTextError, match="contradicts"):
        st.parse_ships_text(contradiction, filename=NOLO)
    no_line = "\n".join(line for line in text(NOLO).splitlines() if "SHIPS Prob RI for 30kt" not in line)
    ri = st.parse_ships_text(no_line, filename=NOLO)
    assert ri.whole_percent["RIOD"] == 19 and ri.rounding["RIOD"] == "half_up"


# ---------------------------------------------------------------------------
# Identity: a file must be the storm and cycle it is named for
# ---------------------------------------------------------------------------

def test_a_file_whose_banner_names_another_storm_is_refused():
    t = text(RACHEL).replace("RACHEL      EP182026  10/02/26", "RACHEL      EP152026  10/02/26")
    with pytest.raises(st.ShipsTextError, match="banner says EP152026"):
        st.parse_ships_text(t, filename=RACHEL)   # the RI-index header still says EP182026
    t = t.replace("RI INDEX EP182026", "RI INDEX EP152026").replace("(AHI) EP182026", "(AHI) EP152026")
    with pytest.raises(st.ShipsTextError, match="requested as EP182026"):
        st.parse_ships_text(t, filename=RACHEL)   # self-consistent text, wrong file name


def test_a_file_whose_banner_names_another_cycle_is_refused():
    t = text(RACHEL).replace("EP182026  10/02/26  12 UTC", "EP182026  10/02/26  06 UTC")
    with pytest.raises(st.ShipsTextError):
        st.parse_ships_text(t, filename=RACHEL)


def test_a_file_saved_under_another_storms_name_is_refused():
    with pytest.raises(st.ShipsTextError, match="requested as EP152026"):
        st.parse_ships_text(text(RACHEL), filename=NOLO)
    with pytest.raises(st.ShipsTextError):
        st.parse_ships_text(text(RACHEL), expected_storm_id="EP182026", expected_cycle=dt.datetime(2026, 10, 2, 6))


def test_an_ri_index_header_that_disagrees_with_the_banner_is_refused():
    t = text(RACHEL).replace("RI INDEX EP182026 RACHEL     10/02/26  12 UTC", "RI INDEX EP172026 RACHEL     10/02/26  12 UTC")
    assert t != text(RACHEL)
    with pytest.raises(st.ShipsTextError, match="RI-index header"):
        st.parse_ships_text(t, filename=RACHEL)


def test_no_banner_no_forecast():
    t = "\n".join(line for line in text(RACHEL).splitlines() if "EP182026  10/02/26" not in line)
    with pytest.raises(st.ShipsTextError, match="banner"):
        st.parse_ships_text(t, filename=RACHEL)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_a_header_without_the_threshold_is_refused():
    t = text(NOLO).replace("| 30/24 |", "| 30/36 |")
    with pytest.raises(st.ShipsTextError, match="no 30/24 column"):
        st.parse_ships_text(t, filename=NOLO)


def test_a_row_with_a_missing_value_is_refused_not_shifted():
    t = text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%", "       DTOPS:     3.0%   23.0%")
    with pytest.raises(st.ShipsTextError, match="values for 8 columns"):
        st.parse_ships_text(t, filename=NOLO)


def test_an_unreadable_value_is_refused():
    t = text(NOLO).replace("       DTOPS:     3.0%   23.0%   15.0%", "       DTOPS:     3.0%   23.0%   1S.0%")
    with pytest.raises(st.ShipsTextError, match="unreadable"):
        st.parse_ships_text(t, filename=NOLO)


def test_two_matrices_and_duplicate_rows_are_refused():
    t = text(NOLO)
    block = t[t.index("Matrix of RI probabilities"):]
    with pytest.raises(st.ShipsTextError, match="blocks"):
        st.parse_ships_text(t + "\n" + block, filename=NOLO)
    dup = t.replace("       SDCON:", "       DTOPS:")
    with pytest.raises(st.ShipsTextError, match="appears twice"):
        st.parse_ships_text(dup, filename=NOLO)


def test_no_matrix_is_refused():
    t = text(NOLO).replace("Matrix of RI probabilities", "Matrix of something else")
    with pytest.raises(st.ShipsTextError, match="no 'Matrix of RI probabilities' block"):
        st.parse_ships_text(t, filename=NOLO)


def test_crlf_files_parse_identically():
    lf = st.parse_ships_text(text(NOLO), filename=NOLO)
    crlf = st.parse_ships_text(text(NOLO).replace("\n", "\r\n"), filename=NOLO)
    assert lf == crlf
