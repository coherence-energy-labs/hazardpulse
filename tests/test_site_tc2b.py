"""TC2b on the storm card (TC1 program amendment 4): a column only when the record carries a good TC2b, its label and
winds from the record, and the table without it byte-identical to TC1's own."""
from __future__ import annotations

import copy

from hazardpulse.site.pages import hurricane


def _storm(tc2b=None):
    tc1 = {"cycle": "2026-10-07T06:00:00Z",
           "TC1": {"12": {"lat": 23.0, "lon": -91.0, "vmax_kt": 42.0}, "24": {"lat": 24.0, "lon": -90.0, "vmax_kt": 50.0}},
           "OFCL": {"12": {"lat": 23.1, "lon": -91.1, "vmax_kt": 45.0}, "24": {"lat": 24.1, "lon": -90.1, "vmax_kt": 55.0}}}
    if tc2b is not None:
        tc1["TC2b"] = tc2b
    return {"storm_id": "AL092026", "tc1": tc1}


GOOD = {"status": "ok", "label": "TC2b", "intensity": {"12": 49.5, "24": 65.0}, "moved_tc1": True, "shift_24h_kt": 15.0}


def test_a_good_tc2b_record_adds_its_column_with_its_own_label_and_winds():
    html = hurricane._tc1_table(_storm(GOOD))
    # winds are whole knots on the card, as TC1's and NHC's are (49.5 -> 50)
    assert "With our RI model (TC2b)" in html and ">65 kt<" in html
    row12 = html[html.find(">12 h<"):html.find(">24 h<")]
    assert row12.count(" kt<") == 3 and ">50 kt<" in row12            # TC1 42, TC2b 50, NHC 45
    assert "moved TC1 by 15 kt at 24 hours" in html
    relabelled = hurricane._tc1_table(_storm(dict(GOOD, label="X9")))       # the label is the record's, not the page's
    assert "With our RI model (X9)" in relabelled and "TC2b" not in relabelled.split("caption")[0]


def test_no_tc2b_or_a_failed_one_leaves_tc1s_table_exactly_as_it_was():
    plain = hurricane._tc1_table(_storm())
    assert "With our RI model" not in plain
    assert hurricane._tc1_table(_storm({"status": "error: x"})) == plain
    agree = dict(GOOD, moved_tc1=False, shift_24h_kt=None, intensity={"12": 42.0, "24": 50.0})
    html = hurricane._tc1_table(_storm(copy.deepcopy(agree)))
    assert "this cycle it agrees with TC1" in html
