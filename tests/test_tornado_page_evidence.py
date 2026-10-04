"""The live tornado pages: the ledger keeps updating, and a storm's numbers are measured, not invented."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hazardpulse.site import build
from hazardpulse.site.data import SiteData
from hazardpulse.site.pages import tornado
from hazardpulse.verification import evidence_pages as ep

ROOT = Path(__file__).resolve().parents[1]


def _entry(ts: str, version: str, h: str) -> str:
    return json.dumps({"timestamp": ts, "model_version": version, "n_storms": 3, "top_probability": 0.1,
                       "prev_hash": "0" * 64, "hash": h * 64}) + "\n"


def _site(tmp_path: Path, ledger: str = "") -> SiteData:
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "data" / "tornado-ledger.jsonl").write_text(ledger, encoding="utf-8")
    (dist / "data" / "live-tornadoes.json").write_text(json.dumps({"model_version": "tornado_v3-bbbbbbbbbbbb"}),
                                                       encoding="utf-8")
    return SiteData(root=tmp_path, dist=dist)


def test_the_ledger_keeps_updating_a_page_it_already_rendered(tmp_path):
    page = (ROOT / "dist" / "verification" / "tornado" / "index.html").read_text(encoding="utf-8")
    d = _site(tmp_path, _entry("2026-10-02T00:00:00Z", "tornado_v3-aaaaaaaaaaaa", "a"))
    first = build._apply_ledger(d, page)
    assert "2026-10-02T00:00:00Z" in first and "tornado_v3-aaaaaaaaaaaa" in first
    assert "earlier model" in first                       # an entry of a model that no longer serves says so
    # the second run is the one that silently did nothing from 2026-03-31: the page was already rendered
    with (d.dist / "data" / "tornado-ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(_entry("2026-10-03T00:00:00Z", "tornado_v3-bbbbbbbbbbbb", "b"))
    second = build._apply_ledger(SiteData(root=tmp_path, dist=d.dist), first)
    assert "2026-10-03T00:00:00Z" in second and "tornado_v3-bbbbbbbbbbbb" in second
    assert second.index("2026-10-03T00:00:00Z") < second.index("2026-10-02T00:00:00Z")   # newest first
    assert second.count("<table>") == first.count("<table>")
    strip = lambda s: ep._marker_re("rows", "hp-ledger").sub("", s)
    assert strip(first) == strip(second)                  # nothing outside the ledger block moved


def test_a_page_without_the_ledger_markers_is_an_error_not_a_silent_no_op(tmp_path):
    with pytest.raises(ep.PageBlockError):
        build._apply_ledger(_site(tmp_path), "<html><body>no markers</body></html>")


_TABLE = {"bins": [{"lo": 0.0, "hi": 0.05, "n": 1460000, "pos": 400, "observed": 0.0003, "mean_forecast": 0.0004,
                    "observed_ci": [0.00025, 0.0003]},
                   {"lo": 0.05, "hi": 1.0, "n": 900, "pos": 120, "observed": 0.1333, "mean_forecast": 0.12,
                    "observed_ci": [0.112, 0.156]}]}


class _Data:
    def __init__(self, ev):
        self.evidence = {"tornado": ev}


def _storm(p60: float = 0.12) -> dict:
    return {"storm_id": "77", "lat": 35.2, "lon": -97.4, "valid_time": "20250506_180039 UTC",
            "tornado_probability": p60, "model_version": "tornado_v3-aaaaaaaaaaaa",
            "v3": {"model": "v3_w", "probability_60min": p60, "probability_30min": 0.07, "probability_90min": 0.15,
                   "probability_ef2plus_60min": 0.02, "nws_warning": {"active": True, "minutes_since_issue": 4.0},
                   "drivers": [{"input": "p_maxllaz", "label": "low-level rotation (max azimuthal shear)",
                                "value": 0.012, "log_odds": 0.84},
                               {"input": "p_ps_tor", "label": "NOAA ProbTor", "value": 12.0, "log_odds": -0.21}],
                   "inputs": {"p_ps_tor": 12.0, "p_mucape": 2500.0, "p_ps": 0.0}}}


def test_a_storm_shows_its_measured_outcome_rate_never_an_invented_one():
    ev = {"model_version": "tornado_v3-aaaaaaaaaaaa", "reliability": _TABLE, "fallback": None}
    html = tornado._row(1, _storm(), _Data(ev))
    assert "Historical analogs" not in html and "percentile" not in html
    assert "Of the 900 storm observations in 2025" in html and "13.33%" in html
    assert "between 5.0% and 100%" in html and "11.20%" in html and "15.60%" in html
    # the quiet storms: a 0.03% rate is printed as such, never as "0.0%"
    quiet = tornado._row(1, _storm(0.001), _Data(ev))
    assert "0.03%" in quiet and "0.0%" not in quiet
    assert "low-level rotation (max azimuthal shear)" in html and "raises the chance" in html and "+0.84" in html
    assert "NOAA ProbTor" in html and "12%" in html and "lowers the chance" in html     # ProbTor is a percent
    assert "NWS tornado warning" in html and "in effect" in html
    # a probability in a range the test never populated: no rate is shown
    empty = {"bins": [{"lo": 0.0, "hi": 0.05, "n": 10, "pos": 0, "observed": 0.0, "mean_forecast": 0.01},
                      {"lo": 0.05, "hi": 1.0, "n": 0, "pos": 0, "observed": None, "mean_forecast": None}]}
    html = tornado._row(1, _storm(), _Data({**ev, "reliability": empty}))
    assert "No storm in the 2025 test was scored in this range" in html


def test_no_research_diagnostic_or_missing_value_masquerades_as_a_storm_property():
    s = _storm()
    s["coherence_diagnostics"] = {"singularity_conditions_met": 4, "tau": 1.57}
    html = tornado._row(1, s, _Data(None))
    assert "CRITICAL" not in html and "singularity" not in html.lower() and "coherence" not in html.lower()
    assert "not reported" in html                         # an attribute the storm lacks is not a zero
    assert "ProbSevere (any severe" in html and "p_ps\"" not in html


def test_a_storm_without_a_bound_table_says_so():
    html = tornado._row(1, _storm(), _Data(None))
    assert "No 2025 test table is bound to the model that scored this storm" in html
