"""hazardpulse.com is one rendering of the published artifacts (hazardpulse.site.build).

Two kinds of test:
* the COMMITTED site: every page is current (a fixed point of the artifacts), obeys the CSP's rules,
  carries complete metadata and the one shell, and every data file is strict JSON;
* a SYNTHETIC site: each page says what its artifacts say -- the right storm's number, a withheld forecast
  withheld, a live score only once events exist, research diagnostics never presented as the forecast.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from hazardpulse.site import build, fmt, maps, places, shell
from hazardpulse.site.data import SiteData, basin_of, official_center
from hazardpulse.site.pages import common, earthquake, hurricane, overview, record, research

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
HTML = sorted(p for p in DIST.rglob("*.html"))


# ------------------------------------------------------------------------------------------------
# the committed site
# ------------------------------------------------------------------------------------------------

def test_the_committed_site_is_a_fixed_point_of_its_artifacts():
    """Every page equals what the artifacts and the shell render now: no stale number, no drifted shell."""
    assert build.check_site() == []


def test_every_registered_page_exists_and_every_page_is_registered():
    for page in shell.PAGES.values():
        assert shell.static_page_file(page.path).exists(), page.path
    on_disk = {"/" + str(p.relative_to(DIST)).replace("\\", "/") for p in HTML}
    registered = {("/404.html" if p.path == "/404.html" else p.path + "index.html") for p in shell.PAGES.values()}
    registered = {r.replace("//", "/") for r in registered}
    assert on_disk == registered, on_disk ^ registered


def test_no_page_has_an_inline_style_or_executable_inline_script():
    """The CSP allows neither: an inline style is silently dropped and logged as an error on every view."""
    for p in HTML:
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"\sstyle\s*=", text, re.I), p
        for m in re.finditer(r"<script\b([^>]*)>", text, re.I):
            attrs = m.group(1)
            assert "src=" in attrs or 'type="application/ld+json"' in attrs, (p, attrs)


def test_every_page_has_complete_metadata_and_the_one_shell():
    header = re.compile(r'<header class="site-header" role="banner">.*?</header>', re.S)
    footer = re.compile(r'<footer class="site-footer" role="contentinfo">.*?</footer>', re.S)
    headers, footers = set(), set()
    for page in shell.PAGES.values():
        text = shell.static_page_file(page.path).read_text(encoding="utf-8")
        assert text.count("<h1") == 1, page.path
        assert f"<title>{fmt.esc(shell.full_title(page))}</title>" in text, page.path
        assert f'<meta name="description" content="{fmt.esc(page.description)}">' in text, page.path
        assert 90 <= len(page.description) <= 160, page.path      # a full search snippet, never cut
        assert '<meta property="og:image" content="https://hazardpulse.com/assets/og-card.png">' in text
        assert ('<meta name="robots" content="noindex">' in text) == page.noindex, page.path
        if not page.noindex:
            assert f'<link rel="canonical" href="https://hazardpulse.com{page.path}">' in text, page.path
        assert 'id="main"' in text and 'class="skip-link"' in text
        h = header.search(text).group(0)
        headers.add(re.sub(r' aria-current="page"', "", h))
        footers.add(footer.search(text).group(0))
    assert len(headers) == 1 and len(footers) == 1          # one header, one footer, everywhere


def test_versioned_assets_carry_the_hash_of_the_file_they_name():
    """/assets is cached for a year: a URL must change whenever its file does."""
    text = (DIST / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'/assets/([^"?#]+)\?v=([0-9a-f]{10})', text)
    assert {"styles.css", "site-shell.js", "hp-mark.svg"} <= {r[0] for r in refs}
    for name, v in refs:
        data = (DIST / "assets" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest()[:10] == v, name


def test_every_data_file_is_strict_json():
    """A bare NaN is valid to Python's json module and invalid to every browser and Worker: the research
    summary's ten NaNs made /api/v1/laic/* a 404 (2026-10-04)."""
    def reject(token):
        raise ValueError(f"non-finite number {token}")
    for p in sorted((DIST / "data").rglob("*.json")):
        json.loads(p.read_text(encoding="utf-8"), parse_constant=reject)
    json.loads((DIST / "speculation-rules.json").read_text(encoding="utf-8"), parse_constant=reject)


def test_the_headers_allow_nothing_from_another_origin_and_prefetch_by_header():
    headers = (DIST / "_headers").read_text(encoding="utf-8")
    csp = re.search(r"Content-Security-Policy: (.*)", headers).group(1)
    assert "http" not in csp and "unsafe-inline" not in csp
    assert 'Speculation-Rules: "/speculation-rules.json"' in headers
    assert "application/speculationrules+json" in headers


def test_the_sitemap_lists_every_indexable_page_and_nothing_else():
    sitemap = (DIST / "sitemap.xml").read_text(encoding="utf-8")
    locs = set(re.findall(r"<loc>https://hazardpulse.com([^<]+)</loc>", sitemap))
    assert locs == {p.path for p in shell.PAGES.values() if not p.noindex}


def test_the_map_base_drawings_are_current():
    for proj in (maps.WORLD, maps.CONUS):
        assert (DIST / "assets" / "maps" / f"{proj.name}.svg").read_text(encoding="utf-8") == maps.base_svg(proj)


def test_every_quality_check_the_engine_runs_is_described_in_words():
    from hazardpulse.gates import engine
    ids = {fn.__name__.split("_", 1)[0].upper() for fn in engine._GATES}
    described = {k.split("_", 1)[0] for k in record.GATE_TEXT}
    assert ids <= described, ids - described


# ------------------------------------------------------------------------------------------------
# formatting, places, maps
# ------------------------------------------------------------------------------------------------

def test_a_chance_never_reads_as_zero_point_zero():
    assert fmt.pct(0.0003) == "&lt;0.1%" and fmt.pct(0.0) == "0%" and fmt.pct(0.21054) == "21.1%"
    assert fmt.pct(None) == "&mdash;" and fmt.pct(float("nan")) == "&mdash;"
    assert fmt.pts(0.019) == "+1.9 pts" and fmt.pts(-0.112) == "−11.2 pts" and fmt.pts(0.0001) == "no change"
    assert fmt.utc("2026-10-04T09:59:43.644490Z") == "4 Oct 2026, 09:59 UTC"
    assert fmt.utc("20261004_073035 UTC", with_date=False) == "07:30 UTC"
    assert [fmt.level(p) for p in (0.0, 0.009, 0.01, 0.2, 0.5)] == ["p1", "p1", "p2", "p4", "p6"]


def test_an_interval_is_shown_only_when_it_informs():
    wide = {"confidence_lo": 0.0, "confidence_hi": 0.25, "uncertainty_class": "wide"}
    tight = {"confidence_lo": 0.08, "confidence_hi": 0.12, "uncertainty_class": "tight",
             "receipt": {"coverage_target": 0.9}}
    assert fmt.interval(wide) == "" and fmt.interval({}) == ""
    assert fmt.interval(tight) == "90% range 8.0%&ndash;12.0%"


def test_places_name_points_the_way_seismologists_and_the_weather_service_do():
    assert places.fe_number(48, 12) == 543 and places.fe_region(48, 12) == "Germany"      # USGS / ObsPy reference
    assert places.fe_region(-30, -60) == "Northeastern Argentina"
    assert places.fe_region(-5, 151) == "New Britain Region, P.N.G."
    assert re.fullmatch(r"(near |\d+ mi [NESW]{1,3} of )Moore, OK", places.describe_us(35.30, -97.47))
    assert places.describe_far(23.5, -175.9).endswith("W of Honolulu, HI")
    assert places.coords(-21, 169, 0) == "21°S 169°E"


def test_the_heat_grid_merges_runs_and_refuses_a_grid_that_does_not_fit_its_domain():
    dom = {"lat_min": -60, "lon_min": -180, "dlat": 2, "dlon": 2, "n_lat": 2, "n_lon": 3}
    svg = maps.heat_grid(maps.WORLD, {"forecast_domain": dom, "probability_grid": "0.2,0.2,0,0.002,0,0.05"})
    assert svg.count("<rect") == 3                  # 0.2,0.2 merged; 0.002 and 0.05 alone; zeros unshaded
    assert re.findall(r'class="heat (h\d)"', svg) == ["h5", "h1", "h4"]    # 20%+, 0.1-0.3%, 3-10%
    with pytest.raises(ValueError):
        maps.heat_grid(maps.WORLD, {"forecast_domain": dom, "probability_grid": "0.1,0.2"})


def test_a_storm_is_credited_to_the_agency_responsible_where_it_is_now():
    assert basin_of({"basin": "EP", "lon": -175.9}) == "Central Pacific"
    assert basin_of({"basin": "EP", "lon": -113.0}) == "East Pacific"
    assert official_center({"basin": "EP", "lon": -175.9})[0] == "Central Pacific Hurricane Center"
    assert official_center({"basin": "WP", "lon": 147.0})[0] == "Joint Typhoon Warning Center"


# ------------------------------------------------------------------------------------------------
# a synthetic site
# ------------------------------------------------------------------------------------------------

ISSUED = "2026-10-04T06:00:00Z"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def site(tmp_path):
    dist = tmp_path / "dist"
    grid = ["0.0001"] * 6
    grid[4] = "0.21"
    _write(dist / "data" / "replay" / "eq_fcst_20261004_0600.json", {
        "forecast_id": "eq_fcst_20261004_0600", "issued_at": ISSUED, "model_version": "eq_S1-test",
        "forecast_domain": {"lat_min": -22, "lon_min": 166, "dlat": 2, "dlon": 2, "n_lat": 2, "n_lon": 3},
        "probability_grid": ",".join(grid), "top_probability": 0.21,
        "source_catalog": {"n_recent_events": 1862},
        "active_cells": [{"lat": -19.0, "lon": 169.0, "probability": 0.21, "n_events": 43, "max_mag": 6.6,
                          "lambda_long": 0.02, "lambda_short": 0.12, "b_value": 1.1,
                          "days_to_criticality": 3296.0, "conditions_met": 2}]})
    _write(dist / "data" / "replay" / "eq_fcst_20261004_0000.json", {
        "forecast_id": "eq_fcst_20261004_0000", "issued_at": "2026-10-04T00:00:00Z", "top_probability": 0.19})
    _write(dist / "data" / "live-pulse.json", {"updated_at": ISSUED, "hazards": [
        {"key": "eq", "forecast_id": "eq_fcst_20261004_0600", "gate_status": "pass"},
        {"key": "hu", "forecast_id": "hu_fcst_20261004_0959"},
        {"key": "to", "forecast_id": "to_fcst_20261004_0743"}]})
    _write(dist / "data" / "live-storms.json", {"updated_at": "2026-10-04T09:59:43Z",
                                                "forecast_id": "hu_fcst_20261004_0959", "storms": [
        {"storm_id": "EP152026", "storm_name": "NOLO", "basin": "EP", "lat": 23.5, "lon": -175.9, "vmax_kt": 95,
         "category": "Category 2", "ri_probability": 0.01, "ri_source": "noaa_aid_stack",
         "ri_source_label": "NOAA DTOPS", "issue_time": "2026-10-04T06:00:00"},
        {"storm_id": "WP262026", "storm_name": "Choi-Wan", "basin": "WP", "lat": 22.9, "lon": 147.0, "vmax_kt": 125,
         "category": "Category 4", "ri_probability": 0.0177, "ri_source": "v8.2",
         "ri_source_label": "HazardPulse v8.2", "issue_time": "2026-10-04T06:00:00"}]})
    _write(dist / "data" / "live-tornadoes.json", {"updated_at": "2026-10-04T07:43:44Z",
                                                   "forecast_id": "to_fcst_20261004_0743", "storms": [
        {"storm_id": "1", "lat": 26.02, "lon": -97.25, "tornado_probability": 0.0018,
         "coherence_diagnostics": {"singularity_conditions_met": 4}, "v3": {"model": "v3_w"}}]})
    _write(dist / "data" / "evidence" / "gate-decisions.json", {"decisions": [
        {"forecast_id": "eq_fcst_20261004_0600", "decision": "pass", "reasons": [], "warnings": [],
         "emitted_at": ISSUED},
        {"forecast_id": "hu_fcst_20261004_0959", "decision": "degrade", "reasons": [],
         "warnings": ["model calibration not yet measured"], "emitted_at": "2026-10-04T09:59:43Z"},
        {"forecast_id": "to_fcst_20261004_0743", "decision": "block",
         "reasons": ["source data is stale (14.0h > 12h)"], "warnings": [], "emitted_at": "2026-10-04T07:43:44Z"}]})
    _write(dist / "data" / "cross-modality-summary.json", {"generated_at": ISSUED, "n_eq_events_used": 200,
        "analyses": [
            {"analysis_id": "geomagnetic_precursor_kp", "n_targets": 200, "n_controls": 193, "delta_mean": 0.03,
             "bootstrap_ci_delta_lo": -0.25, "bootstrap_ci_delta_hi": 0.31},
            {"analysis_id": "solar_flare_precursor_xray", "n_targets": 0, "delta_mean": None,
             "bootstrap_ci_delta_lo": None, "bootstrap_ci_delta_hi": None},
            {"analysis_id": "imf_bz_precursor", "n_targets": 200, "n_controls": 193, "delta_mean": 0.9,
             "bootstrap_ci_delta_lo": 0.2, "bootstrap_ci_delta_hi": 1.5}]})
    _write(dist / "data" / "verification-summary.json", {"score_as_of": ISSUED, "system": {"alerts": []},
                                                         "hazards": []})
    d = SiteData(root=tmp_path, dist=dist)
    d.__dict__["evidence"] = {"earthquake": None, "hurricane": None, "tornado": None, "_errors": []}
    return d


def test_the_hurricane_headline_names_the_storm_its_number_belongs_to(site):
    """The home card said "NOLO is being tracked" beside Choi-Wan's 1.8% (2026-10-04)."""
    head = site.headlines["hu"]
    assert head.where == "Choi-Wan" and head.probability == pytest.approx(0.0177)
    assert head.source == "HazardPulse v8.2"
    cards = overview.hazard_cards(site)
    assert "rapid intensification: Choi-Wan" in cards and "West Pacific" in cards and "NOLO" not in cards


def test_the_change_is_against_the_previous_forecast_in_points(site):
    head = site.headlines["eq"]
    assert head.previous == pytest.approx(0.19)
    assert common.since_previous(head).startswith("+2.0 pts since the previous forecast")


def test_quality_check_outcomes_are_the_recorded_ones_and_a_blocked_forecast_is_withheld(site):
    assert site.headlines["hu"].gate == "degrade" and site.headlines["to"].gate == "block"
    assert "Published with warnings" in common.gate_notice(site.headlines["hu"])
    withheld = build._withheld(site, "/live/tornado/")
    assert withheld and "This forecast is withheld" in withheld and "Input data older than the limit" in withheld
    assert "0.2%" not in withheld and "<0.1%" not in withheld
    assert build._withheld(site, "/live/earthquake/") is None


def test_the_earthquake_page_explains_the_forecast_and_keeps_research_statistics_apart(site):
    html = earthquake.page(site)
    assert "Earthquakes cannot be predicted" in html
    assert "account for 86% of this cell&rsquo;s modelled rate" in html and "since 1973" in html
    assert "Research statistics (not used by the forecast)" in html
    for banned in ("criticality", "Days to criticality", "singularity", "Threat brief", "Operational posture"):
        assert banned not in html, banned
    assert "No final test is bound to the served model in this build" in html


def test_the_research_page_reads_each_verdict_from_its_interval(site):
    html = research.page(site)
    assert "No difference found" in html                              # interval contains zero
    assert "Not run: no data for these events in this window" in html  # n = 0, no statistics
    assert "Higher before events in this sample" in html              # interval above zero
    assert "Not run yet: needs a curated list of events" in html      # hypotheses never run are listed too
    assert "NaN" not in html


def test_the_status_page_is_built_from_the_records_not_typed(site):
    html = record.status(site)
    # as of the newest forecast (09:59): earthquake 4 h old (limit 12), tornado 2 h (limit 6) -> all current
    assert html.count('<span class="chip good">Current</span>') == 3 and "Overdue</span>" not in html
    assert "Published with warnings" in html and "Blocked" in html
    assert "Calibration on live forecasts not yet measured" in html
    assert "99.4%" not in html                                        # the old page's typed uptime


def test_the_status_page_flags_a_forecast_past_its_window(site):
    _write(site.dist / "data" / "live-storms.json", {"updated_at": "2026-10-05T09:00:00Z",
                                                     "forecast_id": "hu_fcst_20261005_0900", "storms": []})
    fresh = SiteData(root=site.root, dist=site.dist)
    fresh.__dict__["evidence"] = site.evidence
    html = record.status(fresh)                                       # now 'as of' 5 Oct 09:00
    assert "Delayed:" in html and "Earthquake" in html.split("Delayed:", 1)[1].split("</p>", 1)[0]


def test_the_hurricane_page_says_where_each_number_comes_from(site):
    html = hurricane.page(site)
    assert "NOAA publishes no rapid-intensification guidance for this basin" in html
    assert "Central Pacific Hurricane Center" in html and "Joint Typhoon Warning Center" in html
    assert html.index("Choi-Wan") < html.index("Nolo")                # highest chance first
