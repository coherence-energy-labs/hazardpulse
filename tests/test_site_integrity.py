from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_critical_public_pages_no_longer_contain_placeholder_content() -> None:
    pages = [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/hurricane/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "Hurricane Alice" not in text
        assert "TEST DATA" not in text
        assert "P(M6+ in 90 days)" not in text
        assert "hazardpulse.io" not in text


def test_key_public_pages_use_hazardpulse_branding() -> None:
    pages = [
        "dist/index.html",
        "dist/404.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/tornado/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "https://coherenceenergylabs.com" not in text, rel_path
        assert ">Coherence Energy Labs<" not in text, rel_path
        assert "Built with Coherence Lang" not in text, rel_path


def test_live_pulse_confidence_ranges_are_missing_or_sane() -> None:
    pulse = json.loads(_read_text("dist/data/live-pulse.json"))
    for hazard in pulse["hazards"]:
        lo = hazard.get("conf_lo")
        hi = hazard.get("conf_hi")
        probability = hazard.get("probability")
        if lo is None or hi is None:
            continue
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= probability <= hi


def test_public_license_asset_exists() -> None:
    assert (ROOT / "dist" / "COMMERCIAL_LICENSE.md").exists()


def test_remaining_dist_metadata_uses_primary_domain() -> None:
    for rel_path in [
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/methods/index.html",
        "dist/registry/index.html",
        "dist/ops/status/index.html",
        "dist/verification/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/tornado/index.html",
        "dist/legal/disclaimer/index.html",
        "dist/feed.xml",
        "dist/sitemap.xml",
        "dist/robots.txt",
    ]:
        assert "hazardpulse.io" not in _read_text(rel_path), rel_path


def test_key_public_pages_have_structural_basics() -> None:
    pages = [
        "dist/index.html",
        "dist/404.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
    ]
    for rel_path in pages:
        text = _read_text(rel_path)
        assert "<title>" in text, rel_path
        assert '<meta name="description"' in text, rel_path
        assert '<link rel="canonical"' in text, rel_path
        assert '<meta property="og:title"' in text, rel_path
        assert '<meta name="twitter:title"' in text, rel_path
        assert 'class="skip-link"' in text, rel_path
        assert 'role="banner"' in text, rel_path
        assert 'id="main"' in text, rel_path
        assert 'role="contentinfo"' in text, rel_path
        assert len(re.findall(r"<h1\b", text)) == 1, rel_path


def test_key_surfaces_publish_structured_data_and_prefetch_rules() -> None:
    for rel_path in [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/hurricane/index.html",
        "dist/evidence/index.html",
    ]:
        text = _read_text(rel_path)
        assert 'type="application/ld+json"' in text, rel_path
        assert 'type="speculationrules"' in text, rel_path


def test_key_public_pages_have_no_encoding_garbage() -> None:
    bad_fragments = ["ï¿½", "�", "Â·", "â€”", "â†’"]
    for rel_path in [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/legal/disclaimer/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
        "dist/404.html",
        "dist/feed.xml",
        "dist/sitemap.xml",
    ]:
        text = _read_text(rel_path)
        for fragment in bad_fragments:
            assert fragment not in text, (rel_path, fragment)


def test_stylesheet_uses_encoding_safe_generated_labels() -> None:
    text = _read_text("dist/assets/styles.css")
    for fragment in [
        'content: " \\25BE";',
        'content: "\\26A0\\00A0ALERT";',
        'content: "\\26A0\\00A0NEARBY HAZARD";',
        'content: "\\26A0\\00A0ADVISORY";',
    ]:
        assert fragment in text, fragment


def test_worker_is_in_deploy_path() -> None:
    worker_path = ROOT / "src" / "worker.js"
    assert worker_path.exists(), "src/worker.js must exist for production deploys"
    wrangler_toml = _read_text("wrangler.toml")
    assert 'main = "./src/worker.js"' in wrangler_toml


def test_personalized_live_pages_are_not_publicly_cached() -> None:
    headers = _read_text("dist/_headers")
    assert "/live/*" in headers
    assert "Cache-Control: private, no-cache, no-store, must-revalidate" in headers
    assert "/data/*" in headers
    assert "X-Robots-Tag: noindex, nofollow" in headers
    assert "X-Robots-Tag: index, follow" not in headers


def test_live_earthquake_forecast_references_existing_replay() -> None:
    pulse = json.loads(_read_text("dist/data/live-pulse.json"))
    eq = next(h for h in pulse["hazards"] if h["key"] == "eq")
    forecast_id = eq.get("forecast_id")
    assert forecast_id, "earthquake forecast_id should be present"
    replay_path = ROOT / "dist" / "data" / "replay" / f"{forecast_id}.json"
    assert replay_path.exists(), replay_path

    earthquake_page = _read_text("dist/live/earthquake/index.html")
    assert f"/data/replay/{forecast_id}.json" in earthquake_page

    index_page = _read_text("dist/index.html")
    assert forecast_id in index_page


def test_live_forecast_ids_reference_existing_replay_artifacts() -> None:
    pulse = json.loads(_read_text("dist/data/live-pulse.json"))
    for hazard in pulse["hazards"]:
        forecast_id = hazard.get("forecast_id")
        if not forecast_id:
            continue
        replay_path = ROOT / "dist" / "data" / "replay" / f"{forecast_id}.json"
        assert replay_path.exists(), replay_path


def test_verification_summary_tracks_storage_and_scoring_status() -> None:
    payload = json.loads(_read_text("dist/data/verification-summary.json"))
    assert payload["generated_at"]
    assert payload["score_as_of"]
    assert "system" in payload
    hazards = {item["key"]: item for item in payload["hazards"]}
    for key in ["eq", "hu", "to"]:
        item = hazards[key]
        assert item["verification_status"]
        assert item["verification_status_label"]
        assert item["forecast_storage"]["n_replay_artifacts"] >= 0
        assert "n_matured_forecasts" in item["forecast_storage"]
        assert "n_scored_forecasts" in item["forecast_storage"]
        assert "recommended_action" in item


def test_verification_rollups_are_written_to_results_storage() -> None:
    for rel_path in [
        "results/verification/system/summary.json",
        "results/verification/earthquake/live_rollup.json",
        "results/verification/hurricane/live_rollup.json",
        "results/verification/tornado/live_rollup.json",
    ]:
        path = ROOT / rel_path
        assert path.exists(), rel_path
        assert path.read_text(encoding="utf-8").strip(), rel_path


def test_verification_page_uses_explicit_scoring_status_language() -> None:
    text = _read_text("dist/verification/index.html")
    assert "Every metric is computed from resolved outcomes - not cherry-picked examples." not in text
    assert "If a live model is not" in text
    assert "scored yet, this page says so directly." in text
    assert "scoring backlogs" in text


def test_evidence_artifacts_use_real_records() -> None:
    evidence_files = [
        "dist/data/evidence/prediction-ledger.json",
        "dist/data/evidence/provenance-envelopes.json",
        "dist/data/evidence/gate-decisions.json",
        "dist/evidence/index.html",
    ]
    forbidden = [
        "example_hash",
        "eq_input_hash",
        "hu_input_hash",
        "to_input_hash",
        "trace_eq_20260313_0300",
        "gate_eq_20260313_0300_001",
        "sha256:eq_",
        "sha256:hu_",
        "sha256:to_",
    ]
    for rel_path in evidence_files:
        text = _read_text(rel_path)
        for token in forbidden:
            assert token not in text, (rel_path, token)


def test_public_html_has_no_inline_executable_scripts() -> None:
    for html_path in (ROOT / "dist").rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        for match in re.finditer(r"<script\b([^>]*)>", text, re.IGNORECASE):
            attrs = match.group(1)
            if "src=" in attrs:
                continue
            if 'type="application/ld+json"' in attrs:
                continue
            if 'type="speculationrules"' in attrs:
                continue
            if 'type="application/json"' in attrs:
                continue
            raise AssertionError(f"unexpected inline executable script in {html_path}")


def test_public_html_theme_toggle_labels_are_consistent() -> None:
    for html_path in (ROOT / "dist").rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        assert 'aria-label="Switch to light mode"' not in text, html_path


def test_public_pages_ship_theme_bootstrap_assets() -> None:
    for rel_path in [
        "dist/404.html",
        "dist/index.html",
        "dist/api/index.html",
        "dist/evidence/index.html",
        "dist/legal/disclaimer/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
        "dist/methods/index.html",
        "dist/ops/status/index.html",
        "dist/registry/index.html",
        "dist/verification/index.html",
        "dist/verification/tornado/index.html",
    ]:
        text = _read_text(rel_path)
        assert '<script src="/assets/site-shell.js?v=2"></script>' in text, rel_path
        assert '<link rel="stylesheet" href="/assets/styles.css?v=9">' in text, rel_path


def test_map_surfaces_include_user_marker_svg_node() -> None:
    for rel_path in [
        "dist/assets/world-map-base.svg",
        "dist/assets/world-map-inline.svg",
        "dist/live/earthquake/index.html",
        "dist/live/tornado/index.html",
    ]:
        text = _read_text(rel_path)
        assert 'class="user-marker"' in text, rel_path
        assert 'class="user-pin"' in text, rel_path
        assert 'class="user-ring"' in text, rel_path


def test_depth_toggle_hides_root_sections_without_leaving_ghost_layout() -> None:
    stylesheet = _read_text("dist/assets/styles.css")
    assert '.depth-content > section[data-depth="simple"]' in stylesheet
    assert '.depth-content > section[data-depth="technical"]' in stylesheet
    assert '.depth-content > div[data-depth="simple"]' in stylesheet
    assert '.depth-content > div[data-depth="technical"]' in stylesheet
    assert "display: none;" in stylesheet


def test_stylesheet_dark_mode_uses_theme_attribute() -> None:
    stylesheet = _read_text("dist/assets/styles.css")
    assert 'body[data-theme="dark"]' in stylesheet
    assert "body:has(.theme-toggle:checked)" not in stylesheet


def test_sitemap_lastmod_matches_live_publish_date() -> None:
    pulse = json.loads(_read_text("dist/data/live-pulse.json"))
    publish_date = pulse["updated_at"][:10]
    sitemap = _read_text("dist/sitemap.xml")
    for route in [
        "https://hazardpulse.com/",
        "https://hazardpulse.com/live/",
        "https://hazardpulse.com/live/earthquake/",
        "https://hazardpulse.com/live/hurricane/",
        "https://hazardpulse.com/live/tornado/",
        "https://hazardpulse.com/evidence/",
    ]:
        assert f"<loc>{route}</loc><lastmod>{publish_date}</lastmod>" in sitemap


def test_worker_api_smoke() -> None:
    result = subprocess.run(
        ["node", str(ROOT / "tests" / "worker_api_check.mjs")],
        capture_output=True,
        text=True,
        check=False,
        # Never inherit stdin: under pytest in a console-less Windows session the inherited
        # handle is invalid and CreateProcess fails with WinError 6 before node even starts.
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tornado_scoring_runs_ml_when_trained_model_exists() -> None:
    """Guard against silent regression to physics-only fallback.

    The pre-trained GBT at results/models/tornado_gbt_v1.json must be used
    whenever it exists. If the latest tornado replay artifact shows
    scoring_tier = "tier2_analytic" or "tier3_ps_only" while the model file
    is present, the ML path is broken.
    """
    model_path = ROOT / "results" / "models" / "tornado_gbt_v1.json"
    if not model_path.exists():
        return  # No model file → nothing to validate

    replay_dir = ROOT / "dist" / "data" / "replay"
    replays = sorted(replay_dir.glob("to_fcst_*.json"))
    assert replays, "No tornado replay artifacts found"

    latest = json.loads(replays[-1].read_text(encoding="utf-8"))
    tier = latest.get("scoring_tier")
    assert tier == "tier1_ml", (
        f"Tornado replay {replays[-1].name} has scoring_tier={tier!r} but "
        "results/models/tornado_gbt_v1.json exists. The trained GBT should "
        "be running. Likely causes: HRRR unreachable, operational_storm "
        "import failed, or tier-selection regression."
    )


def test_hurricane_training_data_present() -> None:
    """The hurricane scorer hard-fails without this file (by design)."""
    training_path = ROOT / "results" / "hurricane_operational_ri_2000_2024_al_sst.jsonl"
    assert training_path.exists(), (
        f"Hurricane training data missing at {training_path}. "
        "Run scripts/build_hurricane_training_data.py to (re)build."
    )
    # Sanity-check: at least 50k cases expected (IBTrACS 2000-2024 × 6 basins).
    line_count = sum(1 for _ in training_path.open(encoding="utf-8"))
    assert line_count > 50_000, (
        f"Hurricane training data is suspiciously small ({line_count} lines). "
        "Rebuild from IBTrACS."
    )
