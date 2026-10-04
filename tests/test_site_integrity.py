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
        if rel_path == "dist/404.html":
            assert '<meta name="robots" content="noindex">' in text   # a 404 is never indexed or canonical
        else:
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
        # prefetch rules travel in a header, not an inline script the CSP would have to allow
        assert 'type="speculationrules"' not in text, rel_path
    assert 'Speculation-Rules: "/speculation-rules.json"' in _read_text("dist/_headers")
    assert json.loads(_read_text("dist/speculation-rules.json"))["prefetch"]


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
    """Generated content is written as CSS escapes, never raw non-ASCII a mis-decoded file would garble."""
    text = _read_text("dist/assets/styles.css")
    assert text.isascii()
    contents = re.findall(r'content:\s*"([^"]*)"', text)
    assert contents and all(c.isascii() for c in contents)
    assert 'content: "\\2192"' in text            # the card arrow, as an escape


def test_worker_is_in_deploy_path() -> None:
    worker_path = ROOT / "src" / "worker.js"
    assert worker_path.exists(), "src/worker.js must exist for production deploys"
    wrangler_toml = _read_text("wrangler.toml")
    assert 'main = "./src/worker.js"' in wrangler_toml


def test_pages_run_through_the_worker_and_static_files_do_not() -> None:
    """Workers serve a file that matches an asset without running the Worker, so the edge personalisation
    and the pages' own headers never ran in production until this routing existed (2026-10-04)."""
    import tomllib
    cfg = tomllib.loads(_read_text("wrangler.toml"))
    patterns = cfg["assets"]["run_worker_first"]
    assert "/*" in patterns
    for static in ("!/assets/*", "!/data/*", "!/speculation-rules.json", "!/sitemap.xml", "!/feed.xml",
                   "!/robots.txt"):
        assert static in patterns, static


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
    assert f'data-forecast="{forecast_id}"' in index_page


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


def test_committed_verification_rollups_never_contradict_the_prospective_scorers() -> None:
    """Audit finding #4: the rollups said "0 scored, no evaluator" for months while
    results/<hazard>_prospective/prospective_summary.json held scored forecasts.

    Checks every committed copy -- the served dist/data rollups and the
    results/verification storage -- against the committed scorer summaries.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("bsa_integrity", ROOT / "scripts" / "build_site_artifacts.py")
    bsa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsa)
    summaries = {
        key: json.loads((ROOT / "results" / f"{name}_prospective" / "prospective_summary.json").read_text(encoding="utf-8"))
        for key, name in (("eq", "earthquake"), ("hu", "hurricane"), ("to", "tornado"))
        if (ROOT / "results" / f"{name}_prospective" / "prospective_summary.json").exists()
    }
    copies = {
        "dist/data/verification-summary.json": json.loads(_read_text("dist/data/verification-summary.json"))["hazards"],
        "results/verification/system/summary.json": json.loads(
            _read_text("results/verification/system/summary.json")
        )["hazards"],
        "dist/data/verification/*.json": [
            json.loads(_read_text(f"dist/data/verification/{key}.json")) for key in ("eq", "hu", "to")
        ],
        "results/verification/*/live_rollup.json": [
            json.loads(_read_text(f"results/verification/{name}/live_rollup.json"))
            for name in ("earthquake", "hurricane", "tornado")
        ],
    }
    problems = {
        where: bsa._verification_rollup_violations(items, summaries, allow_lag=True)
        for where, items in copies.items()
    }
    assert not any(problems.values()), problems


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
        # versioned by content hash (tests/test_site_pages.py checks each hash against its file)
        assert re.search(r'<script src="/assets/site-shell\.js\?v=[0-9a-f]{10}"></script>', text), rel_path
        assert re.search(r'<link rel="stylesheet" href="/assets/styles\.css\?v=[0-9a-f]{10}">', text), rel_path


def test_map_surfaces_include_user_marker_svg_node() -> None:
    """Each map carries the visitor marker, with its projection for the edge worker to place it."""
    for rel_path in [
        "dist/index.html",
        "dist/live/index.html",
        "dist/live/earthquake/index.html",
        "dist/live/hurricane/index.html",
        "dist/live/tornado/index.html",
    ]:
        text = _read_text(rel_path)
        assert 'class="user-marker"' in text, rel_path
        assert 'class="user-pin"' in text, rel_path
        assert 'class="user-ring"' in text, rel_path
        assert re.search(r'class="user-marker"[^>]*data-lon0="-?[\d.]+" data-lat0="-?[\d.]+" data-sx="[\d.]+" '
                         r'data-sy="[\d.]+"', text), rel_path


def test_no_page_hides_content_behind_a_depth_toggle() -> None:
    """The old Simple/Technical toggle hid the legal terms and half of every page by default; every page now
    leads with plain language and puts detail in labelled disclosures."""
    for html_path in (ROOT / "dist").rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        assert "data-depth=" not in text and 'id="depth-technical"' not in text, html_path


def test_stylesheet_dark_mode_uses_theme_attribute() -> None:
    stylesheet = _read_text("dist/assets/styles.css")
    assert ':root[data-theme="dark"]' in stylesheet
    assert ':root:not([data-theme="light"])' in stylesheet          # the system setting, unless overridden
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


def test_tornado_scoring_runs_the_best_trained_model_present() -> None:
    """Guard against silent regression to a weaker tier.

    The newest tornado replay must come from the best trained model in the repo: the v3 suite
    (results/models/tornado_v3_w.json or tornado_v3.json, docs/TORNADO_MODEL_PROGRAM.md) when its
    payloads exist, else the GBT at results/models/tornado_gbt_v1.json. A replay from a lower tier
    while a better model is present means the serving path is broken. (Until 2026-10-03 this
    demanded tier1_ml whenever the GBT existed, so v3 going live failed it.)
    """
    models = ROOT / "results" / "models"
    if (models / "tornado_v3_w.json").exists() or (models / "tornado_v3.json").exists():
        want, prefix = "tier1_v3", "tornado_v3-"
    elif (models / "tornado_gbt_v1.json").exists():
        want, prefix = "tier1_ml", None
    else:
        return  # no trained model -> nothing to validate

    replay_dir = ROOT / "dist" / "data" / "replay"
    replays = sorted(replay_dir.glob("to_fcst_*.json"))
    assert replays, "No tornado replay artifacts found"

    latest = json.loads(replays[-1].read_text(encoding="utf-8"))
    tier = latest.get("scoring_tier")
    assert tier == want, (
        f"Tornado replay {replays[-1].name} has scoring_tier={tier!r}, expected {want!r} for the "
        "trained models present. Likely causes: a payload failed to load, the warnings and fallback "
        "paths both failed, or a tier-selection regression."
    )
    if prefix:
        assert str(latest.get("model_version", "")).startswith(prefix), latest.get("model_version")


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
