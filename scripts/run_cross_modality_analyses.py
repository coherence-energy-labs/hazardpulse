#!/usr/bin/env python3
"""Run all six cross-modality analyses and publish a summary.

Targets the LAIC framework + lightning + CME-RI hypotheses on the
HazardPulse + Signalbook corpus.

Outputs:
  - results/cross_modality/{analysis_name}.json   (per-analysis dataclass)
  - dist/data/cross-modality-summary.json         (worker-served aggregate)

Wiring:
  - .github/workflows/cross-modality-analyses.yml schedules this
    weekly + on workflow_dispatch.
  - dist/verification/cross-modality/index.html renders from the
    published summary.

Sample sizes are configurable; the default 200 events × 6 analyses
finishes in roughly ~10 minutes on a CI runner thanks to the cached
SWPC + GLM tables. The full ~1500-event population test runs in
~90 minutes (use --full).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.analyses import (  # noqa: E402
    run_earthquake_geomagnetic_precursor,
    run_earthquake_imf_bz_precursor,
    run_earthquake_solar_flare_precursor,
    run_hurricane_lightning_correlation,
    run_tornado_lightning_leadup,
    run_cme_hurricane_intensification,
)
from hazardpulse.data.earthquake import load_usgs_catalog  # noqa: E402
from hazardpulse.earthquake.definitive_model import (  # noqa: E402
    decluster_gardner_knopoff, _event_epoch,
)


def _epoch_to_dt(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(tzinfo=None)


def _load_eq_targets(*, max_events: int) -> list[dt.datetime]:
    """Load M6+ declustered mainshock times from the cached USGS catalog."""
    catalog = load_usgs_catalog(min_year=2005, max_year=2025, min_mag=4.0)
    if not catalog:
        return []
    mainshocks, _ = decluster_gardner_knopoff(catalog)
    m6_plus = [m for m in mainshocks if (m.get("mag") or 0) >= 6.0]
    times: list[dt.datetime] = []
    for m in m6_plus:
        t_str = m.get("time", "")
        if not isinstance(t_str, str):
            continue
        epoch = _event_epoch(m)
        if epoch <= 0:
            continue
        times.append(_epoch_to_dt(epoch))
    times.sort()
    if max_events and len(times) > max_events:
        # Even spacing across the window, not random subsample
        step = len(times) // max_events
        times = times[::max(step, 1)][:max_events]
    return times


def _finite(obj):
    """The same structure with every non-finite float (NaN, inf) replaced by None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _strict_json(obj) -> str:
    """JSON every parser accepts: an analysis with no data has null statistics, never a bare NaN token
    (which Python writes by default and the Worker's JSON.parse rejects -- /api/v1/laic/* was a 404)."""
    return json.dumps(_finite(obj), indent=2, allow_nan=False)


def _result_to_dict(result, hazard: str, analysis_id: str) -> dict:
    """Convert any analysis dataclass to the publish schema."""
    base = dataclasses.asdict(result)
    base["hazard"] = hazard
    base["analysis_id"] = analysis_id
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-eq", type=int, default=200,
                        help="Max M6+ events for earthquake analyses (default 200)")
    parser.add_argument("--full", action="store_true",
                        help="Use full ~1500 M6+ population (slower)")
    parser.add_argument("--skip-lightning", action="store_true",
                        help="Skip GLM lightning analyses (require live S3 fetch)")
    parser.add_argument("--out-summary", type=Path,
                        default=PROJECT_ROOT / "dist" / "data" / "cross-modality-summary.json")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "cross_modality")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    max_eq = None if args.full else args.max_eq

    print("Cross-modality analyses suite")
    print(f"  Max EQ events: {max_eq if max_eq else 'full population'}")
    print(f"  Lightning analyses: {'SKIPPED' if args.skip_lightning else 'enabled'}")
    print()

    print("Loading M6+ mainshocks from USGS catalog...")
    eq_times = _load_eq_targets(max_events=max_eq or 5000)
    print(f"  Got {len(eq_times)} M6+ events for analyses")

    analyses: list[dict] = []

    # ---- Earthquake LAIC analyses ----
    if eq_times:
        print()
        print("Running earthquake_geomagnetic_precursor (Sobolev/Hayakawa Kp)...")
        r = run_earthquake_geomagnetic_precursor(eq_times)
        d = _result_to_dict(r, "earthquake", "geomagnetic_precursor_kp")
        (args.out_dir / "earthquake_geomagnetic_precursor.json").write_text(
            _strict_json(d) + "\n", encoding="utf-8",
        )
        analyses.append(d)
        print(f"  delta={r.delta_mean:+.3f} CI [{r.bootstrap_ci_delta_lo:+.3f}, "
              f"{r.bootstrap_ci_delta_hi:+.3f}]  p={r.welch_p_two_sided:.3f}")

        print()
        print("Running earthquake_solar_flare_precursor (Freund/Pulinets X-ray)...")
        r = run_earthquake_solar_flare_precursor(eq_times)
        d = _result_to_dict(r, "earthquake", "solar_flare_precursor_xray")
        (args.out_dir / "earthquake_solar_flare_precursor.json").write_text(
            _strict_json(d) + "\n", encoding="utf-8",
        )
        analyses.append(d)
        print(f"  delta={r.delta_mean:+.3f} CI [{r.bootstrap_ci_delta_lo:+.3f}, "
              f"{r.bootstrap_ci_delta_hi:+.3f}]  p={r.welch_p_two_sided:.3f}")

        print()
        print("Running earthquake_imf_bz_precursor (LAIC IMF Bz)...")
        r = run_earthquake_imf_bz_precursor(eq_times)
        d = _result_to_dict(r, "earthquake", "imf_bz_precursor")
        (args.out_dir / "earthquake_imf_bz_precursor.json").write_text(
            _strict_json(d) + "\n", encoding="utf-8",
        )
        analyses.append(d)
        print(f"  delta={r.delta_mean:+.3f} CI [{r.bootstrap_ci_delta_lo:+.3f}, "
              f"{r.bootstrap_ci_delta_hi:+.3f}]  p={r.welch_p_two_sided:.3f}")

    # ---- Hurricane CME ----
    print()
    print("Running cme_hurricane_intensification (Thakur+ CME-RI)...")
    # Build a few RI events from matured hurricane forecasts
    # For now use a small synthetic sample: any matured hurricane forecast
    # whose predictions[].ri_occurred is True.
    hu_replay_dir = PROJECT_ROOT / "results" / "hurricane_prospective"
    ri_times: list[dt.datetime] = []
    no_ri_times: list[dt.datetime] = []
    pf_path = hu_replay_dir / "per_forecast_scores.jsonl"
    if pf_path.exists():
        for line in pf_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            issued = rec.get("issued_at")
            if not issued:
                continue
            # A no-storm forecast or one whose storms have no verifiable best track is
            # NOT a "no RI" observation (score_hurricane_prospective: n_verified).
            if rec.get("null_forecast") or not rec.get("n_verified", 0):
                continue
            t = dt.datetime.fromisoformat(issued.replace("Z", "+00:00")).replace(tzinfo=None)
            if rec.get("n_ri_events", 0) > 0:
                ri_times.append(t)
            else:
                no_ri_times.append(t)
    if ri_times and no_ri_times:
        r = run_cme_hurricane_intensification(ri_times, no_ri_times)
        d = _result_to_dict(r, "hurricane", "cme_intensification")
        (args.out_dir / "cme_hurricane_intensification.json").write_text(
            _strict_json(d) + "\n", encoding="utf-8",
        )
        analyses.append(d)
        print(f"  RI={r.n_ri_events}  no-RI={r.n_no_ri_events}  "
              f"delta={r.delta_mean:+.3f} p={r.welch_p:.3f}")
    else:
        print(f"  Insufficient matured RI data (RI={len(ri_times)}, no-RI={len(no_ri_times)}); "
              "analysis skipped this run.")

    # ---- Lightning analyses ----
    if not args.skip_lightning:
        print()
        print("Lightning analyses (GLM): require live cyclone + tornado event lists; "
              "skipped here. Use scripts/run_lightning_analyses.py with curated event lists.")
        # Placeholder so the summary records the analysis was scheduled
        for slug, hazard, label in [
            ("hurricane_lightning_correlation", "hurricane",
             "Fierro+2014 GLM-RI (placeholder; needs curated event list)"),
            ("tornado_lightning_leadup", "tornado",
             "Steiger+2007 flash-jump (placeholder; needs SPC+GLM join)"),
        ]:
            analyses.append({
                "analysis_id": slug,
                "hazard": hazard,
                "name": slug,
                "status": "needs_curated_input",
                "notes": label,
            })

    # ---- Publish summary ----
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "n_analyses": len(analyses),
        "max_eq_events_requested": max_eq,
        "n_eq_events_used": len(eq_times),
        "analyses": analyses,
    }
    args.out_summary.write_text(_strict_json(payload) + "\n", encoding="utf-8")
    # the research page renders from this summary: rebuild the site so the page and the file agree
    from hazardpulse.site import build as site_build
    changed = site_build.build_site()
    print(f"Site: {len(changed)} files re-rendered")
    print()
    print(f"Wrote {args.out_summary} ({len(analyses)} analyses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
