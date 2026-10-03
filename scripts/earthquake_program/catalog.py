#!/usr/bin/env python3
"""Earthquake forecast program, step 0: the catalog every candidate is fitted and scored on.

Two sources, both audited before use (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 3):

1. ``comcat_m4.5_<year>.csv`` -- every ComCat event with M >= 4.5, 1973 through the
   cutoff, pulled ONE YEAR PER REQUEST with the repository's limit-aware fetch
   (:func:`hazardpulse.data.usgs_fdsn.fetch_window` bisects any window whose answer
   reaches the 20,000-row limit, so a year can never be silently cut). A manifest per
   year records the request count and the largest response. This file feeds the
   long-term and short-term rate models (M >= 5.0), the targets (M >= 6.0) and the
   M >= 4.5 features.
2. The repository's audited M2.5+ year files (``.cache/earthquake/usgs``), used only by
   the M2.5 count features of candidate C and by the served-model baseline D, which
   reads M2.5+ sequences. Every year is run through
   :func:`hazardpulse.data.usgs_fdsn.audit_year_catalog`; a year that fails is refused.

Output: ``<cache>/program_catalog.npz`` (M4.5+, sorted by time) and
``<cache>/program_catalog_m25.npz`` (M2.5+, 2000 ->), plus ``catalog_audit.json`` in
the results directory.

    python scripts/earthquake_program/catalog.py --cutoff 2026-10-01
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data import usgs_fdsn  # noqa: E402

PROGRAM_CACHE = REPO / ".cache" / "earthquake" / "program"
RESULTS = REPO / "results" / "earthquake_program"
DEFAULT_M25_DIR = REPO / ".cache" / "earthquake" / "usgs"
USER_AGENT = "HazardPulse-research/1.0 (earthquake forecast program; polite yearly paging)"
FIRST_YEAR = 1973
MIN_MAG = 4.5
PAUSE_S = 1.0


def _fetch_text(url: str, retries: int = 4) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # network: retry with backoff, then raise
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"USGS request failed after {retries} tries: {url}: {last}")


def year_path(year: int) -> Path:
    return PROGRAM_CACHE / f"comcat_m{MIN_MAG:.1f}_{year}.csv"


def fetch_year(year: int, cutoff: dt.datetime, *, force: bool = False) -> Path:
    """Every M>=4.5 event of ``year`` (to ``cutoff`` for the cutoff year); cached with a manifest."""
    dest = year_path(year)
    man_path = usgs_fdsn.manifest_path_for(dest)
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = min(dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc), cutoff)
    if dest.exists() and man_path.exists() and not force:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if man.get("coverage_end") == end.strftime("%Y-%m-%dT%H:%M:%SZ"):
            return dest
    stats: dict = {}
    fieldnames, rows = usgs_fdsn.fetch_window(
        start, end, min_magnitude=MIN_MAG, fetch_text=_fetch_text,
        pause_seconds=PAUSE_S, stats=stats)
    if not rows:
        raise usgs_fdsn.USGSCatalogIncompleteError(f"no M{MIN_MAG}+ events returned for {year}")
    rows.sort(key=lambda r: r.get("time", ""))
    PROGRAM_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)
    manifest = {
        "year": year,
        "min_magnitude": MIN_MAG,
        "coverage_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_events": len(rows),
        "row_limit": usgs_fdsn.USGS_FDSN_ROW_LIMIT,
        "requests": stats.get("requests"),
        "bisections": stats.get("splits"),
        "max_rows_per_response": stats.get("max_rows_per_response"),
        "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetcher": "hazardpulse.data.usgs_fdsn.fetch_window (yearly, bisect-on-limit)",
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  {year}: {len(rows):,} events, {stats.get('requests')} request(s), "
          f"max rows/response {stats.get('max_rows_per_response')}")
    return dest


def _epoch(text: str) -> float:
    return usgs_fdsn.parse_event_time(text).timestamp()


def read_rows(path: Path, min_mag: float) -> tuple[list[dict], list[str]]:
    out: list[dict] = []
    times: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            times.append(row.get("time", ""))
            try:
                mag = float(row["mag"])
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if mag < min_mag:
                continue
            try:
                depth = float(row.get("depth") or "nan")
            except ValueError:
                depth = float("nan")
            out.append({"t": _epoch(row["time"]), "lat": lat, "lon": lon, "mag": mag,
                        "depth": depth, "id": row.get("id", ""), "type": row.get("type", "")})
    return out, times


def to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda r: r["t"])
    # de-duplicate by id (adjacent year windows are half-open, but be explicit)
    seen: set[str] = set()
    kept = []
    for r in rows:
        key = r["id"] or f"{r['t']}|{r['lat']}|{r['lon']}|{r['mag']}"
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return {
        "t": np.array([r["t"] for r in kept], dtype=np.float64),
        "lat": np.array([r["lat"] for r in kept], dtype=np.float64),
        "lon": np.array([r["lon"] for r in kept], dtype=np.float64),
        "mag": np.array([r["mag"] for r in kept], dtype=np.float64),
        "depth": np.array([r["depth"] for r in kept], dtype=np.float64),
        "id": np.array([r["id"] for r in kept]),
        "type": np.array([r["type"] for r in kept]),
    }


def audit_m45(arrays: dict[str, np.ndarray], first_year: int, cutoff: dt.datetime) -> dict:
    """Year/month completeness of the M>=5 subset: a global month with no M>=5 event is a hole."""
    t = arrays["t"]
    m = arrays["mag"]
    years = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc).year for x in t])
    months = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc).month for x in t])
    per_year = {}
    empty_months = []
    for y in range(first_year, cutoff.year + 1):
        sel = years == y
        per_year[str(y)] = {
            "m4.5": int((sel & (m >= 4.5)).sum()),
            "m5": int((sel & (m >= 5.0)).sum()),
            "m6": int((sel & (m >= 6.0)).sum()),
            "m7": int((sel & (m >= 7.0)).sum()),
        }
        last_month = 12 if y < cutoff.year else cutoff.month - (1 if cutoff.day == 1 else 0)
        for mo in range(1, last_month + 1):
            if not ((sel & (months == mo) & (m >= 5.0)).any()):
                empty_months.append(f"{y}-{mo:02d}")
    return {"per_year": per_year, "months_without_m5": empty_months}


def build_m25(m25_dir: Path, first_year: int, last_year: int) -> tuple[dict, dict]:
    rows: list[dict] = []
    audit: dict[str, list[str]] = {}
    now = dt.datetime.now(dt.timezone.utc)
    for year in range(first_year, last_year + 1):
        path = m25_dir / f"usgs_catalog_{year}.csv"
        if not path.exists():
            audit[str(year)] = ["missing"]
            continue
        yr_rows, times = read_rows(path, 2.5)
        problems = usgs_fdsn.audit_year_catalog(times, year, manifest=usgs_fdsn.read_manifest(path), now=now)
        audit[str(year)] = problems
        rows.extend(yr_rows)
    return to_arrays(rows), audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2026-10-01", help="UTC date; the catalog ends strictly before it")
    ap.add_argument("--m25-dir", type=Path, default=DEFAULT_M25_DIR,
                    help="directory of the audited usgs_catalog_<year>.csv M2.5+ files")
    ap.add_argument("--m25-first-year", type=int, default=2000)
    ap.add_argument("--m25-last-year", type=int, default=2025)
    ap.add_argument("--skip-m25", action="store_true")
    args = ap.parse_args(argv)

    cutoff = dt.datetime.fromisoformat(args.cutoff).replace(tzinfo=dt.timezone.utc)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROGRAM_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"ComCat M{MIN_MAG}+ {FIRST_YEAR} -> {cutoff:%Y-%m-%d} (one request per year, bisect on limit)")
    rows: list[dict] = []
    manifests = {}
    for year in range(FIRST_YEAR, cutoff.year + 1):
        path = fetch_year(year, cutoff)
        yr_rows, _ = read_rows(path, MIN_MAG)
        rows.extend(yr_rows)
        manifests[str(year)] = json.loads(usgs_fdsn.manifest_path_for(path).read_text(encoding="utf-8"))
    arrays = to_arrays(rows)
    arrays = {k: v[arrays["t"] < cutoff.timestamp()] for k, v in arrays.items()}
    np.savez_compressed(PROGRAM_CACHE / "program_catalog.npz", **arrays)
    audit = audit_m45(arrays, FIRST_YEAR, cutoff)
    clipped = [y for y, m in manifests.items() if int(m.get("max_rows_per_response") or 0) >= usgs_fdsn.USGS_FDSN_ROW_LIMIT]
    report = {
        "cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "m45": {
            "n_events": int(arrays["t"].size),
            "n_m5": int((arrays["mag"] >= 5.0).sum()),
            "n_m6": int((arrays["mag"] >= 6.0).sum()),
            "years_with_a_response_at_the_row_limit": clipped,
            "max_rows_per_response": max(int(m.get("max_rows_per_response") or 0) for m in manifests.values()),
            "requests": sum(int(m.get("requests") or 0) for m in manifests.values()),
            **audit,
        },
    }
    print(f"  M4.5+: {report['m45']['n_events']:,} events, M5+ {report['m45']['n_m5']:,}, "
          f"M6+ {report['m45']['n_m6']:,}; months without an M5: {audit['months_without_m5'][:10]}")

    if not args.skip_m25:
        m25, m25_audit = build_m25(args.m25_dir, args.m25_first_year, args.m25_last_year)
        bad = {y: p for y, p in m25_audit.items() if p}
        np.savez_compressed(PROGRAM_CACHE / "program_catalog_m25.npz", **m25)
        report["m25"] = {
            "source_dir": str(args.m25_dir),
            "years": [args.m25_first_year, args.m25_last_year],
            "n_events": int(m25["t"].size),
            "audit_problems": bad,
            "per_year": {str(y): int(((m25["t"] >= dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp())
                                      & (m25["t"] < dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc).timestamp())).sum())
                         for y in range(args.m25_first_year, args.m25_last_year + 1)},
        }
        # cross-check: the M>=4.5 events of the M2.5 files against the fresh M4.5 pull, per year
        cross = {}
        for y in range(args.m25_first_year, args.m25_last_year + 1):
            lo = dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp()
            hi = dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc).timestamp()
            a = int(((m25["t"] >= lo) & (m25["t"] < hi) & (m25["mag"] >= 5.0)).sum())
            b = int(((arrays["t"] >= lo) & (arrays["t"] < hi) & (arrays["mag"] >= 5.0)).sum())
            cross[str(y)] = {"m5_in_m25_files": a, "m5_in_fresh_pull": b, "ratio": round(a / b, 4) if b else None}
        report["m25"]["m5_cross_check"] = cross
        print(f"  M2.5+: {m25['t'].size:,} events; audit problems: {bad or 'none'}")
    (RESULTS / "catalog_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {RESULTS / 'catalog_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
