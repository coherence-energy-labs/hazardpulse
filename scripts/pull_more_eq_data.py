#!/usr/bin/env python3
"""Pull a FULLER USGS catalog: global M2.0+ (vs the M2.5+ default), one year file each.

More small events = more foreshocks = better precursor signal, especially for the
'silent' big quakes. Read by load_usgs_catalog when HAZARDPULSE_USGS_FULL=1.

Uses the shared limit-aware fetcher (hazardpulse.data.usgs_fdsn): month by month,
bisecting any window whose response reaches the FDSN row limit, and RAISING on a
failed request. The previous version skipped failed months with a print and kept
cap-hit months, which is how usgs_M2.0_2005.csv lost Oct-Dec and 2006 lost Jan-Feb.
Resumable: a year whose file passes the completeness audit is not re-pulled.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hazardpulse.data import usgs_fdsn  # noqa: E402

OUT = REPO / ".cache" / "earthquake" / "usgs_full"
UA = "hazardpulse/0.1 (research; +https://github.com/coherence-energy-labs/hazardpulse)"
MINMAG = 2.0


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def pull_year(year: int, out_dir: Path, now: dt.datetime, *, force: bool = False) -> Path:
    dest = out_dir / f"usgs_M{MINMAG}_{year}.csv"
    if not force and dest.exists():
        with dest.open(encoding="utf-8", errors="replace") as fh:
            times = [row.get("time", "") for row in csv.DictReader(fh)]
        problems = usgs_fdsn.audit_year_catalog(times, year, manifest=usgs_fdsn.read_manifest(dest),
                                                now=now)
        fresh = year < now.year or bool(usgs_fdsn.read_manifest(dest))
        if not problems and fresh:
            print(f"  [{year}] cached, complete ({len(times)} events)", flush=True)
            return dest
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = min(dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc), now)
    stats: dict = {}
    fieldnames, rows = usgs_fdsn.fetch_catalog(start, end, min_magnitude=MINMAG, fetch_text=_get,
                                               pause_seconds=1.0, stats=stats)
    problems = usgs_fdsn.audit_event_times((r.get("time", "") for r in rows), period_start=start,
                                           period_end=end, proven_unclipped=True)
    if problems:
        raise usgs_fdsn.USGSCatalogIncompleteError(f"M{MINMAG} {year}: {'; '.join(problems)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.partial")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)
    manifest = {
        "year": year, "min_magnitude": MINMAG,
        "coverage_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_events": len(rows), "row_limit": usgs_fdsn.USGS_FDSN_ROW_LIMIT,
        "requests": stats.get("requests", 0), "bisections": stats.get("splits", 0),
        "max_rows_per_response": stats.get("max_rows_per_response", 0),
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    usgs_fdsn.manifest_path_for(dest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  [{year}] wrote {len(rows)} events ({stats.get('requests')} requests, "
          f"{stats.get('splits')} bisections) -> {dest.name}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-year", type=int, default=2000)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    failures = []
    for year in range(args.min_year, min(args.max_year or now.year, now.year) + 1):
        try:
            pull_year(year, args.out_dir, now, force=args.force)
        except Exception as exc:
            failures.append(f"{year}: {exc}")
            print(f"  [{year}] FAILED: {exc}", flush=True)
    if failures:
        print("INCOMPLETE: " + " | ".join(failures))
        return 1
    print(f"DONE. M{MINMAG}+ catalog complete in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
