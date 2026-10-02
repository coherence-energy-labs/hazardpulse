"""Ladder arms A and B: the PRE-FIX definitive pipeline, verbatim, on the rebuilt cache.

Run from the PRISTINE worktree (hp_base, origin/main aedf08cd1) with PYTHONPATH=src:

    python ladder_legacy.py A <out_dir>   # old code + stamp shim only
    python ladder_legacy.py B <out_dir>   # A + the label fix only

Stamp shim (both arms): the old builder can only parse ISO stamps (it keys on
the letter T, which also matches "UTC"); the rebuilt cache holds native
"YYYYMMDD_HHMMSS UTC" stamps. The shim rewrites them to ISO on load -- the
same instants -- so the old code sees what the March-2026 cache gave it.

Label shim (arm B only): SPC reports converted to absolute UTC (tz=3 -> +6 h),
keyed by UTC date, the next UTC day's reports appended with hour + 24, and
compute_label without the +24 h wrap. Everything else -- features, solver,
torsion, 5:1 test downsampling, i.i.d. bootstrap -- is the old code.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import sys
from pathlib import Path

import hazardpulse
from hazardpulse.tornado import definitive_model as old

arm, out_dir = sys.argv[1], Path(sys.argv[2])
print("hazardpulse from", hazardpulse.__file__, "arm", arm, flush=True)

_NATIVE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\s*UTC$")
_orig_load = old.load_cached_probsevere


def _iso_stamps(date_str, *a, **k):
    steps = _orig_load(date_str, *a, **k)
    if steps is None:
        return None
    for ts in steps:
        m = _NATIVE.match(str(ts.get("valid_time", "")))
        if m:
            y, mo, d, hh, mi, ss = m.groups()
            ts["valid_time"] = f"{y}-{mo}-{d}T{hh}:{mi}:{ss}Z"
    return steps


old.load_cached_probsevere = _iso_stamps

if arm == "B":
    def utc_reports(spc_csv_path):
        by_day: dict[str, list[dict]] = {}
        with open(spc_csv_path, encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                try:
                    yr, mo, dy = int(r["yr"]), int(r["mo"]), int(r["dy"])
                    slat, slon = float(r["slat"]), float(r["slon"])
                    tz = int(r["tz"])
                    hh, mi = (int(x) for x in r["time"].split(":")[:2])
                except (ValueError, KeyError):
                    continue
                if abs(slat) < 1.0 or abs(slon) < 1.0 or tz not in (3, 9):
                    continue
                t = dt.datetime(yr, mo, dy, hh, mi) + dt.timedelta(hours=6 if tz == 3 else 0)
                by_day.setdefault(t.strftime("%Y%m%d"), []).append(
                    {"slat": slat, "slon": slon, "hour": t.hour + t.minute / 60.0, "mag": int(r["mag"])}
                )
        # The old loader returns {date: reports}; load_all_data looks up the
        # storm's own date. Append the NEXT UTC day's reports with hour + 24 so
        # windows crossing 00Z see them on the same axis.
        out: dict[str, list[dict]] = {}
        for day in set(by_day) | {
            (dt.datetime.strptime(d, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d") for d in by_day
        }:
            nxt = (dt.datetime.strptime(day, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
            out[day] = list(by_day.get(day, [])) + [
                dict(r, hour=r["hour"] + 24.0) for r in by_day.get(nxt, [])
            ]
        return out

    def label_no_wrap(storm_lat, storm_lon, storm_hour, reports, forward_minutes=old.FORWARD_WINDOW_MIN):
        if storm_hour < 0:
            return -1
        for tor in reports:
            if tor.get("hour", -1) < 0:
                continue
            if old.haversine_km(storm_lat, storm_lon, tor["slat"], tor["slon"]) > old.LABEL_RADIUS_KM:
                continue
            if 0 <= tor["hour"] - storm_hour <= forward_minutes / 60.0:
                return 1
        return 0

    old.load_spc_tornado_reports = utc_reports
    old.compute_label = label_no_wrap

res = old.main(output_dir=out_dir, verbose=True)
print("ARM", arm, "test AUC full/enhanced/baseline:",
      round(res["full"]["auc"], 4), round(res["enhanced"]["auc"], 4), round(res["baseline"]["auc"], 4),
      "excluded", res["data_summary"]["total_excluded"], flush=True)
