#!/usr/bin/env python3
"""Rebuild the shadow forecasts of test cycles the live scorer missed before catch-up existed
(docs/HURRICANE_RI_V9_PROGRAM.md, amendment 7 rule 3).

    PYTHONPATH=src python scripts/rebuild_hurricane_cycles.py EP152026:2026100400 EP182026:2026100400

Rule 3 allows a missed cycle to be rebuilt from archived inputs ONLY by a procedure that, run blind on
the cycles that already have qualifying live records, reproduces all four shadows' probabilities
exactly (to 1e-12). This script is that procedure and its control, in one run:

1. CONTROL. Every live test record of the storms being rebuilt (cycle_records' rule-1 selection over
   the forecast files) is rebuilt by the procedure below WITHOUT reading the record; only then is each
   rebuilt shadow compared with the stored one -- its probability, its whole curve, its model
   probabilities and its full-precision inputs. Anything beyond 1e-12 (or any differing input) and
   nothing is written.
2. PROCEDURE, per cycle t: the storm's a-deck as it stands now (only cycle t's lines enter the
   features), the SHIPS text from the archive (``results/hurricane_ships_archive``) -- refused unless
   the directory listing shows it last modified before t + 3 h 30 min, so the archived text is the one a
   run at that time would have read -- and the GMGSI images at t + 2 h and t - 4 h from NOAA's archive.
   The case and the shadows are the live scorer's own (``build_live_case(..., cycle=t)``,
   ``shadow_forecasts``).
3. OUTPUT: ``results/hurricane_prospective/rebuilt/<forecast id>.json`` -- the rebuilt records
   (``rebuilt: true``), what each read (a-deck SHA-256, SHIPS file and SHA-256, IR image keys), and the
   control's full result. The prospective test reads them as rule 3 records and reports every result
   with and without them (rule 5).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.data.http import fetch_text  # noqa: E402
from hazardpulse.hurricane import cycle_records as cr  # noqa: E402
from hazardpulse.hurricane import ships_text  # noqa: E402

REPLAY = ROOT / "dist" / "data" / "replay"
REBUILT = ROOT / "results" / "hurricane_prospective" / "rebuilt"
ARCHIVE = ROOT / "results" / "hurricane_ships_archive"
TOL = 1e-12
PROBABILITY_FIELDS = ("probability", "model_probability")
CURVE_FIELDS = ("probabilities", "model_probabilities", "noaa_24h")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compare(stored: dict, rebuilt: dict) -> list[str]:
    """Every way a rebuilt shadow differs from the stored one (empty = reproduced)."""
    out = []
    for f in PROBABILITY_FIELDS:
        a, b = stored.get(f), rebuilt.get(f)
        if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > TOL):
            out.append(f"{f}: {a} vs {b}")
    for f in CURVE_FIELDS:
        sa, sb = stored.get(f) or {}, rebuilt.get(f) or {}
        for k in sorted(set(sa) | set(sb)):
            a, b = sa.get(k), sb.get(k)
            if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > TOL):
                out.append(f"{f}[{k}]: {a} vs {b}")
    ia, ib = stored.get("inputs") or {}, rebuilt.get("inputs") or {}
    for n in sorted(set(ia) | set(ib)):
        if ia.get(n) != ib.get(n):
            out.append(f"input {n}: {ia.get(n)} vs {ib.get(n)}")
    if stored.get("model_version") != rebuilt.get("model_version"):
        out.append(f"model_version: {stored.get('model_version')} vs {rebuilt.get('model_version')}")
    return out


class Procedure:
    """The rebuild procedure, with an account of every input it read."""

    def __init__(self, fs, storms: list[str], listing: dict[str, str]):
        self.fs = fs
        self.models = (fs.load_v9_model(), fs.load_v10_model(), fs.load_challengers())
        self.decks, self.deck_sha = {}, {}
        for sid in storms:
            raw = fs.fetch_bytes(f"{fs.REALTIME_ADECK_INDEX}a{sid.lower()}.dat.gz", namespace="atcf_realtime",
                                 use_cache=False)
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
            self.decks[sid] = fs.parse_atcf_deck(text)
            self.deck_sha[sid] = hashlib.sha256(raw).hexdigest()
        self.listing = listing
        self.read: dict[tuple[str, dt.datetime], dict] = {}

    def ships(self, sid: str, cycle: dt.datetime):
        """The SHIPS text as it stood at t + 3 h 30 min: NHC's listing must show it last modified before
        then, and NHC's current text must be byte-identical to a version in our archive -- so the text
        read is the one a run at t + 3 h 30 would have read, and it is one we keep."""
        name = ships_text.filename_for(sid, cycle)
        listed = self.listing.get(name)
        note = self.read.setdefault((sid, cycle), {})
        note["ships_text"] = {"file": name, "listed_modified": listed}
        if listed is None or listed >= (cycle + cr.ADVISORY_DELAY).strftime("%Y-%m-%d %H:%M"):
            raise SystemExit(f"{name} was last modified {listed}, not before t + 3 h 30 min: the text a run at "
                             "that time read cannot be shown to be the one NHC serves now")
        now_text = self.fs.fetch_bytes(ships_text.STEXT_ROOT + name, namespace="ships_text", use_cache=False)
        sha = hashlib.sha256(now_text).hexdigest()
        archived = [p for p in sorted(ARCHIVE.glob(f"*/{name}.gz")) + sorted(ARCHIVE.glob(f"*/versions/{name}.*.gz"))
                    if hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == sha]
        if not archived:
            raise SystemExit(f"{name}: NHC's text is not a version in the archive")
        note["ships_text"].update(sha256=sha, archived_as=archived[0].relative_to(ARCHIVE).as_posix())
        return now_text.decode("utf-8", "replace"), name

    def ir(self, sid, cycle, records):
        feats, read = self.fs.live_ir_read(sid, cycle, records)
        self.read.setdefault((sid, cycle), {})["ir_images"] = read
        return feats, read

    def run(self, sid: str, cycle: dt.datetime) -> dict:
        v9, v10, ch = self.models
        case = self.fs.build_live_case(sid, self.decks[sid], cycle=cycle)
        if case is None:
            raise SystemExit(f"{sid} has no analysis at {cycle} in its a-deck")
        shadows = self.fs.shadow_forecasts(case, v9, v10, self.ships, lambda s: self.decks[s], ch, self.ir)
        return {"case": case, "shadows": shadows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cycles", nargs="+", help="STORMID:YYYYMMDDHH, e.g. EP152026:2026100400")
    args = ap.parse_args(argv)
    targets = []
    for spec in args.cycles:
        sid, dtg = spec.split(":")
        targets.append((sid.upper(), dt.datetime.strptime(dtg, "%Y%m%d%H")))
    storms = sorted({s for s, _ in targets})
    fs = _load("fas_rebuild", "scripts/fetch_and_score.py")
    v9p = _load("v9p_rebuild", "scripts/score_hurricane_v9_prospective.py")
    index = fetch_text(ships_text.STEXT_ROOT, namespace="ships_index", use_cache=False)
    archiver = _load("ash_rebuild", "scripts/archive_ships_text.py")
    proc = Procedure(fs, storms, archiver.listed_modified(index))
    keys = fs._shadow_keys(*proc.models[:2], proc.models[2])

    # 1. control: every live test record of these storms, rebuilt blind, compared only afterwards
    sel = v9p.select_test_records(REPLAY, rebuilt_dir=None)
    control = [c for c in sel.chosen.values() if c.storm_id in storms and not c.catch_up and not c.rebuilt]
    rebuilt_control = {c.key: proc.run(c.storm_id, c.cycle)["shadows"] for c in control}
    results, failures = [], 0
    for c in control:
        stored = c.ref[2]
        for k in keys:
            diffs = compare(stored.get(k) or {}, rebuilt_control[c.key].get(k) or {})
            failures += bool(diffs)
            results.append({"storm_id": c.storm_id, "cycle": cr.format_utc(c.cycle), "record": c.ref[0],
                            "shadow": k, "reproduced": not diffs, "differences": diffs[:10]})
    print(f"control: {len(control)} live test records x {len(keys)} shadows, "
          f"{len(results) - failures} reproduced, {failures} not")
    if len(control) < 6 or failures:
        print("  the procedure does not meet rule 3 on these records: nothing is rebuilt")
        for r in results:
            if not r["reproduced"]:
                print("   ", r["storm_id"], r["cycle"], r["shadow"], r["differences"][:3])
        return 1

    # 2. the missed cycles, by the same procedure; a cycle with a test record is never rebuilt
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)
    records = []
    existing = v9p.select_test_records(REPLAY, rebuilt_dir=REBUILT)
    for sid, t in targets:
        if (sid, t) in existing.chosen:
            raise SystemExit(f"{sid} {t} already has a test record ({existing.chosen[(sid, t)].ref[0]})")
        got = proc.run(sid, t)
        case = got["case"]
        records.append({"storm_id": sid, "storm_name": case.get("storm_name"), "basin": case.get("basin"),
                        "issue_time": case["issue_time"], "lat": case.get("analysis_lat"),
                        "lon": case.get("analysis_lon"), "vmax_kt": case.get("analysis_vmax_kt"),
                        "rebuilt": True, "lag_hours": cr.lag_hours(now, t), "published": False,
                        "inputs_read": {"adeck_sha256": proc.deck_sha[sid], **proc.read.get((sid, t), {})},
                        **got["shadows"]})
    fid = f"hu_rebuilt_{now:%Y%m%d_%H%M}"
    body = {"forecast_id": fid, "issued_at": cr.format_utc(now),
            "kind": "rebuilt test records (docs/HURRICANE_RI_V9_PROGRAM.md amendment 7 rule 3)",
            "procedure": "scripts/rebuild_hurricane_cycles.py",
            "control": {"tolerance": TOL, "records": len(control), "shadows": len(results),
                        "reproduced": len(results) - failures, "results": results},
            cr.REBUILT_KEY: records}
    body["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in body.items()}, sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    REBUILT.mkdir(parents=True, exist_ok=True)
    (REBUILT / f"{fid}.json").write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    for r in records:
        print(f"rebuilt {r['storm_id']} {r['issue_time']}: " + ", ".join(
            f"{k} {r[k].get('probability')}" for k in keys if isinstance(r.get(k), dict)))
    print(f"wrote {REBUILT / (fid + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
