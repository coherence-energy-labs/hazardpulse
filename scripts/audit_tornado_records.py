#!/usr/bin/env python3
"""Audit the tornado forecast records: every v3 forecast that stored its inputs is recomputed from
them by the payload its ``model_version`` names, and must give back its own 60-min probability.

    PYTHONPATH=src python scripts/audit_tornado_records.py [--strict]

(The tornado ledger's hash chain is checked by the site build; this adds the recomputation.)
Records written before inputs were stored are counted as not covered, never as passing. Writes
``results/tornado_prospective/record_audit.json``; ``--strict`` exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402
from hazardpulse.tornado import v3_serving as vs  # noqa: E402

OUT = ROOT / "results" / "tornado_prospective" / "record_audit.json"
TOL = 1e-12


def payloads(models_dir: Path) -> dict[str, dict]:
    out = {}
    for name in (vs.MAIN_FILE, vs.FALLBACK_FILE):
        p = models_dir / name
        if p.exists():
            payload = lp.load(p)
            out[lp.model_version(payload)] = payload
    return out


def audit(root: Path = ROOT) -> dict:
    known = payloads(root / "results" / "models")
    res = {"files_with_inputs": 0, "checked": 0, "matched": 0, "mismatched": [], "artifact_not_in_repo": 0,
           "files_without_inputs": 0}
    for p in sorted((root / "dist" / "data" / "replay").glob("to_fcst_*.json")):
        text = p.read_text(encoding="utf-8")
        if '"inputs"' not in text:                 # written before inputs were stored: not covered
            res["files_without_inputs"] += 1
            continue
        res["files_with_inputs"] += 1
        for s in json.loads(text).get("storms") or []:
            v3 = s.get("v3") or {}
            if not v3.get("inputs"):
                continue
            payload = known.get(str(v3.get("model_version")))
            if payload is None:
                res["artifact_not_in_repo"] += 1
                continue
            res["checked"] += 1
            got = vs.recompute_p60(payload, v3["inputs"])
            if abs(got - float(v3["probability_60min"])) <= TOL:
                res["matched"] += 1
            else:
                res["mismatched"].append(f"{p.stem} {s.get('storm_id')}: {got} vs {v3['probability_60min']}")
    res["ok"] = not res["mismatched"]
    res["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    res = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"tornado records: {res['matched']}/{res['checked']} recomputed exactly; not covered: "
          f"{res['files_without_inputs']} files without inputs, {res['artifact_not_in_repo']} without their payload")
    return 1 if (args.strict and not res["ok"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
