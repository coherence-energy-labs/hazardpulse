#!/usr/bin/env python3
"""Audit every hurricane forecast record: can each published and shadow number still be shown to
be what was issued, and how it was made?

    PYTHONPATH=src python scripts/audit_hurricane_records.py [--strict]

1. chain    -- ``dist/data/hurricane-ledger.jsonl``: every entry's hash recomputes, and every
               ``prev_hash`` is the previous entry's hash;
2. content  -- every ledger entry's ``content_sha256`` equals the hash of its replay file's
               forecast (id + storms), so no stored storm record, shadow or input changed later;
3. recompute -- every shadow that stored its inputs gives back its own model probabilities when
               those inputs are run through the artifact its ``model_version`` names.

Records written before a check existed (no ledger entry, no stored inputs) are counted as not
covered, never as passing. Writes ``results/hurricane_prospective/record_audit.json``; with
``--strict`` exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import ri_v9, ri_v10  # noqa: E402
from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402

LEDGER = ROOT / "dist" / "data" / "hurricane-ledger.jsonl"
REPLAY = ROOT / "dist" / "data" / "replay"
OUT = ROOT / "results" / "hurricane_prospective" / "record_audit.json"
TOL = 5e-5          # stored probabilities are rounded to 4 decimals


CATCH_UP_KEY = "shadow_catch_up"      # a forecast file's catch-up records (amendment 7 rule 2)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def content(art: dict) -> dict:
    """What a forecast file said, as its ledger entry hashed it (fetch_and_score.forecast_content):
    its id, its storms, and its catch-up records when it has any."""
    body = {"forecast_id": art["forecast_id"], "storms": art["storms"]}
    if art.get(CATCH_UP_KEY):
        body[CATCH_UP_KEY] = art[CATCH_UP_KEY]
    return body


def known_models(root: Path = ROOT) -> dict[str, tuple[str, dict]]:
    """``{model_version: (kind, artifact)}`` for every hurricane shadow artifact in the repo."""
    out = {}
    v9_path = root / "results" / "models" / ri_v9.MODEL_PATH.name
    if v9_path.exists():
        payload, version = ri_v9.load(v9_path)
        out[version] = ("v9", payload)
    for name in (ri_v10.MODEL_PATH.name, ri_v10.V10_2_PATH.name, ri_v10.V10_3_PATH.name):
        p = root / "results" / "models" / name
        if p.exists():
            art, version = ri_v10.load(p)
            out[version] = ("v10", art)
    return out


def check_chain(rows: list[dict]) -> list[str]:
    bad, prev = [], "0" * 64
    for i, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != "hash"}
        if sha(body) != row.get("hash"):
            bad.append(f"entry {i} ({row.get('forecast_id')}): hash does not recompute")
        if row.get("prev_hash") != prev:
            bad.append(f"entry {i} ({row.get('forecast_id')}): prev_hash is not the previous entry's hash")
        prev = row.get("hash")
    return bad


def recompute_shadow(kind: str, art: dict, shadow: dict) -> float:
    """Largest |recomputed - stored| over the shadow's model probabilities."""
    inputs = shadow["inputs"]
    if kind == "v10":
        got = ri_v10.recompute(art, inputs)
        return max(abs(got[k] - float(v)) for k, v in (shadow.get("model_probabilities") or {}).items())
    row = np.array([[np.nan if inputs.get(n) is None else float(inputs[n]) for n in art["feature_names"]]])
    return abs(float(lp.predict_proba(art, row)[0]) - float(shadow["model_probability"]))


def audit(root: Path = ROOT) -> dict:
    ledger = root / "dist" / "data" / "hurricane-ledger.jsonl"
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if ledger.exists() else []
    chain_bad = check_chain(rows)
    content_bad, content_ok, no_replay = [], 0, []
    for row in rows:
        p = root / "dist" / "data" / "replay" / f"{row['forecast_id']}.json"
        if not p.exists():
            no_replay.append(row["forecast_id"])
            continue
        art = json.loads(p.read_text(encoding="utf-8"))
        if sha(content(art)) == row.get("content_sha256"):
            content_ok += 1
        else:
            content_bad.append(row["forecast_id"])
    models = known_models(root)
    rec = {"checked": 0, "matched": 0, "mismatched": [], "no_inputs": 0, "artifact_not_in_repo": 0}
    files = sorted((root / "dist" / "data" / "replay").glob("hu_fcst_*.json"))
    for p in files:
        art = json.loads(p.read_text(encoding="utf-8"))
        for s in (art.get("storms") or []) + (art.get(CATCH_UP_KEY) or []):
            for key, sh in s.items():
                if not (key.endswith("_shadow") and isinstance(sh, dict) and sh.get("status") == "ok"):
                    continue
                if not sh.get("inputs"):
                    rec["no_inputs"] += 1
                    continue
                m = models.get(str(sh.get("model_version")))
                if m is None:
                    rec["artifact_not_in_repo"] += 1
                    continue
                rec["checked"] += 1
                worst = recompute_shadow(m[0], m[1], sh)
                if worst <= TOL:
                    rec["matched"] += 1
                else:
                    rec["mismatched"].append(f"{art['forecast_id']} {s.get('storm_id')} {key}: |d| {worst:.2e}")
    ok = not chain_bad and not content_bad and not rec["mismatched"]
    return {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "ok": ok,
            "ledger": {"entries": len(rows), "chain_mismatches": chain_bad,
                       "content_matches": content_ok, "content_mismatches": content_bad,
                       "entries_without_replay": no_replay},
            "replay_files": len(files), "recompute": rec,
            "note": "records written before the ledger (2026-10-03) or before inputs were stored are "
                    "counted as not covered, never as passing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    res = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    led, rec = res["ledger"], res["recompute"]
    print(f"hurricane records: ledger {led['entries']} entries, chain mismatches {len(led['chain_mismatches'])}, "
          f"content {led['content_matches']} ok / {len(led['content_mismatches'])} changed; shadows recomputed "
          f"{rec['matched']}/{rec['checked']} (not covered: {rec['no_inputs']} without inputs, "
          f"{rec['artifact_not_in_repo']} without their artifact)")
    return 1 if (args.strict and not res["ok"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
