#!/usr/bin/env python3
"""RECOMPUTE every ledger entry's SHA-256 from its own content. Reject anything that lies.

WHY THIS EXISTS

HazardPulse ships two hash chains -- `dist/data/earthquake-ledger.jsonl` and
`dist/data/tornado-ledger.jsonl` -- and the product claim resting on them is "this prediction was
recorded before the event." Until this script existed, the ONLY read-time check in the repo was
`build_site_artifacts._count_link_mismatches`, which compared each row's stored `prev_hash` to the
previous row's stored `hash` and never recomputed a digest from the row's content. Measured
2026-08-06 against the live `origin/main` ledgers, that check reports **0 mismatches** for all of:

  * rewriting entry 249's `top_probability` from 0.78 to 0.99 and backdating its timestamp to 2020,
  * a full re-chain of all 499 entries with arbitrary content (self-consistent, so it "passes"),
  * dropping the last 50 entries.

A chain that only checks that pointers agree with each other is not a chain; it is a linked list.
This script closes all three by (1) recomputing each digest from the row's canonical content,
(2) verifying the linkage, and (3) committing the length and head to a separate anchor file so
truncation and wholesale re-chaining are detectable.

THE WRITE-TIME DIGEST (read from the writers, not guessed)

  earthquake  scripts/fetch_and_score_earthquake.py:1552 and :2044
              sha256(json.dumps(entry_without_hash, sort_keys=True))          <- DEFAULT separators
  tornado     scripts/fetch_and_score_tornado.py:3540
              sha256(json.dumps(entry_without_hash, sort_keys=True, separators=(",", ":")))

The two ledgers genuinely disagree on separators, so the spec is per-ledger. Both reproduce
100% of the live rows (499/499 and 1367/1367) under the specs below.

WHAT THE WRITE-TIME DIGEST DOES *NOT* COVER  (stated explicitly, as required)

`forecast_id` was retrofitted onto rows that had already been hashed: it sits AFTER `hash` in the
JSON of the earliest rows and is absent from the digest body. Measured on origin/main:

    earthquake-ledger.jsonl   rows 0..12    (13 rows)   forecast_id NOT covered by the row digest
    tornado-ledger.jsonl      rows 0..169   (170 rows)  forecast_id NOT covered by the row digest

Every later row covers every field it carries. Two mechanisms close the legacy gap rather than
tolerating it:

  * `LEGACY_UNCOVERED_ROWS` is PINNED. A row at or beyond the pinned prefix MUST verify with every
    field included -- a new uncovered field can never appear without failing this gate.
  * the anchor's `chain_commitment` digests the COMPLETE canonical row (including `hash` and
    `forecast_id`), so mutating an uncovered field on a legacy row still breaks the anchor.

USAGE

    python scripts/verify_ledger_chain.py                  # verify worktree (or HEAD blobs)
    python scripts/verify_ledger_chain.py --ref origin/main # verify a ref's blobs
    python scripts/verify_ledger_chain.py --base origin/main  # + append-only chain continuity
    python scripts/verify_ledger_chain.py --update-anchor     # regenerate evidence/ledger-anchors.json
    python scripts/verify_ledger_chain.py --self-test         # prove the verifier rejects forgeries
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The anchor deliberately lives OUTSIDE `dist/`: `dist/` is in .gitignore, so a new file there
#: would need `git add -f` and would silently never reach the repo -- an anchor that is not
#: committed anchors nothing.
ANCHOR_PATH = ROOT / "evidence" / "ledger-anchors.json"

GENESIS = "0" * 64
ANCHOR_DOMAIN = b"hazardpulse-ledger-anchor-v1\n"


@dataclass(frozen=True)
class LedgerSpec:
    """How ONE ledger's digest is computed. Taken from its writer, verified against every live row."""

    path: str                       # repo-relative
    writer: str                     # file:line of the append that defines the digest
    compact: bool                   # True -> separators=(",", ":") at write time
    legacy_uncovered_rows: int      # leading rows whose digest predates `uncovered_fields`
    uncovered_fields: tuple[str, ...] = ("forecast_id",)

    def dumps(self, body: dict) -> str:
        kwargs: dict = {"sort_keys": True}
        if self.compact:
            kwargs["separators"] = (",", ":")
        return json.dumps(body, **kwargs)

    def digest(self, body: dict) -> str:
        return hashlib.sha256(self.dumps(body).encode("utf-8")).hexdigest()


LEDGERS: tuple[LedgerSpec, ...] = (
    LedgerSpec(
        path="dist/data/earthquake-ledger.jsonl",
        writer="scripts/fetch_and_score_earthquake.py:2044",
        compact=False,
        legacy_uncovered_rows=13,
    ),
    LedgerSpec(
        path="dist/data/tornado-ledger.jsonl",
        writer="scripts/fetch_and_score_tornado.py:3540",
        compact=True,
        legacy_uncovered_rows=170,
    ),
)


@dataclass
class Report:
    path: str
    n_rows: int = 0
    n_legacy: int = 0
    head_hash: str | None = None
    first_hash: str | None = None
    chain_commitment: str | None = None
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# ---------------------------------------------------------------------------
# reading (worktree, or a git blob when dist/ is sparse-checked-out away)
# ---------------------------------------------------------------------------


def blob_at(ref: str, path: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def read_source(path: str, ref: str | None) -> tuple[str | None, str]:
    """Returns (text, origin). Explicit `ref` wins; otherwise worktree, then HEAD's blob."""
    if ref:
        return blob_at(ref, path), f"git {ref}"
    on_disk = ROOT / path
    if on_disk.exists():
        return on_disk.read_text(encoding="utf-8"), "worktree"
    text = blob_at("HEAD", path)
    if text is not None:
        return text, "git HEAD (not in the worktree -- sparse checkout)"
    return None, "missing"


def parse_rows(text: str, report: Report) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            report.violations.append(f"line {lineno}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            report.violations.append(f"line {lineno}: expected a JSON object, got {type(row).__name__}")
            continue
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# the actual verification
# ---------------------------------------------------------------------------


def canonical_row_bytes(row: dict) -> bytes:
    """The COMPLETE row, including `hash` and any field the write-time digest failed to cover."""
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")


def chain_commitment(rows: list[dict]) -> str:
    digest = hashlib.sha256(ANCHOR_DOMAIN)
    for row in rows:
        digest.update(canonical_row_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_rows(rows: list[dict], spec: LedgerSpec, report: Report) -> None:
    """Recompute every digest, verify linkage, ordering and uniqueness."""
    previous_hash = GENESIS
    seen_hashes: dict[str, int] = {}
    previous_ts = ""
    for index, row in enumerate(rows):
        stored = row.get("hash")
        if not isinstance(stored, str) or len(stored) != 64:
            report.violations.append(f"row {index}: missing or malformed `hash`")
            previous_hash = stored if isinstance(stored, str) else previous_hash
            continue

        # (1) THE CHECK THAT DID NOT EXIST: recompute the digest from the row's own content.
        full_body = {k: v for k, v in row.items() if k != "hash"}
        if spec.digest(full_body) == stored:
            pass
        elif index < spec.legacy_uncovered_rows:
            legacy_body = {k: v for k, v in full_body.items() if k not in spec.uncovered_fields}
            if legacy_body != full_body and spec.digest(legacy_body) == stored:
                report.n_legacy += 1
            else:
                report.violations.append(
                    f"row {index}: FORGED CONTENT -- recomputed digest does not match the stored "
                    f"hash {stored[:16]}... (neither the full body nor the legacy body)")
        else:
            report.violations.append(
                f"row {index}: FORGED CONTENT -- recomputed digest does not match the stored "
                f"hash {stored[:16]}...; every row at or beyond index "
                f"{spec.legacy_uncovered_rows} must cover every field it carries")

        # (2) linkage.
        declared_prev = row.get("prev_hash")
        if declared_prev != previous_hash:
            where = "genesis prev_hash must be 64 zeros" if index == 0 else f"row {index - 1}'s hash"
            report.violations.append(
                f"row {index}: BROKEN LINK -- prev_hash {str(declared_prev)[:16]}... != {where} "
                f"({previous_hash[:16]}...)")

        # (3) a repeated digest means a duplicated or reordered entry.
        if stored in seen_hashes:
            report.violations.append(
                f"row {index}: DUPLICATE entry -- identical digest already at row {seen_hashes[stored]}")
        else:
            seen_hashes[stored] = index

        # (4) a ledger of "recorded before the event" may not travel backwards in time.
        timestamp = str(row.get("timestamp", ""))
        if timestamp and previous_ts and timestamp < previous_ts:
            report.violations.append(
                f"row {index}: OUT OF ORDER -- timestamp {timestamp} precedes row {index - 1}'s "
                f"{previous_ts}")
        previous_ts = timestamp or previous_ts
        previous_hash = stored

    report.n_rows = len(rows)
    report.head_hash = rows[-1].get("hash") if rows else None
    report.first_hash = rows[0].get("hash") if rows else None
    report.chain_commitment = chain_commitment(rows)


def verify_anchor(report: Report, anchor: dict | None) -> None:
    """A length/head commitment: without it, dropping the tail leaves a perfectly valid chain."""
    if anchor is None:
        report.violations.append(
            f"NO ANCHOR for {report.path}. A truncated chain is still a valid chain -- without a "
            f"committed length and head, dropping the tail is undetectable. "
            f"Run: python scripts/verify_ledger_chain.py --update-anchor")
        return
    expected = {
        "n_entries": report.n_rows,
        "head_hash": report.head_hash,
        "first_hash": report.first_hash,
        "chain_commitment": report.chain_commitment,
    }
    for key, value in expected.items():
        recorded = anchor.get(key)
        if recorded != value:
            report.violations.append(
                f"ANCHOR MISMATCH on {key}: anchor says {recorded!r}, the ledger computes {value!r}"
                + (" -- entries were REMOVED" if key == "n_entries"
                   and isinstance(recorded, int) and report.n_rows < recorded else "")
                + ". If the ledger legitimately GREW (a scoring run, or a merge that brought in "
                  "newer entries), re-run --update-anchor and review the diff; if it did not, this "
                  "artifact has been altered.")


def verify_continuity(spec: LedgerSpec, rows: list[dict], base_ref: str, report: Report) -> None:
    """Append-only ACROSS COMMITS: the base chain must survive intact as a prefix of this one."""
    base_text = blob_at(base_ref, spec.path)
    if base_text is None:
        report.notes.append(f"no {spec.path} at {base_ref}: new ledger, continuity not applicable")
        return
    base_rows = [json.loads(x) for x in base_text.splitlines() if x.strip()]
    if len(rows) < len(base_rows):
        report.violations.append(
            f"APPEND-ONLY VIOLATION vs {base_ref}: {len(base_rows)} entries there, {len(rows)} here "
            f"({len(base_rows) - len(rows)} DESTROYED)")
    for index in range(min(len(rows), len(base_rows))):
        if rows[index].get("hash") != base_rows[index].get("hash"):
            report.violations.append(
                f"HISTORY REWRITTEN vs {base_ref}: row {index} has digest "
                f"{str(rows[index].get('hash'))[:16]}... but {base_ref} recorded "
                f"{str(base_rows[index].get('hash'))[:16]}...")
            break


def load_anchors(ref: str | None) -> dict:
    text, _ = read_source(ANCHOR_PATH.relative_to(ROOT).as_posix(), ref)
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data.get("ledgers", {}) if isinstance(data, dict) else {}


def check_all(ref: str | None, base: str | None) -> list[Report]:
    anchors = load_anchors(ref)
    reports: list[Report] = []
    for spec in LEDGERS:
        report = Report(path=spec.path)
        text, origin = read_source(spec.path, ref)
        if text is None:
            report.violations.append(
                f"{spec.path} NOT FOUND (worktree and git). A ledger that cannot be read has not "
                f"been verified; refusing to report green.")
            reports.append(report)
            continue
        report.notes.append(f"read from {origin}; digest per {spec.writer}")
        rows = parse_rows(text, report)
        verify_rows(rows, spec, report)
        verify_anchor(report, anchors.get(spec.path))
        if base:
            verify_continuity(spec, rows, base, report)
        reports.append(report)
    return reports


def build_anchor_document(ref: str | None) -> dict:
    ledgers: dict[str, dict] = {}
    for spec in LEDGERS:
        text, _ = read_source(spec.path, ref)
        if text is None:
            raise SystemExit(f"cannot anchor {spec.path}: not found in the worktree or in git")
        report = Report(path=spec.path)
        rows = parse_rows(text, report)
        verify_rows(rows, spec, report)
        if report.violations:
            raise SystemExit(
                f"REFUSING to anchor {spec.path}: it does not verify.\n  "
                + "\n  ".join(report.violations[:10]))
        ledgers[spec.path] = {
            "n_entries": report.n_rows,
            "first_hash": report.first_hash,
            "head_hash": report.head_hash,
            "chain_commitment": report.chain_commitment,
            "digest_separators": "compact" if spec.compact else "default",
            "legacy_rows_not_covering_forecast_id": spec.legacy_uncovered_rows,
        }
    return {
        "mode": "append_only",
        "schema": "hazardpulse.ledger-anchors.v1",
        "explain": (
            "Length and head commitment for each hash-chain ledger. `chain_commitment` is "
            "sha256 over the domain tag then every complete canonical row (sort_keys, compact "
            "separators, INCLUDING each row's own `hash` and any field the write-time digest "
            "does not cover). Truncating, reordering or rewriting any row changes it."
        ),
        "ledgers": ledgers,
    }


def refresh_anchor(ref: str | None = None) -> dict:
    """Rewrite the anchor from the current ledgers. RAISES if a ledger does not verify.

    The appenders call this so the length/head commitment cannot go stale on the first real
    append -- a stale anchor fails the gate for the wrong reason and trains people to ignore it.
    """
    document = build_anchor_document(ref)
    ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


# ---------------------------------------------------------------------------
# self-test: a gate nobody has watched fail is not a gate
# ---------------------------------------------------------------------------


def _forge_chain(spec: LedgerSpec, n: int = 6, probability: float = 0.1) -> list[dict]:
    """Build a chain the way the production writers build one, so the fixture is not a strawman."""
    rows: list[dict] = []
    previous = GENESIS
    for i in range(n):
        entry = {
            "forecast_id": f"fx_{i:04d}",
            "timestamp": f"2026-01-{i + 1:02d}T00:00:00Z",
            "model_version": "test_v1",
            "top_probability": round(probability + i / 100, 4),
            "prev_hash": previous,
        }
        entry["hash"] = spec.digest(entry)
        rows.append(entry)
        previous = entry["hash"]
    return rows


_NO_ANCHOR_ARG = object()


def _verify(rows: list[dict], spec: LedgerSpec, anchor=_NO_ANCHOR_ARG) -> Report:
    """`anchor=None` means THE ANCHOR IS MISSING (fatal); omitting it skips the anchor stage."""
    report = Report(path=spec.path)
    verify_rows(rows, spec, report)
    if anchor is not _NO_ANCHOR_ARG:
        verify_anchor(report, anchor)
    return report


def self_test() -> int:
    """Four forgeries the old check accepted. Each must be REJECTED here."""
    failures: list[str] = []
    for spec in LEDGERS:
        clean = _forge_chain(spec)
        baseline = _verify(clean, spec)
        if not baseline.ok:
            failures.append(f"{spec.path}: an HONEST chain was rejected: {baseline.violations}")
            continue
        anchor = {
            "n_entries": baseline.n_rows,
            "first_hash": baseline.first_hash,
            "head_hash": baseline.head_hash,
            "chain_commitment": baseline.chain_commitment,
        }

        # 1. mutate a middle entry, leaving every stored hash and pointer untouched.
        mutated = copy.deepcopy(clean)
        mutated[2]["top_probability"] = 0.99
        if _verify(mutated, spec, anchor).ok:
            failures.append(f"{spec.path}: a MUTATED middle entry was accepted")
        # ...and it must be caught by RECOMPUTING THE DIGEST, not merely by the anchor. Without
        # this line the whole digest stage can be deleted and this self-test still prints GREEN
        # (measured: it did).
        no_anchor = _verify(mutated, spec)
        if not any("FORGED CONTENT" in v for v in no_anchor.violations):
            failures.append(f"{spec.path}: the per-row digest recomputation is NOT LIVE -- a mutated "
                            f"entry survived every check except the anchor: {no_anchor.violations}")

        # 2. drop the tail. The remaining chain is internally perfect.
        truncated = copy.deepcopy(clean)[:3]
        if _verify(truncated, spec, anchor).ok:
            failures.append(f"{spec.path}: a TRUNCATED ledger was accepted")
        # The truncated chain passes every per-row check: linkage and digests are all intact.
        # That is exactly WHY the anchor exists, so a MISSING anchor must itself be fatal.
        if _verify(truncated, spec).ok is not True:
            failures.append(f"{spec.path}: the truncation fixture is a strawman -- it fails the "
                            f"per-row checks, so it does not prove the anchor is what catches it")
        if not any("NO ANCHOR" in v for v in _verify(truncated, spec, None).violations):
            failures.append(f"{spec.path}: a missing anchor was not reported as fatal")

        # 3. forge an entire history from scratch, correctly chained.
        forged = _forge_chain(spec, n=6, probability=0.9)
        report = _verify(forged, spec, anchor)
        if report.ok:
            failures.append(f"{spec.path}: a WHOLLY FORGED chain was accepted")
        if not any("ANCHOR MISMATCH" in v for v in report.violations):
            failures.append(f"{spec.path}: the forged chain was rejected for the WRONG reason: "
                            f"{report.violations}")

        # 4. reorder two adjacent entries. The LINKAGE stage must catch this on its own.
        reordered = copy.deepcopy(clean)
        reordered[2], reordered[3] = reordered[3], reordered[2]
        if _verify(reordered, spec, anchor).ok:
            failures.append(f"{spec.path}: a REORDERED ledger was accepted")
        if not any("BROKEN LINK" in v for v in _verify(reordered, spec).violations):
            failures.append(f"{spec.path}: the linkage check is NOT LIVE -- a reordered ledger was "
                            f"caught only by the anchor")

        # 5. anti-strawman: the mutation must be invisible to the OLD prev_hash-only check,
        #    or this self-test is not testing the defect that was actually found.
        def old_check(rows: list[dict]) -> int:
            mismatches, previous = 0, GENESIS
            for i, row in enumerate(rows):
                if str(row.get("prev_hash", "")) != (previous if i else GENESIS):
                    mismatches += 1
                previous = str(row.get("hash", previous))
            return mismatches
        if old_check(mutated) != 0 or old_check(truncated) != 0 or old_check(forged) != 0:
            failures.append(f"{spec.path}: the fixtures are strawmen -- the OLD check already "
                            f"caught them, so they do not reproduce the reported defect")

    if not LEDGERS:
        print("SELF-TEST FAILED: no ledger specs are registered, so this gate checks nothing")
        return 1
    if failures:
        print("SELF-TEST FAILED:", *failures, sep="\n  ")
        return 1
    print(f"SELF-TEST GREEN: for {len(LEDGERS)} ledger spec(s), an honest chain is accepted and "
          f"mutate-middle / truncate-tail / forge-whole-chain / reorder are all REJECTED -- while "
          f"the old prev_hash-only check reports 0 mismatches for every one of them.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default=None, help="verify the ledgers as they exist at this git ref")
    ap.add_argument("--base", default=None,
                    help="also require the chain at this ref to survive as an intact prefix")
    ap.add_argument("--update-anchor", action="store_true",
                    help="rewrite evidence/ledger-anchors.json from the current ledgers")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.update_anchor:
        document = refresh_anchor(args.ref)
        print(f"Wrote {ANCHOR_PATH.relative_to(ROOT).as_posix()}")
        for path, entry in document["ledgers"].items():
            print(f"  {path}: {entry['n_entries']} entries, head {entry['head_hash'][:16]}..., "
                  f"commitment {entry['chain_commitment'][:16]}...")
        return 0

    if not LEDGERS:
        print("LEDGER GATE VACUOUS: no ledgers registered", file=sys.stderr)
        return 1

    reports = check_all(args.ref, args.base)
    failed = False
    for report in reports:
        for note in report.notes:
            print(f"  note: {report.path}: {note}")
        if report.ok:
            print(f"OK  {report.path}: {report.n_rows} entries, every digest RECOMPUTED and matched, "
                  f"chain linked, anchor agrees"
                  + (f" ({report.n_legacy} legacy rows whose digest predates `forecast_id`; "
                     f"pinned and covered by the anchor)" if report.n_legacy else ""))
        else:
            failed = True
            print(f"\nFAIL {report.path} ({len(report.violations)} violation(s)):")
            for violation in report.violations[:20]:
                print(f"  - {violation}")
            if len(report.violations) > 20:
                print(f"  ... and {len(report.violations) - 20} more")
    if failed:
        print("\nLEDGER VERIFICATION FAILED. These artifacts back the claim 'this prediction was "
              "recorded before the event'. Do not publish them.")
        return 1
    print(f"\nLEDGER GATE GREEN - {len(reports)} chain(s) verified by RECOMPUTING every digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
