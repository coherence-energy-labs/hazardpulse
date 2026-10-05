#!/usr/bin/env python3
"""An append-only evidence artifact must never LOSE records. Fail the build if one does.

WHY THIS EXISTS

HazardPulse's entire proposition is "this prediction was recorded before the event." The evidence
artifacts under `dist/data/evidence/` are how that is demonstrated, and `prediction-ledger.json`
declares its own contract in the file: `"mode": "append_only"`.

Nothing enforced it. Measured 2026-08-01, `origin/platform-trust-program` (654 commits, prepared for
a merge to `main`) carries REGENERATED copies of four of those artifacts, each frozen at the record
count from the day the branch forked, while `main` has kept accumulating:

    gate-decisions.json        1256 on the branch   1803 on main
    prediction-ledger.json     1336                 1883
    provenance-envelopes.json  1256                 1803
    replay-index.json          1147                 1694

A merge resolving any of those toward the branch would delete **547 records** from each. Not corrupt
them -- delete them, cleanly, in a commit that looks like an ordinary data update. On a PUBLIC repo
whose product is verifiable prediction provenance, an append-only ledger that silently shrinks is
indistinguishable from backdating, and it is the single most damaging thing that could happen here.
The branch is not malicious; it simply regenerated files a live system owns. That is exactly why a
human reviewing 654 commits would wave it through.

WHAT IT CHECKS

Artifacts are discovered by their OWN declaration -- any JSON carrying `"mode": "append_only"` -- so
a new evidence file is protected the moment it declares itself, with nothing to remember to register.
Files matching the same evidence shape are also checked, because three of the four above do not yet
carry the marker (a gap this script reports rather than silently tolerating).

    python scripts/check_append_only_monotonic.py                 # vs origin/main
    python scripts/check_append_only_monotonic.py --base HEAD~1
    python scripts/check_append_only_monotonic.py --self-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Files known to be append-only evidence. The `"mode": "append_only"` marker is the real contract;
#: this list covers the ones that do not carry it yet, and the gate REPORTS that gap so the marker
#: gets added rather than the list quietly becoming the source of truth.
KNOWN_EVIDENCE = (
    "dist/data/evidence/prediction-ledger.json",
    "dist/data/evidence/gate-decisions.json",
    "dist/data/evidence/provenance-envelopes.json",
    "dist/data/evidence/replay-index.json",
)

COUNT_KEYS = ("entries", "records", "items", "decisions", "envelopes", "predictions")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout


def is_ancestor(a: str, b: str) -> bool:
    """``a`` is ``b`` or in its history."""
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT,
                          capture_output=True, check=False).returncode == 0


def effective_base(base: str, head: str) -> tuple[str | None, str | None, str]:
    """``(ref, since, why)``: the state ``head`` must not have shrunk from, and -- for a change not yet in it
    -- the merge base, so that only the ledger files the change itself modified are compared.

    ``head`` already in ``base``'s history (a push to main; or main moved on while this ran): ``base`` is
    either ``head`` itself -- a comparison that cannot fail -- or a LATER commit, a shrink by construction (the
    merge of PR #28 read one record "destroyed" in four ledgers on 2026-10-05, because a scorer pushed during
    its test run). The question is then what ``head`` did to the state it was built on: its first parent.
    ``ref`` is None when there is none (a root commit cannot have lost a record).

    ``head`` not yet in it (a pull request's merge ref, a branch): would merging it shrink ``base``? Only a
    ledger file the change modified since the merge base can land; a file it never touched keeps main's copy,
    however far main has moved since (PR #29's merge ref was built before two scorer commits and read one
    record "destroyed" for files it never touched). A stale fork that REGENERATED the ledgers modified them,
    so it is still compared with the live branch -- the case this gate exists for."""
    if not is_ancestor(head, base):
        since = git("merge-base", head, base).strip() or None
        return base, since, (f"{head} merges into {base}: ledger files changed since "
                             f"{since[:12] if since else 'the (unknown) merge base'} are compared with it")
    parent = git("rev-parse", "--verify", "--quiet", f"{head}^1").strip()
    why = (f"{head} is already in {base}'s history, so it is checked against the commit it was built on "
           f"({head}^1)")
    return (parent or None), None, why


def blob_at(ref: str, path: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def record_count(text: str) -> tuple[int | None, bool]:
    """Returns (count, declares_append_only). None means 'no countable record list'."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, False
    if isinstance(data, list):
        return len(data), False
    if not isinstance(data, dict):
        return None, False
    declared = data.get("mode") == "append_only"
    for key in COUNT_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return len(value), declared
        if isinstance(value, int):
            return value, declared
    return None, declared


def discover(ref: str) -> list[str]:
    """Every tracked JSON that declares append_only, plus the known evidence artifacts."""
    found = set(KNOWN_EVIDENCE)
    for path in git("ls-tree", "-r", "--name-only", ref).splitlines():
        if path.endswith(".json") and path.startswith("dist/data/evidence/"):
            found.add(path)
    return sorted(found)


def check(base: str, head: str, since: str | None = None) -> tuple[list[str], list[str], int, list[str]]:
    """``(violations, notes, n_compared, untouched)``: ``untouched`` are the files the change did not modify
    since ``since`` (not compared: a merge keeps the live copy)."""
    violations: list[str] = []
    notes: list[str] = []
    untouched: list[str] = []
    n_compared = 0
    for path in discover(head):
        base_text, head_text = blob_at(base, path), blob_at(head, path)
        if base_text is None or head_text is None:
            continue                                  # new or removed file: not a shrink
        if since is not None and blob_at(since, path) == head_text:
            untouched.append(path)
            continue
        base_n, _ = record_count(base_text)
        head_n, head_declares = record_count(head_text)
        if base_n is None or head_n is None:
            continue
        n_compared += 1
        if not head_declares:
            notes.append(f"{path}: treated as append-only evidence but does NOT declare "
                         f'"mode": "append_only" -- add the marker so the contract is in the file')
        if head_n < base_n:
            violations.append(
                f"{path}: {base_n} records at {base} -> {head_n} at {head} "
                f"({base_n - head_n} DESTROYED). An append-only ledger may grow, never shrink.")
    return violations, notes, n_compared, untouched


def self_test() -> int:
    """The gate must refuse a shrink and accept growth, or it is checking nothing."""
    grew = json.dumps({"mode": "append_only", "entries": [1, 2, 3]})
    shrank = json.dumps({"mode": "append_only", "entries": [1]})
    undeclared = json.dumps({"entries": [1]})

    n_grew, d_grew = record_count(grew)
    n_shrank, _ = record_count(shrank)
    if (n_grew, d_grew) != (3, True):
        print(f"SELF-TEST FAILED: counted {n_grew} declared={d_grew}, expected 3/True")
        return 1
    if n_shrank != 1:
        print(f"SELF-TEST FAILED: counted {n_shrank} for a 1-entry ledger")
        return 1
    if record_count(undeclared)[1] is not False:
        print("SELF-TEST FAILED: an undeclared file was reported as declaring append_only")
        return 1
    if record_count("not json at all")[0] is not None:
        print("SELF-TEST FAILED: unparseable content produced a count")
        return 1
    if n_shrank >= n_grew:
        print("SELF-TEST FAILED: the shrink fixture is not smaller, so it proves nothing")
        return 1
    print("SELF-TEST GREEN: a shrink is counted as smaller, an undeclared file is flagged, "
          "and unparseable content yields no count instead of a zero.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="origin/main", help="ref the ledger must not have shrunk from")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    checked = discover(args.head)
    if not checked:
        # An empty check passes for any input. Say so instead of printing a confident green.
        print("APPEND-ONLY GATE VACUOUS: no evidence artifacts discovered", file=sys.stderr)
        return 1

    base, since, why = effective_base(args.base, args.head)
    print(f"  base: {why}")
    if base is None:
        print(f"APPEND-ONLY GATE GREEN - {args.head} is a root commit: there is no earlier state to have lost "
              "records from.")
        return 0
    violations, notes, compared, untouched = check(base, args.head, since)
    for note in notes:
        print(f"  note: {note}")
    if violations:
        print(f"\nAPPEND-ONLY VIOLATION ({len(violations)}):", *violations, sep="\n  ")
        print("\nResolve these files toward the LIVE branch. A regenerated evidence artifact from an "
              "older fork point is stale data, not a change.")
        return 1
    print(f"APPEND-ONLY GATE GREEN - {compared} evidence artifact(s) compared with {base}, none lost records"
          + (f"; {len(untouched)} not modified by this change (a merge keeps the live copy)" if untouched else "")
          + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
