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
#:
#: The two `.jsonl` files are the ACTUAL SHA-256 hash chains (499 and 1367 entries on origin/main as
#: of 2026-08-06). Until this commit they were covered by NOTHING: `discover()` only globbed
#: `dist/data/evidence/**.json`, and the real chains are `.jsonl` files outside that prefix. The gate
#: printed "GATE GREEN - 4 evidence artifact(s) checked" while the two artifacts the product's
#: central claim actually rests on were not among them.
KNOWN_EVIDENCE = (
    "dist/data/evidence/prediction-ledger.json",
    "dist/data/evidence/gate-decisions.json",
    "dist/data/evidence/provenance-envelopes.json",
    "dist/data/evidence/replay-index.json",
    "dist/data/earthquake-ledger.jsonl",
    "dist/data/tornado-ledger.jsonl",
    "evidence/ledger-anchors.json",
)

COUNT_KEYS = ("entries", "records", "items", "decisions", "envelopes", "predictions")

#: The anchor written by scripts/verify_ledger_chain.py: one entry per chain, each with its own
#: length. Summing them would let a shrink in one chain hide behind growth in the other.
ANCHOR_PATH = "evidence/ledger-anchors.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout


def blob_at(ref: str, path: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def jsonl_count(text: str) -> tuple[int | None, bool]:
    """A .jsonl hash chain: one record per line. Any unparseable line makes the count meaningless."""
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        try:
            json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None, False
    return len(lines), True          # a hash chain IS append-only by construction


def anchor_counts(text: str) -> dict[str, int]:
    """Per-chain lengths from the anchor. Per-chain, never summed."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    ledgers = data.get("ledgers") if isinstance(data, dict) else None
    if not isinstance(ledgers, dict):
        return {}
    return {name: entry["n_entries"] for name, entry in ledgers.items()
            if isinstance(entry, dict) and isinstance(entry.get("n_entries"), int)}


def record_count(text: str, path: str = "") -> tuple[int | None, bool]:
    """Returns (count, declares_append_only). None means 'no countable record list'."""
    if path.endswith(".jsonl"):
        return jsonl_count(text)
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
    """Every tracked append-only artifact: the evidence JSON, the .jsonl hash chains, the anchor.

    Discovery is by declaration (`"mode": "append_only"`) OR by living in the evidence directory
    OR by being a `*-ledger.jsonl` hash chain. The last clause is the one that was missing: the
    real chains are `.jsonl` and live in `dist/data/`, not `dist/data/evidence/`.
    """
    found = set(KNOWN_EVIDENCE)
    for path in git("ls-tree", "-r", "--name-only", ref).splitlines():
        if path.endswith(".json") and path.startswith("dist/data/evidence/"):
            found.add(path)
        if path.endswith("-ledger.jsonl") and path.startswith("dist/data/"):
            found.add(path)
    return sorted(found)


def check(base: str, head: str) -> tuple[list[str], list[str], int]:
    """Returns (violations, notes, n_compared). n_compared is how many artifacts were REALLY
    compared -- discovery alone proves nothing if every comparison was skipped."""
    violations: list[str] = []
    notes: list[str] = []
    compared = 0
    for path in discover(head):
        base_text, head_text = blob_at(base, path), blob_at(head, path)
        if head_text is None:
            if path in KNOWN_EVIDENCE and base_text is not None:
                violations.append(
                    f"{path}: present at {base}, GONE at {head}. Deleting an append-only evidence "
                    f"artifact destroys every record in it.")
            else:
                notes.append(f"{path}: absent at {head}; nothing to compare")
            continue
        if base_text is None:
            notes.append(f"{path}: new at {head} (absent at {base}); nothing to compare")
            continue
        base_n, _ = record_count(base_text, path)
        head_n, head_declares = record_count(head_text, path)
        if base_n is None or head_n is None:
            notes.append(f"{path}: no countable record list at one of the refs; NOT COMPARED")
            continue
        compared += 1
        if not head_declares:
            notes.append(f"{path}: treated as append-only evidence but does NOT declare "
                         f'"mode": "append_only" -- add the marker so the contract is in the file')
        if head_n < base_n:
            violations.append(
                f"{path}: {base_n} records at {base} -> {head_n} at {head} "
                f"({base_n - head_n} DESTROYED). An append-only ledger may grow, never shrink.")

    # The anchor carries one length PER CHAIN. Checking only the file-level record count would let a
    # shrink in one chain hide behind growth in the other.
    base_anchor, head_anchor = blob_at(base, ANCHOR_PATH), blob_at(head, ANCHOR_PATH)
    if base_anchor and head_anchor:
        base_map, head_map = anchor_counts(base_anchor), anchor_counts(head_anchor)
        for name, base_n in base_map.items():
            head_n = head_map.get(name)
            if head_n is None:
                violations.append(f"{ANCHOR_PATH}: chain {name} was anchored at {base} and is GONE "
                                  f"at {head}")
            elif head_n < base_n:
                violations.append(f"{ANCHOR_PATH}: chain {name} anchored at {base_n} entries, now "
                                  f"{head_n} ({base_n - head_n} DESTROYED)")
            else:
                compared += 1
    return violations, notes, compared


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

    # The hash chains are .jsonl. Counting them as if they were JSON documents returns None, which
    # the comparison treats as "nothing to compare" -- that is how 1866 chained records went
    # unchecked while the gate printed GREEN.
    chain = '{"a": 1}\n{"a": 2}\n\n{"a": 3}\n'
    n_chain, chain_declares = record_count(chain, "dist/data/earthquake-ledger.jsonl")
    if (n_chain, chain_declares) != (3, True):
        print(f"SELF-TEST FAILED: a 3-entry .jsonl chain counted as {n_chain}/{chain_declares}")
        return 1
    if record_count(chain)[0] is not None:
        print("SELF-TEST FAILED: a .jsonl chain must not be counted as a JSON document")
        return 1
    if record_count('{"a": 1}\nnot json\n', "x.jsonl")[0] is not None:
        print("SELF-TEST FAILED: an unparseable chain line produced a count anyway")
        return 1
    for known in ("dist/data/earthquake-ledger.jsonl", "dist/data/tornado-ledger.jsonl",
                  ANCHOR_PATH):
        if known not in KNOWN_EVIDENCE:
            print(f"SELF-TEST FAILED: {known} is not in the discovery set, so the real hash chain "
                  f"is not covered by this gate")
            return 1

    # Per-chain anchor lengths must never be summed.
    anchor_base = json.dumps({"ledgers": {"a": {"n_entries": 100}, "b": {"n_entries": 100}}})
    anchor_head = json.dumps({"ledgers": {"a": {"n_entries": 150}, "b": {"n_entries": 50}}})
    base_map, head_map = anchor_counts(anchor_base), anchor_counts(anchor_head)
    if not (base_map == {"a": 100, "b": 100} and head_map == {"a": 150, "b": 50}):
        print(f"SELF-TEST FAILED: anchor counts parsed as {base_map} / {head_map}")
        return 1
    if sum(base_map.values()) != sum(head_map.values()):
        print("SELF-TEST FAILED: the hidden-shrink fixture must have an unchanged TOTAL, or it "
              "does not prove that per-chain comparison is what catches it")
        return 1
    if not any(head_map[k] < v for k, v in base_map.items()):
        print("SELF-TEST FAILED: the hidden-shrink fixture does not actually shrink a chain")
        return 1

    print("SELF-TEST GREEN: a shrink is counted as smaller, an undeclared file is flagged, "
          "unparseable content yields no count instead of a zero, a .jsonl hash chain is counted "
          "by line, both live chains are in the discovery set, and a per-chain shrink that leaves "
          "the TOTAL unchanged is still visible.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="origin/main", help="ref the ledger must not have shrunk from")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    # A tree compared to ITSELF can never shrink, so it always passes. That is not a green gate,
    # it is a broken one -- and it was the exact invocation the push-to-main workflow used
    # (`--base origin/main --head HEAD` on a push to main resolves to one commit). Compare the
    # RESOLVED commits, not the ref strings: "origin/main" and "HEAD" look different and are not.
    base_sha = git("rev-parse", args.base).strip()
    head_sha = git("rev-parse", args.head).strip()
    if not base_sha or not head_sha:
        print(f"APPEND-ONLY GATE: cannot resolve {args.base!r} / {args.head!r}. Failing closed.",
              file=sys.stderr)
        return 1
    if base_sha == head_sha:
        print(f"APPEND-ONLY GATE MISCONFIGURED: --base {args.base!r} and --head {args.head!r} are "
              f"the SAME commit ({base_sha[:12]}). Comparing a tree to itself proves nothing.",
              file=sys.stderr)
        return 1

    checked = discover(args.head)
    if not checked:
        # An empty check passes for any input. Say so instead of printing a confident green.
        print("APPEND-ONLY GATE VACUOUS: no evidence artifacts discovered", file=sys.stderr)
        return 1

    violations, notes, compared = check(args.base, args.head)
    for note in notes:
        print(f"  note: {note}")
    if violations:
        print(f"\nAPPEND-ONLY VIOLATION ({len(violations)}):", *violations, sep="\n  ")
        print("\nResolve these files toward the LIVE branch. A regenerated evidence artifact from an "
              "older fork point is stale data, not a change.")
        return 1
    if not compared:
        # Discovery is not verification. Every artifact can be discovered and every comparison
        # skipped (missing at a ref, uncountable, wrong extension) and the gate would still be
        # green -- which is precisely how the .jsonl chains went unchecked.
        print(f"APPEND-ONLY GATE VACUOUS: {len(checked)} artifact(s) discovered but ZERO were "
              f"actually compared against {args.base}.", file=sys.stderr)
        return 1
    print(f"APPEND-ONLY GATE GREEN - {compared} of {len(checked)} discovered artifact(s) compared "
          f"against {args.base}; none lost records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
