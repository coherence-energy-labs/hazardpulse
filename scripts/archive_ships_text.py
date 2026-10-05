#!/usr/bin/env python3
"""Keep every SHIPS text NHC publishes -- every VERSION of it (docs/MODEL_IMPROVEMENT_LEDGER.md, H4;
docs/HURRICANE_RI_V9_PROGRAM.md amendment 7, rule 3).

    python scripts/archive_ships_text.py [--limit N]
    python scripts/archive_ships_text.py --backfill-from-git     (once: retrieval bounds for old files)

NHC's ``atcf/stext`` directory holds only the current season, and our forecast files keep only the
RI probabilities. The full text also carries the operational environment predictors (shear, SST,
ocean heat content, humidity, ...) that a future model can train on in exactly the form it will
read live, but only if the files were kept.

NHC REWRITES a text after first publishing it: Rachel 2026-10-03 18Z was first published with a
DTOPS line (DTOPS 0%) and rewritten by 18:50 without it, so the first version and the final one give
different NOAA numbers. Keeping only the first version seen made a rebuild of what a forecast read
impossible to check. So:

* ``<year>/<name>.gz`` is the FIRST version archived (gzip with no timestamp, so the bytes depend
  only on the text). It is never rewritten.
* every later DISTINCT version is ``<year>/versions/<name>.<retrieved UTC>.gz``. Never rewritten,
  never deleted.
* ``<year>/versions.jsonl`` (append-only) logs every retrieval: the file it is stored in, the text's
  SHA-256, when it was retrieved, and the directory listing's last-modified time at that moment
  (``kind`` "version" for new bytes, "seen" for bytes already kept).

A listed file is fetched again only when the listing says it may have changed since we last read it
(its last-modified minute is at or after our last retrieval's minute, or differs from the one we last
recorded), so a run reads a handful of files, not the season.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.data.http import fetch_bytes, fetch_text  # noqa: E402
from hazardpulse.hurricane import ships_text  # noqa: E402

ARCHIVE = ROOT / "results" / "hurricane_ships_archive"
NAME = re.compile(r"(\d{2})(\d{6})([A-Z]{2}\d{4})_ships\.txt")
# a listing row: the file's link, then (same row) its last-modified time "YYYY-MM-DD HH:MM"
ROW = re.compile(r'href="(\d{8}[A-Z]{2}\d{4}_ships\.txt)"[^\n]*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')
MANIFEST = "versions.jsonl"
VERSIONS_DIR = "versions"
# a listing without modification times: re-read a file whose cycle is this recent at most this often
NO_LISTING_TIME_WINDOW = dt.timedelta(hours=48)
NO_LISTING_TIME_RECHECK = dt.timedelta(hours=6)


def listed(index_html: str) -> list[str]:
    """SHIPS text file names in a directory listing; anything else (sort links, parent, other
    files, a crafted path) is ignored."""
    return sorted({m.group(0) for m in NAME.finditer(index_html)
                   if f'href="{m.group(0)}"' in index_html})


def listed_modified(index_html: str) -> dict[str, str]:
    """``{name: "YYYY-MM-DD HH:MM"}``: the listing's last-modified time of each file (UTC)."""
    return {name: stamp for name, stamp in ROW.findall(index_html)}


def archive_path(name: str, root: Path = ARCHIVE) -> Path:
    m = NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"not a SHIPS text name: {name!r}")
    return root / f"20{m.group(1)}" / f"{name}.gz"


def version_path(name: str, retrieved_at: dt.datetime, root: Path = ARCHIVE) -> Path:
    base = archive_path(name, root)
    return base.parent / VERSIONS_DIR / f"{name}.{retrieved_at:%Y%m%dT%H%M%SZ}.gz"


def cycle_of(name: str) -> dt.datetime:
    m = NAME.fullmatch(name)
    return dt.datetime.strptime("20" + m.group(1) + m.group(2), "%Y%m%d%H")


def deterministic_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def _z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo is None else t.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _minute(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def manifest(root: Path = ARCHIVE) -> dict[str, list[dict]]:
    """``{name: [log entries, oldest first]}`` over every year's versions.jsonl."""
    out: dict[str, list[dict]] = {}
    for p in sorted(root.glob(f"*/{MANIFEST}")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out.setdefault(e["name"], []).append(e)
    return out


def versions(name: str, root: Path = ARCHIVE) -> list[dict]:
    """Every distinct stored version of a text, oldest first, each with its retrieval time."""
    return [e for e in manifest(root).get(name, []) if e.get("kind") == "version"]


def _append(root: Path, name: str, entry: dict) -> None:
    p = archive_path(name, root).parent / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stored_sha(path: Path) -> str:
    return _sha(gzip.decompress(path.read_bytes()))


def needs_read(name: str, log: list[dict], listed_at: str | None, now: dt.datetime) -> bool:
    """Whether a file already archived may have changed since we last read it."""
    if not log:
        return True                         # archived before versions were logged: read it once
    last = log[-1]
    seen_at = _parse(last.get("retrieved_at"))
    if listed_at is None:                   # a listing without times: re-read recent cycles, sparingly
        recent = now - cycle_of(name) <= NO_LISTING_TIME_WINDOW
        return recent and (seen_at is None or now - seen_at >= NO_LISTING_TIME_RECHECK)
    if seen_at is None:                     # only an upper bound on when the first version was read
        bound = _parse(last.get("archived_by_commit_at"))
        return bound is None or listed_at >= _minute(bound)
    return listed_at != last.get("listed_modified") or listed_at >= _minute(seen_at)


def sync(index_html: str, fetch, root: Path = ARCHIVE, limit: int | None = None, pause: float = 0.0,
         now: dt.datetime | None = None) -> dict:
    """Archive every listed SHIPS text the archive lacks, and every new version of one it has;
    returns counts. Nothing already archived is ever rewritten or removed."""
    names = listed(index_html)
    stamps = listed_modified(index_html)
    log = manifest(root)
    missing = [n for n in names if not archive_path(n, root).exists()]
    now_ = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    recheck = [n for n in names if n not in missing and needs_read(n, log.get(n, []), stamps.get(n), now_)]
    todo = missing + recheck
    todo = todo if limit is None else todo[:limit]
    failed, archived, new_versions, same = [], 0, 0, 0
    for n in todo:
        try:
            data = fetch(ships_text.STEXT_ROOT + n)
        except Exception as exc:  # noqa: BLE001 -- one missing file must not stop the rest
            failed.append(f"{n}: {type(exc).__name__}")
            continue
        if not data.strip():
            failed.append(f"{n}: empty")
            continue
        got_at = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)
        entry = {"name": n, "sha256": _sha(data), "retrieved_at": _z(got_at), "listed_modified": stamps.get(n)}
        base = archive_path(n, root)
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_bytes(deterministic_gzip(data))
            entry.update(kind="version", file=_rel(base, root))
            archived += 1
        else:
            if not log.get(n):              # the base file predates the log: log it first, time unknown
                legacy = {"name": n, "kind": "version", "file": _rel(base, root), "sha256": _stored_sha(base),
                          "retrieved_at": None, "listed_modified": None, "archived_by_commit_at": None,
                          "note": "archived before retrieval times were logged; retrieval time not recorded"}
                _append(root, n, legacy)
                log[n] = [legacy]
            kept = {e["sha256"]: e["file"] for e in log[n] if e.get("kind") == "version"}
            if entry["sha256"] in kept:
                entry.update(kind="seen", file=kept[entry["sha256"]])
                same += 1
            else:
                vp = version_path(n, got_at, root)
                if vp.exists():             # same second, different bytes: never overwrite
                    failed.append(f"{n}: a version already stored at {vp.name}")
                    continue
                vp.parent.mkdir(parents=True, exist_ok=True)
                vp.write_bytes(deterministic_gzip(data))
                entry.update(kind="version", file=_rel(vp, root))
                new_versions += 1
        _append(root, n, entry)
        log.setdefault(n, []).append(entry)
        if pause:
            time.sleep(pause)
    return {"listed": len(names), "already": len(names) - len(missing), "archived_now": archived,
            "rechecked": len([n for n in todo if n in recheck]), "new_versions": new_versions,
            "unchanged": same, "failed": failed}


def backfill_from_git(root: Path = ARCHIVE, repo: Path = ROOT) -> int:
    """One-off: log every base file archived before versions were logged, with its text's SHA-256 and
    the time of the commit that added it -- an upper bound on when it was retrieved (the archive step
    runs just before the hurricane workflow commits). Files already in the log are left alone."""
    out = subprocess.run(["git", "-C", str(repo), "log", "--diff-filter=A", "--format=C %cI", "--name-only",
                          "--", str(root.relative_to(repo)).replace("\\", "/")],
                         check=True, capture_output=True, text=True).stdout
    added: dict[str, str] = {}
    when = None
    for line in out.splitlines():
        if line.startswith("C "):
            when = _z(_parse(line[2:].strip()))
        elif line.strip().endswith("_ships.txt.gz") and "/" + VERSIONS_DIR + "/" not in line:
            # git lists newest first; a base file is added once (it is never rewritten or removed)
            added.setdefault(Path(line.strip()).name[:-3], when)
    log = manifest(root)
    n = 0
    for base in sorted(root.glob("*/*_ships.txt.gz")):
        name = base.name[:-3]
        if log.get(name):
            continue
        _append(root, name, {"name": name, "kind": "version", "file": _rel(base, root), "sha256": _stored_sha(base),
                             "retrieved_at": None, "listed_modified": None,
                             "archived_by_commit_at": added.get(name),
                             "note": "archived before retrieval times were logged; retrieved at or before "
                                     "archived_by_commit_at"})
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backfill-from-git", action="store_true")
    args = ap.parse_args(argv)
    if args.backfill_from_git:
        print(f"SHIPS text archive: {backfill_from_git()} files logged from git history")
        return 0
    index = fetch_text(ships_text.STEXT_ROOT, namespace="ships_index", use_cache=False)
    res = sync(index, lambda url: fetch_bytes(url, namespace="ships_archive", use_cache=False),
               limit=args.limit, pause=0.05)
    print(f"SHIPS text archive: {res['listed']} listed, {res['archived_now']} archived now, "
          f"{res['rechecked']} re-read ({res['new_versions']} new versions, {res['unchanged']} unchanged), "
          f"{len(res['failed'])} failed" + (f" ({'; '.join(res['failed'][:5])})" if res["failed"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
