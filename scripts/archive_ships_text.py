#!/usr/bin/env python3
"""Keep every SHIPS text NHC publishes (docs/MODEL_IMPROVEMENT_LEDGER.md, H4).

    python scripts/archive_ships_text.py [--limit N]

NHC's ``atcf/stext`` directory holds only the current season, and our forecast files keep only the
RI probabilities. The full text also carries the operational environment predictors (shear, SST,
ocean heat content, humidity, ...) that a future model can train on in exactly the form it will
read live, but only if the files were kept. This copies every SHIPS text the directory lists and
the archive lacks into ``results/hurricane_ships_archive/<year>/<name>.gz`` (gzip with no
timestamp, so the bytes depend only on the text), and never rewrites a file it already has.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.data.http import fetch_bytes, fetch_text  # noqa: E402
from hazardpulse.hurricane import ships_text  # noqa: E402

ARCHIVE = ROOT / "results" / "hurricane_ships_archive"
NAME = re.compile(r"(\d{2})(\d{6})([A-Z]{2}\d{4})_ships\.txt")


def listed(index_html: str) -> list[str]:
    """SHIPS text file names in a directory listing; anything else (sort links, parent, other
    files, a crafted path) is ignored."""
    return sorted({m.group(0) for m in NAME.finditer(index_html)
                   if f'href="{m.group(0)}"' in index_html})


def archive_path(name: str, root: Path = ARCHIVE) -> Path:
    m = NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"not a SHIPS text name: {name!r}")
    return root / f"20{m.group(1)}" / f"{name}.gz"


def deterministic_gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def sync(index_html: str, fetch, root: Path = ARCHIVE, limit: int | None = None, pause: float = 0.0) -> dict:
    """Archive every listed SHIPS text the archive lacks; returns counts."""
    names = listed(index_html)
    missing = [n for n in names if not archive_path(n, root).exists()]
    new = missing if limit is None else missing[:limit]
    failed = []
    for n in new:
        try:
            data = fetch(ships_text.STEXT_ROOT + n)
        except Exception as exc:  # noqa: BLE001 -- one missing file must not stop the rest
            failed.append(f"{n}: {type(exc).__name__}")
            continue
        if not data.strip():
            failed.append(f"{n}: empty")
            continue
        path = archive_path(n, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(deterministic_gzip(data))
        if pause:
            time.sleep(pause)
    return {"listed": len(names), "already": len(names) - len(missing),
            "archived_now": len(new) - len(failed), "failed": failed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    index = fetch_text(ships_text.STEXT_ROOT, namespace="ships_index", use_cache=False)
    res = sync(index, lambda url: fetch_bytes(url, namespace="ships_archive", use_cache=False),
               limit=args.limit, pause=0.05)
    print(f"SHIPS text archive: {res['listed']} listed, {res['archived_now']} archived now, "
          f"{len(res['failed'])} failed" + (f" ({'; '.join(res['failed'][:5])})" if res["failed"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
