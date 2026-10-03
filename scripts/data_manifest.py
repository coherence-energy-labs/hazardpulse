#!/usr/bin/env python3
"""Pin the research data behind every model: one manifest per dataset, every file with its size
and SHA-256, committed to the repository.

    PYTHONPATH=src python scripts/data_manifest.py            # write the manifests
    PYTHONPATH=src python scripts/data_manifest.py --verify   # check the caches against them

The datasets live in local caches (gitignored: they are large and rebuilt from public archives).
A manifest makes them identifiable: any result's input table hash (``dev_table_sha256`` and the
like) can be found in it, and a restored or re-downloaded copy can be proven identical -- or
shown to differ, file by file -- with ``--verify``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "data_manifests"


def datasets() -> dict[str, dict]:
    import hurricane_ri_v9 as v9
    return {
        "hurricane_research": {
            "description": "Hurricane RI development and 2026 tables, the NHC a/e-decks they were built from, "
                           "and the storm-centred GMGSI infrared crops (amendments 1-6)",
            "roots": {"hurricane_ri_v9": v9.CACHE / "hurricane_ri_v9", "hurricane_ri_stack": v9.CACHE / "hurricane_ri_stack",
                      "atcf_adecks": v9.CACHE / "atcf_adecks", "hurricane_ir/crops": v9.CACHE / "hurricane_ir" / "crops"},
            "sources": ["NHC ATCF archive (a-, b-, e-decks)", "NHC atcf/stext SHIPS texts (2026)",
                        "NOAA GMGSI longwave IR (s3://noaa-gmgsi-pds)", "IBTrACS (case truth)"],
        },
        "earthquake_program": {
            "description": "The earthquake program's frozen catalogs, per-split forecasts and trees "
                           "(docs/EARTHQUAKE_FORECAST_PROGRAM.md sections 9-10)",
            "roots": {"earthquake/program": ROOT / ".cache" / "earthquake" / "program"},
            "sources": ["USGS ComCat (FDSN), M>=4.5 1973-2026 and M2.5+ 2000-2025"],
        },
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(roots: dict[str, Path]) -> list[list]:
    rows = []
    for label, root in roots.items():
        if not root.exists():
            continue
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            rows.append([f"{label}/{p.relative_to(root).as_posix()}", p.stat().st_size, file_sha256(p)])
    return rows


def aggregate(rows: list[list]) -> str:
    return hashlib.sha256("".join(f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in rows).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, spec in datasets().items():
        path = OUT / f"{name}.json"
        rows = scan(spec["roots"])
        if args.verify:
            want = {r[0]: r for r in json.loads(path.read_text(encoding="utf-8"))["files"]}
            have = {r[0]: r for r in rows}
            missing = sorted(set(want) - set(have))
            changed = sorted(k for k in set(want) & set(have) if want[k][2] != have[k][2])
            extra = sorted(set(have) - set(want))
            bad += len(missing) + len(changed)
            print(f"{name}: {len(want)} pinned; {len(missing)} missing, {len(changed)} changed, {len(extra)} new")
            continue
        manifest = {"dataset": name, "description": spec["description"], "sources": spec["sources"],
                    "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "n_files": len(rows), "total_bytes": sum(r[1] for r in rows),
                    "aggregate_sha256": aggregate(rows), "columns": ["path", "bytes", "sha256"], "files": rows}
        path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{name}: {len(rows)} files, {manifest['total_bytes'] / 1e6:.0f} MB, aggregate {manifest['aggregate_sha256'][:16]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
