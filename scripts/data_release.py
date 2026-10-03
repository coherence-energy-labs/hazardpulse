#!/usr/bin/env python3
"""Package the pinned research datasets for a public release (results/data_manifests/).

    PYTHONPATH=src python scripts/data_release.py --out <dir>

One ``<dataset>.tar.gz`` per manifest, holding exactly the manifest's files -- each re-hashed while it
is packed (a file that no longer matches its manifest stops the run) -- plus the manifest itself
(``MANIFEST.json``) and a ``README.txt`` with sources, licences and how to restore. Archives are
byte-reproducible: members sorted as in the manifest, every timestamp and owner zeroed, gzip
without a time stamp. Restore: extract into the cache root the README names, then
``python scripts/data_manifest.py --verify``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_manifest as dm  # noqa: E402

LICENCE = ("Derived from public data: NOAA/NHC ATCF decks and SHIPS texts, NOAA GMGSI imagery and USGS "
           "ComCat are US-government public domain. HazardPulse's derived tables and model outputs: "
           "AGPL-3.0-only, as the repository.")


def _info(name: str, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = size, 0, 0o644, 0, 0, "", ""
    return ti


def readme(name: str, manifest: dict, restore_root: str) -> str:
    return "\n".join([
        f"HazardPulse research dataset: {name}",
        "",
        manifest["description"],
        "",
        f"Files: {manifest['n_files']:,} ({manifest['total_bytes'] / 1e6:.0f} MB uncompressed)",
        f"Aggregate SHA-256 (of the manifest's path/bytes/sha256 lines): {manifest['aggregate_sha256']}",
        "Sources: " + "; ".join(manifest["sources"]),
        "Licence: " + LICENCE,
        "",
        "Restore:",
        f"  1. extract this archive into {restore_root}",
        "  2. python scripts/data_manifest.py --verify   (every file checked against the committed manifest)",
        "",
        "The manifest committed in the repository (results/data_manifests/) is the reference; MANIFEST.json",
        "here is a copy of it.",
        "",
    ])


def build(name: str, spec: dict, manifest: dict, out_dir: Path) -> dict:
    roots = spec["roots"]
    path = out_dir / f"hazardpulse-data-{name}.tar.gz"
    restore_root = {"hurricane_research": "the hazardpulse cache root (hurricane_ri_v9.CACHE, e.g. <repo>/.cache)",
                    "earthquake_program": "<repo>/.cache"}[name]
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for extra, data in (("MANIFEST.json", json.dumps(manifest, separators=(",", ":")).encode("utf-8")),
                            ("README.txt", readme(name, manifest, restore_root).encode("utf-8"))):
            tar.addfile(_info(extra, len(data)), io.BytesIO(data))
        for rel, size, sha in manifest["files"]:
            label = next(l for l in sorted(roots, key=len, reverse=True) if rel.startswith(l + "/"))
            src = roots[label] / rel[len(label) + 1:]
            data = src.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise SystemExit(f"{rel} no longer matches its manifest: refuse to package")
            tar.addfile(_info(rel, size), io.BytesIO(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"dataset": name, "asset": path.name, "bytes": path.stat().st_size, "sha256": digest,
            "manifest": f"results/data_manifests/{name}.json", "aggregate_sha256": manifest["aggregate_sha256"],
            "n_files": manifest["n_files"], "path": str(path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for name, spec in dm.datasets().items():
        manifest = json.loads((dm.OUT / f"{name}.json").read_text(encoding="utf-8"))
        r = build(name, spec, manifest, out)
        report.append(r)
        print(f"{r['asset']}: {r['n_files']:,} files, {r['bytes'] / 1e6:.0f} MB, sha256 {r['sha256'][:16]}")
    (out / "release.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
