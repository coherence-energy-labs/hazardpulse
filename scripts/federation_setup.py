#!/usr/bin/env python3
"""Bootstrap HazardPulse as a Signalbook federation peer.

Generates an Ed25519 keypair, writes a Signalbook-compatible
``federation.toml`` config, and registers HazardPulse's atlas tables
(prediction-ledger, evidence/, model_weights_registry.json) as
queryable surfaces.

After this runs, the operator can join the federation by:

    signalbook federated server start --config ~/.hazardpulse/federation.toml

Other peers can then query HazardPulse's atlas with cryptographically
verified Ed25519 signatures, last-writer-wins conflict resolution,
delta-sync watermarks, and trust-level enforcement.

Run once per node:

    python scripts/federation_setup.py [--node-id hazardpulse-prod-1]
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
from pathlib import Path

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _generate_keypair() -> tuple[bytes, bytes]:
    """Return (private_key_bytes, public_key_bytes_b64). Ed25519."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError(
            "cryptography library is required: pip install cryptography"
        )
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv_bytes, base64.b64encode(pub_bytes)


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--node-id", default=f"hazardpulse-{os.uname().nodename if hasattr(os, 'uname') else 'win'}",
        help="Short identifier for this peer in the federation.",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        default=Path.home() / ".hazardpulse",
        help="Directory to write federation config + keypair.",
    )
    parser.add_argument(
        "--peers-db", type=Path, default=None,
        help="SQLite path for federation peers table (default <config>/peers.db)",
    )
    parser.add_argument(
        "--trust-level", default="trusted",
        choices=("trusted", "readonly", "untrusted"),
        help="Trust level applied to incoming peer rows by default.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config.",
    )
    args = parser.parse_args(argv)

    if not HAS_CRYPTOGRAPHY:
        print("ERROR: cryptography library not installed.", file=sys.stderr)
        print("       Install: pip install cryptography", file=sys.stderr)
        return 1

    config_dir = args.config_dir
    config_path = config_dir / "federation.toml"
    key_path = config_dir / "federation_local.pem"
    peers_db = args.peers_db or (config_dir / "peers.db")

    if config_path.exists() and not args.force:
        print(f"  Config already exists at {config_path}. Use --force to overwrite.")
        return 0

    print(f"Generating Ed25519 keypair for node '{args.node_id}'...")
    priv_bytes, pub_b64 = _generate_keypair()
    pub_str = pub_b64.decode("ascii")

    config_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(priv_bytes)
    try:
        key_path.chmod(0o600)
    except (NotImplementedError, OSError):
        pass  # Windows permission model differs

    # Atlas surface declarations — HazardPulse atlases that peers can query.
    atlas_tables = [
        {
            "name": "hazardpulse_prediction_ledger",
            "modality": "natural_hazard_prediction",
            "path": str(PROJECT_ROOT / "dist" / "data" / "evidence" / "prediction-ledger.json"),
            "schema_url": "https://hazardpulse.com/api/contracts/prediction-ledger.json",
            "row_count_estimate": 250,
        },
        {
            "name": "hazardpulse_provenance_envelopes",
            "modality": "audit_trail",
            "path": str(PROJECT_ROOT / "dist" / "data" / "evidence" / "provenance-envelopes.json"),
            "schema_url": "https://hazardpulse.com/api/contracts/provenance.json",
            "row_count_estimate": 250,
        },
        {
            "name": "hazardpulse_model_weights_registry",
            "modality": "model_weights",
            "path": str(PROJECT_ROOT / "results" / "models" / "model_weights_registry.json"),
            "schema_url": "https://hazardpulse.com/api/contracts/model-registry.json",
            "row_count_estimate": 3,
        },
        {
            "name": "hazardpulse_replay_index",
            "modality": "natural_hazard_prediction",
            "path": str(PROJECT_ROOT / "dist" / "data" / "evidence" / "replay-index.json"),
            "schema_url": "https://hazardpulse.com/api/contracts/replay-index.json",
            "row_count_estimate": 250,
        },
    ]

    toml_lines = [
        "# HazardPulse federation config (Signalbook-compatible).",
        f"# Generated {dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat()}Z",
        "",
        "[local_identity]",
        f'node_id = "{args.node_id}"',
        f'public_key = "{pub_str}"',
        f'private_key_path = "{key_path.as_posix()}"',
        f'private_key_env = "HAZARDPULSE_FEDERATION_PRIVATE_KEY"',
        "",
        "[sync_policy]",
        "default_interval_seconds = 300",
        "max_rows_per_peer_per_sync = 1000",
        f'default_trust_level = "{args.trust_level}"',
        "",
        f'peers_db_path = "{peers_db.as_posix()}"',
        "",
        "[server]",
        'bind_host = "0.0.0.0"',
        "bind_port = 8443",
        "tls_required = true",
        "max_request_bytes = 10485760  # 10 MB",
        "",
        "# --- HazardPulse atlas tables exposed to federation queries ---",
    ]
    for table in atlas_tables:
        toml_lines.extend([
            "",
            "[[atlas_table]]",
            f'name = "{table["name"]}"',
            f'modality = "{table["modality"]}"',
            f'path = "{table["path"]}"',
            f'schema_url = "{table["schema_url"]}"',
            f'row_count_estimate = {table["row_count_estimate"]}',
        ])

    config_path.write_text("\n".join(toml_lines) + "\n", encoding="utf-8")

    # Provenance fingerprint for the keypair (so we can publish the
    # node's pub-key to a directory for trust bootstrapping)
    fingerprint_path = config_dir / "node_fingerprint.json"
    fingerprint = {
        "node_id": args.node_id,
        "public_key_b64": pub_str,
        "key_algorithm": "Ed25519",
        "atlas_tables": [t["name"] for t in atlas_tables],
        "trust_level": args.trust_level,
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "host_os": sys.platform,
    }
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")

    # Also publish a public-safe fingerprint to the worker-served path so
    # peers can discover this node via /api/v1/federation/atlas.
    public_fp = dict(fingerprint)
    public_fp["atlas_tables"] = [
        {"name": t["name"], "modality": t["modality"], "schema_url": t["schema_url"]}
        for t in atlas_tables
    ]
    public_fp_path = PROJECT_ROOT / "dist" / "data" / "federation-fingerprint.json"
    public_fp_path.parent.mkdir(parents=True, exist_ok=True)
    public_fp_path.write_text(json.dumps(public_fp, indent=2) + "\n", encoding="utf-8")
    print(f"  Public fingerprint (worker-served): {public_fp_path}")

    print(f"  Private key:    {key_path}")
    print(f"  Config:         {config_path}")
    print(f"  Peers DB:       {peers_db}")
    print(f"  Fingerprint:    {fingerprint_path}")
    print(f"  Public key b64: {pub_str}")
    print()
    print("Atlas tables exposed to federation queries:")
    for t in atlas_tables:
        print(f"  - {t['name']}  ({t['modality']})")
    print()
    print("Next steps:")
    print(f"  1. Publish node fingerprint to your peer directory.")
    print(f"  2. Add trusted peers to {peers_db} (signalbook federated peers add ...).")
    print(f"  3. Start the federation server:")
    print(f"     signalbook federated server start --config {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
