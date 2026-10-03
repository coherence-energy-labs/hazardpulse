"""The data side of the records: published research snapshots agree with the committed manifests,
and the verification workflow audits both hazards' forecast records strictly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "results" / "data_manifests"


def _aggregate(rows) -> str:
    return hashlib.sha256("".join(f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in rows).encode("utf-8")).hexdigest()


def test_every_manifest_is_internally_consistent():
    for p in sorted(MANIFESTS.glob("*.json")):
        if p.name == "releases.json":
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        assert m["n_files"] == len(m["files"]) and m["total_bytes"] == sum(r[1] for r in m["files"]), p.name
        assert m["aggregate_sha256"] == _aggregate(m["files"]), p.name


def test_every_published_snapshot_names_a_committed_manifest_with_its_aggregate():
    rel = json.loads((MANIFESTS / "releases.json").read_text(encoding="utf-8"))["releases"]
    assert rel
    for r in rel:
        assert r["url"].endswith(f"/releases/tag/{r['tag']}")
        for a in r["assets"]:
            m = json.loads((ROOT / a["manifest"]).read_text(encoding="utf-8"))
            assert a["aggregate_sha256"] == m["aggregate_sha256"] and a["n_files"] == m["n_files"], a["asset"]
            assert len(a["sha256"]) == 64 and a["download"].endswith(f"/{r['tag']}/{a['asset']}")


def test_a_results_input_table_is_in_the_pinned_data():
    m = json.loads((MANIFESTS / "hurricane_research.json").read_text(encoding="utf-8"))
    hashes = {r[2] for r in m["files"]}
    for res in ("hurricane_ri_v10_challengers.json", "hurricane_ri_v10_ir.json"):
        sel = json.loads((ROOT / "results" / "calibration" / res).read_text(encoding="utf-8"))
        assert sel["dev_table_sha256"] in hashes, res


def test_the_verification_workflow_audits_both_hazards_strictly():
    wf = (ROOT / ".github" / "workflows" / "verification-score.yml").read_text(encoding="utf-8")
    for script in ("audit_hurricane_records.py", "audit_tornado_records.py"):
        assert f"python scripts/{script} --strict" in wf
    commit = wf.split("Commit and push", 1)[1]
    assert "results/hurricane_prospective/" in commit and "results/tornado_prospective/" in commit
