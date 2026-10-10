"""The public model registry (dist/data/model-registry.json): every model that makes a live number
is in it, its hashes are of the bytes git stores, each lineage file it names exists with the hash it
gives, and the committed file is exactly what the repository's models and results produce."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "dist" / "data" / "model-registry.json"

# every artifact a live scorer loads (fetch_and_score*.py)
LIVE_ARTIFACTS = (
    # v8.3 is the served other-basins model (amendment 15); v8.2 is still loaded live, as J1's base
    "hurricane_ri_stack_v1.json", "hurricane_ri_v8_3.json", "hurricane_ri_v8_2.json", "hurricane_ri_v9.json",
    "hurricane_ri_v10.json",
    "hurricane_ri_v10_2.json", "hurricane_ri_v10_3.json", "hurricane_ri_v10_4.json", "hurricane_ri_j1.json",
    "earthquake_operational_v1.json",
    "earthquake_gear1_stack_v1.json", "tornado_v3_w.json", "tornado_v3.json", "tornado_v3_w_30.json",
    "tornado_v3_w_90.json", "tornado_v3_w_ef2.json",
)


@pytest.fixture(scope="module")
def reg_mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("publish_model_registry_t", ROOT / "scripts" / "publish_model_registry.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def committed():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_the_committed_registry_is_what_the_repository_produces(reg_mod, committed):
    """A model or results change that was not re-published fails here (python
    scripts/publish_model_registry.py rewrites it only when its content changed)."""
    assert reg_mod.normalized(committed) == reg_mod.normalized(reg_mod.build_registry())


def test_every_live_model_is_registered_with_the_hash_of_its_stored_bytes(committed):
    assert not any("\\" in e["weights_path"] for e in committed["entries"])       # no OS-specific paths
    by_path = {e["weights_path"]: e for e in committed["entries"]}
    for name in LIVE_ARTIFACTS:
        path = ROOT / "results" / "models" / name
        if not path.exists():
            continue
        e = by_path.get(f"results/models/{name}")
        assert e is not None, f"{name} makes live numbers but is not in the registry"
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        assert e["sha256"] == hashlib.sha256(canonical).hexdigest() and e["size_bytes"] == len(canonical), name


def test_every_lineage_file_exists_with_the_hash_the_registry_gives(committed):
    n = 0
    for e in committed["entries"]:
        for item in (e.get("lineage") or {}).get("selection_and_evaluation") or []:
            p = ROOT / item["path"]
            assert p.exists(), item["path"]
            assert item["sha256"] == hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), item["path"]
            n += 1
    assert n >= 10


def test_our_shadow_models_name_their_preregistration_and_where_their_forecasts_are_kept(committed):
    from hazardpulse.hurricane import ri_j1

    shadows = [e for e in committed["entries"] if (e.get("lineage") or {}).get("role") == "shadow"]
    j1_label = json.loads(ri_j1.MODEL_PATH.read_text(encoding="utf-8"))["label"]
    assert {e["name"].split(" (")[0] for e in shadows} >= {f"HazardPulse Hurricane RI {v}"
                                                            for v in ("v9.1", "v10.1", "v10.2", "v10.3", j1_label)}
    nhc_entrants = json.loads((ROOT / "results" / "hurricane_prospective" / "v9_shadow.json")
                              .read_text(encoding="utf-8"))["entrants"]
    for e in shadows:
        lin = e["lineage"]
        assert lin["prereg_tag"] and lin["live_records"]["ledger"] == "dist/data/hurricane-ledger.jsonl"
        # each shadow names the prospective test that scores it, and is an entrant that test knows
        if lin["prospective"]["file"] == "results/hurricane_prospective/v9_shadow.json":
            assert lin["prospective"]["entrant"] in nhc_entrants, e["name"]
        else:
            assert lin["prospective"] == {"file": "results/hurricane_prospective/j1_shadow.json", "entrant": j1_label}
            assert lin["live_records"]["key"] == ri_j1.SHADOW_KEY and e["output_schema"]["label"] == j1_label
    try:                                    # tags are present in a full clone; CI's shallow one has none
        tags = set(subprocess.run(["git", "tag", "-l", "prereg-*"], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.split())
    except (OSError, subprocess.CalledProcessError):
        tags = set()
    if tags:
        for e in committed["entries"]:
            tag = (e.get("lineage") or {}).get("prereg_tag")
            assert tag is None or tag in tags, f"{e['name']} names a pre-registration tag that does not exist: {tag}"


def test_a_crlf_checkout_hashes_like_the_stored_bytes(reg_mod, tmp_path):
    lf, crlf = tmp_path / "a.json", tmp_path / "b.json"
    lf.write_bytes(b'{"a":1}\n')
    crlf.write_bytes(b'{"a":1}\r\n')
    assert reg_mod._hash_file(lf) == reg_mod._hash_file(crlf)
    binary_lf, binary_crlf = tmp_path / "a.npz", tmp_path / "b.npz"
    binary_lf.write_bytes(b"x\n")
    binary_crlf.write_bytes(b"x\r\n")
    assert reg_mod._hash_file(binary_lf) != reg_mod._hash_file(binary_crlf)      # binaries are not folded
