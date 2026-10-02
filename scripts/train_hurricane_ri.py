#!/usr/bin/env python3
"""Train a hurricane RI model recipe once and write its pinned serving artifact.

The daily scorer (scripts/fetch_and_score.py) never trains: it loads the artifact named by
its MODEL_ARTIFACT, which this script produces. Each artifact records the SHA-256 of every
data file it was fitted on and the exact config, and the scorer refuses it if either
changed -- so re-run this after rebuilding a training set or changing the config, then
commit the artifact (results/models/ is gitignored):

    python scripts/train_hurricane_ri.py --recipe hurricane_ri_v8_2
    git add -f results/models/hurricane_ri_v8_2.json

    python scripts/train_hurricane_ri.py --recipe hurricane_ri_v8_2 --verify   # refit, require bit-identity

Recipes (see hazardpulse.hurricane.ri_model.RECIPES): hurricane_ri_v8_1 (the original pin,
comparison only), hurricane_ri_v8_1_1 and hurricane_ri_v8_2 (held-out converged calibration).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import ri_model  # noqa: E402


def _first_difference(a, b, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path or '<root>'}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path or '<root>'}: keys differ ({sorted(set(a) ^ set(b))})"
        for key in a:
            diff = _first_difference(a[key], b[key], f"{path}.{key}" if path else key)
            if diff:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _first_difference(x, y, f"{path}[{i}]")
            if diff:
                return diff
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", choices=sorted(ri_model.RECIPES), default="hurricane_ri_v8_2")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="refit and compare with the existing artifact instead of writing it")
    args = parser.parse_args(argv)
    out = args.out or ri_model.ARTIFACTS[args.recipe]

    t0 = time.perf_counter()
    model = ri_model.fit_recipe(args.recipe, log=lambda m: print(m, flush=True))
    train_s = time.perf_counter() - t0
    print(f"  {args.recipe} trained in {train_s:.1f}s", flush=True)
    ri_model.attach_provenance(model, data=ri_model.recipe_data_binding(args.recipe), train_seconds=train_s)
    ri_model.validate_structure(model)

    if args.verify:
        committed = json.loads(out.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(model, allow_nan=False))  # what the scorer actually serves
        diff = _first_difference(ri_model.model_parameters(committed), ri_model.model_parameters(fresh))
        prov_c, prov_f = committed.get("provenance", {}), fresh["provenance"]
        for key in ("data", "config", "trainer_fingerprint"):
            if prov_c.get(key) != prov_f.get(key):
                diff = diff or f"provenance.{key}: {prov_c.get(key)!r} != {prov_f.get(key)!r}"
        if diff:
            print(f"VERIFY FAILED: refit differs from {out}: {diff}")
            return 1
        print(f"VERIFY OK: refit is bit-identical to {out} "
              f"({len(fresh['gbt_d3']['trees'])}+{len(fresh['gbt_d4']['trees'])} trees, "
              f"{len(fresh['bagged'])} bags, calibration {fresh['calibration']['method']})")
        return 0

    path = ri_model.save_model(model, out)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print("Commit it with: git add -f " + str(path.relative_to(ROOT)).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
