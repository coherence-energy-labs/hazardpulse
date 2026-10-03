"""Generate the declared LightGBM random search (protocol search item 3) -- 40 configurations from a
fixed seed, written before any experiment so the space cannot be tuned to a result.

    python scripts/audit_20261001/experiments/make_search.py   # -> 03_lgbm_search.json
"""
import json
from pathlib import Path

import numpy as np

SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63, 127, 255],
    "min_child_samples": [20, 50, 100, 200, 500],
    "feature_fraction": [0.4, 0.55, 0.7, 0.85, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0],
    "min_split_gain": [0.0, 0.01, 0.1],
    "max_depth": [-1, 6, 10],
}
SEED = 20261002
N = 40

rng = np.random.RandomState(SEED)
exps = []
for i in range(N):
    params = {k: v[rng.randint(len(v))] for k, v in SPACE.items()}
    params = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
              for k, v in params.items()}
    exps.append({"name": f"s_lgbm_{i:02d}", "model": "lgbm", "blocks_from": "best", "params": params,
                 "calibration": "platt", "neg_per_pos": 30, "save_preds": True, "eval_dev": False})
out = Path(__file__).with_name("03_lgbm_search.json")
out.write_text(json.dumps(exps, indent=1), encoding="utf-8")
print(out, len(exps))
