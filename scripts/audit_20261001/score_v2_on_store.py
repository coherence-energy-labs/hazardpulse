"""The served v2 model's probability for every row of the v3 feature store (the bar to beat).

    PYTHONPATH=src python scripts/audit_20261001/score_v2_on_store.py 2023-01-01 2024-12-31

Rebuilds v2's own 41-feature vector (definitive_model blocks P/E/H/C, the 80 km g3 analysis
chosen by the causal policy) for each store row, IN THE STORE'S ROW ORDER (asserted on the
storm id and time of every row), and scores it with results/models/tornado_gbt_v1.json
through the payload's float32 normalisation and Platt calibration. A row with no causal
analysis gets NaN (v2 cannot score it; it is never zero-filled). Writes
STORE/_v2/<day>.npy (float32) next to the day file, with the payload's sha256 in
STORE/_v2/payload.json; a day is redone when that hash changes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from hazardpulse.data import hrrr as H
from hazardpulse.data.probsevere import load_cached_probsevere
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado.coherence_engine import compute_coherence_fields, compute_derived_hrrr

ROOT = H.PROJECT_ROOT
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", str(ROOT / ".cache" / "probsevere_v2")))
STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", str(ROOT / ".cache" / "feature_store_v3")))
PAYLOAD = Path(__file__).resolve().parents[2] / "results" / "models" / "tornado_gbt_v1.json"
OUT = STORE / "_v2"


def batch_scores(payload: dict, X: np.ndarray) -> np.ndarray:
    """predict_proba_from_payload for a matrix: same float32 normalisation, same tree walk."""
    f32 = np.float32
    mu = np.asarray(payload["normalization"]["means"], f32)
    sd = np.asarray(payload["normalization"]["stds"], f32)
    Z = ((X.astype(f32) - mu) / sd).astype(np.float64)
    F = np.full(len(Z), float(payload["init_pred"]))
    lr = float(payload["learning_rate"])
    for tree in payload["trees"]:
        vals = np.empty(len(Z))
        # descend with the rows grouped by node (a per-row walk is too slow for a year of storms)
        stack = [(tree, np.arange(len(Z)))]
        while stack:
            node, idx = stack.pop()
            if node.get("leaf", False):
                vals[idx] = float(node["val"])
                continue
            go_left = Z[idx, node["feat"]] <= node["thresh"]
            stack.append((node["left"], idx[go_left]))
            stack.append((node["right"], idx[~go_left]))
        F += lr * vals
    return dm.apply_calibration(F, payload.get("calibration"))


def score_day(d: str, payload: dict) -> str:
    store_file = STORE / f"{d}.npz"
    if not store_file.exists():
        return f"{d} no-store"
    out = OUT / f"{d}.npy"
    if out.exists():
        return f"{d} cached"
    steps = load_cached_probsevere(d, cache_dir=PS_V2)
    month = int(d[4:6])
    an80 = {}
    for h in dm.HRRR_ANALYSIS_HOURS:
        g = H.load_cached_hrrr(d, hour=h)
        if g is not None:
            an80[h] = (g, compute_derived_hrrr(g), compute_coherence_fields(g, month=month))
    step_min = dm.probsevere_step_minutes(steps)
    idx = dm.index_storms_by_id(steps)
    rows, feats, sids, ts = [], [], [], []
    for si, st in enumerate(steps):
        t = dm.parse_probsevere_valid_time(st.get("valid_time", ""))
        if t is None:
            continue
        h80 = dm.select_analysis_hour(t, sorted(an80))
        a = an80.get(h80) if h80 is not None else None
        for storm in st.get("storms", []):
            sids.append(str(storm.get("id", "")))
            ts.append(int(t.timestamp()))
            if a is None:
                feats.append(None)
                continue
            hist = dm.build_storm_history(steps, storm.get("id"), si, id_index=idx)
            feats.append(np.concatenate([
                dm.extract_block_p(storm), dm.extract_block_e(storm, hist, step_min),
                dm.extract_block_h(storm, a[0], a[1]), dm.extract_block_c(storm, a[2], a[0])]))
    with np.load(store_file) as z:
        if not (np.array_equal(z["sid"], np.asarray(sids)) and np.array_equal(z["t"], np.asarray(ts, np.int64))):
            raise RuntimeError(f"{d}: row order differs from the feature store")
    p = np.full(len(feats), np.nan)
    have = [i for i, f in enumerate(feats) if f is not None]
    if have:
        X = np.stack([feats[i] for i in have]).astype(np.float32)
        p[have] = batch_scores(payload, X)
        # the batch path must equal the ONLY live scoring path, on a sample of rows
        names = payload["feature_names"]
        for i in have[:: max(1, len(have) // 25)]:
            live, _ = dm.predict_proba_from_payload(payload, dict(zip(names, map(float, feats[i]))))
            if abs(live - p[i]) > 1e-12:
                raise RuntimeError(f"{d}: batch {p[i]} != live {live}")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{d}.tmp.npy"
    np.save(tmp, p.astype(np.float32))
    os.replace(tmp, out)
    return f"{d} n={len(p)} scored={len(have)}"


def main() -> int:
    start, end = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()
    if len(payload["feature_names"]) != dm.N_FEAT_FULL or payload["feature_names"] != dm.ALL_FEATURE_NAMES_FULL:
        raise SystemExit("payload feature order differs from definitive_model.ALL_FEATURE_NAMES_FULL")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = OUT / "payload.json"
    if meta.exists() and json.loads(meta.read_text())["sha256"] != sha:
        for f in OUT.glob("*.npy"):
            f.unlink()
    meta.write_text(json.dumps({"sha256": sha, "path": str(PAYLOAD), "model_version": dm.model_version_of_payload(PAYLOAD)}),
                    encoding="utf-8")
    d = start
    while d <= end:
        print(score_day(d.strftime("%Y%m%d"), payload), flush=True)
        d += dt.timedelta(days=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
