"""A LightGBM binary model as a self-contained JSON payload, scored in pure NumPy.

The served tornado path must not depend on the LightGBM runtime: the live scorer walks trees
from a payload checked into the repository, exactly as the v2 NumPy GBT was served. This
module exports a trained ``lightgbm.Booster`` to such a payload and scores it.

Split semantics are LightGBM's own (``Tree::NumericalDecision``):

* a NaN input is replaced by 0.0 unless the node's missing type is ``NaN``;
* if the missing type is ``Zero`` and |x| <= 1e-35, or ``NaN`` and x is NaN, the row follows
  ``default_left``;
* otherwise ``x <= threshold`` goes left.

Inputs are compared in float64, as LightGBM promotes them. ``predict_raw`` must equal
``booster.predict(X, raw_score=True)`` -- the parity test in tests/test_lgbm_payload.py
checks it to 1e-12 on rows built to hit every branch type.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "hazardpulse_lgbm_payload/1"
_ZERO = 1e-35
_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _flatten(tree: dict) -> dict[str, list]:
    """One tree as parallel node arrays; a leaf is a node whose ``feature`` is -1."""
    feat, thr, dleft, miss, left, right, value, node_value = [], [], [], [], [], [], [], []

    def add(node: dict) -> int:
        i = len(feat)
        feat.append(-1)
        thr.append(0.0)
        dleft.append(False)
        miss.append(0)
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        node_value.append(0.0)
        if "leaf_value" in node and "split_feature" not in node:
            value[i] = node_value[i] = float(node["leaf_value"])
            return i
        node_value[i] = float(node["internal_value"])
        if node.get("decision_type", "<=") != "<=":
            raise ValueError(f"unsupported decision type {node.get('decision_type')!r} (categorical splits)")
        feat[i] = int(node["split_feature"])
        thr[i] = float(node["threshold"])
        dleft[i] = bool(node["default_left"])
        miss[i] = _MISSING[str(node.get("missing_type", "None"))]
        left[i] = add(node["left_child"])
        right[i] = add(node["right_child"])
        return i

    add(tree["tree_structure"])
    return {"feature": feat, "threshold": thr, "default_left": dleft, "missing_type": miss,
            "left": left, "right": right, "value": value, "node_value": node_value}


def export_booster(booster, feature_names: list[str], *, calibration: dict, provenance: dict) -> dict:
    """``booster`` -> payload. ``feature_names`` are the model's input columns, in order."""
    dump = booster.dump_model()
    if dump.get("objective", "").split()[0] != "binary":
        raise ValueError(f"expected a binary objective, got {dump.get('objective')!r}")
    if len(feature_names) != dump["max_feature_idx"] + 1:
        raise ValueError(f"{len(feature_names)} names for {dump['max_feature_idx'] + 1} model inputs")
    n_used = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else booster.num_trees()
    trees = [_flatten(t) for t in dump["tree_info"][:n_used]]
    return {"schema": SCHEMA, "feature_names": list(feature_names), "n_trees": len(trees),
            "trees": trees, "calibration": calibration, "provenance": provenance}


def _tree_predict(tree: dict, X: np.ndarray, contrib: np.ndarray | None = None) -> np.ndarray:
    """Leaf value per row. With ``contrib`` (n_rows x n_features), also adds each split's change in
    node value (child minus parent) to the split feature's column -- Saabas path attribution."""
    feat = np.asarray(tree["feature"], np.int64)
    thr = np.asarray(tree["threshold"], np.float64)
    dleft = np.asarray(tree["default_left"], bool)
    miss = np.asarray(tree["missing_type"], np.int64)
    left = np.asarray(tree["left"], np.int64)
    right = np.asarray(tree["right"], np.int64)
    value = np.asarray(tree["value"], np.float64)
    nval = np.asarray(tree["node_value"], np.float64) if contrib is not None else None
    node = np.zeros(X.shape[0], np.int64)
    active = feat[node] >= 0
    while active.any():
        idx = np.flatnonzero(active)
        nd = node[idx]
        x = X[idx, feat[nd]]
        mt = miss[nd]
        isnan = np.isnan(x)
        x = np.where(isnan & (mt != 2), 0.0, x)
        go_default = ((mt == 1) & (np.abs(x) <= _ZERO)) | ((mt == 2) & isnan)
        go_left = np.where(go_default, dleft[nd], x <= thr[nd])
        child = np.where(go_left, left[nd], right[nd])
        if contrib is not None:
            np.add.at(contrib, (idx, feat[nd]), nval[child] - nval[nd])
        node[idx] = child
        active[idx] = feat[node[idx]] >= 0
    return value[node]


def contributions(payload: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """``(bias, contrib)``: per row, the raw score = bias + contrib.sum(axis=1) EXACTLY, with
    contrib[:, j] the part of the score the path attribution assigns to input j. bias is the sum
    of the trees' root values (the score before any split)."""
    X = np.asarray(X, np.float64)
    if any("node_value" not in t for t in payload["trees"]):
        raise ValueError("payload has no node values (exported before contributions existed)")
    contrib = np.zeros_like(X)
    bias = sum(float(t["node_value"][0]) for t in payload["trees"])
    for tree in payload["trees"]:
        _tree_predict(tree, X, contrib)
    return np.full(X.shape[0], bias), contrib


def predict_interval(payload: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """Venn-Abers band [p0, p1] for each row (payload["interval"] = VennAbersCalibrator.to_dict()
    fitted on the model's leave-one-year-out raw scores); NaN bands if the payload has none."""
    if not payload.get("interval"):
        n = np.asarray(X).shape[0]
        return np.full(n, np.nan), np.full(n, np.nan)
    from hazardpulse.trust.venn_abers import VennAbersCalibrator
    va = VennAbersCalibrator.from_dict(payload["interval"])
    _, p0, p1 = va.predict(predict_raw(payload, X))
    return np.asarray(p0, np.float64), np.asarray(p1, np.float64)


def predict_raw(payload: dict, X) -> np.ndarray:
    """Raw additive score (log-odds) for each row of ``X`` (columns = payload feature_names)."""
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or X.shape[1] != len(payload["feature_names"]):
        raise ValueError(f"X has shape {X.shape}; the payload expects {len(payload['feature_names'])} columns")
    out = np.zeros(X.shape[0], np.float64)
    for tree in payload["trees"]:
        out += _tree_predict(tree, X)
    return out


def predict_proba(payload: dict, X) -> np.ndarray:
    """Calibrated probability: Platt on the raw score (``calibration`` = {"a", "b"})."""
    cal = payload["calibration"]
    z = float(cal["a"]) * predict_raw(payload, X) + float(cal["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def model_version(payload: dict, prefix: str = "tornado_v3") -> str:
    """Identity bound to the content (canonical JSON, so line endings and key order cannot split it)."""
    return f"{prefix}-{hashlib.sha256(canonical_bytes(payload)).hexdigest()[:12]}"


def save(payload: dict, path: str | Path) -> str:
    for t in payload["trees"]:
        if any(not math.isfinite(v) for v in t["threshold"] + t["value"]):
            raise ValueError("non-finite threshold or leaf value")
    Path(path).write_bytes(canonical_bytes(payload))
    return model_version(payload)


def load(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {payload.get('schema')!r}, expected {SCHEMA!r}")
    return payload
