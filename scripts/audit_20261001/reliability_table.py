"""Fine reliability tables for the served tornado models' 2025 final runs (descriptive; read-only).

    PYTHONPATH=src python scripts/audit_20261001/reliability_table.py [run ...]   # default: v3_plus_W v3_primary

The final runs' own reliability (tornado_lab metrics) uses ten equal-width bins, so 99.8% of the
2025 storm observations fall in its first bin (0-10%) -- too coarse to say how storms scored at
2% turned out. This writes, per run, log-spaced bins over the SAME saved 2025 predictions and
labels: n, tornadic count, mean forecast, observed rate and its Jeffreys 95% interval. No model,
threshold or choice depends on it; the live page uses it to tell a reader how 2025 storm
observations that the model scored like a given storm turned out (served_evidence.reliability_bin).
Each table records the sha256 of the predictions it was computed from.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

spec = importlib.util.spec_from_file_location("tornado_lab", Path(__file__).with_name("tornado_lab.py"))
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

EDGES = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def table(p: np.ndarray, y: np.ndarray, edges=EDGES) -> list[dict]:
    """Bins [lo, hi) (the last closed). Empty bins are kept with n = 0 so a reader sees the gap."""
    if p.shape != y.shape:
        raise ValueError(f"{p.shape} predictions for {y.shape} labels")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("predictions must be finite probabilities")
    idx = np.clip(np.searchsorted(np.asarray(edges), p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        pos = int(y[m].sum())
        row = {"lo": edges[b], "hi": edges[b + 1], "n": n, "pos": pos,
               "mean_forecast": float(p[m].mean()) if n else None,
               "observed": pos / n if n else None, "observed_ci": None}
        if n:
            lo = 0.0 if pos == 0 else float(beta.ppf(0.025, pos + 0.5, n - pos + 0.5))
            hi = 1.0 if pos == n else float(beta.ppf(0.975, pos + 0.5, n - pos + 0.5))
            row["observed_ci"] = [lo, hi]
        out.append(row)
    return out


def run(name: str) -> Path:
    final = json.loads((lab.OUT / f"final_{name}.json").read_text(encoding="utf-8"))
    label = final["exp"]["label"]
    pred_path = lab.LAB / "preds" / f"{name}_final.npy"
    p = np.load(pred_path).astype(np.float64)
    _, Y, meta = lab.load("final")
    y = lab.get_y(Y, meta, label).astype(np.float64)
    rows = table(p, y)
    if sum(r["n"] for r in rows) != p.size or sum(r["pos"] for r in rows) != int(y.sum()):
        raise SystemExit(f"{name}: bins do not partition the {p.size} predictions")
    f25 = final["final_2025"]
    if int(y.sum()) != int(f25["pos"]) or p.size != int(f25["n"]):
        raise SystemExit(f"{name}: {p.size} rows / {int(y.sum())} tornadic, the final run says "
                         f"{f25['n']} / {f25['pos']}: not the same test")
    out = lab.OUT / f"reliability_{name}_final.json"
    out.write_text(json.dumps({
        "run": name, "split": "final", "label": label, "n": int(p.size), "pos": int(y.sum()),
        "interval": "Jeffreys 95% (Beta(pos + 1/2, n - pos + 1/2) quantiles)",
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "edges": list(EDGES), "bins": rows,
    }, indent=1), encoding="utf-8")
    print(f"{name}: " + " | ".join(f"{r['lo']:g}-{r['hi']:g}: n={r['n']} obs="
                                   f"{'--' if r['observed'] is None else format(r['observed'], '.4f')}"
                                   for r in rows), flush=True)
    return out


def main() -> int:
    for name in sys.argv[1:] or ["v3_plus_W", "v3_primary"]:
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
