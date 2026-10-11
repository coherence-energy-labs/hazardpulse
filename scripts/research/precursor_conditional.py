#!/usr/bin/env python3
"""Does a confirmed precursor signal carry information beyond what the served model already reads? (diagnostic)

    python scripts/research/precursor_atlas_2.py --past-only --dump
    python scripts/research/precursor_conditional.py

Conditional (matched-set) logistic regression on run 2b's raw values: each mainshock against its own 20 past-only
controls, so place drops out. Model 1 holds the served model's analogues only -- counts (F1, F3) and the largest recent
magnitude (M1). Model 2 adds A1 (moment over the decade) and Z1 (rate ratio). Model 3 adds B2 (b-value change) on the
sets where it exists. A coefficient that stays clear of 0 next to the analogues is information the served model's
inputs do not hold in this form. Diagnostic only: whether it improves the forecast is decided by an amendment of the
earthquake program on its own data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "precursor_atlas" / "atlas_2b_raw.json"
OUT = ROOT / "results" / "precursor_atlas" / "conditional_2b.json"


def transform(d: dict) -> dict:
    return {"F1": math.log1p(d["F1"]), "F3": math.log1p(d["F3"]), "M1": d["M1"],
            "A1": math.log10(d["A1"]) if d["A1"] and math.isfinite(d["A1"]) else math.nan,
            "Z1": math.log10(d["Z1"]) if d["Z1"] and math.isfinite(d["Z1"]) else math.nan,
            "B2": d["B2"]}


def sets(rows, names):
    out = []
    for r in rows:
        ev = transform(r["raw_event"])
        cs = [transform(c) for c in r["raw_controls"]]
        xs = [ev] + cs
        keep = [x for x in xs if all(math.isfinite(x[n]) for n in names)]
        if not all(math.isfinite(ev[n]) for n in names) or len(keep) < 6:
            continue
        out.append(np.array([[x[n] for n in names] for x in keep]))     # row 0 is the event
    return out


def fit(groups, names):
    allx = np.vstack(groups)
    mu, sd = allx.mean(0), allx.std(0) + 1e-12
    gs = [(g - mu) / sd for g in groups]

    def nll(beta):
        v, grad = 0.0, np.zeros_like(beta)
        for g in gs:
            e = g @ beta
            m = e.max()
            w = np.exp(e - m)
            s = w.sum()
            v -= e[0] - (m + math.log(s))
            grad -= g[0] - (w[:, None] * g).sum(0) / s
        return v, grad
    r = optimize.minimize(nll, np.zeros(len(names)), jac=True, method="BFGS")
    H = np.linalg.inv(r.hess_inv) if hasattr(r, "hess_inv") else None
    se = np.sqrt(np.diag(r.hess_inv))
    return {"n_sets": len(gs), "names": names, "beta_per_sd": r.x.tolist(), "se": se.tolist(),
            "z": (r.x / se).tolist(), "nll": float(r.fun), "converged": bool(r.success)}


def main() -> int:
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    rows = rows.get("rows") or rows.get("events") or rows.get("records")
    if rows is None or not rows or "raw_event" not in rows[0]:
        raise SystemExit("run precursor_atlas_2.py --past-only --dump first (raw values are missing)")
    res = {}
    for key, names in (("1_served_analogues", ["F1", "F3", "M1"]), ("2_plus_A1_Z1", ["F1", "F3", "M1", "A1", "Z1"]),
                       ("3_plus_B2", ["F1", "F3", "M1", "A1", "Z1", "B2"])):
        f = fit(sets(rows, names), names)
        res[key] = f
        print(key, f"n_sets {f['n_sets']}", ", ".join(f"{n} {b:+.2f} (z {z:+.1f})" for n, b, z in zip(names, f["beta_per_sd"], f["z"])))
    # likelihood-ratio for A1+Z1 on the same sets as model 2
    names2, names1 = ["F1", "F3", "M1", "A1", "Z1"], ["F1", "F3", "M1"]
    g2 = sets(rows, names2)
    f2 = fit(g2, names2)
    f1 = fit([g[:, :3] for g in g2], names1)
    lr = 2 * (f1["nll"] - f2["nll"])
    res["lr_A1_Z1_given_analogues"] = {"lr": lr, "df": 2, "p": math.exp(-lr / 2)}
    print(f"likelihood ratio, A1 + Z1 beyond F1, F3, M1 (same {f2['n_sets']} sets): {lr:.1f} on 2 df, p {math.exp(-lr / 2):.1e}")
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
