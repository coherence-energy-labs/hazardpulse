"""Hurricane RI v10.1 against EVERY public RI guidance on the same development cases.

    PYTHONPATH=src python scripts/hurricane_ri_v10_vs_all.py

The amendment-2 selection compared v10 with DTOPS only (the aid a pre-registered 2020-2024 comparison
found best). "Best of NOAA on other seasons" is not a measurement on these cases, so this reports the
served v10.1 (V2 + the early-guidance gate, forward chaining 2022-2025, each season predicted from
earlier ones) against each of NOAA's five published RI aids -- SHIPS-RII (RIOD), the RI logistic
(RIOL), the RI Bayesian (RIOB), the RI consensus (RIOC) and DTOPS (DTOP) -- on the cases where that
aid was issued, paired by storm.

This is a development-period comparison (descriptive, not a pre-registered claim); the claim for v10.1
is the prospective test in amendment 2. Controls: V2's pooled 30-kt log loss must equal the frozen
selection's, and the gated forecast at a gate-closed case must be DTOPS's own value, or the run stops.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import hurricane_ri_v10 as v10  # noqa: E402
import hurricane_ri_v9 as v9  # noqa: E402
from hazardpulse.hurricane import ri_v10  # noqa: E402

OUT = ROOT / "results" / "calibration" / "hurricane_ri_v10_vs_all.json"
AIDS = {"RIOD": "SHIPS-RII", "RIOL": "RI logistic", "RIOB": "RI Bayesian", "RIOC": "RI consensus",
        "DTOP": "DTOPS"}
# deterministic intensity guidance read as an RI call (forecast 24-h rise >= 30 kt); per-model hurricane
# guidance is not kept in the development table, only its mean and its most aggressive member
CALLS = {"ofcl_dv24": "NHC official forecast", "dv24_HCCA": "HCCA", "dv24_IVCN": "IVCN",
         "dv24_DSHP": "SHIPS (DSHP)", "dv24_LGEM": "LGEM", "dv24_NNIC": "NNIC", "dv24_AEMI": "GEFS mean",
         "dv24_regional_mean": "hurricane-model mean", "dv24_regional_max": "most aggressive hurricane model",
         "dv24_global_mean": "global-model mean"}


def aid_at(rows, tech: str, k: int) -> np.ndarray:
    """The aid's own published probability at k kt / 24 h, NaN where it was not issued."""
    return np.array([(r["pcts"].get(f"{tech}_{k}/24") / 100.0) if r["pcts"].get(f"{tech}_{k}/24") is not None
                     else np.nan for r in rows])


def gate_open(rows) -> np.ndarray:
    return np.array([all(math.isfinite(r["f"][f"dv24_{a}"]) for a in ri_v10.GATE_AIDS) for r in rows])


def multi_paired(dv, ours: dict, aid: dict, groups, reps=v9.REPS, seed=v9.SEED) -> dict:
    """Storm-bootstrap 95% interval of (ours - aid) in the per-case Brier sum over v10.MULTI."""
    b = sum((ours[k] - (dv >= k)) ** 2 for k in v10.MULTI)
    a = sum((aid[k] - (dv >= k)) ** 2 for k in v10.MULTI)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d = []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        d.append(b[idx].mean() - a[idx].mean())
    return {"ours": float(b.mean()), "aid": float(a.mean()), "d": float(b.mean() - a.mean()),
            "d_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def pod_at_matched_pofd(y, p, call) -> tuple[float, float, float]:
    """(aid POD, our POD at no more false alarms than the aid's, aid POFD) -- the fair test of a
    probability against a yes/no call: pick our threshold so that we raise at most as many false
    alarms as the call did, then count the RI events each catches."""
    neg, pos = y == 0, y == 1
    pofd, pod = float(call[neg].mean()), float(call[pos].mean())
    fa = int(math.floor(pofd * neg.sum() + 1e-9))
    s = np.sort(p[neg])[::-1]
    t = s[fa] if fa < len(s) else -np.inf          # call yes when p > t: at most fa negatives exceed it
    return pod, float((p[pos] > t).mean()), pofd


def matched_paired(y, p, call, groups, reps=v9.REPS, seed=v9.SEED) -> dict:
    pod_a, pod_o, pofd = pod_at_matched_pofd(y, p, call)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    d = []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        a, o, _ = pod_at_matched_pofd(y[idx], p[idx], call[idx])
        d.append(o - a)
    return {"aid_pod": pod_a, "aid_pofd": pofd, "ours_pod": pod_o, "d_pod": pod_o - pod_a,
            "d_pod_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def main() -> int:
    rows = v10.load(v10.DEV10)
    sel = json.loads(v10.SEL10.read_text(encoding="utf-8"))
    if sel["carried"] != "V2":
        raise SystemExit(f"selection carried {sel['carried']}, this report is written for V2")
    preds, test_rows = {k: [] for k in v10.MULTI}, []
    for season in v9.FOLDS:
        train = [r for r in rows if r["season"] < season]
        test = [r for r in rows if r["season"] == season]
        out, _ = v10.fit_predict("V2", train, test, ks_out=v10.MULTI)
        for k in v10.MULTI:
            preds[k].append(out[k])
        test_rows.extend(test)
        v9.log(f"fold {season} done")
    P = {k: np.concatenate(v) for k, v in preds.items()}
    y = np.array([r["y"] for r in test_rows])
    dv = np.array([r["dv"] for r in test_rows])
    groups = np.array([r["sid"] for r in test_rows])
    ll_v2 = v9.summary(y, P[30], False)["log_loss"]
    if abs(ll_v2 - sel["pooled"]["V2"]["log_loss"]) > 1e-9:
        raise SystemExit(f"control failed: V2 pooled LL {ll_v2} != selection's {sel['pooled']['V2']['log_loss']}")

    # the served v10.1: V2 where the early guidance is present, else DTOPS's own value (SHIPS-RII if none)
    gate = gate_open(test_rows)
    served = {k: np.where(gate, P[k], v10.dtops_at(test_rows, k)) for k in v10.MULTI}
    closed = np.flatnonzero(~gate)
    if len(closed) and not np.allclose(served[30][closed], v10.dtops_at(test_rows, 30)[closed], equal_nan=True):
        raise SystemExit("control failed: a gate-closed case does not carry DTOPS's own value")
    n_nan = {k: int(np.isnan(served[k]).sum()) for k in v10.MULTI}

    report = {"phase": "vs-all", "program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 2, descriptive)",
              "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "dev_table_sha256": v9.sha(v10.DEV10), "selection_sha256": v9.sha(v10.SEL10),
              "model": "v10.1 = V2 with the early-guidance gate (DSHP, IVCN, NNIC present, else DTOPS)",
              "seasons": list(v9.FOLDS), "n": int(len(y)), "events": int(y.sum()),
              "gate_open": int(gate.sum()), "served_nan_by_k": n_nan,
              "control_V2_log_loss": ll_v2, "aids": {}}
    lines = [f"v10.1 on {len(y)} cases ({int(y.sum())} RI events), gate open on {int(gate.sum())}"]
    for tech, label in AIDS.items():
        a30 = aid_at(test_rows, tech, 30)
        m = np.isfinite(a30) & np.isfinite(served[30])
        res = {"label": label, "n": int(m.sum()), "events": int(y[m].sum()),
               "storms": int(len(np.unique(groups[m])))}
        if m.sum() and y[m].sum():
            res["ours_30"] = v9.summary(y[m], served[30][m], False)
            res["aid_30"] = v9.summary(y[m], a30[m], True)
            res["vs_30"] = v9.paired(y[m], a30[m], True, served[30][m], False, groups[m])
        ak = {k: aid_at(test_rows, tech, k) for k in v10.MULTI}
        mm = np.all([np.isfinite(ak[k]) & np.isfinite(served[k]) for k in v10.MULTI], axis=0)
        res["n_multi"] = int(mm.sum())
        if mm.sum():
            res["vs_multi"] = multi_paired(dv[mm], {k: served[k][mm] for k in v10.MULTI},
                                           {k: ak[k][mm] for k in v10.MULTI}, groups[mm])
        report["aids"][tech] = res
        if "vs_30" in res:
            w, mu = res["vs_30"], res.get("vs_multi")
            line = (f"vs {label:13s} n {res['n']:4d} ev {res['events']:3d} | 30/24 LL {res['ours_30']['log_loss']:.4f}"
                    f" vs {res['aid_30']['log_loss']:.4f} d {w['d_log_loss']:+.4f} [{w['d_log_loss_ci'][0]:+.4f},"
                    f" {w['d_log_loss_ci'][1]:+.4f}] | Brier d {w['d_brier']:+.5f} [{w['d_brier_ci'][0]:+.5f},"
                    f" {w['d_brier_ci'][1]:+.5f}] | AUC {res['ours_30']['auc']:.3f} vs {res['aid_30']['auc']:.3f}")
            if mu:
                line += (f" | 4-thr Brier {mu['ours']:.4f} vs {mu['aid']:.4f} d {mu['d']:+.4f}"
                         f" [{mu['d_ci'][0]:+.4f}, {mu['d_ci'][1]:+.4f}] (n {res['n_multi']})")
            lines.append(line)
        else:
            lines.append(f"vs {label:13s} n {res['n']} -- not comparable")
    report["beats_every_aid_95"] = all(
        r.get("vs_30", {}).get("d_log_loss_ci", [0, 0])[1] < 0 for r in report["aids"].values())
    report["calls"] = {}
    for name, label in CALLS.items():
        v = np.array([r["f"][name] for r in test_rows])
        m = np.isfinite(v) & np.isfinite(served[30])
        res = {"label": label, "n": int(m.sum()), "events": int(y[m].sum()), "calls": int((v[m] >= 30).sum())}
        res.update(matched_paired(y[m], served[30][m], (v[m] >= 30).astype(float), groups[m]))
        report["calls"][name] = res
        lines.append(f"vs {label:34s} n {res['n']:4d} ev {res['events']:3d} RI calls {res['calls']:3d} | POFD "
                     f"{res['aid_pofd']:.4f}: POD {res['aid_pod']:.3f} vs ours {res['ours_pod']:.3f} "
                     f"d {res['d_pod']:+.3f} [{res['d_pod_ci'][0]:+.3f}, {res['d_pod_ci'][1]:+.3f}]")
    report["beats_every_call_95"] = all(r["d_pod_ci"][0] > 0 for r in report["calls"].values())
    for l in lines:
        v9.log(l)
    v9.log(f"30/24 log loss better than every aid at 95%: {report['beats_every_aid_95']}; "
           f"catches more RI than every yes/no call at its own false-alarm rate (95%): {report['beats_every_call_95']}")
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    v9.log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
