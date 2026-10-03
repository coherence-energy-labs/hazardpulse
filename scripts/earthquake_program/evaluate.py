#!/usr/bin/env python3
"""Program step 4: score every candidate on one split (docs/EARTHQUAKE_FORECAST_PROGRAM.md
sections 4 and 7).

    python scripts/earthquake_program/evaluate.py --split choose   # applies the selection rule
    python scripts/earthquake_program/evaluate.py --split dev      # confirmation, no re-selection
    python scripts/earthquake_program/evaluate.py --split final    # read ONCE, after choose.json

FINAL refuses to run before ``choose.json`` holds a decision, and refuses to overwrite an
existing ``final.json``: the held-out years are read once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import features_c as F  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

CHAIN = ["A", "B", "C0", "C1"]
ALL = CHAIN + ["D"]


def declustered_m6(events: of.EventSet) -> of.EventSet:
    """Gardner-Knopoff (1974, both directions) mainshocks among M>=6: an M6+ event is dropped
    if it lies in the window of a larger M6+ event that is not itself dropped."""
    sel = events.select(events.mag >= 6.0)
    n = len(sel)
    order = np.argsort(-sel.mag, kind="stable")
    d_km, t_days = of.gk_window(sel.mag)
    u = of._unit(sel.lat, sel.lon)
    dropped = np.zeros(n, dtype=bool)
    for i in order:
        if dropped[i]:
            continue
        near_t = np.abs(sel.t - sel.t[i]) <= t_days[i] * C.SEC_DAY
        smaller = sel.mag < sel.mag[i]
        cand = np.nonzero(near_t & smaller & ~dropped)[0]
        if cand.size:
            dist = of.EARTH_RADIUS_KM * np.arccos(np.clip(u[cand] @ u[i], -1, 1))
            dropped[cand[dist <= d_km[i]]] = True
    return sel.select(~dropped)


def active_masks(issue: np.ndarray) -> np.ndarray:
    r25 = C.load_raw("program_catalog_m25.npz")
    c25 = F.CellCatalog(r25["t"], r25["lat"], r25["lon"], r25["mag"])
    return np.stack([F.active_cells(t, c25) for t in issue])


def score_split(split: str, cands: list[str], block: str = "month", target_events: of.EventSet | None = None) -> dict:
    fit = json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))
    p0 = float(fit["fit"]["p0"])
    issue = C.split_issue_times(split)
    events = C.load_events(4.5) if target_events is None else target_events
    Y = C.binary_targets(events, issue)
    counts = of.target_matrix(events, issue).toarray()
    act = active_masks(issue)
    sc_all = C.SplitScorer(issue, Y, p0=p0, block=block)
    sc_act = C.SplitScorer(issue, Y, p0=p0, mask=act, block=block)
    out = {"split": split, "block": block, "p0_reference": p0, "n_issue_times": int(issue.size),
           "first_issue": str(np.datetime64(int(issue[0]), "s")),
           "last_issue": str(np.datetime64(int(issue[-1]), "s")),
           "n_cell_times": int(Y.size), "n_positive_cell_windows": int(Y.sum()),
           "n_target_events": int(counts.sum()),
           "positives_in_active_cells": int((Y & act).sum()), "candidates": {}, "paired": {}}
    stats_all, stats_act = {}, {}
    for cand in cands:
        P = C.load_pred(cand, split)
        if P.shape != Y.shape:
            raise RuntimeError(f"{cand}/{split}: forecast shape {P.shape} != outcomes {Y.shape}")
        s_all = sc_all.stats(P)
        s_act = sc_act.stats(P)
        stats_all[cand], stats_act[cand] = s_all, s_act
        res = sc_all.summary(s_all)
        res["auc_active_cells"] = sc_act.summary(s_act)["auc"]
        res["positives_forecast_zero"] = int((Y & (P <= 0.0)).sum())
        res["poisson_ig_per_event_site_reference"] = C.poisson_ig_site_reference(P, counts)
        res["reliability"] = C.reliability(P, Y)
        out["candidates"][cand] = res
        print(f"  {cand:3s} IG {res['ig_per_target']['value']:+.4f} {res['ig_per_target']['ci95']}  "
              f"AUC {res['auc']['value']:.4f} {res['auc']['ci95']}  BSS {res['bss']['value']:+.4f}  "
              f"AUC(active) {res['auc_active_cells']['value']:.4f}  sum p/sum y {res['calib_ratio']['value']:.3f}")
    for i, x in enumerate(cands):
        for y in cands[:i]:
            out["paired"][f"{x}-{y}"] = sc_all.paired(stats_all[x], stats_all[y])
    return out


def apply_rule(res: dict) -> dict:
    current = "A"
    steps = []
    for x in CHAIN[1:] + ["D"]:
        key = f"{x}-{current}"
        pr = res["paired"][key] if key in res["paired"] else None
        if pr is None:
            raise RuntimeError(f"missing paired comparison {key}")
        ig, auc = pr["ig_per_target"], pr["auc"]
        if ig["ci95"][0] > 0:
            verdict, why = True, "IG interval above 0"
        elif ig["ci95"][0] <= 0 <= ig["ci95"][1] and auc["ci95"][0] > 0 and ig["diff"] >= 0:
            verdict, why = True, "IG interval contains 0; AUC interval above 0 and IG point >= 0"
        else:
            verdict, why = False, "not shown better"
        steps.append({"candidate": x, "against": current, "ig_diff": ig["diff"], "ig_ci95": ig["ci95"],
                      "auc_diff": auc["diff"], "auc_ci95": auc["ci95"], "replaces": verdict, "reason": why})
        print(f"  {x} vs {current}: dIG {ig['diff']:+.4f} {ig['ci95']}  dAUC {auc['diff']:+.4f} {auc['ci95']} -> "
              f"{'REPLACES' if verdict else 'stays ' + current} ({why})")
        if verdict:
            current = x
    return {"chosen": current, "steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["choose", "dev", "final"])
    args = ap.parse_args(argv)
    out_path = C.RESULTS / f"{args.split}.json"
    if args.split == "final":
        choose = C.RESULTS / "choose.json"
        if not choose.exists() or "decision" not in json.loads(choose.read_text(encoding="utf-8")):
            raise SystemExit("FINAL refused: no recorded CHOOSE decision (run --split choose first)")
        if out_path.exists():
            raise SystemExit(f"FINAL refused: {out_path} exists -- the final split is read once")
    print(f"scoring {args.split}")
    res = score_split(args.split, ALL)
    if args.split == "choose":
        print("selection rule:")
        res["decision"] = apply_rule(res)
    else:
        decision = json.loads((C.RESULTS / "choose.json").read_text(encoding="utf-8"))["decision"]
        res["chosen_on_choose"] = decision["chosen"]
    if args.split == "dev":
        chosen = res["chosen_on_choose"]
        idx = CHAIN.index(chosen) if chosen in CHAIN else len(CHAIN)
        if idx > 0:
            simpler = CHAIN[idx - 1]
            pr = res["paired"].get(f"{chosen}-{simpler}")
            res["dev_warning"] = bool(pr and pr["ig_per_target"]["ci95"][1] < 0)
            print(f"  DEV check {chosen} vs {simpler}: dIG {pr['ig_per_target']['diff']:+.4f} "
                  f"{pr['ig_per_target']['ci95']} -> warning={res['dev_warning']}")
    if args.split == "final":
        print("sensitivity: quarter blocks")
        res["quarter_blocks"] = score_split("final", ALL, block="quarter")
        print("secondary: declustered M6+ mainshock targets")
        res["declustered_mainshock_targets"] = score_split("final", ALL, target_events=declustered_m6(C.load_events(4.5)))
    C.write_json(out_path.name, res)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
