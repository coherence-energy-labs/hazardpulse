#!/usr/bin/env python3
"""Earthquake program amendment E3 (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 11): the served S1 with every
non-earthquake removed from its inputs (E3s), against S1 as served, on DEV (decides) and FINAL (further read).

    python scripts/earthquake_program/e3_types.py [--workers N] [--splits dev,final]

Both arms run through the live code path (``forecast_with_stack``: the frozen catalog plus the simulated live
fetch of ``build_artifact``). The control arm must reproduce the evaluated S1 (the stack on C0's stored
forecasts) to the program's live-parity tolerance at every issue time, or the run stops. Writes
``results/earthquake_program/e3_types.json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import build_artifact as ba  # noqa: E402
import common as C  # noqa: E402
import gear1_stack as gs  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

BASE = C.REPO / "results" / "models" / "earthquake_operational_v1.json"
STACK = C.REPO / "results" / "models" / "earthquake_gear1_stack_v1.json"
LIVE_TOL = 1e-9                     # build_stack.LIVE_TOL: the program's live-parity tolerance
MARGIN = -0.005                     # nats per target (section 11)
N_SITES = 8                         # report the cells holding the most removed M5+ events
EQ = "earthquake"

_STATE: dict = {}


def load_typed(name: str) -> dict[str, np.ndarray]:
    """``common.load_raw`` (canonical rounding, total order) with the ComCat ``type`` carried along."""
    z = np.load(C.PROGRAM_CACHE / name)
    t, lat, lon, mag = of._canon(z["t"], z["lat"], z["lon"], z["mag"])
    order = np.lexsort((mag, lon, lat, t))
    return {"t": t[order], "lat": lat[order], "lon": lon[order], "mag": mag[order],
            "depth": z["depth"][order].astype(np.float64), "type": np.asarray(z["type"]).astype(str)[order]}


def typed_frozen_mask(art: of.LoadedArtifact) -> np.ndarray:
    """True for each frozen event whose ComCat type is "earthquake": types from program_catalog.npz, carried
    through EventSet.from_arrays's own rounding, filter and order, and the arrays checked equal to the
    artifact's element by element."""
    raw = load_typed("program_catalog.npz")
    t, lat, lon, mag, typ = raw["t"], raw["lat"], raw["lon"], raw["mag"], raw["type"]
    keep = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(mag) & (mag >= of.INPUT_MIN_MAG)
    keep &= t < art.cutoff
    t, lat, lon, mag, typ = t[keep], lat[keep], lon[keep], mag[keep], typ[keep]
    fz = art.frozen
    for name, mine, theirs in (("t", t, fz.t), ("lat", lat, fz.lat), ("lon", lon, fz.lon), ("mag", mag, fz.mag)):
        if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
            raise SystemExit(f"frozen catalog {name} does not match program_catalog.npz: types cannot be attached")
    return typ == EQ


def _init(splits: list[str]) -> None:
    base = of.load_artifact(BASE)
    stack = of.load_stack(STACK, base)
    mask = typed_frozen_mask(base)
    raw25 = C.load_raw("program_catalog_m25.npz")             # exactly what build_artifact's parity read
    typed25 = load_typed("program_catalog_m25.npz")
    for k in ("t", "lat", "lon", "mag", "depth"):
        if not np.array_equal(typed25[k], raw25[k], equal_nan=True):
            raise SystemExit(f"typed M2.5 catalog {k} differs from common.load_raw's")
    eq25 = typed25["type"] == EQ
    raw25_eq = {k: v[eq25] for k, v in raw25.items()}
    _STATE.update(base=base, stack=stack, base_eq=dataclasses.replace(base, frozen=base.frozen.select(mask)),
                  raw25=raw25, raw25_eq=raw25_eq)


def _one(t: float) -> tuple[float, np.ndarray, np.ndarray]:
    s = _STATE
    p_ctl = of.forecast_with_stack(s["base"], s["stack"], ba._simulated_live(s["raw25"], t), t)["probability"]
    p_e3 = of.forecast_with_stack(s["base_eq"], s["stack"], ba._simulated_live(s["raw25_eq"], t), t)["probability"]
    return t, np.asarray(p_ctl, np.float64), np.asarray(p_e3, np.float64)


def site_cells(art: of.LoadedArtifact, mask: np.ndarray) -> dict[str, dict]:
    """The cells holding the most removed (non-earthquake) M5+ frozen events -- found from the catalog, not
    typed -- with how many each holds."""
    removed = art.frozen.select(~mask)
    cells = of.cell_index(removed.lat, removed.lon)
    ids, counts = np.unique(cells, return_counts=True)
    top = np.argsort(-counts, kind="stable")[:N_SITES]
    return {f"r{int(ids[i]) // of.N_LON}c{int(ids[i]) % of.N_LON}": {"cell": int(ids[i]),
                                                                    "removed_m5_events": int(counts[i])}
            for i in top}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", default="dev,final")
    args = ap.parse_args(argv)
    splits = args.splits.split(",")
    t0 = time.time()

    _init(splits)
    base, stack = _STATE["base"], _STATE["stack"]
    mask = typed_frozen_mask(base)
    stack_rep = json.loads((C.RESULTS / "gear1_e1.json").read_text(encoding="utf-8"))
    co = stack_rep["coefficients_fitted_on_choose"]["S1"]
    theta = np.array([co["a"], co["c"], co["b"]])
    g_log10 = gs.gear1_log_map(json.loads((C.RESULTS / "gear1_cells.json").read_text(encoding="utf-8"))["cells_per_year"])
    p0 = float(json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))["fit"]["p0"])
    events = C.load_events(4.5)
    sites = site_cells(base, mask)

    report = {"amendment": "E3 (section 11)", "base": base.model_version, "stack": stack.model_version,
              "frozen_events": int(len(mask)), "frozen_non_earthquakes_removed": int((~mask).sum()),
              "margin_nats_per_target": MARGIN, "splits": {}}
    for split in splits:
        issue = C.split_issue_times(split)
        want = gs.predict_logistic(theta, gs.logit(C.load_pred("C0", split)), g_log10)   # the evaluated S1
        P_ctl = np.empty((len(issue), of.N_CELLS))
        P_e3 = np.empty_like(P_ctl)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(splits,)) as ex:
            for k, (t, pc, pe) in enumerate(ex.map(_one, [float(x) for x in issue])):
                P_ctl[k], P_e3[k] = pc, pe
                if (k + 1) % 10 == 0:
                    print(f"  {split} {k + 1}/{len(issue)} ({time.time() - t0:.0f}s)", flush=True)
        worst = float(np.max(np.abs(P_ctl - want)))
        if worst > LIVE_TOL:
            raise SystemExit(f"{split}: the control does not reproduce the evaluated S1 ({worst:.2e} > {LIVE_TOL}): "
                             "nothing is compared")
        Y = C.binary_targets(events, issue)
        sc = C.SplitScorer(issue, Y, p0=p0)
        st = {"S1": sc.stats(P_ctl), "E3s": sc.stats(P_e3)}
        site = {name: {**info, "S1_mean": float(P_ctl[:, info["cell"]].mean()),
                       "E3s_mean": float(P_e3[:, info["cell"]].mean())} for name, info in sites.items()}
        report["splits"][split] = {
            "role": {"dev": "decides", "final": "further read, never used to decide"}.get(split, "reported"),
            "n_issue_times": int(len(issue)), "n_positive": int(Y.sum()),
            "control_max_abs_vs_evaluated_S1": worst,
            "models": {k: sc.summary(s) for k, s in st.items()},
            "paired_E3s_minus_S1": sc.paired(st["E3s"], st["S1"]),
            "cells_with_most_removed_events": site,
            "cells_changed": int(np.sum(np.any(P_e3 != P_ctl, axis=0)))}
        m = report["splits"][split]
        m["seconds"] = round(time.time() - t0, 1)
        C.write_json(f"e3_types_{split}.json", {**{k: v for k, v in report.items() if k != "splits"},
                                                 "splits": {split: m}})      # saved as soon as it is done
        print(f"{split}: control max |S1 - evaluated| {worst:.1e}; IG S1 {m['models']['S1']['ig_per_target']['value']:.4f} "
              f"E3s {m['models']['E3s']['ig_per_target']['value']:.4f}; dIG "
              f"{m['paired_E3s_minus_S1']['ig_per_target']['diff']:+.5f} "
              f"{[round(v, 5) for v in m['paired_E3s_minus_S1']['ig_per_target']['ci95']]}", flush=True)
    if "dev" in report["splits"]:
        lo = report["splits"]["dev"]["paired_E3s_minus_S1"]["ig_per_target"]["ci95"][0]
        report["rule"] = {"E3s_replaces_S1": bool(lo >= MARGIN), "dev_dIG_ci95_low": lo, "margin": MARGIN}
        print("RULE:", report["rule"], flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    C.write_json("e3_types.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
