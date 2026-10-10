"""Program TC1, amendment 7 (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md): R48, our RI model's 48-h exceedance curve,
and TC2c+O, TC2b+O with the second day bracketed by it.

    PYTHONPATH=src python scripts/hurricane_r48.py build    # R48: out-of-fold curves 2022-2025, the 2026 model, its gate
    PYTHONPATH=src python scripts/hurricane_r48.py tc2c     # the registered test, DEV and 2026

R48 is H8 (the RI program's v10.4 model) with a 48-h label: the same rows and inputs, the same monotone constraints
(the threshold lowers the probability), parameters, seeds and early stopping. Controls (each stops the run): the
24-h out-of-fold curves of the same rows reproduce amendment 8's log loss (the TC2 curves file records it); R48's
out-of-fold log loss over its thresholds is below climatology's; the DEV TC2b+O numbers reproduce amendment 5's.
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

import hurricane_tc1 as tc1  # noqa: E402
import hurricane_tc2 as t2  # noqa: E402
from hazardpulse.hurricane import consensus as cs  # noqa: E402
from hazardpulse.hurricane import tc2  # noqa: E402

OUT_DIR = ROOT / "results" / "hurricane_tc2"
CURVES48_DEV = OUT_DIR / "r48_curves_dev.json"
CURVES48_2026 = OUT_DIR / "r48_curves_2026.json"
BUILD_OUT = OUT_DIR / "r48_build.json"
TC2C_OUT = OUT_DIR / "tc2c.json"
ARTIFACT = ROOT / "results" / "models" / "hurricane_ri_r48.json"
K48 = tc2.R48_THRESHOLDS_KT
TC_STATUS = frozenset(("TD", "TS", "HU", "SD", "SS"))


def _ri():
    import hurricane_ri_g1 as g1
    return g1


def dv48(row, truth) -> float | None:
    """Best-track wind at t + 48 h minus at t; None unless both fixes are tropical or subtropical."""
    tr = truth.get(str(row["atcf_id"]).upper())
    if not tr:
        return None
    t = dt.datetime.strptime(row["dtg"], "%Y%m%d%H")
    a, b = tr.get(t), tr.get(t + dt.timedelta(hours=48))
    if not a or not b or a[3] not in TC_STATUS or b[3] not in TC_STATUS:
        return None
    if not (math.isfinite(a[2]) and math.isfinite(b[2])):
        return None
    return float(b[2] - a[2])


def _rows48(rows, names, ks=K48):
    import hurricane_ri_v9 as v9
    X = v9.design(rows, names)
    dv = np.array([r["dv48"] for r in rows])
    Xs = np.vstack([np.hstack([X, np.full((len(X), 1), float(k))]) for k in ks])
    ys = np.concatenate([(dv >= k).astype(float) for k in ks])
    return Xs, ys


def fit48(train, names, mono=None):
    """H8's learner on 48-h labels (``hurricane_ri_v10_challengers.fit`` with the label and thresholds changed).
    ``mono``: the constraints, threshold column included; H8's own for its names when not given."""
    import lightgbm as lgb
    g1 = _ri()
    ch, v9 = g1.ch, g1.v9
    mono = ch.monotone(names, True) if mono is None else mono
    last = max(r["season"] for r in train)
    Xa, ya = _rows48(train, names)
    Xi, yi = _rows48([r for r in train if r["season"] < last], names)
    Xv, yv = _rows48([r for r in train if r["season"] == last], names)
    models, rounds = [], []
    for s in g1.v10.SEEDS:
        p = dict(v9.GBT_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s,
                 monotone_constraints=mono)
        b = lgb.train(p, lgb.Dataset(Xi, yi), 2000, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        r = max(1, int(b.best_iteration or 1))
        rounds.append(r)
        models.append(lgb.train(p, lgb.Dataset(Xa, ya), r))
    return models, rounds


def predict48(models, rows, names):
    import hurricane_ri_v9 as v9
    X = v9.design(rows, names)
    return {k: np.mean([m.predict(np.hstack([X, np.full((len(X), 1), float(k))])) for m in models], axis=0) for k in K48}


def _ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def build() -> int:
    g1 = _ri()
    ch, v9, v10, h9 = g1.ch, g1.v9, g1.v10, g1.h9
    c24 = json.loads(t2.CURVES_DEV.read_text(encoding="utf-8"))
    if abs(float(c24["control_log_loss"]) - t2.H8_LL) > 1e-12:
        raise SystemExit("control failed: the 24-h curves file does not record amendment 8's log loss")
    rows = v10.load(v10.DEV10)
    h9.prepare(rows, h9.h8.adeck_dev)
    names = list(ch.CANDS["H8"]["names"])
    truth = tc1.load_truth(range(2020, 2026))
    labelled = []
    for r in rows:
        d = dv48(r, truth)
        if d is not None:
            r["dv48"] = d
            labelled.append(r)
    out, ys, ps, clim = {}, {k: [] for k in K48}, {k: [] for k in K48}, {k: [] for k in K48}
    for season in v9.FOLDS:
        train = [r for r in labelled if r["season"] < season]
        test = [r for r in labelled if r["season"] == season]
        models, _ = fit48(train, names)
        P = predict48(models, test, names)
        for k in K48:
            y = np.array([r["dv48"] >= k for r in test], float)
            base = float(np.mean([r["dv48"] >= k for r in train]))
            ys[k].extend(y), ps[k].extend(P[k]), clim[k].extend([base] * len(test))
        for i, r in enumerate(test):
            out[t2._key(r["atcf_id"], r["dtg"])] = {"curve": {str(k): float(P[k][i]) for k in K48},
                                                     "gate_ok": t2._gate(r), "season": r["season"]}
    ll_model = float(np.mean([_ll(np.array(ys[k]), np.array(ps[k])) for k in K48]))
    ll_clim = float(np.mean([_ll(np.array(ys[k]), np.array(clim[k])) for k in K48]))
    per_k = {str(k): {"events": int(np.sum(ys[k])), "n": len(ys[k]), "log_loss": _ll(np.array(ys[k]), np.array(ps[k])),
                      "climatology": _ll(np.array(ys[k]), np.array(clim[k])), "mean_forecast": float(np.mean(ps[k])),
                      "event_rate": float(np.mean(ys[k]))} for k in K48}
    if not ll_model < ll_clim:
        raise SystemExit(f"R48's gate failed: out-of-fold log loss {ll_model:.5f} is not below climatology's {ll_clim:.5f}")
    # 2026: fitted on every development season, applied to the 2026 rows (their inputs only)
    models, rounds = fit48(labelled, names)
    c26 = v10.cases_2026()
    h9.prepare(c26, h9.h8.adeck_2026)
    P26 = predict48(models, c26, names)
    out26 = {t2._key(r["atcf_id"], r["dtg"]): {"curve": {str(k): float(P26[k][i]) for k in K48}, "gate_ok": t2._gate(r)}
             for i, r in enumerate(c26)}
    _export(models, names, rounds, ll_model, ll_clim, len(labelled))
    meta = {"program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 7)", "prereg_tag": "prereg-r48-tc2c"}
    CURVES48_DEV.write_text(json.dumps({**meta, "model": "R48, out of fold per season", "curves": out}), encoding="utf-8")
    CURVES48_2026.write_text(json.dumps({**meta, "model": "R48 fitted on 2020-2025", "curves": out26}), encoding="utf-8")
    BUILD_OUT.write_text(json.dumps({**meta, "rows_labelled": len(labelled), "rows_dev": len(rows),
                                     "oof_log_loss": ll_model, "oof_climatology_log_loss": ll_clim,
                                     "per_threshold": per_k, "rounds_2026_model": rounds}, indent=1), encoding="utf-8")
    tc1.log(f"R48: {len(labelled)} of {len(rows)} rows have a 48-h label; out-of-fold log loss {ll_model:.5f} vs "
            f"climatology {ll_clim:.5f} (gate passed); per threshold: " + ", ".join(
                f"{k}: {v['log_loss']:.4f}/{v['climatology']:.4f} ({v['events']} ev)" for k, v in per_k.items()))
    return 0


def _export(models, names, rounds, ll, ll_clim, n) -> None:
    """R48 with v10.4's schema and input names, so ``ri_v10.recompute`` scores a v10.4 record's inputs with it."""
    from hazardpulse.hurricane import ri_v10
    from hazardpulse.tornado import lgbm_payload as lp
    members = [lp.export_booster(m, names + ["threshold_kt"], calibration={"method": "identity", "a": 1.0, "b": 0.0},
                                 provenance={}) for m in models]
    art = {"schema": ri_v10.SCHEMA, "model_name": "hurricane_ri_r48", "label": "R48", "thresholds_kt": list(K48),
           "feature_names": names, "members": members,
           "provenance": {"program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 7)", "prereg_tag": "prereg-r48-tc2c",
                          "event": "V(t+48 h) - V(t) >= k kt (best track; tropical or subtropical at both times)",
                          "trained": "NHC cycles 2020-2025 with a 48-h label", "rows": n, "rounds": rounds,
                          "inputs": "H8's (v10.4's) 67 inputs", "oof_2022_2025": {"log_loss": ll, "climatology": ll_clim}}}
    ARTIFACT.write_bytes(ri_v10.canonical_bytes(art))


def _tc2c_product(decks, run_o, c24, c48):
    """TC2b+O as amendment 5 builds it, then the second day bracketed by R48 (amendment 7)."""
    prod_b, _ = t2.tc2_product(decks, run_o, c24, tc2.TC2B_TAPER_END_H)
    by_storm = {d.storm: d for d in decks}
    cycles: dict = {}
    for (storm, t, lead), (fc, _w) in prod_b.items():
        cycles.setdefault((storm, t), {})[lead] = fc
    out, stats = {}, {"cycles": 0, "with_curve": 0, "up": 0, "down": 0}
    shifts = []
    for (storm, t), by_lead in cycles.items():
        stats["cycles"] += 1
        c = c48.get(t2._key(storm, t))
        curve = {int(k): v for k, v in c["curve"].items()} if c and c.get("gate_ok") else None
        an = by_storm[storm].analysis(t)
        v0 = an[2] if an is not None and math.isfinite(an[2]) else None
        proj, info = tc2.project_second_day(by_lead, v0, curve)
        stats["with_curve"] += curve is not None
        if info.get("applied"):
            stats["up" if info["shift_48h"] > 0 else "down"] += 1
            shifts.append(info["shift_48h"])
        for lead, v in proj.items():
            out[(storm, t, lead)] = (v, None)
    sh = np.array(shifts or [0.0])
    stats.update({"mean_abs_shift_48h_kt": float(np.abs(sh).mean()) if shifts else 0.0,
                  "max_up_kt": float(sh.max()), "max_down_kt": float(sh.min())})
    return prod_b, out, stats


def _ri48(case_list) -> set:
    keys = set()
    for d, tr, t in case_list:
        a, b = tr.get(t), tr.get(t + dt.timedelta(hours=48))
        if a and b and math.isfinite(a[2]) and math.isfinite(b[2]) and b[2] - a[2] >= 50:
            keys.add((d.storm, t))
    return keys


def _evaluate(decks, truth_of, seasons, c24, c48):
    case_list = tc1.cases(decks, truth_of, seasons)
    run_o = cs.run(decks, tc1.chosen("intensity", official=True), "intensity", keep_weights=False)
    prod_b, prod_c, stats = _tc2c_product(decks, run_o, c24, c48)
    products = {"TC2b+O": prod_b, "TC2c+O": prod_c, "TC1+O": run_o, "OFCL": "OFCL", "HCCA": "HCCA", "IVCN": "IVCN"}
    errs = tc1.score(case_list, "intensity", t2.REPORT_LEADS, products)
    ri24, ri48 = t2._ri_subset(case_list), _ri48(case_list)

    def sub(keys, ld):
        return {p: (float(np.mean([v for k, v in errs[p][ld].items() if k in keys])) if any(k in keys for k in errs[p][ld]) else None,
                    sum(1 for k in errs[p][ld] if k in keys)) for p in ("TC2b+O", "TC2c+O", "OFCL")}
    return {"errors": {p: tc1.mean_error(e, tc1.SCORED_LEADS) for p, e in errs.items()},
            "TC2c+O-TC2b+O": tc1.paired(errs["TC2c+O"], errs["TC2b+O"], tc1.SCORED_LEADS, tc1.REPORT_LEVEL),
            "reported": {f"TC2c+O-{q}": tc1.paired(errs["TC2c+O"], errs[q], tc1.SCORED_LEADS, tc1.REPORT_LEVEL)
                         for q in ("OFCL", "HCCA", "IVCN")},
            "ri24_subset": {"cycles": len(ri24), "24": sub(ri24, 24)},
            "ri48_subset": {"cycles": len(ri48), "48": sub(ri48, 48)}, "shift_stats": stats}


def _two_seasons(c48: dict, c48_26: dict):
    """TC2b+O and the second-day bracket from ``c48`` on DEV, then from ``c48_26`` on 2026, after the control that
    DEV TC2b+O reproduces amendment 5. Returns (the control value, DEV, 2026, the 2026 storms)."""
    c24 = json.loads(t2.CURVES_DEV.read_text(encoding="utf-8"))["curves"]
    c24_26 = json.loads(t2.CURVES_2026.read_text(encoding="utf-8"))["curves"]
    seasons = tc1.WARMUP + tc1.CHOOSE + tc1.DEV
    truth = tc1.load_truth(seasons)
    decks = tc1.load_decks(tc1.ALL_TECHS, seasons)
    dev = _evaluate(decks, lambda d: tc1.truth_for(d, truth), tc1.DEV, c24, c48)
    want = json.loads(t2.TC2BO_OUT.read_text(encoding="utf-8"))["dev"]["errors"]["TC2b+O"]["mean_over_leads"]
    got = dev["errors"]["TC2b+O"]["mean_over_leads"]
    if abs(got - want) > 1e-9:
        raise SystemExit(f"control failed: DEV TC2b+O {got!r} != amendment 5's {want!r}")
    del decks
    storms = tc1.fetch_2026()
    decks26 = tc1.load_decks(tc1.ALL_TECHS, seasons + (2026,))
    t26 = {s: tc1.btk_truth(s) for s in storms}
    s26 = _evaluate(decks26, lambda d: t26.get(d.storm, {}), (2026,), c24_26, c48_26)
    return got, dev, s26, storms


def _better(r) -> bool:
    """Amendment 7's rule for one season: the second-day product below TC2b+O at 48 h and over 24-120 h."""
    d = r["TC2c+O-TC2b+O"]
    return bool(d.get("n_storms") and d["per_lead"]["48"]["d"] < 0 and d["mean_over_leads"]["d"] < 0)


def tc2c() -> int:
    c48 = json.loads(CURVES48_DEV.read_text(encoding="utf-8"))["curves"]
    c48_26 = json.loads(CURVES48_2026.read_text(encoding="utf-8"))["curves"]
    got, dev, s26, storms = _two_seasons(c48, c48_26)
    carried = _better(dev) and _better(s26)
    TC2C_OUT.write_text(json.dumps({
        "phase": "TC2c+O test (amendment 7)", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 7)",
        "prereg_tag": "prereg-r48-tc2c", "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "control_tc2bo_dev_intensity": got, "storms_2026": storms, "carried": "TC2c+O" if carried else "TC2b+O",
        "dev": dev, "season_2026": s26}, indent=1), encoding="utf-8")
    tc1.log(f"control passed: DEV TC2b+O {got:.6f}")
    for name, r in (("DEV", dev), ("2026", s26)):
        d = r["TC2c+O-TC2b+O"]
        e = r["errors"]
        tc1.log(f"{name}: " + ", ".join(f"{p} {v['mean_over_leads']:.3f}" for p, v in e.items() if v["mean_over_leads"])
                + f" | TC2c+O - TC2b+O mean {d['mean_over_leads']['d']:+.3f} [{d['mean_over_leads']['ci'][0]:+.3f}, "
                f"{d['mean_over_leads']['ci'][1]:+.3f}]; " + ", ".join(
                    f"{ld} h {v['d']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for ld, v in d["per_lead"].items())
                + f" | RI48 {json.dumps(r['ri48_subset'])} | shifts {json.dumps(r['shift_stats'])}")
    tc1.log(f"CARRIED TC2c+O: {carried}")
    return 0


# ---------------------------------------------------------------------------
# Amendment 8: R48g -- R48 told the guidance's 48-h forecasts and TC1+O's own 48-h change
# ---------------------------------------------------------------------------

R48_OOF_LL = 0.2702259918074902                      # amendment 7's out-of-fold log loss (r48_build.json)
R48G_CURVES_DEV = OUT_DIR / "r48g_curves_dev.json"
R48G_CURVES_2026 = OUT_DIR / "r48g_curves_2026.json"
R48G_BUILD = OUT_DIR / "r48g_build.json"
R48G_SCREEN = OUT_DIR / "r48g_screen.json"
R48_SHIFTED_DEV, R48_MEAN_ABS_SHIFT_DEV, R48_DEV_48H = 485, 8.271870854224955, 1.045   # amendment 7's outcome


def _g_names() -> tuple[str, ...]:
    from hazardpulse.hurricane import ri_v9_features as fx
    return (*fx.aid_change_names(48, 50), "tc1o_dv48")


def _g_mono(names: list[str]) -> list[int]:
    """H8's constraints for H8's names; R48g's inputs raise the probability as their 24-h counterparts do, the spread
    unconstrained as ``dv24_spread`` is; the threshold column lowers it."""
    ch = _ri().ch
    g = set(_g_names())
    base = ch.monotone([n for n in names if n not in g], True)[:-1]
    return base + [0 if n == "dv48_spread" else 1 for n in names if n in g] + [-1]


def _tc1o_dv48(decks) -> dict[str, float]:
    """TC1+O's 48-h intensity minus V0 (the CARQ intensity at t), per cycle, by one online pass over ``decks``."""
    run_o = cs.run(decks, tc1.chosen("intensity", official=True), "intensity", keep_weights=False)
    by_storm = {d.storm: d for d in decks}
    out = {}
    for (storm, t, lead), (fc, _w) in run_o.items():
        if lead != 48:
            continue
        an = by_storm[storm].analysis(t)
        if an is not None and math.isfinite(an[2]) and fc is not None and math.isfinite(float(fc)):
            out[t2._key(storm, t)] = float(fc) - an[2]
    return out


def add_g48(rows, adeck_path_of, tc1o: dict) -> dict:
    """Write R48g's 14 inputs into each row's features. Control 1 (amendment 8) runs here: the same builder at 24 h
    must reproduce the row's stored 24-h guidance inputs, or the run stops."""
    import gzip
    from hazardpulse.hurricane import atcf
    from hazardpulse.hurricane import ri_v9_features as fx
    names24 = fx.aid_change_names(24, 30)
    by_storm: dict[str, list] = {}
    for r in rows:
        by_storm.setdefault(r["atcf_id"], []).append(r)
    cov = {"rows": len(rows), "decks_missing": 0, "control_values_checked": 0, "control_mismatches": 0,
           "tc1o_dv48_present": 0}
    first_bad = None
    for aid, rs in by_storm.items():
        p = adeck_path_of(aid)
        recs = atcf.parse_atcf_deck(gzip.decompress(p.read_bytes()).decode("utf-8", "replace")) if p.exists() else []
        cov["decks_missing"] += int(not p.exists())
        for r in rs:
            table = fx.cycle_table(recs, dt.datetime.strptime(r["dtg"], "%Y%m%d%H"))
            f24 = fx.aid_change_features(table, 24, 30)
            for n in names24:
                a, b = float(r["f"].get(n, math.nan)), f24[n]
                cov["control_values_checked"] += 1
                if not ((math.isnan(a) and math.isnan(b)) or a == b):
                    cov["control_mismatches"] += 1
                    first_bad = first_bad or (aid, r["dtg"], n, a, b)
            r["f"].update(fx.aid_change_features(table, 48, 50))
            r["f"]["tc1o_dv48"] = tc1o.get(t2._key(aid, r["dtg"]), math.nan)
            cov["tc1o_dv48_present"] += int(math.isfinite(r["f"]["tc1o_dv48"]))
        del recs
    if cov["control_mismatches"]:
        raise SystemExit(f"control 1 failed: the builder at 24 h differs from the stored inputs ({cov}; first {first_bad})")
    for n in _g_names():
        cov[f"present_{n}"] = sum(1 for r in rows if math.isfinite(r["f"][n]))
    return cov


def build_g() -> int:
    g1 = _ri()
    ch, v9, v10, h9 = g1.ch, g1.v9, g1.v10, g1.h9
    rows = v10.load(v10.DEV10)
    h9.prepare(rows, h9.h8.adeck_dev)
    names = list(ch.CANDS["H8"]["names"])
    names_g = names + list(_g_names())
    mono_g = _g_mono(names_g)
    storms = tc1.fetch_2026()
    seasons = tc1.WARMUP + tc1.CHOOSE + tc1.DEV
    tc1o = _tc1o_dv48(tc1.load_decks(tc1.ALL_TECHS, seasons + (2026,)))
    cov = add_g48(rows, h9.h8.adeck_dev, tc1o)
    truth = tc1.load_truth(range(2020, 2026))
    labelled = []
    for r in rows:
        d = dv48(r, truth)
        if d is not None:
            r["dv48"] = d
            labelled.append(r)
    out, ys = {}, {k: [] for k in K48}
    ps = {m: {k: [] for k in K48} for m in ("R48", "R48g")}
    clim = {k: [] for k in K48}
    for season in v9.FOLDS:
        train = [r for r in labelled if r["season"] < season]
        test = [r for r in labelled if r["season"] == season]
        P = predict48(fit48(train, names)[0], test, names)
        Pg = predict48(fit48(train, names_g, mono_g)[0], test, names_g)
        for k in K48:
            ys[k].extend(np.array([r["dv48"] >= k for r in test], float))
            ps["R48"][k].extend(P[k]), ps["R48g"][k].extend(Pg[k])
            clim[k].extend([float(np.mean([r["dv48"] >= k for r in train]))] * len(test))
        for i, r in enumerate(test):
            out[t2._key(r["atcf_id"], r["dtg"])] = {"curve": {str(k): float(Pg[k][i]) for k in K48},
                                                     "gate_ok": t2._gate(r), "season": r["season"]}
        tc1.log(f"R48g fold {season} done")

    def ll(m):
        return float(np.mean([_ll(np.array(ys[k]), np.array(ps[m][k])) for k in K48]))
    ll_r48, ll_g = ll("R48"), ll("R48g")
    ll_clim = float(np.mean([_ll(np.array(ys[k]), np.array(clim[k])) for k in K48]))
    if abs(ll_r48 - R48_OOF_LL) > 1e-12:
        raise SystemExit(f"control 2 failed: R48 rebuilt {ll_r48!r} != amendment 7's {R48_OOF_LL!r}")
    per_k = {str(k): {"events": int(np.sum(ys[k])), "n": len(ys[k]),
                      "R48g": _ll(np.array(ys[k]), np.array(ps["R48g"][k])),
                      "R48": _ll(np.array(ys[k]), np.array(ps["R48"][k])),
                      "climatology": _ll(np.array(ys[k]), np.array(clim[k])),
                      "mean_forecast_R48g": float(np.mean(ps["R48g"][k])), "event_rate": float(np.mean(ys[k]))}
             for k in K48}
    meta = {"program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 8)", "prereg_tag": "prereg-r48g"}
    gate = ll_g < ll_r48
    build_rec = {**meta, "rows_labelled": len(labelled), "rows_dev": len(rows), "coverage_dev": cov,
                 "control_r48_oof_log_loss": ll_r48, "oof_log_loss_R48g": ll_g, "oof_climatology_log_loss": ll_clim,
                 "gate_R48g_below_R48": gate, "per_threshold": per_k, "inputs_added": list(_g_names()),
                 "constraints_added": dict(zip(_g_names(), mono_g[len(names):-1]))}
    tc1.log(f"control 2 passed: R48 {ll_r48:.6f}; R48g {ll_g:.6f} (climatology {ll_clim:.6f}); gate {gate}; "
            + ", ".join(f"{k}: {v['R48g']:.4f}/{v['R48']:.4f}" for k, v in per_k.items()))
    if not gate:
        R48G_BUILD.write_text(json.dumps(build_rec, indent=1), encoding="utf-8")
        raise SystemExit(f"R48g's gate failed: out-of-fold log loss {ll_g:.6f} is not below R48's {ll_r48:.6f}")
    models, rounds = fit48(labelled, names_g, mono_g)
    c26 = v10.cases_2026()
    h9.prepare(c26, h9.h8.adeck_2026)
    cov26 = add_g48(c26, h9.h8.adeck_2026, tc1o)
    P26 = predict48(models, c26, names_g)
    out26 = {t2._key(r["atcf_id"], r["dtg"]): {"curve": {str(k): float(P26[k][i]) for k in K48}, "gate_ok": t2._gate(r)}
             for i, r in enumerate(c26)}
    R48G_CURVES_DEV.write_text(json.dumps({**meta, "model": "R48g, out of fold per season", "curves": out}), encoding="utf-8")
    R48G_CURVES_2026.write_text(json.dumps({**meta, "model": "R48g fitted on 2020-2025", "curves": out26}), encoding="utf-8")
    R48G_BUILD.write_text(json.dumps({**build_rec, "coverage_2026": cov26, "storms_2026_fetched": storms,
                                      "rounds_2026_model": rounds}, indent=1), encoding="utf-8")
    return 0


def screen_g() -> int:
    """Amendment 8's screen: TC2c'+O (R48g's curve) against TC2b+O on DEV and 2026. It can kill R48g, never carry it."""
    c48 = json.loads(R48G_CURVES_DEV.read_text(encoding="utf-8"))["curves"]
    c48_26 = json.loads(R48G_CURVES_2026.read_text(encoding="utf-8"))["curves"]
    build_rec = json.loads(R48G_BUILD.read_text(encoding="utf-8"))
    got, dev, s26, storms = _two_seasons(c48, c48_26)
    passes = _better(dev) and _better(s26)
    st = dev["shift_stats"]
    d48 = dev["TC2c+O-TC2b+O"]["per_lead"]["48"]["d"]
    predictions = {
        "1_R48g_log_loss_below_R48": bool(build_rec["gate_R48g_below_R48"]),
        "2_fewer_and_smaller_DEV_shifts": bool(st["up"] + st["down"] < R48_SHIFTED_DEV
                                               and st["mean_abs_shift_48h_kt"] < R48_MEAN_ABS_SHIFT_DEV),
        "3_DEV_48h_below_half_of_amendment_7": bool(d48 < R48_DEV_48H / 2)}
    R48G_SCREEN.write_text(json.dumps({
        "phase": "R48g screen (amendment 8): TC2c'+O = TC2c+O with R48g's curve; reported under amendment 7's names",
        "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md (amendment 8)", "prereg_tag": "prereg-r48g",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "control_tc2bo_dev_intensity": got, "storms_2026": storms, "screen_passed": passes,
        "decides": "a pass sends TC2c'+O to live shadow, judged only on cycles after prereg-r48g; a fail kills R48g",
        "predictions_of_the_cause": predictions, "dev": dev, "season_2026": s26}, indent=1), encoding="utf-8")
    tc1.log(f"control 3 passed: DEV TC2b+O {got:.6f}")
    for name, r in (("DEV", dev), ("2026", s26)):
        d = r["TC2c+O-TC2b+O"]
        tc1.log(f"{name}: TC2c'+O - TC2b+O mean {d['mean_over_leads']['d']:+.3f} [{d['mean_over_leads']['ci'][0]:+.3f}, "
                f"{d['mean_over_leads']['ci'][1]:+.3f}]; " + ", ".join(
                    f"{ld} h {v['d']:+.3f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for ld, v in d["per_lead"].items())
                + f" | RI48 {json.dumps(r['ri48_subset'])} | shifts {json.dumps(r['shift_stats'])}")
    tc1.log(f"predictions {predictions}; SCREEN PASSED: {passes}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build", "tc2c", "build_g", "screen_g"))
    a = ap.parse_args(argv)
    return {"build": build, "tc2c": tc2c, "build_g": build_g, "screen_g": screen_g}[a.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
