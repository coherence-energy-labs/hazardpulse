#!/usr/bin/env python3
"""Tornado program amendment 11 (docs/TORNADO_MODEL_PROGRAM.md, T2b): each served payload's Platt map refitted on
NOAA's new ProbSevere format.

    python scripts/tornado_program/t2b_recalibrate.py control-loyo OOF_DIR      # served (a, b) from their own LOYO scores
    python scripts/tornado_program/t2b_recalibrate.py control-live [REPLAY]     # the margin path gives the live numbers
    python scripts/tornado_program/t2b_recalibrate.py build 2025-08-05 2025-10-31   # rows per UTC day ($T2B_DIR)
    python scripts/tornado_program/t2b_recalibrate.py fit                       # the artifact and the fit's facts

The rows are amendment 10's (``t2_decide.day_rows``: the 30-minute slot files, block P as served, the training
labels on the storm's own track from SPC's preliminary reports, block W from the IEM archive), with all three
horizons kept. The fit is ``definitive_model.fit_platt`` -- the code that fitted every served calibration
(``tornado_lab._final_core``) -- on the payload's raw margin, unweighted, every new-format row of 2025-08-05 20:48Z
.. 2026-09-30. ``fit`` writes ``results/models/tornado_v3_recal_t2b.json`` (the artifact the live shadow reads) and
``results/tornado_program/t2b_fit.json`` (the fit's facts; in-sample, descriptive, decide nothing).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit_20261001"))

from hazardpulse.tornado import definitive_model as dm  # noqa: E402
from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402
from hazardpulse.tornado import t2b_shadow as tb  # noqa: E402
from hazardpulse.tornado import v3_serving as vs  # noqa: E402
from hazardpulse.verification import nws_warnings as nw  # noqa: E402


def _t2():
    spec = importlib.util.spec_from_file_location("t2_decide_for_t2b", ROOT / "scripts" / "tornado_program" / "t2_decide.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t2 = _t2()

T2B_DIR = Path(os.environ.get("T2B_DIR", str(ROOT / ".cache" / "t2b")))
ROWS = T2B_DIR / "rows"
MODELS = ROOT / "results" / "models"
ARTIFACT = MODELS / tb.ARTIFACT_FILE
FACTS = ROOT / "results" / "tornado_program" / "t2b_fit.json"
CONTROL_LOYO = ROOT / "results" / "tornado_program" / "t2b_control_loyo.json"
#: the window (amendment 11): the first new-format scan (amendment 10) through 2026-09-30
FIRST_NEW_FORMAT = dt.datetime(2025, 8, 5, 20, 48, tzinfo=dt.timezone.utc)
WINDOW = ("20250805", "20260930")
#: candidate -> the saved leave-one-year-out score set its served calibration was fitted on (tornado_lab._save_oof)
LOYO_SETS = {"p60_w": "v3_plus_W.npz", "p60": "v3_primary.npz", "p30": "v3_plus_W_30.npz", "p90": "v3_plus_W_90.npz"}
REPRODUCE_TOL = 1e-9          # tornado_lab.oof_only's standard
HORIZON_COL = {lab: i for i, lab in enumerate(t2.HORIZON_LABELS)}
CODE_FILES = ("scripts/tornado_program/t2b_recalibrate.py", "scripts/tornado_program/t2_decide.py",
              "scripts/score_tornado_prospective.py", "src/hazardpulse/tornado/definitive_model.py",
              "src/hazardpulse/tornado/lgbm_payload.py", "src/hazardpulse/tornado/storm_features.py",
              "src/hazardpulse/tornado/t2b_shadow.py", "src/hazardpulse/data/probsevere.py",
              "src/hazardpulse/verification/nws_warnings.py")


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str | None:
    import subprocess
    try:
        got = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return got.stdout.strip() or None if got.returncode == 0 else None


def _payload(k: str) -> dict:
    return lp.load(MODELS / tb.CANDIDATES[k])


# --------------------------------------------------------------------------- control 1: the served (a, b)
def control_loyo(oof_dir: Path) -> int:
    """Each served calibration reproduced by ``fit_platt`` on its own saved leave-one-year-out scores (2021-2024),
    within 1e-9. Writes the result (with each score set's SHA-256) beside the fit's facts."""
    res = {"amendment": 11, "control": "served (a, b) from their own LOYO scores, fit_platt", "tolerance": REPRODUCE_TOL,
           "candidates": {}}
    ok = True
    for k, fname in LOYO_SETS.items():
        f = Path(oof_dir) / fname
        payload = _payload(k)
        with np.load(f) as z:
            s, y = z["score"].astype(np.float64), z["y"].astype(np.float64)
        cal = dm.fit_platt(s, y)
        served = payload["calibration"]
        da, db = float(cal["a"]) - float(served["a"]), float(cal["b"]) - float(served["b"])
        good = abs(da) <= REPRODUCE_TOL and abs(db) <= REPRODUCE_TOL
        ok &= good
        res["candidates"][k] = {"payload": tb.CANDIDATES[k], "payload_sha256": vs.model_digest(payload),
                                "loyo_scores": fname, "loyo_scores_sha256": _sha256_file(f),
                                "n": int(len(y)), "events": int(y.sum()),
                                "served": {"a": float(served["a"]), "b": float(served["b"])},
                                "refit": {"a": float(cal["a"]), "b": float(cal["b"])},
                                "difference": {"a": da, "b": db}, "bit_identical": da == 0.0 and db == 0.0,
                                "reproduced": good}
        print(f"{k:6s} {fname:18s} n={len(y)} events={int(y.sum())} served a={served['a']!r} b={served['b']!r} "
              f"refit a={cal['a']!r} b={cal['b']!r} -> {'reproduced' if good else 'NOT REPRODUCED'}", flush=True)
    res["ok"] = bool(ok)
    CONTROL_LOYO.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_LOYO.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


# --------------------------------------------------------------------------- control 2: the margin path, live
def control_live(replay_dir: Path) -> int:
    """On every live v3 record that stored its inputs, ``proba_from_margin(served (a, b), predict_raw)`` -- the
    fit's margin path -- gives back the record's ``probability_60min`` (the payload that served it) and every
    product probability the coherence rule did not clip, within ``t2_decide.MAX_PROB_ULP`` ulp. Exit 1 otherwise."""
    payloads = {k: _payload(k) for k in tb.CANDIDATES}
    by_model = {"v3_w": "p60_w", "v3": "p60"}
    items: dict[str, list[tuple[str, dict, float]]] = {k: [] for k in tb.CANDIDATES}
    for f in sorted(Path(replay_dir).glob("to_fcst_*.json")):
        text = f.read_text(encoding="utf-8")
        if '"inputs"' not in text:
            continue
        for s in json.loads(text).get("storms") or []:
            v3 = s.get("v3") or {}
            if not v3.get("inputs") or v3.get("model") not in by_model:
                continue
            if v3.get("model_version") != lp.model_version(payloads[by_model[v3["model"]]]):
                continue                     # a record of another payload: not this control's
            tag = f"{f.stem} {s.get('storm_id')}"
            items[by_model[v3["model"]]].append((tag, v3["inputs"], float(v3["probability_60min"])))
            for k, field in tb.PRODUCT_FIELDS.items():
                if v3.get(field) is not None and k not in (v3.get("coherence_clipped") or []):
                    items[k].append((tag, v3["inputs"], float(v3[field])))
    bad, hist, n = [], {}, 0
    for k, rows in items.items():
        if not rows:
            continue
        payload = payloads[k]
        F = lp.predict_raw(payload, tb.input_matrix(payload, [r[1] for r in rows]))
        for j, (tag, _inp, p) in enumerate(rows):
            u = t2.ulp_distance(float(tb.proba_from_margin(payload["calibration"], F[j:j + 1])[0]), p)
            hist[u] = hist.get(u, 0) + 1
            n += 1
            if u > t2.MAX_PROB_ULP:
                bad.append(f"{tag} {k}: {u} ulp from the record")
    counts = {k: len(v) for k, v in items.items()}
    print(f"control-live: {n} recorded probabilities ({counts}); ulp distances {dict(sorted(hist.items()))}; "
          f"beyond {t2.MAX_PROB_ULP} ulp: {len(bad)}", flush=True)
    for b in bad[:20]:
        print("  ", b)
    return 1 if (bad or n == 0) else 0


# --------------------------------------------------------------------------- rows
def build(start: dt.date, end: dt.date) -> int:
    """Amendment 10's rows (``t2_decide.day_rows``) for every UTC day of [start, end], into $T2B_DIR/rows."""
    t2.ROWS = ROWS
    t2.PS_DIR = T2B_DIR / "probsevere"
    ROWS.mkdir(parents=True, exist_ok=True)
    reports, failed = t2.preliminary_reports(start, end)
    # every year a query in [start, end] can need (t2_decide.build passes only {start - 1, end}: a window that
    # crosses a new year would lose the year between)
    warnings = nw.load_tor_warnings(list(range(start.year - 1, end.year + 1)))
    log = {"start": start.isoformat(), "end": end.isoformat(), "spc_failed_convective_days": failed, "days": {}}
    d = start
    while d <= end:
        ds = d.strftime("%Y%m%d")
        out = ROWS / f"{ds}.npz"
        if not out.exists():
            r = t2.day_rows(ds, reports, warnings)
            if r is None:
                log["days"][ds] = "no ProbSevere"
            else:
                np.savez_compressed(out, **r)
                log["days"][ds] = f"{len(r['y'])} obs, {int(r['y'].sum())} storm_60"
        else:
            log["days"][ds] = "built"
        print(ds, log["days"][ds], flush=True)
        d += dt.timedelta(days=1)
    (T2B_DIR / f"build_log_{start.strftime('%Y%m%d')}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


def _days(start: str, end: str) -> list[str]:
    d0 = dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    d1 = dt.date(int(end[:4]), int(end[4:6]), int(end[6:]))
    return [(d0 + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


def load_window(start: str = WINDOW[0], end: str = WINDOW[1]) -> tuple[dict, dict]:
    """Every row of the window and the facts of how it was assembled. Stops (SystemExit) on a day never built."""
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(T2B_DIR.glob("build_log_*.json"))]
    if not logs:
        raise SystemExit(f"no build_log_*.json in {T2B_DIR}: run `build` first")
    failed = sorted({d for lg in logs for d in lg["spc_failed_convective_days"]})
    built = {d: v for lg in logs for d, v in lg["days"].items()}
    missing = [d for d in _days(start, end) if d not in built]
    if missing:
        raise SystemExit(f"{len(missing)} days of {start}..{end} were never built (first: {missing[:5]})")
    bad = set()
    for iso in failed:   # convective day c holds UTC c 12Z .. c+1 12Z: it labels UTC days c-1 (late), c and c+1
        c = dt.date.fromisoformat(iso)
        bad |= {(c + dt.timedelta(days=k)).strftime("%Y%m%d") for k in (-1, 0, 1)}
    parts, dropped = [], []
    for ds in _days(start, end):
        f = ROWS / f"{ds}.npz"
        if not f.exists():
            continue
        if ds in bad:
            dropped.append(ds)
            continue
        with np.load(f) as z:
            parts.append({k: z[k] for k in ("P", "y", "y3", "t", "sid", "w_active", "w_minutes")})
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    facts = {"spc_failed_convective_days": failed, "days_dropped_unlabelable": dropped,
             "days_without_probsevere": sorted(d for d, v in built.items() if v == "no ProbSevere" and start <= d <= end)}
    return cat, facts


def data_digest(rows: dict) -> str:
    """SHA-256 of the fit's rows in their order: t (int64), sid (UTF-8, NUL-free, newline-joined), P (float32),
    w_active, w_minutes (float32), y3 (int8) -- each little-endian, concatenated in that order."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rows["t"], "<i8").tobytes())
    h.update("\n".join(str(s) for s in rows["sid"]).encode("utf-8"))
    h.update(np.ascontiguousarray(rows["P"], "<f4").tobytes())
    h.update(np.ascontiguousarray(rows["w_active"], "<f4").tobytes())
    h.update(np.ascontiguousarray(rows["w_minutes"], "<f4").tobytes())
    h.update(np.ascontiguousarray(rows["y3"], "i1").tobytes())
    return h.hexdigest()


def margins(payload: dict, rows: dict, chunk: int = 200_000) -> np.ndarray:
    """The payload's raw margin on every row, its inputs exactly as served (``t2_decide.columns``)."""
    X = t2.columns(payload, rows, absent_nan=False)
    return np.concatenate([lp.predict_raw(payload, X[i:i + chunk]) for i in range(0, len(X), chunk)])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def describe(y: np.ndarray, p: np.ndarray) -> dict:
    return {"mean_forecast": float(p.mean()), "mean_over_base_rate": float(p.mean() / y.mean()),
            "brier": float(np.mean((p - y) ** 2)), "log_loss": log_loss(y, p), "auc": float(dm.compute_auc(y, p))}


def fit(start: str = WINDOW[0], end: str = WINDOW[1]) -> int:
    raw, facts = load_window(start, end)
    t = raw["t"]
    cut = int(FIRST_NEW_FORMAT.timestamp())
    first_day = (t >= int(dt.datetime(2025, 8, 5, tzinfo=dt.timezone.utc).timestamp())) & (t < cut)
    old_on_first_day = int(np.count_nonzero(raw["P"][first_day, t2.P_NAMES.index("p_ps")]))
    # the cut separates the formats: 2025-08-05 before 20:48Z is old-format (some p_ps nonzero) ...
    if not old_on_first_day:
        raise SystemExit("no 2025-08-05 row before 20:48Z has p_ps != 0: the cut does not separate the formats")
    keep = t >= cut
    rows = {k: v[keep] for k, v in raw.items()}
    # ... and every kept row is new-format: the two attributes NOAA removed read 0.0
    for n in t2.ABSENT:
        nz = int(np.count_nonzero(rows["P"][:, t2.P_NAMES.index(n)]))
        if nz:
            raise SystemExit(f"{n} is nonzero on {nz} kept rows: not the new format amendment 11 registered")
    day = (rows["t"] // 86400).astype(np.int64)
    n_days = int(len(np.unique(day)))
    sha = data_digest(rows)
    commit = os.environ.get("GITHUB_SHA") or _git_head()
    code = {"commit": commit, "files_sha256": {f: _sha256_file(ROOT / f) for f in CODE_FILES}}
    window = {"first_row": dt.datetime.fromtimestamp(int(rows["t"].min()), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "last_row": dt.datetime.fromtimestamp(int(rows["t"].max()), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "registered": "2025-08-05T20:48Z .. 2026-09-30 (every 30-minute slot)", "days": n_days}
    cands, described = {}, {}
    for k in tb.CANDIDATES:
        payload = _payload(k)
        label = tb.LABELS[k]
        y = rows["y3"][:, HORIZON_COL[label]].astype(np.float64)
        F = margins(payload, rows)
        cal = dm.fit_platt(F, y)                       # raises PlattNotConverged: then no artifact
        if not float(cal["a"]) > 0.0:
            raise SystemExit(f"{k}: fitted a = {cal['a']!r} <= 0: void (the map would not be increasing)")
        served = tb.proba_from_margin(payload["calibration"], F)
        recal = tb.proba_from_margin(cal, F)
        ev_days = int(len(np.unique(day[y == 1])))
        cands[k] = {"a": float(cal["a"]), "b": float(cal["b"]), "method": "platt", "label": label,
                    "payload": tb.CANDIDATES[k], "payload_sha256": vs.model_digest(payload),
                    "served": {"a": float(payload["calibration"]["a"]), "b": float(payload["calibration"]["b"])},
                    "n": int(len(y)), "events": int(y.sum()), "event_days_utc": ev_days}
        described[k] = {"label": label, "n": int(len(y)), "events": int(y.sum()), "base_rate": float(y.mean()),
                        "served": describe(y, served), "recalibrated": describe(y, recal),
                        "margin": {"mean": float(F.mean()), "p50": float(np.median(F)),
                                   "p99": float(np.percentile(F, 99)), "max": float(F.max())}}
        print(f"{k:6s} {label}: n={len(y)} events={int(y.sum())} a={cal['a']!r} b={cal['b']!r}  mean forecast "
              f"served {served.mean():.3e} recal {recal.mean():.3e} base rate {y.mean():.3e}", flush=True)
    art = {"schema": tb.SCHEMA, "program": tb.PROGRAM, "method": "Platt (definitive_model.fit_platt) on the "
           "payload's raw margin (lgbm_payload.predict_raw), unweighted, every row",
           "reports": "SPC preliminary filtered tornado reports (YYMMDD_rpts_filtered_torn.csv), timed as "
                      "score_tornado_prospective times them",
           "rows": "t2_decide.day_rows: 30-minute slot files, block P as served (p_ps, p_vil_density 0.0), block W "
                   "from the IEM archive, storm_30/60/90 on the storm's own track",
           "window": window, "data_sha256": sha, "data_sha256_definition": data_digest.__doc__.strip(),
           "code": code, "candidates": cands, **facts}
    tb.validate(art)
    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_bytes(tb.canonical_bytes(art))
    out = {"amendment": 11, "program": "docs/TORNADO_MODEL_PROGRAM.md",
           "artifact": f"results/models/{tb.ARTIFACT_FILE}", "artifact_sha256": tb.digest(art),
           "note": "in-sample on the fit window: descriptive, decides nothing (the rule is prospective)",
           "window": window, "n_rows": int(len(rows["t"])), "rows_before_cut_on_2025_08_05": int(first_day.sum()),
           "old_format_rows_before_cut_with_p_ps": old_on_first_day, "data_sha256": sha, "code": code, **facts,
           "candidates": described}
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    FACTS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {ARTIFACT} ({tb.digest(art)[:12]}) and {FACTS}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "control-loyo":
        return control_loyo(Path(argv[1]))
    if cmd == "control-live":
        return control_live(Path(argv[1]) if len(argv) > 1 else ROOT / "dist" / "data" / "replay")
    if cmd == "build":
        return build(dt.date.fromisoformat(argv[1]), dt.date.fromisoformat(argv[2]))
    if cmd == "fit":
        return fit()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
