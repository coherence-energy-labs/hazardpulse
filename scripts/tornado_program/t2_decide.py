#!/usr/bin/env python3
"""Tornado program amendment 10 (docs/TORNADO_MODEL_PROGRAM.md): which v3 serves NOAA's new ProbSevere
format, decided on 2026-01-01 .. 2026-09-30.

    python scripts/tornado_program/t2_decide.py control [replay_dir]            # (a) vs the live records, first
    python scripts/tornado_program/t2_decide.py build 2026-01-01 2026-09-30     # one rows file per UTC day
    python scripts/tornado_program/t2_decide.py decide                          # scores, paired comparisons, rule

Every piece is the program's own: the 30-minute slot files (``probsevere.slot_start_keys``, the training
selection), the 28 ProbSevere attributes the model reads (``storm_features.block_p``), the ``storm_60`` label on
the storm's own track (``storm_features.labels``, as ``build_feature_store.build_day``), SPC's preliminary
reports timed as the prospective verifier times them (``score_tornado_prospective``), the warning state as block
W (``verification.nws_warnings``), the payloads scored by ``lgbm_payload`` (as served), and the day-bootstrap
paired comparison of ``tornado_lab``.

Writes ``$T2_DIR`` (default ``.cache/t2_2026``): ``rows/<day>.npz`` and ``build_log.json``; ``decide`` writes
``results/tornado_program/t2_2026.json``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit_20261001"))

from hazardpulse.data import probsevere as ps  # noqa: E402
from hazardpulse.tornado import definitive_model as dm  # noqa: E402
from hazardpulse.tornado import lgbm_payload as lp  # noqa: E402
from hazardpulse.tornado import storm_features as sf  # noqa: E402
from hazardpulse.verification import nws_warnings as nw  # noqa: E402

UTC = dt.timezone.utc
T2_DIR = Path(os.environ.get("T2_DIR", str(ROOT / ".cache" / "t2_2026")))
PS_DIR = T2_DIR / "probsevere"
ROWS = T2_DIR / "rows"
MODELS = ROOT / "results" / "models"
OUT = ROOT / "results" / "tornado_program" / "t2_2026.json"
P_NAMES = [f"p_{a}" for a in sf.PS_ATTRS]                 # the store's P-block names, in order
assert P_NAMES == list(sf.FEATURE_NAMES[: len(P_NAMES)]), "block P is the store's first 28 columns"
STORM_60 = sf.LABEL_NAMES.index("storm_60")
ABSENT = ("p_ps", "p_vil_density")                          # read as 0.0 since the format change
W_NAMES = ("w_tor_warning_active", "w_minutes_since_issue")
CANDIDATES = {   # name -> (+W payload, fallback payload, the two absent inputs as NaN?)
    "a_served": ("tornado_v3_w.json", "tornado_v3.json", False),
    "b_nan": ("tornado_v3_w.json", "tornado_v3.json", True),
    "d_drop2": ("candidates/tornado_v3_drop2_w.json", "candidates/tornado_v3_drop2.json", False),
}
N_BOOT = 2000
SEED = 42


def _day(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


# --------------------------------------------------------------------------- reports
def preliminary_reports(start: dt.date, end: dt.date) -> tuple[dict[str, list[dict]], list[str]]:
    """SPC preliminary filtered tornado reports for UTC [start, end + 1 day], in ``load_spc_tornado_reports``'s
    format (keyed by UTC date; ``time_utc`` POSIX seconds of the UTC instant), plus the convective days whose file
    could not be read."""
    import score_tornado_prospective as stp
    t0 = dt.datetime(start.year, start.month, start.day)
    t1 = dt.datetime(end.year, end.month, end.day) + dt.timedelta(days=2)
    reports, failed = stp.fetch_spc_reports_range(t0, t1)
    by_date: dict[str, list[dict]] = {}
    for r in reports:
        t = stp.parse_utc(r["time"]).replace(tzinfo=UTC)          # parse_utc is naive UTC: make it explicit
        by_date.setdefault(t.strftime("%Y%m%d"), []).append({
            "slat": float(r["lat"]), "slon": float(r["lon"]), "mag": int(r["mag"]), "time_utc": t.timestamp(),
            "hour": t.hour + t.minute / 60.0, "local_date": ""})
    return by_date, failed


# --------------------------------------------------------------------------- rows
def day_rows(d: str, reports: dict[str, list[dict]], warnings) -> dict | None:
    """Every storm observation of day ``d``: P block, storm_60, time, place, id, warning state."""
    steps = ps.fetch_probsevere_day(d, cache_dir=PS_DIR)
    if not steps:
        return None
    nxt = (dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
    window = dm.reports_in_label_window(reports, d)
    tracks: dict[str, list] = {}
    if window:   # as build_day: every slot of each id on d and d+1 (a label may use the future)
        for st in steps + (ps.fetch_probsevere_day(nxt, cache_dir=PS_DIR) or []):
            tv = dm.parse_probsevere_valid_time(st.get("valid_time", ""))
            if tv is None:
                continue
            for s_ in st.get("storms", []):
                tracks.setdefault(str(s_.get("id")), []).append((tv.timestamp(), s_))
    P, Y, T, LAT, LON, SID = [], [], [], [], [], []
    for ts in steps:
        t = dm.parse_probsevere_valid_time(ts.get("valid_time", ""))
        if t is None:
            continue
        for storm in ts.get("storms", []):
            lab, _ef, _lead = sf.labels(storm, t.timestamp(), window, track=tracks.get(str(storm.get("id"))))
            P.append(sf.block_p(storm))
            Y.append(int(lab[STORM_60]))
            T.append(int(t.timestamp()))
            LAT.append(float(storm.get("lat", 0.0)))
            LON.append(float(storm.get("lon", 0.0)))
            SID.append(str(storm.get("id", "")))
    if not P:
        return None
    lat, lon, tt = np.asarray(LAT, np.float64), np.asarray(LON, np.float64), np.asarray(T, np.float64)
    active, _, since = nw.tor_warning_state(lat, lon, tt, warnings=warnings, require_coverage=False)
    active = np.asarray(active, bool)
    return {"P": np.stack(P).astype(np.float32), "y": np.asarray(Y, np.int8), "t": np.asarray(T, np.int64),
            "lat": lat.astype(np.float32), "lon": lon.astype(np.float32), "sid": np.asarray(SID),
            "w_active": active.astype(np.float32),
            "w_minutes": np.where(active, np.asarray(since, np.float64), np.nan).astype(np.float32),
            "n_steps": np.int16(len(steps)), "n_reports_window": np.int32(len(window))}


def build(start: dt.date, end: dt.date) -> int:
    ROWS.mkdir(parents=True, exist_ok=True)
    reports, failed = preliminary_reports(start, end)
    warnings = nw.load_tor_warnings([start.year - 1, end.year])
    log = {"start": start.isoformat(), "end": end.isoformat(), "spc_failed_convective_days": failed,
           "days": {}}
    d = start
    while d <= end:
        ds = _day(d)
        out = ROWS / f"{ds}.npz"
        if not out.exists():
            r = day_rows(ds, reports, warnings)
            if r is None:
                log["days"][ds] = "no ProbSevere"
            else:
                np.savez_compressed(out, **r)
                log["days"][ds] = f"{len(r['y'])} obs, {int(r['y'].sum())} storm_60"
        else:
            log["days"][ds] = "built"
        print(ds, log["days"][ds], flush=True)
        d += dt.timedelta(days=1)
    (T2_DIR / f"build_log_{_day(start)}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- scoring
def load_rows(start: str, end: str, failed_days: list[str]):
    """All rows of [start, end]; a day whose label window reaches a convective day SPC could not serve is
    dropped (its storms could not be labelled honestly), and counted."""
    bad = set()
    for iso in failed_days:   # convective day c holds UTC c 12Z .. c+1 12Z: labels of UTC days c and c+1
        c = dt.date.fromisoformat(iso)
        bad |= {_day(c), _day(c + dt.timedelta(days=1)), _day(c - dt.timedelta(days=1))}
    parts, days, dropped = [], [], []
    for f in sorted(ROWS.glob("*.npz")):
        ds = f.stem
        if not (start <= ds <= end):
            continue
        if ds in bad:
            dropped.append(ds)
            continue
        with np.load(f) as z:
            parts.append({k: z[k] for k in ("P", "y", "w_active", "w_minutes")})
            days.append(np.full(len(z["y"]), int(ds), np.int64))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cat["day"] = np.concatenate(days)
    return cat, dropped


def columns(payload: dict, rows: dict, absent_nan: bool) -> np.ndarray:
    """The payload's inputs by name, exactly as served: P from block_p (the absent pair 0.0 as parsed, or NaN),
    the warning state as block W."""
    name_col = {n: i for i, n in enumerate(P_NAMES)}
    X = np.empty((len(rows["y"]), len(payload["feature_names"])), np.float64)
    for j, n in enumerate(payload["feature_names"]):
        if n in name_col:
            v = rows["P"][:, name_col[n]].astype(np.float64)
            X[:, j] = np.nan if (absent_nan and n in ABSENT) else v
        elif n == "w_tor_warning_active":
            X[:, j] = rows["w_active"]
        elif n == "w_minutes_since_issue":
            X[:, j] = rows["w_minutes"]
        else:
            raise ValueError(f"{n}: not a P or W input -- this payload reads something else")
    return X


def predict(payload: dict, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    return np.concatenate([lp.predict_proba(payload, X[i:i + chunk]) for i in range(0, len(X), chunk)])


def paired(y: np.ndarray, day: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict:
    """B minus A, whole UTC days resampled (the same draws for both): tornado_lab.compare's arithmetic."""
    import tornado_lab as lab
    uniq, idx = np.unique(day, return_inverse=True)
    D = len(uniq)
    M = lab.day_bootstrap_counts(D, N_BOOT, SEED)
    Ua, P, N = lab.day_pair_matrix(y, pa, idx, D)
    Ub, _, _ = lab.day_pair_matrix(y, pb, idx, D)
    d_auc = (np.einsum("bi,ij,bj->b", M, Ub, M) - np.einsum("bi,ij,bj->b", M, Ua, M)) / ((M @ P) * (M @ N))
    sa = np.bincount(idx, weights=(pa - y) ** 2, minlength=D)
    sb = np.bincount(idx, weights=(pb - y) ** 2, minlength=D)
    n_d = np.bincount(idx, minlength=D).astype(np.float64)
    d_bri = (M @ sb - M @ sa) / (M @ n_d)
    return {"delta_auc": dm.compute_auc(y, pb) - dm.compute_auc(y, pa),
            "delta_auc_ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
            "delta_brier": float(np.mean((pb - y) ** 2) - np.mean((pa - y) ** 2)),
            "delta_brier_ci": [float(np.percentile(d_bri, 2.5)), float(np.percentile(d_bri, 97.5))],
            "n_days": int(D)}


def decide(start: str = "20260101", end: str = "20260930") -> int:
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(T2_DIR.glob("build_log_*.json"))]
    if not logs:
        raise SystemExit(f"no build_log_*.json in {T2_DIR}: run `build` first")
    failed = sorted({d for lg in logs for d in lg["spc_failed_convective_days"]})
    built = {d: v for lg in logs for d, v in lg["days"].items()}
    missing = [d for d in (_day(dt.date(int(start[:4]), int(start[4:6]), int(start[6:])) + dt.timedelta(days=i))
                           for i in range((dt.date(int(end[:4]), int(end[4:6]), int(end[6:]))
                                           - dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))).days + 1))
               if d not in built]
    if missing:
        raise SystemExit(f"{len(missing)} days of {start}..{end} were never built (first: {missing[:5]})")
    rows, dropped = load_rows(start, end, failed)
    y = rows["y"].astype(np.float64)
    # the defect must be present on every row, or this is not the regime the amendment is about
    for n in ABSENT:
        nz = int(np.count_nonzero(rows["P"][:, P_NAMES.index(n)]))
        if nz:
            raise SystemExit(f"{n} is nonzero on {nz} rows: the format is not the one amendment 10 registered")
    res = {"amendment": 10, "program": "docs/TORNADO_MODEL_PROGRAM.md", "period": [start, end],
           "n": int(len(y)), "pos": int(y.sum()), "n_days": int(len(np.unique(rows["day"]))),
           "days_dropped_unlabelable": dropped, "spc_failed_convective_days": failed,
           "days_without_probsevere": sorted(d for d, v in built.items() if v == "no ProbSevere"),
           "labels": "SPC preliminary filtered reports",
           "candidates": {}, "paired_vs_a": {}}
    preds = {}
    for name, (fw, ff, nan) in CANDIDATES.items():
        for variant, fname in (("plus_W", fw), ("fallback", ff)):
            payload = json.loads((MODELS / fname).read_text(encoding="utf-8"))
            p = predict(payload, columns(payload, rows, nan))
            preds[(name, variant)] = p
            res["candidates"][f"{name}/{variant}"] = {
                "payload": fname, "model_sha256": lp_digest(payload), "auc": dm.compute_auc(y, p),
                "brier": dm.compute_brier(y, p), "bss": dm.compute_bss(y, p), "mean_forecast": float(p.mean())}
            print(f"{name:8s} {variant:8s} AUC {res['candidates'][f'{name}/{variant}']['auc']:.4f}", flush=True)
    # (b) is (a) by construction (every split on the two inputs has missing type None: NaN reads as 0.0);
    # check it on every row instead of assuming it
    for variant in ("plus_W", "fallback"):
        if not np.array_equal(preds[("a_served", variant)], preds[("b_nan", variant)]):
            raise SystemExit(f"(b) differs from (a) ({variant}): the payload's missing-value rule is not the "
                             "one amendment 10 recorded")
    res["b_equals_a_on_every_row"] = True
    for name in ("b_nan", "d_drop2"):
        for variant in ("plus_W", "fallback"):
            res["paired_vs_a"][f"{name}/{variant}"] = paired(y, rows["day"], preds[("a_served", variant)],
                                                             preds[(name, variant)])
    # the rule (amendment 10), on the +W variants: interval above 0 and Brier point <= 0
    qual = {n: r for n in ("b_nan", "d_drop2")
            for r in [res["paired_vs_a"][f"{n}/plus_W"]]
            if r["delta_auc_ci"][0] > 0 and r["delta_brier"] <= 0}
    winner = max(qual, key=lambda n: qual[n]["delta_auc"]) if qual else "a_served"
    res["decision"] = {"served": winner, "qualified": sorted(qual),
                       "rule": "AUC(X)-AUC(a) 95% day-bootstrap interval above 0 and Brier point <= 0 (+W)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["paired_vs_a"], indent=1), "\nDECISION:", res["decision"], flush=True)
    return 0


def lp_digest(payload: dict) -> str:
    import hashlib
    return hashlib.sha256(lp.canonical_bytes(payload)).hexdigest()


# --------------------------------------------------------------------------- control
#: probability agreement the control demands, in units in the last place. The stored inputs reproduce exactly,
#: but NumPy's float64 exp is not one function across builds and CPUs: re-run on Windows 22 of 1,430 live
#: probabilities differed, on a GitHub runner 40 (different storms), all by <= 2 ulp (amendment 10).
MAX_PROB_ULP = 4


def ulp_distance(a: float, b: float) -> int:
    """Units in the last place between two positive float64 values."""
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def control(replay_dir: Path, first: str = "20261003", last: str = "20261005") -> int:
    """(a) must reproduce the live records: the file each record's storms were read from, parsed by the same
    module, gives the stored 28 P inputs EXACTLY; the served payload on the stored inputs gives the stored
    probability to within MAX_PROB_ULP (the records keep no raw score, and exp is not reproducible across
    builds). Any input difference, or any probability further apart, stops (exit 1)."""
    payloads = {"v3_w": json.loads((MODELS / "tornado_v3_w.json").read_text(encoding="utf-8")),
                "v3": json.loads((MODELS / "tornado_v3.json").read_text(encoding="utf-8"))}
    files = sorted(p for p in replay_dir.glob("to_fcst_*.json") if first <= p.stem[8:16] <= last)
    keys_by_day: dict[str, list[str]] = {}
    steps_cache: dict[str, dict] = {}
    n = n_input_ok = n_prob_ok = 0
    bad: list[str] = []
    ulps: list[int] = []
    for f in files:
        art = json.loads(f.read_text(encoding="utf-8"))
        for s in art.get("storms", []):
            v3 = s.get("v3") or {}
            inp, p60 = v3.get("inputs"), v3.get("probability_60min")
            if not inp or p60 is None:
                continue
            vt = dm.parse_probsevere_valid_time(s.get("valid_time", ""))
            day = vt.strftime("%Y%m%d")
            if day not in keys_by_day:
                keys_by_day[day] = ps._list_s3_files(day)[0]
            key = next((k for k in keys_by_day[day] if ps.key_time(k) == vt.replace(tzinfo=None)
                        or ps.key_time(k) == vt), None)
            if key is None:
                bad.append(f"{f.name} {s.get('storm_id')}: no S3 file at {vt}")
                continue
            if key not in steps_cache:
                got = ps._fetch_steps([key])
                steps_cache[key] = {str(x.get("id")): x for x in (got[0]["storms"] if got else [])}
            storm = steps_cache[key].get(str(s.get("storm_id")))
            n += 1
            if storm is None:
                bad.append(f"{f.name} {s.get('storm_id')}: not in {key}")
                continue
            mine = sf.block_p(storm)
            stored = np.array([np.nan if inp.get(k) is None else inp[k] for k in P_NAMES], np.float32)
            if np.array_equal(mine, stored, equal_nan=True):
                n_input_ok += 1
            else:
                diff = [P_NAMES[i] for i in np.flatnonzero(~((mine == stored) | (np.isnan(mine) & np.isnan(stored))))]
                bad.append(f"{f.name} {s.get('storm_id')}: inputs differ {diff[:5]}")
            payload = payloads.get(v3.get("model"))
            if payload is None:
                bad.append(f"{f.name} {s.get('storm_id')}: served model {v3.get('model')!r} unknown")
                continue
            row = np.array([[np.nan if inp.get(k) is None else float(inp[k]) for k in payload["feature_names"]]])
            u = ulp_distance(float(lp.predict_proba(payload, row)[0]), float(p60))
            ulps.append(u)
            if u <= MAX_PROB_ULP:
                n_prob_ok += 1
            else:
                bad.append(f"{f.name} {s.get('storm_id')}: probability {u} ulp from the record")
    hist = {int(k): int(v) for k, v in zip(*np.unique(ulps, return_counts=True))} if ulps else {}
    print(f"control: {n} live storm observations in {len(files)} records; inputs identical {n_input_ok}, "
          f"probability within {MAX_PROB_ULP} ulp {n_prob_ok} (ulp distances {hist}); failures {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    return 1 if (bad or n == 0) else 0


def main(argv: list[str]) -> int:
    cmd = argv[0]
    if cmd == "build":
        return build(dt.date.fromisoformat(argv[1]), dt.date.fromisoformat(argv[2]))
    if cmd == "decide":
        return decide(*argv[1:3]) if len(argv) > 2 else decide()
    if cmd == "control":
        return control(Path(argv[1]) if len(argv) > 1 else ROOT / "dist" / "data" / "replay")
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
