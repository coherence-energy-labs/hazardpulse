#!/usr/bin/env python3
"""The prospective test of tornado program amendment 11 (T2b; docs/TORNADO_MODEL_PROGRAM.md): each served payload's
calibration against its recalibration for NOAA's new ProbSevere format, on the live v3 record.

The record and its labels are the prospective tornado verifier's (``score_tornado_prospective.py``): it labels every
matured v3 record once per run (``storm_features.labels`` on the storm's archived track, SPC's preliminary reports,
from the storm's valid time) and hands the storm forecasts that carry this artifact's ``t2b_shadow`` to ``update``
here, which writes ``results/tornado_prospective/t2b_shadow.json``:

* per candidate (``p60_w`` primary, ``p60``, ``p30``, ``p90``), one forecast per storm per valid hour (the first
  issued), the served probability against the recalibrated one, both as recorded and both from the same margin;
* a running record: no claim is read from it before a look;
* the looks, each applied once by the first run at least 3 days after its date (waiting, up to 14 days, while a
  record issued before it is unscorable for a transient reason) and FROZEN: never recomputed.

The rule (per candidate, at a decision look): the recalibration replaces the served calibration iff its Brier score
AND its log loss are both below the served's AND the 95 % percentile interval of Brier(recal) - Brier(served),
2,000 draws resampling whole convective days (12Z-12Z), seed 42, lies wholly below 0. No claim with fewer than 30
tornadic storm forecasts or 10 convective days holding one. Ranking is unchanged by construction (a > 0).

    python scripts/score_tornado_t2b_prospective.py --check-frozen     # CI: no frozen look changed (vs git HEAD)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit_20261001"))

from hazardpulse.tornado import t2b_shadow as tb  # noqa: E402

OUT = ROOT / "results" / "tornado_prospective" / "t2b_shadow.json"
OUT_REL = "results/tornado_prospective/t2b_shadow.json"
LOOKS = ("2027-07-01", "2028-07-01")          # decision looks
DESCRIPTIVE_READS = ("2027-01-01",)            # frozen, decides nothing (the cool season alone)
FREEZE_AFTER = dt.timedelta(days=3)
WAIT_AT_MOST = dt.timedelta(days=14)
N_BOOT = 2000
SEED = 42
LEVEL = 0.95
MIN_EVENTS = 30
MIN_EVENT_DAYS = 10
RULE = ("replace iff Brier(recal) < Brier(served) AND log loss(recal) < log loss(served) AND the 95% interval of "
        "Brier(recal) - Brier(served) (percentile, 2,000 draws, whole convective days 12Z-12Z, seed 42) lies wholly "
        "below 0; no claim with fewer than 30 tornadic storm forecasts or 10 convective days holding one")


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def convective_day(t: dt.datetime) -> str:
    """The SPC convective day (12Z to 12Z) an instant falls in, by its starting date."""
    return (t - dt.timedelta(hours=12)).date().isoformat()


def carries(artifact: dict, art_sha: str) -> bool:
    """The forecast record carries shadows of THIS artifact (a record of any other digest is not counted)."""
    desc = artifact.get(tb.SHADOW_KEY) or {}
    return desc.get("status") == "ok" and desc.get("artifact_sha256") == art_sha


def rows_from_forecast(artifact: dict, labels: dict, art_sha: str) -> list[dict]:
    """Every (storm forecast, candidate) of one labelled forecast record that carries this artifact's shadow."""
    if not carries(artifact, art_sha):
        return []
    y_by_label = {"storm_30": labels["y_storm_30"], "storm_60": labels["y_true"], "storm_90": labels["y_storm_90"]}
    issued = _iso(labels["issued_at"])
    out = []
    for i, s in enumerate(artifact.get("storms") or []):
        sh = s.get(tb.SHADOW_KEY)
        if not sh:
            continue
        vt = labels["storm_times"][i]
        for k in tb.CANDIDATES:
            if k not in sh:
                continue
            y = float(y_by_label[tb.LABELS[k]][i])
            if not np.isfinite(y):
                continue                   # not labelled at this horizon (not a v3 storm): never a negative
            out.append({"candidate": k, "storm_id": str(s.get("storm_id", s.get("id"))), "valid_time": _iso(vt),
                        "issued_at": issued, "cday": convective_day(vt), "y": int(y),
                        "served": float(sh[k]["served"]), "recal": float(sh[k]["recal"])})
    return out


def first_per_storm_hour(rows: list[dict]) -> list[dict]:
    """One forecast per (candidate, storm, valid hour): the first issued -- the verifier's pooling rule."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r["issued_at"], r["candidate"], r["storm_id"], r["valid_time"])):
        key = (r["candidate"], r["storm_id"], r["valid_time"][:13])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def _summary(y: np.ndarray, p: np.ndarray) -> dict:
    from hazardpulse.tornado.definitive_model import compute_auc
    auc = compute_auc(y, p) if 0 < y.sum() < len(y) else float("nan")
    return {"brier": float(np.mean((p - y) ** 2)), "log_loss": log_loss(y, p), "mean_forecast": float(p.mean()),
            "auc": None if not np.isfinite(auc) else float(auc)}


def brier_interval(y: np.ndarray, served: np.ndarray, recal: np.ndarray, cday: np.ndarray) -> list[float]:
    """95 % percentile interval of Brier(recal) - Brier(served), whole convective days resampled (the same draws
    for both): ``tornado_lab``'s day bootstrap with the convective day as the block."""
    import tornado_lab as lab
    _, idx = np.unique(cday, return_inverse=True)
    D = int(idx.max()) + 1
    M = lab.day_bootstrap_counts(D, N_BOOT, SEED)
    sa = np.bincount(idx, weights=(served - y) ** 2, minlength=D)
    sb = np.bincount(idx, weights=(recal - y) ** 2, minlength=D)
    n_d = np.bincount(idx, minlength=D).astype(np.float64)
    d = (M @ sb - M @ sa) / (M @ n_d)
    lo, hi = (1.0 - LEVEL) / 2.0 * 100.0, (1.0 + LEVEL) / 2.0 * 100.0
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))]


def evaluate(rows: list[dict], decide: bool) -> dict:
    """One candidate's record. ``decide``: apply the rule (a decision look); otherwise no claim is made."""
    rows = first_per_storm_hour(rows)
    if not rows:
        return {"n": 0, "claim": None if decide else "none before a look", "reason": "no data"}
    y = np.array([r["y"] for r in rows], np.float64)
    s = np.array([r["served"] for r in rows], np.float64)
    q = np.array([r["recal"] for r in rows], np.float64)
    cday = np.array([r["cday"] for r in rows])
    events = int(y.sum())
    event_days = int(len(set(cday[y == 1].tolist())))
    res = {"n": int(len(y)), "events": events, "convective_days": int(len(set(cday.tolist()))),
           "event_convective_days": event_days, "base_rate": float(y.mean()),
           "first_valid": rows[0]["valid_time"] if rows else None,
           "served": _summary(y, s), "recalibrated": _summary(y, q),
           "d_brier": float(np.mean((q - y) ** 2) - np.mean((s - y) ** 2)),
           "d_log_loss": log_loss(y, q) - log_loss(y, s),
           "ranking": "unchanged by construction (a > 0: the same order of the same margin); reported, not decided on"}
    enough = events >= MIN_EVENTS and event_days >= MIN_EVENT_DAYS
    if enough:
        res["d_brier_ci95"] = brier_interval(y, s, q, cday)
    if not decide:
        res["claim"] = "none before a look"
        return res
    if not enough:
        res["claim"] = None
        res["reason"] = (f"{events} tornadic storm forecasts on {event_days} convective days: below the registered "
                         f"{MIN_EVENTS} / {MIN_EVENT_DAYS}; the served calibration stays")
        res["replace"] = False
        return res
    better_brier = res["recalibrated"]["brier"] < res["served"]["brier"]
    better_ll = res["recalibrated"]["log_loss"] < res["served"]["log_loss"]
    res["replace"] = bool(better_brier and better_ll and res["d_brier_ci95"][1] < 0.0)
    res["claim"] = "replace the served calibration" if res["replace"] else "the served calibration stays"
    res["rule_parts"] = {"brier_below": bool(better_brier), "log_loss_below": bool(better_ll),
                         "interval_below_0": bool(res["d_brier_ci95"][1] < 0.0)}
    return res


def rows_digest(rows: list[dict]) -> str:
    keys = ("candidate", "storm_id", "valid_time", "issued_at", "y", "served", "recal")
    canon = [[r[k] for k in keys] for r in first_per_storm_hour(rows)]
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode("utf-8")).hexdigest()


def update(rows: list[dict], now: dt.datetime, pending_issued: list[str], art_sha: str,
           out_path: Path = OUT) -> dict:
    """Write the running record and freeze every look that is due. ``pending_issued``: issue times of the records
    carrying this artifact's shadow that could not be labelled this run (retried next run)."""
    prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    looks = dict(prev.get("looks") or {}) if prev.get("artifact_sha256") in (None, art_sha) else {}
    superseded = list(prev.get("superseded") or [])
    if prev.get("artifact_sha256") not in (None, art_sha):
        # another artifact's record (the registered one is fixed at its commit): kept, never mixed into this one
        superseded.append({k: prev.get(k) for k in ("artifact_sha256", "looks", "first_record")})
    pending: dict[str, str] = {}
    schedule = sorted([(d, "descriptive") for d in DESCRIPTIVE_READS] + [(d, "decision") for d in LOOKS])
    for day, kind in schedule:
        if day in looks:
            continue                       # frozen: never recomputed
        due = dt.datetime.fromisoformat(day)
        if now < due + FREEZE_AFTER:
            continue
        cutoff = f"{day}T00:00:00Z"
        waiting = [t for t in pending_issued if t < cutoff]
        if waiting and now < due + WAIT_AT_MOST:
            pending[day] = (f"{len(waiting)} records issued before {day} could not be labelled this run; retried "
                            f"until {(due + WAIT_AT_MOST).date()}")
            continue
        sel = [r for r in rows if r["issued_at"] < cutoff]
        entry = {"kind": kind, "frozen_at": _iso(now), "records_before": cutoff,
                 "records_unscorable_excluded": len(waiting), "rows_sha256": rows_digest(sel), "candidates": {}}
        for k in tb.CANDIDATES:
            earlier = [d for d, kd in schedule if kd == "decision" and d < day]
            if kind == "decision" and any(((looks.get(d) or {}).get("candidates") or {}).get(k, {}).get("replace")
                                          for d in earlier):
                entry["candidates"][k] = {"skipped": "replaced at an earlier look"}
                continue
            entry["candidates"][k] = evaluate([r for r in sel if r["candidate"] == k], decide=(kind == "decision"))
        looks[day] = entry
    first = min((r["issued_at"] for r in rows), default=None)
    res = {"program": "docs/TORNADO_MODEL_PROGRAM.md (amendment 11, T2b)", "artifact": f"results/models/{tb.ARTIFACT_FILE}",
           "artifact_sha256": art_sha, "scored_as_of": _iso(now), "first_record": first,
           "candidates": {k: {"payload": tb.CANDIDATES[k], "label": tb.LABELS[k]} for k in tb.CANDIDATES},
           "primary": "p60_w", "rule": RULE, "look_dates": list(LOOKS), "descriptive_reads": list(DESCRIPTIVE_READS),
           "records_unscorable_this_run": len(pending_issued),
           "running": {k: evaluate([r for r in rows if r["candidate"] == k], decide=False) for k in tb.CANDIDATES},
           "looks": looks, "looks_pending": pending,
           "note": "before a look this is a running record: no claim is read from it. A look is frozen once written."}
    if superseded:
        res["superseded"] = superseded
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    run = res["running"]["p60_w"]
    print(f"  T2b: {sum(r.get('n', 0) for r in res['running'].values())} storm forecasts with a shadow "
          f"(p60_w: {run.get('n', 0)}, {run.get('events', 0)} tornadic); looks frozen {sorted(looks)}; wrote {out_path}")
    return res


def check_frozen(committed: dict, new: dict) -> list[str]:
    """Every look in the committed record is in the new one, unchanged."""
    problems = []
    for day, look in (committed.get("looks") or {}).items():
        if committed.get("artifact_sha256") != new.get("artifact_sha256"):
            if not any(s.get("artifact_sha256") == committed.get("artifact_sha256") and
                       (s.get("looks") or {}).get(day) == look for s in new.get("superseded") or []):
                problems.append(f"look {day} of artifact {str(committed.get('artifact_sha256'))[:12]} was dropped")
            continue
        if (new.get("looks") or {}).get(day) != look:
            problems.append(f"look {day} changed after it was frozen")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-frozen", action="store_true",
                    help="exit 1 if a look frozen in the committed record (git HEAD) changed in the working tree")
    a = ap.parse_args(argv)
    if not a.check_frozen:
        raise SystemExit("the record is written by scripts/score_tornado_prospective.py (it labels the v3 record "
                         "once per run); this script's command line only checks it: --check-frozen")
    if not OUT.exists():
        print(f"T2b: {OUT_REL} not written (no artifact, or no run yet): nothing frozen to check")
        return 0
    new = json.loads(OUT.read_text(encoding="utf-8"))
    got = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{OUT_REL}"], capture_output=True, text=True)
    if got.returncode != 0:
        print(f"T2b: {OUT_REL} is not committed yet: nothing frozen to check")
        return 0
    problems = check_frozen(json.loads(got.stdout), new)
    for p in problems:
        print("T2b FROZEN LOOK CHANGED:", p)
    print(f"T2b: frozen looks {sorted(new.get('looks') or {})}; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
