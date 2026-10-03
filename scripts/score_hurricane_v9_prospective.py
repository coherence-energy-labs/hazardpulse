"""The v9.1 prospective test (docs/HURRICANE_RI_V9_PROGRAM.md, amendment 1).

    PYTHONPATH=src python scripts/score_hurricane_v9_prospective.py [--as-of YYYY-MM-DD]

Every live NHC forecast record since 2026-10-04 00Z carries ``ri_v9_shadow``: the v9.1 probability
and the DTOPS (else SHIPS-RII) value it is compared with, both fixed when the forecast was made.
A cycle is scored once t + 24 h has passed and the operational best track has both fixes. Before a
look date the output is a running record; at a look (2026-12-01, 2027-12-01) the claim rule is
applied once, with 97.5% storm-bootstrap intervals, and frozen into the output file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import atcf, ri_v9  # noqa: E402

REPLAY = ROOT / "dist" / "data" / "replay"
OUT = ROOT / "results" / "hurricane_prospective" / "v9_shadow.json"
BTK = "https://ftp.nhc.noaa.gov/atcf/btk/b{low}.dat"
LOOKS = ("2026-12-01", "2027-12-01")
LEVEL_AT_LOOK = 0.975
START = dt.datetime.fromisoformat(ri_v9.PROSPECTIVE_START.replace("Z", ""))


def _iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", ""))


def collect(replay_dir: Path = REPLAY) -> list[dict]:
    """One record per (storm, cycle): the first forecast that carried a v9.1 shadow for it."""
    seen: dict[tuple[str, str], dict] = {}
    for path in sorted(replay_dir.glob("hu_fcst_*.json")):
        art = json.loads(path.read_text(encoding="utf-8"))
        for s in art.get("storms") or []:
            sh = s.get("ri_v9_shadow") or {}
            if sh.get("status") != "ok" or sh.get("probability") is None or not sh.get("cycle"):
                continue
            cyc = _iso(sh["cycle"])
            a = sh.get("dtops_pct") if sh.get("dtops_pct") is not None else sh.get("riod_pct")
            if cyc < START or a is None:
                continue
            key = (str(s["storm_id"]).upper(), sh["cycle"])
            seen.setdefault(key, {"storm_id": key[0], "cycle": sh["cycle"], "p": float(sh["probability"]),
                                  "a": float(a) / 100.0, "gate_ok": bool(sh.get("gate_ok")),
                                  "source": sh.get("source"), "model_version": sh.get("model_version"),
                                  "forecast_id": art.get("forecast_id")})
    return sorted(seen.values(), key=lambda r: (r["cycle"], r["storm_id"]))


def fetch_best_track(storm_id: str) -> dict[dt.datetime, float]:
    url = BTK.format(low=storm_id[:2].lower() + storm_id[2:])
    req = urllib.request.Request(url, headers={"User-Agent": "HazardPulse-verification/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict[dt.datetime, float] = {}
    for rec in atcf.parse_atcf_deck(text):
        if rec.model == "BEST" and rec.tau_hours == 0 and rec.cycle not in out and rec.vmax_kt:
            out[rec.cycle] = float(rec.vmax_kt)
    return out


def score(records: list[dict], now: dt.datetime, best_track=fetch_best_track) -> list[dict]:
    tracks: dict[str, dict] = {}
    out = []
    for r in records:
        t = _iso(r["cycle"])
        if t + dt.timedelta(hours=24) > now:
            continue
        if r["storm_id"] not in tracks:
            try:
                tracks[r["storm_id"]] = best_track(r["storm_id"])
            except Exception:  # noqa: BLE001 -- not scorable this run; tried again next run
                tracks[r["storm_id"]] = {}
        bt = tracks[r["storm_id"]]
        v0, v24 = bt.get(t), bt.get(t + dt.timedelta(hours=24))
        if v0 is None or v24 is None:
            continue
        out.append({**r, "v_t": v0, "v_t24": v24, "y": int(v24 - v0 >= 30.0)})
    return out


def evaluate(scored: list[dict], level: float) -> dict:
    import hurricane_ri_v9 as v9
    if not scored:
        return {"n": 0}
    y = np.array([r["y"] for r in scored])
    p, a = np.array([r["p"] for r in scored]), np.array([r["a"] for r in scored])
    groups = np.array([r["storm_id"] for r in scored])
    res = {"v9_1": v9.summary(y, p, False), "dtops": v9.summary(y, a, True),
           "storms": int(len(set(groups))), "storms_with_an_event": int(len({g for g, t in zip(groups, y) if t})),
           "gated_cycles": int(sum(not r["gate_ok"] for r in scored))}
    if y.sum() and (1 - y).sum():
        d = v9.paired(y, a, True, p, False, groups, level=level)
        res["vs_dtops"] = d
        res["claim"] = bool((d["d_log_loss_ci"][1] < 0 or d["d_brier_ci"][1] < 0)
                            and d["d_log_loss"] <= 0 and d["d_brier"] <= 0)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: now, UTC)")
    args = ap.parse_args(argv)
    now = (dt.datetime.fromisoformat(args.as_of) if args.as_of
           else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    records = collect()
    scored = score(records, now)
    looks = dict(prev.get("looks") or {})
    for day in LOOKS:
        if now >= dt.datetime.fromisoformat(day) and day not in looks:
            frozen = [r for r in scored if _iso(r["cycle"]) + dt.timedelta(hours=24) <= dt.datetime.fromisoformat(day)]
            looks[day] = {"as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "level": LEVEL_AT_LOOK,
                          **evaluate(frozen, LEVEL_AT_LOOK)}
    out = {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 1)", "start": ri_v9.PROSPECTIVE_START,
           "scored_as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "shadow_records": len(records),
           "matured_and_scored": len(scored), "running_95": evaluate(scored, 0.95),
           "looks": looks, "look_dates": list(LOOKS),
           "note": "running numbers are descriptive; the claim is made only at a look date"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    run = out["running_95"]
    print(f"v9.1 prospective: {len(records)} shadow records, {len(scored)} matured and scored"
          + (f"; running LL v9.1 {run['v9_1']['log_loss']:.4f} vs DTOPS {run['dtops']['log_loss']:.4f}"
             if run.get("n") != 0 else ""))
    for day, lk in looks.items():
        print(f"  look {day}: claim={lk.get('claim')} (n={lk.get('v9_1', {}).get('n', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
