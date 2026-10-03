"""The prospective test of our RI models against NOAA DTOPS (docs/HURRICANE_RI_V9_PROGRAM.md,
amendments 1 and 2).

    PYTHONPATH=src python scripts/score_hurricane_v9_prospective.py [--as-of YYYY-MM-DD]

Every live NHC forecast record since 2026-10-04 00Z carries the shadows -- ``ri_v9_shadow``
(v9.1: P(RI 30 kt / 24 h)) and ``ri_v10_shadow`` (v10.1: the 24-h exceedance curve) -- each with the
NOAA values it is compared with, all fixed when the forecast was made. A cycle is scored once
t + 24 h has passed and the operational best track has both fixes. Before a look date the output is
a running record; at a look (2026-12-01, 2027-12-01) each entrant's claim rule is applied once and
frozen into the output file:

* v9.1  -- 30/24 log loss or Brier vs DTOPS, 97.5% interval wholly below 0, both points <= 0;
* v10.1 -- the sum of Brier scores over the 25/30/35/40-kt 24-h thresholds vs NOAA's own published
  values, 98.75% interval wholly below 0, and its 30/24 log loss point <= DTOPS's;
* v10.2 (amendment 4, the amendment-3 challenger, ``ri_v10_2_shadow``) -- v10.1's rule at 99.375%;
* v10.3 (amendment 6, the amendment-5 IR challenger, ``ri_v10_3_shadow``) -- v10.1's rule at 99.6875%.

Each entrant's error budget is half the previous one's (2.5%, 1.25%, 0.625%, ...), so however many
challengers enter, the family's total stays below 5%.
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
MULTI = (25, 30, 35, 40)
ENTRANTS = {"v9_1": {"key": "ri_v9_shadow", "level": 0.975, "metric": "30kt"},
            "v10_1": {"key": "ri_v10_shadow", "level": 0.9875, "metric": "multi"},
            "v10_2": {"key": "ri_v10_2_shadow", "level": 0.99375, "metric": "multi"},
            "v10_3": {"key": "ri_v10_3_shadow", "level": 0.996875, "metric": "multi"}}


def _iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", ""))


def collect(replay_dir: Path = REPLAY, key: str = "ri_v9_shadow") -> list[dict]:
    """One record per (storm, cycle): the first forecast that carried this shadow for it."""
    seen: dict[tuple[str, str], dict] = {}
    for path in sorted(replay_dir.glob("hu_fcst_*.json")):
        art = json.loads(path.read_text(encoding="utf-8"))
        for s in art.get("storms") or []:
            sh = s.get(key) or {}
            if sh.get("status") != "ok" or sh.get("probability") is None or not sh.get("cycle"):
                continue
            cyc = _iso(sh["cycle"])
            a = sh.get("dtops_pct") if sh.get("dtops_pct") is not None else sh.get("riod_pct")
            if cyc < START or a is None:
                continue
            rec = {"storm_id": str(s["storm_id"]).upper(), "cycle": sh["cycle"], "p": float(sh["probability"]),
                   "a": float(a) / 100.0, "gate_ok": bool(sh.get("gate_ok")), "source": sh.get("source"),
                   "model_version": sh.get("model_version"), "forecast_id": art.get("forecast_id")}
            if sh.get("probabilities") is not None:
                rec["p_k"] = {str(k): sh["probabilities"].get(str(k)) for k in MULTI}
                rec["a_k"] = {str(k): (sh.get("noaa_24h") or {}).get(str(k)) for k in MULTI}
            seen.setdefault((rec["storm_id"], sh["cycle"]), rec)
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


def score(records: list[dict], now: dt.datetime, best_track=fetch_best_track, tracks=None) -> list[dict]:
    tracks = {} if tracks is None else tracks
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
        out.append({**r, "v_t": v0, "v_t24": v24, "dv": v24 - v0, "y": int(v24 - v0 >= 30.0)})
    return out


def _multi_brier(scored, which):
    tot = np.zeros(len(scored))
    for i, r in enumerate(scored):
        for k in MULTI:
            p = ((r.get("p_k") if which == "ours" else r.get("a_k")) or {}).get(str(k))
            tot[i] += ((0.0 if p is None else float(p)) - float(r["dv"] >= k)) ** 2
    return tot


def evaluate(scored: list[dict], level: float, metric: str = "30kt") -> dict:
    import hurricane_ri_v9 as v9
    if not scored:
        return {"n": 0}
    y = np.array([r["y"] for r in scored])
    p, a = np.array([r["p"] for r in scored]), np.array([r["a"] for r in scored])
    groups = np.array([r["storm_id"] for r in scored])
    res = {"ours": v9.summary(y, p, False), "dtops": v9.summary(y, a, True),
           "storms": int(len(set(groups))), "storms_with_an_event": int(len({g for g, t in zip(groups, y) if t})),
           "gated_cycles": int(sum(not r["gate_ok"] for r in scored)), "metric": metric, "level": level}
    if not (y.sum() and (1 - y).sum()):
        return res
    d30 = v9.paired(y, a, True, p, False, groups, level=level)
    res["vs_dtops_30kt"] = d30
    if metric == "30kt":
        res["claim"] = bool((d30["d_log_loss_ci"][1] < 0 or d30["d_brier_ci"][1] < 0)
                            and d30["d_log_loss"] <= 0 and d30["d_brier"] <= 0)
        return res
    ours, base = _multi_brier(scored, "ours"), _multi_brier(scored, "noaa")
    rng = np.random.default_rng(v9.SEED)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draws = []
    for _ in range(v9.REPS):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        draws.append(ours[idx].mean() - base[idx].mean())
    q = (1 - level) / 2
    res["multi_threshold_brier"] = {"ours": float(ours.mean()), "noaa": float(base.mean()),
                                    "d": float(ours.mean() - base.mean()),
                                    "d_ci": [float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q))]}
    res["claim"] = bool(res["multi_threshold_brier"]["d_ci"][1] < 0 and d30["d_log_loss"] <= 0)
    return res


CHALLENGES = (("v10_2", "v10_1"), ("v10_3", "v10_1"), ("v10_3", "v10_2"))   # descriptive (amendments 4, 6)


def versus(challenger: list[dict], champion: list[dict]) -> dict:
    """Challenger minus champion on the cycles both scored: the four-threshold Brier sum per cycle,
    storm-bootstrap 95% interval. Descriptive: it decides which one the site shows, never a claim."""
    import hurricane_ri_v9 as v9
    champ = {(r["storm_id"], r["cycle"]): r for r in champion}
    pairs = [(r, champ[(r["storm_id"], r["cycle"])]) for r in challenger if (r["storm_id"], r["cycle"]) in champ]
    if not pairs:
        return {"n": 0}
    b = _multi_brier([p for p, _ in pairs], "ours") - _multi_brier([c for _, c in pairs], "ours")
    groups = np.array([p["storm_id"] for p, _ in pairs])
    rng = np.random.default_rng(v9.SEED)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draws = [b[np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])].mean()
             for _ in range(v9.REPS)]
    return {"n": len(pairs), "storms": int(len(uniq)), "d_brier4": float(b.mean()),
            "d_brier4_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: now, UTC)")
    args = ap.parse_args(argv)
    now = (dt.datetime.fromisoformat(args.as_of) if args.as_of
           else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    tracks: dict = {}
    out = {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendments 1, 2 and 4)", "start": ri_v9.PROSPECTIVE_START,
           "scored_as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "look_dates": list(LOOKS),
           "note": "running numbers are descriptive; a claim is made only at a look date", "entrants": {}}
    by_entrant: dict[str, list[dict]] = {}
    for name, spec in ENTRANTS.items():
        records = collect(key=spec["key"])
        scored = score(records, now, tracks=tracks)
        by_entrant[name] = scored
        looks = dict(((prev.get("entrants") or {}).get(name) or {}).get("looks") or {})
        for day in LOOKS:
            cut = dt.datetime.fromisoformat(day)
            if now >= cut and day not in looks:
                frozen = [r for r in scored if _iso(r["cycle"]) + dt.timedelta(hours=24) <= cut]
                looks[day] = {"as_of": now.strftime("%Y-%m-%dT%H:%MZ"), **evaluate(frozen, spec["level"], spec["metric"])}
        out["entrants"][name] = {"shadow_records": len(records), "matured_and_scored": len(scored),
                                 "running_95": evaluate(scored, 0.95, spec["metric"]), "looks": looks,
                                 "claim_rule": spec}
        run = out["entrants"][name]["running_95"]
        print(f"{name}: {len(records)} shadow records, {len(scored)} matured and scored"
              + (f"; running 30/24 LL ours {run['ours']['log_loss']:.4f} vs DTOPS {run['dtops']['log_loss']:.4f}"
                 if run.get("n") != 0 else ""))
        for day, lk in looks.items():
            print(f"  look {day}: claim={lk.get('claim')} (n={lk.get('ours', {}).get('n', 0)})")
    out["challenger_vs_champion"] = {f"{a}_vs_{b}": versus(by_entrant[a], by_entrant[b]) for a, b in CHALLENGES}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
