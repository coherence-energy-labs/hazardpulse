"""The prospective test of our RI models against NOAA DTOPS (docs/HURRICANE_RI_V9_PROGRAM.md,
amendments 1 and 2).

    PYTHONPATH=src python scripts/score_hurricane_v9_prospective.py [--as-of YYYY-MM-DD]

Every live NHC forecast record since 2026-10-04 00Z carries the shadows -- ``ri_v9_shadow``
(v9.1: P(RI 30 kt / 24 h)) and ``ri_v10_shadow`` (v10.1: the 24-h exceedance curve) -- each with the
NOAA values it is compared with, all fixed when the forecast was made.

Which record is a cycle's test record is amendment 7 (``hazardpulse.hurricane.cycle_records``): the
unit is the storm-cycle, its record the FIRST made at or after t + 3 h 30 min judged by the record's own
time (earlier records read preliminary inputs; later ones are duplicates), a catch-up record
(``catch_up: true``, written by the live scorer for a cycle it missed) only if made before t + 12 h.
A cycle is scored once t + 24 h has passed and the operational best track has both fixes, against the
change from the storm's own t (rule 4). Every result is also reported without catch-up and rebuilt
records (rule 5). Before a look date the output is a running record; at a look (2026-12-01,
2027-12-01) each entrant's claim rule is applied once, on the full test set, and frozen into the
output file:

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
from hazardpulse.hurricane import cycle_records as cr  # noqa: E402

REPLAY = ROOT / "dist" / "data" / "replay"
# amendment 7 rule 3: missed cycles rebuilt by a procedure that reproduced the live records exactly
# (scripts/rebuild_hurricane_cycles.py)
REBUILT = ROOT / "results" / "hurricane_prospective" / "rebuilt"
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


_DEFAULT = object()


def candidates(replay_dir: Path | None = None, rebuilt_dir=_DEFAULT) -> list[cr.Candidate]:
    """Every forecast record of a test-population storm-cycle (an NHC-basin numbered storm, t >= the
    start) that carries the shadows: the published storms of each forecast file and its catch-up
    records (amendment 7 rule 2), each with the time its file was made (cycle_records.file_candidates,
    the reader the live scorer's catch-up uses too); and the rule 3 records of ``rebuilt_dir`` (by
    default REBUILT when the default replay directory is read, none for another one)."""
    if rebuilt_dir is _DEFAULT:
        rebuilt_dir = REBUILT if replay_dir is None else None
    paths = sorted((replay_dir or REPLAY).glob("hu_fcst_*.json"))
    if rebuilt_dir is not None:
        paths += sorted(Path(rebuilt_dir).glob("hu_rebuilt_*.json"))
    out = []
    for path in paths:
        art = json.loads(path.read_text(encoding="utf-8"))
        out.extend(c for c in cr.file_candidates(art) if cr.is_nhc_numbered(c.storm_id) and c.cycle >= START)
    return out


def select_test_records(replay_dir: Path | None = None, rebuilt_dir=_DEFAULT) -> cr.Selection:
    """Amendment 7 rules 1-3: at most one test record per storm-cycle -- the first made at or after
    t + 3 h 30 min (a catch-up record only before t + 12 h; a rule 3 rebuild where no run recorded the
    cycle). The same record serves every entrant."""
    return cr.select(candidates(replay_dir, rebuilt_dir))


def collect(replay_dir: Path | None = None, key: str = "ri_v9_shadow", selection: cr.Selection | None = None) -> list[dict]:
    """Each storm-cycle's test record (``select_test_records``), read for one entrant's shadow. A cycle
    whose test record lacks this shadow, or carries no NOAA value to compare with, is not scored for it:
    a later record of the same cycle never stands in (it is a duplicate)."""
    sel = select_test_records(replay_dir) if selection is None else selection
    out = []
    for (sid, t), c in sel.chosen.items():
        fid, kind, s = c.ref
        sh = s.get(key) or {}
        if sh.get("status") != "ok" or sh.get("probability") is None:
            continue
        a = sh.get("dtops_pct") if sh.get("dtops_pct") is not None else sh.get("riod_pct")
        if a is None:
            continue
        cycle = t.strftime("%Y-%m-%dT%H:00:00Z")
        rec = {"storm_id": sid, "cycle": cycle, "p": float(sh["probability"]),
               "a": float(a) / 100.0, "gate_ok": bool(sh.get("gate_ok")), "source": sh.get("source"),
               "model_version": sh.get("model_version"), "forecast_id": fid,
               "record_made_at": cr.format_utc(c.made_at), "lag_hours": cr.lag_hours(c.made_at, t),
               "catch_up": c.catch_up, "rebuilt": c.rebuilt}
        if sh.get("probabilities") is not None:
            rec["p_k"] = {str(k): sh["probabilities"].get(str(k)) for k in MULTI}
            rec["a_k"] = {str(k): (sh.get("noaa_24h") or {}).get(str(k)) for k in MULTI}
        out.append(rec)
    return sorted(out, key=lambda r: (r["cycle"], r["storm_id"]))


def without_catch_up(records: list[dict]) -> list[dict]:
    """Rule 5: the same test set without catch-up and rebuilt records, so the amendment's effect shows."""
    return [r for r in records if not r.get("catch_up") and not r.get("rebuilt")]


def fetch_best_track(storm_id: str) -> dict[dt.datetime, float]:
    url = BTK.format(low=storm_id[:2].lower() + storm_id[2:])
    req = urllib.request.Request(url, headers={"User-Agent": "HazardPulse-verification/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    return cr.best_track_intensity(atcf.parse_atcf_deck(text))


def score(records: list[dict], now: dt.datetime, best_track=None, tracks=None) -> list[dict]:
    """Rule 4: the best-track change from the storm's own t to t + 24 h, once both fixes exist."""
    best_track = best_track or fetch_best_track
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
        o = cr.outcome(tracks[r["storm_id"]], t)
        if not o.decided:
            continue
        out.append({**r, "v_t": o.v_t, "v_t24": o.v_t24, "dv": o.dv, "y": int(o.ri)})
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
    selection = select_test_records()
    chosen = list(selection.chosen.values())
    out = {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendments 1, 2, 4, 6 and 7)", "start": ri_v9.PROSPECTIVE_START,
           "scored_as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "look_dates": list(LOOKS),
           "note": "running numbers are descriptive; a claim is made only at a look date, on the full test set "
                   "(amendment 7 rules 1-3); the same numbers without catch-up and rebuilt records are reported "
                   "beside it (rule 5)",
           "test_set": {"unit": "storm-cycle: the first record made at or after t + 3 h 30 min (rule 1); a "
                                "catch-up record only if made before t + 12 h (rule 2)",
                        "test_records": len(chosen),
                        "catch_up_records": sum(c.catch_up for c in chosen),
                        "rebuilt_records": sum(c.rebuilt for c in chosen),
                        "records_set_aside": selection.counts(),
                        "cycles_with_only_records_set_aside": [f"{s} {cr.format_utc(t)}" for s, t
                                                               in selection.cycles_without_a_record()]},
           "entrants": {}}
    by_entrant: dict[str, list[dict]] = {}
    for name, spec in ENTRANTS.items():
        records = collect(key=spec["key"], selection=selection)
        scored = score(records, now, tracks=tracks)
        by_entrant[name] = scored
        prev_entrant = (prev.get("entrants") or {}).get(name) or {}
        looks = dict(prev_entrant.get("looks") or {})
        for day in LOOKS:
            cut = dt.datetime.fromisoformat(day)
            if now >= cut and day not in looks:
                frozen = [r for r in scored if _iso(r["cycle"]) + dt.timedelta(hours=24) <= cut]
                looks[day] = {"as_of": now.strftime("%Y-%m-%dT%H:%MZ"), **evaluate(frozen, spec["level"], spec["metric"]),
                              "without_catch_up_or_rebuilt": evaluate(without_catch_up(frozen), spec["level"],
                                                                      spec["metric"])}
        out["entrants"][name] = {"shadow_records": len(records), "matured_and_scored": len(scored),
                                 "catch_up_or_rebuilt_scored": len(scored) - len(without_catch_up(scored)),
                                 "running_95": evaluate(scored, 0.95, spec["metric"]),
                                 "running_95_without_catch_up_or_rebuilt": evaluate(without_catch_up(scored), 0.95,
                                                                                    spec["metric"]),
                                 "looks": looks, "claim_rule": spec}
        run = out["entrants"][name]["running_95"]
        print(f"{name}: {len(records)} test records, {len(scored)} matured and scored"
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
