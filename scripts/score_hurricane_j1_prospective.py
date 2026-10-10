"""The prospective test of J1 against v8.2 in the JTWC basins (docs/HURRICANE_RI_V9_PROGRAM.md, amendments 13b,
14 and 15).

    PYTHONPATH=src python scripts/score_hurricane_j1_prospective.py [--as-of YYYY-MM-DD]

Every live record of a West Pacific, North Indian or Southern Hemisphere storm made after J1 was deployed carries
``ri_j1_shadow`` with v8.2's own number at full precision, both fixed when the forecast was made. Until amendment 15
v8.2 was the published number (``ri_probability``, ``ri_source`` "v8.2"); since then v8.3 is published and v8.2 --
J1's registered comparator, unchanged -- is recorded beside it (``v8_2``). ``comparator_checks`` binds the
comparator to v8.2 either way. The test unit, the record that represents each storm-cycle and the outcome are
amendment 7's,
exactly as for the NHC entrants (``hazardpulse.hurricane.cycle_records``):
- one record per storm-cycle: the first made at or after t + 3 h 30 min (a catch-up record only before t + 12 h);
- scored once t + 24 h has passed and the operational best track (UCAR RAL's open b-deck) has both fixes, against
  the storm's own change from t (30 kt or more is RI).

J1 is the first entrant of the JTWC family, so it carries that family's first error budget, 2.5 %, under v9.1's
rule. At a look (2026-12-01, 2027-12-01), applied once on the full test set and frozen into the output, its claim
is met iff J1 - v8.2 has a 97.5 % storm-bootstrap interval wholly below 0 on the 30/24 log loss OR the Brier
score, AND both point estimates are <= 0. A met claim makes J1 the published JTWC-basin number. Before a look the
output is a running record, and no claim is read from it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hazardpulse.hurricane import cycle_records as cr  # noqa: E402
from hazardpulse.hurricane import ri_j1  # noqa: E402

REPLAY = ROOT / "dist" / "data" / "replay"
OUT = ROOT / "results" / "hurricane_prospective" / "j1_shadow.json"
LOOKS = ("2026-12-01", "2027-12-01")
LEVEL = 0.975
JTWC_PREFIXES = ("WP", "IO", "SH")
# J1's registered comparator (amendments 13b and 14), kept by amendment 15 when v8.3 became the published number
COMPARATOR = "hurricane_ri_v8_2"
COMPARATOR_ARTIFACT = ROOT / "results" / "models" / f"{COMPARATOR}.json"
COMPARATOR_TOL = 1e-12
_CALIBRATION: dict = {}


def comparator_from_input(sh: dict) -> float | None:
    """v8.2's calibrated probability recomputed from J1's own first input (``v82_logit``, v8.2's ensemble as a
    logit) with v8.2's calibration, read from v8.2's artifact. J1's input and its comparator then come from one
    model: had the scorer fed J1 another model's numbers, this disagrees with the recorded comparator (v8.3's
    calibration is not v8.2's)."""
    x = (sh.get("inputs") or {}).get("v82_logit")
    if x is None:
        return None
    if not _CALIBRATION:
        cal = json.loads(COMPARATOR_ARTIFACT.read_text(encoding="utf-8"))["calibration"]
        if cal.get("method") != "logistic_on_logit_newton":
            raise ValueError(f"{COMPARATOR_ARTIFACT.name}: calibration {cal.get('method')!r}")
        _CALIBRATION.update(a=float(cal["a"]), b=float(cal["b"]))
    z = _CALIBRATION["a"] * float(x) + _CALIBRATION["b"]
    return 1.0 / (1.0 + float(np.exp(-np.clip(z, -500, 500))))


def is_jtwc(storm_id: str) -> bool:
    sid = str(storm_id).upper()
    return sid[:2] in JTWC_PREFIXES and sid[2:4].isdigit()


def select_test_records(replay_dir: Path | None = None) -> cr.Selection:
    out = []
    for path in sorted((replay_dir or REPLAY).glob("hu_fcst_*.json")):
        art = json.loads(path.read_text(encoding="utf-8"))
        out.extend(c for c in cr.file_candidates(art) if is_jtwc(c.storm_id))
    return cr.select(out)


def comparator_checks(s: dict, sh: dict) -> bool:
    """Whether a record's comparator is v8.2's own number, at full precision (``v8_2_model_probability``): a
    published 0.0000 on an RI event would otherwise cost v8.2 ~28 nats by rounding alone.

    * Until amendment 15, v8.2 was the published number (``ri_source`` "v8.2"): the full-precision value must round
      to what was published.
    * Since amendment 15, v8.3 is published and J1's comparator stays v8.2: the shadow must name v8.2 as the model
      its comparator came from; the value must round to v8.2's number recorded beside the published one (``v8_2``,
      written by the scorer outside the shadow); and it must be v8.2's calibration of J1's own v8.2 input
      (``comparator_from_input``, to 1e-12). A record whose v8.2 is missing or is another model is not scored --
      the published v8.3 never stands in for J1's comparator."""
    a = sh.get("v8_2_model_probability")
    if a is None:
        return False
    if s.get("ri_source") == "v8.2":
        return s.get("ri_probability") is not None and round(float(a), 4) == round(float(s["ri_probability"]), 4)
    beside = s.get("v8_2") or {}
    again = comparator_from_input(sh)
    return (sh.get("v8_2_model_version") == COMPARATOR and beside.get("model_version") == COMPARATOR
            and beside.get("ri_probability") is not None
            and round(float(a), 4) == round(float(beside["ri_probability"]), 4)
            and again is not None and abs(again - float(a)) <= COMPARATOR_TOL)


def collect(replay_dir: Path | None = None, selection: cr.Selection | None = None) -> list[dict]:
    """Each JTWC storm-cycle's test record that carries J1's shadow and v8.2's number, J1's comparator. A cycle
    whose test record predates J1 (no shadow) is not scored; a later record never stands in for it."""
    sel = select_test_records(replay_dir) if selection is None else selection
    out = []
    for (sid, t), c in sel.chosen.items():
        fid, _kind, s = c.ref
        sh = s.get(ri_j1.SHADOW_KEY) or {}
        if sh.get("status") != "ok" or sh.get("model_probability") is None:
            continue
        if not comparator_checks(s, sh):
            continue
        a = sh["v8_2_model_probability"]
        region = ri_j1.storm_region(str(s.get("basin", "")), s.get("lon"))
        out.append({"storm_id": sid, "cycle": t.strftime("%Y-%m-%dT%H:00:00Z"), "region": region,
                    "in_scope": region in ri_j1.scope(),
                    "p": float(sh["model_probability"]), "a": float(a),
                    "model_version": sh.get("model_version"), "ir_features_read": sh.get("ir_features_read"),
                    "forecast_id": fid, "record_made_at": cr.format_utc(c.made_at),
                    "lag_hours": cr.lag_hours(c.made_at, t), "catch_up": c.catch_up})
    return sorted(out, key=lambda r: (r["cycle"], r["storm_id"]))


def fetch_best_track(storm_id: str) -> dict[dt.datetime, float]:
    import score_hurricane_prospective as shp
    records, _ = shp.fetch_best_track_with_source(storm_id)
    return cr.best_track_intensity(records)


def score(records: list[dict], now: dt.datetime, best_track=None) -> list[dict]:
    best_track = best_track or fetch_best_track
    tracks: dict[str, dict] = {}
    out = []
    for r in records:
        t = dt.datetime.fromisoformat(r["cycle"].replace("Z", ""))
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


def evaluate(scored: list[dict], level: float = LEVEL) -> dict:
    import hurricane_ri_v9 as v9
    if not scored:
        return {"n": 0}
    y = np.array([r["y"] for r in scored])
    p, a = np.array([r["p"] for r in scored]), np.array([r["a"] for r in scored])
    groups = np.array([r["storm_id"] for r in scored])
    res = {"j1": v9.summary(y, p, False), "v8_2": v9.summary(y, a, False), "storms": int(len(set(groups))),
           "storms_with_an_event": int(len({g for g, t in zip(groups, y) if t})), "level": level}
    if not (y.sum() and (1 - y).sum()):
        return res
    d = v9.paired(y, a, False, p, False, groups, level=level)
    res["j1_minus_v8_2"] = d
    res["claim"] = bool((d["d_log_loss_ci"][1] < 0 or d["d_brier_ci"][1] < 0)
                        and d["d_log_loss"] <= 0 and d["d_brier"] <= 0)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args(argv)
    now = (dt.datetime.fromisoformat(a.as_of) if a.as_of else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    recs = collect()
    scored = score(recs, now)
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    looks = dict(prev.get("looks") or {})
    claim_set = [r for r in scored if r["in_scope"]]     # amendment 14: the claim counts only the entrant's scope
    for day in LOOKS:
        if now >= dt.datetime.fromisoformat(day) and day not in looks:
            looks[day] = {"frozen_at": now.strftime("%Y-%m-%dT%H:%MZ"), "scope": list(ri_j1.scope()),
                          **evaluate([r for r in claim_set if r["cycle"] < f"{day}T00:00:00Z"])}
    by_region = {g: evaluate([r for r in scored if r["region"] == g]) for g in ("WP", "NI", "SI", "SP")}
    res = {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendments 13b, 14)", "entrant": "J1",
           "comparator": "v8.2 (J1's registered comparator; the published number until amendment 15)", "rule": "v9.1's rule at 97.5 % (the JTWC family's first budget, 2.5 %)",
           "scope": list(ri_j1.scope()), "scored_as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "look_dates": list(LOOKS),
           "records_with_j1": len(recs), "running": evaluate(claim_set), "running_all_jtwc": evaluate(scored),
           "by_region": by_region, "looks": looks,
           "note": "before a look this is a running record; no claim is read from it. The claim counts only the "
                   "entrant's scope; every JTWC region is reported beside it"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    run = res["running"]
    print(f"J1 prospective: {len(recs)} test records carry J1; {run.get('n', len(scored))} scored "
          f"({run.get('j1', {}).get('events', 0) if isinstance(run.get('j1'), dict) else 0} RI); wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
