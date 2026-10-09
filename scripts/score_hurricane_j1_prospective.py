"""The prospective test of J1 against the published v8.2 in the JTWC basins (docs/HURRICANE_RI_V9_PROGRAM.md,
amendment 13b).

    PYTHONPATH=src python scripts/score_hurricane_j1_prospective.py [--as-of YYYY-MM-DD]

Every live record of a West Pacific, North Indian or Southern Hemisphere storm made after J1 was deployed carries
``ri_j1_shadow`` beside the published v8.2 number (``ri_probability``, ``ri_source`` "v8.2"), both fixed when the
forecast was made. The test unit, the record that represents each storm-cycle and the outcome are amendment 7's,
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


def is_jtwc(storm_id: str) -> bool:
    sid = str(storm_id).upper()
    return sid[:2] in JTWC_PREFIXES and sid[2:4].isdigit()


def select_test_records(replay_dir: Path | None = None) -> cr.Selection:
    out = []
    for path in sorted((replay_dir or REPLAY).glob("hu_fcst_*.json")):
        art = json.loads(path.read_text(encoding="utf-8"))
        out.extend(c for c in cr.file_candidates(art) if is_jtwc(c.storm_id))
    return cr.select(out)


def collect(replay_dir: Path | None = None, selection: cr.Selection | None = None) -> list[dict]:
    """Each JTWC storm-cycle's test record that carries J1's shadow and v8.2's published number. A cycle whose
    test record predates J1 (no shadow) is not scored; a later record never stands in for it."""
    sel = select_test_records(replay_dir) if selection is None else selection
    out = []
    for (sid, t), c in sel.chosen.items():
        fid, _kind, s = c.ref
        sh = s.get(ri_j1.SHADOW_KEY) or {}
        if sh.get("status") != "ok" or sh.get("model_probability") is None:
            continue
        if s.get("ri_source") != "v8.2" or s.get("ri_probability") is None:
            continue
        # v8.2 at full precision (the published number before display rounding): a published 0.0000 on an RI
        # event would otherwise cost v8.2 ~28 nats by rounding alone. It must round to what was published.
        a = sh.get("v8_2_model_probability")
        if a is None or round(float(a), 4) != round(float(s["ri_probability"]), 4):
            continue
        out.append({"storm_id": sid, "cycle": t.strftime("%Y-%m-%dT%H:00:00Z"),
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
    for day in LOOKS:
        if now >= dt.datetime.fromisoformat(day) and day not in looks:
            looks[day] = {"frozen_at": now.strftime("%Y-%m-%dT%H:%MZ"),
                          **evaluate([r for r in scored if r["cycle"] < f"{day}T00:00:00Z"])}
    res = {"program": "docs/HURRICANE_RI_V9_PROGRAM.md (amendment 13b)", "entrant": "J1",
           "comparator": "v8.2 as published", "rule": "v9.1's rule at 97.5 % (the JTWC family's first budget, 2.5 %)",
           "scored_as_of": now.strftime("%Y-%m-%dT%H:%MZ"), "look_dates": list(LOOKS),
           "records_with_j1": len(recs), "running": evaluate(scored), "looks": looks,
           "note": "before a look this is a running record; no claim is read from it"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    run = res["running"]
    print(f"J1 prospective: {len(recs)} test records carry J1; {run.get('n', len(scored))} scored "
          f"({run.get('j1', {}).get('events', 0) if isinstance(run.get('j1'), dict) else 0} RI); wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
