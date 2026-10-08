"""Program TC1 (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md): our own track and intensity forecast.

    PYTHONPATH=src python scripts/hurricane_tc1.py control     # the verifier must reproduce NHC's published OFCL errors
    PYTHONPATH=src python scripts/hurricane_tc1.py select      # the config, on CHOOSE (2021-2022)
    PYTHONPATH=src python scripts/hurricane_tc1.py dev         # the claims, on DEV (2023-2025)
    PYTHONPATH=src python scripts/hurricane_tc1.py read2026    # the 2026 season so far, fetched fresh (no claim)

``control`` scores NHC's official forecast (OFCL) with this program's verifier and compares every
(basin, kind, season, lead) cell with NHC's own published error tables
(results/hurricane_tc1/nhc_ofcl_published.json). It is the verifier's oracle: if our verification rules or our
truth differ from NHC's, it shows here, on numbers NHC has already published, before any TC1 number exists.
``select``, ``dev`` and ``read2026`` refuse to run unless the control has passed.

Verification rules (NHC's): a forecast made at synoptic time t for lead tau is verified when the best track
classes the system as a tropical or subtropical cyclone (TD, TS, HU, SD, SS) at both t and t + tau. Truth is the
best track (IBTrACS's USA columns) at synoptic hours. Track error is the great-circle distance (n mi); intensity
error the absolute difference (kt).
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from hazardpulse.hurricane import consensus as cs  # noqa: E402

CACHE = Path("C:/Users/Josh/Projects/hazardpulse/.cache")
DECKS = CACHE / "atcf_adecks"
FRESH_2026 = CACHE / "hurricane_tc1" / "2026"            # every 2026 deck, fetched by read2026
DECKS_2026 = FRESH_2026 / "adeck"
OUT_DIR = ROOT / "results" / "hurricane_tc1"
PUBLISHED = OUT_DIR / "nhc_ofcl_published.json"
CONTROL = OUT_DIR / "verifier_control.json"
SELECTION = OUT_DIR / "selection.json"
DEV_OUT = OUT_DIR / "dev.json"
READ_2026 = OUT_DIR / "season_2026.json"
TC_STATUS = frozenset(("TD", "TS", "HU", "SD", "SS"))
SEASONS = range(2020, 2026)
COMPARATORS = ("OFCL", "TVCN", "IVCN", "HCCA", "GDMI")
NHC_ATCF = "https://ftp.nhc.noaa.gov/atcf"

# the registered design (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md)
WARMUP, CHOOSE, DEV = (2020,), (2021, 2022), (2023, 2024, 2025)
SCORED_LEADS = (24, 48, 72, 96, 120)
GRID = tuple(cs.Config(half_life_days=h, shrink=s, debias=b, storm_boost=k)
             for h in (20.0, 60.0, 180.0) for s in (0.5, 1.0) for b in (False, True) for k in (0.0, 4.0))
EQUAL = cs.Config(half_life_days=None)
CLAIM_LEVEL = 0.9875                 # four primary claims share 5% (Bonferroni)
REPORT_LEVEL = 0.95
SEED, REPS = 20261008, 4000
SECOND = {"track": "TVCN", "intensity": "IVCN"}       # NHC's simple consensus for each kind
ALL_TECHS = tuple(sorted(set(cs.TRACK_MEMBERS) | set(cs.INTENSITY_MEMBERS) | set(COMPARATORS)))


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------------------------------------------

def ibtracs_path(basin: str) -> Path:
    import hashlib

    import build_hurricane_training_data as bh
    return CACHE / "ibtracs" / f"{hashlib.md5(bh.IBTRACS_URL.format(basin=basin).encode()).hexdigest()}.csv"


def load_truth(seasons=SEASONS) -> dict[str, dict[dt.datetime, tuple[float, float, float, str]]]:
    """``{ATCF id: {synoptic time: (lat, lon, wind kt, status)}}`` from IBTrACS NA and EP (USA columns). A storm in
    both files keeps the first copy of each time."""
    lo, hi = min(seasons), max(seasons)
    out: dict[str, dict] = {}
    for basin in ("NA", "EP"):
        with ibtracs_path(basin).open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rd = csv.reader(fh)
            col = {h.strip(): i for i, h in enumerate(next(rd))}
            next(rd, None)                                         # units row
            for row in rd:
                try:
                    season = int(row[col["SEASON"]])
                except ValueError:
                    continue
                if not lo <= season <= hi:
                    continue
                aid = row[col["USA_ATCF_ID"]].strip().upper()
                if len(aid) != 8 or aid[:2] not in ("AL", "EP", "CP"):
                    continue
                try:
                    t = dt.datetime.strptime(row[col["ISO_TIME"]].strip(), "%Y-%m-%d %H:%M:%S")
                    lat, lon = float(row[col["USA_LAT"]]), float(row[col["USA_LON"]])
                except ValueError:
                    continue
                if t.hour % 6 or t.minute:
                    continue
                try:
                    wind = float(row[col["USA_WIND"]])
                except ValueError:
                    wind = math.nan
                out.setdefault(aid, {}).setdefault(t, (lat, lon, wind, row[col["USA_STATUS"]].strip().upper()))
    return out


# ---------------------------------------------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------------------------------------------

def deck_paths(seasons=SEASONS) -> list[Path]:
    """Every numbered AL/EP/CP storm's a-deck (invests and tests, numbers 80-99, are not storms)."""
    out = []
    for season in seasons:
        root = DECKS_2026 if season == 2026 else DECKS / str(season)
        for p in sorted(root.glob("a*.dat.gz")):
            n = p.name
            if n[1:3] in ("al", "ep", "cp") and n[3:5].isdigit() and int(n[3:5]) < 80:
                out.append(p)
    return out


def load_decks(techs, seasons=SEASONS) -> list[cs.StormDeck]:
    decks = []
    for p in deck_paths(seasons):
        text = gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
        decks.append(cs.parse_deck(p.name[1:9], text, techs))
        del text
    return decks


def truth_for(deck: cs.StormDeck, truth: dict) -> dict:
    """The best track a deck verifies against. A storm that crossed basins can carry a second ATCF id (Bonnie
    2022 was AL02, then EP04) while IBTrACS keeps the whole track under the first: a deck whose id is not in
    IBTrACS takes the track that its own CARQ fixes follow -- the most fixes within 111 km at the same synoptic
    times, at least three."""
    own = truth.get(deck.storm.upper())
    if own is not None:
        return own
    season = deck.storm[-4:]
    fixes = [(t, deck.analysis(t)) for t in deck.analysis_times]
    best, hits = {}, 2
    for aid, tr in truth.items():
        if aid[-4:] != season:
            continue
        n = sum(1 for t, a in fixes if t in tr and math.isfinite(a[0]) and math.isfinite(a[1])
                and cs.great_circle_km(a[0], a[1], tr[t][0], tr[t][1]) <= 111.0)
        if n > hits:
            best, hits = tr, n
    return best


def area_at(lat: float, lon: float) -> str:
    """Whose forecast it is, by where the storm is: ``AL`` (the Atlantic, NHC), ``EP`` (the eastern Pacific east
    of 140 W, NHC) or ``CP`` (140 W to the dateline, CPHC). The Atlantic-Pacific divide follows the Central
    American and Mexican coasts in coarse steps (a storm over the isthmus is the rare ambiguous case)."""
    if lon <= -140.0 or lon > 0:
        return "CP"
    divide = (-77.5 if lat <= 8.5 else -83.0 if lat <= 10.0 else -86.0 if lat <= 13.0
              else -94.5 if lat <= 16.5 else -100.0 if lat <= 30.0 else -110.0)
    return "EP" if lon < divide else "AL"


def verify(tr: dict, t: dt.datetime, lead: int, fc, kind: str) -> float | None:
    """The error of one forecast against its storm's best track ``tr`` under NHC's rules, or None when it is not
    verified."""
    a, b = tr.get(t), tr.get(t + dt.timedelta(hours=lead))
    if a is None or b is None or a[3] not in TC_STATUS or b[3] not in TC_STATUS:
        return None
    if kind == "track":
        if not (math.isfinite(fc[0]) and math.isfinite(fc[1])):
            return None
        return cs.great_circle_km(fc[0], fc[1], b[0], b[1]) / cs.KM_PER_NM
    if not (math.isfinite(fc[2]) and math.isfinite(b[2])):
        return None
    return abs(fc[2] - b[2])


# published cells that contradict NHC's own other table for the same forecasts; each is shown, never pooled
TABLE_INCONSISTENT = {
    ("AL", "track", 2020, 72): "NHC's track table gives 318 cases (79.9 n mi); its intensity table gives 275 for the "
                               "same forecasts at 72 h, and every other 2020 lead has equal counts in both tables. "
                               "This verifier finds 275 (98.1 n mi), and matches the intensity cell exactly "
                               "(275, 10.9 kt)",
}


def control() -> int:
    """Score OFCL with this verifier; compare with NHC's published tables, cell by cell."""
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    leads = pub["leads"]
    log("loading truth and OFCL")
    truth = load_truth()
    decks = load_decks(("OFCL",))
    errs: dict[tuple, list[float]] = {}
    for d in decks:
        season = int(d.storm[-4:])
        tr = truth_for(d, truth)
        for t in d.cycles:
            if t not in tr:
                continue
            basin = area_at(tr[t][0], tr[t][1])
            if basin == "CP":
                continue
            for lead in leads:
                fc = d.get(t, "OFCL", lead)
                if fc is None:
                    continue
                for kind in ("track", "intensity"):
                    e = verify(tr, t, lead, fc, kind)
                    if e is not None:
                        errs.setdefault((basin, kind, season, lead), []).append(e)
    cells = []
    for table, years in pub["tables"].items():
        basin, kind = table.split("_")
        for year, row in years.items():
            for lead, (p_err, p_n) in zip(leads, row):
                ours = errs.get((basin, kind, int(year), lead), [])
                cells.append({"basin": basin, "kind": kind, "season": int(year), "lead": lead,
                              "nhc_error": p_err, "nhc_n": p_n, "ours_error": float(np.mean(ours)) if ours else None,
                              "ours_n": len(ours)})
    # pooled over the finalized seasons (2020-2024) per basin, kind and lead: N within 2%, mean error within
    # 2% (track) or 0.2 kt (intensity); 2025's best track was still being finalized when the tables were made
    for c in cells:
        why = TABLE_INCONSISTENT.get((c["basin"], c["kind"], c["season"], c["lead"]))
        if why:
            c["excluded"] = why
    pooled, ok = [], True
    for basin in ("AL", "EP"):
        for kind in ("track", "intensity"):
            for lead in leads:
                cs_ = [c for c in cells if c["basin"] == basin and c["kind"] == kind and c["lead"] == lead
                       and c["season"] <= 2024 and "excluded" not in c]
                n_nhc = sum(c["nhc_n"] for c in cs_)
                n_ours = sum(c["ours_n"] for c in cs_)
                e_nhc = sum(c["nhc_error"] * c["nhc_n"] for c in cs_) / n_nhc
                e_ours = sum((c["ours_error"] or 0.0) * c["ours_n"] for c in cs_) / max(n_ours, 1)
                tol = 0.02 * e_nhc if kind == "track" else 0.2
                passed = abs(n_ours - n_nhc) <= 0.02 * n_nhc and abs(e_ours - e_nhc) <= tol
                ok &= passed
                pooled.append({"basin": basin, "kind": kind, "lead": lead, "nhc_n": n_nhc, "ours_n": n_ours,
                               "nhc_error": round(e_nhc, 2), "ours_error": round(e_ours, 2), "pass": passed})
                log(f"{basin} {kind:9s} {lead:3d} h: N {n_ours:5d} vs NHC {n_nhc:5d} | error {e_ours:6.2f} vs "
                    f"{e_nhc:6.2f}  {'ok' if passed else 'FAIL'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verifier_control.json").write_text(json.dumps({
        "what": "OFCL scored by TC1's verifier vs NHC's published tables (2020-2024 pooled is the control; 2025 shown)",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "truth": "IBTrACS v04r01 USA columns (cache 2026-04-13), synoptic hours", "passed": ok,
        "pooled_2020_2024": pooled, "cells": cells}, indent=1), encoding="utf-8")
    log(f"verifier control {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------

def cases(decks, truth_of, seasons) -> list[tuple]:
    """``(deck, storm best track, t)`` for every cycle NHC would verify from: an NHC area (not CP) at t, in the
    seasons."""
    out = []
    for d in decks:
        if int(d.storm[-4:]) not in seasons:
            continue
        tr = truth_of(d)
        for t in d.cycles:
            if t in tr and area_at(tr[t][0], tr[t][1]) != "CP":
                out.append((d, tr, t))
    return out


def score(case_list, kind: str, leads, products: dict) -> dict:
    """``{product: {lead: {(storm, t): error}}}`` -- a product is a TC1 run (``{(storm, t, lead): (forecast,
    _)}``) or an aid's name, read from the deck."""
    out = {p: {ld: {} for ld in leads} for p in products}
    for d, tr, t in case_list:
        for lead in leads:
            for p, src in products.items():
                if isinstance(src, str):
                    fc = d.get(t, src, lead)
                else:
                    got = src.get((d.storm, t, lead))
                    fc = None if got is None else ((got[0][0], got[0][1], math.nan) if kind == "track"
                                                   else (math.nan, math.nan, got[0]))
                if fc is None:
                    continue
                e = verify(tr, t, lead, fc, kind)
                if e is not None:
                    out[p][lead][(d.storm, t)] = e
    return out


def mean_error(errs: dict, leads) -> dict:
    per = {ld: (float(np.mean(list(errs[ld].values()))) if errs[ld] else None, len(errs[ld])) for ld in leads}
    return {"per_lead": {str(k): {"error": v[0], "n": v[1]} for k, v in per.items()},
            "mean_over_leads": float(np.mean([v[0] for v in per.values()])) if all(v[0] is not None for v in per.values())
            else None}


def paired(a: dict, b: dict, leads, level: float) -> dict:
    """``a - b`` on the cases both verified (homogeneous per lead): per lead and the mean over the leads, with
    storm-block bootstrap intervals at ``level``."""
    rng = np.random.default_rng(SEED)
    storms = sorted({k[0] for ld in leads for k in set(a[ld]) & set(b[ld])})
    if not storms:
        return {"n": 0}
    ix = {s: i for i, s in enumerate(storms)}
    sums = np.zeros((len(leads), len(storms)))
    cnts = np.zeros((len(leads), len(storms)))
    for j, ld in enumerate(leads):
        for k in set(a[ld]) & set(b[ld]):
            sums[j, ix[k[0]]] += a[ld][k] - b[ld][k]
            cnts[j, ix[k[0]]] += 1
    point = sums.sum(1) / np.maximum(cnts.sum(1), 1)
    draws = []
    for _ in range(REPS):
        w = np.bincount(rng.integers(0, len(storms), len(storms)), minlength=len(storms))
        draws.append((sums @ w) / np.maximum(cnts @ w, 1))
    draws = np.array(draws)
    q = (1 - level) / 2
    mean_draws = draws.mean(1)
    return {"n_storms": len(storms), "level": level,
            "per_lead": {str(ld): {"d": float(point[j]), "n": int(cnts[j].sum()),
                                   "ci": [float(np.quantile(draws[:, j], q)), float(np.quantile(draws[:, j], 1 - q))]}
                         for j, ld in enumerate(leads)},
            "mean_over_leads": {"d": float(point.mean()),
                                "ci": [float(np.quantile(mean_draws, q)), float(np.quantile(mean_draws, 1 - q))]}}


def _control_passed() -> None:
    if not CONTROL.exists() or not json.loads(CONTROL.read_text(encoding="utf-8")).get("passed"):
        raise SystemExit("the verifier control has not passed: run `hurricane_tc1.py control` first")


def _config_dict(cfg: cs.Config) -> dict:
    return {"half_life_days": cfg.half_life_days, "shrink": cfg.shrink, "debias": cfg.debias,
            "storm_boost": cfg.storm_boost, "prior_n": cfg.prior_n, "include_official": cfg.include_official}


def select() -> int:
    """Per kind, the grid config with the lowest mean (over the scored leads) mean error on CHOOSE. The online
    state runs from the start of the warm-up season."""
    _control_passed()
    seasons = WARMUP + CHOOSE
    log(f"loading decks and truth {seasons[0]}-{seasons[-1]}")
    truth = load_truth(seasons)
    decks = load_decks(ALL_TECHS, seasons)
    case_list = cases(decks, lambda d: truth_for(d, truth), CHOOSE)
    out = {}
    for kind in ("track", "intensity"):
        table = []
        for i, cfg in enumerate((EQUAL,) + GRID):
            fcs = cs.run(decks, cfg, kind, keep_weights=False)
            me = mean_error(score(case_list, kind, SCORED_LEADS, {"c": fcs})["c"], SCORED_LEADS)
            table.append({"index": i, "config": _config_dict(cfg), **me})
            log(f"{kind} config {i:2d}: mean over leads {me['mean_over_leads']:.3f}")
        best = min(table[1:], key=lambda r: (r["mean_over_leads"], r["index"]))
        out[kind] = {"chosen": best, "equal_weight": table[0], "grid": table}
        log(f"{kind}: chose config {best['index']} {best['config']}")
    SELECTION.write_text(json.dumps({
        "phase": "TC1 select", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md", "prereg_tag": "prereg-tc1",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": {"warmup": WARMUP, "choose": CHOOSE}, "scored_leads": SCORED_LEADS, **out}, indent=1),
        encoding="utf-8")
    log(f"wrote {SELECTION}")
    return 0


def chosen(kind: str, official: bool = False) -> cs.Config:
    c = json.loads(SELECTION.read_text(encoding="utf-8"))[kind]["chosen"]["config"]
    return cs.Config(**{**c, "include_official": official})


def evaluate(decks, truth_of, seasons, level: float) -> dict:
    """TC1, TC1+O and the equal-weight reference against OFCL, HCCA, NHC's simple consensus and GDMI."""
    case_list = cases(decks, truth_of, seasons)
    out = {}
    for kind in ("track", "intensity"):
        products = {"TC1": cs.run(decks, chosen(kind), kind, keep_weights=False),
                    "TC1+O": cs.run(decks, chosen(kind, official=True), kind, keep_weights=False),
                    "EQ": cs.run(decks, EQUAL, kind, keep_weights=False),
                    "OFCL": "OFCL", "HCCA": "HCCA", SECOND[kind]: SECOND[kind], "GDMI": "GDMI"}
        errs = score(case_list, kind, SCORED_LEADS, products)
        res = {"errors": {p: mean_error(e, SCORED_LEADS) for p, e in errs.items()},
               "claims": {f"{p}-OFCL": paired(errs[p], errs["OFCL"], SCORED_LEADS, level) for p in ("TC1", "TC1+O")},
               "reported": {f"{p}-{q}": paired(errs[p], errs[q], SCORED_LEADS, REPORT_LEVEL)
                            for p in ("TC1", "TC1+O") for q in ("HCCA", SECOND[kind], "GDMI")}}
        res["reported"]["EQ-OFCL"] = paired(errs["EQ"], errs["OFCL"], SCORED_LEADS, REPORT_LEVEL)
        for name, c in res["claims"].items():
            c["claim"] = bool(c.get("n_storms") and c["mean_over_leads"]["ci"][1] < 0)
            log(f"{kind} {name}: mean over leads {c['mean_over_leads']['d']:+.2f} "
                f"[{c['mean_over_leads']['ci'][0]:+.2f}, {c['mean_over_leads']['ci'][1]:+.2f}] -> claim {c['claim']}")
        out[kind] = res
    return out


def dev() -> int:
    _control_passed()
    if not SELECTION.exists():
        raise SystemExit("no selection: run `hurricane_tc1.py select` first")
    seasons = WARMUP + CHOOSE + DEV
    log(f"loading decks and truth {seasons[0]}-{seasons[-1]}")
    truth = load_truth(seasons)
    decks = load_decks(ALL_TECHS, seasons)
    res = evaluate(decks, lambda d: truth_for(d, truth), DEV, CLAIM_LEVEL)
    DEV_OUT.write_text(json.dumps({
        "phase": "TC1 dev", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md", "prereg_tag": "prereg-tc1",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": DEV, "scored_leads": SCORED_LEADS, "claim_level": CLAIM_LEVEL,
        "configs": {k: _config_dict(chosen(k)) for k in ("track", "intensity")}, **res}, indent=1), encoding="utf-8")
    log(f"wrote {DEV_OUT}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# 2026, the season so far (a further read: no claim)
# ---------------------------------------------------------------------------------------------------------------

def _fetch(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hazardpulse-tc1"}), timeout=60) as r:
        return r.read()


def fetch_2026() -> list[str]:
    """Every numbered 2026 AL/EP/CP storm's a-deck and b-deck, as NHC publishes them now."""
    import re
    index = _fetch(f"{NHC_ATCF}/aid_public/").decode("utf-8", "replace")
    names = sorted(set(re.findall(r"a((?:al|ep|cp)(\d\d)2026)\.dat\.gz", index)))
    storms = [s for s, n in names if int(n) < 80]
    (FRESH_2026 / "adeck").mkdir(parents=True, exist_ok=True)
    (FRESH_2026 / "btk").mkdir(parents=True, exist_ok=True)
    for s in storms:
        (FRESH_2026 / "adeck" / f"a{s}.dat.gz").write_bytes(_fetch(f"{NHC_ATCF}/aid_public/a{s}.dat.gz"))
        (FRESH_2026 / "btk" / f"b{s}.dat").write_bytes(_fetch(f"{NHC_ATCF}/btk/b{s}.dat"))
    return storms


def btk_truth(storm: str) -> dict:
    """The operational best track of a 2026 storm, ``{synoptic time: (lat, lon, wind, status)}``."""
    from hazardpulse.hurricane.atcf import _parse_lat, _parse_lon
    out = {}
    for line in (FRESH_2026 / "btk" / f"b{storm}.dat").read_text(encoding="utf-8", errors="replace").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 11 or f[4] != "BEST":
            continue
        try:
            t = dt.datetime.strptime(f[2], "%Y%m%d%H")
            lat, lon, wind = _parse_lat(f[6]), _parse_lon(f[7]), float(f[8])
        except (ValueError, TypeError):
            continue
        if t.hour % 6 or lat is None or lon is None:
            continue
        out.setdefault(t, (lat, lon, wind, f[10].upper()))
    return out


def read2026() -> int:
    _control_passed()
    if not SELECTION.exists():
        raise SystemExit("no selection: run `hurricane_tc1.py select` first")
    storms = fetch_2026()
    log(f"fetched {len(storms)} 2026 storms' decks")
    seasons = WARMUP + CHOOSE + DEV + (2026,)
    decks = load_decks(ALL_TECHS, seasons)
    t26 = {s: btk_truth(s) for s in storms}
    res = evaluate(decks, lambda d: t26.get(d.storm, {}), (2026,), REPORT_LEVEL)
    READ_2026.write_text(json.dumps({
        "phase": "TC1 2026 further read", "program": "docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md",
        "prereg_tag": "prereg-tc1", "declared": "the season in progress; operational best tracks; no claim",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "storms": storms,
        "scored_leads": SCORED_LEADS, **res}, indent=1), encoding="utf-8")
    log(f"wrote {READ_2026}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("control", "select", "dev", "read2026"))
    phase = ap.parse_args(argv).phase
    return {"control": control, "select": select, "dev": dev, "read2026": read2026}[phase]()


if __name__ == "__main__":
    raise SystemExit(main())
