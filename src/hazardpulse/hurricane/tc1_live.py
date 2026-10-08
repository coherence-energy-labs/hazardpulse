"""TC1 live (docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md, amendment 1): our track and intensity forecast for every
active NHC storm, issued by the hurricane scorer.

Each run replays the current season's decks from the models saved after the last complete season
(``results/hurricane_tc1/state.json``, made and checked by ``scripts/hurricane_tc1.py snapshot``). That is the
backtest's own pass: the live forecast is what the backtest would have issued from the same decks. A forecast at
synoptic time t is made only once t + 3 h 30 min has passed, as the RI program's test records are.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from hazardpulse.hurricane import consensus as cs

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "results" / "hurricane_tc1" / "state.json"
SELECTION = ROOT / "results" / "hurricane_tc1" / "selection.json"
ISSUE_DELAY = dt.timedelta(hours=3, minutes=30)
PRODUCTS = ("TC1", "TC1+O")
TECHS = tuple(sorted(set(cs.TRACK_MEMBERS) | set(cs.INTENSITY_MEMBERS) | {cs.OFFICIAL}))


class StateError(RuntimeError):
    """The saved models cannot be used for this forecast (missing, another selection, another season)."""


def sha256(path: Path) -> str:
    """The digest of a JSON file's content, not its bytes: a Windows checkout writes CRLF where the runner reads
    LF, and a byte hash would then refuse the same file on one machine and accept it on the other."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_state(path: Path = STATE, selection: Path = SELECTION) -> dict:
    """The saved models, refused unless they were made from the selection on disk."""
    if not path.exists():
        raise StateError(f"no saved TC1 models at {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("selection_sha256") != sha256(selection):
        raise StateError("the saved TC1 models were made from another selection")
    sel = json.loads(selection.read_text(encoding="utf-8"))
    for key, m in doc["models"].items():
        product, kind = key.split("/")
        want = dict(sel[kind]["chosen"]["config"], include_official=(product == "TC1+O"))
        if m["config"] != want:
            raise StateError(f"{key}: saved config {m['config']} is not the selected {want}")
    return doc


def issue_until(now: dt.datetime) -> dt.datetime:
    """The latest synoptic time whose forecast may be issued at ``now`` (t + 3 h 30 passed)."""
    t = now - ISSUE_DELAY
    return t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0)


def forecast(state: Mapping, decks: Iterable[cs.StormDeck], storms: Iterable[str], now: dt.datetime) -> dict:
    """``{storm: record}`` for each requested storm (lower-case ATCF id) that has an analysis at or before
    ``issue_until(now)``: the latest such cycle, TC1 and TC1+O, track and intensity at every lead, beside NHC's
    official forecast from the same deck. The decks must be the whole current season's."""
    season = now.year
    if int(state["season_end"]) != season - 1:
        raise StateError(f"the saved models end with {state['season_end']}; season {season} needs {season - 1}")
    decks = [d for d in decks if d.storm.endswith(str(season))]
    by_id = {d.storm: d for d in decks}
    until = issue_until(now)
    cycles = {}
    for s in storms:
        d = by_id.get(s)
        times = [t for t in d.analysis_times if t <= until] if d else []
        if times:
            cycles[s] = max(times)
    if not cycles:
        return {}
    wanted = set(cycles.items())
    runs = {}
    for key, saved in state["models"].items():
        runs[key] = cs.replay(cs.OnlineConsensus.from_dict(saved), decks, cycles=wanted, until=until)
    out = {}
    for s, t in cycles.items():
        d = by_id[s]
        rec = {"cycle": t.strftime("%Y-%m-%dT%H:00:00Z"), "issued_after": until.strftime("%Y-%m-%dT%H:00:00Z"),
               "season_state_end": state["season_end"], "selection_sha256": state["selection_sha256"]}
        for product in PRODUCTS:
            leads = {}
            for lead in cs.LEADS:
                tr = runs[f"{product}/track"].get((s, t, lead))
                iv = runs[f"{product}/intensity"].get((s, t, lead))
                if tr is None and iv is None:
                    continue
                leads[str(lead)] = {"lat": None if tr is None else round(tr[0][0], 2),
                                    "lon": None if tr is None else round(tr[0][1], 2),
                                    "vmax_kt": None if iv is None else round(iv[0], 1),
                                    "track_weights": None if tr is None else _top(tr[1]),
                                    "intensity_weights": None if iv is None else _top(iv[1])}
            rec[product] = leads
        official = {}
        for lead in cs.LEADS:
            f = d.get(t, cs.OFFICIAL, lead)
            if f is not None:
                official[str(lead)] = {"lat": _num(f[0]), "lon": _num(f[1]), "vmax_kt": _num(f[2])}
        rec["OFCL"] = official
        out[s] = rec
    return out


def _num(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 2)


def _top(weights: Mapping[str, float] | None, n: int = 4) -> dict[str, float] | None:
    if not weights:
        return None
    return {m: round(w, 3) for m, w in sorted(weights.items(), key=lambda kv: -kv[1])[:n] if w > 0}
