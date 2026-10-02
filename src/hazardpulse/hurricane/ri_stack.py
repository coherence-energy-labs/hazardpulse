"""NOAA RI-aid stack: the half that is shared by the fitter and the live scorer.

The pre-registered program (docs/HURRICANE_RI_PROGRAM.md) chooses one of six candidates for
P(V(t+24 h) - V(t) >= 30 kt) from NOAA's operational RI guidance -- SHIPS-RII (ATCF tech RIOD),
its logistic (RIOL) and Bayesian (RIOB) versions, their consensus (RIOC) and DTOPS (DTOP) --
optionally with v8.2. ``scripts/hurricane_ri_stack.py`` fits and selects; this module holds what
the served path must compute EXACTLY as the fitter did:

* the input representation: whole percent (what the e-decks carry; live SHIPS text is rounded
  half-up to it by :mod:`hazardpulse.hurricane.ships_text`), logits on the probability clipped
  to [0.005, 0.995]; the v8.2 probability's logit uses ``ri_model._logit`` (clip 1e-6);
* the design columns of a candidate on an availability pattern, intercept LAST (the column
  order of ``benchmark_hurricane_vs_ships.fit_stack``);
* the pattern fallback (Amendment 1, item 2): a cycle whose own pattern has no fit uses the
  largest fitted sub-pattern, ties keeping the inputs earliest in ``INPUT_ORDER``;
* the served artifact: canonical JSON, its identity ``hurricane_ri_stack_v1-<sha256[:12]>``
  bound to the file content (line endings normalised, so a CRLF checkout is the same model).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hazardpulse.hurricane import operational_ri as ori
from hazardpulse.hurricane import ri_model

SCHEMA = "hazardpulse.hurricane_ri_stack/1"
MODEL_NAME = "hurricane_ri_stack_v1"
ARTIFACT_PATH = ri_model.RESULTS / "models" / f"{MODEL_NAME}.json"

AIDS = ("RIOD", "RIOL", "RIOB", "DTOP")      # the logit-pool aids, in column order
V82 = "V82"                                   # v8.2 on live CARQ inputs (candidate E)
INPUT_ORDER = AIDS + (V82,)                   # also the tie order of the pattern fallback
REQUIRED_INPUT = "RIOD"                       # every case holds it (the case definition)
ALL_TECHS = ("RIOD", "RIOL", "RIOB", "RIOC", "DTOP", "SDCN")
PCT_CLIP = 0.005                              # half the whole-percent reporting resolution
MIN_EVENTS_PER_FIT = 20                       # events AND non-events a pattern fit needs

# Where each ATCF tech is printed in NHC's live SHIPS text ("Matrix of RI probabilities").
# RIOD = SHIPS-RII and DTOP = DTOPS are MEASURED identities (NHC discussion quotes, see
# benchmark_hurricane_vs_ships.RI_TECHS); RIOL/RIOB = Logistic/Bayesian are HYPOTHESIS
# (tech letter and consensus membership); RIOC = mean(RIOD, RIOL, RIOB) is MEASURED.
SHIPS_TEXT_ROW = {"RIOD": "SHIPS-RII", "RIOL": "Logistic", "RIOB": "Bayesian",
                  "RIOC": "Consensus", "DTOP": "DTOPS", "SDCN": "SDCON"}

CANDIDATES: dict[str, dict[str, Any]] = {
    "A": {"kind": "raw_aid", "aid": "DTOP", "fallback": "RIOD", "n_params": 0,
          "description": "DTOPS raw (SHIPS-RII where DTOPS is missing: Amendment 1, item 1)"},
    "B": {"kind": "raw_aid", "aid": "RIOD", "fallback": None, "n_params": 0,
          "description": "SHIPS-RII raw"},
    "C": {"kind": "raw_aid", "aid": "RIOC", "fallback": "RIOD", "n_params": 0,
          "description": "RIOC raw, NOAA's RI consensus (SHIPS-RII where RIOC is missing)"},
    "D": {"kind": "logit_pool", "inputs": AIDS, "basin": False, "n_params": 5,
          "description": "logistic pool of logit RIOD, RIOL, RIOB, DTOP + intercept, per availability pattern"},
    "E": {"kind": "logit_pool", "inputs": AIDS + (V82,), "basin": False, "n_params": 6,
          "description": "D + logit(v8.2 on live CARQ inputs), per availability pattern"},
    "F": {"kind": "logit_pool", "inputs": AIDS, "basin": True, "n_params": 7,
          "description": "D + Atlantic indicator + indicator x logit(DTOPS), per availability pattern"},
}


class StackArtifactError(RuntimeError):
    """The served stack artifact is missing, malformed, or not what it claims to be."""


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

def aid_logit(pct: np.ndarray | Iterable[float]) -> np.ndarray:
    """Logit of a whole-percent aid, the probability clipped to [0.005, 0.995] (equals
    ``benchmark_hurricane_vs_ships.logit_of(pct / 100, True)``; a test pins the identity)."""
    p = np.asarray(pct, dtype=float) / 100.0
    return ri_model._logit(np.clip(p, PCT_CLIP, 1.0 - PCT_CLIP))


def v82_logit(p: np.ndarray | Iterable[float]) -> np.ndarray:
    """Logit of a v8.2 probability (``logit_of(p, False)``: ri_model's own 1e-6 clip)."""
    return ri_model._logit(np.asarray(p, dtype=float))


def is_atlantic(basin: str) -> float:
    return 1.0 if str(basin).upper() == "AL" else 0.0


def available_inputs(aids: Mapping[str, Any], v82: float | None, inputs: Iterable[str]) -> tuple[str, ...]:
    """The candidate's inputs present at a cycle, in ``INPUT_ORDER``."""
    have = {k for k, v in aids.items() if v is not None}
    if v82 is not None and math.isfinite(float(v82)):
        have.add(V82)
    wanted = set(inputs)
    return tuple(name for name in INPUT_ORDER if name in wanted and name in have)


def column_names(pattern: Iterable[str], basin: bool) -> list[str]:
    """Design columns of a pattern, WITHOUT the intercept (which is appended last)."""
    pattern = tuple(pattern)
    cols = [f"logit_{name}" for name in pattern]
    if basin:
        cols.append("basin_al")
        if "DTOP" in pattern:
            cols.append("basin_al_x_logit_DTOP")
    return cols


def design_columns(pattern: Iterable[str], basin: bool, rows: list[Mapping[str, Any]]) -> list[np.ndarray]:
    """Column arrays for ``rows`` (each with ``aids`` {tech: pct}, ``v82`` and ``basin``).

    Every row must carry every input of the pattern; a missing one is a caller error.
    """
    pattern = tuple(pattern)
    cols: list[np.ndarray] = []
    for name in pattern:
        if name == V82:
            cols.append(v82_logit([r["v82"] for r in rows]))
        else:
            cols.append(aid_logit([r["aids"][name] for r in rows]))
    if basin:
        al = np.array([is_atlantic(r["basin"]) for r in rows], dtype=float)
        cols.append(al)
        if "DTOP" in pattern:
            cols.append(al * cols[pattern.index("DTOP")])
    return cols


def sub_patterns(pattern: Iterable[str]) -> list[tuple[str, ...]]:
    """Every sub-pattern of ``pattern`` that keeps the required input, in fallback order:
    larger first; among equal sizes, the one keeping inputs earliest in ``INPUT_ORDER``."""
    pattern = tuple(name for name in INPUT_ORDER if name in set(pattern))
    if REQUIRED_INPUT not in pattern:
        return []
    rest = [n for n in pattern if n != REQUIRED_INPUT]
    out: list[tuple[str, ...]] = []
    for k in range(len(rest), -1, -1):
        for combo in itertools.combinations(rest, k):   # lexicographic in INPUT_ORDER positions
            keep = set(combo) | {REQUIRED_INPUT}
            out.append(tuple(n for n in INPUT_ORDER if n in keep))
    return out


def all_patterns(inputs: Iterable[str]) -> list[tuple[str, ...]]:
    """Every subset of ``inputs`` that contains the required input (fallback order)."""
    return sub_patterns(inputs)


def resolve_pattern(own: Iterable[str], fitted: Iterable[Iterable[str]]) -> tuple[str, ...] | None:
    """The fit a cycle uses: its own pattern if fitted, else the first fitted sub-pattern."""
    have = {tuple(p) for p in fitted}
    for p in sub_patterns(own):
        if p in have:
            return p
    return None


# ---------------------------------------------------------------------------
# The served artifact
# ---------------------------------------------------------------------------

def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """One line of sorted-key JSON + LF: no line ending inside for a checkout to rewrite."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def save_artifact(payload: Mapping[str, Any], path: str | Path = ARTIFACT_PATH) -> str:
    """Validate, write atomically, and return the model_version bound to the written bytes."""
    validate_artifact(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(canonical_bytes(payload))
    part.replace(path)
    return model_version_of(path)


def model_version_of(path: str | Path) -> str:
    """``hurricane_ri_stack_v1-<sha256[:12]>`` of the artifact's content (CRLF normalised, as
    ``ri_model.sha256_file``), the identity every forecast, the prospective calibration pool and
    the trust layer carry -- the convention of ``definitive_model.model_version_of_payload``."""
    return f"{MODEL_NAME}-{ri_model.sha256_file(path)[:12]}"


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise StackArtifactError(message)


def validate_artifact(payload: Mapping[str, Any]) -> None:
    _need(payload.get("schema") == SCHEMA, f"unknown schema {payload.get('schema')!r}")
    cand = payload.get("candidate")
    _need(cand in CANDIDATES, f"unknown candidate {cand!r}")
    spec = CANDIDATES[cand]
    _need(payload.get("kind") == spec["kind"], f"candidate {cand} is {spec['kind']}, artifact says {payload.get('kind')!r}")
    _need(payload.get("logit_clip") == [PCT_CLIP, 1.0 - PCT_CLIP], "logit_clip differs from the program's [0.005, 0.995]")
    if spec["kind"] == "raw_aid":
        _need(payload.get("aid") == spec["aid"] and payload.get("fallback") == spec["fallback"],
              "raw-aid artifact names a different aid or fallback than its candidate")
        return
    _need(list(payload.get("inputs") or []) == list(spec["inputs"]), "inputs differ from the candidate's")
    _need(bool(payload.get("basin")) == bool(spec["basin"]), "basin flag differs from the candidate's")
    fits = payload.get("patterns") or []
    _need(len(fits) > 0, "no pattern fits")
    seen = set()
    for fit in fits:
        pattern = tuple(fit.get("pattern") or [])
        _need(pattern and pattern[0] == REQUIRED_INPUT, f"pattern {pattern} lacks {REQUIRED_INPUT}")
        _need(all(n in spec["inputs"] for n in pattern), f"pattern {pattern} has an input the candidate lacks")
        _need(list(pattern) == [n for n in INPUT_ORDER if n in set(pattern)], f"pattern {pattern} not in INPUT_ORDER")
        _need(pattern not in seen, f"pattern {pattern} fitted twice")
        seen.add(pattern)
        cols = column_names(pattern, spec["basin"]) + ["intercept"]
        _need(fit.get("columns") == cols, f"pattern {pattern}: columns {fit.get('columns')} != {cols}")
        coef = fit.get("coef") or []
        _need(len(coef) == len(cols) and all(isinstance(c, (int, float)) and math.isfinite(c) for c in coef),
              f"pattern {pattern}: coefficients missing or non-finite")
    _need((REQUIRED_INPUT,) in seen, "no fit for the minimal pattern (RIOD alone): no fallback exists")


def load_artifact(path: str | Path = ARTIFACT_PATH) -> tuple[dict[str, Any], str]:
    """``(payload, model_version)``; refuses a missing, unparsable or inconsistent file."""
    path = Path(path)
    if not path.exists():
        raise StackArtifactError(f"RI stack artifact missing at {path} (scripts/hurricane_ri_stack.py --phase final)")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StackArtifactError(f"unreadable RI stack artifact {path}: {exc}") from exc
    validate_artifact(payload)
    return payload, model_version_of(path)


def predict(payload: Mapping[str, Any], aids: Mapping[str, Any], basin: str,
            v82: float | None = None) -> tuple[float, dict[str, Any]] | None:
    """P(RI) for one cycle from the artifact, or None when the needed aids are absent.

    ``aids``: {ATCF tech: whole percent or None}. Returns ``(probability, info)`` with the
    inputs and pattern actually used.
    """
    aids = {k: (None if v is None else int(v)) for k, v in aids.items()}
    for v in aids.values():
        if v is not None and not 0 <= v <= 100:
            raise ValueError(f"aid value {v} outside 0..100")
    if payload["kind"] == "raw_aid":
        if aids.get(payload["aid"]) is not None:
            used = payload["aid"]
        elif payload.get("fallback") and aids.get(payload["fallback"]) is not None:
            used = payload["fallback"]
        else:
            return None
        return aids[used] / 100.0, {"kind": "raw_aid", "used": used, "inputs": {used: aids[used]}}
    if aids.get(REQUIRED_INPUT) is None:
        return None
    own = available_inputs(aids, v82, payload["inputs"])
    fits = {tuple(f["pattern"]): f for f in payload["patterns"]}
    pattern = resolve_pattern(own, fits)
    if pattern is None:
        return None
    fit = fits[pattern]
    row = {"aids": aids, "v82": v82, "basin": basin}
    cols = design_columns(pattern, bool(payload["basin"]), [row])
    x = np.array([c[0] for c in cols] + [1.0])
    prob = float(ori.sigmoid(x @ np.asarray(fit["coef"], dtype=float)))
    used = {n: (v82 if n == V82 else aids[n]) for n in pattern}
    return prob, {"kind": "logit_pool", "own_pattern": list(own), "pattern": list(pattern),
                  "fallback": tuple(own) != pattern, "inputs": used}
