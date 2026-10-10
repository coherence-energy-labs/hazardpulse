"""Tornado program amendment 11 (T2b): the served payloads' Platt maps refitted on NOAA's new ProbSevere format,
run live in SHADOW beside the served probabilities -- recorded, never published -- until the registered rule
(docs/TORNADO_MODEL_PROGRAM.md, amendment 11) says otherwise.

A recalibration changes no input and no tree: it is a new (a, b) on the payload's own raw margin
(``lgbm_payload.predict_raw``). So the shadow of a storm forecast is a function of the record's stored inputs alone:

* per candidate that applies to the storm -- ``p60_w`` and the 30/90-min products when the +W model served,
  ``p60`` (the no-warnings fallback, which reads only block P) on every storm -- the payload's raw ``margin``, the
  served payload's own probability from it (``served``; for a product, before the coherence clip), and the
  recalibrated probability (``recal``);
* ``probability_60min``: the recalibrated probability of the model that served the storm.

The forecast record states what produced it (``descriptor``): the artifact's SHA-256 and, per candidate, the
recalibration's (a, b), the served (a, b) and the payload's SHA-256. A candidate whose artifact entry names another
payload than the one loaded is refused, never applied. No artifact: no shadow, and the forecast is unchanged.

One module for the live scorer (``attach``) and the record audit (``recompute``), so what a record carries and what
an audit recomputes are the same arithmetic: ``proba_from_margin`` is ``lgbm_payload.predict_proba``'s, applied to
a margin already summed, and gives the served probability bit for bit (tests/test_tornado_t2b.py).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import v3_serving as vs

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "results" / "models"
ARTIFACT_FILE = "tornado_v3_recal_t2b.json"
SCHEMA = "hazardpulse_tornado_recalibration/1"
SHADOW_KEY = "t2b_shadow"
PROGRAM = "docs/TORNADO_MODEL_PROGRAM.md amendment 11"
#: candidate -> the served payload file it recalibrates (the keys are V3Suite.versions()'s)
CANDIDATES: dict[str, str] = {"p60_w": vs.MAIN_FILE, "p60": vs.FALLBACK_FILE,
                              "p30": vs.PRODUCT_FILES["p30"], "p90": vs.PRODUCT_FILES["p90"]}
#: candidate -> the training label its live record is judged on
LABELS: dict[str, str] = {"p60_w": "storm_60", "p60": "storm_60", "p30": "storm_30", "p90": "storm_90"}
#: ``v3.model`` -> the candidate that recalibrates the 60-min payload which served the storm
SERVED_BY: dict[str, str] = {"v3_w": "p60_w", "v3": "p60"}
#: product candidate -> the record's (coherence-clipped) published field
PRODUCT_FIELDS: dict[str, str] = {"p30": "probability_30min", "p90": "probability_90min"}


class ArtifactError(ValueError):
    """The recalibration artifact is malformed: never applied."""


def canonical_bytes(obj) -> bytes:
    """The artifact's file bytes (single-line canonical JSON, as ``lgbm_payload.save`` writes a payload): its
    SHA-256 is the same on every checkout."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(art: Mapping) -> str:
    return hashlib.sha256(canonical_bytes(art)).hexdigest()


def validate(art: Mapping) -> dict:
    """The artifact, checked: schema, and per candidate a finite (a, b) with a > 0 (an increasing map: ranking
    unchanged) and the payload digest it was fitted for."""
    if art.get("schema") != SCHEMA:
        raise ArtifactError(f"schema {art.get('schema')!r}, expected {SCHEMA!r}")
    cands = art.get("candidates")
    if not isinstance(cands, Mapping) or not cands:
        raise ArtifactError("no candidates")
    for k, c in cands.items():
        if k not in CANDIDATES:
            raise ArtifactError(f"unknown candidate {k!r}")
        a, b = c.get("a"), c.get("b")
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (a, b)):
            raise ArtifactError(f"{k}: non-finite calibration ({a!r}, {b!r})")
        if not float(a) > 0.0:
            raise ArtifactError(f"{k}: a = {a!r} <= 0 would not be an increasing map")
        if not isinstance(c.get("payload_sha256"), str) or len(c["payload_sha256"]) != 64:
            raise ArtifactError(f"{k}: no payload digest")
    return dict(art)


def load(path: str | Path | None = None) -> dict | None:
    """The artifact, or None when it is absent (no shadow). A present but malformed artifact raises."""
    p = Path(path) if path is not None else MODELS / ARTIFACT_FILE
    if not p.exists():
        return None
    return validate(json.loads(p.read_bytes().decode("utf-8")))


def proba_from_margin(cal: Mapping, F) -> np.ndarray:
    """``lgbm_payload.predict_proba``'s arithmetic on a margin already summed: for a 1-row ``X``,
    ``proba_from_margin(payload["calibration"], predict_raw(payload, X))`` IS ``predict_proba(payload, X)``,
    bit for bit."""
    z = float(cal["a"]) * np.asarray(F, np.float64) + float(cal["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def input_matrix(payload: Mapping, inputs: Iterable[Mapping]) -> np.ndarray:
    """Rows of the payload's inputs from stored records, exactly as ``v3_serving.recompute_p60`` builds one."""
    names = payload["feature_names"]
    rows = [[np.nan if inp.get(n) is None else float(inp[n]) for n in names] for inp in inputs]
    return np.asarray(rows, np.float64).reshape(len(rows), len(names))


def applicable(storm: Mapping) -> list[str]:
    """The candidates a stored storm forecast can carry: ``p60`` on every v3 storm that stored its inputs (the
    fallback reads only block P); ``p60_w`` and the products present in the record when the +W model served."""
    v3 = storm.get("v3") or {}
    if not v3.get("inputs"):
        return []
    if v3.get("model") == "v3_w":
        return ["p60_w", "p60"] + [k for k, f in PRODUCT_FIELDS.items() if v3.get(f) is not None]
    return ["p60"]


def _one(cal: Mapping, margin: np.ndarray) -> float:
    return float(proba_from_margin(cal, margin)[0])


class Shadow:
    """The artifact bound to the payloads being served. ``payloads``: candidate -> loaded payload."""

    def __init__(self, art: Mapping, payloads: Mapping[str, dict | None]):
        self.art = validate(art)
        self.sha256 = digest(self.art)
        self.bound: dict[str, dict] = {}
        self.refused: dict[str, str] = {}
        for k, c in self.art["candidates"].items():
            p = payloads.get(k)
            if p is None:
                self.refused[k] = "its payload is not loaded"
            elif vs.model_digest(p) != c["payload_sha256"]:
                self.refused[k] = f"fitted for payload {c['payload_sha256'][:12]}, not the served {vs.model_digest(p)[:12]}"
            else:
                self.bound[k] = p

    @classmethod
    def from_suite(cls, suite: vs.V3Suite, art: Mapping) -> "Shadow":
        return cls(art, {"p60_w": suite.main, "p60": suite.fallback,
                         "p30": suite.products.get("p30"), "p90": suite.products.get("p90")})

    @property
    def active(self) -> bool:
        return bool(self.bound)

    def descriptor(self) -> dict:
        """What produced every shadow of a forecast record (stored once per record)."""
        out = {"status": "ok" if self.active else "refused", "program": PROGRAM, "published": False,
               "artifact": f"results/models/{ARTIFACT_FILE}", "artifact_sha256": self.sha256,
               "candidates": {k: {"a": float(self.art["candidates"][k]["a"]),
                                  "b": float(self.art["candidates"][k]["b"]),
                                  "served_a": float(p["calibration"]["a"]), "served_b": float(p["calibration"]["b"]),
                                  "payload": CANDIDATES[k], "payload_sha256": self.art["candidates"][k]["payload_sha256"]}
                              for k, p in self.bound.items()}}
        if self.refused:
            out["refused"] = dict(self.refused)
        return out

    def compute(self, storms: list[Mapping]) -> list[dict | None]:
        """The shadow of each storm forecast (None when no candidate applies), from its stored inputs."""
        per: list[dict] = [{} for _ in storms]
        for k, payload in self.bound.items():
            idx = [i for i, s in enumerate(storms) if k in applicable(s)]
            if not idx:
                continue
            F = lp.predict_raw(payload, input_matrix(payload, [storms[i]["v3"]["inputs"] for i in idx]))
            cal = self.art["candidates"][k]
            for j, i in enumerate(idx):
                m = F[j:j + 1]
                per[i][k] = {"margin": float(F[j]), "served": _one(payload["calibration"], m), "recal": _one(cal, m)}
        out: list[dict | None] = []
        for s, models in zip(storms, per):
            if not models:
                out.append(None)
                continue
            sh: dict = {}
            served_by = SERVED_BY.get(str((s.get("v3") or {}).get("model")))
            if served_by in models:
                sh["probability_60min"] = models[served_by]["recal"]
            sh.update({k: models[k] for k in CANDIDATES if k in models})
            out.append(sh)
        return out

    def attach(self, storms: list[dict]) -> int:
        """Add ``t2b_shadow`` to every storm forecast a candidate applies to; nothing else is touched. Returns how
        many storms carry one."""
        n = 0
        for s, sh in zip(storms, self.compute(storms)):
            if sh is not None:
                s[SHADOW_KEY] = sh
                n += 1
        return n


def recompute(art: Mapping, payloads_by_sha: Mapping[str, dict], descriptor: Mapping,
              storms: list[Mapping]) -> dict:
    """The record audit for one forecast record: every stored shadow recomputed from the storm's stored inputs.

    Margins must be EXACT (a sum of tree leaves in a fixed order is exact IEEE arithmetic on any machine);
    probabilities within 1e-12 (NumPy's ``exp`` differs by an ulp or two across builds, amendment 10). Within the
    record, where the same process computed both: the served model's ``served`` must equal ``v3.probability_60min``
    EXACTLY, ``probability_60min`` the serving candidate's ``recal``, and a product's ``served`` its published
    value unless the coherence rule clipped it (then the published value is p60). The descriptor must state the
    artifact's own (a, b) and payload digests."""
    res = {"storms": 0, "values_checked": 0, "mismatched": []}
    cands = art["candidates"]
    for k, d in (descriptor.get("candidates") or {}).items():
        c = cands.get(k)
        if c is None or float(d.get("a")) != float(c["a"]) or float(d.get("b")) != float(c["b"]) \
                or d.get("payload_sha256") != c["payload_sha256"]:
            res["mismatched"].append(f"descriptor {k}: not the artifact's calibration or payload")
    rows: dict[str, list[tuple[int, Mapping]]] = {}
    for i, s in enumerate(storms):
        sh = s.get(SHADOW_KEY)
        if not sh:
            continue
        res["storms"] += 1
        expected = applicable(s)
        for k in CANDIDATES:
            if k in sh and k not in expected:
                res["mismatched"].append(f"storm {s.get('storm_id')}: carries {k}, which does not apply to it")
            if k in sh:
                rows.setdefault(k, []).append((i, sh[k]))
    for k, items in rows.items():
        d = (descriptor.get("candidates") or {}).get(k)
        payload = payloads_by_sha.get((d or {}).get("payload_sha256", ""))
        if d is None or payload is None:
            res["mismatched"].append(f"{k}: {len(items)} shadows whose payload is not known")
            continue
        X = input_matrix(payload, [storms[i]["v3"]["inputs"] for i, _ in items])
        F = lp.predict_raw(payload, X)
        for j, (i, rec) in enumerate(items):
            tag = f"storm {storms[i].get('storm_id')} {k}"
            res["values_checked"] += 3
            m = F[j:j + 1]
            if float(rec["margin"]) != float(F[j]):
                res["mismatched"].append(f"{tag}: margin {rec['margin']!r} recomputes as {float(F[j])!r}")
            if abs(_one(payload["calibration"], m) - float(rec["served"])) > 1e-12:
                res["mismatched"].append(f"{tag}: served {rec['served']!r} does not recompute")
            if abs(_one(cands[k], m) - float(rec["recal"])) > 1e-12:
                res["mismatched"].append(f"{tag}: recal {rec['recal']!r} does not recompute")
    for s in storms:
        sh = s.get(SHADOW_KEY)
        if not sh:
            continue
        v3 = s.get("v3") or {}
        tag = f"storm {s.get('storm_id')}"
        served_by = SERVED_BY.get(str(v3.get("model")))
        if served_by in sh:
            res["values_checked"] += 2
            if float(sh[served_by]["served"]) != float(v3.get("probability_60min")):
                res["mismatched"].append(f"{tag}: the shadow's served {served_by} is not v3.probability_60min")
            if sh.get("probability_60min") != sh[served_by]["recal"]:
                res["mismatched"].append(f"{tag}: probability_60min is not the serving model's recalibration")
        for k, field in PRODUCT_FIELDS.items():
            if k not in sh:
                continue
            res["values_checked"] += 1
            published = float(v3[field])
            if k in (v3.get("coherence_clipped") or []):
                ok = published == float(v3["probability_60min"])
            else:
                ok = published == float(sh[k]["served"])
            if not ok:
                res["mismatched"].append(f"{tag}: {k} published {published!r} is not its shadow's served value")
    return res
