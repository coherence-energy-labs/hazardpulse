"""Tornado v3 forecast records keep the exact inputs every served product read, so a past forecast
recomputes from its record alone (through JSON, at full precision)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado import v3_serving as vs

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
pytestmark = pytest.mark.skipif(not (MODELS / vs.MAIN_FILE).exists(), reason="v3 payloads not present")


@pytest.fixture(scope="module")
def suite():
    return vs.V3Suite.load(MODELS)


def _fv(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fv = rng.normal(size=len(sf.FEATURE_NAMES)) * 10
    fv[rng.random(len(fv)) < 0.15] = np.nan                       # some inputs missing, as live
    return fv


@pytest.mark.parametrize("warning", [None, (1.0, 12.5), (0.0, float("nan"))])
def test_every_served_product_recomputes_from_its_stored_inputs(suite, warning):
    for seed in range(5):
        fv = _fv(seed)
        rec = json.loads(json.dumps(vs.record_inputs(fv, warning)))        # through the JSON record
        models = ({"p60": suite.main, **suite.products} if warning is not None else {"p60": suite.fallback})
        for key, payload in models.items():
            if payload is None:
                continue
            want = float(lp.predict_proba(payload, suite._columns(payload, fv, warning))[0])
            assert vs.recompute_p60(payload, rec) == want, key            # bit for bit


def test_a_changed_stored_input_changes_the_recomputed_forecast(suite):
    fv = _fv(7)
    rec = vs.record_inputs(fv, (1.0, 5.0))
    base = vs.recompute_p60(suite.main, rec)
    changed = [n for n in suite.main["feature_names"] if rec.get(n) is not None
               and vs.recompute_p60(suite.main, dict(rec, **{n: rec[n] + 50.0})) != base]
    assert changed                     # the record's inputs are what the forecast depends on
