"""The S2 stack (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 12): GEAR1's weight b + d logit(p_C0). A v1 stack (S1)
keeps computing exactly what it computed, so every forecast it issued stays recomputable."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.earthquake import operational_forecast as of

ROOT = Path(__file__).resolve().parents[1]
SERVED_V1 = ROOT / "results" / "models" / "earthquake_gear1_stack_v1.json"
BASE = types.SimpleNamespace(model_version="eq_operational_C0_v1-test")


def _g(seed=0):
    return np.random.default_rng(seed).uniform(-7.0, -0.5, of.N_CELLS)


def _p(seed=1):
    return np.random.default_rng(seed).uniform(1e-9, 0.3, of.N_CELLS)


def test_v2_applies_the_interaction_and_v1_does_not(tmp_path):
    g, p = _g(), _p()
    co = dict(a=-0.5386595234045181, c=0.7461788191936157, b=-0.07815908648879437)
    d = -0.07326573161498508
    of.write_stack(tmp_path / "s2.json", model_name="eq_S2_test", base_model_version=BASE.model_version, g_log10=g,
                   provenance={}, d=d, **co)
    of.write_stack(tmp_path / "s1.json", model_name="eq_S1_test", base_model_version=BASE.model_version, g_log10=g,
                   provenance={}, **co)
    v2, v1 = of.load_stack(tmp_path / "s2.json", BASE), of.load_stack(tmp_path / "s1.json", BASE)
    assert v2.meta["schema"] == of.STACK_SCHEMA_V2 and v2.d == d
    assert v1.meta["schema"] == of.STACK_SCHEMA and v1.d == 0.0 and "d" not in v1.meta["coefficients"]
    z = np.log(p) - np.log1p(-p)
    want2 = 0.5 * (1.0 + np.tanh(0.5 * (co["a"] + co["c"] * z + co["b"] * g + d * z * g)))   # the code's sigmoid form
    np.testing.assert_allclose(of.apply_stack(v2, p), want2, rtol=1e-12, atol=0)
    old = co["a"] + co["c"] * (np.log(p) - np.log1p(-p)) + co["b"] * g            # S1's code before E4, verbatim
    np.testing.assert_array_equal(of.apply_stack(v1, p), 0.5 * (1.0 + np.tanh(0.5 * old)))
    assert np.max(np.abs(of.apply_stack(v2, p) - of.apply_stack(v1, p))) > 1e-3      # the term does something


@pytest.mark.parametrize("schema,coefs", [(of.STACK_SCHEMA_V2, {"a": 0.1, "c": 1.0, "b": 0.2}),
                                          (of.STACK_SCHEMA, {"a": 0.1, "c": 1.0, "b": 0.2, "d": -0.07}),
                                          ("hazardpulse/earthquake-operational-stack/v3", {"a": 0.1, "c": 1.0, "b": 0.2})])
def test_a_stack_whose_coefficients_do_not_match_its_schema_is_refused(tmp_path, schema, coefs):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": schema, "model_name": "eq_bad", "base_model_version": BASE.model_version,
                             "coefficients": coefs, "g_log10": [float(x) for x in _g()]}), encoding="utf-8")
    with pytest.raises(of.OperationalArtifactError):
        of.load_stack(p, BASE)


def test_a_non_finite_interaction_is_refused(tmp_path):
    with pytest.raises(of.OperationalArtifactError, match="finite"):
        of.write_stack(tmp_path / "x.json", model_name="eq_x", base_model_version=BASE.model_version, a=0.0, c=1.0,
                       b=0.2, g_log10=_g(), provenance={}, d=float("nan"))


@pytest.mark.skipif(not SERVED_V1.exists(), reason="S1's stack not present")
def test_s1s_published_stack_still_loads_as_s1():
    meta = json.loads(SERVED_V1.read_text(encoding="utf-8"))
    stack = of.load_stack(SERVED_V1, types.SimpleNamespace(model_version=meta["base_model_version"]))
    assert stack.d == 0.0 and stack.model_version == of.stack_model_version(SERVED_V1)
