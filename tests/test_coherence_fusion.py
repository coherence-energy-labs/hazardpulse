"""Coherence fusion: Boltzmann weights on verified information cost, with forgetting -- pinned before any outcome."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hazardpulse.verification import coherence_fusion as cf


def _cases(n=400, seed=0, good="A", bad="B"):
    """Two sources: ``good`` tracks the true probability, ``bad`` is noise around 0.5. One case a day."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = float(rng.uniform(0.02, 0.6))
        y = int(rng.uniform() < q)
        out.append({"t": float(i), "probs": {good: q, bad: float(rng.uniform(0.3, 0.7))}, "y": y})
    return out


def _ll(cases, fused, start=0):
    return float(np.mean([cf.log_loss(f, c["y"]) for c, f in list(zip(cases, fused))[start:]]))


def test_the_coherent_source_earns_the_weight_and_the_fusion_follows_it():
    cases = _cases()
    fused = cf.run(cases, cf.FusionConfig(eta=1.0, tau_days=None), ["A", "B"], verify_after_days=1.0)
    f = cf.CoherenceFusion(cf.FusionConfig(eta=1.0, tau_days=None))
    for c, p in zip(cases, fused):
        f.verify(c["t"] + 1, c["probs"], p, c["y"], ["A", "B"])
    w = f.weights(1e6, ["A", "B"])
    assert w["A"] > 0.99
    good_only = [c["probs"]["A"] for c in cases]
    assert _ll(cases, fused, 100) == pytest.approx(_ll(cases, good_only, 100), abs=0.01)
    eq = cf.run(cases, cf.FusionConfig(equal=True), ["A", "B"], 1.0)
    assert _ll(cases, fused, 100) < _ll(cases, eq, 100)


def test_weights_are_boltzmann_in_the_energy_gap():
    f = cf.CoherenceFusion(cf.FusionConfig(eta=0.5, tau_days=None))
    f.energy = {"A": 10.0, "B": 12.0}
    f.t_days = 0.0
    w = f.weights(0.0, ["A", "B"])
    assert w["B"] / w["A"] == pytest.approx(math.exp(-0.5 * 2.0))


def test_a_silent_source_pays_the_fusions_own_loss():
    f = cf.CoherenceFusion(cf.FusionConfig(eta=1.0, tau_days=None))
    f.verify(1.0, {"A": 0.9}, fused=0.8, y=1, sources=["A", "B"])
    assert f.energy["A"] == pytest.approx(-math.log(0.9)) and f.energy["B"] == pytest.approx(-math.log(0.8))


def test_forgetting_lets_a_source_recover_and_never_forgetting_does_not():
    f_mem = cf.CoherenceFusion(cf.FusionConfig(eta=1.0, tau_days=None))
    f_fgt = cf.CoherenceFusion(cf.FusionConfig(eta=1.0, tau_days=5.0))
    for f in (f_mem, f_fgt):
        f.energy, f.t_days = {"A": 0.0, "B": 20.0}, 0.0
    assert f_mem.weights(100.0, ["A", "B"])["B"] < 1e-8
    assert f_fgt.weights(100.0, ["A", "B"])["B"] == pytest.approx(0.5, abs=1e-3)       # 20 e^-20 ~ 0


def test_pools_and_alignment():
    f = cf.CoherenceFusion(cf.FusionConfig(equal=True))
    p_lin, _ = f.fuse(0.0, {"A": 0.1, "B": 0.5})
    assert p_lin == pytest.approx(0.3)
    p_log, _ = cf.CoherenceFusion(cf.FusionConfig(equal=True, pool="log")).fuse(0.0, {"A": 0.1, "B": 0.5})
    assert p_log == pytest.approx(1 / (1 + 3.0))                                        # geometric odds 1/3
    p_al, _ = cf.CoherenceFusion(cf.FusionConfig(equal=True, pool="log", align=2.0)).fuse(0.0, {"A": 0.1, "B": 0.1})
    assert p_al == pytest.approx(1 / (1 + 81.0))                                        # agreement amplified
    assert cf.CoherenceFusion(cf.FusionConfig()).fuse(0.0, {"A": math.nan}) == (None, {})


def test_a_fused_forecast_never_sees_an_outcome_before_it_verifies():
    cases = _cases(n=120, seed=3)
    cfg = cf.FusionConfig(eta=0.5, tau_days=30.0, pool="log", align=1.2)
    a = cf.run(cases, cfg, ["A", "B"], verify_after_days=1.0)
    flipped = [dict(c, y=1 - c["y"]) if c["t"] >= 60 else c for c in cases]
    b = cf.run(flipped, cfg, ["A", "B"], verify_after_days=1.0)
    assert a[:61] == b[:61]                       # the case at t=60 verifies at t=61: nothing before it can know
    assert a[62:] != b[62:]                       # and the check can fail
