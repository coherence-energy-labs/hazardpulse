"""The live earthquake scorer must hand its models the history they were trained on.

Regression for the 2026-10 finding: FEATURE_HISTORY_DAYS was 400, but the deep GRU
sequences (training ``_seq_one``: ``t0 = ref - 5 * 365 d``) and Block S
(5 x 365.25 d) read five years, so most served sequences were silently shorter than
anything seen in training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hazardpulse.earthquake.deep_serve import SEQUENCE_LOOKBACK_DAYS, DeepEQScorer

ROOT = Path(__file__).resolve().parents[1]
SEC_DAY = 86400.0


def _script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _catalog(ref: float, years: float = 6.0, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = np.sort(ref - rng.uniform(0, years * 365.25, n) * SEC_DAY)
    return SimpleNamespace(
        times=times,
        lats=35.0 + rng.normal(0, 0.5, n),
        lons=140.0 + rng.normal(0, 0.5, n),
        mags=rng.uniform(2.5, 6.0, n),
        depths=rng.uniform(0, 100, n),
    )


def _clip(cat, t_min: float):
    keep = cat.times >= t_min
    return SimpleNamespace(**{k: getattr(cat, k)[keep] for k in ("times", "lats", "lons", "mags", "depths")})


def _scorer(K: int, radius_km: float) -> DeepEQScorer:
    s = DeepEQScorer.__new__(DeepEQScorer)          # build_sequence needs only K and radius
    s.K, s.radius_km = K, radius_km
    return s


def test_feature_history_covers_every_served_lookback():
    eq = _script("hp_fse_hist_test", "scripts/fetch_and_score_earthquake.py")
    from hazardpulse.earthquake.definitive_model import BLOCK_S_LOOKBACK_DAYS

    assert eq.FEATURE_HISTORY_DAYS >= SEQUENCE_LOOKBACK_DAYS
    assert eq.FEATURE_HISTORY_DAYS >= BLOCK_S_LOOKBACK_DAYS


@pytest.mark.parametrize("K,radius", [(192, 500.0), (384, 100.0)])   # nowcast, short-term
def test_served_history_window_does_not_change_any_sequence(K, radius):
    """A catalog cut at FEATURE_HISTORY_DAYS must build exactly the sequences the full
    catalog builds -- i.e. the live fetch window never truncates a model input.

    The catalog is SPARSE (~60 events/yr near the cell, fewer than K in 400 d): in a
    busy cell the K most recent events all fall inside any window, so only a quiet
    cell -- most of the live grid -- can expose a short window."""
    eq = _script("hp_fse_hist_test2", "scripts/fetch_and_score_earthquake.py")
    ref = 1.79e9
    cat = _catalog(ref, n=360)
    served = _clip(cat, ref - eq.FEATURE_HISTORY_DAYS * SEC_DAY)
    sc = _scorer(K, radius)
    X_full, m_full = sc.build_sequence(cat, 35.0, 140.0, ref)
    X_srv, m_srv = sc.build_sequence(served, 35.0, 140.0, ref)
    # the 5-year window holds more events than 400 days would (the case that matters)
    in_400d = int((cat.times >= ref - 400 * SEC_DAY).sum())
    assert in_400d < m_full.sum() <= K
    np.testing.assert_array_equal(m_srv, m_full)
    np.testing.assert_allclose(X_srv, X_full)


def test_serving_sequence_matches_the_training_builder():
    """deep_serve.build_sequence must reproduce scripts/deep_sequence_earthquake._seq_one."""
    train = _script("hp_deepseq_train_test", "scripts/deep_sequence_earthquake.py")
    ref = 1.79e9
    cat = _catalog(ref, seed=5)
    K, R = 96, 300.0
    train._SEQ_CAT, train._SEQ_K, train._SEQ_R = cat, K, R
    X_tr, m_tr = train._seq_one((35.2, 139.8, ref))
    X_sv, m_sv = _scorer(K, R).build_sequence(cat, 35.2, 139.8, ref)
    np.testing.assert_array_equal(m_sv, m_tr)
    np.testing.assert_allclose(X_sv, X_tr, rtol=1e-5, atol=1e-5)    # training stores float32
