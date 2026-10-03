"""The earthquake program's month-block scorer (scripts/earthquake_program/common.py).

The selection rule and the final table rest on ``SplitScorer``: at the identity resample
it must reproduce the tie-aware AUC, the Bernoulli information gain and the Brier score
computed directly, with and without a cell mask; and the split/issue-time rules must keep
every outcome window inside its split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.core.metrics import roc_auc

PROGRAM = Path(__file__).resolve().parents[1] / "scripts" / "earthquake_program"


@pytest.fixture(scope="module")
def C():
    sys.path.insert(0, str(PROGRAM))
    spec = importlib.util.spec_from_file_location("eq_program_common_test", PROGRAM / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("masked", [False, True])
def test_split_scorer_reproduces_the_direct_metrics(C, masked):
    rng = np.random.default_rng(1)
    n_issue, n_cell = 40, 400
    issue = np.array([C.ISSUE_ORIGIN.timestamp() + 7 * 86400 * k for k in range(n_issue)])
    P = rng.uniform(0, 0.05, (n_issue, n_cell))
    P[:, :60] = np.round(P[:, :60], 2)                      # heavy ties
    Y = rng.random((n_issue, n_cell)) < 2 * P
    mask = (rng.random((n_issue, n_cell)) < 0.3) if masked else None
    p0 = 0.01
    sc = C.SplitScorer(issue, Y, p0=p0, mask=mask, n_boot=100)
    res = sc.summary(sc.stats(P))
    m = np.ones_like(Y) if mask is None else mask
    pc = np.clip(P, 1e-12, 1 - 1e-12)
    ll = np.sum(np.where(Y, np.log(pc), np.log1p(-pc))[m])
    ll0 = np.sum(np.where(Y, np.log(p0), np.log1p(-p0))[m])
    assert res["auc"]["value"] == pytest.approx(roc_auc(Y[m].astype(float), P[m]), abs=1e-12)
    assert res["ig_per_target"]["value"] == pytest.approx((ll - ll0) / Y[m].sum(), rel=1e-10)
    assert res["brier"]["value"] == pytest.approx(np.mean(((P - Y) ** 2)[m]), rel=1e-12)
    lo, hi = res["auc"]["ci95"]
    assert lo <= res["auc"]["value"] <= hi


def test_paired_difference_of_a_monotone_rescaling_has_zero_auc_change(C):
    rng = np.random.default_rng(2)
    issue = np.array([C.ISSUE_ORIGIN.timestamp() + 7 * 86400 * k for k in range(30)])
    P = rng.uniform(0, 0.05, (30, 300))
    Y = rng.random((30, 300)) < 2 * P
    sc = C.SplitScorer(issue, Y, p0=0.01, n_boot=100)
    pr = sc.paired(sc.stats(P), sc.stats(P * 0.5))
    assert pr["auc"]["diff"] == 0.0
    assert pr["auc"]["ci95"] == [0.0, 0.0]
    assert pr["ig_per_target"]["diff"] > 0          # halving a calibrated-ish forecast loses log score


def test_every_issue_window_stays_inside_its_split(C):
    issue = C.all_issue_times()
    for t in issue:
        name = C.split_of(t)
        lo, hi = C.SPLITS[name]
        assert lo.timestamp() <= t and t + 30 * 86400 <= hi.timestamp()
    counts = {n: int(sum(C.split_of(t) == n for t in issue)) for n in C.SPLITS}
    assert counts == {"fit": 674, "choose": 153, "dev": 100, "final": 153}
