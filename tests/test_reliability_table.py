"""The fine 2025 reliability table: bins partition the predictions; intervals are Jeffreys."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rt():
    spec = importlib.util.spec_from_file_location(
        "reliability_table_t", ROOT / "scripts" / "audit_20261001" / "reliability_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bins_partition_the_predictions_and_the_top_edge_is_closed(rt):
    p = np.array([0.0, 0.0009, 0.001, 0.0015, 0.05, 0.4999, 0.5, 1.0])
    y = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=float)
    rows = rt.table(p, y)
    assert sum(r["n"] for r in rows) == p.size and sum(r["pos"] for r in rows) == int(y.sum())
    by = {(r["lo"], r["hi"]): r for r in rows}
    assert by[(0.0, 0.001)]["n"] == 2                  # 0.001 itself starts the next bin
    assert by[(0.001, 0.002)]["n"] == 2 and by[(0.001, 0.002)]["pos"] == 1
    assert by[(0.05, 0.1)]["n"] == 1
    assert by[(0.3, 0.5)]["n"] == 1 and by[(0.5, 1.0)]["n"] == 2   # 1.0 belongs to the last bin
    assert by[(0.002, 0.005)]["n"] == 0 and by[(0.002, 0.005)]["observed"] is None


def test_the_interval_is_jeffreys_and_a_zero_count_bin_starts_at_zero(rt):
    p = np.full(1000, 0.015)
    y = np.zeros(1000)
    y[:7] = 1
    r = next(r for r in rt.table(p, y) if r["n"])
    assert r["observed"] == pytest.approx(0.007)
    assert r["observed_ci"][0] == pytest.approx(beta.ppf(0.025, 7.5, 993.5))
    assert r["observed_ci"][1] == pytest.approx(beta.ppf(0.975, 7.5, 993.5))
    none = next(r for r in rt.table(p, np.zeros(1000)) if r["n"])
    assert none["observed_ci"][0] == 0.0 and 0 < none["observed_ci"][1] < 0.01


def test_bad_inputs_are_refused(rt):
    with pytest.raises(ValueError):
        rt.table(np.array([0.1, np.nan]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        rt.table(np.array([0.1, 1.2]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        rt.table(np.array([0.1]), np.array([0.0, 1.0]))
