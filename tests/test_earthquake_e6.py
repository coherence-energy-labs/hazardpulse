"""Amendment E6's input (docs/EARTHQUAKE_FORECAST_PROGRAM.md section 14): the 3x3 neighbourhood moment sum."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _e6():
    sys.path.insert(0, str(ROOT / "scripts" / "earthquake_program"))
    spec = importlib.util.spec_from_file_location("gear1_e6_t", ROOT / "scripts" / "earthquake_program" / "gear1_e6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_neighbourhood_wraps_in_longitude_and_stops_at_the_edge_rows():
    e6 = _e6()
    g = np.zeros((5, 6))
    g[2, 0] = 1.0                                          # western edge column: wraps to the eastern one
    out = e6.nbr3(g)
    hit = {(r, c) for r, c in zip(*np.nonzero(out))}
    assert hit == {(r, c) for r in (1, 2, 3) for c in (5, 0, 1)}
    assert np.all(out[list(zip(*hit))[0], list(zip(*hit))[1]] == 1.0)
    g = np.zeros((5, 6))
    g[0, 3] = 2.0                                          # bottom row: no row below it
    out = e6.nbr3(g)
    assert {(r, c) for r, c in zip(*np.nonzero(out))} == {(r, c) for r in (0, 1) for c in (2, 3, 4)}
    assert out.sum() == 2.0 * 6                            # 2 rows x 3 columns see it
    g = np.random.default_rng(1).random((5, 6))
    brute = np.array([[sum(g[rr, cc % 6] for rr in (r - 1, r, r + 1) if 0 <= rr < 5 for cc in (c - 1, c, c + 1))
                       for c in range(6)] for r in range(5)])
    assert np.allclose(e6.nbr3(g), brute)
