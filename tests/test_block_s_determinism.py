"""Block S features must be deterministic: same cell -> same features, always.

compute_block_s once drew its clustering subsample from the global unseeded RNG, so
the SAME (lat, lon, time) produced different features each call -- the live scorer
and the training pipeline disagreed (train/serve skew) and the result could not be
cached or replayed. This guards the seeded-RNG fix.

Needs the USGS catalog cache (.cache/earthquake/usgs); skipped where absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_CACHE = REPO / ".cache" / "earthquake" / "usgs"
pytestmark = pytest.mark.skipif(
    not _CACHE.is_dir() or not any(_CACHE.glob("*.csv")),
    reason="USGS catalog cache not present",
)


@pytest.fixture(scope="module")
def cat_and_catalog():
    import hazardpulse.data.earthquake as eq_data
    import hazardpulse.earthquake.definitive_model as dm
    from hazardpulse.data.usgs_fdsn import USGSCatalogIncompleteError

    # A unit test never downloads: with no downloader, load_usgs_catalog cannot repair
    # an incomplete cache (it refuses instead), so skip rather than hit the network.
    mp = pytest.MonkeyPatch()
    mp.setattr(eq_data, "_load_download_module", lambda: None)
    try:
        catalog = dm.load_usgs_catalog(min_year=2000, max_year=2024, min_mag=2.5)
    except USGSCatalogIncompleteError as exc:
        pytest.skip(f"USGS catalog cache incomplete (rebuild with the downloader): {exc}")
    finally:
        mp.undo()
    return dm, dm.CatalogArrays(catalog, verbose=False), catalog


def test_block_s_is_deterministic(cat_and_catalog):
    dm, cat, _ = cat_and_catalog
    a = np.asarray(dm.compute_block_s(36.5, -120.0, 1.65e9, cat))
    b = np.asarray(dm.compute_block_s(36.5, -120.0, 1.65e9, cat))
    assert np.array_equal(a, b), "same cell must yield identical Block S features"


def test_block_s_varies_by_cell(cat_and_catalog):
    dm, cat, _ = cat_and_catalog
    a = np.asarray(dm.compute_block_s(36.5, -120.0, 1.65e9, cat))
    b = np.asarray(dm.compute_block_s(34.0, -118.0, 1.65e9, cat))
    # different cells should differ somewhere (seeding is per-cell, not a constant)
    assert not np.array_equal(a, b)


def test_parallel_worker_matches_serial(cat_and_catalog):
    dm, cat, catalog = cat_and_catalog
    dm._par_worker_init()   # sets the per-worker globals in-process
    rng = np.random.RandomState(7)
    for _ in range(3):
        sm = {
            "latitude": float(rng.uniform(33, 41)),
            "longitude": float(rng.uniform(-123, -116)),
            "ref_epoch": 1.65e9 + float(rng.uniform(-2e7, 2e7)),
            "label": int(rng.uniform() < 0.3),
        }
        s = dm.compute_block_s(sm["latitude"], sm["longitude"], sm["ref_epoch"], cat)
        Xs = np.zeros(dm.N_FEAT_S, np.float32) if s is None else np.asarray(s, np.float32)
        Xc = np.nan_to_num(
            dm.compute_block_c(catalog, sm["latitude"], sm["longitude"], sm["ref_epoch"]),
            nan=0.0,
        ).astype(np.float32)
        Xx = np.nan_to_num(dm.compute_block_x(Xs, Xc), nan=0.0).astype(np.float32)
        Ws, Wc, Wx, _, _ = dm._par_worker_extract(sm)
        assert np.array_equal(Xs, Ws) and np.array_equal(Xc, Wc) and np.array_equal(Xx, Wx)
