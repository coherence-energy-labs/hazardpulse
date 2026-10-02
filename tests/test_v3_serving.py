"""v3 serving: the live path must rebuild training's features and the final run's probabilities."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf
from hazardpulse.tornado import v3_serving as vs

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "results" / "models"


def test_nested_probabilities_are_served_coherent():
    out = vs.coherent(0.20, {"p30": 0.25, "p90": 0.15, "p_ef2": 0.30})
    assert out["p30"] == 0.20 and out["p90"] == 0.20 and out["p_ef2"] == 0.20
    assert sorted(out["coherence_clipped"]) == ["p30", "p90", "p_ef2"]
    ok = vs.coherent(0.20, {"p30": 0.10, "p90": 0.30, "p_ef2": 0.05})
    assert (ok["p30"], ok["p90"], ok["p_ef2"], ok["coherence_clipped"]) == (0.10, 0.30, 0.05, [])


@pytest.mark.skipif(not (MODELS / vs.FALLBACK_FILE).exists(), reason="v3 payload not exported")
def test_every_served_input_has_a_plain_name():
    for f in (vs.FALLBACK_FILE, vs.MAIN_FILE, *vs.PRODUCT_FILES.values()):
        if (MODELS / f).exists():
            names = lp.load(MODELS / f)["feature_names"]
            raw = [n for n in names if vs.plain_name(n) == n]
            assert raw == [], (f, raw)


STORE = Path(os.environ.get("HAZARDPULSE_FEATURE_STORE", "C:/Users/Josh/Projects/hazardpulse/.cache/feature_store_v3"))
PS_V2 = Path(os.environ.get("HAZARDPULSE_PROBSEVERE_V2_CACHE", "C:/Users/Josh/Projects/hazardpulse/.cache/probsevere_v2"))
DAY = "20250315"


@pytest.mark.skipif(not ((STORE / f"{DAY}.npz").exists() and (PS_V2 / f"{DAY}.json.gz").exists()
                         and (MODELS / vs.FALLBACK_FILE).exists() and (STORE / "_lab" / "final_X.npy").exists()),
                    reason="needs the local v3 feature store, ProbSevere cache and exported payload")
def test_live_path_rebuilds_the_stored_rows_and_the_final_probabilities():
    """Train/serve parity on real 2025 storms: the serving path (raw ProbSevere + HRRR -> features
    -> payload) must equal the feature-store row and the final run's probability for that row."""
    from hazardpulse.data import hrrr as H
    from hazardpulse.data.probsevere import load_cached_probsevere
    from hazardpulse.tornado import definitive_model as dm
    from hazardpulse.tornado.coherence_engine import compute_derived_hrrr

    steps = load_cached_probsevere(DAY, cache_dir=PS_V2)
    z = np.load(STORE / f"{DAY}.npz")
    suite = vs.V3Suite.load(MODELS)
    step_min = dm.probsevere_step_minutes(steps)
    idx = dm.index_storms_by_id(steps)
    # the day's offset inside the assembled final split, to find the final run's probabilities
    days = sorted(p.stem for p in STORE.glob("2025*.npz"))
    offset = sum(np.load(STORE / f"{d}.npz")["Y"].shape[0] for d in days[:days.index(DAY)])
    final_p = np.load(STORE / "_lab" / "preds" / "v3_primary_final.npy")
    payload = suite.fallback
    cols = [sf.FEATURE_NAMES.index(n) for n in payload["feature_names"]]
    rng = np.random.RandomState(0)
    rows = rng.choice(len(z["sid"]), size=40, replace=False)
    grids = {}
    for r in rows:
        si = int(z["step"][r])
        storm = next(s for s in steps[si]["storms"] if str(s.get("id")) == str(z["sid"][r]))
        hour = int(z["analysis"][r])
        h80 = None
        if hour >= 0:
            hh = hour if hour < 100 else hour - 100
            if hh not in grids:
                g = H.load_cached_hrrr(DAY, hour=hh)
                grids[hh] = (g, compute_derived_hrrr(g))
            h80 = grids[hh]
        hist = vs.storm_history(steps, storm.get("id"), si, id_index=idx)
        out = suite.score_storm(storm, hist, step_min, *(h80 if h80 else (None, None)), warning=None)
        fv = sf.feature_vector(storm, hist, step_min, h80=h80)
        np.testing.assert_array_equal(fv[cols].astype(np.float32), z["X"][r][cols])   # same features
        assert out["p60"] == pytest.approx(float(final_p[offset + r]), abs=2e-6)     # same probability
        assert out["model"] == "v3" and out["drivers"]
