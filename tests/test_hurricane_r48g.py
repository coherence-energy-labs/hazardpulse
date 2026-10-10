"""R48g's inputs (TC1 program, amendment 8): the guidance's change to any lead, by the builder H8's 24-h inputs use."""
from __future__ import annotations

import datetime as dt
import math
import random

from hazardpulse.hurricane import atcf
from hazardpulse.hurricane import ri_v9_features as fx


def _table(seed: int, taus=(-24, -12, 0, 12, 24, 36, 48), drop: float = 0.25):
    rng = random.Random(seed)
    techs = ("CARQ", "OFCL", *fx.SPREAD_AIDS)
    t = {}
    for tech in techs:
        for tau in taus:
            if tech == "CARQ" and tau > 0:
                continue
            if tech != "CARQ" and tau <= 0:
                continue
            if tech != "CARQ" and rng.random() < drop:
                continue
            t[(tech, tau)] = fx.Fix(float(rng.randint(25, 140)), 980.0, 21.5, -60.0)
    return t


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_at_24_h_the_builder_is_adeck_features_exactly():
    """Control 1 in miniature: at tau 24 (30 kt) the new builder returns adeck_features' own guidance inputs."""
    names = fx.aid_change_names(24, 30)
    assert set(names) == {n for n in fx.O_NAMES if n.startswith("dv24_") or n == "frac_ge30"} | set(fx.H_NAMES)
    for seed in range(200):
        tab = _table(seed)
        old, new = fx.adeck_features(tab, "AL"), fx.aid_change_features(tab, 24, 30)
        assert set(new) == set(names)
        assert all(_same(old[n], new[n]) for n in names), seed


def test_at_48_h_it_reads_the_48_h_forecasts_and_nothing_at_24_h():
    tab = {("CARQ", 0): fx.Fix(50.0, None, 20.0, -50.0), ("DSHP", 24): fx.Fix(60.0, None, None, None),
           ("DSHP", 48): fx.Fix(75.0, None, None, None), ("LGEM", 48): fx.Fix(70.0, None, None, None),
           ("HWFI", 48): fx.Fix(110.0, None, None, None), ("OFCL", 48): fx.Fix(80.0, None, None, None),
           ("OFCL", 36): fx.Fix(72.0, None, None, None), ("OFCL", 24): fx.Fix(65.0, None, None, None)}
    f = fx.aid_change_features(tab, 48, 50)
    assert f["dv48_DSHP"] == 25.0 and f["dv48_LGEM"] == 20.0 and math.isnan(f["dv48_IVCN"])
    assert f["dv48_regional_mean"] == 60.0 and f["dv48_regional_max"] == 60.0 and math.isnan(f["dv48_global_mean"])
    assert f["frac48_ge50"] == 1 / 3                          # DSHP +25, LGEM +20, HWFI +60
    assert f["dv48_spread"] > 0 and f["ofcl_dv48"] == 30.0 and f["ofcl_dv36"] == 22.0
    assert set(f) == set(fx.aid_change_names(48, 50))


def _r48():
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("hurricane_r48_t", root / "scripts" / "hurricane_r48.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DECK = """AL, 09, 2026100912, 03, CARQ,   0, 250N,  850W,  50,  995, XX
AL, 09, 2026100912, 03, DSHP,  24, 260N,  860W,  62, 0, XX
AL, 09, 2026100912, 03, DSHP,  48, 270N,  870W,  77, 0, XX
AL, 09, 2026100912, 03, OFCL,  12, 255N,  855W,  55, 0, XX
AL, 09, 2026100912, 03, OFCL,  24, 260N,  860W,  60, 0, XX
AL, 09, 2026100912, 03, OFCL,  36, 265N,  865W,  68, 0, XX
AL, 09, 2026100912, 03, OFCL,  48, 270N,  870W,  75, 0, XX
"""


def test_control_1_stops_the_run_on_a_stored_input_the_builder_does_not_reproduce(tmp_path):
    """add_g48 checks every row's stored 24-h guidance inputs against the builder before writing R48g's: a row built
    the same way passes and gets its 48-h inputs, one stored value off by 1 kt stops the run."""
    import gzip
    import pytest
    r48 = _r48()
    p = tmp_path / "aal092026.dat.gz"
    p.write_bytes(gzip.compress(DECK.encode()))
    stored = fx.adeck_features(fx.cycle_table(atcf.parse_atcf_deck(DECK), dt.datetime(2026, 10, 9, 12)), "AL")
    assert stored["dv24_DSHP"] == 12.0 and stored["ofcl_dv12"] == 5.0          # the deck parses as written
    good = {"atcf_id": "AL092026", "dtg": "2026100912", "f": dict(stored)}
    cov = r48.add_g48([good], lambda aid: p, {"AL092026|2026100912": 23.5})
    assert cov["control_mismatches"] == 0 and cov["control_values_checked"] == len(fx.aid_change_names(24, 30))
    assert good["f"]["dv48_DSHP"] == 27.0 and good["f"]["ofcl_dv48"] == 25.0 and good["f"]["ofcl_dv36"] == 18.0
    assert good["f"]["tc1o_dv48"] == 23.5
    bad = {"atcf_id": "AL092026", "dtg": "2026100912", "f": dict(stored, dv24_DSHP=13.0)}
    with pytest.raises(SystemExit, match="control 1 failed"):
        r48.add_g48([bad], lambda aid: p, {})


def test_r48g_constraints_mirror_the_24_h_inputs():
    r48 = _r48()
    names = ["v0", "dv24_DSHP", "dv24_spread", *r48._g_names()]
    mono = r48._g_mono(names)
    assert len(mono) == len(names) + 1 and mono[-1] == -1
    by = dict(zip(names, mono))
    assert by["dv48_spread"] == 0 and by["dv48_DSHP"] == 1 and by["tc1o_dv48"] == 1 and by["ofcl_dv36"] == 1
    assert by["dv24_DSHP"] == 1 and by["dv24_spread"] == 0           # H8's own, unchanged
    assert len(r48._g_names()) == 14


def test_no_analysis_intensity_means_no_change_is_defined():
    tab = _table(3)
    tab[("CARQ", 0)] = fx.Fix(None, 990.0, 20.0, -50.0)
    assert all(math.isnan(v) for v in fx.aid_change_features(tab, 48, 50).values())
