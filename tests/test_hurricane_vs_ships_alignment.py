"""The SHIPS-RII benchmark's alignment must pair a v8.2 case with NOAA's probability for the SAME
storm, the SAME synoptic time and the SAME event (30 kt in 24 h from t), or refuse to pair it.

Every check here has a negative control next to it -- the value a plausible wrong alignment
would produce -- so a regression that loosens the key (nearest time, first ATCF id of the storm,
the 25 or 35 kt record, a 30 kt record over another window) fails a test instead of quietly
moving the benchmark.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bm():
    name = "hp_benchmark_vs_ships_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "benchmark_hurricane_vs_ships.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# One cycle of an e-deck, in the archive's own layout: the 30 kt / 24 h record sits among
# records that share its tau (25/35/40 kt at 24 h), its dV (30 kt from 12 h), or its tech.
EDECK_AL04 = "\n".join([
    "AL, 04, 2022070112, RI, RIOD,  12, 120N,  700W,  11,  20,  65,    ,            0,           12, ",
    "AL, 04, 2022070112, RI, RIOD,  24, 120N,  700W,  44,  25,  70,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOD,  24, 120N,  700W,  31,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOD,  24, 120N,  700W,  57,  30,  75,    ,           12,           24, ",
    "AL, 04, 2022070112, RI, RIOD,  24, 120N,  700W,  19,  35,  80,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, DTOP,  24, 120N,  701W,  48,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOC,  24, 120N,  700W,  33,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOC,  24, 120N,  700W,  34,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOL,  24, 120N,  700W,  29,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, RI, RIOL,  24, 120N,  700W,  29,  30,  75,    ,            0,           24, ",
    "AL, 04, 2022070112, IN, IVCN,  24, 120N,  700W,  60,  30,    ,    4, ",
    "AL, 04, 2022070118, RI, RIOD,  24, 122N,  710W,  62,  30,  80,    ,            0,           24, ",
])
EDECK_EP04 = "\n".join([
    "EP, 04, 2022070212, RI, RIOD,  24, 112N,  860W,   7,  30, 115,    ,            0,           24, ",
])


def _table(bm):
    recs = []
    for text, sid in ((EDECK_AL04, "AL042022"), (EDECK_EP04, "EP042022")):
        r, _ = bm.parse_edeck_ri(text, sid)
        recs.extend(r)
    return bm.index_ri(recs)


def test_selects_only_the_30kt_24h_record_from_t0(bm):
    table, stats = _table(bm)
    riod = table[("AL042022", "2022070112")]["RIOD"]
    assert riod.prob_pct == 31            # 30 kt, tau 24, window 0-24
    assert riod.v_final_kt - riod.dv_kt == 45  # the real-time initial intensity it was issued on
    # negative controls: the neighbouring records a loose selector would pick instead
    t25, _ = bm.index_ri(bm.parse_edeck_ri(EDECK_AL04, "AL042022")[0], dv_kt=25)
    t35, _ = bm.index_ri(bm.parse_edeck_ri(EDECK_AL04, "AL042022")[0], dv_kt=35)
    w12, _ = bm.index_ri(bm.parse_edeck_ri(EDECK_AL04, "AL042022")[0], start_tau=12)
    assert {t25[("AL042022", "2022070112")]["RIOD"].prob_pct,
            t35[("AL042022", "2022070112")]["RIOD"].prob_pct,
            w12[("AL042022", "2022070112")]["RIOD"].prob_pct} == {44, 19, 57}
    assert stats == {"identical_repeats": 1, "conflicting_keys_dropped": 1, "cycles": 3}


def test_conflicting_repeats_are_dropped_identical_repeats_kept(bm):
    table, _ = _table(bm)
    techs = table[("AL042022", "2022070112")]
    assert "RIOC" not in techs            # 33 vs 34 for one key: no guessing which run was operational
    assert techs["RIOL"].prob_pct == 29   # the same value twice is one record
    assert techs["DTOP"].prob_pct == 48


def _fix(bm, t, atcf, status="HU", d2l=500.0, wind=50.0):
    return bm.IbFix(t, atcf, wind, status, d2l, 12.0, -70.0)


T1 = dt.datetime(2022, 7, 1, 12)
T2 = dt.datetime(2022, 7, 2, 12)


def _ib(bm):
    # Bonnie-like: one IBTrACS SID, ATCF id AL04 over the Atlantic, EP04 after crossing
    return {"SID1": {T1: _fix(bm, T1, "AL042022"), T2: _fix(bm, T2, "EP042022"),
                     dt.datetime(2022, 7, 1, 18): _fix(bm, dt.datetime(2022, 7, 1, 18), "AL042022")},
            "SIDW": {T1: _fix(bm, T1, "WP012022")},
            "SIDX": {T1: _fix(bm, T1, "")}}


def _case(sid, t):
    return {"storm_id": sid, "issue_time": t.strftime("%Y-%m-%d %H:%M:%S"), "ri_label_30kt": 0}


FILES = {"AL042022": 9, "EP042022": 1, "AL052022": 0}


def test_exact_storm_and_time_key_with_per_fix_atcf_id(bm):
    table, _ = _table(bm)
    ib = _ib(bm)
    a1 = bm.align_case(_case("SID1", T1), ib, table, FILES)
    a2 = bm.align_case(_case("SID1", T2), ib, table, FILES)
    assert (a1["reason"], a1["atcf_id"], a1["dtg"], a1["techs"]["RIOD"].prob_pct) == ("matched", "AL042022", "2022070112", 31)
    # after the basin crossing the case belongs to EP04: the storm's FIRST id would find nothing
    assert (a2["reason"], a2["atcf_id"], a2["techs"]["RIOD"].prob_pct) == ("matched", "EP042022", 7)
    assert ("AL042022", "2022070212") not in table


def test_no_nearest_time_matching(bm):
    table, _ = _table(bm)
    ib = _ib(bm)
    ib["SID1"][dt.datetime(2022, 7, 1, 6)] = _fix(bm, dt.datetime(2022, 7, 1, 6), "AL042022")
    # 06Z has no e-deck cycle; 12Z (6 h later) does: it must NOT be borrowed
    a = bm.align_case(_case("SID1", dt.datetime(2022, 7, 1, 6)), ib, table, FILES)
    assert a["reason"] == "no_ri_at_cycle" and a["techs"] == {}
    # and the 18Z case gets ITS cycle's value, not 12Z's
    b = bm.align_case(_case("SID1", dt.datetime(2022, 7, 1, 18)), ib, table, FILES)
    assert b["techs"]["RIOD"].prob_pct == 62


def test_refusal_reasons(bm):
    table, _ = _table(bm)
    ib = _ib(bm)
    ib["SID5"] = {T1: _fix(bm, T1, "AL052022")}
    ib["SID6"] = {T1: _fix(bm, T1, "AL062022")}
    ib["SID1"][dt.datetime(2022, 7, 1, 15)] = _fix(bm, dt.datetime(2022, 7, 1, 15), "AL042022")
    reasons = {
        "non_synoptic": _case("SID1", dt.datetime(2022, 7, 1, 15)),
        "no_ibtracs_fix": _case("SID1", dt.datetime(2022, 7, 3, 0)),
        "no_usa_atcf_id": _case("SIDX", T1),
        "jtwc_basin_no_public_ri": _case("SIDW", T1),
        "edeck_has_no_ri_records": _case("SID5", T1),
        "no_edeck_file": _case("SID6", T1),
    }
    for want, case in reasons.items():
        assert bm.align_case(case, ib, table, FILES)["reason"] == want, want
    # a cycle where NOAA's other aids exist but SHIPS-RII does not is not a SHIPS-RII match
    only_dtop, _ = bm.index_ri([r for r in bm.parse_edeck_ri(EDECK_AL04, "AL042022")[0] if r.tech == "DTOP"])
    assert bm.align_case(_case("SID1", T1), ib, only_dtop, FILES)["reason"] == "primary_tech_missing"


def test_duplicate_rows_collapse_to_one_case_and_disagreeing_rows_refuse(bm):
    a = {"storm_id": "S", "issue_time": "2022-07-01 12:00:00", "ri_label_30kt": 1, "x": 1.0}
    b = dict(a, issue_time="2022-07-01 18:00:00")
    uniq, mult = bm.unique_cases([a, dict(a), b])
    assert len(uniq) == 2 and mult[("S", "2022-07-01 12:00:00")] == 2
    with pytest.raises(ValueError):
        bm.unique_cases([a, dict(a, x=2.0)])


def test_filename_and_line_ids(bm):
    assert bm.storm_id_from_filename("eal092022.dat.gz") == "AL092022"
    assert bm.storm_id_from_filename("aep182023.dat.gz") == "EP182023"
    assert bm.storm_id_from_filename("eal092022.dat.gz.part") is None
    recs, stats = bm.parse_edeck_ri(EDECK_EP04.replace("EP, 04", "CP, 04"), "EP042022")
    assert recs[0].atcf_id == "CP042022" and stats["line_id_differs_from_file"] == 1


def test_ibtracs_reader_keeps_first_copy_and_reads_the_atcf_id(bm):
    header = ("SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,WMO_AGENCY,"
              "TRACK_TYPE,DIST2LAND,LANDFALL,IFLAG,USA_AGENCY,USA_ATCF_ID,USA_LAT,USA_LON,USA_RECORD,"
              "USA_STATUS,USA_WIND,USA_PRES")
    units = " ,Year, , , , , , ,degrees_north,degrees_east,kts,mb, , ,km,km, , , ,degrees_north,degrees_east, , ,kts,mb"
    row1 = "SID1,2022,1,NA,CS,BONNIE,2022-07-01 12:00:00,TS,12.0,-70.0,40,1005,hurdat_atl,main,250,250,O,hurdat_atl,AL042022,12.0,-70.0, ,TS,40,1005"
    row1b = row1.replace("AL042022", "ZZ992022")
    row2 = "SID9,2022,1,NA,CS,X,2022-07-01 12:00:00,TS,12.0,-70.0,40,1005,hurdat_atl,main,0,0,O,hurdat_atl,AL992022,12.0,-70.0, ,TS,40,1005"
    out = bm.parse_ibtracs_fixes([header, units, row1, row1b, row2], {"SID1"})
    assert list(out) == ["SID1"]
    fix = out["SID1"][T1]
    assert (fix.atcf_id, fix.usa_wind, fix.usa_status, fix.dist2land_km) == ("AL042022", 40.0, "TS", 250.0)


def test_rii_developmental_sample_mask(bm):
    def track(statuses, d2l):
        return {T1 + dt.timedelta(hours=3 * k): bm.IbFix(T1 + dt.timedelta(hours=3 * k), "AL042022", 50.0, s, d, 12.0, -70.0)
                for k, (s, d) in enumerate(zip(statuses, d2l))}

    over_water = track(["HU"] * 9, [300.0] * 9)
    assert bm.rii_sample_mask(over_water, T1) is True
    assert bm.rii_sample_mask(track(["HU"] * 9, [300.0] * 4 + [0.0] + [300.0] * 4), T1) is False   # landfall inside
    assert bm.rii_sample_mask(track(["HU"] * 8 + ["EX"], [300.0] * 9), T1) is False                # not tropical at t+24
    assert bm.rii_sample_mask(track(["HU"] * 8, [300.0] * 8), T1) is None                           # no t+24 fix


# --- anchors in the real archive: the tech identities the benchmark relies on -----------------

def _cache_root() -> Path:
    return Path(os.environ.get("HAZARDPULSE_CACHE_ROOT", ROOT / ".cache"))


def _real(bm, sid: str):
    path = _cache_root() / "atcf_adecks" / sid[-4:] / f"e{sid.lower()}.dat.gz"
    if not path.exists():
        pytest.skip(f"{path} not cached (python scripts/benchmark_hurricane_vs_ships.py --download)")
    return bm.parse_edeck_ri(gzip.decompress(path.read_bytes()).decode("utf-8", "replace"), sid)[0]


def test_riod_is_the_ships_rii_nhc_quoted(bm):
    """NHC Ian discussions #7 and #14 quote SHIPS-RII values; only RIOD carries them."""
    recs = _real(bm, "AL092022")

    def techs(dtg, dv, tau):
        t, _ = bm.index_ri(recs, tau=tau, dv_kt=dv, stop_tau=tau)
        return {k: v.prob_pct for k, v in t[("AL092022", dtg)].items()}

    q7 = techs("2022092418", 65, 72)
    q14a, q14b = techs("2022092612", 35, 24), techs("2022092612", 45, 36)
    assert (q7["RIOD"], q14a["RIOD"], q14b["RIOD"]) == (66, 73, 79)
    for other in ("RIOB", "RIOL", "RIOC"):   # negative control: no other tech fits all three quotes
        assert (q7[other], q14a[other], q14b[other]) != (66, 73, 79)


def test_dtop_is_dtops_nhc_quoted(bm):
    """NHC Helene discussion #10 (18Z 25 Sep 2024): DTOPS 'at least a 90 percent chance' of 35 kt/24 h."""
    recs = _real(bm, "AL092024")
    t, _ = bm.index_ri(recs, dv_kt=35)
    probs = {k: v.prob_pct for k, v in t[("AL092024", "2024092518")].items()}
    assert probs["DTOP"] >= 90
    assert all(v < 90 for k, v in probs.items() if k != "DTOP")
