"""The ProbSevere input-format guard: which of the served model's inputs the live feed no longer carries.

NOAA changed the ProbSevere JSON on 2025-08-06 (25 -> 48 storm attributes; PS, VIL_DENSITY and MAXRC_ICECF
removed; MAXRC_EMISS and AVG_BEAM_HGT no longer in their documented string formats). The parser
(``hazardpulse.data.probsevere._parse_storms``) never reports a missing attribute: the core ones are read as
0.0, the others are left missing. So since that day the served model has received ``p_ps = 0`` and
``p_vil_density = 0`` for every live storm -- 196 and 475 of the 8,826 splits of the served +W trees
(``tornado_v3_w.json``; 171 and 486 of 9,212 in the fallback) -- and nothing said so. (The three string
attributes carry no split in either served model.)

This module does NOT change what the model receives (that is a model change, and needs the pre-registered
evaluation of docs/MODEL_IMPROVEMENT_LEDGER.md, branch T2). It makes the gap visible: for every run it maps
the newest file's attribute census (``probsevere.property_census``) onto the served model's inputs and
records which were absent and what the parser fed instead, into the forecast record, its provenance envelope
and, in plain words, the tornado page.
"""
from __future__ import annotations

from hazardpulse.tornado.v3_serving import plain_name

FORMAT_CHANGE = "2025-08-06"

# model input -> (where the parser reads it, in order of preference; what the parser feeds when all are absent)
# sources: ("properties", KEY) or ("models", NAME) -- the latter is feature["models"][NAME]["PROB"]
ZERO = "zero"            # read as 0.0 (probsevere._float default)
MISSING = "missing"      # left missing (NaN): the trees' missing-value branch
SOURCES: dict[str, tuple[tuple[tuple[str, str], ...], str]] = {
    "p_ps": ((("properties", "PS"),), ZERO),
    "p_ps_tor": ((("models", "probtor"), ("properties", "PROBTOR"), ("properties", "PS_TOR")), ZERO),
    "p_ps_hail": ((("models", "probhail"), ("properties", "PROBHAIL"), ("properties", "PS_HAIL")), ZERO),
    "p_ps_wind": ((("models", "probwind"), ("properties", "PROBWIND"), ("properties", "PS_WIND")), ZERO),
    "p_ps_severe": ((("models", "probsevere"), ("properties", "PROBSEVERE"), ("properties", "PS")), ZERO),
    "p_mucape": ((("properties", "MUCAPE"),), ZERO),
    "p_mlcape": ((("properties", "MLCAPE"),), ZERO),
    "p_mlcin": ((("properties", "MLCIN"),), ZERO),
    "p_ebshear": ((("properties", "EBSHEAR"),), ZERO),
    "p_srh01": ((("properties", "SRH01KM"), ("properties", "SRH01")), ZERO),
    "p_mesh": ((("properties", "MESH"),), ZERO),
    "p_vil_density": ((("properties", "VIL_DENSITY"), ("properties", "VILD")), ZERO),
    "p_flash_rate": ((("properties", "FLASH_RATE"), ("properties", "FLASHRATE")), ZERO),
    "p_flash_density": ((("properties", "FLASH_DENSITY"), ("properties", "FLASHDENSITY")), ZERO),
    "p_maxllaz": ((("properties", "MAXLLAZ"),), ZERO),
    "p_p98llaz": ((("properties", "P98LLAZ"),), ZERO),
    "p_p98mlaz": ((("properties", "P98MLAZ"),), ZERO),
    "p_lja": ((("properties", "LJA"),), ZERO),
    "p_size": ((("properties", "SIZE"),), ZERO),
    "p_motion_east": ((("properties", "MOTION_EAST"),), ZERO),
    "p_motion_south": ((("properties", "MOTION_SOUTH"),), ZERO),
    "p_pwat": ((("properties", "PWAT"),), MISSING),
    "p_cape_m10m30": ((("properties", "CAPE_M10M30"),), MISSING),
    "p_meanwind_1_3kmagl": ((("properties", "MEANWIND_1-3kmAGL"),), MISSING),
    "p_wetbulb_0c_hgt": ((("properties", "WETBULB_0C_HGT"),), MISSING),
    "p_avg_beam_hgt": ((("properties", "AVG_BEAM_HGT"),), MISSING),
    "p_maxrc_emiss": ((("properties", "MAXRC_EMISS"),), MISSING),
    "p_maxrc_icecf": ((("properties", "MAXRC_ICECF"),), MISSING),
}
# the string-valued attributes and the format the parser documents for them
STRING_FORMATS = {"p_maxrc_emiss": ("MAXRC_EMISS", "'2251Z 1.5%/min (weak)'"),
                  "p_maxrc_icecf": ("MAXRC_ICECF", "'2241Z 0.01/min (weak)'"),
                  "p_avg_beam_hgt": ("AVG_BEAM_HGT", "'4.09 kft / 1.25 km'")}


def _carried(census: dict, source: tuple[str, str]) -> int:
    where, key = source
    table = census.get("models" if where == "models" else "properties") or {}
    return int(table.get(key, 0) or 0)


def split_counts(payloads) -> dict[str, int]:
    """Per input, the most splits any of the given LightGBM payloads makes on it (0 = the model ignores it)."""
    out: dict[str, int] = {}
    for p in payloads:
        if not p:
            continue
        names = p["feature_names"]
        counts: dict[str, int] = {n: 0 for n in names}
        for t in p["trees"]:
            for j in t["feature"]:
                if j >= 0:
                    counts[names[j]] += 1
        for n, c in counts.items():
            out[n] = max(out.get(n, 0), c)
    return out


def input_gaps(census: dict | None, model_inputs, splits: dict[str, int] | None = None) -> dict:
    """The served model's inputs the newest ProbSevere file did not carry, per run.

    ``census``: ``probsevere.property_census`` of the newest file (None when the run read the day cache and
    the census is unknown). ``model_inputs``: the feature names of every served payload; ``splits``: their
    ``split_counts`` (recorded per item, so a reader can tell an input the trees use from one they ignore).
    Returns a record with ``absent`` (inputs no object carried; ``fed`` = "zero" or "missing"), ``partial``
    (carried by some objects only) and ``changed_format`` (present, but not in the documented string format,
    so the parser read the bare number)."""
    names = [n for n in dict.fromkeys(model_inputs) if n in SOURCES]
    splits = splits or {}
    if census is None:
        return {"census": "unavailable", "note": "the run read cached slot data; the feed format was not checked",
                "absent": [], "partial": [], "changed_format": []}
    n = int(census.get("n_objects") or 0)
    absent, partial, changed = [], [], []
    for name in names:
        sources, fed = SOURCES[name]
        carried = max((_carried(census, s) for s in sources), default=0)
        item = {"input": name, "label": plain_name(name), "source": [s[1] for s in sources], "fed": fed}
        if name in splits:
            item["model_splits"] = int(splits[name])
        if n and carried == 0:
            absent.append(item)
        elif n and carried < n:
            partial.append({**item, "n_objects_without": n - carried})
        elif n and name in STRING_FORMATS:
            key, documented = STRING_FORMATS[name]
            ok = int((census.get("documented_string_format") or {}).get(key, 0) or 0)
            if ok < carried:
                changed.append({**item, "source": [key], "fed": "the bare number", "documented_format": documented,
                                "n_objects_not_in_format": carried - ok})
    return {"census": "newest file", "feed_valid_time": census.get("valid_time"), "n_objects": n,
            "n_feed_attributes": int(census.get("n_attributes") or 0), "absent": absent, "partial": partial,
            "changed_format": changed}


def has_gaps(gaps: dict | None) -> bool:
    return bool(gaps) and bool(gaps.get("absent") or gaps.get("partial") or gaps.get("changed_format"))
