"""Serve the v3 tornado suite for live ProbSevere storms -- the same features training saw.

Per storm observation, from the payloads exported by scripts/audit_20261001/export_v3_payload.py:

* ``p60``: P(a tornado from THIS storm within 60 min) -- the +W model (NWS warning state as an
  input) when the live warnings feed answered, else the no-warnings model (``tornado_v3.json``);
* ``p30`` / ``p90`` / ``p_ef2``: the same event within 30 / 90 min, and an EF2+ tornado within
  60 min (+W models; omitted, never guessed, when the warnings feed is down);
* ``band``: the Venn-Abers interval [lo, hi] of the 60-min model (fitted on leave-one-year-out
  scores), when its payload carries one;
* ``drivers``: the inputs that moved this storm's score most (path attribution, exact and
  additive in log-odds), in plain words.

The four probabilities describe nested events, so they are served COHERENT: p30 <= p60 <= p90 and
p_ef2 <= p60 (each product is clipped into the range its nesting implies; how often that bites is
measured on 2025 and recorded in the payload provenance by the exporter's caller).

Features come from ``storm_features.feature_vector`` with the storm's history from
``definitive_model.build_storm_history`` and the day's ProbSevere cadence -- the functions that
built every training row -- and the 80 km HRRR analysis (grids + derived) the scorer loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado import lgbm_payload as lp
from hazardpulse.tornado import storm_features as sf

MAIN_FILE = "tornado_v3_w.json"
FALLBACK_FILE = "tornado_v3.json"
PRODUCT_FILES = {"p30": "tornado_v3_w_30.json", "p90": "tornado_v3_w_90.json", "p_ef2": "tornado_v3_w_ef2.json"}
W_NAMES = ("w_tor_warning_active", "w_minutes_since_issue")

_PLAIN = {
    "p_ps": "ProbSevere severe probability", "p_ps_tor": "NOAA ProbTor", "p_ps_hail": "NOAA ProbHail",
    "p_ps_wind": "NOAA ProbWind", "p_ps_severe": "ProbSevere composite", "p_mucape": "most-unstable CAPE",
    "p_mlcape": "mixed-layer CAPE", "p_mlcin": "mixed-layer CIN", "p_ebshear": "effective bulk shear",
    "p_srh01": "0-1 km helicity", "p_mesh": "max hail size (MESH)", "p_vil_density": "VIL density",
    "p_flash_rate": "lightning flash rate", "p_flash_density": "lightning flash density",
    "p_maxllaz": "low-level rotation (max azimuthal shear)", "p_p98llaz": "low-level rotation (98th pct)",
    "p_p98mlaz": "mid-level rotation (98th pct)", "p_lja": "lightning jump", "p_size": "storm size",
    "p_motion_east": "storm motion (east)", "p_motion_south": "storm motion (south)",
    "p_pwat": "precipitable water", "p_cape_m10m30": "CAPE in the -10..-30 C layer",
    "p_meanwind_1_3kmagl": "1-3 km mean wind", "p_wetbulb_0c_hgt": "wet-bulb zero height",
    "p_avg_beam_hgt": "radar beam height", "p_maxrc_emiss": "satellite cloud-top growth",
    "p_maxrc_icecf": "satellite glaciation rate",
    "e_age_min": "storm age", "e_n_steps": "track length", "e_rot_sustained_min": "sustained rotation (minutes)",
    "e_motion_speed": "storm speed", "e_motion_dir_deg": "storm heading",
    "h80_hrrr_cape": "HRRR CAPE", "h80_hrrr_cin": "HRRR CIN", "h80_hrrr_srh01": "HRRR 0-1 km helicity",
    "h80_hrrr_srh03": "HRRR 0-3 km helicity", "h80_hrrr_shear06": "HRRR 0-6 km shear",
    "h80_hrrr_shear01": "HRRR 0-1 km shear", "h80_hrrr_pwat": "HRRR precipitable water",
    "h80_hrrr_t2m": "HRRR 2 m temperature", "h80_hrrr_td2m": "HRRR 2 m dew point",
    "h80_hrrr_refc": "HRRR composite reflectivity", "h80_hrrr_stp": "HRRR significant tornado parameter",
    "h80_hrrr_srh05_est": "HRRR 0-500 m helicity",
    "w_tor_warning_active": "NWS tornado warning in effect", "w_minutes_since_issue": "minutes since the warning",
}
_TREND = {"delta": "change", "max": "track maximum", "mean": "track mean", "slope": "trend"}
_SERIES = {"maxllaz": "low-level rotation", "p98llaz": "low-level rotation (98th pct)", "ps_tor": "ProbTor",
           "ps": "ProbSevere", "mesh": "hail size", "flash_rate": "flash rate", "size": "storm size"}


def plain_name(name: str) -> str:
    if name in _PLAIN:
        return _PLAIN[name]
    if name.startswith("e_"):
        for series, words in _SERIES.items():
            for suffix, how in _TREND.items():
                if name == f"e_{series}_{suffix}":
                    return f"{words}, {how}"
    return name


@dataclass
class V3Suite:
    main: dict | None = None
    fallback: dict | None = None
    products: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, models_dir: str | Path) -> "V3Suite":
        d = Path(models_dir)

        def opt(name):
            return lp.load(d / name) if (d / name).exists() else None
        return cls(main=opt(MAIN_FILE), fallback=opt(FALLBACK_FILE),
                   products={k: p for k, f in PRODUCT_FILES.items() if (p := opt(f)) is not None})

    @property
    def available(self) -> bool:
        return self.main is not None or self.fallback is not None

    def versions(self) -> dict[str, str]:
        out = {}
        for key, p in (("p60_w", self.main), ("p60", self.fallback), *self.products.items()):
            if p is not None:
                out[key] = lp.model_version(p)
        return out

    @staticmethod
    def _columns(payload: dict, fv: np.ndarray, w: tuple[float, float] | None) -> np.ndarray:
        idx = {n: i for i, n in enumerate(sf.FEATURE_NAMES)}
        row = []
        for n in payload["feature_names"]:
            if n in idx:
                row.append(fv[idx[n]])
            elif n in W_NAMES:
                if w is None:
                    raise ValueError("this payload needs the NWS warning state")
                row.append(w[W_NAMES.index(n)])
            else:
                raise ValueError(f"unknown model input {n!r}")
        return np.asarray([row], np.float64)

    def score_storm(self, storm: dict, history: list[dict], step_minutes: float,
                    hrrr: dict | None, derived_hrrr: dict | None,
                    warning: tuple[float, float] | None, n_drivers: int = 5) -> dict:
        """One storm. ``warning`` = (active 0/1, minutes since issue or NaN), or None when the live
        warnings feed did not answer (then the no-warnings model serves and products are omitted)."""
        h80 = (hrrr, derived_hrrr) if hrrr is not None and derived_hrrr is not None else None
        fv = sf.feature_vector(storm, history, step_minutes, h80=h80, c80=None, fields9=None)
        use_w = warning is not None and self.main is not None
        model = self.main if use_w else self.fallback
        if model is None:
            raise ValueError("no usable v3 payload (the +W model needs the warnings feed; no fallback loaded)")
        X = self._columns(model, fv, warning if use_w else None)
        p60 = float(lp.predict_proba(model, X)[0])
        lo, hi = (float(v[0]) for v in lp.predict_interval(model, X))
        out: dict = {"p60": p60, "model": "v3_w" if use_w else "v3", "model_version": lp.model_version(model),
                     "band": [lo, hi] if np.isfinite(lo) and np.isfinite(hi) else None,
                     "hrrr_used": h80 is not None, "warning": None if warning is None else
                     {"active": bool(warning[0] > 0.5),
                      "minutes_since_issue": None if not np.isfinite(warning[1]) else float(warning[1])}}
        try:
            bias, contrib = lp.contributions(model, X)
            order = np.argsort(-np.abs(contrib[0]))[:n_drivers]
            out["drivers"] = [{"input": model["feature_names"][j], "label": plain_name(model["feature_names"][j]),
                               "value": None if not np.isfinite(X[0, j]) else float(X[0, j]),
                               "log_odds": round(float(contrib[0, j]), 4)} for j in order if contrib[0, j] != 0]
        except ValueError:
            out["drivers"] = []
        if use_w:
            raw = {}
            for key, payload in self.products.items():
                raw[key] = float(lp.predict_proba(payload, self._columns(payload, fv, warning))[0])
            out.update(coherent(p60, raw))
        return out


def coherent(p60: float, raw: dict[str, float]) -> dict:
    """Nested events, served coherent: p30 <= p60 <= p90 and p_ef2 <= p60. Returns the clipped
    products and which were clipped (so the rate can be reported)."""
    out: dict = {}
    clipped = []
    if "p30" in raw:
        out["p30"] = min(raw["p30"], p60)
        if raw["p30"] > p60:
            clipped.append("p30")
    if "p90" in raw:
        out["p90"] = max(raw["p90"], p60)
        if raw["p90"] < p60:
            clipped.append("p90")
    if "p_ef2" in raw:
        out["p_ef2"] = min(raw["p_ef2"], p60)
        if raw["p_ef2"] > p60:
            clipped.append("p_ef2")
    out["coherence_clipped"] = clipped
    return out


def storm_history(time_steps: list[dict], storm_id, latest_idx: int, id_index=None) -> list[dict]:
    """The training definition of a storm's history (bounded lookback, same id)."""
    return dm.build_storm_history(time_steps, storm_id, latest_idx, id_index=id_index)
