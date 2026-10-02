"""Replay the LIVE tier-1 scoring path on a real historical evening, from the cache.

Uses fetch_and_score_tornado.score_storms (the function cron runs) with the
installed payload, the matched HRRR analysis (latest at or before the step,
<= 3 h), and the corrected UTC labels to say which storms really produced a
tornado within 60 min / 40 km.
"""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("live", ROOT / "scripts" / "fetch_and_score_tornado.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)

from hazardpulse.data.hrrr import load_cached_hrrr  # noqa: E402
from hazardpulse.data.probsevere import load_cached_probsevere  # noqa: E402
from hazardpulse.tornado import definitive_model as dm  # noqa: E402
from hazardpulse.tornado.coherence_engine import compute_coherence_fields  # noqa: E402

date, upto = sys.argv[1], sys.argv[2]  # e.g. 20240427 2230
steps = [s for s in load_cached_probsevere(date)
         if dm.parse_probsevere_valid_time(s["valid_time"]).strftime("%H%M") <= upto]
t_obs = dm.parse_probsevere_valid_time(steps[-1]["valid_time"])
hour = dm.select_analysis_hour(t_obs, dm.HRRR_ANALYSIS_HOURS)
hrrr = load_cached_hrrr(date, hour=hour)
coh = compute_coherence_fields(hrrr, month=int(date[4:6]))
payload = live.load_pretrained_gbt()
print("model", live.MODEL_VERSION, "obs", t_obs.isoformat(), "analysis", f"{hour:02d}Z")
scored = live.score_storms(steps, hrrr, coh, None, t_obs.replace(tzinfo=None),
                           scoring_tier="tier1_ml", coherence_source="hrrr",
                           pretrained_gbt=payload)
live.refresh_risk_bands(scored)

reports = dm.load_spc_tornado_reports(ROOT / ".cache" / "spc" / "1950-2024_actual_tornadoes.csv")
window = dm.reports_in_label_window(reports, date)
y = np.array([dm.compute_label(s["lat"], s["lon"], t_obs, window) for s in scored], dtype=float)
p = np.array([s["tornado_probability"] for s in scored])
print(f"{len(scored)} storms; {int(y.sum())} produced a tornado within 60 min / 40 km")
print("AUC on this one time step:", round(dm.compute_auc(y, p), 4) if 0 < y.sum() < len(y) else "n/a")
print("mean p:", round(float(p.mean()), 5), " tier/calibrated:", scored[0]["scoring_tier"], scored[0]["model_scores"].get("calibrated"))
print("top 8:")
for s, yy in sorted(zip(scored, y), key=lambda t: -t[0]["tornado_probability"])[:8]:
    print(f"  {s['tornado_probability']:.3f} {s['risk_band']:9s} ({s['lat']:.2f},{s['lon']:.2f}) tornado_next_60min={int(yy)}")
print("bands:", {b: sum(1 for s in scored if s['risk_band'] == b) for b in ('minimal', 'low', 'moderate', 'high', 'very_high')})
