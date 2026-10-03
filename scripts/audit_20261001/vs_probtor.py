"""Does the served tornado model beat NOAA's own ProbTor on the same storms?

Rebuilds the 2024 test population exactly as definitive_model does (same labels,
same causal-analysis exclusion), scores it with the SERVED payload, and compares
against ProbSevere's ProbTor (ps_tor, 0-100: NOAA's probability of a tornado in
the next 60 min) and the any-severe ProbSevere score (ps). Day-clustered CIs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from hazardpulse.data.hrrr import load_cached_hrrr
from hazardpulse.data.probsevere import load_cached_probsevere, scan_probsevere_cache
from hazardpulse.tornado import definitive_model as dm
from hazardpulse.tornado.coherence_engine import compute_coherence_fields, compute_derived_hrrr

spc = Path(sys.argv[1])
payload_path = Path(sys.argv[2])
n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
t0 = time.time()

reports = dm.load_spc_tornado_reports(spc)
gbt, norm, names = dm.load_model(payload_path)
cal = json.loads(payload_path.read_text(encoding="utf-8"))["calibration"]

rows, ys, pstor, ps, days, missing_pstor = [], [], [], [], [], 0
test_days = [d for d in scan_probsevere_cache() if dm.date_in_range(d, dm.TEST_START, dm.TEST_END)]
for k, d in enumerate(test_days):
    steps = load_cached_probsevere(d)
    if not steps:
        continue
    month = int(d[4:6])
    analyses = {}
    for h in dm.HRRR_ANALYSIS_HOURS:
        g = load_cached_hrrr(d, hour=h)
        if g is not None:
            analyses[h] = (g, compute_derived_hrrr(g), compute_coherence_fields(g, month=month))
    window = dm.reports_in_label_window(reports, d)
    step_min = dm.probsevere_step_minutes(steps)
    idx = dm.index_storms_by_id(steps)
    hours = sorted(analyses)
    for si, ts in enumerate(steps):
        t = dm.parse_probsevere_valid_time(ts.get("valid_time", ""))
        h = dm.select_analysis_hour(t, hours)
        an = analyses.get(h) if h is not None else None
        for storm in ts.get("storms", []):
            y = dm.compute_label(float(storm.get("lat", 0)), float(storm.get("lon", 0)), t, window)
            if y == -1 or an is None:
                continue
            g, der, coh = an
            hist = dm.build_storm_history(steps, storm.get("id"), si, id_index=idx)
            feat = np.concatenate([
                dm.extract_block_p(storm), dm.extract_block_e(storm, hist, step_min),
                dm.extract_block_h(storm, g, der), dm.extract_block_c(storm, coh, g),
            ])
            rows.append(feat)
            ys.append(float(y))
            if storm.get("ps_tor") is None:
                missing_pstor += 1
            pstor.append(float(storm.get("ps_tor") or 0.0) / 100.0)
            ps.append(float(storm.get("ps") or 0.0) / 100.0)
            days.append(int(d))
    if (k + 1) % 50 == 0:
        print(f"  {k + 1}/{len(test_days)} days, {len(ys)} samples, {time.time() - t0:.0f}s", flush=True)

X = np.stack(rows).astype(np.float32)
y = np.array(ys)
g = np.array(days)
pstor, ps = np.array(pstor), np.array(ps)
p_model = dm.apply_calibration(gbt.decision_function(norm.transform(X)), cal)
print(f"\nn={len(y)} positives={int(y.sum())} base_rate={y.mean():.5f} missing ps_tor={missing_pstor} "
      f"({time.time() - t0:.0f}s)")


def report(name, p):
    ci = dm.cluster_bootstrap_auc_ci(y, p, g, n_boot=n_boot)
    print(f"  {name:28s} AUC {dm.compute_auc(y, p):.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  "
          f"Brier {dm.compute_brier(y, p):.6f}  BSS {dm.compute_bss(y, p):+.4f}  mean p {p.mean():.5f}")


report("served model (calibrated)", p_model)
report("NOAA ProbTor (ps_tor/100)", pstor)
report("ProbSevere any-severe (ps/100)", ps)
cal_tor = dm.fit_platt(np.log(np.clip(pstor, 1e-4, 1 - 1e-4) / (1 - np.clip(pstor, 1e-4, 1 - 1e-4))), y)
print("  (ProbTor re-calibrated IN-SAMPLE on test -- an upper bound for its Brier, not a fair forecast)")
lt = np.log(np.clip(pstor, 1e-4, 1 - 1e-4) / (1 - np.clip(pstor, 1e-4, 1 - 1e-4)))
report("ProbTor, Platt in-sample", dm.apply_calibration(lt, cal_tor))
d = dm.paired_cluster_bootstrap_test(y, pstor, p_model, g, n_boot=n_boot)
print(f"\n  model - ProbTor: delta AUC {d['delta_auc']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
      f"p(model <= ProbTor) = {d['p_value']:.4f}   ({d['n_clusters']} days)")
# Where does the model's edge live? Storms ProbTor rates near zero vs the rest.
lo = pstor < 0.01
for name, m in (("ProbTor < 1%", lo), ("ProbTor >= 1%", ~lo)):
    if 0 < y[m].sum() < m.sum():
        print(f"  {name:14s} n={int(m.sum()):7d} tornadic={int(y[m].sum()):5d}  "
              f"AUC model {dm.compute_auc(y[m], p_model[m]):.4f}  ProbTor {dm.compute_auc(y[m], pstor[m]):.4f}")
print(f"done {time.time() - t0:.0f}s")
