"""Set the singularity-condition-3 threshold on |torsion| from TRAINING days only.

Label-free: the threshold is a climatological exceedance quantile of |torsion|
over CONUS grid cells on training-split (2021-2022) HRRR 18z days, restricted
to cells where the coherence source already exceeds damping (S/Gamma > 1, i.e.
condition 1 -- the cells where condition 3 can matter). No tornado report and
no val/test day enters the number.

Also reports, for comparison, the same quantiles of the OLD torsion
(shear_06 * curl(grad tau) / 25) -- which is ~1e-8 everywhere.
"""
import sys

import numpy as np

import hazardpulse
from hazardpulse.data.hrrr import load_cached_hrrr
from hazardpulse.data.probsevere import scan_probsevere_cache
from hazardpulse.tornado import coherence_engine as ce
from hazardpulse.tornado import definitive_model as dm

print(hazardpulse.__file__)
q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
vals_all, vals_c1, ndays, residuals = [], [], 0, []
for d in scan_probsevere_cache():
    if not dm.date_in_range(d, dm.TRAIN_START, dm.TRAIN_END):
        continue
    used = False
    for h in dm.HRRR_ANALYSIS_HOURS:
        g = load_cached_hrrr(d, hour=h)
        if g is None:
            continue
        f = ce.compute_coherence_fields(g, month=int(d[4:6]))
        vals_all.append(np.abs(f["torsion"]).ravel())
        vals_c1.append(np.abs(f["torsion"][f["S_over_Gamma"] > 1.0]))
        used = True
    ndays += used
a = np.concatenate(vals_all)
c1 = np.concatenate(vals_c1)
print(f"training days {ndays}; cells all={a.size} cond1={c1.size}")
for qq in (0.5, 0.75, 0.9, 0.95, 0.99):
    print(f"  q{qq:.2f}: all {np.quantile(a, qq):.5f}  cond1 {np.quantile(c1, qq):.5f}")
print(f"THRESHOLD (cond1 cells, q={q}): {np.quantile(c1, q):.4f}")
print("fraction of cond1 cells above old 0.1:", float((c1 > 0.1).mean()),
      " above old NPE 0.05:", float((c1 > 0.05).mean()))
