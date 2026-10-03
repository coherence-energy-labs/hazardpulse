# Model improvement ledger

**Standing rule (Josh, 2026-10-03):** "just cause we beat something now doesnt mean we stop, we keep
testing and optimizing and solving and figuring out how we can make our models better always."

A win against today's incumbent is a checkpoint. This file is the queue that makes "always"
concrete. It holds:

- the champion of each hazard;
- the open branches, ranked by expected gain per unit of cost, each with the one experiment that
  decides it;
- every kill, with its witness and what survives narrower.

Read it before starting model work, and update it when a branch is decided.

## The loop

1. Take the top open branch.
2. **Pre-register** the candidates, the control and the carried rule, then commit and tag before
   any candidate number exists.
3. **Measure** on the development seasons. Every run carries a control that reproduces the
   champion bit for bit, or it stops.
4. **Carried challenger** -> live shadow beside the champion. It gets its own prospective entry at
   half the previous entrant's error budget, so the family total stays below 5%.
5. **Promotion** happens only by the rule written before the challenger scored its first cycle.
   The site shows numbers bound to artifacts, never typed.
6. **Kills** are written here with the smallest witness and what survives.

## Champions (2026-10-03)

| hazard | published | ours, shown or in shadow | evidence |
|---|---|---|---|
| Hurricane RI, NHC basins | NOAA DTOPS | v10.1 shown beside it; **v10.2 challenger** in shadow | `docs/HURRICANE_RI_V9_PROGRAM.md` amendments 2-4 |
| Tornado | v3 (+NWS warning state) | -- | `docs/TORNADO_MODEL_PROGRAM.md` |
| Earthquake M6+ | **S1 = C0 + GEAR1** (since 2026-10-03) | -- | `docs/EARTHQUAKE_FORECAST_PROGRAM.md` section 10 |

## Open branches, ranked

### Hurricane RI

The cheap levers on the current inputs are exhausted (amendment 3): monotone constraints gave
-0.0007 in log loss, and revisions -0.0002. **The next material gain must be new information.**

1. **H1. Convective structure from geostationary IR, one code path for training and live.**
   - Source: NOAA GMGSI longwave IR global mosaic (`s3://noaa-gmgsi-pds/GMGSI_LW/`). It is hourly
     and archived from 2021. The 12Z image landed at 12:39Z, well before our t + 3 h 30 run.
     Files are ~7.5 MB.
   - Features at t: annular brightness-temperature statistics around the CARQ centre at 0-50,
     50-200 and 100-300 km (mean, std, fraction colder than -50 C and -70 C), azimuthal symmetry,
     and their 6-h trend. These are the quantities SHIPS's GOES predictors summarise. Here they are
     direct inputs, not filtered through NOAA's probabilities.
   - Decides: whether satellite convective structure adds to NOAA's RI probabilities.
   - Experiment: V5 + IR on folds 2022-2025, training from 2021. A control is required: V5 must
     reproduce itself on the 2021+ training window.
   - Cost: about 4,000 cycles x 7.5 MB, streamed (crop, then delete). Never keep the images (see
     the scratchpad disk-bomb law).
2. **H2. Environment from GFS analyses** (`s3://noaa-gfs-bdp-pds`, 2021+, real-time).
   - Features: 850-200 hPa shear, 700-500 hPa RH, SST and potential intensity at t, computed by one
     code for training and live.
   - Decides: whether the raw environment adds to what NOAA's RI aids already compress from it.
   - Cost: GRIB2 byte-range reads by `.idx`.
3. **H3. Inner-core passive microwave** (the 37/89 GHz ring, a strong RI precursor that DTOPS does
   not use).
   - Training data exists: TC-PRIMED final, 1987-2025, `s3://noaa-nesdis-tcprimed-pds`.
   - **Blocked live:** no real-time source has been found. TC-PRIMED "preliminary" is batch (the
     2025 files were uploaded 2026-07-24; 2026 EP is empty).
   - Unblocks if: a real-time swath feed is found (ATMS SDR on NOAA's AWS is the candidate, but it
     is coarse) and our own processing reproduces TC-PRIMED's features on overlapping years.
4. **H4. DONE (2026-10-03, PR #13): every SHIPS text is archived**
   (`results/hurricane_ships_archive/<year>/`). The 2026 season is backfilled (1,023 texts), then
   synced by every hurricane run. Once two or more seasons have accrued, a model can train on
   SHIPS's operational environment predictors (shear, SST, OHC, RH).
5. **H5. Anytime-valid prospective tests** for new entrants: e-processes or confidence sequences
   for forecast comparison (Henzi and Ziegel 2022; Choe and Ramdas 2023).
   - A challenger could then be judged continuously instead of only at fixed looks, without
     spending more error budget for peeking.
   - Needs care with overlapping 24-h windows (lag-4 dependence within a storm).
6. **H6. The top-end tie with yes/no calls** (HCCA, IVCN, hurricane-model mean at their own
   false-alarm rates). M did not close it (0.215 vs HCCA's 0.236).
   - Re-test after H1 and H2. If it persists, it is a property of these inputs, not of the model.

### Earthquake

1. **E1. DONE (2026-10-03): S1 = C0 + GEAR1 is served** (`docs/EARTHQUAKE_FORECAST_PROGRAM.md`
   section 10).
   - Gain: DEV S1 - S0 = +0.016 [+0.002, +0.030] nats per target; FINAL (second read) +0.013
     [+0.000, +0.026].
   - GEAR1 *alone* loses to our causal smoothed seismicity by 0.2-0.3 nats. The best public global
     model is a useful input, not a better forecast.
   - Open from it: GEAR1's strain-rate term only enters as one global weight. A per-region or
     per-depth weight is the next experiment (pre-register it; fit on CHOOSE).
2. **E2.** The margin over smoothed seismicity is at the edge of zero.
   - Decide whether this is power (number of M6+ quakes) or a real limit, with a per-year
     breakdown and a power estimate before any new features.

### Tornado

1. **T1. The warnings tie.** Without the warning state as an input, v3 only ties the NWS warnings
   at their own false-alarm rate (POD 0.230 vs 0.240, d -0.010 [-0.043, +0.015]). The warning
   state is information we cannot create; the question is what forecasters see that ProbSevere's
   28 attributes do not.
   - Candidate: MRMS rotation tracks and azimuthal shear (`s3://noaa-mrms-pds`, archive plus
     real-time), timed honestly.
   - Amendment 8's lesson applies: every input must exist at the forecast's issue time.

## Killed (witness -> what survives)

| claim | smallest witness | survives narrower |
|---|---|---|
| Extend hurricane training before 2020 with NOAA's RI probabilities | NHC e-decks carry no RI records before 2020: `eal062018` has 0 RI lines, `eal182021` has 488 per aid (every 2015-2019 deck sampled had 0) | O+H-only history (no NOAA probabilities) from 2015 |
| v10 is under-confident, so recalibrate it upward | 2026, served model: slope 0.99, intercept -0.19 | nothing. Dev intercepts tracked each season's RI rate (weather) |
| Storm-specific guidance errors predict RI (E) | dev: +0.0021 log loss vs v9 | nothing. The interpolated aids absorb the error |
| Cycle-to-cycle revisions predict RI (R) | dev: -0.0002 [-0.0013, +0.0009] | nothing |
| TC-PRIMED as a live input | 2025 "preliminary" uploaded 2026-07-24; 2026 EP empty | training data only |
| GEAR1 as a better long-term earthquake map than ours | AG - A_ch: DEV -0.21 [-0.33, -0.08], FINAL -0.31 [-0.41, -0.21] nats per target | GEAR1 as an added term (S1, served) |

**Blocked, not killed:** CIRA's SHIPS developmental data. `rammb-data.cira.colostate.edu` returns
403 (nginx) to this machine, even with browser headers. Untested from a US CI runner.
