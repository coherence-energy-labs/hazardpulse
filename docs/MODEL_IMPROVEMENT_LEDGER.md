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
| Hurricane RI, NHC basins | NOAA DTOPS | v10.1 shown beside it; challengers **v10.2** (monotone) and **v10.3** (+ satellite IR) in shadow | `docs/HURRICANE_RI_V9_PROGRAM.md` amendments 2-6 |
| Tornado | v3 (+NWS warning state) | -- | `docs/TORNADO_MODEL_PROGRAM.md` |
| Earthquake M6+ | **S1 = C0 + GEAR1** (since 2026-10-03) | -- | `docs/EARTHQUAKE_FORECAST_PROGRAM.md` section 10 |

## Open branches, ranked

### Hurricane RI

The cheap levers on the current inputs are exhausted (amendment 3): monotone constraints gave
-0.0007 in log loss, and revisions -0.0002. **The next material gain must be new information.**
The first one, satellite IR (H1), delivered.

**Ops note (2026-10-03):** the earthquake and verification scorers commit without deploying (bot
pushes do not trigger `deploy.yml`). Their changes reach the site only at the next hurricane or
tornado deploy or merge. Cheap fix: give `earthquake-score.yml` the same deploy step as
`hurricane-score.yml`.

1. **H1. DONE (2026-10-03): v10.3 = V5 + satellite IR, carried and in shadow** (amendments 5-6).
   - Data: NOAA GMGSI longwave, hourly, archived from 2021-07-12. Crops are made by the same
     library code for training and live.
   - Result: dev log loss 0.1418 vs V5 0.1436 (-0.0018 [-0.0054, +0.0015]). On 2026 (fourth read)
     0.1178 vs 0.1262 (**-0.0084 [-0.0152, -0.0018]**), the first new-information gain.
   - Lesson: one bucket connection gives ~0.9 MB/s and eight give ~10, so parallelise any bulk
     S3 pull.
   - Next from it: a finer inner core. GMGSI is 8 km, so native GOES ABI band 13 (2 km, on AWS,
     real-time) would resolve eyes and rings properly. Also a learned representation (a small CNN
     on the crops) in place of the 14 hand-made statistics, with the same controls.
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
3. **E3. An earthquakes-only catalog** (recorded 2026-10-05 from the integrity audit; not yet
   pre-registered). Every model in the programme (A, B, C0, S1) is fitted and scored on ComCat
   rows of every event type and every depth.
   - Witness, explosions: 70 of the 7,498 M6+ rows 1973 to 2026-10-01 in the programme's own
     `comcat_m4.5_<year>.csv` files are not earthquakes (69 `nuclear explosion`, 1 `explosion`;
     41 in Kazakhstan, 12 Nevada, 8 Russia, 6 China; the last one North Korea, 2017-09-03). They
     enter the long-term map as M5+ seismicity and the targets as M6+ "earthquakes". In the served
     forecast `eq_fcst_20261004_2100` the Semipalatinsk test-site cell (row 54, col 129) has
     P = 5.90e-4, 51x the central-Kazakhstan cell (row 54, col 125, 1.15e-5) and 44x the grid
     median (1.34e-5); the catalog's last M6+ explosion there is 1989-10-19.
   - Reproduce: count `type` among M >= 6 rows of `.cache/earthquake/program/comcat_m4.5_*.csv`
     before the cutoff; read the two cells from that replay's `probability_grid`.
   - Witness, depth: there is no depth limit. 1,540 of the 7,498 M6+ rows (20.5%) are deeper than
     70 km (1,038 at 70-300 km, 502 deeper), while GEAR1, the term S1 adds, is a forecast of
     shallow (0-70 km) seismicity.
   - Experiment: candidates (a) `type == "earthquake"` only, (b) (a) plus a depth split
     (shallow targets scored against GEAR1-shallow, intermediate/deep against the rest), with the
     served S1 as the control reproduced bit for bit. Pre-register and tag first; refit every rate
     and the trees on CHOOSE, decide on DEV, report FINAL as a further read.
   - Carried rule: as E1 (DEV paired IG 95% interval above zero against the control). The
     targets change with (a), so the paired comparison scores both arms on the earthquakes-only
     targets and reports the unfiltered targets beside them.

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
