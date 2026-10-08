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

**Ops note (2026-10-03, resolved):** the earthquake and verification scorers used to commit without
deploying, because bot pushes do not trigger `deploy.yml`. Both now deploy themselves with the same
Wrangler step as the other scorers (`earthquake-score.yml`, `verification-score.yml`).

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
   - H8 widened it on dev (0.209 -> 0.194): the balanced response improves calibration, not the top end.
7. **H8. DONE (2026-10-08): the coherence equation as the vortex's balanced response, carried small;
   v10.4 in shadow** (amendments 8-9, PR #36, `hurricane_ri_h8.json`).
   - **What it is.** The coherence operator in its Rossby-adjustment form, with `ell(r) = c / I(r)`
     (Schubert and Hack 1982). This is the first use of the operator here that is the physics itself.
     The earlier uses were analogies:
     - tornado: the coherence field "adds nothing measurable once timed honestly"
       (`TORNADO_MODEL_PROGRAM.md`);
     - earthquake: the CFT block's ablation lift was 0.742 -> 0.749 AUC
       (`earthquake_nowcast_validation.md`), and the served map does not use it.
   - **Result.** dev dLL -0.00016 [-0.00089, +0.00054], dBrier4 -0.00013 [-0.00083, +0.00060]; 2026 (further
     read) dLL -0.0006 [-0.0028, +0.0020]; 2.4% of the split gain. About a tenth of IR's gain; every interval
     covers 0.
   - **Why small.** First-principles account; HYPOTHESIS until a test separates the two parts.
     1. The heating inside the RMW is mostly the IR model's 0-50 km ring again, since a typical RMW is
        20-40 nm (37-74 km).
     2. What is new is RMW-relative placement plus inertial stability, and both ride on CARQ's RMW. That
        RMW is in 5-nm steps, often carried forward between fixes, and poorly known for weak systems,
        which are where RI begins.
   - **Next from it, ranked.**
     1. **An RMW the image measures:** the radius of the eyewall's coldest ring, or of the strongest
        radial gradient. It would replace CARQ's in `ell(r)`. This separates "the physics adds nothing"
        from "the input is too coarse".
     2. **The same operator on 2 km GOES ABI band 13** (H1's next step). At 8 km a 15-nm RMW is three
        pixels.
     3. **The tangential-wind tendency itself** (the Sawyer-Eliassen secondary circulation), not the
        balanced heating. It is one more elliptic solve on the same grid.
     - Each needs its own registration; amendment 8's dev data is read.

### Hurricane track and intensity (program TC1, `docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md`)

1. **TC1. DONE (2026-10-08): our own track and intensity forecast** (tag `prereg-tc1`).
   - **The verifier comes first.** Our verifier reproduces NHC's published OFCL errors: 28/28 pooled cells,
     and 115/168 single cells with NHC's exact case count. It caught two of our own rule bugs (area by deck id;
     renamed crossing storms) before any skill number existed.
   - **Result (DEV 2023-2025):** track 90.74 n mi vs OFCL 92.08 and HCCA 97.61. No claim against OFCL
     (-2.76 [-8.18, +1.95] at 98.75%). Ahead of HCCA from 72 h (-6.24 [-11.14, -1.36], reported). Intensity
     ties.
   - **2026 so far:** track ties OFCL (-0.90). Equal weights would be +15.25 worse, so the online weighting
     does the work.
2. **TC1-live. Live, on the site, beside NHC's forecast.** An amendment before the first live forecast:
   - a persisted online state, updated each run;
   - the live state equal to the backtest's on the same decks;
   - each test record made at t + 3 h 30, as the RI program's are.
   - Open risk: the backtest used final decks, so an aid that arrived after t + 3 h 30 was counted. Measure
     that on the live record.
3. **TC2. Intensity conditioned on our RI probability.** Intensity is where TC1 adds nothing, and RI is where
   every consensus fails.
4. **TC3. Coherence steering (screened-Poisson PV inversion of the GFS analysis) -- ranked LOW, with the
   reason.** The inversion is exact physics for the steering flow, but the information is already in TC1's
   members: each global model integrates its own analysis's steering, and TC1 weights them by how they verify.
   - What could still help: steering *strength* as a regime signal. With weak steering the members spread and
     the right weights differ.
   - **The cheap test that decides it.** Do TC1's DEV track errors grow with weak analysis steering (850-200 hPa
     layer mean from GFS at t), beyond what the members' spread already shows? If not, TC3 is dead without
     building the inversion.
5. **G1. Genesis probability for invests, against NHC's outlook (TWO).** Feasibility checked 2026-10-08.
   - NHC's ATCF archive keeps no invest decks: `atcf/archive/2024/` has 57 AL files, none numbered 90-99.
   - Invest decks exist only in the season's real-time `aid_public` and `btk`.
   - History must come from UCAR RAL's real-time archive (to be confirmed for 2020+).
   - NHC's 2-day and 7-day probabilities are in its graphical TWO shapefile archive (`nhc.noaa.gov/gis`), as
     outlook areas. Matching an area to an invest and to the storm it became is the core of the work.
   - Comparator: NHC's own probability, Brier and reliability, on invests that did and did not form.

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
3. **E3. An earthquakes-only catalog** (recorded 2026-10-05 from the integrity audit).
   - **DECIDED 2026-10-06: S1 stays served; the rule is not met.** Section 11.1 of the earthquake program.
     - DEV (decides): IG(E3s) - IG(S1) -0.0026 [-0.0066, +0.0002]; the lower end is below the -0.005 margin.
     - FINAL (further read): +0.0015 [-0.0005, +0.0037].
     - The test-site cells fall 14-53x toward their neighbours. Nevada and Hawaii fall only about 2x, since
       both have real seismicity; Hawaii's removed events are volcanic.
     - **E3a (open, HYPOTHESIS): remove only the man-made types** (nuclear explosion, explosion, mine collapse,
       rock burst), keeping volcanic eruptions, and judge it on forecasts made after its registration.
   - Registered as section 11 of the earthquake program (tag `earthquake-amendment-e3`, PR #34), run in
     `research-earthquake-e3.yml`.
   - The registered candidate is the smallest correction: the served S1 with every input restricted to
     earthquakes, nothing refitted. It is judged by non-inferiority on DEV (IG lower bound >= -0.005 nats per
     target).
   - **Corrected count:** the inputs carry far more than the M6+ explosions below. 645 of 293,699 M4.5+ events
     are not earthquakes; 494 of them are M5+ inputs of the served map, plus 7,594 in its M2.5+ feed. No
     M6+ non-earthquake occurs from 2018 on, so the targets are unchanged.
   - **Found by the data, not typed:** Hawaii's cell holds 54 "volcanic eruption" events, and Hawaii also has
     real M6+ earthquakes. The rule decides whether removing them hurts.
   - **The original finding (2026-10-05):** every model in the programme (A, B, C0, S1) is fitted and scored
     on ComCat rows of every event type and every depth.
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
2. **T2. The 2025-08 ProbSevere format change** (opened 2026-10-05, tornado audit item 3).
   - **DECIDED 2026-10-06: the served model stays.** Tornado program amendment 10 (tag `tornado-amendment-10`,
     PRs #32-#33) ran on 2026-01..09 in `research-tornado-t2.yml`: 1,262,285 observations, 1,500 tornadic.
     - Served +W AUC 0.9699, equal to its 2025 final (0.9702). **The format change costs no ranking skill**; the
       within-2025 drop below was the season.
     - (d) against (a): dAUC -0.0003 [-0.0012, +0.0007], so the rule is not met. But dBrier is -1.13e-5
       [-2.13e-5, -1.9e-6] (BSS +0.107 against +0.098), and the served mean forecast is 0.81x the base rate.
     - **T2b (open): recalibrate the served model for the new format.** It needs its own registration, judged
       on forecasts made after it, since 2026-01..09 is now read. Candidate: Platt refitted on new-format data
       (2025-08-05..2026-09-30); control: the served calibration; decided on the live record.
   - Registered and run as:
     - The switch was **2025-08-05**, between 14:00Z (the last scan with `PS`) and 20:48Z; not 08-06 as below.
       No scan carries both formats.
     - **(b) is (a), by construction.** Every split on the two inputs has missing type None (196 + 475 in the
       +W payload), where NaN reads as 0.0. The "missing-value branch" below does not exist.
     - **(c) is not run.** VIL density needs the 18 dBZ echo top, which the new file lacks.
     - **(d)**, the chosen configuration without the two inputs, is built (`results/models/candidates/`).
       Validation 2023: AUC 0.96175 against the served configuration's 0.96215 (Brier equal).
     - The control demands exact inputs and probabilities within 4 ulp. Every one of 1,430 live input vectors
       reproduces exactly; NumPy's `exp` differs by <= 2 ulp across machines. The record audit's 1e-12
       tolerance already covers that, so no record change is needed.
     - The 30/90-min and EF2+ products read the same two inputs. (d) was not carried, so they keep their
       payloads too. T2b's recalibration question covers them as well.
   - **What changed.** On 2025-08-06 NOAA changed the ProbSevere JSON from 25 to 48 storm attributes.
     `PS`, `VIL_DENSITY` and `MAXRC_ICECF` were removed, and `MAXRC_EMISS` and `AVG_BEAM_HGT` are no
     longer in their documented string formats. New attributes include `MaxFED`, `DCAPE`, `VIL`,
     `EchoTop_50` and `LCL`.
   - **What the model gets.** The parser reads a missing `PS` or `VIL_DENSITY` as 0.0. So for every live
     storm since the change, the served model has received `p_ps = 0` and `p_vil_density = 0`. These
     two inputs carry 196 and 475 of the 8,826 splits in `tornado_v3_w.json`. The model was trained on
     2020-10..2024-12, entirely before the change. The three string attributes have no splits.
   - **Now visible, not changed.** Since 2026-10-05 every run records which inputs were absent and what
     the model received (`input_gaps` in the forecast record and its provenance envelope;
     `hazardpulse.tornado.input_guard`). The tornado page states the gap in plain words. Nothing the
     model receives has changed: that would be a model change.
   - **Witnesses.**
     - 2026-10-05 audit, as reported (not re-measured): NOAA ProbTor AUC 0.947 -> 0.843 across the
       change; ours 0.971 before vs 0.949 [0.928, 0.964] after.
     - Re-measured here on every 2025 storm observation of the v3 feature store (`storm_60` label; AUC
       with ties counted half; 200-bootstrap 95% intervals):
       - before 2025-08-06: 929,002 observations, 1,400 positive. Served fallback (`tornado_v3.json`, no
         NWS input) 0.970 [0.966, 0.974]; ProbTor 0.883. `p_ps == 0` on 29.6% of rows,
         `p_vil_density == 0` on 0.6%.
       - from 2025-08-06: 540,975 observations, 179 positive. Fallback 0.947 [0.931, 0.961]; ProbTor
         0.843. `p_ps == 0` and `p_vil_density == 0` on **100%** of rows.
   - **The confound.** The after-period is August-December: the quiet season, with a base rate of 0.033%
     against 0.151% before. NOAA's own ProbTor dropped too, and it has no zero-filled inputs of ours. A
     before/after split cannot separate the format change from the season.
   - **Deciding experiment.** Pre-register it first: the rule, a control that reproduces the served
     numbers bit for bit, and the season-matched comparison.
     1. Score the FROZEN served models on 2026-01..09 (new format throughout, all seasons) three ways,
        with every other input identical:
        - (a) missing = 0, as served;
        - (b) missing = NaN, the trees' missing-value branches;
        - (c) each absent attribute rebuilt from new-format attributes, for example VIL density from
          `VIL` and an echo-top height. Each candidate mapping is written into the pre-registration and
          checked on files that carry both forms, if any exist, before any scoring.
     2. Only then, a refit across both formats with the new attributes (`MaxFED`, `DCAPE`, `VIL`,
        `EchoTop_50`, `LCL`), chosen on 2023-2024 plus 2025-08..12, and scored once on 2026.
     - Promotion follows the program's carried-challenger rule: shadow first, then the rule written
       before the challenger's first cycle.
   - **Measured 2026-10-05, ruling out the cheap repair:** old `PS` is not the `probsevere` model
     probability the parser still reads (`p_ps_severe`). On the training store they are equal on 25.9% of
     rows (corr 0.83), and `p_ps` was already 0 on 32% of rows while `p_ps_severe` never is. So mapping
     the surviving field into `PS` would feed the model a quantity it never saw. In tonight's live file the
     new `ProbSevere` property equalled `models.probsevere.PROB` on all 78 storms, all at 0 (inconclusive).
3. **T3. A live nowcast at the edge** (opened 2026-10-05).
   - **Decision 2026-10-06: deferred behind T2.** An edge copy of a model that is fed two zeros is not the next
     step; the model is. The free plan's 10 ms of Worker CPU also cannot score an outbreak. Reopen once T2 is
     decided.
   - **The fact.** The served `tornado_v3_w` reads 30 inputs (`results/models/tornado_v3_w.json`
     `feature_names`): the current scan's 28 ProbSevere attributes, whether an NWS tornado warning
     covers the storm, and minutes since it was issued. That is 339 trees and a Platt map (a 1.006,
     b -6.391). Every input is in NOAA's 2-minute ProbSevere file and the NWS alerts the Worker already
     reads. No HRRR, track history or coherence field enters it.
   - **What it would give.** The Worker could score every tracked storm on the newest file, so the 60-min
     forecast is minutes old, not 30-120.
   - **Deciding work.**
     1. A JavaScript tree evaluator and feature extractor, owned by a parity oracle: every recorded v3
        storm (inputs and full-precision `probability_60min` are in each record, ~300,000 since
        2026-10-03) must reproduce exactly.
     2. Feature extraction checked against archived raw files for recorded runs.
   - **Decisions first.** Worker CPU: the free plan's 10 ms per request cannot score an outbreak's
     hundreds of storms. And every number the site shows must stay recorded, so edge numbers need a
     record (or the edge shows the batch record plus a fresher view, labelled as such).
   - **Correction recorded.** The same day this was first "killed" because the record's input vector has
     161 features, 107 from HRRR and the coherence engine. That counted what the record computes, not what
     the model reads. The kill was itself killed by the payload's feature list.

### Hurricane RI, serving

1. **H7. DONE (2026-10-05): WP/IO/SH v8.2 scored with its storm history** (opened the same day,
   hurricane audit).
   - **The problem.** Live WP ran from one JTWC warning, so 9 of v8.2's 17 inputs were median-imputed.
     Live witness: Choi-Wan intensified 45, 45 and 40 kt in 24 h while v8.2 published 5.7%, 5.4% and 3.2%.
   - **The repair** (`scripts/fetch_and_score.py`: `with_track_history`, `jtwc_live_case`,
     `unavailable_inputs`). Each JTWC storm takes its best-track fixes up to the warning's own cycle from
     UCAR RAL's real-time b-deck. It is never given a later fix, and never another storm's track: the
     track must reach within 12 h of the warning's cycle, close to its position. The model is unchanged;
     its inputs are restored. Two bugs found on the way, both fixed:
     - SH storm ids now take the season year (July-June; RAL: `bsh012026` began 2025-07-16). The calendar
       year would have fetched the previous season's storm of the same number. It would also have left
       every Jul-Dec SH storm unverifiable in the prospective test.
     - A storm warned across its season's turn keeps the year it formed in.
   - **Measured on the 2022-2024 WP test cycles** (2,110 cycles, 157 events; `hurricane_v82_test_composition.py`,
     each input pattern derived by running the live code on real fixtures):

     | inputs | log loss |
     |---|---|
     | every input (the current fix on time) | 0.2057 |
     | this cycle's fix late (pressure and its 3 changes filled) | 0.2134 |
     | one warning, the old live path | 0.2589 |
     | climatology | 0.2649 |

   - **Descriptive only.** The served calibration saw these cycles, and the test's history is the
     post-season track, not the working one.
   - **RAL lag, one day only (2026-10-05):** the 00Z fixes were in the file by 01:47, and the 12Z fix by
     15:17 (`bwp262026.dat`, Last-Modified 15:17:16Z). Both are inside the scorer's t + 3 h 30 min, but the
     12Z fix only by 13 min.
   - **Live at 17:00Z** (each with history vs the warning alone):
     - Choi-Wan: 0 inputs missing, 0.10%, vs 1.31% with 9 missing.
     - Koguma: 0.0974 vs 0.0602. At 12 h old its 24 h changes do not exist; training lacked them for
       young storms too, so they are recorded as `inputs_before_first_fix`, not as missing.
   - **What stays open.** Each record now says whether its history was on time (`ri_inputs.analysis_model`
     BEST vs JTWC, `track_source`), so the live record measures RAL's lag from here on. Resurrect if
     12Z cycles start landing late: move the hurricane slot to t + 3 h 45 min.

## Platform (2026-10-05 audit)

- **A clock for the scorers.** GitHub fired no scheduled run of any workflow here from 00:23Z to past
  03:30Z on 2026-10-05, with nothing on its status page. The scorers' own gated crons and
  `scheduler.yml` are in place, but both depend on GitHub's cron. The production Worker's 10-minute
  Cloudflare cron (PR #26) dispatches the scheduler once a token exists: `GH_DISPATCH_TOKEN`
  (fine-grained, this repository, Actions read/write).
  - **Solved 2026-10-06 without a token (PR #31).** `scheduler.yml` is its own clock: each run dispatches
    the next one ~10 minutes later. The concurrency group keeps it to one chain, and a cron run restarts a
    broken chain. MEASURED: the run started at 00:52:01Z dispatched its successor at 01:01:26Z. The cron alone
    had fired nothing from 18:20Z to 00:12Z. The Worker's cron stays as a dormant third clock.
  - **The queue (PR #29).** A GitHub concurrency group holds ONE pending job, and a newcomer cancels it, so
    scorers join only while the queue has room.
- **Where the records live.** `docs/DATA_AND_RECORDS.md` covers growth, about 1 GB a year in git at the
  new cadence, and the options (R2 with hashes in git is recommended).
  - **Decision 2026-10-06: they stay in git for now.** The public, append-only history is part of what is
    being shown.
  - MEASURED: GitHub reports the repository at 196 MB. The growth comes from evidence ledgers rewritten
    whole on every run (`gate-decisions.json` is 4.1 MB).
  - **Revisit when the repository passes 1 GB or any growing file passes 25 MB.** At about 1 GB a year that
    is roughly ten months out.
  - The largest file, `results/hurricane_operational_ri_2000_2024_al_sst.jsonl` (86 MB), is static, but it is
    near GitHub's 100 MB limit.
- **The 2026 tornado-warning file stopped the IEM loader (fixed, PR #32).** JKL 24 was issued twice on
  2026-08-11, and IEM carried the first warning's expiry onto the second. An expiry earlier than a row's own
  issuance now uses its INIT_EXP. It applies to 2 rows in 2026 and 0 in 2020-2025.

## Killed (witness -> what survives)

| claim | smallest witness | survives narrower |
|---|---|---|
| Extend hurricane training before 2020 with NOAA's RI probabilities | NHC e-decks carry no RI records before 2020: `eal062018` has 0 RI lines, `eal182021` has 488 per aid (every 2015-2019 deck sampled had 0) | O+H-only history (no NOAA probabilities) from 2015 |
| v10 is under-confident, so recalibrate it upward | 2026, served model: slope 0.99, intercept -0.19 | nothing. Dev intercepts tracked each season's RI rate (weather) |
| Storm-specific guidance errors predict RI (E) | dev: +0.0021 log loss vs v9 | nothing. The interpolated aids absorb the error |
| Cycle-to-cycle revisions predict RI (R) | dev: -0.0002 [-0.0013, +0.0009] | nothing |
| TC-PRIMED as a live input | 2025 "preliminary" uploaded 2026-07-24; 2026 EP empty | training data only |
| GEAR1 as a better long-term earthquake map than ours | AG - A_ch: DEV -0.21 [-0.33, -0.08], FINAL -0.31 [-0.41, -0.21] nats per target | GEAR1 as an added term (S1, served) |
| Repair the 2025 format change by mapping the new `ProbSevere` into old `PS` | training store: `p_ps == p_ps_severe` on 25.9% of rows, corr 0.83; `p_ps` = 0 on 32% where `p_ps_severe` never is | T2's evaluation and refit |
| Repair the format change with the trees' missing-value branch (T2 option b) | every split on `p_ps` / `p_vil_density` has missing type None (671 of them in +W): NaN reads as 0.0, so (b) returns the served probabilities on every row | (d): a model that never reads the two inputs |
| Taking every non-earthquake out of the served earthquake map's inputs costs nothing (E3s, non-inferiority at -0.005) | DEV: IG(E3s) - IG(S1) -0.0026 [-0.0066, +0.0002] | man-made types only (E3a), keeping volcanic events; judged on forecasts after registration |
| The served tornado model is degraded by the 2025 format change (ranking) | 2026, new format all year: AUC 0.9699, equal to the 2025 final's 0.9702; the within-2025 drop was the season | calibration: the served mean forecast is 0.81x the base rate (T2b) |
| The 22 Windows-only probability mismatches are a platform `exp` difference | on a GitHub runner 40 mismatched -- different storms; disabling AVX2/AVX-512 here changed nothing | probabilities reproduce within 2 ulp on every machine tried, inputs exactly; the audit's 1e-12 covers it |
| A live calibrator from the matured live record, as soon as one exists | tornado 2026-10-04: fitted on 794 storm-forecasts with 0 tornadoes, it published 567-1,822x the model's chance; held-out Brier 467x worse | a calibrator with >= 30 distinct events and a held-out win over the model (`calibrator_admissible`, PR #22) |

**Blocked, not killed:** CIRA's SHIPS developmental data. `rammb-data.cira.colostate.edu` returns
403 (nginx) to this machine, even with browser headers. Untested from a US CI runner.
