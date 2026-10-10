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

## Champions (2026-10-10)

| hazard | published | ours, shown or in shadow | evidence |
|---|---|---|---|
| Hurricane RI, NHC basins | NOAA DTOPS | **v10.1 shown** beside it (named only in `results/hurricane_prospective/shown_model.json`; amendment 4 decides it at each look). In shadow: **v9.1**, the one entrant whose claim would switch what is *published* (amendment 1, 97.5%); challengers **v10.2** (monotone), **v10.3** (+ satellite IR) and **v10.4** (+ the balanced response, H8) | `docs/HURRICANE_RI_V9_PROGRAM.md` amendments 1-9; live record `results/hurricane_prospective/v9_shadow.json` |
| Hurricane RI, other basins (JTWC: WP, IO, SH) | **HazardPulse v8.3** (since 2026-10-10: v8.2's recipe on de-duplicated rows, non-inferior by amendment 15's rule; v8.2 kept as J1's base) | **J1 in test** beside it (amendments 13b and 14): recorded on every JTWC storm (`ri_j1_shadow`), shown only on storm cards in its scope, WP and SI (`ri_j1.scope()`). Never the published number unless its prospective claim is met at a look (v9.1's rule at 97.5%, the JTWC family's first budget; 2026-12-01, 2027-12-01; live in-scope cycles only) | `results/calibration/hurricane_ri_evaluation.json`, `hurricane_ri_v8_2_test_composition.json`; J1: `hurricane_ri_j1.json`, `hurricane_ri_j2.json` (scope); live record `results/hurricane_prospective/j1_shadow.json` |
| Hurricane track and intensity, NHC basins | -- (NHC's official forecast is the authority) | **TC1 shown beside OFCL** on every NHC storm card since 2026-10-08 (`hurricane_tc1-<state digest>`); **TC1+O** recorded, not shown; **TC2b** (TC1's intensity held to our RI model's median, amendments 2-4) recorded since 2026-10-10, in test; **TC2b+O** (the same on TC1+O, amendments 5-6; DEV mean 12.543 kt vs OFCL 12.627, no claim) recorded and shown on storm cards since 2026-10-10, in test | `docs/HURRICANE_TRACK_INTENSITY_PROGRAM.md` amendment 1; `results/hurricane_tc1/` |
| Tornado | v3 (+NWS warning state) | **T2b in shadow** (amendment 11: the served payloads' margins Platt-refitted on new-format data; `t2b_shadow` on every v3 forecast; decided at 2027-07-01 / 2028-07-01) | `docs/TORNADO_MODEL_PROGRAM.md` (amendment 10: stays served after the format change) |
| Earthquake M6+ | **S2 = C0 + GEAR1 weighted by the cell's activity** (since 2026-10-10, amendment E4; S1 served 2026-10-03 to 2026-10-10) | -- | `docs/EARTHQUAKE_FORECAST_PROGRAM.md` sections 10-12 |

The site shows each of these from its artifact (methods, registry, the hurricane page), and since 2026-10-09 the
methods page carries the research record below -- carried and not carried -- each row read from its results file.

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
     real-time) would resolve eyes and rings properly. **Tested as G1 (below): not carried.** Still open: a
     learned representation (a small CNN on the crops) in place of the hand-made statistics, with the same
     controls.
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
8. **H9. DONE (2026-10-08): the storm's coherence state, NOT carried** (amendment 10, `hurricane_ri_h9.json`).
   dev dLL +0.00275 [+0.00111, +0.00455] against H8, worse beyond its interval. Killed below, with what survives.
9. **Fusion. DONE (2026-10-08): coherence fusion of every RI source, claim NOT met** (amendment 11,
   `hurricane_ri_fusion.json`). Fusion minus H8 on DEV 2024-2025 +0.0036 [-0.0014, +0.0082]. Killed below.
10. **G1. DONE (2026-10-09): the inner core from GOES 2 km, hourly, NOT carried** (amendments 12, 12b, tag
    `prereg-hurricane-ri-amend12b`, `hurricane_ri_g1.json`).
    - **The input was sound first.** Amendment 12's eye rule over-called eyes in weak storms; 12b's eye, by ADT's
      closed-ring definition, passed every registered gate against CIMSS ADT before any outcome was read:
      held-out (2024-2026) HSS 0.834 (gate 0.80), eye fraction below 50 kt 0.011 (gate 0.03), centre median
      9.6 km, eye-temperature Spearman 0.959, 29,010 matched crops. `results/goes/adt_check.json` is that check's
      own output, committed 2026-10-09 from the CI artifact `goes-g1-adt-check` of run 37930285529 (commit
      bec84a879), SHA-256 `252916ba2bbdff5b7b2f3bd9a37c48ab2f5a8bfc21c0318db11ec331d976faab`.
    - **Result.** G1 - H8 on dev: dLL +0.00061 [-0.00027, +0.00155], dBrier4 **+0.00116 [+0.00008, +0.00233]**:
      worse on both, so not carried; H8 (v10.4) stands. The noise control (the 12 columns shuffled within
      season, seeds 1-3) gave dLL -0.00005 to +0.00052. 2026 further read: -0.0017 [-0.0053, +0.0019], the other
      direction, spanning 0.
    - **Reading.** The third null over H8 in a row (H9, fusion, G1). With 193 development events, H8 sits at the
      floor the data can certify (a dLL of about +-0.0009). The next real gain needs more independent RI events
      (earlier seasons, more basins) or a different contract, not another input family on the same 193.
11. **J1. DONE (2026-10-09): satellite IR for the JTWC basins, CARRIED on the pooled hindcast; in test beside v8.2
    in its scope, WP + SI** (amendments 13, 13a, 13b, 14; tags `prereg-hurricane-ri-amend13`, `-amend14`;
    `hurricane_ri_j1.json`, `hurricane_ri_j2.json`).
    - **What it is.** LightGBM on v8.2's score, v8.2's 17 inputs, amendment 5's 14 GMGSI IR features and basin
      flags, for the basins where we publish our own v8.2 because no NOAA RI guidance exists.
    - **Result (2024 + 2025 JTWC, 3,546 cycles, 158 events).** J1 - v8.2 dLL -0.00761 [-0.01472, -0.00034],
      dBrier -0.00083 [-0.00304, +0.00133]: carried. IR's own contribution J1 - B -0.00892 [-0.01380, -0.00390];
      the shuffled-IR noise control gave -0.00026 to +0.00067. 2026 further read (no claim): 0.2150 -> 0.1953.
    - **The independent pass (amendment 14)** refuted the claim as first worded ("J1 beats v8.2 in the JTWC
      basins"): South Pacific +0.0164 [+0.0051, +0.0362] on 1 RI event in 391 cycles, North Indian worse on Brier
      with 0 events, and a pooled interval that excludes 0 by only 0.0003 (crops moved 15 km, or one storm dropped,
      and it includes 0). The claim now stands as measured: pooled, hindcast, crops on best-track positions.
    - **J2** (separate SI/SP flags) **NOT carried** on 2026 (dLL vs J1 +0.00038 [-0.00112, +0.00172]); the
      shared-flag diagnosis is falsified (2026 SP, 20 events: the J family -0.048 vs v8.2).
    - **Live (13b + 14).** J1 is recorded on every WP/IO/SH storm and shown, on the site, only for storms in its
      registered scope (development point estimate <= 0: WP and SI). Its claim counts only in-scope live cycles;
      the hindcast interval is not evidence for promotion. The site shows it from the artifacts (storm cards,
      methods `#hurricane-j1`, research record, registry).
    - **Next from it.** Bringing SP into scope needs a new registration judged on data after it (2026 points that
      way). NI stays untested on RI events. A v8.2 retrain on de-duplicated rows is its own item (serving, below).

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
2. **TC1-live. DONE (2026-10-08): live, on the site, beside NHC's forecast** (program amendment 1).
   - The scorer replays the season from the saved models (`results/hurricane_tc1/state.json`); the snapshot was
     refused unless its replay equals one pass over 2020-2026 bit for bit (6,618 track and 6,895 intensity
     forecasts per product).
   - Each NHC storm card shows TC1 beside OFCL, lead by lead. Since 2026-10-09 TC1 has a model identity,
     `hurricane_tc1-<12 hex>` (the content digest of `state.json`), a registry entry, and a methods entry with
     its DEV claims, the HCCA comparison stated as descriptive at 95%, TC1+O, and the 2026 further read.
   - Open risk: the backtest used final decks, so an aid that arrived after t + 3 h 30 was counted. Measured at
     the looks (2026-12-01, 2027-12-01) on the live record.
3. **TC2. DONE (2026-10-10): TC2 NOT carried, TC2b CARRIED and live in test** (TC1 program amendments 2-4,
   `results/hurricane_tc2/`).
   - **The rule.** NHC scores intensity by MAE, which the median minimizes, and our RI model's 24-h exceedance curve
     brackets that median with nothing fitted. TC1's 24-h change is moved into the bracket only when it lies outside.
   - **TC2** carried that shift to 120 h. On DEV it was better at 24 h (-0.20 [-0.38, -0.04]) and worse from 72 h;
     mean +0.07, so not carried.
   - **TC2b** tapers the shift to 0 by 72 h, a taper chosen from DEV and so tested on 2026: mean -0.074, 24 h -0.15.
     Carried, neither interval excluding 0.
   - **RI cycles, 24 h error:** 2026, TC1 25.1 -> TC2b 20.0 kt (OFCL 19.0); DEV, TC1 23.6 -> 20.7 kt (OFCL 18.5).
   - **Next from it.** A curve for longer leads, our RI model at 48 h, would let the bracket reach where TC2's
     persistence failed. Tried as R48 / TC2c+O (item 3b): not carried.
3a. **TC2b+O. DONE (2026-10-10): CARRIED on DEV and 2026, live** (amendments 5-6, `results/hurricane_tc2/tc2bo.json`).
   - The same rule on TC1+O, our best base intensity forecast. Control: TC1+O's DEV 12.598598 kt reproduced.
   - **TC2b+O - TC1+O, mean 24-120 h:** DEV -0.055 [-0.109, -0.012]; 2026 -0.062 [-0.199, +0.053].
     24 h: DEV -0.169 [-0.34, -0.02]; 2026 -0.126 [-0.57, +0.23].
   - **Against OFCL (reported, no claim):** DEV 12.543 vs 12.627, -0.069 [-0.61, +0.47]. This is our first
     intensity forecast with a DEV mean below NHC's official one. 2026: +0.392 [-0.77, +1.20].
   - **RI cycles, 24 h:** DEV 23.2 -> 20.5 kt (OFCL 18.5); 2026 24.0 -> 19.4 kt (OFCL 18.3).
   - **Live (checked in the API's 2026-10-10 09:33Z forecast):** `tc1["TC2b+O"]` on every NHC storm, with its base named
     (`TC1+O`). Storm cards show TC2b+O's winds where the record has them, and TC2b's otherwise.
3b. **R48 / TC2c+O. DONE (2026-10-10): R48 has skill; TC2c+O NOT carried** (amendment 7, tag `prereg-r48-tc2c`,
   `results/hurricane_tc2/r48_build.json`, `tc2c.json`).
   - **R48** is the 48-h exceedance curve, 15-65 kt, on H8's 67 inputs. Out-of-fold log loss 0.270 vs
     climatology 0.420, so its gate passed.
   - **TC2c+O** holds the second day to R48's median bracket. Control: DEV TC2b+O 12.543423 reproduced.
     DEV TC2c+O - TC2b+O, mean +0.216 [+0.098, +0.355], 48 h +1.045 [+0.56, +1.64]; 2026 mean +0.073,
     48 h +0.260. RI48 cycles worse in both seasons. Not carried, and not served.
   - **Why (hypotheses, untested):** R48 sees neither the track nor land, and it was trained only on storms that
     survived 48 h. The bracket pulls toward a median the storm reaches only if it stays over water.
   - **Resurrect if:** a 48-h curve conditioned on TC1's forecast track, with dissipation and landfall as
     outcomes rather than exclusions.
4. **TC3. Coherence steering (screened-Poisson PV inversion of the GFS analysis) -- ranked LOW, with the
   reason.** The inversion is exact physics for the steering flow, but the information is already in TC1's
   members: each global model integrates its own analysis's steering, and TC1 weights them by how they verify.
   - What could still help: steering *strength* as a regime signal. With weak steering the members spread and
     the right weights differ.
   - **The cheap test that decides it.** Do TC1's DEV track errors grow with weak analysis steering (850-200 hPa
     layer mean from GFS at t), beyond what the members' spread already shows? If not, TC3 is dead without
     building the inversion.
5. **GEN1. Genesis probability for invests, against NHC's outlook (TWO).** Feasibility checked 2026-10-08.
   (Named G1 until 2026-10-09; renamed because G1 is also the 2 km GOES test of the RI program, amendment 12.)
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
     per-depth weight is the next experiment (pre-register it; fit on CHOOSE). Done as E4.
1a. **E4. DONE (2026-10-10): S2 -- GEAR1's weight follows the cell's activity, CARRIED** (section 12, tag
   `prereg-earthquake-e4`, `results/earthquake_program/gear1_e4.json`).
   - **The form:** `logit p = a + c z + b g + d z g`, with z = logit(p_C0). It is the one-parameter version of
     "per region", along the axis the physics names: strain should matter where the catalog is quiet.
   - **Result:** DEV S2 - S1 **+0.0139 [+0.0015, +0.0255]** nats per target; FINAL second read +0.0087
     [+0.0001, +0.0169].
   - **The registered prediction held:** d = -0.073, so GEAR1's weight is 0.73 in a typical cell and 0.11
     where M6+ earthquakes happen.
   - **Size:** GEAR1's total contribution about doubles (S1 - S0 was +0.016).
   - **Serving (section 12.2):** stack schema v2, built with formula and live parity on a runner. The scorer,
     evidence and registry read one constant (`hazardpulse.earthquake.served`). A named stack that is missing now
     stops the run; it used to fall back to C0 alone.
   - **Open from it:** DEV has decided E1, E3 and E4. The prospective record (section 8) is the independent
     check.
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
     - **T2b: REGISTERED, FITTED, LIVE IN SHADOW (2026-10-10, amendment 11, tag `tornado-amendment-11`, PR #46).**
       - The candidate is Platt refitted on 1,782,183 new-format observations. For `p60_w` the coefficients move
         from (1.006, -6.391) to (1.068, -6.346); the in-sample mean forecast moves from 0.84x to 1.00x the base rate.
       - Controls: the served calibrations are reproduced bit for bit.
       - It is decided on the live record at 2027-07-01 and 2028-07-01: Brier AND log loss lower, with the 95%
         convective-day interval of dBrier below 0.
       - EF2+ waits for SPC's final ratings.
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
2. **DONE (2026-10-10, amendment 15, PR #45): v8.3 = v8.2 retrained on de-duplicated rows, published.**
   - Test on untouched 2025-2026 JTWC data: dLL +0.00027 [-0.00070, +0.00137], which is non-inferior at +0.002.
     The fit is correct and no better.
   - NI (0 events) is slightly worse beyond its interval.
   - J1 stays on v8.2's numbers, from v8.2's own artifact.
   - The history of the finding:
     **(found 2026-10-09, amendment 13a): v8.2's training data holds 5,950 duplicate basin-crossing rows.**
   - The v8.2 builder concatenates IBTrACS's per-basin files, each carrying the whole track of every storm that
     enters its basin, so a basin-crossing storm appears once per basin it touches: 5,950 byte-identical rows
     (8.5% of 69,722) from 178 storms, 814 RI-positive; 5,255 of them in v8.2's training years (2000-2021).
   - The served v8.2 was trained with those storms counted twice. J1 de-duplicates its own rows
     (`hurricane_ri_j1.dedupe`); v8.2 is unchanged.
   - **A v8.2 retrain on de-duplicated data needs its own registration** (it is a model change): the decision rule,
     the control that reproduces the served v8.2 first, and how J1, which takes v8.2's score as an input, follows.

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
| A storm's coherence state (exponentially forgotten memory of core convection, symmetry, heating retention, intensity tendency over 36 h) adds to RI (amendment 10, H9) | dev dLL **+0.00275 [+0.00111, +0.00455]** vs H8 (worse); the same features shuffled cost ~6x less; the retention memory alone +0.00155 | persistence of deep core convection without the RMW-derived memory (-0.00049 [-0.00108, +0.00006], single-feature diagnostic; needs its own registration on unread data) |
| Coherence fusion of every RI source (Boltzmann weights on forgotten verified log loss, amendment 11) beats the best single source | dev 30/24 LL fusion 0.1527 vs H8 0.1490, d +0.0036 [-0.0014, +0.0082]; NOAA's aids 0.02-0.11 worse than our models, which already take them as inputs | the law as a weighting rule: it beats equal-weight pools by ~0.013. It wins where sources are independent and comparable (TC1 track vs HCCA), not where the best one already holds the rest |
| An adaptive coherence reach (each M5+ kernel width = distance to its k-th nearest earlier event, l ~ rho^-1/2) beats A's fixed 20 km kernel (earthquake, FIT-stage probe 2026-10-08) | FIT IG 2.4391-2.5029 vs fixed 2.5166 (control reproduced 2.517); every variant worse, narrower always better | nothing at 2-degree cells: sub-50 km reach is sub-cell in active regions, and widening in sparse ones spreads mass off the faults where M6+ recur |
| Hourly 2 km inner-core structure from GOES (core and ring convection, eye, eyewall edge, symmetry, persistence, trends) adds RI information beyond H8 (amendment 12b, G1) | dev G1 - H8: dLL +0.00061 [-0.00027, +0.00155], dBrier4 +0.00116 [+0.00008, +0.00233]; the shuffled-column noise control gave dLL -0.00005 to +0.00052. The input passed ADT's gates first (held-out HSS 0.834) | nothing at 24 h on 2020-2025's 193 events: the operational aids already ingest GOES IR. More independent events (earlier seasons, more basins) or another contract |
| J1 beats v8.2 in the JTWC basins (amendment 13's claim as first worded) | South Pacific, 2024-2025: J1 - v8.2 dLL +0.0164 [+0.0051, +0.0362] on 1 RI event in 391 cycles; the outcome's per-basin table had merged SI and SP into "SH" (amendment 14, independent pass) | the pooled hindcast claim as measured (crops on best-track positions), and J1 live only in its scope, WP + SI; promotion only from the live record |
| J1's SP loss comes from its one shared `is_sh` flag (fix: J2, separate SI/SP flags) | separate flags moved development SP by 0.0005 (+0.0164 -> +0.0159); J2 - J1 on 2026 +0.00038 [-0.00112, +0.00172], not carried; 2026 SP (20 events): the J family -0.048 vs v8.2 | the witness itself (2024-2025 SP worse, on 1 event); SP into scope only by a new registration judged on later data |
| A live calibrator from the matured live record, as soon as one exists | tornado 2026-10-04: fitted on 794 storm-forecasts with 0 tornadoes, it published 567-1,822x the model's chance; held-out Brier 467x worse | a calibrator with >= 30 distinct events and a held-out win over the model (`calibrator_admissible`, PR #22) |
| TC2's 24-h intensity shift persists to 120 h (TC1 amendment 2) | DEV 72-120 h worse (+0.13 to +0.27 kt); mean +0.07 kt over 24-120 h | the bracket within the curve's own horizon (TC2b: carried on 2026) |
| A skilful 48-h RI curve (R48) brackets the 48-h intensity median well enough to correct the second day (TC2c+O, TC1 amendment 7) | DEV TC2c+O - TC2b+O mean +0.216 [+0.098, +0.355], 48 h +1.045 [+0.56, +1.64]; 2026 worse too; R48's own log loss 0.270 vs climatology 0.420, so the curve has skill and the median rule fails | TC2b+O within 72 h; a 48-h curve conditioned on the forecast track, with dissipation and landfall as outcomes |
| A published file can be checked for strict JSON after the bot commits it | 2026-10-10 03:39Z: TC2b's -Infinity reached live-storms.json; /api/v1/live/hurricane was 404 for ~1 h although the strict-JSON test existed | the writers enforce it (`strict_json`, `allow_nan=False`) |

**Blocked, not killed:** CIRA's SHIPS developmental data. `rammb-data.cira.colostate.edu` returns
403 (nginx) to this machine, even with browser headers. Untested from a US CI runner.
