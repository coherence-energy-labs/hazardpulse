# Program TC1 -- our own hurricane track and intensity forecast

Registered 2026-10-08 ~18:45Z, before any TC1 forecast error, or any HCCA, TVCN, IVCN or GDMI error, has been
computed. Tag: `prereg-tc1`.

## Why

HazardPulse forecasts one hurricane quantity, the chance of rapid intensification (the RI program). It has no
answer to "where will it go, and how strong will it get?" -- the first two questions anyone asks of a storm.

NHC's official forecast (OFCL) is the bar. Its best guidance is a corrected consensus fitted on past seasons
(HCCA).

**What this is an instance of: prediction with expert advice whose experts change.** The aid suite is not
stationary:
- HWRF and HMON gave way to HAFS in 2023;
- Google DeepMind's GDMI arrived in 2025;
- consensus aids are renamed (TVCA, TVCE and TVCX are gone from the 2026 public decks).

A combination fitted once on past seasons must wait a season or more to weight a new aid.

TC1 instead re-estimates its weights at every cycle from what has already verified. Each aid's errors at each
lead are measured against the operational analysis (CARQ) at the verifying time, across the basin and within
this storm, with exponential forgetting. A new aid enters on its first verified forecast, shrunk toward the
pool until its own record speaks. The combination is Bates and Granger's minimum-variance weights on the
members' error covariance, shrunk toward its diagonal and constrained non-negative.

## The forecast

- **Code:** `src/hazardpulse/hurricane/consensus.py`, the same code for the backtest and, later, live.
  `tests/test_hurricane_consensus.py` pins:
  - what the weights learn (inverse-variance weights on independent aids);
  - that a constant bias is removed;
  - that a forecast never sees the future (rewriting every later analysis leaves every earlier forecast bit
    for bit);
  - the geometry and the parser.
- **Members** are early (interpolated) aids that are public in real time. ECMWF-derived aids (EMXI, EEMN,
  FSSE, ...) are in the archive but not in NHC's public real-time decks, so a forecast that used them could
  not run live.
  - **Track:** AVNI, AEMI, HFAI, HFBI, HWFI, HMNI, CTCI, UKXI, CMCI, CEMI, NVGI, GDMI, TVCN, HCCA.
  - **Intensity:** DSHP, LGEM, NNIC, NNIB, HFAI, HFBI, HWFI, HMNI, CTCI, GDMI, IVCN, HCCA.
- **Leads:** 12-120 h; at least two members are needed (TVCN's own rule).
- **Products:**
  - **TC1** -- the members only: an independent forecast, a peer of OFCL;
  - **TC1+O** -- with OFCL as one more member. It is issued after the advisory (t + 3 h 30), so it asks
    whether the official forecast can be improved.
- **Causality.** A forecast at t uses only forecasts that verified by t (t0 + tau <= t), scored against
  CARQ at that time: the analysis that existed then. The final best track never enters the weights.

## The verifier, and its control (run before this registration)

**NHC's rules.** A forecast at synoptic time t for lead tau is verified when the best track classes the
system as TD, TS, HU, SD or SS at both t and t + tau. A forecast belongs to the area the storm is in at t:
- the Atlantic or the eastern Pacific east of 140 W (NHC);
- 140 W to the dateline (CPHC) is never scored.

Track error is great-circle n mi; intensity error is absolute kt.

**Truth:**
- 2020-2025: IBTrACS v04r01's USA columns (cache of 2026-04-13), at synoptic hours. A deck whose ATCF id is
  not in IBTrACS takes the track its own CARQ fixes follow. Bonnie 2022 was AL02 and then EP04, and IBTrACS
  keeps the whole track under AL02.
- 2026: NHC's operational b-decks.

**The control** (`scripts/hurricane_tc1.py control`, `results/hurricane_tc1/verifier_control.json`). The
verifier scores OFCL, and the result is compared with NHC's own published error tables
(`results/hurricane_tc1/nhc_ofcl_published.json`, transcribed from NHC's annual OFCL error PDFs).
- **Pass rule:** pooled over 2020-2024, per basin, kind and lead, N within 2% and mean error within 2%
  (track) or 0.2 kt (intensity).
- **Result: passed, all 28 pooled cells.** 115 of 168 single cells have NHC's exact case count.
  - AL track 24 h: 1,624 cases vs NHC's 1,612; 34.38 vs 34.42 n mi.
  - EP intensity 48 h: 764 vs 768; 12.88 vs 12.87 kt.
- **How it got there.** It failed twice, and both failures were in this verifier:
  1. Area by the deck's id, not by where the storm was: CPHC's forecasts west of 140 W were counted.
  2. Renamed crossing storms found no truth.

  Each was fixed at its root and re-run. Neither fix touches any TC1 number, since none existed.
- **One published cell is excluded, and shown.** NHC's 2020 Atlantic 72-h *track* cell gives 318 cases
  (79.9 n mi).
  - NHC's own *intensity* table gives 275 cases for the same forecasts at 72 h.
  - Every other 2020 lead has equal counts in the two tables.
  - This verifier finds 275 (98.1 n mi), and matches the intensity cell exactly (275, 10.9 kt).

## Design

- **Seasons:**
  - **WARM-UP** 2020: the online state starts, nothing is scored.
  - **CHOOSE** 2021-2022.
  - **DEV** 2023-2025.
  - **2026**, the season in progress: a further read against operational best tracks, no claim.

  The online state runs continuously through all of them. A season is assigned by the storm's ATCF year.
- **Grid (24 configs).**
  - Forgetting half-life: 20, 60 or 180 days.
  - Covariance shrinkage: 0.5 or 1.0 (diagonal: inverse-MSE weights).
  - Debias: off or on.
  - This storm's own verifications weighted 1x or 5x.
  - A new aid's prior weight: 5 pseudo-verifications, fixed.
  - The **equal-weight** reference (EQ) is reported beside the grid.
- **Selection (CHOOSE):** per kind, the config with the lowest mean over 24, 48, 72, 96 and 120 h of the
  mean error. Ties go to the earlier config in grid order. Track and intensity are chosen separately.
  TC1+O uses the same config with OFCL added.
- **Primary claims (DEV, AL and EP pooled, 98.75% each, so four claims share 5%):**
  1. TC1 track better than OFCL;
  2. TC1 intensity better than OFCL;
  3. TC1+O track better than OFCL;
  4. TC1+O intensity better than OFCL.

  Each claim is met iff the storm-block bootstrap interval of the mean over the five leads of the per-lead
  mean difference (product minus OFCL, on the cases both verified) lies wholly below 0. The bootstrap uses
  4,000 draws, seed 20261008.
- **Reported, 95%, descriptive:**
  - each lead;
  - TC1 and TC1+O against HCCA, against TVCN (track) or IVCN (intensity), and against GDMI (2025 only:
    GDMI's first season);
  - EQ against OFCL.

## Afterwards

- The forecast goes on the site whatever the claims say, as "HazardPulse's track and intensity forecast"
  beside NHC's. Its DEV numbers are stated plainly. "More accurate than NHC's official forecast" is said only
  where a claim is met.
- Going live is its own amendment, written before the first live forecast. The online state is persisted and
  updated every scorer run, and the live state must reproduce the backtest's on the same decks.
- The RI probability as an input to the intensity forecast is a later candidate (TC2). RI is where every
  intensity consensus fails, and where HazardPulse already has a model.

## Known before the result

- **OFCL is hard to beat.** Forecasters see every aid, including the ECMWF aids TC1 may not use. HCCA
  already weights the aids with past seasons' errors. A TC1 that ties OFCL is the likely outcome; a TC1+O
  gain would be small.
- **The CHOOSE seasons have neither HAFS nor GDMI.** A config chosen there may not be the best one for
  2023-2025.
- **One season of GDMI (2025)** supports the GDMI comparison; it is descriptive.
- **The weights learn against CARQ, not the best track.** CARQ's position at the verifying time carries
  analysis error of its own, mostly at weak, disorganized systems.
- **Disclosed:** before this registration, the only errors computed were OFCL's, in the control. They
  reproduce NHC's published numbers. The engine was timed on the 2020 warm-up season, where no error was
  computed. The aid inventory (which aids exist in which season) reads inputs only.

## Outcome (2026-10-08)

**Selection (CHOOSE 2021-2022, `selection.json`).** Errors are the mean over 24-120 h.
- **Track:** config 17 (half-life 180 d, shrinkage 0.5, no debias, no storm weight), 84.72 n mi. Equal
  weights give 87.21; the worst config 96.20. Every debiasing config was worse than equal weights: an aid's
  recent mean error does not carry forward.
- **Intensity:** config 18 (half-life 180 d, shrinkage 0.5, no debias, this storm 5x), 11.52 kt. Equal
  weights give 11.57. On intensity the weighting barely matters.

**DEV 2023-2025, AL and EP pooled (`dev.json`)** -- mean error over the five leads (n mi; kt):

| | TC1 | TC1+O | OFCL | HCCA | TVCN / IVCN | equal weights |
|---|---|---|---|---|---|---|
| track | **90.74** | **90.47** | 92.08 | 97.61 | 97.56 | 94.49 |
| intensity | 12.83 | **12.60** | 12.63 | 13.22 | 12.78 | 12.75 |

The table pools each product's own verified cases. The comparisons below are paired, on shared cases only.

**Primary claims: none met.**

| claim (98.75%) | mean over leads | interval |
|---|---|---|
| TC1 track vs OFCL | **-2.76 n mi** | [-8.18, +1.95] |
| TC1+O track vs OFCL | **-3.05 n mi** | [-8.22, +1.40] |
| TC1 intensity vs OFCL | +0.25 kt | [-0.63, +1.11] |
| TC1+O intensity vs OFCL | -0.00 kt | [-0.69, +0.70] |

**Reported (95%, descriptive):**
- **TC1 track vs HCCA: -6.24 [-11.14, -1.36]**, better at 72 h (-5.8), 96 h (-9.5) and 120 h (-15.6). HCCA is
  NHC's corrected consensus; TC1 beats it from 72 h on.
- TC1 track vs TVCN: -6.82 [-12.79, -1.88].
- TC1 track vs OFCL at 72 h alone: -4.1 [-7.5, -0.8].
- TC1 vs GDMI on 2025, GDMI's one season: -0.56 [-11.39, +8.10], a tie with the strongest single aid.
- The weighting is worth about 3.7 n mi over equal weights of the same members: EQ - OFCL +0.98, against
  TC1's -2.76.
- Intensity: ties throughout. Against HCCA, TC1+O is -0.52 [-1.09, +0.09], led by 120 h (-2.3 [-3.8, -0.8]).

**2026 so far** (31 storms fetched 2026-10-08 18:53Z, operational best tracks; `season_2026.json`; no claim,
and the numbers move as the season's decks update):
- **Track:**
  - TC1 vs OFCL: -0.90 [-3.95, +2.22];
  - TC1+O vs OFCL: -1.73 [-3.98, +0.57];
  - TC1 vs HCCA: -1.34 [-8.92, +7.92];
  - TC1 vs GDMI: +0.14 [-10.40, +14.04].

  Equal weights would have been far worse this season: EQ - OFCL +15.25 [+3.73, +30.97]. The weighting is
  doing the work.
- **Intensity:**
  - TC1 vs OFCL: +1.04 [-0.64, +2.22];
  - TC1 vs HCCA: +1.68 [-0.22, +3.77].

  HCCA's 2026 intensity is unusually good: 10.41 kt, against OFCL's 11.26.

**What it means.**
- **TC1's track forecast is at the level of NHC's official forecast**, and on 2023-2025 it is better than
  NHC's best objective aid (HCCA) beyond 48 h. It is automatic and issued 30 minutes after the advisory.
- It is not shown to beat OFCL: every interval against OFCL includes 0.
- Its intensity forecast ties the official forecast. It adds nothing on intensity that NHC's consensus does
  not already have.
- **Next:**
  1. TC1 goes live and on the site (its own amendment). Its track is worth showing beside NHC's.
  2. TC2: intensity conditioned on our RI probability. Intensity is where TC1 adds nothing, and RI is where
     every consensus fails.
