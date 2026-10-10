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

## Amendment 1 -- TC1 live (2026-10-08, before the first live forecast)

- **What runs.** The hurricane scorer issues TC1 and TC1+O, track and intensity, for every NHC-area storm it
  scores (`src/hazardpulse/hurricane/tc1_live.py`; `scripts/fetch_and_score.py` `attach_tc1`).
  - Each run replays the current season's decks from the models saved after 2025
    (`results/hurricane_tc1/state.json`, made by `hurricane_tc1.py snapshot`).
  - That is the backtest's own pass. The snapshot is refused unless replaying the fetched 2026 decks from it
    gives what one pass over 2020-2026 gives, bit for bit. It passed on 6,618 track and 6,895 intensity
    forecasts per product.
  - The state carries content digests of `selection.json` and `dev.json`. The scorer refuses a state made from
    another selection; the site refuses DEV numbers the state was not made from.
- **When.** A storm's forecast is for its latest synoptic time t with t + 3 h 30 passed, from the decks as
  they stand at the run.
- **Strict inputs.** Every numbered storm deck of the season that NHC's index lists must be read. If one cannot
  be, TC1 is not issued that run and the log says why. A replay without one storm's verifications would be a
  different forecast under TC1's name.
- **Fail-safe.** TC1 never touches the published RI number. Each storm record carries:
  - `tc1`, with the cycle;
  - TC1 and TC1+O positions and winds at each lead, with each lead's top weights;
  - NHC's official forecast from the same deck.

  The record is kept in the run's forecast file.
- **On the site.** Each NHC storm card shows TC1 beside NHC's official forecast, lead by lead. The hurricane
  page states TC1's DEV numbers, read from `dev.json` bound to the state. "Beats the official forecast" is said
  only if a primary claim was met. None was: the track result is stated as a tie with OFCL, and a gain over HCCA.
- **The live record (prospective, descriptive).** At each look (2026-12-01 and 2027-12-01), the live TC1 and
  OFCL records are scored with this program's verifier against NHC's best tracks. Same rules, homogeneous
  cases, storm-block intervals.
  - **This measures the open risk.** The backtest read final decks, so an aid that arrived after t + 3 h 30
    counted there; the live record has only what existed at the run.
  - The live record makes no claim. A claim needs its own registration.
- **Each season.** In January, `snapshot` is re-run with the finished season added (`season_end` + 1). The
  scorer refuses a state that does not end with the previous season.

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

## Amendment 2 -- TC2: TC1's intensity, held to our RI model's median (2026-10-10, before any TC2 number exists)

**Why.** Intensity is where TC1 adds nothing, and rapid intensification is where every intensity consensus fails.
Isaias 2026 showed both at once:
- on 7 October our in-test RI models gave 62-68% for a 30 kt rise within 24 h, NOAA's DTOPS 15-28%, and the
  storm rose 30-35 kt from four consecutive cycles;
- TC1's intensity trailed NHC's: 24-h errors of 17.8 kt against NHC's official 13.3 kt.

**What this is an instance of: the median of a forecast distribution under an absolute-error score.** NHC scores
intensity by mean absolute error, and the forecast that minimizes expected absolute error is the median of the
predictive distribution. Our RI model (v10.4, the H8 model of the RI program, amendments 8-9) issues a calibrated
24-h exceedance curve: P(V(t+24 h) - V(t) >= k) for k = 15, 20, 25, 30, 35, 40, 45 kt. That curve brackets the median
of the 24-h change **without anything being fitted**:
- if P(>= k) >= 0.5, the median is at least k;
- if P(>= k) < 0.5, the median is below k.

**TC2 is TC1 projected into that bracket.** Track is TC1's, unchanged. Intensity, at each cycle t:
1. **The curve.** v10.4's curve for the cycle, used only where its gate passes (the DSHP, IVCN and NNIC 24-h
   forecasts all exist, `ri_v10.GATE_AIDS`). It is made non-increasing in k by a running minimum. Where there is
   no curve, or the gate fails, TC2 is TC1.
2. **The bracket:** L = the largest k with P(>= k) >= 0.5 (none: no lower bound); U = the smallest k with
   P(>= k) < 0.5 (none: no upper bound).
3. **TC1's 24-h change:** dV = V_TC1(24 h) - V0, where V0 is the CARQ intensity at t, the analysis TC1 uses.
4. **The shift:** s = clip(dV, L, U) - dV. It is zero whenever TC1 already lies inside our bracket.
5. **At each lead tau:**
   - for tau <= 24 h, V_TC2 = V_TC1(tau) + s * tau / 24, so the change accrues linearly over the first day;
   - beyond 24 h, V_TC2 = V_TC1(tau) + s * f(tau), with f(tau) = min(1, max(0, (V_TC1(tau) - 26.7) /
     (V_TC1(24 h) - 26.7))). This is 0 if V_TC1(24 h) <= 26.7.

   So the shift persists while TC1 holds the storm's strength, and decays as TC1 decays it. 26.7 kt is the
   background intensity of Kaplan and DeMaria's (1995) inland decay model, a published constant: an extra
   amount of wind at landfall decays the way the rest of the excess above the background does.
6. **Nothing is fitted.** TC2 has no free parameter, so the development seasons are a clean test of it.

**The curves.**
- **DEV 2023-2025:** out of fold, from the RI program's own machinery. Each season's curve comes from H8 fitted only
  on earlier seasons (folds 2022-2025, `scripts/hurricane_ri_g1.py`'s fold construction).
  - **Control:** those folds' pooled 30/24 log loss must reproduce amendment 8's 0.14161766094567663 to 1e-12, or
    the run stops.
  - Cycles are matched by storm and synoptic time. A TC1 cycle without an H8 row is TC1.
- **2026:** H8 fitted on 2020-2025, which is the v10.4 artifact's model.
- **Live, if carried:** the curve the v10.4 shadow records in the cycle's own forecast record.

**The test** (`scripts/hurricane_tc2.py`, the TC1 program's verifier, rules, truth and leads).
- **Primary:** DEV 2023-2025, AL and EP pooled, intensity. TC2 against TC1 on the cases both verify (TC2 exists
  exactly where TC1 does).
- **Carried iff** the mean over 24-120 h of the per-lead mean absolute error difference (TC2 minus TC1) is below 0
  AND the 24-h difference is below 0 (point estimates).
- **Reported (95% storm-block bootstrap, 4,000 draws, seed 20261008, as in this program):**
  - each lead, 12 h included;
  - TC2 against OFCL, HCCA and IVCN;
  - **the RI subset**, cycles whose best track rose >= 30 kt over the next 24 h: TC2, TC1 and OFCL at 12 and 24 h;
  - how often the shift moves TC1 up and down, and by how much;
  - **2026**, operational best tracks: a further read, no claim.

**A carried TC2.** An amendment written before its first live forecast puts TC2's intensity in the live record
beside TC1 and NHC's. The site's TC1 table shows it as TC2's intensity, labelled in test.

**Known before the result.**
- **v10.4 was designed with 2022-2025 in view.** Its features were chosen against those folds, so the out-of-fold
  curves are honest per fold, but the model family is not naive about those seasons. TC2's rule itself is new and
  has nothing fitted.
- **The bracket only binds where the curve is decisive.** P(>= 15) is often below 0.5 for a steady storm, so U = 15
  clips any TC1 rise above 15 kt down to 15. Upward shifts happen only when our model is at least 50% confident
  of a >= 15 kt rise. Most cycles keep TC1's number; the effect is concentrated where TC1 and our RI model disagree.
- **V0 is CARQ's, the label's V(t) is the best track's.** The difference is usually a few kt.
- **Only intensity at 24 h is bracketed directly.** Longer leads carry the 24-h correction by the rule above, not
  by a curve of their own.

### Amendment 2 outcome (2026-10-10): TC2 NOT carried -- better at 24 h, worse from 72 h

`results/hurricane_tc2/dev.json`.

**Controls:**
- the out-of-fold H8 curves reproduced amendment 8's log loss 0.14161766094567663 exactly;
- the 2026 curves reproduced 0.11715873924102127 exactly;
- the TC1 intensity run reproduced TC1's registered DEV error, 12.832416 kt.

**DEV 2023-2025, AL and EP pooled.** Intensity error, mean over 24-120 h: TC1 12.832, **TC2 12.900**, OFCL 12.627,
HCCA 13.223, IVCN 12.781.

| TC2 - TC1 (kt) | d | 95% |
|---|---|---|
| 24 h | **-0.197** | **[-0.38, -0.04]** |
| 48 h | -0.126 | [-0.36, +0.08] |
| 72 h | +0.134 | [-0.27, +0.52] |
| 96 h | +0.252 | [-0.27, +0.74] |
| 120 h | +0.274 | [-0.34, +0.82] |
| mean, 24-120 h | +0.067 | [-0.28, +0.38] |

The mean is above 0, so **TC2 is not carried.**

**What worked: the RI subset** (157 cycles that rose >= 30 kt in 24 h).

| lead | TC1 | **TC2** | OFCL | HCCA |
|---|---|---|---|---|
| 24 h | 23.6 kt | **20.7 kt** | 18.5 kt | 19.8 kt |
| 12 h | 11.7 kt | **10.7 kt** | 9.0 kt | 10.3 kt |

**How often the rule acted.** Of 2,825 cycles with a gated curve, it raised TC1 on 379 and lowered it on 44. The mean
shift was 4.9 kt, between -11.4 and +18.8 kt.

**The root cause.**
- The curve speaks only to the next 24 h, and the bracket is right there: the 24-h gain is significant.
- Carrying the whole shift on to 120 h (the registered persistence) helps through 48 h and hurts from 72 h. By
  day 3 the consensus has caught up with the intensification, so the persisted shift overshoots.
- What survives is the bracket at 24 h. The extension beyond the curve's own horizon is falsified.

## Amendment 3 -- TC2b: the bracket only where the curve speaks (2026-10-10, before any TC2 or TC2b number on 2026)

**The change.** TC2b is TC2 with the shift tapered to zero by 72 h:
- tau <= 24 h: unchanged, s * tau / 24;
- 24 h < tau: s * f_KD(tau) * max(0, (72 - tau) / 48), where f_KD is amendment 2's decay factor.

So 48 h keeps half the shift, and 72 h and beyond equal TC1. **The taper's end (72 h) was chosen from amendment 2's
DEV per-lead pattern** (helpful through 48 h, harmful from 72 h). That makes DEV unfit to test it: TC2b's test is
the 2026 season.

**The test** is 2026, AL and EP pooled, operational best tracks (`hurricane_tc2.py read2026`), the curves from H8
fitted on 2020-2025 (the v10.4 artifact's model, controlled above).
- **Primary:** TC2b - TC1, intensity.
- **Carried iff** the 24-h difference is below 0 AND the mean over 24-120 h is below 0 (point estimates).
  Beyond 48 h TC2b equals TC1, so the mean mostly carries the 24-h and 48-h effects.
- **Reported:**
  - TC2 (amendment 2's rule) on 2026, beside TC2b;
  - TC2b against OFCL, HCCA and IVCN;
  - the RI subset;
  - shift statistics;
  - 95% storm-block intervals.
- **2026 is the season in progress.** No TC2 or TC2b number on 2026 exists at this registration. Only TC1's own
  2026 further read has been computed, in the outcome above.

**A carried TC2b** goes into the live record by its own amendment before its first live forecast, as amendment 2
said for TC2.
- **Next:**
  1. TC1 goes live and on the site (its own amendment). Its track is worth showing beside NHC's.
  2. TC2: intensity conditioned on our RI probability. Intensity is where TC1 adds nothing, and RI is where
     every consensus fails.
