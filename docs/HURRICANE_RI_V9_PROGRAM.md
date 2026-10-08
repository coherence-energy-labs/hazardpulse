# Hurricane RI v9 -- beyond DTOPS: pre-registered protocol (written 2026-10-03, before any 2026 outcome)

Goal: a rapid-intensification probability of our own that is better than NOAA's DTOPS -- the best
public RI guidance, which beat every alternative in `docs/HURRICANE_RI_PROGRAM.md` -- decided by a
held-out season no model or choice has seen. If nothing beats DTOPS, DTOPS keeps being served and
that is the reported result.

## What is already known (carried forward, not re-tested)

- 2025 (601 NHC cycles, 44 events), read once by the previous programme: DTOPS AUC 0.928, our v8.2
  0.849; a logit pool of SHIPS-RII/Logistic/Bayesian/DTOPS did not beat DTOPS; v8.2 adds -0.0001 to
  such a pool (2022-2024). Conclusion carried: re-weighting NOAA's 30/24 probabilities is exhausted,
  and v8.2's inputs (analysis intensity, its tendencies, aid intensity forecasts at a few lead
  times, climatological potential intensity) add nothing to them.
- What no previous model used: the other RI thresholds every NOAA aid publishes (20/12 .. 45/36),
  the full set of EARLY intensity guidance in the a-deck (NOAA's statistical models, the regional
  hurricane models HWRF/HMON/HAFS, the global models, the GEFS mean, NHC's consensus aids including
  the neural-network consensus NNIC) as 24-h intensity CHANGES with their agreement and spread,
  and the NHC official forecast.

Seen before this protocol (declared): the count of 2026 SHIPS-text files per storm (no content),
one invest's SHIPS text (format only), and the site's live DTOPS values for three storms on
2026-10-03 (no outcome). No 2026 best track and no 2026 aid-vs-outcome comparison has been read.

## Contract

- **Event**: V(t+24 h) - V(t) >= 30 kt (the previous programme's event and SHIPS-RII's threshold).
- **Cycle**: an NHC-basin (AL/EP/CP) synoptic time t (00/06/12/18Z) of a numbered storm whose
  SHIPS-RII 30/24 probability exists (e-deck RIOD record; live: the SHIPS text's SHIPS-RII value).
- **Inputs at t** -- only what exists when NHC's advisory for t is issued (t + 3 h):
  - O (ours): from the a-deck at t. CARQ: V0 (tau 0), dV over the past 12 h and 24 h (CARQ tau
    -12/-24), MSLP0, |latitude|, an Atlantic indicator (CP counts as EP). EARLY aids' 24-h change
    dV24 = V(tau 24) - V0: DSHP, LGEM, IVCN, HCCA, NNIC individually; the mean and the max over the
    regional models {HWFI, HMNI, HFAI, HFBI, CTCI}; the mean over the global models {AVNI, EMXI,
    EGRI, NVGI}; AEMI (GEFS mean); the spread (std) of all these individual dV24 (>= 3 present) and
    the fraction of them >= 30 kt. Early ("I"/statistical/consensus) aids only: they are built from
    the previous cycle and exist at t; late runs are not inputs.
  - N (NOAA RI aids): logit of SHIPS-RII, Logistic, Bayesian, Consensus and DTOPS at 20/12, 25/24,
    30/24, 35/24, 40/24, 45/36 (whole percent clipped to [0.005, 0.995]). Training: the e-deck RI
    records (RIOD, RIOL, RIOB, RIOC, DTOP). Live and 2026: the SHIPS text's "Matrix of RI
    probabilities", same rows and columns, rounded as `ships_text.parse_ships_text` does. The
    ECMWF-based aids (EIOx), SDCON and DTPE are excluded: they are not in the live text or not in
    every season.
  - H (human): the NHC official forecast's dV24 and dV12 (OFCL tau 24/12 minus V0), issued with
    the advisory at t + 3 h. A product using H is published after the advisory.
  - Missing anything = NaN (never a fill that looks like data).

## Candidates (fixed now)

- **A** DTOPS 30/24 as issued (SHIPS-RII where DTOPS is missing) -- the incumbent.
- **B** our model on O only ("ours alone": no NOAA RI probability, no human forecast).
- **C** O + N.
- **D** O + N + H.

Each of B, C, D in two declared model classes:
- `lr`: L2 logistic regression (C = 1.0) on features standardised with training-fold statistics,
  NaN imputed by the training median plus one missing indicator per feature that has any NaN;
- `gbt`: LightGBM, binary log loss, unweighted, learning rate 0.03, 7 leaves, max depth 3, min 30
  rows per leaf, feature and bagging fraction 0.8 (bagging every iteration), L2 5.0, seed 0; the
  number of trees is chosen inside each training window by early stopping (patience 100, at most
  2,000) on its LAST training season with the earlier seasons as training, then the model is
  refitted on the whole window with that number.

Seven candidates: A, B_lr, B_gbt, C_lr, C_gbt, D_lr, D_gbt.

## Development (seasons 2020-2025)

Case table: the previous programme's cases (2020-2024 and 2025, IBTrACS truth; 4,692 cycles),
each enriched with O, N and H from the archived a- and e-decks. Forward chaining: for each
y in 2022, 2023, 2024, 2025, fit on 2020..y-1 and score y. Criterion: pooled mean log loss over
the four scored seasons (A's whole percent clipped to [0.005, 0.995]); Brier, AUC, calibration
reported. Bootstrap: whole storms (IBTrACS SID), 2,000 draws, seed 20261003, percentile 95%.

**Carried to the final**: the candidate other than A with the lowest pooled development log loss
(an exact tie goes to the earlier in the order B_lr, B_gbt, C_lr, C_gbt, D_lr, D_gbt). Also
carried, for reporting only: the better of B_lr/B_gbt ("ours alone").

## Final (the 2026 season, read once)

- Cases: every 2026 AL/EP/CP numbered-storm cycle (storm number 1-49) with a SHIPS text file in
  `atcf/stext` whose SHIPS-RII 30/24 value is present, and truth.
- Truth: the 2026 b-deck best-track intensity (`atcf/btk`, BEST, tau 0) at t and t + 24 h, both
  present and > 0, as published when the final runs. (Development truth is the post-season
  reanalysis in IBTrACS; 2026 truth is the operational best track -- a stated difference.)
- Features: the 2026 a-deck from `atcf/aid_public` and the SHIPS text, through the same builder.
- The carried candidate is refitted on 2020-2025 (its class's rules) and scored once on 2026,
  with A, the "ours alone" model, SHIPS-RII and the NOAA consensus beside it.
- **Claim rule**: "better than DTOPS" only if the paired storm-bootstrap 95% interval of
  LL(candidate) - LL(A) or of Brier(candidate) - Brier(A) on 2026 lies wholly below 0, AND both
  point differences are <= 0. Then the candidate is served for the NHC basins (DTOPS remains its
  input where it uses N); otherwise DTOPS stays served and the result is reported as it is.
- An independent adversary attacks a claimed result before it is published.

## Outcome (2026-10-03)

**Development** (forward chaining 2022-2025, 2,871 cycles, 193 events; `results/calibration/
hurricane_ri_v9_selection.json`): carried D_gbt, pooled LL 0.1459 vs DTOPS 0.1647, dLL -0.0189
[-0.0294, -0.0084], dBrier -0.0053 [-0.0090, -0.0017], AUC 0.932 vs 0.905; better than DTOPS on LL,
Brier and AUC in each of the four seasons. "Ours alone" (B_lr) LL 0.1513, dLL -0.0134 [-0.0233, -0.0018].

**Final, 2026 read once** (586 cycles, 35 events from 9 storms; `hurricane_ri_v9_final.json`):
D_gbt LL 0.1316 vs DTOPS 0.1495, dLL -0.0180 [-0.0427, +0.0106]; Brier 0.0349 vs 0.0423, dBrier
-0.0074 [-0.0149, +0.0001]; AUC 0.924 vs 0.893. Better on every point estimate, but no interval
lies wholly below 0: **the claim rule is NOT met. DTOPS stays served.** "Ours alone" did not hold
up (LL 0.1597, AUC 0.864).

Found after the read (exploratory -- it cannot turn this result into a claim): the 2026 public
a-deck of CP01 carries none of the early aids (only late global and ensemble runs), while every
development CP case had them (65/65). On those 54 cycles (4 events) D_gbt scored LL 0.2218 vs
DTOPS 0.1305; on the 532 cycles with the aids, dLL -0.0291 [-0.0475, -0.0087], dBrier -0.0101
[-0.0163, -0.0040]. The model is good where its inputs exist and poor where they do not -- a gate,
designed after this read, needs its own unseen test (v9.1 below).

## Amendment 1 -- v9.1, a PROSPECTIVE test (2026-10-03, after the 2026 read; fixed before any cycle it scores)

No untouched past season exists: DTOPS has no e-deck record before 2020 (the 2019 e-decks hold no RI
records at all). The gated model is therefore tested on cycles that have not happened yet.

- **v9.1** = D_gbt exactly as refitted for the final (2020-2025, 166 trees, `results/models/
  hurricane_ri_v9.json`, scored in NumPy) wherever the cycle has the early aids it was trained on --
  all three of DSHP, IVCN and NNIC have a 24-h forecast at the cycle; otherwise DTOPS 30/24 (SHIPS-RII
  where DTOPS is missing). The gate was designed after the 2026 read; that is why this test exists.
- **Serving until the claim**: DTOPS stays the published number. v9.1 runs in SHADOW on every live
  NHC cycle (computed after the advisory, t + 3 h 30 min, so the official forecast exists as in
  training), is written into the forecast record and its replay, and is never shown as the forecast.
- **Test set**: every NHC-basin numbered-storm cycle with synoptic time >= 2026-10-04 00Z that the
  live scorer forecast in shadow, scored against the best track at t and t + 24 h.
- **Two looks**: the end of the 2026 season (2026-12-01) and the end of 2027 (2027-12-01). At each,
  the claim rule above with 97.5% intervals (two looks at 0.025 each). The first look that meets it
  switches serving to v9.1; if neither does, DTOPS stays and the result is reported.
- The model is not refitted, re-tuned or re-gated during the test.

## Amendment 2 -- v10 (2026-10-03, written before any v10 number is computed)

Three things the v9 representation leaves on the table, each a hypothesis to be measured:

1. **The outcome is a number, not a bit.** v9 learns P(dV >= 30) from ~280 events; every one of the
   ~4,700 development cycles has a measured 24-h change dV. v10 learns the whole exceedance curve
   P(dV >= k) for k in {15, 20, 25, 30, 35, 40, 45} kt with ONE model on threshold-stacked rows
   (k is an input, monotone decreasing), so every cycle informs the 30-kt tail and the product is
   a coherent set of thresholds.
2. **How the guidance has been doing on THIS storm.** E group, at cycle t, from the a-deck: for X
   in {DSHP, LGEM, IVCN, HCCA, NNIC, OFCL}, err12_X = V_CARQ(t) - X's forecast made at t-12 h for
   tau 12, and err24_X likewise from t-24 h for tau 24 (positive = the aid under-forecast); their
   means over the aids present; dv_past6 = V0 - CARQ(t-6 h). Models that keep under-forecasting a
   storm are a classic sign that RI has begun and the guidance has not caught it.
3. **Robustness to missing guidance, in the model, not a gate.** Each training row gets one copy
   with the early-aid and E features masked (NaN), so the model learns the regime CP01 exposed.

Candidates (LightGBM, the v9 parameters, each the average of seeds 0-4): V1 = v9's inputs + E
(binary 30 kt); V2 = v9's inputs, threshold-stacked; V3 = v9's inputs + E, stacked; V4 = V3 with
the masking copies. Reference R0 = v9 D_gbt as frozen. Selection: forward chaining 2022-2025
exactly as v9; the V with the lowest pooled 30/24 log loss is carried if it beats R0's pooled log
loss (else v9.1 stays the entrant). Also reported: the sum of Brier scores over the 24-h
thresholds 25/30/35/40 kt, the thresholds DTOPS publishes.

2026 is a SECOND read for v10, declared: its design was informed by the 2026 read (CP01). It is
reported, never used to claim.

The claim for v10 is PROSPECTIVE only, beside v9.1, on cycles >= the first cycle v10 runs in
shadow, at the same looks (2026-12-01, 2027-12-01): primary = the sum of Brier scores over the
25/30/35/40-kt 24-h thresholds, v10 vs DTOPS's own four published probabilities, paired storm
bootstrap; claim if its 98.75% interval lies wholly below 0 AND v10's 30/24 log loss point is
<= DTOPS's (98.75%: two looks for a second entrant, so v9.1 + v10 together stay near 0.05).

### Amendment 2 outcome (2026-10-03)

Development (forward chaining 2022-2025; control: R0 reproduced v9's pooled LL 0.1459 exactly;
`results/calibration/hurricane_ri_v10_selection.json`):

| | 30/24 LL | vs R0 | 4-threshold Brier vs DTOPS (0.1722) |
|---|---|---|---|
| R0 (v9) | 0.1459 | -- | -- |
| V1 (+E, binary) | 0.1480 | +0.0021 [-0.0003, +0.0047] | -- |
| **V2 (stacked)** | **0.1443** | -0.0015 [-0.0057, +0.0027] | **0.1554, -0.0168 [-0.0277, -0.0063]** |
| V3 (stacked +E) | 0.1446 | -0.0013 | 0.1564, -0.0158 [-0.0269, -0.0053] |
| V4 (V3 + masking) | 0.1445 | -0.0014 | 0.1568, -0.0154 [-0.0268, -0.0046] |

**Carried: V2.** Killed: the E features (storm-specific guidance errors) add nothing -- DERIVED
reason: NOAA's interpolated "I" aids are already shifted to the storm's current intensity, so
their recent error is absorbed before the model sees it. The exceedance representation buys a
small, non-significant 30-kt gain and the full threshold product.

2026, DECLARED second read (no claim; `hurricane_ri_v10_2026_second_read.json`): V2 LL 0.1291 vs
DTOPS 0.1495 (dLL -0.0204 [-0.0477, +0.0127]), AUC 0.925, 4-threshold Brier d -0.0202 [-0.0445,
+0.0057]. On the 54 CP01 cycles without guidance V2 scored 0.2408 vs DTOPS 0.1305 (V4's masking:
0.1951) -- the fragility is narrowed by masking but not closed, so v10 is served live with
amendment 1's gate: **v10.1 = V2 where DSHP, IVCN and NNIC are present, else DTOPS (each threshold
DTOPS's own, SHIPS-RII where DTOPS is missing).** v10.1 enters the prospective test as declared
above (first cycle = the first it runs in shadow).

### Amendment 2, descriptive addendum -- v10.1 against every public RI aid (2026-10-03)

Descriptive, NOT a claim: the claim for v10.1 is the prospective test above. The selection compared
v10 with DTOPS alone, because a pre-registered 2020-2024 comparison had found DTOPS NOAA's best. That
is a statement about other seasons, not a measurement on these cases, so
`scripts/hurricane_ri_v10_vs_all.py` (-> `results/calibration/hurricane_ri_v10_vs_all.json`)
compares v10.1 **as served**, V2 with the gate, with every RI probability NOAA publishes, paired by
storm on the cycles where each aid was issued. The forward chaining is the same as in the selection.
Two controls stop the run if they fail:

- V2's pooled log loss must equal the selection's, bit for bit.
- Every gate-closed cycle must carry DTOPS's own value.

The gate was open on 2,825 of the 2,871 cycles (193 RI events).

| NOAA RI aid | cycles | aid 30/24 LL | v10.1 | dLL [95%] | 4-threshold Brier aid / v10.1, d [95%] |
|---|---|---|---|---|---|
| SHIPS-RII (RIOD) | 2,871 | 0.1995 | 0.1445 | -0.0551 [-0.0687, -0.0390] | 0.1986 / 0.1555, -0.0431 [-0.0572, -0.0292] |
| RI logistic (RIOL) | 2,871 | 0.1970 | 0.1445 | -0.0525 [-0.0664, -0.0396] | 0.2048 / 0.1555, -0.0494 [-0.0654, -0.0346] |
| RI Bayesian (RIOB) | 2,871 | 0.2324 | 0.1445 | -0.0879 [-0.1232, -0.0583] | 0.2093 / 0.1555, -0.0538 [-0.0762, -0.0331] |
| RI consensus (RIOC) | 2,871 | 0.1834 | 0.1445 | -0.0390 [-0.0506, -0.0275] | 0.1903 / 0.1555, -0.0348 [-0.0487, -0.0219] |
| DTOPS (DTOP) | 2,740 | 0.1656 | 0.1479 | -0.0177 [-0.0290, -0.0072] | 0.1748 / 0.1594, -0.0154 [-0.0268, -0.0047] |

Every interval lies below zero, both in 30-kt log loss and in the four-threshold Brier
(MEASURED, development period).

**Intensity forecasts read as yes/no RI calls** (forecast dV24 >= 30 kt). The fair test of a
probability against a single call: set our threshold so that we raise no more false alarms than the
call did, then count the RI events each catches. Ties at the threshold count against us. The per-model
hurricane guidance is not kept in the development table, only its mean and its most aggressive member.

| call | RI calls | its false-alarm rate | its POD | v10.1 POD | d [95%] |
|---|---|---|---|---|---|
| NHC official forecast | 94 | 1.32% | 0.342 | 0.353 | +0.011 [-0.116, +0.104] |
| HCCA | 61 | 0.63% | 0.236 | 0.209 | -0.026 [-0.172, +0.108] |
| IVCN | 17 | 0.11% | 0.073 | 0.047 | -0.026 [-0.099, +0.070] |
| SHIPS (DSHP) | 63 | 1.29% | 0.150 | 0.342 | **+0.192 [+0.056, +0.311]** |
| LGEM | 28 | 0.42% | 0.088 | 0.176 | +0.088 [-0.060, +0.191] |
| NNIC | 76 | 0.94% | 0.264 | 0.316 | +0.052 [-0.072, +0.107] |
| GEFS mean | 0 | 0.00% | 0.000 | 0.031 | +0.031 [+0.000, +0.086] |
| hurricane-model mean | 31 | 0.31% | 0.120 | 0.094 | -0.026 [-0.123, +0.136] |
| most aggressive hurricane model | 142 | 2.63% | 0.387 | 0.471 | +0.084 [-0.012, +0.216] |
| global-model mean | 2 | 0.00% | 0.010 | 0.031 | +0.021 [-0.014, +0.079] |

- v10.1 catches significantly more RI than the SHIPS call.
- It is within noise of every other call. Its point estimate is below on three of them (HCCA, IVCN,
  hurricane-model mean), by about 5 of 191 events each.
- These operating points allow 3 to 70 false alarms in four seasons, so the test cannot separate
  gaps this size. The development data cannot decide whether a variant closes them, and neither can
  the prospective test for years. So no variant was searched for.
- What this rules out is the unqualified phrase "beats every forecast". What survives is "beats
  every RI probability NOAA publishes, and ties the best yes/no calls at their own false-alarm
  rates".

**Calibration check -- KILLED: "recalibrate v10 upward".**
- Dev observation: out of sample, v10's 2022-2025 forecasts look under-confident at the top. The
  45-60% bin verified at 69% (n 61), and the pooled mean is 0.058 against a rate of 0.067.
- Per season, the logistic calibration of logit(p) is:

  | season | slope | intercept | RI rate |
  |---|---|---|---|
  | 2022 | 0.93 | -0.25 | 5.1% |
  | 2023 | 1.12 | +0.73 | 6.8% |
  | 2024 | 1.13 | +0.61 | 7.9% |
  | 2025 | 1.41 | +0.55 | 7.3% |

  The intercept moves with the season's RI rate, which is weather, not a model property.
- Smallest witness: 2026, the served model. Slope 0.99, intercept -0.19, mean 0.067 against a rate
  of 0.060. A calibrator fit on 2022-2025 would have moved 2026 the wrong way.
- Survives narrower: nothing. In-sample the top bin is also compressed (about 0.70 forecast against
  0.89-1.00 observed), but out of sample its sign changes between seasons (2022: 0.65 against 0.43).
- Resurrect if: a within-season signal, such as the season's verified RI rate so far, is shown on
  the development seasons to predict the intercept.

**Open:** the prospective comparison against all five aids. It needs no new recording: each published
forecast file (`dist/data/replay/hu_fcst_*.json`) keeps, for the cycle the shadow scored, the SHIPS
text's 30/24 values for RIOD, RIOL, RIOB, RIOC and DTOP (`ri_inputs.ships_text.whole_percent`).
This was checked on hu_fcst_20261003_1223.

## Amendment 3 -- challengers to v10.1 (2026-10-03, written before any challenger number is computed)

Standing rule (Josh, 2026-10-03): a win is a checkpoint, not a stop. From here on, every
improvement is a CHALLENGER to the live champion (v10.1). It is selected on the development seasons
under a rule fixed in advance. It then earns any promotion on cycles it never saw.

**What motivates these two** (the descriptive addendum above): at the false-alarm rates of the
HCCA, IVCN and hurricane-model-mean RI calls, v10.1 only ties them. The top of its ranking is
where it is weakest.

- **M -- monotone constraints.** A higher guidance forecast or a higher NOAA probability may never
  lower ours. The constraint is +1 on every guidance dV24 (DSHP, LGEM, IVCN, HCCA, NNIC, AEMI,
  hurricane-model mean and max, global mean), on frac_ge30, on OFCL dV24 and dV12, and on all 30
  NOAA RI logits. The threshold column stays -1, and every other feature is unconstrained.
- **R -- revisions since the previous cycle** (t - 6 h, same storm). Twelve features:
  - for DSHP, LGEM, IVCN, HCCA, NNIC, the hurricane-model mean and OFCL: dV24(t) - dV24(t-6);
  - frac_ge30(t) - frac_ge30(t-6);
  - logit(t) - logit(t-6) of DTOPS, SHIPS-RII and the RI consensus at 30/24;
  - the observed dV over the past 6 h (CARQ).
  - A missing previous cycle gives NaN. Live, everything at t-6 comes from that cycle's a-deck and
    SHIPS text, both public well before t.

**Candidates:** V5 = V2 + M; V6 = V2 + R; V7 = V2 + R + M. Data, folds (2022-2025), seeds,
LightGBM settings and stacking are V2's.

**Control:** V2, recomputed by the same script, must reproduce its selection log loss
(0.1443435895033004), or the run stops.

**Carried rule:** the candidate with the lowest pooled 30/24 log loss, among those whose pooled
30/24 log loss AND pooled four-threshold Brier (25/30/35/40) are both below V2's. An exact tie goes
to the earlier in V5, V6, V7. If none qualifies, V2 stays champion and the candidates are recorded
as kills.

**Reported for every candidate:** paired dLL vs V2 (95%, storm bootstrap), and POD at the HCCA
call's false-alarm rate.

**2026:** a declared THIRD read. It is reported and no claim is made from it.

**A carried candidate (v10.2)** runs live in shadow beside v10.1. Its promotion over v10.1 is
decided only on cycles it never saw, under a rule written (amendment 4) before it scores its first
cycle.

### Amendment 3 outcome (2026-10-03)

`scripts/hurricane_ri_v10_challengers.py` -> `results/calibration/hurricane_ri_v10_challengers.json`.
The control passed: V2 reproduced 0.1443435895033004. 4,400 of 4,692 cycles had the previous cycle.

| | dev 30/24 LL | vs V2 [95%] | dev Brier4 | vs V2 [95%] | AUC | POD at HCCA's FAR (HCCA 0.236) | 2026 LL | 2026 Brier4 |
|---|---|---|---|---|---|---|---|---|
| V2 (v10.1) | 0.1443 | -- | 0.1554 | -- | 0.9304 | 0.209 | 0.1291 | 0.1274 |
| **V5 (+M)** | **0.1436** | -0.0007 [-0.0022, +0.0008] | **0.1544** | -0.0010 [-0.0027, +0.0005] | 0.9322 | 0.215 | 0.1262 | 0.1261 |
| V6 (+R) | 0.1441 | -0.0002 [-0.0013, +0.0009] | 0.1549 | -0.0005 [-0.0017, +0.0007] | 0.9314 | 0.204 | 0.1274 | 0.1255 |
| V7 (+R+M) | 0.1440 | -0.0003 [-0.0018, +0.0013] | 0.1548 | -0.0006 [-0.0021, +0.0009] | 0.9320 | 0.204 | 0.1258 | 0.1255 |

The 2026 DTOPS reference is LL 0.1495 and Brier4 0.1476 (586 cycles, third read).

**Carried: V5 -> v10.2** (all three were eligible; V5 has the lowest log loss).
- The gain is small, and no interval excludes 0. It goes the same way on both metrics and on the
  2026 third read. The prior is structural: a model should not lower RI odds because the guidance
  forecasts more strengthening.
- **R adds nothing** (-0.0002). KILLED: "how the guidance changed since the last cycle predicts RI
  beyond the guidance itself".
- What survives narrower: nothing. The revision is a difference of two inputs the model already
  sees at t and, through the previous cycle's case, implicitly in training.
- **The top-end gap to HCCA's yes/no call is not closed by M** (0.215 against 0.236). With the
  inputs we have, the cheap levers are exhausted.
- **The next material gain needs new information**, not a reshaped model. See
  `docs/MODEL_IMPROVEMENT_LEDGER.md`.

## Amendment 4 -- v10.2 in the prospective test, and which model the site shows (2026-10-03, before v10.2 scores any cycle)

- **v10.2** = V5 with v10.1's gate. Artifact `results/models/hurricane_ri_v10_2.json`, label "v10.2",
  v10.1's schema and inputs. It runs live in shadow (`ri_v10_2_shadow`) from the same single read
  as v9.1 and v10.1. Its first scored cycle is the first at or after 2026-10-04 00Z, the shared
  start.
- **Its claim** is v10.1's rule (four-threshold Brier vs NOAA's own values, plus the 30/24 log-loss
  point) at **99.375%**, at the same looks (2026-12-01, 2027-12-01).
- **Error budget:** each new entrant gets half the previous one's (v9.1 2.5%, v10.1 1.25%, v10.2
  0.625%, the next 0.3125%, ...). The family total stays below 5% however many challengers ever
  enter, and no later entrant can spend an earlier one's budget.
- **Challenger vs champion, descriptive:** the prospective scorer reports v10.2 minus v10.1 on the
  cycles both scored (four-threshold Brier, storm bootstrap).
- **Which model the site's "HazardPulse model" column shows:** v10.1 until the first look. At each
  look, in this order:
  1. the entrant whose claim is met (the newest if several);
  2. else the newest carried challenger, provided its descriptive comparison with the shown model
     has a point estimate <= 0;
  3. else the shown model stays.

  The switch is made by a commit at the look, citing the scorer's frozen output.

## Amendment 5 -- convective structure from geostationary IR (2026-10-03, before any IR feature is compared with an outcome)

**Why.** Amendment 3 exhausted the cheap levers on our inputs, so the next gain needs new
information (`docs/MODEL_IMPROVEMENT_LEDGER.md`, H1). Satellite convective structure enters
NOAA's RI aids through their GOES predictors. Here it becomes a direct input.

**Source:** NOAA GMGSI longwave IR, `s3://noaa-gmgsi-pds/GMGSI_LW/`.
- Global, hourly, at about 0.072 degrees (8 km); 8-bit counts, where higher means colder.
- Archived from **2021-07-12**, and live about 35-40 minutes after each hour.
- The product's file naming changed between 2024 and 2025 (to `v3r0_blend`). The grid, the
  encoding and the label are the same; drift is checked below.

**Images and centre.**
- Two images per cycle t:
  - **t + 2 h**, the newest image certain to exist when the live forecast runs at t + 3 h 30;
  - **t - 4 h**, six hours earlier, for trends.
- The centre is the CARQ position at t, extrapolated along the t - 6 h -> t motion to each image's
  hour.
- Crops are +-4 degrees (`scripts/hurricane_ir_crops.py`). A missing image gives NaN.

**IR features (14):** `src/hazardpulse/hurricane/ir_features.py`, the same code for training and
live. All are in counts. A region with fewer than 50% valid pixels is NaN.
- Means in the 0-50, 50-200 and 200-300 km rings.
- Std of 50-200 km.
- Fraction of pixels with count >= 195 in 50-200 km and in 0-300 km.
- Fraction >= 215 in 0-100 km.
- Azimuthal asymmetry: the std of the eight octant means in 50-200 km.
- Eye contrast: the 25-75 km mean minus the 0-25 km minimum.
- Max count in 0-50 km.
- The t+2h minus t-4h change of the 0-50 km mean, the 50-200 km mean, the 50-200 km cold
  fraction, and the asymmetry.

**Candidate:** **V8 = V5 + IR**, with V5's settings and monotone set (the IR features are
unconstrained). Training rows before 2021-07-12 carry NaN IR. Every scored fold (2022-2025) has IR.

**Control:** V5, recomputed by the same script, must reproduce its amendment-3 log loss
(0.14360848294226844, from `hurricane_ri_v10_challengers.json`), or the run stops. The value was
first typed here as "0.143633..."; it was corrected in a separate commit before any IR result.

**Carried rule (amendment 3's):** V8 is carried iff its pooled 30/24 log loss AND its pooled
four-threshold Brier are both below V5's.
- Reported: paired 95% intervals vs V5, and POD at the HCCA call's false-alarm rate.
- Reported: a **drift check**. For each IR feature, the 2025 median minus the 2022-2024 median, in
  pooled-SD units; a shift above 1 SD is flagged. This is descriptive and cannot change the rule.

**2026:** a declared fourth read (no claim).

**A carried V8 (v10.3):** runs in shadow with the IR read live at hour t + 2 h. It enters the
prospective test at half of v10.2's error budget (99.6875%), under amendment 4's display rule.

### Amendment 5 outcome (2026-10-03)

**Data.** `scripts/hurricane_ir_crops.py` made 8,185 crops from 2,431 image hours. 21 hours have
no image in NOAA's archive, and those cycles are NaN as registered.
- IR is present on 2,829 of the 2,871 scored development cycles, and on all 586 cycles of 2026.
- Refactor control: the crop code was moved into the library (`hazardpulse.hurricane.ir_source`)
  that the live scorer uses. A stored crop was reproduced bit for bit from a fresh fetch.

**Control:** V5 reproduced 0.14360848294226844.

| | dev 30/24 LL | dev Brier4 | AUC | POD at HCCA's FAR (HCCA 0.236) | 2026 LL (4th read) | 2026 Brier4 |
|---|---|---|---|---|---|---|
| V5 (v10.2) | 0.1436 | 0.1544 | 0.9322 | 0.215 | 0.1262 | -- |
| **V8 = V5 + IR** | **0.1418** | **0.1527** | **0.9334** | 0.209 | **0.1178** | -- |

**V8 - V5:**
- dev: dLL -0.0018 [-0.0054, +0.0015], dBrier4 -0.0016 [-0.0062, +0.0026];
- 2026: **dLL -0.0084 [-0.0152, -0.0018]**, dBrier4 -0.0076 [-0.0151, -0.0006].

**Drift check:** no feature is flagged. The largest 2025-vs-2022-2024 median shift is 0.20 SD.

**Carried: V8 -> v10.3.**
- IR is the first new-information gain, 2.5x the monotone gain on dev.
- On 2026, the season trained with the most IR history, it is the largest gain of any challenger.
  Its interval excludes 0 there. It is a declared fourth read, so it supports the result without
  being a claim.
- The top-end gap to the HCCA call is not closed by IR either (0.209 vs 0.236). Ledger H6 stays
  open.

## Amendment 6 -- v10.3 in the prospective test (2026-10-03, before v10.3 scores any cycle)

- **v10.3** = V8 with v10.1's gate. Artifact `results/models/hurricane_ri_v10_3.json`, label
  "v10.3", inputs ONH + IR.
- **Live IR.** The scorer reads GMGSI at hour t + 2 h and t - 4 h, once per hour per run. It crops
  and featurises them with the training code (`ir_source`, `ir_features`).
  - A missing image or a fetch error gives NaN inputs, as a missing image did in training. It is
    recorded in the shadow as `ir`.
  - A missing IR *reader* (h5py) means v10.3 is **not scored**. Recording forecasts without IR
    under its name would be a silently different model.
- **Its claim** is v10.1's rule at **99.6875%**, half of v10.2's error budget, at the same looks.
  Its first scored cycle is the first after it is deployed.
- **Descriptive comparisons:** v10.3 vs v10.1 and v10.3 vs v10.2 on shared cycles. Amendment 4's
  display rule applies unchanged: the newest carried challenger is v10.3.

## Amendment 7 -- what counts as a test cycle (2026-10-05 ~02Z, before any test cycle has matured)

Written before the first test outcome can exist: the earliest test cycle (2026-10-04 06Z) matures at
2026-10-05 06Z. No outcome of any test cycle has been looked at.

**Why.** Amendment 1 defined the test set as "every cycle ... that the live scorer forecast in shadow".
That made GitHub's scheduler part of the test design. An audit (2026-10-05) found:
- **Missed cycles.** Scheduled runs were dropped or started hours late. From 2026-10-01 00Z to 10-04 18Z,
  Nolo, Rachel and Choi-Wan each missed 9 of 16 cycles, and EP15 and EP18 both missed 10-04 00Z.
- **No catch-up.** The scorer reads only each storm's latest cycle, so a missed cycle never returns.
- **Preliminary inputs.** A run that lands soon after a cycle reads preliminary inputs, as Rachel
  10-03 18Z did at t + 46 min: no OFCL, no IR, and a SHIPS text NHC later rewrote, so DTOPS 0% became
  SHIPS-RII 13%. Nolo 10-03 12Z is the same case: v10.1 gave 0.076 before the advisory and 0.181 after.
- **Repeats.** Repeat runs of one cycle were kept as separate records (Nolo 12Z: 6).

None of these choices depends on outcomes, but all of them change which records are scored.

1. **The unit is the storm-cycle.** Each NHC-basin numbered storm at synoptic time t has at most one
   test record. That record is the first one made at or after t + 3 h 30 min, judged by the time the
   record itself carries.
   - Records made earlier read preliminary inputs, so they are never test records.
   - Later records of the same cycle are duplicates and are not scored.
2. **Catch-up.** From this amendment on, each scorer run also forecasts, in shadow, every cycle that
   meets all of these:
   - the storm is active;
   - t >= 2026-10-04 00Z and t + 3 h 30 min has passed;
   - the cycle has no test record yet.

   It reads the deck and the SHIPS text as they stand at that run, in the same single read. A catch-up
   record carries `catch_up: true` and its lag (record time minus t). It counts only if made before
   t + 12 h, which keeps it within one cycle of operational timing.
3. **Cycles missed before this amendment.** These are EP15 and EP18 at 10-04 00Z, and any other cycle
   between 10-04 00Z and the first run under rule 2.
   - They may be rebuilt from archived inputs (a-deck, the SHIPS archive, GMGSI), but only by a
     procedure that, run blind on the 6 cycles that already have qualifying live records (EP15 and
     EP18 at 10-04 06, 12 and 18Z), reproduces all four shadows' probabilities exactly (to 1e-12).
   - If the procedure cannot do that, those cycles stay missing. Rebuilt records carry `rebuilt: true`.
4. **Outcomes** are the best-track intensity change from t to t + 24 h, for the storm's own synoptic
   time t. The run time never enters. This applies to the prospective scorer and to the verifier of
   the published numbers.
5. **Reported both ways.** At each look the claim is evaluated on the full test set (rules 1-3). The
   result is also reported without catch-up and rebuilt records, so the effect of this amendment is
   visible. The claim rule, the looks and the error budgets of amendments 1, 4 and 6 are unchanged.

### Amendment 7, implementation record (2026-10-05)

- Rules 1, 2 and 4 live in one module, `hazardpulse.hurricane.cycle_records`, used by the live scorer
  (catch-up), the prospective test and the verifier of the published numbers.
- Rule 3: `scripts/rebuild_hurricane_cycles.py` was run blind on the 6 qualifying live records
  (EP15 and EP18 at 10-04 06, 12 and 18Z). It reproduced 24 of 24 shadows: probability, curve, model
  probabilities and every full-precision input identical (max |d| 0.0). EP15 and EP18 at 10-04 00Z
  were then rebuilt (`results/hurricane_prospective/rebuilt/hu_rebuilt_20261005_0251.json`,
  `rebuilt: true`). The file records the control and every input read.
- On the records of 2026-10-05 ~03Z, the test set is 8 storm-cycles: EP15 and EP18 at 10-04 00Z
  (rebuilt), 06, 12 and 18Z.
- Disclosure: before the rebuild, while checking the published-number verifier (~02:20Z), this
  implementation fetched the operational best tracks of EP15 and EP18. They then held fixes through
  10-05 00Z, which is the 24-h fix of the two rebuilt cycles. Their 24-h change was not computed.
  One endpoint was printed: EP18 at 85 kt at 10-05 00Z, shown as the old verifier's misaligned end
  fix for the 06Z record. The rebuild has no choice an outcome could steer: the procedure is fixed
  and its control is exact. The disclosure is here so a reader can judge.

## Amendment 8 -- H8: the coherence equation as the vortex's balanced response (2026-10-08 ~17:40Z, before any H8 feature is compared with an outcome)

**Why.** IR (amendment 5) says where the cold cloud is. It does not say whether the vortex can use
it. A balanced vortex spins up efficiently only from heating held by its own inertial stability:
heating inside the local Rossby radius `ell = c / I` (Schubert and Hack 1982; Vigh and Schubert
2009). Heating outside that radius radiates away as gravity waves.
- That balance is the coherence equation of `hazardpulse.coherence.tau_c_solver` in its
  Rossby-adjustment form, with a coherence length that varies in space.
- The earlier uses of the operator here were analogies, and both were killed in scope: the tornado
  environment field at 9 km and the earthquake CFT signatures (`docs/MODEL_IMPROVEMENT_LEDGER.md`,
  Killed). In H8 the operator is the physics itself.

**Features (4):** `src/hazardpulse/hurricane/balanced_response.py`, the same code for training and
live. Every constant is declared in that file and none is fitted to an outcome.
- **Heating.** `S = max(count - COLD, 0)` (ir_features' COLD), on a storm-centred isotropic 8 km
  grid of +-400 km. Each cell is the mean of its valid pixels; count 255 is missing.
- **Coherence length.** `ell(r) = c / I(r)`, with `c` = 50 m/s, from a modified Rankine vortex
  (decay 0.5) with `I^2 = (f + 2v/r)(f + zeta)`, floored at `f^2`. Its inputs are all at analysis time t:
  - Vmax = `v0`;
  - the RMW, read from the cycle's CARQ line (ATCF field 20, nm; 0 means unknown);
  - `f` at `abs_lat`.
- **Balanced state.** Solve `lap(tau) - tau / ell^2 + S / ell^2 = 0`, zero at the grid edge, to a
  relative residual of 1e-8. The solver raises rather than return an uncertified field.
- **The four features:**
  - `h8_core_balanced`: the mean of `tau` within the RMW;
  - `h8_core_heating`: the mean of `S` within the RMW;
  - `h8_core_retention`: balanced / (heating + 1);
  - `h8_core_balanced_d6`: core balanced at t + 2 h minus at t - 4 h, with the same vortex.
  Images and centres are amendment 5's.
- **NaN** when:
  - the image is missing;
  - Vmax or the RMW is missing;
  - fewer than 50% of the grid cells within 300 km hold a valid pixel.
- **Inputs measured before this registration** (no outcome read):
  - CARQ RMW is present on 4,662 of the 4,692 development cycles (missing: 9 in 2020, 20 in 2022, 1 in
    2024) and on all 586 cycles of 2026.
  - One solve takes about 0.2 s.
- `tests/test_balanced_response.py` pins the physics before any outcome:
  - `ell` is short in the core and tends to `c / f` far away;
  - heating inside the RMW is retained more than 10x better than the same heating 150 km out;
  - a stronger, tighter vortex retains more;
  - missing data give NaN, never a number.

**Candidate:** **H8 = V8 + the 4 H8 features**, with V8's settings and monotone set (the H8 features
are unconstrained). Rows without IR carry NaN H8.

**Control:** V8, recomputed by the same script, must reproduce amendment 5's development log loss
(0.1417825586152726, from `hurricane_ri_v10_ir.json`) to 1e-12, or the run stops.

**Carried rule (amendment 5's):** H8 is carried iff its pooled 30/24 log loss AND its pooled
four-threshold Brier are both below V8's.
- Reported: paired 95% intervals vs V8, and POD at the HCCA call's false-alarm rate.
- Reported: amendment 5's drift check for the H8 features.
- Reported: the share of split gain the H8 features take in the model fitted on all development
  seasons.
- All three are descriptive and cannot change the rule.

**2026:** a declared further read (no claim).

**A carried H8 (v10.4).** Before it scores any cycle, an amendment like amendment 6 registers it:
- live inputs: the IR images and the CARQ RMW;
- its claim at half of v10.3's error budget (99.84375%), under amendment 4's display rule.

**Known before the result:**
- **H8 overlaps IR.** For a typical RMW, core heating is close to `ir_mean_0_50`. What is new is
  where the heating sits relative to the RMW, and the vortex's inertial stability. The expected
  gain is small; noise the size of amendment 5's dev interval (about +-0.0035 in LL) can decide
  the point-estimate rule either way.
- **The constants are textbook values** (c, the Rankine decay), not tuned.
- **The RMW is an operational estimate.** It is in 5-nm steps and poorly known for weak systems.
- **The grid edge.** It sits at 400 km, where `ell` far out is about 1,000 km. The core features
  are governed by the core's `ell` (about 15-60 km). The edge is a declared choice, not a tuned one.

### Amendment 8 outcome (2026-10-08)

**Run** (`scripts/hurricane_ri_h8.py`, `results/calibration/hurricane_ri_h8.json`).
- H8 is present on 3,501 of the 4,692 development cycles: every cycle with IR and an RMW. It is
  present on 2,828 of the 2,871 scored cycles, and on all 586 cycles of 2026.
- A first run was stopped before it produced any feature. The harness held every storm's whole
  a-deck in memory and passed 2.4 GB. It now keeps only each deck's CARQ lines; the RMW is
  identical on all 5,278 cycles (commit f0789f700). The feature code is unchanged since the tag.

**Control:** V8 reproduced 0.1417825586152726.

| | dev 30/24 LL | dev Brier4 | AUC | POD at HCCA's FAR (HCCA 0.236) | 2026 LL (further read) | 2026 Brier4 | 2026 AUC |
|---|---|---|---|---|---|---|---|
| V8 (v10.3) | 0.1418 | 0.1527 | 0.9334 | **0.209** | 0.1178 | 0.1185 | **0.9410** |
| **H8 = V8 + balanced response** | **0.1416** | **0.1526** | **0.9340** | 0.194 | **0.1172** | **0.1172** | 0.9388 |

**H8 - V8:**
- dev: dLL -0.00016 [-0.00089, +0.00054], dBrier4 -0.00013 [-0.00083, +0.00060];
- 2026: dLL -0.0006 [-0.0028, +0.0020], dBrier4 -0.0013 [-0.0029, +0.0001].

**Descriptive.** The four H8 features take 2.4% of the split gain in the model fitted on every
development season. Drift is not flagged; the largest shift is 0.31 SD.

**Carried: H8 -> v10.4**, by the registered rule. What the numbers say:
- The balanced response adds information the model uses, and the gain points the same way on 2026,
  the season with the most IR history.
- It is about a tenth of IR's gain. Every interval covers 0.
- It lowers the top end on dev (POD at HCCA's false-alarm rate 0.209 -> 0.194) and the 2026 AUC
  (0.9410 -> 0.9388). It improves calibration more than ranking.
- The ledger records why it is small: the core heating overlaps the IR rings, and CARQ's RMW is
  coarse.

As registered, v10.4 enters the prospective test only through an amendment written before it scores
any cycle (amendment 9).

## Amendment 9 -- v10.4 in the prospective test (2026-10-08, before v10.4 scores any cycle)

- **v10.4** = H8 with v10.1's gate. Artifact `results/models/hurricane_ri_v10_4.json`, label "v10.4",
  inputs ONH + IR + H8. It is refit on every development season by `scripts/hurricane_ri_h8.py export`,
  which refuses unless:
  - the refit reproduces amendment 8's 2026 log loss to 1e-12;
  - V8 there is the served v10.3;
  - the artifact reproduces the boosters to 1e-9.
- **Live H8.** The scorer computes H8 once per cycle with the training code (`balanced_response`):
  - from the same two crops as v10.3's IR, cut from the run's cached GMGSI images;
  - with `v0` and `abs_lat` from the training feature builder;
  - with the RMW from the cycle's first CARQ tau-0 line that gives one.
  The shadow records `h8`: "ok", or which inputs were missing and why. A missing input gives NaN, as in
  training. Without the IR reader (h5py), v10.4 is not scored at all, as for v10.3.
- **Its claim** is v10.1's rule at **99.84375%**, half of v10.3's error budget, at the same looks.
  Its first scored cycle is the first after it is deployed.
- **Descriptive comparisons:** v10.4 vs v10.1, and v10.4 vs v10.3 (the model it was selected
  against), on shared cycles.
- Amendment 4's display rule applies unchanged: the newest carried challenger is v10.4.

## Known uncertainty, stated before the result

- The e-deck RI value and the SHIPS-text value are the same quantity rounded to whole percent;
  identity is verified per aid on the cycles where both exist for any season, else assumed
  (the previous programme found no 2025 discrepancy at 30/24).
- Aid suites change between seasons (HWRF/HMON -> HAFS in 2023); grouped means are used so a
  renamed model does not become a missing feature.
- 2026 is one season, of a size the test has to be: development events are ~250, the 2026 test
  has whatever RI events occurred. A real but small improvement may not reach the claim rule.
