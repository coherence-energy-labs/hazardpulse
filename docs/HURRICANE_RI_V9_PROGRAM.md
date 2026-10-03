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

## Known uncertainty, stated before the result

- The e-deck RI value and the SHIPS-text value are the same quantity rounded to whole percent;
  identity is verified per aid on the cycles where both exist for any season, else assumed
  (the previous programme found no 2025 discrepancy at 30/24).
- Aid suites change between seasons (HWRF/HMON -> HAFS in 2023); grouped means are used so a
  renamed model does not become a missing feature.
- 2026 is one season, of a size the test has to be: development events are ~250, the 2026 test
  has whatever RI events occurred. A real but small improvement may not reach the claim rule.
