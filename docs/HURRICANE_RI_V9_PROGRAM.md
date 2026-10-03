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

## Known uncertainty, stated before the result

- The e-deck RI value and the SHIPS-text value are the same quantity rounded to whole percent;
  identity is verified per aid on the cycles where both exist for any season, else assumed
  (the previous programme found no 2025 discrepancy at 30/24).
- Aid suites change between seasons (HWRF/HMON -> HAFS in 2023); grouped means are used so a
  renamed model does not become a missing feature.
- 2026 is one season, of a size the test has to be: development events are ~250, the 2026 test
  has whatever RI events occurred. A real but small improvement may not reach the claim rule.
