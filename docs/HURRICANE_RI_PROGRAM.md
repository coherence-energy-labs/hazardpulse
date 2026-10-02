# Hurricane RI program -- pre-registered protocol (written 2026-10-02, before any 2025 number)

Goal: the best achievable 24 h rapid-intensification probability for live storms, any
combination of sources, decided by held-out numbers. Written BEFORE the 2025 season is
scored so the choice cannot bend toward it. Amendments are appended, dated, with the reason.

## Why this program exists (measured, results/calibration/hurricane_vs_ships.json)

On the 2,221 held-out 2022-2024 NHC-basin cases, NOAA's operational aids beat v8.2:
v8.2 AUC 0.845, SHIPS-RII 0.882, RI consensus 0.874, DTOPS 0.906; a logistic stack of
RIOD/RIOL/RIOB/DTOP 0.909, and v8.2 adds -0.0001 to it. Live v8.2 also read OFCL tau 0 (no
pressure) until b28c2b3ef: AUC 0.824 live vs 0.832 with CARQ inputs.

## Contract

- **Event**: V(t+24 h) - V(t) >= 30 kt, V = IBTrACS USA_WIND (HURDAT2 for AL/EP/CP), both
  fixes present -- the v8.2 evaluation's truth and SHIPS-RII's own threshold.
- **Cases**: every NHC-basin (AL/EP/CP) synoptic cycle (atcf_id, YYYYMMDDHH) whose e-deck
  holds a 30 kt / 24 h RIOD record, 2020-2025, with truth.
- **Inputs**: the e-deck RI records with TAU=24, dV=30, start 0, stop 24 for RIOD (SHIPS-RII),
  RIOL (logistic), RIOB (Bayesian), DTOP (DTOPS), whole percent. LIVE, the same numbers come
  from the SHIPS text (`atcf/stext/{YYMMDDHH}{BB}{NN}{YY}_ships.txt`, "Matrix of RI
  probabilities", rows SHIPS-RII / Logistic / Bayesian / DTOPS, column 30/24), rounded to
  whole percent so training and serving share one representation. Logits use the
  probability clipped to [0.005, 0.995] (half the reporting resolution).
- **Splits**: development seasons 2020-2024; FINAL season 2025, read once by the final step.

## Candidates

A. DTOPS raw. B. SHIPS-RII raw. C. RIOC raw (NOAA's published consensus).
D. Logit pool: logistic regression on the clipped logits of RIOD, RIOL, RIOB, DTOP plus an
   intercept (Newton, the benchmark's `fit_stack`). Theory says a pool of calibrated
   forecasts needs weights summing above 1 (Ranjan & Gneiting 2010); the fit decides.
E. D plus logit(v8.2) on CARQ live inputs (the benchmark's model C: members <= 2018,
   calibration 2019-2021, so no development season is in-sample for it).
F. D plus a basin indicator (AL vs EP/CP) and its interaction with the DTOPS logit.

Missing aids at a cycle: each of D/E/F is also fitted on every availability pattern that
occurs (e.g. no DTOP); a cycle uses the fit for its own pattern. Patterns are reported.

## Selection (development seasons only)

Forward-chaining: for y in 2022, 2023, 2024, fit on 2020..y-1 and score y. Pooled mean log
loss is the criterion; AUC, Brier, BSS and reliability are reported. Parsimony rule: a
candidate with more parameters is chosen over a simpler one only if the paired
storm-bootstrap interval (2,000 draws, whole storms) of the log-loss difference excludes 0.

## Final

The chosen candidate is refitted on 2020-2024 and scored ONCE on 2025, against A, B, C and
v8.2 (live CARQ inputs), with storm-bootstrap intervals and paired differences. "Better than
DTOPS" is claimed only if the paired interval on Brier or log loss excludes 0. An
independent adversary attacks the result before it is reported.

## Served

NHC basins: the chosen model whenever the cycle's SHIPS text is present; otherwise, and in
the JTWC basins (no public RI guidance), v8.2 on CARQ/JTWC inputs. Every served number
carries its source (`ri_source`: "noaa_aid_stack" or "v8.2"), and the site names NOAA's
SHIPS-RII and DTOPS as the inputs.

## Known uncertainty, stated before the result

- e-deck = SHIPS text identity is MEASURED for RIOD (three NHC discussion quotes) and DTOP
  (one), not on a full season: no season has both files public (stext holds only 2026,
  the archive holds e-decks only through 2025). The rounding convention is unverified
  (<= 1 percentage point). The 2026 e-decks, once archived, settle it.
- DTOPS depends on deterministic model runs (HWRF/HMON until 2023, HAFS after); a change in
  its members can move its skill between seasons.
