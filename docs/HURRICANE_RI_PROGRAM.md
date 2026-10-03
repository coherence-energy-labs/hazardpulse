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

## Amendment 1 (2026-10-02) -- written before the development selection was run and before any 2025 number

Candidates, criterion, splits and the parsimony rule are unchanged. What follows is what the
text above leaves undefined, fixed before any result could steer it.

1. **A raw aid that is missing at a cycle.** A (DTOPS) and C (RIOC) are not defined at cycles
   whose e-deck lacks that aid (DTOP is absent at about 15% of the 2020-2024 cycles that hold a
   RIOD record, per the benchmark's tech counts), so as written they cannot be scored on the
   case set and no paired comparison exists. Rule: A and C fall back to SHIPS-RII (RIOD) at such
   cycles -- RIOD is present at every case by the case definition, and a served raw-aid product
   would have to do the same. The number of fallback cycles is reported.
2. **Pattern fits.** The fit for availability pattern P uses every training case at which all
   of P's inputs are present (a cycle with more aids also trains its sub-patterns), P's inputs
   only, plus an intercept. A fit exists only with >= 20 events and >= 20 non-events and a
   converged Newton fit (`ri_model.newton_logistic`). A cycle whose own pattern has no fit uses
   the largest sub-pattern that has one (ties: keep the inputs earliest in the order RIOD, RIOL,
   RIOB, DTOP, V82). For E, v8.2 is an input like an aid, so "no v8.2 case at the cycle" is part
   of the pattern; for F the basin indicator is in every pattern and its DTOPS interaction only
   in patterns containing DTOP. The served artifact stores a fit for every subset containing
   RIOD that meets the same rule, not only the subsets observed.
3. **Parsimony, operationally.** Parameters per full pattern: A = B = C = 0 < D = 5 < E = 6 <
   F = 7. The zero-parameter candidate with the lowest pooled log loss is the first incumbent
   (no rule ranks equal-parameter candidates other than the criterion; an exact tie goes to the
   earlier letter). Then D, E, F in that order: a challenger replaces the incumbent only if the
   upper end of the 95% paired storm-bootstrap interval of LL(challenger) - LL(incumbent) is
   below 0. Otherwise -- including any tie -- the simpler incumbent stays.
4. **Log loss of a whole-percent forecast** (A, B, C) is taken on the probability clipped to
   [0.005, 0.995], the protocol's clip; a 0% or 100% aid that misses would otherwise score an
   infinite loss. AUC and Brier use the raw value. Stack outputs are sigmoids and are not clipped.
5. **Storm unit** for every bootstrap: the IBTrACS SID (a storm whose ATCF id changes across
   basins is one storm). 2,000 draws, seed 20261002, percentile 95% (`ri_evaluation.storm_bootstrap`).
6. **Truth mapping.** A case (atcf_id, YYYYMMDDHH) maps to the unique IBTrACS SID whose fix at t
   carries USA_ATCF_ID = atcf_id; V(t+24 h) is read on that SID; USA_WIND must be present and
   > 0 at both fixes. A key that maps to two SIDs is dropped and counted. No status or land
   filter (the contract has none).
7. **BSS reference** = the event rate of the case's own fit seasons (2020..y-1 in forward
   chaining; 2020-2024 in the final), an out-of-sample climatology.
8. **2025 isolation.** The select phase reads no 2025 deck and no 2025 IBTrACS wind; the final
   phase downloads the 2025 e- and a-decks. The IBTrACS cache (downloaded 2026-04-13) was
   checked for 2025 by counting rows whose USA_WIND is non-empty, never reading a value: AL 13
   storms / 731 rows, EP+CP 20 storms / 975 rows, every row with USA_WIND, USA_AGENCY
   hurdat_atl / hurdat_epa -- the same 33 storms the NHC 2025 archive holds. No fresh IBTrACS is
   fetched, so 2020-2025 truth comes from one file, the one v8.2 was built from.
9. **v8.2 on live CARQ inputs** = `fetch_and_score.build_live_case` with the served analysis
   priority (CARQ first since b28c2b3ef; OFCL only where CARQ is absent, counted) on the archived
   a-deck truncated at t (`benchmark_hurricane_vs_ships.operational_cases`). In the final, the
   chosen candidate is compared with A, B, C on every 2025 case, and with v8.2 on the 2025 cases
   where that builder yields a case at the cycle (the n is reported, and the whole table is
   repeated on that subset).

## Amendment 2 (2026-10-02) -- after a dry run of the case MAPPING on 2020-2024, before any candidate was scored

The mapping of Amendment 1, item 6 was dry-run on the development e-decks (counts only; no
candidate metric was computed). Of the cycles with a RIOD record, 1,143 precede the first
IBTrACS fix of their storm (pre-genesis invest cycles: no best track, so no truth -- correctly
dropped) and 76 follow its last fix; but 69 belong to ATCF ids that never occur in IBTrACS:
EP042022 and EP182022, the East Pacific stages of Bonnie and Julia, which NHC's operational
decks renumbered on crossing from the Atlantic while IBTrACS/HURDAT2 keep AL022022 and AL132022
for the whole track. Dropping them would silently lose the Pacific half of every basin-crosser.
Rule added: when no fix at t carries the case's ATCF id, the case maps to the unique SID with a
fix at t within 100 km of the e-deck's own RIOD position (none or several -> dropped, counted as
"matched_by_position"). Measured on 2020-2024: it recovers 52 cycles, all of EP042022 -> AL022022
(49) and EP182022 -> AL132022 (3), and the recovered set is identical at 50, 150 and 300 km, so
the radius is not doing the choosing.

## Erratum and record (2026-10-02, written AFTER the final -- changes nothing above)

- Erratum, candidate E: "no development season is in-sample for it" is false. Model C's
  calibration window 2019-2021 contains the development seasons 2020-2021, so E's training rows
  for those seasons carry an in-sample v8.2 calibration. E was not chosen; nothing else is touched.
- Records: results/calibration/hurricane_ri_stack_selection.json (choice A), ..._final.json
  (2025, read once), ..._adversary.json (the required independent attack: CONFIRMED, scoped to the
  601 cycles with an archived RIOD record -- Melissa's RI, 9 of the season's 53 events, is missing
  from NOAA's e-deck archive).
