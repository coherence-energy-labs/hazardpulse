# Tornado model program -- pre-registered protocol (written 2026-10-02, before any v3 experiment)

Goal: the best achievable storm-object tornado model, any combination of methods, with
every claim decided by held-out numbers. This file is written BEFORE the experiments so the
protocol cannot be bent toward a result. Amendments are appended, dated, with the reason.

## Contract

- **Primary target `storm_60`**: a tornado report starts within R of the storm's centroid
  advected by its own motion to the report time, within 60 min; R = clip(sqrt(size/pi) + 5
  km, 8, 25 km). "This storm produces a tornado" -- what the site claims and what NOAA's
  ProbTor predicts. Secondary targets: `storm_30`, `storm_90`, and the neighbourhood labels
  `nbhd_30/60/90` (40 km of the current centroid; the v2 label).
- **Population**: every ProbSevere storm observation (30-min slots) of every UTC day with
  data, tornado days and quiet days alike. Training alone subsamples negatives, with weights.
- **Splits**: train 2020-10-15..2022; validation 2023 (every choice); development test 2024;
  FINAL test 2025, read once by the final pipeline and by nothing else.
- **Metrics**: ROC-AUC (Mann-Whitney, ties 1/2) and PR-AUC for ranking; Brier, BSS vs the
  population base rate, and 10-bin reliability for calibration. Every interval and paired
  comparison resamples whole UTC days (1,000 draws unless stated).

## Bars to beat (same storms, same labels, dev and final)

NOAA ProbTor (ps_tor/100); ProbSevere any-severe (ps/100); STP at 80 km and at 9 km; the
served v2 model; NWS tornado warnings (POD at the warnings' own false-alarm rate, on the
model's ROC curve). A model is "better" only if the paired day-clustered interval of the
difference excludes zero in BOTH test years.

## Declared search (all scored on validation 2023)

1. Feature blocks (LightGBM): P; P+E; P+E+H80; P+E+H80+C80; P+E+H9; P+E+H9+C9; all;
   all minus C9; all minus C9-pde (Gaussian control only); all minus C9-gauss (PDE only);
   all minus ProbTor/ProbHail/ProbWind (NOAA's model outputs).
2. Model families on the best block set: LightGBM, XGBoost, CatBoost, L2 logistic, the
   in-repo NumPy GBT.
3. LightGBM hyper-parameters: random search, 40 configurations, scored by validation AUC.
4. Calibration: none / Platt / Venn-Abers (in-repo `trust.venn_abers`), fitted on validation.
5. Negatives per positive in training: 10 / 30 / 100.
6. Training label: storm_60 vs nbhd_60, each evaluated on both.
7. Ensembles of the families and of seeds, weights chosen on validation.
8. Learning curve: 25 / 50 / 100% of training days.

## The coherence framework, tested fairly

C9 solves the variable-coefficient Helmholtz field D lap(tau) - Gamma tau + S = 0 at 9 km
(the storm scale; 80 km cannot resolve a mesocyclone) at three coherence lengths (9, 27,
81 km), with its gradient, alignment and tilting torsion. The CONTROL is the same
diagnostics from a Gaussian smoothing of S/Gamma at the same scales. Verdicts:
- coherence adds information iff `all` beats `all minus C9` (paired, both test years);
- the PDE operator matters iff `PDE only` beats `Gaussian only`.

## Stress tests (dev, then final)

Region (Plains / Midwest / Southeast / elsewhere), season, local night, storm size, lead
time, EF2+ tornadoes, analysis missing; a label-shuffle null (must give AUC 0.5); feature
availability at prediction time (every input <= observation time, asserted in code).

## Amendment 1 (2026-10-02, before any experiment): the storm label is chosen by attribution, not by a model

Reason: the first built day (2021-03-25, an Alabama outbreak) credited only 5 of its 10
UTC-day tornado reports to any storm under `storm_60`. The misses sit 20-23 km from the
CENTROID of large objects (448-560 km2, R 17-18 km): a supercell's tornado forms at the
rear-flank edge of the echo, not at its centre, so a centroid radius drops exactly the big
storms. ProbSevere ships each object's polygon. Candidates:
- L0 the current label (advected centroid, R = clip(sqrt(size/pi) + 5, 8, 25) km);
- L1 the report lies inside, or within 5 km of, the storm's POLYGON advected by the storm's
  own motion to the report time, report 0-60 min after the observation;
- L2 the same with 10 km;
- (reference only) nbhd_60, 40 km of the current centroid.
Measured on every 2021 day with ProbSevere data (development data; labels only, no model):
coverage = share of SPC tornado reports credited to at least one storm observation at lead
0-60 min; ambiguity = mean number of distinct storm ids credited per credited report.
Rule: the candidate with the highest coverage among those with ambiguity <= 1.3; candidates
within 2 coverage points of each other are a tie and the smaller buffer wins (L0 counts as
the smallest). The chosen label becomes `storm_30/60/90`; the others stay in the store as
secondary labels so the choice can be audited.

## Amendment 2 (2026-10-02, after a one-day smoke test of amendment 1, before the full-year measurement)

The one-day smoke test (2021-03-25, 10 reports) put the median report 4.7 km OUTSIDE its
storm's advected polygon: advecting a polygon with ProbSevere's motion for up to 60 min
moves it by the motion error times the lead (5 m/s for 60 min is 18 km). A label may use
the future, and the archive knows where the storm actually went: the same storm id appears
in later slots. Added candidates:
- T5: the report lies within 5 km of the polygon of the SAME storm id at that id's slot
  nearest the report time (within 15 min); if the id has no slot within 15 min (track
  ended), the polygon of its last slot before the report, advected by the residual time;
- T10: the same with 10 km.
Tie rule refined: candidates within 2 coverage points are ordered by lower ambiguity, then
by smaller buffer. The measurement and the rule are otherwise those of amendment 1.

## Amendment 3 (2026-10-02, before any experiment): human forecasters as an input

"Any combination" includes the NWS. Whether a storm sits inside an active tornado warning
polygon at the observation time, and for how long it has been warned, is known live (the
warning is public the moment it is issued) and causal (only warnings issued at or before
the observation count). Block W = (active_now, minutes_since_issue), from the IEM storm-based
warning archive (`hazardpulse.verification.nws_warnings`). Added to the declared search as
`best + W` against `best`, decided on validation like every block. NWS warnings also stay a
bar: their POD at their own false-alarm rate against every model's ROC curve. A model that
uses W is reported separately from one that does not, because it is no longer independent
of the warnings it is compared with.

## Amendment 4 (2026-10-02, before any ladder result): how "best" is read off the ladder

- The block set carried forward (`experiments/best_blocks.json`) is the arm with the highest
  validation ROC-AUC (out-of-fold Platt probabilities); the NOAA-free arms are ablations and
  are not eligible. If a smaller arm is within 0.001 AUC of the top one, the smaller arm is
  carried (fewer live inputs, same skill).
- Later choices (family, hyper-parameters, calibration, negatives per positive, label,
  ensembles) are read the same way, calibration by validation Brier instead of AUC.
- The coherence verdicts of the section above are paired day-bootstrap comparisons
  (`tornado_lab.py compare`) on validation, then on dev 2024 and final 2025; a block "adds
  information" only if the interval excludes zero in BOTH test years.

## Amendment 5 (2026-10-02, after the first ladder run, before any choice was taken from it)

The first ladder (12 arms, validation only) is VOID as evidence about features: every arm
early-stopped at 22-45 trees at learning rate 0.03. Root cause, measured: the training rows
carried weights that restore the population (negatives x ~30), so a positive's gradient is
~1 while its Hessian is ~0.0011; Newton leaf steps on positive-heavy leaves are huge, the
population-weighted log loss used for early stopping turns up within a few dozen trees, and
a ~40-tree model cannot use 159 inputs -- hence a flat ladder (0.946-0.952) on which the
28 ProbSevere columns "won". Independent check: the served v2 model (class-balanced
training) ranks the same validation rows at AUC 0.959, above every arm; ProbTor 0.867.
Fix: training weights are class-balanced (every positive weight n_neg/n_pos over the sampled
rows; negatives 1) and early stopping monitors AUC on the validation sample. Probabilities
still come only from the calibrator fitted afterwards, so the population is restored there,
as for v2. The void run's results are kept under results/lab/void_population_weights/.

## Amendment 6 (2026-10-02, after the final): the served model is the +W variant, by the owner's decision

Validation (amendment 4) chose the primary v3 (P+E+H80); the +W secondary (amendment 3) was
carried to the test years as declared. Both test years: +W over primary +0.0016 [+0.0006,
+0.0026] (2024) and +0.0014 [+0.0004, +0.0025] (2025) AUC with Brier better; +W over the
served v2 +0.0048 (2024) and +0.0031 [+0.0006, +0.0057] (2025) with Brier better. The owner
chose to serve the strongest model ("the best one possible"). This is a choice informed by
the test years and is recorded as such; the primary is kept as the automatic fallback when
the live NWS warnings feed is unavailable.

## Amendment 7 (2026-10-02, before fitting them): products beyond one probability

Declared before any of them is fitted or scored, with the served configuration (+W, the
chosen LightGBM parameters, Platt) and no new choices:
- P(tornado within 30 min) and P(within 90 min): labels `storm_30`, `storm_90`;
- P(EF2+ tornado within 60 min): positive iff `storm_60` and the matched report's EF >= 2;
- a per-storm probability interval: Venn-Abers on the leave-one-year-out scores;
- per-storm contributions: Saabas path attribution of the raw score (exact, additive).
Each probability is evaluated like the main model: trained 2020-10..2022, calibrated on
2023, scored on dev 2024; then refitted 2020-10..2024 with LOYO calibration and scored on
2025 once. The 60-min model's 2025 read does not choose anything about these.

## Amendment 8 (2026-10-02, after the independent adversary pass): HRRR publication latency

The adversary found that the store gave every row the latest analysis VALID at or before the
observation, but an analysis is published later. Measured on the eight most recent 00/06/12/18Z
HRRR-Zarr analyses: 100.6-100.7 min after valid time, every time. So ~56% of rows read an
analysis that did not exist yet -- a leak the "every input <= observation time" check could not
see, because it compared valid times. Fix at the root, one rule for training, evaluation and the
live scorer (`hazardpulse.data.hrrr_availability`): an analysis is usable iff valid + 100 min <= t;
the newest usable 3-hourly one is taken if at most 4 h 40 min old (the previous day's 21Z covers the
early hours). The live scorer is held to the same 3-hourly cadence.

`rebuild_h80_available.py` recomputes every row's H80 block through
`definitive_model.extract_block_h` under the new rule, after reproducing the stored H80 bit for bit
under the old rule on every row. v2 is rescored under the same rule. Then the affected steps are
rerun on corrected inputs, with the choices unchanged except where the declared rules say otherwise:
- the block decision P vs P+E vs P+E+H80 (amendment 4's rule; H80 is what moved);
- the validation run of the chosen configuration (its early-stopped rounds feed the final);
- dev 2024 (primary and +W), the finals (primary and +W, 2025 read a second time as a bug-fix
  rerun, recorded here) and the amendment-7 products.
The leaky results stay in results/lab/ for comparison; the corrected ones are in results/lab_avail/.

Reporting fixes adopted from the same pass:
- ProbTor is compared both as published and with its integer/zero ties broken by ProbSevere's own
  any-severe probability (published ProbTor is 0 for 93% of storms);
- v2 is compared on its own 40 km neighbourhood label as well as on the per-storm label;
- the NWS-warnings bar re-estimates the matched threshold inside every bootstrap replicate and
  resamples multi-day events (consecutive tornado days), not single UTC days.

## Amendment 9 (2026-10-03, before the corrected EF2+ final): the Platt fit is the maximum-likelihood fit

On the EF2+ product's leave-one-year-out scores, `definitive_model.fit_platt` (plain Newton) DIVERGED:
a = 3.1e16, b = -1.1e16. The superseded (pre-amendment-8) EF2+ final therefore served probabilities
of 0 or 1 and scored a 2025 Brier skill of -57.7 -- found by reading that superseded final. The
mechanism: the starting intercept logit(base rate) - mean(score) assumes the scores sit near the
right scale; EF2+'s negatives sit near -10 with a heavy right tail, so the first step overshoots
(log-loss 0.063 -> 1.68) and the iterates oscillate away.

The method is unchanged (Platt by maximum likelihood); the implementation now finds the maximum:
Newton with a backtracking line search on the convex log-loss, keeping the full step whenever it
does not increase the loss, so every fit that converged before is reproduced bit for bit (checked
on all seven saved LOYO score sets, leaky and corrected); a fit that does not converge raises
instead of returning a calibration. EF2+ under the fix: a = 0.687, b = -6.535 on the superseded
scores, mean forecast = base rate. The corrected EF2+ final runs with it; no choice is changed.

## Outcome of the corrected programme (amendments 8-9; 2025 read a second time, once)

Choices on validation 2023, corrected inputs (`scripts/audit_20261001/experiments/avail/`):
- Blocks (amendment 4's rule): P 0.96105, P+E 0.96159, P+E+H80 0.96173 -- all within 0.001 of
  the top, so the smallest, **P (ProbSevere's 28 attributes)**, is carried. Timed as it is
  published, the 80 km HRRR block adds about +0.0001; the +0.0038 measured before amendment 8 was
  almost entirely the latency leak.
- Configuration: LightGBM, 40 configurations searched (median 0.96131), carried s_lgbm_28 (0.96215;
  31 leaves, depth 6, L2 20); Platt; 30 negatives per positive; the per-storm label.
- Served (amendment 6's decision stands; its figures, from the P+E+H80 leaky era, are superseded):
  P + the NWS warning state, with P alone as the fallback when the warnings feed is down.
- Development year 2024: P 0.9683 [0.9623, 0.9732] BSS +0.086; +W 0.9691 [0.9631, 0.9741] BSS
  +0.114; at the warnings' false-alarm rate P alone -1.0 points [-4.3, +1.5], +W +4.1 [+1.9, +5.4].

Final 2025, every ProbSevere storm observation (from `results/lab_avail/` by
`scripts/audit_20261001/report_finals.py`; intervals: day bootstrap, 2,000 replicates):

| Final run | AUC [95% CI] | PR-AUC | BSS | n / tornadic |
|---|---|---|---|---|
| v3_plus_W (storm_60), served | 0.9702 [0.9636, 0.9758] | 0.1750 | +0.1067 | 1,469,977 / 1,579 |
| v3_primary (storm_60), fallback | 0.9693 [0.9628, 0.9748] | 0.1664 | +0.0966 | 1,469,977 / 1,579 |
| v3_plus_W_30 (storm_30) | 0.9743 [0.9675, 0.9795] | 0.1394 | +0.0840 | 1,469,977 / 978 |
| v3_plus_W_90 (storm_90) | 0.9681 [0.9614, 0.9736] | 0.1894 | +0.1114 | 1,469,977 / 1,993 |
| v3_plus_W_ef2 (storm_60_ef2) | 0.9942 [0.9891, 0.9971] | 0.1512 | +0.0608 | 1,469,977 / 284 |
| NOAA ProbTor, as issued | 0.8790 [0.8504, 0.9022] | 0.1257 | -0.1733 | same storms |
| NOAA ProbTor, ties broken (Platt) | 0.9391 [0.9242, 0.9508] | 0.1287 | +0.0728 | same storms |
| ProbSevere (Platt) | 0.8709 [0.8312, 0.8997] | 0.0393 | +0.0180 | same storms |
| STP 80 km (Platt) | 0.8446 [0.8172, 0.8665] | 0.0106 | -0.0119 | same storms |
| v2, as served | 0.9636 [0.9561, 0.9691] | 0.1247 | +0.0581 | 1,466,605 / 1,576 |

| Paired by day (2025) | label | dAUC [95% CI] | dBrier [95% CI] |
|---|---|---|---|
| v3_plus_W - ProbTor as issued | storm_60 | +0.0913 [+0.0718, +0.1152] | -3.00e-04 [-0.000407, -0.000212] |
| v3_plus_W - ProbTor ties broken | storm_60 | +0.0311 [+0.0222, +0.0425] | -3.64e-05 [-0.000055, -0.000021] |
| v3_plus_W - v2 | storm_60 | +0.0067 [+0.0034, +0.0104] | -5.23e-05 [-0.000083, -0.000029] |
| v3_plus_W - v2 (v2's own 40 km event) | nbhd_60 | +0.0011 [-0.0045, +0.0076] | -2.92e-05 [-0.000049, -0.000012] |
| v3_plus_W - v3_primary | storm_60 | +0.0009 [+0.0002, +0.0017] | -1.08e-05 [-0.000024, +0.000001] |
| v3_primary - ProbTor ties broken | storm_60 | +0.0302 [+0.0214, +0.0414] | -2.56e-05 [-0.000036, -0.000016] |
| v3_plus_W_30 - ProbTor as issued | storm_30 | +0.0804 [+0.0617, +0.1054] | -3.80e-04 [-0.000515, -0.000266] |
| v3_plus_W_90 - ProbTor as issued | storm_90 | +0.0986 [+0.0786, +0.1233] | -2.71e-04 [-0.000373, -0.000191] |
| v3_plus_W_ef2 - ProbTor as issued | storm_60_ef2 | +0.0176 [+0.0063, +0.0385] | -4.98e-04 [-0.000665, -0.000354] |

At the NWS tornado warnings' own false-alarm rate (threshold re-matched in every replicate,
multi-day events resampled together): +W catches 30.5% of tornadic storm observations vs the
warnings' 22.7%, +7.7 points [+3.8, +10.1]; P alone 27.5%, +4.7 [-0.2, +8.5]. With 2024's -1.0
[-4.3, +1.5], the model WITHOUT the warning input is not distinguishable from the warnings in
either year; what is shown is that it adds detection on top of them.

Reading, stated plainly: v3 beats NOAA's ProbTor on the same storms in ranking (even after its
ties are broken) and in Brier; it ties v2 on v2's own event and beats it on the per-storm event
it was built for; the HRRR environment and the coherence field add nothing measurable once timed
honestly. All of the above is one test year read once; live verification of the served model
starts with its first matured forecasts.

## Amendment 10 (2026-10-06, before any candidate number): NOAA's 2025 ProbSevere format change (ledger T2)

**The defect.** NOAA changed the ProbSevere file between 2025-08-05 14:00Z (the last scan carrying `PS` and
`VIL_DENSITY`) and 20:48Z (the first scan with the new attributes); no scan carries both (MEASURED on the
store's cache: 2025-08-05 has 36 of 48 slots, 2,255 storms with a nonzero `ps` before, 1,430 new-format storms
after, 0 with both). The parser reads the two absent attributes as 0.0, so since then every served v3
forecast has received `p_ps = 0` and `p_vil_density = 0`. The served payloads were fitted on 2020-10..2024,
where `p_vil_density` was 0 on 0.6% of rows; those two inputs carry 196 and 475 of the +W model's 8,826 splits.

**Candidates** (the +W model with the no-warnings fallback, as served; every other input identical):
- **(a) SERVED**, the control: `tornado_v3_w.json` / `tornado_v3.json`, the two inputs 0.0, as served.
- **(b) NAN**: the same payloads, the two inputs missing (NaN): each node's own missing-value rule.
- **(d) DROP2**: the chosen configuration (`chosen_on_validation.json`, primary and +W) without `p_ps` and
  `p_vil_density`. Its round count is chosen as the original's was, by early stopping on validation 2023
  (`tornado_lab.py run`); then the unchanged final pipeline (leave-one-year-out Platt, refit on
  2020-10..2024). The only code change: `_final_core` and `_loyo_scores` honour the experiment's `drop` list
  (as `run` already does).
- **(c) REBUILT is not run.** Witness: no scan carries both forms, so no mapping can be checked before scoring;
  and `VIL_DENSITY` is VIL over the 18 dBZ echo top, while the new file has only the 50 dBZ one.

**The decision set: 2026-01-01 .. 2026-09-30**, every ProbSevere storm observation at the program's 30-minute
slots, the new format throughout. No choice has ever been made on it.
- Labels: the program's `storm_60` labeller (`storm_features.labels`, the storm's own track) on SPC's
  preliminary filtered tornado reports (`YYMMDD_rpts_filtered_torn.csv`, the prospective verifier's source). The
  final 2026 database does not exist yet. Preliminary reports carry location and time errors that the final
  database corrects; that affects every candidate alike (the comparison is paired), but these absolute numbers
  are not comparable with the 2025 final's.
- Warning state: the IEM storm-based warning archive (`verification.nws_warnings`), as for block W.
- **Control, before any comparison:** on the live records of 2026-10-03 .. 2026-10-05, the pipeline's (a) inputs
  and probabilities must reproduce the records' stored `inputs` and `probability_60min` bit for bit for every
  storm observation matched by id and scan time. Any mismatch stops the run.

**Rule** (paired day bootstrap, 2,000 draws, as section Metrics): a candidate X replaces (a) iff the 95% interval
of AUC(X) - AUC(a) lies entirely above 0 AND the Brier point difference is <= 0. If both (b) and (d) qualify,
the higher AUC point is served. If neither does, (a) stays and the cost of the format change is recorded with
its interval. The served candidate's predecessor stays in the live record beside it, so the switch is also
verified on forecasts made after it.

**Clarifications (2026-10-06, still before any candidate number):**
- The rule is applied to the +W variants: they are what is served whenever the NWS feed answers. The fallback
  variants are reported beside them.
- The control runs where the live scorer runs (GitHub's ubuntu runner). Run on Windows first, it found all
  1,430 live input vectors identical, but 22 of the 1,430 probabilities 1-2 units in the last place apart.
  The model file and the scoring code are unchanged since before those forecasts; the records keep no raw
  score, so where the difference arises is a HYPOTHESIS: NumPy's `exp`, built per platform. The control on
  the live platform tests it; if it holds, recomputing a record exactly needs the live platform -- a
  reproducibility gap of its own, recorded in the ledger.
- `scripts/tornado_program/t2_decide.py` (`control`, `build`, `decide`) implements this amendment.
- **(b) is (a), by construction** (PROVEN on the payloads): every split on the two inputs has missing type
  None -- 196 and 475 in the +W payload, 171 and 486 in the fallback -- and under LightGBM's rule
  (`lgbm_payload._tree_predict`) a NaN at such a split is read as 0.0. The trees learnt no missing-value
  branch for inputs that were never missing in training, so "missing" and "zero" are the same input. (b) stays
  scored, and `decide` checks that its predictions equal (a)'s on every row; it cannot qualify. The decision
  is (a) against (d). (The ledger's T2 entry offered (b) as "the trees' missing-value branches"; for these two
  inputs there are none.)
- **The control, narrowed after its first run (2026-10-06, still before any decision number).** The platform
  hypothesis above was FALSIFIED. On the runner too, every one of the 1,430 input vectors was identical, but
  40 probabilities differed -- different storms from Windows' 22 -- all by at most 2 units in the last
  place. Disabling NumPy's AVX2/AVX-512 paths here changed nothing. The function is the same; the
  floating-point `exp` beneath it is not one function across builds and machines, and the records keep no raw
  score that would pin it down. The control now demands the inputs EXACTLY and each probability within 4 ulp
  (`MAX_PROB_ULP`; measured 0 ulp on 1,408, 1 on 16, 2 on 6); with 0 ulp it fails, as it should. The build and
  decide jobs never ran (they wait on the control), so no candidate number existed when this was changed.
  Root fix, recorded in the ledger: a record should carry the raw score, which is exact IEEE arithmetic.
- **Disclosure:** a one-day smoke test of the scoring code, on 2026-04-27 (a day of the decision set),
  scored (a) and (b): identical, AUC 0.985 each on 3,810 observations (24 tornadic). No number for (d) has been
  computed on 2026 data.

**Descriptive, decides nothing: the defect on identical storms.** On 2025-01-01 .. 2025-08-04 (old format;
out of sample for the served payloads; part of the 2025 final, already read), (a) and (b) are scored with
the two inputs as recorded, as 0.0 and as NaN. The paired differences are the cost of the zero-fill with no
change of season.

**Next, not part of this decision:** a refit with the new attributes (`maxfed`, `dcape`, `vil`, `echotop_50`,
`lcl`, ...) needs new-format training data, of which 2026 is the decision set here. It is a later amendment.

### Outcome of amendment 10 (2026-10-06): (a) stays served

Run `research-tornado-t2.yml` 37400897999 on `ce2b3c75f`; result `results/tornado_program/t2_2026.json`.
- **Data:** 1,262,285 storm observations, 1,500 tornadic, 273 days. No day was dropped: every SPC preliminary
  file was read, and every day had ProbSevere.
- **Control:** passed on the runner. All 1,430 live inputs were exact; probabilities were 0 ulp apart on 1,390,
  1 ulp on 29 and 2 ulp on 11.
- **(b) = (a)** on every row, as proven.

| +W variant (served path) | AUC | Brier | BSS | mean forecast (base rate 1.19e-3) |
|---|---|---|---|---|
| (a) served | 0.9699 | 1.071e-3 | +0.098 | 9.68e-4 (0.81x) |
| (d) without `p_ps`, `p_vil_density` | 0.9696 | 1.060e-3 | +0.107 | 1.56e-3 (1.31x) |

Paired, (d) - (a), day bootstrap 2,000:

| variant | dAUC [95% CI] | dBrier [95% CI] |
|---|---|---|
| +W | -0.0003 [-0.0012, +0.0007] | -1.13e-5 [-2.13e-5, -1.9e-6] |
| fallback | +0.0002 [-0.0005, +0.0010] | -2.28e-5 [-3.66e-5, -9.6e-6] |

**The rule is not met** (the AUC interval contains 0), so (a) stays served. What this establishes:
- **The format change costs no ranking skill.** The served model's 2026 AUC (0.9699, new format all year)
  equals its 2025 final (0.9702). The within-2025 drop the audit saw (0.970 before the change, 0.947 after) was
  the August-December season, as suspected.
- **It may cost calibration.** (d)'s Brier is better, with an interval that excludes 0, and the served model's
  mean forecast is 19% under the base rate. But (d) is 31% over it, and no rule here was written for
  calibration. A recalibration of the served model for the new format is a new question with its own
  registration: 2026-01..09 has now been read, so it is judged on forecasts made after it is registered.

## Amendment 11 (2026-10-10, before any candidate number): T2b, the served models' calibration for the new format

**The question.** Amendment 10 kept the served models: the format change costs no ranking skill. It left one
question open. On 2026-01..09 the served +W model's mean forecast was 0.81x the base rate (9.68e-4 against
1.19e-3), and no rule there was written for calibration. This amendment changes no input and no tree. It asks
only whether each served payload's Platt map, refitted on the new format, gives better probabilities from the SAME
raw score. The fit window contains 2026-01..09, which has been read, so the answer is decided only on forecasts
made after this registration.

**Candidates**, one per served payload that the live record can label, each decided on its own:
- `p60_w`: `tornado_v3_w.json`, the published 60-min probability whenever the NWS feed answers. **Primary.**
- `p60`: `tornado_v3.json`, the fallback, served when the feed does not answer. MEASURED: 213 of the 5,660 v3 storm
  forecasts of 2026-10-03..10 (four feed outages).
- `p30`, `p90`: `tornado_v3_w_30.json`, `tornado_v3_w_90.json`.

Each candidate is that payload's raw margin (`lgbm_payload.predict_raw`, the sum of its trees, unchanged) mapped by
a new Platt fit. The fit uses `definitive_model.fit_platt`: maximum likelihood, Newton with a backtracking line
search (amendment 9). That is the code that fitted every served calibration (`tornado_lab._final_core`), and like
them the fit is unweighted on the whole population.

**Fit data.** Every new-format storm observation at the program's 30-minute slots (`probsevere.slot_start_keys`),
from 2025-08-05 20:48Z (the first new-format scan, amendment 10) through 2026-09-30.
- Rows come from amendment 10's row builder (`t2_decide.day_rows`'s steps), keeping all three horizons:
  - block P from `storm_features.block_p`, with the two absent attributes 0.0, as served;
  - block W from the IEM warning archive (`verification.nws_warnings`), as for every +W fit.
- On 2025-08-05 only rows at or after 20:48Z are kept.
- **Labels: the training labels `storm_30`, `storm_60`, `storm_90`** (`storm_features.labels` on the storm's own
  track).
  - The reports are SPC's preliminary filtered tornado reports throughout the window, timed as the prospective
    verifier times them. Each map is therefore fitted against the report source its live record is judged on.
  - Training used the final database, but the final 2026 database does not exist yet.
  - A day whose label window reaches a convective day SPC cannot serve is dropped and counted, as in amendment 10.
- **Checks; each one stops the fit:**
  - every day of the window was built;
  - `p_ps` and `p_vil_density` are 0.0 on every kept row;
  - at least one 2025-08-05 row before the cut has `p_ps` != 0, so the cut does separate the two formats;
  - the fit converges (`PlattNotConverged` otherwise);
  - the fitted `a` > 0. Only then is the map increasing, so it cannot change the ranking.
- **Output:**
  - `results/models/tornado_v3_recal_t2b.json`, the artifact: per candidate a, b, the payload's SHA-256, n, events
    and days; plus the window, the data's SHA-256, and the code's SHA-256s and commit.
  - `results/tornado_program/t2b_fit.json`: the fit's facts.
  - In-sample numbers are descriptive and decide nothing.
- The artifact is fixed at its commit. A record that carries any other artifact digest does not count.

**Controls. Each one runs before the fit is used, and any failure stops it.**
1. **The served (a, b) reproduce.** `fit_platt` is run on each payload's own saved leave-one-year-out scores (2021-2024):
   `_lab_avail/oof/v3_plus_W.npz`, `v3_primary.npz`, `v3_plus_W_30.npz`, `v3_plus_W_90.npz`. It must return the
   payload's a and b within 1e-9 (`tornado_lab.oof_only`'s standard). These files exist only on the machine that
   built the store (88 MB each, not in the repository). This control runs there, and its result is committed with
   the files' SHA-256s.
2. **The fit's margin path reproduces the served probabilities.** On every live v3 record that stored its inputs,
   `sigmoid(a * predict_raw + b)` with the served (a, b) must give back:
   - the record's `probability_60min`, from the payload that served it;
   - each `probability_30min` / `_90min` that the coherence rule did not clip.

   The tolerance is 4 ulp (amendment 10's `MAX_PROB_ULP`), checked on the runner.
3. **Amendment 10's control** (`t2_decide.py control`): the P inputs parsed from S3 must equal the live records'
   stored inputs exactly.

**Live shadow.**
- **When it starts:** with the first live tornado run whose checkout carries the artifact.
- **What each v3 storm forecast carries** (`t2b_shadow`), per candidate that applies to that storm:
  - which candidates apply: `p60_w` and the products when the +W model served; `p60` on every storm, since the
    fallback reads only block P, which every record stores;
  - the payload's raw margin;
  - the served payload's own probability from that margin (before the products' coherence clip);
  - the recalibrated probability.
- **`t2b_shadow.probability_60min`** is the recalibrated probability of the model that served the storm.
- **The record also states** what produced the shadow: the artifact's SHA-256, and each candidate's (a, b) and
  payload SHA-256.
- **It is recorded, never published.** No published field changes: `tornado_probability`, `risk_band`, every `v3`
  field, the band and the receipt.
- **The record audit** recomputes each margin exactly from the stored inputs, and each probability within 1e-12.
- **Without the artifact**, or with an artifact fitted for another payload, there is no shadow and the forecast is
  unchanged.

**The rule** (prospective; per candidate):
- **Record:** every v3 forecast record from the first one carrying this artifact's shadow up to the look date. It
  is labelled as the prospective verifier labels v3 (`score_tornado_prospective`): `storm_features.labels` on the
  storm's archived track, SPC's preliminary reports, counted from the storm's valid time, at the candidate's own
  horizon. One forecast is kept per storm per valid hour (the first issued). A record that cannot be labelled is
  left out and counted.
- **Compared:** the served probability against the recalibrated one, both as recorded and both from the same
  recorded margin.
- **Replace iff:** the recalibration's Brier score AND its log loss are both below the served calibration's,
  AND the 95% interval of Brier(recalibrated) - Brier(served) lies wholly below 0.
  - The interval is a percentile bootstrap, 2,000 draws, seed 42.
  - It resamples whole **convective days** (12Z-12Z; the SPC file each label comes from). The program's other
    comparisons resample UTC days. But most tornadoes fall between 21Z and 03Z, so a UTC day cuts an evening
    outbreak at 00Z into two blocks that are not independent, and that narrows the interval. For a decision, the
    block is the unit of dependence.
- **No claim** with fewer than 30 tornadic storm forecasts, or fewer than 10 convective days holding one. A
  percentile interval over a handful of event clusters is not a 95% interval. The served calibration then stays.
- **Ranking is unchanged by construction.** With a > 0 the map is strictly increasing in the margin, so AUC is the
  served model's own, up to float64 ties. AUC is reported, never decided on.

**Looks.** Each look is applied once and then frozen into the output. It counts the records issued before the
look date. It is computed by the first scoring run at least 3 days after that date, by which time every window has
matured and SPC's files are posted. If any record is still unscorable for a transient reason, the run waits, for at
most 14 days.
- **2027-07-01**, after the 2027 spring peak. In 2025, 1,400 of the 1,579 tornadic storm observations fell before
  2025-08-06 (base rate 0.151% before, 0.033% after; ledger T2).
- **2028-07-01**, on the whole record to then, for each candidate the first look did not replace.
- **No earlier decision.** A look at 2027-01-01 would judge only the cool season, which is a different base rate
  and a different regime from the spring that carries most of the risk, on a few dozen events. A frozen
  descriptive read is made at 2027-01-01, and it decides nothing.
- Two decision looks, each at 2.5% one-sided, bound at 5% per candidate the chance of adopting a calibration that
  is no better.
- Before a look the output is a running record, and no claim is read from it.

**After a decision.**
- A met rule replaces that payload's `calibration`. That makes a new payload file and so a new model version.
  Everything else in the payload stays, including the Venn-Abers band.
- The replaced calibration then runs as the shadow for a year, so the switch is itself verified on forecasts made
  after it.
- If the rule is still not met at 2028-07-01, the candidate is closed.

**Not recalibrated now: EF2+ (`tornado_v3_w_ef2.json`).** Its label needs each report's rating, and SPC's
preliminary reports carry none. F_Scale is `UNK` on every row of 250519 (40 reports), 260427 (25) and 260316 (4);
the rating appears only in the free-text comment. So EF2+ can be neither fitted on new-format data nor judged on
the live record until SPC's final database covers the window. That is a later amendment.

**Implementation:**
- `scripts/tornado_program/t2b_recalibrate.py`: control, build, fit. It runs on GitHub's runner via
  `.github/workflows/research-tornado-t2b.yml`.
- `hazardpulse.tornado.t2b_shadow`: the shadow; also used by the record audit.
- `scripts/score_tornado_t2b_prospective.py`: the rule, called by the prospective tornado verifier.

## Final pipeline (fixed now)

The configuration chosen on validation is refitted on 2020-10..2024 with the validation-
chosen rounds, calibrated by a calibrator fitted on out-of-fold (leave-one-year-out) scores,
and scored ONCE on 2025. That model is the one served. An independent adversary attacks
the final claim before it is reported.

## Data record (facts measured while building the store, not choices)

- NOAA's `noaa-mrms-pds` bucket holds NO ProbSevere objects for 2021-05-15 .. 2021-05-24
  (10 days; S3 listing of `ProbSevere/2021051*` and `2021052*` checked 2026-10-02) nor for
  2020-11-04. Those days have no storm observations and are absent from every split; their
  tornadoes are unobservable to a storm-object model.
- A day whose HRRR analyses fail to arrive is built with what exists and recorded per row
  (`analysis`); its store file carries an input fingerprint, so a later fetch that fills
  the hole rebuilds it. 2022-01-06 got 0 of 8 analyses in the first pass (retried).
- HRRR fetch failures came in two kinds, told apart since the pass records the child's
  error: real archive holes ("nothing found at path"; e.g. 2024-11-15/16/18 18-21Z,
  2025-05-29 18Z, 2025-06-20 06-09Z, 2025-10-20 06-18Z) and read timeouts when 8 parallel
  dates saturated the link (27% of dates in one stretch; 4 workers fixed it).
- The train and validation memmaps are FROZEN as every validation choice saw them
  (assembled 2026-10-02 before the gap retries). Retry-filled 2020-2023 days are not
  re-assembled into them; dev 2024 and final 2025 are assembled after the retries. The
  final refit reads the same frozen train/val memmaps plus dev.
