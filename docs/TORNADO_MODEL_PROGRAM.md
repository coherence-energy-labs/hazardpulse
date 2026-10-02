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
