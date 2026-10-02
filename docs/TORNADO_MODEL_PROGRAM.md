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

## Final pipeline (fixed now)

The configuration chosen on validation is refitted on 2020-10..2024 with the validation-
chosen rounds, calibrated by a calibrator fitted on out-of-fold (leave-one-year-out) scores,
and scored ONCE on 2025. That model is the one served. An independent adversary attacks
the final claim before it is reported.
