# Earthquake operational forecast program (pre-registered)

Status: **protocol frozen 2026-10-02 before any candidate was fitted.** Sections 1-8 are the
registration and are not edited after the first fit; results are appended in section 9
onward with the step that produced them. Evidence classes: **MEASURED** (run and recorded
here), **DERIVED** (from code or documents), **HYPOTHESIS**.

## 0. Why this program exists

The served earthquake probability is labelled on the site as "Estimated M6.0+ probability
in the next 30 days" for each 2-degree cell (`scripts/fetch_and_score_earthquake.py`,
page text and `variableMeasured`). DERIVED from the code and the replay artifacts:

- the model registry advertises `earthquake_gbt_v1.json` with **AUC 0.907**, a
  CASE-CONTROL NOWCAST (positives scored at the mainshock setting vs same-place
  controls); on a gridded forward test that model scored **0.509**
  (`docs/earthquake_nowcast_validation.md`, `scripts/backtest_operational_grid.py`) --
  no operational skill (audit finding E5);
- the probability actually published is NOT that GBT: tier precedence in
  `score_grid_cells` puts the deep GRU year-ahead nowcast first
  (`eq_deep_nowcast_m5.0_2025_K192.serve.npz`, trained on "M5+ within 500 km / 365 d"
  case-control samples), every replay artifact of 2026-10-01 carries
  `model_id = deep_gru_k192`, and its raw score is mapped by the trust-layer calibrator
  fitted in-sample on matured live forecasts;
- only "active" cells (>= 5 M2.5+ events in the cell in the last 30 days, 82 of 11,700 on
  2026-10-01 22:00Z) receive a probability; every other cell is published as **0.0**, so
  any M6+ outside an active cell is a forecast of "impossible" that happened;
- every earthquake model was trained on the USGS catalog truncated at 20,000 events per
  year request (E1: 23% of M6+ missing), fixed 2026-10-01 (monthly/bisecting paging).

The question: what is the best probability this site can HONESTLY serve for the quantity
it already names, measured by a comparison fixed in advance?

## 1. The contract (what is forecast)

For every cell `c` of the site's grid -- 65 rows x 180 columns, latitude -60..70,
longitude -180..180, 2 x 2 degrees (`coherence_engine` constants; 11,700 cells) -- and
every issue time `t`:

> **P( at least one ComCat event with magnitude >= 6.0 has its epicentre in cell `c`
> during [t, t + 30 days) ),** using only catalog events with origin time strictly
> before `t`.

- **All M6+, not declustered mainshocks.** Chosen because it is exactly what the site
  publishes and what its own verifier scores: `scripts/score_earthquake_prospective.py`
  fetches every ComCat M >= 6.0 event in `[issued_at, issued_at + 30 d)` and assigns it
  to a cell with `latlon_to_grid_cell`. A forecast of declustered mainshocks would be
  verified against a different event set than the one it is published for, and
  declustering is itself model-dependent (window choice). Large aftershocks are real
  M6+ shaking and are part of what a reader of "M6.0+ in the next 30 days" expects.
  A declustered-mainshock target is reported on the FINAL split as a secondary
  diagnostic only (how much of any skill is clustering).
- **Cell assignment** is `coherence_engine.latlon_to_grid_cell` (int truncation toward
  zero, then clipping): events north of 70N or south of 60S fall into the edge rows,
  exactly as the verifier counts them. The vectorised `operational_forecast.cell_index`
  is tested equal to it.
- **Any ComCat event type** at M >= 6 counts (the verifier does not filter types).
- A forecast is a probability for **every** cell (no default zeros).

## 2. Issue times and splits

Issue times `t_k = 2005-01-03T00:00Z + 7k days` (Mondays, every 7 days). An issue time
belongs to a split only if its whole 30-day window lies inside the split's years, so no
outcome is shared across splits:

| split | issue times | role |
|---|---|---|
| FIT | 2005-01-03 <= t, t + 30 d <= 2018-01-01 | every parameter of every candidate |
| CHOOSE | 2018-01-01 <= t, t + 30 d <= 2021-01-01 | the selection rule (section 7) |
| DEV | 2021-01-01 <= t, t + 30 d <= 2023-01-01 | confirmation, no re-selection |
| FINAL | 2023-01-01 <= t, t + 30 d <= 2026-01-01 | read ONCE, after the choice is recorded |

The FIT split starts in 2005 because candidate C reads M2.5+ counts over 5-year windows
and the audited M2.5+ files start in 2000; every candidate is fitted on the same issue
times. Long-term inputs reach back to 1973 (section 3).

## 3. Data

- **Program catalog** (`scripts/earthquake_program/catalog.py`): every ComCat event with
  M >= 4.5, 1973-01-01 to 2026-10-01, pulled one year per request through
  `hazardpulse.data.usgs_fdsn.fetch_window` (bisects any response at the 20,000-row
  limit), one manifest per year. MEASURED 2026-10-02: 54 requests, largest response
  9,584 rows (no year near the limit, no bisection needed), 293,699 events, 89,861 with
  M >= 5, 7,498 with M >= 6, **no calendar month 1973-2026/09 without an M >= 5 event**
  (`results/earthquake_program/catalog_audit.json`).
- **M2.5+ files** (2000-2025, the repository's re-pulled `usgs_catalog_<year>.csv`):
  every year passes `usgs_fdsn.audit_year_catalog`; MEASURED: their M >= 5 counts equal
  the fresh M4.5 pull's in **every** year 2000-2025 (ratio 1.0000). Used only by C's
  M2.5 count features, Block S, and the served baseline D.
- Inputs to the rate models: M >= 5.0 (globally near-complete since the 1970s; M >= 4.5
  is not complete outside dense networks before ~2000 -- MEASURED here as 3,015 M4.5+ in
  1973 vs 7,878 in 2005 while M5+ moved 1,454 -> 1,844).
- The 2026 file of the old cache (no manifest, fetched 2026-06-26) is not used.

## 4. Metrics

All metrics pool every (issue time, cell) pair of a split -- 11,700 cells per issue
time, no cell excluded.

- **Primary: information gain per target** `IG = (LL_X - LL_0) / N_pos`, with the
  Bernoulli log-likelihood `LL = sum y log p + (1 - y) log(1 - p)` (p clipped to
  [1e-12, 1 - 1e-12], the site verifier's clip), `N_pos` the number of positive
  cell-windows, and the reference `0` = the constant probability `p0` = the FIT split's
  positive fraction for every cell ("uniform rate per cell", the CSEP-style
  uniform reference; the Bernoulli form is the binary-likelihood score of the contract's
  own "at least one" event). Units: nats per positive cell-window.
- **ROC-AUC** over all cell-times, tie-aware (`hazardpulse.core.metrics.roc_auc`).
- **Brier** and **BSS** against the same constant `p0`.
- **Reliability**: bins with edges 0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1
  (n, mean forecast, observed frequency), and calibration-in-the-large `sum p / sum y`.
- Secondary (reported, never used to choose): AUC among the live scorer's active
  cells (>= 5 M2.5+ events in the cell in the 30 days before t) -- the set D scores;
  Poisson information gain per earthquake with the site verifier's reference (uniform
  per cell, total = observed count), for comparability with
  `results/earthquake_prospective`; on FINAL, the same metrics against declustered
  mainshock targets (Gardner-Knopoff on the M >= 5 catalog).
- **Intervals**: block bootstrap over calendar months of the issue time (all issue times
  of a resampled month travel together), 2,000 resamples, percentile 95%. Paired
  differences use the same resamples for both models. Sensitivity on FINAL: quarter
  blocks. (Consecutive issue times share 23 of 30 outcome days and aftershock sequences
  last months, so resampling single issue times would understate the variance.)

## 5. Candidates (declared before fitting)

Complexity order: **A < B < C0 < C1**; D is the incumbent.

**A -- long-term smoothed seismicity** (the standard reference forecast).
Every M >= 5.0 epicentre from 1973-01-01 to t (strictly before) is spread over the grid
by `(r^2 + d^2)^-1.5` (great-circle r; 16 sub-points per cell weighted by cos(lat);
support radius clip(20 d, 300, 1500) km; each event's kernel normalised to 1 over the
domain). `s_t` = the normalised sum. `lambda = mu ((1 - eps) s_t + eps * area/total area)`,
`P = 1 - exp(-lambda)`. Fitted on FIT by maximum Bernoulli likelihood: `mu > 0`,
`0 < eps < 1` continuous; `d` in {10, 20, 35, 50, 75, 100, 150} km and input in
{all M >= 5, causally declustered M >= 5} by the maximum FIT likelihood. Causal
declustering = an event is dropped if it lies inside the Gardner-Knopoff window of an
EARLIER event at least as large (no foreshock rule, so no later event can change it).

**B -- A + short-term clustering (ETAS-style).**
`lambda_B = mu_B ((1 - eps_B) s_t + eps_B area/total) + sum_i K 10^(alpha (M_i - 5)) Omega(t - t_i) g_i(c)`,
sum over every M >= 5.0 event before t since 1973, `Omega(D) = integral_D^(D+30) (s + c)^-p ds`
(days), `g_i` the same kernel at width `d0 10^(0.5 (M_i - 5))`. `s_t` is A's chosen map
(same d, same input). Fitted on FIT by maximum Bernoulli likelihood of `1 - exp(-lambda_B)`:
`mu_B, eps_B, K, alpha in [0, 3], c in [1e-4, 10] d, p in [0.5, 3]` continuous;
`d0` in {5, 10, 20, 40} km by maximum FIT likelihood.

**C0 -- gradient-boosted trees on causal per-cell features, A and B as inputs.**
LightGBM binary log-loss on: log10 lambda_A, log10 lambda_B, log10(B's short-term part +
1e-9); M >= 4.5 counts in the cell over 7 d, 30 d, 365 d, 5 y and in the 3 x 3
neighbourhood over 30 d, 365 d; M2.5+ counts in the cell over 30 d, 365 d; largest
magnitude in the 3 x 3 neighbourhood over 30 d and 365 d; days since the last M >= 6 in
the 3 x 3 neighbourhood and its magnitude; days since the last M >= 7 in the 5 x 5
neighbourhood and its magnitude; mean depth and fraction deeper than 70 km of the cell's
M >= 4.5 events over 5 y; Aki b-value of M >= 5 events in the 3 x 3 neighbourhood over
10 y (n >= 30, else missing); the live "active" flag. No latitude/longitude. Rows: every
FIT cell-time that is positive or active, plus a seeded 5% sample of the other negatives
weighted 20. Fixed hyperparameters: 15 leaves, learning rate 0.03, min 200 rows per leaf,
feature/bagging fraction 0.8, L2 1.0; rounds by early stopping (100) on FIT issue times
2015-2017 when trained on 2005-2014, then refitted on all FIT rows with that round count.

**C1 -- C0 + Block S.** The 61 `definitive_model.compute_block_s` features on the M2.5+
catalog (events strictly before t, 5 y, 500 km) for active cells (missing elsewhere).
Cost-bounded: MEASURED 15.8 ms per cell-time, so it is computed for active cells only.

**D -- the served model, scored on this contract.** Exactly the live primary
probability: active cells -> the deep GRU K192 (`eq_deep_nowcast_m5.0_2025_K192.serve.npz`)
on the 1,827 days of M2.5+ before t -> the trust-layer calibrator
(`results/calibration/earthquake_calibration.json`); every other cell 0.0. Caveats
recorded in advance: the calibrator was fitted (2026-10-01) on matured live forecasts
that overlap FINAL (2025), which can only flatter D's log score and Brier there; the
GBT `earthquake_gbt_v1.json` (the registry's 0.907/0.509 model) is NOT re-scored -- its
Block C re-parses the whole history list per call (~1 s per cell-time; > 5 CPU-hours for
CHOOSE + FINAL) and it is not what the site publishes.

## 6. Fitting discipline

- Every parameter, every discrete choice (A's width and input, B's d0, C's round count)
  is chosen on FIT only. CHOOSE is touched only by the selection rule; DEV only for
  confirmation; FINAL once.
- Candidate maps are produced by `src/hazardpulse/earthquake/operational_forecast.py`
  (`RateEngine`), the same code the live scorer calls.

## 7. Selection rule (on CHOOSE)

Walk the chain A -> B -> C0 -> C1 with `current = A`. Candidate X replaces `current` iff
the paired month-block 95% interval of `IG(X) - IG(current)` lies entirely above 0; if
that interval contains 0, X replaces `current` only if the paired interval of
`AUC(X) - AUC(current)` lies entirely above 0 AND the IG point difference is >= 0.
Otherwise `current` stays and the walk continues with the next candidate against it.
D replaces the chain winner only under the same rule. **If nothing beats A, A is served.**

DEV is reported for every candidate; a pre-declared warning (no re-selection) is raised
if the chosen model's DEV IG is below the next-simpler candidate's with an interval
excluding 0.

## 8. Serving rule

The served artifact is the chosen candidate with its FIT parameters -- exactly the
object evaluated, no refit -- plus the frozen M >= 5 catalog 1973-01-01 .. 2026-10-01
(exclusive); live events from the cutoff on come from the scorer's own USGS fetch (the
scorer refuses if its fetch does not reach back to the cutoff). Its `model_version` is
`<name>-<first 12 hex of the SHA-256 of the artifact with CRLF normalised to LF>`. The
live scorer publishes the forecast for every cell (the replay artifact carries the full
grid, so the verifier scores the real forecast instead of a default 0), and the trust
calibrator fitted to the old model is NOT applied to the new one (it is bound to another
model_version).

---

## 9. Results (appended after the registration)

The registration above (sections 0-8) had SHA-256 `64b7ed0c1a34aa63...f488a430` when the
first candidate was fitted and was not edited afterwards. All numbers below are
**MEASURED** unless marked; intervals are 95% month-block bootstrap (2,000 resamples).

### 9.1 Execution log (no protocol change)

- `fit_ab.py` was restarted once for speed (map sums moved from CSR row-slicing copies to
  `bincount` over the CSR buffers; A at d = 10 km gave LL -39374.45 in both runs), then cut
  by a 30-minute tool limit after two of B's four widths; `--resume` reused the recorded
  rows after recomputing A's chosen map and checking its likelihood.
- Every analytic B gradient agreed with central differences (relative error <= 6.4e-7), and
  the FIT likelihood recomputed from the saved 11,700-cell maps equalled the fitted one for
  A (-39360.68) and B (-38666.69): the pair-based likelihood is the full-map likelihood.
- After CHOOSE selected C0, C's feature code moved byte-identically from
  `scripts/earthquake_program/features_c.py` to `src/hazardpulse/earthquake/operational_features.py`
  (diff of everything below the header: empty).

### 9.2 FIT (674 issue times 2005-2017, 7,176 positive cell-windows, p0 = 9.10e-4)

A, information gain per target (nats) by kernel width and input:

| d (km) | 10 | 20 | 35 | 50 | 75 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| all M5+ | 2.515 | **2.517** | 2.489 | 2.450 | 2.375 | 2.310 | 2.190 |
| causally declustered | 2.462 | 2.459 | 2.424 | 2.381 | 2.302 | 2.238 | 2.121 |

Chosen A: d = 20 km, all M5+, mu = 10.78 expected positive cells per window, eps = 0.0038.
Declustering the input loses ~0.06 nats: the aftershock-rich places are where M6+ recur.

B by d0: 2.613 (5 km), 2.599 (10), 2.583 (20), 2.567 (40). Chosen d0 = 5 km:
K = 9.2e-4, alpha = 0.53, c = 0.010 d, p = 0.78, mu_B = 3.32, eps_B = 0.015 (6.5 M pairs).
With p < 1 and mu_B a third of A's mu, the triggered sum also carries decade-scale memory
of past events (HYPOTHESIS for the interpretation; the parameters are MEASURED).

C (461,406 rows, weighted population 7.90 M = 674 x 11,700 within sampling): C0 21
features, 156 rounds, internal validation log-loss 0.004480; C1 82 features, 141 rounds,
0.004509. C0's largest gains: log10 lambda_short, log10 lambda_B, log10 lambda_A, M4.5+
count in the cell over 5 y, days since the last M6+ nearby, b-value, mean depth.

D: 40,825 active cell-times re-scored over CHOOSE/DEV/FINAL; 9 edge cells where the deep
model returns nothing (their events lie beyond the latitude band) were scored 0 (live they
would fall to the GBT/heuristic tier).

### 9.3 CHOOSE (153 issue times 2018-2020; 1,790,100 cell-times; 1,512 positive; 1,686 M6+)

| | IG/target | AUC | BSS | AUC active cells | sum p / sum y |
|---|---|---|---|---|---|
| A | 2.572 [2.469, 2.687] | 0.9674 [0.9642, 0.9706] | +0.0227 | 0.760 | 1.08 |
| B | 2.597 [2.493, 2.702] | 0.9677 [0.9645, 0.9708] | +0.0240 | 0.761 | 1.08 |
| C0 | 2.607 [2.509, 2.706] | 0.9692 [0.9664, 0.9722] | +0.0237 | 0.742 | 1.00 |
| C1 | 2.605 [2.506, 2.714] | 0.9691 [0.9662, 0.9721] | +0.0239 | 0.737 | 0.99 |
| D | -13.355 [-14.246, -12.406] | 0.6324 [0.6147, 0.6515] | -0.0286 | 0.672 | 0.44 |

Rule walk (paired, X - current):

| step | dIG | dAUC | verdict |
|---|---|---|---|
| B vs A | +0.025 [-0.003, +0.053] | +0.0003 [-0.0008, +0.0012] | A stays |
| C0 vs A | +0.035 [-0.021, +0.084] | +0.0019 [+0.0003, +0.0032] | **C0 replaces A** (IG interval contains 0; AUC interval above 0; IG point >= 0) |
| C1 vs C0 | -0.002 [-0.043, +0.037] | -0.0001 [-0.0007, +0.0004] | C0 stays |
| D vs C0 | -15.96 [-16.79, -15.08] | -0.337 [-0.354, -0.319] | C0 stays |

**Decision: C0** (`results/earthquake_program/choose.json`).

### 9.4 DEV (100 issue times 2021-2022; 980 positive)

IG: A 2.638, B 2.718, C0 2.717, C1 2.711, D -12.45. C0 - A +0.079 [+0.010, +0.143];
C0 - B -0.0005 [-0.072, +0.072]: the pre-declared warning did not fire.

### 9.5 FINAL, read once (153 issue times 2023-01-02 .. 2025-12-01; 1,790,100 cell-times; 1,384 positive cell-windows; 1,637 M6+ events)

| | IG/target (nats) | AUC | BSS vs uniform | AUC active cells | sum p / sum y |
|---|---|---|---|---|---|
| A long-term | 2.533 [2.406, 2.652] | 0.9668 [0.9622, 0.9710] | +0.017 [+0.013, +0.022] | 0.745 [0.708, 0.779] | 1.18 [1.07, 1.30] |
| B + clustering | 2.603 [2.471, 2.728] | 0.9679 [0.9631, 0.9722] | +0.022 [+0.015, +0.029] | 0.773 [0.736, 0.806] | 1.20 [1.10, 1.33] |
| **C0 served** | **2.593 [2.469, 2.712]** | **0.9681 [0.9633, 0.9723]** | **+0.021 [+0.013, +0.028]** | 0.767 [0.733, 0.798] | 1.14 [1.03, 1.26] |
| C1 (+ Block S) | 2.605 [2.472, 2.731] | 0.9686 [0.9641, 0.9727] | +0.022 [+0.014, +0.030] | 0.774 [0.738, 0.803] | 1.14 [1.04, 1.27] |
| D served until now | -13.669 [-14.625, -12.689] | 0.6247 [0.6056, 0.6443] | -0.029 [-0.039, -0.019] | 0.680 [0.645, 0.710] | 0.51 [0.46, 0.56] |

Paired on FINAL: C0 - A dIG +0.061 [+0.0003, +0.120], dAUC +0.0013 [+0.0000, +0.0026];
B - A +0.070 [+0.028, +0.120]; C0 - B -0.010 [-0.056, +0.038]; C1 - C0 +0.012
[-0.014, +0.044]; D - C0 -16.26 [-17.15, -15.34]. Quarter blocks widen the intervals
slightly (C0 IG [2.413, 2.728], AUC [0.9615, 0.9738]) and change no conclusion. The site
verifier's Poisson score per earthquake (uniform reference with the observed count):
A 2.51, B 2.60, C0 2.58, D -13.24 -- the live record of D in
`results/earthquake_prospective/prospective_summary.json` (600 matured forecasts,
2026-10-01) is -15.5 (OBSERVED), the same failure.

C0 reliability on FINAL (forecast bin: n, mean forecast, observed frequency):
< 1e-4: 1,396,896, 2.3e-5, 1.3e-5 · 1e-4-3e-4: 89,086, 1.8e-4, 9.0e-5 · 3e-4-1e-3: 110,820,
5.7e-4, 5.4e-4 · 1e-3-3e-3: 85,326, 1.7e-3, 1.5e-3 · 3e-3-0.01: 71,047, 5.8e-3, 4.5e-3 ·
0.01-0.03: 28,810, 0.015, 0.018 · 0.03-0.1: 7,695, 0.053, 0.037 · 0.1-0.3: 387, 0.136, 0.129 ·
>= 0.3: 33, 0.41, 0.30.

Secondary, declustered M6+ mainshock targets (IG reference is still the all-M6+ p0, so read
the differences): A 2.351, B 2.350, C0 2.364, C1 2.386 -- the dynamic models' gain over A
is clustering (aftershock-type M6+); on mainshocks B adds nothing and C0 +0.013.

### 9.6 What this establishes

- **The served model had no operational value on this contract** (MEASURED on CHOOSE, DEV,
  FINAL): AUC 0.62-0.65 and an information gain of -12.5 to -13.7 nats per target (worse
  than a uniform forecast, ~16 nats below C0), because it publishes 0 for every cell
  outside the ~100 recently active ones -- 1,028 of the 1,384 FINAL positive cell-windows
  (74%) fell in cells it called impossible.
- **Most of the honest skill is where earthquakes have happened** (A: 2.53 nats per target,
  AUC 0.967). Clustering adds ~0.06-0.07 nats (B, C0 on FINAL, intervals excluding 0 vs A);
  among active cells the ranking AUC rises from 0.745 to 0.77.
- **B and C0 are indistinguishable** on DEV and FINAL. C0 is served because the registered
  rule chose it on CHOOSE (through the AUC clause); switching to the simpler B after seeing
  FINAL would be the selection-after-test this program exists to prevent. A re-registration
  could adopt B's simplicity; it would cost nothing measurable.
- **Block S (C1) adds nothing** over C0 (CHOOSE dIG -0.002; FINAL +0.012 [-0.014, +0.044]).
- **Calibration drifts with the global rate**: every model over-forecasts 2023-2025
  (sum p / sum y 1.14-1.20; 2024 had 99 M6+ against a 2005-2017 mean of 158 per year).

### 9.7 What is served now, and how

- `results/models/earthquake_operational_v1.json` (3.3 MB, one JSON line):
  `model_version = eq_operational_C0_v1-d28c62a2e35b` (SHA-256 of the LF-normalised file,
  `d28c62a2e35b9e19...`). It holds B's rate model, A's long-term parameters, the 156 trees
  as a `hazardpulse_lgbm_payload/1` (scored in NumPy by `hazardpulse.tornado.lgbm_payload`,
  LightGBM's split semantics), the 21 feature names, and the 89,861 ComCat M5+ events
  1973-01-01 .. 2026-10-01 as the frozen catalog.
- Parity: at one issue time per split the live code path (`forecast_from_artifact`, frozen
  catalog + a simulated 1,827-day live fetch) reproduced the evaluated probability grid
  **bit for bit**; the NumPy trees equal LightGBM's output exactly (max |dp| = 0) on 175
  stored rows. CI re-checks the rate grids (SHA-256) and the stored rows on every run.
- `scripts/fetch_and_score_earthquake.py` loads the artifact (fails closed without it),
  computes all 11,700 cells, lists the active cells plus the 25 highest-probability cells,
  writes the full grid into each replay artifact (`probability_grid`), stamps every
  forecast, ledger entry and page with the content-bound model_version, attaches receipts
  (`hazardpulse/forecast/v1`, no interval claimed), and does not apply the trust
  calibrator fitted to the old model. `scripts/score_earthquake_prospective.py` scores the
  grid and pools calibration data for one model_version only.
- End-to-end live run (2026-10-03 01:00Z, real USGS fetch): 137,117 M2.5+ history events,
  89,877 M5+ inputs, sum of P 10.4 expected cells with an M6+ in 30 days, top cell 21.7%
  (21S 169E, after a recent M6.6); 98/98 receipts verify. The forecast step costs 31.5 s and
  147 MiB peak.
- The model registry lists the served entry with its FINAL numbers bound; the GBT v1 entry
  is marked NOT SERVED.

### 9.8 Uncertainty and limits

- **Final catalog vs live catalog**: the evaluation used the revised ComCat (final
  magnitudes, complete aftershock lists); live inputs are preliminary and short-term
  aftershock catalogs are incomplete for hours to days after large events. The clustering
  term is most exposed. HYPOTHESIS: live skill somewhat below FINAL; the prospective grid
  scoring now measures it.
- **Interval dependence**: month blocks treat months as independent; aftershock sequences
  span months. Quarter blocks gave slightly wider intervals and the same conclusions; the
  C0 - A IG interval on FINAL only just excludes 0.
- **Non-stationary rate**: the fitted level (2005-2017) over-forecast 2023-2025 by ~14%.
  No recalibration was registered, so none is applied.
- **Choice margin**: C0 was chosen on CHOOSE through AUC with an IG interval containing 0;
  its FINAL advantage over A is real but small, and it equals B.
- **D re-creation**: scored at cell centres with the 2026-10-01 calibrator, which was
  fitted on matured live forecasts overlapping 2025 (it can only flatter D); 9 edge-cell
  fallbacks scored 0.
- **AUC over all cells is mostly geography**; the active-cell AUC (0.74-0.77) is the better
  ranking measure and is secondary by registration.
- The frozen catalog must be rebuilt before 2031-10 (the scorer refuses once its 1,827-day
  fetch no longer reaches the cutoff). The secondary deep fields `prob_30d_local` and
  `prob_op_m5_30d` are still published and were not evaluated here.
- Once 30 days of operational forecasts mature, `verification-score.yml` will fit a
  Venn-Abers calibrator bound to the new model_version and the scorer will apply it; with
  few matured forecasts it may widen intervals or degrade calibration (not tested).

### 9.9 Reproduce

```
python scripts/earthquake_program/catalog.py --cutoff 2026-10-01 --m25-dir <.cache/earthquake/usgs>
python scripts/earthquake_program/fit_ab.py                 # A grid, B grid, maps, per-split forecasts
python scripts/earthquake_program/blocks_c1.py --workers 4  # Block S for active cells (C1)
python scripts/earthquake_program/fit_c.py                  # C0, C1
python scripts/earthquake_program/score_d.py --workers 4    # the served model, re-created
python scripts/earthquake_program/evaluate.py --split choose
python scripts/earthquake_program/evaluate.py --split dev
python scripts/earthquake_program/evaluate.py --split final # refuses a second read
python scripts/earthquake_program/build_artifact.py         # served artifact + parity checks
python -m pytest -q tests/test_earthquake_operational_forecast.py \
    tests/test_earthquake_operational_serving.py tests/test_earthquake_program_metrics.py
```
