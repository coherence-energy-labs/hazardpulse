# Precursor atlas: does anything happen in the Earth system before large earthquakes? (registered 2026-10-11, before any result)

**Why.** After the Panama M7.7 (2026-10-09; `docs/EARTHQUAKE_FORECAST_PROGRAM.md` section 13) Josh asked whether anything
anywhere -- a volcano, the tides, other earthquakes, the Sun, the planet's spin -- changed before it. Any single
earthquake will show *something* beforehand, because many signals exist. A connection is real only if it shows up
across many earthquakes, against the same place at other times, and survives a test on earthquakes it was not
found on.

**Events.** Shallow (<= 70 km) M7.0+ mainshocks in ComCat, 1990-01-01 .. 2026-10-09: no event of equal or larger
magnitude within 150 km in the 30 days before (the definition of `scripts/earthquake_program/aftershock_base_rates.py`).

**Controls.** For each event, 20 times at the same epicentre, drawn uniformly from [t - 3 y, t + 3 y]:
- a draw is rejected within 180 days of t;
- a draw is rejected when an M7+ follows within 500 km in the 30 days after it, so a control is never itself the
  eve of a large earthquake;
- the seed is fixed: 20,261,011 plus the event's index.

The +-3 y window keeps slow trends (the solar cycle, decadal changes in the rate of large earthquakes, the length of
day) from appearing as connections.

**Signals.** Each is computed at the epicentre and time, from data before that time only.

| id | signal | source |
|---|---|---|
| T1 | days to the nearest new or full moon (smaller = spring tide) | low-precision lunar ephemeris (Meeus, *Astronomical Algorithms*, ch. 47) |
| T2 | Earth-Moon distance (smaller = stronger tide) | same |
| Q1 | M7+ within 1,500 km (beyond 150 km) in the 120 days before | ComCat M6+ |
| Q2 | M6+ within 500 km in the 365 days before | ComCat M6+ |
| Q3 | M7+ anywhere on Earth in the 30 days before | ComCat M6+ |
| V1 | eruptions starting within 500 km in the 365 days before | Smithsonian GVP Holocene eruptions and volcanoes |
| V2 | eruptions starting within 1,000 km in the 90 days before | same |
| G1 | the highest Kp in the 72 hours before | GFZ Potsdam Kp since 1932 |
| G2 | the mean Kp over the 27 days before | same |
| S1 | mean daily sunspot number over the 27 days before | SILSO (Royal Observatory of Belgium) |
| R1 | length of day, the 30-day mean before | IERS EOP C04 |
| R2 | length of day, the change over the year before | same |

**Controls of the instrument (added 2026-10-11 after the tag, still before any result).** Neither is one of the 12
signals or in the BH family. If either fails, every result is void.
- **P+ (positive control): M6+ within 150 km in the 30 days before.** Large earthquakes cluster after nearby
  moderate ones, so P+ must come out with mean u above 0.5 at p < 0.01 on all events. If it does not, the design
  cannot see a real connection.
- **P- (negative control): a seeded random number per time.** It must not reach p < 0.01.

**The statistic.** For each event and signal, u is the event value's mid-rank percentile among its own 20 controls.
- With no connection, u is uniform and its mean is 0.5.
- The test is z = (mean u - 0.5) / sqrt(1 / (12 n)), two-sided.
- The probability gain reported is the share of events above their controls' 80th percentile, divided by 0.2.

**Discovery and confirmation (the rule, fixed now).**
- **DISCOVER:** events of 1990-2009. Benjamini-Hochberg across the 12 signals at q = 0.10.
- **CONFIRM:** events of 2010-2026. A discovered signal must go the same way, one-sided p < 0.05.
- **A connection is claimed** only for a signal that is discovered and confirmed. Every other result is reported as
  found, with its interval: "no connection found at this sample size" is a result too.

**Then, for a confirmed signal only:** its probability gain sets whether it is worth an earthquake-program amendment.
A gain of 1.2 on a 0.2% chance moves nothing that saves a life, and the report must say so.

**Not testable here yet, and why.**
- Ionospheric electron content: daily global IGS maps exist from 1998, but the download is large and runs on a
  runner.
- Outgoing longwave radiation, GNSS displacement, rainfall and groundwater: large or regional data.
- Animal behaviour and coastline or island changes: no systematic global record exists.

These are recorded as the next tier, not dismissed.

**Run.** `python scripts/research/precursor_atlas.py` (fetches its public inputs into `.cache/precursor/`). Results go
to `results/precursor_atlas/atlas.json`.

## Outcome 1 (2026-10-11): no connection found for any of the 12 signals; the instrument works

`results/precursor_atlas/atlas.json`.
- **386 events:** 207 in 1990-2009 (discovery) and 179 in 2010-2026 (confirmation).
- **The ephemeris:** it put the 2024-04-08 eclipse new moon 22 minutes off and the 2024-04-23 full moon 11 minutes
  off, with the Moon at 359,780 km.

**The instrument passed.**
- **P+ (nearby M6+ activity in the 30 days before), the known connection:** mean u 0.567, p 5.5e-6. The design sees a
  real connection of that size.
- **P- (noise):** mean u 0.505, p 0.74.

**The 12 signals** (mean u over all events, where 0.5 means no connection):

| signal | all events | 1990-2009 | 2010-2026 | verdict |
|---|---|---|---|---|
| spring tide (nearness to new or full moon) | 0.483 (p 0.24) | 0.480 | 0.486 | no connection |
| Moon near (perigee) | 0.532 (p 0.030) | 0.546 (p 0.021) | 0.515 (p 0.48) | not discovered after BH, not replicated |
| M7+ within 150-1,500 km, 120 days before | 0.510 (p 0.48) | 0.498 | 0.524 | no connection |
| M6+ within 500 km, 365 days before | 0.514 (p 0.33) | 0.501 | 0.530 | no connection |
| M7+ anywhere, 30 days before | 0.511 (p 0.47) | 0.486 | 0.539 (p 0.068) | no connection |
| eruptions within 500 km, 365 days before | 0.486 (p 0.35) | 0.481 | 0.492 | no connection |
| eruptions within 1,000 km, 90 days before | 0.498 (p 0.89) | 0.500 | 0.496 | no connection |
| highest Kp, 72 h before | 0.511 (p 0.47) | 0.499 | 0.524 | no connection |
| mean Kp, 27 days before | 0.512 (p 0.40) | 0.519 | 0.504 | no connection |
| sunspot number, 27 days before | 0.494 (p 0.68) | 0.475 | 0.516 | no connection |
| length of day | 0.512 (p 0.43) | 0.528 | 0.493 | no connection |
| length of day, change over the year | 0.509 (p 0.52) | 0.496 | 0.525 | no connection |

**What this rules out, and what it does not (DERIVED from the sample size).**
- With 386 events, the standard error of mean u is 0.015. A shift of about 0.04 would have shown at p < 0.01, which
  is smaller than P+'s.
- So no signal here carries a connection as strong as nearby moderate seismicity, the one this program already uses.
- Weaker connections (probability gains of about 1.2 or less) are not excluded. At a 0.2% chance they would not move
  a forecast to anything that saves a life.

**Panama, the anecdote that shows the trap.** Among its own 20 control times, the M7.7 struck:
- nearer a spring tide than all 20 (u 1.00);
- after more regional M7+ activity than 18 of them (u 0.88; Colombia M7.4 in August, Venezuela M7.2 in June);
- at a longer day than all 20.

Each of these looks like a connection in one earthquake. Across 386, none is: spring tide 0.483, regional M7+ 0.510,
length of day 0.512.

**The next tier** (recorded, not dismissed):
- ionospheric electron content (IGS global maps, 1998+);
- outgoing longwave radiation;
- GNSS displacement (Nevada Geodetic Laboratory, about 2000+);
- satellite surface change (InSAR, coastlines).

They need larger downloads on a runner, and each is tested by this same protocol and controls.

## Run 2 -- the texture of seismicity before large earthquakes (registered 2026-10-11, before any run-2 result)

**Why.** Run 1 tested forcings from outside the fault: tides, the Sun, the planet's spin, volcanoes, distant
earthquakes. None connected. A fault fails when its own stress meets its strength, so run 2 looks at what the rocks
around it record. Seismicity has a shape -- a size mix, quiet spells, acceleration, deeper activity -- that run 1's
counts of large events ignore.

**Data.** ComCat M4.5+, 1984-today, fetched by year into `.cache/precursor/`.

**Design.** Run 1's, unchanged: the same 386 events, the same 20 same-place controls each within +-3 y, the same
statistic, P+ and P- as instrument controls, BH at q = 0.10 on 1990-2009, and confirmation on 2010-2026 (same sign,
one-sided p < 0.05).

**The 10 signals.** At the epicentre, from events before the time; shallow (<= 70 km) unless stated.

| id | signal |
|---|---|
| F1 | M4.5+ within 100 km, 7 days before |
| F2 | M4.5+ within 100 km, 30 days before |
| F3 | M4.5+ within 200 km, 365 days before |
| B1 | minus the b-value (Aki-Utsu, Mc 4.5) of M4.5+ within 200 km over the 3 years before (needs 30 events) |
| B2 | minus the change in b: the 3 years before, against the 3-10 years before (each needs 30 events) |
| Z1 | quiescence or activation: the M4.5+ rate within 200 km in the year before, over the mean yearly rate of the 5 years before that (each count + 0.5) |
| A1 | acceleration: seismic moment of M4.5+ within 200 km in the year before, over the mean yearly moment of the 10 years before |
| D1 | deeper activity: M4.5+ at 70-300 km within 300 km, the year before over the mean yearly rate of the 5 years before |
| M1 | the largest magnitude within 200 km in the 365 days before (4.4 if none) |
| L1 | days since the last M6.5+ within 150 km, capped at 10 years. A high u means "due" (renewal); a low u means "soon after" (clustering) |

## Run 2 outcome as registered, and the flaw found in reading it (2026-10-11)

**As registered** (`results/precursor_atlas/atlas_2.json`): the instrument passes, with P+ at mean u 0.567 and P- at
0.524 (p 0.10).

**Five signals are discovered on 1990-2009 and confirmed on 2010-2026:**

| signal | mean u | p |
|---|---|---|
| A1, moment in the last year over the decade before | 0.628 | 4e-18 |
| L1, days since the last M6.5+ | 0.623 | 6e-17 |
| F1, M4.5+ within 100 km in 7 days | 0.601 | 6e-12 |
| Z1, the rate in the last year over the 5 before | 0.585 | 6e-9 |
| F2, M4.5+ within 100 km in 30 days | 0.581 | 4e-8 |

F3, B1, B2, D1 and M1 show no connection.

**The flaw.** Half of each event's controls fall after it (t + 180 d to t + 3 y). For every signal built from past
seismicity, those controls see the event itself and its aftershock sequence as their own recent history.
- **L1** is biased toward the event: a later control's "last M6.5+" is the event itself, so its time since is short.
  L1's result may be this flaw alone.
- **A1, Z1, F1, F2, F3, M1, B1 and B2** are biased against the event, because later controls carry its aftershocks.
  A positive result for these survives the flaw.
- **Run 1's Q1, Q2 and Q3** were biased the same way, against the event.

## Run 2b -- past-only controls (registered 2026-10-11, before any run-2b result)

Runs 1 and 2 are re-run with one change to the controls. Everything else is unchanged: signals, statistic, rule,
instrument controls.
- **Draws:** 20 per event from [t - 6 y, t - 180 d].
- **Screen 1 (the mainshock rule, mirrored):** no M7+ within 150 km in the 30 days before the draw.
- **Screen 2 (unchanged):** no M7+ within 500 km in the 30 days after it.
- **Which result decides:** run 2b's. Run 2's numbers stay in the record as the flawed design's.
- **One data rule, added before running 2b.** Past-only controls reach back to 1985, so a signal whose history would
  start before the catalog's (1984-01-01) is missing for events and controls alike. The years each needs: B1 3, B2
  10, Z1 6, D1 6, A1 11, L1 10. Run 1's signals need at most a year. Run 2 as registered stays as it ran.

## Run 2b outcome (2026-10-11, decides): the region wakes up before many large earthquakes

`results/precursor_atlas/atlas_1b.json`, `atlas_2b.json`. The instrument passes: P+ at mean u 0.575 (p 3e-7), P- at
0.505 and 0.524.

**Confirmed: discovered on 1990-2009 and confirmed on 2010-2026, with past-only controls.**

| signal | mean u | probability gain, top fifth | p |
|---|---|---|---|
| A1, moment in the last year over the decade before (200 km) | **0.640** | **2.28** | 4e-17 |
| M1, the largest magnitude within 200 km in the year before | 0.628 | 2.01 | 4e-18 |
| F3, M4.5+ within 200 km in the year before | 0.622 | 1.88 | 8e-17 |
| F2, M4.5+ within 100 km in the 30 days before | 0.617 | 1.61 | 2e-15 |
| F1, M4.5+ within 100 km in the 7 days before | 0.614 | 1.37 | 9e-15 |
| B2, the b-value falling (3 years against the 3-10 before) | 0.577 | 1.83 | 6e-4 (n 167) |
| Z1, the rate in the last year over the 5 before | 0.568 | 1.84 | 1e-5 |
| Q2 (run 1), M6+ within 500 km in the year before | 0.557 | 1.18 | ~1e-4 |

46% of large mainshocks fell in the top fifth of their own region's recent moment history, where 20% would fall by
chance.

**Killed by 2b.**
- **L1 ("due" after a long quiet):** 0.508. Run 2's 0.623 was the flaw: later controls saw the event as their last
  M6.5+.
- **Sunspots and geomagnetic activity** are discovered with opposite signs in the two halves (sunspots 0.372, then
  0.605). That is a solar-cycle artifact of past-only controls, and the confirmation rule refuses it.
- **Every other run-1 signal** (tides, perigee, distant and global large earthquakes, eruptions, the length of day):
  no connection.

**Panama was the other kind of earthquake:** A1 0.25, Z1 0.10, M1 0.30. Its region was quieter than its own past, so
these signals would have lowered its chance.

**Beyond what the served model reads (diagnostic, conditional matched-set logistic;
`scripts/research/precursor_conditional.py`, `results/precursor_atlas/conditional_2b.json`).**
- The served C0 already reads counts (7 d, 30 d, 365 d, 5 y), the largest recent magnitude and the b-value level.
- **Next to those, A1 keeps a clear coefficient:** +0.61 per s.d., z +2.9 (311 matched sets). A1 and Z1 together add
  a likelihood ratio of 59.6 on 2 df to F1, F3 and M1.
- **Part of that is the 5-year activity level, which C0 already has as a count.** The moment weighting, A1, is the
  part C0 does not hold in any form.
- **B2 adds nothing clear** next to them (z +1.3, n 176).
- This is a diagnostic chosen after seeing run 2b. Whether A1 improves the forecast is decided by an earthquake-program
  amendment on its own data.

## Run 3 -- the ground signalling the sky: heat radiated before large earthquakes (registered 2026-10-11, before any run-3 result)

**Why.** A family of published claims ("lithosphere-atmosphere-ionosphere coupling") holds that the crust signals the
atmosphere before a large earthquake. The best-known measurements are:
- thermal anomalies, as outgoing longwave radiation (OLR) over the epicentre;
- ionospheric electron content.

hazardpulse's research page tests space-weather versions of this; run 3 tests the surface one.

**Data.** NOAA's interpolated daily OLR, 2.5 x 2.5 degrees, 1974 onward (`olr.day.mean.nc`, NOAA PSL). The value is
the grid point nearest the epicentre. Events after the file's last day have no value.

**Design.** Run 2b's: the same events, past-only controls (6 y to 180 d before, the mainshock rule mirrored), the
statistic, P+ and P- as instrument controls, BH at q = 0.10 on 1990-2009, and confirmation on 2010-onward (same
sign, one-sided p < 0.05).

**Signals.**

| id | signal |
|---|---|
| O1 | the largest daily OLR over the 7 days before, minus the median of the 30 days before that window (a short-term thermal anomaly) |
| O2 | the mean OLR of the 7 days before, minus the same calendar days' mean over the 5 years before (an anomaly against season) |

**Ionospheric electron content is run 3b.** Its archives (CODE, IGS mirrors) did not answer from the local machine.
It runs on a runner that checks its own access first, by the same design.

## Run 3 outcome (2026-10-11): no thermal precursor

`results/precursor_atlas/atlas_3.json`. NOAA's OLR file ends 2022-12-31, so 342 of the 386 events have values.
- **The instrument passes:** P+ 0.575 (p 3e-7), P- 0.529 (p 0.05).
- **O1, the OLR spike over the 7 days before:** mean u 0.512, p 0.44 (1990-2009 0.511; 2010-2022 0.513). No
  connection.
- **O2, OLR against the same days of the 5 years before:** 0.500, p 0.98. No connection.

The thermal-anomaly claim does not appear at the epicentre in daily 2.5-degree OLR across 342 large earthquakes.
Finer or other thermal products (land-surface temperature, hourly geostationary infrared) are not tested here.
