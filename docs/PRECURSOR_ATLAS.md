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
