# Coherence Language twin of the tornado scorer -- UNMAINTAINED, not the served path

Status (2026-10-02): these `.cl` files are a research port written 2026-07; nothing in CI,
the scorers or the site runs them. The served tornado model is the Python pipeline
(`scripts/fetch_and_score_tornado.py` -> `src/hazardpulse/tornado/definitive_model.py`).
The twin has NOT received the 2026-10-01/02 audit fixes (docs/AUDIT_2026-10-01.md), and it
differs from the served path in ways that change its numbers:

| Twin (`tornado/*.cl`) | Served Python |
|---|---|
| "torsion" = SRH x (dtau/di * dtau/dj), neither a curl nor a tilting term | tilting torsion (shear x grad tau) / 25 (`coherence_engine.compute_tilting_torsion`), T3 |
| Helmholtz by 200 fixed Jacobi sweeps, no residual check | certified PCG to a 1e-8 relative residual (`tau_c_solver`), S1 |
| 80 km grid indexed as if HRRR were lat/lon | g3: native LCC points binned to true lat/lon, winds rotated to earth, T4 |
| no label clock (scoring only) | SPC CST -> UTC instants, T1 |
| scores with the weights it is handed | refuses a payload whose feature order or geometry differs |

Bringing it in line means porting those five items and a parity test against the Python
scorer on archived storms. Until then, treat any number it produces as unverified.

Fixed 2026-10-02: the STP fallback in `scorer.cl` was written across lines that began with
`*`, which Coherence parses as new statements, so the SRH and shear factors were silently
discarded (`stp = min(mucape/1500, 2)` alone). Wrapped in parentheses; the language's own
analyzer reports SEM001 once on the old file and zero times on the new one.
