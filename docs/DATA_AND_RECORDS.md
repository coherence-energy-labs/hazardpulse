# Data and records: what is kept, where, and how it is checked

Every number HazardPulse publishes, and every number its models produce in shadow, should be
possible to look up later. It should also be possible to show it is unchanged and to trace how it
was made: from the data, through the decision that chose the model, to the model and the inputs
of that forecast. This file lists what is kept, where, and which check would catch a gap. It also
lists what is not kept yet.

## Live forecasts

| what | where | kept | checked by |
|---|---|---|---|
| Every forecast run, every hazard: the published numbers, the model version behind each, and for hurricanes every shadow model's forecast | `dist/data/replay/{eq,to,hu}_fcst_*.json` (in git) | forever (nothing prunes them) | `tests/test_site_integrity*`, the site build |
| Earthquake: the full 11,700-cell probability grid, the model and artifact hashes, and the input event count | `eq_fcst_*.json` (`probability_grid`, `operational_model`) | forever | the prospective scorer reads the grid |
| Hurricane shadows: **the exact input vector** of every model (since 2026-10-03), plus the IR features for v10.3 | `hu_fcst_*.json`, `inputs` in each `ri_*_shadow` | forever | `scripts/audit_hurricane_records.py` recomputes each forecast from its inputs |
| Tornado v3: **the exact feature vector and warning state** every served product read (since 2026-10-03) | `to_fcst_*.json`, `storms[].v3.inputs` | forever | `scripts/audit_tornado_records.py` recomputes each 60-min forecast |
| Hash-chained ledgers, one entry per run (`hash`, `prev_hash`). The hurricane ledger also carries a content hash of the forecast. | `dist/data/{earthquake,tornado,hurricane}-ledger.jsonl` | append-only | the verification page reports link mismatches; the hurricane audit checks each entry against its replay file |
| The live inputs NOAA publishes for hurricanes: every SHIPS text | `results/hurricane_ships_archive/<year>/` | forever (NHC keeps only the current season) | `tests/test_ships_archive.py` |

## How each forecast was scored

| what | where |
|---|---|
| Prospective scores, published and shadow models, each hazard | `results/{earthquake,tornado,hurricane}_prospective/` |
| The hurricane challengers' prospective test: looks, claim rules, challenger vs champion | `results/hurricane_prospective/v9_shadow.json` |
| Record audits (chain, content, recomputation) | `results/{hurricane,tornado}_prospective/record_audit.json` |

The verification workflow runs the scorers and both audits with `--strict`, so a broken record fails
the job, and commits the results.

## How each model was made

| what | where | checked by |
|---|---|---|
| The pre-registrations: candidates, controls and rules, written before any result | `docs/*_PROGRAM.md`. Tags `prereg-*` mark the commit of each registration. | the tag commits predate the results |
| Selection and evaluation results, each recording the hash of the table it used | `results/calibration/*.json`, `results/earthquake_program/*.json` | the controls in each script reproduce the previous champion bit for bit |
| The models themselves, frozen; the model version is a hash of the artifact | `results/models/*.json` | each live scorer refuses an artifact whose version changed while it ran |
| **The registry: every live model with its lineage** (pre-registration tag, program, the results files with their hashes, and where its forecasts and scores are kept) | `dist/data/model-registry.json` (served at `/api/v1/registry/models`) | `tests/test_model_registry.py` (the committed file must equal what the repository produces) |
| The queue of what is being tried next, and every dead end with its evidence | `docs/MODEL_IMPROVEMENT_LEDGER.md` | |

## The research data

The training and evaluation datasets are too large for git (about 1 GB). Each file is **pinned** by
size and SHA-256 in `results/data_manifests/<dataset>.json` (committed). They are **published** as
release assets: `results/data_manifests/releases.json` lists each archive with its SHA-256. The
first is release `data-2026-10-03`.

To restore and verify:
1. Download the archives and extract them into `<repo>/.cache`.
2. Run `python scripts/data_manifest.py --verify`.

To publish a new snapshot after the data changes:
1. `python scripts/data_manifest.py`
2. `python scripts/data_release.py --out <dir>`
3. Create a new release with those archives, and add its record to `releases.json`.

## Not kept yet (known gaps)

- **Earthquake live catalog at forecast time.** The record keeps the model, the artifact hashes and
  the input event count, but not the ComCat events themselves. ComCat revises magnitudes, so an
  exact recomputation needs the events as fetched. The fix is to store the events fetched since
  the artifact's frozen-catalog cutoff with each run (small), plus a periodic M2.5+ snapshot for
  C0's features.
- **Records written before 2026-10-03** have no stored inputs. The audits count them as "not covered",
  never as passing.
- **Raw satellite images** are not kept, only the storm crops used in training and the 14 IR features
  in each live record. The images themselves stay available from NOAA's public bucket.
