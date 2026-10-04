# The website

hazardpulse.com is one rendering of the published artifacts. Nothing on it is typed by hand except prose.

## How a page is made

```
scorer  ->  dist/data/*.json, dist/data/replay/*.json     (what was forecast)
        ->  build_site_artifacts()                         (evidence indexes, quality checks, verification)
        ->  hazardpulse.site.build.build_site()            (every page, from those files)
```

* `src/hazardpulse/site/shell.py`: the one `<head>`, header and footer, and the page registry (`PAGES`:
  title, description, indexing). Every page is `shell.document(path, main)`.
* `src/hazardpulse/site/data.py`: `SiteData`, everything the pages show, read from the artifacts. A page
  is a pure function of it.
* `src/hazardpulse/site/pages/`: one module per generated page (home, live overview, earthquake,
  hurricane, tornado, track record, evidence, status, research).
* Hand-kept pages (methods, registry, tornado test results, API, legal, 404) keep their `<main>` in
  `dist/` and are re-wrapped in the current shell on every build. Their model numbers live between
  `<!-- hp-evidence:NAME -->` markers and are rendered by `hazardpulse.verification.evidence_pages` from
  the results bound to each served model file.
* `fmt.py` (numbers, times, the one chance colour scale), `places.py` (Flinn-Engdahl regions, nearest
  towns), `maps.py` (static SVG maps; base drawings in `/assets/maps`, markers inline), `hazards.py`
  (each hazard's event, window, schedule and official sources).

The output is a fixed point: building twice changes nothing. `python -m hazardpulse.site.build --check`
lists any page that differs from what the artifacts say, and `tests/test_site_pages.py` fails on one.

## Rules the build and the tests enforce

* No inline `style` attribute and no inline executable script: the Content-Security-Policy allows only
  this origin (`src/worker.js`, `dist/_headers`), so either would be dropped and logged on every view.
* Every asset URL carries the hash of its file (`?v=`): `/assets` is cached for a year.
* Every data file is strict JSON (no `NaN`), so the Worker and every browser can parse it.
* A forecast whose quality checks ended in `block` is withheld on its page, with the reason.
* A live skill score is quoted only once 10 or more events have been observed for that model version.
* Research diagnostics are never presented as the forecast; band words (one scorer's "watch" is an NWS
  term) are not used; a chance below 0.1% reads `<0.1%`, never `0.0%`.

## Edge personalisation (`src/worker.js`)

Pages run through the Worker (`run_worker_first` in `wrangler.toml`); assets, data files and the root text
files are served directly. Workers serve any file that matches an asset without running the Worker, so
until 2026-10-04 none of the code below had ever run on a page. `_headers` does not apply to a response the
Worker returns, so the Worker sets every header a page needs itself.

For each HTML response the Worker reads one small file the build writes for it (`/data/area-index.json`:
the earthquake grid and storm positions) and uses the visitor's approximate location (Cloudflare `request.cf`), for
that response only, to place the location marker on the maps (the projection travels on each map's
`.user-marker`), to fill "Near you" on the home page (the visitor's own grid-cell earthquake chance, the
nearest tropical cyclone and tracked thunderstorm), and to show a factual notice when an active tropical
cyclone is within 500 km or a nearby storm is under an NWS tornado warning. An earthquake number never
raises a notice. Nothing is stored.

## Geographic data

`scripts/build_site_geodata.py` builds `src/hazardpulse/site/data/` from public-domain sources: Census
Bureau places (US) and Natural Earth populated places (elsewhere), the Flinn-Engdahl regions (USGS,
1995 revision), and Natural Earth 1:110m land and US state outlines.
