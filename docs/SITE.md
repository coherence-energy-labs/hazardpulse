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
* Every asset URL carries the hash of its file (`?v=`): `/assets` is cached for a year. That includes the
  stylesheet's own font URLs, which the build rewrites so they match the preload exactly.
* Every data file is strict JSON (no `NaN`), so the Worker and every browser can parse it.
* A forecast whose quality checks ended in `block` is withheld on its page, with the reason.
* A live skill score is quoted only once 10 or more events have been observed for that model version.
* No model label, year range or claim is typed around the evidence blocks or in page code
  (`tests/test_site_claims_bound.py`). Until 2026-10-09 the check compared only the text between the markers, and
  prose beside a block contradicted it. Now: a sentence the results contradict (`evidence_pages.CONTRADICTED`) fails
  `check_pages` and `--check` wherever it appears; a `vN.N` label or a year range in a string literal of the page
  code, or in a hand-kept page outside its blocks, fails the tests unless an artifact justifies it (`ALLOWED`); and
  every model label on any page must be one an artifact names.
* Which of our hurricane RI models is shown beside NOAA's number is named in one file,
  `results/hurricane_prospective/shown_model.json` (`"shown": "<entrant>"`). Its label, artifact, version, the key of
  its live shadow forecasts and its challengers are derived from it (`served_evidence.shown_model`); switching it at a
  look (amendment 4) is that one line, and every page follows.
* Research diagnostics are never presented as the forecast; band words (one scorer's "watch" is an NWS
  term) are not used; a chance below 0.1% reads `<0.1%`, never `0.0%`.

## Edge personalisation (`src/worker.js`)

Pages run through the Worker (`run_worker_first` in `wrangler.toml`); assets, data files and the root text
files are served directly. Workers serve any file that matches an asset without running the Worker, so
until 2026-10-04 none of the code below had ever run on a page. `_headers` does not apply to a response the
Worker returns, so the Worker sets every header a page needs itself.

For each HTML response the Worker reads one small file the build writes for it (`/data/area-index.json`:
the earthquake grid, each storm's position, strength and advisory time, the tracked thunderstorms) and
uses the visitor's approximate location (Cloudflare `request.cf`), for that response only, to place the
location marker on the maps (the projection travels on each map's `.user-marker`), to centre the globe, to
fill "Near you", and to show a factual notice at the top of the page when an official NWS alert of Severe
or Extreme severity is in effect at the visitor's point, an active tropical cyclone is within 500 km, or a
nearby storm is under an NWS tornado warning. An earthquake never raises a notice. Nothing is stored.

## The live layer (home and `/live/`)

What is happening now sits beside what is likely next. The Worker reads the agencies' public feeds:

| Feed | What it gives | Where |
|---|---|---|
| USGS `2.5_day.geojson`, `4.5_week.geojson` | every M2.5+ earthquake of the last day; the week's largest | counts, feed, globe, "Near you" |
| NHC `CurrentStorms.json` | active storms in the Atlantic, East and Central Pacific; the advisory bin names the issuing centre | counts, feed, globe, "Near you" |
| NWS `alerts/active?event=Tornado Warning` | US tornado warnings in effect (polygon centres) | counts, feed, globe |
| NWS `alerts/active?point=lat,lon` | the alerts at the visitor's point, rounded to 0.1 degree; US only | "Near you", the notice |

Storms outside NOAA's basins come from the HazardPulse forecast feed, and only while their position is
under 24 hours old. Each storm's position is its best-track fix for the cycle, from UCAR RAL, or the JTWC
warning's position while that fix is not yet published (`ri_inputs.analysis_model` says which). Tracked thunderstorms are shown only while the tornado forecast that
tracked them is under 3 hours old.

The server renders everything first (`HTMLRewriter`: the counters `[data-live]`, `#live-feed`, `#near-you`,
the notice, the globe's centre), so the page is complete without JavaScript. `/assets/app.js` then draws
the globe (WebGL: a land mask, `/assets/maps/land-2048.webp`, and the 30-day earthquake forecast as a heat
layer from `/data/globe.json`, ~17 KB, under the live markers), polls `/api/v1/now` every minute and
`/api/v1/near` every five while the page is visible, and keeps the relative times current. Without WebGL the static forecast map stays.

Rules, each with a test in `tests/worker_api_check.mjs`:

* Unknown is never zero. A count whose agency did not answer is `null` in the API and a dash on the page;
  "Quiet" and "None nearby" are said only of a source that answered.
* Each feed is fetched at most once a minute per server: concurrent requests share one fetch, a failure is
  remembered for the minute, a copy past its minute is served at once and refreshed after the response,
  and a request is abandoned after 2.5 s. A last good copy is served for at most 15 minutes.
* A page waits at most 350 ms for the live data (the first request on a cold server took 1.0 s on the
  preview). Past that it is sent with dashes and the feed's placeholder, the notice still comes from the
  build's index, `app.js` fills the rest, and the fetches finish behind the response.
* Agency text is escaped after it is shortened, never before; an agency link is used only if it is https.
* A missing number in a feed is missing, not 0 (`Number(null)` is 0 in JavaScript; `isFinite(null)` is true).

## Geographic data

`scripts/build_site_geodata.py` builds `src/hazardpulse/site/data/` from public-domain sources: Census
Bureau places (US) and Natural Earth populated places (elsewhere), the Flinn-Engdahl regions (USGS,
1995 revision), and Natural Earth 1:110m land and US state outlines. Given Natural Earth's 1:50m land
(`ne_50m_land.geojson`), it also draws the globe's land mask, `dist/assets/maps/land-2048.webp`.
