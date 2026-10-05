import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workerSource = readFileSync(path.join(root, "src", "worker.js"), "utf8");
const transformedSource = workerSource.replace(
  "export default",
  "globalThis.__worker_default ="
);

// records the handlers each page response registers, so a test can read what the page would show
let lastRewriter = null;
class HTMLRewriterStub {
  constructor() {
    this.handlers = [];
    lastRewriter = this;
  }

  on(selector, handler) {
    this.handlers.push([selector, handler]);
    return this;
  }

  transform(response) {
    return response;
  }
}

function mimeTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".xml") return "application/xml; charset=utf-8";
  if (ext === ".txt") return "text/plain; charset=utf-8";
  if (ext === ".md") return "text/markdown; charset=utf-8";
  if (ext === ".svg") return "image/svg+xml";
  if (ext === ".png") return "image/png";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".js") return "application/javascript; charset=utf-8";
  return "application/octet-stream";
}

function resolveAssetPath(urlPath) {
  let normalized = urlPath;
  if (!normalized || normalized === "/") normalized = "/index.html";
  if (normalized.endsWith("/")) normalized += "index.html";
  return path.join(root, "dist", ...normalized.split("/").filter(Boolean));
}

// the agencies the live layer reads; each test sets the fixtures and the mode (ok | fail | hang)
const UPSTREAM_HOSTS = new Set(["earthquake.usgs.gov", "www.nhc.noaa.gov", "api.weather.gov"]);
const upstream = { mode: "ok", fixtures: {}, calls: [] };
// GitHub, as the Worker's cron dispatcher sees it
const githubCalls = [];
let githubStatus = 204;

const context = vm.createContext({
  console,
  URL,
  Headers,
  Request,
  Response,
  AbortController,
  setTimeout,
  clearTimeout,
  HTMLRewriter: HTMLRewriterStub,
  fetch: async (input, init = {}) => {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
    if (url.hostname === "api.github.com") {
      githubCalls.push({ url: request.url, method: init.method || request.method, auth: (init.headers || {}).Authorization,
                         body: init.body });
      return new Response(null, { status: githubStatus });
    }
    if (UPSTREAM_HOSTS.has(url.hostname)) {
      upstream.calls.push({ url: request.url, ua: request.headers.get("user-agent") });
      if (upstream.mode === "fail") return new Response("upstream error", { status: 503 });
      if (upstream.mode === "hang") {
        return new Promise((_, reject) => {
          const signal = init.signal || request.signal;
          signal.addEventListener("abort", () => reject(new Error("aborted")));
        });
      }
      const body = upstream.fixtures[request.url];
      if (!body) return new Response("not found", { status: 404 });
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/geo+json" } });
    }
    if (url.hostname !== "hazardpulse.com") {
      throw new Error(`unexpected fetch host: ${url.hostname}`);
    }
    const filePath = resolveAssetPath(url.pathname);
    if (!existsSync(filePath)) {
      return new Response("not found", { status: 404 });
    }
    return new Response(readFileSync(filePath), {
      status: 200,
      headers: { "Content-Type": mimeTypeFor(filePath) },
    });
  },
  globalThis: {},
});

vm.runInContext(transformedSource, context, { filename: "worker.js" });
const worker = context.globalThis.__worker_default;
const workerTest = context.globalThis.__hazardpulse_worker_test;
const handlerFor = (selector) => (lastRewriter.handlers.find(([s]) => s === selector) || [])[1];
assert(worker && typeof worker.fetch === "function");
assert(workerTest);

const env = {
  ASSETS: {
    async fetch(request) {
      const url = new URL(request.url);
      if (url.hostname !== "hazardpulse.com") {
        throw new Error(`unexpected asset host: ${url.hostname}`);
      }
      const filePath = resolveAssetPath(url.pathname);
      if (!existsSync(filePath)) {
        return new Response("not found", { status: 404 });
      }
      return new Response(readFileSync(filePath), {
        status: 200,
        headers: { "Content-Type": mimeTypeFor(filePath) },
      });
    },
  },
};

async function expectJsonRoute(pathname, predicate) {
  const response = await worker.fetch(new Request(`https://hazardpulse.com${pathname}`), env);
  assert.equal(response.status, 200, `${pathname} should return 200`);
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("X-Frame-Options"), "DENY");
  assert.equal(response.headers.get("X-Robots-Tag"), "noindex, nofollow");
  const json = await response.json();
  assert.equal(json.meta.version, "v1");
  predicate(json.data);
}

function assertHtmlSecurityHeaders(
  response,
  expectedStatus = 200,
  expectedCacheControl = "private, no-cache, no-store, must-revalidate"
) {
  assert.equal(response.status, expectedStatus);
  assert.match(response.headers.get("Content-Security-Policy") || "", /frame-ancestors 'none'/);
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("X-Frame-Options"), "DENY");
  assert.equal(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin");
  assert.equal(response.headers.get("Cache-Control"), expectedCacheControl);
}

const homeResponse = await worker.fetch(new Request("https://hazardpulse.com/"), env);
assertHtmlSecurityHeaders(homeResponse);
const homeCsp = homeResponse.headers.get("Content-Security-Policy") || "";
// _headers is not applied to a Worker response: the page's own headers come from the Worker
assert.equal(homeResponse.headers.get("Speculation-Rules"), '"/speculation-rules.json"');
assert.equal(homeResponse.headers.get("X-Build-Mode"), null);
assert.match(await homeResponse.text(), /HazardPulse/);

const siteShellResponse = await worker.fetch(
  new Request("https://hazardpulse.com/assets/site-shell.js"),
  env
);
assert.equal(siteShellResponse.status, 200);
assert.match(siteShellResponse.headers.get("Content-Type") || "", /javascript/);
assert.match(await siteShellResponse.text(), /data-theme/);

assert.equal(
  workerTest.hasReliableGeo({
    latitude: 0,
    longitude: 0,
    city: null,
    country: null,
    region: null,
    timezone: null,
    continent: null,
  }),
  false
);
assert.equal(
  workerTest.hasReliableGeo({
    latitude: 40.7128,
    longitude: -74.006,
    city: "New York",
    country: "US",
    region: "New York",
    timezone: "America/New_York",
    continent: "NA",
  }),
  true
);
assert.equal(
  workerTest.hasReliableGeo({
    latitude: 35.4676,
    longitude: -97.5164,
    city: null,
    country: null,
    region: null,
    timezone: null,
    continent: null,
  }),
  true
);

assert.equal(
  workerTest.readThemePreference(
    new Request("https://hazardpulse.com/", {
      headers: { cookie: "hp_theme=dark; foo=bar" },
    })
  ),
  "dark"
);

const invalidGeo = workerTest.normalizeGeo({ latitude: "0", longitude: "0" });
assert.equal(invalidGeo.isReliable, false);

const validGeo = workerTest.normalizeGeo({
  latitude: "40.7128",
  longitude: "-74.0060",
  city: "New York",
  country: "US",
  region: "New York",
  timezone: "America/New_York",
  continent: "NA",
});

function attrs(handler, initial = {}) {
  const set = new Map();
  handler.element({
    getAttribute: (name) => (name in initial ? initial[name] : null),
    setAttribute: (name, value) => set.set(name, value),
  });
  return set;
}

// the theme reaches <html> before first paint; the toggle reports its state
assert.equal(attrs(new workerTest.ThemeRootHandler("dark")).get("data-theme"), "dark");
assert.equal(attrs(new workerTest.ThemeRootHandler(null)).has("data-theme"), false);
const toggle = attrs(new workerTest.ThemeToggleHandler("dark"));
assert.equal(toggle.get("aria-pressed"), "true");
assert.equal(toggle.get("aria-label"), "Switch to light theme");
assert.equal(attrs(new workerTest.ThemeToggleHandler("light")).get("aria-pressed"), "false");

// the visitor marker uses the projection carried by the map's own marker element
const worldProj = { "data-lon0": "-180", "data-lat0": "80", "data-sx": "2.66667", "data-sy": "2.66667" };
const marker = attrs(new workerTest.UserMarkerHandler(validGeo), worldProj);
const m = (marker.get("transform") || "").match(/^translate\(([\d.]+) ([\d.]+)\)$/);
assert.ok(m, marker.get("transform"));
assert.ok(Math.abs(Number(m[1]) - (-74.006 + 180) * 2.66667) < 0.2);
assert.ok(Math.abs(Number(m[2]) - (80 - 40.7128) * 2.66667) < 0.2);
assert.match(marker.get("aria-label") || "", /Your approximate location/);
assert.equal(attrs(new workerTest.UserMarkerHandler(invalidGeo), worldProj).has("transform"), false);
assert.equal(attrs(new workerTest.UserMarkerHandler(validGeo), {}).has("transform"), false);   // no projection, no guess

// the visitor's own earthquake cell is the published grid value for that cell
const replay = {
  forecast_domain: { lat_min: -60, lon_min: -180, dlat: 2, dlon: 2, n_lat: 65, n_lon: 180 },
  probability_grid: Array.from({ length: 65 * 180 }, (_, i) => (i === 50 * 180 + 52 ? "0.0123" : "0.0001")).join(","),
};
const cell = workerTest.earthquakeCell(40.7128, -74.006, replay);
assert.equal(cell.p, 0.0123);
assert.equal(cell.lat, 41);
assert.equal(cell.lon, -75);   // the cell spanning 76W-74W
assert.equal(workerTest.earthquakeCell(75, 10, replay).outside, true);

// the banner is factual, hazard-appropriate, and never raised by an earthquake number
const quiet = workerTest.summarizeArea(validGeo, { eq: replay, eqGate: "pass", storms: [], tornadoes: [] });
assert.equal(quiet.banner, null);
const highEq = { ...replay, probability_grid: replay.probability_grid.replace("0.0123", "0.9") };
assert.equal(workerTest.summarizeArea(validGeo, { eq: highEq, eqGate: "pass", storms: [], tornadoes: [] }).banner, null);
const stormArea = workerTest.summarizeArea(validGeo, {
  eq: replay, eqGate: "pass", tornadoes: [],
  storms: [{ storm_id: "AL092026", storm_name: "IRENE", basin: "AL", category: "Category 1", lat: 39.0, lon: -72.0, ri_probability: 0.2,
             position_time: new Date(Date.now() - 3 * 3600_000).toISOString() }],
});
assert.equal(stormArea.banner.kind, "hurricane");
const bannerHtml = workerTest.bannerHtml(stormArea);
assert.match(bannerHtml, /Irene \(Hurricane, Category 1\) is about \d+ km \(\d+ mi\) [NESW]+ of your approximate location/);
assert.match(bannerHtml, /National Hurricane Center/);
assert.doesNotMatch(bannerHtml, /style=|imminent|take action/i);
const farStorm = workerTest.summarizeArea(validGeo, {
  eq: replay, eqGate: "pass", tornadoes: [],
  storms: [{ storm_id: "EP152026", storm_name: "NOLO", basin: "EP", lat: 23.5, lon: -175.9, ri_probability: 0.9,
             position_time: new Date(Date.now() - 3 * 3600_000).toISOString() }],
});
assert.equal(farStorm.banner, null);
assert.deepEqual(workerTest.officialCenter({ basin: "EP", lon: -175.9 })[0], "Central Pacific Hurricane Center");
const okc = workerTest.normalizeGeo({ latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" });
const twister = (p, active) => ({ storm_id: "1", lat: 35.5, lon: -97.6, tornado_probability: p, v3: { nws_warning: { active } } });
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.02, false)] }).banner, null);
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.02, true)] }).banner.kind, "tornado");
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.25, false)] }).banner.kind, "tornado");
// the build's area index carries the warning state as a flag
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [],
  tornadoes: [{ lat: 35.5, lon: -97.6, tornado_probability: 0.01, warned: true }] }).banner.kind, "tornado");
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [],
  tornadoes: [{ lat: 35.5, lon: -97.6, tornado_probability: 0.01, warned: false }] }).banner, null);

// the index the build writes is what the worker reads
const areaIndex = JSON.parse(readFileSync(path.join(root, "dist", "data", "area-index.json"), "utf8"));
assert.ok(areaIndex.eq && areaIndex.eq.forecast_domain && typeof areaIndex.eq.probability_grid === "string");
assert.ok(Array.isArray(areaIndex.storms) && Array.isArray(areaIndex.tornadoes));
const fromIndex = workerTest.summarizeArea(validGeo, {
  eq: areaIndex.eq, eqGate: areaIndex.eq.gate, storms: areaIndex.storms, tornadoes: areaIndex.tornadoes });
assert.ok(fromIndex.eqCell && Number.isFinite(fromIndex.eqCell.p));

// "Near you": empty (so hidden) without a reliable location; facts, not advice, with one
assert.equal(workerTest.nearHtml({ reliable: false }, invalidGeo), "");
const areaHtml = workerTest.nearHtml(stormArea, validGeo);
assert.match(areaHtml, /New York, US/);
assert.match(areaHtml, /Chance of an M6\+ in your 2&deg; cell, next 30 days: <span class="chance p2">1\.2%<\/span>/);
assert.doesNotMatch(areaHtml, /style=/);

// ---------------------------------------------------------------------------------------------
// The live layer: upstream agency feeds (fixtures), their failure modes, and what the edge renders from them
// ---------------------------------------------------------------------------------------------

const LIVE_URLS = workerTest.LIVE_SOURCES;
// objects built inside the sandbox carry its prototypes; compare their content
const plain = (x) => JSON.parse(JSON.stringify(x));
const NOW = Date.now();
const iso = (ms) => new Date(ms).toISOString();
const MIN = 60_000;
const HOUR = 3600_000;

function quakeFeature(id, mag, place, ageMs, lon, lat, depth, extra = {}) {
  return {
    id,
    properties: { mag, place, time: NOW - ageMs, url: `https://earthquake.usgs.gov/earthquakes/eventpage/${id}`, ...extra },
    geometry: { type: "Point", coordinates: [lon, lat, depth] },
  };
}

upstream.fixtures = {
  [LIVE_URLS.quakesDay]: {
    features: [
      // agency text is data: markup in it must arrive escaped, a non-https link must not arrive at all
      quakeFeature("us1", 5.1, '<script>alert(1)</script> & "Fiji"', 10 * MIN, 178.1, -17.9, 560.4,
        { url: "javascript:alert(1)", tsunami: 1 }),
      quakeFeature("nc1", 2.7, "5 km NE of Brooklyn, NY", 50 * MIN, -73.95, 40.68, 8.2),
      quakeFeature("ak1", 3.0, "40 km N of Anchorage, Alaska", 5 * HOUR, -150.0, 61.5, 30.0),
      { id: "bad", properties: { mag: null, place: "no magnitude", time: NOW }, geometry: { coordinates: [0, 0, 0] } },
    ],
  },
  [LIVE_URLS.quakesWeek]: {
    features: [
      quakeFeature("w1", 4.6, "Tonga", 2 * 24 * HOUR, -174.0, -20.0, 100),
      quakeFeature("w2", 6.3, "Kuril Islands", 3 * 24 * HOUR, 153.0, 47.0, 30),
    ],
  },
  [LIVE_URLS.storms]: {
    activeStorms: [
      { id: "ep182026", binNumber: "EP3", name: "Rachel", classification: "HU", intensity: "90", pressure: "965",
        latitudeNumeric: 20.1, longitudeNumeric: -114.3, movementDir: 270, movementSpeed: 6,
        lastUpdate: iso(NOW - 2 * HOUR), publicAdvisory: { url: "https://www.nhc.noaa.gov/text/MIATCPEP3.shtml" } },
      // on the dateline NOAA reports +180; the advisory comes from Honolulu
      { id: "ep152026", binNumber: "CP2", name: "Nolo", classification: "HU", intensity: "100", pressure: "959",
        latitudeNumeric: 24.2, longitudeNumeric: 180, movementDir: 280, movementSpeed: 17,
        lastUpdate: iso(NOW - 2 * HOUR), publicAdvisory: { url: "https://www.nhc.noaa.gov/text/HFOTCPCP2.shtml" } },
      // no position: dropped, never drawn at 0, 0 (Number(null) is 0)
      { id: "al992026", binNumber: "AT5", name: "Ghost", classification: "TD", intensity: null,
        latitudeNumeric: null, longitudeNumeric: null, lastUpdate: iso(NOW - HOUR) },
    ],
  },
  [LIVE_URLS.tornadoWarnings]: {
    features: [{
      properties: {
        id: "urn:oid:tor1", event: "Tornado Warning", headline: "Tornado Warning issued by NWS Norman OK",
        areaDesc: "Cleveland, OK; McClain, OK", severity: "Extreme", sent: iso(NOW - 15 * MIN), ends: iso(NOW + 30 * MIN),
        senderName: "NWS Norman OK",
      },
      geometry: { type: "Polygon", coordinates: [[[-97.6, 35.2], [-97.3, 35.2], [-97.3, 35.4], [-97.6, 35.4], [-97.6, 35.2]]] },
    }],
  },
  // the NWS alerts at Oklahoma City's point, rounded to 0.1 degree
  "https://api.weather.gov/alerts/active?point=35.5,-97.5": {
    features: [
      { properties: { id: "a-minor", event: "Special Weather Statement", headline: "Strong storms nearby", areaDesc: "Oklahoma, OK",
        severity: "Minor", sent: iso(NOW - 20 * MIN), senderName: "NWS Norman OK" }, geometry: null },
      { properties: { id: "a-severe", event: "Severe Thunderstorm Warning", headline: "Hail & 70 mph winds <expected>",
        areaDesc: "Oklahoma, OK", severity: "Severe", sent: iso(NOW - 5 * MIN), ends: iso(NOW + 40 * MIN),
        senderName: "NWS Norman OK" }, geometry: null },
    ],
  },
};

// the build's area index, as the edge reads it: storms where NOAA's feed does not reach, with their age
const realIndex = JSON.parse(readFileSync(path.join(root, "dist", "data", "area-index.json"), "utf8"));
const fcstId = (kind, ms) => {
  const d = new Date(ms);
  const p = (n) => String(n).padStart(2, "0");
  return `${kind}_fcst_${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}_${p(d.getUTCHours())}${p(d.getUTCMinutes())}`;
};
const fixtureIndex = {
  forecasts: { ...realIndex.forecasts, to: fcstId("to", NOW - 30 * MIN) },
  eq: realIndex.eq,
  storms: [
    { storm_id: "EP182026", storm_name: "RACHEL", basin: "EP", category: "Category 2", lat: 20.1, lon: -114.0,
      vmax_kt: 90, ri_probability: 0.07, position_time: iso(NOW - 3 * HOUR) },
    { storm_id: "AL202026", storm_name: "ZETA", basin: "AL", category: "Category 1", lat: 25.0, lon: -60.0,
      vmax_kt: 70, ri_probability: 0.2, position_time: iso(NOW - 3 * HOUR) },
    { storm_id: "WP262026", storm_name: "Choi-Wan", basin: "WP", category: "Category 4", lat: 24.4, lon: 147.1,
      vmax_kt: 135, ri_probability: 0.0125, position_time: iso(NOW - 6 * HOUR) },
    { storm_id: "WP252026", storm_name: "STALE", basin: "WP", category: "", lat: 30.0, lon: 140.0,
      vmax_kt: 40, ri_probability: 0.01, position_time: iso(NOW - 40 * HOUR) },
  ],
  tornadoes: [{ lat: 35.5, lon: -97.6, tornado_probability: 0.4, warned: true }],
};
const envFixture = {
  ASSETS: {
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname === "/data/area-index.json") {
        return new Response(JSON.stringify(fixtureIndex), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return env.ASSETS.fetch(request);
    },
  },
};
const pageRequest = new Request("https://hazardpulse.com/");
const okcGeoEarly = () => workerTest.normalizeGeo({ latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" });

function resetLive(mode = "ok") {
  upstream.mode = mode;
  upstream.calls.length = 0;
  workerTest.resetLiveState();
}

// 1. the live picture from the agencies' feeds
resetLive("ok");
const live = await workerTest.getLive(envFixture, pageRequest);
assert.deepEqual(plain(live.sources), { usgs: "ok", usgs_week: "ok", nhc: "ok", nws: "ok", forecasts: "ok" });
assert.equal(live.counts.quakes_day, 3, "the feature without a magnitude is dropped");
assert.equal(live.counts.quakes_day_m45, 1);
assert.equal(live.counts.quakes_week_m45, 2);
assert.deepEqual(plain(live.quakes.day.map((q) => q.id)), ["us1", "nc1", "ak1"], "newest first");
assert.equal(live.quakes.week_major[0].id, "w2", "largest first");
assert.deepEqual(plain(live.storms.map((s) => s.name)), ["Rachel", "Nolo", "Choi-Wan"]);
const [rachel, nolo, choi] = live.storms;
assert.equal(rachel.source, "National Hurricane Center");
assert.equal(rachel.kind, "Hurricane");
assert.equal(rachel.category, "Category 2");
assert.equal(rachel.ri_probability, 0.07, "NOAA's position, the HazardPulse forecast's chance");
assert.equal(nolo.source, "Central Pacific Hurricane Center", "the bin, not the +180 longitude, names the centre");
assert.equal(nolo.url, "https://www.nhc.noaa.gov/text/HFOTCPCP2.shtml");
assert.equal(choi.kind, "Super typhoon");
assert.equal(choi.wind_kt, 135);
assert.equal(choi.source, "Joint Typhoon Warning Center");
assert.ok(Number.isFinite(choi.updated));
assert.equal(live.counts.storms, 3, "the 40-hour-old position and the NOAA-basin index storm are not live");
assert.equal(live.counts.tornado_warnings, 1);
assert.ok(Math.abs(live.tornado_warnings[0].lat - 35.3) < 1e-9 && Math.abs(live.tornado_warnings[0].lon + 97.45) < 1e-9,
  "the polygon's centre, its closing vertex counted once");
assert.equal(live.counts.tracked_thunderstorms, 1);
for (const call of upstream.calls) assert.match(call.ua || "", /^HazardPulse\/1\.0 \(\+https:\/\/hazardpulse\.com/);

// a minute's memo: a second page view does not ask the agencies again
const callsBefore = upstream.calls.length;
await workerTest.getLive(envFixture, pageRequest);
assert.equal(upstream.calls.length, callsBefore);

// 2. what the API serves: no internals, the same fragments the page gets
const pub = workerTest.publicLive(live);
assert.equal("_index" in pub, false);
assert.equal(pub.html.stats.quakes_day, "3");
assert.equal(pub.html.stats.storms, "3");
const feed = pub.html.feed;
assert.doesNotMatch(feed, /<script/);
assert.match(feed, /&lt;script&gt;alert\(1\)&lt;\/script&gt; &amp; &quot;Fiji&quot;/);
assert.doesNotMatch(feed, /javascript:/);
assert.match(feed, /href="https:\/\/earthquake\.usgs\.gov\/" rel="noopener">&lt;script/);
assert.match(feed, /tsunami message issued/);
assert.match(feed, /^<li class="feed-item feed-eq feed-big"><span class="feed-mag">M5\.1<\/span>/, "the newest event leads");
assert.ok(feed.indexOf("Tornado Warning: Cleveland, OK") < feed.indexOf("Brooklyn"), "newest first across hazards");
assert.match(feed, /Nolo \(Hurricane, Category 3\)/);
assert.match(feed, /Choi-Wan \(Super typhoon, Category 4\)/);
assert.match(feed, /<time datetime="[^"]+" data-relative>10 min ago<\/time>/);
assert.doesNotMatch(feed, /style=/);
const emptyFeed = workerTest.feedHtml({ quakes: { day: [] }, storms: [], tornado_warnings: [] });
assert.match(emptyFeed, /^<li class="feed-empty">/);

// 3. the agencies down: unknown is unknown -- never "0 earthquakes", never "Quiet"
resetLive("fail");
const down = await workerTest.getLive(envFixture, pageRequest);
assert.deepEqual(plain(down.sources),
  { usgs: "unavailable", usgs_week: "unavailable", nhc: "unavailable", nws: "unavailable", forecasts: "ok" });
assert.equal(down.counts.quakes_day, null);
assert.equal(down.counts.tornado_warnings, null);
assert.equal(down.counts.storms, 3, "the forecast feed still counts the storms");
assert.deepEqual(plain(workerTest.statsOf(down)), { storms: "3", tracked_thunderstorms: "1" }, "the page keeps its dashes");
assert.match(workerTest.feedHtml(down),
  /Not answering right now: the USGS earthquake feed, the National Hurricane Center, the National Weather Service\./);
const downNear = workerTest.nearHtml(
  workerTest.summarizeArea(okcGeoEarly(), workerTest.areaDataOf(fixtureIndex), down, null), okcGeoEarly(), down);
assert.match(downNear, /<p class="near-figure">&mdash;<\/p><p>The USGS earthquake feed is not answering right now\.<\/p>/);
assert.doesNotMatch(downNear, /Quiet/);
// ...and a failure is remembered for its minute: a dead agency costs one timeout a minute, not one per view
const callsWhileDown = upstream.calls.length;
assert.equal(callsWhileDown, 4);
await workerTest.getLive(envFixture, pageRequest);
assert.equal(upstream.calls.length, callsWhileDown);
assert.deepEqual(plain(down.storms.map((s) => s.name).sort()), ["Choi-Wan", "Rachel", "Zeta"],
  "without NOAA's feed its basins come from the forecast feed; the stale storm still does not");
assert.equal(down.storms.find((s) => s.name === "Zeta").kind, "Hurricane");
assert.equal(down.storms.find((s) => s.name === "Zeta").source, "National Hurricane Center");

// ...and a feed that fails after answering once serves its last good copy
resetLive("ok");
const firstCopy = await workerTest.cachedJson(LIVE_URLS.quakesDay, 0);
upstream.mode = "fail";
const secondCopy = await workerTest.cachedJson(LIVE_URLS.quakesDay, 0);
assert.ok(firstCopy && secondCopy === firstCopy, "the last good copy, not nothing");
// ...but only while it can still be called "now"
workerTest.liveMemo.get(LIVE_URLS.quakesDay).okAt = Date.now() - workerTest.LIVE_MAX_AGE_MS - MIN;
assert.equal(await workerTest.cachedJson(LIVE_URLS.quakesDay, 0), null, "an old copy is reported unavailable");

// a copy past its minute is served at once and refreshed behind the response
resetLive("ok");
await workerTest.cachedJson(LIVE_URLS.storms);
const oldCopy = workerTest.liveMemo.get(LIVE_URLS.storms);
oldCopy.checkedAt -= 2 * MIN;
oldCopy.value = { activeStorms: [], marker: "old" };
const background = [];
const ctx = { waitUntil: (p) => background.push(p) };
const served = await workerTest.cachedJson(LIVE_URLS.storms, 60_000, ctx);
assert.equal(served.marker, "old", "no wait for the agency");
assert.equal(background.length, 1);
await Promise.all(background);
assert.equal(workerTest.liveMemo.get(LIVE_URLS.storms).value.activeStorms.length, 3, "refreshed after the response");

// a point's alerts are held at most LIVE_MAX_AGE_MS (the privacy page's promise): the next refresh drops it
const pointUrl = "https://api.weather.gov/alerts/active?point=35.5,-97.5";
await workerTest.cachedJson(pointUrl);
assert.ok(workerTest.liveMemo.has(pointUrl));
workerTest.liveMemo.get(pointUrl).checkedAt = Date.now() - workerTest.LIVE_MAX_AGE_MS - MIN;
await workerTest.cachedJson(LIVE_URLS.quakesWeek, 0);
assert.equal(workerTest.liveMemo.has(pointUrl), false, "an old point answer is gone from memory");

// concurrent requests on a cold server share one fetch
resetLive("ok");
await Promise.all(Array.from({ length: 8 }, () => workerTest.cachedJson(LIVE_URLS.quakesDay)));
assert.equal(upstream.calls.length, 1);

// ...and a hanging agency cannot hold a page: the request is abandoned after 2.5 s
resetLive("hang");
const t0 = Date.now();
const hung = await workerTest.cachedJson(`${LIVE_URLS.quakesDay}?hang`, 0);
const waited = Date.now() - t0;
assert.equal(hung, null);
assert.ok(waited >= 2000 && waited < 4500, `abandoned after ${waited} ms`);
resetLive("ok");

// 4. official alerts at the visitor's point: US only, the point rounded before it leaves
const okcGeo = workerTest.normalizeGeo({ latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" });
const alerts = await workerTest.nwsPointAlerts(okcGeo);
assert.deepEqual(plain(alerts.map((a) => a.severity)), ["Severe", "Minor"], "most severe first");
assert.ok(upstream.calls.some((c) => c.url === "https://api.weather.gov/alerts/active?point=35.5,-97.5"));
const tokyoGeo = workerTest.normalizeGeo({ latitude: "35.6895", longitude: "139.6917", city: "Tokyo", country: "JP" });
upstream.calls.length = 0;
assert.equal(await workerTest.nwsPointAlerts(tokyoGeo), null);
assert.equal(upstream.calls.length, 0, "no request about a visitor outside the US");

// 5. "Near you" and the banner
const okcArea = workerTest.summarizeArea(okcGeo, workerTest.areaDataOf(fixtureIndex), live, alerts);
assert.equal(okcArea.banner.kind, "alert", "an official Severe alert outranks everything");
const okcBanner = workerTest.bannerHtml(okcArea);
assert.match(okcBanner, /<strong class="emergency-title">Severe Thunderstorm Warning for your area\.<\/strong>/);
assert.match(okcBanner, /Hail &amp; 70 mph winds &lt;expected&gt;/);
assert.match(okcBanner, /issued by the NWS Norman OK/);
const okcNear = workerTest.nearHtml(okcArea, okcGeo, live);
assert.match(okcNear, /Near you &middot; Oklahoma City, US/);
assert.match(okcNear, /<div class="near-tile near-alert near-bad"><p class="near-kicker">Official alerts<\/p><p class="near-figure">2<\/p>/);
assert.match(okcNear, /Severe Thunderstorm Warning in effect until \d\d:\d\d UTC/);
assert.match(okcNear, /to the nearest tracked storm \(tracked <time datetime="[^"]+" data-relative>3\d min ago<\/time>\); its chance of a tornado within the hour: <span class="chance p5">40\.0%<\/span>\. It is under a National Weather Service tornado warning/);

// a tracking list hours old no longer says where storms are: no distance, no tornado banner
const staleTracking = { ...fixtureIndex, forecasts: { ...fixtureIndex.forecasts, to: fcstId("to", NOW - 5 * HOUR) } };
const staleArea = workerTest.summarizeArea(okcGeo, workerTest.areaDataOf(staleTracking), live, []);
assert.equal(staleArea.tornado, null);
assert.equal(staleArea.banner, null, "a warned storm from five hours ago is not announced as near");
assert.match(workerTest.nearHtml(staleArea, okcGeo, live), /Storm tracking has not updated recently/);
assert.equal(workerTest.forecastTimeOf("to_fcst_20261004_2059"), Date.UTC(2026, 9, 4, 20, 59));
assert.equal(workerTest.forecastTimeOf("nonsense"), null);
assert.doesNotMatch(okcNear, /style=|<script/);

const tokyoArea = workerTest.summarizeArea(tokyoGeo, workerTest.areaDataOf(fixtureIndex), live, null);
assert.equal(tokyoArea.banner, null, "a storm about 1,400 km away is shown, not announced");
const tokyoNear = workerTest.nearHtml(tokyoArea, tokyoGeo, live);
assert.doesNotMatch(tokyoNear, /Official alerts/, "no alerts tile where we cannot read official alerts");
assert.match(tokyoNear, /<p class="near-figure">1,4\d0 km<\/p><p>to Choi-Wan \(Super typhoon, Category 4\), SSE of you/);
assert.match(tokyoNear, /href="https:\/\/www\.metoc\.navy\.mil\/jtwc\/jtwc\.html" rel="noopener">Joint Typhoon Warning Center<\/a>/);
assert.match(tokyoNear, /No thunderstorm tracked within 300 km/);

// an earthquake never raises the banner, however close and however large
const fijiGeo = workerTest.normalizeGeo({ latitude: "-17.9", longitude: "178.1", city: "Suva", country: "FJ" });
const fijiArea = workerTest.summarizeArea(fijiGeo, workerTest.areaDataOf(fixtureIndex), live, null);
assert.equal(fijiArea.banner, null);
assert.equal(fijiArea.quake.item.id, "us1");
assert.match(workerTest.nearHtml(fijiArea, fijiGeo, live), /<p class="near-figure">M5\.1<\/p><p>0 km \(0 mi\)/);

const nobody = workerTest.summarizeArea(invalidGeo, workerTest.areaDataOf(fixtureIndex), live, null);
assert.equal(workerTest.nearHtml(nobody, invalidGeo, live), "");
assert.equal(workerTest.bannerHtml(nobody), "");

// 6. the handlers the page rewrite uses
const globeAttrs = attrs(new workerTest.GlobeHandler(validGeo));
assert.equal(globeAttrs.get("data-user-lat"), "40.7");
assert.equal(globeAttrs.get("data-user-lon"), "-74.0");
assert.equal(attrs(new workerTest.GlobeHandler(invalidGeo)).size, 0);
function content(handler, attributes = {}) {
  let out = null;
  handler.element({
    getAttribute: (n) => attributes[n] ?? null,
    setInnerContent: (h, opts) => { out = { h, html: Boolean(opts && opts.html) }; },
  });
  return out;
}
assert.deepEqual(content(new workerTest.LiveStatHandler(pub.html.stats), { "data-live": "quakes_day" }), { h: "3", html: false });
assert.equal(content(new workerTest.LiveStatHandler(pub.html.stats), { "data-live": "nonsense" }), null);
assert.deepEqual(content(new workerTest.HtmlFillHandler("<li>x</li>")), { h: "<li>x</li>", html: true });
assert.equal(content(new workerTest.HtmlFillHandler("")), null, "nothing to say leaves the element alone");

// 7. the routes
const nowResponse = await worker.fetch(new Request("https://hazardpulse.com/api/v1/now"), envFixture);
assert.equal(nowResponse.status, 200);
assert.equal(nowResponse.headers.get("Cache-Control"), "public, max-age=60");
const nowJson = await nowResponse.json();
assert.equal(nowJson.data.counts.quakes_day, 3);
assert.equal("_index" in nowJson.data, false);
assert.equal(typeof nowJson.data.html.feed, "string");

function withCf(url, cf) {
  const r = new Request(url);
  Object.defineProperty(r, "cf", { value: cf });
  return r;
}
const nearResponse = await worker.fetch(
  withCf("https://hazardpulse.com/api/v1/near", { latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" }),
  envFixture);
assert.equal(nearResponse.status, 200);
assert.equal(nearResponse.headers.get("Cache-Control"), "private, no-store", "a visitor's own area is never cached");
const nearJson = await nearResponse.json();
assert.equal(nearJson.data.reliable, true);
assert.match(nearJson.data.html.near, /Oklahoma City/);
assert.match(nearJson.data.html.banner, /Severe Thunderstorm Warning for your area/);
assert.equal("lat" in nearJson.data || "latitude" in nearJson.data, false, "the response does not echo the location");
const nearNobody = await (await worker.fetch(new Request("https://hazardpulse.com/api/v1/near"), envFixture)).json();
assert.deepEqual(nearNobody.data, { reliable: false, html: { near: "", banner: "" } });

// a cold server does not hold the page for the agencies: past the budget the page goes without the live
// data (app.js fills it), the notice still stands from the build's index, and the fetches carry on behind
resetLive("hang");
const background2 = [];
const coldCtx = { waitUntil: (p) => background2.push(p) };
const nearChoiWan = { latitude: "25.0", longitude: "148.0", city: "Chichijima", country: "JP" };
const t1 = Date.now();
const coldPage = await worker.fetch(withCf("https://hazardpulse.com/", nearChoiWan), envFixture, coldCtx);
const coldMs = Date.now() - t1;
assert.equal(coldPage.status, 200);
assert.ok(coldMs < 1500, `the page waited ${coldMs} ms for the agencies`);
assert.equal(handlerFor("#near-you").html, "", "Near you is left for app.js");
assert.equal(handlerFor("#live-feed").html, "", "the feed keeps its placeholder");
assert.equal(Object.keys(handlerFor("[data-live]").stats).length, 0, "the counters keep their dashes");
assert.match(handlerFor(".emergency-banner").html, /Choi-Wan \(Super typhoon, Category 4\) is about \d+ km/,
  "the notice still stands, from the build's index");
assert.ok(background2.length >= 1, "the fetches carry on behind the response");
// a slow page says where its time went
assert.match(coldPage.headers.get("Server-Timing") || "", /^live;dur=\d+;desc="past the budget, filled by app\.js"$/);
await Promise.all(background2);
// ...and a warm server renders all of it into the page
resetLive("ok");
await workerTest.getLive(envFixture, pageRequest);
const warmPage = await worker.fetch(withCf("https://hazardpulse.com/",
  { latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" }), envFixture, coldCtx);
assert.match(warmPage.headers.get("Server-Timing") || "", /^live;dur=\d+;desc="live data in the page"$/);
assert.match(handlerFor("#near-you").html, /Near you &middot; Oklahoma City, US/);
assert.equal(handlerFor("[data-live]").stats.quakes_day, "3");
assert.match(handlerFor("#live-feed").html, /^<li class="feed-item/);
assert.match(handlerFor(".emergency-banner").html, /Severe Thunderstorm Warning for your area/);

// the pages still render with every agency down
resetLive("fail");
const downHome = await worker.fetch(new Request("https://hazardpulse.com/"), envFixture);
assertHtmlSecurityHeaders(downHome);
resetLive("ok");

// 8. the small rules
// a storm whose stored chance rounded to 0 is "<0.1%", never "0%"
const roundedZero = workerTest.summarizeArea(okcGeo, { eq: null, storms: [], known: true,
  tornadoes: [{ lat: 35.5, lon: -97.6, tornado_probability: 0, warned: false }] }, live, []);
const roundedZeroNear = workerTest.nearHtml(roundedZero, okcGeo, live);
assert.match(roundedZeroNear, /its chance of a tornado within the hour: <span class="chance p1">&lt;0\.1%<\/span>/);
assert.doesNotMatch(roundedZeroNear, />0%</);
assert.equal(workerTest.freshForecastStorm({ position_time: iso(NOW - 2 * HOUR) }), true);
assert.equal(workerTest.freshForecastStorm({ position_time: iso(NOW - 25 * HOUR) }), false);
assert.equal(workerTest.freshForecastStorm({ position_time: iso(NOW + 3 * HOUR) }), false, "a position from the future is wrong");
assert.equal(workerTest.freshForecastStorm({}), false, "no time, no claim that it is current");
assert.equal(workerTest.safeUrl("https://www.nhc.noaa.gov/x"), "https://www.nhc.noaa.gov/x");
assert.equal(workerTest.safeUrl("javascript:alert(1)"), "https://www.weather.gov/");
assert.equal(workerTest.safeUrl("http://example.com/"), "https://www.weather.gov/");
assert.equal(workerTest.safeUrl('https://a.b/"onmouseover=x'), "https://www.weather.gov/");
assert.equal(workerTest.clipText("a & b", 160), "a &amp; b");
assert.equal(workerTest.clipText("x".repeat(10) + "&", 11), "x".repeat(10) + "&amp;", "short enough: whole");
assert.equal(workerTest.clipText("abc & defghij", 6), "abc &amp;\u2026", "cut before escaping, never inside an entity");
assert.equal(workerTest.officialCenter({ basin: "EP", lon: 180 })[0], "Central Pacific Hurricane Center");
assert.equal(workerTest.officialCenter({ basin: "EP", lon: 178.5 })[0], "Joint Typhoon Warning Center");
assert.equal(workerTest.officialCenter({ basin: "EP", lon: -120 })[0], "National Hurricane Center");
assert.equal(workerTest.officialCenter({ basin: "CP", lon: null })[0], "Central Pacific Hurricane Center");
assert.equal(workerTest.stormOfForecast({ storm_id: "SH052027", storm_name: "ALFRED", basin: "SH", vmax_kt: 80 }).kind, "Tropical cyclone");
assert.equal(workerTest.stormOfForecast({ storm_id: "WP012027", storm_name: "X", basin: "WP", vmax_kt: 70 }).kind, "Typhoon");
assert.equal(workerTest.stormOfForecast({ storm_id: "WP012027", storm_name: "X", basin: "WP", vmax_kt: null, category: "Category 2" }).kind, "Typhoon");
assert.equal(workerTest.stormOfForecast({ storm_id: "AL012027", storm_name: "X", basin: "AL", vmax_kt: 40 }).kind, "Tropical storm");

await expectJsonRoute("/api/v1/live/pulse", (data) => {
  assert.ok(Array.isArray(data.hazards));
});

await expectJsonRoute("/api/v1/live/hurricane", (data) => {
  assert.ok("n_active_storms" in data);
});

await expectJsonRoute("/api/v1/live/tornado", (data) => {
  assert.ok(Array.isArray(data.storms));
});

await expectJsonRoute("/api/v1/live/earthquake", (data) => {
  assert.ok(data.summary);
});

await expectJsonRoute("/api/v1/forecast/eq_fcst_20260402_0000", (data) => {
  assert.equal(data.forecast_id, "eq_fcst_20260402_0000");
});

await expectJsonRoute("/api/v1/verification/summary", (data) => {
  assert.ok(Array.isArray(data.hazards));
});

await expectJsonRoute("/api/v1/registry/models", (data) => {
  assert.ok(Array.isArray(data.models));
});

const provenancePayload = JSON.parse(
  readFileSync(path.join(root, "dist", "data", "evidence", "provenance-envelopes.json"), "utf8")
);
if (Array.isArray(provenancePayload.envelopes) && provenancePayload.envelopes.length > 0) {
  const firstEnvelope = provenancePayload.envelopes[0];
  await expectJsonRoute(`/api/v1/evidence/${firstEnvelope.provenance_id}`, (data) => {
    assert.equal(data.provenance_id, firstEnvelope.provenance_id);
  });
}

const gatePayload = JSON.parse(
  readFileSync(path.join(root, "dist", "data", "evidence", "gate-decisions.json"), "utf8")
);
if (Array.isArray(gatePayload.decisions) && gatePayload.decisions.length > 0) {
  const firstDecision = gatePayload.decisions[0];
  await expectJsonRoute(`/api/v1/gates/${firstDecision.gate_decision_id}`, (data) => {
    assert.equal(data.gate_decision_id, firstDecision.gate_decision_id);
  });
}

const sseResponse = await worker.fetch(
  new Request("https://hazardpulse.com/stream/live/pulse"),
  env
);
assert.equal(sseResponse.status, 200);
assert.match(sseResponse.headers.get("Content-Type") || "", /text\/event-stream/);
assert.equal(sseResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");
assert.match(await sseResponse.text(), /event: live_pulse/);

const redirectResponse = await worker.fetch(
  new Request("https://hazardpulse.com/commercial-license/"),
  env
);
assert.equal(redirectResponse.status, 302);
assert.match(redirectResponse.headers.get("Location") || "", /COMMERCIAL_LICENSE\.md$/);
assert.equal(redirectResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");

const missingResponse = await worker.fetch(
  new Request("https://hazardpulse.com/does-not-exist"),
  env
);
assertHtmlSecurityHeaders(missingResponse, 404, "no-store");
assert.equal(missingResponse.headers.get("X-Robots-Tag"), "noindex, nofollow");
assert.match(await missingResponse.text(), /Page not found/);

// the live status is computed at request time from the build's index, never a constant "ok"
const statusIndex = { hazards: [
  { key: "eq", forecast_id: "eq_fcst_1", issued_at: "2026-10-04T06:00:00Z", quality_checks: "pass", overdue_after_hours: 12 },
  { key: "to", forecast_id: "to_fcst_1", issued_at: "2026-10-04T07:00:00Z", quality_checks: "degrade", overdue_after_hours: 6 },
] };
const fresh = workerTest.opsSnapshot(statusIndex, new Date("2026-10-04T09:00:00Z"));
assert.equal(fresh.status, "published_with_warnings");
assert.deepEqual(fresh.hazards.map((h) => h.overdue), [false, false]);
assert.equal(fresh.hazards[1].age_hours, 2);
const late = workerTest.opsSnapshot(statusIndex, new Date("2026-10-04T14:00:00Z"));
assert.equal(late.status, "delayed");
assert.deepEqual(late.hazards.map((h) => h.overdue), [false, true]);
assert.equal(workerTest.opsSnapshot(null).status, "unknown");
await expectJsonRoute("/api/v1/ops/status", (data) => {
  assert.ok(Array.isArray(data.hazards) && data.hazards.length === 3);
  assert.ok(["ok", "published_with_warnings", "delayed"].includes(data.status));
});

// a live page viewed after its forecast's window says so; within the window it says nothing
function ageNote(now) {
  let html = "";
  new workerTest.ForecastAgeHandler(new Date(now)).element({
    getAttribute: (n) => ({ "data-issued": "2026-10-04T07:43:44Z", "data-window-minutes": "60",
                            "data-schedule": "every 2 hours" })[n] ?? null,
    setInnerContent: (h) => { html = h; },
  });
  return html;
}
assert.equal(ageNote("2026-10-04T08:30:00Z"), "");
assert.match(ageNote("2026-10-04T11:00:00Z"), /issued 3 hours ago, so its 60-minute window has passed/);
assert.match(ageNote("2026-10-04T08:50:00Z"), /issued 66 minutes ago/);

// the API documentation page is a page, not an API route (it 404ed once pages ran through the Worker)
const apiDocs = await worker.fetch(new Request("https://hazardpulse.com/api/"), env);
assertHtmlSecurityHeaders(apiDocs);
assert.match(await apiDocs.text(), /Every forecast, as data/);

const unknownApiResponse = await worker.fetch(
  new Request("https://hazardpulse.com/api/v1/unknown"),
  env
);
assert.equal(unknownApiResponse.status, 404);
const unknownApiJson = await unknownApiResponse.json();
assert.equal(unknownApiJson.error.code, "not_found");
assert.ok(unknownApiJson.error.trace_id);

// the API's rate limit answers in the same envelope as every other error
let limited = null;
for (let i = 0; i < 130 && !limited; i += 1) {
  const r = await worker.fetch(new Request("https://hazardpulse.com/api/v1/ops/status",
    { headers: { "CF-Connecting-IP": "203.0.113.9" } }), env);
  if (r.status === 429) limited = r;
}
assert.ok(limited, "no 429 after 130 requests from one address");
const limitedJson = await limited.json();
assert.equal(limitedJson.error.code, "rate_limited");
assert.ok(limitedJson.error.trace_id);

// the HTML CSP allows nothing from another origin
assert.doesNotMatch(homeCsp, /https?:\/\//);
assert.doesNotMatch(homeCsp, /unsafe-inline/);

// the published contracts list IS the Worker's routes: every public route is listed and every listed path
// answers (four routes had gone unlisted before 2026-10-05)
const contracts = JSON.parse(readFileSync(path.join(root, "dist", "data", "api-contracts.json"), "utf8"));
const listed = new Set(contracts.endpoints.map((e) => e.path));
const NOT_PUBLIC = { "/api/v1/federation/atlas": "operator endpoint: answers only where a federation node is configured" };
const exactRoutes = [...workerSource.matchAll(/path === "(\/(?:api\/v1|stream)\/[^"]+)"/g)].map((m) => m[1]);
assert.ok(exactRoutes.length >= 12, `found only ${exactRoutes.length} routes: the route pattern no longer matches the Worker`);
for (const r of exactRoutes) {
  if (!(r in NOT_PUBLIC)) assert.ok(listed.has(r), `route ${r} is not in api-contracts.json`);
}
for (const t of ["/api/v1/forecast/{forecast_id}", "/api/v1/replay/{forecast_id}", "/api/v1/evidence/{provenance_id}",
  "/api/v1/gates/{gate_decision_id}", "/api/v1/laic/{hazard}", "/api/v1/live/earthquake", "/api/v1/live/hurricane",
  "/api/v1/live/tornado"]) {
  assert.ok(listed.has(t), `${t} is not in api-contracts.json`);
}
resetLive("ok");
for (const p of listed) {
  if (p.includes("{")) continue;
  const r = await worker.fetch(new Request(`https://hazardpulse.com${p}`, { headers: { "CF-Connecting-IP": "198.51.100.7" } }), envFixture);
  assert.notEqual(r.status, 404, `${p} is listed but does not answer`);
}
const apiPage = readFileSync(path.join(root, "dist", "api", "index.html"), "utf8");
for (const p of ["/api/v1/now", "/api/v1/near"]) assert.ok(apiPage.includes(`<code>${p}</code>`), `${p} is not documented`);

// the scorers' clock: Cloudflare's cron asks GitHub to run scheduler.yml -- only with a token, exactly once
{
  githubCalls.length = 0;
  const pending = [];
  await worker.scheduled({ cron: "*/10 * * * *" }, {}, { waitUntil: (p) => pending.push(p) });
  await Promise.all(pending);
  assert.equal(githubCalls.length, 0, "no token, no request");
  pending.length = 0;
  await worker.scheduled({ cron: "*/10 * * * *" }, { GH_DISPATCH_TOKEN: "t0ken" }, { waitUntil: (p) => pending.push(p) });
  await Promise.all(pending);
  assert.equal(githubCalls.length, 1);
  assert.equal(githubCalls[0].url, workerTest.SCHEDULER_DISPATCH);
  assert.equal(githubCalls[0].method, "POST");
  assert.equal(githubCalls[0].auth, "Bearer t0ken");
  assert.deepEqual(JSON.parse(githubCalls[0].body), { ref: "main" });
  githubStatus = 401;                                   // a refused token is reported, never thrown
  const refused = await workerTest.dispatchScheduler({ GH_DISPATCH_TOKEN: "bad" });
  assert.equal(refused.dispatched, false);
  assert.equal(refused.status, 401);
  githubStatus = 204;
}
const wranglerToml = readFileSync(path.join(root, "wrangler.toml"), "utf8");
assert.match(wranglerToml, /\[triggers\]\s*crons = \["\*\/10 \* \* \* \*"\]/);
assert.match(wranglerToml, /\[env\.preview\][^[]*triggers = \{ crons = \[\] \}/, "the preview must not dispatch too");

console.log("worker api smoke checks passed");
