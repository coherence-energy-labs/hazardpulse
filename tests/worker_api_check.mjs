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

class HTMLRewriterStub {
  on() {
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

const context = vm.createContext({
  console,
  URL,
  Headers,
  Request,
  Response,
  HTMLRewriter: HTMLRewriterStub,
  fetch: async (input) => {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
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
  storms: [{ storm_id: "AL092026", storm_name: "IRENE", basin: "AL", category: "Category 1", lat: 39.0, lon: -72.0, ri_probability: 0.2 }],
});
assert.equal(stormArea.banner.kind, "hurricane");
let bannerHtml = "";
new workerTest.EmergencyBannerHandler(stormArea).element({ setInnerContent: (h) => { bannerHtml = h; } });
assert.match(bannerHtml, /Irene \(Category 1\) is about \d+ km \(\d+ mi\) [NESW]+ of your approximate location/);
assert.match(bannerHtml, /National Hurricane Center/);
assert.doesNotMatch(bannerHtml, /style=|imminent|take action/i);
const farStorm = workerTest.summarizeArea(validGeo, {
  eq: replay, eqGate: "pass", tornadoes: [],
  storms: [{ storm_id: "EP152026", storm_name: "NOLO", basin: "EP", lat: 23.5, lon: -175.9, ri_probability: 0.9 }],
});
assert.equal(farStorm.banner, null);
assert.deepEqual(workerTest.officialCenter({ basin: "EP", lon: -175.9 })[0], "Central Pacific Hurricane Center");
const okc = workerTest.normalizeGeo({ latitude: "35.4676", longitude: "-97.5164", city: "Oklahoma City", country: "US" });
const twister = (p, active) => ({ storm_id: "1", lat: 35.5, lon: -97.6, tornado_probability: p, v3: { nws_warning: { active } } });
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.02, false)] }).banner, null);
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.02, true)] }).banner.kind, "tornado");
assert.equal(workerTest.summarizeArea(okc, { eq: null, storms: [], tornadoes: [twister(0.25, false)] }).banner.kind, "tornado");

// "Near you": empty (so hidden) without a reliable location; facts, not advice, with one
let areaHtml = "unset";
new workerTest.YourAreaHandler({ reliable: false }, invalidGeo).element({ setInnerContent: (h) => { areaHtml = h; } });
assert.equal(areaHtml, "unset");
new workerTest.YourAreaHandler(stormArea, validGeo).element({ setInnerContent: (h) => { areaHtml = h; } });
assert.match(areaHtml, /New York, US/);
assert.match(areaHtml, /chance of a magnitude 6\+ earthquake in your 2&deg; grid cell/);
assert.match(areaHtml, /<span class="chance p2">1\.2%<\/span>/);
assert.doesNotMatch(areaHtml, /style=/);

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

console.log("worker api smoke checks passed");
