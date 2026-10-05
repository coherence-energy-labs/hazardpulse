/**
 * HazardPulse Cloudflare Worker
 *
 * - Serves static assets from dist/
 * - Exposes public beta JSON/SSE API routes backed by dist/data/*
 * - Applies edge geolocation personalization to HTML responses
 */

const API_VERSION = "v1";
const PRIMARY_DOMAIN = "https://hazardpulse.com";
const INTERNAL_ASSET_HEADER = "x-hazardpulse-internal-asset";
const THEME_COOKIE_NAME = "hp_theme";
const HTML_CACHE_CONTROL = "private, no-cache, no-store, must-revalidate";
const ALLOWED_ORIGINS = new Set([
  "https://hazardpulse.com",
  "https://www.hazardpulse.com",
  "https://hazardpulse-preview.workers.dev",
]);
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX_REQUESTS = 120;
const _rateLimitMap = new Map();
let _rateLimitLastPrune = 0;

// Everything the site loads comes from this origin: self-hosted fonts, no map tiles, no inline style or script.
const HTML_CONTENT_SECURITY_POLICY =
  "default-src 'self'; " +
  "script-src 'self'; " +
  "style-src 'self'; " +
  "img-src 'self' data:; " +
  "font-src 'self'; " +
  "connect-src 'self'; " +
  "frame-ancestors 'none'; base-uri 'self'; form-action 'self' mailto:; object-src 'none'; upgrade-insecure-requests";

function isFiniteCoordinate(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function hasReliableGeo(geo) {
  if (!geo) return false;
  if (!isFiniteCoordinate(geo.latitude) || !isFiniteCoordinate(geo.longitude)) {
    return false;
  }
  if (geo.latitude < -90 || geo.latitude > 90) return false;
  if (geo.longitude < -180 || geo.longitude > 180) return false;
  if (Math.abs(geo.latitude) < 0.25 && Math.abs(geo.longitude) < 0.25) {
    return false;
  }
  return true;
}

function normalizeGeo(cf = {}) {
  const latitude =
    cf.latitude !== undefined && cf.latitude !== null ? Number(cf.latitude) : null;
  const longitude =
    cf.longitude !== undefined && cf.longitude !== null ? Number(cf.longitude) : null;

  const geo = {
    latitude: isFiniteCoordinate(latitude) ? latitude : null,
    longitude: isFiniteCoordinate(longitude) ? longitude : null,
    city: cf.city || null,
    country: cf.country || null,
    region: cf.region || null,
    continent: cf.continent || null,
    timezone: cf.timezone || null,
  };
  geo.isReliable = hasReliableGeo(geo);
  return geo;
}

function readThemePreference(request) {
  const cookieHeader = request.headers.get("cookie") || "";
  const match = cookieHeader.match(
    new RegExp(`(?:^|;\\s*)${THEME_COOKIE_NAME}=(dark|light)(?:;|$)`, "i")
  );
  return match ? match[1].toLowerCase() : null;
}

function withSecurityHeaders(
  response,
  { cacheControl, xRobotsTag, contentSecurityPolicy, vary } = {}
) {
  const secured = new Response(response.body, response);
  secured.headers.set(
    "Strict-Transport-Security",
    "max-age=63072000; includeSubDomains; preload"
  );
  secured.headers.set(
    "Content-Security-Policy",
    contentSecurityPolicy || HTML_CONTENT_SECURITY_POLICY
  );
  secured.headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  secured.headers.set("Cross-Origin-Opener-Policy", "same-origin");
  secured.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), interest-cohort=(), payment=(), usb=(), bluetooth=(), serial=()"
  );
  secured.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  secured.headers.set("X-Content-Type-Options", "nosniff");
  secured.headers.set("X-Frame-Options", "DENY");
  // _headers is not applied to a response the Worker returns, so every header a page needs is set here
  const type = secured.headers.get("Content-Type") || "";
  if (type.includes("text/html")) {
    secured.headers.set("Speculation-Rules", '"/speculation-rules.json"');
  }
  if (cacheControl) secured.headers.set("Cache-Control", cacheControl);
  if (cacheControl && cacheControl.includes("no-store")) {
    secured.headers.set("CDN-Cache-Control", "no-store");
    secured.headers.set("Cloudflare-CDN-Cache-Control", "no-store");
    secured.headers.set("Surrogate-Control", "no-store");
  }
  if (xRobotsTag) secured.headers.set("X-Robots-Tag", xRobotsTag);
  if (vary) secured.headers.set("Vary", vary);
  return secured;
}

function isLikelyDocumentRequest(request, pathname) {
  if (pathname === "/") return true;
  if (pathname.endsWith("/")) return true;
  const lastSegment = pathname.split("/").pop() || "";
  if (!lastSegment.includes(".")) return true;
  const accept = request.headers.get("accept") || "";
  return accept.includes("text/html");
}

async function notFoundResponse(env, request) {
  if (isLikelyDocumentRequest(request, new URL(request.url).pathname)) {
    const notFoundAsset = await fetchAsset(env, "/404.html");
    if (notFoundAsset.ok) {
      return withSecurityHeaders(
        new Response(notFoundAsset.body, {
          status: 404,
          headers: notFoundAsset.headers,
        }),
        {
          cacheControl: "no-store",
          xRobotsTag: "noindex, nofollow",
          contentSecurityPolicy: HTML_CONTENT_SECURITY_POLICY,
        }
      );
    }
  }

  return withSecurityHeaders(
    new Response("Not found", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    }),
    {
      cacheControl: "no-store",
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// a link that arrived in an agency's feed: only an https URL is used (escaping alone would still let a
// javascript: URL through into an href)
function safeUrl(value, fallback = "https://www.weather.gov/") {
  const s = String(value || "").trim();
  return /^https:\/\/[^\s"'<>]+$/i.test(s) ? s : fallback;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const radiusKm = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return radiusKm * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function _corsOrigin(request) {
  const origin = request && request.headers && request.headers.get("Origin");
  if (origin && ALLOWED_ORIGINS.has(origin)) return origin;
  return "https://hazardpulse.com";
}

function _checkRateLimit(request) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  const now = Date.now();
  let entry = _rateLimitMap.get(ip);
  if (!entry || now - entry.start > RATE_LIMIT_WINDOW_MS) {
    entry = { start: now, count: 0 };
    _rateLimitMap.set(ip, entry);
  }
  entry.count++;
  // forget every address whose minute is over: a count lives no longer than it is needed
  if (now - _rateLimitLastPrune > RATE_LIMIT_WINDOW_MS) {
    for (const [key, value] of _rateLimitMap) {
      if (now - value.start > RATE_LIMIT_WINDOW_MS) _rateLimitMap.delete(key);
    }
    _rateLimitLastPrune = now;
  }
  if (_rateLimitMap.size > 10000) _rateLimitMap.clear();
  return entry.count <= RATE_LIMIT_MAX_REQUESTS;
}

function jsonResponse(body, status = 200, cacheControl = "public, max-age=300", request = null) {
  return withSecurityHeaders(
    new Response(JSON.stringify(body, null, 2), {
      status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": cacheControl,
        "Access-Control-Allow-Origin": _corsOrigin(request),
        "Vary": "Origin",
      },
    }),
    {
      cacheControl,
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function sseResponse(eventName, payload, request = null) {
  const body = `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
  return withSecurityHeaders(
    new Response(body, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": _corsOrigin(request),
        "Vary": "Origin",
      },
    }),
    {
      cacheControl: "no-cache",
      xRobotsTag: "noindex, nofollow",
    }
  );
}

function apiEnvelope(data, cacheTtl, extraMeta = {}) {
  return {
    meta: {
      version: API_VERSION,
      generated_at: new Date().toISOString(),
      cache_ttl: cacheTtl,
      ...extraMeta,
    },
    data,
  };
}

function buildTraceId() {
  try {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return `hp_${globalThis.crypto.randomUUID()}`;
    }
  } catch {}
  return `hp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function errorEnvelope(code, message, status = 404) {
  return jsonResponse(
    {
      meta: {
        version: API_VERSION,
        generated_at: new Date().toISOString(),
      },
      error: { code, message, trace_id: buildTraceId() },
    },
    status,
    "no-cache"
  );
}

function assetRequest(pathname, baseRequest) {
  const baseUrl = baseRequest ? new URL(baseRequest.url) : new URL(PRIMARY_DOMAIN);
  const headers = new Headers(baseRequest ? baseRequest.headers : undefined);
  headers.set(INTERNAL_ASSET_HEADER, "1");
  headers.delete("if-none-match");
  headers.delete("if-modified-since");
  return new Request(new URL(pathname, baseUrl).toString(), {
    method: "GET",
    headers,
  });
}

async function fetchAsset(env, pathname, baseRequest) {
  const request = assetRequest(pathname, baseRequest);
  try {
    const assetResponse = await env.ASSETS.fetch(request);
    if (assetResponse.ok) return assetResponse;
  } catch {}

  try {
    return await fetch(request);
  } catch {
    return new Response(null, { status: 404 });
  }
}

async function fetchAssetJson(env, pathname, baseRequest) {
  const response = await fetchAsset(env, pathname, baseRequest);
  if (!response.ok) return null;
  try {
    const text = await response.text();
    return JSON.parse(text.replace(/^\uFEFF/, ""));
  } catch {
    return null;
  }
}

async function fetchAssetText(env, pathname, baseRequest) {
  const response = await fetchAsset(env, pathname, baseRequest);
  if (!response.ok) return null;
  return response.text();
}

// ---------------------------------------------------------------------------------------------
// Live feeds: what is happening now, from the agencies that observe it. Earthquakes from the USGS (the
// feed updates every minute), active tropical cyclones from the National Hurricane Center, tornado
// warnings from the National Weather Service.
//
// Each feed is checked at most once a minute per server, however many visitors arrive: concurrent requests
// share one fetch, a failure is remembered for the minute too (a dead agency costs one timeout a minute,
// not one per page view), and a copy past its minute is served at once while the next one is fetched behind
// the response. A failed check keeps the last good copy -- but only while it is recent enough to call
// "now"; older than that, the feed is reported unavailable rather than quietly stale.
// ---------------------------------------------------------------------------------------------

const LIVE_UA = "HazardPulse/1.0 (+https://hazardpulse.com; josh@coherenceenergylabs.com)";
const LIVE_TTL_MS = 60_000;
const LIVE_TIMEOUT_MS = 2500; // a slow agency must never stall a page for longer
const LIVE_MAX_AGE_MS = 15 * 60_000; // the oldest copy still shown as live
const LIVE_SOURCES = {
  quakesDay: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
  quakesWeek: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson",
  storms: "https://www.nhc.noaa.gov/CurrentStorms.json",
  tornadoWarnings: "https://api.weather.gov/alerts/active?event=Tornado%20Warning",
};
const _liveMemo = new Map(); // url -> { value, okAt, checkedAt }
const _liveInflight = new Map(); // url -> the one fetch in flight

async function fetchJsonOnce(url) {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), LIVE_TIMEOUT_MS);
  try {
    const res = await fetch(
      new Request(url, { headers: { "User-Agent": LIVE_UA, Accept: "application/geo+json, application/json" } }),
      { signal: abort.signal, cf: { cacheTtl: LIVE_TTL_MS / 1000, cacheEverything: true } }
    );
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function refreshJson(url) {
  let pending = _liveInflight.get(url);
  if (!pending) {
    pending = fetchJsonOnce(url).then((value) => {
      const now = Date.now();
      const prev = _liveMemo.get(url);
      // nothing outlives its usefulness: an answer not checked for LIVE_MAX_AGE_MS is dropped (the privacy
      // page says a point's alerts are held at most that long), and the per-point keys cannot grow unbounded
      for (const [key, e] of _liveMemo) if (now - e.checkedAt > LIVE_MAX_AGE_MS) _liveMemo.delete(key);
      if (_liveMemo.size > 400) _liveMemo.clear();
      _liveMemo.set(url, value !== null
        ? { value, okAt: now, checkedAt: now }
        : { value: prev ? prev.value : null, okAt: prev ? prev.okAt : 0, checkedAt: now });
    }).finally(() => _liveInflight.delete(url));
    _liveInflight.set(url, pending);
  }
  return pending;
}

function usableCopy(entry, now) {
  return entry && entry.value !== null && now - entry.okAt <= LIVE_MAX_AGE_MS ? entry.value : null;
}

// ctx (the request's ExecutionContext) lets a past-its-minute copy be served at once and refreshed after
// the response; without it the refresh is awaited
async function cachedJson(url, ttlMs = LIVE_TTL_MS, ctx = null) {
  const now = Date.now();
  const entry = _liveMemo.get(url);
  if (entry && now - entry.checkedAt < ttlMs) return usableCopy(entry, now);
  const copy = usableCopy(entry, now);
  if (copy !== null && ctx && typeof ctx.waitUntil === "function") {
    ctx.waitUntil(refreshJson(url));
    return copy;
  }
  await refreshJson(url);
  return usableCopy(_liveMemo.get(url), Date.now());
}

// a feed's number: missing is NaN, never 0 (Number(null) is 0 -- an event with no magnitude would pass as
// an M0.0, one with no time as 1970)
function feedNumber(v) {
  if (v === null || v === undefined || (typeof v === "string" && !v.trim())) return NaN;
  return Number(v);
}

function quakeOf(f) {
  const p = (f && f.properties) || {};
  const c = (f && f.geometry && f.geometry.coordinates) || [];
  return {
    id: String(f.id || ""),
    mag: feedNumber(p.mag),
    place: String(p.place || "Unknown location"),
    time: feedNumber(p.time),
    lon: feedNumber(c[0]),
    lat: feedNumber(c[1]),
    depth_km: feedNumber(c[2]),
    url: String(p.url || ""),
    tsunami: Boolean(p.tsunami),
    alert: p.alert || null,
  };
}

function validQuake(q) {
  return [q.mag, q.time, q.lat, q.lon].every(Number.isFinite);
}

const STORM_KIND = {
  HU: "Hurricane", TY: "Typhoon", STY: "Super typhoon", TS: "Tropical storm", TD: "Tropical depression",
  STS: "Subtropical storm", SD: "Subtropical depression", PTC: "Potential tropical cyclone", PC: "Post-tropical cyclone",
};

function categoryOf(kt) {
  const v = feedNumber(kt);
  if (!Number.isFinite(v)) return "";
  if (v >= 137) return "Category 5";
  if (v >= 113) return "Category 4";
  if (v >= 96) return "Category 3";
  if (v >= 83) return "Category 2";
  if (v >= 64) return "Category 1";
  return "";
}

function stormOfNhc(s, forecastById) {
  const id = String(s.id || "").toUpperCase();
  const f = forecastById.get(id) || {};
  const kind = STORM_KIND[String(s.classification || "").toUpperCase()] || "Tropical cyclone";
  return {
    id,
    name: stormName({ storm_name: s.name, storm_id: id }),
    kind,
    category: categoryOf(s.intensity),
    wind_kt: feedNumber(s.intensity),
    pressure_mb: feedNumber(s.pressure),
    lat: feedNumber(s.latitudeNumeric),
    lon: feedNumber(s.longitudeNumeric),
    basin: id.slice(0, 2),
    moving: Number.isFinite(feedNumber(s.movementDir)) ? { dir_deg: feedNumber(s.movementDir), speed_mph: feedNumber(s.movementSpeed) } : null,
    updated: s.lastUpdate ? Date.parse(s.lastUpdate) : null,
    url: (s.publicAdvisory && s.publicAdvisory.url) || "https://www.nhc.noaa.gov/",
    // the advisory bin names the issuing centre (CP1-CP5: Honolulu). Longitude cannot: a storm at the
    // dateline is reported at +180, east of every threshold.
    source: /^CP/.test(String(s.binNumber || "").toUpperCase()) || id.slice(0, 2) === "CP"
      ? "Central Pacific Hurricane Center"
      : "National Hurricane Center",
    ri_probability: Number.isFinite(feedNumber(f.ri_probability)) ? feedNumber(f.ri_probability) : null,
  };
}

// a storm from the HazardPulse forecast feed is shown as live only while its position is this fresh: the
// feed runs every 6 hours, so a position older than a day means the storm has dissipated or the feed stalled
const FORECAST_STORM_MAX_AGE_MS = 24 * 3600_000;

// each basin's own terms: hurricanes in NOAA's basins, typhoons in the West Pacific (JTWC), and a tropical
// cyclone everywhere else JTWC covers. Without a wind speed, a Saffir-Simpson category still says the storm
// is at hurricane strength.
function forecastStormKind(basin, kt, category = "") {
  const known = Number.isFinite(kt);
  const strong = known ? kt >= 64 : /^Category [1-5]$/.test(String(category));
  const weaker = known ? (kt >= 34 ? "Tropical storm" : "Tropical depression") : "Tropical cyclone";
  if (basin === "AL" || basin === "EP" || basin === "CP") return strong ? "Hurricane" : weaker;
  if (basin === "WP") {
    if (known && kt >= 130) return "Super typhoon";
    return strong ? "Typhoon" : weaker;
  }
  return "Tropical cyclone";
}

function freshForecastStorm(s, now = Date.now()) {
  const t = Date.parse(String(s.position_time || ""));
  return Number.isFinite(t) && now - t <= FORECAST_STORM_MAX_AGE_MS && t - now <= 3600_000;
}

function stormOfForecast(s) {
  const kt = feedNumber(s.vmax_kt);
  const basin = String(s.basin || "").toUpperCase();
  const t = Date.parse(String(s.position_time || ""));
  return {
    id: String(s.storm_id || "").toUpperCase(),
    name: stormName(s),
    kind: forecastStormKind(basin, kt, s.category),
    category: String(s.category || categoryOf(kt) || ""),
    wind_kt: kt,
    pressure_mb: null,
    lat: feedNumber(s.lat),
    lon: feedNumber(s.lon),
    basin,
    moving: null,
    updated: Number.isFinite(t) ? t : null,
    url: officialCenter(s)[1],
    source: officialCenter(s)[0],
    ri_probability: Number.isFinite(feedNumber(s.ri_probability)) ? feedNumber(s.ri_probability) : null,
  };
}

function polygonCentroid(geometry) {
  if (!geometry || !Array.isArray(geometry.coordinates)) return null;
  const ring = geometry.type === "Polygon" ? geometry.coordinates[0]
    : geometry.type === "MultiPolygon" ? (geometry.coordinates[0] || [])[0] : null;
  if (!Array.isArray(ring) || !ring.length) return null;
  // a GeoJSON ring repeats its first vertex at the end; counting it twice would pull the centre toward it
  const first = ring[0], last = ring[ring.length - 1];
  const pts = ring.length > 1 && first[0] === last[0] && first[1] === last[1] ? ring.slice(0, -1) : ring;
  let lon = 0, lat = 0;
  for (const [x, y] of pts) { lon += Number(x); lat += Number(y); }
  const c = { lat: lat / pts.length, lon: lon / pts.length };
  return Number.isFinite(c.lat) && Number.isFinite(c.lon) ? c : null;
}

function warningOf(f) {
  const p = (f && f.properties) || {};
  const c = polygonCentroid(f && f.geometry);
  return {
    id: String(p.id || f.id || ""),
    event: String(p.event || "Warning"),
    headline: String(p.headline || p.event || ""),
    area: String(p.areaDesc || ""),
    severity: String(p.severity || ""),
    urgency: String(p.urgency || ""),
    sent: p.sent ? Date.parse(p.sent) : null,
    expires: p.ends || p.expires ? Date.parse(p.ends || p.expires) : null,
    sender: String(p.senderName || "National Weather Service"),
    lat: c ? c.lat : null,
    lon: c ? c.lon : null,
    url: "https://www.weather.gov/",
  };
}

// How long a page waits for the live data before it is sent without it. Measured on the preview
// (2026-10-05): a warm server answers from memory in well under this; the first request on a cold one took
// 1.0 s, nearly all of it the agencies' feeds.
const PAGE_LIVE_BUDGET_MS = 350;

// the work's result, or null once budgetMs has passed; the work itself carries on behind the response
async function withinBudget(work, budgetMs, ctx = null) {
  const guarded = work.catch(() => null);
  if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(guarded);
  let timer = null;
  const late = new Promise((resolve) => { timer = setTimeout(() => resolve(null), budgetMs); });
  try {
    return await Promise.race([guarded, late]);
  } finally {
    clearTimeout(timer);
  }
}

// the build's area index changes only with a deploy (which starts new isolates), so one parse a minute
// serves every request in between
let _indexMemo = null;

async function areaIndexOf(env, request) {
  const now = Date.now();
  if (_indexMemo && _indexMemo.value && now - _indexMemo.at < LIVE_TTL_MS) return _indexMemo.value;
  const value = await fetchAssetJson(env, "/data/area-index.json", request);
  _indexMemo = { at: now, value };
  return value;
}

// A count is a number only when its source answered: an agency that did not answer is "unknown" (null),
// never 0 -- "0 earthquakes in the last 24 hours" would be a false statement.
async function getLive(env, request, ctx = null) {
  const [day, week, nhc, warnings, idx] = await Promise.all([
    cachedJson(LIVE_SOURCES.quakesDay, LIVE_TTL_MS, ctx),
    cachedJson(LIVE_SOURCES.quakesWeek, LIVE_TTL_MS, ctx),
    cachedJson(LIVE_SOURCES.storms, LIVE_TTL_MS, ctx),
    cachedJson(LIVE_SOURCES.tornadoWarnings, LIVE_TTL_MS, ctx),
    areaIndexOf(env, request),
  ]);
  const dayOk = Boolean(day && Array.isArray(day.features));
  const weekOk = Boolean(week && Array.isArray(week.features));
  const nhcOk = Boolean(nhc && Array.isArray(nhc.activeStorms));
  const nwsOk = Boolean(warnings && Array.isArray(warnings.features));
  const idxOk = Boolean(idx && typeof idx === "object");
  const quakesDay = (dayOk ? day.features : []).map(quakeOf).filter(validQuake).sort((a, b) => b.time - a.time);
  const quakesWeek = (weekOk ? week.features : []).map(quakeOf).filter(validQuake).sort((a, b) => b.mag - a.mag);
  const forecastStorms = idxOk && Array.isArray(idx.storms) ? idx.storms : [];
  const forecastById = new Map(forecastStorms.map((s) => [String(s.storm_id || "").toUpperCase(), s]));
  const nhcStorms = (nhcOk ? nhc.activeStorms : []).map((s) => stormOfNhc(s, forecastById));
  const nhcIds = new Set(nhcStorms.map((s) => s.id));
  // NOAA covers the Atlantic and the eastern and central Pacific; every other active storm comes from the
  // HazardPulse forecast feed (positions from the Joint Typhoon Warning Center's advisories). If NOAA's feed
  // is unreachable, its basins come from the forecast feed too rather than vanishing.
  const noaaBasins = nhcOk ? ["AL", "EP", "CP"] : [];
  const otherStorms = forecastStorms
    .filter((s) => !nhcIds.has(String(s.storm_id || "").toUpperCase()) && !noaaBasins.includes(String(s.basin || "").toUpperCase()))
    .filter((s) => freshForecastStorm(s))
    .map(stormOfForecast);
  const storms = [...nhcStorms, ...otherStorms].filter((s) => Number.isFinite(s.lat) && Number.isFinite(s.lon));
  const tornadoWarnings = (nwsOk ? warnings.features : []).map(warningOf);
  const tracked = idxOk && Array.isArray(idx.tornadoes) ? idx.tornadoes : [];
  const known = (ok, n) => (ok ? n : null);
  return {
    as_of: new Date().toISOString(),
    counts: {
      quakes_day: known(dayOk, quakesDay.length),
      quakes_day_m45: known(dayOk, quakesDay.filter((q) => q.mag >= 4.5).length),
      quakes_week_m45: known(weekOk, quakesWeek.length),
      // without NOAA's feed its basins come from the forecast feed, so the count is still a count
      storms: known(nhcOk || idxOk, storms.length),
      tornado_warnings: known(nwsOk, tornadoWarnings.length),
      tracked_thunderstorms: known(idxOk, tracked.length),
    },
    quakes: { day: quakesDay, week_major: quakesWeek.slice(0, 12) },
    storms,
    tornado_warnings: tornadoWarnings,
    sources: {
      usgs: dayOk ? "ok" : "unavailable",
      usgs_week: weekOk ? "ok" : "unavailable",
      nhc: nhcOk ? "ok" : "unavailable",
      nws: nwsOk ? "ok" : "unavailable",
      forecasts: idxOk ? "ok" : "unavailable",
    },
    _index: idx,
  };
}

function publicLive(live) {
  const { _index, ...rest } = live;
  return { ...rest, html: { feed: feedHtml(live), stats: statsOf(live) } };
}

// NWS alerts in effect at the visitor's point (US only). The point is rounded to 0.1 degree (about 11 km)
// before it is sent, and is never stored.
async function nwsPointAlerts(geo, ctx = null) {
  if (!geo.isReliable || !["US", "PR", "VI", "GU", "AS", "MP"].includes(String(geo.country || "").toUpperCase())) {
    return null;
  }
  const lat = (Math.round(geo.latitude * 10) / 10).toFixed(1);
  const lon = (Math.round(geo.longitude * 10) / 10).toFixed(1);
  const d = await cachedJson(`https://api.weather.gov/alerts/active?point=${lat},${lon}`, LIVE_TTL_MS, ctx);
  if (!d || !Array.isArray(d.features)) return null;
  const rank = { Extreme: 0, Severe: 1, Moderate: 2, Minor: 3, Unknown: 4 };
  return d.features.map(warningOf).sort((a, b) => (rank[a.severity] ?? 5) - (rank[b.severity] ?? 5));
}

// ---------------------------------------------------------------------------------------------
// Personalisation: the visitor's approximate location (request.cf), used for this one response and
// never stored. Everything shown is a fact -- a distance, a name, an official alert, a published chance --
// and never an instruction of ours: official agencies issue warnings, HazardPulse does not.
// ---------------------------------------------------------------------------------------------

const AREA = {
  hurricaneBannerKm: 500, // an active tropical cyclone this close puts a notice at the top of every page
  tornadoBannerKm: 50, // ...as does a tracked storm this close under an NWS tornado warning
  tornadoBannerChance: 0.1, // ...or with at least this chance of a tornado within 60 minutes
  quakeNearbyKm: 1000, // "Near you" names the nearest earthquake of the last day within this distance
  stormNearbyKm: 2000, // ...the nearest tropical cyclone within this distance
  tornadoNearbyKm: 300, // ...and the nearest tracked thunderstorm within this distance
};

// the site's one chance scale (hazardpulse.site.fmt.LEVELS)
const LEVELS = [
  [0.5, "p6"],
  [0.3, "p5"],
  [0.15, "p4"],
  [0.05, "p3"],
  [0.01, "p2"],
  [0, "p1"],
];
const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];

function chanceLevel(p) {
  const v = Number(p) || 0;
  for (const [lo, cls] of LEVELS) if (v >= lo) return cls;
  return "p1";
}

// a hazard chance: never "0%" -- no model here outputs an exact zero, so a 0 is rounding (the tornado scorer
// stores 4 decimals) or a default, and "0%" would promise what nobody can ("Near you" said a storm 170 km
// away had a 0% chance of a tornado, 2026-10-05)
function formatChance(p) {
  const v = Number(p);
  if (!Number.isFinite(v)) return "&mdash;";
  if (v < 0.001) return "&lt;0.1%";
  if (v >= 1) return "100%";
  return `${(v * 100).toFixed(1)}%`;
}

function chanceHtml(p) {
  return `<span class="chance ${chanceLevel(p)}">${formatChance(p)}</span>`;
}

function bearing(fromLat, fromLon, toLat, toLon) {
  const p1 = (fromLat * Math.PI) / 180;
  const p2 = (toLat * Math.PI) / 180;
  const dl = ((toLon - fromLon) * Math.PI) / 180;
  const x = Math.sin(dl) * Math.cos(p2);
  const y = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  const deg = ((Math.atan2(x, y) * 180) / Math.PI + 360) % 360;
  return COMPASS[Math.floor((deg + 11.25) / 22.5) % 16];
}

function distanceText(km) {
  const k = km >= 100 ? Math.round(km / 10) * 10 : Math.round(km);
  const mi = Math.round(km / 1.609344);
  return `${k.toLocaleString("en-US")} km (${mi.toLocaleString("en-US")} mi)`;
}

function agoText(ms, now = Date.now()) {
  if (!Number.isFinite(ms)) return "";
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} h ${m % 60 ? `${m % 60} min ` : ""}ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

// agency text, shortened BEFORE it is escaped (cutting escaped text can split an entity such as &amp;)
function clipText(text, max) {
  const chars = Array.from(String(text || ""));
  return escapeHtml(chars.length > max ? `${chars.slice(0, max - 1).join("").trimEnd()}…` : chars.join(""));
}

function timeTag(ms, now = Date.now()) {
  if (!Number.isFinite(ms)) return "";
  const iso = new Date(ms).toISOString();
  return `<time datetime="${iso}" data-relative>${agoText(ms, now)}</time>`;
}

function stormName(storm) {
  const name = String(storm.storm_name || "").trim();
  if (!name || ["INVEST", "UNNAMED", "NONAME"].includes(name.toUpperCase())) {
    return String(storm.storm_id || "An unnamed storm");
  }
  return name === name.toUpperCase() ? name.charAt(0) + name.slice(1).toLowerCase() : name;
}

// the agency that issues the official forecast where the storm is now. A live storm names its own (NHC's
// advisory bin says who writes the advisory). Otherwise by position: an East Pacific storm west of 140 W has
// passed to the Central Pacific Hurricane Center, whose area runs to the dateline -- NOAA reports a storm on
// it at +180 -- and west of the dateline it has left NOAA's area. hazardpulse.site.data.basin_of is the
// same rule.
function officialCenter(storm) {
  if (storm.source && storm.url) return [String(storm.source), String(storm.url)];
  const NHC = ["National Hurricane Center", "https://www.nhc.noaa.gov/"];
  const CPHC = ["Central Pacific Hurricane Center", "https://www.nhc.noaa.gov/?cpac"];
  const JTWC = ["Joint Typhoon Warning Center", "https://www.metoc.navy.mil/jtwc/jtwc.html"];
  const basin = String(storm.basin || "").toUpperCase();
  const lon = storm.lon === null || storm.lon === undefined || storm.lon === "" ? NaN : Number(storm.lon);
  if (basin === "AL") return NHC;
  if (basin === "EP" || basin === "CP") {
    if (!Number.isFinite(lon)) return basin === "CP" ? CPHC : NHC;
    if (lon > 0 && lon < 180) return JTWC;
    return lon < -140 || lon >= 180 ? CPHC : NHC;
  }
  return JTWC;
}

// the visitor's own grid cell in the earthquake forecast: the value the forecast publishes for it
function earthquakeCell(lat, lon, replay) {
  if (!replay) return null;
  const d = replay.forecast_domain;
  const row = Math.floor((lat - Number(d.lat_min)) / Number(d.dlat));
  const col = Math.floor((lon - Number(d.lon_min)) / Number(d.dlon));
  if (row < 0 || row >= Number(d.n_lat) || col < 0 || col >= Number(d.n_lon)) return { outside: true };
  const values = String(replay.probability_grid).split(",");
  const p = Number(values[row * Number(d.n_lon) + col]);
  if (!Number.isFinite(p)) return null;
  return {
    p,
    lat: Number(d.lat_min) + (row + 0.5) * Number(d.dlat),
    lon: Number(d.lon_min) + (col + 0.5) * Number(d.dlon),
  };
}

function nearest(lat, lon, items, maxKm) {
  let best = null;
  for (const item of items) {
    const ilat = feedNumber(item.lat);
    const ilon = feedNumber(item.lon);
    if (!Number.isFinite(ilat) || !Number.isFinite(ilon)) continue;
    const km = haversineKm(lat, lon, ilat, ilon);
    if (km <= maxKm && (!best || km < best.km)) best = { item, km };
  }
  return best;
}

function tornadoWarned(storm) {
  if (!storm) return false;
  if (typeof storm.warned === "boolean") return storm.warned;
  return Boolean(storm.v3 && storm.v3.nws_warning && storm.v3.nws_warning.active);
}

// data: the build's area index ({eq, eqGate, storms, tornadoes}); live: getLive() (optional);
// alerts: NWS alerts at the visitor's point (optional; null outside the US)
function summarizeArea(geo, data, live = null, alerts = null) {
  if (!geo.isReliable) return { reliable: false };
  const lat = geo.latitude;
  const lon = geo.longitude;
  // without the live picture, the build's index storms -- only those whose position is current -- in the
  // live shape every renderer reads
  const storms = live && live.storms
    ? live.storms
    : (data.storms || []).filter((s) => freshForecastStorm(s)).map(stormOfForecast);
  const out = {
    reliable: true,
    lat,
    lon,
    // what we could see: "nothing nearby" is said only of a source that answered
    known: {
      quakes: Boolean(live && live.sources && live.sources.usgs === "ok"),
      storms: live && live.counts ? live.counts.storms !== null : data.known !== false,
      // an index without a forecast time (a test's, or a caller's own) is taken as current
      tornadoes: data.known !== false && data.tornadoesCurrent !== false,
    },
    trackedAt: data.trackedAt ?? null,
    eqCell: data.eqGate === "block" ? null : earthquakeCell(lat, lon, data.eq),
    quake: live ? nearest(lat, lon, live.quakes.day, AREA.quakeNearbyKm) : null,
    quakesNearbyDay: live ? live.quakes.day.filter((q) => haversineKm(lat, lon, q.lat, q.lon) <= AREA.quakeNearbyKm).length : null,
    storm: nearest(lat, lon, storms, AREA.stormNearbyKm),
    tornado: null,
    alerts,
    banner: null,
  };
  out.tornado = out.known.tornadoes ? nearest(lat, lon, data.tornadoes || [], AREA.tornadoNearbyKm) : null;
  const official = (alerts || []).find((a) => a.severity === "Extreme" || a.severity === "Severe");
  const near = nearest(lat, lon, storms, AREA.hurricaneBannerKm);
  if (official) {
    out.banner = { kind: "alert", item: official };
  } else if (near) {
    out.banner = { kind: "hurricane", ...near };
  } else if (out.known.tornadoes) {
    const close = (data.tornadoes || [])
      .map((s) => ({ item: s, km: haversineKm(lat, lon, feedNumber(s.lat), feedNumber(s.lon)) }))
      .filter(
        (x) =>
          Number.isFinite(x.km) &&
          x.km <= AREA.tornadoBannerKm &&
          (tornadoWarned(x.item) || Number(x.item.tornado_probability) >= AREA.tornadoBannerChance)
      )
      .sort((a, b) => Number(b.item.tornado_probability) - Number(a.item.tornado_probability));
    if (close.length) out.banner = { kind: "tornado", ...close[0] };
  }
  return out;
}

function placeText(geo) {
  const city = escapeHtml(geo.city || geo.region || "");
  const country = escapeHtml(geo.country || "");
  return [city, country].filter(Boolean).join(", ") || "your area";
}

function stormLabel(s) {
  const strength = s.category ? `${escapeHtml(s.kind)}, ${escapeHtml(s.category)}` : escapeHtml(s.kind || "Tropical cyclone");
  return `${escapeHtml(s.name)} (${strength})`;
}

// ---- server-rendered fragments (the same HTML the page gets at the edge and the app refreshes) ----

// only the counts that are known: an unknown one is left out, and the page keeps its dash
function statsOf(live) {
  const out = {};
  for (const key of ["quakes_day", "quakes_day_m45", "storms", "tornado_warnings", "tracked_thunderstorms"]) {
    const v = live.counts[key];
    if (Number.isFinite(v)) out[key] = String(v);
  }
  return out;
}

function feedHtml(live, now = Date.now()) {
  const items = [];
  for (const q of live.quakes.day.slice(0, 14)) {
    const big = q.mag >= 4.5 ? " feed-big" : "";
    items.push({
      t: q.time,
      html:
        `<li class="feed-item feed-eq${big}"><span class="feed-mag">M${q.mag.toFixed(1)}</span>` +
        `<span class="feed-main"><a href="${escapeHtml(safeUrl(q.url, "https://earthquake.usgs.gov/"))}" rel="noopener">${escapeHtml(q.place)}</a>` +
        `<span class="feed-meta">Earthquake${Number.isFinite(q.depth_km) ? ` &middot; ${Math.round(q.depth_km)} km deep` : ""} &middot; ${timeTag(q.time, now)}` +
        `${q.tsunami ? " &middot; <strong>tsunami message issued</strong>" : ""}</span></span></li>`,
    });
  }
  for (const s of live.storms) {
    items.push({
      t: s.updated || now - 3 * 3600_000,
      html:
        `<li class="feed-item feed-hu"><span class="feed-mag">${Number.isFinite(s.wind_kt) ? `${Math.round(s.wind_kt)} kt` : "TC"}</span>` +
        `<span class="feed-main"><a href="${escapeHtml(safeUrl(s.url, "https://www.nhc.noaa.gov/"))}" rel="noopener">${stormLabel(s)}</a>` +
        `<span class="feed-meta">${escapeHtml(s.source)}` +
        `${Number.isFinite(s.ri_probability) ? ` &middot; chance of rapid intensification in 24 h: ${formatChance(s.ri_probability)}` : ""}` +
        `${s.updated ? ` &middot; advisory ${timeTag(s.updated, now)}` : ""}</span></span></li>`,
    });
  }
  for (const w of live.tornado_warnings.slice(0, 10)) {
    items.push({
      t: w.sent || now,
      html:
        `<li class="feed-item feed-to"><span class="feed-mag">TOR</span>` +
        `<span class="feed-main"><a href="${escapeHtml(safeUrl(w.url))}" rel="noopener">${escapeHtml(w.event)}: ${clipText(w.area, 140)}</a>` +
        `<span class="feed-meta">${escapeHtml(w.sender)} &middot; ${timeTag(w.sent, now)}</span></span></li>`,
    });
  }
  items.sort((a, b) => b.t - a.t);
  // a feed that did not answer is said to be silent, never taken for a quiet world
  const sources = live.sources || {};
  const silent = [
    sources.usgs === "unavailable" ? "the USGS earthquake feed" : "",
    sources.nhc === "unavailable" ? "the National Hurricane Center" : "",
    sources.nws === "unavailable" ? "the National Weather Service" : "",
  ].filter(Boolean);
  const note = silent.length
    ? `<li class="feed-empty">Not answering right now: ${silent.join(", ")}. What it reports will appear here when it does.</li>`
    : "";
  if (!items.length) {
    return note || '<li class="feed-empty">Nothing reported in the last 24 hours.</li>';
  }
  return items.slice(0, 16).map((i) => i.html).join("") + note;
}

function nearHtml(area, geo, live = null) {
  if (!area || !area.reliable) return "";
  const tiles = [];
  // 1. official alerts (US) -- relayed as issued
  if (Array.isArray(area.alerts)) {
    if (area.alerts.length) {
      const a = area.alerts[0];
      tiles.push(
        `<div class="near-tile near-alert near-${a.severity === "Extreme" || a.severity === "Severe" ? "bad" : "warn"}">` +
          `<p class="near-kicker">Official alerts</p><p class="near-figure">${area.alerts.length}</p>` +
          `<p>${escapeHtml(a.event)} in effect${a.expires ? ` until ${new Date(a.expires).toUTCString().slice(17, 22)} UTC` : ""}: ` +
          `${clipText(a.headline, 160)}</p><a href="https://www.weather.gov/" rel="noopener">National Weather Service</a></div>`
      );
    } else {
      tiles.push(
        `<div class="near-tile near-good"><p class="near-kicker">Official alerts</p><p class="near-figure">None</p>` +
          `<p>The National Weather Service has no alerts in effect at your location.</p></div>`
      );
    }
  }
  // 2. earthquakes: what happened, then what is likely ("Quiet" only from a feed that answered)
  const known = area.known || { quakes: true, storms: true, tornadoes: true };
  const q = area.quake;
  const cell = area.eqCell;
  tiles.push(
    `<div class="near-tile"><p class="near-kicker">Earthquakes</p>` +
      (q
        ? `<p class="near-figure">M${q.item.mag.toFixed(1)}</p><p>${distanceText(q.km)} ${bearing(area.lat, area.lon, q.item.lat, q.item.lon)} of you, ${timeTag(q.item.time)}: ${escapeHtml(q.item.place)}.` +
          `${area.quakesNearbyDay > 1 ? ` ${area.quakesNearbyDay} earthquakes M2.5+ within ${AREA.quakeNearbyKm.toLocaleString("en-US")} km today.` : ""}</p>`
        : known.quakes
          ? `<p class="near-figure">Quiet</p><p>No earthquake of M2.5+ within ${AREA.quakeNearbyKm.toLocaleString("en-US")} km in the last 24 hours.</p>`
          : `<p class="near-figure">&mdash;</p><p>The USGS earthquake feed is not answering right now.</p>`) +
      (cell && !cell.outside
        ? `<p class="near-forecast">Chance of an M6+ in your 2&deg; cell, next 30 days: ${chanceHtml(cell.p)}</p>`
        : "") +
      `<a href="/live/earthquake/">Earthquake forecast</a></div>`
  );
  // 3. tropical cyclones
  const s = area.storm;
  if (s) {
    const [center, url] = officialCenter(s.item);
    tiles.push(
      `<div class="near-tile"><p class="near-kicker">Tropical cyclones</p><p class="near-figure">${distanceText(s.km).split(" (")[0]}</p>` +
        `<p>to ${stormLabel(s.item)}, ${bearing(area.lat, area.lon, s.item.lat, s.item.lon)} of you.` +
        `${Number.isFinite(Number(s.item.ri_probability)) && s.item.ri_probability !== null ? ` Chance it intensifies rapidly in 24 h: ${chanceHtml(s.item.ri_probability)}.` : ""}</p>` +
        `<a href="${escapeHtml(safeUrl(url))}" rel="noopener">${escapeHtml(center)}</a></div>`
    );
  } else if (known.storms) {
    tiles.push(
      `<div class="near-tile near-good"><p class="near-kicker">Tropical cyclones</p><p class="near-figure">None nearby</p>` +
        `<p>No active tropical cyclone within ${AREA.stormNearbyKm.toLocaleString("en-US")} km.` +
        `${live && live.counts.storms ? ` ${live.counts.storms} active elsewhere.` : ""}</p><a href="/live/hurricane/">All active storms</a></div>`
    );
  } else {
    tiles.push(
      `<div class="near-tile"><p class="near-kicker">Tropical cyclones</p><p class="near-figure">&mdash;</p>` +
        `<p>Storm positions are not available right now.</p><a href="https://www.nhc.noaa.gov/" rel="noopener">National Hurricane Center</a></div>`
    );
  }
  // 4. thunderstorms (from the latest tornado forecast, and only while it is recent enough to say where they are)
  const t = area.tornado;
  const trackedWhen = Number.isFinite(area.trackedAt) ? ` (tracked ${timeTag(area.trackedAt)})` : "";
  tiles.push(
    `<div class="near-tile"><p class="near-kicker">Thunderstorms</p>` +
      (t
        ? `<p class="near-figure">${distanceText(t.km).split(" (")[0]}</p><p>to the nearest tracked storm${trackedWhen}; its chance of a tornado within the hour: ${chanceHtml(t.item.tornado_probability)}` +
          `${tornadoWarned(t.item) ? ". It is under a National Weather Service tornado warning" : ""}.</p>`
        : known.tornadoes
          ? `<p class="near-figure">None nearby</p><p>No thunderstorm tracked within ${AREA.tornadoNearbyKm} km${trackedWhen}. Storm tracking covers the US and its edges.</p>`
          : `<p class="near-figure">&mdash;</p><p>Storm tracking has not updated recently, so it cannot say where storms are now.</p>`) +
      `<a href="/live/tornado/">Tornado forecast</a></div>`
  );
  return (
    `<div class="near-head"><p class="near-place"><span class="near-pin" aria-hidden="true"></span>Near you &middot; ${placeText(geo)}</p>` +
    `<p class="near-note">From your approximate location, used for this page view only.</p></div>` +
    `<div class="near-grid">${tiles.join("")}</div>`
  );
}

function bannerHtml(area) {
  const b = area && area.banner;
  if (!b) return "";
  let html = "";
  if (b.kind === "alert") {
    const a = b.item;
    html =
      `<strong class="emergency-title">${escapeHtml(a.event)} for your area.</strong> ` +
      `${clipText(a.headline, 220)} &mdash; issued by the ${escapeHtml(a.sender)}. ` +
      `Follow its instructions: <a href="https://www.weather.gov/" rel="noopener">weather.gov</a>.`;
  } else if (b.kind === "hurricane") {
    const s = b.item;
    const [center, url] = officialCenter(s);
    html =
      `<strong class="emergency-title">Tropical cyclone nearby.</strong> ${stormLabel(s)} is about ${distanceText(b.km)} ` +
      `${bearing(area.lat, area.lon, s.lat, s.lon)} of your approximate location. For its track and any warnings for your ` +
      `area, follow the <a href="${escapeHtml(safeUrl(url))}" rel="noopener">${escapeHtml(center)}</a> and your local officials.`;
  } else if (b.kind === "tornado") {
    const s = b.item;
    const warned = tornadoWarned(s);
    html =
      `<strong class="emergency-title">${warned ? "Tornado warning nearby." : "Storm nearby."}</strong> ` +
      `A thunderstorm about ${distanceText(b.km)} from your approximate location ` +
      (warned
        ? "is under a National Weather Service tornado warning. "
        : `has a ${formatChance(s.tornado_probability)} chance of producing a tornado in the next hour (HazardPulse research forecast). `) +
      `Check <a href="https://www.weather.gov/" rel="noopener">weather.gov</a> for the warnings in effect where you are.`;
  }
  return html
    ? `<div class="container emergency-inner"><span class="emergency-dot" aria-hidden="true"></span><p>${html}</p></div>`
    : "";
}

class ThemeRootHandler {
  constructor(themePreference) {
    this.themePreference = themePreference;
  }

  element(el) {
    if (this.themePreference === "dark" || this.themePreference === "light") {
      el.setAttribute("data-theme", this.themePreference);
    }
  }
}

class ThemeToggleHandler {
  constructor(themePreference) {
    this.themePreference = themePreference;
  }

  element(el) {
    const dark = this.themePreference === "dark";
    el.setAttribute("aria-pressed", dark ? "true" : "false");
    el.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
}

// places the visitor marker on a map; the map's projection travels on the marker itself
// (data-lon0 / data-lat0 / data-sx / data-sy: x = (lon - lon0) * sx, y = (lat0 - lat) * sy)
class UserMarkerHandler {
  constructor(geo) {
    this.geo = geo;
  }

  element(el) {
    try {
      const { geo } = this;
      if (!geo.isReliable) return;
      const raw = ["data-lon0", "data-lat0", "data-sx", "data-sy"].map((n) => el.getAttribute(n));
      // Number(null) is 0: a missing attribute must not become a projection
      if (raw.some((v) => v === null || String(v).trim() === "")) return;
      const [lon0, lat0, sx, sy] = raw.map(Number);
      if (![lon0, lat0, sx, sy].every(Number.isFinite)) return;
      const x = (geo.longitude - lon0) * sx;
      const y = (lat0 - geo.latitude) * sy;
      if (x < 0 || y < 0) return;
      el.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
      el.setAttribute(
        "aria-label",
        `Your approximate location: ${geo.latitude.toFixed(1)}, ${geo.longitude.toFixed(1)}`
      );
    } catch {
      // a marker that cannot be placed stays hidden off the map
    }
  }
}

// the globe turns to face the visitor first (approximate location, rounded to a degree)
class GlobeHandler {
  constructor(geo) {
    this.geo = geo;
  }

  element(el) {
    if (!this.geo.isReliable) return;
    el.setAttribute("data-user-lat", (Math.round(this.geo.latitude * 10) / 10).toFixed(1));
    el.setAttribute("data-user-lon", (Math.round(this.geo.longitude * 10) / 10).toFixed(1));
  }
}

class HtmlFillHandler {
  constructor(html) {
    this.html = html;
  }

  element(el) {
    if (this.html) el.setInnerContent(this.html, { html: true });
  }
}

class LiveStatHandler {
  constructor(stats) {
    this.stats = stats || {};
  }

  element(el) {
    const key = el.getAttribute("data-live");
    if (key && Object.prototype.hasOwnProperty.call(this.stats, key)) el.setInnerContent(this.stats[key]);
  }
}

// the build's area index as the summary's input: the earthquake grid, the forecast feed's storms and the
// tracked thunderstorms
// a forecast id carries its issue time: to_fcst_20261004_2059 -> 2026-10-04T20:59Z
function forecastTimeOf(id) {
  const m = /_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$/.exec(String(id || ""));
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) : null;
}

// tracked thunderstorms move ~50 km an hour: a list older than this no longer says where they are
const TRACKED_STORMS_MAX_AGE_MS = 3 * 3600_000;

function areaDataOf(idx, now = Date.now()) {
  if (!idx) return { eq: null, eqGate: null, storms: [], tornadoes: [], known: false, tornadoesCurrent: false };
  const eq = idx.eq && idx.eq.forecast_domain && idx.eq.probability_grid ? idx.eq : null;
  const trackedAt = forecastTimeOf(idx.forecasts && idx.forecasts.to);
  return {
    eq,
    eqGate: eq ? eq.gate : null,
    storms: Array.isArray(idx.storms) ? idx.storms : [],
    tornadoes: Array.isArray(idx.tornadoes) ? idx.tornadoes : [],
    known: true,
    trackedAt,
    tornadoesCurrent: trackedAt !== null && now - trackedAt <= TRACKED_STORMS_MAX_AGE_MS,
  };
}

// a live page is rendered when its forecast is published; when it is viewed after the forecast's own window
// has passed (a 60-minute tornado forecast seen three hours later), the page says so. The window runs from the
// DATA's valid time when the page gives one (data-valid): a tornado forecast made from radar data observed 19
// minutes before it was issued has 41 minutes left at issue, and counting from the issue time left it expired
// with no notice 6.3% of the time (the 2026-10-05 audit). data-format="short" is the one-line form a hazard card
// carries (home and /live/).
function minutesText(min) {
  return min >= 120 ? `${Math.floor(min / 60)} hours` : `${Math.max(1, Math.round(min))} minutes`;
}

class ForecastAgeHandler {
  constructor(now = new Date()) {
    this.now = now;
  }

  element(el) {
    const issued = Date.parse(el.getAttribute("data-issued") || "");
    const validRaw = el.getAttribute("data-valid");
    const valid = validRaw ? Date.parse(validRaw) : NaN;
    const windowMin = Number(el.getAttribute("data-window-minutes"));
    if (!Number.isFinite(issued) || !Number.isFinite(windowMin) || windowMin <= 0) return;
    const fromValid = Number.isFinite(valid) && valid <= issued;
    const start = fromValid ? valid : issued;
    const ageMin = (this.now.getTime() - start) / 60000;
    if (ageMin <= windowMin) return;
    const span = windowMin >= 120 ? `${Math.round(windowMin / 60)}-hour` : `${windowMin}-minute`;
    const endedAgo = minutesText(ageMin - windowMin);
    if (el.getAttribute("data-format") === "short") {
      el.setInnerContent(` &middot; <strong>out of date: its ${span} window ended ${endedAgo} ago</strong>`, {
        html: true,
      });
      return;
    }
    const schedule = escapeHtml(el.getAttribute("data-schedule") || "");
    const lead = fromValid
      ? `This forecast was made from data observed at ${new Date(valid).toISOString().slice(11, 16)} UTC, so its ` +
        `${span} window ended ${endedAgo} ago.`
      : `This forecast was issued ${minutesText(ageMin)} ago, so its ${span} window has passed.`;
    el.setInnerContent(
      `<strong>${lead}</strong> ` +
        `A new one is published ${schedule || "on schedule"}; until then, treat these numbers as out of date.`,
      { html: true }
    );
  }
}

// The live status: each hazard's latest forecast, its age now, whether it is overdue, and its quality-check
// outcome, from the index the site build writes (hazardpulse.site.build.status_index). It used to answer
// status "ok" unconditionally.
function opsSnapshot(index, now = new Date()) {
  const hazards = (index && Array.isArray(index.hazards) ? index.hazards : []).map((h) => {
    const issued = h.issued_at ? Date.parse(h.issued_at) : NaN;
    const ageHours = Number.isFinite(issued) ? (now.getTime() - issued) / 3600000 : null;
    const overdue = ageHours === null || ageHours > Number(h.overdue_after_hours);
    return { ...h, age_hours: ageHours === null ? null : Math.round(ageHours * 10) / 10, overdue };
  });
  let status = "ok";
  if (!hazards.length) status = "unknown";
  else if (hazards.some((h) => h.overdue || h.quality_checks === "block")) status = "delayed";
  else if (hazards.some((h) => h.quality_checks !== "pass")) status = "published_with_warnings";
  return { status, as_of: now.toISOString(), domain: PRIMARY_DOMAIN, hazards };
}

async function buildOpsSnapshot(env, request) {
  return opsSnapshot(await fetchAssetJson(env, "/data/status.json", request));
}

async function handleApiRequest(request, env, ctx = null) {
  if (!_checkRateLimit(request)) {
    return errorEnvelope("rate_limited", "Too many requests from this address. Try again in a minute.", 429);
  }

  const url = new URL(request.url);
  const path = url.pathname;

  // what is happening now (shared by every visitor; cached a minute)
  if (path === "/api/v1/now") {
    const live = await getLive(env, request, ctx);
    return jsonResponse(apiEnvelope(publicLive(live), 60), 200, "public, max-age=60");
  }

  // the visitor's own area (from their approximate location; never cached, never stored)
  if (path === "/api/v1/near") {
    const geo = normalizeGeo(request.cf || {});
    const live = await getLive(env, request, ctx);
    const alerts = geo.isReliable ? await nwsPointAlerts(geo, ctx) : null;
    const area = geo.isReliable ? summarizeArea(geo, areaDataOf(live._index), live, alerts) : { reliable: false };
    return jsonResponse(
      apiEnvelope({ reliable: Boolean(area.reliable), html: { near: nearHtml(area, geo, live), banner: bannerHtml(area) } }, 0),
      200,
      "private, no-store"
    );
  }

  if (path === "/api/v1/live/pulse") {
    const pulse = await fetchAssetJson(env, "/data/live-pulse.json", request);
    if (!pulse) return errorEnvelope("not_found", "Live pulse data is unavailable.");
    return jsonResponse(apiEnvelope(pulse, 300), 200, "public, max-age=300");
  }

  const liveMatch = path.match(/^\/api\/v1\/live\/(earthquake|hurricane|tornado)$/);
  if (liveMatch) {
    const hazard = liveMatch[1];
    if (hazard === "hurricane") {
      const storms = await fetchAssetJson(env, "/data/live-storms.json", request);
      if (!storms) return errorEnvelope("not_found", "Live hurricane data is unavailable.");
      return jsonResponse(apiEnvelope(storms, 300, { hazard }), 200, "public, max-age=300");
    }
    if (hazard === "tornado") {
      const tornadoes = await fetchAssetJson(env, "/data/live-tornadoes.json", request);
      if (!tornadoes) return errorEnvelope("not_found", "Live tornado data is unavailable.");
      return jsonResponse(apiEnvelope(tornadoes, 300, { hazard }), 200, "public, max-age=300");
    }

    const pulse = await fetchAssetJson(env, "/data/live-pulse.json", request);
    if (!pulse || !Array.isArray(pulse.hazards)) {
      return errorEnvelope("not_found", "Live earthquake data is unavailable.");
    }
    const eq = pulse.hazards.find((item) => item.key === "eq") || null;
    const forecastId = eq && eq.forecast_id;
    const replay = forecastId
      ? await fetchAssetJson(env, `/data/replay/${forecastId}.json`, request)
      : null;
    return jsonResponse(
      apiEnvelope({ summary: eq, replay }, 300, { hazard }),
      200,
      "public, max-age=300"
    );
  }

  const forecastMatch = path.match(/^\/api\/v1\/forecast\/([A-Za-z0-9_.-]+)$/);
  if (forecastMatch) {
    const forecastId = forecastMatch[1];
    const artifact = await fetchAssetJson(env, `/data/replay/${forecastId}.json`, request);
    if (!artifact) {
      return errorEnvelope("not_found", `Forecast ${forecastId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(artifact, 31536000, { forecast_id: forecastId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const replayMatch = path.match(/^\/api\/v1\/replay\/([A-Za-z0-9_.-]+)$/);
  if (replayMatch) {
    const forecastId = replayMatch[1];
    const artifact = await fetchAssetJson(env, `/data/replay/${forecastId}.json`, request);
    if (!artifact) {
      return errorEnvelope("not_found", `Replay artifact ${forecastId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(artifact, 31536000, { forecast_id: forecastId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const evidenceMatch = path.match(/^\/api\/v1\/evidence\/([A-Za-z0-9_.-]+)$/);
  if (evidenceMatch) {
    const provenanceId = evidenceMatch[1];
    const envelopes = await fetchAssetJson(
      env,
      "/data/evidence/provenance-envelopes.json",
      request
    );
    const match =
      envelopes &&
      Array.isArray(envelopes.envelopes) &&
      envelopes.envelopes.find((item) => item.provenance_id === provenanceId);
    if (!match) {
      return errorEnvelope("not_found", `Provenance envelope ${provenanceId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(match, 31536000, { provenance_id: provenanceId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  const gatesMatch = path.match(/^\/api\/v1\/gates\/([A-Za-z0-9_.-]+)$/);
  if (gatesMatch) {
    const gateDecisionId = gatesMatch[1];
    const gates = await fetchAssetJson(env, "/data/evidence/gate-decisions.json", request);
    const match =
      gates &&
      Array.isArray(gates.decisions) &&
      gates.decisions.find((item) => item.gate_decision_id === gateDecisionId);
    if (!match) {
      return errorEnvelope("not_found", `Gate decision ${gateDecisionId} was not found.`);
    }
    return jsonResponse(
      apiEnvelope(match, 31536000, { gate_decision_id: gateDecisionId }),
      200,
      "public, max-age=31536000, immutable"
    );
  }

  if (path === "/api/v1/verification/summary") {
    const summary = await fetchAssetJson(env, "/data/verification-summary.json", request);
    if (!summary) return errorEnvelope("not_found", "Verification summary is unavailable.");
    return jsonResponse(apiEnvelope(summary, 3600), 200, "public, max-age=3600");
  }

  if (path === "/api/v1/registry/models") {
    const registry = await fetchAssetJson(env, "/data/model-registry.json", request);
    if (!registry) return errorEnvelope("not_found", "Model registry is unavailable.");
    return jsonResponse(apiEnvelope(registry, 3600), 200, "public, max-age=3600");
  }

  if (path === "/api/v1/laic/summary") {
    const laic = await fetchAssetJson(env, "/data/cross-modality-summary.json", request);
    if (!laic) return errorEnvelope("not_found", "LAIC analyses summary is unavailable.");
    return jsonResponse(apiEnvelope(laic, 3600), 200, "public, max-age=3600");
  }

  const laicMatch = path.match(/^\/api\/v1\/laic\/(earthquake|hurricane|tornado|cme)$/);
  if (laicMatch) {
    const slug = laicMatch[1];
    const summary = await fetchAssetJson(env, "/data/cross-modality-summary.json", request);
    if (!summary || !Array.isArray(summary.analyses)) {
      return errorEnvelope("not_found", `No LAIC analyses for ${slug}.`);
    }
    const matches = summary.analyses.filter((a) => a.hazard === slug);
    if (!matches.length) return errorEnvelope("not_found", `No LAIC analyses for ${slug}.`);
    return jsonResponse(apiEnvelope({ hazard: slug, analyses: matches }, 3600),
                        200, "public, max-age=3600");
  }

  if (path === "/api/v1/alerts/recent") {
    const recent = await fetchAssetJson(env, "/data/alerts-recent.json", request);
    if (!recent) return errorEnvelope("not_found", "Recent alerts are unavailable.");
    return jsonResponse(apiEnvelope(recent, 60), 200, "public, max-age=60");
  }

  if (path === "/api/v1/verification/tornado-recovery") {
    const recovery = await fetchAssetJson(env, "/data/tornado-recovery.json", request);
    if (!recovery) return errorEnvelope("not_found", "Tornado recovery data unavailable.");
    return jsonResponse(apiEnvelope(recovery, 600), 200, "public, max-age=600");
  }

  if (path === "/api/v1/federation/atlas") {
    // Public atlas surface — lists tables this node exposes to peers.
    // Each peer can query these via signed Ed25519 requests (over the
    // separate signalbook federated server). This endpoint is the
    // discovery/handshake hint, not the signed query channel.
    const fingerprint = await fetchAssetJson(env, "/data/federation-fingerprint.json", request);
    if (!fingerprint) {
      return errorEnvelope(
        "not_configured",
        "Federation node is not configured. Run scripts/federation_setup.py."
      );
    }
    return jsonResponse(apiEnvelope(fingerprint, 3600), 200, "public, max-age=3600");
  }

  if (path === "/api/v1/ops/status") {
    const snapshot = await buildOpsSnapshot(env, request);
    return jsonResponse(apiEnvelope(snapshot, 60), 200, "public, max-age=60");
  }

  return null;
}

async function handleStreamRequest(request, env) {
  const path = new URL(request.url).pathname;

  if (path === "/stream/live/pulse") {
    const pulse = await fetchAssetJson(env, "/data/live-pulse.json", request);
    if (!pulse) return errorEnvelope("not_found", "Live pulse data is unavailable.");
    return sseResponse("live_pulse", apiEnvelope(pulse, 300));
  }

  if (path === "/stream/ops/status") {
    const snapshot = await buildOpsSnapshot(env, request);
    return sseResponse("ops_status", apiEnvelope(snapshot, 300));
  }

  return null;
}

// ---------------------------------------------------------------------------------------------
// The scorers' clock. GitHub's own cron is best-effort, and for this repository it was far from it:
// 2026-10-01..05 a 2-hour tornado cron ran every 5.5 h at the median, the hurricane scorer went 23 h
// without a run, and on 2026-10-05 GitHub fired no scheduled run of any workflow from 00:23Z for hours
// (status page: all systems operational). Cloudflare's cron fires on time, so every 10 minutes this asks
// GitHub to run scheduler.yml, which decides from the run history what is due (scripts/ci/schedule.py)
// and never doubles a run. It needs a GitHub token allowed to run workflows in this repository
// (`wrangler secret put GH_DISPATCH_TOKEN`: a fine-grained token, this repository only, Actions: read and
// write); without one it does nothing.
// ---------------------------------------------------------------------------------------------

const SCHEDULER_DISPATCH = "https://api.github.com/repos/coherence-energy-labs/hazardpulse/actions/workflows/scheduler.yml/dispatches";

async function dispatchScheduler(env) {
  if (!env || !env.GH_DISPATCH_TOKEN) return { dispatched: false, reason: "no GH_DISPATCH_TOKEN" };
  const res = await fetch(SCHEDULER_DISPATCH, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_DISPATCH_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "hazardpulse-scheduler",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main" }),
  });
  // 204 is success; anything else is logged (visible in the Worker's logs), never thrown into the runtime
  if (res.status !== 204) console.log(`scheduler dispatch failed: HTTP ${res.status}`);
  return { dispatched: res.status === 204, status: res.status };
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(dispatchScheduler(env));
  },

  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    if (
      request.headers.get(INTERNAL_ASSET_HEADER) === "1" &&
      (pathname.startsWith("/data/") ||
        pathname.startsWith("/assets/") ||
        pathname.endsWith(".json") ||
        pathname.endsWith(".xml") ||
        pathname.endsWith(".txt") ||
        pathname.endsWith(".md"))
    ) {
      return env.ASSETS.fetch(request);
    }

    if (pathname === "/commercial-license" || pathname === "/commercial-license/") {
      return withSecurityHeaders(
        Response.redirect(`${PRIMARY_DOMAIN}/COMMERCIAL_LICENSE.md`, 302),
        {
          cacheControl: "public, max-age=3600",
          xRobotsTag: "noindex, nofollow",
        }
      );
    }

    // /api/ itself is the API's documentation page; everything below it is the API
    if (pathname.startsWith("/api/") && pathname !== "/api/" && pathname !== "/api/index.html") {
      const apiResponse = await handleApiRequest(request, env, ctx);
      return apiResponse || errorEnvelope("not_found", "Unknown API endpoint.");
    }

    if (pathname.startsWith("/stream/")) {
      const streamResponse = await handleStreamRequest(request, env);
      return streamResponse || errorEnvelope("not_found", "Unknown stream endpoint.");
    }

    if (
      pathname.startsWith("/assets/") ||
      pathname.startsWith("/data/") ||
      pathname.endsWith(".xml") ||
      pathname.endsWith(".txt") ||
      pathname.endsWith(".json") ||
      pathname.endsWith(".md")
    ) {
      const assetResponse = await env.ASSETS.fetch(request);
      if (!assetResponse.ok) return notFoundResponse(env, request);
      const xRobotsTag =
        pathname.startsWith("/data/") ||
        pathname.endsWith(".json") ||
        pathname.endsWith(".md")
          ? "noindex, nofollow"
          : undefined;
      return withSecurityHeaders(assetResponse, { xRobotsTag });
    }

    let response;
    try {
      response = await env.ASSETS.fetch(request);
    } catch {
      return withSecurityHeaders(
        new Response("Service temporarily unavailable", {
          status: 502,
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        }),
        {
          cacheControl: "no-store",
          xRobotsTag: "noindex, nofollow",
        }
      );
    }

    if (!response.ok) {
      return notFoundResponse(env, request);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html")) {
      return withSecurityHeaders(response);
    }

    const geo = normalizeGeo(request.cf || {});
    const themePreference = readThemePreference(request);
    // the live picture (shared, cached a minute) and, with a location, the official alerts at the visitor's
    // point -- fetched together, and waited for only PAGE_LIVE_BUDGET_MS. A warm server answers from memory;
    // a cold one sends the page without them (dashes, the feed's placeholder) and app.js fills them in, while
    // the fetches finish behind the response and warm the server for the next visitor.
    const liveStart = Date.now();
    const live = await withinBudget(
      Promise.all([getLive(env, request, ctx), geo.isReliable ? nwsPointAlerts(geo, ctx) : null]),
      PAGE_LIVE_BUDGET_MS,
      ctx
    );
    const liveMs = Date.now() - liveStart;
    const [liveNow, alerts] = live || [null, null];
    let area = { reliable: false };
    if (geo.isReliable) {
      // past the budget the notice still stands, from the build's own index (a local file, fast); only
      // "Near you" waits for app.js
      area = liveNow
        ? summarizeArea(geo, areaDataOf(liveNow._index), liveNow, alerts)
        : summarizeArea(geo, areaDataOf(await areaIndexOf(env, request)), null, null);
    }

    const transformed = new HTMLRewriter()
      .on("html", new ThemeRootHandler(themePreference))
      .on(".theme-toggle", new ThemeToggleHandler(themePreference))
      .on(".user-marker", new UserMarkerHandler(geo))
      .on(".emergency-banner", new HtmlFillHandler(bannerHtml(area)))
      .on("#near-you", new HtmlFillHandler(liveNow ? nearHtml(area, geo, liveNow) : ""))
      .on("#live-feed", new HtmlFillHandler(liveNow ? feedHtml(liveNow) : ""))
      .on("[data-live]", new LiveStatHandler(liveNow ? statsOf(liveNow) : {}))
      .on("#globe", new GlobeHandler(geo))
      .on(".forecast-age", new ForecastAgeHandler())
      .transform(response);

    const page = withSecurityHeaders(new Response(transformed.body, transformed), {
      cacheControl: HTML_CACHE_CONTROL,
      xRobotsTag: "index, follow",
      contentSecurityPolicy: HTML_CONTENT_SECURITY_POLICY,
      vary: "CF-IPCountry, Accept-Encoding",
    });
    // how long the page waited for the live data, readable in any browser's network panel: a slow page
    // (one took 3.2 s on 2026-10-05 and could not be attributed afterwards) now says where its time went
    page.headers.set(
      "Server-Timing",
      `live;dur=${liveMs};desc="${liveNow ? "live data in the page" : "past the budget, filled by app.js"}"`
    );
    return page;
  },
};

if (typeof globalThis !== "undefined") {
  globalThis.__hazardpulse_worker_test = {
    ThemeRootHandler,
    ThemeToggleHandler,
    UserMarkerHandler,
    GlobeHandler,
    LiveStatHandler,
    HtmlFillHandler,
    ForecastAgeHandler,
    summarizeArea,
    areaDataOf,
    earthquakeCell,
    officialCenter,
    opsSnapshot,
    dispatchScheduler,
    SCHEDULER_DISPATCH,
    getLive,
    publicLive,
    nwsPointAlerts,
    cachedJson,
    liveMemo: _liveMemo,
    resetLiveState() {
      _liveMemo.clear();
      _liveInflight.clear();
      _indexMemo = null;
    },
    LIVE_MAX_AGE_MS,
    statsOf,
    forecastTimeOf,
    LIVE_SOURCES,
    stormOfNhc,
    stormOfForecast,
    freshForecastStorm,
    safeUrl,
    clipText,
    feedHtml,
    nearHtml,
    bannerHtml,
    hasReliableGeo,
    normalizeGeo,
    readThemePreference,
  };
}
