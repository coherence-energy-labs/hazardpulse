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
// Personalisation at the edge: the visitor's approximate location (request.cf), used for this one
// response and never stored. Everything below is factual -- distances, names and published chances --
// and never an instruction: official agencies issue warnings, HazardPulse does not.
// ---------------------------------------------------------------------------------------------

const AREA = {
  hurricaneBannerKm: 500, // an active tropical cyclone this close puts a notice at the top of every page
  tornadoBannerKm: 50, // ...as does a tracked storm this close under an NWS tornado warning
  tornadoBannerChance: 0.1, // ...or with at least this chance of a tornado within 60 minutes
  stormNearbyKm: 2000, // "Near you" names the nearest tropical cyclone within this distance
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

function formatChance(p) {
  const v = Number(p);
  if (!Number.isFinite(v)) return "&mdash;";
  if (v <= 0) return "0%";
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

function stormName(storm) {
  const name = String(storm.storm_name || "").trim();
  if (!name || ["INVEST", "UNNAMED", "NONAME"].includes(name.toUpperCase())) {
    return String(storm.storm_id || "An unnamed storm");
  }
  return name === name.toUpperCase() ? name.charAt(0) + name.slice(1).toLowerCase() : name;
}

// the agency that issues the official forecast where the storm is now (an East Pacific storm west of
// 140 W has passed to the Central Pacific Hurricane Center)
function officialCenter(storm) {
  const basin = String(storm.basin || "").toUpperCase();
  const lon = Number(storm.lon);
  if (basin === "AL" || ((basin === "EP" || basin === "CP") && lon >= -140)) {
    return ["National Hurricane Center", "https://www.nhc.noaa.gov/"];
  }
  if (basin === "EP" || basin === "CP") {
    return ["Central Pacific Hurricane Center", "https://www.nhc.noaa.gov/?cpac"];
  }
  return ["Joint Typhoon Warning Center", "https://www.metoc.navy.mil/jtwc/jtwc.html"];
}

// one small file the site build writes for this purpose (hazardpulse.site.build.area_index): the earthquake
// grid and the storm positions, ~0.15 MB, instead of parsing 1.5 MB of forecast files on every page view
async function loadAreaData(env, request) {
  const idx = await fetchAssetJson(env, "/data/area-index.json", request);
  if (!idx) return { eq: null, eqGate: null, storms: [], tornadoes: [] };
  const eq = idx.eq && idx.eq.forecast_domain && idx.eq.probability_grid ? idx.eq : null;
  return {
    eq,
    eqGate: eq ? eq.gate : null,
    storms: Array.isArray(idx.storms) ? idx.storms : [],
    tornadoes: Array.isArray(idx.tornadoes) ? idx.tornadoes : [],
  };
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
    const ilat = Number(item.lat);
    const ilon = Number(item.lon);
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

function summarizeArea(geo, data) {
  if (!geo.isReliable) return { reliable: false };
  const lat = geo.latitude;
  const lon = geo.longitude;
  const out = {
    reliable: true,
    lat,
    lon,
    eqCell: data.eqGate === "block" ? null : earthquakeCell(lat, lon, data.eq),
    storm: nearest(lat, lon, data.storms, AREA.stormNearbyKm),
    tornado: nearest(lat, lon, data.tornadoes, AREA.tornadoNearbyKm),
    banner: null,
  };
  const near = nearest(lat, lon, data.storms, AREA.hurricaneBannerKm);
  if (near) {
    out.banner = { kind: "hurricane", ...near };
  } else {
    const close = data.tornadoes
      .map((s) => ({ item: s, km: haversineKm(lat, lon, Number(s.lat), Number(s.lon)) }))
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

// a live page is rendered when its forecast is published; when it is viewed after the forecast's own window
// has passed (a 60-minute tornado forecast seen three hours later), the page says so
class ForecastAgeHandler {
  constructor(now = new Date()) {
    this.now = now;
  }

  element(el) {
    const issued = Date.parse(el.getAttribute("data-issued") || "");
    const windowMin = Number(el.getAttribute("data-window-minutes"));
    if (!Number.isFinite(issued) || !Number.isFinite(windowMin) || windowMin <= 0) return;
    const ageMin = (this.now.getTime() - issued) / 60000;
    if (ageMin <= windowMin) return;
    const schedule = escapeHtml(el.getAttribute("data-schedule") || "");
    const ago = ageMin >= 120 ? `${Math.floor(ageMin / 60)} hours` : `${Math.round(ageMin)} minutes`;
    const span = windowMin >= 120 ? `${Math.round(windowMin / 60)}-hour` : `${windowMin}-minute`;
    el.setInnerContent(
      `<strong>This forecast was issued ${ago} ago, so its ${span} window has passed.</strong> ` +
        `A new one is published ${schedule || "on schedule"}; until then, treat these numbers as out of date.`,
      { html: true }
    );
  }
}

class EmergencyBannerHandler {
  constructor(area) {
    this.area = area;
  }

  element(el) {
    const b = this.area && this.area.banner;
    if (!b) return;
    let html = "";
    if (b.kind === "hurricane") {
      const s = b.item;
      const [center, url] = officialCenter(s);
      const strength = escapeHtml(s.category || "");
      html =
        `<strong class="emergency-title">Tropical cyclone nearby.</strong> ${escapeHtml(stormName(s))}` +
        (strength ? ` (${strength})` : "") +
        ` is about ${distanceText(b.km)} ${bearing(this.area.lat, this.area.lon, s.lat, s.lon)} of your ` +
        `approximate location. For its track and any warnings for your area, follow the ` +
        `<a href="${url}" rel="noopener">${center}</a> and your local officials.`;
    } else if (b.kind === "tornado") {
      const s = b.item;
      const warned = tornadoWarned(s);
      html =
        `<strong class="emergency-title">${warned ? "Tornado warning nearby." : "Storm nearby."}</strong> ` +
        `A thunderstorm about ${distanceText(b.km)} from your approximate location ` +
        (warned
          ? "is under a National Weather Service tornado warning. "
          : `has a ${formatChance(s.tornado_probability)} chance of producing a tornado in the next hour ` +
            "(HazardPulse research forecast). ") +
        `Check <a href="https://www.weather.gov/" rel="noopener">weather.gov</a> for the warnings in effect where you are.`;
    }
    if (!html) return;
    el.setInnerContent(
      `<div class="container emergency-inner"><span class="emergency-dot" aria-hidden="true"></span><p>${html}</p></div>`,
      { html: true }
    );
  }
}

class YourAreaHandler {
  constructor(area, geo) {
    this.area = area;
    this.geo = geo;
  }

  element(el) {
    const { area, geo } = this;
    if (!area || !area.reliable) return; // the section stays empty, and hidden
    const cards = [];

    const c = area.eqCell;
    if (c && !c.outside) {
      cards.push(
        `<div class="card hz-eq"><span class="hazard-tag eq"><span class="hazard-dot eq" aria-hidden="true"></span>Earthquake</span>` +
          `<p class="area-figure">${chanceHtml(c.p)}</p>` +
          `<p>chance of a magnitude 6+ earthquake in your 2&deg; grid cell in the next 30 days.</p>` +
          `<a class="card-cta" href="/live/earthquake/">Earthquake forecast</a></div>`
      );
    } else if (c && c.outside) {
      cards.push(
        `<div class="card hz-eq"><span class="hazard-tag eq"><span class="hazard-dot eq" aria-hidden="true"></span>Earthquake</span>` +
          `<p>Your location is outside the forecast&rsquo;s grid (60&deg;S to 70&deg;N).</p></div>`
      );
    }

    const s = area.storm;
    if (s) {
      const [center, url] = officialCenter(s.item);
      cards.push(
        `<div class="card hz-hu"><span class="hazard-tag hu"><span class="hazard-dot hu" aria-hidden="true"></span>Tropical cyclone</span>` +
          `<p class="area-figure">${distanceText(s.km)}</p>` +
          `<p>to ${escapeHtml(stormName(s.item))}${s.item.category ? ` (${escapeHtml(s.item.category)})` : ""}, the nearest active tropical cyclone. ` +
          `Official forecast: <a href="${url}" rel="noopener">${center}</a>.</p>` +
          `<a class="card-cta" href="/live/hurricane/">Hurricane forecast</a></div>`
      );
    } else {
      cards.push(
        `<div class="card hz-hu"><span class="hazard-tag hu"><span class="hazard-dot hu" aria-hidden="true"></span>Tropical cyclone</span>` +
          `<p>No active tropical cyclone within ${AREA.stormNearbyKm.toLocaleString("en-US")} km.</p></div>`
      );
    }

    const t = area.tornado;
    if (t) {
      cards.push(
        `<div class="card hz-to"><span class="hazard-tag to"><span class="hazard-dot to" aria-hidden="true"></span>Thunderstorms</span>` +
          `<p class="area-figure">${chanceHtml(t.item.tornado_probability)}</p>` +
          `<p>chance of a tornado within the next hour from the nearest tracked thunderstorm, about ${distanceText(t.km)} away` +
          (tornadoWarned(t.item) ? ", which is under a National Weather Service tornado warning" : "") +
          `.</p><a class="card-cta" href="/live/tornado/">Tornado forecast</a></div>`
      );
    } else {
      cards.push(
        `<div class="card hz-to"><span class="hazard-tag to"><span class="hazard-dot to" aria-hidden="true"></span>Thunderstorms</span>` +
          `<p>No thunderstorm tracked within ${AREA.tornadoNearbyKm} km. Tornado forecasts cover the contiguous US.</p></div>`
      );
    }

    el.setInnerContent(
      `<div class="container"><div class="section-head"><div><h2 id="your-area-heading">Near you</h2>` +
        `<p>Based on your approximate location, ${placeText(geo)}, worked out for this page view only and never stored. ` +
        `For warnings, follow your local officials.</p></div></div>` +
        `<div class="your-area-grid">${cards.join("")}</div></div>`,
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

async function handleApiRequest(request, env) {
  if (!_checkRateLimit(request)) {
    return errorEnvelope("rate_limited", "Too many requests from this address. Try again in a minute.", 429);
  }

  const url = new URL(request.url);
  const path = url.pathname;

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

export default {
  async fetch(request, env) {
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
      const apiResponse = await handleApiRequest(request, env);
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
    // the hazard data is read only when there is a location to compare it with
    const area = geo.isReliable ? summarizeArea(geo, await loadAreaData(env, request)) : { reliable: false };

    const transformed = new HTMLRewriter()
      .on("html", new ThemeRootHandler(themePreference))
      .on(".theme-toggle", new ThemeToggleHandler(themePreference))
      .on(".user-marker", new UserMarkerHandler(geo))
      .on(".emergency-banner", new EmergencyBannerHandler(area))
      .on(".your-area-section", new YourAreaHandler(area, geo))
      .on(".forecast-age", new ForecastAgeHandler())
      .transform(response);

    return withSecurityHeaders(new Response(transformed.body, transformed), {
      cacheControl: HTML_CACHE_CONTROL,
      xRobotsTag: "index, follow",
      contentSecurityPolicy: HTML_CONTENT_SECURITY_POLICY,
      vary: "CF-IPCountry, Accept-Encoding",
    });
  },
};

if (typeof globalThis !== "undefined") {
  globalThis.__hazardpulse_worker_test = {
    ThemeRootHandler,
    ThemeToggleHandler,
    UserMarkerHandler,
    EmergencyBannerHandler,
    YourAreaHandler,
    summarizeArea,
    opsSnapshot,
    ForecastAgeHandler,
    earthquakeCell,
    officialCenter,
    hasReliableGeo,
    normalizeGeo,
    readThemePreference,
  };
}
