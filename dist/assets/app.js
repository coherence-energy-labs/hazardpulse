/* HazardPulse live app: the globe, and keeping the live numbers current.
   Progressive: every page works without it -- the edge already rendered the live counts, the latest events
   and "Near you"; this script turns the globe on and refreshes them every minute. No third-party code,
   nothing stored, and text from the agencies' feeds is only ever inserted as text. */
(function () {
  "use strict";

  var reduceMotion = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  var LIVE_MS = 60000;
  var NEAR_MS = 300000;
  var TRACKED_MAX_AGE_MS = 3 * 3600000; // the edge's limit too: older tracked storms are not shown as now

  // a real number: JSON's null is not 0 (the global isFinite(null) is true -- a storm whose wind is
  // unknown would read "0 kt", a storm with no position would be drawn at 0, 0)
  function fin(v) { return typeof v === "number" && isFinite(v); }

  // a forecast id carries its issue time: to_fcst_20261004_2059 -> 2026-10-04T20:59Z
  function forecastTime(id) {
    var m = /_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$/.exec(String(id || ""));
    return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) : NaN;
  }

  // ------------------------------------------------------------------ times
  function ago(ms) {
    if (!isFinite(ms)) return "";
    var s = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (s < 60) return "just now";
    var m = Math.round(s / 60);
    if (m < 60) return m + " min ago";
    var h = Math.floor(m / 60);
    if (h < 24) return h + " h " + (m % 60 ? (m % 60) + " min " : "") + "ago";
    var d = Math.floor(h / 24);
    return d + " day" + (d === 1 ? "" : "s") + " ago";
  }

  function refreshTimes() {
    var els = document.querySelectorAll("time[data-relative]");
    for (var i = 0; i < els.length; i++) {
      var t = Date.parse(els[i].getAttribute("datetime"));
      if (isFinite(t)) els[i].textContent = ago(t);
    }
  }

  // ------------------------------------------------------------------ live numbers
  function animateNumber(el, to) {
    var from = parseInt(el.textContent.replace(/[^0-9]/g, ""), 10);
    if (!isFinite(from) || reduceMotion || from === to) { el.textContent = String(to); return; }
    var start = null;
    function step(ts) {
      if (start === null) start = ts;
      var k = Math.min(1, (ts - start) / 700);
      var e = 1 - Math.pow(1 - k, 3);
      el.textContent = String(Math.round(from + (to - from) * e));
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    el.classList.add("live-bump");
    setTimeout(function () { el.classList.remove("live-bump"); }, 1200);
  }

  function getJSON(url) {
    return fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } }).then(function (r) {
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    });
  }

  var globe = null;

  function applyLive(data) {
    if (!data) return;
    var stats = (data.html && data.html.stats) || {};
    var els = document.querySelectorAll("[data-live]");
    for (var i = 0; i < els.length; i++) {
      var key = els[i].getAttribute("data-live");
      // a count the edge left out is unknown (its agency did not answer): a dash, never a stale number
      if (Object.prototype.hasOwnProperty.call(stats, key)) animateNumber(els[i], parseInt(stats[key], 10) || 0);
      else els[i].textContent = "\u2014";
    }
    var feed = document.getElementById("live-feed");
    if (feed && data.html && typeof data.html.feed === "string") feed.innerHTML = data.html.feed;
    var clocks = document.querySelectorAll("[data-live-clock]");
    for (var j = 0; j < clocks.length; j++) {
      clocks[j].textContent = "last checked " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    if (globe) globe.setLive(data);
    refreshTimes();
  }

  function applyNear(data) {
    if (!data || !data.html) return;
    var near = document.getElementById("near-you");
    if (near && data.html.near) near.innerHTML = data.html.near;
    var banner = document.querySelector(".emergency-banner");
    if (banner && typeof data.html.banner === "string") banner.innerHTML = data.html.banner;
    refreshTimes();
  }

  function pollLive() {
    getJSON("/api/v1/now").then(function (env) { applyLive(env.data); }).catch(function () {});
  }

  function pollNear() {
    if (!document.getElementById("near-you")) return;
    getJSON("/api/v1/near").then(function (env) { applyNear(env.data); }).catch(function () {});
  }

  // ------------------------------------------------------------------ globe
  var VS = "attribute vec2 a; varying vec2 v; void main(){ v = a; gl_Position = vec4(a, 0.0, 1.0); }";
  var FS = [
    "precision highp float;",
    "varying vec2 v;",
    "uniform float u_r;",
    "uniform mat3 u_rot;",
    "uniform sampler2D u_land;",
    "uniform sampler2D u_heat;",
    "uniform float u_heatOn;",
    "uniform vec2 u_heatLat;",
    "uniform vec3 u_ocean;",
    "uniform vec3 u_landc;",
    "uniform vec3 u_coast;",
    "uniform vec3 u_glow;",
    "const float PI = 3.14159265;",
    "vec3 heatColor(float h){",
    "  vec3 a = vec3(1.0, 0.82, 0.32); vec3 b = vec3(1.0, 0.48, 0.14); vec3 c = vec3(0.92, 0.12, 0.18);",
    "  return h < 0.5 ? mix(a, b, h * 2.0) : mix(b, c, (h - 0.5) * 2.0);",
    "}",
    "void main(){",
    "  vec2 p = v / u_r;",
    "  float r2 = dot(p, p);",
    "  if (r2 > 1.0) {",
    "    float d = sqrt(r2) - 1.0;",
    "    float a = exp(-d * 10.0) * 0.5 * (1.0 - smoothstep(0.12, 0.2, d));",
    "    gl_FragColor = vec4(u_glow * a, a);",
    "    return;",
    "  }",
    "  float z = sqrt(1.0 - r2);",
    "  vec3 n = vec3(p.x, p.y, z);",
    "  vec3 w = u_rot * n;",
    "  float lat = asin(clamp(w.y, -1.0, 1.0));",
    "  float lon = atan(w.x, w.z);",
    "  vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);",
    "  float land = texture2D(u_land, uv).r;",
    "  vec3 col = mix(u_ocean, u_landc, smoothstep(0.25, 0.75, land));",
    "  float coast = 1.0 - abs(land - 0.5) * 2.0;",
    "  col = mix(col, u_coast, clamp(coast, 0.0, 1.0) * 0.55);",
    "  float latd = degrees(lat); float lond = degrees(lon);",
    "  float gl = min(abs(mod(latd + 15.0, 30.0) - 15.0), abs(mod(lond + 15.0, 30.0) - 15.0) * cos(lat));",
    "  col += vec3(0.05, 0.08, 0.13) * (1.0 - smoothstep(0.12, 0.35, gl / max(z, 0.25)));",
    "  if (u_heatOn > 0.5 && lat > u_heatLat.x && lat < u_heatLat.y) {",
    "    vec2 huv = vec2(lon / (2.0 * PI) + 0.5, (lat - u_heatLat.x) / (u_heatLat.y - u_heatLat.x));",
    "    float h = texture2D(u_heat, huv).r;",
    "    col = mix(col, heatColor(h), smoothstep(0.04, 0.55, h) * 0.9);",
    "  }",
    "  vec3 L = normalize(vec3(-0.55, 0.5, 0.7));",
    "  float diff = max(dot(n, L), 0.0);",
    "  col *= 0.42 + 0.78 * diff;",
    "  col += u_glow * pow(1.0 - z, 3.0) * 0.75;",
    "  gl_FragColor = vec4(col, 1.0);",
    "}"
  ].join("\n");

  function compile(gl, type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) || "shader");
    return sh;
  }

  function rad(d) { return d * Math.PI / 180; }

  // view -> world rotation: Ry(lon0) * Rx(-lat0), as a row-major 3x3
  function rotation(lat0, lon0) {
    var a = rad(-lat0), c = rad(lon0);
    var ca = Math.cos(a), sa = Math.sin(a), cc = Math.cos(c), sc = Math.sin(c);
    // Rx(a) = [1,0,0; 0,ca,-sa; 0,sa,ca]; Ry(c) = [cc,0,sc; 0,1,0; -sc,0,cc]
    return [
      cc, sc * sa, sc * ca,
      0, ca, -sa,
      -sc, cc * sa, cc * ca
    ];
  }

  function worldOf(lat, lon) {
    var p = rad(lat), l = rad(lon);
    return [Math.cos(p) * Math.sin(l), Math.sin(p), Math.cos(p) * Math.cos(l)];
  }

  function Globe(el) {
    var wrap = el.closest(".globe-wrap");
    var glc = el.querySelector(".globe-gl");
    var ov = el.querySelector(".globe-overlay");
    var tip = el.querySelector(".globe-tip");
    var gl = glc.getContext("webgl", { antialias: true, premultipliedAlpha: true, alpha: true });
    if (!gl) throw new Error("no webgl");
    var ctx = ov.getContext("2d");
    var prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link");
    gl.useProgram(prog);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    var U = {};
    ["u_r", "u_rot", "u_land", "u_heat", "u_heatOn", "u_heatLat", "u_ocean", "u_landc", "u_coast", "u_glow"].forEach(function (n) {
      U[n] = gl.getUniformLocation(prog, n);
    });
    gl.uniform3f(U.u_ocean, 0.022, 0.05, 0.115);
    gl.uniform3f(U.u_landc, 0.17, 0.27, 0.41);
    gl.uniform3f(U.u_coast, 0.45, 0.66, 0.92);
    gl.uniform3f(U.u_glow, 0.33, 0.6, 1.0);
    gl.uniform1i(U.u_land, 0);
    gl.uniform1i(U.u_heat, 1);
    gl.uniform1f(U.u_heatOn, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    function texture(unit) {
      var t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }
    var landTex = texture(0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    var heatTex = texture(1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([0]));

    var userLat = parseFloat(el.getAttribute("data-user-lat"));
    var userLon = parseFloat(el.getAttribute("data-user-lon"));
    var hasUser = isFinite(userLat) && isFinite(userLon);
    var view = {
      lat: hasUser ? Math.max(-45, Math.min(45, userLat)) : 18,
      lon: hasUser ? userLon : -60,
      vlon: 0, vlat: 0
    };
    var markers = [];
    var hits = [];
    var tracked = [];
    var dpr = 1, size = 0, radiusPx = 0;
    var R_NDC = 0.86;
    var dragging = null, lastInput = 0, lastFrame = 0, running = false, pinned = null;
    var onScreen = true, pageVisible = !document.hidden;

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      var css = el.clientWidth;
      size = Math.max(1, Math.round(css * dpr));
      glc.width = glc.height = ov.width = ov.height = size;
      gl.viewport(0, 0, size, size);
      radiusPx = R_NDC * size / 2;
      draw(performance.now());
    }

    // world -> screen (device pixels); null when on the far side
    function project(lat, lon, m) {
      var w = worldOf(lat, lon);
      // view = M^T * world
      var x = m[0] * w[0] + m[3] * w[1] + m[6] * w[2];
      var y = m[1] * w[0] + m[4] * w[1] + m[7] * w[2];
      var z = m[2] * w[0] + m[5] * w[1] + m[8] * w[2];
      if (z < 0.06) return null;
      return { x: size / 2 + x * radiusPx, y: size / 2 - y * radiusPx, z: z };
    }

    function quakeColor(q, now) {
      var age = (now - q.time) / 3600000;
      if (age < 1) return "255,196,64";
      if (age < 6) return "255,140,60";
      return "255,110,80";
    }

    function draw(t) {
      var m = rotation(view.lat, view.lon);
      gl.uniformMatrix3fv(U.u_rot, false, [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
      gl.uniform1f(U.u_r, R_NDC);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      ctx.clearRect(0, 0, size, size);
      var now = Date.now();
      var s = dpr;
      hits = [];
      // tracked thunderstorms: faint points
      for (var i = 0; i < tracked.length; i++) {
        var tp = project(tracked[i].lat, tracked[i].lon, m);
        if (!tp) continue;
        ctx.fillStyle = "rgba(255,170,90," + (0.35 * tp.z + 0.15) + ")";
        ctx.beginPath(); ctx.arc(tp.x, tp.y, 1.4 * s, 0, 6.2832); ctx.fill();
      }
      for (var k = 0; k < markers.length; k++) {
        var mk = markers[k];
        var p = project(mk.lat, mk.lon, m);
        if (!p) continue;
        var alpha = Math.min(1, p.z * 1.6);
        if (mk.type === "quake") {
          var r = (2.2 + Math.max(0, mk.mag - 2.5) * 1.9) * s;
          var rgb = quakeColor(mk, now);
          if (!reduceMotion && now - mk.time < 3 * 3600000) {
            var ph = ((t / 1600) + (mk.time % 997) / 997) % 1;
            ctx.strokeStyle = "rgba(" + rgb + "," + (0.55 * (1 - ph) * alpha) + ")";
            ctx.lineWidth = 1.5 * s;
            ctx.beginPath(); ctx.arc(p.x, p.y, r + ph * 14 * s, 0, 6.2832); ctx.stroke();
          }
          ctx.fillStyle = "rgba(" + rgb + "," + (0.85 * alpha) + ")";
          ctx.strokeStyle = "rgba(10,15,30," + (0.6 * alpha) + ")";
          ctx.lineWidth = 1 * s;
          ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
          hits.push({ x: p.x, y: p.y, r: Math.max(r, 7 * s), m: mk });
        } else if (mk.type === "storm") {
          var sr = 10 * s;
          var spin = reduceMotion ? 0 : (t / 900) * (mk.lat < 0 ? 1 : -1);
          var hot = mk.wind_kt >= 96 ? "255,92,138" : mk.wind_kt >= 64 ? "120,180,255" : "150,205,255";
          ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(spin);
          ctx.strokeStyle = "rgba(" + hot + "," + alpha + ")"; ctx.lineWidth = 2.2 * s; ctx.lineCap = "round";
          for (var arm = 0; arm < 2; arm++) {
            ctx.beginPath(); ctx.arc(0, 0, sr, arm * Math.PI, arm * Math.PI + 1.9); ctx.stroke();
          }
          ctx.fillStyle = "rgba(" + hot + "," + alpha + ")";
          ctx.beginPath(); ctx.arc(0, 0, 3.2 * s, 0, 6.2832); ctx.fill();
          ctx.restore();
          ctx.font = "600 " + (11.5 * s) + "px Inter, system-ui, sans-serif";
          ctx.fillStyle = "rgba(230,238,250," + alpha + ")";
          ctx.shadowColor = "rgba(0,0,0,0.8)"; ctx.shadowBlur = 4 * s;
          ctx.fillText(mk.name, p.x + sr + 5 * s, p.y + 4 * s);
          ctx.shadowBlur = 0;
          hits.push({ x: p.x, y: p.y, r: 14 * s, m: mk });
        } else if (mk.type === "warning") {
          var d = 6 * s;
          ctx.fillStyle = "rgba(255,70,90," + alpha + ")";
          ctx.strokeStyle = "rgba(255,255,255," + (0.8 * alpha) + ")"; ctx.lineWidth = 1.2 * s;
          ctx.beginPath(); ctx.moveTo(p.x, p.y - d); ctx.lineTo(p.x + d, p.y); ctx.lineTo(p.x, p.y + d); ctx.lineTo(p.x - d, p.y); ctx.closePath();
          ctx.fill(); ctx.stroke();
          hits.push({ x: p.x, y: p.y, r: 9 * s, m: mk });
        }
      }
      if (hasUser) {
        var u = project(userLat, userLon, m);
        if (u) {
          var ph2 = reduceMotion ? 0.5 : (t / 2000) % 1;
          ctx.strokeStyle = "rgba(94,234,212," + (0.7 * (1 - ph2)) + ")"; ctx.lineWidth = 2 * s;
          ctx.beginPath(); ctx.arc(u.x, u.y, (5 + ph2 * 16) * s, 0, 6.2832); ctx.stroke();
          ctx.fillStyle = "#5eead4"; ctx.strokeStyle = "#04121f"; ctx.lineWidth = 1.5 * s;
          ctx.beginPath(); ctx.arc(u.x, u.y, 4.5 * s, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.font = "700 " + (11 * s) + "px Inter, system-ui, sans-serif";
          ctx.fillStyle = "#d9fffa"; ctx.shadowColor = "rgba(0,0,0,0.8)"; ctx.shadowBlur = 4 * s;
          ctx.fillText("You", u.x + 8 * s, u.y - 7 * s); ctx.shadowBlur = 0;
          hits.push({ x: u.x, y: u.y, r: 10 * s, m: { type: "user" } });
        }
      }
    }

    function loop(t) {
      running = false;
      if (!onScreen || !pageVisible) return;
      var idle = Date.now() - lastInput > 3500;
      var coasting = Math.abs(view.vlon) > 0.002 || Math.abs(view.vlat) > 0.002;
      var turning = !dragging && !coasting && idle && !reduceMotion && !pinned;
      // the slow idle turn and the pulses need no more than 30 frames a second; a drag gets every frame
      if (!dragging && !coasting && t - lastFrame < 32) { schedule(); return; }
      var dt = lastFrame ? Math.min(64, t - lastFrame) : 16;
      lastFrame = t;
      if (!dragging) {
        if (coasting) {
          view.lon += view.vlon; view.lat = clampLat(view.lat + view.vlat);
          view.vlon *= 0.94; view.vlat *= 0.94;
        } else if (turning) {
          view.lon += 0.0022 * dt;
          if (!tip.hidden) tip.hidden = true; // a hover tip would no longer point at its marker
        }
      }
      draw(t);
      if (!reduceMotion || dragging || coasting) schedule();
    }

    function schedule() {
      if (!running) { running = true; requestAnimationFrame(loop); }
    }

    function clampLat(v) { return Math.max(-70, Math.min(70, v)); }

    function degPerPx() { return 180 / (Math.PI * radiusPx / dpr); }

    function hitAt(clientX, clientY) {
      var rect = ov.getBoundingClientRect();
      var x = (clientX - rect.left) * dpr, y = (clientY - rect.top) * dpr;
      var best = null;
      for (var i = 0; i < hits.length; i++) {
        var dx = hits[i].x - x, dy = hits[i].y - y, d2 = dx * dx + dy * dy;
        if (d2 <= hits[i].r * hits[i].r && (!best || d2 < best.d2)) best = { h: hits[i], d2: d2 };
      }
      return best ? best.h : null;
    }

    function line(text, cls) {
      var p = document.createElement("p");
      if (cls) p.className = cls;
      p.textContent = text;
      return p;
    }

    function showTip(h, pin) {
      if (!h) { if (!pinned) tip.hidden = true; return; }
      var mk = h.m;
      tip.textContent = "";
      if (mk.type === "user") {
        tip.appendChild(line("You are here", "globe-tip-title"));
        tip.appendChild(line("Your approximate location, from your connection. Not stored."));
      } else if (mk.type === "quake") {
        tip.appendChild(line("M" + mk.mag.toFixed(1) + " earthquake", "globe-tip-title"));
        tip.appendChild(line(mk.place));
        tip.appendChild(line(ago(mk.time) + (fin(mk.depth_km) ? " \u00b7 " + Math.round(mk.depth_km) + " km deep" : ""), "globe-tip-meta"));
      } else if (mk.type === "storm") {
        tip.appendChild(line(mk.name, "globe-tip-title"));
        tip.appendChild(line(mk.kind + (mk.category ? ", " + mk.category : "") + (fin(mk.wind_kt) ? " \u00b7 " + Math.round(mk.wind_kt) + " kt" : "")));
        if (fin(mk.ri_probability)) {
          tip.appendChild(line("Chance of rapid intensification, 24 h: " + (mk.ri_probability < 0.001 ? "<0.1%" : (mk.ri_probability * 100).toFixed(1) + "%"), "globe-tip-meta"));
        }
        tip.appendChild(line("Source: " + mk.source, "globe-tip-meta"));
      } else if (mk.type === "warning") {
        tip.appendChild(line(mk.event, "globe-tip-title"));
        tip.appendChild(line(mk.area.slice(0, 160)));
        tip.appendChild(line("Issued by the " + mk.sender, "globe-tip-meta"));
      }
      if (mk.url && /^https:\/\//.test(mk.url)) {
        var a = document.createElement("a");
        a.href = mk.url; a.rel = "noopener"; a.textContent = "Official details";
        tip.appendChild(a);
      }
      tip.hidden = false;
      var x = h.x / dpr, y = h.y / dpr, w = el.clientWidth;
      tip.style.left = "0px"; // measure at a known place first
      var tw = tip.offsetWidth;
      tip.style.left = Math.max(4, Math.min(w - tw - 4, x - tw / 2)) + "px";
      tip.style.top = Math.max(4, y + 16) + "px";
      pinned = pin ? h : null;
    }

    ov.addEventListener("pointerdown", function (e) {
      if (!pinned) tip.hidden = true;
      dragging = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
      ov.setPointerCapture(e.pointerId);
      lastInput = Date.now();
      view.vlon = view.vlat = 0;
      schedule();
    });
    ov.addEventListener("pointermove", function (e) {
      if (dragging && dragging.id === e.pointerId) {
        var dx = e.clientX - dragging.x, dy = e.clientY - dragging.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragging.moved = true;
        var k = degPerPx();
        view.lon -= dx * k; view.lat = clampLat(view.lat + dy * k);
        view.vlon = -dx * k * 0.6; view.vlat = dy * k * 0.6;
        dragging.x = e.clientX; dragging.y = e.clientY;
        lastInput = Date.now();
        if (pinned) { pinned = null; tip.hidden = true; }
        schedule();
      } else if (e.pointerType === "mouse") {
        var over = hitAt(e.clientX, e.clientY);
        lastInput = Date.now();
        if (!pinned) showTip(over, false);
        ov.style.cursor = over ? "pointer" : "grab";
      }
    });
    function endDrag(e) {
      if (!dragging) return;
      var tapped = !dragging.moved;
      dragging = null;
      lastInput = Date.now();
      if (tapped) {
        var h = hitAt(e.clientX, e.clientY);
        if (h) showTip(h, true); else { pinned = null; tip.hidden = true; }
      }
      schedule();
    }
    ov.addEventListener("pointerup", endDrag);
    ov.addEventListener("pointercancel", endDrag);
    ov.addEventListener("pointerleave", function () { if (!pinned) tip.hidden = true; });
    ov.tabIndex = 0;
    ov.addEventListener("keydown", function (e) {
      var step = 6;
      if (e.key === "ArrowLeft") view.lon -= step;
      else if (e.key === "ArrowRight") view.lon += step;
      else if (e.key === "ArrowUp") view.lat = clampLat(view.lat + step);
      else if (e.key === "ArrowDown") view.lat = clampLat(view.lat - step);
      else return;
      e.preventDefault();
      lastInput = Date.now();
      draw(performance.now());
    });

    // the land texture, the forecast heat and the live markers
    var img = new Image();
    img.onload = function () {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, landTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE, gl.UNSIGNED_BYTE, img);
      el.classList.add("globe-ready");
      draw(performance.now());
    };
    img.src = el.getAttribute("data-land");

    getJSON(el.getAttribute("data-area") || "/data/area-index.json").then(function (idx) {
      var trackedAt = forecastTime(idx && idx.forecasts && idx.forecasts.to);
      if (idx && Array.isArray(idx.tornadoes) && Date.now() - trackedAt <= TRACKED_MAX_AGE_MS) {
        tracked = idx.tornadoes.filter(function (s) { return fin(s.lat) && fin(s.lon); });
      }
      var eq = idx && idx.eq;
      if (!eq || eq.gate === "block" || !eq.forecast_domain || typeof eq.probability_grid !== "string") return;
      var dom = eq.forecast_domain, vals = eq.probability_grid.split(",");
      var nx = dom.n_lon | 0, ny = dom.n_lat | 0;
      // the shader maps the grid's columns onto the whole globe from 180 W: any other grid is not drawn
      if (vals.length !== nx * ny || dom.lon_min !== -180 || Math.abs(nx * dom.dlon - 360) > 1e-6) return;
      var data = new Uint8Array(nx * ny);
      for (var i = 0; i < vals.length; i++) {
        var p = parseFloat(vals[i]);
        // logarithmic: 0.05% -> 0, 25% -> 255
        var v = p > 0 ? (Math.log(p) / Math.LN10 + 3.3) / 2.7 : 0;
        data[i] = Math.max(0, Math.min(255, Math.round(v * 255)));
      }
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, heatTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, nx, ny, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
      gl.uniform2f(U.u_heatLat, rad(dom.lat_min), rad(dom.lat_min + ny * dom.dlat));
      gl.uniform1f(U.u_heatOn, 1);
      draw(performance.now());
    }).catch(function () {});

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        onScreen = entries[entries.length - 1].isIntersecting;
        if (onScreen) schedule();
      }).observe(el);
    }
    document.addEventListener("visibilitychange", function () {
      pageVisible = !document.hidden;
      if (pageVisible) schedule();
    });
    glc.addEventListener("webglcontextlost", function (e) {
      // the GPU took the context back: stop drawing and show the map rather than a frozen globe
      e.preventDefault();
      onScreen = false;
      el.hidden = true;
      if (wrap) wrap.classList.remove("has-globe");
    });
    window.addEventListener("resize", resize);

    el.hidden = false;
    if (wrap) wrap.classList.add("has-globe");
    resize();
    schedule();

    return {
      setLive: function (live) {
        var next = [];
        (live.quakes && live.quakes.day || []).forEach(function (q) { if (fin(q.lat) && fin(q.lon) && fin(q.mag)) next.push({ type: "quake", lat: q.lat, lon: q.lon, mag: q.mag, time: q.time, place: q.place, depth_km: q.depth_km, url: q.url }); });
        (live.storms || []).forEach(function (st) { if (fin(st.lat) && fin(st.lon)) next.push({ type: "storm", lat: st.lat, lon: st.lon, name: st.name, kind: st.kind, category: st.category, wind_kt: st.wind_kt, ri_probability: st.ri_probability, source: st.source, url: st.url }); });
        (live.tornado_warnings || []).forEach(function (w) { if (fin(w.lat) && fin(w.lon)) next.push({ type: "warning", lat: w.lat, lon: w.lon, event: w.event, area: w.area, sender: w.sender, url: w.url }); });
        next.sort(function (a, b) { return (a.type === "quake" ? a.mag : 99) - (b.type === "quake" ? b.mag : 99); });
        markers = next;
        draw(performance.now());
        schedule();
      }
    };
  }

  // ------------------------------------------------------------------ start
  function start() {
    var el = document.getElementById("globe");
    if (el) {
      try { globe = Globe(el); } catch (e) { globe = null; } // no WebGL: the static map stays
    }
    if (document.querySelector("[data-live]") || el) {
      pollLive();
      setInterval(function () { if (!document.hidden) pollLive(); }, LIVE_MS);
    }
    var near = document.getElementById("near-you");
    if (near) {
      // a page sent before the live data arrived (a cold server) has an empty "Near you": fill it now
      if (!near.firstElementChild) pollNear();
      setInterval(function () { if (!document.hidden) pollNear(); }, NEAR_MS);
    }
    refreshTimes();
    setInterval(refreshTimes, 30000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
