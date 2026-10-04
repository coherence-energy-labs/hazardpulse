/* HazardPulse site shell: the colour theme. Loaded in <head> so the saved theme applies before the
   first paint. The edge also sets data-theme from the cookie, so a page renders in the right theme
   even before this runs. Nothing else lives here: every page works without JavaScript. */
(function () {
  "use strict";
  var KEY = "hp_theme";
  var root = document.documentElement;

  function saved() {
    var m = document.cookie.match(/(?:^|;\s*)hp_theme=(dark|light)(?:;|$)/);
    if (m) return m[1];
    try {
      var s = window.localStorage.getItem(KEY);
      if (s === "dark" || s === "light") return s;
    } catch (e) {}
    return null;
  }

  function systemDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function current() {
    var t = root.getAttribute("data-theme");
    if (t === "dark" || t === "light") return t;
    return systemDark() ? "dark" : "light";
  }

  function apply(theme, persist) {
    root.setAttribute("data-theme", theme);
    if (persist) {
      try { window.localStorage.setItem(KEY, theme); } catch (e) {}
      document.cookie = "hp_theme=" + theme + "; path=/; max-age=31536000; SameSite=Lax" +
        (location.protocol === "https:" ? "; Secure" : "");
    }
    var buttons = document.querySelectorAll(".theme-toggle");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      buttons[i].setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    }
  }

  var s = saved();
  if (s) root.setAttribute("data-theme", s);

  document.addEventListener("DOMContentLoaded", function () {
    apply(current(), false);
    document.addEventListener("click", function (ev) {
      var btn = ev.target.closest && ev.target.closest(".theme-toggle");
      if (!btn) return;
      apply(current() === "dark" ? "light" : "dark", true);
    });
  });
})();
