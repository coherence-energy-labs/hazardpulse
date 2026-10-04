"""Build every page of hazardpulse.com from the published artifacts.

    python -m hazardpulse.site.build            # render into dist/
    python -m hazardpulse.site.build --check    # exit 1 if any page differs from what the artifacts say

``build_site`` is the ONE renderer: every scorer reaches it through ``build_site_artifacts()``, so whichever
hazard published last, every page is rendered from the same files by the same code. Its output is a fixed
point -- building twice changes nothing -- which is what ``check_site`` (and tests/test_site_pages.py) asserts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Callable

from hazardpulse.site import fmt, maps, shell
from hazardpulse.site.data import SiteData
from hazardpulse.site.pages import earthquake, hurricane, overview, record, research, tornado

GENERATED: dict[str, Callable[[SiteData], str]] = {
    "/": overview.home,
    "/live/": overview.live,
    "/live/earthquake/": earthquake.page,
    "/live/hurricane/": hurricane.page,
    "/live/tornado/": tornado.page,
    "/verification/": record.verification,
    "/evidence/": record.evidence,
    "/ops/status/": record.status,
    "/verification/cross-modality/": research.page,
}

# pages whose evidence blocks (hp-evidence markers) are re-rendered from the served-model evidence
EVIDENCE_PAGES = ("methods/index.html", "registry/index.html", "verification/tornado/index.html")


def page_file(dist: Path, path: str) -> Path:
    return shell.static_page_file(path, dist)


def _withheld(d: SiteData, path: str) -> str | None:
    """A live page whose current forecast was BLOCKED by the quality checks is withheld (the gate engine's
    contract: block -> not published to public surfaces). The page says why, and nothing else."""
    key = {"/live/earthquake/": "eq", "/live/hurricane/": "hu", "/live/tornado/": "to"}.get(path)
    if key is None or d.headlines[key].gate != "block":
        return None
    from hazardpulse.site.hazards import HAZARDS
    from hazardpulse.site.pages import common
    head = d.headlines[key]
    reasons = "".join(f"<li>{fmt.esc(common.plain_gate_note(n))}</li>" for n in head.gate_notes)
    hero = common.hero(f"{HAZARDS[key].name}", "This forecast is withheld",
                       "The current forecast failed a blocking quality check, so it is not shown. The next forecast "
                       "will replace it when it passes.",
                       meta=[("Forecast", f"<code>{fmt.esc(head.forecast_id)}</code>"),
                             ("Issued", fmt.time_tag(head.issued_at))])
    body = common.section("why", "Why", f'<div class="notice notice-bad"><ul>{reasons}</ul></div>'
                          + common.official_notice(HAZARDS[key]))
    return f'<main id="main" class="page">{hero}{body}</main>'


def render(d: SiteData) -> dict[str, str]:
    """Every generated page, as the full document it should be."""
    out = {}
    for path, fn in GENERATED.items():
        main = _withheld(d, path) or fn(d)
        out[path] = shell.document(path, main, extra_head=shell.jsonld_for(path, d))
    return out


def _write(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def sitemap_and_feed(d: SiteData, dist: Path) -> list[str]:
    changed = []
    lastmod = (fmt.parse_time(d.pulse.get("updated_at")) or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%d")
    urls = []
    for p in shell.PAGES.values():
        if p.noindex:
            continue
        live = p.path in ("/", "/live/", "/live/earthquake/", "/live/hurricane/", "/live/tornado/", "/evidence/",
                          "/verification/", "/ops/status/")
        urls.append(f"  <url><loc>{shell.DOMAIN}{p.path}</loc><lastmod>{lastmod}</lastmod>"
                    f"<changefreq>{'hourly' if live else 'weekly'}</changefreq>"
                    f"<priority>{'1.0' if p.path == '/' else ('0.9' if live else '0.6')}</priority></url>")
    sitemap = ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               + "\n".join(urls) + "\n</urlset>\n")
    if _write(dist / "sitemap.xml", sitemap):
        changed.append("sitemap.xml")
    items = []
    for key, name, unit in (("eq", "Earthquake", "chance of an M6+ earthquake in a 2-degree cell, next 30 days"),
                            ("hu", "Hurricane", "chance of rapid intensification, next 24 hours"),
                            ("to", "Tornado", "chance a tracked US storm produces a tornado, next 60 minutes")):
        h = d.headlines[key]
        if not h.forecast_id or h.probability is None:
            continue
        t = fmt.parse_time(h.issued_at) or dt.datetime.now(dt.timezone.utc)
        where = fmt.esc(h.where)
        items.append(
            f"    <item>\n      <title>{name}: {fmt.pct_plain(h.probability)} highest {unit}</title>\n"
            f"      <link>{shell.DOMAIN}/live/{['earthquake', 'hurricane', 'tornado'][['eq', 'hu', 'to'].index(key)]}/</link>\n"
            f'      <guid isPermaLink="false">{fmt.esc(h.forecast_id)}</guid>\n'
            f"      <pubDate>{t.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>\n"
            f"      <description>{fmt.esc(name)} forecast {fmt.esc(h.forecast_id)}: highest {unit} is "
            f"{fmt.pct_plain(h.probability)}{(' (' + where + ')') if where else ''}.</description>\n    </item>")
    feed = ('<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0">\n  <channel>\n'
            "    <title>HazardPulse forecasts</title>\n"
            f"    <link>{shell.DOMAIN}/</link>\n"
            "    <description>Each new HazardPulse forecast as it is published: earthquakes (30 days), hurricane "
            "rapid intensification (24 hours) and US tornadoes (60 minutes). Research forecasts, not "
            "warnings.</description>\n    <language>en</language>\n"
            + "\n".join(items) + "\n  </channel>\n</rss>\n")
    if _write(dist / "feed.xml", feed):
        changed.append("feed.xml")
    return changed


LEDGER_PAGE = "verification/tornado/index.html"


def tornado_ledger(d: SiteData, n: int = 20) -> str:
    """The latest tornado ledger entries, newest first, as a table (block ``hp-ledger:rows``)."""
    import json as _json
    from hazardpulse.site.pages import common
    path = d.dist / "data" / "tornado-ledger.jsonl"
    entries = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(_json.loads(line))
                except ValueError:
                    continue
    served = str(d.tornadoes.get("model_version") or "")
    rows = []
    for e in reversed(entries[-n:]):
        h = str(e.get("hash") or "")
        model = str(e.get("model_version") or "")
        note = "" if model == served else ' <span class="chip neutral">earlier model</span>'
        rows.append([fmt.time_tag(e.get("timestamp")), f"{int(e.get('n_storms') or 0):,}",
                     f"<code>{fmt.esc(model)}</code>{note}", fmt.pct(e.get("top_probability")),
                     f'<code title="{fmt.esc(h)}">{fmt.esc(h[:12])}&hellip;</code>' if h else "&mdash;"])
    if not rows:
        return '<p class="empty">No ledger entries yet.</p>'
    return (common.table(["Issued", "Storms", "Model", "Highest storm chance", "Entry hash"], rows,
                         caption=f"The {len(rows)} most recent tornado forecasts", num_cols=(1, 3))
            + f'<p class="section-foot"><a href="/data/tornado-ledger.jsonl">The full ledger ({len(entries):,} entries, '
              "JSONL)</a></p>")


def _apply_ledger(d: SiteData, page: str) -> str:
    from hazardpulse.verification import evidence_pages
    return evidence_pages.apply_block(page, "rows", tornado_ledger(d), prefix="hp-ledger")


def evidence_blocks(d: SiteData, dist: Path) -> list[str]:
    from hazardpulse.verification import evidence_pages
    present = [rel for rel in EVIDENCE_PAGES if (dist / rel).exists()]
    if not present:
        return []
    ev = {k: v for k, v in d.evidence.items() if not k.startswith("_")}
    changed = evidence_pages.render_pages(dist, d.root, ev=ev)
    f = dist / LEDGER_PAGE
    if f.exists():
        old = f.read_text(encoding="utf-8")
        new = _apply_ledger(d, old)
        if new != old:
            f.write_text(new, encoding="utf-8", newline="\n")
            changed.append(LEDGER_PAGE + " (ledger)")
    return changed


def build_site(dist: Path | None = None, root: Path | None = None) -> list[str]:
    """Render everything; returns what changed."""
    root = root or shell.ROOT
    dist = dist or (root / "dist")
    d = SiteData(root=root, dist=dist)
    changed = maps.write_assets(dist)
    for path, html in render(d).items():
        if _write(page_file(dist, path), html):
            changed.append(path)
    changed += evidence_blocks(d, dist)
    changed += shell.rewrap_static_pages(dist)
    changed += sitemap_and_feed(d, dist)
    return changed


def check_site(dist: Path | None = None, root: Path | None = None) -> list[str]:
    """Pages that differ from what the artifacts and the shell say (empty = the site is current)."""
    root = root or shell.ROOT
    dist = dist or (root / "dist")
    d = SiteData(root=root, dist=dist)
    stale = []
    for path, html in render(d).items():
        f = page_file(dist, path)
        if not f.exists() or f.read_text(encoding="utf-8") != html:
            stale.append(path)
    for p in shell.PAGES.values():
        if p.static and page_file(dist, p.path).exists():
            if shell.rewrapped(p.path, dist) != page_file(dist, p.path).read_text(encoding="utf-8"):
                stale.append(p.path)
    from hazardpulse.verification import evidence_pages
    ev = {k: v for k, v in d.evidence.items() if not k.startswith("_")}
    stale += [f"/{rel}" for rel in evidence_pages.check_pages(dist, root, ev=ev)]
    f = dist / LEDGER_PAGE
    if f.exists() and _apply_ledger(d, f.read_text(encoding="utf-8")) != f.read_text(encoding="utf-8"):
        stale.append(f"/{LEDGER_PAGE} (ledger)")
    return stale


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report stale pages instead of writing")
    args = ap.parse_args(argv)
    if args.check:
        stale = check_site()
        for s in stale:
            print(f"stale: {s}", file=sys.stderr)
        return 1 if stale else 0
    changed = build_site()
    print(f"site: {len(changed)} files changed" + (": " + ", ".join(changed) if changed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
