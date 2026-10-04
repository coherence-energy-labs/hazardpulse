#!/usr/bin/env python3
"""Render the social share card (dist/assets/og-card.png, 1200x630) from HTML with the site's own fonts.

    python scripts/make_og_card.py      # needs playwright + chromium
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dist" / "assets"


def card_html() -> str:
    font = base64.b64encode((ASSETS / "fonts" / "inter-latin.woff2").read_bytes()).decode()
    logo = base64.b64encode((ASSETS / "hp-mark.svg").read_bytes()).decode()
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: Inter; font-weight: 400 800; src: url(data:font/woff2;base64,{font}) format("woff2"); }}
html, body {{ margin: 0; width: 1200px; height: 630px; }}
body {{ font-family: Inter, sans-serif; background: #0b1220; color: #e6ebf3; position: relative; overflow: hidden; }}
.glow {{ position: absolute; width: 900px; height: 900px; right: -320px; top: -360px;
  background: radial-gradient(circle, rgba(245,158,11,.22), rgba(245,158,11,0) 62%); }}
.grid {{ position: absolute; inset: 0; background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
  linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px); background-size: 48px 48px; }}
.wrap {{ position: absolute; inset: 72px 80px; display: flex; flex-direction: column; }}
.brand {{ display: flex; align-items: center; gap: 16px; font-size: 30px; font-weight: 750; letter-spacing: -.02em; }}
.brand img {{ width: 48px; height: 48px; border-radius: 12px; }}
h1 {{ margin: auto 0 0; font-size: 76px; line-height: 1.03; letter-spacing: -.035em; font-weight: 800; max-width: 900px; }}
p {{ margin: 28px 0 0; font-size: 28px; line-height: 1.4; color: #aab6ca; max-width: 920px; }}
.hz {{ display: flex; gap: 28px; margin-top: 40px; font-size: 22px; font-weight: 600; color: #c3cddc; }}
.hz span {{ display: inline-flex; align-items: center; gap: 10px; }}
.dot {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; }}
</style></head><body><div class="glow"></div><div class="grid"></div><div class="wrap">
<div class="brand"><img src="data:image/svg+xml;base64,{logo}" alt="">HazardPulse</div>
<h1>Hazard forecasts you can verify.</h1>
<p>Every number traceable to its data, its model and its track record against the official guidance.</p>
<div class="hz"><span><i class="dot" style="background:#2dd4bf"></i>Earthquakes</span><span><i class="dot" style="background:#60a5fa"></i>Hurricanes</span><span><i class="dot" style="background:#fb923c"></i>Tornadoes</span></div>
</div></body></html>"""


def main() -> int:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1200, "height": 630})
        page.set_content(card_html())
        page.wait_for_timeout(300)
        page.screenshot(path=str(ASSETS / "og-card.png"))
        b.close()
    print(f"wrote {ASSETS / 'og-card.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
