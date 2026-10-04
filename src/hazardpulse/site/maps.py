"""The site's maps: static SVG, no JavaScript, no third-party tiles.

Each map is a base drawing kept once in /assets/maps (land, graticule, US states -- cached for a year
and pulled into every page with ``<use>``, so it inherits the page's light or dark colours) plus the
page's own markers drawn inline. Every marker is a link to its row in the page's table, and every
map has a text alternative: the table under it lists the same things.

The visitor's location marker (``.user-marker``) is placed by the edge worker, which reads the map's
projection from the marker's data attributes -- so a map may use any linear projection here without
the worker carrying a copy of its constants.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hazardpulse.site import fmt
from hazardpulse.site.places import DATA

ASSET_DIR = "maps"


@dataclass(frozen=True)
class Projection:
    """Plate carrée with independent x/y scales (an equirectangular projection with a standard
    parallel): x = (lon - lon0) * sx, y = (lat0 - lat) * sy."""
    name: str
    lon0: float
    lat0: float
    lon1: float
    lat1: float
    sx: float
    sy: float

    @property
    def width(self) -> float:
        return round((self.lon1 - self.lon0) * self.sx, 1)

    @property
    def height(self) -> float:
        return round((self.lat0 - self.lat1) * self.sy, 1)

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.sx, (self.lat0 - lat) * self.sy

    def contains(self, lat: float, lon: float, margin: float = 0.0) -> bool:
        return (self.lat1 - margin <= lat <= self.lat0 + margin) and (self.lon0 - margin <= lon <= self.lon1 + margin)

    def data_attrs(self) -> str:
        return (f'data-lon0="{self.lon0:g}" data-lat0="{self.lat0:g}" data-sx="{self.sx:.6g}" '
                f'data-sy="{self.sy:.6g}"')


WORLD = Projection("world", -180.0, 80.0, 180.0, -60.0, 960 / 360, 960 / 360)
_CONUS_SX = 960 / 59.0
CONUS = Projection("conus", -125.0, 50.0, -66.0, 24.0, _CONUS_SX, _CONUS_SX / math.cos(math.radians(37.0)))


@lru_cache(maxsize=None)
def _rings(name: str) -> tuple:
    return tuple(json.loads((DATA / name).read_text(encoding="utf-8")))


def _path(rings, proj: Projection, *, margin: float) -> str:
    """One SVG path for every ring that touches the projection's frame."""
    parts = []
    for ring in rings:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        if max(lons) < proj.lon0 - margin or min(lons) > proj.lon1 + margin:
            continue
        if max(lats) < proj.lat1 - margin or min(lats) > proj.lat0 + margin:
            continue
        pts = [proj.xy(lat, lon) for lon, lat in ring]
        seg = [f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"]
        last = (round(pts[0][0], 1), round(pts[0][1], 1))
        for x, y in pts[1:]:
            here = (round(x, 1), round(y, 1))
            if here != last:
                seg.append(f"L{here[0]:.1f} {here[1]:.1f}")
                last = here
        if len(seg) >= 3:
            parts.append("".join(seg) + "Z")
    return "".join(parts)


def _graticule(proj: Projection, step: int) -> str:
    lines = []
    lon = math.ceil(proj.lon0 / step) * step
    while lon <= proj.lon1:
        x, _ = proj.xy(0, lon)
        lines.append(f"M{x:.1f} 0V{proj.height:.1f}")
        lon += step
    lat = math.floor(proj.lat0 / step) * step
    while lat >= proj.lat1:
        _, y = proj.xy(lat, 0)
        lines.append(f"M0 {y:.1f}H{proj.width:.1f}")
        lat -= step
    return "".join(lines)


def base_svg(proj: Projection) -> str:
    """The cached base drawing of a map: ``#land``, ``#graticule`` and (US map) ``#states``. No fill or
    stroke is set here, so each ``<use>`` takes its colours from the page's stylesheet."""
    margin = 10.0
    groups = [f'<g id="land"><path d="{_path(_rings("land_110m.json"), proj, margin=margin)}"/></g>',
              f'<g id="graticule"><path d="{_graticule(proj, 30 if proj is WORLD else 5)}"/></g>']
    if proj is CONUS:
        groups.append(f'<g id="states"><path d="{_path(_rings("us_states_110m.json"), proj, margin=margin)}"/></g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {proj.width:g} {proj.height:g}">'
            + "".join(groups) + "</svg>\n")


def write_assets(dist: Path) -> list[str]:
    """Write the base drawings to /assets/maps; returns the files that changed."""
    out_dir = dist / "assets" / ASSET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for proj in (WORLD, CONUS):
        path = out_dir / f"{proj.name}.svg"
        svg = base_svg(proj)
        if not path.exists() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8", newline="\n")
            changed.append(f"assets/{ASSET_DIR}/{proj.name}.svg")
    return changed


def _base_href(proj: Projection) -> str:
    from hazardpulse.site.shell import asset
    return asset(f"{ASSET_DIR}/{proj.name}.svg")


def figure(proj: Projection, markers: str, *, title: str, desc: str, caption: str = "", legend: str = "",
           ident: str = "map", show_user: bool = True) -> str:
    """A whole map: base layers by ``<use>``, the page's markers, the visitor marker, a caption."""
    href = _base_href(proj)
    layers = [f'<rect class="map-ocean" width="{proj.width:g}" height="{proj.height:g}"/>',
              f'<use class="map-graticule" href="{href}#graticule"/>',
              f'<use class="map-land" href="{href}#land"/>']
    if proj is CONUS:
        layers.append(f'<use class="map-states" href="{href}#states"/>')
    user = ""
    if show_user:
        user = (f'<g class="user-marker" transform="translate(-100 -100)" {proj.data_attrs()} '
                'aria-label="Your approximate location"><circle class="user-ring" r="9"/>'
                '<circle class="user-pin" r="4"/></g>')
    cap = ""
    if caption or legend:
        cap = f'<figcaption>{legend}{f"<p>{caption}</p>" if caption else ""}</figcaption>'
    return (f'<figure class="map map-{proj.name}">'
            f'<svg class="map-svg" viewBox="0 0 {proj.width:g} {proj.height:g}" role="img" '
            f'aria-labelledby="{ident}-title {ident}-desc" preserveAspectRatio="xMidYMid meet">'
            f'<title id="{ident}-title">{fmt.esc(title)}</title><desc id="{ident}-desc">{fmt.esc(desc)}</desc>'
            + "".join(layers) + f'<g class="map-markers">{markers}</g>{user}</svg>{cap}</figure>')


# ------------------------------------------------------------------------------------------------
# markers
# ------------------------------------------------------------------------------------------------

def pin(proj: Projection, lat: float, lon: float, label: str, *, href: str, title: str, cls: str = "",
        r: float = 7.5) -> str:
    """A numbered, linked marker."""
    if not proj.contains(lat, lon, margin=0.5):
        return ""
    x, y = proj.xy(lat, lon)
    return (f'<a class="pin {cls}" href="{href}"><title>{fmt.esc(title)}</title>'
            f'<g transform="translate({x:.1f} {y:.1f})"><circle r="{r:g}"/>'
            f'<text text-anchor="middle" dy="0.35em">{fmt.esc(label)}</text></g></a>')


def dot(proj: Projection, lat: float, lon: float, *, cls: str, title: str, r: float = 3.0,
        href: str | None = None) -> str:
    if not proj.contains(lat, lon, margin=0.5):
        return ""
    x, y = proj.xy(lat, lon)
    body = f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:g}"><title>{fmt.esc(title)}</title></circle>'
    return f'<a href="{href}">{body}</a>' if href else body


def storm(proj: Projection, lat: float, lon: float, name: str, *, href: str, title: str, cls: str) -> str:
    """A tropical cyclone: a ringed marker with its name beside it."""
    if not proj.contains(lat, lon, margin=0.5):
        return ""
    x, y = proj.xy(lat, lon)
    anchor = "end" if x > proj.width - 120 else "start"
    dx = -11 if anchor == "end" else 11
    return (f'<a class="storm {cls}" href="{href}"><title>{fmt.esc(title)}</title>'
            f'<g transform="translate({x:.1f} {y:.1f})"><circle class="storm-halo" r="10"/>'
            f'<circle class="storm-eye" r="4.5"/>'
            f'<text x="{dx}" dy="0.35em" text-anchor="{anchor}">{fmt.esc(name)}</text></g></a>')


# earthquake heat scale: the grid holds every cell's chance; most are far below 1%, so the map
# uses its own finer, logarithmic steps (printed in the map's legend)
HEAT: tuple[tuple[float, str, str], ...] = (
    (0.10, "h5", "10% or more"),
    (0.03, "h4", "3&ndash;10%"),
    (0.01, "h3", "1&ndash;3%"),
    (0.003, "h2", "0.3&ndash;1%"),
    (0.001, "h1", "0.1&ndash;0.3%"),
)


def heat_class(p: float) -> str | None:
    for lo, cls, _ in HEAT:
        if p >= lo:
            return cls
    return None


def heat_legend() -> str:
    items = "".join(f'<li><span class="swatch heat {cls}" aria-hidden="true"></span>{text}</li>'
                    for _, cls, text in reversed(HEAT))
    return ('<div class="legend" role="group" aria-label="Map colour scale"><span class="legend-title">'
            f'Chance of an M6+ earthquake in the cell, next 30 days</span><ul>{items}</ul></div>')


def heat_grid(proj: Projection, replay: dict) -> str:
    """The whole probability grid as shaded cells, horizontal runs of one class merged into one rect."""
    dom = replay.get("forecast_domain") or {}
    grid = replay.get("probability_grid")
    if not dom or not grid:
        return ""
    values = [float(v) for v in str(grid).split(",")]
    n_lat, n_lon = int(dom["n_lat"]), int(dom["n_lon"])
    if len(values) != n_lat * n_lon:
        raise ValueError(f"probability grid has {len(values)} values for a {n_lat}x{n_lon} domain")
    dlat, dlon = float(dom["dlat"]), float(dom["dlon"])
    lat_min, lon_min = float(dom["lat_min"]), float(dom["lon_min"])
    rects = []
    for r in range(n_lat):
        lat_top = lat_min + (r + 1) * dlat
        run_cls, run_start = None, 0
        for c in range(n_lon + 1):
            cls = heat_class(values[r * n_lon + c]) if c < n_lon else None
            if cls != run_cls:
                if run_cls is not None:
                    x0, y0 = proj.xy(lat_top, lon_min + run_start * dlon)
                    x1, y1 = proj.xy(lat_top - dlat, lon_min + c * dlon)
                    rects.append(f'<rect class="heat {run_cls}" x="{x0:.1f}" y="{y0:.1f}" '
                                 f'width="{x1 - x0:.1f}" height="{y1 - y0:.1f}"/>')
                run_cls, run_start = cls, c
    return '<g class="heat-grid">' + "".join(rects) + "</g>"
