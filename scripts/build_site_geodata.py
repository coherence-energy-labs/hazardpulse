#!/usr/bin/env python3
"""Build the site's geographic tables (place names, regions, map outlines) from public-domain sources.

    python scripts/build_site_geodata.py --sources DIR

Writes, under src/hazardpulse/site/data/:

  places.csv          name, region, country, lat, lon, population
                        * US: Census Bureau 2023 place gazetteer (coordinates) joined to the Vintage 2023
                          population estimates; incorporated places of >= 2,500 people.
                        * Elsewhere: Natural Earth 10m populated places (public domain).
  flinn_engdahl.json  the Flinn-Engdahl seismic and geographic regions (Young et al. 1996, the 1995
                        revision; distributed by the USGS): the region names and the per-quadrant
                        longitude breakpoints by whole degree of latitude.
  land_110m.json      Natural Earth 1:110m land polygons, as [lon, lat] rings rounded to 0.01 degree.
  us_states_110m.json Natural Earth 1:110m US state outlines (lakes cut out), the same format.

DIR must hold the downloaded source files:
  2023_Gaz_place_national.txt   https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/
  sub-est2023.csv               https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/cities/totals/
  ne_10m_populated_places_simple.geojson
                                https://github.com/nvkelso/natural-earth-vector (geojson/)
  names.asc quadsidx.asc nesect.asc nwsect.asc sesect.asc swsect.asc
                                the USGS Flinn-Engdahl files (also shipped in ObsPy's geodetics/data)
  ne_110m_land.geojson ne_110m_admin_1_states_provinces_lakes.geojson
                                https://github.com/nvkelso/natural-earth-vector (geojson/)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "hazardpulse" / "site" / "data"
US_MIN_POPULATION = 2500

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# Census place names carry their legal description ("Abbeville city"); the label drops it.
LSAD_SUFFIX = re.compile(
    r"\s+(city and borough|consolidated government \(balance\)|metropolitan government \(balance\)|"
    r"unified government \(balance\)|metro government \(balance\)|urban county|city|town|village|borough|"
    r"municipality|plantation|corporation|comunidad|zona urbana|CDP)$")
RENAME = {"Urban Honolulu": "Honolulu"}   # the Census name of Honolulu's census-designated place
BALANCE = re.compile(r"\s*\(balance\)$|-.*\(balance\)$")


def us_places(src: Path) -> list[dict]:
    coords = {}
    with (src / "2023_Gaz_place_national.txt").open(encoding="latin-1") as fh:
        rows = csv.reader(fh, delimiter="\t")
        header = [h.strip() for h in next(rows)]
        ix = {h: i for i, h in enumerate(header)}
        for r in rows:
            coords[r[ix["GEOID"]].strip()] = (float(r[ix["INTPTLAT"]]), float(r[ix["INTPTLONG"]].strip()))
    out = []
    text = (src / "sub-est2023.csv").read_bytes().decode("latin-1")
    for r in csv.DictReader(io.StringIO(text)):
        if r["SUMLEV"] != "162" or r["STNAME"] not in STATES:
            continue
        geoid = r["STATE"] + r["PLACE"]
        pop = int(r["POPESTIMATE2023"] or 0)
        if pop < US_MIN_POPULATION or geoid not in coords:
            continue
        name = BALANCE.sub("", LSAD_SUFFIX.sub("", r["NAME"].strip())).strip()
        name = RENAME.get(name, name)
        lat, lon = coords[geoid]
        out.append({"name": name, "region": STATES[r["STNAME"]], "country": "United States",
                    "lat": round(lat, 4), "lon": round(lon, 4), "population": pop})
    return out


def world_places(src: Path) -> list[dict]:
    data = json.loads((src / "ne_10m_populated_places_simple.geojson").read_text(encoding="utf-8"))
    out = []
    for f in data["features"]:
        p = f["properties"]
        if p.get("adm0_a3") == "USA":
            continue  # the Census table covers all 50 states and DC at a finer grain
        if p.get("featurecla") in ("Meteorological Station", "Scientific station", "Historic place"):
            continue
        out.append({"name": p["name"], "region": p.get("adm1name") or "", "country": p.get("adm0name") or "",
                    "lat": round(float(p["latitude"]), 4), "lon": round(float(p["longitude"]), 4),
                    "population": int(p.get("pop_max") or 0)})
    return out


def flinn_engdahl(src: Path) -> dict:
    names = [line.strip() for line in (src / "names.asc").read_text(encoding="ascii").splitlines() if line.strip()]
    counts = [int(v) for v in (src / "quadsidx.asc").read_text(encoding="ascii").split()]
    quads = {}
    for qi, quad in enumerate(("ne", "nw", "se", "sw")):
        per_lat = counts[qi * 91:(qi + 1) * 91]
        flat = [int(v) for v in (src / f"{quad}sect.asc").read_text(encoding="ascii").split()]
        lons, nums = flat[0::2], flat[1::2]
        rows, start = [], 0
        for n in per_lat:
            rows.append([lons[start:start + n], nums[start:start + n]])
            start += n
        assert start == len(lons), (quad, start, len(lons))
        quads[quad] = rows
    assert len(names) == 757, len(names)
    return {"source": "Flinn-Engdahl regionalisation, 1995 revision (Young et al. 1996, PEPI 96:223-297); "
                      "USGS distribution", "names": names, "quadrants": quads}


def rings(geojson: Path, keep=lambda props: True) -> list[list[list[float]]]:
    """Every outer and inner ring of the kept features, as [lon, lat] pairs rounded to 0.01 degree,
    consecutive duplicates (after rounding) dropped."""
    data = json.loads(geojson.read_text(encoding="utf-8"))
    out = []
    for f in data["features"]:
        if not keep(f.get("properties") or {}):
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                pts = []
                for lon, lat in ring:
                    pt = [round(lon, 2), round(lat, 2)]
                    if not pts or pts[-1] != pt:
                        pts.append(pt)
                if len(pts) >= 4:
                    out.append(pts)
    return out


# WebP at quality 88: 85 KB against the PNG's 135 KB. Measured in the shader's own terms (its
# smoothstep(0.25, 0.75) land weight): the largest error is 0.14 at a few coastline pixels, and 0.008% of
# pixels change side of 0.5 (2026-10-05).
GLOBE_TEXTURE = ROOT / "dist" / "assets" / "maps" / "land-2048.webp"
GLOBE_TEXTURE_QUALITY = 88


def globe_texture(geojson: Path, out: Path = GLOBE_TEXTURE, width: int = 2048, supersample: int = 3) -> None:
    """The globe's land mask: an equirectangular 8-bit image (land 255, sea 0; WebP) of Natural Earth's 1:50m land,
    drawn at ``supersample`` times the size and filtered down so coastlines are anti-aliased. The globe's
    shader samples it; nothing else does."""
    from PIL import Image, ImageDraw

    w, h = width * supersample, width * supersample // 2
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    def xy(ring):
        return [((lon + 180.0) / 360.0 * w, (90.0 - lat) / 180.0 * h) for lon, lat in ring]

    data = json.loads(geojson.read_text(encoding="utf-8"))
    for f in data["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            draw.polygon(xy(poly[0]), fill=255)
            for hole in poly[1:]:
                draw.polygon(xy(hole), fill=0)
    img = img.resize((width, width // 2), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "WEBP", quality=GLOBE_TEXTURE_QUALITY, method=6)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", type=Path, required=True)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    places = us_places(args.sources) + world_places(args.sources)
    places.sort(key=lambda p: (-p["population"], p["country"], p["region"], p["name"]))
    with (OUT / "places.csv").open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "region", "country", "lat", "lon", "population"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(places)
    fe = flinn_engdahl(args.sources)
    (OUT / "flinn_engdahl.json").write_text(json.dumps(fe, separators=(",", ":")) + "\n", encoding="utf-8")
    land = rings(args.sources / "ne_110m_land.geojson")
    (OUT / "land_110m.json").write_text(json.dumps(land, separators=(",", ":")) + "\n", encoding="utf-8")
    states = rings(args.sources / "ne_110m_admin_1_states_provinces_lakes.geojson",
                   keep=lambda p: p.get("iso_a2") == "US" and p.get("postal") not in ("AK", "HI"))
    (OUT / "us_states_110m.json").write_text(json.dumps(states, separators=(",", ":")) + "\n", encoding="utf-8")
    if (args.sources / "ne_50m_land.geojson").exists():
        globe_texture(args.sources / "ne_50m_land.geojson")
        print(f"globe texture: {GLOBE_TEXTURE.relative_to(ROOT)} ({GLOBE_TEXTURE.stat().st_size:,} bytes)")
    n_us = sum(p["country"] == "United States" for p in places)
    print(f"places.csv: {len(places)} places ({n_us} from the Census table); flinn_engdahl.json: {len(fe['names'])} "
          f"regions; land_110m.json: {len(land)} rings; us_states_110m.json: {len(states)} rings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
