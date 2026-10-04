"""Human-readable locations: "12 mi SW of Moore, OK", "Vanuatu Islands".

Two public-domain tables, built by scripts/build_place_names.py:

* ``places.csv`` -- US incorporated places of 2,500+ people (Census Bureau 2023) and Natural Earth's
  populated places elsewhere. ``describe_us`` and ``describe_far`` name a point by its bearing and
  distance from the nearest place large enough for the scale of the hazard.
* ``flinn_engdahl.json`` -- the Flinn-Engdahl regions (1995 revision) that the USGS and every seismic
  network use to name where an earthquake is; ``fe_region`` gives the region of a point.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
EARTH_KM = 6371.0088
KM_PER_MI = 1.609344
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


@dataclass(frozen=True)
class Place:
    name: str
    region: str
    country: str
    lat: float
    lon: float
    population: int

    def label(self) -> str:
        if self.country == "United States":
            return f"{self.name}, {self.region}"
        return f"{self.name}, {self.country}" if self.country and self.country != self.name else self.name


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(h)))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """16-point compass direction FROM point 1 TO point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    deg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return COMPASS[int((deg + 11.25) // 22.5) % 16]


@lru_cache(maxsize=1)
def _places() -> tuple[Place, ...]:
    with (DATA / "places.csv").open(encoding="utf-8", newline="") as fh:
        return tuple(Place(r["name"], r["region"], r["country"], float(r["lat"]), float(r["lon"]),
                           int(r["population"])) for r in csv.DictReader(fh))


@lru_cache(maxsize=8)
def _index(min_population: int) -> dict[tuple[int, int], list[Place]]:
    grid: dict[tuple[int, int], list[Place]] = {}
    for p in _places():
        if p.population >= min_population:
            grid.setdefault((math.floor(p.lat), math.floor(p.lon)), []).append(p)
    return grid


def nearest(lat: float, lon: float, *, min_population: int = 0, max_km: float = 2500.0) -> tuple[Place, float] | None:
    """The nearest place of at least ``min_population`` within ``max_km``, searching outward ring by
    ring of 1-degree bins until a ring cannot hold anything closer than the best found."""
    grid = _index(min_population)
    best: tuple[Place, float] | None = None
    cy, cx = math.floor(lat), math.floor(lon)
    max_ring = int(max_km / 111.0 / max(0.05, math.cos(math.radians(min(abs(lat) + 1, 89))))) + 2
    for ring in range(0, max_ring + 1):
        # the closest any bin in this ring can be is (ring - 1) degrees of latitude away
        if best is not None and (ring - 1) * 111.0 > best[1]:
            break
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dy), abs(dx)) != ring:
                    continue
                bx = ((cx + dx + 180) % 360) - 180
                for p in grid.get((cy + dy, bx), ()):
                    d = haversine_km(lat, lon, p.lat, p.lon)
                    if d <= max_km and (best is None or d < best[1]):
                        best = (p, d)
    return best


def _distance_phrase(lat: float, lon: float, place: Place, km: float, *, miles: bool, near_km: float) -> str:
    if km < near_km:
        return f"near {place.label()}"
    direction = bearing(place.lat, place.lon, lat, lon)
    if miles:
        return f"{round(km / KM_PER_MI):,} mi {direction} of {place.label()}"
    return f"{_round_km(km):,} km {direction} of {place.label()}"


def _round_km(km: float) -> int:
    return int(round(km, -1)) if km >= 100 else int(round(km))


def describe_us(lat: float, lon: float) -> str:
    """A US storm: miles and a 16-point direction from the nearest town of 2,500+ people, the way the
    National Weather Service words a location ("12 mi SW of Moore, OK")."""
    hit = nearest(lat, lon, min_population=2500, max_km=400.0)
    if hit is None:
        return coords(lat, lon)
    return _distance_phrase(lat, lon, hit[0], hit[1], miles=True, near_km=5.0)


def describe_far(lat: float, lon: float, *, min_population: int = 100_000) -> str:
    """A tropical cyclone or an ocean point: kilometres from the nearest sizeable city."""
    hit = nearest(lat, lon, min_population=min_population, max_km=6000.0)
    if hit is None:
        return coords(lat, lon)
    return _distance_phrase(lat, lon, hit[0], hit[1], miles=False, near_km=25.0)


def coords(lat: float, lon: float, decimals: int = 1) -> str:
    """``21.0°S 169.0°E`` (decimals=1) or ``21°S 169°E`` (decimals=0)."""
    la = f"{abs(lat):.{decimals}f}°{'N' if lat >= 0 else 'S'}"
    lo = f"{abs(lon):.{decimals}f}°{'E' if lon >= 0 else 'W'}"
    return f"{la} {lo}"


@lru_cache(maxsize=1)
def _fe() -> dict:
    return json.loads((DATA / "flinn_engdahl.json").read_text(encoding="utf-8"))




def _title(name: str) -> str:
    words = name.lower().split(" ")
    out = []
    for i, w in enumerate(words):
        if "." in w.rstrip(".,") and len(w) <= 7:   # P.N.G., U.S.A.: abbreviations stay upper case
            out.append(w.upper())
        elif i > 0 and w in ("of", "the", "and"):
            out.append(w)
        else:
            out.append("-".join(part[:1].upper() + part[1:] for part in w.split("-")))
    return " ".join(out).replace(" ,", ",")


def fe_number(lat: float, lon: float) -> int:
    """The Flinn-Engdahl region number of a point (the USGS lookup: whole-degree latitude rows of
    longitude breakpoints, one table per quadrant)."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"not a coordinate: {lat}, {lon}")
    if lon == -180:
        lon = 180.0
    quad = ("n" if lat >= 0 else "s") + ("e" if lon >= 0 else "w")
    lons, nums = _fe()["quadrants"][quad][int(abs(lat))]
    alon = int(abs(lon))
    idx = 0
    for i, start in enumerate(lons):
        if start > alon:
            break
        idx = i
    return nums[idx]


def fe_region(lat: float, lon: float) -> str:
    """``Vanuatu Islands``, ``Off East Coast of Kamchatka``."""
    return _title(_fe()["names"][fe_number(lat, lon) - 1])
