"""What each hazard's forecast is, in one place: the event, its window, its schedule, its official sources.

Every page that names a hazard's event or window reads it from here, so the home page, the live pages,
the status page and the methods page cannot describe the same forecast three different ways (the
site said "formation in 24h" on three pages for months after the tornado model became a 60-minute one).
``scripts/check_liveness.py`` reads ``MAX_AGE_HOURS`` from here too.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hazard:
    key: str                  # eq / hu / to: the key in live-pulse.json and the replay ids
    name: str                 # "Earthquake"
    path: str                 # the live page
    event: str                # what the probability is the chance OF
    window: str               # "30 days"
    window_short: str         # "30 d"
    coverage: str             # where the forecast exists
    schedule: str             # how often a new forecast is issued
    max_age_hours: int        # older than this and the forecast is overdue (liveness check, status page)
    official: tuple           # (label, url) of the authorities to follow


EARTHQUAKE = Hazard(
    key="eq", name="Earthquake", path="/live/earthquake/",
    event="at least one magnitude 6+ earthquake in a 2&deg; grid cell",
    window="30 days", window_short="30 d",
    coverage="the globe on a 2&deg; grid",
    schedule="every 6 hours", max_age_hours=12,
    official=(("USGS Earthquake Hazards Program", "https://earthquake.usgs.gov/"),
              ("your national geological survey", None)),
)
HURRICANE = Hazard(
    key="hu", name="Hurricane", path="/live/hurricane/",
    event="rapid intensification: maximum sustained winds rising 30 knots or more",
    window="24 hours", window_short="24 h",
    coverage="every active tropical cyclone worldwide",
    schedule="every 6 hours, after each forecast cycle", max_age_hours=14,
    official=(("National Hurricane Center", "https://www.nhc.noaa.gov/"),
              ("Central Pacific Hurricane Center", "https://www.nhc.noaa.gov/?cpac"),
              ("Joint Typhoon Warning Center", "https://www.metoc.navy.mil/jtwc/jtwc.html")),
)
TORNADO = Hazard(
    key="to", name="Tornado", path="/live/tornado/",
    event="a tornado from a tracked thunderstorm",
    window="60 minutes", window_short="60 min",
    coverage=("every thunderstorm NOAA&rsquo;s ProbSevere system tracks on radar over and near the contiguous "
              "United States"),
    schedule="every 2 hours, or every 30 minutes while tornado risk is elevated", max_age_hours=6,
    official=(("National Weather Service", "https://www.weather.gov/"),
              ("Storm Prediction Center", "https://www.spc.noaa.gov/")),
)

HAZARDS: dict[str, Hazard] = {h.key: h for h in (EARTHQUAKE, HURRICANE, TORNADO)}
MAX_AGE_HOURS: dict[str, int] = {k: h.max_age_hours for k, h in HAZARDS.items()}


def official_links(h: Hazard) -> str:
    parts = [f'<a href="{url}" rel="noopener">{label}</a>' if url else label for label, url in h.official]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]
