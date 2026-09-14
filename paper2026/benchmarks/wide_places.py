"""Ask OSM which administrative areas each feed sits in, for manual review.

Clustering feeds by distance would need a threshold nobody can defend. Asking OSM
instead gives an external definition of "city" -- an administrative relation --
and the same relation is what later clips both sides of the comparison, so the
GTFS side and the OSM side are cut by one geometry rather than two.

What this module does *not* do is decide. ``admin_level`` is not comparable across
countries: a city is level 8 in Germany and France, 4 in a city state, 6 or 10
elsewhere. So the whole stack of enclosing areas is written out, a candidate is
suggested, and the reviewer overrides it where the suggestion is wrong. Feeds
whose stops are spread over more than a city are flagged separately: for them the
centroid names a point between towns rather than a city, and the network will have
to be carved out of a regional feed -- which is itself recorded, because how often
that is necessary is part of what the study reports.

Writes ``results/wide_tier/places_review.csv``.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
from wide_paths import PAPER_DIR
from wide_paths import PLACES_REVIEW_CSV as REVIEW_CSV
from wide_paths import WIDE_DIR, use_paper_cache
from wide_profile import PROFILE_CSV

from iduedu import config
from iduedu.overpass.cache import cache_load, cache_save_async
from iduedu.overpass.downloaders import _overpass_request

logger = logging.getLogger(__name__)


#: Beyond this radius the feed describes a region, not a city, and its centroid
#: is not a reliable name for anything. Chosen from the measured distribution:
#: the median feed reaches 16.6 km, the 90th percentile 162 km.
CITY_SPREAD_KM = 60.0

#: Overpass area identifiers are offset by these constants.
RELATION_AREA_OFFSET = 3_600_000_000
WAY_AREA_OFFSET = 2_400_000_000

#: Levels that can plausibly denote a city somewhere in the world. Kept wide on
#: purpose -- narrowing it is the reviewer's job, not the query's.
CITY_ADMIN_LEVELS = {4, 5, 6, 7, 8, 9, 10}


#: Kept as a name because four stages import it; the cache itself is declared in
#: ``wide_paths`` so there is one of it.
use_benchmark_cache = use_paper_cache


def _areas_at(latitude: float, longitude: float) -> list[dict]:
    query = f"[out:json][timeout:{config.timeout}];is_in({latitude},{longitude})->.a;area.a;out tags;"
    cache_key = f"{config.overpass_url}\nPOST\n{query}"
    cached = cache_load("is_in", cache_key)
    if cached is not None:
        return cached.get("elements", [])
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    payload = response.json()
    cache_save_async("is_in", cache_key, payload)
    return payload.get("elements", [])


def _settlements_near(latitude: float, longitude: float, radius_m: int = 30000) -> list[dict]:
    """Settlement nodes around a point.

    The administrative stack alone cannot answer "which city is this": the deepest
    enclosing boundary is a neighbourhood (Abidjan resolves to Deux Plateaux) and
    the shallowest is a province. A settlement node carries ``place`` and usually
    ``population``, which is what actually names a city.
    """
    query = (
        f"[out:json][timeout:{config.timeout}];"
        f'node(around:{radius_m},{latitude},{longitude})["place"~"^(city|town|municipality)$"];'
        f"out body;"
    )
    cache_key = f"{config.overpass_url}\nPOST\n{query}"
    cached = cache_load("settlements", cache_key)
    if cached is not None:
        return cached.get("elements", [])
    response = _overpass_request("POST", config.overpass_url, data={"data": query})
    payload = response.json()
    cache_save_async("settlements", cache_key, payload)
    return payload.get("elements", [])


def _pick_settlement(elements: list[dict], latitude: float, longitude: float) -> dict | None:
    """The settlement a feed most plausibly belongs to: biggest first, nearest to break ties."""
    ranked = []
    for node in elements:
        tags = node.get("tags", {}) or {}
        try:
            population = int(str(tags.get("population", "")).replace(" ", "").replace(",", ""))
        except ValueError:
            population = 0
        place_rank = {"city": 0, "municipality": 1, "town": 2}.get(tags.get("place", ""), 3)
        distance = math.hypot(
            (float(node.get("lat", latitude)) - latitude) * 111.0,
            (float(node.get("lon", longitude)) - longitude) * 111.0 * math.cos(math.radians(latitude)),
        )
        ranked.append(
            {
                "name": tags.get("int_name") or tags.get("name:en") or tags.get("name", ""),
                "place": tags.get("place", ""),
                "population": population,
                "distance_km": round(distance, 1),
                "osm": f"node/{node.get('id')}",
                "_key": (place_rank, -population, distance),
            }
        )
    if not ranked:
        return None
    return min(ranked, key=lambda item: item["_key"])


def _osm_reference(area_id: int) -> str:
    if area_id > RELATION_AREA_OFFSET:
        return f"relation/{area_id - RELATION_AREA_OFFSET}"
    if area_id > WAY_AREA_OFFSET:
        return f"way/{area_id - WAY_AREA_OFFSET}"
    return f"area/{area_id}"


def describe(latitude: float, longitude: float) -> dict:
    """Return the administrative stack around a point and a suggested city."""
    areas = _areas_at(latitude, longitude)
    administrative = []
    for area in areas:
        tags = area.get("tags", {}) or {}
        level = tags.get("admin_level")
        if level is None or not str(level).strip().isdigit():
            continue
        administrative.append(
            {
                "level": int(level),
                "name": tags.get("int_name") or tags.get("name:en") or tags.get("name", ""),
                "local_name": tags.get("name", ""),
                "osm": _osm_reference(int(area.get("id", 0))),
                "country": tags.get("ISO3166-1:alpha2", "") or tags.get("ISO3166-1", ""),
            }
        )
    administrative.sort(key=lambda item: item["level"])

    country = next((item["country"] for item in administrative if item["country"]), "")
    settlement = _pick_settlement(_settlements_near(latitude, longitude), latitude, longitude)

    # The boundary to clip with is the one that carries the settlement's name; a
    # neighbourhood boundary sharing a point with the city is not the city.
    boundary = None
    if settlement and settlement["name"]:
        matches = [item for item in administrative if item["name"] and item["name"] == settlement["name"]]
        boundary = matches[-1] if matches else None
    if boundary is None:
        # Fall back to the deepest plausible administrative level, and say so.
        candidates = [item for item in administrative if item["level"] in CITY_ADMIN_LEVELS and item["name"]]
        boundary = candidates[-1] if candidates else None

    return {
        "osm_country": country,
        "settlement": settlement["name"] if settlement else "",
        "settlement_place": settlement["place"] if settlement else "",
        "settlement_population": settlement["population"] if settlement else 0,
        "settlement_distance_km": settlement["distance_km"] if settlement else "",
        "boundary_name": boundary["name"] if boundary else "",
        "boundary_osm": boundary["osm"] if boundary else "",
        "boundary_admin_level": boundary["level"] if boundary else "",
        "boundary_matches_settlement": bool(
            settlement and boundary and settlement["name"] and settlement["name"] == boundary["name"]
        ),
        "admin_stack": " > ".join(f"{item['name'] or item['local_name']}[{item['level']}]" for item in administrative),
        "admin_stack_json": json.dumps(administrative, ensure_ascii=False),
    }


def run(limit: int | None = None) -> pd.DataFrame:
    use_benchmark_cache()
    profile = pd.read_csv(PROFILE_CSV)
    profile = profile[profile["ok"].astype(str).str.lower().isin({"true", "1"})]
    if limit:
        profile = profile.head(limit)

    rows = []
    for index, feed in enumerate(profile.itertuples(index=False), start=1):
        row = {
            "feed_id": feed.feed_id,
            "n_stops": feed.n_stops,
            "n_routes": feed.n_routes,
            "n_trips": feed.n_trips,
            "spread_km": feed.spread_km,
            "centre_lat": feed.centre_lat,
            "centre_lon": feed.centre_lon,
            "scale": "city" if feed.spread_km <= CITY_SPREAD_KM else "regional",
            "needs_clipping": feed.spread_km > CITY_SPREAD_KM,
        }
        try:
            row.update(describe(feed.centre_lat, feed.centre_lon))
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the sweep
            logger.warning("%s: %s", feed.feed_id, exc)
            row.update({"osm_country": "", "settlement": "", "boundary_osm": "", "admin_stack": f"ERROR {exc}"})
        rows.append(row)
        logger.info(
            "[%d/%d] %-26s %-9s %-24s %s",
            index,
            len(profile),
            str(feed.feed_id)[:26],
            row["scale"],
            (row.get("settlement") or "?")[:24],
            "" if row.get("boundary_matches_settlement") else "(boundary differs)",
        )

    frame = pd.DataFrame(rows)
    frame["resolved_city"] = ""  # filled in during resolution, by hand where automatic matching is unsure
    frame["resolved_osm"] = ""
    frame["resolution_note"] = ""
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
    return frame


def _report(frame: pd.DataFrame) -> None:
    named = frame["settlement"].astype(str) != ""
    print(f"feeds located: {len(frame)}")
    print(f"  named a settlement           {int(named.sum())}")
    print(f"  no settlement within 30 km   {int((~named).sum())}")
    print(f"  regional, need clipping      {int(frame['needs_clipping'].sum())}")
    print(f"  distinct settlements         {frame.loc[named, 'settlement'].nunique()}")
    agreed = frame["boundary_matches_settlement"].astype(str).str.lower().isin({"true", "1"})
    print(f"  boundary agrees with name    {int(agreed.sum())} of {len(frame)}")

    shared = frame.loc[named].groupby("settlement").size()
    shared = shared[shared > 1].sort_values(ascending=False)
    print(f"\nsettlements described by more than one feed: {len(shared)}")
    for city, count in shared.head(15).items():
        print(f"  {count}  {city}")

    print("\nadmin level of the chosen boundary:")
    print(frame["boundary_admin_level"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N feeds")
    parser.add_argument("--report-only", action="store_true", help="read the existing review file")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if arguments.report_only and REVIEW_CSV.exists():
        frame = pd.read_csv(REVIEW_CSV).fillna("")
    else:
        frame = run(arguments.limit)
    _report(frame)
    print(f"\nwritten: {Path(REVIEW_CSV).relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
