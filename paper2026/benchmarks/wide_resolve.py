"""Turn located feeds into cities: one boundary and one member list per city.

This is the verification pass over ``wide_places.py``, and it exists because the
first resolution was wrong in four ways that only showed up in the output:

* **Population outranked distance.** Both Hong Kong feeds resolved to Shenzhen,
  26 km away, because Shenzhen is larger. Now the nearest settlement wins among
  those of comparable size, so a bigger neighbour cannot capture a city.
* **National feeds were labelled by their median stop.** The Norwegian feed
  (142 349 stops, 939 km across) came out as a village in the countryside. A feed spread over more
  than a city is resolved at its densest cluster of stops instead, which is where
  its main network actually is, and is marked as needing to be cut down.
* **Boundaries as coarse as a whole country were accepted.** Five feeds were
  offered ``admin_level=2`` for clipping. Anything above level 5 is refused.
* One feed was lost to an Overpass 504 and is simply retried.

The output is ``results/wide_tier/cities.csv``: one row per city with the
boundary to clip by, the feeds that belong to it, and the flags the paper needs --
how many feeds had to be merged, and whether the network had to be carved out of
a larger one.
"""

import argparse
import json
import logging
import math
import unicodedata
import zipfile
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from wide_feeds import FEED_CACHE
from wide_paths import CITIES_CSV
from wide_paths import CITY_MEMBERS_CSV as MEMBERS_CSV
from wide_paths import PAPER_DIR, PROFILE_CSV, WIDE_DIR
from wide_places import CITY_SPREAD_KM, REVIEW_CSV, _settlements_near, describe, use_benchmark_cache
from wide_profile import _read_member

from iduedu import get_4326_boundary

logger = logging.getLogger(__name__)

#: A settlement is a serious candidate if it is within this share of the largest
#: population nearby. Among those, the nearest one wins -- which is what stops a
#: larger neighbour from capturing a city across a border.
POPULATION_TOLERANCE = 0.3

#: Boundaries at or above this level are countries, states and provinces. Clipping
#: a city network by one of them is not clipping at all.
MIN_USABLE_ADMIN_LEVEL = 5


def pick_settlement(candidates: list[dict], latitude: float, longitude: float) -> dict | None:
    """Nearest settlement of comparable size, rather than simply the largest."""
    ranked = []
    for node in candidates:
        tags = node.get("tags", {}) or {}
        try:
            population = int(str(tags.get("population", "")).replace(" ", "").replace(",", ""))
        except ValueError:
            population = 0
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
            }
        )
    ranked = [item for item in ranked if item["name"]]
    if not ranked:
        return None
    largest = max(item["population"] for item in ranked)
    serious = [item for item in ranked if item["population"] >= largest * POPULATION_TOLERANCE] or ranked
    return min(serious, key=lambda item: (item["distance_km"], -item["population"]))


def dense_core(feed_id: str, cell_degrees: float = 0.05) -> tuple[float, float] | None:
    """Centre of the densest cluster of stops.

    For a national feed the median stop is a point in the countryside; the densest
    cell is where the network it mostly describes actually is.
    """
    path = FEED_CACHE / f"{feed_id}.zip"
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            stops = _read_member(archive, "stops.txt")
    except zipfile.BadZipFile:
        return None
    if stops is None or "stop_lat" not in stops:
        return None
    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    usable = latitudes.between(-90, 90) & longitudes.between(-180, 180) & ~(latitudes.eq(0) & longitudes.eq(0))
    latitudes, longitudes = latitudes[usable], longitudes[usable]
    if latitudes.empty:
        return None
    keys = pd.Series(
        list(zip((latitudes / cell_degrees).round().astype(int), (longitudes / cell_degrees).round().astype(int)))
    )
    best = keys.value_counts().idxmax()
    # Comparing an object array of tuples against a tuple broadcasts elementwise;
    # map the comparison instead.
    inside = keys.map(lambda key: key == best).to_numpy()
    return float(np.median(latitudes.to_numpy()[inside])), float(np.median(longitudes.to_numpy()[inside]))


#: Levels 9 and 10 are boroughs, wards and quarters almost everywhere. Choosing one
#: for a city means clipping its network to a single district, so they are only
#: taken when the boundary is actually named after the city.
MAX_UNNAMED_ADMIN_LEVEL = 8


def _comparable(name: str) -> str:
    """Fold a name so that a latin transliteration matches a boundary tagged in another script."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in folded.casefold() if character.isalnum())


#: Below this length a city name is too generic to be looked for inside a longer
#: boundary name without catching something unrelated.
MIN_CONTAINED_NAME = 4


def name_matches(city_name: str, item: dict) -> bool:
    """Does this boundary carry the city's name?

    Equality alone is too strict: administrative units are named after the city
    with a qualifier around it, so Jakarta failed to match "Daerah Khusus Ibukota
    Jakarta" and was clipped to a unit holding 22% of its stops, and Bangkok to
    one holding 45%. Containment is checked in both directions, because the
    qualifier can sit on either side.
    """
    wanted = _comparable(city_name)
    if not wanted:
        return False
    for candidate in (_comparable(item.get("name", "")), _comparable(item.get("local_name", ""))):
        if not candidate:
            continue
        if candidate == wanted:
            return True
        if len(wanted) >= MIN_CONTAINED_NAME and wanted in candidate:
            return True
        if len(candidate) >= MIN_CONTAINED_NAME and candidate in wanted:
            return True
    return False


def usable_boundary(admin_stack_json: str, settlement_name: str) -> dict | None:
    """The administrative boundary that stands for the city.

    Preference is: a boundary named after the settlement, then the deepest
    boundary that is still city-sized. Taking the deepest level unconditionally
    handed Jakarta, Bogota and Tokyo a district boundary, because their names are
    tagged in a script the settlement name did not match.
    """
    try:
        stack = json.loads(admin_stack_json) if admin_stack_json else []
    except json.JSONDecodeError:
        return None
    named_any_level = [item for item in stack if item.get("name") and name_matches(settlement_name, item)]
    # A name match outranks the level rule. City states carry their city boundary at
    # level 2 or 4, so refusing coarse levels outright clipped Singapore to a
    # planning area (0% of its stops survived) and Hong Kong to a district (5%).
    if named_any_level:
        return named_any_level[-1]

    usable = [item for item in stack if item.get("level", 0) >= MIN_USABLE_ADMIN_LEVEL and item.get("name")]
    if not usable:
        return None

    city_sized = [item for item in usable if item["level"] <= MAX_UNNAMED_ADMIN_LEVEL]
    return (city_sized or usable)[-1]


#: A boundary is accepted when it still contains this share of the feed's stops.
#: Administrative "city proper" units are often far smaller than the network they
#: serve -- clipping by the comuna of Santiago kept 1.2% of its stops and by the
#: chosen Jakarta unit 0.1% -- so the boundary is chosen against the data instead
#: of by administrative level alone.
MIN_STOP_SHARE = 0.7


def feed_stops(feed_id: str) -> np.ndarray | None:
    """Longitude/latitude array of a feed's usable stops."""
    path = FEED_CACHE / f"{feed_id}.zip"
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            stops = _read_member(archive, "stops.txt")
    except zipfile.BadZipFile:
        return None
    if stops is None or "stop_lat" not in stops:
        return None
    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    usable = latitudes.between(-90, 90) & longitudes.between(-180, 180) & ~(latitudes.eq(0) & longitudes.eq(0))
    if not usable.any():
        return None
    return np.column_stack([longitudes[usable].to_numpy(), latitudes[usable].to_numpy()])


def choose_boundary(
    stack: list[dict], points: np.ndarray | None, city_name: str = "", regional: bool = False
) -> tuple[dict | None, float]:
    """Smallest boundary that still holds most of the network.

    Walks the administrative stack from the most specific outwards and stops at
    the first boundary retaining ``MIN_STOP_SHARE`` of the stops. Only relations
    are considered: ``get_4326_boundary`` takes a relation id, and feeding it a
    way id raises.

    The retention rule applies to **city-scale feeds only**. For a regional or
    national source a low retention is the intended outcome -- carving Oslo out of
    the Norwegian feed keeps 2% of its stops, and that is the point -- so widening
    the boundary until the whole source fits would undo the carving and return the
    country.
    """
    candidates = [item for item in stack if str(item.get("osm", "")).startswith("relation/")]
    # Widening must stop before the boundary stops being a city. Without this the
    # rule walked out to admin_level 2 and handed Mumbai the whole of India and
    # Oslo the whole of Norway -- at which point nothing is clipped at all.
    candidates = [
        item for item in candidates if item.get("level", 0) >= MIN_USABLE_ADMIN_LEVEL or name_matches(city_name, item)
    ]
    candidates.sort(key=lambda item: item.get("level", 0), reverse=True)
    if points is None or not len(points):
        return (candidates[0] if candidates else None), 0.0

    if regional:
        # The city is named, not fitted: the source deliberately covers far more.
        chosen = usable_boundary(json.dumps(candidates, ensure_ascii=False), city_name)
        return chosen, 0.0

    sample = points if len(points) <= 4000 else points[np.linspace(0, len(points) - 1, 4000).astype(int)]
    geometry = gpd.GeoSeries([Point(x, y) for x, y in sample], crs=4326)

    best, best_share = None, 0.0
    for item in candidates:
        try:
            polygon = get_4326_boundary(osm_id=int(str(item["osm"]).split("/")[1]))
        except Exception as exc:  # noqa: BLE001 - a missing boundary must not stop the sweep
            logger.debug("boundary %s unavailable: %s", item.get("osm"), exc)
            continue
        share = float(geometry.within(polygon).mean())
        if share > best_share:
            best, best_share = item, share
        if share >= MIN_STOP_SHARE:
            return item, share
    return best, best_share


def resolve() -> pd.DataFrame:
    use_benchmark_cache()
    places = pd.read_csv(REVIEW_CSV).fillna("")
    profile = pd.read_csv(PROFILE_CSV).set_index("feed_id")

    rows = []
    for record in places.to_dict("records"):
        feed_id = record["feed_id"]
        spread = float(record.get("spread_km") or 0)
        latitude, longitude = float(record["centre_lat"]), float(record["centre_lon"])
        regional = spread > CITY_SPREAD_KM

        # A regional feed is resolved where its stops actually are, not at the median.
        anchor = (latitude, longitude)
        if regional:
            core = dense_core(feed_id)
            if core:
                anchor = core

        stack_json = record.get("admin_stack_json", "")
        if anchor != (latitude, longitude) or not record.get("settlement"):
            described = describe(*anchor)  # cached unless the anchor moved
            stack_json = described["admin_stack_json"]
            settlement = {"name": described["settlement"], "population": described["settlement_population"]}
        else:
            candidates = _settlements_near(latitude, longitude)
            picked = pick_settlement(candidates, latitude, longitude)
            settlement = picked or {"name": record.get("settlement", ""), "population": 0}

        try:
            stack = json.loads(stack_json) if stack_json else []
        except json.JSONDecodeError:
            stack = []
        boundary, stop_share = choose_boundary(stack, feed_stops(feed_id), settlement["name"], regional)
        rows.append(
            {
                "feed_id": feed_id,
                "city_name": settlement["name"],
                "population": settlement.get("population", 0),
                "boundary_osm": boundary["osm"] if boundary else "",
                "boundary_level": boundary["level"] if boundary else "",
                "boundary_name": boundary.get("name", "") if boundary else "",
                "boundary_stop_share": round(stop_share, 4),
                "anchor_lat": round(anchor[0], 6),
                "anchor_lon": round(anchor[1], 6),
                "spread_km": spread,
                "regional_source": regional,
                "n_stops": int(profile.loc[feed_id, "n_stops"]) if feed_id in profile.index else 0,
                "modes": profile.loc[feed_id, "modes"] if feed_id in profile.index else "",
            }
        )
        logger.info("%-28s -> %-22s %s", str(feed_id)[:28], settlement["name"][:22], "(regional)" if regional else "")

    return pd.DataFrame(rows)


def assemble(feeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group feeds into cities and produce the two indicators the paper needs."""
    feeds = feeds[feeds["city_name"] != ""].copy()
    feeds["city_key"] = (
        feeds["city_name"].str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.lower()
    )
    feeds["city_key"] = feeds["city_key"].str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

    modes_by_city: dict[str, set[str]] = defaultdict(set)
    for record in feeds.to_dict("records"):
        for part in str(record["modes"]).split(";"):
            if part.strip():
                modes_by_city[record["city_key"]].add(part.split(":")[0])

    cities = (
        feeds.groupby(["city_key", "city_name"])
        .agg(
            n_feeds_merged=("feed_id", "count"),
            n_stops=("n_stops", "sum"),
            population=("population", "max"),
            carved_from_regional=("regional_source", "any"),
            boundary_osm=("boundary_osm", lambda values: next((v for v in values if v), "")),
            boundary_level=("boundary_level", lambda values: next((v for v in values if v != ""), "")),
        )
        .reset_index()
    )
    cities["osm_id"] = cities["boundary_osm"].str.extract(r"/(\d+)$", expand=False).fillna("")
    cities["gtfs_modes"] = cities["city_key"].map(lambda key: ";".join(sorted(modes_by_city.get(key, ()))))
    cities["has_usable_boundary"] = cities["osm_id"] != ""
    return cities.sort_values("n_stops", ascending=False).reset_index(drop=True), feeds


def _report(cities: pd.DataFrame, members: pd.DataFrame) -> None:
    print(f"cities: {len(cities)}   feeds placed: {len(members)}")
    print(f"  with a usable boundary        {int(cities['has_usable_boundary'].sum())}")
    print(f"  assembled from several feeds  {int((cities['n_feeds_merged'] > 1).sum())}")
    print(f"  carved out of a regional feed {int(cities['carved_from_regional'].sum())}")
    print(f"  million-plus                  {int((cities['population'] >= 1_000_000).sum())}")
    print("\nboundary admin level: " + str(cities["boundary_level"].value_counts(dropna=False).to_dict()))
    print("\ncities from more than one feed:")
    multi = cities[cities["n_feeds_merged"] > 1].sort_values("n_feeds_merged", ascending=False)
    print(multi[["city_name", "n_feeds_merged", "n_stops", "gtfs_modes"]].to_string(index=False))
    print("\nlargest 15 by stops:")
    print(
        cities.head(15)[["city_name", "n_feeds_merged", "n_stops", "carved_from_regional", "osm_id"]].to_string(
            index=False
        )
    )
    without = cities[~cities["has_usable_boundary"]]
    if not without.empty:
        print(f"\nno usable boundary ({len(without)}):")
        print(without[["city_name", "n_stops", "boundary_level"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if arguments.report_only and CITIES_CSV.exists():
        cities, members = pd.read_csv(CITIES_CSV), pd.read_csv(MEMBERS_CSV)
    else:
        feeds = resolve()
        cities, members = assemble(feeds)
        WIDE_DIR.mkdir(parents=True, exist_ok=True)
        cities.to_csv(CITIES_CSV, index=False, encoding="utf-8")
        members.to_csv(MEMBERS_CSV, index=False, encoding="utf-8")
    _report(cities, members)
    print(f"\nwritten: {CITIES_CSV.relative_to(PAPER_DIR)}, {MEMBERS_CSV.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
