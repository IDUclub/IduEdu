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
import re
import unicodedata
import zipfile
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box as box_geometry
from wide_feeds import FEED_CACHE
from wide_paths import CITIES_CSV
from wide_paths import CITY_MEMBERS_CSV as MEMBERS_CSV
from wide_paths import PAPER_DIR, PROFILE_CSV, WIDE_DIR
from wide_places import CITY_SPREAD_KM, REVIEW_CSV, _settlements_near, describe, prefetch_points, use_benchmark_cache
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


#: The study compares cities, not provinces (author, 14.10.2026): no unit larger than
#: this is a city boundary, whatever its level or name. It keeps the largest city
#: states and capitals -- Moscow (2,563 km2), Saint Petersburg (2,284), Hong Kong
#: (1,114), Berlin (891), Singapore (734) -- and turns away what had come in as a
#: "city": the province of Barcelona (7,737; level 6 is the province in Spain), the
#: department of La Paz (131,257), the metropolitan provinces of Turkey (Konya
#: 41,092, Ankara 25,699), the Regierungsbezirk Stuttgart (10,557). A walk graph of
#: such a unit is also hours of Overpass for countryside the timetable does not serve.
CITY_MAX_KM2 = 3000.0
CITY_STATE_MAX_KM2 = CITY_MAX_KM2

#: Units whose OSM polygon takes in open sea, so that its area says nothing about
#: the city's size. Each is checked by hand against the official land area.
SEA_INCLUDED = {
    # Tokyo Metropolis: 42,124 km2 as drawn, 2,194 km2 of land (the polygon covers the
    # territorial sea around the Izu and Ogasawara islands)
    "relation/1543125",
}

#: City-governorates: the governorate is the city's own administrative unit, its area
#: a little over CITY_MAX_KM2 because it takes in desert on the city's edge (author,
#: 15.10.2026: "Cairo and Alexandria can surely stay").
CITY_GOVERNORATES = {
    "relation/4103336",  # Cairo Governorate, 3,007 km2
    "relation/3061846",  # Alexandria Governorate, 4,191 km2
}

_AREA_CACHE: dict[str, float] = {}


def _unit_area_km2(item: dict) -> float:
    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    key = str(item["osm"])
    if key not in _AREA_CACHE:
        osm_id = int(key.split("/")[1])
        polygon = gpd.GeoSeries([get_boundary_by_osm_id(osm_id)], crs=4326)
        _AREA_CACHE[key] = float(polygon.to_crs(_EQUAL_AREA).area.iloc[0] / 1e6)
    return _AREA_CACHE[key]


#: A unit above the city level whose bounding box is larger than this is not a city
#: state, and its polygon -- a state's or a country's, megabytes that Overpass often
#: refuses to serve -- is not fetched to prove it. A box is never smaller than its
#: polygon; the margin keeps irregular city states (Moscow's box is 2.4 times its area).
BBOX_REJECT_KM2 = 15 * CITY_MAX_KM2


def _bbox_km2(item: dict) -> float:
    from wide_batch import cached_single

    osm_id = int(str(item["osm"]).split("/")[1])
    elements = cached_single("bbox", f"rel({osm_id});out bb;")
    bounds = next(element["bounds"] for element in elements if "bounds" in element)
    box = box_geometry(bounds["minlon"], bounds["minlat"], bounds["maxlon"], bounds["maxlat"])
    return float(gpd.GeoSeries([box], crs=4326).to_crs(_EQUAL_AREA).area.iloc[0] / 1e6)


def _polygon_cached(item: dict) -> bool:
    from wide_prefetch import single_key as boundary_key

    from iduedu.overpass.cache import cache_load

    return cache_load("boundary", boundary_key(int(str(item["osm"]).split("/")[1]))) is not None


def prefetch_bboxes(stacks: list[str]) -> None:
    """Boxes of every unit above the city level whose polygon is not cached, in batches."""
    from wide_batch import cached_batch

    ids = set()
    for stack_json in stacks:
        try:
            stack = json.loads(stack_json) if stack_json else []
        except json.JSONDecodeError:
            continue
        for item in stack:
            osm = str(item.get("osm", ""))
            if osm.startswith("relation/") and item.get("level", 0) < MIN_USABLE_ADMIN_LEVEL:
                if not _polygon_cached(item):
                    ids.add(int(osm.split("/")[1]))
    logger.info("bounding boxes: %d units above the city level without a cached polygon", len(ids))
    logger.info("bbox: %s", cached_batch("bbox", [f"rel({i});out bb;" for i in sorted(ids)], batch=50))


def city_sized(item: dict) -> bool:
    """Whether the unit is small enough to be a city (``CITY_MAX_KM2``)."""
    if str(item.get("osm", "")) in SEA_INCLUDED | CITY_GOVERNORATES:
        return True
    try:
        if (
            item.get("level", 0) < MIN_USABLE_ADMIN_LEVEL
            and not _polygon_cached(item)
            and _bbox_km2(item) > BBOX_REJECT_KM2
        ):
            return False
        return _unit_area_km2(item) <= CITY_MAX_KM2
    except Exception:  # noqa: BLE001 - an unmeasurable unit is not taken on trust
        return False


def usable_boundary(admin_stack_json: str, settlement_name: str) -> dict | None:
    """The administrative unit that stands for a city carved out of a regional feed.

    Author's decision (14.10.2026, option (a)): such a city is limited to the city
    level. Choosing the deepest unit whose name merely contained the settlement's
    gave Buenos Aires its province (306,440 km2) and Sao Paulo its state (269,623):
    the densest cell of a national feed lies in the suburbs, outside the city proper.
    In order:

    1. a unit at the city level (``MIN_USABLE_ADMIN_LEVEL``..``MAX_UNNAMED_ADMIN_LEVEL``)
       bearing the city's name exactly;
    2. a coarser unit bearing the city's name exactly -- the city states (Berlin,
       Singapore, Hong Kong);
    3. the deepest unit at the city level at the densest cell (Buenos Aires -> the
       partido there);
    4. a deeper unit bearing the city's name exactly (the town inside an Australian
       shire larger than a city);
    5. nothing: a city with no city-sized unit is not clipped to a region instead.

    Every step takes only units no larger than ``CITY_MAX_KM2``: in Spain level 6 is
    the province, and step 1 had handed Barcelona and A Coruna theirs.
    """
    try:
        stack = json.loads(admin_stack_json) if admin_stack_json else []
    except json.JSONDecodeError:
        return None
    # coarse to deep, whatever order the caller passed (choose_boundary hands it deepest first)
    relations = sorted(
        (item for item in stack if item.get("name") and str(item.get("osm", "")).startswith("relation/")),
        key=lambda item: item.get("level", 0),
    )
    city_level = [
        item for item in relations if MIN_USABLE_ADMIN_LEVEL <= item.get("level", 0) <= MAX_UNNAMED_ADMIN_LEVEL
    ]
    exact_city_level = [item for item in city_level if _same_name(settlement_name, item)]
    coarse_named = [
        item
        for item in relations
        if item.get("level", 0) < MIN_USABLE_ADMIN_LEVEL and _same_name(settlement_name, item)
    ]
    deeper_named = [
        item
        for item in relations
        if item.get("level", 0) > MAX_UNNAMED_ADMIN_LEVEL and _same_name(settlement_name, item)
    ]
    # Sizes are measured only when a step reaches the unit: measuring every unit of the
    # stack up front fetched the polygon of every country and state on the way.
    # The units of a stack nest, so going outwards the first one too large ends a step.
    for step in (exact_city_level, coarse_named, city_level):
        for item in reversed(step):
            if city_sized(item):
                return item
            break
    for item in deeper_named:
        if city_sized(item):
            return item
    return None


#: A boundary is accepted when it still contains this share of the feed's stops.
#: Administrative "city proper" units are often far smaller than the network they
#: serve -- clipping by the comuna of Santiago kept 1.2% of its stops and by the
#: chosen Jakarta unit 0.1% -- so the boundary is chosen against the data instead
#: of by administrative level alone.
MIN_STOP_SHARE = 0.7

#: A unit more than this many times larger than the area its retained stops span is
#: a county, province or region, not the city (author, 14.10.2026, brief task 4):
#: Buenos Aires had been given the province (306,440 km2), Rialto San Bernardino
#: County (52,072 km2) -- Truckee's artefact again. Exempt when the unit bears the
#: city's own name exactly (Tokyo is "Tokyo"; Detroit's 88-stop people mover sits
#: in the right 371 km2 "Detroit"), never by substring ("Provincia de Buenos Aires").
MAX_AREA_RATIO = 10.0
_EQUAL_AREA = 6933


def _plain_name(name) -> str:
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _same_name(city_name: str, item: dict) -> bool:
    city = _plain_name(city_name)
    return bool(city) and city in {_plain_name(item.get("name")), _plain_name(item.get("local_name"))}


def _areas_km2(osm_id: int, kept: gpd.GeoSeries) -> tuple[float, float]:
    """The unit's own area (its parts, not their hull) and the hull of the stops it keeps."""
    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    unit = gpd.GeoSeries([get_boundary_by_osm_id(osm_id)], crs=4326).to_crs(_EQUAL_AREA).area.iloc[0] / 1e6
    served = 0.0
    if len(kept) >= 3:
        served = gpd.GeoSeries([kept.union_all().convex_hull], crs=4326).to_crs(_EQUAL_AREA).area.iloc[0] / 1e6
    return float(unit), float(served)


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
    # Widening must stop before the boundary stops being a city: the rule had walked out
    # to admin_level 2 and handed Mumbai the whole of India, and a level filter with a
    # name exemption handed Barcelona its province (level 6 in Spain) and La Paz its
    # department. The test is the unit's size (CITY_MAX_KM2), whatever its level or
    # name; a coarse unit within it is a city state -- the District of Columbia for
    # Washington (its rail feed had been clipped to Ward 2), Brussels-Capital, Vienna.
    # (tested in the sweep below, deepest first, so a country is measured only when
    # nothing smaller would do)
    candidates.sort(key=lambda item: item.get("level", 0), reverse=True)
    if points is None or not len(points):
        return next((item for item in candidates if city_sized(item)), None), 0.0

    if regional:
        # The city is named, not fitted: the source deliberately covers far more.
        chosen = usable_boundary(json.dumps(candidates, ensure_ascii=False), city_name)
        return chosen, 0.0

    sample = points if len(points) <= 4000 else points[np.linspace(0, len(points) - 1, 4000).astype(int)]
    geometry = gpd.GeoSeries([Point(x, y) for x, y in sample], crs=4326)

    best, best_share = None, 0.0
    # The name exemption belongs to the deepest unit bearing the city's name only. A
    # coarser unit with the same name is a province named after its capital: Ciudad
    # Real[8] is the city, Ciudad Real[6] the province, and the exemption let the
    # province in whenever the city kept under 70% of the stops.
    named_levels = [item.get("level", 0) for item in candidates if _same_name(city_name, item)]
    deepest_named = max(named_levels) if named_levels else None
    oversized = []
    for item in candidates:
        if not city_sized(item):
            # the units of a stack nest: every coarser one is larger still
            break
        osm_id = int(str(item["osm"]).split("/")[1])
        try:
            polygon = get_4326_boundary(osm_id=osm_id)
        except Exception as exc:  # noqa: BLE001 - a missing boundary must not stop the sweep
            logger.debug("boundary %s unavailable: %s", item.get("osm"), exc)
            continue
        inside = geometry.within(polygon)
        share = float(inside.mean())
        try:
            unit_km2, served_km2 = _areas_km2(osm_id, geometry[inside])
        except Exception:  # noqa: BLE001 - without its own geometry the unit cannot be measured
            unit_km2, served_km2 = float("nan"), float("nan")
        item = {**item, "area_km2": round(unit_km2, 1), "served_km2": round(served_km2, 3)}
        too_large = unit_km2 > MAX_AREA_RATIO * served_km2 if served_km2 == served_km2 else False
        exempt = _same_name(city_name, item) and item.get("level", 0) == deepest_named
        if too_large and not exempt:
            # the next unit inward stands instead; this one is kept only as a last resort
            oversized.append((item, share))
            continue
        if share > best_share or best is None:
            best, best_share = item, share
        if share >= MIN_STOP_SHARE:
            return item, share
    if best is None and oversized:
        # every unit was set aside: the last resort is the smallest municipal one, not
        # a ward (the same floor as below)
        municipal = [(item, share) for item, share in oversized if item.get("level", 0) <= MAX_UNNAMED_ADMIN_LEVEL]
        item, share = (municipal or oversized)[0]
        return {**item, "oversized": True}, share
    # The replacement inward stops at the municipality (author, 14.10.2026): a ward
    # or a quarter is never the city. Stellenbosch Local Municipality (93% of the
    # stops) had been replaced by Ward 22 (6%), the City of Sydney by Moore Park.
    # When only such units lie inside a unit the size rule set aside, that unit stays,
    # marked as oversized.
    # When the municipality itself is larger than a city (an Australian shire, a Turkish
    # metropolitan province), the unit inside it is taken instead and marked.
    sub_municipal = (
        best is not None and best.get("level", 0) > MAX_UNNAMED_ADMIN_LEVEL and not _same_name(city_name, best)
    )
    if sub_municipal and best_share < MIN_STOP_SHARE:
        municipal = [(item, share) for item, share in oversized if item.get("level", 0) <= MAX_UNNAMED_ADMIN_LEVEL]
        if municipal:
            item, share = municipal[0]
            return {**item, "oversized": True}, share
    return best, best_share


def resolve() -> pd.DataFrame:
    use_benchmark_cache()
    places = pd.read_csv(REVIEW_CSV).fillna("")
    profile = pd.read_csv(PROFILE_CSV).set_index("feed_id")

    # Regional feeds are re-described at their densest cell, two questions each;
    # ask them all in batched requests first (wide_batch), then the loop reads the cache.
    cores = {}
    for record in places.to_dict("records"):
        if float(record.get("spread_km") or 0) > CITY_SPREAD_KM:
            core = dense_core(record["feed_id"])
            if core:
                cores[record["feed_id"]] = core
    prefetch_points(list(cores.values()))
    stacks = [
        (
            describe(*cores[record["feed_id"]])["admin_stack_json"]
            if record["feed_id"] in cores
            else record.get("admin_stack_json", "")
        )
        for record in places.to_dict("records")
    ]
    prefetch_bboxes(stacks)

    def one(record: dict) -> dict:
        feed_id = record["feed_id"]
        spread = float(record.get("spread_km") or 0)
        latitude, longitude = float(record["centre_lat"]), float(record["centre_lon"])
        regional = spread > CITY_SPREAD_KM

        # A regional feed is resolved where its stops actually are, not at the median.
        anchor = (latitude, longitude)
        if regional:
            core = cores.get(feed_id)
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
        row = {
            "feed_id": feed_id,
            "city_name": settlement["name"],
            "population": settlement.get("population", 0),
            "boundary_osm": boundary["osm"] if boundary else "",
            "boundary_level": boundary["level"] if boundary else "",
            "boundary_name": boundary.get("name", "") if boundary else "",
            "boundary_stop_share": round(stop_share, 4),
            "boundary_area_km2": boundary.get("area_km2", "") if boundary else "",
            "served_area_km2": boundary.get("served_km2", "") if boundary else "",
            "boundary_oversized": bool(boundary and boundary.get("oversized")),
            "anchor_lat": round(anchor[0], 6),
            "anchor_lon": round(anchor[1], 6),
            "spread_km": spread,
            "regional_source": regional,
            "n_stops": int(profile.loc[feed_id, "n_stops"]) if feed_id in profile.index else 0,
            "modes": profile.loc[feed_id, "modes"] if feed_id in profile.index else "",
        }
        logger.info("%-28s -> %-22s %s", str(feed_id)[:28], settlement["name"][:22], "(regional)" if regional else "")
        return row

    # One feed whose Overpass answers keep failing must not end a stage that takes
    # hours: it is set aside, retried once at the end, and listed if it still fails.
    rows, failed = [], []
    for record in places.to_dict("records"):
        try:
            rows.append(one(record))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: %s; retried at the end", record["feed_id"], type(exc).__name__)
            failed.append(record)
    still = []
    for record in failed:
        try:
            rows.append(one(record))
        except Exception as exc:  # noqa: BLE001
            still.append({"feed_id": record["feed_id"], "reason": f"{type(exc).__name__}: {exc}"[:200]})
    if still:
        pd.DataFrame(still).to_csv(WIDE_DIR / "resolve_failures.csv", index=False, encoding="utf-8")
        logger.warning("%d feeds still unresolved after a second attempt; see resolve_failures.csv", len(still))
    return pd.DataFrame(rows)


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def assemble(feeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group feeds into cities and produce the two indicators the paper needs.

    A city is a boundary. Feeds were once grouped by the name of their nearest
    settlement and the city then took the first boundary among them -- which put
    Île-de-France Mobilités (56,361 stops, 10% of them inside Paris) and a 41-stop
    Rosny-sous-Bois network into one "Paris" clipped by Rosny-sous-Bois. Now feeds
    resolved to the same boundary form one city, named after the settlement of its
    largest feed; feeds without a boundary are grouped by name, as before.

    ``boundary_stop_share`` is the share of the city-scale feeds' stops that the
    boundary keeps, over all of them -- not the best single feed, which let 41 stops
    vouch for 56,361. ``boundary_fallback`` marks a city whose boundary kept less than
    ``MIN_STOP_SHARE``: the rule chose the best boundary there was (brief, tasks 4
    and 9: the 70% rule decides the boundary, not whether a city is studied).
    """
    feeds = feeds[feeds["city_name"] != ""].copy()
    feeds["n_stops"] = pd.to_numeric(feeds["n_stops"], errors="coerce").fillna(0)
    feeds["boundary_stop_share"] = pd.to_numeric(feeds["boundary_stop_share"], errors="coerce").fillna(0)
    feeds["group"] = feeds["boundary_osm"].where(feeds["boundary_osm"] != "", "name:" + feeds["city_name"].map(_key))

    # name each group after its largest feed; a name two boundaries share gets the
    # relation id of the smaller one appended
    largest = feeds.sort_values(["n_stops", "feed_id"], ascending=[False, True]).drop_duplicates(subset=["group"])
    # A city with a boundary is named after it: 27 municipal bus networks around
    # Nagoya (Kuwana, Chita, Seto...) all had Nagoya as their nearest large
    # settlement, and none of them is Nagoya. The settlement name stays for the
    # feeds without a boundary.
    boundary_named = largest["boundary_name"].astype(str).str.strip()
    largest = largest.assign(
        display=boundary_named.where((largest["boundary_osm"] != "") & (boundary_named != ""), largest["city_name"])
    )
    names = largest.set_index("group")["display"]
    sizes = feeds.groupby("group")["n_stops"].sum()
    order = sizes.sort_values(ascending=False).index
    taken: dict[str, str] = {}
    keys: dict[str, str] = {}
    for group in order:
        key = _key(names[group])
        if key in taken:
            key = f"{key}_{str(group).rsplit('/', 1)[-1]}" if "/" in str(group) else f"{key}_noboundary"
        taken[key] = group
        keys[group] = key
    feeds["city_key"] = feeds["group"].map(keys)
    # the settlement each feed was named after is kept: "is this boundary the city's own
    # unit" compares the settlement with the boundary, which the city's name no longer can
    feeds["settlement"] = feeds["city_name"]
    feeds["city_name"] = feeds["group"].map(names)

    modes_by_city: dict[str, set[str]] = defaultdict(set)
    for record in feeds.to_dict("records"):
        for part in str(record["modes"]).split(";"):
            if part.strip():
                modes_by_city[record["city_key"]].add(part.split(":")[0])

    regional = feeds["regional_source"].astype(str).str.lower().isin({"true", "1"})
    city_scale = feeds[~regional].assign(kept=lambda f: f["n_stops"] * f["boundary_stop_share"])
    share = city_scale.groupby("city_key")["kept"].sum() / city_scale.groupby("city_key")["n_stops"].sum()

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
    cities["boundary_stop_share"] = cities["city_key"].map(share).round(4)
    cities["boundary_fallback"] = cities["boundary_stop_share"].notna() & (
        cities["boundary_stop_share"] < MIN_STOP_SHARE
    )
    # Size of the unit against the area its stops span (brief, item 3: predictors).
    # The served area is the widest member's: a city's feeds all share one boundary.
    for column in ("boundary_area_km2", "served_area_km2"):
        if column in feeds:
            feeds[column] = pd.to_numeric(feeds[column], errors="coerce")
    if "boundary_area_km2" in feeds:
        cities["boundary_area_km2"] = cities["city_key"].map(feeds.groupby("city_key")["boundary_area_km2"].max())
        cities["served_area_km2"] = cities["city_key"].map(feeds.groupby("city_key")["served_area_km2"].max())
        cities["area_ratio"] = (cities["boundary_area_km2"] / cities["served_area_km2"]).round(2)
        cities["boundary_oversized"] = cities["city_key"].map(
            feeds.groupby("city_key")["boundary_oversized"].agg(lambda v: v.astype(str).str.lower().eq("true").any())
        )
    feeds = feeds.drop(columns="group")
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
