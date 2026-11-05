"""The study cities: in every country, the five cities with the most complete GTFS data.

Author, 15.10.2026: the cohort is at most ``PER_COUNTRY`` cities per country --
those whose timetables describe them most completely -- with OSM and GTFS graphs
for every one of them. Moscow and Saint Petersburg are always in.

Everything up to the boundaries is decided offline, from what earlier stages
stored:

* **where a feed is** -- ``places_review.csv`` (``wide_places``): the settlement and
  the administrative stack at the feed's median stop, or, for a regional feed, at
  its densest cell (``wide_resolve.dense_core``, answered from the cache);
* **candidate cities** -- feeds grouped by country and settlement;
* **completeness** -- the number of distinct stops of the country's feeds within
  ``SCORE_RADIUS_KM`` of the city's anchor. Counted around the city rather than per
  feed, it credits a city served by a regional network (Berlin by VBB) as fully as
  one with its own. Feeds whose calendar has ended count too: the field-mapped
  African feeds carry nominal 2018--2019 calendars, and requiring current service
  had removed every city of 15 African countries.

Per country the candidates are taken in order of completeness; a candidate is
skipped when its boundary is one already taken, holds an anchor already taken or
lies inside a boundary already taken (two names for one city, or a suburb).

Boundaries follow ``wide_resolve.usable_boundary`` (the city's own unit, at most
``CITY_MAX_KM2``). Every polygon they need that is not cached is fetched first in
batched requests (``wide_prefetch.prefetch``; boxes of coarse units with
``wide_batch.cached_batch``). A city's feeds are then every feed of its country
with a stop inside its boundary.

Writes ``cities.csv`` and ``city_members.csv`` (the formats ``wide_city_feed``,
``wide_osm`` and ``wide_gtfs`` read) and ``pick5_candidates.csv`` with every
candidate, its score and why it was or was not taken.
"""

import argparse
import json
import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from wide_paths import CITIES_CSV, CITY_MEMBERS_CSV, PROFILE_CSV, WIDE_DIR, use_paper_cache

logger = logging.getLogger(__name__)

PER_COUNTRY = 5
SCORE_RADIUS_KM = 10.0
MIN_SCORE = 20
#: candidates per country whose boundaries are prepared in the batch
SHORTLIST = 12
MUST_KEEP = {"moscow_feed", "spb_feed"}


def _plain(name) -> str:
    from wide_resolve import _plain_name

    return _plain_name(name)


#: A national or regional feed is one candidate city per dense cluster of its stops,
#: not only at its densest cell: the Dutch, Swedish or Norwegian national feed is
#: the whole country's data, and a single anchor gave such countries one city.
EXTRA_CORES = 6
CORE_MIN_STOPS = 100
CORE_MIN_KM = 30.0


def dense_cores(feed_id: str, cell_degrees: float = 0.05) -> list[tuple[float, float]]:
    """Centres of the densest stop cells, at least CORE_MIN_KM apart, densest first."""
    from wide_resolve import feed_stops

    stops = feed_stops(feed_id)
    if stops is None:
        return []
    cells = pd.DataFrame(
        {
            "lat": stops[:, 1],
            "lon": stops[:, 0],
            "i": np.round(stops[:, 1] / cell_degrees).astype(int),
            "j": np.round(stops[:, 0] / cell_degrees).astype(int),
        }
    )
    counts = cells.groupby(["i", "j"]).agg(n=("lat", "size"), lat=("lat", "median"), lon=("lon", "median"))
    counts = counts[counts["n"] >= CORE_MIN_STOPS].sort_values("n", ascending=False)
    cores: list[tuple[float, float]] = []
    for row in counts.itertuples():
        here = (row.lat, row.lon)
        if all(_km(here, other) >= CORE_MIN_KM for other in cores):
            cores.append(here)
        if len(cores) >= EXTRA_CORES:
            break
    return cores


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(h)))


def feeds_table() -> pd.DataFrame:
    """One row per place a feed describes: its city, or each dense cluster of a regional feed."""
    from wide_places import CITY_SPREAD_KM, REVIEW_CSV, describe, prefetch_points
    from wide_resolve import dense_core

    places = pd.read_csv(REVIEW_CSV, keep_default_na=False, encoding="utf-8-sig")
    profile = pd.read_csv(PROFILE_CSV, keep_default_na=False).set_index("feed_id")
    outcomes = pd.read_csv(WIDE_DIR / "feed_outcomes.csv", keep_default_na=False).set_index("feed_id")
    today = time.strftime("%Y%m%d")

    def country_of(record) -> str:
        feed_id = record["feed_id"]
        country = outcomes.at[feed_id, "country_code"] if feed_id in outcomes.index else ""
        return country or record.get("osm_country", "")

    records = places.to_dict("records")
    regional = {r["feed_id"]: float(r.get("spread_km") or 0) > CITY_SPREAD_KM for r in records}
    # countries short of candidates: their regional feeds are looked at cluster by cluster
    settlements = pd.DataFrame(
        {"country": [country_of(r) for r in records], "settlement": [r.get("settlement", "") for r in records]}
    )
    short = set(settlements.groupby("country")["settlement"].nunique().loc[lambda n: n < PER_COUNTRY * 2].index)
    cores: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        if not regional[record["feed_id"]]:
            continue
        first = dense_core(record["feed_id"])
        extra = dense_cores(record["feed_id"]) if country_of(record) in short else []
        cores[record["feed_id"]] = ([first] if first else []) + [
            c for c in extra if not first or _km(c, first) >= CORE_MIN_KM
        ]
    prefetch_points([point for points in cores.values() for point in points[1:]])

    rows = []
    for record in records:
        feed_id = record["feed_id"]
        places_here = [
            (
                (float(record["centre_lat"]), float(record["centre_lon"])),
                record.get("settlement", ""),
                record.get("admin_stack_json", ""),
            )
        ]
        if regional[feed_id] and cores.get(feed_id):
            places_here = []
            for core in cores[feed_id]:
                described = describe(*core)  # cached: places, resolve and the batch above
                places_here.append((core, described["settlement"], described["admin_stack_json"]))
        service_to = str(profile.at[feed_id, "service_to"]) if feed_id in profile.index else ""
        for rank, (anchor, settlement, stack) in enumerate(places_here):
            rows.append(
                {
                    "feed_id": feed_id,
                    "core_rank": rank,
                    "country_code": country_of(record),
                    "regional": regional[feed_id],
                    "anchor_lat": anchor[0],
                    "anchor_lon": anchor[1],
                    "settlement": settlement,
                    "population": (
                        pd.to_numeric(record.get("settlement_population"), errors="coerce") if rank == 0 else np.nan
                    ),
                    "admin_stack_json": stack,
                    "n_stops": pd.to_numeric(record.get("n_stops"), errors="coerce"),
                    "modes": profile.at[feed_id, "modes"] if feed_id in profile.index else "",
                    "current": (service_to[:8] >= today) if service_to else True,
                }
            )
    return pd.DataFrame(rows)


def _xyz(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


#: country -> (distinct stops of all its feeds, their KD-tree); filled by ``score``
COUNTRY_STOPS: dict[str, tuple[np.ndarray, cKDTree]] = {}
#: radius of the city core the boundary is fitted to when no unit bears the city's name
CORE_RADIUS_KM = 5.0
MIN_CORE_SHARE = 0.7
#: two picks closer than this are one metropolis (Sydney's Fairfield and Blacktown)
MIN_PICK_SPACING_KM = 25.0


def score(candidates: pd.DataFrame, feeds: pd.DataFrame) -> pd.Series:
    """Distinct current stops within SCORE_RADIUS_KM of each candidate's anchor."""
    from wide_resolve import feed_stops

    scores = pd.Series(0, index=candidates.index)
    chord = 2 * np.sin(SCORE_RADIUS_KM / 6371.0 / 2)
    for country, group in candidates.groupby("country_code"):
        points = []
        for feed_id in feeds.loc[feeds["country_code"] == country, "feed_id"].unique():
            stops = feed_stops(feed_id)
            if stops is not None:
                points.append(stops)
        if not points:
            continue
        stops = np.unique(np.round(np.vstack(points), 4), axis=0)
        tree = cKDTree(_xyz(stops[:, 1], stops[:, 0]))
        COUNTRY_STOPS[country] = (stops, tree)
        hits = tree.query_ball_point(_xyz(group["anchor_lat"], group["anchor_lon"]), chord, return_length=True)
        scores.loc[group.index] = hits
    return scores


def candidates_table(feeds: pd.DataFrame) -> pd.DataFrame:
    feeds = feeds[feeds["country_code"] != ""].copy()
    feeds["key"] = feeds["country_code"] + "|" + feeds["settlement"].map(_plain)
    rows = []
    for key, group in feeds.groupby("key"):
        # anchored where its own city-scale feeds are; else at the regional core
        own = group[~group["regional"]]
        lead = (own if not own.empty else group).sort_values("n_stops", ascending=False).iloc[0]
        rows.append(
            {
                "key": key,
                "country_code": lead["country_code"],
                "settlement": lead["settlement"],
                "population": group["population"].max(),
                "anchor_lat": lead["anchor_lat"],
                "anchor_lon": lead["anchor_lon"],
                "admin_stack_json": lead["admin_stack_json"],
                "feeds_here": "+".join(group["feed_id"].unique()),
                "own_feeds": "+".join(own["feed_id"].unique()),
                "carved": own.empty,
                "must_keep": bool(set(group["feed_id"]) & MUST_KEEP),
            }
        )
    table = pd.DataFrame(rows)
    table["score"] = score(table, feeds)
    return table.sort_values(["country_code", "must_keep", "score"], ascending=[True, False, False])


def prepare_boundaries(shortlist: pd.DataFrame) -> None:
    """Fetch, in batches, every polygon and box the boundary choice may need."""
    from wide_batch import cached_batch
    from wide_prefetch import prefetch
    from wide_resolve import MIN_USABLE_ADMIN_LEVEL, _polygon_cached

    city_ids, coarse_ids = set(), set()
    for stack_json in shortlist["admin_stack_json"]:
        for item in json.loads(stack_json or "[]"):
            osm = str(item.get("osm", ""))
            if not osm.startswith("relation/") or _polygon_cached(item):
                continue
            (coarse_ids if item.get("level", 0) < MIN_USABLE_ADMIN_LEVEL else city_ids).add(int(osm.split("/")[1]))
    logger.info("boxes of %d coarse units", len(coarse_ids))
    logger.info("%s", cached_batch("bbox", [f"rel({i});out bb;" for i in sorted(coarse_ids)], batch=50))
    logger.info("polygons of %d city-level units", len(city_ids))
    logger.info("%s", prefetch(sorted(city_ids), batch=20))


def boundary_for(candidate: pd.Series) -> tuple[dict | None, float]:
    """The city's own administrative unit, never larger than ``CITY_MAX_KM2``.

    1. The unit bearing the city's name: the deepest at the city level, else a city
       state of that name (Berlin, Seoul).
    2. Otherwise the unit fitted to the city's core, among units down to the municipal
       level (8): the smallest holding ``MIN_CORE_SHARE`` of the country's stops within
       ``CORE_RADIUS_KM`` of the anchor, else the one holding most. No area-ratio test
       here: in Turkey level 8 is the neighbourhood (mahalle), and the ratio test had
       set the districts aside for it. Taking simply the deepest
       city-level unit had made five Istanbul neighbourhoods (level 8 mahalle) five
       "cities" of Turkey; the per-feed fit of ``wide_resolve`` had handed Berlin its
       quarter Mitte because a small city-scale feed sat there.
    """
    from wide_resolve import MAX_UNNAMED_ADMIN_LEVEL, MIN_USABLE_ADMIN_LEVEL, _same_name, city_sized

    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    stack = json.loads(candidate["admin_stack_json"] or "[]")
    relations = sorted(
        (item for item in stack if item.get("name") and str(item.get("osm", "")).startswith("relation/")),
        key=lambda item: item.get("level", 0),
    )
    named = [item for item in relations if _same_name(candidate["settlement"], item)]
    city_named = [i for i in named if MIN_USABLE_ADMIN_LEVEL <= i.get("level", 0) <= MAX_UNNAMED_ADMIN_LEVEL]
    coarse_named = [i for i in named if i.get("level", 0) < MIN_USABLE_ADMIN_LEVEL]
    stops, tree = COUNTRY_STOPS.get(candidate["country_code"], (None, None))
    core = None
    if stops is not None:
        chord = 2 * np.sin(CORE_RADIUS_KM / 6371.0 / 2)
        near = tree.query_ball_point(_xyz([candidate["anchor_lat"]], [candidate["anchor_lon"]])[0], chord)
        core = stops[near] if near else None

    def holds_core(item: dict) -> bool:
        # A relation whose polygon cannot be closed from its members comes out as a
        # sliver: Mumbai's (relation 7888990) as 0.1 km2 of coast, holding none of the
        # city's stops. Such a unit is passed over.
        if core is None:
            return True
        polygon = get_boundary_by_osm_id(int(item["osm"].split("/")[1]))
        return (
            polygon is not None and not polygon.is_empty and shapely.contains_xy(polygon, core[:, 0], core[:, 1]).any()
        )

    for step in (city_named, coarse_named):
        for item in reversed(step):
            if city_sized(item):
                if holds_core(item):
                    return item, float("nan")
                continue
            break  # the units of a stack nest: a coarser one is larger still
    if core is None:
        return None, float("nan")
    best, best_share = None, -1.0
    # deepest first; the units of a stack nest, so the first too large ends the walk
    for item in sorted(
        (i for i in relations if i.get("level", 0) <= MAX_UNNAMED_ADMIN_LEVEL), key=lambda i: -i.get("level", 0)
    ):
        if not city_sized(item):
            break
        polygon = get_boundary_by_osm_id(int(item["osm"].split("/")[1]))
        if polygon is None or polygon.is_empty:
            continue
        share = float(shapely.contains_xy(polygon, core[:, 0], core[:, 1]).mean())
        if share >= MIN_CORE_SHARE:
            return item, share
        if share > best_share:
            best, best_share = item, share
    return best, max(best_share, 0.0)


def unit_at_settlement(candidate: pd.Series) -> dict | None:
    """The city's own unit, looked up where the city's settlement node is.

    A regional feed's densest cell can lie outside the city it is named after -- the
    Uruguayan intercity network's in Canelones, next to Montevideo -- so its stack
    lacks the city's unit. Used only when the anchor gave no city-sized boundary; the
    one new point is asked for in the batched form.
    """
    from wide_places import _settlements_near, describe, prefetch_points
    from wide_resolve import _same_name, city_sized

    nodes = _settlements_near(candidate["anchor_lat"], candidate["anchor_lon"])
    node = next(
        (
            n
            for n in nodes
            if "lat" in n
            and _same_name(
                candidate["settlement"],
                {"name": n.get("tags", {}).get("name"), "local_name": n.get("tags", {}).get("name:en")},
            )
        ),
        None,
    )
    if node is None:
        return None
    prefetch_points([(node["lat"], node["lon"])])
    stack = json.loads(describe(node["lat"], node["lon"])["admin_stack_json"] or "[]")
    named = sorted(
        (i for i in stack if str(i.get("osm", "")).startswith("relation/") and _same_name(candidate["settlement"], i)),
        key=lambda i: -i.get("level", 0),
    )
    return next((i for i in named if city_sized(i)), None)


def pick(table: pd.DataFrame) -> pd.DataFrame:
    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    table = table.copy()
    for column in ("decision", "reason", "boundary_osm", "boundary_name", "boundary_level", "stop_share"):
        table[column] = pd.Series([""] * len(table), index=table.index, dtype=object)
    for country, group in table.groupby("country_code", sort=False):
        taken: list[tuple[str, object, tuple[float, float], str]] = []
        for index, candidate in group.iterrows():
            if len(taken) >= PER_COUNTRY and not candidate["must_keep"]:
                table.loc[index, ["decision", "reason"]] = ["drop", "country has five"]
                continue
            # A country's best city stays whatever its count (author, 15.10.2026: Armenia,
            # Kosovo and China are worth keeping, few stops included); the floor only
            # stops thin cities from filling a country's other places.
            if candidate["score"] < (MIN_SCORE if taken else 1) and not candidate["must_keep"]:
                table.loc[index, ["decision", "reason"]] = ["drop", f"under {MIN_SCORE} current stops nearby"]
                continue
            try:
                boundary, share = boundary_for(candidate)
                polygon = get_boundary_by_osm_id(int(boundary["osm"].split("/")[1])) if boundary else None
            except Exception as exc:  # noqa: BLE001
                table.loc[index, ["decision", "reason"]] = ["drop", f"boundary: {type(exc).__name__}"]
                continue
            if boundary is None or polygon is None or polygon.is_empty:
                boundary = unit_at_settlement(candidate)
                polygon = get_boundary_by_osm_id(int(boundary["osm"].split("/")[1])) if boundary else None
                share = float("nan")
            if boundary is None or polygon is None or polygon.is_empty:
                table.loc[index, ["decision", "reason"]] = ["drop", "no city-sized boundary"]
                continue
            here = shapely.Point(candidate["anchor_lon"], candidate["anchor_lat"])
            near = next(
                (
                    name
                    for _, _, anchor, name in taken
                    if _km(anchor, (candidate["anchor_lat"], candidate["anchor_lon"])) < MIN_PICK_SPACING_KM
                ),
                None,
            )
            if near and not candidate["must_keep"]:
                table.loc[index, ["decision", "reason"]] = ["drop", f"same metropolis as {near}"]
                continue
            clash = next(
                (
                    name
                    for osm, other, anchor, name in taken
                    if osm == boundary["osm"]
                    or other.contains(here)
                    or polygon.contains(shapely.Point(anchor[1], anchor[0]))
                ),
                None,
            )
            if clash:
                table.loc[index, ["decision", "reason"]] = ["drop", f"same city as {clash}"]
                continue
            taken.append(
                (boundary["osm"], polygon, (candidate["anchor_lat"], candidate["anchor_lon"]), candidate["settlement"])
            )
            table.loc[
                index, ["decision", "reason", "boundary_osm", "boundary_name", "boundary_level", "stop_share"]
            ] = [
                "keep",
                "must keep" if candidate["must_keep"] else f"rank {len(taken)} in country",
                boundary["osm"],
                boundary.get("name", ""),
                boundary.get("level", ""),
                round(share, 4),
            ]
    return table


def members(kept: pd.DataFrame, feeds: pd.DataFrame) -> pd.DataFrame:
    """Every feed of the city's country with a stop inside its boundary."""
    from wide_resolve import feed_stops

    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    rows = []
    cache: dict[str, np.ndarray | None] = {}
    for city in kept.itertuples():
        polygon = get_boundary_by_osm_id(int(str(city.boundary_osm).split("/")[1]))
        shapely.prepare(polygon)
        minx, miny, maxx, maxy = polygon.bounds
        for feed in feeds[feeds["country_code"] == city.country_code].drop_duplicates(subset=["feed_id"]).itertuples():
            if feed.feed_id not in cache:
                cache[feed.feed_id] = feed_stops(feed.feed_id)
            stops = cache[feed.feed_id]
            if stops is None:
                continue
            box = (stops[:, 0] >= minx) & (stops[:, 0] <= maxx) & (stops[:, 1] >= miny) & (stops[:, 1] <= maxy)
            if not box.any():
                continue
            inside = int(shapely.contains_xy(polygon, stops[box, 0], stops[box, 1]).sum())
            if inside:
                rows.append(
                    {
                        "feed_id": feed.feed_id,
                        "city_key": city.city_key,
                        "stops_inside": inside,
                        "n_stops": len(stops),
                        "regional_source": feed.regional,
                        "current": feed.current,
                        "modes": feed.modes,
                    }
                )
    return pd.DataFrame(rows)


def _city_key(name: str, used: set[str], osm: str) -> str:
    from wide_resolve import _key

    key = _key(name) or "city"
    if key in used:
        key = f"{key}_{osm.split('/')[1]}"
    used.add(key)
    return key


def write(table: pd.DataFrame, feeds: pd.DataFrame) -> None:

    kept = table[table["decision"] == "keep"].copy()
    used: set[str] = set()
    kept["city_key"] = [
        _city_key(name, used, osm)
        for name, osm in zip(
            kept["boundary_name"].where(kept["boundary_name"] != "", kept["settlement"]), kept["boundary_osm"]
        )
    ]
    member_table = members(kept, feeds)
    modes = member_table.groupby("city_key")["modes"].apply(
        lambda values: ";".join(sorted({m.split(":")[0] for v in values for m in str(v).split(";") if m}))
    )
    cities = pd.DataFrame(
        {
            "city_key": kept["city_key"],
            "city_name": kept["boundary_name"].where(kept["boundary_name"] != "", kept["settlement"]),
            "settlement": kept["settlement"],
            "country_code": kept["country_code"],
            "n_feeds_merged": kept["city_key"].map(member_table.groupby("city_key").size()).fillna(0).astype(int),
            "n_stops": kept["city_key"]
            .map(member_table.groupby("city_key")["stops_inside"].sum())
            .fillna(0)
            .astype(int),
            "score_stops_10km": kept["score"],
            "population": kept["population"],
            "carved_from_regional": kept["carved"],
            "boundary_osm": kept["boundary_osm"],
            "boundary_level": kept["boundary_level"],
            "osm_id": kept["boundary_osm"].str.split("/").str[1],
            "boundary_stop_share": kept["stop_share"],
            "gtfs_modes": kept["city_key"].map(modes).fillna(""),
            "has_usable_boundary": True,
            "pick_reason": kept["reason"],
        }
    )
    cities.to_csv(CITIES_CSV, index=False)
    member_table.to_csv(CITY_MEMBERS_CSV, index=False)
    table.drop(columns=["admin_stack_json"]).to_csv(WIDE_DIR / "pick5_candidates.csv", index=False)
    print(
        f"cities: {len(cities)} in {cities['country_code'].nunique()} countries; "
        f"feeds used: {member_table['feed_id'].nunique()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-fetch", action="store_true", help="choose only with what is cached")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    use_paper_cache()
    feeds = feeds_table()
    logger.info("%d feeds, %d current", len(feeds), int(feeds["current"].sum()))
    table = candidates_table(feeds)
    logger.info("%d candidate cities in %d countries", len(table), table["country_code"].nunique())
    shortlist = table.groupby("country_code").head(SHORTLIST)
    shortlist = pd.concat([shortlist, table[table["must_keep"]]]).drop_duplicates(subset=["key"])
    if not arguments.no_fetch:
        prepare_boundaries(shortlist)
    table = pick(table)
    write(table, feeds)


if __name__ == "__main__":
    main()
