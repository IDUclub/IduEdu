"""From feeds to the cities the study compares.

Every GTFS feed of the run comes from some catalogue (Mobility Database,
Transitland Atlas, DigitalTransport4Africa, Transitous; recorded in
``selection.csv``) and has been given a city and a boundary by ``wide_resolve``.
This module turns that into the study's cities in two steps.

**1. One boundary, one city.** ``wide_resolve`` names a feed's city after its
nearest settlement, so feeds clipped by the same administrative unit could come
out as two cities over one territory. Cities resolved to the same boundary are
merged (see ``merge``). Feeds from different catalogues meet in one city this
way; which catalogues each city draws on is reported. Nested cities stay separate
and are flagged (``inside_city``).

**2. Which cities.** Every usable city is compared -- the comparison runs worldwide,
wherever both sides can be built (author's decision). Usable means:

* a boundary (an OSM relation) and at least ``STUB_STOPS`` stop.

The 70% rule is not a filter here. It decides which administrative unit a feed
belongs to (brief, task 9); where no unit keeps ``MIN_STOP_SHARE`` of a city's
stops, ``wide_resolve`` takes the best there is and the city carries
``boundary_fallback`` -- the Truckee case. Whether to leave such cities out is the
author's call (brief, task 4); the flag lets it be made without recomputing.

Two spread limits remain available but are off by default (``None``):
``COUNTRY_CAP`` cities per country and ``MIN_SPACING_KM`` between chosen cities.
When set, World Bank regions take turns, then countries inside a region (fewest
cities first), and inside a country the larger settlement (OSM population), then
the larger feed, goes first -- the round robin of ``wide_selection.select``. The
turn a city was taken in is kept either way, so a capped subset can be drawn from
``cohort_selection.csv`` later without recomputing anything.

The country is the one OSM puts the feed in (``wide_country``), the catalogue's
only where OSM gives none.

Writes ``cohort_merges.csv`` (what was merged into what), ``cohort_selection.csv``
(every city with its decision and reason, so the thresholds can be moved and the
choice recomputed offline), and with ``--apply`` the chosen cities as the run's
``cities.csv`` / ``city_members.csv``. The resolver's full output is kept as
``cities_resolved.csv`` / ``city_members_resolved.csv``.
"""

import argparse
import math
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd
from wide_paths import CITIES_CSV, CITY_MEMBERS_CSV, SELECTION_CSV, WIDE_DIR

STUB_STOPS = 1
MIN_STOP_SHARE = 0.7
COUNTRY_CAP: int | None = None
MIN_SPACING_KM: float | None = None

REPORT_CSV = WIDE_DIR / "cohort_selection.csv"
MERGE_CSV = WIDE_DIR / "cohort_merges.csv"
RESOLVED_CITIES_CSV = WIDE_DIR / "cities_resolved.csv"
RESOLVED_MEMBERS_CSV = WIDE_DIR / "city_members_resolved.csv"


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _resolved() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The resolver's output, preserved before --apply first overwrites it."""
    if not RESOLVED_CITIES_CSV.exists():
        shutil.copy2(CITIES_CSV, RESOLVED_CITIES_CSV)
        shutil.copy2(CITY_MEMBERS_CSV, RESOLVED_MEMBERS_CSV)
    return (
        pd.read_csv(RESOLVED_CITIES_CSV, keep_default_na=False),
        pd.read_csv(RESOLVED_MEMBERS_CSV, keep_default_na=False),
    )


def feeds_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every resolved feed with its catalogue, country and numbers; and the cities."""
    from wide_country import feed_countries

    cities, members = _resolved()
    selection = pd.read_csv(SELECTION_CSV, keep_default_na=False)[["feed_id", "catalog", "country_code"]]
    members = members.merge(selection, on="feed_id", how="left").merge(feed_countries(), on="feed_id", how="left")
    members["osm_country"] = members["osm_country"].fillna("").astype(str).str.upper().str.strip()
    members["country"] = members["osm_country"].where(members["osm_country"] != "", members["country_code"])
    for column in ("n_stops", "boundary_stop_share", "anchor_lat", "anchor_lon", "population"):
        members[column] = pd.to_numeric(members[column], errors="coerce")
    members["regional"] = members["regional_source"].astype(str).str.lower().isin({"true", "1"})
    members["resolved_city"] = members["city_key"]
    return members, cities


def _boundaries(cities: pd.DataFrame, exact: bool = False):
    """Each city's boundary, from the run's Overpass cache -- offline: a cache miss
    raises instead of sending a request.

    By default the polygon the pipeline clips with (``get_4326_boundary``, which
    turns a multipolygon into its convex hull). ``exact=True`` gives the relation's
    own geometry, holes, exclaves and islands as they are, plus ``hull_ratio``: how
    much larger the clipping hull is than the territory.
    """
    import geopandas as gpd
    from wide_city_feed import parse_osm_id
    from wide_paths import use_paper_cache

    from iduedu import get_4326_boundary
    from iduedu.overpass import downloaders
    from iduedu.overpass.downloaders import get_boundary_by_osm_id

    use_paper_cache()

    def refuse(*args, **kwargs):
        raise RuntimeError("offline: boundary not in the cache")

    original, downloaders._overpass_http = downloaders._overpass_http, refuse
    records = []
    try:
        for row in cities.itertuples(index=False):
            osm_id = parse_osm_id(row.osm_id)
            if osm_id is None:
                continue
            try:
                if exact:
                    geometry = get_boundary_by_osm_id(osm_id)
                    hull = get_4326_boundary(territory=geometry)
                    records.append({"city_key": row.city_key, "geometry": geometry, "hull": hull})
                else:
                    records.append({"city_key": row.city_key, "geometry": get_4326_boundary(osm_id=osm_id)})
            except Exception:  # noqa: BLE001 - a missing boundary only means nothing is tested against it
                continue
    finally:
        downloaders._overpass_http = original
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    frame["area_km2"] = frame.to_crs(6933).area / 1e6
    if exact and not frame.empty:
        hull_area = gpd.GeoSeries(frame["hull"], crs=4326).to_crs(6933).area / 1e6
        frame["hull_ratio"] = (hull_area / frame["area_km2"]).round(2)
        frame = frame.drop(columns="hull")
    return frame


def _usable(members: pd.DataFrame, cities: pd.DataFrame) -> pd.Series:
    """city_key -> usable before merging: boundary, stops, and the 70% rule."""
    has_boundary = set(cities.loc[cities["osm_id"].astype(str).str.strip() != "", "city_key"])
    rows = {}
    for key, group in members.groupby("city_key"):
        city_scale = group[~group["regional"]]
        share = city_scale["boundary_stop_share"].max() if len(city_scale) else math.nan
        rows[key] = bool(
            key in has_boundary
            and group["n_stops"].sum() >= STUB_STOPS
            and (group["regional"].all() or (share == share and share >= MIN_STOP_SHARE))
        )
    return pd.Series(rows, dtype=bool)


def merge(members: pd.DataFrame, cities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cities resolved to the same boundary are one city.

    ``wide_resolve`` groups feeds by settlement name, so two feeds clipped by the
    same administrative unit could stay two "cities" -- Royan and La Rochelle both
    resolved to one arrondissement -- and that territory would be compared twice.
    The boundary, not the name, is what the comparison cuts both sides with, so a
    shared boundary means one city; the one with the most stops names it.

    Merging by containment was tried first and was wrong: a feed whose own boundary
    is a region (Bordeaux carved out of a regional feed with Nouvelle-Aquitaine)
    swallowed a dozen separate cities. Nested cities stay separate territories;
    ``flag_nested`` records where one lies inside another.
    """
    osm = cities.set_index("city_key")["osm_id"].astype(str).str.strip()
    stops = members.groupby("city_key")["n_stops"].sum()
    frame = pd.DataFrame({"osm_id": osm, "stops": stops.reindex(osm.index).fillna(0)})
    frame = frame[frame["osm_id"] != ""].rename_axis("city_key").reset_index()
    frame = frame.sort_values(["stops", "city_key"], ascending=[False, True], kind="stable")
    representative = frame.drop_duplicates(subset=["osm_id"]).set_index("osm_id")["city_key"]
    target = dict(zip(frame["city_key"], frame["osm_id"].map(representative)))

    members = members.copy()
    members["city_key"] = members["city_key"].map(lambda key: target.get(key, key))
    moved = members[members["city_key"] != members["resolved_city"]]
    log = moved[["feed_id", "catalog", "resolved_city", "city_key"]].rename(columns={"city_key": "merged_into"})
    log.to_csv(MERGE_CSV, index=False, encoding="utf-8")
    return members, log


def flag_nested(table: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """For every kept city, the smallest other kept city whose boundary holds its anchor.

    Not a merge: a commune and the department around it are two territories. The
    test uses the relation's exact geometry, not the clipping hull, so a city with
    far-off islands (Tokyo and Ogasawara) does not appear to contain its
    neighbours. ``hull_ratio`` is recorded for every city all the same: the hull is
    what clips both sides of the comparison.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    table = table.copy()
    table["inside_city"] = ""
    table["inside_level"] = ""
    keep = table[table["decision"] == "keep"]
    polygons = _boundaries(cities[cities["city_key"].isin(set(keep["city_key"]))], exact=True)
    if polygons.empty:
        return table
    table["boundary_area_km2"] = table["city_key"].map(polygons.set_index("city_key")["area_km2"].round(1))
    table["hull_ratio"] = table["city_key"].map(polygons.set_index("city_key")["hull_ratio"])
    points = gpd.GeoDataFrame(
        {"city_key": keep["city_key"].to_numpy()},
        geometry=[Point(x, y) for x, y in zip(keep["anchor_lon"], keep["anchor_lat"])],
        crs=4326,
    )
    hits = gpd.sjoin(points, polygons.rename(columns={"city_key": "container"}), predicate="within")
    area = polygons.set_index("city_key")["area_km2"]
    hits = hits[(hits["city_key"] != hits["container"]) & (hits["area_km2"] > hits["city_key"].map(area).fillna(0))]
    smallest = hits.sort_values("area_km2").drop_duplicates(subset=["city_key"]).set_index("city_key")["container"]
    levels = cities.set_index("city_key")["boundary_level"]
    table["inside_city"] = table["city_key"].map(smallest).fillna("")
    table["inside_level"] = table["inside_city"].map(levels).fillna("")
    return table


def _plain(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def resolve_containers(table: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Decide every kept city whose boundary holds other cities of the study.

    Author's decision (14.10.2026, brief task 4) with one refinement:

    * If the boundary bears the city's own name -- Tokyo is "Tokyo", Kocaeli
      "Kocaeli" -- it is that city's territory, and a city resolved inside it
      (Hachioji inside Tokyo-to) is part of it: those cities are merged into it.
    * Otherwise the "city" is a regional network given a regional boundary -- the
      Pas-de-Calais around Calais and Dunkirk, Charente-Maritime around Royan -- and
      is left out; the cities inside it stay.

    Returns the table and the merges ``{nested city: container}``.
    """
    boundary_names = members.groupby("resolved_city")["boundary_name"].first()
    # the settlement of the city's largest feed: a city is now named after its boundary,
    # so its name always equals the boundary's and cannot tell a capital from its province
    largest = members.assign(_n=pd.to_numeric(members["n_stops"], errors="coerce").fillna(0)).sort_values(
        "_n", ascending=False
    )
    settlements = largest.drop_duplicates(subset=["resolved_city"]).set_index("resolved_city")[
        "settlement" if "settlement" in members else "city_name"
    ]
    containers = set(table.loc[table["decision"] == "keep", "inside_city"]) - {""}
    merges: dict[str, str] = {}
    for container in sorted(containers):
        row = table.index[table["city_key"] == container]
        if not len(row) or table.at[row[0], "decision"] != "keep":
            continue
        own = _plain(settlements.get(container, "")) == _plain(boundary_names.get(container, ""))
        # a country or a macro-region is never a city to merge into, whatever its name
        level = pd.to_numeric(table.at[row[0], "boundary_level"], errors="coerce")
        own = own and level == level and level >= 4
        nested = table.index[(table["inside_city"] == container) & (table["decision"] == "keep")]
        if own:
            for index in nested:
                merges[table.at[index, "city_key"]] = container
                table.loc[index, ["decision", "reason"]] = [
                    "drop",
                    f"merged into {container} (inside its own boundary)",
                ]
        else:
            table.loc[row, ["decision", "reason"]] = ["drop", "boundary holds other cities of the study"]
    return table, merges


def city_table(members: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """One row per (merged) city with what the rule needs."""
    from wide_selection import world_bank_countries

    names = members.drop_duplicates(subset=["resolved_city"]).set_index("resolved_city")["city_name"]
    rows = []
    for key, group in members.groupby("city_key"):
        own = group[group["resolved_city"] == key]
        city_scale = own[~own["regional"]] if len(own) else group[~group["regional"]]
        countries = group["country"].replace("", np.nan).dropna()
        catalogs = group["catalog"].replace("", np.nan).dropna()
        rows.append(
            {
                "city_key": key,
                "city_name": names.get(key, group["city_name"].iloc[0]),
                "n_feeds": len(group),
                "n_cities_merged": int(group["resolved_city"].nunique()),
                "catalogs": ";".join(f"{c}:{n}" for c, n in catalogs.value_counts().sort_index().items()),
                "n_catalogs": int(catalogs.nunique()),
                "n_stops": float(group["n_stops"].sum()),
                "population": float(group["population"].max()),
                "country_code": countries.mode().iloc[0] if len(countries) else "",
                "anchor_lat": float((own if len(own) else group)["anchor_lat"].median()),
                "anchor_lon": float((own if len(own) else group)["anchor_lon"].median()),
                "all_regional": bool((own if len(own) else group)["regional"].all()),
                # share of all the city-scale feeds' stops kept by the boundary, not the
                # best single feed: 41 Rosny stops once vouched for 56,361 of Ile-de-France
                "best_stop_share": (
                    float(
                        (city_scale["n_stops"] * city_scale["boundary_stop_share"]).sum() / city_scale["n_stops"].sum()
                    )
                    if len(city_scale) and city_scale["n_stops"].sum() > 0
                    else math.nan
                ),
            }
        )
    extra = [
        c
        for c in (
            "boundary_area_km2",
            "served_area_km2",
            "area_ratio",
            "boundary_stop_share",
            "boundary_fallback",
            "boundary_oversized",
        )
        if c in cities
    ]
    table = pd.DataFrame(rows).merge(
        cities[["city_key", "osm_id", "boundary_level", "boundary_osm", *extra]].rename(
            columns={"boundary_stop_share": "city_stop_share"}
        ),
        on="city_key",
        how="left",
    )

    # The boundary's own facts, read from the cached is_in answers (wide_country): its
    # country -- a carved city's densest cell can lie across a border (Geneva) -- and its
    # population, where OSM tags one; otherwise the settlement's, but only when the
    # settlement is the unit itself (Helsinki in Helsinki, not New York for Hoboken).
    from wide_country import boundary_facts

    facts = boundary_facts().set_index("osm_id")
    osm = table["osm_id"].astype(str)
    boundary_country = osm.map(facts["country"]).fillna("")
    table["country_from"] = np.where(boundary_country != "", "boundary", "feeds")
    table["country_code"] = boundary_country.where(boundary_country != "", table["country_code"])
    settlements = members.sort_values("n_stops", ascending=False).drop_duplicates(subset=["city_key"])
    settlement_name = table["city_key"].map(
        settlements.set_index("city_key")["settlement" if "settlement" in members else "city_name"]
    )
    unit_population = pd.to_numeric(osm.map(facts["population"]), errors="coerce")
    same_unit = [_plain(a) == _plain(b) for a, b in zip(settlement_name.fillna(""), table["city_name"].fillna(""))]
    table["settlement_population"] = table["population"]
    table["population"] = unit_population.where(unit_population.notna(), table["population"].where(same_unit))
    table["population_from"] = np.where(
        unit_population.notna(), "boundary tag", np.where(same_unit, "settlement of the same name", "")
    )

    regions = world_bank_countries()[["country_code", "country_name", "region", "income_group"]]
    table = table.merge(regions, on="country_code", how="left")
    # Taiwan is not in the World Bank's country list; its own classification puts it
    # in East Asia & Pacific and in the high-income group (footnote in the tables).
    taiwan = table["country_code"] == "TW"
    table.loc[taiwan, ["country_name", "region", "income_group"]] = ["Taiwan", "East Asia & Pacific", "High income"]
    table["wb_note"] = np.where(taiwan, "region and income assigned by hand (not in the World Bank list)", "")
    for column, fill in (("region", "Unknown"), ("income_group", "Unknown"), ("country_name", "")):
        table[column] = table[column].fillna(fill).replace("", fill)
    table["region"] = table["region"].str.strip()
    return table


def choose(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["decision"] = ""
    table["reason"] = ""
    table["turn"] = np.nan

    no_boundary = table["osm_id"].fillna("").astype(str).str.strip() == ""
    table.loc[no_boundary, ["decision", "reason"]] = ["drop", "no usable boundary"]
    stub = (table["decision"] == "") & (table["n_stops"] < STUB_STOPS)
    table.loc[stub, ["decision", "reason"]] = ["drop", f"fewer than {STUB_STOPS} stop"]
    # flag, not filter: see the module docstring
    table["boundary_fallback"] = ~table["all_regional"] & (table["best_stop_share"].fillna(0) < MIN_STOP_SHARE)

    per_country: dict[str, int] = {}
    anchors: list[tuple[float, float]] = []
    candidates = table[table["decision"] == ""].sort_values(["population", "n_stops"], ascending=False, kind="stable")
    queues = {
        region: {country: list(rows.index) for country, rows in part.groupby("country_code", sort=True)}
        for region, part in candidates.groupby("region", sort=True)
    }

    def spaced(index: int) -> bool:
        lat, lon = table.at[index, "anchor_lat"], table.at[index, "anchor_lon"]
        if MIN_SPACING_KM is None or not anchors:
            return True
        a = np.array(anchors, dtype=float)
        return bool(_haversine_km(lat, lon, a[:, 0], a[:, 1]).min() >= MIN_SPACING_KM)

    def take_one(region: str, turn: int) -> bool:
        countries = queues[region]
        for country in sorted(countries, key=lambda c: (per_country.get(c, 0), c)):
            indices = countries[country]
            while indices:
                index = indices.pop(0)
                if COUNTRY_CAP is not None and per_country.get(country, 0) >= COUNTRY_CAP:
                    table.loc[index, ["decision", "reason"]] = ["drop", f"country cap {COUNTRY_CAP}"]
                    continue
                if not spaced(index):
                    table.loc[index, ["decision", "reason"]] = ["drop", f"within {MIN_SPACING_KM:.0f} km"]
                    continue
                reason = "usable" if COUNTRY_CAP is None and MIN_SPACING_KM is None else "spread rule"
                table.loc[index, ["decision", "reason", "turn"]] = ["keep", reason, turn]
                per_country[country] = per_country.get(country, 0) + 1
                anchors.append((table.at[index, "anchor_lat"], table.at[index, "anchor_lon"]))
                return True
        return False

    turn = 0
    while True:
        turn += 1
        if not any([take_one(region, turn) for region in sorted(queues)]):
            break
    return table


def report(table: pd.DataFrame, log: pd.DataFrame) -> None:
    keep = table[table["decision"] == "keep"]
    print(f"cities after merging: {len(table)}  (feeds re-assigned by boundary: {len(log)})")
    print(f"kept: {len(keep)} cities in {keep['country_code'].nunique()} countries")
    print("\nwhy left out:\n" + table[table.decision == "drop"]["reason"].value_counts().to_string())
    regions = keep.groupby("region").agg(cities=("city_key", "size"), countries=("country_code", "nunique"))
    print("\nkept, by World Bank region:\n" + regions.to_string())
    print("\nkept, by income:\n" + keep["income_group"].value_counts().to_string())
    print(
        "\nkept, by catalogues drawn on:\n"
        + keep["catalogs"].str.replace(r":\d+", "", regex=True).value_counts().to_string()
    )
    print(f"kept cities with feeds from more than one catalogue: {int((keep['n_catalogs'] > 1).sum())}")


def apply(table: pd.DataFrame, members: pd.DataFrame, cities: pd.DataFrame) -> None:
    """Write the chosen cities, merges included, as the run's cities.csv / city_members.csv."""
    keep = set(table.loc[table["decision"] == "keep", "city_key"])
    chosen = members[members["city_key"].isin(keep)].drop(
        columns=["catalog", "country_code", "osm_country", "country", "regional"], errors="ignore"
    )

    def modes_of(values) -> str:
        return ";".join(sorted({p.split(":")[0] for v in values for p in str(v).split(";") if p.strip()}))

    grouped = chosen.assign(n_stops=pd.to_numeric(chosen["n_stops"], errors="coerce").fillna(0)).groupby("city_key")
    summary = grouped.agg(
        n_feeds_merged=("feed_id", "count"), n_stops=("n_stops", "sum"), gtfs_modes=("modes", modes_of)
    )
    out = cities[cities["city_key"].isin(keep)].set_index("city_key")
    out.update(summary)
    out.reset_index().to_csv(CITIES_CSV, index=False, encoding="utf-8")
    chosen.to_csv(CITY_MEMBERS_CSV, index=False, encoding="utf-8")
    print(f"\nwritten {CITIES_CSV.name}: {len(out)} cities, {CITY_MEMBERS_CSV.name}: {len(chosen)} feeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the chosen cities as the run's cities.csv")
    parser.add_argument("--country-cap", type=int, default=None, help="most cities per country (default: none)")
    parser.add_argument(
        "--min-spacing-km", type=float, default=None, help="least distance between cities (default: none)"
    )
    arguments = parser.parse_args()
    members, cities = feeds_table()
    members, log = merge(members, cities)
    global COUNTRY_CAP, MIN_SPACING_KM
    COUNTRY_CAP, MIN_SPACING_KM = arguments.country_cap, arguments.min_spacing_km
    table = flag_nested(choose(city_table(members, cities)), cities)
    table, merges = resolve_containers(table, members)
    if merges:
        # the merged cities' feeds now belong to the city that holds them
        members["city_key"] = members["city_key"].map(lambda key: merges.get(key, key))
        extra = members[members["resolved_city"].map(lambda key: merges.get(key, "")) != ""]
        log = pd.concat(
            [
                log,
                extra[["feed_id", "catalog", "resolved_city", "city_key"]].rename(columns={"city_key": "merged_into"}),
            ],
            ignore_index=True,
        )
        log.to_csv(MERGE_CSV, index=False, encoding="utf-8")
    table.to_csv(REPORT_CSV, index=False, encoding="utf-8")
    report(table, log)
    if arguments.apply:
        apply(table, members, cities)


if __name__ == "__main__":
    main()
