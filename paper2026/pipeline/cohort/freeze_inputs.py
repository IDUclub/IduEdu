"""Freeze what the cohort stages chose into ``inputs/`` (run once, 19.10.2026).

The cohort was drawn from live catalogues and Overpass (see ``README.md`` here);
neither returns the same answers twice. The pipeline therefore starts from what
that drawing produced, written once as plain tables:

* ``cities.csv`` -- the 265 study cities with country, World Bank region and income
  group, population, boundary relation and its area, and why each was picked;
* ``city_members.csv`` -- which feed serves which city;
* ``feeds.csv`` -- every member feed: catalogue, URL, licence, sha256 of the archive;
* ``boundaries.geojson`` -- each city's boundary polygon (EPSG:4326), the one that
  clips both sources, taken from the Overpass cache without a network request;
* ``geofabrik_regions.csv`` -- the Geofabrik extract each city is cut from, with
  its replication timestamp;
* ``route_type_fixes.csv`` -- route types a source codes by its own scheme;
* ``cohort/`` -- the drawing's own tables (sampling frame, downloads, archive
  profiles, feed outcomes): the attrition counts and the selection map read them.

    python pipeline/cohort/freeze_inputs.py
"""

import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INPUTS_DIR, PAPER_DIR, setup  # noqa: E402

SOURCE = PAPER_DIR / "results" / "wide_tier"
OVERPASS_CACHE = PAPER_DIR / ".iduedu_cache"

CITY_COLUMNS = [
    "city_key", "city_name", "settlement", "country_code", "country_name", "region", "income_group",
    "population", "population_from", "osm_id", "boundary_level", "boundary_area_km2", "boundary_stop_share",
    "carved_from_regional", "n_stops", "gtfs_modes", "pick_reason",
]  # fmt: skip

#: Korea's national feed (KTDB) numbers its modes in an order of its own; the route
#: names leave no doubt (city buses "마을버스", KTX, island ferries, flights).
ROUTE_TYPE_FIXES = [
    ("trs-kr_korea-ad1d68", "0", "3", "city and village buses"),
    ("trs-kr_korea-ad1d68", "1", "1", "metro and light metro"),
    ("trs-kr_korea-ad1d68", "2", "4", "island ferries"),
    ("trs-kr_korea-ad1d68", "3", "200", "intercity and express coaches"),
    ("trs-kr_korea-ad1d68", "4", "2", "conventional rail"),
    ("trs-kr_korea-ad1d68", "5", "200", "airport limousine buses"),
    ("trs-kr_korea-ad1d68", "6", "2", "high-speed rail (KTX, SRT)"),
    ("trs-kr_korea-ad1d68", "7", "", "domestic flights: dropped"),
]


def cached_boundaries(osm_ids: dict[str, int]) -> gpd.GeoDataFrame:
    """Boundary polygons from the Overpass cache; a polygon not cached fails loudly."""
    import iduedu.overpass.downloaders as downloaders
    from iduedu import config, get_4326_boundary

    def refuse(*_, **__):
        raise RuntimeError("boundary not in the cache; freezing makes no network request")

    config.set_overpass_cache(enabled=True, cache_dir=str(OVERPASS_CACHE))
    downloaders._overpass_request = refuse  # pylint: disable=protected-access
    rows = [
        {"city_key": key, "osm_id": osm_id, "geometry": get_4326_boundary(osm_id=osm_id)}
        for key, osm_id in osm_ids.items()
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def main() -> None:
    logger = setup("freeze")
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    chosen = pd.read_csv(SOURCE / "cities.csv", keep_default_na=False)
    fields = pd.read_csv(SOURCE / "city_fields.csv", keep_default_na=False).drop_duplicates("city_key", keep="last")
    extra = [
        c for c in ("country_name", "region", "income_group", "population", "population_from", "boundary_area_km2")
    ]
    table = chosen.drop(columns=[c for c in extra if c in chosen], errors="ignore").merge(
        fields[["city_key"] + extra], on="city_key", how="left"
    )
    table["osm_id"] = pd.to_numeric(table["osm_id"], errors="coerce").astype("Int64")
    table[[c for c in CITY_COLUMNS if c in table]].to_csv(INPUTS_DIR / "cities.csv", index=False)

    members = pd.read_csv(SOURCE / "city_members.csv", keep_default_na=False)
    members = members[members["city_key"].isin(table["city_key"])]
    members.to_csv(INPUTS_DIR / "city_members.csv", index=False)

    selection = pd.read_csv(SOURCE / "selection.csv", keep_default_na=False).drop_duplicates("feed_id", keep="last")
    downloads = pd.read_csv(SOURCE / "downloads.csv", keep_default_na=False).drop_duplicates("feed_id", keep="last")
    feeds = pd.DataFrame({"feed_id": sorted(set(members["feed_id"]))})
    feeds = feeds.merge(
        selection[["feed_id", "catalog", "provider", "name", "country_code", "url", "license"]],
        on="feed_id",
        how="left",
    ).merge(downloads[["feed_id", "sha256", "bytes", "fetched_at", "status"]], on="feed_id", how="left")
    feeds.to_csv(INPUTS_DIR / "feeds.csv", index=False)

    osm_ids = dict(zip(table["city_key"], table["osm_id"].astype(int)))
    cached_boundaries(osm_ids).to_file(INPUTS_DIR / "boundaries.geojson", driver="GeoJSON")

    plan = pd.read_csv(SOURCE / "geofabrik_plan.csv", keep_default_na=False)
    snapshot = pd.read_csv(SOURCE / "osm_snapshot.csv", keep_default_na=False)
    plan = plan.merge(snapshot[["region", "snapshot_utc", "bytes"]], on="region", how="left")
    plan[plan["city_key"].isin(table["city_key"])].to_csv(INPUTS_DIR / "geofabrik_regions.csv", index=False)

    # the record of the drawing, for the attrition counts and the selection map
    (INPUTS_DIR / "cohort").mkdir(exist_ok=True)
    for name in ("selection.csv", "downloads.csv", "feed_profile.csv", "feed_outcomes.csv"):
        if (SOURCE / name).exists():
            shutil.copy(SOURCE / name, INPUTS_DIR / "cohort" / name)

    fixes = pd.DataFrame(ROUTE_TYPE_FIXES, columns=["feed_id", "published", "gtfs", "what"])
    fixes.to_csv(INPUTS_DIR / "route_type_fixes.csv", index=False)

    logger.info(
        "inputs: %d cities, %d member rows, %d feeds (%d without a sha256), %d boundaries, %d regions",
        len(table), len(members), len(feeds), int(feeds["sha256"].fillna("").eq("").sum()), len(osm_ids),
        plan["region"].nunique(),
    )  # fmt: skip


if __name__ == "__main__":
    main()
