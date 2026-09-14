"""Second download pass: complete each city with every catalog feed that lands in it.

The selection took at most a couple of feeds per country, which maximises country
spread but leaves most cities described by **one operator**. Comparing that
against an OSM graph holding every operator is the "partial reference" defect that
cost the previous run five cities and a separate table of caveats -- only now at
feed level rather than archive level.

So: for every city that has a boundary, take every feed in the merged catalog
whose known position falls inside it, and fetch the ones we do not have.

Two limits are worth stating rather than hiding, because both end up in the paper:

* Only 1796 of 4473 catalog entries carry any coordinates. Cities are completed
  where the catalog is filled in and stay partial where it is not, so **the
  completeness of the reference depends on the completeness of the catalog**.
  ``completed_from_catalog`` records which cities this pass could actually touch.
* A cap per city keeps Tokyo or Los Angeles from pulling a hundred suburban
  operators and tilting the sample back towards rich countries -- exactly the
  skew the regional quotas exist to prevent.
"""

import argparse
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from wide_catalogs import merged_catalog
from wide_city_feed import parse_osm_id
from wide_paths import CITIES_CSV
from wide_paths import CITY_COMPLETION_CSV as REPORT_CSV
from wide_paths import SELECTION_CSV, WIDE_DIR
from wide_places import use_benchmark_cache
from wide_selection import merge_into_selection

from iduedu import get_4326_boundary

logger = logging.getLogger(__name__)

DEFAULT_CAP = 15


def find_additions(cap: int = DEFAULT_CAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Catalog feeds that fall inside a city we already have, and are not downloaded yet."""
    use_benchmark_cache()
    cities = pd.read_csv(CITIES_CSV).fillna("")
    selection = pd.read_csv(SELECTION_CSV)
    known = set(selection["feed_id"].astype(str))

    catalog = merged_catalog()
    located = catalog[catalog["lat"].notna() & catalog["lon"].notna()].copy()
    located = located[~located["feed_id"].astype(str).isin(known)]
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(located["lon"], located["lat"])], crs=4326)
    located = located.set_geometry(points)
    logger.info("%d located catalog feeds not yet downloaded", len(located))

    additions, report = [], []
    for _, city in cities.iterrows():
        osm_id = parse_osm_id(city.get("osm_id", ""))
        row = {"city_key": city["city_key"], "city_name": city["city_name"], "found": 0, "added": 0, "reason": ""}
        if osm_id is None:
            row["reason"] = "no boundary"
            report.append(row)
            continue
        try:
            boundary = get_4326_boundary(osm_id=osm_id)
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"{type(exc).__name__}"
            report.append(row)
            continue
        inside = located[located.geometry.within(boundary)]
        row["found"] = len(inside)
        if not inside.empty:
            # Prefer the ones the catalog knows most about, then keep it bounded.
            chosen = inside.sort_values(["municipality", "feed_id"], ascending=[False, True]).head(cap)
            row["added"] = len(chosen)
            additions.append(chosen.drop(columns="geometry"))
        report.append(row)
        logger.info(
            "%-26s found %3d, adding %3d %s", str(city["city_name"])[:26], row["found"], row["added"], row["reason"]
        )

    frame = pd.concat(additions, ignore_index=True).drop_duplicates(subset=["feed_id"]) if additions else pd.DataFrame()
    return frame, pd.DataFrame(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="most feeds to add per city")
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame, report = find_additions(arguments.cap)
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8")

    touched = int((report["found"] > 0).sum())
    print(f"\ncities that gained candidates: {touched} of {len(report)}")
    print(f"feeds to add: {len(frame)}")
    if not frame.empty:
        print(frame.groupby("catalog").size().to_string())
    if arguments.dry_run:
        print("\nDry run: selection.csv untouched.")
        return
    print(f"\nadded to selection: {merge_into_selection(frame)}")


if __name__ == "__main__":
    main()
