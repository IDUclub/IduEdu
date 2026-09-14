"""Does transit change anything in the cities whose two estimates coincide?

Six cities report an OSM estimate exactly equal to the schedule reference. Two
explanations fit: the two transit layers happen to serve the same destinations, or
neither layer is used at all and both variants are really the pedestrian network.
Only the second is consistent with a ratio equal to 1 to the tenth decimal across
100 origins, but "consistent with" is not a measurement.

This computes the same cumulative-opportunity measure over the walking graph
alone, with the same points and the same seed, and compares it with the two
intermodal variants. If walking alone reaches exactly as far, transit contributed
nothing and the ratio of 1.000 says so.

    python wide_walkonly.py --cities alor_setar blue_downs
"""

import argparse
import logging

import numpy as np
import pandas as pd
from bench_common import sweep
from wide_access import (
    N_DESTINATIONS,
    N_ORIGINS,
    SEED,
    THRESHOLDS_MIN,
    _nearest_nodes,
    _reach,
    _sample_points,
)
from wide_cohort import measurable, restrict
from wide_paths import ACCESSIBILITY_CSV, WIDE_DIR, osm_graph_path

from iduedu import read_urban_graph

logger = logging.getLogger(__name__)

REPORT_CSV = WIDE_DIR / "walk_only.csv"


def measure_city(city_key: str) -> dict:
    walk_path = osm_graph_path(city_key, "walk")
    if not walk_path.exists():
        return {"city_key": city_key, "status": "missing_graph"}

    walk = read_urban_graph(walk_path)
    points = _sample_points(walk.nodes_gdf, N_DESTINATIONS, SEED)
    origins = points[:N_ORIGINS]
    counts = _reach(walk, _nearest_nodes(walk, origins), _nearest_nodes(walk, points))

    row = {"city_key": city_key, "status": "ok"}
    for threshold in THRESHOLDS_MIN:
        row[f"walk_mean_{threshold}"] = round(float(np.array(counts[threshold], dtype=float).mean()), 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", help="default is every city whose two estimates coincide")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    access = pd.read_csv(ACCESSIBILITY_CSV).drop_duplicates(subset=["city_key"], keep="last")
    access = access.loc[access["status"].eq("ok")]
    identical = access.loc[access["osm_mean_30"].sub(access["gtfs_mean_30"]).abs().lt(1e-9)]
    keys = arguments.cities or sorted(set(identical["city_key"].map(str)) & set(measurable()))
    logger.info("%d cities where the two estimates coincide", len(keys))

    sweep(
        restrict(keys, arguments.cities),
        measure_city,
        REPORT_CSV,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: f"{rows[0].get('status')} walk-only 30 min {rows[0].get('walk_mean_30')}",
    )

    report = pd.read_csv(REPORT_CSV).drop_duplicates(subset=["city_key"], keep="last")
    merged = report.merge(access[["city_key", "osm_mean_30", "gtfs_mean_30"]], on="city_key", how="left")
    merged["transit adds"] = (merged["osm_mean_30"] - merged["walk_mean_30"]).round(2)
    print()
    print(merged[["city_key", "walk_mean_30", "osm_mean_30", "gtfs_mean_30", "transit adds"]].to_string(index=False))


if __name__ == "__main__":
    main()
