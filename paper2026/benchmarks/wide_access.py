"""Accessibility from OSM alone against accessibility from the timetable.

This is the question the study exists to answer: if a city has no feed, how wrong
is an estimate built from OpenStreetMap? Two intermodal graphs are built over the
**same** pedestrian network, so the only thing that differs is where the transit
comes from:

* **A** -- OSM public transport, run times from the kinematic model;
* **D** -- the city's own GTFS, run times and headways from the schedule.

The measure is cumulative opportunities with location as the opportunity: a fixed
sample of points is drawn once per city from the walking network, and each origin
counts how many of them it can reach within 15, 30 and 45 minutes. Using the same
points as origins and destinations keeps the two variants comparable without
needing a global dataset of jobs or schools, which does not exist at this sample's
coverage.

Both variants keep every component rather than the largest one. Trimming would
remove different pieces on each side, and the difference between the two would
then include the trimming rather than the data.
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from bench_common import already_done, sweep
from scipy.spatial import cKDTree
from wide_cohort import measurable, restrict, with_gtfs_graph
from wide_paths import ACCESSIBILITY_CSV as ACCESS_CSV
from wide_paths import ALL_OSM_MODES, gtfs_graph_path, osm_graph_path

from iduedu import read_urban_graph
from iduedu.graph.shortest_paths import single_source_dijkstra_path_length
from iduedu.graph_builders.intermodal_builders import join_pt_walk_graph

logger = logging.getLogger(__name__)

THRESHOLDS_MIN = (15, 30, 45)
N_DESTINATIONS = 1500
N_ORIGINS = 100
SEED = 20260811


def _sample_points(walk_nodes, count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    size = min(count, len(walk_nodes))
    chosen = generator.choice(len(walk_nodes), size, replace=False)
    geometry = walk_nodes.geometry.iloc[chosen]
    return np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])


def _nearest_nodes(graph, points: np.ndarray) -> np.ndarray:
    nodes = graph.nodes_gdf
    positions = np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()])
    _, indices = cKDTree(positions).query(points)
    return nodes.index.to_numpy()[indices]


def _reach(graph, origin_nodes: np.ndarray, destination_nodes: np.ndarray) -> dict[int, list[int]]:
    """For each origin, how many destinations fall inside each time threshold."""
    counts = {threshold: [] for threshold in THRESHOLDS_MIN}
    destinations = pd.Index(destination_nodes)
    for origin in origin_nodes:
        try:
            lengths = single_source_dijkstra_path_length(
                graph, source_node=origin, weight="time_min", cutoff=float(max(THRESHOLDS_MIN))
            )
        except Exception:  # noqa: BLE001 - an unreachable origin must not end the city
            for threshold in THRESHOLDS_MIN:
                counts[threshold].append(0)
            continue
        reached = lengths.reindex(destinations).to_numpy(dtype=float)
        for threshold in THRESHOLDS_MIN:
            counts[threshold].append(int(np.count_nonzero(reached <= threshold)))
    return counts


def measure_city(city_key: str) -> dict:
    row = {"city_key": city_key}
    walk_path = osm_graph_path(city_key, "walk")
    pt_path = osm_graph_path(city_key, "pt", ALL_OSM_MODES)
    gtfs_path = gtfs_graph_path(city_key)
    if not (walk_path.exists() and pt_path.exists() and gtfs_path.exists()):
        row["status"] = "missing_graph"
        return row

    started = time.perf_counter()
    walk = read_urban_graph(walk_path)
    osm_pt = read_urban_graph(pt_path)
    gtfs_pt = read_urban_graph(gtfs_path)
    if walk.nodes_gdf.empty or osm_pt.nodes_gdf.empty or gtfs_pt.nodes_gdf.empty:
        row["status"] = "empty_graph"
        return row

    points = _sample_points(walk.nodes_gdf, N_DESTINATIONS, SEED)
    origins = points[:N_ORIGINS]

    variants = {}
    for name, transit in (("A", osm_pt), ("D", gtfs_pt)):
        joined = join_pt_walk_graph(transit, walk, keep_largest_subgraph=False)
        variants[name] = _reach(joined, _nearest_nodes(joined, origins), _nearest_nodes(joined, points))

    row["n_origins"] = len(origins)
    row["n_destinations"] = len(points)
    for threshold in THRESHOLDS_MIN:
        osm_reach = np.array(variants["A"][threshold], dtype=float)
        reference = np.array(variants["D"][threshold], dtype=float)
        row[f"osm_mean_{threshold}"] = round(float(osm_reach.mean()), 2)
        row[f"gtfs_mean_{threshold}"] = round(float(reference.mean()), 2)
        # The share of the reference the OSM estimate recovers. Reported as a ratio
        # of means rather than a mean of ratios: origins that reach nothing in the
        # reference would otherwise divide by zero and be silently dropped.
        row[f"ratio_{threshold}"] = round(float(osm_reach.mean() / reference.mean()), 4) if reference.mean() else None
    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    row["status"] = "ok"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

    # A schedule with no run times cannot serve as a reference, so ``with_gtfs_graph``
    # is taken without ``include_untimed``. Cities that have no OSM graph stay in:
    # "OSM has nothing here" is a finding, and the row records it.
    keys = restrict(sorted(set(measurable()) & with_gtfs_graph()), arguments.cities)
    if arguments.limit:
        # The limit counts cities still to do, not cities in the cohort.
        done = set() if arguments.force else already_done(ACCESS_CSV)
        keys = [key for key in keys if key not in done][: arguments.limit]

    sweep(
        keys,
        measure_city,
        ACCESS_CSV,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: (
            f"{rows[0].get('status')} ratio 30 min {rows[0].get('ratio_30')} "
            f"({rows[0].get('osm_mean_30')} vs {rows[0].get('gtfs_mean_30')})"
        ),
    )


if __name__ == "__main__":
    main()
