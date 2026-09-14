"""Build the schedule side of the comparison: one reference graph per city feed.

This is the expensive step three later ones stand on, which is why it is its own
module and its own cache:

* **waiting times** (T-64) come from the ``boarding`` edges of these graphs
  rather than from a headway formula written here -- so the number the paper
  reports is the number the model actually uses;
* **segment matching** (T-38) needs GTFS run times next to OSM run times, and
  the graph is where a feed's segments acquire length and time;
* **accessibility** (T-69) uses these graphs as the reference the OSM estimate is
  measured against.

No service date and no time window. With all three filters omitted every trip in
the feed participates, which is the aggregate all-service graph the study wants:
a window would silently rank cities by how well their operators fill
``calendar.txt`` rather than by their service.

Feeds come from ``paper2026/city_feeds/*.zip`` -- already merged across a city's
sources and clipped to its OSM boundary, so the graph and its OSM counterpart
cover the same ground.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from bench_common import sweep
from wide_cohort import measurable, restrict
from wide_paths import (
    ALL_OSM_MODES,
    GTFS_GRAPH_DIR,
    GTFS_GRAPHS_CSV,
    city_feed_path,
    gtfs_graph_path,
    osm_graph_path,
)

from iduedu import UrbanGraph, get_gtfs_public_transport_graph, read_urban_graph, write_urban_graph

logger = logging.getLogger(__name__)

#: A pattern-stop with a single scheduled departure has no measurable headway.
#: Left at ``None`` the library drops its boarding edge, which quietly removes
#: once-a-day service from the graph -- and once-a-day service is exactly what
#: distinguishes the cities this study is about. The count of such stops is
#: recorded per city so the choice can be revisited with evidence.
SINGLE_DEPARTURE_WAIT_MIN = 60.0


def _city_crs(city_key: str):
    """Adopt the walk graph's CRS so the two sides can be joined later.

    Both sides estimate a local UTM zone, but from different extents -- the feed's
    stops on one side, the OSM boundary on the other -- and near a zone edge they
    disagree. Arequipa, Buenos Aires and Tokyo each landed one zone apart, which
    ``join_pt_walk_graph`` rejects outright. Taking the CRS from the walk graph
    settles it by construction rather than per city.
    """
    for path in (osm_graph_path(city_key, "walk"), osm_graph_path(city_key, "pt", ALL_OSM_MODES)):
        if path.exists():
            try:
                return read_urban_graph(path).nodes_gdf.crs
            except Exception:  # noqa: BLE001 - an unreadable cache must not decide the CRS
                continue
    return None


#: Edge types that are service, not access.
TRAVEL_TYPES = {"bus", "trolleybus", "tram", "subway", "train", "ferry", "monorail", "funicular", "coach"}


def _summarise(graph: UrbanGraph) -> dict:
    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    summary = {"n_nodes": len(nodes), "n_edges": len(edges)}
    if not edges.empty and "type" in edges.columns:
        counts = edges["type"].value_counts().to_dict()
        summary["edge_types"] = ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        boarding = edges.loc[edges["type"].eq("boarding"), "time_min"]
        if not boarding.empty:
            summary["wait_median_min"] = round(float(boarding.median()), 3)

        # A segment that takes no time is worse than a segment that is missing:
        # it hands the network free travel. Curitiba writes both timepoints of
        # every trip at the same clock minute -- 66 840 trips, all nominally
        # instantaneous -- and interpolation dutifully spreads that zero across
        # all fifty stops. Counted here so no later stage consumes it unaware.
        travel = edges.loc[edges["type"].isin(TRAVEL_TYPES), "time_min"]
        if not travel.empty:
            zero = int((travel <= 0).sum())
            summary["n_travel_edges"] = len(travel)
            summary["zero_time_edges"] = zero
            summary["zero_time_share"] = round(zero / len(travel), 4)
    if not nodes.empty and "type" in nodes.columns:
        modes = sorted(set(nodes["type"].dropna().map(str)) - {"platform", "station", "entrance", "boarding_area"})
        summary["node_types"] = ";".join(modes)
    return summary


def build_city(feed: Path, force: bool = False) -> dict:
    """Build one city's reference graph and return the manifest row."""
    city_key = feed.stem
    destination = gtfs_graph_path(city_key)
    row = {"city_key": city_key, "path": str(destination), "n_nodes": 0, "n_edges": 0}

    if destination.exists() and not force:
        row["status"] = "cached"
        return row

    started = time.perf_counter()
    try:
        graph = get_gtfs_public_transport_graph(
            feed, crs=_city_crs(city_key), single_departure_wait_min=SINGLE_DEPARTURE_WAIT_MIN
        )
    except Exception as error:  # noqa: BLE001 - one bad feed must not end the sweep
        row["status"] = "failed"
        row["reason"] = f"{type(error).__name__}: {error}"[:300]
        logger.warning(f"{city_key}: {row['reason']}")
        return row

    row["elapsed_s"] = round(time.perf_counter() - started, 1)
    if graph.nodes_gdf.empty:
        row["status"] = "empty"
        return row

    GTFS_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    write_urban_graph(graph, destination)
    row.update(_summarise(graph))
    row["crs"] = str(graph.nodes_gdf.crs)
    # A graph whose every segment is instantaneous carries no schedule at all.
    # It still holds stops and headways, so it is kept and marked rather than
    # dropped: coverage and waiting times remain measurable, run times do not.
    row["status"] = "no_usable_times" if row.get("zero_time_share") == 1.0 else "ok"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", help="city keys; default is every merged feed on disk")
    parser.add_argument("--force", action="store_true", help="rebuild graphs already cached")
    parser.add_argument("--limit", type=int, help="stop after this many cities")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    keys = restrict(measurable(), arguments.cities)
    if arguments.limit:
        keys = keys[: arguments.limit]

    # A failed build is retried; only a graph that exists or is provably empty is
    # finished. This is why the stage states its own done set.
    done: set[str] = set()
    if GTFS_GRAPHS_CSV.exists():
        manifest = pd.read_csv(GTFS_GRAPHS_CSV)
        done = {str(key) for key in manifest.loc[manifest["status"].isin(["ok", "empty"]), "city_key"]}

    sweep(
        keys,
        lambda city_key: build_city(city_feed_path(city_key), arguments.force),
        GTFS_GRAPHS_CSV,
        force=arguments.force,
        done_keys=done,
        logger=logger,
        describe=lambda rows: (
            f"{rows[0]['status']} {rows[0].get('n_nodes', 0)} nodes, {rows[0].get('n_edges', 0)} edges"
        ),
    )


if __name__ == "__main__":
    main()
