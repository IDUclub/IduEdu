#!/usr/bin/env python3
"""Simplification ablation: the time of a traversal on a simplified and an unsimplified graph.

``bench_build.py`` measures what simplification adds to the build time; this script
measures what it saves afterwards. For every city and network, IduEdu builds both
graphs (the builds are not timed here), and a full single-source shortest-path
traversal is timed on each from the same places. Together with the build times, this
gives the number of traversals after which a simplified walk graph has made up for its
longer build; on drive networks simplification shortens the build as well.

The sources are the same places on both graphs: ``N_SOURCES`` nodes drawn from the
unsimplified graph with a fixed seed and snapped to the nearest node of the simplified
graph. A traversal is ``UrbanGraph.single_source_dijkstra`` with distances only and no
cutoff, so it reaches every vertex of the largest component and includes building the
returned Series, as a user of the library would see it. One traversal per graph runs
first and is not recorded: it compiles the Numba kernels.

The graphs are built from the Overpass responses that the build stage cached, so this
stage downloads nothing when it runs after ``bench_build.py``. One CSV row per (city,
network, simplify, attempt); resume-safe.

Usage:
    python bench_simplify.py
    python bench_simplify.py --smoke
    python bench_simplify.py --areas "Helsinki" --networks walk
"""

import argparse
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from bench_common import AREAS, RESULTS_DIR, bbox_from_pbf, dump_environment, force_cleanup, make_key, resolve_area_pbf
from timing_paths import use_paper_cache

OUT_CSV = RESULTS_DIR / "simplify_benchmark.csv"
KEY_COLUMNS = ["area", "network", "simplify", "attempt"]
NETWORKS = ["walk", "drive"]
N_SOURCES = 32
SEED = 20261028
SMOKE_OSM_ID = 1114252

#: Measurements per cell: one unless BENCH_ATTEMPTS (or ``run_all.py --attempts``) asks for
#: more. Every row is keyed by its attempt, so a later run adds attempts to the same CSV.
ATTEMPTS = max(1, int(os.environ.get("BENCH_ATTEMPTS", "1")))


def existing_keys() -> set:
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
    return {make_key(row[c] for c in KEY_COLUMNS) for _, row in df.iterrows()}


def append(row: dict) -> None:
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)


def build(polygon, network: str, simplify: bool):
    from iduedu import get_drive_graph, get_walk_graph

    builder = get_walk_graph if network == "walk" else get_drive_graph
    return builder(territory=polygon, simplify=simplify, keep_largest_subgraph=True)


def sources(raw, simplified) -> tuple[list, list]:
    """The same places on both graphs: nodes of the raw graph, snapped onto the simplified one."""
    from iduedu import nearest_nodes

    rng = np.random.default_rng(SEED)
    picked = raw.nodes_gdf.iloc[rng.choice(len(raw.nodes_gdf), size=N_SOURCES, replace=False)]
    points = gpd.GeoDataFrame(geometry=picked.geometry.values, crs=raw.nodes_gdf.crs)
    return list(picked.index), list(nearest_nodes(simplified, points).values)


def time_traversals(graph, nodes: list) -> float:
    start = time.perf_counter()
    for node in nodes:
        graph.single_source_dijkstra(node, weight="time_min", output="distances")
    return time.perf_counter() - start


def measure_area(existing: set, area: str, polygon, networks: list[str]) -> None:
    for network in networks:
        pending = [
            (simplify, attempt)
            for simplify in (False, True)
            for attempt in range(1, ATTEMPTS + 1)
            if make_key([area, network, simplify, attempt]) not in existing
        ]
        if not pending:
            print(f"  [skip] {area} {network}: recorded")
            continue
        print(f"  [build] {area} {network} (not timed)")
        graphs = {False: build(polygon, network, False), True: build(polygon, network, True)}
        raw_nodes, simplified_nodes = sources(graphs[False], graphs[True])
        nodes = {False: raw_nodes, True: simplified_nodes}
        for simplify, graph in graphs.items():
            graph.single_source_dijkstra(nodes[simplify][0], weight="time_min", output="distances")
        for simplify, attempt in pending:
            graph = graphs[simplify]
            total = time_traversals(graph, nodes[simplify])
            append(
                dict(
                    area=area,
                    network=network,
                    simplify=simplify,
                    attempt=attempt,
                    n_nodes=len(graph.nodes_gdf),
                    n_edges=len(graph.edges_gdf),
                    n_sources=N_SOURCES,
                    time_total_sec=round(total, 4),
                    time_per_traversal_sec=round(total / N_SOURCES, 5),
                )
            )
            existing.add(make_key([area, network, simplify, attempt]))
            print(
                f"  [run ] {area} {network} simplify={simplify} attempt={attempt}: {total / N_SOURCES:.3f} s per traversal"
            )
        for graph in graphs.values():
            force_cleanup(graph)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one small district, quick harness check")
    parser.add_argument("--areas", default=None, help="comma-separated subset of areas")
    parser.add_argument("--networks", default=",".join(NETWORKS))
    args = parser.parse_args()

    use_paper_cache()
    dump_environment("simplify")
    existing = existing_keys()
    networks = [n.strip() for n in args.networks.split(",") if n.strip()]

    if args.smoke:
        from iduedu import get_4326_boundary

        measure_area(existing, "smoke", get_4326_boundary(osm_id=SMOKE_OSM_ID), networks)
        print(f"[done] smoke results -> {OUT_CSV}")
        return

    areas = [a.strip() for a in args.areas.split(",")] if args.areas else AREAS
    for area in areas:
        try:
            polygon = bbox_from_pbf(resolve_area_pbf(area)).polygon_4326
            print(f"\n=== {area} ===")
            measure_area(existing, area, polygon, networks)
        except Exception as error:  # noqa: BLE001 - one city must not end the sweep
            print(f"[fail] {area}: {type(error).__name__}: {error}", flush=True)
    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
