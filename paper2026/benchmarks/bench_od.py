#!/usr/bin/env python3
"""
B3 — OD-matrix benchmark: IduEdu vs NetworKit vs igraph.

Two settings, selected by --mode:
  rect   : origins = residential buildings (growing |O|), destinations = schools (fixed |D|)
  square : origins = destinations = the same building subset (|O| = |D|)

Honest fairness protocol
------------------------
The task is a *geospatial* OD query: given spatial objects (buildings, schools),
compute shortest-path times between them on the city graph. The starting graph
representation differs per tool exactly as it does in practice:

- IduEdu consumes its own ``UrbanGraph`` (the output of its builders) and
  ``od_matrix`` snaps the objects to nodes internally via the persistent
  GeoDataFrame R-tree. So for IduEdu there is no conversion and no separate snap.
- Competitors start from a NetworkX ``MultiDiGraph`` — the representation the
  osmnx/networkx ecosystem produces (UrbanGraph did not exist before). They pay
  the real costs a networkit/igraph user pays:
    * convert : NetworkX -> the library's own graph (min-collapse parallel edges);
    * snap    : build a spatial index (scipy cKDTree) over node coordinates and
                query the object points to nearest nodes.

The NetworkX graph is derived from the *same* UrbanGraph, so all tools route on an
identical network. We record snap / convert / od separately and their end-to-end
total. (Plain NetworkX routing is not part of the OD comparison — it has no
batched OD API and is orders of magnitude slower; NetworkX is used only as the
common source representation the competitors convert from.)

Resume-safe CSV: one row per (library, mode, n_sources, threshold, attempt).

Usage:
    python bench_od.py --mode rect
    python bench_od.py --mode square
    python bench_od.py --mode rect --smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from bench_common import (
    RESULTS_DIR,
    append_row,
    bbox_from_pbf,
    dump_environment,
    force_cleanup,
    gdf_points_xy,
    load_existing_keys,
    make_key,
    measure,
    node_coords,
    nx_to_igraph,
    nx_to_networkit,
    resolve_area_pbf,
    urban_to_networkx,
)

OUT_CSV = RESULTS_DIR / "od_benchmark.csv"
KEY_COLUMNS = ["library", "mode", "n_sources", "threshold_min", "attempt"]

DEV_DIR = Path(__file__).resolve().parents[2]  # dev/
ORIGINS_PATH = DEV_DIR / "buildings_spb.parquet"
DEST_PATH = DEV_DIR / "school.geojson"
GRAPH_PATH = RESULTS_DIR / "spb_intermodal.urbangraph"
SMOKE_GRAPH_PATH = RESULTS_DIR / "smoke.urbangraph"

ATTEMPTS = 3
SEED = 42
SLEEP_SEC = 1.0
WEIGHT = "time_min"

# Rectangular OD grows |O| far past |D| (cheap: min(|O|,|D|) Dijkstra runs via graph
# reversal); square OD needs |O| runs, so it is capped lower — matches the v1 grid.
N_SOURCES_RECT = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
N_SOURCES_SQUARE = [128, 256, 512, 1024, 2048, 4096, 8192]
THRESHOLDS_MIN = [5, 15, 30, 60]

# Exclusive caps: competitors become impractically slow past these |O|.
MAX_IGRAPH_SOURCES = 8192
MAX_NETWORKIT_SOURCES = 65536
SMOKE_OSM_ID = 1114252


# ----------------------------
# Graph preparation
# ----------------------------


def load_or_build_graph(smoke: bool):
    from iduedu import get_intermodal_graph, read_urban_graph, write_urban_graph

    path = SMOKE_GRAPH_PATH if smoke else GRAPH_PATH
    if path.exists():
        print(f"[graph] loading {path}")
        return read_urban_graph(path)

    if smoke:
        from iduedu import get_4326_boundary

        polygon = get_4326_boundary(osm_id=SMOKE_OSM_ID)
    else:
        pbf_path = resolve_area_pbf("Saint Petersburg")
        polygon = bbox_from_pbf(pbf_path).polygon_4326

    print("[graph] building intermodal graph (one-off, cached to disk afterwards)")
    graph = get_intermodal_graph(territory=polygon, keep_largest_subgraph=True)

    # Keep only routing-relevant columns: intermodal node attributes may hold list
    # values (merged PT `route` lists) that the parquet graph I/O cannot serialize.
    from iduedu import UrbanGraph

    node_cols = [graph.nodes_gdf.geometry.name]
    edge_keep = ["u", "v", "k", "geometry", "length_meter", "time_min", "oneway", "type"]
    graph = UrbanGraph(
        nodes_gdf=graph.nodes_gdf[node_cols].copy(),
        edges_gdf=graph.edges_gdf[[c for c in edge_keep if c in graph.edges_gdf.columns]].copy(),
        is_multigraph=graph.is_multigraph,
        is_directed=graph.is_directed,
        edge_direction_column=graph.edge_direction_column,
        crs=graph.crs,
        graph_type=graph.type,
    )
    write_urban_graph(graph, path)
    return graph


# ----------------------------
# OD computations
# ----------------------------


def od_iduedu(urban_graph, from_gdf, to_gdf, threshold):
    """IduEdu end-to-end: snapping is internal (GeoDataFrame R-tree), no conversion."""
    from iduedu import od_matrix

    return od_matrix(
        urban_graph,
        gdf_origins=from_gdf,
        gdf_destinations=to_gdf,
        weight=WEIGHT,
        threshold=threshold,
    )


def od_networkit(nk_graph, sources: list[int], targets: list[int]) -> np.ndarray:
    import networkit as nk

    spsp = nk.distance.SPSP(nk_graph, sources, targets)
    spsp.run()
    return np.asarray(spsp.getDistances(), dtype=float)


def od_igraph(ig_graph, sources: list[int], targets: list[int]) -> np.ndarray:
    uniq: dict[int, int] = {}
    back = []
    for t in targets:
        back.append(uniq.setdefault(int(t), len(uniq)))
    dist = ig_graph.distances(source=sources, target=list(uniq), weights="weight")
    return np.asarray(dist, dtype=float)[:, back]


def inf_share(mat) -> float:
    arr = np.asarray(mat, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(~np.isfinite(arr)))


def snap_points(node_xy: np.ndarray, node_ids: np.ndarray, points_xy: np.ndarray):
    """Build a fresh cKDTree over node coords and return (build+query time, nearest node ids)."""
    from scipy.spatial import cKDTree

    m_build = measure(cKDTree, node_xy)
    kdtree = m_build.result
    m_query = measure(kdtree.query, points_xy)
    _, positions = m_query.result
    return m_build.time_sec + m_query.time_sec, node_ids[positions]


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rect", "square"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--libraries", default="iduedu,networkit,igraph")
    args = parser.parse_args()
    mode = args.mode
    libraries = {x.strip() for x in args.libraries.split(",")}

    dump_environment(f"od_{mode}")
    existing = load_existing_keys(OUT_CSV, KEY_COLUMNS)
    if existing:
        print(f"[resume] {len(existing)} measurements already in {OUT_CSV}")

    import geopandas as gpd

    graph = load_or_build_graph(args.smoke)
    n_nodes, n_edges = len(graph.nodes_gdf), len(graph.edges_gdf)
    print(f"[graph] nodes={n_nodes:,} edges={n_edges:,}")

    # Competitor input: the equivalent NetworkX graph (built once, not charged —
    # it is the representation their users already have from osmnx).
    need_competitors = bool(libraries - {"iduedu"})
    nx_graph = None
    node_ids = node_xy = None
    if need_competitors:
        print("[graph] deriving competitor NetworkX MultiDiGraph")
        nx_graph = urban_to_networkx(graph, WEIGHT)
        node_ids, node_xy = node_coords(nx_graph)

    if args.smoke:
        rng = np.random.default_rng(SEED)
        nodes = graph.nodes_gdf
        origins = nodes.sample(min(600, len(nodes)), random_state=SEED)[["geometry"]].reset_index(drop=True)
        dest = nodes.sample(min(40, len(nodes)), random_state=SEED + 1)[["geometry"]].reset_index(drop=True)
        n_sources_list = [64, 256]
        thresholds = [15]
        attempts = 1
    else:
        origins = gpd.read_parquet(ORIGINS_PATH)
        dest = gpd.read_file(DEST_PATH)
        n_sources_list = N_SOURCES_SQUARE if mode == "square" else N_SOURCES_RECT
        thresholds = THRESHOLDS_MIN
        attempts = ATTEMPTS
    print(f"[data] origins={len(origins):,} destinations={len(dest):,}")

    # Warm-up (not recorded): Numba JIT for iduedu, competitor imports/first calls.
    print("[warm] iduedu numba kernels + competitor imports")
    warm_gdf = origins.iloc[:8]
    od_iduedu(graph, warm_gdf, warm_gdf, None)
    od_iduedu(graph, warm_gdf, warm_gdf, 5)
    if need_competitors:
        warm_xy = gdf_points_xy(warm_gdf, graph.crs)
        _, warm_ids = snap_points(node_xy, node_ids, warm_xy)
        if "networkit" in libraries:
            g_nk, pos_nk = nx_to_networkit(nx_graph)
            wp = [pos_nk[int(x)] for x in warm_ids]
            od_networkit(g_nk, wp, wp)
        if "igraph" in libraries:
            g_ig, pos_ig = nx_to_igraph(nx_graph)
            wp = [pos_ig[int(x)] for x in warm_ids]
            od_igraph(g_ig, wp, wp)

    rng = np.random.default_rng(SEED)
    selections = {
        n: np.sort(rng.choice(len(origins), size=n, replace=False)) for n in n_sources_list if n <= len(origins)
    }

    for n_sources, sel in selections.items():
        from_gdf = origins.iloc[sel].copy()
        to_gdf = from_gdf if mode == "square" else dest
        n_targets = len(to_gdf)

        for attempt in range(1, attempts + 1):
            # --- IduEdu: full + thresholds (snap internal, no conversion) ---
            if "iduedu" in libraries:
                for threshold in [None, *thresholds]:
                    key = make_key(["iduedu", mode, n_sources, threshold, attempt])
                    if key in existing:
                        print(f"  [skip] {key}")
                        continue
                    print(f"  [run ] iduedu {mode} |O|={n_sources} thr={threshold} attempt={attempt}")
                    m = measure(od_iduedu, graph, from_gdf, to_gdf, threshold)
                    append_row(
                        OUT_CSV,
                        dict(
                            library="iduedu",
                            mode=mode,
                            n_sources=n_sources,
                            n_targets=n_targets,
                            threshold_min="NA" if threshold is None else threshold,
                            attempt=attempt,
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            time_snap_sec=0.0,
                            time_convert_sec=0.0,
                            time_od_sec=round(m.time_sec, 3),
                            time_total_sec=round(m.time_sec, 3),
                            inf_share=round(inf_share(m.result), 5),
                        ),
                    )
                    existing.add(key)
                    force_cleanup(m)
                    time.sleep(SLEEP_SEC)

            # --- Competitors: convert (nx->lib) + snap (cKDTree) + od ---
            competitor_specs = []
            if "networkit" in libraries and n_sources < MAX_NETWORKIT_SOURCES:
                competitor_specs.append("networkit")
            if "igraph" in libraries and n_sources < MAX_IGRAPH_SOURCES:
                competitor_specs.append("igraph")

            pending = [
                lib for lib in competitor_specs if make_key([lib, mode, n_sources, None, attempt]) not in existing
            ]
            if not pending:
                continue

            # Snapping (build cKDTree + query) is identical for all competitors:
            # measure once, charge each. Returns nearest NetworkX node ids.
            from_xy = gdf_points_xy(from_gdf, graph.crs)
            to_xy = gdf_points_xy(to_gdf, graph.crs)
            t_snap_src, from_nx = snap_points(node_xy, node_ids, from_xy)
            t_snap_dst, to_nx = snap_points(node_xy, node_ids, to_xy)
            t_snap = t_snap_src + t_snap_dst

            for lib_name in pending:
                key = make_key([lib_name, mode, n_sources, None, attempt])
                print(f"  [run ] {lib_name} {mode} |O|={n_sources} attempt={attempt}")

                if lib_name == "networkit":
                    m_conv = measure(nx_to_networkit, nx_graph)
                    g_nk, pos = m_conv.result
                    t_convert = m_conv.time_sec
                    src = [pos[int(x)] for x in from_nx]
                    dst = [pos[int(x)] for x in to_nx]
                    m_od = measure(od_networkit, g_nk, src, dst)
                    force_cleanup(g_nk, m_conv)
                else:  # igraph
                    m_conv = measure(nx_to_igraph, nx_graph)
                    g_ig, pos = m_conv.result
                    t_convert = m_conv.time_sec
                    src = [pos[int(x)] for x in from_nx]
                    dst = [pos[int(x)] for x in to_nx]
                    m_od = measure(od_igraph, g_ig, src, dst)
                    force_cleanup(g_ig, m_conv)

                append_row(
                    OUT_CSV,
                    dict(
                        library=lib_name,
                        mode=mode,
                        n_sources=n_sources,
                        n_targets=n_targets,
                        threshold_min="NA",
                        attempt=attempt,
                        n_nodes=n_nodes,
                        n_edges=n_edges,
                        time_snap_sec=round(t_snap, 3),
                        time_convert_sec=round(t_convert, 3),
                        time_od_sec=round(m_od.time_sec, 3),
                        time_total_sec=round(t_snap + t_convert + m_od.time_sec, 3),
                        inf_share=round(inf_share(m_od.result), 5),
                    ),
                )
                existing.add(key)
                force_cleanup(m_od)
                time.sleep(SLEEP_SEC)

        force_cleanup(from_gdf)

    print(f"\n[done] results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
