#!/usr/bin/env python3
"""
B4 — Validity checks (critical for reviewers: speed claims mean nothing if the
results differ).

V1. OD correctness: IduEdu `od_matrix` vs NetworkX Dijkstra on the identical
    CSR-derived graph. Reports max |Δ| over a sampled OD block. IduEdu quantizes
    weights internally, so a sub-0.01-minute tolerance is expected, not exact 0.

V2. Build similarity: IduEdu vs OSMnx walk graphs on the same polygon —
    node/edge counts, total edge length (km), share of nodes in the largest
    component. The counts differ by design (different simplification rules);
    total network length is the comparable invariant.

Output: results/validity_report.json (+ console summary).

Usage:
    python bench_validity.py [--smoke]     # smoke == same small territory, quick
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from bench_common import RESULTS_DIR, dump_environment, measure, urban_to_networkx
from bench_od import load_or_build_graph

OUT_JSON = RESULTS_DIR / "validity_report.json"

WEIGHT = "time_min"
N_SAMPLE_SOURCES = 30
N_SAMPLE_TARGETS = 50
SEED = 11
SMOKE_OSM_ID = 1114252


def check_od_correctness(graph) -> dict:
    import networkx as nx

    from iduedu import od_matrix

    # Reference graph = the honest NetworkX MultiDiGraph (same node ids as UrbanGraph).
    nx_graph = urban_to_networkx(graph, WEIGHT)
    nodelist = graph.nodes_gdf.index.to_numpy()

    rng = np.random.default_rng(SEED)
    src_ids = rng.choice(nodelist, size=min(N_SAMPLE_SOURCES, len(nodelist)), replace=False)
    dst_ids = rng.choice(nodelist, size=min(N_SAMPLE_TARGETS, len(nodelist)), replace=False)

    iduedu_mat = od_matrix(
        graph,
        origins_nodes=src_ids.tolist(),
        destination_nodes=dst_ids.tolist(),
        weight=WEIGHT,
    ).to_numpy(dtype=float)

    nx_mat = np.full_like(iduedu_mat, np.inf)
    for i, sp in enumerate(src_ids):
        dist = nx.single_source_dijkstra_path_length(nx_graph, int(sp), weight="weight")
        for j, dp in enumerate(dst_ids):
            nx_mat[i, j] = dist.get(int(dp), np.inf)

    both_finite = np.isfinite(iduedu_mat) & np.isfinite(nx_mat)
    finite_mismatch = int((np.isfinite(iduedu_mat) != np.isfinite(nx_mat)).sum())
    max_abs_diff = float(np.abs(iduedu_mat[both_finite] - nx_mat[both_finite]).max()) if both_finite.any() else 0.0

    report = dict(
        n_sources=len(src_ids),
        n_targets=len(dst_ids),
        finite_pairs=int(both_finite.sum()),
        reachability_mismatches=finite_mismatch,
        max_abs_diff_min=round(max_abs_diff, 6),
    )
    print(
        f"[V1] OD correctness: max_abs_diff={report['max_abs_diff_min']} min, "
        f"reachability mismatches={finite_mismatch}"
    )
    return report


def check_build_similarity(polygon) -> dict:
    import osmnx as ox

    from iduedu import get_walk_graph

    ox.settings.use_cache = True

    m_idu = measure(get_walk_graph, territory=polygon, simplify=True, keep_largest_subgraph=True)
    graph_idu = m_idu.result
    total_km_idu = float(graph_idu.edges_gdf["length_meter"].sum() / 1000)

    m_ox = measure(ox.graph_from_polygon, polygon=polygon, network_type="walk", simplify=True, retain_all=False)
    graph_ox = m_ox.result
    graph_ox_proj = ox.project_graph(graph_ox)
    total_km_ox = float(sum(d["length"] for _, _, d in graph_ox_proj.edges(data=True)) / 1000)
    # OSMnx walk graphs duplicate every edge in both directions; IduEdu stores
    # undirected walk edges once. Halve the OSMnx total for comparability.
    total_km_ox_undirected = total_km_ox / 2

    report = dict(
        iduedu=dict(
            nodes=len(graph_idu.nodes_gdf),
            edges=len(graph_idu.edges_gdf),
            total_km=round(total_km_idu, 1),
            build_sec=round(m_idu.time_sec, 2),
        ),
        osmnx=dict(
            nodes=graph_ox.number_of_nodes(),
            edges=graph_ox.number_of_edges(),
            total_km_directed=round(total_km_ox, 1),
            total_km_undirected=round(total_km_ox_undirected, 1),
            build_sec=round(m_ox.time_sec, 2),
        ),
        total_length_ratio=round(total_km_idu / total_km_ox_undirected, 4) if total_km_ox_undirected else None,
    )
    print(
        f"[V2] total length: iduedu={report['iduedu']['total_km']} km, "
        f"osmnx/2={report['osmnx']['total_km_undirected']} km, "
        f"ratio={report['total_length_ratio']}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    dump_environment("validity")

    from iduedu import get_4326_boundary

    graph = load_or_build_graph(args.smoke)
    polygon = get_4326_boundary(osm_id=SMOKE_OSM_ID)  # V2 always runs on the small AOI

    report = dict(
        od_correctness=check_od_correctness(graph),
        build_similarity=check_build_similarity(polygon),
    )
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] report -> {OUT_JSON}")


if __name__ == "__main__":
    main()
