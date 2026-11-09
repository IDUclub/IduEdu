"""Public shortest-path helpers and conversion of node routes to graph edges."""

from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import reverse as reverse_geometry

from iduedu.graph.routing import (
    ShortestPathResult,
    dijkstra_parallel,
    iter_od_matrix,
    multi_source_dijkstra,
    multi_source_dijkstra_nearest_source,
    od_matrix,
    single_source_dijkstra,
    single_source_dijkstra_path,
)
from iduedu.graph.urban_graph import UrbanGraph

PATH_ORDER_COLUMN = "path_order"
PATH_U_COLUMN = "path_u"
PATH_V_COLUMN = "path_v"


def path_to_edges(
    urban_graph: UrbanGraph,
    path: Iterable[Any],
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
) -> gpd.GeoDataFrame:
    """Convert an ordered node path to ordered graph edges.

    For parallel edges the first edge with the minimum selected ``weight`` is
    used, matching the default adjacency-matrix aggregation rule. Edge
    attributes are preserved and reverse traversals receive reversed geometry.

    Args:
        urban_graph: Spatial urban graph containing edge geometries.
        path: Ordered graph node ids, for example from
            :func:`single_source_dijkstra_path` or a path-matrix cell.
        weight: Edge weight used to choose between parallel edges.

    Returns:
        GeoDataFrame of traversed edges with ``path_order``, ``path_u`` and
        ``path_v`` columns.
    """

    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if not isinstance(urban_graph.edges_gdf, gpd.GeoDataFrame):
        raise TypeError("graph.edges_gdf must be GeoDataFrame to build route geometry")
    if weight not in urban_graph.edges_gdf.columns:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")

    path = list(path)
    missing_nodes = [node for node in path if node not in urban_graph.nodes_gdf.index]
    if missing_nodes:
        preview = missing_nodes[:10]
        raise ValueError(
            f"path contains nodes that are absent in graph: {preview}" + (" ..." if len(missing_nodes) > 10 else "")
        )

    empty_result = urban_graph.edges_gdf.iloc[0:0].copy()
    empty_result[PATH_ORDER_COLUMN] = pd.Series(dtype=np.int64)
    empty_result[PATH_U_COLUMN] = pd.Series(dtype=object)
    empty_result[PATH_V_COLUMN] = pd.Series(dtype=object)
    if len(path) < 2:
        return empty_result

    edges = urban_graph.edges_gdf
    if edges[weight].isna().any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] contains NaN")
    if (edges[weight] < 0).any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] must contain only non-negative values")

    # Scan the edge table once: only edges between path nodes can be steps.
    steps = {}
    for path_order, step in enumerate(zip(path[:-1], path[1:])):
        steps.setdefault(step, []).append(path_order)
    if urban_graph.edge_direction_column is not None:
        reverse_allowed = ~edges[urban_graph.edge_direction_column].astype(bool).to_numpy()
    else:
        reverse_allowed = np.full(len(edges), not urban_graph.is_directed)
    near = np.flatnonzero((edges["u"].isin(path) & edges["v"].isin(path)).to_numpy())
    # Per step keep the minimum (weight, edge position): the first lightest edge.
    best = {}
    for position, edge_u, edge_v, edge_weight, can_reverse in zip(
        near.tolist(),
        edges["u"].iloc[near].tolist(),
        edges["v"].iloc[near].tolist(),
        edges[weight].iloc[near].tolist(),
        reverse_allowed[near].tolist(),
    ):
        candidates = [((edge_u, edge_v), False)]
        if can_reverse and edge_u != edge_v:
            candidates.append(((edge_v, edge_u), True))
        for step, stored_reverse in candidates:
            for path_order in steps.get(step, ()):
                if path_order not in best or (edge_weight, position) < best[path_order][:2]:
                    best[path_order] = (edge_weight, position, stored_reverse)

    for path_order, (path_u, path_v) in enumerate(zip(path[:-1], path[1:])):
        if path_order not in best:
            raise ValueError(f"path contains an edge that is absent or disallowed in graph: ({path_u!r}, {path_v!r})")
    chosen = [best[path_order] for path_order in range(len(path) - 1)]
    selected = edges.iloc[[position for _, position, _ in chosen]].copy()
    stored_reverse = np.array([reverse for _, _, reverse in chosen])
    if stored_reverse.any():
        geometry = selected.geometry.to_numpy().copy()
        geometry[stored_reverse] = reverse_geometry(geometry[stored_reverse])
        selected[selected.geometry.name] = gpd.GeoSeries(geometry, crs=edges.crs).values
    selected[PATH_ORDER_COLUMN] = np.arange(len(path) - 1, dtype=np.int64)
    selected[PATH_U_COLUMN] = path[:-1]
    selected[PATH_V_COLUMN] = path[1:]
    return gpd.GeoDataFrame(selected, geometry=edges.geometry.name, crs=edges.crs)
