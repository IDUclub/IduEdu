"""Public shortest-path helpers and conversion of node routes to graph edges."""

from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import reverse as reverse_geometry

from iduedu.graph.routing import (
    ShortestPathResult,
    dijkstra_parallel,
    dijkstra_path_length_parallel,
    iter_od_matrix,
    multi_source_dijkstra,
    multi_source_dijkstra_nearest_source,
    multi_source_dijkstra_path,
    multi_source_dijkstra_path_length,
    od_matrix,
    single_source_dijkstra,
    single_source_dijkstra_path,
    single_source_dijkstra_path_length,
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

    selected_edges = []
    for path_order, (path_u, path_v) in enumerate(zip(path[:-1], path[1:])):
        forward_mask = (edges["u"] == path_u) & (edges["v"] == path_v)
        reverse_mask = (edges["u"] == path_v) & (edges["v"] == path_u)

        if urban_graph.edge_direction_column is not None:
            reverse_mask &= ~edges[urban_graph.edge_direction_column].astype(bool)
        elif urban_graph.is_directed:
            reverse_mask &= False

        candidates = edges.loc[forward_mask | reverse_mask]
        if candidates.empty:
            raise ValueError(f"path contains an edge that is absent or disallowed in graph: ({path_u!r}, {path_v!r})")

        candidate_weights = candidates[weight].to_numpy()
        selected_position = int(np.flatnonzero(candidate_weights == candidate_weights.min())[0])
        selected = candidates.iloc[[selected_position]].copy()
        stored_reverse = bool(
            selected.iloc[0]["u"] == path_v and selected.iloc[0]["v"] == path_u and not (path_u == path_v)
        )
        if stored_reverse:
            selected.geometry = selected.geometry.map(reverse_geometry)

        selected[PATH_ORDER_COLUMN] = path_order
        selected[PATH_U_COLUMN] = path_u
        selected[PATH_V_COLUMN] = path_v
        selected_edges.append(selected)

    return gpd.GeoDataFrame(
        pd.concat(selected_edges, axis=0), geometry=urban_graph.edges_gdf.geometry.name, crs=urban_graph.edges_gdf.crs
    )
