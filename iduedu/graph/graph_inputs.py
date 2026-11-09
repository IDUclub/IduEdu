from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from iduedu.graph.urban_graph import UrbanGraph


def _object_points(urban_graph: UrbanGraph, gdf: gpd.GeoDataFrame, what: str) -> gpd.GeoSeries:
    """Representative points of ``gdf`` in the graph CRS, after checking the object table."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"gdf must be GeoDataFrame, got {type(gdf).__name__}")
    if gdf.empty:
        raise ValueError("gdf must not be empty")
    if gdf.index.has_duplicates:
        raise ValueError("gdf.index must be unique")
    if gdf.crs is None:
        raise ValueError(f"gdf must have CRS to calculate {what}")
    local_crs = getattr(urban_graph.nodes_gdf, "crs", None)
    if local_crs is None:
        local_crs = getattr(urban_graph.edges_gdf, "crs", None)
    if local_crs is None:
        raise ValueError("UrbanGraph does not have CRS on nodes_gdf or edges_gdf")
    points_geom = gdf.geometry
    if points_geom.crs != local_crs:
        points_geom = points_geom.to_crs(local_crs)
    return points_geom.representative_point()


def nearest_edges(
    urban_graph: UrbanGraph,
    gdf: gpd.GeoDataFrame,
    *,
    max_dist: float | None = None,
) -> pd.DataFrame:
    """Return the nearest graph edge for each input geometry, with the distance and the position along it.

    Objects are matched by their ``representative_point()`` in the graph CRS, as in :func:`nearest_nodes`.
    This does not change the graph; use :func:`project_objects2urban_graph` to insert objects into it.

    Args:
        urban_graph: Graph whose ``edges_gdf`` contains line geometries.
        gdf: GeoDataFrame with geometries to match to graph edges.
        max_dist: Optional search radius in graph CRS units; objects farther from every edge get missing
            values. ``None`` searches without a limit.

    Returns:
        DataFrame indexed like ``gdf`` with columns ``edge`` (label of the edge in ``edges_gdf``), ``u``,
        ``v`` (and ``k`` for multigraphs), ``dist`` (distance from the object to the edge) and ``along``
        (distance from the start of the edge geometry to the closest point on it). When several edges are
        equally near, one of them is returned.

    Raises:
        TypeError: If ``gdf`` is not a GeoDataFrame.
        ValueError: If the graph has no edges or geometries cannot be matched safely.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """
    edges_gdf = urban_graph.edges_gdf
    if not isinstance(edges_gdf, gpd.GeoDataFrame) or edges_gdf.empty:
        raise ValueError("graph has no edge geometries to match objects to")
    if max_dist is not None and (not np.isfinite(max_dist) or max_dist < 0):
        raise ValueError("max_dist must be a non-negative number or None")
    points_geom = _object_points(urban_graph, gdf, "nearest graph edges")

    matches, distances = edges_gdf.sindex.nearest(
        points_geom.values, return_all=False, max_distance=max_dist, return_distance=True
    )
    object_pos, edge_pos = matches
    found = edges_gdf.iloc[edge_pos]

    def column(values):
        # Missing rows for objects beyond max_dist; integer ids stay integers (nullable).
        series = pd.Series(values, index=gdf.index[object_pos]).reindex(gdf.index)
        if pd.api.types.is_integer_dtype(np.asarray(values).dtype):
            series = series.astype("Int64")
        return series

    keys = ["u", "v"] + (["k"] if urban_graph.is_multigraph and "k" in edges_gdf.columns else [])
    result = pd.DataFrame({"edge": column(found.index.to_numpy())}, index=gdf.index)
    for key in keys:
        result[key] = column(found[key].to_numpy())
    result["dist"] = column(np.asarray(distances, dtype=float))
    result["along"] = column(shapely.line_locate_point(found.geometry.values, points_geom.values[object_pos]))
    return result


def nearest_nodes(
    urban_graph: UrbanGraph,
    gdf: gpd.GeoDataFrame,
    *,
    graph_node_column: str = "graph_node_id",
) -> pd.Series:
    """Return the nearest graph node id for each input geometry.

    Args:
        urban_graph: Graph whose ``nodes_gdf`` contains point geometries.
        gdf: GeoDataFrame with geometries to match to graph nodes.
        graph_node_column: Name assigned to the returned ``Series``.

    Returns:
        Series indexed like ``gdf`` with nearest node ids as values.

    Raises:
        TypeError: If ``gdf`` is not a GeoDataFrame.
        ValueError: If graph or object geometries cannot be matched safely.

    See also:
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    nodes_gdf = urban_graph.nodes_gdf
    if not isinstance(nodes_gdf, gpd.GeoDataFrame):
        raise TypeError("UrbanGraph nodes_gdf must be GeoDataFrame to calculate nearest graph nodes")
    if nodes_gdf.empty:
        raise ValueError("graph is empty")
    if nodes_gdf.index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")
    points_geom = _object_points(urban_graph, gdf, "nearest graph nodes")

    matches = nodes_gdf.sindex.nearest(points_geom.values, return_all=False)
    order = np.argsort(matches[0], kind="stable")
    nearest_positions = matches[1][order]
    if len(nearest_positions) != len(gdf):
        raise ValueError("Could not find nearest graph node for some input geometries")

    node_ids = nodes_gdf.index.to_numpy()[nearest_positions]
    return pd.Series(node_ids, index=gdf.index, name=graph_node_column)


def resolve_graph_nodes_input(
    *,
    urban_graph: UrbanGraph,
    nodes: Iterable[Any] | None,
    gdf: gpd.GeoDataFrame | pd.DataFrame | None,
    graph_node_column: str,
    nodes_name: str,
    gdf_name: str,
) -> pd.Series:
    """Normalize ``nodes`` or ``gdf`` inputs into graph node ids.

    If ``gdf`` already contains ``graph_node_column``, values are read from
    that column. Otherwise ``gdf`` must be a GeoDataFrame and each geometry is
    matched to the nearest graph node. Plain ``nodes`` input is validated and
    returned as a Series indexed by node ids.
    """

    if nodes is not None and gdf is not None:
        raise ValueError(f"Pass either {nodes_name} or {gdf_name}, not both")
    if nodes is None and gdf is None:
        raise ValueError(f"Pass {nodes_name} or {gdf_name}")

    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if len(urban_graph.nodes_gdf) == 0:
        raise ValueError("graph is empty")

    graph_node_index = urban_graph.nodes_gdf.index
    if graph_node_index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")

    if gdf is not None:
        if not isinstance(gdf, (gpd.GeoDataFrame, pd.DataFrame)):
            raise TypeError(f"{gdf_name} must be DataFrame or GeoDataFrame, got {type(gdf).__name__}")
        if gdf.empty:
            raise ValueError(f"{gdf_name} must not be empty")
        if gdf.index.has_duplicates:
            raise ValueError(f"{gdf_name}.index must be unique")
        if graph_node_column in gdf.columns:
            if gdf[graph_node_column].isna().any():
                raise ValueError(f"{gdf_name}[{graph_node_column!r}] contains NaN")
            result = gdf[graph_node_column].copy()
        else:
            if not isinstance(gdf, gpd.GeoDataFrame):
                raise KeyError(f"{gdf_name} has no node column {graph_node_column!r} and no geometry")
            result = nearest_nodes(urban_graph, gdf, graph_node_column=graph_node_column)
    else:
        resolved_nodes = list(nodes)
        if not resolved_nodes:
            raise ValueError(f"{nodes_name} must not be empty")
        if pd.isna(pd.Series(resolved_nodes)).any():
            raise ValueError(f"{nodes_name} must not contain NaN")
        result = pd.Series(resolved_nodes, index=pd.Index(resolved_nodes), name=graph_node_column)

    missing_nodes = [node for node in result.to_numpy() if node not in graph_node_index]
    if missing_nodes:
        preview = missing_nodes[:10]
        raise ValueError(
            f"{nodes_name} contain nodes that are absent in graph: {preview}"
            + (" ..." if len(missing_nodes) > 10 else "")
        )

    return result
