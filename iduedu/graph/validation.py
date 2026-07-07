from typing import Any

import geopandas as gpd
import pandas as pd
import pyproj


def gdf_crs(frame) -> Any | None:
    """Return a GeoDataFrame CRS or ``None`` for non-geospatial frames."""
    if not isinstance(frame, gpd.GeoDataFrame):
        return None
    try:
        return frame.crs
    except AttributeError:
        return None


def sync_graph_crs(graph) -> None:
    """Synchronize graph, node and edge CRS metadata.

    Raises:
        ValueError: If node and edge CRS values conflict with the graph CRS.
    """
    nodes_crs = gdf_crs(graph.nodes_gdf)
    edges_crs = gdf_crs(graph.edges_gdf)
    inferred_crs = graph.crs or nodes_crs or edges_crs

    if inferred_crs is None:
        graph.crs = None
        return

    inferred_crs = pyproj.CRS.from_user_input(inferred_crs)

    for name, frame_crs in (("nodes_gdf", nodes_crs), ("edges_gdf", edges_crs)):
        if frame_crs is not None and frame_crs != inferred_crs:
            raise ValueError(f"{name}.crs={frame_crs} does not match graph crs={inferred_crs}")

    if isinstance(graph.nodes_gdf, gpd.GeoDataFrame) and nodes_crs is None and graph.nodes_gdf.active_geometry_name:
        graph.nodes_gdf = graph.nodes_gdf.set_crs(inferred_crs)
    if isinstance(graph.edges_gdf, gpd.GeoDataFrame) and edges_crs is None and graph.edges_gdf.active_geometry_name:
        graph.edges_gdf = graph.edges_gdf.set_crs(inferred_crs)

    graph.crs = inferred_crs


def validate_nodes(graph) -> None:
    """Validate the node table contract of an ``UrbanGraph``.

    Raises:
        TypeError: If ``nodes_gdf`` is not a DataFrame-like object.
        ValueError: If node ids are duplicated or geometries are invalid.
    """
    nodes = graph.nodes_gdf
    if not isinstance(nodes, (gpd.GeoDataFrame, pd.DataFrame)):
        raise TypeError(f"nodes_gdf must be DataFrame or GeoDataFrame, got {type(nodes).__name__}")
    if nodes.index.has_duplicates:
        raise ValueError("nodes_gdf.index must be unique")
    if isinstance(nodes, gpd.GeoDataFrame):
        if nodes.geometry.isna().any():
            raise ValueError("nodes_gdf.geometry contains NaN")
        if (~nodes.geometry.geom_type.isin(["Point"])).any():
            raise ValueError("All nodes_gdf geometries must be Point")


def validate_edges(graph) -> None:
    """Validate the edge table contract of an ``UrbanGraph``.

    Raises:
        TypeError: If ``edges_gdf`` is not a DataFrame-like object.
        ValueError: If required columns, topology keys or geometries are invalid.
    """
    edges = graph.edges_gdf
    if not isinstance(edges, (gpd.GeoDataFrame, pd.DataFrame)):
        raise TypeError(f"edges_gdf must be DataFrame or GeoDataFrame, got {type(edges).__name__}")
    if edges.empty:
        return
    required = {"u", "v", "geometry", "length_meter", "time_min"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edges_gdf missing required columns: {sorted(missing)}")
    if edges[["u", "v"]].isna().any().any():
        raise ValueError("edges_gdf columns ['u', 'v'] must not contain NaN")
    if graph.edge_direction_column is not None:
        if graph.edge_direction_column not in edges.columns:
            raise ValueError(f"edges_gdf missing edge_direction_column {graph.edge_direction_column!r}")
        if edges[graph.edge_direction_column].isna().any():
            raise ValueError(f"edges_gdf[{graph.edge_direction_column!r}] contains NaN")
        values = set(edges[graph.edge_direction_column].dropna().unique())
        if not values <= {False, True, 0, 1}:
            raise ValueError(f"edges_gdf[{graph.edge_direction_column!r}] must contain only boolean values")
    if graph.is_multigraph:
        if "k" not in edges.columns:
            raise ValueError("edges_gdf must contain 'k' for multigraph")
        if edges[["u", "v", "k"]].duplicated().any():
            raise ValueError("edges_gdf must have unique ['u','v','k']")
    else:
        if edges[["u", "v"]].duplicated().any():
            raise ValueError("edges_gdf must have unique ['u','v'] for non-multigraph")
    if isinstance(edges, gpd.GeoDataFrame):
        if edges.geometry.isna().any():
            raise ValueError("edges_gdf.geometry contains NaN")
        if (~edges.geometry.geom_type.isin(["LineString"])).any():
            raise ValueError("All edges_gdf geometries must be LineString")


def validate_nodes_edges(graph) -> None:
    """Validate consistency between node ids, edge endpoints and CRS values."""
    nodes = graph.nodes_gdf
    edges = graph.edges_gdf

    if isinstance(nodes, gpd.GeoDataFrame) and isinstance(edges, gpd.GeoDataFrame):
        nodes_crs = gdf_crs(nodes)
        edges_crs = gdf_crs(edges)
        if nodes_crs is not None and edges_crs is not None and nodes_crs != edges_crs:
            raise ValueError(f"nodes and edges crs mismatch: nodes.crs={nodes_crs}, edges.crs={edges_crs}")

    if edges.empty:
        return

    edge_nodes = pd.Index(pd.concat([edges["u"], edges["v"]], ignore_index=True).unique())
    missing_nodes = edge_nodes.difference(nodes.index)

    if not missing_nodes.empty:
        raise ValueError(f"Some edge endpoints are missing in nodes_gdf.index: {missing_nodes.tolist()[:10]}")


def validate_graph(graph) -> None:
    """Validate all node, edge, topology and CRS contracts of an ``UrbanGraph``.

    Raises:
        TypeError: If graph tables use unsupported types.
        ValueError: If graph table contracts are violated.
    """

    validate_nodes(graph)
    validate_edges(graph)
    sync_graph_crs(graph)
    validate_nodes_edges(graph)
