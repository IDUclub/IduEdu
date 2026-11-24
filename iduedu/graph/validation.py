from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely


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
    validate_node_table(graph.nodes_gdf)


def validate_node_table(nodes):
    """Validate an UrbanGraph node table."""
    if not isinstance(nodes, (gpd.GeoDataFrame, pd.DataFrame)):
        raise TypeError(f"nodes_gdf must be DataFrame or GeoDataFrame, got {type(nodes).__name__}")
    if nodes.index.has_duplicates:
        raise ValueError("nodes_gdf.index must be unique")
    if isinstance(nodes, gpd.GeoDataFrame):
        _validate_geometry(nodes, "nodes_gdf", shapely.GeometryType.POINT, "Point")


def _validate_geometry(frame, name, expected_id, expected_type):
    type_ids = shapely.get_type_id(frame.geometry.array)
    if (type_ids != expected_id).any():
        if (type_ids == shapely.GeometryType.MISSING).any():
            raise ValueError(f"{name}.geometry contains NaN")
        raise ValueError(f"All {name} geometries must be {expected_type}")


def validate_edges(graph) -> None:
    """Validate the edge table contract of an ``UrbanGraph``.

    Raises:
        TypeError: If ``edges_gdf`` is not a DataFrame-like object.
        ValueError: If required columns, topology keys or geometries are invalid.
    """
    validate_edge_table(graph.edges_gdf, graph.is_multigraph, graph.edge_direction_column)


def validate_edge_table(edges, is_multigraph, direction_column):
    """Validate an UrbanGraph edge table."""
    if not isinstance(edges, (gpd.GeoDataFrame, pd.DataFrame)):
        raise TypeError(f"edges_gdf must be DataFrame or GeoDataFrame, got {type(edges).__name__}")
    if edges.empty:
        return
    required = {"u", "v", "geometry", "length_meter", "time_min"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edges_gdf missing required columns: {sorted(missing)}")
    if edges["u"].isna().any() or edges["v"].isna().any():
        raise ValueError("edges_gdf columns ['u', 'v'] must not contain NaN")
    if direction_column is not None:
        validate_edge_direction(edges, direction_column)
    if is_multigraph:
        if "k" not in edges.columns:
            raise ValueError("edges_gdf must contain 'k' for multigraph")
        if edges.duplicated(subset=["u", "v", "k"]).any():
            raise ValueError("edges_gdf must have unique ['u','v','k']")
    else:
        if edges.duplicated(subset=["u", "v"]).any():
            raise ValueError("edges_gdf must have unique ['u','v'] for non-multigraph")
    if isinstance(edges, gpd.GeoDataFrame):
        _validate_geometry(edges, "edges_gdf", shapely.GeometryType.LINESTRING, "LineString")


def validate_edge_direction(edges, direction_column):
    """Validate the graph's direction column in the given rows."""
    if direction_column not in edges.columns:
        raise ValueError(f"edges_gdf missing edge_direction_column {direction_column!r}")
    direction = edges[direction_column]
    if direction.isna().any():
        raise ValueError(f"edges_gdf[{direction_column!r}] contains NaN")
    if not set(direction.unique()) <= {False, True}:
        raise ValueError(f"edges_gdf[{direction_column!r}] must contain only boolean values")


def validate_nodes_edges(graph) -> None:
    """Validate consistency between node ids, edge endpoints and CRS values."""
    nodes = graph.nodes_gdf
    edges = graph.edges_gdf

    if isinstance(nodes, gpd.GeoDataFrame) and isinstance(edges, gpd.GeoDataFrame):
        nodes_crs = gdf_crs(nodes)
        edges_crs = gdf_crs(edges)
        if nodes_crs is not None and edges_crs is not None and nodes_crs != edges_crs:
            raise ValueError(f"nodes and edges crs mismatch: nodes.crs={nodes_crs}, edges.crs={edges_crs}")

    validate_edge_endpoints(nodes.index, edges)


def validate_edge_endpoints(node_ids, edges):
    """Require both endpoints of each edge to be present among the node ids."""
    if edges.empty:
        return

    if (
        pd.api.types.is_integer_dtype(node_ids.dtype)
        and node_ids.dtype == edges["u"].dtype == edges["v"].dtype
        and (node_ids.get_indexer(edges["u"]) >= 0).all()
        and (node_ids.get_indexer(edges["v"]) >= 0).all()
    ):
        return
    edge_nodes = pd.Index(pd.concat([edges["u"], edges["v"]], ignore_index=True).unique())
    missing_nodes = edge_nodes.difference(node_ids)

    if not missing_nodes.empty:
        raise ValueError(f"Some edge endpoints are missing in nodes_gdf.index: {missing_nodes.tolist()[:10]}")


def matching_edge_positions(edges, keys, key_cols):
    """Return candidate edge positions and their matching key positions (or -1)."""
    key_index = pd.MultiIndex.from_frame(keys[key_cols])
    if edges["u"].dtype == keys["u"].dtype and pd.api.types.is_integer_dtype(edges["u"].dtype):
        # Factorize only edges whose start node is among the changed keys.
        positions = np.flatnonzero(edges["u"].isin(keys["u"]))
        edge_index = pd.MultiIndex.from_arrays([edges[key].iloc[positions] for key in key_cols])
    else:
        positions = np.arange(len(edges))
        edge_index = pd.MultiIndex.from_frame(edges[key_cols])
    return positions, key_index.get_indexer(edge_index)


def validate_applied_changes(source, graph, changes, key_cols):
    """Check the resulting rows affected by changes to a valid graph."""
    source_nodes, source_edges = source.nodes_gdf, source.edges_gdf
    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    updates = changes.edges_to_update
    active_geometry = getattr(edges, "active_geometry_name", None)
    updated_geometry = active_geometry is not None and updates is not None and active_geometry in updates.columns
    key_dtypes_changed = not source_edges.dtypes.reindex(key_cols).equals(edges.dtypes.reindex(key_cols))
    added_to_empty_graph = not edges.empty and source_edges.empty

    # A change to the schema or key dtypes can affect existing rows during concatenation.
    if (
        nodes.index.dtype != source_nodes.index.dtype
        or key_dtypes_changed
        or added_to_empty_graph
        or getattr(source_edges, "active_geometry_name", None) != active_geometry
        or updated_geometry
    ):
        graph.validate()
        return

    if changes.nodes_gdf is not None and not changes.nodes_gdf.empty:
        validate_node_table(nodes.iloc[-len(changes.nodes_gdf) :])
    direction = graph.edge_direction_column
    if changes.edges_gdf is not None and not changes.edges_gdf.empty:
        first_added = len(edges) - len(changes.edges_gdf)
        added = edges.iloc[first_added:]
        validate_edge_table(added, graph.is_multigraph, direction)
        retained = edges.iloc[:first_added]
        if (matching_edge_positions(retained, added, key_cols)[1] >= 0).any():
            if graph.is_multigraph:
                raise ValueError("edges_gdf must have unique ['u','v','k']")
            raise ValueError("edges_gdf must have unique ['u','v'] for non-multigraph")
        validate_edge_endpoints(nodes.index, added)

    if direction is not None and updates is not None and not updates.empty and direction in updates.columns:
        positions, matched = matching_edge_positions(edges, updates, key_cols)
        validate_edge_direction(edges.iloc[positions[matched >= 0]], direction)


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
