import importlib
import importlib.util
import warnings
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, Point

from iduedu.graph.urban_graph import UrbanGraph


def _require_networkx():
    if importlib.util.find_spec("networkx") is None:
        raise ImportError(
            "NetworkX is required for conversions to/from NetworkX graphs. "
            "Install optional dependency `networkx` to use this function."
        )

    return importlib.import_module("networkx")


def nx_graph2urban_graph(
    nx_graph: Any,
    restore_edge_geom: bool = True,
    *,
    check_oneway: bool = True,
    oneway_column: str = "oneway",
) -> UrbanGraph:
    """Convert a NetworkX graph to an ``UrbanGraph``.

    This helper is intended for compatibility with libraries that still produce
    NetworkX graphs. Nodes must contain ``x`` and ``y`` coordinates, the graph must
    expose a ``crs`` attribute, and edges are expected to carry ``geometry``,
    ``length_meter`` and ``time_min`` attributes.

    Args:
        nx_graph: NetworkX graph, directed graph, multigraph or multidigraph.
            Directed graphs are accepted for compatibility, but a warning is emitted
            because stable behavior for directed NetworkX inputs is not guaranteed.
        restore_edge_geom: If ``True``, empty edge geometries are restored as
            straight lines between endpoint nodes. If ``False``, edges with empty
            geometries are removed.
        check_oneway: If ``True`` and ``oneway_column`` exists on edges, that column
            is used as the UrbanGraph edge direction column.
        oneway_column: Boolean edge attribute that marks one-way movement.

    Returns:
        Converted ``UrbanGraph`` instance.

    Raises:
        TypeError: If ``nx_graph`` is not a NetworkX graph.
        ValueError: If required CRS, node or edge data is missing or invalid.
    """

    nx = _require_networkx()

    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            f"graph must be nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph, " f"got {type(nx_graph).__name__}"
        )
    if nx_graph.is_directed():
        warnings.warn(
            "Directed NetworkX graphs are accepted for compatibility, but stable behavior is not guaranteed. "
            "Prefer Graph/MultiGraph with edge geometry oriented from u to v and one-way movement stored in a "
            f"boolean edge column such as {oneway_column!r}.",
            UserWarning,
            stacklevel=2,
        )

    try:
        crs = nx_graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have crs attribute and no crs was provided") from exc
    if nx_graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if nx_graph.number_of_edges() == 0:
        raise ValueError("graph has no edges")

    try:
        node_index, node_data = zip(*nx_graph.nodes(data=True))
        node_geoms = [Point(data["x"], data["y"]) for data in node_data]
    except KeyError as exc:
        raise ValueError(f"NetworkX nodes must have required {exc.args[0]!r} attribute") from exc

    nodes_gdf = gpd.GeoDataFrame(index=node_index, crs=crs, geometry=node_geoms)

    if nx_graph.is_multigraph():
        edges_df = pd.DataFrame(list(nx_graph.edges(data=True, keys=True)), columns=["u", "v", "k", "data"])
    else:
        edges_df = pd.DataFrame(list(nx_graph.edges(data=True)), columns=["u", "v", "data"])

    edge_data_expanded = pd.json_normalize(edges_df["data"])
    edges_df = pd.concat([edges_df.drop(columns=["data"]), edge_data_expanded], axis=1)
    if "geometry" not in edges_df.columns:
        edges_df["geometry"] = LineString()
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs=crs)
    edges_gdf["geometry"] = edges_gdf["geometry"].fillna(LineString())
    edge_direction_column = oneway_column if check_oneway and oneway_column in edges_gdf.columns else None

    empty_geometry = edges_gdf.geometry.is_empty
    if restore_edge_geom:
        edges_gdf.loc[empty_geometry, "geometry"] = [
            LineString((nodes_gdf.loc[u, "geometry"], nodes_gdf.loc[v, "geometry"]))
            for u, v in edges_gdf.loc[empty_geometry, ["u", "v"]].itertuples(index=False, name=None)
        ]
    else:
        edges_gdf = edges_gdf.loc[~empty_geometry].copy()
        if edges_gdf.empty:
            raise ValueError("graph has no edges with non-empty geometry")

    non_linestring = edges_gdf.geometry.geom_type != "LineString"
    if non_linestring.any():
        bad_types = edges_gdf.loc[non_linestring, "geometry"].geom_type.head(10).tolist()
        raise ValueError(f"All edge geometries must be LineString, got {bad_types}")

    node_xy = pd.DataFrame(
        {"node_x": nodes_gdf.geometry.x, "node_y": nodes_gdf.geometry.y},
        index=nodes_gdf.index,
    )
    edge_node_xy = edges_gdf[["u", "v"]].join(node_xy.add_prefix("u_"), on="u").join(node_xy.add_prefix("v_"), on="v")
    missing_node_xy = edge_node_xy[["u_node_x", "u_node_y", "v_node_x", "v_node_y"]].isna().any(axis=1)
    if missing_node_xy.any():
        missing_preview = edges_gdf.loc[missing_node_xy, ["u", "v"]].head(10).to_dict("records")
        raise ValueError(f"Some edge endpoints are missing node coordinates: {missing_preview}")

    coords = edges_gdf.geometry.get_coordinates().to_numpy()
    counts = edges_gdf.geometry.count_coordinates().to_numpy()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]

    u_xy = edge_node_xy[["u_node_x", "u_node_y"]].to_numpy(dtype=float)
    v_xy = edge_node_xy[["v_node_x", "v_node_y"]].to_numpy(dtype=float)
    endpoint_tolerance = np.maximum(1e-6, 1e-9 * np.maximum(edges_gdf.geometry.length.to_numpy(), 1.0))

    start_matches_u = np.isclose(starts, u_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    swap_uv = ~start_matches_u
    if swap_uv.any():
        edges_gdf.loc[swap_uv, ["u", "v"]] = edges_gdf.loc[swap_uv, ["v", "u"]].to_numpy()
        u_xy[swap_uv], v_xy[swap_uv] = v_xy[swap_uv].copy(), u_xy[swap_uv].copy()

    start_matches_u = np.isclose(starts, u_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    end_matches_v = np.isclose(ends, v_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    bad_endpoint_match = ~(start_matches_u & end_matches_v)
    if bad_endpoint_match.any():
        bad_idx = np.flatnonzero(bad_endpoint_match)[:10]
        start_distance = np.hypot(*(starts - u_xy).T)
        end_distance = np.hypot(*(ends - v_xy).T)
        bad_edges = edges_gdf.iloc[bad_idx][["u", "v"]].assign(
            start_to_u_distance=start_distance[bad_idx],
            end_to_v_distance=end_distance[bad_idx],
        )
        raise ValueError(
            "Some edge geometry endpoints do not match node coordinates: " f"{bad_edges.to_dict('records')}"
        )

    return UrbanGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        is_multigraph=nx_graph.is_multigraph(),
        is_directed=nx_graph.is_directed() or edge_direction_column is not None,
        edge_direction_column=edge_direction_column,
    )


def urban_graph2nx_graph(urban_graph: UrbanGraph) -> Any:
    """Convert an ``UrbanGraph`` to a NetworkX graph.

    NetworkX is not required for normal IduEdu workflows, but this conversion is
    useful at integration boundaries with tools that consume NetworkX graph objects.

    Args:
        urban_graph: Tabular urban graph to convert.

    Returns:
        ``networkx.Graph``, ``networkx.DiGraph``, ``networkx.MultiGraph`` or
        ``networkx.MultiDiGraph`` with node and edge attributes copied from the
        UrbanGraph tables.
    """

    nx = _require_networkx()

    nodes_gdf = urban_graph.nodes_gdf.copy()
    edges_gdf = urban_graph.edges_gdf.copy()

    graph_attrs = {
        "crs": edges_gdf.crs if isinstance(edges_gdf, gpd.GeoDataFrame) else None,
        "edge_direction_column": urban_graph.edge_direction_column,
    }

    is_multi = urban_graph.is_multigraph
    directed = urban_graph.is_directed and urban_graph.edge_direction_column is None

    nx_graph = (
        nx.MultiDiGraph()
        if directed and is_multi
        else (nx.MultiGraph() if is_multi else nx.DiGraph() if directed else nx.Graph())
    )
    nx_graph.graph.update(graph_attrs)

    nodes_gdf["x"] = nodes_gdf.geometry.x
    nodes_gdf["y"] = nodes_gdf.geometry.y
    df_nodes = nodes_gdf.drop(columns=nodes_gdf.active_geometry_name)

    if is_multi:
        edges_gdf = edges_gdf.set_index(["u", "v", "k"], drop=True)
        attr_names = edges_gdf.columns.to_list()

        for (u, v, k), attr_vals in zip(edges_gdf.index, edges_gdf.to_numpy(), strict=True):
            data_all = zip(attr_names, attr_vals, strict=True)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notna(val)}
            nx_graph.add_edge(u, v, key=k, **data)

    else:
        edges_gdf = edges_gdf.set_index(["u", "v"], drop=True)
        attr_names = edges_gdf.columns.to_list()

        for (u, v), attr_vals in zip(edges_gdf.index, edges_gdf.to_numpy(), strict=True):
            data_all = zip(attr_names, attr_vals, strict=True)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notna(val)}
            nx_graph.add_edge(u, v, **data)

    nx_graph.add_nodes_from(set(df_nodes.index) - set(nx_graph.nodes))

    for col in df_nodes.columns:
        nx.set_node_attributes(nx_graph, values=df_nodes[col].dropna().to_dict(), name=col)

    return nx_graph
