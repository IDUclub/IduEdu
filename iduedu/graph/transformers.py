from typing import Literal

import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from iduedu.config import config
from iduedu.graph.components import largest_component
from iduedu.graph.editors import subgraph_by_nodes
from iduedu.graph.urban_graph import UrbanGraph

logger = config.logger


def estimate_crs_for_bounds(minx, miny, maxx, maxy) -> CRS:
    """
    Estimate a local UTM CRS for the given lon/lat bounds.
    """
    x_center = np.mean([minx, maxx])
    y_center = np.mean([miny, maxy])
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=x_center,
            south_lat_degree=y_center,
            east_lon_degree=x_center,
            north_lat_degree=y_center,
        ),
    )
    return CRS.from_epsg(utm_crs_list[0].code)


def keep_largest_connected_component(
    graph: UrbanGraph,
    *,
    mode: Literal["auto", "connected", "weak", "strong"] = "auto",
) -> UrbanGraph:
    """
    Keep only the largest component of an ``UrbanGraph``.

    With ``mode="auto"``, directed graphs use the largest strongly connected
    component and undirected graphs use the largest connected component.

    See also:
        https://iduclub.github.io/IduEdu/examples/connectivity.html
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")

    component = largest_component(graph, mode=mode)
    if len(component) == len(graph.nodes_gdf):
        return graph.copy()

    component_name = {
        "auto": "strongly connected" if graph.is_directed else "connected",
        "connected": "connected",
        "weak": "weakly connected",
        "strong": "strongly connected",
    }[mode]
    removed = len(graph.nodes_gdf) - len(component)
    logger.warning(
        f"Removing {removed} nodes outside the largest {component_name} component. "
        f"Retaining {len(component)} of {len(graph.nodes_gdf)} nodes."
    )
    return subgraph_by_nodes(graph, component)


def to_directed(
    graph: UrbanGraph,
    *,
    edge_direction_column: str = "oneway",
    default_direction_value: bool = False,
) -> UrbanGraph:
    """
    Return a directed copy of ``UrbanGraph`` using a boolean edge direction column.

    ``default_direction_value=False`` makes existing edges traversable in both
    directions while still storing the graph as directed for adjacency building.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")

    edges = graph.edges_gdf.copy()
    if not edges.empty:
        if edge_direction_column not in edges.columns:
            edges[edge_direction_column] = bool(default_direction_value)
        else:
            missing_direction = edges[edge_direction_column].isna()
            if missing_direction.any():
                edges.loc[missing_direction, edge_direction_column] = bool(default_direction_value)
            edges[edge_direction_column] = edges[edge_direction_column].astype(bool)

    return UrbanGraph(
        nodes_gdf=graph.nodes_gdf.copy(),
        edges_gdf=edges,
        is_multigraph=graph.is_multigraph,
        is_directed=True,
        edge_direction_column=edge_direction_column,
        adjacency_weight=graph.adjacency_weight,
        crs=graph.crs,
        graph_type=graph.type,
    )


def to_undirected(graph: UrbanGraph) -> UrbanGraph:
    """
    Return an undirected copy of ``UrbanGraph``.

    Direction columns are kept as regular edge attributes but are not used for
    adjacency construction.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")

    return UrbanGraph(
        nodes_gdf=graph.nodes_gdf.copy(),
        edges_gdf=graph.edges_gdf.copy(),
        is_multigraph=graph.is_multigraph,
        is_directed=False,
        adjacency_weight=graph.adjacency_weight,
        crs=graph.crs,
        graph_type=graph.type,
    )


def simplify_multiedges(
    graph: UrbanGraph, *, weight: str = "time_min", rule: Literal["min", "max"] = "min"
) -> UrbanGraph:
    """Collapse a multigraph ``UrbanGraph`` into a simple graph.

    For each node pair, one edge is selected according to ``weight`` and ``rule``.
    ``rule="min"`` keeps the edge with the smallest weight, while ``rule="max"``
    keeps the edge with the largest weight.

    Args:
        graph: Source urban graph.
        weight: Edge column used to choose the representative edge.
        rule: Selection rule, either ``"min"`` or ``"max"``.

    Returns:
        New ``UrbanGraph`` with ``is_multigraph=False``.

    Raises:
        TypeError: If ``graph`` is not an ``UrbanGraph``.
        KeyError: If ``weight`` is absent from ``edges_gdf``.
        ValueError: If weights contain ``NaN`` or ``rule`` is unsupported.

    See also:
        https://iduclub.github.io/IduEdu/examples/graph_operations.html
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")
    if not isinstance(weight, str):
        raise TypeError(f"weight must be str, got {type(weight).__name__}")
    if rule not in {"min", "max"}:
        raise ValueError(f"rule must be 'min' or 'max', got {rule!r}")

    if not graph.is_multigraph:
        return UrbanGraph(
            nodes_gdf=graph.nodes_gdf.copy(),
            edges_gdf=graph.edges_gdf.copy(),
            is_multigraph=False,
            is_directed=graph.is_directed,
            edge_direction_column=graph.edge_direction_column,
        )

    edges = graph.edges_gdf.copy()

    if weight not in edges.columns:
        raise KeyError(f"edges_gdf has no weight column {weight!r}")
    if edges[weight].isna().any():
        raise ValueError(f"edges_gdf[{weight!r}] contains NaN")

    if graph.is_directed:
        group_cols = ["u", "v"]
        work = edges
    else:
        work = edges.copy()
        uv_min = pd.concat([work["u"], work["v"]], axis=1).min(axis=1)
        uv_max = pd.concat([work["u"], work["v"]], axis=1).max(axis=1)
        work["_u_simplify"] = uv_min
        work["_v_simplify"] = uv_max
        group_cols = ["_u_simplify", "_v_simplify"]

    ascending = rule == "min"

    selected_idx = (
        work.sort_values(weight, ascending=ascending, kind="mergesort").groupby(group_cols, sort=False).head(1).index
    )

    simple_edges = edges.loc[selected_idx].copy()

    if "k" in simple_edges.columns:
        simple_edges = simple_edges.drop(columns="k")

    return UrbanGraph(
        nodes_gdf=graph.nodes_gdf.copy(),
        edges_gdf=simple_edges,
        is_multigraph=False,
        is_directed=graph.is_directed,
        edge_direction_column=graph.edge_direction_column,
    )
