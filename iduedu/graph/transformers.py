from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

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


def consolidate_intersections(
    graph: UrbanGraph,
    tolerance: float = 10.0,
    *,
    max_span: float | None = None,
    return_mapping: bool = False,
) -> UrbanGraph | tuple[UrbanGraph, pd.Series]:
    """Merge groups of nearby nodes, such as the several nodes of one junction of dual carriageways, into one.

    Two nodes closer than ``tolerance`` belong to one cluster, and clusters chain (single linkage). A cluster
    is then split into the parts that edges inside the cluster connect, so nodes that are merely close (a
    bridge above a road) stay apart. Each part with several nodes becomes one node at the mean position of its
    members, keeping the id and attributes of its first member in ``nodes_gdf`` order; single nodes are
    unchanged. Edges inside a part are removed. The other edges keep their geometry with the end points moved
    to the new nodes; ``length_meter`` is the new geometry length and ``time_min`` is scaled with it.

    Chaining can join long runs of closely spaced nodes (dense footpath networks); a part whose bounding box
    is wider than ``max_span`` is therefore left unmerged.

    Meant for network statistics and centralities, where one junction drawn as several nodes inflates the
    counts. It changes routing: turns and one-way links inside a junction disappear.

    Args:
        graph: Graph with point nodes and line edges in a projected (metric) CRS.
        tolerance: Distance in CRS units under which two nodes are joined.
        max_span: Largest bounding-box diagonal of a part that is still merged, in CRS units. ``None`` means
            ``4 * tolerance``.
        return_mapping: Also return the mapping from old to new node ids.

    Returns:
        New ``UrbanGraph`` of the same kind (directed or not, multigraph or not; a simple graph keeps the
        lighter of the parallel edges that merging creates), and with ``return_mapping`` the ``Series`` from
        every old node id to its new one.

    Raises:
        TypeError: If ``graph`` is not an ``UrbanGraph``.
        ValueError: If ``tolerance`` is negative, the CRS is geographic or an edge is not a LineString.
    """
    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number")
    crs = graph.nodes_gdf.crs or graph.crs
    if crs is not None and CRS.from_user_input(crs).is_geographic:
        raise ValueError("consolidate_intersections needs a projected CRS; reproject the graph first")

    nodes, edges = graph.nodes_gdf, graph.edges_gdf
    n = len(nodes)
    xy = np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()]).reshape(-1, 2)
    positions = pd.Index(nodes.index)
    u, v = positions.get_indexer(edges["u"]), positions.get_indexer(edges["v"])

    pairs = cKDTree(xy).query_pairs(r=tolerance, output_type="ndarray") if n else np.empty((0, 2), dtype=int)
    near = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, cluster = csgraph.connected_components(near, directed=False)
    inside = (cluster[u] == cluster[v]) & (u != v)
    links = sparse.coo_matrix((np.ones(int(inside.sum())), (u[inside], v[inside])), shape=(n, n))
    _, part = csgraph.connected_components(links, directed=False)

    if max_span is None:
        max_span = 4 * tolerance
    if n:
        low = np.full((part.max() + 1, 2), np.inf)
        high = np.full((part.max() + 1, 2), -np.inf)
        np.minimum.at(low, part, xy)
        np.maximum.at(high, part, xy)
        too_wide = np.hypot(*(high - low).T) > max_span
        if too_wide.any():
            logger.debug(f"consolidate_intersections: {int(too_wide.sum())} groups wider than {max_span} left as is")
            loose = too_wide[part]
            part = np.where(loose, part.max() + 1 + np.arange(n), part)
            part = np.unique(part, return_inverse=True)[1]
    first = pd.Series(np.arange(n)).groupby(part).transform("min").to_numpy()  # first member of each part
    size = np.bincount(part)[part] if n else np.zeros(0, dtype=int)
    merged = size > 1
    new_xy = xy.copy()
    if merged.any():
        sums = np.zeros((part.max() + 1, 2))
        np.add.at(sums, part, xy)
        new_xy[merged] = sums[part[merged]] / size[merged, None]
    mapping = pd.Series(nodes.index.to_numpy()[first], index=nodes.index, name="node")

    keep_nodes = np.flatnonzero(first == np.arange(n))
    new_nodes = nodes.iloc[keep_nodes].copy()
    moved = merged[keep_nodes]
    if moved.any():
        geometry = new_nodes.geometry.to_numpy().copy()
        geometry[moved] = shapely.points(new_xy[keep_nodes[moved]])
        new_nodes = new_nodes.set_geometry(gpd.GeoSeries(geometry, index=new_nodes.index, crs=nodes.crs))

    drop = (part[u] == part[v]) & merged[u]
    new_edges = edges.loc[~drop].copy()
    eu, ev = u[~drop], v[~drop]
    if len(new_edges):
        geoms = new_edges.geometry.to_numpy()
        if not (shapely.get_type_id(geoms) == 1).all():
            raise ValueError("every edge geometry must be a LineString")
        coords, owner = shapely.get_coordinates(geoms, return_index=True)
        counts = np.bincount(owner, minlength=len(geoms))
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        ends = starts + counts - 1
        # An edge geometry may run from v to u; its start then belongs to v.
        start_is_u = np.hypot(*(coords[starts] - xy[eu]).T) <= np.hypot(*(coords[starts] - xy[ev]).T)
        start_node = np.where(start_is_u, eu, ev)
        end_node = np.where(start_is_u, ev, eu)
        coords[starts] = np.where(merged[start_node, None], new_xy[start_node], coords[starts])
        coords[ends] = np.where(merged[end_node, None], new_xy[end_node], coords[ends])
        new_geoms = shapely.linestrings(coords, indices=owner)
        old_length = shapely.length(geoms)
        new_edges = new_edges.set_geometry(gpd.GeoSeries(new_geoms, index=new_edges.index, crs=edges.crs))
        new_edges["u"] = mapping.to_numpy()[eu]
        new_edges["v"] = mapping.to_numpy()[ev]
        if "length_meter" in new_edges.columns:
            new_length = shapely.length(new_geoms)
            if "time_min" in new_edges.columns:
                ratio = np.divide(new_length, old_length, out=np.ones_like(new_length), where=old_length > 0)
                new_edges["time_min"] = new_edges["time_min"].to_numpy() * ratio
            new_edges["length_meter"] = new_length

    multigraph = graph.is_multigraph or bool(new_edges.duplicated(["u", "v"]).any())
    if multigraph:
        new_edges["k"] = new_edges.groupby(["u", "v"], sort=False).cumcount()
    result = UrbanGraph(
        nodes_gdf=new_nodes,
        edges_gdf=new_edges,
        is_multigraph=multigraph,
        is_directed=graph.is_directed,
        edge_direction_column=graph.edge_direction_column,
        adjacency_weight=graph.adjacency_weight,
        crs=graph.crs,
        graph_type=graph.type,
    )
    if multigraph and not graph.is_multigraph:
        result = simplify_multiedges(result, weight=graph.adjacency_weight)
        result.type = graph.type
    return (result, mapping) if return_mapping else result
