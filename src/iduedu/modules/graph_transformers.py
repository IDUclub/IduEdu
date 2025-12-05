import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from pyproj.aoi import AreaOfInterest

# pylint: disable=no-name-in-module
from pyproj.database import query_utm_crs_info
from shapely import LineString, MultiLineString, Point, Polygon, from_wkt, line_merge, node
from shapely.geometry.base import BaseGeometry


def clip_nx_graph(graph: nx.Graph, polygon: Polygon) -> nx.Graph:
    """
    Clip a NetworkX graph by a polygon and return the induced subgraph.

    Builds a GeoDataFrame of node points using `graph.graph["crs"]`, clips it by the given
    polygon, then returns the node-induced subgraph (keeping original node/edge attributes).

    Parameters:
        graph (nx.Graph): Graph with node coords stored as `x`, `y` and a CRS in `graph.graph["crs"]`.
        polygon (Polygon): Clipping polygon in the same CRS as the graph.

    Returns:
        (nx.Graph): Subgraph containing only nodes whose points fall inside the polygon.

    Notes:
        Edges are preserved only if both endpoints remain in the subgraph.
    """
    crs = graph.graph["crs"]
    points = gpd.GeoDataFrame(
        data=[{"id": p_id, "geometry": Point(data["x"], data["y"])} for p_id, data in graph.nodes(data=True)], crs=crs
    ).clip(polygon, True)
    clipped = graph.subgraph(points["id"].tolist())
    return clipped


def _fmt_top_sizes(sizes, top_k: int = 5) -> str:
    ss = sorted(sizes, reverse=True)
    if len(ss) <= top_k:
        return "[" + ", ".join(map(str, ss)) + "]"
    return "[" + ", ".join(map(str, ss[:top_k])) + ", …]"


def keep_largest_strongly_connected_component(graph: nx.DiGraph, *, top_k_wcc_sizes: int = 5) -> nx.DiGraph:
    """
    Keep only the largest strongly connected component of a directed graph.

    Logs the sizes of weakly connected components (WCC) for visibility, then removes all
    nodes outside the largest strongly connected component (SCC) and returns the pruned copy.

    Parameters:
        graph (nx.DiGraph): Directed graph to prune (a copy is made).
        top_k_wcc_sizes (int): How many largest WCC sizes to show in the warning.

    Returns:
        (nx.DiGraph): Graph restricted to the largest SCC.

    Notes:
        - Uses `nx.weakly_connected_components` for a quick disconnectedness summary.
        - Nodes from all SCCs except the largest are removed.
    """
    graph = graph.copy()

    weakly_connected_components = list(nx.weakly_connected_components(graph))
    if len(weakly_connected_components) > 1:
        sizes = [len(c) for c in weakly_connected_components]
        logger.debug(
            f"Graph contains {len(weakly_connected_components)} weakly connected components. "
            f"This means the graph has disconnected groups if edge directions are ignored. "
            f"Component sizes:: {_fmt_top_sizes(sizes, top_k=top_k_wcc_sizes)}"
        )

    all_scc = sorted(nx.strongly_connected_components(graph), key=len)
    nodes_to_del = set().union(*all_scc[:-1])

    if nodes_to_del:
        logger.warning(
            f"Removing {len(nodes_to_del)} nodes from {len(all_scc) - 1} smaller strongly connected components. "
            f"These are subgraphs where nodes are internally reachable but isolated from the rest. "
            f"Retaining only the largest strongly connected component ({len(all_scc[-1])} nodes)."
        )
        graph.remove_nodes_from(nodes_to_del)

    return graph


def estimate_crs_for_bounds(minx, miny, maxx, maxy) -> CRS:
    """
    Estimate a local UTM CRS for the given lon/lat bounds.

    Uses the bounds center to query a suitable WGS84 UTM CRS via `pyproj.query_utm_crs_info`
    and returns it as a `pyproj.CRS`.

    Parameters:
        minx (float): Western longitude.
        miny (float): Southern latitude.
        maxx (float): Eastern longitude.
        maxy (float): Northern latitude.

    Returns:
        (pyproj.CRS): UTM CRS suited for metric distance calculations near the bounds center.
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
    crs = CRS.from_epsg(utm_crs_list[0].code)
    logger.debug(f"Estimated CRS for territory {crs}")
    return crs


def _edges_to_gdf(graph: nx.Graph, crs) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.
    """
    graph_df = pd.DataFrame(list(graph.edges(data=True)), columns=["u", "v", "data"])
    edge_data_expanded = pd.json_normalize(graph_df["data"])
    graph_df = pd.concat([graph_df.drop(columns=["data"]), edge_data_expanded], axis=1)
    graph_df = gpd.GeoDataFrame(graph_df, geometry="geometry", crs=crs).set_index(["u", "v"])
    graph_df["geometry"] = graph_df["geometry"].fillna(LineString())
    return graph_df


def _nodes_to_gdf(graph: nx.Graph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as nodes.
    """

    ind, data = zip(*graph.nodes(data=True))
    node_geoms = (Point(d["x"], d["y"]) for d in data)
    gdf_nodes = gpd.GeoDataFrame(data, index=ind, crs=crs, geometry=list(node_geoms))

    return gdf_nodes


def _restore_edges_geom(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    edges_wout_geom = edges_gdf[edges_gdf["geometry"].is_empty].reset_index()
    edges_wout_geom["geometry"] = [
        LineString((s, e))
        for s, e in zip(
            nodes_gdf.loc[edges_wout_geom["u"], "geometry"], nodes_gdf.loc[edges_wout_geom["v"], "geometry"]
        )
    ]
    edges_wout_geom.set_index(["u", "v"], inplace=True)
    edges_gdf.update(edges_wout_geom)
    return edges_gdf


def graph_to_gdf(
    graph: nx.MultiDiGraph, edges: bool = True, nodes: bool = True, restore_edge_geom=False
) -> gpd.GeoDataFrame | None:
    """
    Convert a NetworkX graph to GeoDataFrames (edges and/or nodes).

    Reads CRS from `graph.graph["crs"]`. Depending on flags, returns only nodes, only edges,
    or a concatenation of both. Optionally reconstructs missing edge geometries from node points.

    Parameters:
        graph (nx.MultiDiGraph): Graph with node coords (`x`, `y`) and optional edge `geometry`.
        edges (bool): If True, include edges.
        nodes (bool): If True, include nodes.
        restore_edge_geom (bool): If True, fill empty edge geometries from node coordinates.

    Returns:
        (gpd.GeoDataFrame | None): Nodes and/or edges as GeoDataFrame(s).
            If both `edges` and `nodes` are False, returns None.

    Raises:
        ValueError: If `graph.graph["crs"]` is missing.

    Notes:
        - Edge GeoDataFrame uses MultiIndex `(u, v)`.
        - When both are requested, the result is a vertical concat of nodes then edges.
    """
    try:
        crs = graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have crs attribute and no crs was provided") from exc
    if not edges and not nodes:
        logger.debug("Neither edges or nodes were selected, graph_to_gdf returning None")
        return None
    if nodes and not edges:
        nodes_gdf = _nodes_to_gdf(graph, crs)
        return nodes_gdf
    if not nodes and edges:
        edges_gdf = _edges_to_gdf(graph, crs)
        if restore_edge_geom:
            nodes_gdf = _nodes_to_gdf(graph, crs)
            edges_gdf = _restore_edges_geom(nodes_gdf, edges_gdf)
        return edges_gdf

    nodes_gdf = _nodes_to_gdf(graph, crs)
    edges_gdf = _edges_to_gdf(graph, crs)
    if restore_edge_geom:
        edges_gdf = _restore_edges_geom(nodes_gdf, edges_gdf)
    full_gdf = pd.concat([nodes_gdf, edges_gdf])
    return full_gdf


def gdf_to_graph(
    gdf: gpd.GeoDataFrame, project_gdf_attr=True, reproject_to_utm_crs=True, speed=5, check_intersections=True
) -> nx.DiGraph:
    """
    Convert a GeoDataFrame of LineStrings into a directed graph (nx.DiGraph).

    Explodes multilines, optionally enforces topological intersections, merges collinear segments,
    transfers selected attributes back to merged lines via centroid-buffer spatial join,
    and constructs a directed graph whose edges correspond to line segments. Lengths are computed
    in meters in a local metric CRS; travel time uses a provided speed.

    Parameters:
        gdf (gpd.GeoDataFrame): Input with LineString geometries (other types are filtered out).
        project_gdf_attr (bool): If True, projects original attributes to merged lines via nearest overlay.
        reproject_to_utm_crs (bool): If True, lengths computed in UTM and optionally reprojected back.
        speed (float): Speed in km/h used to compute `time_min` for each edge.
        check_intersections (bool): If True, uses `shapely.node` before `line_merge` to enforce proper splits.

    Returns:
        (nx.DiGraph): Directed graph with:
            - node attributes: `x`, `y`;
            - edge attributes: `geometry`, `length_meter`, `time_min`, plus projected attributes;

            Graph attribute `graph["crs"]` is set to the (possibly reprojected) CRS.

    Raises:
        ValueError: If the input contains no valid LineStrings.

    Notes:
        - Attribute projection aggregates multi-matches via a uniqueness reducer (`unique_list`).
        - `speed` is internally converted to meters/minute.
    """

    def unique_list(agg_vals):
        agg_vals = list(set(agg_vals.dropna()))
        if len(agg_vals) == 1:
            return agg_vals[0]
        return agg_vals

    original_crs = gdf.crs
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    gdf = gdf.explode(ignore_index=True)
    gdf = gdf[gdf.geom_type == "LineString"]

    if len(gdf) == 0:
        raise ValueError("Provided GeoDataFrame contains no valid LineStrings")

    if check_intersections:
        lines = line_merge(node(MultiLineString(gdf.geometry.to_list())))
    else:
        lines = line_merge(MultiLineString(gdf.geometry.to_list()))

    if isinstance(lines, LineString):
        lines = MultiLineString([lines])

    lines = gpd.GeoDataFrame(geometry=list(lines.geoms), crs=gdf.crs)

    if len(gdf.columns) > 1 and project_gdf_attr:
        lines_centroids = lines.copy()
        lines_centroids.geometry = lines_centroids.apply(
            lambda row: row.geometry.line_interpolate_point(row.geometry.length / 2), axis=1
        ).buffer(0.05, resolution=2)
        lines_with_attrs = gpd.sjoin(lines_centroids, gdf, how="left", predicate="intersects")
        aggregated_attrs = (
            lines_with_attrs.drop(columns=["geometry", "index_right"])  # убрать геометрию буфера
            .groupby(lines_with_attrs.index)
            .agg(unique_list)
        )
        lines = pd.concat([lines, aggregated_attrs], axis=1)

    lines["length_meter"] = np.round(lines.length, 2)
    if not reproject_to_utm_crs:
        lines = lines.to_crs(original_crs)

    coords = lines.geometry.get_coordinates()
    coords_grouped_by_index = coords.reset_index(names="old_index").groupby("old_index")
    start_coords = coords_grouped_by_index.head(1).apply(lambda a: (a.x, a.y), axis=1).rename("start")
    end_coords = coords_grouped_by_index.tail(1).apply(lambda a: (a.x, a.y), axis=1).rename("end")
    coords = pd.concat([start_coords.reset_index(), end_coords.reset_index()], axis=1)[["start", "end"]]
    lines = pd.concat([lines, coords], axis=1)
    unique_coords = pd.concat([coords["start"], coords["end"]], ignore_index=True).unique()
    coord_to_index = {coord: idx for idx, coord in enumerate(unique_coords)}

    lines["u"] = lines["start"].map(coord_to_index)
    lines["v"] = lines["end"].map(coord_to_index)

    speed = speed * 1000 / 60
    lines["time_min"] = np.round(lines["length_meter"] / speed, 2)

    graph = nx.Graph()
    for coords, node_id in coord_to_index.items():
        x, y = coords
        graph.add_node(node_id, x=float(x), y=float(y))

    columns_to_attr = lines.columns.difference(["start", "end", "u", "v"])
    for _, row in lines.iterrows():
        edge_attrs = {}
        for col in columns_to_attr:
            edge_attrs[col] = row[col]
        graph.add_edge(row.u, row.v, **edge_attrs)

    graph.graph["crs"] = lines.crs
    graph.graph["speed_m_min"] = speed
    return nx.DiGraph(graph)


def write_gml(graph: nx.Graph, gml_path: str) -> nx.Graph:
    """
    Write a NetworkX graph to GML, coercing node coordinates to plain floats.

    Ensures node attributes `x` and `y` are Python `float`, then writes the graph using
    `stringizer=str` so any non-primitive attribute values are serialized as strings.

    Parameters:
        graph (nx.Graph): Input graph. Not mutated — a sanitized copy is written.
        gml_path (str): Output GML file path.

    Returns:
        (nx.Graph): The sanitized copy of the graph that was written to disk.
    """
    graph = graph.copy()

    # Nodes: x/y to float; node geometry to WKT if present
    for n, data in graph.nodes(data=True):
        if "x" in data:
            try:
                data["x"] = float(data["x"])
            except Exception:
                raise ValueError(f"Node {n} has non-numeric x={data['x']!r}")
        if "y" in data:
            try:
                data["y"] = float(data["y"])
            except Exception:
                raise ValueError(f"Node {n} has non-numeric y={data['y']!r}")

    nx.write_gml(graph, gml_path, stringizer=lambda v: str(v))
    return graph


def read_gml(gml_path: str, **nx_kwargs) -> nx.Graph:
    """
    Read a GML file into a NetworkX graph and cast edge `geometry` from WKT strings to shapely.

    Loads the graph via `networkx.read_gml` and, when an edge attribute `geometry` is a string,
    attempts to parse it with `shapely.wkt.from_wkt`. Non-parsable strings are left unchanged.

    Parameters:
        gml_path (str): Path to the GML file.
        **nx_kwargs: Additional keyword arguments forwarded to `networkx.read_gml`.

    Returns:
        (nx.Graph): The graph with edge `geometry` parsed to shapely objects where possible.
    """
    graph = nx.read_gml(gml_path, **nx_kwargs)

    if graph.is_multigraph():
        for u, v, k, data in graph.edges(keys=True, data=True):
            if "geometry" in data and isinstance(data["geometry"], str):
                try:
                    data["geometry"] = from_wkt(data["geometry"])
                except Exception:
                    pass
    else:
        for u, v, data in graph.edges(data=True):
            if "geometry" in data and isinstance(data["geometry"], str):
                try:
                    data["geometry"] = from_wkt(data["geometry"])
                except Exception:
                    pass

    return graph


def reproject_graph(graph: nx.Graph, target_crs) -> nx.Graph:
    """
    Reproject node coordinates (`x`, `y`) and edge geometries to a new CRS (in place).

    Builds GeoDataFrames for nodes and for edges that have shapely `geometry`, applies
    `GeoDataFrame.to_crs(target_crs)`, writes transformed coordinates/geometries back to the graph,
    and updates `graph.graph["crs"]` to the resulting target CRS.

    Parameters:
        graph (nx.Graph): Graph with current CRS in `graph["crs"]`; nodes carry `x`, `y`
            in that CRS, edges may carry shapely `geometry` in the same CRS.
        target_crs: Target CRS accepted by GeoPandas (EPSG int, string like `"EPSG:3857"`,
            or a `pyproj.CRS`).

    Returns:
        (nx.Graph): The same graph instance (mutated in place) with updated coordinates/geometries and CRS.

    Raises:
        ValueError: If `graph.graph["crs"]` is missing.

    Notes:
        - Only nodes with both `x` and `y` are updated.
        - Edges without shapely geometry are left unchanged.
        - If an edge `geometry` is stored as a WKT string, it is not reprojected; parse it first.
    """
    try:
        current_crs = graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have 'crs' attribute") from exc

    nodes_items = [(n, d) for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if nodes_items:
        node_ids = [n for n, _ in nodes_items]
        node_points = [Point(float(d["x"]), float(d["y"])) for _, d in nodes_items]
        nodes_gdf = gpd.GeoDataFrame(index=node_ids, geometry=node_points, crs=current_crs).to_crs(target_crs)
        target_crs = nodes_gdf.crs
        for nid, geom in nodes_gdf.geometry.items():
            graph.nodes[nid]["x"] = float(geom.x)
            graph.nodes[nid]["y"] = float(geom.y)

    if graph.is_multigraph():
        edge_records = [
            (u, v, k, data)
            for u, v, k, data in graph.edges(keys=True, data=True)
            if isinstance(data.get("geometry"), BaseGeometry)
        ]
        if edge_records:
            idx = [(u, v, k) for u, v, k, _ in edge_records]
            geoms = [data["geometry"] for _, _, _, data in edge_records]
            edges_gdf = gpd.GeoDataFrame(
                index=pd.MultiIndex.from_tuples(idx, names=["u", "v", "k"]), geometry=geoms, crs=current_crs
            ).to_crs(target_crs)
            target_crs = edges_gdf.crs
            for (u, v, k), geom in edges_gdf.geometry.items():
                graph.edges[u, v, k]["geometry"] = geom
    else:
        edge_records = [
            (u, v, data) for u, v, data in graph.edges(data=True) if isinstance(data.get("geometry"), BaseGeometry)
        ]
        if edge_records:
            idx = [(u, v) for u, v, _ in edge_records]
            geoms = [data["geometry"] for _, _, data in edge_records]
            edges_gdf = gpd.GeoDataFrame(
                index=pd.MultiIndex.from_tuples(idx, names=["u", "v"]), geometry=geoms, crs=current_crs
            ).to_crs(target_crs)
            target_crs = edges_gdf.crs
            for (u, v), geom in edges_gdf.geometry.items():
                graph.edges[u, v]["geometry"] = geom
    graph.graph["crs"] = target_crs
    return graph
