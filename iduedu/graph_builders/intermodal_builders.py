import concurrent.futures
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, Polygon

from iduedu import config
from iduedu.graph.editors import apply_urban_graph_changes, project_objects2urban_graph
from iduedu.graph.transformers import keep_largest_connected_component
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders.drive_walk_builders import get_walk_graph
from iduedu.graph_builders.public_transport_builders import get_public_transport_graph
from iduedu.overpass.downloaders import get_4326_boundary

logger = config.logger

PLATFORM_NODE_TYPES = {"platform", "subway_platform", "subway_entry_exit", "subway_entry", "subway_exit"}
DEFAULT_WALK_SPEED_M_PER_MIN = 5 * 1000 / 60


def join_pt_walk_graph(
    public_transport_g: UrbanGraph,
    walk_g: UrbanGraph,
    max_dist: float = 20,
    keep_largest_subgraph: bool = True,
    *,
    add_link_edge: bool = True,
    walk_speed_m_per_min: float | None = None,
) -> UrbanGraph:
    """
    Join public-transport and pedestrian ``UrbanGraph`` objects into an intermodal graph.

    Public-transport platform-like nodes are projected to the walk graph via
    :func:`project_objects2urban_graph`. Platforms farther than ``max_dist``
    from the walking network are left as regular PT nodes.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
        https://iduclub.github.io/IduEdu/examples/objects_and_nearest_nodes.html
    """

    if public_transport_g.crs != walk_g.crs:
        raise ValueError(f"CRS mismatch: public_transport_g.crs={public_transport_g.crs}, walk_g.crs={walk_g.crs}")

    if public_transport_g.nodes_gdf.empty:
        logger.warning("Public transport graph is empty. Returning walk graph unchanged.")
        return walk_g
    if walk_g.nodes_gdf.empty:
        logger.warning("Walk graph is empty. Returning public transport graph unchanged.")
        return public_transport_g

    logger.info("Composing intermodal graph...")
    if walk_speed_m_per_min is not None:
        walk_speed = walk_speed_m_per_min
    else:
        walk_speed = DEFAULT_WALK_SPEED_M_PER_MIN

    walk_directed = walk_g.to_directed(edge_direction_column="oneway", default_direction_value=False)

    pt_nodes = public_transport_g.nodes_gdf.copy()
    platform_mask = (
        pt_nodes["type"].isin(PLATFORM_NODE_TYPES)
        if "type" in pt_nodes.columns
        else pd.Series(False, index=pt_nodes.index)
    )
    pt_platforms = pt_nodes.loc[platform_mask].copy()

    if pt_platforms.empty:
        logger.warning("Public transport graph has no platform-like nodes. Joining graphs without projection.")
        projected_walk = walk_directed
        object2node_map = pd.Series(dtype=int)
    else:
        changes, object2node_map = project_objects2urban_graph(
            walk_directed,
            pt_platforms,
            walk_speed,
            max_dist=max_dist,
            add_link_edge=add_link_edge,
        )
        projected_walk = apply_urban_graph_changes(walk_directed, changes)
        walk_edges = projected_walk.edges_gdf.copy()
        if "type" not in walk_edges.columns:
            walk_edges["type"] = "walk"
        else:
            walk_edges["type"] = walk_edges["type"].fillna("walk")
        walk_edges["oneway"] = walk_edges["oneway"].map(lambda value: False if pd.isna(value) else bool(value))
        projected_walk = UrbanGraph(
            nodes_gdf=projected_walk.nodes_gdf.copy(),
            edges_gdf=walk_edges,
            is_multigraph=projected_walk.is_multigraph,
            is_directed=projected_walk.is_directed,
            edge_direction_column=projected_walk.edge_direction_column,
            adjacency_weight=projected_walk.adjacency_weight,
            crs=projected_walk.crs,
            graph_type=projected_walk.type,
        )

        if not object2node_map.empty:
            walk_nodes = projected_walk.nodes_gdf.copy()
            mapped_platforms = object2node_map.dropna().astype(int).rename("target_node").to_frame()
            node_attrs = pt_nodes.drop(columns=pt_nodes.geometry.name).join(mapped_platforms, how="inner")
            attr_columns = [column for column in node_attrs.columns if column != "target_node"]

            if attr_columns:
                for column in attr_columns:
                    is_sequence = node_attrs[column].map(lambda value: isinstance(value, (list, tuple, set)))
                    node_attrs.loc[is_sequence, column] = node_attrs.loc[is_sequence, column].map(list)

                def collapse_values(values):
                    """Collapse equal values to a scalar and preserve multiple values as a list."""
                    unique_values = list(dict.fromkeys(values.dropna()))
                    if len(unique_values) == 1:
                        return unique_values[0]
                    return unique_values

                attrs = pd.DataFrame(index=pd.Index(node_attrs["target_node"].drop_duplicates(), name="target_node"))
                for column in attr_columns:
                    attr_long = node_attrs[["target_node", column]].explode(column)
                    attr_long = attr_long.dropna(subset=[column])
                    if attr_long.empty:
                        continue
                    attrs[column] = attr_long.groupby("target_node", sort=False)[column].agg(collapse_values)

                for column in attrs.columns:
                    if column not in walk_nodes.columns:
                        walk_nodes[column] = pd.NA
                    walk_nodes.loc[attrs.index, column] = attrs[column]

            projected_walk = UrbanGraph(
                nodes_gdf=walk_nodes,
                edges_gdf=projected_walk.edges_gdf.copy(),
                is_multigraph=projected_walk.is_multigraph,
                is_directed=projected_walk.is_directed,
                edge_direction_column=projected_walk.edge_direction_column,
                adjacency_weight=projected_walk.adjacency_weight,
                crs=projected_walk.crs,
                graph_type=projected_walk.type,
            )

    projected_mapping = object2node_map.dropna().astype(int).to_dict()
    pt_nodes_remapped = pt_nodes.loc[~pt_nodes.index.isin(projected_mapping.keys())].copy()

    relabel_mapping = dict(projected_mapping)
    start_node_id = 0 if projected_walk.nodes_gdf.empty else int(pd.Index(projected_walk.nodes_gdf.index).max()) + 1
    new_index = pd.RangeIndex(start_node_id, start_node_id + len(pt_nodes_remapped))
    relabel_mapping.update(dict(zip(pt_nodes_remapped.index, new_index)))
    pt_nodes_remapped.index = new_index

    pt_edges_remapped = public_transport_g.edges_gdf.copy()
    if not pt_edges_remapped.empty:
        pt_edges_remapped["u"] = pt_edges_remapped["u"].map(relabel_mapping)
        pt_edges_remapped["v"] = pt_edges_remapped["v"].map(relabel_mapping)
        pt_edges_remapped = pt_edges_remapped.dropna(subset=["u", "v"]).copy()
        if not pt_edges_remapped.empty:
            pt_edges_remapped[["u", "v"]] = pt_edges_remapped[["u", "v"]].astype(int)
            if public_transport_g.edge_direction_column is not None:
                pt_edges_remapped[public_transport_g.edge_direction_column] = pt_edges_remapped[
                    public_transport_g.edge_direction_column
                ].astype(bool)
            if public_transport_g.is_multigraph:
                pt_edges_remapped["k"] = pt_edges_remapped.groupby(["u", "v"], sort=False).cumcount()

    nodes = gpd.GeoDataFrame(
        pd.concat([projected_walk.nodes_gdf, pt_nodes_remapped], axis=0, sort=False),
        geometry=projected_walk.nodes_gdf.geometry.name,
        crs=projected_walk.crs,
    )
    edges = gpd.GeoDataFrame(
        pd.concat([projected_walk.edges_gdf, pt_edges_remapped], axis=0, ignore_index=True, sort=False),
        geometry=projected_walk.edges_gdf.geometry.name,
        crs=projected_walk.crs,
    )
    if not edges.empty:
        edges["oneway"] = edges["oneway"].astype(bool)
        edges["k"] = edges.groupby(["u", "v"], sort=False).cumcount()

    intermodal = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        adjacency_weight=projected_walk.adjacency_weight,
        crs=projected_walk.crs,
        graph_type="intermodal",
    )

    if keep_largest_subgraph:
        intermodal = keep_largest_connected_component(intermodal)
    logger.debug("Done composing intermodal UrbanGraph.")
    return intermodal


def get_intermodal_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    osm_edge_tags: list[str] | None = None,
    max_dist: float = 30,
    keep_largest_subgraph: bool = True,
    add_link_edge: bool = True,
    walk_kwargs: dict[str, Any] | None = None,
    pt_kwargs: dict[str, Any] | None = None,
) -> UrbanGraph:
    """
    Build an intermodal (PT+walking) ``UrbanGraph`` for a territory.

    The function resolves a boundary polygon (by ``osm_id`` or ``territory``),
    runs pedestrian graph construction (:func:`get_walk_graph`) and public
    transport graph construction (:func:`get_public_transport_graph`) in
    parallel, then connects public-transport platforms to nearby walk edges via
    :func:`join_pt_walk_graph` using the ``max_dist`` snapping radius. Edge
    lengths (meters) and travel times (minutes) come from the underlying
    builders and from the walk-edge splits created during projection.

    Parameters:
        osm_id (int | None): OSM relation/area id for the territory; provide this or ``territory``.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326.
        clip_by_territory (bool): If True, both PT and Walk graphs are clipped to the boundary.
        osm_edge_tags (list[str] | None): Subset of OSM tags to retain (forwarded to both builders).
        max_dist (float): Max distance in meters to connect PT platforms to walk edges.
        keep_largest_subgraph (bool): If True, keep only the largest strongly connected component after joining.
        add_link_edge (bool): If True, add platform-to-walk connector edges during projection.
        walk_kwargs (dict[str, Any] | None): Extra keyword args for :func:`get_walk_graph` (e.g., ``walk_speed``,
            ``simplify``, ``osm_edge_tags``, ``keep_largest_subgraph``). Walk graph keep_largest_subgraph defaults to
            False unless explicitly set here.
        pt_kwargs (dict[str, Any] | None): Extra keyword args for :func:`get_public_transport_graph`
            (e.g., ``transport_types``, ``osm_edge_tags``, ``transport_registry``).

    Returns:
        Intermodal ``UrbanGraph`` combining public transport and pedestrian
        edges. The graph CRS equals the builders' local projected CRS.

    Notes:
        - ``clip_by_territory`` and ``osm_edge_tags`` are propagated to both builders unless
          overridden in ``walk_kwargs``/``pt_kwargs``.
        - If the PT graph is empty for the area, the function returns the walking graph alone (with a warning).
        - Joining requires both sub-graphs to share the same CRS; builders derive a local projected CRS
          from the boundary's extent, so lengths/times are in meters/minutes.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
    """
    boundary = get_4326_boundary(osm_id=osm_id, territory=territory)

    walk_kwargs = dict(walk_kwargs or {})
    pt_kwargs = dict(pt_kwargs or {})

    for kwargs in (walk_kwargs, pt_kwargs):
        if osm_edge_tags is not None:
            kwargs.setdefault("osm_edge_tags", osm_edge_tags)
        kwargs.setdefault("clip_by_territory", clip_by_territory)

    walk_kwargs.setdefault("keep_largest_subgraph", False)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        walk_future = executor.submit(get_walk_graph, territory=boundary, **walk_kwargs)
        logger.debug("Started downloading and parsing walk graph...")

        pt_future = executor.submit(get_public_transport_graph, territory=boundary, **pt_kwargs)
        logger.debug("Started downloading and parsing public transport graph...")

        pt_g = pt_future.result()
        logger.debug("Public transport graph done!")

        walk_g = walk_future.result()
        logger.debug("Walk graph done!")

    if pt_g.nodes_gdf.empty:
        logger.warning("Public transport graph is empty! Returning only walk graph.")
        return walk_g

    walk_speed = walk_kwargs.get("walk_speed")
    intermodal = join_pt_walk_graph(
        pt_g,
        walk_g,
        max_dist=max_dist,
        keep_largest_subgraph=keep_largest_subgraph,
        add_link_edge=add_link_edge,
        walk_speed_m_per_min=walk_speed,
    )
    return intermodal
