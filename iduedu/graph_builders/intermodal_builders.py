import concurrent.futures
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon

from iduedu import config
from iduedu.constants.network_enums import WALK_ACCESS_NODE_TYPES
from iduedu.graph.editors import (
    _apply_graph_changes_to_tables,
    _join_graph_tables,
    _project_objects,
)
from iduedu.graph.routes import join_routes, used_routes
from iduedu.graph.transformers import _directed_edges, _restrict_to_largest_strong_component
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders.drive_walk_builders import get_walk_graph
from iduedu.graph_builders.public_transport_builders import get_public_transport_graph
from iduedu.overpass.downloaders import get_4326_boundary

logger = config.logger

DEFAULT_WALK_SPEED_M_PER_MIN = 5 * 1000 / 60


def _normalize_oneway(values: pd.Series, default: bool) -> pd.Series:
    """Boolean ``oneway`` values: missing ones take ``default``, the rest their truth value."""
    if values.dtype == bool:
        return values
    if values.empty:
        return values.map(bool)
    if isinstance(values.dtype, pd.CategoricalDtype):
        # Mapping categories can preserve the categorical dtype.
        return values.map(lambda value: default if pd.isna(value) else bool(value))
    # Use scalar truthiness for extension dtypes too.
    return values.astype(object).mask(values.isna(), default).astype(bool)


def _collapse_platform_attribute(attr_long: pd.DataFrame, column: str) -> pd.Series:
    """Keep the first occurrence of each value and collect only multi-value nodes into lists."""
    unique = attr_long.drop_duplicates(["target_node", column]).set_index("target_node")[column]
    multiple = unique.index.duplicated(keep=False)
    if multiple.any():
        lists = unique[multiple].groupby(level=0, sort=False).agg(list)
        if multiple.all():
            return lists
        return pd.concat([unique[~multiple].astype(object), lists])
    return unique.infer_objects() if pd.api.types.is_object_dtype(unique) else unique


def _transfer_platform_attributes(
    walk_nodes: gpd.GeoDataFrame, pt_nodes: gpd.GeoDataFrame, object2node_map: pd.Series
) -> None:
    """Copy the attributes of projected platforms onto their walk nodes, in place."""
    mapped_platforms = object2node_map.rename("target_node").to_frame()
    node_attrs = pt_nodes.drop(columns=pt_nodes.geometry.name).join(mapped_platforms, how="inner")
    targets = pd.Index(node_attrs["target_node"].drop_duplicates(), name="target_node")
    for column in node_attrs.columns.drop("target_node"):
        # Explode uses the iteration order of list-like values, including lists, tuples and sets.
        attr_long = node_attrs[["target_node", column]].explode(column).dropna(subset=[column])
        if attr_long.empty:
            continue
        attrs = _collapse_platform_attribute(attr_long, column).reindex(targets)
        if column not in walk_nodes.columns:
            walk_nodes[column] = pd.NA
        walk_nodes.loc[targets, column] = attrs


def _walk_key_offsets(walk_edges: gpd.GeoDataFrame, pt_edges: gpd.GeoDataFrame):
    """Count walk edges per ``(u, v)`` pair in the order of the public-transport edge table."""

    if walk_edges.empty:
        return 0
    candidates = walk_edges.loc[walk_edges["u"].isin(pt_edges["u"]), ["u", "v"]]
    counts = candidates.groupby(["u", "v"], sort=False).size().rename("offset").reset_index()
    offsets = pt_edges[["u", "v"]].merge(counts, on=["u", "v"], how="left")["offset"]
    return offsets.fillna(0).astype(int).to_numpy()


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

    walk_edges = _directed_edges(walk_g.edges_gdf, "oneway", False)
    walk_nodes = walk_g.nodes_gdf

    pt_nodes = public_transport_g.nodes_gdf
    platform_mask = (
        pt_nodes["type"].isin(WALK_ACCESS_NODE_TYPES)
        if "type" in pt_nodes.columns
        else pd.Series(False, index=pt_nodes.index)
    )
    pt_platforms = pt_nodes.loc[platform_mask]

    if pt_platforms.empty:
        logger.warning("Public transport graph has no platform-like nodes. Joining graphs without projection.")
        object2node_map = pd.Series(dtype=int)
        walk_routes = used_routes(walk_g.routes, walk_edges)
    else:
        changes, object2node_map = _project_objects(
            walk_nodes,
            walk_edges,
            pt_platforms,
            walk_speed,
            crs=walk_g.crs,
            is_multigraph=walk_g.is_multigraph,
            is_directed=True,
            direction_column="oneway",
            max_dist=max_dist,
            add_link_edge=add_link_edge,
        )
        unprojected = pt_platforms.index.difference(object2node_map.index)
        if len(unprojected) > 0:
            # Routes usually extend past the walk graph territory, so distant platforms are expected.
            unprojected_by_type = pt_platforms.loc[unprojected, "type"].value_counts().to_dict()
            logger.info(
                f"{len(unprojected)} of {len(pt_platforms)} platform-like nodes are farther than {max_dist} m "
                f"from the walk network and stay unconnected: {unprojected_by_type}"
            )
        walk_nodes, walk_edges = _apply_graph_changes_to_tables(walk_nodes, walk_edges, changes)
        walk_routes = None
        if "type" not in walk_edges.columns:
            walk_edges["type"] = "walk"
        else:
            walk_edges["type"] = walk_edges["type"].fillna("walk")
        walk_edges["oneway"] = _normalize_oneway(walk_edges["oneway"], default=False)

    if not walk_edges.empty and not walk_g.is_multigraph:
        walk_edges["k"] = walk_edges.groupby(["u", "v"], sort=False).cumcount()

    object2node_map = object2node_map.dropna().astype(int)
    if not object2node_map.empty:
        _transfer_platform_attributes(walk_nodes, pt_nodes, object2node_map)
    projected_mapping = object2node_map.to_dict()
    pt_nodes_remapped = pt_nodes.loc[~pt_nodes.index.isin(projected_mapping.keys())].copy()

    relabel_mapping = dict(projected_mapping)
    # PT nodes keep their ids where the walk graph does not use them (coordinate ids of both graphs stay stable);
    # the others continue after the largest id of both.
    walk_ids = walk_nodes.index
    taken = pt_nodes_remapped.index.isin(walk_ids)
    if not pd.api.types.is_integer_dtype(pt_nodes_remapped.index) or not pd.api.types.is_integer_dtype(walk_ids):
        start_node_id = 0 if walk_ids.empty else int(walk_ids.max()) + 1
        new_index = pd.RangeIndex(start_node_id, start_node_id + len(pt_nodes_remapped))
    elif taken.any():
        kept = pt_nodes_remapped.index[~taken]
        start_node_id = int(walk_ids.append(kept).max()) + 1 if len(walk_ids) or len(kept) else 0
        fresh = np.arange(start_node_id, start_node_id + int(taken.sum()), dtype=np.int64)
        new_ids = pt_nodes_remapped.index.to_numpy(dtype=np.int64, copy=True)
        new_ids[taken] = fresh
        new_index = pd.Index(new_ids)
    else:
        new_index = pt_nodes_remapped.index
    relabel_mapping.update(dict(zip(pt_nodes_remapped.index, new_index)))
    pt_nodes_remapped.index = new_index

    pt_edges_remapped = public_transport_g.edges_gdf
    if not pt_edges_remapped.empty:
        # Positions instead of a dict map: a missing endpoint would turn the ids into floats, which cannot hold
        # the coordinate ids above 2**53.
        old_ids = pd.Index(list(relabel_mapping.keys()))
        target_ids = np.asarray(list(relabel_mapping.values()), dtype=np.int64)
        u_position = old_ids.get_indexer(pt_edges_remapped["u"])
        v_position = old_ids.get_indexer(pt_edges_remapped["v"])
        kept_edges = (u_position >= 0) & (v_position >= 0)
        pt_edges_remapped = pt_edges_remapped[kept_edges].copy()
        pt_edges_remapped["u"] = target_ids[u_position[kept_edges]]
        pt_edges_remapped["v"] = target_ids[v_position[kept_edges]]
    if not pt_edges_remapped.empty:
        if "oneway" in pt_edges_remapped.columns:
            pt_edges_remapped["oneway"] = _normalize_oneway(pt_edges_remapped["oneway"], default=True)
        else:
            pt_edges_remapped["oneway"] = True
        # Edge keys must be unique across both tables, so PT keys continue the walk keys of the same pair.
        pt_edges_remapped["k"] = pt_edges_remapped.groupby(["u", "v"], sort=False).cumcount() + _walk_key_offsets(
            walk_edges, pt_edges_remapped
        )

    # Platforms projected onto the walk network keep their walk node ids, so those nodes belong to
    # both parts; the table join merges the duplicated index and keeps the walk row.
    projected_targets = pd.Index(sorted(set(projected_mapping.values())))
    pt_part_nodes = gpd.GeoDataFrame(
        pd.concat([walk_nodes.loc[projected_targets], pt_nodes_remapped], axis=0, sort=False),
        geometry=walk_nodes.geometry.name,
        crs=walk_g.crs,
    )
    nodes, edges = _join_graph_tables(walk_nodes, walk_edges, pt_part_nodes, pt_edges_remapped, walk_g.crs)
    pt_routes = used_routes(public_transport_g.routes, pt_edges_remapped)
    intermodal = UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        adjacency_weight=walk_g.adjacency_weight,
        crs=walk_g.crs,
        graph_type="intermodal",
        routes=join_routes(walk_routes, pt_routes),
    )

    if keep_largest_subgraph:
        nodes_before = intermodal.nodes_gdf
        _restrict_to_largest_strong_component(intermodal)
        dropped = nodes_before.index.difference(intermodal.nodes_gdf.index)
        if len(dropped) > 0 and "type" in nodes_before.columns:
            # Typed nodes are the public-transport ones: losing them means a mode lost its
            # connection to the walk network, which is worth reporting louder than the total count.
            dropped_by_type = nodes_before.loc[dropped, "type"].dropna().value_counts().to_dict()
            if dropped_by_type:
                logger.warning(f"Public transport nodes dropped with the smaller components: {dropped_by_type}")
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
    transport graph construction (:func:`iduedu.get_public_transport_graph`) in
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
        pt_kwargs (dict[str, Any] | None): Extra keyword args for :func:`iduedu.get_public_transport_graph`
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
