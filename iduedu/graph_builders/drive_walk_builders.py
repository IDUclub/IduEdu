from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import LineString, Polygon, linestrings
from shapely.geometry.multipolygon import MultiPolygon

from iduedu import _rust, config
from iduedu.constants.highway_enums import HighwayType
from iduedu.constants.network_enums import Network
from iduedu.graph.transformers import estimate_crs_for_bounds, keep_largest_connected_component
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.overpass.downloaders import get_4326_boundary, get_network_by_filters, osm_source_label

logger = config.logger


def _assign_edge_keys(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edges = edges.copy()
    edges["k"] = edges.groupby(["u", "v"], sort=False).cumcount()
    return edges


def _build_nodes_and_uv(edges: gpd.GeoDataFrame, crs: CRS) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1

    starts = coords[first_idx]
    ends = coords[last_idx]

    edges = edges.copy()
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n_edges = len(edges)
    edges["u"] = labels[:n_edges].astype(int)
    edges["v"] = labels[n_edges:].astype(int)

    if len(pd.Index(uniques)) == 0:
        return gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(geometry=[])
    coords = np.asarray(list(pd.Index(uniques)), dtype=float)
    nodes = gpd.GeoDataFrame(
        index=pd.RangeIndex(len(coords)), geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), crs=crs
    )

    return nodes, edges


def _get_highway_properties(highway) -> tuple[str, float]:
    """
    Map OSM ``highway`` class(es) to a regulatory category and a default speed.

    Accepts either a single string or a list of classes. When multiple classes are present,
    picks the lowest (most permissive) regulatory status and the minimum default speed among them.

    Parameters:
        highway (str | list[str]): OSM highway class or list of classes (e.g., "primary", "residential", ...).

    Returns:
        (tuple[str, float]): A pair ``(category, maxspeed_mpm)`` where:
            - ``category`` (str): regulatory class label, e.g., "local" | "regional" | "federal".
            - ``maxspeed_mpm`` (float): default speed for that class in **meters per minute**.

    Notes:
        If the class is unknown, returns defaults for ``UNCLASSIFIED``.
    """
    default = (HighwayType.UNCLASSIFIED.reg_status, HighwayType.UNCLASSIFIED.max_speed)

    if not highway:
        return default

    highway_list = highway if isinstance(highway, list) else [highway]
    reg_values, speed_values = [], []
    for ht in highway_list:
        try:
            enum_value = HighwayType[ht.upper()]
        except KeyError:
            continue
        reg_values.append(enum_value.reg_status)
        speed_values.append(enum_value.max_speed)

    if not reg_values or not speed_values:
        return default

    rank = {"local": 0, "regional": 1, "federal": 2}
    lowest_reg = min(reg_values, key=lambda s: rank.get(s, 999))

    min_speed = min(speed_values) if speed_values else default[1]

    return lowest_reg, min_speed


#: Tags that always separate edges, whether or not they are kept as columns: a bridge or a tunnel does not
#: follow the ground, so it must not be merged with the road it continues.
STRUCTURE_TAGS = ("bridge", "tunnel")


def _build_edges_from_overpass(
    polygon: Polygon, way_filter: str, needed_tags: set[str], local_crs: CRS, simplify: bool = True
) -> gpd.GeoDataFrame:
    """
    Download OSM ways by filter, cut them into edges and project them to a local CRS.

    Every way is split into its straight segments. With ``simplify=True`` the segments are merged back into
    edges that run from junction to junction: two segments are joined at a point only when they are the
    only segments there, follow one another, and carry the same tags (``needed_tags`` and the bridge and
    tunnel tags). An edge therefore never mixes the attributes of different ways, and bridges and tunnels
    stay separate edges. With ``simplify=False`` every segment is an edge.

    Parameters:
        polygon (Polygon): Boundary polygon in EPSG:4326 used to query OSM data.
        way_filter (str): Overpass filter applied to ways (e.g., ``[ "highway" ~ "motorway|trunk|primary|..." ]``).
        needed_tags (set[str]): OSM tags to put on the edges.
        local_crs (CRS): Projected CRS of the result.
        simplify (bool): Merge segments into junction-to-junction edges.

    Returns:
        GeoDataFrame: Line features in the local CRS with ``geometry``, ``way_idx`` (row of the source way)
        and one column per tag in ``needed_tags``.
    """
    data = get_network_by_filters(polygon, way_filter)
    if len(data) == 0:
        return gpd.GeoDataFrame()

    ways = data[data["type"] == "way"].copy()
    if len(ways) == 0:
        return gpd.GeoDataFrame()

    # Collecting the coordinates of each line (lon, lat): Overpass gives point dicts, a .pbf file arrays.
    coord_entries = [
        (idx, pts if isinstance(pts, np.ndarray) else np.asarray([(p["lon"], p["lat"]) for p in pts], dtype="f8"))
        for idx, pts in ways["geometry"].items()
        if len(pts) >= 2
    ]
    if not coord_entries:
        return gpd.GeoDataFrame()
    way_indices, coords_list = zip(*coord_entries)
    way_indices = np.asarray(way_indices)
    lonlat = np.concatenate(coords_list, axis=0)
    counts = np.array([a.shape[0] for a in coords_list], dtype=np.int64)
    way_of_point = np.repeat(np.arange(len(counts)), counts)

    # Points with the same coordinates are one node; segments join consecutive points of a way.
    _, node_of = np.unique(lonlat, axis=0, return_inverse=True)
    node_of = node_of.ravel()
    start = np.flatnonzero((way_of_point[1:] == way_of_point[:-1]) & (node_of[1:] != node_of[:-1]))
    if len(start) == 0:
        return gpd.GeoDataFrame()
    way_of_segment = way_of_point[start]

    tags = ways["tags"]
    tag_keys = sorted(set(needed_tags) | set(STRUCTURE_TAGS))
    if len(tag_keys) > 50:
        tags_expanded = pd.json_normalize(tags.tolist())
        tags_expanded.index = ways.index
        tags_df = tags_expanded.reindex(columns=tag_keys)
    else:
        cols = {k: tags.map(lambda d, kk=k: d.get(kk) if isinstance(d, dict) else None) for k in tag_keys}
        tags_df = pd.DataFrame(cols, index=ways.index)

    if simplify:
        # Ways with equal tags share a key; only segments with the same key are merged.
        way_tags = tags_df.loc[way_indices]
        key_of_way = way_tags.groupby(tag_keys, dropna=False, sort=False).ngroup().to_numpy()
        order, offsets = _rust.merge_chains(
            node_of[start].astype(np.uint32),
            node_of[start + 1].astype(np.uint32),
            key_of_way[way_of_segment].astype(np.uint32),
            int(node_of.max()) + 1,
        )
    else:
        order, offsets = np.arange(len(start)), np.arange(len(start) + 1)

    # The points of an edge: the start of its first segment, then the end of every segment.
    n_edges = len(offsets) - 1
    edge_of_segment = np.repeat(np.arange(n_edges), np.diff(offsets))
    ordered = start[order]
    points = np.empty(len(ordered) + n_edges, dtype=np.int64)
    edge_of_point = np.empty(len(points), dtype=np.int64)
    first = offsets[:-1] + np.arange(n_edges)
    points[first] = ordered[offsets[:-1]]
    edge_of_point[first] = np.arange(n_edges)
    rest = np.arange(len(ordered)) + edge_of_segment + 1
    points[rest] = ordered + 1
    edge_of_point[rest] = edge_of_segment

    x, y = Transformer.from_crs(4326, local_crs, always_xy=True).transform(lonlat[:, 0], lonlat[:, 1])
    geoms = linestrings(np.column_stack([x, y])[points], indices=edge_of_point)
    edges = gpd.GeoDataFrame(
        {"way_idx": way_indices[way_of_segment[order[offsets[:-1]]]]}, geometry=geoms, crs=local_crs
    )
    return edges.join(tags_df[sorted(needed_tags)], on="way_idx")


def get_drive_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    simplify: bool = True,
    add_road_category: bool = True,
    clip_by_territory: bool = False,
    keep_largest_subgraph: bool = True,
    network_type: Literal["drive", "drive_service", "custom"] = "drive",
    custom_filter: str | None = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
) -> UrbanGraph:
    """
    Build a drivable UrbanGraph from OpenStreetMap within a given territory.

    The function downloads OSM ways via Overpass (or reads them from a local OSM file in pbf mode, see
    ``config.set_osm_source``), segments them into directed edges, optionally merges
    contiguous segments, normalizes OSM one-way semantics into a bool ``oneway`` edge column, and computes
    per-edge length (meters) and travel time (minutes). Node coordinates are unique line endpoints in a
    local projected CRS. Edge attributes can include selected OSM tags and a derived road category/speed.

    Parameters:
        osm_id (int | None): OSM relation/area ID of the territory boundary. Provide this or ``territory``.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 or GeoDataFrame.
            Used when ``osm_id`` is not given.
        simplify (bool): If True, merges the segments of the ways into edges that run from junction to
            junction. Segments are merged only when they carry the same tags, so an edge keeps the attributes
            of its ways, and bridges and tunnels stay separate edges. If False, keeps raw per-segment edges.
        add_road_category (bool): If True, adds a derived ``category`` (e.g., local/regional/federal) and a default
            speed (``maxspeed_mpm``) inferred from the OSM ``highway`` class.
        clip_by_territory (bool): If True, clips edges by the exact boundary geometry before graph construction.
        keep_largest_subgraph (bool): If True, returns only the largest strongly connected component.
        network_type: Preset of Overpass filters to select drivable ways.
            Use "custom" together with ``custom_filter`` to pass your own Overpass ``way`` filter.
        custom_filter (str | None): Custom Overpass way filter used in custom mode.
        osm_edge_tags (list[str] | None): Which OSM tags to retain on edges. Overrides defaults from config.
            The tags oneway, maxspeed, and highway are always added.

    Returns:
        UrbanGraph: Directed multigraph of the road network. Edges include
        geometry in the local CRS, length, travel time, direction metadata and selected OSM tags. The graph
        keeps the local projected CRS and selected network type as metadata.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html

    """
    polygon4326 = get_4326_boundary(osm_id=osm_id, territory=territory)

    filters = {"drive": Network.DRIVE.filter, "drive_service": Network.DRIVE_SERVICE.filter, "custom": custom_filter}
    try:
        road_filter = filters[network_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {network_type!r}")

    if network_type == "custom" and road_filter is None:
        raise ValueError("For road_type='custom' you must provide custom_filter")

    if osm_edge_tags is None:
        needed_tags = set(config.drive_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    tags_to_retrieve = set(needed_tags) | {"oneway", "maxspeed", "highway"}

    local_crs = estimate_crs_for_bounds(*polygon4326.bounds)

    logger.info(f"Retrieving drive network via {osm_source_label()}...")
    edges_gdf = _build_edges_from_overpass(
        polygon4326, road_filter, needed_tags=tags_to_retrieve, local_crs=local_crs, simplify=simplify
    )

    if len(edges_gdf) == 0:
        logger.warning("No edges found, returning empty graph")
        return UrbanGraph.empty(
            crs=local_crs, is_multigraph=True, is_directed=True, edge_direction_column="oneway", graph_type=network_type
        )

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges_gdf = edges_gdf.clip(clip_poly_gdf, keep_geom_type=True).reset_index(drop=True)
        if len(edges_gdf) == 0:
            logger.warning("No edges left after clipping, returning empty graph")
            return UrbanGraph.empty(
                crs=local_crs,
                is_multigraph=True,
                is_directed=True,
                edge_direction_column="oneway",
                graph_type=network_type,
            )

    if "oneway" not in edges_gdf.columns:
        edges_gdf["oneway"] = "no"

    edges_gdf["oneway"] = edges_gdf["oneway"].fillna("no")
    oneway_norm = edges_gdf["oneway"].astype(str).str.lower().str.strip()
    reverse_mask = oneway_norm.eq("-1")
    if reverse_mask.any():
        edges_gdf.loc[reverse_mask, "geometry"] = edges_gdf.loc[reverse_mask, "geometry"].apply(
            lambda geom: LineString(list(geom.coords)[::-1])
        )
    edges_gdf["oneway"] = oneway_norm.isin({"yes", "true", "1", "-1"})

    nodes_gdf, edges_gdf = _build_nodes_and_uv(edges_gdf, local_crs)

    if "highway" not in edges_gdf.columns:
        edges_gdf["highway"] = None
    edges_gdf[["category", "maxspeed_mpm"]] = edges_gdf["highway"].apply(
        lambda h: pd.Series(_get_highway_properties(h))
    )

    if "maxspeed" in edges_gdf.columns:
        maxspeed_osm_mpm = (pd.to_numeric(edges_gdf["maxspeed"], errors="coerce") * 1000.0 / 60.0).round(3)
        edges_gdf["speed_mpm"] = maxspeed_osm_mpm.fillna(edges_gdf["maxspeed_mpm"])
    else:
        edges_gdf["speed_mpm"] = edges_gdf["maxspeed_mpm"]

    edges_gdf["length_meter"] = edges_gdf.geometry.length.round(3)
    edges_gdf["time_min"] = (edges_gdf["length_meter"] / edges_gdf["speed_mpm"]).round(3)
    edges_gdf["type"] = "drive"

    edges_gdf = _assign_edge_keys(edges_gdf)

    edge_attr_cols = set(needed_tags) | {"u", "v", "k", "geometry", "length_meter", "time_min", "type", "oneway"}
    if add_road_category:
        edge_attr_cols.add("category")
    edges_gdf = edges_gdf[[col for col in edges_gdf.columns if col in edge_attr_cols]].copy()

    graph = UrbanGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=local_crs,
        graph_type=network_type,
    )
    if keep_largest_subgraph:
        graph = keep_largest_connected_component(graph)
    logger.debug("Drive graph built.")
    return graph


def get_walk_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    simplify: bool = True,
    clip_by_territory: bool = False,
    keep_largest_subgraph: bool = True,
    walk_speed: float = 5 * 1000 / 60,  # m/min
    network_type: Literal["walk", "custom"] = "walk",
    custom_filter: str | None = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
) -> UrbanGraph:
    """
    Build a pedestrian UrbanGraph from OpenStreetMap within a given territory.

    The function fetches OSM ways via Overpass (or a local OSM file in pbf mode) using a walking filter, splits
    each way into line segments,
    and computes per-edge length (meters) and traversal time (minutes) using a given walking speed (m/min).
    The graph is stored as an undirected multigraph because pedestrian movement is treated as bidirectional.

    Parameters:
        osm_id (int | None): OSM relation/area ID for the boundary. Provide this or ``territory``.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry (EPSG:4326) or a GeoDataFrame
            to define the area when ``osm_id`` is not given.
        simplify (bool): If True, merges the segments of the ways into edges that run from junction to
            junction. Segments are merged only when they carry the same tags, so an edge keeps the attributes
            of its ways, and bridges and tunnels stay separate edges. If False, keeps raw per-segment edges.
        clip_by_territory (bool): If True, clips edges by the provided boundary before graph construction.
        keep_largest_subgraph (bool): If True, retains only the largest strongly connected component.
        walk_speed (float): Walking speed in meters per minute used to compute time for each edge.
        network_type: Preset of Overpass filters. Use custom mode with custom_filter
            to pass your own way filter.
        custom_filter (str | None): Custom Overpass filter string used in custom mode.
        osm_edge_tags (list[str] | None): List of OSM edge tags to retain (overrides defaults). Only these keys
            are joined from element tags.

    Returns:
        UrbanGraph: Undirected multigraph of the walking network. Edges include
        geometry in the local CRS, length, travel time and selected OSM tags. The graph keeps the local
        projected CRS and selected network type as metadata.

    Notes:
        All walking edges are treated as bidirectional by the undirected UrbanGraph adjacency builder.
        Lengths are measured in meters in a local projected CRS estimated from the territory bounds.

    See also:
        https://iduclub.github.io/IduEdu/examples/get_any_graph.html
    """
    polygon4326 = get_4326_boundary(osm_id=osm_id, territory=territory)

    filters = {
        "walk": Network.WALK.filter,
        "custom": custom_filter,
    }
    try:
        road_filter = filters[network_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {network_type!r}")
    if network_type == "custom" and road_filter is None:
        raise ValueError("For road_type='custom' you must provide custom_filter")

    if osm_edge_tags is None:
        needed_tags = set(config.walk_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    local_crs = estimate_crs_for_bounds(*polygon4326.bounds)

    logger.info(f"Downloading walk network via {osm_source_label()} ...")
    edges_gdf = _build_edges_from_overpass(
        polygon4326, road_filter, needed_tags=needed_tags, local_crs=local_crs, simplify=simplify
    )

    if len(edges_gdf) == 0:
        logger.warning("No edges found, returning empty graph")
        return UrbanGraph.empty(crs=local_crs, is_multigraph=True, is_directed=False, graph_type=network_type)

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges_gdf = edges_gdf.clip(clip_poly_gdf, keep_geom_type=True).explode(ignore_index=True)
        if len(edges_gdf) == 0:
            logger.warning("No edges left after clipping, returning empty graph")
            return UrbanGraph.empty(crs=local_crs, is_multigraph=True, is_directed=False, graph_type=network_type)

    nodes_gdf, edges_gdf = _build_nodes_and_uv(edges_gdf, local_crs)

    edges_gdf["length_meter"] = edges_gdf.geometry.length.round(3)
    edges_gdf["time_min"] = (edges_gdf["length_meter"] / float(walk_speed)).round(3)
    edges_gdf["type"] = "walk"

    edges_gdf = _assign_edge_keys(edges_gdf)

    edge_attrs = set(needed_tags) | {"u", "v", "k", "geometry", "length_meter", "time_min", "type"}
    edges_gdf = edges_gdf[[col for col in edges_gdf.columns if col in edge_attrs]].copy()

    graph = UrbanGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        is_multigraph=True,
        is_directed=False,
        crs=local_crs,
        graph_type=network_type,
    )
    if keep_largest_subgraph:
        graph = keep_largest_connected_component(graph)
    logger.debug("Walk graph built.")
    return graph
