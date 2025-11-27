import time
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import LineString, MultiLineString, Polygon, line_merge, linestrings
from shapely.geometry.multipolygon import MultiPolygon

from iduedu import config
from iduedu.enums.highway_enums import HighwayType
from iduedu.enums.network_enums import Network
from iduedu.modules.graph_transformers import estimate_crs_for_bounds, keep_largest_strongly_connected_component
from iduedu.modules.overpass_downloaders import get_4326_boundary, get_network_by_filters

logger = config.logger


def _get_highway_properties(highway) -> tuple[str, float]:
    """
    Map OSM `highway` class(es) to a regulatory category and a default speed.

    Accepts either a single string or a list of classes. When multiple classes are present,
    picks the lowest (most permissive) regulatory status and the minimum default speed among them.

    Parameters:
        highway (str | list[str]): OSM highway class or list of classes (e.g., "primary", "residential", ...).

    Returns:
        (tuple[str, float]): A pair `(category, maxspeed_mpm)` where:
            - `category` (str): regulatory class label, e.g., "local" | "regional" | "federal".
            - `maxspeed_mpm` (float): default speed for that class in **meters per minute**.

    Notes:
        If the class is unknown, returns defaults for `UNCLASSIFIED`.
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


def _build_edges_from_overpass(
    polygon: Polygon,
    way_filter: str,
    needed_tags: set[str],
    simplify: bool = True,
) -> tuple[GeoDataFrame, CRS]:
    """
    Download OSM ways by filter, segment into edges, and project to a local CRS.

    The function queries the ways within `polygon` using an Overpass `way_filter`, converts each way
    into consecutive line segments (one edge per segment), and projects the result to a local metric CRS.
    If `simplify=True`, contiguous segments are merged (`line_merge`), and OSM attributes are transferred
    to merged lines via midpoint nearest join.

    Parameters:
        polygon (Polygon): Boundary polygon in EPSG:4326 used to query OSM data.
        way_filter (str): Overpass filter applied to ways (e.g., `[ "highway" ~ "motorway|trunk|primary|..." ]`).
        simplify (bool): If True, merges contiguous segments into longer edges and reattaches attributes
            using a nearest midpoint join.

    Returns:
        (tuple[gpd.GeoDataFrame, CRS]): A pair `(edges, local_crs)` where:
            - `edges` (GeoDataFrame): Line features in local CRS with columns:
                `geometry` (LineString), `way_idx` (source way index), `id` (OSM way id), `tags` (dict).
            - `local_crs` (pyproj.CRS): Estimated local projected CRS suitable for metric length computations.

    Notes:
        Attributes on merged edges are inferred from the nearest original segment around the midpoint,
        which may drop or aggregate original per-segment variability.
    """
    data = get_network_by_filters(polygon, way_filter)
    if len(data) == 0:
        return gpd.GeoDataFrame(), CRS.from_epsg(4326)
    logger.info("Downloading network via Overpass done!")
    ways = data[data["type"] == "way"].copy()

    # Собираем координаты каждой линии (lon, lat)
    coords_list = [np.asarray([(p["lon"], p["lat"]) for p in pts], dtype="f8") for pts in ways["geometry"].values]

    # сегментация на отрезки
    starts = np.concatenate([a[:-1] for a in coords_list], axis=0)
    ends = np.concatenate([a[1:] for a in coords_list], axis=0)

    lengths = np.array([a.shape[0] for a in coords_list], dtype=int)
    seg_counts = np.maximum(lengths - 1, 0)
    way_idx = np.repeat(ways.index.values, seg_counts)

    coords = np.stack([starts, ends], axis=1)
    geoms = linestrings(coords)

    # локальная проекция
    local_crs = estimate_crs_for_bounds(*polygon.bounds)

    edges = gpd.GeoDataFrame({"way_idx": way_idx}, geometry=geoms, crs=4326).to_crs(local_crs)

    tags = ways["tags"]

    tag_keys: set[str] = set(needed_tags)
    if len(tag_keys) > 50:
        tags_expanded = pd.json_normalize(tags.tolist())
        tags_expanded.index = ways.index
        tags_df = tags_expanded.loc[:, tags_expanded.columns.intersection(tag_keys)]
    else:
        cols = {k: tags.map(lambda d, kk=k: d.get(kk) if isinstance(d, dict) else None) for k in tag_keys}
        tags_df = pd.DataFrame(cols, index=ways.index)

    edges = edges.join(tags_df, on="way_idx")

    existing_columns = [c for c in needed_tags if c in edges.columns]
    if simplify:
        # сшиваем ребра и переносим атрибуты через midpoints -> nearest
        merged_lines = line_merge(MultiLineString(edges.geometry.to_list()), directed=True)
        list_of_merged = list(merged_lines.geoms) if merged_lines.geom_type == "MultiLineString" else [merged_lines]

        lines = gpd.GeoDataFrame(geometry=list_of_merged, crs=local_crs)

        mid = lines.copy()
        mid.geometry = mid.interpolate(lines.length / 2)

        joined = gpd.sjoin_nearest(mid[["geometry"]], edges, how="left", max_distance=1)
        joined = joined.reset_index().drop_duplicates(subset="index").set_index("index")
        lines = lines.join(joined[existing_columns])
        edges = lines

    return edges, local_crs


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
    keep_edge_geometry: bool = True,
) -> nx.MultiDiGraph:
    """
    Build a drivable road network (nx.MultiDiGraph) from OpenStreetMap within a given territory.

    The function downloads OSM ways via Overpass, segments them into directed edges, optionally merges
    contiguous segments, duplicates two-way streets in reverse, and computes per-edge length (meters)
    and travel time (minutes). Node coordinates are unique line endpoints in a local projected CRS.
    Edge attributes can include selected OSM tags and a derived road category/speed.

    Parameters:
        osm_id (int | None): OSM relation/area ID of the territory boundary. Provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry in EPSG:4326 or GeoDataFrame.
            Used when `osm_id` is not given.
        simplify (bool): If True, merges contiguous collinear segments and transfers attributes
            back to merged lines using nearest midpoints. If False, keeps raw per-segment edges.
        add_road_category (bool): If True, adds a derived `category` (e.g., local/regional/federal) and a default
            speed (`maxspeed_mpm`) inferred from the OSM `highway` class.
        clip_by_territory (bool): If True, clips edges by the exact boundary geometry before graph construction.
        keep_largest_subgraph (bool): If True, returns only the largest strongly connected component.
        network_type (Literal["drive","drive_service","custom"]): Preset of Overpass filters to select drivable ways.
            Use "custom" together with `custom_filter` to pass your own Overpass `way` filter.
        custom_filter (str | None): Custom Overpass filter (e.g., `["highway"~"motorway|trunk|…"]`) used when
            `network_type="custom"`.
        osm_edge_tags (list[str] | None): Which OSM tags to retain on edges. Overrides defaults from config.
            The tags `oneway`, `maxspeed`, and `highway` are always added.
        keep_edge_geometry (bool): If True, stores shapely geometries on edges (`geometry` attribute).

    Returns:
        (nx.MultiDiGraph): Directed multigraph of the road network. Each edge carries:
            - `geometry` (if `keep_edge_geometry=True`) in local CRS,
            - `length_meter` (float), `time_min` (float),
            - `type="drive"`,
            - selected OSM tags (incl. `highway`, `maxspeed`, `oneway`, optional `category`, etc.).

            Graph-level attributes: `graph["crs"]` (local projected CRS), `graph["type"]` (network_type).

    Raises:
        ValueError: If `network_type` is unknown, or `network_type="custom"` without `custom_filter`.
    """
    polygon4326 = get_4326_boundary(osm_id=osm_id, territory=territory)

    filters = {
        "drive": Network.DRIVE.filter,
        "drive_service": Network.DRIVE_SERVICE.filter,
        "custom": custom_filter,
    }
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

    logger.info("Downloading drive network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(
        polygon4326, road_filter, needed_tags=tags_to_retrieve, simplify=simplify
    )

    if len(edges) == 0:
        logger.warning("No edges found, returning empty graph")
        return nx.MultiDiGraph()

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True)

    # двусторонние — дублируем с реверсом
    if "oneway" not in edges.columns:
        two_way = edges.copy()
    else:
        two_way = edges[edges["oneway"] != "yes"].copy()
    two_way.geometry = two_way.geometry.reverse()
    edges = pd.concat([edges, two_way], ignore_index=True)

    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]
    edges["u"] = u
    edges["v"] = v

    edges[["category", "maxspeed_mpm"]] = edges["highway"].apply(lambda h: pd.Series(_get_highway_properties(h)))

    if "maxspeed" in edges.columns:
        maxspeed_osm_mpm = (pd.to_numeric(edges["maxspeed"], errors="coerce") * 1000.0 / 60.0).round(3)
        edges["speed_mpm"] = maxspeed_osm_mpm.fillna(edges["maxspeed_mpm"])
    else:
        edges["speed_mpm"] = edges["maxspeed_mpm"]

    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / edges["speed_mpm"]).round(3)
    edges["type"] = "drive"

    graph = nx.MultiDiGraph()

    graph.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(uniques))

    if add_road_category:
        needed_tags |= {"category"}
    if keep_edge_geometry:
        needed_tags |= {"geometry"}
    needed_tags |= {"length_meter", "time_min", "type"}
    edge_attr_cols = list(tag for tag in needed_tags if tag in edges.columns)
    attrs_iter = edges[edge_attr_cols].to_dict("records")
    graph.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if keep_largest_subgraph:
        graph = keep_largest_strongly_connected_component(graph)

    mapping = {old: new for new, old in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["type"] = network_type
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
    keep_edge_geometry: bool = True,
) -> nx.MultiDiGraph:
    """
    Build a pedestrian network (nx.MultiDiGraph) from OpenStreetMap within a given territory.

    The function fetches OSM ways via Overpass using a walking filter, splits each way into directed
    line segments, duplicates all segments in reverse, and computes per-edge length (meters) and traversal time
    (minutes) using a given walking speed (m/min). Node coordinates are unique segment endpoints in a local
    projected CRS. Selected OSM tags can be attached to edges.

    Parameters:
        osm_id (int | None): OSM relation/area ID for the boundary. Provide this or `territory`.
        territory (Polygon | MultiPolygon | gpd.GeoDataFrame | None): Boundary geometry (EPSG:4326) or a GeoDataFrame
            to define the area when `osm_id` is not given.
        simplify (bool): If True, merges contiguous segments (via internal line merging) and transfers attributes
            back to merged lines using nearest midpoints; if False, keeps raw per-segment edges.
        clip_by_territory (bool): If True, clips edges by the provided boundary before graph construction.
        keep_largest_subgraph (bool): If True, retains only the largest strongly connected component.
        walk_speed (float): Walking speed in meters per minute used to compute `time_min` for each edge.
        network_type (Literal["walk","custom"]): Preset of Overpass filters. Use "custom" together with `custom_filter`
            to pass your own way filter.
        custom_filter (str | None): Custom Overpass filter string used when `network_type="custom"`.
        osm_edge_tags (list[str] | None): List of OSM edge tags to retain (overrides defaults). Only these keys
            are joined from element tags.
        keep_edge_geometry (bool): If True, stores shapely `geometry` on edges in the local projected CRS.

    Returns:
        (nx.MultiDiGraph): Directed multigraph of the walking network. Each edge carries:
            - `geometry` (if `keep_edge_geometry=True`), local CRS,
            - `length_meter` (float), `time_min` (float),
            - `type="walk"`,
            - selected OSM tags (as requested).

            Graph attributes include: `graph["crs"]` (local projected CRS),
            `graph["walk_speed"]` (float), and `graph["type"]` (network_type).

    Raises:
        ValueError: If `network_type` is unknown, or `network_type="custom"` without `custom_filter`.

    Notes:
        All walking edges are treated as bidirectional by duplicating geometries in reverse; u/v nodes
        are assigned by factorizing unique segment endpoints. Lengths are measured in meters in a local
        projected CRS estimated from the territory bounds.
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

    logger.info("Downloading walk network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon4326, road_filter, needed_tags=needed_tags, simplify=simplify)

    if len(edges) == 0:
        logger.warning("No edges found, returning empty graph")
        return nx.MultiDiGraph()

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True).explode(ignore_index=True)

    two_way = edges.copy()
    two_way.geometry = two_way.geometry.reverse()
    edges = pd.concat([edges, two_way], ignore_index=True)

    # u/v и узлы
    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]

    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / float(walk_speed)).round(3)
    edges["type"] = "walk"

    # Сборка графа
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(uniques))

    edge_attrs = set(needed_tags)
    if keep_edge_geometry:
        edge_attrs |= {"geometry"}
    edge_attrs |= {"length_meter", "time_min", "type"}
    edge_attrs = [attr for attr in edge_attrs if attr in edges.columns]

    attrs_iter = edges[edge_attrs].to_dict("records")
    graph.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if keep_largest_subgraph:
        graph = keep_largest_strongly_connected_component(graph)

    mapping = {old: new for new, old in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["walk_speed"] = float(walk_speed)
    graph.graph["type"] = network_type
    logger.debug("Walk graph built.")
    return graph
