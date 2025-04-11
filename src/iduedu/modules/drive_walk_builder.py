import re

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import MultiPolygon, Polygon, unary_union
from tqdm.auto import tqdm

from iduedu import config
from iduedu.enums.drive_enums import HighwayType
from iduedu.modules.downloaders import get_boundary
from iduedu.utils.utils import estimate_crs_for_bounds, remove_weakly_connected_nodes

logger = config.logger

BASE_FILTER = "['highway'~'" + "|".join([h.value for h in HighwayType]) + "']"


def highway_type_to_reg(highway_type) -> int:
    """
    Determine the reg_status based on highway type.
    """
    try:
        if isinstance(highway_type, list):
            reg_values = [HighwayType[ht.upper()].reg_status for ht in highway_type]
            return min(reg_values)
        return HighwayType[highway_type.upper()].reg_status
    except KeyError:
        return 3


def determine_reg(name_roads, highway_type=None) -> int:
    """
    Determine the reg_status based on road_name.
    """

    if isinstance(name_roads, list):
        for item in name_roads:
            if re.match(r"^[МАР]", str(item)):
                return 1
            if re.match(r"^\d.*[A-Za-zА-Яа-я]", str(item)):
                return 2
        return 3
    if pd.isna(name_roads):
        # Выставление значения по типу дороги, если значение NaN
        if highway_type:
            return highway_type_to_reg(highway_type)
        return 3
    if re.match(r"^[МАР]", str(name_roads)):
        return 1
    if re.match(r"^\d.*[A-Za-zА-Яа-я]", str(name_roads)):
        return 2
    return 3


def get_max_speed(highway_types) -> float:
    """
    Determine the speed based on road_name.
    """
    # Проверяем, является ли highway_types списком.
    try:
        if isinstance(highway_types, list):
            max_speeds = []
            for ht in highway_types:
                try:
                    highway_enum = HighwayType[ht.upper()]
                    max_speeds.append(highway_enum.max_speed)
                except KeyError:
                    logger.debug(f"{ht} not found in HighwayType enum, skipping.")
            if max_speeds:
                return max(max_speeds)
            logger.debug("No valid highway types provided, returning 40 km/h.")
            return 40 * 1000 / 60
        return HighwayType[highway_types.upper()].max_speed
    except KeyError:
        return 40 * 1000 / 60


def get_drive_graph_by_poly(
    polygon: Polygon | MultiPolygon, additional_edgedata=None, road_filter: str = None
) -> nx.MultiDiGraph:
    if additional_edgedata is None:
        additional_edgedata = []
    if not road_filter:
        road_filter = BASE_FILTER
    if isinstance(polygon, MultiPolygon):
        polygon = unary_union(polygon)
        if isinstance(polygon, MultiPolygon):
            polygon = polygon.convex_hull
    logger.info("Downloading drive graph from OSM, it may take a while for large territory ...")
    graph = ox.graph_from_polygon(
        polygon,
        network_type="drive",
        custom_filter=road_filter,
        truncate_by_edge=False,
    )
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    nodes, edges = ox.graph_to_gdfs(graph)
    edges: gpd.GeoDataFrame
    edges.reset_index(inplace=True)
    if "ref" not in edges.columns:
        edges["ref"] = pd.NA
    edges["reg"] = edges.apply(lambda row: determine_reg(row["ref"], row["highway"]), axis=1)

    nodes.to_crs(local_crs, inplace=True)
    nodes.geometry = nodes.geometry.set_precision(0.00001)
    nodes[["x", "y"]] = nodes.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1, result_type="expand")
    edges.to_crs(local_crs, inplace=True)
    edges.geometry = edges.geometry.set_precision(0.00001)
    edges["maxspeed"] = edges["highway"].apply(get_max_speed)

    edges[["length_meter", "time_min"]] = edges.apply(
        lambda row: (round(row.geometry.length, 3), round(row.geometry.length / row.maxspeed, 3)),
        axis=1,
        result_type="expand",
    )
    edgesdata = ["u", "v", "key", "length_meter", "time_min", "geometry"] + additional_edgedata

    edges = edges[edgesdata]

    edges.set_index(["u", "v", "key"], inplace=True)
    graph = ox.graph_from_gdfs(nodes, edges)
    graph = remove_weakly_connected_nodes(graph)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["type"] = "drive"
    logger.debug("Done!")
    return graph


def get_drive_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    additional_edgedata=None,
):
    """
    Generate a road network graph for driving within the specified territory or polygon.
    Optionally, include additional edge data such as highway type, max speed, registration status and road name.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory to build the graph for. Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to build the graph for. Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon to define the area for the road network. Must be in CRS 4326.
    additional_edgedata : list[str], optional
        List of additional edge data attributes to include in the graph. Possible values include
        ['highway', 'maxspeed', 'reg', 'ref', 'name'] or any other, that exist in OSM. Defaults to None.

    Returns
    -------
    networkx.Graph
        A road network graph for the specified territory or polygon, with optional additional edge data.

    Examples
    --------
    >>> drive_graph = get_drive_graph(osm_id=1114252)
    >>> drive_graph = get_drive_graph(territory_name="Санкт-Петербург", additional_edgedata=['highway', 'maxspeed'])
    >>> drive_graph = get_drive_graph(polygon=some_polygon, additional_edgedata=['name', 'ref'])

    Notes
    -----
    Road speeds are defined in `iduedu.enums.drive_enums.py`.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    polygon = get_boundary(osm_id, territory_name, polygon)

    return get_drive_graph_by_poly(polygon, additional_edgedata=additional_edgedata)


def get_walk_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    walk_speed: float = 5 * 1000 / 60,
):
    """
    Generate a pedestrian road network graph within the specified territory or polygon.
    The graph's edges includes calculated walking times based on the specified walking speed.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory to build the walking graph for.
        Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to build the walking graph for. Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon to define the area for the pedestrian network. Must be in CRS 4326.
    walk_speed : float, optional
        Walking speed in meters per minute. Defaults to 5 km/h (approximately 83.33 meters per minute).

    Returns
    -------
    networkx.Graph
        A pedestrian road network graph with edge lengths and walking times for the specified territory or polygon.

    Examples
    --------
    >>> walk_graph = get_walk_graph(osm_id=1114252)
    >>> walk_graph = get_walk_graph(territory_name="Санкт-Петербург", walk_speed=5)
    >>> walk_graph = get_walk_graph(polygon=some_polygon)

    Notes
    -----
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    polygon = get_boundary(osm_id, territory_name, polygon)

    logger.info("Downloading walk graph from OSM, it may take a while for large territory ...")
    graph = ox.graph_from_polygon(polygon, network_type="walk", truncate_by_edge=False, simplify=True)
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    nodes, edges = ox.graph_to_gdfs(graph)
    nodes.to_crs(local_crs, inplace=True)
    nodes.geometry = nodes.geometry.set_precision(0.00001)
    nodes[["x", "y"]] = nodes.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1, result_type="expand")
    nodes = nodes[["x", "y", "geometry"]]
    edges.reset_index(inplace=True)
    edges.to_crs(local_crs, inplace=True)
    edges.geometry = edges.geometry.set_precision(0.00001)
    tqdm.pandas(desc="Calculating the weights of the walk graph", disable=not config.enable_tqdm_bar)
    edges[["length_meter", "time_min"]] = edges.progress_apply(
        lambda row: (round(row.geometry.length, 3), round(row.geometry.length / walk_speed, 3)),
        axis=1,
        result_type="expand",
    )
    edges["type"] = "walk"
    edges = edges[
        [
            "u",
            "v",
            "key",
            "length_meter",
            "time_min",
            "type",
            "geometry",
        ]
    ]
    edges.set_index(["u", "v", "key"], inplace=True)
    graph = ox.graph_from_gdfs(nodes, edges)
    graph = remove_weakly_connected_nodes(graph)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["walk_speed"] = walk_speed
    graph.graph["type"] = "walk"
    logger.debug("Done!")
    return graph
