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
from iduedu.utils.utils import estimate_crs_for_bounds, keep_largest_strongly_connected_component

logger = config.logger

BASE_FILTER = "['highway'~'" + "|".join([h.value for h in HighwayType]) + "']"


def get_highway_properties(highway) -> tuple[int, float]:
    """
    Determine the reg_status and speed based on highway type.
    :return (reg_status, max_speed).
    """
    default_speed = 40 * 1000 / 60  # default: LOCAL, 40 км/ч

    if not highway:
        return 3, default_speed

    highway_list = highway if isinstance(highway, list) else [highway]

    reg_values = []
    speed_values = []

    for ht in highway_list:
        try:
            enum_value = HighwayType[ht.upper()]
            reg_values.append(enum_value.reg_status)
            speed_values.append(enum_value.max_speed)
        except KeyError:
            continue

    if not reg_values or not speed_values:
        return 3, default_speed

    return min(reg_values), max(speed_values)


def get_drive_graph_by_poly(
    polygon: Polygon | MultiPolygon,
    additional_edgedata=None,
    road_filter: str = None,
    retain_all: bool = False,
    **osmnx_kwargs,
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
    osmnx_kwargs["retain_all"] = retain_all
    graph = ox.graph_from_polygon(polygon, network_type="drive", custom_filter=road_filter, **osmnx_kwargs)
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    nodes, edges = ox.graph_to_gdfs(graph)
    edges: gpd.GeoDataFrame
    edges.reset_index(inplace=True)

    edges[["reg", "maxspeed"]] = edges["highway"].apply(lambda h: pd.Series(get_highway_properties(h)))

    nodes.to_crs(local_crs, inplace=True)
    nodes.geometry = nodes.geometry.set_precision(0.00001)
    nodes[["x", "y"]] = nodes.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1, result_type="expand")
    edges.to_crs(local_crs, inplace=True)
    edges.geometry = edges.geometry.set_precision(0.00001)

    edges[["length_meter", "time_min"]] = edges.apply(
        lambda row: (round(row.geometry.length, 3), round(row.geometry.length / row.maxspeed, 3)),
        axis=1,
        result_type="expand",
    )
    edgesdata = ["u", "v", "key", "length_meter", "time_min", "geometry"] + additional_edgedata

    edges = edges[edgesdata]

    edges.set_index(["u", "v", "key"], inplace=True)
    graph = ox.graph_from_gdfs(nodes, edges)
    if not retain_all:
        graph = keep_largest_strongly_connected_component(graph)
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
    retain_all: bool = False,
    **osmnx_kwargs,
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
    retain_all: bool, optional
        If True, return the entire graph even if it is not connected.
        If False, retain only the largest weakly connected component.
    **osmnx_kwargs
        Additional keyword arguments to pass to osmnx.graph.graph_from_polygon().
        See https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_polygon
    Returns
    -------
    networkx.Graph
        A road network graph for the specified territory or polygon, with optional additional edge data.

    Examples
    --------
    >>> drive_graph = get_drive_graph(osm_id=1114252)
    >>> drive_graph = get_drive_graph(territory_name="Санкт-Петербург", additional_edgedata=['highway', 'maxspeed'])
    >>> drive_graph = get_drive_graph(polygon=some_polygon, additional_edgedata=['name', 'ref'],simplify=False)

    Notes
    -----
    Road speeds are defined in `iduedu.enums.drive_enums.py`.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    polygon = get_boundary(osm_id, territory_name, polygon)

    return get_drive_graph_by_poly(
        polygon, additional_edgedata=additional_edgedata, retain_all=retain_all, **osmnx_kwargs
    )


def get_walk_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    walk_speed: float = 5 * 1000 / 60,
    retain_all: bool = False,
    **osmnx_kwargs,
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
    retain_all: bool, optional
        If True, return the entire graph even if it is not connected.
        If False, retain only the largest weakly connected component.
    **osmnx_kwargs
        Additional keyword arguments to pass to osmnx.graph.graph_from_polygon().
        See https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_polygon

    Returns
    -------
    networkx.Graph
        A pedestrian road network graph with edge lengths and walking times for the specified territory or polygon.

    Examples
    --------
    >>> walk_graph = get_walk_graph(osm_id=1114252)
    >>> walk_graph = get_walk_graph(territory_name="Санкт-Петербург", walk_speed=5)
    >>> walk_graph = get_walk_graph(polygon=some_polygon,simplify=False)

    Notes
    -----
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    polygon = get_boundary(osm_id, territory_name, polygon)

    logger.info("Downloading walk graph from OSM, it may take a while for large territory ...")
    osmnx_kwargs["retain_all"] = retain_all
    if "simplify" not in osmnx_kwargs:
        osmnx_kwargs["simplify"] = True
    print(osmnx_kwargs)
    graph = ox.graph_from_polygon(polygon, network_type="walk", **osmnx_kwargs)
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
    if not retain_all:
        graph = keep_largest_strongly_connected_component(graph)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["walk_speed"] = walk_speed
    graph.graph["type"] = "walk"
    logger.debug("Done!")
    return graph
