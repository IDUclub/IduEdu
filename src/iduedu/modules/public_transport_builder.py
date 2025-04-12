import multiprocessing

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from iduedu import config
from iduedu.enums.pt_enums import PublicTrasport
from iduedu.modules.routes_parser import parse_overpass_to_edgenode
from iduedu.utils.utils import clip_nx_graph, estimate_crs_for_bounds

from .downloaders import (
    get_boundary,
    get_routes_by_poly,
)

logger = config.logger


def _graph_data_to_nx(graph_df, keep_geometry: bool = True) -> nx.DiGraph:
    platforms = graph_df[graph_df["type"] == "platform"]
    platforms = platforms.groupby("point", as_index=False).agg({"node_id": list, "route": list})
    platforms["type"] = "platform"

    stops = graph_df[(graph_df["type"] != "platform") & (graph_df["u"].isna())][["point", "node_id", "route", "type"]]
    stops = stops.groupby(["point", "route", "type"], as_index=False).agg({"node_id": list})
    stops["route"] = stops["route"].apply(lambda x: [x])

    all_nodes = pd.concat([platforms, stops], ignore_index=True).reset_index(drop=True)
    mapping = {}
    for i, row in all_nodes.iterrows():
        index = row.name
        node_id_list = row["node_id"]
        for node_id in node_id_list:
            mapping[node_id] = int(index)

    def replace_with_mapping(value):
        return mapping.get(value)

    # Применение функции замены к каждому столбцу DataFrame
    graph_df["u"] = graph_df["u"].apply(replace_with_mapping)
    graph_df["v"] = graph_df["v"].apply(replace_with_mapping)
    graph_df["node_id"] = graph_df["node_id"].apply(replace_with_mapping)

    edges = graph_df[~graph_df["u"].isna()][["route", "type", "u", "v", "geometry"]].copy()

    def calc_len_time(row):
        if row.type == "boarding":
            return 0, 0
        length = round(row.geometry.length, 3)
        return length, round(length / PublicTrasport[row.type.upper()].avg_speed, 3)

    edges[["length_meter", "time_min"]] = edges.apply(
        calc_len_time,
        axis=1,
        result_type="expand",
    )
    graph = nx.DiGraph()
    for i, node in all_nodes.iterrows():
        route = list(set(node["route"]))
        if len(route) == 1:
            route = route[0]
        graph.add_node(i, x=node["point"][0], y=node["point"][1], type=node["type"], route=route)
    for i, edge in edges.iterrows():
        graph.add_edge(
            edge["u"],
            edge["v"],
            route=edge["route"],
            type=edge["type"],
            geometry=edge["geometry"] if keep_geometry else None,
            length_meter=edge["length_meter"],
            time_min=edge["time_min"],
        )

    return graph


def _get_multi_routes_by_poly(args):
    polygon, transport = args
    return get_routes_by_poly(polygon, transport)


def _multi_process_row(args):
    row, local_crs = args
    return parse_overpass_to_edgenode(row, local_crs)


def get_single_public_transport_graph(
    public_transport_type: str | PublicTrasport,
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds: bool = False,
    keep_geometry: bool = True,
):
    """
    Generate a graph of a specific type of public transport network within a given territory or polygon.
    The graph can optionally be clipped by the specified bounds and retain original geometries.

    Parameters
    ----------
    public_transport_type : str | PublicTransport
        Type of public transport to generate the graph for. Examples include "bus", "tram", "subway", etc.
    osm_id : int, optional
        OpenStreetMap ID of the territory. Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to generate the transport network for. Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon to define the area for the transport network. Must be in CRS 4326.
    clip_by_bounds : bool, optional
        If True, clips the resulting graph to the bounds of the provided polygon. Defaults to False.
    keep_geometry : bool, optional
        If True, retains the original geometry of the transport routes. Defaults to True.

    Returns
    -------
    networkx.Graph
        A public transport network graph for the specified transport type and territory or polygon.

    Raises
    ------
    ValueError
        If no valid `osm_id`, `territory_name`, or `polygon` is provided.

    Warnings
    --------
    Logs a warning if no public transport routes are found in the specified area.

    Examples
    --------
    >>> bus_graph = get_single_public_transport_graph(public_transport_type="bus", osm_id=1114252)
    >>> tram_graph = get_single_public_transport_graph(territory_name="Санкт-Петербург", public_transport_type="tram")
    >>> metro_graph = get_single_public_transport_graph(polygon=poly, public_transport_type="bus", clip_by_bounds=True)

    Notes
    -----
    The function uses OpenStreetMap data and processes public transport routes in parallel.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    polygon = get_boundary(osm_id, territory_name, polygon)

    overpass_data = get_routes_by_poly(polygon, public_transport_type)
    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return nx.Graph()
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    if not config.enable_tqdm_bar:
        logger.debug("Parsing routes")

    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        n_cpus = multiprocessing.cpu_count()
        rows = [(row, local_crs) for _, row in overpass_data.iterrows()]
        edgenode_for_routes = process_map(
            _multi_process_row, rows, desc="Parsing routes", chunksize=1, disable=not config.enable_tqdm_bar
        )
    else:
        tqdm.pandas(desc="Parsing routes data", disable=not config.enable_tqdm_bar)
        edgenode_for_routes = overpass_data.progress_apply(
            lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1
        ).tolist()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)

    to_return = _graph_data_to_nx(graph_df, keep_geometry=keep_geometry)
    to_return.graph["crs"] = local_crs
    to_return.graph["type"] = public_transport_type

    if clip_by_bounds:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        return clip_nx_graph(to_return, polygon)

    logger.debug("Done!")
    return to_return


def get_all_public_transport_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds: bool = False,
    keep_geometry: bool = True,
    transport_types: list[PublicTrasport] = None,
) -> nx.Graph:
    """
    Generate a public transport network graph that includes all types of transport within a specified territory.
    The graph can optionally be clipped by bounds and retain original geometries.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory. Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to generate the public transport network for. Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon to define the area for the transport network. Must be in CRS 4326.
    clip_by_bounds : bool, optional
        If True, clips the resulting graph to the bounds of the provided polygon. Defaults to False.
    keep_geometry : bool, optional
        If True, retains the original geometry of the transport routes. Defaults to True.
    transport_types: list[PublicTransport], optional
        By default `[PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]`,
        can be any combination of PublicTransport Enums.

    Returns
    -------
    networkx.Graph
        A public transport network graph for all transport types (e.g., bus, tram, subway) in the specified territory.

    Raises
    ------
    ValueError
        If no valid `osm_id`, `territory_name`, or `polygon` is provided.

    Warnings
    --------
    Logs a warning if no public transport routes are found in the specified area.

    Examples
    --------
    >>> all_transport_graph = get_all_public_transport_graph(osm_id=1114252)
    >>> all_transport_graph = get_all_public_transport_graph(territory_name="Санкт-Петербург", clip_by_bounds=True)
    >>> all_transport_graph = get_all_public_transport_graph(polygon=some_polygon, keep_geometry=False)

    Notes
    -----
    This function retrieves routes for all public transport types using OpenStreetMap data.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    if transport_types is None:
        transport_types = [PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]
    else:
        for transport_type in transport_types:
            if not isinstance(transport_type, PublicTrasport):
                raise ValueError(f"transport_type {transport_type} is not a valid transport type.")

    polygon = get_boundary(osm_id, territory_name, polygon)

    transports = [transport.value for transport in transport_types]
    args_list = [(polygon, transport) for transport in transports]

    if not config.enable_tqdm_bar:
        logger.debug("Downloading pt routes")

    overpass_data = pd.concat(
        thread_map(
            _get_multi_routes_by_poly,
            args_list,
            desc="Downloading public transport routes from OSM",
            disable=not config.enable_tqdm_bar,
        ),
        ignore_index=True,
    ).reset_index(drop=True)

    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return nx.Graph()

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    if not config.enable_tqdm_bar:
        logger.debug("Parsing public transport routes")
    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        rows = [(row, local_crs) for _, row in overpass_data.iterrows()]
        edgenode_for_routes = process_map(
            _multi_process_row,
            rows,
            desc="Parsing public transport routes",
            chunksize=1,
            disable=not config.enable_tqdm_bar,
        )
    else:
        tqdm.pandas(desc="Parsing public transport routes", disable=not config.enable_tqdm_bar)
        edgenode_for_routes = overpass_data.progress_apply(
            lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1
        ).tolist()

    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = _graph_data_to_nx(graph_df, keep_geometry=keep_geometry)
    to_return.graph["crs"] = local_crs
    to_return.graph["type"] = "public_trasport"

    if clip_by_bounds:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        to_return = clip_nx_graph(to_return, polygon)

    logger.debug("Done!")
    return to_return
