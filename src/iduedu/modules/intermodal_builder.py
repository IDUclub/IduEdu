import concurrent.futures
import time

import networkx as nx
from shapely import MultiPolygon, Polygon

from iduedu import config
from iduedu.enums.pt_enums import PublicTrasport
from iduedu.modules.downloaders import get_boundary
from iduedu.modules.drive_walk_builder import get_walk_graph
from iduedu.modules.pt_walk_joiner import join_pt_walk_graph
from iduedu.modules.public_transport_builder import get_all_public_transport_graph

logger = config.logger


def get_intermodal_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds: bool = False,
    keep_routes_geom: bool = True,
    max_dist: float = 30,
    transport_types: list[PublicTrasport] = None,
) -> nx.Graph:
    """
    Generate an intermodal transport graph that combines public transport and pedestrian networks,
    with platforms serving as connection points between the two graphs.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory. Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to generate the intermodal transport network for.
        Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon defining the area for the intermodal network. Must be in CRS 4326.
    clip_by_bounds : bool, optional
        If True, clips the public transport network to the bounds of the provided polygon. Defaults to False.
    keep_routes_geom : bool, optional
    max_dist : float, optional
        Maximum distance (in meters) to search for connections between platforms and pedestrian edges. Defaults to 30.
    transport_types: list[PublicTransport], optional
        By default `[PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]`,
        can be any combination of PublicTransport Enums.
    Returns
    -------
    nx.Graph
        An intermodal network graph combining public transport and pedestrian routes, where public transport platforms
        are linked to nearby walking routes.

    Raises
    ------
    ValueError
        If no valid `osm_id`, `territory_name`, or `polygon` is provided.

    Warnings
    --------
    Logs a warning if the public transport graph is empty and only returns the pedestrian graph.

    Examples
    --------
    >>> intermodal_graph = get_intermodal_graph(osm_id=1114252, clip_by_bounds=True)
    >>> intermodal_graph = get_intermodal_graph(territory_name="Санкт-Петербург", polygon=some_polygon)

    Notes
    -----
    The function concurrently downloads and processes both the pedestrian and public transport graphs,
    then combines them using platforms as connection points.
    If the public transport graph is empty, only the pedestrian graph is returned.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    boundary = get_boundary(osm_id, territory_name, polygon)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        walk_graph_future = executor.submit(get_walk_graph, polygon=boundary)
        logger.debug("Started downloading and parsing walk graph...")

        # Sleep to not get 429 to many requests
        time.sleep(0.5)
        pt_graph_future = executor.submit(
            get_all_public_transport_graph,
            polygon=boundary,
            clip_by_bounds=clip_by_bounds,
            keep_geometry=keep_routes_geom,
            transport_types=transport_types,
        )
        logger.debug("Started downloading and parsing public trasport graph...")

        pt_g = pt_graph_future.result()
        logger.debug("Public trasport graph done!")

        walk_g = walk_graph_future.result()
        logger.debug("Walk graph done!")

    if len(pt_g.nodes()) == 0:
        logger.warning("Public trasport graph is empty! Returning only walk graph.")
        return walk_g

    intermodal = join_pt_walk_graph(pt_g, walk_g, max_dist=max_dist)
    return intermodal
