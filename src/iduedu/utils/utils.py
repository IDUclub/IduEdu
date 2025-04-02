import geopandas as gpd
import networkx as nx
import numpy as np
from loguru import logger
from pyproj import CRS
from pyproj.aoi import AreaOfInterest

# pylint: disable=no-name-in-module
from pyproj.database import query_utm_crs_info
from shapely import Point, Polygon


def clip_nx_graph(graph: nx.Graph, polygon: Polygon) -> nx.Graph:
    crs = graph.graph["crs"]
    points = gpd.GeoDataFrame(
        data=[{"id": p_id, "geometry": Point(data["x"], data["y"])} for p_id, data in graph.nodes(data=True)], crs=crs
    ).clip(polygon, True)
    clipped = graph.subgraph(points["id"].tolist())
    return clipped


def remove_weakly_connected_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    graph = graph.copy()

    weakly_connected_components = list(nx.weakly_connected_components(graph))
    if len(weakly_connected_components) > 1:
        logger.warning(
            f"Found {len(weakly_connected_components)} disconnected subgraphs in the network. "
            f"These are isolated groups of nodes with no connections between them. "
            f"Size of components: {[len(c) for c in weakly_connected_components]}"
        )

    all_scc = sorted(nx.strongly_connected_components(graph), key=len)
    nodes_to_del = set().union(*all_scc[:-1])

    if nodes_to_del:
        logger.warning(
            f"Removing {len(nodes_to_del)} nodes that form {len(all_scc) - 1} trap components. "
            f"These are groups where you can enter but can't exit (or vice versa). "
            f"Keeping the largest strongly connected component ({len(all_scc[-1])} nodes)."
        )
        graph.remove_nodes_from(nodes_to_del)

    return graph


def estimate_crs_for_bounds(minx, miny, maxx, maxy):
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
