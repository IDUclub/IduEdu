import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from shapely import LineString
from shapely.geometry.point import Point


def _edges_to_gdf(graph: nx.Graph, crs) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.
    """
    e_ind_source, e_ind_target, e_data = zip(*graph.edges(data=True))
    index_matrix = np.array([e_ind_source, e_ind_target]).transpose()
    final_index = [tuple(i) for i in index_matrix]
    lines = (LineString(d["geometry"]) if not isinstance(d["geometry"], float) else None for d in e_data)
    gdf_edges = gpd.GeoDataFrame(e_data, index=final_index, crs=crs, geometry=list(lines))

    return gdf_edges


def _nodes_to_gdf(graph: nx.Graph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as nodes.
    """

    ind, data = zip(*graph.nodes(data=True))
    node_geoms = (Point(d["x"], d["y"]) for d in data)
    gdf_nodes = gpd.GeoDataFrame(data, index=ind, crs=crs, geometry=list(node_geoms))

    return gdf_nodes


def graph_to_gdf(
    graph: nx.MultiDiGraph,
    edges: bool = True,
    nodes: bool = True,
) -> gpd.GeoDataFrame | None:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to convert.
    edges: bool, default to True
        Keep edges in GoeDataFrame.
    nodes: bool, default to True
        Keep nodes in GoeDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Graph representation in GeoDataFrame format
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
        return edges_gdf

    nodes_gdf = _nodes_to_gdf(graph, crs)
    edges_gdf = _edges_to_gdf(graph, crs)
    full_gdf = pd.concat([nodes_gdf, edges_gdf])
    return full_gdf
