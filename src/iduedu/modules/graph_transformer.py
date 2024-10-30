import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely import LineString
from shapely.geometry.point import Point


def _edges_to_gdf(G: nx.MultiDiGraph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.
    """

    e_ind_source, e_ind_target, e_data = zip(*G.edges(data=True))
    index_matrix = np.array([e_ind_source, e_ind_target]).transpose()
    final_index = [tuple(i) for i in index_matrix]
    lines = (LineString(d['geometry']) for d in e_data)
    gdf_edges = gpd.GeoDataFrame(e_data, index=final_index, crs=32636, geometry=list(lines))

    return gdf_edges

def _nodes_to_gdf(G: nx.MultiDiGraph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as nodes.
    """

    ind, data = zip(*G.nodes(data=True))
    node_geoms = (Point(d["x"], d["y"]) for d in data)
    gdf_nodes = gpd.GeoDataFrame(data, index=ind, crs=crs, geometry=list(node_geoms))

    return gdf_nodes


def graph_to_gdf(
        G: nx.MultiDiGraph,
        crs: int | None = None,
        edges: bool = True,
        nodes: bool = True,
) -> gpd.GeoDataFrame | None:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph to convert.
    crs: int
        Crs with which graph is provided.
    edges: bool, default to True
        Keep edges in GoeDataFrame.
    nodes: bool, default to True
        Keep nodes in GoeDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Graph representation in GeoDataFrame format
    """
    if crs is None:
        try:
            crs = G.graph['crs']
        except:
            raise ValueError("Graph does not have crs attribute and no crs was provided")
    if not edges and not nodes:
        logger.debug("Neither edges or nodes were selected, graph_to_gdf returning None")
        return None
    else:
        if nodes and not edges:
            nodes_gdf = _nodes_to_gdf(G, crs)
            return nodes_gdf
        elif not nodes and edges:
            edges_gdf = _edges_to_gdf(G, crs)
            return  edges_gdf
        else:
            nodes_gdf = _nodes_to_gdf(G, crs)
            edges_gdf = _edges_to_gdf(G, crs)
            full_gdf = pd.concat([nodes_gdf, edges_gdf])
            return  full_gdf