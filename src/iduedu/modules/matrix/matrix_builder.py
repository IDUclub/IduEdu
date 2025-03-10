from heapq import heappop, heappush
from typing import Literal

import geopandas as gpd
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from scipy.spatial import KDTree

from iduedu import config
from iduedu.modules.matrix.numba_csr_matrix import UI32CSRMatrix

logger = config.logger


@nb.njit(
    [
        nb.types.Array(nb.int32, 2, "C")(
            UI32CSRMatrix.class_type.instance_type,
            nb.types.Array(nb.int32, 1, "C"),
            nb.types.Array(nb.int32, 1, "C"),
            nb.int32,
        )
    ],
    cache=True,
    parallel=True,
)
def dijkstra_numba_parallel(numba_matrix: UI32CSRMatrix, sources: np.array, targets: np.array, cutoff: np.int32):
    distance_matrix = np.full((len(sources), len(targets)), -1, dtype=np.int32)

    for i in nb.prange(len(sources)):
        source = sources[i]
        distances = np.full(numba_matrix.tot_rows, -1, dtype=np.int32)
        seen = np.copy(distances)
        fringe = []
        seen[source] = np.int32(0)
        fringe.append((np.int32(0), np.int32(source)))
        while fringe:
            d, v = heappop(fringe)
            if distances[v] != -1:
                continue
            distances[v] = d
            for u, weight in zip(numba_matrix.get_nnz(v), numba_matrix.get_row(v)):
                uv_dist = np.int32(d + weight)
                if uv_dist > cutoff:
                    continue
                if seen[u] == -1 or uv_dist < seen[u]:
                    seen[u] = uv_dist
                    heappush(fringe, (uv_dist, np.int32(u)))

        distance_matrix[i, :] = distances[targets]
    return distance_matrix


def get_closest_nodes(gdf_from: gpd.GeoDataFrame, to_nx_graph: nx.Graph, return_dist=False) -> list | pd.Series:
    """
    Find the closest nodes in a NetworkX graph for each geometry in a GeoDataFrame.

    For each geometry in the input GeoDataFrame, this function identifies the nearest node in the provided graph
    based on Euclidean distance. The CRS of the GeoDataFrame is automatically aligned with the graph's CRS
    if they differ.

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        A GeoDataFrame containing the points for which the closest graph nodes are to be found.
    to_nx_graph : nx.Graph
        A NetworkX graph with geographic data, where each node has 'x' and 'y' attributes representing
        its coordinates in the graph's CRS.
    return_dist : bool, optional
        If True, returns the distances to the closest nodes along with the node IDs. Defaults to False.

    Returns
    -------
    list | pd.Series
        - If `return_dist=False` (default), returns a list of the closest node IDs for each point in `gdf_from`.
        - If `return_dist=True`, returns a pandas Series where the index is the closest node ID and the values
          are the corresponding distances.

    Examples
    --------
    >>> closest_nodes = get_closest_nodes(points_gdf, graph)
    >>> closest_nodes_with_dist = get_closest_nodes(points_gdf, graph, return_dist=True)
    """
    try:
        graph_crs = CRS.from_epsg(to_nx_graph.graph["crs"])
    except CRSError:
        graph_crs = to_nx_graph.graph["crs"]

    if gdf_from.crs != graph_crs:
        gdf_from = gdf_from.to_crs(graph_crs)

    nodes_with_data = list(to_nx_graph.nodes(data=True))
    coordinates = [(data["x"], data["y"]) for node, data in nodes_with_data]
    tree = KDTree(coordinates)
    target_coord = [(p.x, p.y) for p in gdf_from.representative_point()]
    distances, indices = tree.query(target_coord)
    nearest_nodes = [nodes_with_data[idx][0] for idx in indices]

    if return_dist:
        return pd.Series(distances, index=nearest_nodes)

    return nearest_nodes


def get_adj_matrix_gdf_to_gdf(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    nx_graph: nx.Graph,
    weight: Literal["length_meter", "time_min"] = "length_meter",
    dtype: np.dtype = np.float16,
    threshold: int = None,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Compute an adjacency matrix representing the shortest path distances between two sets of points (GeoDataFrames)
    based on a provided graph. Distances are calculated using the specified edge weight attribute.

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        The GeoDataFrame containing the origin points for the distance matrix calculation.
    gdf_to : gpd.GeoDataFrame
        The GeoDataFrame containing the destination points for the distance matrix calculation.
    nx_graph : nx.Graph
        A NetworkX graph with geographic data where each edge has the specified `weight` attribute (e.g., 'length_meter').
    weight : {"length_meter", "time_min"}, optional
        The edge attribute to use for calculating the shortest paths. Defaults to 'length_meter'.
    dtype : np.dtype, optional
        The data type for the adjacency matrix. Defaults to np.float16, which can be changed to avoid precision loss.
    threshold : int, optional
        The maximum path distance to consider when calculating distances. Paths longer than this value are ignored.
        Default is None, which sets the threshold to the maximum integer value (essentially no threshold).
    max_workers : int, optional
        The maximum number of threads to use during computation. By default, uses all available CPU cores (os.cpu_count()).

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the adjacency matrix where rows correspond to `gdf_from` and columns to `gdf_to`.
        Each value in the matrix represents the shortest path distance between the corresponding points,
        based on the provided graph. If no path exists or the path is beyond the threshold, the entry is set to np.inf.

    Notes
    -----
    - The function computes the closest graph nodes for both `gdf_from` and `gft_to` before calculating the matrix.
    - If `gdf_from` and `gft_to` are equal, the distance matrix will be square.
    - Ensure that all edges in the graph have the specified `weight` attribute; otherwise, the calculation may fail.

    Examples
    --------
    >>> adj_matrix = get_adj_matrix_gdf_to_gdf(origins_gdf, destinations_gdf, graph, weight='time_min', dtype=np.float32)
    """
    try:
        graph_crs = CRS.from_epsg(nx_graph.graph["crs"])
    except CRSError:
        graph_crs = nx_graph.graph["crs"]

    if gdf_from.crs != graph_crs:
        gdf_from = gdf_from.to_crs(graph_crs)

    if gdf_to.crs != graph_crs:
        gdf_to = gdf_to.to_crs(graph_crs)

    logger.debug("Preparing graph sparse matrix")

    transposed = False

    if gdf_from.equals(gdf_to):
        closest_nodes_from = get_closest_nodes(gdf_from, nx_graph)
        closest_nodes_to = closest_nodes_from
        sparse_row_scipy = _get_sparse_row(nx_graph, weight)
    else:
        dif = len(gdf_from) - len(gdf_to)
        if dif <= 0:  # straight
            closest_nodes_from = get_closest_nodes(gdf_from, nx_graph)
            closest_nodes_to = get_closest_nodes(gdf_to, nx_graph)
            sparse_row_scipy = _get_sparse_row(nx_graph, weight)

        else:  # reversed
            transposed = True
            closest_nodes_from = get_closest_nodes(gdf_to, nx_graph)
            closest_nodes_to = get_closest_nodes(gdf_from, nx_graph)
            sparse_row_scipy = _get_sparse_row(nx_graph, weight).transpose().tocsc().tocsr()

    csr_matrix = UI32CSRMatrix(*_get_numba_matrix_attr(sparse_row_scipy))
    del sparse_row_scipy

    if max_workers is not None:
        nb.set_num_threads(max_workers)

    # *** Warm-up ***
    dijkstra_numba_parallel(
        numba_matrix=csr_matrix,
        sources=np.array(closest_nodes_to[:1], dtype=np.int32),
        targets=np.array(closest_nodes_from[:1], dtype=np.int32),
        cutoff=np.int32(0),
    )

    if threshold is None:
        threshold = np.iinfo(np.int32).max
    else:
        threshold = threshold * 100

    logger.debug("Starting the gdf-to-gdf matrix calculation")

    adj_matrix = dijkstra_numba_parallel(
        numba_matrix=csr_matrix,
        sources=np.array(closest_nodes_from, dtype=np.int32),
        targets=np.array(closest_nodes_to, dtype=np.int32),
        cutoff=np.int32(threshold),
    )
    if transposed:
        adj_matrix = adj_matrix.transpose()

    adj_matrix = pd.DataFrame(adj_matrix / 100, columns=gdf_to.index, index=gdf_from.index, dtype=dtype)

    return adj_matrix.where(adj_matrix >= 0, np.inf)


def _get_sparse_row(nx_graph, weight):
    sparse_row_scipy = nx.to_scipy_sparse_array(nx_graph, weight=weight)
    sparse_row_scipy.data = np.round(sparse_row_scipy.data * 100).astype(np.uint32)
    return sparse_row_scipy


def _get_numba_matrix_attr(sparse_row_scipy):
    values = sparse_row_scipy.data.astype(np.uint32)
    col_index = sparse_row_scipy.indices.astype(np.uint32)
    row_index = sparse_row_scipy.indptr.astype(np.uint32)
    return values, col_index, row_index
