from heapq import heappop, heappush
from typing import Any, Iterable, Literal

import geopandas as gpd
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from scipy.spatial import KDTree

from iduedu import config
from iduedu.modules.graph_transformers import keep_largest_strongly_connected_component
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
def dijkstra_numba_parallel(
    numba_matrix: UI32CSRMatrix, sources: np.array, targets: np.array, cutoff: np.int32
):  # pragma: no cover
    """
    Parallel multi-source Dijkstra on a CSR matrix (Numba-accelerated).

    Runs Dijkstra from each `source` over a directed graph encoded as a uint32-weight CSR
    (`UI32CSRMatrix`). Distances are accumulated as **int32** in the same units as the matrix
    edge weights (this implementation expects weights pre-scaled by ×100; see helpers below).
    For each source, distances to `targets` are returned; unreachable entries are `-1`.

    Parameters:
        numba_matrix (UI32CSRMatrix): Directed graph in CSR form with non-negative integer weights.
            Attributes used: `.tot_rows`, `.get_row(i)` (weights), `.get_nnz(i)` (neighbor indices).
        sources (np.ndarray[int32]): Node indices to start from.
        targets (np.ndarray[int32]): Node indices to read distances for.
        cutoff (np.int32): Maximum distance; paths exceeding this are ignored.

    Returns:
        (np.ndarray[int32]): Matrix of shape (len(sources), len(targets)); `-1` marks “no path”.
    """
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


def get_closest_nodes(gdf_from: gpd.GeoDataFrame, to_nx_graph: nx.Graph) -> tuple[list[Any], float | Iterable | int]:
    """
    Find the nearest graph node for each geometry in a GeoDataFrame.

    Reprojects `gdf_from` to the graph CRS if needed, then builds a KD-tree over node
    coordinates (`x`, `y`) and queries nearest nodes for point representatives of the
    input geometries (handles points/lines/polygons via `representative_point()`).

    Parameters:
        gdf_from (gpd.GeoDataFrame): Input geometries to snap (any geometry type).
        to_nx_graph (nx.Graph): Graph with node attributes `x`, `y` and `graph["crs"]` set.

    Returns:
        (list, np.ndarray): Tuple of:
            - `nearest_nodes`: list of node IDs (in the graph index space),
            - `distances`: NumPy array of Euclidean distances in the graph CRS units.
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

    return nearest_nodes, distances


def get_adj_matrix_gdf_to_gdf(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    nx_graph: nx.Graph,
    weight: Literal["length_meter", "time_min"] = "length_meter",
    dtype: np.dtype = np.float16,
    add_dist_tofrom_node=True,
    threshold: int = None,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Compute a shortest-path matrix between two GeoDataFrames over a weighted NetworkX graph.

    Each origin (row of `gdf_from`) and destination (column of `gdf_to`) is snapped to its nearest
    graph node (KD-tree), then distances are computed with a batched, Numba-parallel Dijkstra on
    the graph’s CSR matrix. Result units follow `weight`:
    - `"length_meter"` → meters,
    - `"time_min"` → minutes.

    Parameters:
        gdf_from (gpd.GeoDataFrame): Origin geometries (any types; snapped via representative points).
        gdf_to (gpd.GeoDataFrame): Destination geometries.
        nx_graph (nx.Graph): Graph with `graph["crs"]` and per-edge `weight` attribute present.
        weight ({"length_meter", "time_min"}): Edge attribute to minimize.
        dtype (np.dtype): Output matrix dtype (default `np.float16` to save memory).
        add_dist_tofrom_node (bool): If True, adds straight-line distances from origin geometry → its snap node
            and from snap node → destination geometry:
            - if `weight="length_meter"`: meters are added,
            - if `weight="time_min"`: meters converted to minutes assuming 5 km/h (~83.33 m/min).
        threshold (int | None): Optional max path threshold in `weight` units; longer paths are treated as missing.
            Internally quantized by ×100 for integer CSR; `None` ⇒ no cutoff.
        max_workers (int | None): If set, limits Numba thread count (`nb.set_num_threads(max_workers)`).

    Returns:
        (pd.DataFrame): Matrix with index = `gdf_from.index`, columns = `gdf_to.index`.
            Values are shortest-path distances; unreachable pairs are `np.inf`.

    Raises:
        ValueError: If the graph lacks `graph["crs"]`.
        CRSError: If the graph CRS is invalid or reprojection fails.

    Notes:
        - The graph is first pruned to its largest strongly connected component to avoid spurious `∞`.
        - For performance, if `len(gdf_from) > len(gdf_to)`, the computation is performed transposed and flipped back.
        - Internally, edge weights are converted to **uint32** by multiplying by 100; results are divided by 100
          before returning.
    """
    try:
        local_crs = nx_graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have crs attribute") from exc

    try:
        gdf_from = gdf_from.to_crs(nx_graph.graph["crs"])
        gdf_to = gdf_to.to_crs(nx_graph.graph["crs"])
    except CRSError as e:
        raise CRSError(f"Graph crs ({local_crs}) has invalid format.") from e

    nx_graph = keep_largest_strongly_connected_component(nx_graph)
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    logger.debug("Preparing graph sparse matrix")
    transposed = False

    if gdf_from.equals(gdf_to):
        closest_nodes_from, dist_from = get_closest_nodes(gdf_from, nx_graph)
        closest_nodes_to, dist_to = closest_nodes_from, dist_from
        sparse_row_scipy = _get_sparse_row(nx_graph, weight)
    else:
        dif = len(gdf_from) - len(gdf_to)
        if dif <= 0:  # straight
            closest_nodes_from, dist_from = get_closest_nodes(gdf_from, nx_graph)
            closest_nodes_to, dist_to = get_closest_nodes(gdf_to, nx_graph)
            sparse_row_scipy = _get_sparse_row(nx_graph, weight)

        else:  # reversed
            transposed = True
            closest_nodes_from, dist_from = get_closest_nodes(gdf_to, nx_graph)
            closest_nodes_to, dist_to = get_closest_nodes(gdf_from, nx_graph)
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
    if add_dist_tofrom_node:
        if weight == "time_min":
            speed = 5 * 1000 / 60
            dist_to = dist_to / speed
            dist_from = dist_from / speed
        dist_from_matrix = np.array(dist_from)[:, np.newaxis]
        dist_to_matrix = np.array(dist_to)[np.newaxis, :]
        mask = adj_matrix > 0
        additional_matrix = (dist_from_matrix + dist_to_matrix).astype(dtype)
        if transposed:
            additional_matrix = additional_matrix.transpose()
        adj_matrix[mask] += additional_matrix
    return adj_matrix.where(adj_matrix >= 0, np.inf)


def _get_sparse_row(nx_graph, weight):
    """
    Convert a (possibly multi-)graph to a SciPy CSR matrix of edge weights (uint32, ×100 scaling).

    If the graph is a Multi(Graph|DiGraph), it is first coerced to a simple Graph/DiGraph
    (parallel edges are **collapsed and their attributes lost**, keeping one arbitrary edge).
    Then `nx.to_scipy_sparse_array(..., weight=weight)` produces a CSR array; its data are
    multiplied by 100 and cast to `uint32` for integer-only NumPy/Numba arithmetic.

    Parameters:
        nx_graph (nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph): Input network.
        weight (str): Name of the edge attribute to use as weight (must be numeric and non-negative).

    Returns:
        (scipy.sparse.csr_array): CSR matrix with `uint32` data scaled by ×100.

    Warnings:
        Collapsing a multigraph to a simple graph discards multi-edge multiplicity and may change
        effective weights if parallel edges existed.
    """
    if nx_graph.is_multigraph():
        if nx_graph.is_directed():
            nx_graph = nx.DiGraph(nx_graph)
        else:
            nx_graph = nx.Graph(nx_graph)
    sparse_row_scipy = nx.to_scipy_sparse_array(nx_graph, weight=weight)
    sparse_row_scipy.data = np.round(sparse_row_scipy.data * 100).astype(np.uint32)
    return sparse_row_scipy


def _get_numba_matrix_attr(sparse_row_scipy):
    values = sparse_row_scipy.data.astype(np.uint32)
    col_index = sparse_row_scipy.indices.astype(np.uint32)
    row_index = sparse_row_scipy.indptr.astype(np.uint32)
    return values, col_index, row_index
