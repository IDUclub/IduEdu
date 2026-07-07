"""Shortest-path helpers for :class:`iduedu.graph.urban_graph.UrbanGraph`.

The module provides public wrappers around numba implementations from
``iduedu._numba``. It handles graph validation, input normalization, adjacency
matrix preparation, zero-copy conversion to numba-compatible CSR structures and
conversion of sparse numba results back to pandas objects.

Distances are returned in the original units of the selected edge ``weight``:
minutes for ``time_min`` and meters for ``length_meter``. Unreachable nodes or
pairs are represented as ``np.inf``.
"""

from typing import Any, Iterable, Literal

import numba as nb
import numpy as np
import pandas as pd
from scipy import sparse

from iduedu._numba.csr import coo_rows_to_arrays, sparse_row2numba_matrix
from iduedu._numba.shortest_paths import (
    dijkstra_numba_od_parallel,
    dijkstra_numba_path_length_parallel,
    multi_source_dijkstra_numba_nearest_source,
    multi_source_dijkstra_numba_path_length,
    single_source_dijkstra_numba_path_length,
)
from iduedu.config import config
from iduedu.graph.graph_inputs import resolve_graph_nodes_input
from iduedu.graph.urban_graph import UrbanGraph

logger = config.logger

NODE_INDEX_NAME = "node"
DIST_COLUMN = "dist"
SOURCE_NODE_COLUMN = "source_node"
SOURCE_NODES_ATTR = "source_nodes"


def _cutoff2float(weight_value_cutoff: float | None) -> np.float32:
    """Convert an optional search cutoff to the numba float32 convention."""

    if weight_value_cutoff is None:
        return np.float32(np.inf)
    return np.float32(weight_value_cutoff)


def _validate_max_workers(max_workers: int | None) -> None:
    if max_workers is None:
        return
    if not isinstance(max_workers, int):
        raise TypeError(f"max_workers must be int | None, got {type(max_workers).__name__}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")


def _node_positions(urban_graph: UrbanGraph, nodes: Iterable[Any]) -> np.ndarray:
    node_to_pos = urban_graph.node_to_adjacency_pos
    nodes = list(nodes)
    missing_nodes = [node for node in nodes if node not in node_to_pos]
    if missing_nodes:
        preview = missing_nodes[:10]
        raise ValueError(
            f"source_nodes contain nodes that are absent in graph: {preview}"
            + (" ..." if len(missing_nodes) > 10 else "")
        )
    return np.fromiter((node_to_pos[node] for node in nodes), dtype=np.int32, count=len(nodes))


def _pos_to_node_array(urban_graph: UrbanGraph) -> np.ndarray:
    pos_to_node = np.empty(len(urban_graph.adjacency_nodelist), dtype=object)
    pos_to_node[:] = urban_graph.adjacency_nodelist
    return pos_to_node


def _prepare_numba_graph(
    urban_graph: UrbanGraph,
    *,
    weight: str,
    cutoff: float | None,
    reverse: bool,
):
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if len(urban_graph.nodes_gdf) == 0:
        raise ValueError("graph is empty")
    if urban_graph.nodes_gdf.index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")
    if weight not in urban_graph.edges_gdf.columns:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")
    if cutoff is not None and cutoff < 0:
        raise ValueError(f"weight_value_cutoff must be >= 0, got {cutoff}")

    graph_nodelist = urban_graph.nodes_gdf.index.to_list()
    if (
        urban_graph.adjacency_matrix is None
        or urban_graph.adjacency_weight != weight
        or urban_graph.adjacency_nodelist != graph_nodelist
    ):
        urban_graph.update_adjacency_matrix(nodelist=graph_nodelist, weight=weight)
    if urban_graph.adjacency_matrix.shape[0] == 0:
        raise ValueError("graph adjacency_matrix is empty")

    # The cached adjacency matrix is already a float32 CSR in the requested
    # units, so it is handed to the numba kernel directly (no copy, no weight
    # conversion). Only the reverse case builds a new matrix, since transposing
    # yields a genuinely different structure.
    sparse_row_scipy = urban_graph.adjacency_matrix
    if reverse and urban_graph.is_directed:
        sparse_row_scipy = sparse_row_scipy.transpose().tocsr()
    return sparse_row2numba_matrix(sparse_row_scipy)


def _path_length_series(
    reachable_pairs,
    *,
    pos_to_node: np.ndarray,
    dtype: np.dtype,
) -> pd.Series:
    if len(reachable_pairs) == 0:
        return pd.Series(
            [],
            index=pd.Index([], name=NODE_INDEX_NAME),
            name=DIST_COLUMN,
            dtype=pd.SparseDtype(dtype, fill_value=np.inf),
        )

    reachable_pairs_arr = np.asarray(reachable_pairs, dtype=np.float64)
    node_positions = reachable_pairs_arr[:, 0].astype(np.int64)
    return pd.Series(
        reachable_pairs_arr[:, 1].astype(dtype),
        index=pd.Index(pos_to_node[node_positions], name=NODE_INDEX_NAME),
        name=DIST_COLUMN,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))


def single_source_dijkstra_path_length(
    urban_graph: UrbanGraph,
    source_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.Series:
    """Compute shortest-path distances from one source node to graph nodes.

    Args:
        urban_graph: Urban graph with node and edge tables.
        source_node: Existing node id from ``graph.nodes_gdf.index``.
        weight: Edge weight column, usually ``"time_min"`` or ``"length_meter"``.
        cutoff: Optional maximum path cost. Nodes beyond the cutoff are omitted from
            the sparse result and are interpreted as ``np.inf``.
        reverse: If ``True`` and the graph is directed, run on the reversed
            adjacency matrix. This is useful for coverage queries such as "which
            nodes can reach this destination".
        dtype: Floating dtype for the returned sparse series.

    Returns:
        Sparse ``Series`` indexed by reachable graph node ids with path distances
        from ``source_node``.

    See also:
        https://iduclub.github.io/IduEdu/examples/shortest_paths.html
    """

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_pos = _node_positions(urban_graph, [source_node])[0]
    reachable_pairs = single_source_dijkstra_numba_path_length(
        numba_adj_matrix, np.int32(source_pos), _cutoff2float(cutoff)
    )
    return _path_length_series(reachable_pairs, pos_to_node=_pos_to_node_array(urban_graph), dtype=dtype)


def multi_source_dijkstra_path_length(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.Series:
    """Compute distance from the nearest source to each reachable graph node.

    All sources are inserted into one Dijkstra queue, so each node receives only the
    best distance to the closest source. Use
    :func:`multi_source_dijkstra_nearest_source` when the winning source id is also
    needed.

    Args:
        urban_graph: Urban graph with node and edge tables.
        source_nodes: Source node ids. Pass either this argument or ``gdf_sources``.
        gdf_sources: DataFrame or GeoDataFrame with source objects. If it contains
            ``graph_node_column``, those node ids are used directly; otherwise
            GeoDataFrame geometries are matched to nearest graph nodes.
        graph_node_column: Column containing graph node ids in ``gdf_sources``.
        weight: Edge weight column.
        cutoff: Optional maximum path cost.
        reverse: If ``True`` and the graph is directed, run on the reversed
            adjacency matrix.
        dtype: Floating dtype for the returned sparse series.

    Returns:
        Sparse ``Series`` indexed by reachable graph node ids. The normalized source
        mapping is stored in ``result.attrs["source_nodes"]``.

    See also:
        https://iduclub.github.io/IduEdu/examples/shortest_paths.html
    """

    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, pd.Index(source_nodes_s.to_numpy()).unique())
    reachable_pairs = multi_source_dijkstra_numba_path_length(
        numba_adj_matrix,
        source_positions,
        _cutoff2float(cutoff),
    )
    result = _path_length_series(reachable_pairs, pos_to_node=_pos_to_node_array(urban_graph), dtype=dtype)
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def multi_source_dijkstra_nearest_source(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """Find the nearest source and its distance for each reachable graph node.

    Arguments are the same as for :func:`multi_source_dijkstra_path_length`.

    Returns:
        ``DataFrame`` indexed by reachable graph node ids with ``source_node`` and
        ``dist`` columns. The normalized source mapping is stored in
        ``result.attrs["source_nodes"]``.

    See also:
        https://iduclub.github.io/IduEdu/examples/shortest_paths.html
    """

    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, pd.Index(source_nodes_s.to_numpy()).unique())
    pos_to_node = _pos_to_node_array(urban_graph)
    reachable_triplets = multi_source_dijkstra_numba_nearest_source(
        numba_adj_matrix, source_positions, _cutoff2float(cutoff)
    )

    if len(reachable_triplets) == 0:
        result = pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: pd.Series([], index=pd.Index([], name=NODE_INDEX_NAME), dtype=object),
                DIST_COLUMN: pd.Series([], index=pd.Index([], name=NODE_INDEX_NAME), dtype=dtype),
            }
        )
    else:
        reachable_triplets_arr = np.asarray(reachable_triplets, dtype=np.float64)
        node_positions = reachable_triplets_arr[:, 0].astype(np.int64)
        source_positions_arr = reachable_triplets_arr[:, 1].astype(np.int64)
        reachable_index = pd.Index(pos_to_node[node_positions], name=NODE_INDEX_NAME)
        result = pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: pos_to_node[source_positions_arr],
                DIST_COLUMN: reachable_triplets_arr[:, 2].astype(dtype),
            },
            index=reachable_index,
        )
    result[DIST_COLUMN] = result[DIST_COLUMN].astype(pd.SparseDtype(dtype, fill_value=np.inf))
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def dijkstra_path_length_parallel(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Run independent Dijkstra searches for each source.

    Unlike :func:`multi_source_dijkstra_path_length`, sources are not merged into one
    queue. The result contains one sparse row per source object or source node,
    which makes this helper suitable for per-origin isochrone calculations.

    Args:
        urban_graph: Urban graph with node and edge tables.
        source_nodes: Source node ids. Pass either this argument or ``gdf_sources``.
        gdf_sources: DataFrame or GeoDataFrame with source objects.
        graph_node_column: Column containing graph node ids in ``gdf_sources``.
        weight: Edge weight column.
        cutoff: Optional maximum path cost.
        reverse: If ``True`` and the graph is directed, run on the reversed
            adjacency matrix.
        dtype: Floating dtype for returned sparse values.
        max_workers: Optional number of numba worker threads.

    Returns:
        Sparse ``DataFrame`` whose rows are source objects and whose columns are
        reachable graph node ids. The normalized source mapping is stored in
        ``result.attrs["source_nodes"]``.

    See also:
        https://iduclub.github.io/IduEdu/examples/shortest_paths.html
    """

    _validate_max_workers(max_workers)
    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, source_nodes_s.to_numpy())
    if max_workers is not None:
        nb.set_num_threads(max_workers)

    reachable_rows = dijkstra_numba_path_length_parallel(numba_adj_matrix, source_positions, _cutoff2float(cutoff))
    rows, cols, values = coo_rows_to_arrays(reachable_rows)

    if len(values) > 0:
        reachable_col_positions, compact_cols = np.unique(cols, return_inverse=True)
    else:
        reachable_col_positions = np.array([], dtype=np.int32)
        compact_cols = np.array([], dtype=np.int32)
    reachable_columns = pd.Index(
        _pos_to_node_array(urban_graph)[reachable_col_positions],
        name=NODE_INDEX_NAME,
    )

    path_matrix = sparse.coo_matrix(
        (values.astype(dtype), (rows, compact_cols)),
        shape=(len(source_nodes_s), len(reachable_columns)),
    ).tocsr()
    dense_result = np.full(path_matrix.shape, np.inf, dtype=dtype)
    if len(values) > 0:
        dense_result[rows, compact_cols] = values.astype(dtype)

    result = pd.DataFrame(
        dense_result,
        index=source_nodes_s.index,
        columns=reachable_columns,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def od_matrix(
    urban_graph: UrbanGraph,
    *,
    gdf_origins: pd.DataFrame | None = None,
    gdf_destinations: pd.DataFrame | None = None,
    origins_nodes: Iterable[Any] | None = None,
    destination_nodes: Iterable[Any] | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    dtype: np.dtype = np.float32,
    threshold: float | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Compute an origin-destination shortest-path matrix on an ``UrbanGraph``.

    Origins and destinations can be supplied either as graph node ids or as tables of
    objects. Tables with ``graph_node_column`` use those node ids directly; otherwise
    GeoDataFrame geometries are matched to nearest graph nodes. The helper builds or
    reuses the graph adjacency matrix for the selected ``weight``.

    Args:
        urban_graph: Urban graph with node and edge tables.
        gdf_origins: Table of origin objects.
        gdf_destinations: Table of destination objects.
        origins_nodes: Origin graph node ids.
        destination_nodes: Destination graph node ids.
        graph_node_column: Node id column used in origin and destination tables.
        weight: Edge weight column, usually ``"time_min"`` or ``"length_meter"``.
        dtype: Floating dtype for returned sparse values.
        threshold: Optional maximum path cost. Pairs without a path or beyond the
            threshold are represented as ``np.inf``.
        max_workers: Optional number of numba worker threads.

    Returns:
        Sparse ``DataFrame``. When object tables are passed, rows and columns follow
        their indexes; when node lists are passed, rows and columns follow those node
        lists.

    Raises:
        TypeError: If the graph type or ``max_workers`` is invalid.
        ValueError: If inputs are empty, the threshold is negative or requested nodes
            are absent from the graph.

    See also:
        https://iduclub.github.io/IduEdu/examples/shortest_paths.html
    """

    _validate_max_workers(max_workers)

    if (gdf_origins is not None and graph_node_column not in gdf_origins.columns) or (
        gdf_destinations is not None and graph_node_column not in gdf_destinations.columns
    ):
        logger.info(
            "OD matrix is calculated between nearest graph nodes. "
            "For more precise object-to-object distances, project objects into UrbanGraph first."
        )

    origin_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=origins_nodes,
        gdf=gdf_origins,
        graph_node_column=graph_node_column,
        nodes_name="origins_nodes",
        gdf_name="gdf_origins",
    )
    destination_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=destination_nodes,
        gdf=gdf_destinations,
        graph_node_column=graph_node_column,
        nodes_name="destination_nodes",
        gdf_name="gdf_destinations",
    )
    origins_nodes = origin_nodes_s.to_list()
    destination_nodes = destination_nodes_s.to_list()

    transposed = len(destination_nodes) < len(origins_nodes)
    if transposed:
        calc_origins = destination_nodes
        calc_destinations = origins_nodes
    else:
        calc_origins = origins_nodes
        calc_destinations = destination_nodes

    csr_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=threshold, reverse=transposed)

    if max_workers is not None:
        nb.set_num_threads(max_workers)

    origins_pos = _node_positions(urban_graph, calc_origins)
    destinations_pos = _node_positions(urban_graph, calc_destinations)

    dijkstra_numba_od_parallel(
        numba_adj_matrix=csr_adj_matrix,
        origins=origins_pos[:1],
        destinations=destinations_pos[:1],
        cutoff=np.float32(0.0),
    )

    coo_rows = dijkstra_numba_od_parallel(
        numba_adj_matrix=csr_adj_matrix,
        origins=origins_pos,
        destinations=destinations_pos,
        cutoff=_cutoff2float(threshold),
    )

    rows, cols, values = coo_rows_to_arrays(coo_rows)
    od_matrix = sparse.coo_matrix(
        (values.astype(dtype), (rows, cols)), shape=(len(calc_origins), len(calc_destinations))
    ).tocsr()

    if transposed:
        od_matrix = od_matrix.T.tocsr()

    return pd.DataFrame.sparse.from_spmatrix(
        od_matrix,
        index=origin_nodes_s.index,
        columns=destination_nodes_s.index,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))
