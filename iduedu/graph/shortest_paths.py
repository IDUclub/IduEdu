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

import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
from scipy import sparse
from shapely import reverse as reverse_geometry

from iduedu._numba.csr import coo_rows_to_arrays, sparse_row2numba_matrix
from iduedu._numba.shortest_paths import (
    dijkstra_numba_od_parallel,
    dijkstra_numba_path_length_parallel,
    dijkstra_numba_path_parallel,
    multi_source_dijkstra_numba_nearest_source,
    multi_source_dijkstra_numba_path_length,
    single_source_dijkstra_numba_path,
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
ORIGIN_NODES_ATTR = "origin_nodes"
DESTINATION_NODES_ATTR = "destination_nodes"
PATH_COLUMN = "path"
PATH_ORDER_COLUMN = "path_order"
PATH_U_COLUMN = "path_u"
PATH_V_COLUMN = "path_v"


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
    if (urban_graph.edges_gdf[weight] < 0).any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] must contain only non-negative values")
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


def _path_positions_to_nodes(
    path_positions,
    *,
    pos_to_node: np.ndarray,
    reverse: bool = False,
) -> list[Any] | None:
    if len(path_positions) == 0:
        return None
    positions = np.asarray(path_positions, dtype=np.int64)
    if reverse:
        positions = positions[::-1]
    return list(pos_to_node[positions])


def _sparse_path_series(paths: list[list[Any] | None], *, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(
        pd.arrays.SparseArray(paths, dtype=pd.SparseDtype(object, fill_value=None)),
        index=index,
        name=name,
    )


def single_source_dijkstra_path(
    urban_graph: UrbanGraph,
    source_node: Any,
    target_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Return one shortest path as an ordered list of graph node ids.

    The search stops when ``target_node`` settles. If ``reverse`` is true, the
    transposed directed graph is searched and the returned path is reversed back
    into the original graph direction.

    Args:
        urban_graph: Urban graph with node and edge tables.
        source_node: Search origin node id.
        target_node: Search destination node id.
        weight: Edge weight column to minimize.
        cutoff: Optional maximum path cost.
        reverse: Search the transposed adjacency matrix on directed graphs.

    Returns:
        Ordered node ids for one shortest path. Returns an empty list when the
        destination is unreachable or beyond ``cutoff``.
    """

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_pos, target_pos = _node_positions(urban_graph, [source_node, target_node])
    path_positions = single_source_dijkstra_numba_path(
        numba_adj_matrix,
        np.int32(source_pos),
        np.int32(target_pos),
        _cutoff2float(cutoff),
    )
    path = _path_positions_to_nodes(
        path_positions,
        pos_to_node=_pos_to_node_array(urban_graph),
        reverse=reverse and urban_graph.is_directed,
    )
    return [] if path is None else path


def multi_source_dijkstra_path(
    urban_graph: UrbanGraph,
    *,
    gdf_origins: pd.DataFrame | None = None,
    gdf_destinations: pd.DataFrame | None = None,
    origins_nodes: Iterable[Any] | None = None,
    destination_nodes: Iterable[Any] | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    mode: Literal["all_to_all", "pairwise"] = "all_to_all",
    threshold: float | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Return shortest node paths between origin and destination inputs.

    Inputs follow :func:`od_matrix`: pass node ids or object tables whose
    geometries are matched to graph nodes. ``all_to_all`` returns a sparse path
    matrix, while ``pairwise`` matches origins and destinations by position and
    returns one sparse ``path`` column.
    """

    def _all_to_all_paths_dataframe(
        path_rows,
        *,
        origin_index: pd.Index,
        destination_index: pd.Index,
        pos_to_node: np.ndarray,
        transposed: bool,
    ) -> pd.DataFrame:
        columns = []
        for destination_i, destination_label in enumerate(destination_index):
            if transposed:
                paths = [
                    _path_positions_to_nodes(path_rows[destination_i][origin_i], pos_to_node=pos_to_node, reverse=True)
                    for origin_i in range(len(origin_index))
                ]
            else:
                paths = [
                    _path_positions_to_nodes(path_rows[origin_i][destination_i], pos_to_node=pos_to_node)
                    for origin_i in range(len(origin_index))
                ]
            columns.append(_sparse_path_series(paths, index=origin_index, name=destination_label))

        result = pd.concat(columns, axis=1)
        result.columns = destination_index
        return result

    def _pairwise_paths_dataframe(
        path_rows,
        *,
        origin_index: pd.Index,
        destination_index: pd.Index,
        pos_to_node: np.ndarray,
    ) -> pd.DataFrame:
        pair_index = pd.MultiIndex.from_arrays(
            [origin_index, destination_index],
            names=["origin", "destination"],
        )
        paths = [
            _path_positions_to_nodes(path_rows[pair_i][0], pos_to_node=pos_to_node) for pair_i in range(len(pair_index))
        ]
        return _sparse_path_series(paths, index=pair_index, name=PATH_COLUMN).to_frame()

    _validate_max_workers(max_workers)
    if mode not in {"all_to_all", "pairwise"}:
        raise ValueError(f"mode must be 'all_to_all' or 'pairwise', got {mode!r}")

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

    if mode == "pairwise" and len(origin_nodes_s) != len(destination_nodes_s):
        raise ValueError(
            "pairwise mode requires the same number of origins and destinations, "
            f"got {len(origin_nodes_s)} and {len(destination_nodes_s)}"
        )

    transposed = mode == "all_to_all" and len(destination_nodes_s) < len(origin_nodes_s)
    if transposed:
        calc_origins = destination_nodes_s.to_numpy()
        calc_destinations = origin_nodes_s.to_numpy()
    else:
        calc_origins = origin_nodes_s.to_numpy()
        calc_destinations = destination_nodes_s.to_numpy()

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=threshold, reverse=transposed)
    origin_positions = _node_positions(urban_graph, calc_origins)
    destination_positions = _node_positions(urban_graph, calc_destinations)
    if max_workers is not None:
        nb.set_num_threads(max_workers)

    path_rows = dijkstra_numba_path_parallel(
        numba_adj_matrix,
        origin_positions,
        destination_positions,
        _cutoff2float(threshold),
        mode == "pairwise",
    )
    pos_to_node = _pos_to_node_array(urban_graph)

    if mode == "pairwise":
        result = _pairwise_paths_dataframe(
            path_rows,
            origin_index=origin_nodes_s.index,
            destination_index=destination_nodes_s.index,
            pos_to_node=pos_to_node,
        )
    else:
        result = _all_to_all_paths_dataframe(
            path_rows,
            origin_index=origin_nodes_s.index,
            destination_index=destination_nodes_s.index,
            pos_to_node=pos_to_node,
            transposed=transposed,
        )

    result.attrs[ORIGIN_NODES_ATTR] = origin_nodes_s
    result.attrs[DESTINATION_NODES_ATTR] = destination_nodes_s
    return result


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
    reachable_pairs = multi_source_dijkstra_numba_path_length(numba_adj_matrix, source_positions, _cutoff2float(cutoff))
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
    reachable_columns = pd.Index(_pos_to_node_array(urban_graph)[reachable_col_positions], name=NODE_INDEX_NAME)

    path_matrix = sparse.coo_matrix(
        (values.astype(dtype), (rows, compact_cols)),
        shape=(len(source_nodes_s), len(reachable_columns)),
    ).tocsr()
    dense_result = np.full(path_matrix.shape, np.inf, dtype=dtype)
    if len(values) > 0:
        dense_result[rows, compact_cols] = values.astype(dtype)

    result = pd.DataFrame(dense_result, index=source_nodes_s.index, columns=reachable_columns).astype(
        pd.SparseDtype(dtype, fill_value=np.inf)
    )
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
    od_sparse_matrix = sparse.coo_matrix(
        (values.astype(dtype), (rows, cols)), shape=(len(calc_origins), len(calc_destinations))
    ).tocsr()

    if transposed:
        od_sparse_matrix = od_sparse_matrix.T.tocsr()

    return pd.DataFrame.sparse.from_spmatrix(
        od_sparse_matrix,
        index=origin_nodes_s.index,
        columns=destination_nodes_s.index,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))


def path_to_edges(
    urban_graph: UrbanGraph,
    path: Iterable[Any],
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
) -> gpd.GeoDataFrame:
    """Convert an ordered node path to ordered graph edges.

    For parallel edges the first edge with the minimum selected ``weight`` is
    used, matching the default adjacency-matrix aggregation rule. Edge
    attributes are preserved and reverse traversals receive reversed geometry.

    Args:
        urban_graph: Spatial urban graph containing edge geometries.
        path: Ordered graph node ids, for example from
            :func:`single_source_dijkstra_path` or a path-matrix cell.
        weight: Edge weight used to choose between parallel edges.

    Returns:
        GeoDataFrame of traversed edges with ``path_order``, ``path_u`` and
        ``path_v`` columns.
    """

    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if not isinstance(urban_graph.edges_gdf, gpd.GeoDataFrame):
        raise TypeError("graph.edges_gdf must be GeoDataFrame to build route geometry")
    if weight not in urban_graph.edges_gdf.columns:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")

    path = list(path)
    missing_nodes = [node for node in path if node not in urban_graph.nodes_gdf.index]
    if missing_nodes:
        preview = missing_nodes[:10]
        raise ValueError(
            f"path contains nodes that are absent in graph: {preview}" + (" ..." if len(missing_nodes) > 10 else "")
        )

    empty_result = urban_graph.edges_gdf.iloc[0:0].copy()
    empty_result[PATH_ORDER_COLUMN] = pd.Series(dtype=np.int64)
    empty_result[PATH_U_COLUMN] = pd.Series(dtype=object)
    empty_result[PATH_V_COLUMN] = pd.Series(dtype=object)
    if len(path) < 2:
        return empty_result

    edges = urban_graph.edges_gdf
    if edges[weight].isna().any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] contains NaN")
    if (edges[weight] < 0).any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] must contain only non-negative values")

    selected_edges = []
    for path_order, (path_u, path_v) in enumerate(zip(path[:-1], path[1:])):
        forward_mask = (edges["u"] == path_u) & (edges["v"] == path_v)
        reverse_mask = (edges["u"] == path_v) & (edges["v"] == path_u)

        if urban_graph.edge_direction_column is not None:
            reverse_mask &= ~edges[urban_graph.edge_direction_column].astype(bool)
        elif urban_graph.is_directed:
            reverse_mask &= False

        candidates = edges.loc[forward_mask | reverse_mask]
        if candidates.empty:
            raise ValueError(f"path contains an edge that is absent or disallowed in graph: ({path_u!r}, {path_v!r})")

        candidate_weights = candidates[weight].to_numpy()
        selected_position = int(np.flatnonzero(candidate_weights == candidate_weights.min())[0])
        selected = candidates.iloc[[selected_position]].copy()
        stored_reverse = bool(
            selected.iloc[0]["u"] == path_v and selected.iloc[0]["v"] == path_u and not (path_u == path_v)
        )
        if stored_reverse:
            selected.geometry = selected.geometry.map(reverse_geometry)

        selected[PATH_ORDER_COLUMN] = path_order
        selected[PATH_U_COLUMN] = path_u
        selected[PATH_V_COLUMN] = path_v
        selected_edges.append(selected)

    return gpd.GeoDataFrame(
        pd.concat(selected_edges, axis=0), geometry=urban_graph.edges_gdf.geometry.name, crs=urban_graph.edges_gdf.crs
    )
