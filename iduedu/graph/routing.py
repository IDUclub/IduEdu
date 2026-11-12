"""Shared Dijkstra workflows for distances, routes and origin-destination tables."""

from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from iduedu import _rust
from iduedu.config import config
from iduedu.graph.graph_inputs import resolve_graph_nodes_input
from iduedu.graph.routing_result import (
    ShortestPathResult,
    _groups,
    _PathStore,
    _PathTree,
    _RouteList,
    _sparse_frame,
)
from iduedu.graph.urban_graph import UrbanGraph

Output = Literal["distances", "paths", "both"]
PathStorage = Literal["eager", "lazy"]
PairMethod = Literal["bidirectional", "astar", "alt", "dijkstra"]
RoutingTable = pd.Series | pd.DataFrame
RoutingOutput = RoutingTable | ShortestPathResult


def _validate_options(output, path_storage, max_workers=None, batch_size=None):
    if output not in {"distances", "paths", "both"}:
        raise ValueError("output must be 'distances', 'paths' or 'both'")
    if path_storage not in {"eager", "lazy"}:
        raise ValueError("path_storage must be 'eager' or 'lazy'")
    if path_storage == "lazy" and output != "both":
        raise ValueError("path_storage='lazy' requires output='both'")
    for name, value in (("max_workers", max_workers), ("batch_size", batch_size)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be int | None, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


def _resolve_cutoff(cutoff, threshold=None):
    if cutoff is not None and threshold is not None:
        raise ValueError("Pass either cutoff or threshold, not both")
    value = threshold if threshold is not None else cutoff
    if value is not None and (np.isnan(value) or value < 0):
        raise ValueError("cutoff must be >= 0 and must not be NaN")
    return value


def _cutoff_float(cutoff):
    return np.float32(np.inf if cutoff is None else cutoff)


# UrbanGraph._csr_cache: [adjacency matrix, CsrGraph, transposed CsrGraph | None, node id array | None,
# (nodes index, edges table, weight) last validated for this matrix | None,
# (nodes table, its geometry array, x, y) of the node coordinates in matrix order | None,
# {landmark count: LandmarkTable} for ALT searches | None,
# copies of the edge columns the matrix was built from (see _matrix_source) | None]
_MATRIX, _FORWARD, _REVERSE, _NODE_IDS, _CHECKED, _COORDS, _LANDMARKS, _SOURCE = range(8)
PAIR_METHODS = ("bidirectional", "astar", "alt", "dijkstra")
#: Landmarks of method="alt": more make the estimate sharper and the tables larger (8 bytes per node
#: and landmark on a directed graph, 4 on an undirected one).
ALT_LANDMARKS = 16


def _cache(urban_graph):
    """Rust view of the cached adjacency matrix, rebuilt only when the matrix object changes."""
    matrix = urban_graph.adjacency_matrix
    cache = urban_graph._csr_cache  # pylint: disable=protected-access
    if cache is None or cache[_MATRIX] is not matrix:
        forward = _rust.CsrGraph(
            np.asarray(matrix.indptr, dtype=np.int32),
            np.asarray(matrix.indices, dtype=np.int32),
            np.asarray(matrix.data, dtype=np.float32),
        )
        cache = urban_graph._csr_cache = [
            matrix,
            forward,
            None,
            None,
            None,
            None,
            None,
            None,
        ]  # pylint: disable=protected-access
    return cache


def _csr_graph(urban_graph, *, reverse):
    """The cached Rust graph, or its transpose for reverse searches of a directed graph."""
    cache = _cache(urban_graph)
    if not (reverse and urban_graph.is_directed):
        return cache[_FORWARD]
    if cache[_REVERSE] is None:
        cache[_REVERSE] = cache[_FORWARD].transpose()
    return cache[_REVERSE]


def _array_id(values):
    """Identity of the memory behind a column: changes when the column is replaced, not when edited in place."""
    interface = values.__array_interface__
    return interface["data"][0], interface["shape"], interface["typestr"]


def _matrix_source(urban_graph, weight):
    """Everything the adjacency matrix of ``weight`` is built from.

    The direction settings, copies of the ``u``, ``v``, weight and direction columns, and the identity of the
    memory behind ``u`` and ``v``.
    """
    edges = urban_graph.edges_gdf
    column = urban_graph.edge_direction_column if urban_graph.is_directed else None
    names = ["u", "v", weight] + ([column] if column is not None else [])
    arrays = tuple(edges[name].to_numpy() for name in names)
    copies = tuple(np.array(values, copy=True) for values in arrays)
    return weight, urban_graph.is_directed, column, copies, (_array_id(arrays[0]), _array_id(arrays[1]))


def _equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "biufc":
        return bool(np.array_equal(a, b))
    return pd.Series(a, copy=False).equals(pd.Series(b, copy=False))


def _same_source(source, urban_graph, weight, *, full):
    """Whether the edge columns still hold what the cached matrix was built from.

    The weight and direction columns are compared value by value, so edits in place (``edges.loc[...] = ...``)
    are found too; this costs about 0.4 ms per million edges. ``u`` and ``v`` are compared by value only with
    ``full``; otherwise a check that their memory was not replaced is enough, because the ends of the edges are
    changed by assigning new columns or a new table.
    """
    if source is None:
        return False
    source_weight, directed, column, copies, ends = source
    current_column = urban_graph.edge_direction_column if urban_graph.is_directed else None
    if (source_weight, directed, column) != (weight, urban_graph.is_directed, current_column):
        return False
    edges = urban_graph.edges_gdf
    names = ["u", "v", weight] + ([column] if column is not None else [])
    if any(name not in edges for name in names):
        return False
    current = [edges[name].to_numpy() for name in names]
    if not full and (_array_id(current[0]), _array_id(current[1])) != ends:
        return False
    checked = range(len(names)) if full else range(2, len(names))
    return all(_equal(current[i], copies[i]) for i in checked)


def _already_validated(urban_graph, weight):
    cache = urban_graph._csr_cache  # pylint: disable=protected-access
    if cache is None or cache[_CHECKED] is None or urban_graph.adjacency_matrix is None:
        return False
    nodes_index, edges, checked_weight = cache[_CHECKED]
    return (
        cache[_MATRIX] is urban_graph.adjacency_matrix
        and urban_graph.adjacency_weight == weight == checked_weight
        and nodes_index is urban_graph.nodes_gdf.index
        and edges is urban_graph.edges_gdf
        and _same_source(cache[_SOURCE], urban_graph, weight, full=False)
    )


def _prepare_graph(urban_graph, *, weight, cutoff, reverse):
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    _resolve_cutoff(cutoff)
    if _already_validated(urban_graph, weight):
        # Same node and edge tables as the last validated call: skip the O(n + m) checks.
        return _csr_graph(urban_graph, reverse=reverse)
    nodes_index, edges = urban_graph.nodes_gdf.index, urban_graph.edges_gdf
    if len(urban_graph.nodes_gdf) == 0:
        raise ValueError("graph is empty")
    if nodes_index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")
    if weight not in edges:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")
    weights = edges[weight]
    if weights.isna().any() or (weights < 0).any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] must contain non-negative values without NaN")
    nodelist = nodes_index.to_list()
    cache = urban_graph._csr_cache  # pylint: disable=protected-access
    # A matrix built outside this function (update_adjacency_matrix, an archive) has no recorded source and is
    # trusted; a recorded source that no longer matches the edge table means the matrix is stale.
    source = cache[_SOURCE] if cache is not None and cache[_MATRIX] is urban_graph.adjacency_matrix else None
    if (
        urban_graph.adjacency_matrix is None
        or urban_graph.adjacency_weight != weight
        or urban_graph.adjacency_nodelist != nodelist
        or (source is not None and not _same_source(source, urban_graph, weight, full=True))
    ):
        urban_graph.update_adjacency_matrix(nodelist=nodelist, weight=weight)
    cache = _cache(urban_graph)
    cache[_CHECKED] = (nodes_index, edges, weight)
    cache[_SOURCE] = _matrix_source(urban_graph, weight)
    return _csr_graph(urban_graph, reverse=reverse)


def _node_coordinates(graph):
    """Planar x and y of the nodes in matrix order (cached with the matrix while the nodes table stays)."""
    cache = _cache(graph)
    nodes = graph.nodes_gdf
    stored = cache[_COORDS]
    if stored is not None and stored[0] is nodes and stored[1] is nodes.geometry.array:
        return stored[2], stored[3]
    geometry = nodes.geometry.reindex(graph.adjacency_nodelist)
    if geometry.isna().any() or geometry.is_empty.any() or (geometry.geom_type != "Point").any():
        raise ValueError("method='astar' needs a point geometry for every node")
    xs = np.ascontiguousarray(geometry.x.to_numpy(dtype=np.float64))
    ys = np.ascontiguousarray(geometry.y.to_numpy(dtype=np.float64))
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError("method='astar' needs finite node coordinates")
    cache[_COORDS] = (nodes, nodes.geometry.array, xs, ys)
    return xs, ys


def _landmarks(graph, max_workers=None):
    """Landmark tables of the current matrix for ALT searches, built on first use and cached with it."""
    cache = _cache(graph)
    if cache[_LANDMARKS] is None:
        cache[_LANDMARKS] = {}
    table = cache[_LANDMARKS].get(ALT_LANDMARKS)
    if table is None:
        table = cache[_FORWARD].landmarks(_csr_graph(graph, reverse=True), ALT_LANDMARKS, max_workers)
        cache[_LANDMARKS][ALT_LANDMARKS] = table
    return table


def _check_pair_method(method):
    if method not in PAIR_METHODS:
        raise ValueError(f"method must be one of {PAIR_METHODS}, got {method!r}")


def _kernel_pairs(graph, sources, targets, *, cutoff, method, routes, max_workers):
    """Costs and routes of the pairs sources[i] -> targets[i] (positions) by a point-to-point search.

    The caller has prepared the graph for the weight (``_prepare_graph``).
    """
    matrix = _csr_graph(graph, reverse=False)
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    targets = np.ascontiguousarray(targets, dtype=np.int32)
    if method == "astar":
        xs, ys = _node_coordinates(graph)
        result, _ = matrix.astar_pairs(xs, ys, sources, targets, _cutoff_float(cutoff), routes, max_workers)
        return result
    if method == "alt":
        table = _landmarks(graph, max_workers)
        return matrix.alt_pairs(table, sources, targets, _cutoff_float(cutoff), routes, max_workers)
    if method == "bidirectional":
        reverse = _csr_graph(graph, reverse=True)
        return matrix.shortest_pairs(reverse, sources, targets, _cutoff_float(cutoff), routes, max_workers)
    # One-sided searches, one per unique source, each stopping at its own targets.
    unique_sources, inverse = np.unique(sources, return_inverse=True)
    order, offsets = _groups(inverse, len(unique_sources))
    searches = _search_many(
        matrix,
        unique_sources,
        targets[order],
        cutoff,
        "both" if routes else "distances",
        max_workers,
        None,
        target_offsets=offsets,
    )
    costs = np.full(len(sources), np.inf, dtype=np.float32)
    paths = [np.empty(0, dtype=np.int32)] * len(sources)
    for tree_index, search in enumerate(searches):
        rows = order[offsets[tree_index] : offsets[tree_index + 1]]
        costs[rows] = _lookup(search, targets[rows])
        if routes:
            # pylint: disable-next=unpacking-non-sequence
            flat, bounds = _rust.reconstruct_paths(search[0], search[2], targets[rows], False)
            for row, start, end in zip(rows, bounds[:-1], bounds[1:]):
                paths[row] = flat[start:end]
    if not routes:
        return costs, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    lengths = np.array([len(path) for path in paths], dtype=np.int64)
    bounds = np.zeros(len(paths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=bounds[1:])
    flat = np.concatenate(paths) if paths else np.empty(0, dtype=np.int32)
    return costs, flat.astype(np.int32, copy=False), bounds


def _positions(graph, nodes):
    nodes = list(nodes)
    missing = [node for node in nodes if node not in graph.node_to_adjacency_pos]
    if missing:
        raise ValueError(f"Requested nodes are absent in graph: {missing[:10]}")
    return np.array([graph.node_to_adjacency_pos[node] for node in nodes], dtype=np.int32)


def _node_ids(graph):
    """Node ids in matrix order as an object array (cached with the matrix; treat it as read-only)."""
    cache = graph._csr_cache  # pylint: disable=protected-access
    if cache is not None and cache[_MATRIX] is graph.adjacency_matrix and cache[_NODE_IDS] is not None:
        return cache[_NODE_IDS]
    result = np.empty(len(graph.adjacency_nodelist), dtype=object)
    result[:] = graph.adjacency_nodelist
    if cache is not None and cache[_MATRIX] is graph.adjacency_matrix:
        cache[_NODE_IDS] = result
    return result


def _sources(graph, nodes, gdf, column):
    return resolve_graph_nodes_input(
        urban_graph=graph,
        nodes=nodes,
        gdf=gdf,
        graph_node_column=column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )


def _targets(graph, target_nodes):
    if target_nodes is None:
        return np.empty(0, dtype=np.int32), None
    nodes = list(target_nodes)
    if not nodes:
        raise ValueError("target_nodes must not be empty; use None for all reachable nodes")
    return _positions(graph, nodes), pd.Index(nodes, name="node")


def _lookup(search, targets, *, field=1, fill=np.inf, dtype=np.float32):
    result = np.full(len(targets), fill, dtype=dtype)
    if len(search[0]):
        offsets = np.searchsorted(search[0], targets)
        valid = offsets < len(search[0])
        matches = np.flatnonzero(valid)
        matches = matches[search[0][offsets[matches]] == targets[matches]]
        result[matches] = search[field][offsets[matches]]
    return result


def _distance_series(values, index, dtype, attrs, name="dist"):
    result = pd.Series(
        pd.arrays.SparseArray(np.asarray(values, dtype=dtype), fill_value=np.inf), index=index, name=name
    )
    result.attrs.update(attrs)
    return result


def _target_columns(searches, targets):
    """Per search: (column positions, costs) of the settled requested targets."""
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0 or len(np.unique(targets)) != len(targets):
        for search in searches:
            values = _lookup(search, targets)
            reachable = np.flatnonzero(np.isfinite(values))
            yield reachable, values[reachable]
        return
    # Unique targets: one gather per search instead of a binary search for every target.
    column_of = np.full(int(targets.max()) + 1, -1, dtype=np.int64)
    column_of[targets] = np.arange(len(targets))
    for search in searches:
        nodes = search[0].astype(np.int64, copy=False)
        columns = np.full(len(nodes), -1, dtype=np.int64)
        inside = nodes < len(column_of)
        columns[inside] = column_of[nodes[inside]]
        keep = np.flatnonzero(columns >= 0)
        yield columns[keep], search[1][keep]


def _distance_frame(searches, targets, inverse, index, columns, dtype, attrs, *, transposed=False):
    # Explicit zero costs are stored; absent entries become infinity, not scipy's implicit zero.
    data, column_indices, offsets = [], [], [0]
    for found, values in _target_columns(searches, targets):
        data.append(values.astype(dtype))
        column_indices.append(found)
        offsets.append(offsets[-1] + len(found))
    matrix = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(column_indices), np.array(offsets)),
        shape=(len(searches), len(targets)),
    )[inverse]
    matrix = (matrix.T if transposed else matrix).tocoo()
    result = _sparse_frame(matrix.row, matrix.col, matrix.data.astype(dtype), index, columns, dtype, np.inf)
    result.attrs.update(attrs)
    return result


def _store(
    searches,
    graph,
    tree_indices,
    origins,
    targets,
    index,
    columns,
    *,
    layout="matrix",
    reverse=False,
    transposed=False,
    attrs=None,
):
    return _PathStore(
        [_PathTree(search[0], search[2]) for search in searches],
        _node_ids(graph),
        tree_indices,
        origins,
        targets,
        index.copy(),
        None if columns is None else columns.copy(),
        layout=layout,
        reverse=reverse,
        transposed=transposed,
        attrs=attrs,
    )


def _finish(output, distances, store, path_storage, nearest_sources=None):
    if output == "distances":
        return distances
    if output == "paths":
        return store.materialize()
    return ShortestPathResult(distances, store, path_storage=path_storage, nearest_sources=nearest_sources)


def _kernel_search_many(graph, *args):
    """One Rust search per source (a module-level seam, so tests can observe the kernel calls)."""
    return graph.search_many(*args)


def _search_many(graph, sources, targets, cutoff, output, max_workers, batch_size, *, target_offsets=None):
    # Source i searches for targets[starts[i]:ends[i]]; without offsets all
    # sources share the whole array, so it is never copied per source.
    targets = np.ascontiguousarray(targets, dtype=np.int32)
    if target_offsets is None:
        starts, ends = np.zeros(len(sources), dtype=np.int64), np.full(len(sources), len(targets), dtype=np.int64)
    else:
        starts, ends = target_offsets[:-1], target_offsets[1:]
    searches = []
    step = len(sources) if batch_size is None else batch_size
    for start in range(0, len(sources), step):
        batch = slice(start, start + step)
        searches.extend(
            _kernel_search_many(
                graph,
                np.ascontiguousarray(sources[batch], dtype=np.int32),
                targets,
                np.ascontiguousarray(starts[batch], dtype=np.int64),
                np.ascontiguousarray(ends[batch], dtype=np.int64),
                _cutoff_float(cutoff),
                output != "distances",
                max_workers,
            )
        )
    return searches


def _series_search(
    graph, sources, *, target_nodes, weight, cutoff, reverse, output, path_storage, dtype, nearest=False
):
    matrix = _prepare_graph(graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = np.unique(_positions(graph, sources.to_numpy()))
    target_positions, index = _targets(graph, target_nodes)
    search = matrix.search(source_positions, target_positions, _cutoff_float(cutoff), output != "distances")
    if index is None:
        target_positions = search[0]
        index = pd.Index(_node_ids(graph)[target_positions], name="node")
    attrs = {"source_nodes": sources.copy()}
    distances = None if output == "paths" else _distance_series(_lookup(search, target_positions), index, dtype, attrs)
    store = (
        None
        if output == "distances"
        else _store(
            [search],
            graph,
            np.zeros(1, dtype=np.int32),
            source_positions,
            target_positions,
            index,
            None,
            layout="series",
            reverse=reverse and graph.is_directed,
            attrs=attrs,
        )
    )
    owners = None
    if nearest:
        owner_positions = _lookup(search, target_positions, field=3, fill=-1, dtype=np.int32)
        ids = _node_ids(graph)
        owners = pd.Series(
            [ids[p] if p >= 0 else None for p in owner_positions], index=index, name="source_node", dtype=object
        )
    return _finish(output, distances, store, path_storage, owners), owners


def single_source_dijkstra(
    urban_graph: UrbanGraph,
    source_node: Any,
    *,
    target_nodes: Iterable[Any] | None = None,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    output: Output = "both",
    path_storage: PathStorage = "eager",
    dtype: np.dtype = np.float32,
) -> RoutingOutput:
    """Search from one node; return distances, paths, or both from one traversal.

    Args:
        urban_graph: Urban graph to search.
        source_node: Source graph node id.
        target_nodes: Requested target ids, or None for all reachable nodes.
            Explicit targets retain their order, including unreachable targets.
        weight: Non-negative edge cost; time_min uses minutes, length_meter meters.
        cutoff: Inclusive maximum total cost. None means no limit.
        reverse: Search the reversed directed graph; routes are returned in the
            original graph direction, from a target towards the source.
        output: distances, paths, or both (a ShortestPathResult).
        path_storage: eager creates paths immediately; lazy requires output=both
            and reconstructs routes through result.get_path(target).
        dtype: Returned distance dtype. Kernel arithmetic remains float32.

    Returns:
        A sparse Series indexed by target node ids, or ShortestPathResult with
        matching distance and path Series. Missing costs/paths are inf/None.
    """
    _validate_options(output, path_storage)
    sources = _sources(urban_graph, [source_node], None, "graph_node_id")
    return _series_search(
        urban_graph,
        sources,
        target_nodes=target_nodes,
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        output=output,
        path_storage=path_storage,
        dtype=dtype,
    )[0]


def multi_source_dijkstra(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    target_nodes: Iterable[Any] | None = None,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    output: Output = "both",
    path_storage: PathStorage = "eager",
    dtype: np.dtype = np.float32,
) -> RoutingOutput:
    """Search from the nearest member of one combined source set.

    Options and Series results follow single_source_dijkstra. Supply either
    source_nodes or gdf_sources; geometries without graph_node_column are
    matched to nearest graph nodes. A combined result also exposes
    nearest_sources, indexed like distances. Ties choose the source appearing
    first in graph node order. This differs from dijkstra_parallel, which
    runs an independent search for each source.
    """
    _validate_options(output, path_storage)
    sources = _sources(urban_graph, source_nodes, gdf_sources, graph_node_column)
    return _series_search(
        urban_graph,
        sources,
        target_nodes=target_nodes,
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        output=output,
        path_storage=path_storage,
        dtype=dtype,
        nearest=True,
    )[0]


def dijkstra_parallel(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    target_nodes: Iterable[Any] | None = None,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    output: Output = "both",
    path_storage: PathStorage = "eager",
    dtype: np.dtype = np.float32,
    max_workers: int | None = None,
    batch_size: int | None = None,
) -> RoutingOutput:
    """Run independent searches, sharing work for sources mapped to the same node.

    Returns sparse tables with one row per input source and columns for
    target_nodes, or the union of reachable nodes when target_nodes is None.
    Source object labels and repeated node ids are preserved. Other options
    follow single_source_dijkstra. max_workers caps the worker threads (default:
    one per CPU). batch_size limits searches in flight; retained
    results and lazy trees still consume memory across all batches.
    """
    _validate_options(output, path_storage, max_workers, batch_size)
    sources = _sources(urban_graph, source_nodes, gdf_sources, graph_node_column)
    matrix = _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _positions(urban_graph, sources.to_numpy())
    unique_sources, inverse = np.unique(source_positions, return_inverse=True)
    target_positions, columns = _targets(urban_graph, target_nodes)
    searches = _search_many(matrix, unique_sources, target_positions, cutoff, output, max_workers, batch_size)
    if columns is None:
        target_positions = np.unique(np.concatenate([search[0] for search in searches]))
        columns = pd.Index(_node_ids(urban_graph)[target_positions], name="node")
    attrs = {"source_nodes": sources.copy()}
    distances = None
    if output != "paths":
        distances = _distance_frame(searches, target_positions, inverse, sources.index, columns, dtype, attrs)
    store = (
        None
        if output == "distances"
        else _store(
            searches,
            urban_graph,
            inverse,
            source_positions,
            target_positions,
            sources.index,
            columns,
            reverse=reverse and urban_graph.is_directed,
            attrs=attrs,
        )
    )
    return _finish(output, distances, store, path_storage)


def _resolve_od(graph, origins_nodes, destination_nodes, gdf_origins, gdf_destinations, graph_node_column):
    if (gdf_origins is not None and graph_node_column not in gdf_origins.columns) or (
        gdf_destinations is not None and graph_node_column not in gdf_destinations.columns
    ):
        config.logger.debug(
            "OD matrix uses nearest graph nodes. Project objects into UrbanGraph for precise distances."
        )
    origins = resolve_graph_nodes_input(
        urban_graph=graph,
        nodes=origins_nodes,
        gdf=gdf_origins,
        graph_node_column=graph_node_column,
        nodes_name="origins_nodes",
        gdf_name="gdf_origins",
    )
    destinations = resolve_graph_nodes_input(
        urban_graph=graph,
        nodes=destination_nodes,
        gdf=gdf_destinations,
        graph_node_column=graph_node_column,
        nodes_name="destination_nodes",
        gdf_name="gdf_destinations",
    )
    return origins, destinations


def _check_od_method(mode, method):
    if method not in ("auto",) + PAIR_METHODS:
        raise ValueError(f"method must be 'auto' or one of {PAIR_METHODS}, got {method!r}")
    if mode == "all_to_all" and method not in ("auto", "dijkstra"):
        raise ValueError(
            "mode='all_to_all' always runs one Dijkstra search per source; method must be 'auto' or 'dijkstra'"
        )


def _od_pairs(graph, origins, destinations, *, weight, dtype, cutoff, max_workers, output, path_storage, method):
    """Pairwise OD table from point-to-point searches, one per pair."""
    _prepare_graph(graph, weight=weight, cutoff=cutoff, reverse=False)
    origin_positions = _positions(graph, origins.to_numpy())
    destination_positions = _positions(graph, destinations.to_numpy())
    costs, flat, offsets = _kernel_pairs(
        graph,
        origin_positions,
        destination_positions,
        cutoff=cutoff,
        method=method,
        routes=output != "distances",
        max_workers=max_workers,
    )
    attrs = {"origin_nodes": origins.copy(), "destination_nodes": destinations.copy()}
    index = pd.MultiIndex.from_arrays([origins.index, destinations.index], names=["origin", "destination"])
    distances = None if output == "paths" else _distance_series(costs, index, dtype, attrs).to_frame()
    store = None
    if output != "distances":
        store = _PathStore(
            _RouteList(flat, offsets),
            _node_ids(graph),
            np.arange(len(index), dtype=np.int64),
            origin_positions,
            destination_positions,
            index.copy(),
            pd.Index(["path"]),
            layout="pairwise",
            attrs=attrs,
        )
    return _finish(output, distances, store, path_storage)


def _od_resolved(
    graph,
    origins,
    destinations,
    *,
    weight,
    dtype,
    cutoff,
    max_workers,
    batch_size,
    output,
    path_storage,
    mode,
    method="auto",
):
    if mode not in {"all_to_all", "pairwise"}:
        raise ValueError("mode must be 'all_to_all' or 'pairwise'")
    if mode == "pairwise" and len(origins) != len(destinations):
        raise ValueError("pairwise mode requires the same number of origins and destinations")
    _check_od_method(mode, method)
    if mode == "pairwise" and method == "auto":
        method = _auto_pair_method(origins)
    if mode == "pairwise" and method != "dijkstra":
        return _od_pairs(
            graph,
            origins,
            destinations,
            weight=weight,
            dtype=dtype,
            cutoff=cutoff,
            max_workers=max_workers,
            output=output,
            path_storage=path_storage,
            method=method,
        )
    transposed = mode == "all_to_all" and destinations.nunique() < origins.nunique()
    matrix = _prepare_graph(graph, weight=weight, cutoff=cutoff, reverse=transposed)
    origin_positions = _positions(graph, origins.to_numpy())
    destination_positions = _positions(graph, destinations.to_numpy())
    calc_origins, calc_targets = (
        (destination_positions, origin_positions) if transposed else (origin_positions, destination_positions)
    )
    unique_sources, inverse = np.unique(calc_origins, return_inverse=True)
    if mode == "pairwise":
        # Each unique origin searches only for the destinations paired with it.
        order, offsets = _groups(inverse, len(unique_sources))
        searches = _search_many(
            matrix, unique_sources, calc_targets[order], cutoff, output, max_workers, batch_size, target_offsets=offsets
        )
    else:
        searches = _search_many(
            matrix, unique_sources, np.unique(calc_targets), cutoff, output, max_workers, batch_size
        )
    attrs = {"origin_nodes": origins.copy(), "destination_nodes": destinations.copy()}
    index = origins.index
    columns = destinations.index
    distances = None
    if mode == "pairwise":
        index = pd.MultiIndex.from_arrays([origins.index, destinations.index], names=["origin", "destination"])
        columns = pd.Index(["path"])
        if output != "paths":
            values = np.full(len(index), np.inf, dtype=np.float32)
            for tree_index, search in enumerate(searches):
                rows = order[offsets[tree_index] : offsets[tree_index + 1]]
                values[rows] = _lookup(search, destination_positions[rows])
            distances = _distance_series(values, index, dtype, attrs).to_frame()
    elif output != "paths":
        distances = _distance_frame(
            searches, calc_targets, inverse, index, columns, dtype, attrs, transposed=transposed
        )
    store = (
        None
        if output == "distances"
        else _store(
            searches,
            graph,
            inverse,
            origin_positions,
            destination_positions,
            index,
            columns,
            layout="pairwise" if mode == "pairwise" else "matrix",
            transposed=transposed,
            reverse=transposed,
            attrs=attrs,
        )
    )
    return _finish(output, distances, store, path_storage)


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
    cutoff: float | None = None,
    output: Output = "distances",
    path_storage: PathStorage = "eager",
    mode: Literal["all_to_all", "pairwise"] = "all_to_all",
    batch_size: int | None = None,
    method: Literal["auto", "bidirectional", "astar", "dijkstra"] = "auto",
) -> RoutingOutput:
    """Compute OD distances, node routes, or both with one search per unique source.

    Args:
        urban_graph: Urban graph to search.
        gdf_origins: Origin objects, using graph_node_column or nearest geometries.
        gdf_destinations: Destination objects, with the same input conventions.
        origins_nodes: Origin graph ids, as an alternative to gdf_origins.
        destination_nodes: Destination graph ids, instead of gdf_destinations.
        graph_node_column: Explicit graph node id column in object tables.
        weight: Edge cost column: time_min (minutes) or length_meter (meters).
        dtype: Returned distance dtype; kernel arithmetic remains float32.
        threshold: Compatible alias for cutoff. Passing both is an error.
        max_workers: Worker threads (default: one per CPU).
        cutoff: Inclusive maximum total cost, or None for no limit.
        output: distances (default, preserving existing calls), paths, or both.
        path_storage: eager or lazy. Lazy requires output=both and retains compact
            predecessor trees. result.get_path(origin, destination) reconstructs
            one path; result.paths materializes and caches the entire path table.
        mode: all_to_all creates matrices; pairwise matches inputs by position
            and returns a MultiIndex table with one dist or path column.
        batch_size: Maximum searches per batch. This limits working memory,
            not the full returned table or retained lazy trees. Use iter_od_matrix
            to consume bounded blocks of origin rows instead.
        method: Search of pairwise tables. ``dijkstra`` runs one search per unique
            origin that stops at its destinations; ``bidirectional``, ``astar`` and
            ``alt`` run one point-to-point search per pair (see :func:`shortest_path`).
            ``auto`` (default) picks ``bidirectional`` when most origins occur once
            and ``dijkstra`` otherwise. All-to-all tables always use ``dijkstra``.

    Returns:
        Sparse DataFrame or ShortestPathResult. Labels and duplicate node inputs
        are preserved. Unreachable and out-of-cutoff pairs contain inf/None.
        Searches use the smaller unique input side when possible, stop once
        requested targets settle, and restore original directed route order.
    """
    _validate_options(output, path_storage, max_workers, batch_size)
    cutoff = _resolve_cutoff(cutoff, threshold)
    _check_od_method(mode, method)
    origins, destinations = _resolve_od(
        urban_graph, origins_nodes, destination_nodes, gdf_origins, gdf_destinations, graph_node_column
    )
    return _od_resolved(
        urban_graph,
        origins,
        destinations,
        weight=weight,
        dtype=dtype,
        cutoff=cutoff,
        max_workers=max_workers,
        batch_size=batch_size,
        output=output,
        path_storage=path_storage,
        mode=mode,
        method=method,
    )


def iter_od_matrix(
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
    cutoff: float | None = None,
    output: Output = "distances",
    path_storage: PathStorage = "eager",
    mode: Literal["all_to_all", "pairwise"] = "all_to_all",
    batch_size: int = 256,
    method: Literal["auto", "bidirectional", "astar", "dijkstra"] = "auto",
) -> Iterator[RoutingOutput]:
    """Yield OD results in origin-row blocks, using the same options as od_matrix.

    batch_size is the number of input origin rows per yielded block. In pairwise
    mode the corresponding destination rows are sliced too. Object-to-node
    matching is done once. Consume and release blocks to bound retained memory;
    collecting them all retains all tables/trees. Do not edit the graph while
    consuming an iterator. Validation occurs when iteration starts.
    """
    _validate_options(output, path_storage, max_workers, batch_size)
    if batch_size is None:
        raise ValueError("iter_od_matrix requires a positive batch_size")
    cutoff = _resolve_cutoff(cutoff, threshold)
    origins, destinations = _resolve_od(
        urban_graph, origins_nodes, destination_nodes, gdf_origins, gdf_destinations, graph_node_column
    )
    if mode not in {"all_to_all", "pairwise"}:
        raise ValueError("mode must be 'all_to_all' or 'pairwise'")
    if mode == "pairwise" and len(origins) != len(destinations):
        raise ValueError("pairwise mode requires the same number of origins and destinations")
    _check_od_method(mode, method)
    if mode == "pairwise" and method == "auto":
        method = _auto_pair_method(origins)
    for start in range(0, len(origins), batch_size):
        yield _od_resolved(
            urban_graph,
            origins.iloc[start : start + batch_size],
            destinations.iloc[start : start + batch_size] if mode == "pairwise" else destinations,
            weight=weight,
            dtype=dtype,
            cutoff=cutoff,
            max_workers=max_workers,
            batch_size=batch_size,
            output=output,
            path_storage=path_storage,
            mode=mode,
            method=method,
        )


def _auto_pair_method(origins):
    # Point-to-point searches pay off when origins rarely repeat; an origin with many destinations is
    # cheaper as one search that stops at all of them.
    return "bidirectional" if 2 * origins.nunique() > len(origins) else "dijkstra"


def _pair_route(urban_graph, source_node, target_node, *, weight, cutoff, method):
    """Cost and node-id route of one pair by a point-to-point search: (inf, []) when not reached."""
    _check_pair_method(method)
    cutoff = _resolve_cutoff(cutoff)
    _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=False)
    positions = _positions(urban_graph, [source_node, target_node])
    costs, flat, offsets = _kernel_pairs(
        urban_graph,
        positions[:1],
        positions[1:],
        cutoff=cutoff,
        method=method,
        routes=True,
        max_workers=1,
    )
    if not np.isfinite(costs[0]):
        return float("inf"), []
    return float(costs[0]), _node_ids(urban_graph)[flat[offsets[0] : offsets[1]]].tolist()


def single_source_dijkstra_path(
    urban_graph: UrbanGraph,
    source_node: Any,
    target_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Return one shortest node route, or [] when unreachable or beyond cutoff.

    With reverse on a directed graph the route runs from target_node to source_node
    along the edge directions. The search is bidirectional (see :func:`shortest_path`).
    """
    if reverse and urban_graph.is_directed:
        source_node, target_node = target_node, source_node
    return _pair_route(urban_graph, source_node, target_node, weight=weight, cutoff=cutoff, method="bidirectional")[1]


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
    """Return nearest source_node and dist columns for each reachable node."""
    sources = _sources(urban_graph, source_nodes, gdf_sources, graph_node_column)
    distances, owners = _series_search(
        urban_graph,
        sources,
        target_nodes=None,
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        output="distances",
        path_storage="eager",
        dtype=dtype,
        nearest=True,
    )
    result = pd.concat([owners, distances], axis=1)
    result.attrs.update(distances.attrs)
    return result
