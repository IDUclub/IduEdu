"""Shared Dijkstra workflows for distances, routes and origin-destination tables."""

import warnings
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from iduedu import _rust
from iduedu.config import config
from iduedu.graph.graph_inputs import resolve_graph_nodes_input
from iduedu.graph.routing_result import ShortestPathResult, _groups, _PathStore, _PathTree, _sparse_frame
from iduedu.graph.urban_graph import UrbanGraph

Output = Literal["distances", "paths", "both"]
PathStorage = Literal["eager", "lazy"]
RoutingTable = pd.Series | pd.DataFrame
RoutingOutput = RoutingTable | ShortestPathResult


def warn_deprecated(name: str, replacement: str, *, stacklevel: int = 3) -> None:
    """Warn that a legacy routing helper will be removed in IduEdu 3.0."""
    warnings.warn(
        f"{name} is deprecated and will be removed in IduEdu 3.0; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


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


def _csr_graph(urban_graph, *, reverse):
    """Rust view of the cached adjacency matrix (the transpose for reverse directed searches), built once."""
    matrix = urban_graph.adjacency_matrix
    cache = urban_graph._csr_cache  # pylint: disable=protected-access
    if cache is None or cache[0] is not matrix:
        forward = _rust.CsrGraph(
            np.asarray(matrix.indptr, dtype=np.int32),
            np.asarray(matrix.indices, dtype=np.int32),
            np.asarray(matrix.data, dtype=np.float32),
        )
        cache = urban_graph._csr_cache = [matrix, forward, None]  # pylint: disable=protected-access
    if not (reverse and urban_graph.is_directed):
        return cache[1]
    if cache[2] is None:
        cache[2] = cache[1].transpose()
    return cache[2]


def _prepare_graph(urban_graph, *, weight, cutoff, reverse):
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if len(urban_graph.nodes_gdf) == 0:
        raise ValueError("graph is empty")
    if urban_graph.nodes_gdf.index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")
    if weight not in urban_graph.edges_gdf:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")
    weights = urban_graph.edges_gdf[weight]
    if weights.isna().any() or (weights < 0).any():
        raise ValueError(f"graph.edges_gdf[{weight!r}] must contain non-negative values without NaN")
    _resolve_cutoff(cutoff)
    nodelist = urban_graph.nodes_gdf.index.to_list()
    if (
        urban_graph.adjacency_matrix is None
        or urban_graph.adjacency_weight != weight
        or urban_graph.adjacency_nodelist != nodelist
    ):
        urban_graph.update_adjacency_matrix(nodelist=nodelist, weight=weight)
    return _csr_graph(urban_graph, reverse=reverse)


def _positions(graph, nodes):
    nodes = list(nodes)
    missing = [node for node in nodes if node not in graph.node_to_adjacency_pos]
    if missing:
        raise ValueError(f"Requested nodes are absent in graph: {missing[:10]}")
    return np.array([graph.node_to_adjacency_pos[node] for node in nodes], dtype=np.int32)


def _node_ids(graph):
    result = np.empty(len(graph.adjacency_nodelist), dtype=object)
    result[:] = graph.adjacency_nodelist
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


def _distance_frame(searches, targets, inverse, index, columns, dtype, attrs, *, transposed=False):
    # Look up all requested targets once per tree. Explicit zero costs are
    # stored; absent entries become infinity, not scipy's implicit zero.
    data, column_indices, offsets = [], [], [0]
    for search in searches:
        values = _lookup(search, targets)
        reachable = np.flatnonzero(np.isfinite(values))
        data.append(values[reachable].astype(dtype))
        column_indices.append(reachable)
        offsets.append(offsets[-1] + len(reachable))
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
        config.logger.info("OD matrix uses nearest graph nodes. Project objects into UrbanGraph for precise distances.")
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


def _od_resolved(
    graph, origins, destinations, *, weight, dtype, cutoff, max_workers, batch_size, output, path_storage, mode
):
    if mode not in {"all_to_all", "pairwise"}:
        raise ValueError("mode must be 'all_to_all' or 'pairwise'")
    if mode == "pairwise" and len(origins) != len(destinations):
        raise ValueError("pairwise mode requires the same number of origins and destinations")
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

    Returns:
        Sparse DataFrame or ShortestPathResult. Labels and duplicate node inputs
        are preserved. Unreachable and out-of-cutoff pairs contain inf/None.
        Searches use the smaller unique input side when possible, stop once
        requested targets settle, and restore original directed route order.
    """
    _validate_options(output, path_storage, max_workers, batch_size)
    cutoff = _resolve_cutoff(cutoff, threshold)
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
    """Return one shortest node route, or [] when unreachable or beyond cutoff."""
    paths = single_source_dijkstra(
        urban_graph,
        source_node,
        target_nodes=[target_node],
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        output="paths",
    )
    path = paths.iloc[0]
    return path if isinstance(path, list) else []


def single_source_dijkstra_path_length(
    urban_graph: UrbanGraph,
    source_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.Series:
    """Return a sparse distance Series indexed by reachable graph nodes.

    .. deprecated:: 2.4.1
        Use ``single_source_dijkstra(..., output="distances")``.
    """
    warn_deprecated("single_source_dijkstra_path_length", 'single_source_dijkstra(..., output="distances")')
    return single_source_dijkstra(
        urban_graph, source_node, weight=weight, cutoff=cutoff, reverse=reverse, dtype=dtype, output="distances"
    )


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
    """Return nearest-source distances; inputs follow multi_source_dijkstra.

    .. deprecated:: 2.4.1
        Use ``multi_source_dijkstra(..., output="distances")``.
    """
    warn_deprecated("multi_source_dijkstra_path_length", 'multi_source_dijkstra(..., output="distances")')
    return multi_source_dijkstra(
        urban_graph,
        source_nodes=source_nodes,
        gdf_sources=gdf_sources,
        graph_node_column=graph_node_column,
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        dtype=dtype,
        output="distances",
    )


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
    """Return independent distance rows; inputs follow dijkstra_parallel.

    .. deprecated:: 2.4.1
        Use ``dijkstra_parallel(..., output="distances")``.
    """
    warn_deprecated("dijkstra_path_length_parallel", 'dijkstra_parallel(..., output="distances")')
    return dijkstra_parallel(
        urban_graph,
        source_nodes=source_nodes,
        gdf_sources=gdf_sources,
        graph_node_column=graph_node_column,
        weight=weight,
        cutoff=cutoff,
        reverse=reverse,
        dtype=dtype,
        max_workers=max_workers,
        output="distances",
    )


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
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for od_matrix(output='paths'), including pairwise mode.

    Despite its historical name this runs independent OD searches, rather than
    the nearest-source search provided by multi_source_dijkstra.

    .. deprecated:: 2.4.1
        Use ``od_matrix(..., output="paths")``.
    """
    warn_deprecated("multi_source_dijkstra_path", 'od_matrix(..., output="paths")')
    return od_matrix(
        urban_graph,
        gdf_origins=gdf_origins,
        gdf_destinations=gdf_destinations,
        origins_nodes=origins_nodes,
        destination_nodes=destination_nodes,
        graph_node_column=graph_node_column,
        weight=weight,
        mode=mode,
        threshold=threshold,
        cutoff=cutoff,
        max_workers=max_workers,
        output="paths",
    )
