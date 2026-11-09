from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from iduedu import _rust


def _positions_of(nodelist: list[Any], edges: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Matrix positions of edge endpoints (``-1`` when an endpoint is not in ``nodelist``)."""
    index = pd.Index(nodelist)
    if index.has_duplicates:
        raise ValueError("nodelist must not contain duplicate node ids")
    return index, index.get_indexer(edges["u"]), index.get_indexer(edges["v"])


def _arcs(graph, nodelist: list[Any], weight: str | None, *, weak: bool = False):
    """Directed arcs ``rows -> cols`` (with weights) that the graph allows between nodes of ``nodelist``.

    Undirected graphs and ``weak`` give both directions of every edge, graphs with a direction column give
    the reverse of every two-way edge, other directed graphs only the stored direction. Parallel edges are
    not merged here: the CSR builder merges every repeated arc, whatever orientation the edge was stored in.
    """
    edges = graph.edges_gdf
    _, u, v = _positions_of(nodelist, edges)
    inside = (u >= 0) & (v >= 0)
    u, v = u[inside].astype(np.int64), v[inside].astype(np.int64)
    w = (
        np.ones(len(u), dtype=np.float32)
        if weight is None
        else edges[weight].to_numpy()[inside].astype(np.float32, copy=False)
    )
    if weak or not graph.is_directed:
        return np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([w, w])
    if graph.edge_direction_column is not None:
        two_way = ~edges[graph.edge_direction_column].to_numpy()[inside].astype(bool)
        return np.concatenate([u, v[two_way]]), np.concatenate([v, u[two_way]]), np.concatenate([w, w[two_way]])
    return u, v, w


def _csr(size: int, rows, cols, data, reduce: str = "min") -> sparse.csr_matrix:
    indptr, indices, values = _rust.build_csr(size, rows, cols, data, reduce)
    return sparse.csr_matrix((values, indices, indptr), shape=(size, size))


def build_adjacency_matrix(
    graph,
    *,
    nodelist: Iterable[Any],
    weight: str,
    multigraph_rule: Literal["min", "max"] = "min",
) -> sparse.csr_matrix:
    """Build a weighted adjacency matrix for an ``UrbanGraph``.

    Every arc the graph allows becomes one matrix entry; several edges between the same pair of nodes
    (parallel edges, or the two orientations of an undirected edge stored twice) are merged with
    ``multigraph_rule``.

    Args:
        graph: Graph-like object with ``nodes_gdf``, ``edges_gdf`` and graph
            direction attributes.
        nodelist: Node ids defining matrix row and column order.
        weight: Edge column used as matrix values.
        multigraph_rule: Aggregation rule for parallel edges.

    Returns:
        SciPy CSR matrix with shape ``(len(nodelist), len(nodelist))`` and float32 values.

    Raises:
        KeyError: If ``weight`` is not present in ``edges_gdf``.
        ValueError: If weights contain missing values, ``multigraph_rule`` is
            unsupported or ``nodelist`` has duplicates.
    """
    nodelist = list(nodelist)

    if len(nodelist) == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if graph.edges_gdf.empty:
        return sparse.csr_matrix((len(nodelist), len(nodelist)), dtype=np.float32)
    if weight not in graph.edges_gdf.columns:
        raise KeyError(f"edges_gdf has no weight column {weight!r}")
    if graph.edges_gdf[weight].isna().any():
        raise ValueError(f"edges_gdf[{weight!r}] contains NaN")
    if multigraph_rule not in {"min", "max"}:
        raise ValueError(f"Unsupported multigraph_rule={multigraph_rule!r}")

    rows, cols, data = _arcs(graph, nodelist, weight)
    return _csr(len(nodelist), rows, cols, data, multigraph_rule)


def build_boolean_adjacency_matrix(graph, *, nodelist: Iterable[Any], weak: bool) -> sparse.csr_matrix:
    """Build a boolean adjacency matrix for component analysis.

    Args:
        graph: Graph-like object with edge topology and direction metadata.
        nodelist: Node ids defining matrix row and column order.
        weak: If true, ignore edge direction.

    Returns:
        Boolean SciPy CSR matrix.
    """
    nodelist = list(nodelist)
    size = len(nodelist)

    if size == 0 or graph.edges_gdf.empty:
        return sparse.csr_matrix((size, size), dtype=bool)

    rows, cols, data = _arcs(graph, nodelist, None, weak=weak)
    matrix = _csr(size, rows, cols, data)
    return sparse.csr_matrix((matrix.data.astype(bool), matrix.indices, matrix.indptr), shape=(size, size))
