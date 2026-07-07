from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse


def build_adjacency_matrix(
    graph,
    *,
    nodelist: Iterable[Any],
    weight: str,
    multigraph_rule: Literal["min", "max"] = "min",
) -> sparse.csr_matrix:
    """Build a weighted adjacency matrix for an ``UrbanGraph``.

    Args:
        graph: Graph-like object with ``nodes_gdf``, ``edges_gdf`` and graph
            direction attributes.
        nodelist: Node ids defining matrix row and column order.
        weight: Edge column used as matrix values.
        multigraph_rule: Aggregation rule for parallel edges.

    Returns:
        SciPy CSR matrix with shape ``(len(nodelist), len(nodelist))``.

    Raises:
        KeyError: If ``weight`` is not present in ``edges_gdf``.
        ValueError: If weights contain missing values or ``multigraph_rule`` is
            unsupported.
    """
    nodelist = list(nodelist)

    if len(nodelist) == 0:
        return sparse.csr_matrix((0, 0))
    if graph.edges_gdf.empty:
        return sparse.csr_matrix((len(nodelist), len(nodelist)))
    if weight not in graph.edges_gdf.columns:
        raise KeyError(f"edges_gdf has no weight column {weight!r}")
    if graph.edges_gdf[weight].isna().any():
        raise ValueError(f"edges_gdf[{weight!r}] contains NaN")
    if multigraph_rule not in {"min", "max"}:
        raise ValueError(f"Unsupported multigraph_rule={multigraph_rule!r}")

    node_to_pos = {node: i for i, node in enumerate(nodelist)}

    edges = graph.edges_gdf
    mask = edges["u"].isin(nodelist) & edges["v"].isin(nodelist)
    edge_cols = ["u", "v", weight]
    if graph.edge_direction_column is not None:
        edge_cols.append(graph.edge_direction_column)
    edges = edges.loc[mask, edge_cols].copy()

    if graph.edge_direction_column is not None:
        forward_edges = edges[["u", "v", weight]].rename(columns={"u": "source", "v": "target"})
        reverse_edges = edges.loc[~edges[graph.edge_direction_column].astype(bool), ["u", "v", weight]].rename(
            columns={"v": "source", "u": "target"}
        )
        graph_edges = pd.concat([forward_edges, reverse_edges], ignore_index=True)
        graph_edges = graph_edges.groupby(["source", "target"], as_index=False, sort=False)[weight].agg(multigraph_rule)
        u = graph_edges["source"].to_numpy()
        v = graph_edges["target"].to_numpy()
        w = graph_edges[weight].to_numpy()
    elif graph.is_multigraph:
        edges = edges.groupby(["u", "v"], as_index=False, sort=False)[weight].agg(multigraph_rule)
        u = edges["u"].to_numpy()
        v = edges["v"].to_numpy()
        w = edges[weight].to_numpy()
    else:
        u = edges["u"].to_numpy()
        v = edges["v"].to_numpy()
        w = edges[weight].to_numpy()

    row = np.fromiter((node_to_pos[x] for x in u), dtype=np.int64, count=len(u))
    col = np.fromiter((node_to_pos[x] for x in v), dtype=np.int64, count=len(v))

    if graph.edge_direction_column is None and not graph.is_directed:
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        data = np.concatenate([w, w])
    else:
        data = w

    data = np.asarray(data, dtype=np.float32)

    return sparse.coo_matrix(
        (data, (row, col)),
        shape=(len(nodelist), len(nodelist)),
    ).tocsr()


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

    node_to_pos = {node: pos for pos, node in enumerate(nodelist)}
    edges = graph.edges_gdf
    edges = edges.loc[edges["u"].isin(nodelist) & edges["v"].isin(nodelist)]

    if edges.empty:
        return sparse.csr_matrix((size, size), dtype=bool)

    u_pos = np.fromiter((node_to_pos[node] for node in edges["u"].to_numpy()), dtype=np.int64, count=len(edges))
    v_pos = np.fromiter((node_to_pos[node] for node in edges["v"].to_numpy()), dtype=np.int64, count=len(edges))

    if weak or not graph.is_directed:
        rows = np.concatenate([u_pos, v_pos])
        cols = np.concatenate([v_pos, u_pos])
    elif graph.edge_direction_column is not None:
        oneway = edges[graph.edge_direction_column].astype(bool).to_numpy()
        rows = np.concatenate([u_pos, v_pos[~oneway]])
        cols = np.concatenate([v_pos, u_pos[~oneway]])
    else:
        rows = u_pos
        cols = v_pos

    data = np.ones(len(rows), dtype=bool)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size), dtype=bool).tocsr()
