"""Simplification of a finished graph: chains of edges through nodes of degree two merged into single edges."""

import re
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from iduedu import _rust
from iduedu.graph.urban_graph import UrbanGraph

_TIME_COLUMN = re.compile(r"^time(_\w+)?_min$")
_STRUCTURAL = {"u", "v", "k"}


def _sum_columns(edges: gpd.GeoDataFrame, sum_columns: Iterable[str] | None) -> list[str]:
    if sum_columns is not None:
        columns = list(sum_columns)
        missing = set(columns) - set(edges.columns)
        if missing:
            raise KeyError(f"sum_columns not in edges_gdf: {sorted(missing)}")
        return columns
    return [c for c in edges.columns if c == "length_meter" or (isinstance(c, str) and _TIME_COLUMN.match(c))]


def _hashable(column: pd.Series) -> pd.Series:
    """Values usable as group keys: lists, sets and dicts become their text."""
    if column.dtype != object:
        return column
    return column.map(lambda value: repr(value) if isinstance(value, (list, tuple, set, dict)) else value)


def _key_ids(edges: gpd.GeoDataFrame, keys: list[str]) -> np.ndarray:
    if not keys:
        return np.zeros(len(edges), dtype=np.uint32)
    table = pd.DataFrame({i: _hashable(edges[column]) for i, column in enumerate(keys)})
    return table.groupby(list(table.columns), dropna=False, sort=False).ngroup().to_numpy(dtype=np.uint32)


def _two_way(graph: UrbanGraph, edges: gpd.GeoDataFrame) -> np.ndarray:
    if not graph.is_directed:
        return np.ones(len(edges), dtype=bool)
    column = graph.edge_direction_column
    if column is None or column not in edges.columns:
        return np.zeros(len(edges), dtype=bool)
    one_way = edges[column].map(lambda value: False if pd.isna(value) else bool(value)).to_numpy(dtype=bool)
    return ~one_way


def _chains(u, v, key_ids, two_way, keep):
    order, reversed_, offsets = _rust.merge_oriented_chains(
        u.astype(np.uint32), v.astype(np.uint32), key_ids, two_way, keep
    )
    return order.astype(np.int64), reversed_, offsets


def _ends(u, v, order, reversed_, offsets):
    first, last = offsets[:-1], offsets[1:] - 1
    tails = np.where(reversed_[first], v[order[first]], u[order[first]])
    heads = np.where(reversed_[last], u[order[last]], v[order[last]])
    return tails, heads


def _colliding_chains(tails, heads, lengths, directed: bool) -> np.ndarray:
    """Merged chains whose ends repeat the ends of another edge, which a graph without parallel edges forbids."""
    a, b = (tails, heads) if directed else (np.minimum(tails, heads), np.maximum(tails, heads))
    pairs = pd.DataFrame({"a": a, "b": b})
    duplicated = pairs.duplicated(keep="first").to_numpy()
    return np.flatnonzero(duplicated & (lengths > 1))


def simplify_graph(
    graph: UrbanGraph,
    *,
    keys: Iterable[str] | None = None,
    keep_nodes: Iterable | None = None,
    sum_columns: Iterable[str] | None = None,
) -> UrbanGraph:
    """Merge chains of edges through nodes of degree two into single edges.

    Two edges are merged at a node when they are the only edges there, their ``keys`` are equal and their
    directions agree. In an undirected graph, and between two-way edges of a directed graph (the direction
    column is ``False``), an edge drawn against the chain is turned, so the merged geometry runs from ``u`` to
    ``v``. One-way edges are merged only head to tail, and a one-way edge is never merged with a two-way one.
    A ring without a junction becomes a loop at its first node.

    Args:
        graph: Any graph.
        keys: Edge columns that must be equal along a chain. By default all columns except ``u``, ``v``, ``k``,
            the geometry and ``sum_columns``; in a directed graph the direction column is among them.
        keep_nodes: Nodes that stay, for example nodes that objects or another table refer to.
        sum_columns: Columns summed along a chain. By default ``length_meter``, ``time_min`` and every
            ``time_<name>_min`` column. The other columns take the values of the first edge.

    Returns:
        New graph with the same flags. Nodes inside merged chains are removed; the other nodes keep their ids.
        In a graph without parallel edges, a chain whose ends another edge already joins stays split at its
        first inner node.

    Raises:
        KeyError: If ``keys`` or ``sum_columns`` name columns the edges do not have.
    """
    edges = graph.edges_gdf
    nodes = graph.nodes_gdf
    if edges.empty:
        return graph
    sums = _sum_columns(edges, sum_columns)
    geometry_name = edges.geometry.name
    if keys is None:
        keys = [c for c in edges.columns if c not in _STRUCTURAL | set(sums) | {geometry_name}]
    else:
        keys = list(keys)
        missing = set(keys) - set(edges.columns)
        if missing:
            raise KeyError(f"keys not in edges_gdf: {sorted(missing)}")

    node_index = pd.Index(nodes.index)
    u = node_index.get_indexer(edges["u"])
    v = node_index.get_indexer(edges["v"])
    keep = np.zeros(len(node_index), dtype=bool)
    if keep_nodes is not None:
        positions = node_index.get_indexer(pd.Index(list(keep_nodes)))
        keep[positions[positions >= 0]] = True
    key_ids = _key_ids(edges, keys)
    two_way = _two_way(graph, edges)

    order, reversed_, offsets = _chains(u, v, key_ids, two_way, keep)
    if not graph.is_multigraph:
        # Without parallel edges two chains between the same nodes cannot both be merged: keep the first inner
        # node of every chain that repeats the ends of another edge, until none does.
        while True:
            tails, heads = _ends(u, v, order, reversed_, offsets)
            colliding = _colliding_chains(tails, heads, np.diff(offsets), graph.is_directed)
            if colliding.size == 0:
                break
            first = order[offsets[colliding]]
            inner = np.where(reversed_[offsets[colliding]], u[first], v[first])
            keep[inner] = True
            order, reversed_, offsets = _chains(u, v, key_ids, two_way, keep)

    new_edges = _merged_edges(edges, order, reversed_, offsets, sums)
    tails, heads = _ends(u, v, order, reversed_, offsets)
    new_edges["u"] = node_index.to_numpy()[tails]
    new_edges["v"] = node_index.to_numpy()[heads]

    last = np.zeros(len(order), dtype=bool)
    last[offsets[1:] - 1] = True
    inner_heads = np.where(reversed_, u[order], v[order])[~last]
    removed = np.zeros(len(node_index), dtype=bool)
    removed[inner_heads] = True
    removed[tails] = False
    new_nodes = nodes[~removed]

    if graph.is_multigraph:
        new_edges["k"] = new_edges.groupby(["u", "v"], sort=False).cumcount()
    columns = [c for c in edges.columns if c in new_edges.columns]
    return UrbanGraph(
        nodes_gdf=new_nodes,
        edges_gdf=new_edges[columns],
        is_multigraph=graph.is_multigraph,
        is_directed=graph.is_directed,
        edge_direction_column=graph.edge_direction_column,
        adjacency_weight=graph.adjacency_weight,
        crs=graph.crs,
        graph_type=graph.type,
    )


def _merged_edges(edges, order, reversed_, offsets, sums) -> gpd.GeoDataFrame:
    n_chains = len(offsets) - 1
    first = order[offsets[:-1]]
    merged = edges.iloc[first].drop(columns=list(_STRUCTURAL & set(edges.columns))).reset_index(drop=True)
    for column in sums:
        values = edges[column].to_numpy(dtype=float)[order]
        merged[column] = np.add.reduceat(values, offsets[:-1]) if len(values) else values

    single = np.diff(offsets) == 1
    if single.all() and not reversed_.any():
        return merged  # nothing merged or turned: the geometries stay

    geometries = edges.geometry.to_numpy()
    coords, owner = shapely.get_coordinates(geometries, return_index=True)
    counts = np.bincount(owner, minlength=len(geometries))
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    chain_of = np.repeat(np.arange(n_chains), np.diff(offsets))
    inner = np.ones(len(order), dtype=bool)
    inner[offsets[:-1]] = False  # every edge after the first drops the point it shares with the previous one
    take = counts[order] - inner
    base = np.where(reversed_, starts[order] + counts[order] - 1 - inner, starts[order] + inner)
    step = np.where(reversed_, -1, 1)
    within = np.arange(take.sum()) - np.repeat(np.cumsum(take) - take, take)
    index = np.repeat(base, take) + np.repeat(step, take) * within
    lines = shapely.linestrings(coords[index], indices=np.repeat(chain_of, take))
    merged[edges.geometry.name] = lines
    return gpd.GeoDataFrame(merged, geometry=edges.geometry.name, crs=edges.crs)
