"""Point-to-point routes: one shortest path, alternative routes and the k shortest paths."""

from typing import Any, Literal

import numpy as np
import pandas as pd

from iduedu.graph.routing import (
    PairMethod,
    _csr_graph,
    _cutoff_float,
    _node_ids,
    _pair_route,
    _positions,
    _prepare_graph,
    _resolve_cutoff,
    _validate_options,
)
from iduedu.graph.urban_graph import UrbanGraph


def shortest_path(
    urban_graph: UrbanGraph,
    source_node: Any,
    target_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    method: PairMethod = "bidirectional",
) -> tuple[float, list[Any]]:
    """Return the cost and the node route of one shortest path from source_node to target_node.

    Args:
        urban_graph: Urban graph to search.
        source_node: Graph id of the start node.
        target_node: Graph id of the end node.
        weight: Edge cost column: time_min (minutes) or length_meter (meters).
        cutoff: Inclusive maximum cost, or None for no limit.
        method: ``bidirectional`` (default) searches from both ends at once and
            stops where they meet. ``astar`` searches from the start towards the
            straight-line direction of the end; it needs point geometries of the
            nodes. ``alt`` is A* with an estimate from the costs to and from a few
            landmark nodes; their tables are built on the first call and kept
            with the graph, so it pays off for many queries on one graph.
            ``dijkstra`` is a one-sided search that stops at the end. All return a
            shortest path; they differ in speed only, and a tie between paths of
            equal cost may be resolved differently.

    Returns:
        ``(cost, path)``: the cost in the units of ``weight`` and the node ids from
        source_node to target_node, or ``(inf, [])`` when the end is not reached
        within cutoff.
    """
    return _pair_route(urban_graph, source_node, target_node, weight=weight, cutoff=cutoff, method=method)


def k_shortest_paths(
    urban_graph: UrbanGraph,
    source_node: Any,
    target_node: Any,
    k: int,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Return up to k shortest loopless paths from source_node to target_node (Yen's algorithm).

    Every next path leaves an earlier one at some node and takes the shortest way
    on from there that repeats no node of the shared part and no way taken
    before. The paths are often small variations of one another (another
    sidewalk, the other side of a square); for routes along other streets use
    :func:`alternative_paths`.

    Args:
        urban_graph: Urban graph to search.
        source_node: Graph id of the start node.
        target_node: Graph id of the end node.
        k: Maximum number of paths, at least 1.
        weight: Edge cost column: time_min (minutes) or length_meter (meters).
        cutoff: Inclusive maximum cost of a path, or None for no limit.
        max_workers: Worker threads for the searches of each step (default: one per CPU).

    Returns:
        DataFrame indexed by ``rank`` (0 is the shortest path) with columns
        ``dist`` (cost in the units of ``weight``) and ``path`` (node ids from
        source_node to target_node), cheapest first. It has fewer than k rows when
        there are fewer loopless paths within cutoff, and none when the end is
        not reached.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _validate_options("distances", "eager", max_workers)
    cutoff = _resolve_cutoff(cutoff)
    matrix = _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=False)
    source, target = _positions(urban_graph, [source_node, target_node])
    costs, flat, offsets = matrix.k_shortest(
        _csr_graph(urban_graph, reverse=True), int(source), int(target), int(k), _cutoff_float(cutoff), max_workers
    )
    ids = _node_ids(urban_graph)
    paths = [ids[flat[start:end]].tolist() for start, end in zip(offsets[:-1], offsets[1:])]
    result = pd.DataFrame(
        {"dist": costs.astype(np.float64), "path": pd.Series(paths, dtype=object)},
        index=pd.RangeIndex(len(paths), name="rank"),
    )
    result.attrs.update({"source_node": source_node, "target_node": target_node, "weight": weight})
    return result


def alternative_paths(
    urban_graph: UrbanGraph,
    source_node: Any,
    target_node: Any,
    k: int = 3,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    penalty: float = 1.5,
    max_overlap: float = 0.7,
    max_cost_ratio: float = 1.4,
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Return up to k distinct routes from source_node to target_node by the penalty method.

    The shortest route comes first. Then the costs of the edges it uses are
    multiplied by ``penalty`` in both directions and the search runs again, so
    the next route leaves those streets wherever a detour exists; every route
    found is penalised in turn. A route is kept when its real cost is at most
    ``max_cost_ratio`` times the shortest one and at most ``max_overlap`` of
    that cost runs on the streets of a route kept earlier. Unlike
    :func:`k_shortest_paths`, whose routes on a street graph often differ by a
    sidewalk, the routes differ by whole streets. Streets are graph edges: on a
    walk graph the sidewalk on the other side of an avenue is another street.

    Args:
        urban_graph: Urban graph to search.
        source_node: Graph id of the start node.
        target_node: Graph id of the end node.
        k: Maximum number of routes, at least 1. At most ``4 * k`` searches run.
        weight: Edge cost column: time_min (minutes) or length_meter (meters).
        penalty: Factor applied to the edges of every route found, above 1.
        max_overlap: Largest share of a route's cost on the streets of an
            earlier route, from 0 (nothing shared) to 1 (no limit).
        max_cost_ratio: Largest cost of a route relative to the shortest one, at least 1.
        cutoff: Inclusive maximum cost of a route, or None for no limit.

    Returns:
        DataFrame indexed by ``rank`` (0 is the shortest route, then by cost)
        with columns ``dist`` (real cost in the units of ``weight``), ``overlap``
        (the largest share of it on the streets of another route of the table)
        and ``path`` (node ids from source_node to target_node). It has fewer
        than k rows when no more routes meet the limits, and none when the end
        is not reached.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if np.isnan(penalty) or penalty <= 1:
        raise ValueError(f"penalty must be > 1, got {penalty}")
    if np.isnan(max_overlap) or max_overlap < 0 or max_overlap > 1:
        raise ValueError(f"max_overlap must be between 0 and 1, got {max_overlap}")
    if np.isnan(max_cost_ratio) or max_cost_ratio < 1:
        raise ValueError(f"max_cost_ratio must be >= 1, got {max_cost_ratio}")
    cutoff = _resolve_cutoff(cutoff)
    matrix = _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=False)
    source, target = _positions(urban_graph, [source_node, target_node])
    costs, overlaps, flat, offsets = matrix.alternatives(
        _csr_graph(urban_graph, reverse=True),
        int(source),
        int(target),
        int(k),
        float(penalty),
        float(max_overlap),
        float(max_cost_ratio),
        _cutoff_float(cutoff),
    )
    ids = _node_ids(urban_graph)
    paths = [ids[flat[start:end]].tolist() for start, end in zip(offsets[:-1], offsets[1:])]
    result = pd.DataFrame(
        {
            "dist": costs.astype(np.float64),
            "overlap": overlaps.astype(np.float64),
            "path": pd.Series(paths, dtype=object),
        },
        index=pd.RangeIndex(len(paths), name="rank"),
    )
    result.attrs.update({"source_node": source_node, "target_node": target_node, "weight": weight})
    return result
