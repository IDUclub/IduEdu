"""Objects around origins along the network: the nearest ones, sums within a travel cost and diversity.

Origins and objects are represented by graph nodes: given as node ids, as a column of node ids in an object
table, or matched to their nearest nodes (project objects into the graph with
:func:`~iduedu.project_objects2urban_graph` for exact distances). Travel costs are measured between those
nodes.

Sums are computed from the smaller side: one cut-off search per origin node that collects the objects it
reaches, or one search per object node, run on the reversed graph, that hands its values to the origins
around it. Both give the same numbers.
"""

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from iduedu.graph.graph_inputs import resolve_graph_nodes_input
from iduedu.graph.routing import _cutoff_float, _positions, _prepare_graph, _resolve_cutoff
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph.within import band_betas, band_label, cost_bands, spread_within, sum_within, value_rows

Direction = Literal["from", "to"]
Decay = Literal["linear", "exp", "inverse"]


def _endpoints(urban_graph, origins_nodes, gdf_origins, object_nodes, gdf_objects, graph_node_column):
    origins = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=origins_nodes,
        gdf=gdf_origins,
        graph_node_column=graph_node_column,
        nodes_name="origins_nodes",
        gdf_name="gdf_origins",
    )
    objects = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=object_nodes,
        gdf=gdf_objects,
        graph_node_column=graph_node_column,
        nodes_name="object_nodes",
        gdf_name="gdf_objects",
    )
    return origins, objects


def _reverse(direction: str) -> bool:
    if direction not in {"from", "to"}:
        raise ValueError("direction must be 'from' (origin to object) or 'to' (object to origin)")
    return direction == "to"


def nearest_objects(
    urban_graph: UrbanGraph,
    *,
    gdf_origins: pd.DataFrame | None = None,
    gdf_objects: pd.DataFrame | None = None,
    origins_nodes: Iterable[Any] | None = None,
    object_nodes: Iterable[Any] | None = None,
    k: int = 1,
    cutoff: float | None = None,
    weight: Literal["length_meter", "time_min"] = "time_min",
    direction: Direction = "from",
    graph_node_column: str = "graph_node_id",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Find the ``k`` nearest objects of every origin along the network.

    Each origin runs one search that stops as soon as ``k`` objects are found, so the cost does not grow
    with the number of objects.

    Args:
        urban_graph: Graph to search.
        gdf_origins: Origin objects, with ``graph_node_column`` or matched to their nearest nodes, or
        origins_nodes: origin node ids.
        gdf_objects: Objects to look for, with the same conventions, or
        object_nodes: their node ids.
        k: Number of objects to return per origin.
        cutoff: Largest travel cost, inclusive; ``None`` for no limit.
        weight: Edge cost column: ``time_min`` (minutes) or ``length_meter`` (metres).
        direction: ``"from"`` measures the way from the origin to the object, ``"to"`` from the object to
            the origin. They differ on directed graphs.
        graph_node_column: Column of object tables that holds graph node ids.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame with a ``(origin, rank)`` MultiIndex (``rank`` starts at 1) and columns ``object`` (label of
        the object in its table, or its node id) and ``dist``. Origins with fewer than ``k`` reachable objects
        have fewer rows; objects on the same node keep the order of the object table.

    Raises:
        ValueError: If ``k`` is not a positive integer, or inputs cannot be resolved to graph nodes.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    cutoff = _resolve_cutoff(cutoff)
    reverse = _reverse(direction)
    origins, objects = _endpoints(urban_graph, origins_nodes, gdf_origins, object_nodes, gdf_objects, graph_node_column)
    csr = _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    n = csr.n_nodes
    origin_pos = _positions(urban_graph, origins.to_numpy())
    object_pos = _positions(urban_graph, objects.to_numpy())

    sources, inverse = np.unique(origin_pos, return_inverse=True)
    counts = np.bincount(object_pos, minlength=n).astype(np.int32)
    nodes, costs, offsets = csr.nearest(sources, counts, int(k), _cutoff_float(cutoff), max_workers)

    # Objects in node order (stable, so objects of one node keep the order of their table).
    by_node = np.argsort(object_pos, kind="stable")
    first_of_node = np.concatenate([[0], np.cumsum(counts)])[:-1]
    per_node = counts[nodes].astype(np.int64)
    total = int(per_node.sum())
    within = np.arange(total) - np.repeat(np.cumsum(per_node) - per_node, per_node)
    found_object = by_node[np.repeat(first_of_node[nodes], per_node) + within]
    found_cost = np.repeat(costs, per_node)
    found_source = np.repeat(np.repeat(np.arange(len(sources)), np.diff(offsets)), per_node)
    starts = np.concatenate([[0], np.cumsum(np.bincount(found_source, minlength=len(sources)))])
    rank = np.arange(total) - starts[found_source] + 1
    keep = rank <= k
    found_object, found_cost, found_source, rank = found_object[keep], found_cost[keep], found_source[keep], rank[keep]

    # Rows of every origin: the rows of its source node, origins in their own order.
    starts = np.concatenate([[0], np.cumsum(np.bincount(found_source, minlength=len(sources)))])
    size = (starts[1:] - starts[:-1])[inverse]
    rows = np.repeat(starts[:-1][inverse], size) + (
        np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
    )
    index = pd.MultiIndex.from_arrays([np.repeat(origins.index.to_numpy(), size), rank[rows]], names=["origin", "rank"])
    return pd.DataFrame({"object": objects.index.to_numpy()[found_object[rows]], "dist": found_cost[rows]}, index=index)


def _sums_at_origins(  # pylint: disable=too-many-locals
    urban_graph, origins, objects, triples, n_cols, bands, *, weight, decay, beta, direction, max_workers
) -> np.ndarray:
    """``(len(origins), bands, n_cols)`` decayed sums of object values around every origin.

    ``triples`` are ``(object position, column, value)`` arrays over the object table.
    """
    reverse = _reverse(direction)
    decay = "flat" if decay is None else decay
    betas = band_betas(bands, decay, beta)
    _prepare_graph(urban_graph, weight=weight, cutoff=None, reverse=reverse)
    n = len(urban_graph.adjacency_nodelist)
    origin_pos = _positions(urban_graph, origins.to_numpy())
    object_pos = _positions(urban_graph, objects.to_numpy())
    origin_nodes, origin_inverse = np.unique(origin_pos, return_inverse=True)
    object_nodes, object_inverse = np.unique(object_pos, return_inverse=True)
    rows, cols, data = triples

    if len(object_nodes) < len(origin_nodes):
        # Fewer places with objects than origins: search from the objects, against the travel direction.
        values = value_rows(len(object_nodes), object_inverse[rows], cols, data, n_cols)
        slots = np.full(n, -1, dtype=np.int32)
        slots[origin_nodes] = np.arange(len(origin_nodes), dtype=np.int32)
        sums = spread_within(
            urban_graph,
            object_nodes,
            values,
            n_cols,
            slots,
            len(origin_nodes),
            bands,
            weight=weight,
            decay=decay,
            betas=betas,
            reverse=not reverse,
            max_workers=max_workers,
        )
    else:
        values = value_rows(n, object_pos[rows], cols, data, n_cols)
        sums = sum_within(
            urban_graph,
            origin_nodes,
            values,
            n_cols,
            bands,
            weight=weight,
            decay=decay,
            betas=betas,
            reverse=reverse,
            max_workers=max_workers,
        )
    return sums[origin_inverse]


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def aggregate_objects(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    cutoff: float | Iterable[float],
    *,
    gdf_origins: pd.DataFrame | None = None,
    gdf_objects: pd.DataFrame | None = None,
    origins_nodes: Iterable[Any] | None = None,
    object_nodes: Iterable[Any] | None = None,
    columns: Iterable[str] | None = None,
    weight: Literal["length_meter", "time_min"] = "time_min",
    decay: Decay | None = None,
    beta: float | Iterable[float] | None = None,
    direction: Direction = "from",
    graph_node_column: str = "graph_node_id",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Count objects and sum their values within a travel cost of every origin.

    With ``decay`` every object counts with a weight that falls with its travel cost ``d``:
    ``"linear"`` is ``1 - d / cutoff``, ``"exp"`` is ``exp(-beta * d)`` and ``"inverse"`` is ``1 / d`` (objects
    at cost 0 are left out). Sums and counts are then weighted, and means are weighted means.

    Args:
        urban_graph: Graph to search.
        cutoff: Largest travel cost, inclusive, or several of them; every cutoff gives its own columns.
        gdf_origins: Origin objects, with ``graph_node_column`` or matched to their nearest nodes, or
        origins_nodes: origin node ids.
        gdf_objects: Objects to aggregate, with the same conventions, or
        object_nodes: their node ids (counts only).
        columns: Numeric columns of ``gdf_objects`` to sum and average. Missing values are left out.
        weight: Edge cost column: ``time_min`` (minutes) or ``length_meter`` (metres).
        decay: Distance decay of the weights, or ``None`` for none.
        beta: Rate of the exponential decay, one value or one per cutoff. By default ``4 / cutoff``, which
            leaves a weight of 0.018 at the cutoff.
        direction: ``"from"`` measures the way from the origin to the object, ``"to"`` from the object to
            the origin. They differ on directed graphs.
        graph_node_column: Column of object tables that holds graph node ids.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame indexed like the origins. For every cutoff ``c``: ``count_c`` and, for each of ``columns``,
        ``<column>_sum_c`` and ``<column>_mean_c`` (``NaN`` where nothing is in reach).

    Raises:
        ValueError: If ``cutoff``, ``decay`` or ``direction`` is invalid, or ``columns`` is passed without
            ``gdf_objects``.
        KeyError: If a column is absent from ``gdf_objects``.
    """
    bands = cost_bands(cutoff, "cutoff")
    origins, objects = _endpoints(urban_graph, origins_nodes, gdf_origins, object_nodes, gdf_objects, graph_node_column)
    columns = list(columns) if columns is not None else []
    if columns and gdf_objects is None:
        raise ValueError("columns need gdf_objects")
    missing = [c for c in columns if c not in gdf_objects.columns] if columns else []
    if missing:
        raise KeyError(f"gdf_objects has no columns {missing}")

    # Column 0 counts every object; each value column is followed by a column counting its known values.
    m = len(objects)
    rows, cols, data = [np.arange(m)], [np.zeros(m, dtype=np.int64)], [np.ones(m)]
    for j, column in enumerate(columns):
        values = pd.to_numeric(gdf_objects[column], errors="coerce").to_numpy(dtype=float)
        known = np.flatnonzero(np.isfinite(values))
        rows += [known, known]
        cols += [np.full(len(known), 1 + 2 * j), np.full(len(known), 2 + 2 * j)]
        data += [values[known], np.ones(len(known))]
    sums = _sums_at_origins(
        urban_graph,
        origins,
        objects,
        (np.concatenate(rows), np.concatenate(cols), np.concatenate(data)),
        1 + 2 * len(columns),
        bands,
        weight=weight,
        decay=decay,
        beta=beta,
        direction=direction,
        max_workers=max_workers,
    )
    out = {}
    for b, band in enumerate(bands):
        suffix = band_label(band)
        out[f"count_{suffix}"] = sums[:, b, 0]
        for j, column in enumerate(columns):
            out[f"{column}_sum_{suffix}"] = sums[:, b, 1 + 2 * j]
            out[f"{column}_mean_{suffix}"] = _ratio(sums[:, b, 1 + 2 * j], sums[:, b, 2 + 2 * j])
    return pd.DataFrame(out, index=origins.index)


def diversity(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    cutoff: float | Iterable[float],
    category: str,
    *,
    gdf_objects: pd.DataFrame,
    gdf_origins: pd.DataFrame | None = None,
    origins_nodes: Iterable[Any] | None = None,
    values: str | None = None,
    hill_order: float = 2.0,
    weight: Literal["length_meter", "time_min"] = "time_min",
    decay: Decay | None = None,
    beta: float | Iterable[float] | None = None,
    direction: Direction = "from",
    graph_node_column: str = "graph_node_id",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Diversity of object categories within a travel cost of every origin.

    The shares ``p`` of the categories are taken from the object counts in reach (or from the sums of
    ``values``, for example floor area), weighted by ``decay`` when it is set.

    Args:
        urban_graph: Graph to search.
        cutoff: Largest travel cost, inclusive, or several of them.
        category: Column of ``gdf_objects`` with the category of every object. Objects without one are
            left out.
        gdf_objects: Objects, with ``graph_node_column`` or matched to their nearest nodes.
        gdf_origins: Origin objects, with the same conventions, or
        origins_nodes: origin node ids.
        values: Numeric column of ``gdf_objects`` that weighs the objects instead of counting them.
        hill_order: Order ``q`` of the Hill number: 0 counts categories, 1 gives ``exp`` of the Shannon
            entropy, 2 (the default) the inverse Simpson index.
        weight: Edge cost column: ``time_min`` (minutes) or ``length_meter`` (metres).
        decay: Distance decay, see :func:`aggregate_objects`.
        beta: Rate of the exponential decay, see :func:`aggregate_objects`.
        direction: ``"from"`` (origin to object) or ``"to"`` (object to origin).
        graph_node_column: Column of object tables that holds graph node ids.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame indexed like the origins. For every cutoff ``c``: ``richness_c`` (categories in reach),
        ``shannon_c`` (entropy ``-Σ p ln p``) and ``hill_c`` (effective number of categories of order
        ``hill_order``); ``NaN`` for the last two where nothing is in reach.

    Raises:
        KeyError: If ``category`` or ``values`` is absent from ``gdf_objects``.
        ValueError: If ``hill_order`` is negative or other arguments are invalid.
    """
    if not isinstance(gdf_objects, pd.DataFrame):
        raise TypeError("gdf_objects must be a DataFrame or GeoDataFrame")
    for column in filter(None, (category, values)):
        if column not in gdf_objects.columns:
            raise KeyError(f"gdf_objects has no column {column!r}")
    if not np.isfinite(hill_order) or hill_order < 0:
        raise ValueError("hill_order must be a non-negative number")
    bands = cost_bands(cutoff, "cutoff")
    origins, objects = _endpoints(urban_graph, origins_nodes, gdf_origins, None, gdf_objects, graph_node_column)
    codes, categories = pd.factorize(gdf_objects[category])
    amount = (
        np.ones(len(objects)) if values is None else pd.to_numeric(gdf_objects[values], errors="coerce").to_numpy(float)
    )
    known = np.flatnonzero((codes >= 0) & np.isfinite(amount) & (amount > 0))
    sums = _sums_at_origins(
        urban_graph,
        origins,
        objects,
        (known, codes[known], amount[known]),
        max(len(categories), 1),
        bands,
        weight=weight,
        decay=decay,
        beta=beta,
        direction=direction,
        max_workers=max_workers,
    )
    out = {}
    for b, band in enumerate(bands):
        suffix = band_label(band)
        amounts = sums[:, b, :]
        total = amounts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(total > 0, amounts / total, 0.0)
            entropy = -np.where(share > 0, share * np.log(share), 0.0).sum(axis=1)
            if hill_order == 1:
                hill = np.exp(entropy)
            else:
                powered = np.where(share > 0, share**hill_order, 0.0).sum(axis=1)
                hill = np.where(powered > 0, powered ** (1.0 / (1.0 - hill_order)), np.nan)
        present = total[:, 0] > 0
        out[f"richness_{suffix}"] = (amounts > 0).sum(axis=1)
        out[f"shannon_{suffix}"] = np.where(present, entropy, np.nan)
        out[f"hill_{suffix}"] = np.where(present, hill, np.nan)
    return pd.DataFrame(out, index=origins.index)
