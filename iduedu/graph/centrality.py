"""Network centralities within travel-cost thresholds: reach, harmonic closeness, gravity and betweenness.

Every measure is local: it looks only at the nodes within a threshold along the network, and several
thresholds are computed from the same searches. On a graph that is cut at the border of a study area the
values near the border are biased (the network beyond it is missing); build the graph with a margin at least
as wide as the largest threshold and pass the study ``area``, so only the nodes inside are reported while
the searches use the whole graph.
"""

from typing import Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from shapely.geometry.base import BaseGeometry

from iduedu.graph.routing import _prepare_graph
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph.within import band_betas, band_label, cost_bands, sum_within, value_rows

MEASURES = ("reach", "harmonic", "gravity", "betweenness")
_DECAY_OF = {"reach": "flat", "harmonic": "inverse", "gravity": "exp"}


def _node_weights(urban_graph: UrbanGraph, node_weights) -> np.ndarray | None:
    """Weights in node order from a column name, a Series indexed by node id or an array."""
    if node_weights is None:
        return None
    nodes = urban_graph.nodes_gdf
    if isinstance(node_weights, str):
        if node_weights not in nodes.columns:
            raise KeyError(f"graph.nodes_gdf has no column {node_weights!r}")
        values = pd.to_numeric(nodes[node_weights], errors="coerce").to_numpy(dtype=float)
    elif isinstance(node_weights, pd.Series):
        values = pd.to_numeric(node_weights.reindex(nodes.index), errors="coerce").to_numpy(dtype=float)
    else:
        values = np.asarray(node_weights, dtype=float)
        if values.shape != (len(nodes),):
            raise ValueError("node_weights must hold one value per node")
    values = np.where(np.isfinite(values), values, 0.0)
    if (values < 0).any():
        raise ValueError("node_weights must be non-negative")
    return values


def _inside(urban_graph: UrbanGraph, area) -> np.ndarray:
    """Positions of the nodes inside ``area`` (all nodes without it)."""
    nodes = urban_graph.nodes_gdf
    if area is None:
        return np.arange(len(nodes))
    if isinstance(area, gpd.GeoDataFrame):
        area = area.geometry
    if isinstance(area, gpd.GeoSeries):
        if area.crs is not None and nodes.crs is not None and area.crs != nodes.crs:
            area = area.to_crs(nodes.crs)
        area = shapely.union_all(area.values)
    if not isinstance(area, BaseGeometry):
        raise TypeError(f"area must be a shapely geometry, GeoSeries or GeoDataFrame, got {type(area).__name__}")
    shapely.prepare(area)
    return np.flatnonzero(shapely.intersects(area, nodes.geometry.values))


def _sample(n: int, sample, seed) -> tuple[np.ndarray, float]:
    """Source positions of a betweenness run and the factor that scales a sample to all sources."""
    if sample is None or sample >= n:
        return np.arange(n, dtype=np.int32), 1.0
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample < 1:
        raise ValueError("sample must be a positive integer or None")
    picked = np.sort(np.random.default_rng(seed).choice(n, size=int(sample), replace=False))
    return picked.astype(np.int32), n / float(sample)


def centrality(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    measures: Iterable[str] = ("reach", "harmonic", "gravity"),
    *,
    cutoffs: float | Iterable[float] | None = None,
    weight: Literal["length_meter", "time_min"] = "length_meter",
    beta: float | Iterable[float] | None = None,
    node_weights: str | pd.Series | np.ndarray | None = None,
    direction: Literal["from", "to"] = "from",
    area: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    sample: int | None = None,
    seed: int | None = 0,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Centralities of the nodes of a graph within travel-cost thresholds.

    Measures, for a node ``i``, the other nodes ``j`` within the threshold at cost ``d`` and their weights
    ``w`` (1 without ``node_weights``):

    - ``reach``: ``Σ w``, how much is within the threshold;
    - ``harmonic``: ``Σ w / d``, closeness that favours near nodes (nodes at cost 0 are left out);
    - ``gravity``: ``Σ w · exp(-beta · d)``, the same with an exponential decay;
    - ``betweenness``: how many shortest paths no longer than the threshold pass through the node, each
      weighted by ``w`` of both of its ends. A path shared by several equally short routes is split between
      them, end points are not counted, and on undirected graphs every pair of nodes counts once.

    Reach, harmonic and gravity come from one cut-off search per reported node. Betweenness needs a search
    from every node of the graph, which takes much longer; ``sample`` estimates it from a random part of them.

    Args:
        urban_graph: Graph to measure.
        measures: Any of ``"reach"``, ``"harmonic"``, ``"gravity"``, ``"betweenness"``.
        cutoffs: Threshold or thresholds in the units of ``weight``; ``None`` for no limit.
        weight: Edge cost column: ``length_meter`` (metres, the default) or ``time_min`` (minutes).
        beta: Decay rate of ``gravity``, one value or one per threshold. By default ``4 / threshold``, which
            leaves a weight of 0.018 at the threshold; required when there is no threshold.
        node_weights: Weights of the nodes: a column of ``nodes_gdf``, a Series indexed by node id or an
            array in node order. Missing values count as 0.
        direction: On directed graphs, ``"from"`` measures the way from the node to the others, ``"to"``
            towards the node. Betweenness does not depend on it.
        area: Report only the nodes inside (a shapely geometry in the graph CRS, or a GeoSeries /
            GeoDataFrame). The searches still use the whole graph.
        sample: For betweenness, the number of randomly chosen source nodes; the result is scaled up to all
            nodes. ``None`` uses every node.
        seed: Seed of the sample.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame indexed by node id with one column per measure and threshold, named ``<measure>_<cutoff>``
        (``reach_400``, ``betweenness_inf``).

    Raises:
        ValueError: If a measure is unknown, or ``cutoffs``, ``beta`` or ``node_weights`` are invalid.
    """
    measures = list(dict.fromkeys(measures))
    unknown = [m for m in measures if m not in MEASURES]
    if unknown or not measures:
        raise ValueError(f"measures must be chosen from {MEASURES}, got {unknown or measures}")
    if direction not in {"from", "to"}:
        raise ValueError("direction must be 'from' or 'to'")
    bands = cost_bands(cutoffs)
    weights = _node_weights(urban_graph, node_weights)
    _prepare_graph(urban_graph, weight=weight, cutoff=None, reverse=False)
    n = len(urban_graph.nodes_gdf)
    reported = _inside(urban_graph, area)
    out: dict[str, np.ndarray] = {}

    local = [m for m in measures if m != "betweenness"]
    if local:
        decays = [_DECAY_OF[m] for m in local]
        betas = band_betas(bands, decays, beta)
        carriers = np.arange(n) if weights is None else np.flatnonzero(weights > 0)
        amount = np.ones(len(carriers)) if weights is None else weights[carriers]
        values = value_rows(
            n,
            np.tile(carriers, len(local)),
            np.repeat(np.arange(len(local)), len(carriers)),
            np.tile(amount, len(local)),
            len(local),
        )
        sums = sum_within(
            urban_graph,
            reported,
            values,
            len(local),
            bands,
            weight=weight,
            decay=decays,
            betas=betas,
            reverse=direction == "to",
            skip_source=True,
            max_workers=max_workers,
        )
        for c, measure in enumerate(local):
            for b, band in enumerate(bands):
                out[f"{measure}_{band_label(band)}"] = sums[:, b, c]

    if "betweenness" in measures:
        nodes_sum, _ = _betweenness(urban_graph, bands, weight, weights, sample, seed, False, max_workers)
        for b, band in enumerate(bands):
            out[f"betweenness_{band_label(band)}"] = nodes_sum[reported, b]

    columns = [f"{m}_{band_label(band)}" for m in measures for band in bands]
    return pd.DataFrame(out, index=urban_graph.nodes_gdf.index[reported].rename("node"))[columns]


def _betweenness(urban_graph, bands, weight, weights, sample, seed, edges, max_workers):
    """Betweenness sums ``(nodes × bands, arcs × bands | None)`` in matrix order, scaled and halved as needed."""
    csr = _prepare_graph(urban_graph, weight=weight, cutoff=None, reverse=False)
    sources, scale = _sample(csr.n_nodes, sample, seed)
    nodes_sum, arcs_sum = csr.betweenness(
        sources,
        np.ascontiguousarray(bands, dtype=np.float32),
        None if weights is None else np.ascontiguousarray(weights, dtype=np.float32),
        edges,
        max_workers,
    )
    # An undirected graph holds both directions of every pair of nodes: count each pair once.
    scale = scale if urban_graph.is_directed else scale / 2.0
    return nodes_sum * scale, None if arcs_sum is None else arcs_sum * scale


def edge_betweenness(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    *,
    cutoffs: float | Iterable[float] | None = None,
    weight: Literal["length_meter", "time_min"] = "length_meter",
    node_weights: str | pd.Series | np.ndarray | None = None,
    sample: int | None = None,
    seed: int | None = 0,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Betweenness of the edges of a graph: how many shortest paths within a threshold run along each edge.

    Paths are counted as in :func:`centrality`. An edge that can be travelled both ways carries the paths of
    both directions. Among parallel edges only the cheapest one carries paths; equally cheap ones share them.

    Args:
        urban_graph: Graph to measure.
        cutoffs: Threshold or thresholds in the units of ``weight``; ``None`` for no limit.
        weight: Edge cost column: ``length_meter`` (metres, the default) or ``time_min`` (minutes).
        node_weights: Weights of the path ends, see :func:`centrality`.
        sample: Number of randomly chosen source nodes; the result is scaled up to all nodes.
        seed: Seed of the sample.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        DataFrame indexed like ``edges_gdf`` with one column ``betweenness_<cutoff>`` per threshold.
    """
    bands = cost_bands(cutoffs)
    weights = _node_weights(urban_graph, node_weights)
    _, arcs_sum = _betweenness(urban_graph, bands, weight, weights, sample, seed, True, max_workers)
    matrix = urban_graph.adjacency_matrix
    edges = urban_graph.edges_gdf
    index = pd.Index(urban_graph.adjacency_nodelist)
    u, v = index.get_indexer(edges["u"]), index.get_indexer(edges["v"])
    cost = edges[weight].to_numpy(dtype=np.float32)

    direction = urban_graph.edge_direction_column
    if not urban_graph.is_directed:
        both = u != v
    elif direction is not None and direction in edges.columns:
        both = ~edges[direction].to_numpy(dtype=bool) & (u != v)
    else:
        both = np.zeros(len(edges), dtype=bool)

    # Position of the arc of each node pair in the matrix (-1 where the matrix has no such arc). An arc is
    # realised by the cheapest edges between its nodes; equally cheap ones share its paths.
    arc_of = sparse.csr_matrix((np.arange(1, matrix.nnz + 1), matrix.indices, matrix.indptr), shape=matrix.shape)
    arc_cost = matrix.data.astype(np.float32)

    def realised(a, b, allowed):
        arc = np.asarray(arc_of[a, b]).ravel() - 1
        return arc, allowed & (arc >= 0) & (cost == arc_cost[np.maximum(arc, 0)])

    forward, forward_ok = realised(u, v, np.ones(len(edges), dtype=bool))
    backward, backward_ok = realised(v, u, both)
    sharing = np.bincount(forward[forward_ok], minlength=matrix.nnz) + np.bincount(
        backward[backward_ok], minlength=matrix.nnz
    )
    values = np.zeros((len(edges), len(bands)))
    for arc, ok in ((forward, forward_ok), (backward, backward_ok)):
        values[ok] += arcs_sum[arc[ok]] / sharing[arc[ok], None]
    return pd.DataFrame(values, index=edges.index, columns=[f"betweenness_{band_label(band)}" for band in bands])


__all__ = ["MEASURES", "centrality", "edge_betweenness"]
