"""Flow assignment along shortest-path forests: trips from sources spread over destinations by attraction and cost.

Every cluster of sources (for example the entrances of one block) sends its summed flow to the nodes it reaches
within ``cutoff``. A reached node takes a share proportional to its attraction times a decay of the travel cost;
the shares travel back along the shortest-path tree, so every edge carries the flow of all the trips that pass
it. The result is an all-or-nothing assignment per search: equilibrium methods (BPR, MSA) repeat it with edge
costs updated between runs.
"""

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from iduedu.graph.centrality import _node_weights, arcs_to_edges
from iduedu.graph.graph_inputs import _positions
from iduedu.graph.routing import _prepare_graph
from iduedu.graph.urban_graph import UrbanGraph


def assign_flows(  # pylint: disable=too-many-locals
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable,
    source_flows: Iterable[float],
    node_weights: str | pd.Series | np.ndarray,
    cluster_offsets: Iterable[int] | None = None,
    decay: Literal["gaussian", "exp", "flat"] = "gaussian",
    mean: float | None = None,
    std: float | None = None,
    beta: float | None = None,
    cutoff: float | None = None,
    min_flow: float = 0.0,
    weight: Literal["length_meter", "time_min"] = "time_min",
    per: Literal["edges", "arcs"] = "edges",
    max_workers: int | None = None,
) -> pd.DataFrame | np.ndarray:
    """Assign flows from sources to attractive nodes along shortest paths and sum them on the edges.

    Sources are grouped into clusters by ``cluster_offsets`` (cluster ``c`` holds the sources
    ``cluster_offsets[c]:cluster_offsets[c + 1]``; by default every source is a cluster of its own). Each cluster
    runs one search from all of its sources at once, every node joining its nearest source (equal costs go to the
    source that comes first in ``nodes_gdf``). Every reached node other than a source is a destination with the weight
    ``node_weights[node] * decay(cost)``:

    - ``"gaussian"``: ``exp(-0.5 * ((cost - mean) / std) ** 2)``, trips of a typical length ``mean``;
    - ``"exp"``: ``exp(-beta * cost)``;
    - ``"flat"``: 1.

    The summed flow of the cluster's sources is split between its destinations in proportion to these weights and
    carried back along the search tree. An arc keeps the flow of one cluster only when it exceeds ``min_flow``
    or when the arc leaves a source and the flow is positive; the kept flows of all clusters are summed.

    Args:
        urban_graph: Graph to load.
        source_nodes: Graph node ids of the sources, grouped by cluster.
        source_flows: Flow sent from every source (trips, vehicles), finite and non-negative.
        node_weights: Attraction of every node: a ``nodes_gdf`` column name, a Series indexed by node id or an array
            in ``nodes_gdf`` order. Missing values count as 0.
        cluster_offsets: ``n_clusters + 1`` ascending positions into ``source_nodes`` starting at 0 and ending at
            ``len(source_nodes)``; ``None`` makes every source a cluster.
        decay: Decay of the destination weight with the travel cost.
        mean: Typical trip cost for ``"gaussian"``, in the units of ``weight``.
        std: Spread of trip costs for ``"gaussian"``, positive.
        beta: Rate of ``"exp"``, per unit of ``weight``.
        cutoff: Largest travel cost of a trip, inclusive; ``None`` for no limit.
        min_flow: Flow of one cluster below which an arc that does not leave a source is ignored.
        weight: Edge cost column: ``time_min`` (minutes, the default) or ``length_meter`` (metres).
        per: ``"edges"`` (default) for a table of the edges; ``"arcs"`` for one value per arc of
            ``urban_graph.adjacency_matrix`` (aligned with its ``data``, after the call), where parallel edges are
            one arc.
        max_workers: Number of threads; ``None`` uses all cores.

    Returns:
        With ``per="arcs"``, a float64 array of the arc flows. Otherwise a DataFrame indexed like ``edges_gdf``
        with ``flow_uv`` (along the edge from ``u`` to ``v``), ``flow_vu`` (against it, for edges travelled both
        ways) and their sum ``flow``. Among parallel edges only the cheapest one carries flow; equally cheap ones
        share it.
    """
    if per not in ("edges", "arcs"):
        raise ValueError(f"per must be 'edges' or 'arcs', got {per!r}")
    csr = _prepare_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=False)
    nodes = np.asarray(list(source_nodes) if not isinstance(source_nodes, np.ndarray) else source_nodes)
    sources = _positions(urban_graph, nodes) if len(nodes) else np.empty(0, dtype=np.int32)
    flows = np.ascontiguousarray(np.asarray(list(source_flows), dtype=np.float64))
    if flows.shape != (len(sources),):
        raise ValueError("source_flows must hold one value per source node")
    if cluster_offsets is None:
        offsets = np.arange(len(sources) + 1, dtype=np.int64)
    else:
        offsets = np.ascontiguousarray(np.asarray(list(cluster_offsets), dtype=np.int64))
    weights = _node_weights(urban_graph, node_weights)
    arc_flows = csr.assign_flows(
        np.ascontiguousarray(sources, dtype=np.int32),
        flows,
        offsets,
        np.ascontiguousarray(weights, dtype=np.float64),
        decay,
        mean=None if mean is None else float(mean),
        std=None if std is None else float(std),
        beta=None if beta is None else float(beta),
        cutoff=None if cutoff is None else float(cutoff),
        min_flow=float(min_flow),
        n_threads=max_workers,
    )
    if per == "arcs":
        return arc_flows
    forward, backward = arcs_to_edges(urban_graph, weight, arc_flows)
    return pd.DataFrame(
        {"flow_uv": forward, "flow_vu": backward, "flow": forward + backward}, index=urban_graph.edges_gdf.index
    )


__all__ = ["assign_flows"]
