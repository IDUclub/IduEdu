"""assign_flows against a plain Python reference of the shortest-path-forest flow assignment."""

import heapq

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from iduedu import UrbanGraph, assign_flows

pytestmark = [pytest.mark.unit, pytest.mark.rust]

CRS = 32636


def _graph(n, arcs, *, multigraph=False):
    """Directed graph on n nodes laid out on a line; arcs are (u, v, cost)."""
    nodes = gpd.GeoDataFrame(geometry=[Point(10.0 * i, 0.0) for i in range(n)], index=range(n), crs=CRS)
    rows = []
    for k, (u, v, cost) in enumerate(arcs):
        geometry = LineString([(10.0 * u, 0.0), (10.0 * v, float(k % 3 + 1))])
        rows.append(
            {"u": u, "v": v, "k": k, "time_min": float(cost), "length_meter": float(cost), "geometry": geometry}
        )
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    if not multigraph:
        edges = edges.drop(columns="k")
    return UrbanGraph(nodes, edges, multigraph, True, crs=CRS)


def _reference(matrix, sources, flows, offsets, weights, decay, cutoff, min_flow):
    """The algorithm in plain Python, one value per arc of ``matrix`` (CSR, float32 costs).

    Equal costs go to the source with the lower node position, the owner being the source node itself.
    """
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data.astype(np.float32)
    n = len(indptr) - 1
    out = np.zeros(len(indices))
    cutoff = np.float32(np.inf if cutoff is None else cutoff)
    for c in range(len(offsets) - 1):
        a, b = offsets[c], offsets[c + 1]
        dist = np.full(n, np.float32(np.inf), dtype=np.float32)
        owner = np.full(n, -1)
        parent = np.full(n, -1)
        parent_arc = np.full(n, -1)
        done = np.zeros(n, dtype=bool)
        heap, order = [], []
        for i in range(a, b):
            s = sources[i]
            if dist[s] == np.inf:
                dist[s], owner[s] = np.float32(0.0), s
                heapq.heappush(heap, (np.float32(0.0), s, s))
        while heap:
            d, i, u = heapq.heappop(heap)
            if done[u]:
                continue
            if d > cutoff:
                break
            if d != dist[u] or i != owner[u]:
                continue
            done[u] = True
            order.append(u)
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                if done[v]:
                    continue
                cand = np.float32(d + data[e])
                if cand > cutoff:
                    continue
                if cand < dist[v] or (cand == dist[v] and i < owner[v]):
                    dist[v], owner[v], parent[v], parent_arc[v] = cand, i, u, e
                    heapq.heappush(heap, (cand, i, v))
        pos = {node: k for k, node in enumerate(order)}
        branch = np.zeros(len(order))
        total = float(np.sum(flows[a:b]))
        score_sum = 0.0
        for k, node in enumerate(order):
            if parent[node] < 0:
                continue
            score = weights[node] * decay(float(dist[node]))
            if score > 0:
                branch[k] = score
                score_sum += score
        if score_sum > 0 and total > 0:
            branch = total * branch / score_sum
            for k in range(len(order) - 1, -1, -1):
                node = order[k]
                if parent[node] < 0:
                    continue
                flow = branch[k]
                branch[pos[parent[node]]] += flow
                if flow > min_flow or (parent[parent[node]] < 0 and flow > 0):
                    out[parent_arc[node]] += flow
    return out


def _random_case(seed, n=40, m=120):
    rng = np.random.default_rng(seed)
    arcs = [
        (int(u), int(v), int(c)) for u, v, c in zip(rng.integers(0, n, m), rng.integers(0, n, m), rng.integers(1, 4, m))
    ]
    arcs = [arc for arc in arcs if arc[0] != arc[1]]
    arcs += [(arcs[0][0], arcs[0][1], 7)]  # a parallel, more expensive edge
    weights = rng.integers(0, 3, n).astype(float)  # many zero attractions
    sources = rng.integers(0, n, 12)
    flows = rng.integers(0, 5, 12).astype(float)
    offsets = np.array([0, 3, 3, 7, 12])  # cluster 1 is empty
    return _graph(n, arcs, multigraph=True), sources, flows, offsets, weights


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("min_flow", [0.0, 0.3])
def test_arc_flows_match_the_reference(seed, min_flow):
    graph, sources, flows, offsets, weights = _random_case(seed)
    result = assign_flows(
        graph,
        source_nodes=sources,
        source_flows=flows,
        cluster_offsets=offsets,
        node_weights=weights,
        mean=3.0,
        std=2.0,
        cutoff=8.0,
        min_flow=min_flow,
        per="arcs",
    )
    decay = lambda t: np.exp(-0.5 * ((t - 3.0) / 2.0) ** 2)  # noqa: E731
    expected = _reference(graph.adjacency_matrix, sources, flows, offsets, weights, decay, 8.0, min_flow)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.sum() > 0


def test_equal_costs_go_to_the_lower_source_position():
    # Sources 3 and 0 reach node 1 at the same cost; node 0 wins whatever the order of the list.
    graph = _graph(4, [(0, 1, 1), (3, 1, 1), (1, 2, 1)])
    result = assign_flows(
        graph,
        source_nodes=[3, 0],
        source_flows=[1.0, 1.0],
        cluster_offsets=[0, 2],
        node_weights=np.array([0.0, 0.0, 1.0, 0.0]),
        decay="flat",
        per="edges",
    )
    assert result["flow_uv"].tolist() == [2.0, 0.0, 2.0]


def test_edge_table_splits_two_way_edges_and_sums_both_directions():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], index=[0, 1, 2], crs=CRS)
    edges = gpd.GeoDataFrame(
        {"u": [0, 1], "v": [1, 2], "time_min": [1.0, 1.0], "length_meter": [10.0, 10.0]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs=CRS,
    )
    graph = UrbanGraph(nodes, edges, False, False, crs=CRS)
    result = assign_flows(
        graph, source_nodes=[0, 2], source_flows=[4.0, 6.0], node_weights=[0.0, 1.0, 0.0], decay="flat"
    )
    assert result.loc[0, "flow_uv"] == 4.0 and result.loc[0, "flow_vu"] == 0.0
    assert result.loc[1, "flow_uv"] == 0.0 and result.loc[1, "flow_vu"] == 6.0
    assert result["flow"].tolist() == [4.0, 6.0]


def test_zero_attraction_or_zero_flow_gives_nothing():
    graph = _graph(3, [(0, 1, 1), (1, 2, 1)])
    for flows, weights in (([0.0], [0.0, 1.0, 1.0]), ([5.0], [0.0, 0.0, 0.0])):
        result = assign_flows(graph, source_nodes=[0], source_flows=flows, node_weights=weights, decay="flat")
        assert result["flow"].sum() == 0.0


def test_exp_decay_and_cutoff():
    graph = _graph(3, [(0, 1, 1), (1, 2, 1)])
    result = assign_flows(
        graph, source_nodes=[0], source_flows=[1.0], node_weights=[0.0, 1.0, 1.0], decay="exp", beta=1.0, per="arcs"
    )
    near, far = np.exp(-1.0), np.exp(-2.0)
    np.testing.assert_allclose(result, [1.0, far / (near + far)])
    cut = assign_flows(
        graph, source_nodes=[0], source_flows=[1.0], node_weights=[0.0, 1.0, 1.0], decay="flat", cutoff=1.0, per="arcs"
    )
    np.testing.assert_allclose(cut, [1.0, 0.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"decay": "gaussian"}, "gaussian"),
        ({"decay": "exp"}, "beta"),
        ({"decay": "nope"}, "decay must be"),
        ({"decay": "flat", "source_flows": [-1.0]}, "source_flows"),
        ({"decay": "flat", "cluster_offsets": [0, 2]}, "cluster_offsets"),
        ({"decay": "flat", "node_weights": [1.0, -1.0, 0.0]}, "non-negative"),
        ({"decay": "flat", "per": "nodes"}, "per must be"),
    ],
)
def test_invalid_inputs_raise(kwargs, message):
    graph = _graph(3, [(0, 1, 1), (1, 2, 1)])
    args = {"source_nodes": [0], "source_flows": [1.0], "node_weights": [0.0, 1.0, 1.0]} | kwargs
    with pytest.raises(ValueError, match=message):
        assign_flows(graph, **args)


def test_node_weights_by_column_name():
    graph = _graph(3, [(0, 1, 1), (1, 2, 1)])
    graph.nodes_gdf["jobs"] = pd.Series([0.0, 0.0, 3.0], index=graph.nodes_gdf.index)
    result = assign_flows(graph, source_nodes=[0], source_flows=[2.0], node_weights="jobs", decay="flat")
    assert result["flow"].tolist() == [2.0, 2.0]
