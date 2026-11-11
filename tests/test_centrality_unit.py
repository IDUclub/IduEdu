import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import box

from iduedu import UrbanGraph, centrality, edge_betweenness, od_matrix
from tests.factories import CRS, _edges, _nodes, multigraph_with_parallel_edges, undirected_line_graph

pytestmark = pytest.mark.unit


def random_graph(size=5, seed=11, *, directed=True, unit=False) -> UrbanGraph:
    """A grid with random lengths (or unit lengths, which makes many equally short paths)."""
    rng = np.random.default_rng(seed)
    coords = {i * size + j: (j * 100.0, i * 100.0) for i in range(size) for j in range(size)}
    rows = []
    for i in range(size):
        for j in range(size):
            node = i * size + j
            for other in ([node + 1] if j + 1 < size else []) + ([node + size] if i + 1 < size else []):
                u, v = (node, other) if rng.random() < 0.5 else (other, node)
                row = {"u": u, "v": v, "length_meter": 100.0 if unit else float(rng.integers(50, 150))}
                if directed:
                    row["oneway"] = bool(rng.random() < 0.3)
                rows.append(row)
    return UrbanGraph(
        _nodes(coords),
        _edges(coords, rows),
        is_multigraph=False,
        is_directed=directed,
        edge_direction_column="oneway" if directed else None,
        crs=CRS,
    )


def _costs(graph, direction="from"):
    nodes = graph.nodes_gdf.index.to_numpy()
    cost = od_matrix(graph, origins_nodes=nodes, destination_nodes=nodes, weight="length_meter")
    cost = cost.sparse.to_dense().to_numpy().astype(float)
    return cost if direction == "from" else cost.T


@pytest.mark.parametrize("direction", ["from", "to"])
def test_reach_harmonic_and_gravity_match_the_cost_matrix(direction):
    graph = random_graph()
    weights = np.random.default_rng(2).uniform(0, 3, len(graph.nodes_gdf))
    weights[::7] = 0.0
    graph.nodes_gdf["people"] = weights
    cost = _costs(graph, direction)
    np.fill_diagonal(cost, np.inf)  # a node is not within reach of itself
    table = centrality(graph, cutoffs=[150, 400], node_weights="people", direction=direction)
    assert table.columns.tolist() == [
        "reach_150",
        "reach_400",
        "harmonic_150",
        "harmonic_400",
        "gravity_150",
        "gravity_400",
    ]
    for cutoff in (150, 400):
        near = cost <= cutoff
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.where(near & (cost > 0), 1.0 / cost, 0.0)
            decayed = np.where(near, np.exp(-4.0 / cutoff * cost), 0.0)
        assert table[f"reach_{cutoff}"].to_numpy() == pytest.approx((near * weights).sum(axis=1), rel=1e-5)
        assert table[f"harmonic_{cutoff}"].to_numpy() == pytest.approx(inverse @ weights, rel=1e-5)
        assert table[f"gravity_{cutoff}"].to_numpy() == pytest.approx(decayed @ weights, rel=1e-5)
    plain = centrality(graph, ["reach"], cutoffs=400, direction=direction)
    assert plain["reach_400"].to_numpy() == pytest.approx((cost <= 400).sum(axis=1))


def test_centrality_options():
    graph = random_graph(directed=False)
    everything = centrality(graph, ["reach", "gravity"], beta=0.01)
    assert everything.columns.tolist() == ["reach_inf", "gravity_inf"]
    assert (everything["reach_inf"] == len(graph.nodes_gdf) - 1).all()
    inside = centrality(graph, ["reach"], cutoffs=200, area=gpd.GeoSeries([box(-10, -10, 110, 110)], crs=CRS))
    assert inside.index.tolist() == [0, 1, 5, 6] and inside.index.name == "node"
    assert (
        inside["reach_200"].tolist()
        == centrality(graph, ["reach"], cutoffs=200).loc[[0, 1, 5, 6], "reach_200"].tolist()
    )
    series = pd.Series({0: 2.0, 6: 5.0})  # other nodes weigh nothing
    weighted = centrality(graph, ["reach"], node_weights=series)
    assert (
        weighted.loc[0, "reach_inf"] == 5.0
        and weighted.loc[6, "reach_inf"] == 2.0
        and weighted.loc[3, "reach_inf"] == 7.0
    )
    with pytest.raises(ValueError, match="measures"):
        centrality(graph, ["closeness"])
    with pytest.raises(ValueError, match="beta"):
        centrality(graph, ["gravity"])
    with pytest.raises(KeyError):
        centrality(graph, node_weights="missing")
    with pytest.raises(ValueError, match="one value per node"):
        centrality(graph, node_weights=[1.0, 2.0])


def _as_networkx(graph):
    nx = pytest.importorskip("networkx")
    result = nx.DiGraph() if graph.is_directed else nx.Graph()
    result.add_nodes_from(graph.nodes_gdf.index)
    for row in graph.edges_gdf.itertuples():
        result.add_edge(row.u, row.v, w=row.length_meter)
        if graph.is_directed and not row.oneway:
            result.add_edge(row.v, row.u, w=row.length_meter)
    return nx, result


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("unit", [True, False])
def test_betweenness_matches_networkx(directed, unit):
    graph = random_graph(directed=directed, unit=unit)
    nx, reference = _as_networkx(graph)
    nodes = centrality(graph, ["betweenness"])
    expected = nx.betweenness_centrality(reference, weight="w", normalized=False)
    assert nodes["betweenness_inf"].to_dict() == pytest.approx(expected, abs=1e-9)

    edges = edge_betweenness(graph)["betweenness_inf"].to_numpy()
    by_arc = nx.edge_betweenness_centrality(reference, weight="w", normalized=False)
    for value, row in zip(edges, graph.edges_gdf.itertuples()):
        if not directed:
            want = by_arc.get((row.u, row.v), by_arc.get((row.v, row.u)))
        else:
            want = by_arc[(row.u, row.v)] + (0.0 if row.oneway else by_arc[(row.v, row.u)])
        assert value == pytest.approx(want, abs=1e-9)


def _brute_force(graph, cutoff, weights=None):
    """Betweenness by listing every shortest path no longer than ``cutoff``."""
    nx, reference = _as_networkx(graph)
    nodes = list(graph.nodes_gdf.index)
    weights = dict(zip(nodes, np.ones(len(nodes)) if weights is None else weights))
    total = dict.fromkeys(nodes, 0.0)
    length = dict(nx.all_pairs_dijkstra_path_length(reference, weight="w"))
    for s, t in itertools.permutations(nodes, 2):
        if t not in length[s] or length[s][t] > cutoff:
            continue
        paths = list(nx.all_shortest_paths(reference, s, t, weight="w"))
        for path in paths:
            for node in path[1:-1]:
                total[node] += weights[s] * weights[t] / len(paths)
    return {node: value if graph.is_directed else value / 2 for node, value in total.items()}


@pytest.mark.parametrize("directed", [True, False])
def test_betweenness_within_thresholds_and_with_weights(directed):
    graph = random_graph(size=4, directed=directed, unit=True)
    table = centrality(graph, ["betweenness"], cutoffs=[200, 300])
    for cutoff in (200, 300):
        assert table[f"betweenness_{cutoff}"].to_dict() == pytest.approx(_brute_force(graph, cutoff), abs=1e-9)
    weights = np.random.default_rng(5).uniform(0, 2, len(graph.nodes_gdf))
    weights[3] = 0.0
    weighted = centrality(graph, ["betweenness"], cutoffs=300, node_weights=weights)
    assert weighted["betweenness_300"].to_dict() == pytest.approx(_brute_force(graph, 300, weights), abs=1e-6)


def test_betweenness_sampling_and_parallel_edges():
    graph = random_graph(directed=False)
    full = centrality(graph, ["betweenness"])
    assert centrality(graph, ["betweenness"], sample=len(graph.nodes_gdf))["betweenness_inf"].equals(
        full["betweenness_inf"]
    )
    estimate = centrality(graph, ["betweenness"], sample=12, seed=4)["betweenness_inf"]
    assert estimate.sum() == pytest.approx(full["betweenness_inf"].sum(), rel=0.5) and (estimate >= 0).all()
    with pytest.raises(ValueError, match="sample"):
        centrality(graph, ["betweenness"], sample=0)

    parallel = multigraph_with_parallel_edges()  # 0 = 1 (times 5 and 1), 1 - 2
    by_time = edge_betweenness(parallel, weight="time_min")["betweenness_inf"]
    assert by_time.tolist() == [0.0, 2.0, 2.0]
    by_length = edge_betweenness(parallel)["betweenness_inf"]  # equal lengths: the parallel edges share
    assert by_length.tolist() == [1.0, 1.0, 2.0]
    line = edge_betweenness(undirected_line_graph(), cutoffs=[10, 30])
    assert line["betweenness_10"].tolist() == [1.0, 1.0, 1.0] and line["betweenness_30"].tolist() == [3.0, 4.0, 3.0]
