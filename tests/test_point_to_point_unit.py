from itertools import islice

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from shapely.geometry import LineString, Point

from iduedu import (
    UrbanGraph,
    alternative_paths,
    k_shortest_paths,
    od_matrix,
    path_to_edges,
    shortest_path,
    single_source_dijkstra_path,
)
from iduedu.graph import routing
from tests.factories import CRS, directed_oneway_graph, disconnected_graph, undirected_line_graph

pytestmark = [pytest.mark.unit, pytest.mark.rust]

METHODS = ["bidirectional", "astar", "alt", "dijkstra"]


def _grid_graph(width=9, *, directed=True, seed=0, zero_every=None):
    """Grid of points 10 m apart; edges cost their length (or a time) times a random factor."""
    rng = np.random.default_rng(seed)
    coords = {i: (10.0 * (i % width), 10.0 * (i // width)) for i in range(width * width)}
    rows = []
    for i in coords:
        r, c = divmod(i, width)
        for j, ok in (
            (i + 1, c + 1 < width),
            (i + width, r + 1 < width),
            (i + width + 1, c + 1 < width and r + 1 < width),
        ):
            if not ok:
                continue
            for u, v in ((i, j), (j, i)) if directed else ((i, j),):
                if directed and rng.random() < 0.15:
                    continue  # some one-way streets
                line = LineString([coords[u], coords[v]])
                factor = float(rng.uniform(1.0, 3.0))
                rows.append(
                    {"u": u, "v": v, "geometry": line, "length_meter": line.length * factor, "time_min": factor}
                )
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    if zero_every:
        edges.loc[edges.index[::zero_every], ["length_meter", "time_min"]] = 0.0
    nodes = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords.values()], index=pd.Index(list(coords), name="node"), crs=CRS
    )
    return UrbanGraph(nodes, edges, is_multigraph=False, is_directed=directed, crs=CRS, graph_type="walk")


def _reference(graph, weight):
    graph.update_adjacency_matrix(weight=weight)
    return scipy_dijkstra(graph.adjacency_matrix, directed=graph.is_directed)


def _path_cost(graph, path, weight):
    return float(path_to_edges(graph, path, weight=weight)[weight].sum()) if len(path) > 1 else 0.0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("weight", ["length_meter", "time_min"])
def test_shortest_path_matches_scipy(method, directed, weight):
    graph = _grid_graph(directed=directed, seed=1)
    expected = _reference(graph, weight)
    nodes = graph.adjacency_nodelist
    for s in range(0, len(nodes), 7):
        for t in range(0, len(nodes), 5):
            cost, path = shortest_path(graph, nodes[s], nodes[t], weight=weight, method=method)
            if np.isinf(expected[s, t]):
                assert (cost, path) == (float("inf"), [])
                continue
            assert cost == pytest.approx(expected[s, t], rel=1e-5)
            assert path[0] == nodes[s] and path[-1] == nodes[t]
            assert _path_cost(graph, path, weight) == pytest.approx(cost, rel=1e-5)


@pytest.mark.parametrize("method", ["bidirectional", "astar", "alt"])
def test_zero_cost_edges_keep_astar_and_bidirectional_exact(method):
    graph = _grid_graph(directed=True, seed=2, zero_every=11)
    expected = _reference(graph, "time_min")
    nodes = graph.adjacency_nodelist
    for s in range(0, len(nodes), 9):
        for t in range(0, len(nodes), 4):
            cost, _ = shortest_path(graph, nodes[s], nodes[t], weight="time_min", method=method)
            assert cost == pytest.approx(expected[s, t], rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_cutoff_unreachable_and_same_node(method):
    graph = undirected_line_graph()
    assert shortest_path(graph, 0, 2, method=method) == (3.0, [0, 1, 2])
    assert shortest_path(graph, 0, 2, cutoff=3, method=method) == (3.0, [0, 1, 2])
    assert shortest_path(graph, 0, 2, cutoff=2.9, method=method) == (float("inf"), [])
    assert shortest_path(graph, 1, 1, cutoff=0, method=method) == (0.0, [1])
    assert shortest_path(disconnected_graph(), 0, 10, method=method) == (float("inf"), [])
    directed = directed_oneway_graph()
    assert shortest_path(directed, 3, 0, method=method) == (float("inf"), [])
    assert shortest_path(directed, 0, 3, method=method) == (7.0, [0, 1, 2, 3])


def test_single_source_dijkstra_path_keeps_its_meaning():
    directed = directed_oneway_graph()
    assert single_source_dijkstra_path(directed, 0, 3) == [0, 1, 2, 3]
    # reverse on a directed graph: the route from the target to the source along the edges.
    assert single_source_dijkstra_path(directed, 3, 0, reverse=True) == [0, 1, 2, 3]
    assert single_source_dijkstra_path(directed, 3, 0) == []
    assert single_source_dijkstra_path(undirected_line_graph(), 0, 3, reverse=True) == [0, 1, 2, 3]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("storage", ["eager", "lazy"])
def test_pairwise_methods_agree(method, storage):
    graph = _grid_graph(seed=3)
    rng = np.random.default_rng(0)
    nodes = np.asarray(graph.nodes_gdf.index)
    origins, destinations = rng.choice(nodes, 40), rng.choice(nodes, 40)
    params = dict(origins_nodes=origins, destination_nodes=destinations, mode="pairwise", cutoff=6.0)
    reference = od_matrix(graph, **params, method="dijkstra")
    result = od_matrix(graph, **params, method=method, output="both", path_storage=storage)
    np.testing.assert_allclose(
        result.distances["dist"].sparse.to_dense().to_numpy(),
        reference["dist"].sparse.to_dense().to_numpy(),
        rtol=1e-5,
    )
    assert list(result.distances.index) == list(reference.index)
    for row, (origin, destination) in enumerate(result.distances.index):
        path = result.get_path_at(row)
        dist = result.distances["dist"].iloc[row]
        if np.isinf(dist):
            assert path is None
        else:
            assert path[0] == origin and path[-1] == destination
            assert _path_cost(graph, path, "time_min") == pytest.approx(dist, rel=1e-5)
    paths = result.paths
    assert paths.shape == (40, 1)
    assert result.get_path(*result.distances.index[0]) == paths["path"].iloc[0] or np.isinf(
        result.distances["dist"].iloc[0]
    )


def test_auto_method_picks_by_repeated_origins(monkeypatch):
    graph = _grid_graph(seed=4)
    calls = []
    original = routing._kernel_pairs

    def spy(*args, **kwargs):
        calls.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(routing, "_kernel_pairs", spy)
    nodes = list(graph.nodes_gdf.index)
    od_matrix(graph, origins_nodes=nodes[:10], destination_nodes=nodes[10:20], mode="pairwise")
    assert calls == ["bidirectional"]
    calls.clear()
    # The same origin for every pair: one search that stops at all destinations.
    od_matrix(graph, origins_nodes=[nodes[0]] * 10, destination_nodes=nodes[10:20], mode="pairwise")
    assert calls == []


def test_methods_are_validated():
    graph = undirected_line_graph()
    with pytest.raises(ValueError, match="method must be one of"):
        shortest_path(graph, 0, 1, method="yen")
    with pytest.raises(ValueError, match="all_to_all"):
        od_matrix(graph, origins_nodes=[0], destination_nodes=[1], method="astar")
    with pytest.raises(ValueError, match="method must be 'auto'"):
        od_matrix(graph, origins_nodes=[0], destination_nodes=[1], mode="pairwise", method="fast")
    with pytest.raises(ValueError, match="absent"):
        shortest_path(graph, 0, 99)
    lines = graph.copy()
    lines.nodes_gdf = lines.nodes_gdf.set_geometry(lines.nodes_gdf.geometry.buffer(1))
    with pytest.raises(ValueError, match="point geometry"):
        shortest_path(lines, 0, 3, method="astar")


def _simple_paths(graph, source, target, weight, k):
    nx = pytest.importorskip("networkx")
    digraph = nx.DiGraph()
    for u, v, w in graph.edges_gdf[["u", "v", weight]].itertuples(index=False):
        pairs = ((u, v),) if graph.is_directed else ((u, v), (v, u))
        for a, b in pairs:
            if not digraph.has_edge(a, b) or digraph[a][b]["w"] > w:
                digraph.add_edge(a, b, w=w)
    return [
        (nx.path_weight(digraph, path, "w"), path)
        for path in islice(nx.shortest_simple_paths(digraph, source, target, "w"), k)
    ]


@pytest.mark.parametrize("directed", [True, False])
def test_k_shortest_paths_matches_networkx(directed):
    graph = _grid_graph(width=5, directed=directed, seed=5)
    nodes = list(graph.nodes_gdf.index)
    for source, target in ((nodes[0], nodes[-1]), (nodes[7], nodes[16]), (nodes[20], nodes[3])):
        expected = _simple_paths(graph, source, target, "length_meter", 6)
        result = k_shortest_paths(graph, source, target, 6, weight="length_meter")
        assert list(result.columns) == ["dist", "path"] and result.index.name == "rank"
        count = min(6, len(expected))
        assert len(result) == count
        np.testing.assert_allclose(result["dist"], [cost for cost, _ in expected[:count]], rtol=1e-5)
        assert result["dist"].is_monotonic_increasing
        for path, dist in zip(result["path"], result["dist"]):
            assert path[0] == source and path[-1] == target and len(set(path)) == len(path)
            assert _path_cost(graph, path, "length_meter") == pytest.approx(dist, rel=1e-5)
        assert len({tuple(path) for path in result["path"]}) == len(result)


def test_k_shortest_paths_edges_and_method():
    graph = undirected_line_graph()
    assert graph.k_shortest_paths(0, 3, 3)["path"].tolist() == [[0, 1, 2, 3]]
    assert k_shortest_paths(graph, 2, 2, 2)["path"].tolist() == [[2]]
    assert k_shortest_paths(graph, 0, 3, 2, cutoff=6.9).empty
    assert k_shortest_paths(disconnected_graph(), 0, 10, 2).empty
    with pytest.raises(ValueError, match="k must be >= 1"):
        k_shortest_paths(graph, 0, 3, 0)
    with pytest.raises(TypeError, match="k must be int"):
        k_shortest_paths(graph, 0, 3, 2.0)
    assert graph.shortest_path(0, 3, method="astar") == (7.0, [0, 1, 2, 3])


def _ladder():
    """Two parallel streets 0-1-2-3 (cheap) and 4-5-6-7, joined at both ends and by rungs 1-5, 2-6."""
    coords = {i: (100.0 * (i % 4), 0.0 if i < 4 else 50.0) for i in range(8)}
    streets = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (4, 5, 1.2), (5, 6, 1.2), (6, 7, 1.2)]
    streets += [(0, 4, 0.5), (3, 7, 0.5), (1, 5, 0.5), (2, 6, 0.5)]
    rows = [
        {"u": u, "v": v, "geometry": LineString([coords[u], coords[v]]), "length_meter": 100.0 * w, "time_min": w}
        for u, v, w in streets
    ]
    nodes = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords.values()], index=pd.Index(list(coords), name="node"), crs=CRS
    )
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    return UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False, crs=CRS, graph_type="walk")


def test_alternative_paths_take_other_streets():
    graph = _ladder()
    result = alternative_paths(graph, 0, 3, 3, penalty=2.0, max_overlap=0.5, max_cost_ratio=2.0)
    assert list(result.columns) == ["dist", "overlap", "path"] and result.index.name == "rank"
    assert result["path"].iloc[0] == [0, 1, 2, 3] and result["dist"].iloc[0] == pytest.approx(3.0)
    assert [0, 4, 5, 6, 7, 3] in result["path"].tolist()
    assert result["dist"].is_monotonic_increasing
    assert result["overlap"].between(0, 1).all()
    for path, dist in zip(result["path"], result["dist"]):
        assert _path_cost(graph, path, "time_min") == pytest.approx(dist)
    # Limits: the cost ratio keeps only the shortest route, the cutoff even that one.
    assert len(graph.alternative_paths(0, 3, 3, max_cost_ratio=1.2)) == 1
    assert alternative_paths(graph, 0, 3, 3, cutoff=2.9).empty
    assert alternative_paths(graph, 2, 2)["path"].tolist() == [[2]]
    assert alternative_paths(disconnected_graph(), 0, 10).empty


def test_alternative_paths_on_a_directed_graph_match_the_undirected_one():
    graph = _grid_graph(width=7, directed=False, seed=8)
    directed = graph.copy()
    directed.is_directed = True
    directed.edges_gdf = pd.concat(
        [graph.edges_gdf, graph.edges_gdf.rename(columns={"u": "v", "v": "u"})], ignore_index=True
    )
    directed.update_adjacency_matrix(weight="time_min")
    nodes = list(graph.nodes_gdf.index)
    a = alternative_paths(graph, nodes[0], nodes[-1], 4)
    b = alternative_paths(directed, nodes[0], nodes[-1], 4)
    assert a["path"].tolist() == b["path"].tolist()
    np.testing.assert_allclose(a["dist"], b["dist"], rtol=1e-6)


def test_alternative_paths_are_validated():
    graph = undirected_line_graph()
    with pytest.raises(ValueError, match="penalty"):
        alternative_paths(graph, 0, 3, penalty=1.0)
    with pytest.raises(ValueError, match="max_overlap"):
        alternative_paths(graph, 0, 3, max_overlap=1.5)
    with pytest.raises(ValueError, match="max_cost_ratio"):
        alternative_paths(graph, 0, 3, max_cost_ratio=0.9)
    with pytest.raises(ValueError, match="k must be"):
        alternative_paths(graph, 0, 3, 0)


def test_alt_landmarks_are_cached_with_the_matrix():
    graph = _grid_graph(seed=9)
    nodes = list(graph.nodes_gdf.index)
    graph.shortest_path(nodes[0], nodes[-1], method="alt")
    tables = graph._csr_cache[routing._LANDMARKS]  # pylint: disable=protected-access
    first = tables[routing.ALT_LANDMARKS]
    assert 0 < len(first.nodes) <= routing.ALT_LANDMARKS and first.nbytes > 0
    graph.shortest_path(nodes[1], nodes[-2], method="alt")
    assert graph._csr_cache[routing._LANDMARKS][routing.ALT_LANDMARKS] is first  # pylint: disable=protected-access
    # Another weight builds another matrix and new tables.
    graph.shortest_path(nodes[0], nodes[-1], weight="length_meter", method="alt")
    assert graph._csr_cache[routing._LANDMARKS][routing.ALT_LANDMARKS] is not first  # pylint: disable=protected-access
