import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components as scipy_components
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from iduedu import UrbanGraph, _rust, single_source_dijkstra
from iduedu.graph import routing
from tests.factories import CRS, _edges, _nodes

pytestmark = [pytest.mark.unit, pytest.mark.rust]

INF = np.float32(np.inf)


def _random_matrix(n=60, density=0.08, seed=0):
    rng = np.random.default_rng(seed)
    matrix = sparse.random(n, n, density=density, random_state=rng, format="csr", dtype=np.float32)
    matrix.data = (matrix.data * 10).astype(np.float32)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _graph(matrix):
    return _rust.CsrGraph(
        matrix.indptr.astype(np.int32), matrix.indices.astype(np.int32), matrix.data.astype(np.float32)
    )


def _nodes_array(*values):
    return np.array(values, dtype=np.int32)


def test_search_matches_scipy_for_every_source():
    matrix = _random_matrix()
    graph = _graph(matrix)
    expected = scipy_dijkstra(matrix.astype(np.float64), directed=True)
    for source in range(matrix.shape[0]):
        nodes, dist, parents, owners = graph.search(_nodes_array(source), _nodes_array(), INF, False)
        row = np.full(matrix.shape[0], np.inf)
        row[nodes] = dist
        np.testing.assert_allclose(row, expected[source], rtol=1e-5)
        assert parents.size == 0
        assert set(owners.tolist()) <= {source}


def test_cutoff_targets_and_predecessors():
    matrix = _random_matrix(seed=3)
    graph = _graph(matrix)
    expected = scipy_dijkstra(matrix.astype(np.float64), directed=True, indices=0)
    cutoff = np.float32(np.nanmedian(expected[np.isfinite(expected)]))
    nodes, dist, _, _ = graph.search(_nodes_array(0), _nodes_array(), cutoff, False)
    assert (dist <= cutoff).all()
    assert set(nodes.tolist()) == set(np.flatnonzero(expected <= cutoff).tolist())

    reachable = np.flatnonzero(np.isfinite(expected))
    targets = reachable[-3:].astype(np.int32)
    nodes, dist, parents, _ = graph.search(_nodes_array(0), targets, INF, False)
    assert nodes.tolist() == sorted(targets.tolist())
    assert parents.size == 0

    tree_nodes, tree_dist, tree_parents, _ = graph.search(_nodes_array(0), targets, INF, True)
    assert set(targets.tolist()) <= set(tree_nodes.tolist())
    flat, offsets = _rust.reconstruct_paths(tree_nodes, tree_parents, targets, False)
    for k, target in enumerate(targets):
        route = flat[offsets[k] : offsets[k + 1]]
        assert route[0] == 0 and route[-1] == target
        cost = sum(matrix[a, b] for a, b in zip(route[:-1], route[1:]))
        assert cost == pytest.approx(tree_dist[np.searchsorted(tree_nodes, target)], rel=1e-5)
    reverse_flat, _ = _rust.reconstruct_paths(tree_nodes, tree_parents, targets[:1], True)
    assert reverse_flat.tolist() == flat[offsets[0] : offsets[1]][::-1].tolist()


def test_multi_source_owner_is_the_nearest_source_with_ties_to_the_lower_position():
    # 0 - 1 - 2 - 3 - 4 with unit weights; sources 0 and 4 tie at node 2.
    matrix = sparse.diags([1, 1, 1, 1], 1, shape=(5, 5), format="csr", dtype=np.float32)
    matrix = (matrix + matrix.T).tocsr()
    nodes, dist, _, owners = _graph(matrix).search(_nodes_array(4, 0), _nodes_array(), INF, False)
    assert nodes.tolist() == [0, 1, 2, 3, 4]
    assert dist.tolist() == [0, 1, 2, 1, 0]
    assert owners.tolist() == [0, 0, 0, 4, 4]


def test_search_many_and_distance_matrix_agree_with_single_searches():
    matrix = _random_matrix(seed=7)
    graph = _graph(matrix)
    sources = _nodes_array(0, 5, 9, 5)
    targets = _nodes_array(1, 2, 3, 40, 59)
    dense = graph.distance_matrix(sources, targets, INF, 2)
    starts = np.array([0, 0, 2, 5], dtype=np.int64)
    ends = np.array([5, 2, 5, 5], dtype=np.int64)
    many = graph.search_many(sources, targets, starts, ends, INF, False, None)
    expected = scipy_dijkstra(matrix.astype(np.float64), directed=True, indices=sources)[:, targets]
    np.testing.assert_allclose(dense, expected, rtol=1e-5)
    for i, (nodes, dist, _, _) in enumerate(many):
        wanted = targets[starts[i] : ends[i]] if ends[i] > starts[i] else np.arange(matrix.shape[0])
        row = dict(zip(nodes.tolist(), dist.tolist()))
        for target in wanted:
            value = expected[i, list(targets).index(target)] if ends[i] > starts[i] else None
            if value is not None and np.isfinite(value):
                assert row[int(target)] == pytest.approx(value, rel=1e-5)
    assert graph.distance_matrix(sources, _nodes_array(), INF).shape == (4, 0)


def test_transpose_runs_reverse_searches():
    matrix = _random_matrix(seed=11)
    graph = _graph(matrix)
    reverse = graph.transpose()
    assert (reverse.n_nodes, reverse.n_arcs) == (graph.n_nodes, graph.n_arcs)
    expected = scipy_dijkstra(matrix.T.tocsr().astype(np.float64), directed=True, indices=3)
    nodes, dist, _, _ = reverse.search(_nodes_array(3), _nodes_array(), INF, False)
    np.testing.assert_allclose(dist, expected[nodes], rtol=1e-5)
    assert len(nodes) == np.isfinite(expected).sum()


def test_invalid_inputs_are_rejected():
    indptr, indices = np.array([0, 1, 1], dtype=np.int32), np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        _rust.CsrGraph(indptr, indices, np.array([-1], dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        _rust.CsrGraph(indptr, np.array([2], dtype=np.int32), np.array([1], dtype=np.float32))
    graph = _rust.CsrGraph(indptr, indices, np.array([1], dtype=np.float32))
    with pytest.raises(ValueError, match="sources out of range"):
        graph.search(_nodes_array(5), _nodes_array(), INF, False)
    with pytest.raises(ValueError, match="target slice"):
        graph.search_many(
            _nodes_array(0), _nodes_array(1), np.array([0], np.int64), np.array([3], np.int64), INF, False, None
        )
    with pytest.raises(ValueError, match="reduce"):
        _rust.build_csr(2, np.array([0], np.int64), np.array([1], np.int64), np.array([1], np.float32), "sum")


def test_build_csr_merges_repeated_arcs_instead_of_summing():
    rows = np.array([0, 0, 1, 0], dtype=np.int64)
    cols = np.array([1, 1, 0, 0], dtype=np.int64)
    data = np.array([3, 1, 2, 5], dtype=np.float32)
    indptr, indices, values = _rust.build_csr(2, rows, cols, data, "min")
    assert indptr.tolist() == [0, 2, 3]
    assert indices.tolist() == [0, 1, 0]
    assert values.tolist() == [5, 1, 2]
    assert _rust.build_csr(2, rows, cols, data, "max")[2].tolist() == [5, 3, 2]


def test_components_match_scipy():
    matrix = _random_matrix(n=80, density=0.02, seed=5)
    symmetric = ((matrix + matrix.T) > 0).astype(np.float32).tocsr()
    labels = _rust.connected_components(symmetric.indptr.astype(np.int32), symmetric.indices.astype(np.int32))
    count, expected = scipy_components(symmetric, directed=False)
    assert len(set(labels.tolist())) == count
    assert pd.crosstab(labels, expected).astype(bool).sum(axis=1).eq(1).all()

    reverse = matrix.T.tocsr()
    strong = _rust.strongly_connected_components(
        matrix.indptr.astype(np.int32),
        matrix.indices.astype(np.int32),
        reverse.indptr.astype(np.int32),
        reverse.indices.astype(np.int32),
    )
    count, expected = scipy_components(matrix, directed=True, connection="strong")
    assert len(set(strong.tolist())) == count
    assert pd.crosstab(strong, expected).astype(bool).sum(axis=1).eq(1).all()


def _parallel_opposite_edges_graph():
    # Two walkways between 0 and 1 stored in opposite orientations, then 1 -> 2.
    coords = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (20.0, 0.0)}
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "k": 0, "length_meter": 10.0, "time_min": 0.5},
            {"u": 1, "v": 0, "k": 0, "length_meter": 90.0, "time_min": 4.0},
            {"u": 1, "v": 2, "k": 0, "length_meter": 10.0, "time_min": 1.0},
        ],
    )
    return UrbanGraph(_nodes(coords), edges, is_multigraph=True, is_directed=False, crs=CRS, graph_type="walk")


def test_parallel_edges_of_opposite_orientation_keep_the_minimum_weight():
    graph = _parallel_opposite_edges_graph()
    matrix = graph.update_adjacency_matrix()
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[1, 0] == pytest.approx(0.5)
    assert graph.to_csr(multigraph_rule="max")[0, 1] == pytest.approx(4.0)
    distances = single_source_dijkstra(graph, 0, output="distances")
    assert distances.sparse.to_dense().to_dict() == {0: 0.0, 1: 0.5, 2: 1.5}


def test_rust_graph_is_cached_rebuilt_after_changes_and_not_pickled():
    graph = _parallel_opposite_edges_graph()
    first = routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False)
    assert routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False) is first
    graph.update_adjacency_matrix(weight="length_meter")
    assert routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False) is not first
    restored = pickle.loads(pickle.dumps(graph))
    assert restored._csr_cache is None
    assert single_source_dijkstra(restored, 0, output="distances").sparse.to_dense().to_dict()[2] == 1.5
