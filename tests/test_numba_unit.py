import numpy as np
import pytest
from scipy import sparse

from iduedu._numba.components import connected_components_numba, strongly_connected_components_numba
from iduedu._numba.csr import sparse_row2numba_bool_matrix, sparse_row2numba_matrix
from iduedu._numba.shortest_paths import (
    dijkstra_numba_search,
    dijkstra_numba_search_parallel,
    reconstruct_numba_compact_path,
)

pytestmark = [pytest.mark.unit, pytest.mark.numba]


def _weighted_matrix():
    return sparse.csr_matrix(
        np.array(
            [
                [0, 1, 5, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )


def _settled(search):
    return {int(node): float(cost) for node, cost in zip(search[0], search[1])}


def _search(matrix, sources, targets=(), cutoff=10, store_predecessors=False):
    return dijkstra_numba_search(
        matrix,
        np.array(sources, dtype=np.int32),
        np.array(targets, dtype=np.int32),
        np.float32(cutoff),
        store_predecessors,
    )


def test_csr_helpers_convert_scipy_rows():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    assert matrix.tot_rows == 4
    assert matrix.get_cols(0).tolist() == [1, 2]
    assert matrix.get_vals(0).tolist() == [1, 5]
    assert matrix.get_vals(0).dtype == np.float32


def test_connected_components_numba_labels_disconnected_boolean_graph():
    adjacency = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        )
    )
    labels = connected_components_numba(sparse_row2numba_bool_matrix(adjacency))

    assert labels.tolist() == [0, 0, 0, 1]


def test_strongly_connected_components_numba_labels_directed_components():
    adjacency = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        )
    )
    labels = strongly_connected_components_numba(
        sparse_row2numba_bool_matrix(adjacency),
        sparse_row2numba_bool_matrix(adjacency.T.tocsr()),
    )

    components = {}
    for pos, label in enumerate(labels.tolist()):
        components.setdefault(label, set()).add(pos)
    assert {frozenset(component) for component in components.values()} == {frozenset({0, 1}), frozenset({2, 3})}


def test_dijkstra_numba_search_respects_shortest_paths_and_cutoff():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    assert _settled(_search(matrix, [0])) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 4.0}
    assert _settled(_search(matrix, [0], cutoff=2)) == {0: 0.0, 1: 1.0}
    assert _settled(_search(matrix, [0], targets=[2, 3])) == {2: 3.0, 3: 4.0}


def test_dijkstra_numba_search_returns_nearest_source_owners():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    nodes, costs, _, owners = _search(matrix, [0, 3])

    assert dict(zip(nodes.tolist(), zip(owners.tolist(), costs.tolist()))) == {
        0: (0, 0.0),
        1: (0, 1.0),
        2: (0, 3.0),
        3: (3, 0.0),
    }


def test_compact_path_reconstruction_handles_direction_cutoff_and_self_route():
    matrix = sparse_row2numba_matrix(_weighted_matrix())
    nodes, _, parents, _ = _search(matrix, [0], store_predecessors=True)
    clipped, _, clipped_parents, _ = _search(matrix, [0], cutoff=3, store_predecessors=True)

    assert reconstruct_numba_compact_path(nodes, parents, np.int32(3), False).tolist() == [0, 1, 2, 3]
    assert reconstruct_numba_compact_path(nodes, parents, np.int32(3), True).tolist() == [3, 2, 1, 0]
    assert reconstruct_numba_compact_path(nodes, parents, np.int32(0), False).tolist() == [0]
    assert reconstruct_numba_compact_path(clipped, clipped_parents, np.int32(3), False).tolist() == []


def test_parallel_search_uses_per_source_target_slices():
    matrix = sparse_row2numba_matrix(_weighted_matrix())
    targets = np.array([2, 3, 1], dtype=np.int32)

    searches = dijkstra_numba_search_parallel(
        matrix,
        np.array([0, 1, 3], dtype=np.int32),
        targets,
        np.array([0, 2, 3], dtype=np.int64),
        np.array([2, 3, 3], dtype=np.int64),
        np.float32(10),
        False,
    )

    # An empty slice (the last source) means all reachable nodes.
    assert [_settled(search) for search in searches] == [{2: 3.0, 3: 4.0}, {1: 0.0}, {3: 0.0}]
