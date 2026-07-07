import numpy as np
import pytest
from scipy import sparse

from iduedu._numba.components import connected_components_numba, strongly_connected_components_numba
from iduedu._numba.csr import coo_rows_to_arrays, sparse_row2numba_bool_matrix, sparse_row2numba_matrix
from iduedu._numba.shortest_paths import (
    dijkstra_numba_od_parallel,
    dijkstra_numba_path_length_parallel,
    multi_source_dijkstra_numba_nearest_source,
    multi_source_dijkstra_numba_path_length,
    single_source_dijkstra_numba_path_length,
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


def _row_to_dict(row):
    return {int(col): float(value) for col, value in row}


def _triplet_row_to_dict(row):
    return {int(node): (int(source), float(distance)) for node, source, distance in row}


def test_numba_jit_coverage_is_enabled_before_numba_import():
    from numba.core import config

    assert config.JIT_COVERAGE == 1


def test_csr_helpers_convert_scipy_rows_and_extract_coo_arrays():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    assert matrix.tot_rows == 4
    assert matrix.get_cols(0).tolist() == [1, 2]
    assert matrix.get_vals(0).tolist() == [1, 5]

    row = single_source_dijkstra_numba_path_length(matrix, np.int32(0), np.float32(10))
    rows, cols, values = coo_rows_to_arrays([row])

    assert rows.tolist() == [0, 0, 0, 0]
    assert cols.tolist() == [0, 1, 2, 3]
    assert values.tolist() == [0.0, 1.0, 3.0, 4.0]
    assert values.dtype == np.float32


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


def test_single_source_dijkstra_numba_respects_shortest_paths_and_cutoff():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    full = single_source_dijkstra_numba_path_length(matrix, np.int32(0), np.float32(10))
    cutoff = single_source_dijkstra_numba_path_length(matrix, np.int32(0), np.float32(2))

    assert _row_to_dict(full) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 4.0}
    assert _row_to_dict(cutoff) == {0: 0.0, 1: 1.0}


def test_multi_source_dijkstra_numba_returns_nearest_distances():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    row = multi_source_dijkstra_numba_path_length(matrix, np.array([0, 3], dtype=np.int32), np.float32(10))

    assert _row_to_dict(row) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 0.0}


def test_multi_source_dijkstra_numba_returns_nearest_source_with_distance():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    row = multi_source_dijkstra_numba_nearest_source(matrix, np.array([0, 3], dtype=np.int32), np.float32(10))

    assert _triplet_row_to_dict(row) == {0: (0, 0.0), 1: (0, 1.0), 2: (0, 3.0), 3: (3, 0.0)}


def test_parallel_od_numba_returns_destination_columns_per_origin():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    rows = dijkstra_numba_od_parallel(
        matrix,
        np.array([0, 3], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.float32(10),
    )

    assert [_row_to_dict(row) for row in rows] == [{0: 3.0, 1: 4.0}, {1: 0.0}]


def test_parallel_path_length_numba_returns_rows_per_origin():
    matrix = sparse_row2numba_matrix(_weighted_matrix())

    rows = dijkstra_numba_path_length_parallel(matrix, np.array([0, 3], dtype=np.int32), np.float32(10))

    assert [_row_to_dict(row) for row in rows] == [{0: 0.0, 1: 1.0, 2: 3.0, 3: 4.0}, {3: 0.0}]
