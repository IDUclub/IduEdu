import numpy as np
import pytest

from iduedu.graph.adjacency import build_adjacency_matrix
from tests.factories import directed_oneway_graph, multigraph_with_parallel_edges, undirected_line_graph

pytestmark = pytest.mark.unit


def test_undirected_adjacency_is_symmetric():
    graph = undirected_line_graph()
    matrix = graph.to_csr(nodelist=[0, 1, 2, 3], weight="time_min").toarray()

    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0
    assert matrix[1, 2] == 2.0
    assert matrix[2, 1] == 2.0
    assert matrix[2, 3] == 4.0
    assert matrix[3, 2] == 4.0
    assert matrix[0, 3] == 0.0


def test_edge_direction_column_adds_reverse_edges_only_for_non_oneway():
    graph = directed_oneway_graph()
    matrix = graph.to_csr(nodelist=[0, 1, 2, 3], weight="time_min").toarray()

    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 0.0
    assert matrix[1, 2] == 2.0
    assert matrix[2, 1] == 2.0
    assert matrix[2, 3] == 4.0
    assert matrix[3, 2] == 0.0


def test_multigraph_adjacency_uses_min_or_max_parallel_weight():
    graph = multigraph_with_parallel_edges()

    min_matrix = graph.to_csr(nodelist=[0, 1, 2], weight="time_min", multigraph_rule="min").toarray()
    max_matrix = graph.to_csr(nodelist=[0, 1, 2], weight="time_min", multigraph_rule="max").toarray()

    assert min_matrix[0, 1] == 1.0
    assert min_matrix[1, 0] == 1.0
    assert max_matrix[0, 1] == 5.0
    assert max_matrix[1, 0] == 5.0


def test_adjacency_filters_edges_outside_nodelist():
    graph = undirected_line_graph()
    matrix = graph.to_csr(nodelist=[0, 1], weight="time_min").toarray()

    assert matrix.shape == (2, 2)
    assert np.array_equal(matrix, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_adjacency_rejects_missing_weight():
    graph = undirected_line_graph()

    with pytest.raises(KeyError, match="weight"):
        graph.to_csr(weight="missing_weight")


def test_adjacency_rejects_nan_weight():
    graph = undirected_line_graph()
    graph.edges_gdf.loc[0, "time_min"] = np.nan

    with pytest.raises(ValueError, match="contains NaN"):
        build_adjacency_matrix(graph, nodelist=[0, 1, 2, 3], weight="time_min")
