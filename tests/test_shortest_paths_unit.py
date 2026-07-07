import numpy as np
import pandas as pd
import pytest

from iduedu.graph.shortest_paths import (
    dijkstra_path_length_parallel,
    multi_source_dijkstra_nearest_source,
    multi_source_dijkstra_path_length,
    od_matrix,
    single_source_dijkstra_path_length,
)
from tests.factories import directed_oneway_graph, undirected_line_graph

pytestmark = [pytest.mark.unit, pytest.mark.numba]


def _dense(series: pd.Series) -> dict:
    return {key: float(value) for key, value in series.sparse.to_dense().items()}


def test_single_source_dijkstra_returns_expected_distances():
    graph = undirected_line_graph()

    result = single_source_dijkstra_path_length(graph, 0, weight="time_min")

    assert _dense(result) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0}


def test_single_source_dijkstra_respects_cutoff():
    graph = undirected_line_graph()

    result = single_source_dijkstra_path_length(graph, 0, weight="time_min", cutoff=3.0)

    assert _dense(result) == {0: 0.0, 1: 1.0, 2: 3.0}
    assert 3 not in result.index


def test_reverse_dijkstra_uses_transposed_directed_graph():
    graph = directed_oneway_graph()

    forward = single_source_dijkstra_path_length(graph, 3, weight="time_min")
    reverse = single_source_dijkstra_path_length(graph, 3, weight="time_min", reverse=True)

    assert _dense(forward) == {3: 0.0}
    assert _dense(reverse) == {0: 7.0, 1: 6.0, 2: 4.0, 3: 0.0}


def test_multi_source_dijkstra_returns_nearest_distance_and_source():
    graph = undirected_line_graph()

    distances = multi_source_dijkstra_path_length(graph, source_nodes=[0, 3], weight="time_min")
    nearest = multi_source_dijkstra_nearest_source(graph, source_nodes=[0, 3], weight="time_min")

    assert _dense(distances) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 0.0}
    assert nearest.loc[0, "source_node"] == 0
    assert nearest.loc[2, "source_node"] == 0
    assert nearest.loc[3, "source_node"] == 3
    assert float(nearest.loc[2, "dist"]) == 3.0


def test_dijkstra_path_length_parallel_keeps_input_source_index():
    graph = undirected_line_graph()
    sources = pd.DataFrame({"graph_node_id": [0, 3]}, index=["left", "right"])

    result = dijkstra_path_length_parallel(graph, gdf_sources=sources, weight="time_min", max_workers=1)

    assert result.index.tolist() == ["left", "right"]
    assert result.columns.tolist() == [0, 1, 2, 3]
    assert float(result.loc["left", 2]) == 3.0
    assert float(result.loc["right", 1]) == 6.0
    assert result.attrs["source_nodes"].to_dict() == {"left": 0, "right": 3}


def test_od_matrix_uses_smaller_side_transpose_path_and_preserves_labels():
    graph = undirected_line_graph()
    origins = pd.DataFrame({"graph_node_id": [0, 1, 3]}, index=["o0", "o1", "o3"])
    destinations = pd.DataFrame({"graph_node_id": [2]}, index=["d2"])

    result = od_matrix(graph, gdf_origins=origins, gdf_destinations=destinations, weight="time_min", max_workers=1)

    assert result.index.tolist() == ["o0", "o1", "o3"]
    assert result.columns.tolist() == ["d2"]
    assert np.allclose(result.sparse.to_dense()["d2"].to_numpy(), np.array([3.0, 2.0, 4.0]))


def test_shortest_paths_reject_missing_source_node():
    graph = undirected_line_graph()

    with pytest.raises(ValueError, match="absent in graph"):
        single_source_dijkstra_path_length(graph, 999)
