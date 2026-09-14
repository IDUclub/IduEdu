import numpy as np
import pandas as pd
import pytest

from iduedu.graph.shortest_paths import (
    dijkstra_path_length_parallel,
    multi_source_dijkstra_nearest_source,
    multi_source_dijkstra_path,
    multi_source_dijkstra_path_length,
    od_matrix,
    path_to_edges,
    single_source_dijkstra_path,
    single_source_dijkstra_path_length,
)
from tests.factories import directed_oneway_graph, multigraph_with_parallel_edges, undirected_line_graph

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


def test_single_source_dijkstra_path_returns_route_and_handles_cutoff():
    graph = undirected_line_graph()

    route = single_source_dijkstra_path(graph, 0, 3, weight="time_min")
    cutoff_route = single_source_dijkstra_path(graph, 0, 3, weight="time_min", cutoff=3.0)
    same_node_route = single_source_dijkstra_path(graph, 2, 2, cutoff=0.0)

    assert route == [0, 1, 2, 3]
    assert cutoff_route == []
    assert same_node_route == [2]


def test_single_source_dijkstra_path_reverse_returns_original_graph_direction():
    graph = directed_oneway_graph()

    route = single_source_dijkstra_path(graph, 3, 0, reverse=True)

    assert route == [0, 1, 2, 3]


def test_multi_source_dijkstra_path_all_to_all_returns_sparse_path_matrix():
    graph = undirected_line_graph()

    result = multi_source_dijkstra_path(
        graph,
        origins_nodes=[0, 3],
        destination_nodes=[1, 2],
        mode="all_to_all",
        max_workers=1,
    )

    assert result.index.tolist() == [0, 3]
    assert result.columns.tolist() == [1, 2]
    assert result.loc[0, 1] == [0, 1]
    assert result.loc[0, 2] == [0, 1, 2]
    assert result.loc[3, 1] == [3, 2, 1]
    assert result.loc[3, 2] == [3, 2]
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in result.dtypes)
    assert result.attrs["origin_nodes"].to_dict() == {0: 0, 3: 3}
    assert result.attrs["destination_nodes"].to_dict() == {1: 1, 2: 2}


def test_multi_source_dijkstra_path_transposes_directed_search_and_restores_paths():
    graph = directed_oneway_graph()

    result = multi_source_dijkstra_path(
        graph,
        origins_nodes=[0, 1, 2],
        destination_nodes=[3],
        mode="all_to_all",
        max_workers=1,
    )

    assert result.loc[0, 3] == [0, 1, 2, 3]
    assert result.loc[1, 3] == [1, 2, 3]
    assert result.loc[2, 3] == [2, 3]


def test_multi_source_dijkstra_path_pairwise_preserves_object_labels():
    graph = undirected_line_graph()
    origins = pd.DataFrame({"graph_node_id": [0, 1]}, index=["o0", "o1"])
    destinations = pd.DataFrame({"graph_node_id": [2, 3]}, index=["d2", "d3"])

    result = multi_source_dijkstra_path(
        graph,
        gdf_origins=origins,
        gdf_destinations=destinations,
        mode="pairwise",
        max_workers=1,
    )

    assert result.index.tolist() == [("o0", "d2"), ("o1", "d3")]
    assert result["path"].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert isinstance(result["path"].dtype, pd.SparseDtype)
    assert result.attrs["origin_nodes"].to_dict() == {"o0": 0, "o1": 1}
    assert result.attrs["destination_nodes"].to_dict() == {"d2": 2, "d3": 3}


def test_multi_source_dijkstra_path_matches_geodataframes_to_nearest_nodes():
    graph = undirected_line_graph()
    origins = graph.nodes_gdf.loc[[0, 3]].copy()
    destinations = graph.nodes_gdf.loc[[1, 2]].copy()
    origins.index = ["left", "right"]
    destinations.index = ["near-left", "near-right"]

    result = multi_source_dijkstra_path(
        graph,
        gdf_origins=origins,
        gdf_destinations=destinations,
        mode="pairwise",
        max_workers=1,
    )

    assert result.index.tolist() == [("left", "near-left"), ("right", "near-right")]
    assert result["path"].tolist() == [[0, 1], [3, 2]]


def test_multi_source_dijkstra_path_threshold_uses_sparse_missing_values():
    graph = undirected_line_graph()

    result = multi_source_dijkstra_path(
        graph,
        origins_nodes=[0],
        destination_nodes=[1, 2, 3],
        threshold=3.0,
        max_workers=1,
    )

    assert result.loc[0, 1] == [0, 1]
    assert result.loc[0, 2] == [0, 1, 2]
    assert pd.isna(result.loc[0, 3])


def test_multi_source_dijkstra_path_validates_mode_and_pair_count():
    graph = undirected_line_graph()

    with pytest.raises(ValueError, match="mode"):
        multi_source_dijkstra_path(graph, origins_nodes=[0], destination_nodes=[1], mode="unknown")
    with pytest.raises(ValueError, match="same number"):
        multi_source_dijkstra_path(
            graph,
            origins_nodes=[0, 1],
            destination_nodes=[2],
            mode="pairwise",
        )


def test_path_to_edges_uses_minimum_parallel_edge_and_preserves_attributes():
    graph = multigraph_with_parallel_edges()

    route_edges = path_to_edges(graph, [0, 1, 2], weight="time_min")

    assert route_edges["path_order"].tolist() == [0, 1]
    assert route_edges["path_u"].tolist() == [0, 1]
    assert route_edges["path_v"].tolist() == [1, 2]
    assert route_edges["k"].tolist() == [1, 0]
    assert route_edges["time_min"].tolist() == [1.0, 2.0]
    assert route_edges.crs == graph.edges_gdf.crs


def test_path_to_edges_reverses_undirected_geometry_and_rejects_disallowed_edge():
    undirected_graph = undirected_line_graph()
    directed_graph = directed_oneway_graph()

    route_edges = path_to_edges(undirected_graph, [2, 1, 0])

    assert list(route_edges.geometry.iloc[0].coords) == [(20.0, 0.0), (10.0, 0.0)]
    assert list(route_edges.geometry.iloc[1].coords) == [(10.0, 0.0), (0.0, 0.0)]
    with pytest.raises(ValueError, match="absent or disallowed"):
        path_to_edges(directed_graph, [3, 2])
