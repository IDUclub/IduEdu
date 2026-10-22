import numba as nb
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from iduedu import (
    ShortestPathResult,
    UrbanGraph,
    dijkstra_parallel,
    iter_od_matrix,
    multi_source_dijkstra,
    od_matrix,
    path_to_edges,
    single_source_dijkstra,
)
from iduedu._numba.shortest_paths import dijkstra_numba_search
from iduedu.graph import routing
from iduedu.graph.shortest_paths import (
    dijkstra_path_length_parallel,
    multi_source_dijkstra_nearest_source,
    multi_source_dijkstra_path,
    multi_source_dijkstra_path_length,
    single_source_dijkstra_path,
    single_source_dijkstra_path_length,
)
from tests.factories import (
    directed_oneway_graph,
    disconnected_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = [pytest.mark.unit, pytest.mark.numba]


def _dense(series: pd.Series) -> dict:
    return {key: float(value) for key, value in series.sparse.to_dense().items()}


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


@pytest.mark.parametrize(
    "factory,mode,storage",
    [
        pytest.param(directed_oneway_graph, "all_to_all", "eager", id="directed-matrix-eager"),
        pytest.param(undirected_line_graph, "all_to_all", "lazy", id="undirected-matrix-lazy"),
        pytest.param(disconnected_graph, "pairwise", "eager", id="unreachable-pairs-eager"),
        pytest.param(directed_oneway_graph, "pairwise", "lazy", id="directed-pairs-lazy"),
    ],
)
def test_joint_matches_separate_outputs(factory, mode, storage):
    graph = factory()
    params = dict(
        origins_nodes=[0, 1, 3] if 3 in graph.nodes_gdf.index else [0, 1, 10],
        destination_nodes=[2, 1, 0],
        mode=mode,
        cutoff=3,
        max_workers=1,
    )
    distances = graph.od_matrix(**params)
    paths = graph.od_matrix(**params, output="paths")
    result = graph.od_matrix(**params, output="both", path_storage=storage)

    assert isinstance(result, ShortestPathResult)
    assert result.paths_materialized == (storage == "eager")
    pd.testing.assert_frame_equal(result.distances, distances)
    pd.testing.assert_frame_equal(result.paths, paths)
    assert result.paths_materialized


@pytest.mark.parametrize("weight,cutoff,cost", [("time_min", 3, 3), ("length_meter", 20, 20)])
def test_cutoff_inclusive_and_legacy_threshold_alias(weight, cutoff, cost):
    graph = undirected_line_graph()
    params = dict(origins_nodes=[0], destination_nodes=[0, 2, 3], weight=weight)
    result = od_matrix(graph, **params, cutoff=cutoff, output="both", path_storage="lazy")
    assert result.distances.loc[0, 0] == 0
    assert result.get_path(0, 0) == [0]
    assert result.distances.loc[0, 2] == cost
    assert result.get_path(0, 2) == [0, 1, 2]
    assert np.isinf(result.distances.loc[0, 3])
    assert result.get_path(0, 3) is None
    pd.testing.assert_frame_equal(result.distances, od_matrix(graph, **params, threshold=cutoff))
    assert graph.multi_source_dijkstra_path(**params, cutoff=cutoff).loc[0, 2] == [0, 1, 2]


@pytest.mark.parametrize("factory", [undirected_line_graph, directed_oneway_graph])
def test_transposed_lazy_paths_and_snapshot(factory):
    graph = factory()
    result = graph.od_matrix(origins_nodes=[0, 1, 2], destination_nodes=[3], output="both", path_storage="lazy")
    # Lazy routes use saved node labels and parents, not the live graph.
    graph.nodes_gdf.index = ["a", "b", "c", "d"]
    graph.edges_gdf.loc[:, "time_min"] = 999
    graph.adjacency_nodelist[:] = ["a", "b", "c", "d"]
    assert result.get_path(0, 3) == [0, 1, 2, 3]
    assert result.get_path(1, 3) == [1, 2, 3]
    assert not result.paths_materialized
    assert result.paths.loc[2, 3] == [2, 3]
    assert result.paths is result.paths


def test_single_source_targets_keep_order_and_missing_entries():
    graph = directed_oneway_graph()
    result = single_source_dijkstra(graph, 3, target_nodes=[0, 3, 2], path_storage="lazy")
    assert result.distances.index.tolist() == [0, 3, 2]
    assert result.get_path(0) is None
    assert result.get_path(3) == [3]
    assert np.isinf(result.distances.loc[2])
    reverse = graph.single_source_dijkstra(3, target_nodes=[0], reverse=True, path_storage="lazy")
    assert reverse.get_path(0) == [0, 1, 2, 3]
    assert reverse.distances.loc[0] == 7
    assert graph.single_source_dijkstra(0, output="paths").loc[2] == [0, 1, 2]


def test_nearest_sources_differ_from_independent_searches():
    graph = undirected_line_graph()
    sources = pd.DataFrame({"graph_node_id": [0, 3, 0]}, index=["left", "right", "duplicate"])
    nearest = multi_source_dijkstra(graph, gdf_sources=sources, target_nodes=[3, 2], path_storage="lazy")
    parallel = dijkstra_parallel(graph, gdf_sources=sources, target_nodes=[3, 2], path_storage="lazy", max_workers=1)
    assert nearest.get_path(3) == [3]
    assert nearest.get_path(2) == [0, 1, 2]
    assert nearest.nearest_sources.to_dict() == {3: 3, 2: 0}
    assert parallel.get_path("left", 3) == [0, 1, 2, 3]
    assert parallel.get_path("right", 3) == [3]
    assert parallel.get_path("duplicate", 3) == [0, 1, 2, 3]
    assert parallel.distances.loc["left", 3] == 7
    pd.testing.assert_frame_equal(
        parallel.distances,
        graph.dijkstra_parallel(
            gdf_sources=sources, target_nodes=[3, 2], output="distances", batch_size=1, max_workers=1
        ),
    )
    assert graph.multi_source_dijkstra(source_nodes=[0, 3], output="paths").loc[2] == [0, 1, 2]


def test_zero_edges_ties_and_self_routes():
    graph = undirected_line_graph()
    graph.edges_gdf.loc[:, "time_min"] = 0.0
    joint = graph.od_matrix(origins_nodes=[0, 3], destination_nodes=[0, 1, 2, 3], cutoff=0, output="both")
    assert (joint.distances.sparse.to_dense().to_numpy() == 0).all()
    assert joint.get_path(0, 3) == [0, 1, 2, 3]
    nearest = graph.multi_source_dijkstra(source_nodes=[3, 0], cutoff=0, path_storage="lazy")
    assert nearest.nearest_sources.tolist() == [0, 0, 0, 0]
    assert nearest.get_path(3) == [0, 1, 2, 3]
    assert nearest.get_path(0) == [0]


def test_duplicate_labels_and_unique_source_optimization(monkeypatch):
    graph = undirected_line_graph()
    calls = []
    original = routing.dijkstra_numba_search_parallel

    def record(matrix, sources, targets, cutoff, store_predecessors):
        calls.append(len(sources))
        return original(matrix, sources, targets, cutoff, store_predecessors)

    monkeypatch.setattr(routing, "dijkstra_numba_search_parallel", record)
    result = graph.od_matrix(
        origins_nodes=[0, 0, 0], destination_nodes=[1, 2, 3, 1], output="both", path_storage="lazy", max_workers=1
    )
    assert calls == [1]
    assert result.distances.shape == (3, 4)
    assert result.get_path_at(2, 3) == [0, 1]
    with pytest.raises(ValueError, match="ambiguous"):
        result.get_path(0, 1)
    pd.testing.assert_frame_equal(
        result.distances, graph.od_matrix(origins_nodes=[0, 0, 0], destination_nodes=[1, 2, 3, 1], max_workers=1)
    )


def test_pairwise_duplicate_sources_share_one_search(monkeypatch):
    graph = undirected_line_graph()
    calls = []
    original = routing.dijkstra_numba_search_parallel

    def record(matrix, sources, targets, cutoff, store_predecessors):
        calls.append((len(sources), list(targets[0])))
        return original(matrix, sources, targets, cutoff, store_predecessors)

    monkeypatch.setattr(routing, "dijkstra_numba_search_parallel", record)
    origins = pd.DataFrame({"graph_node_id": [0, 0]}, index=["a", "b"])
    destinations = pd.DataFrame({"graph_node_id": [2, 3]}, index=["x", "y"])
    result = graph.od_matrix(
        gdf_origins=origins,
        gdf_destinations=destinations,
        mode="pairwise",
        output="both",
        path_storage="lazy",
        max_workers=1,
    )
    assert calls == [(1, [2, 3])]
    assert result.get_path("a", "x") == [0, 1, 2]
    assert result.get_path("b", "y") == [0, 1, 2, 3]
    assert result.distances["dist"].tolist() == [3, 7]


def test_early_stop_discards_tentative_distances_and_distance_only_parents():
    graph = undirected_line_graph()
    matrix = routing._prepare_numba_graph(graph, weight="time_min", cutoff=None, reverse=False)
    sources, targets = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    tree = dijkstra_numba_search(matrix, sources, targets, np.float32(np.inf), True)
    distances = dijkstra_numba_search(matrix, sources, targets, np.float32(np.inf), False)
    assert tree[0].tolist() == [0, 1]
    assert distances[0].tolist() == [1]
    assert len(distances[2]) == 0


def test_lazy_lookup_does_not_repeat_search(monkeypatch):
    graph = undirected_line_graph()
    result = graph.od_matrix(origins_nodes=[0], destination_nodes=[2, 3], output="both", path_storage="lazy")

    def fail(*args, **kwargs):
        raise AssertionError("Lazy route lookup must not search again")

    monkeypatch.setattr(routing, "dijkstra_numba_search_parallel", fail)
    assert result.get_path(0, 3) == [0, 1, 2, 3]
    assert not result.paths_materialized
    assert result.paths.loc[0, 2] == [0, 1, 2]


@pytest.mark.parametrize("mode", ["all_to_all", "pairwise"])
def test_streamed_blocks_match_full_result(mode):
    graph = directed_oneway_graph()
    params = dict(
        origins_nodes=[0, 1, 2, 3],
        destination_nodes=[3, 2, 1, 0],
        mode=mode,
        cutoff=6,
        output="both",
        path_storage="lazy",
        max_workers=1,
    )
    full = graph.od_matrix(**params)
    blocks = list(iter_od_matrix(graph, **params, batch_size=2))
    assert [block.distances.shape[0] for block in blocks] == [2, 2]
    assert not any(block.paths_materialized for block in blocks)
    # Block metadata describes its own source slice, rather than the entire input.
    # pandas compares attrs with == during concat, which is not defined for
    # differently labelled Series mappings. Drop metadata before combining.
    for block in blocks:
        block.distances.attrs = {}
        block.paths.attrs = {}
    for table, expected in (
        (pd.concat([b.distances for b in blocks]), full.distances),
        (pd.concat([b.paths for b in blocks]), full.paths),
    ):
        table.attrs, expected.attrs = {}, {}
        pd.testing.assert_frame_equal(table, expected)
    assert len(list(graph.iter_od_matrix(**params, batch_size=3))) == 2


def test_geodataframes_string_node_ids_and_parallel_edges():
    graph = multigraph_with_parallel_edges()
    result = graph.od_matrix(origins_nodes=[0], destination_nodes=[2], output="both")
    assert path_to_edges(graph, result.get_path(0, 2))["k"].tolist() == [1, 0]
    assert result.distances.loc[0, 2] == 3
    graph = undirected_line_graph()
    mapping = {i: f"n{i}" for i in range(4)}
    graph.nodes_gdf.index = graph.nodes_gdf.index.map(mapping)
    graph.edges_gdf["u"] = graph.edges_gdf["u"].map(mapping)
    graph.edges_gdf["v"] = graph.edges_gdf["v"].map(mapping)
    origins, destinations = graph.nodes_gdf.iloc[:2].copy(), graph.nodes_gdf.iloc[2:].copy()
    origins.index, destinations.index = ["a", "b"], ["c", "d"]
    result = graph.od_matrix(gdf_origins=origins, gdf_destinations=destinations, output="both", path_storage="lazy")
    assert result.get_path("a", "d") == ["n0", "n1", "n2", "n3"]


def test_random_directed_graph_agrees_with_scipy():
    rng = np.random.default_rng(15)
    size = 14
    weights = rng.integers(0, 8, size=(size, size))
    edges = [(u, v, float(weights[u, v])) for u in range(size) for v in range(size) if u != v and rng.random() < 0.18]
    graph = UrbanGraph(
        pd.DataFrame(index=range(size)),
        pd.DataFrame(
            [(u, v, w, w, None) for u, v, w in edges], columns=["u", "v", "time_min", "length_meter", "geometry"]
        ),
        is_multigraph=False,
        is_directed=True,
    )
    graph.update_adjacency_matrix(weight="time_min")
    expected = scipy_dijkstra(graph.adjacency_matrix, indices=np.arange(size), limit=9)
    result = graph.od_matrix(
        origins_nodes=range(size),
        destination_nodes=range(size),
        cutoff=9,
        output="both",
        path_storage="lazy",
        max_workers=1,
    )
    np.testing.assert_allclose(result.distances.sparse.to_dense().to_numpy(), expected)
    for u in range(size):
        for v in range(size):
            route = result.get_path(u, v)
            if np.isfinite(expected[u, v]):
                assert route[0] == u and route[-1] == v
                assert sum(graph.adjacency_matrix[a, b] for a, b in zip(route[:-1], route[1:])) == expected[u, v]
            else:
                assert route is None


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"output": "invalid"}, ValueError, "output"),
        ({"path_storage": "invalid"}, ValueError, "path_storage"),
        ({"path_storage": "lazy"}, ValueError, "requires"),
        ({"cutoff": 1, "threshold": 1}, ValueError, "either"),
        ({"cutoff": -1}, ValueError, "cutoff"),
        ({"cutoff": np.nan}, ValueError, "NaN"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"max_workers": True}, TypeError, "max_workers"),
        ({"mode": "invalid"}, ValueError, "mode"),
    ],
)
def test_invalid_od_options(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        undirected_line_graph().od_matrix(origins_nodes=[0], destination_nodes=[1], **kwargs)


def test_invalid_targets_inputs_and_lookup_positions():
    graph = undirected_line_graph()
    with pytest.raises(ValueError, match="target_nodes"):
        graph.single_source_dijkstra(0, target_nodes=[])
    with pytest.raises(ValueError, match="absent"):
        graph.single_source_dijkstra(0, target_nodes=[999])
    with pytest.raises(ValueError, match="same number"):
        graph.od_matrix(origins_nodes=[0], destination_nodes=[1, 2], mode="pairwise")
    with pytest.raises(ValueError, match="batch_size"):
        next(graph.iter_od_matrix(origins_nodes=[0], destination_nodes=[1], batch_size=None))
    result = graph.single_source_dijkstra(0)
    with pytest.raises(IndexError):
        result.get_path_at(-1)
    with pytest.raises(TypeError):
        result.get_path_at(0.5)
    with pytest.raises(TypeError):
        result.get_path(0, 1)
    with pytest.raises(KeyError):
        result.get_path(999)


def test_worker_count_restored_on_success_and_failure(monkeypatch):
    graph = undirected_line_graph()
    previous = nb.get_num_threads()
    graph.od_matrix(origins_nodes=[0], destination_nodes=[1], max_workers=1)
    assert nb.get_num_threads() == previous

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(routing, "dijkstra_numba_search_parallel", fail)
    with pytest.raises(RuntimeError, match="kernel failed"):
        graph.od_matrix(origins_nodes=[0], destination_nodes=[1], max_workers=1)
    assert nb.get_num_threads() == previous


def test_parallel_workers_and_batches_preserve_results():
    if nb.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Two Numba workers are unavailable")
    graph = disconnected_graph()
    params = dict(origins_nodes=[0, 1, 2, 10, 11], destination_nodes=[0, 2, 10, 11], output="both")
    serial = graph.od_matrix(**params, max_workers=1)
    parallel = graph.od_matrix(**params, max_workers=2, batch_size=2)
    pd.testing.assert_frame_equal(serial.distances, parallel.distances)
    pd.testing.assert_frame_equal(serial.paths, parallel.paths)


def test_isolated_nodes_unreachable_columns_and_float64_storage():
    graph = UrbanGraph(
        pd.DataFrame(index=["a", "b"]),
        pd.DataFrame(columns=["u", "v", "geometry", "time_min", "length_meter"]),
        is_multigraph=False,
        is_directed=True,
    )
    result = graph.od_matrix(
        origins_nodes=["a"], destination_nodes=["a", "b"], output="both", dtype=np.float64, path_storage="lazy"
    )
    assert result.get_path("a", "a") == ["a"]
    assert result.get_path("a", "b") is None
    assert result.distances.loc["a", "a"] == 0
    assert np.isinf(result.distances.loc["a", "b"])
    assert all(dtype.subtype == np.dtype("float64") for dtype in result.distances.dtypes)
    nearest = graph.multi_source_dijkstra(source_nodes=["a"], target_nodes=["b"])
    assert nearest.nearest_sources.iloc[0] is None


def test_legacy_path_matrix_preserves_labels_mode_and_threshold():
    graph = undirected_line_graph()
    origins = pd.DataFrame({"graph_node_id": [0, 3]}, index=["left", "right"])
    destinations = pd.DataFrame({"graph_node_id": [2, 0]}, index=["near", "far"])
    result = multi_source_dijkstra_path(
        graph,
        gdf_origins=origins,
        gdf_destinations=destinations,
        mode="pairwise",
        threshold=3,
        max_workers=1,
    )
    assert result.index.tolist() == [("left", "near"), ("right", "far")]
    assert result.iloc[0, 0] == [0, 1, 2]
    assert pd.isna(result.iloc[1, 0])
    assert result.attrs["origin_nodes"].to_dict() == {"left": 0, "right": 3}
    assert result.attrs["destination_nodes"].to_dict() == {"near": 2, "far": 0}
