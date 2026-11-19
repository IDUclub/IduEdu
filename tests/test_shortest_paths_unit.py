import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
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
from iduedu.graph import routing
from iduedu.graph.shortest_paths import (
    multi_source_dijkstra_nearest_source,
    single_source_dijkstra_path,
)
from tests.factories import (
    directed_oneway_graph,
    disconnected_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = [pytest.mark.unit, pytest.mark.rust]


def _dense(series: pd.Series) -> dict:
    return {key: float(value) for key, value in series.sparse.to_dense().items()}


def _assert_distance_array(result, expected, distance_format):
    assert result.dtype == expected.dtype == np.float64
    assert result.shape == expected.shape
    if distance_format == "dense":
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
    else:
        assert isinstance(result, csr_array)
        entries = result.tocoo()
        assert result.nnz == np.isfinite(expected).sum()
        np.testing.assert_array_equal(entries.data, expected[entries.row, entries.col])
        assert np.count_nonzero(entries.data == 0) == np.count_nonzero(expected == 0)
        reconstructed = np.full(expected.shape, np.inf)
        reconstructed[entries.row, entries.col] = entries.data
        np.testing.assert_array_equal(reconstructed, expected)


def _assert_combined_array_result(result, distance_only, frame, path_storage, target_labels=None):
    assert isinstance(result, ShortestPathResult)
    assert result.paths_materialized == (path_storage == "eager")
    distances = result.distances
    assert type(distances) is type(distance_only)
    assert distances.shape == distance_only.shape
    assert distances.dtype == distance_only.dtype
    if isinstance(distance_only, csr_array):
        for field in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(getattr(distances, field), getattr(distance_only, field))
    else:
        np.testing.assert_array_equal(distances, distance_only)

    path_frame = frame.paths
    rows = np.arange(len(path_frame))
    columns = np.arange(1 if isinstance(path_frame, pd.Series) else path_frame.shape[1])
    index = path_frame.index
    if target_labels is not None:
        if isinstance(path_frame, pd.Series):
            rows = index.get_indexer(target_labels)
            index = target_labels
        else:
            columns = path_frame.columns.get_indexer(target_labels)
    for row, frame_row in enumerate(rows):
        for column, frame_col in enumerate(columns):
            expected = frame.get_path_at(frame_row, frame_col) if frame_row >= 0 and frame_col >= 0 else None
            assert result.get_path_at(row, column) == expected
    assert result.paths_materialized == (path_storage == "eager")
    assert type(result.paths) is type(path_frame)
    pd.testing.assert_index_equal(result.paths.index, index)
    if isinstance(path_frame, pd.DataFrame):
        pd.testing.assert_index_equal(
            result.paths.columns, path_frame.columns if target_labels is None else target_labels
        )
    if target_labels is None:
        if isinstance(path_frame, pd.Series):
            pd.testing.assert_series_equal(result.paths, path_frame)
        else:
            pd.testing.assert_frame_equal(result.paths, path_frame)


@pytest.mark.parametrize("distance_format", ["dense", "sparse"])
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("path_storage", [None, "lazy", "eager"])
def test_od_matrix_distance_arrays(distance_format, batch_size, path_storage):
    graph = directed_oneway_graph()
    options = {
        "origins_nodes": np.array([3, 0, 2, 0, 1], dtype=graph.nodes_gdf.index.dtype),
        "destination_nodes": np.array([2, 0, 2], dtype=graph.nodes_gdf.index.dtype),
        "cutoff": 2,
        "dtype": np.float64,
    }

    def blocks(**result_options):
        if batch_size is None:
            return [graph.od_matrix(**options, **result_options)]
        return list(graph.iter_od_matrix(**options, **result_options, batch_size=batch_size))

    expected = blocks()
    actual = blocks(distance_format=distance_format)
    if batch_size is not None:
        assert [block.shape for block in actual] == [(4, 3), (1, 3)]
    for result, frame in zip(actual, expected, strict=True):
        _assert_distance_array(result, frame.to_numpy(np.float64, na_value=np.inf), distance_format)
    if path_storage is not None:
        combined = blocks(distance_format=distance_format, output="both", path_storage=path_storage)
        frame_blocks = blocks(output="both")
        assert combined[0].get_path(2, 0) == frame_blocks[0].get_path(2, 0)
        for result, distances, frame in zip(combined, actual, frame_blocks, strict=True):
            _assert_combined_array_result(result, distances, frame, path_storage)


@pytest.mark.parametrize("distance_format", ["dense", "sparse"])
@pytest.mark.parametrize("targets", [[11, 0, 2, 1, 0], None])
@pytest.mark.parametrize("path_storage", [None, "lazy", "eager"])
def test_parallel_distance_arrays(distance_format, targets, path_storage):
    graph = disconnected_graph()
    graph.nodes_gdf = graph.nodes_gdf.loc[[11, 2, 0, 1, 10]]
    options = {
        "source_nodes": [0, 1, 0],
        "target_nodes": targets,
        "cutoff": 1,
        "dtype": np.float64,
    }
    frame = dijkstra_parallel(graph, **options, output="distances")
    target_labels = None
    if targets is None:
        frame = frame.reindex(columns=graph.nodes_gdf.index, fill_value=np.inf)
        target_labels = graph.nodes_gdf.index.rename("node")
    result = graph.dijkstra_parallel(**options, output="distances", distance_format=distance_format)
    _assert_distance_array(result, frame.to_numpy(np.float64, na_value=np.inf), distance_format)
    if path_storage is not None:
        combined = graph.dijkstra_parallel(
            **options, output="both", distance_format=distance_format, path_storage=path_storage
        )
        frame_combined = dijkstra_parallel(graph, **options, output="both")
        assert combined.get_path(1, 1) == frame_combined.get_path(1, 1) == [1]
        assert combined.get_path(1, 11) is None
        _assert_combined_array_result(combined, result, frame_combined, path_storage, target_labels)


@pytest.mark.parametrize(
    "function, sources",
    [(single_source_dijkstra, {"source_node": 0}), (multi_source_dijkstra, {"source_nodes": [0, 1, 0]})],
)
@pytest.mark.parametrize("targets", [[11, 0, 2, 1, 0], None])
@pytest.mark.parametrize("path_storage", [None, "lazy", "eager"])
def test_series_distance_arrays(function, sources, targets, path_storage):
    graph = disconnected_graph()
    graph.nodes_gdf = graph.nodes_gdf.loc[[11, 2, 0, 1, 10]]
    options = {**sources, "target_nodes": targets, "cutoff": 1, "dtype": np.float64}
    frame = function(graph, **options, output="distances")
    target_labels = None
    if targets is None:
        frame = frame.reindex(graph.nodes_gdf.index, fill_value=np.inf)
        target_labels = graph.nodes_gdf.index.rename("node")
    method = getattr(graph, function.__name__)
    distances = method(**options, output="distances", distance_format="dense")
    _assert_distance_array(distances, frame.to_numpy(np.float64, na_value=np.inf), "dense")
    with pytest.raises(ValueError, match="sparse.*unavailable"):
        method(**options, output="distances" if path_storage is None else "both", distance_format="sparse")
    if path_storage is not None:
        frame_combined = function(graph, **options, output="both")
        result = method(**options, output="both", distance_format="dense", path_storage=path_storage)
        assert result.get_path(1) == frame_combined.get_path(1)
        assert result.get_path(11) is None
        _assert_combined_array_result(result, distances, frame_combined, path_storage, target_labels)
        if function is multi_source_dijkstra:
            expected = frame_combined.nearest_sources
            if target_labels is not None:
                expected = expected.reindex(target_labels)
                expected = expected.where(expected.notna(), None)
            pd.testing.assert_series_equal(result.nearest_sources, expected)


@pytest.mark.parametrize("method", ["dijkstra", "bidirectional"])
@pytest.mark.parametrize("path_storage", [None, "lazy", "eager"])
def test_pairwise_distance_arrays(method, path_storage):
    graph = disconnected_graph()
    options = {
        "origins_nodes": [0, 0, 1, 11, 11],
        "destination_nodes": [0, 2, 2, 0, 10],
        "mode": "pairwise",
        "cutoff": 1,
        "dtype": np.float64,
        "method": method,
    }
    frame = od_matrix(graph, **options)
    expected = frame.to_numpy(np.float64, na_value=np.inf)[:, 0]
    distances = od_matrix(graph, **options, distance_format="dense")
    _assert_distance_array(distances, expected, "dense")
    blocks = list(iter_od_matrix(graph, **options, distance_format="dense", batch_size=2))
    assert [block.shape for block in blocks] == [(2,), (2,), (1,)]
    np.testing.assert_array_equal(np.concatenate(blocks), expected)
    output = "distances" if path_storage is None else "both"
    with pytest.raises(ValueError, match="sparse.*pairwise"):
        od_matrix(graph, **options, output=output, distance_format="sparse")
    with pytest.raises(ValueError, match="sparse.*pairwise"):
        list(iter_od_matrix(graph, **options, output=output, distance_format="sparse"))
    if path_storage is not None:
        result = od_matrix(graph, **options, output="both", distance_format="dense", path_storage=path_storage)
        frame_combined = od_matrix(graph, **options, output="both")
        assert result.get_path(1, 2) == frame_combined.get_path(1, 2) == [1, 2]
        _assert_combined_array_result(result, distances, frame_combined, path_storage)
        actual = iter_od_matrix(
            graph, **options, batch_size=2, output="both", distance_format="dense", path_storage=path_storage
        )
        frame_blocks = iter_od_matrix(graph, **options, batch_size=2, output="both")
        for result, distance_block, frame_block in zip(actual, blocks, frame_blocks, strict=True):
            _assert_combined_array_result(result, distance_block, frame_block, path_storage)


@pytest.mark.parametrize(
    "function, inputs",
    [
        (od_matrix, {"origins_nodes": [0], "destination_nodes": [1]}),
        (iter_od_matrix, {"origins_nodes": [0], "destination_nodes": [1]}),
        (dijkstra_parallel, {"source_nodes": [0]}),
        (single_source_dijkstra, {"source_node": 0}),
        (multi_source_dijkstra, {"source_nodes": [0]}),
    ],
)
@pytest.mark.parametrize("distance_format", ["dense", "sparse"])
def test_array_distance_formats_reject_paths_only(function, inputs, distance_format):
    with pytest.raises(ValueError, match="require output='distances'"):
        result = function(disconnected_graph(), **inputs, output="paths", distance_format=distance_format)
        if function is iter_od_matrix:
            list(result)


def test_single_source_dijkstra_respects_cutoff():
    graph = undirected_line_graph()

    result = single_source_dijkstra(graph, 0, weight="time_min", cutoff=3.0, output="distances")

    assert _dense(result) == {0: 0.0, 1: 1.0, 2: 3.0}
    assert 3 not in result.index


def test_reverse_dijkstra_uses_transposed_directed_graph():
    graph = directed_oneway_graph()

    forward = single_source_dijkstra(graph, 3, weight="time_min", output="distances")
    reverse = single_source_dijkstra(graph, 3, weight="time_min", reverse=True, output="distances")

    assert _dense(forward) == {3: 0.0}
    assert _dense(reverse) == {0: 7.0, 1: 6.0, 2: 4.0, 3: 0.0}


def test_multi_source_dijkstra_returns_nearest_distance_and_source():
    graph = undirected_line_graph()

    distances = multi_source_dijkstra(graph, source_nodes=[0, 3], weight="time_min", output="distances")
    nearest = multi_source_dijkstra_nearest_source(graph, source_nodes=[0, 3], weight="time_min")

    assert _dense(distances) == {0: 0.0, 1: 1.0, 2: 3.0, 3: 0.0}
    assert nearest.loc[0, "source_node"] == 0
    assert nearest.loc[2, "source_node"] == 0
    assert nearest.loc[3, "source_node"] == 3
    assert float(nearest.loc[2, "dist"]) == 3.0


def test_dijkstra_parallel_keeps_input_source_index():
    graph = undirected_line_graph()
    sources = pd.DataFrame({"graph_node_id": [0, 3]}, index=["left", "right"])

    result = dijkstra_parallel(graph, gdf_sources=sources, weight="time_min", max_workers=1, output="distances")

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
        single_source_dijkstra(graph, 999)


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
    assert graph.od_matrix(**params, cutoff=cutoff, output="paths").loc[0, 2] == [0, 1, 2]


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
    original = routing._kernel_search_many

    def record(matrix, sources, *args):
        calls.append(len(sources))
        return original(matrix, sources, *args)

    monkeypatch.setattr(routing, "_kernel_search_many", record)
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
    original = routing._kernel_search_many

    def record(matrix, sources, targets, starts, ends, *args):
        calls.append((len(sources), sorted(targets[starts[0] : ends[0]])))
        return original(matrix, sources, targets, starts, ends, *args)

    monkeypatch.setattr(routing, "_kernel_search_many", record)
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
    matrix = routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False)
    sources, targets = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    tree = matrix.search(sources, targets, np.float32(np.inf), True)
    distances = matrix.search(sources, targets, np.float32(np.inf), False)
    assert tree[0].tolist() == [0, 1]
    assert distances[0].tolist() == [1]
    assert len(distances[2]) == 0


def test_lazy_lookup_does_not_repeat_search(monkeypatch):
    graph = undirected_line_graph()
    result = graph.od_matrix(origins_nodes=[0], destination_nodes=[2, 3], output="both", path_storage="lazy")

    def fail(*args, **kwargs):
        raise AssertionError("Lazy route lookup must not search again")

    monkeypatch.setattr(routing, "_kernel_search_many", fail)
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


def test_max_workers_reaches_the_kernel_and_kernel_errors_propagate(monkeypatch):
    graph = undirected_line_graph()
    workers = []
    original = routing._kernel_search_many

    def record(matrix, *args):
        workers.append(args[-1])
        return original(matrix, *args)

    monkeypatch.setattr(routing, "_kernel_search_many", record)
    graph.od_matrix(origins_nodes=[0], destination_nodes=[1], max_workers=1)
    graph.od_matrix(origins_nodes=[0], destination_nodes=[1])
    assert workers == [1, None]

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(routing, "_kernel_search_many", fail)
    with pytest.raises(RuntimeError, match="kernel failed"):
        graph.od_matrix(origins_nodes=[0], destination_nodes=[1], max_workers=1)


def test_parallel_workers_and_batches_preserve_results():
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


def test_pairwise_path_table_preserves_labels_and_threshold():
    graph = undirected_line_graph()
    origins = pd.DataFrame({"graph_node_id": [0, 3]}, index=["left", "right"])
    destinations = pd.DataFrame({"graph_node_id": [2, 0]}, index=["near", "far"])
    result = od_matrix(
        graph,
        gdf_origins=origins,
        gdf_destinations=destinations,
        mode="pairwise",
        threshold=3,
        max_workers=1,
        output="paths",
    )
    assert result.index.tolist() == [("left", "near"), ("right", "far")]
    assert result.iloc[0, 0] == [0, 1, 2]
    assert pd.isna(result.iloc[1, 0])
    assert result.attrs["origin_nodes"].to_dict() == {"left": 0, "right": 3}
    assert result.attrs["destination_nodes"].to_dict() == {"near": 2, "far": 0}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_tables_keep_requested_dtype_with_unreachable_cells(dtype):
    graph = disconnected_graph()
    tables = [
        graph.dijkstra_parallel(source_nodes=[0, 10], cutoff=1, output="distances", dtype=dtype),
        graph.od_matrix(origins_nodes=[0, 10], destination_nodes=[1, 2, 11], cutoff=1, dtype=dtype),
        graph.od_matrix(origins_nodes=[0, 1, 2, 10], destination_nodes=[11], dtype=dtype),
    ]
    for table in tables:
        assert all(column == pd.SparseDtype(dtype, np.inf) for column in table.dtypes)
    assert tables[0].loc[0, 1] == 1 and np.isinf(tables[0].loc[0, 11]) and np.isinf(tables[0].loc[10, 0])
    assert tables[1].loc[10, 11] == 1 and np.isinf(tables[1].loc[0, 2])
    assert tables[2][11].tolist() == [np.inf, np.inf, np.inf, 1]


@pytest.mark.parametrize("origins,destinations", [([0, 1, 0, 10], [2, 0, 11]), ([0, 1, 2, 10, 11], [1, 1])])
def test_materialized_path_matrix_matches_lazy_cells(origins, destinations):
    # The second case runs transposed: searches start from the destinations.
    graph = disconnected_graph()
    params = dict(origins_nodes=origins, destination_nodes=destinations, output="both")
    lazy = graph.od_matrix(**params, path_storage="lazy")
    paths = graph.od_matrix(**params).paths
    assert paths.shape == (len(origins), len(destinations))
    assert (paths.dtypes == pd.SparseDtype(object, None)).all()
    for row in range(len(origins)):
        for column in range(len(destinations)):
            cell = paths.iloc[row, column]
            assert (cell if isinstance(cell, list) else None) == lazy.get_path_at(row, column)
    paths.iloc[0, 0].append("edited")
    assert paths.iloc[2, 0] != paths.iloc[0, 0]


def test_pairwise_repeated_unordered_origins_match_matrix_cells():
    graph = disconnected_graph()
    origins, destinations = [2, 0, 10, 0, 2, 1], [0, 2, 11, 1, 2, 10]
    pairwise = graph.od_matrix(origins_nodes=origins, destination_nodes=destinations, mode="pairwise", output="both")
    matrix = graph.od_matrix(origins_nodes=origins, destination_nodes=destinations, output="both")
    for i in range(len(origins)):
        assert pairwise.distances.iloc[i, 0] == matrix.distances.iloc[i, i]
        assert pairwise.get_path_at(i) == matrix.get_path_at(i, i)


def test_path_to_edges_out_and_back_route_reuses_edges_in_order():
    graph = undirected_line_graph()
    route = path_to_edges(graph, [0, 1, 2, 1, 0])
    assert route["path_order"].tolist() == [0, 1, 2, 3]
    assert list(zip(route["path_u"], route["path_v"])) == [(0, 1), (1, 2), (2, 1), (1, 0)]
    assert route.index.tolist() == [0, 1, 1, 0]
    assert list(route.geometry.iloc[2].coords) == [(20.0, 0.0), (10.0, 0.0)]
    with pytest.raises(ValueError, match=r"absent or disallowed in graph: \(0, 2\)"):
        path_to_edges(graph, [0, 1, 0, 2])


def _cost_to_end(graph, weight="time_min"):
    return _dense(single_source_dijkstra(graph, 0, weight=weight, output="distances"))[3]


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda edges: edges.__setitem__("time_min", edges["time_min"] * 2), id="column-assigned"),
        pytest.param(lambda edges: edges.loc.__setitem__((1, "time_min"), 4.0), id="loc-in-place"),
    ],
)
def test_weight_edits_after_a_search_rebuild_the_matrix(edit):
    graph = undirected_line_graph()
    assert _cost_to_end(graph) == 7.0
    edit(graph.edges_gdf)
    assert _cost_to_end(graph) == pytest.approx(float(graph.edges_gdf["time_min"].sum()))


def test_new_edge_table_and_new_edge_ends_rebuild_the_matrix():
    graph = undirected_line_graph()
    assert _cost_to_end(graph) == 7.0
    edges = graph.edges_gdf.copy()
    edges["time_min"] = [1.0, 1.0, 1.0]
    graph.edges_gdf = edges
    assert _cost_to_end(graph) == 3.0
    graph.edges_gdf["v"] = [1, 3, 2]  # 1 -> 3 directly, 2 hangs on 3
    assert _cost_to_end(graph) == 2.0


def test_direction_edits_rebuild_the_matrix():
    graph = directed_oneway_graph()
    reverse = single_source_dijkstra(graph, 3, weight="time_min", output="distances")
    assert _dense(reverse) == {3: 0.0}
    graph.edges_gdf.loc[2, "oneway"] = False
    assert _dense(single_source_dijkstra(graph, 3, weight="time_min", output="distances")) == {3: 0.0, 2: 4.0, 1: 6.0}


def test_bad_weights_written_after_a_search_are_rejected():
    graph = undirected_line_graph()
    _cost_to_end(graph)
    graph.edges_gdf.loc[0, "time_min"] = np.nan
    with pytest.raises(ValueError, match="non-negative values without NaN"):
        _cost_to_end(graph)


def test_unchanged_tables_reuse_the_matrix_and_a_manual_rebuild_is_trusted():
    graph = undirected_line_graph()
    first = routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False)
    assert routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False) is first
    graph.edges_gdf.loc[0, "time_min"] = 3.0
    matrix = graph.update_adjacency_matrix(weight="time_min")
    routing._prepare_graph(graph, weight="time_min", cutoff=None, reverse=False)
    assert graph.adjacency_matrix is matrix
    assert _cost_to_end(graph) == 9.0
