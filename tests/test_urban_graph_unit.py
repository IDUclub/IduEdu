import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from iduedu import multi_source_dijkstra_path, nearest_nodes, path_to_edges, single_source_dijkstra_path, validate_graph
from iduedu.graph.adjacency import build_adjacency_matrix
from iduedu.graph.urban_graph import UrbanGraph
from tests.factories import (
    CRS,
    _edges,
    _nodes,
    directed_oneway_graph,
    disconnected_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = pytest.mark.unit


def test_empty_graph_has_expected_schema_and_csr_shape():
    graph = UrbanGraph.empty(crs=CRS, is_multigraph=True, edge_direction_column="oneway", graph_type="walk")

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert list(graph.edges_gdf.columns) == ["u", "v", "k", "geometry", "length_meter", "time_min", "oneway"]
    assert graph.to_csr().shape == (0, 0)


def test_update_adjacency_matrix_updates_cached_state_and_copy_preserves_it():
    graph = undirected_line_graph()

    matrix = graph.update_adjacency_matrix(weight="time_min")
    copied = graph.copy()

    assert graph.adjacency_matrix is matrix
    assert graph.adjacency_nodelist == [0, 1, 2, 3]
    assert graph.node_to_adjacency_pos == {0: 0, 1: 1, 2: 2, 3: 3}
    assert copied.adjacency_matrix is not graph.adjacency_matrix
    assert copied.adjacency_matrix.toarray().tolist() == graph.adjacency_matrix.toarray().tolist()


def test_constructor_rejects_duplicate_node_index():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1, 0)],
        index=[1, 1],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(columns=["u", "v", "geometry", "length_meter", "time_min"], geometry="geometry", crs=CRS)

    with pytest.raises(ValueError, match="index must be unique"):
        UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False, crs=CRS)


def test_constructor_rejects_missing_edge_endpoint():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=[0, 1], crs=CRS)
    edges = gpd.GeoDataFrame(
        [{"u": 0, "v": 999, "geometry": LineString([(0, 0), (1, 0)]), "length_meter": 1.0, "time_min": 1.0}],
        geometry="geometry",
        crs=CRS,
    )

    with pytest.raises(ValueError, match="missing"):
        UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False, crs=CRS)


def test_constructor_rejects_duplicate_non_multigraph_edges():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=[0, 1], crs=CRS)
    edges = gpd.GeoDataFrame(
        [
            {"u": 0, "v": 1, "geometry": LineString([(0, 0), (1, 0)]), "length_meter": 1.0, "time_min": 1.0},
            {"u": 0, "v": 1, "geometry": LineString([(0, 0), (1, 0)]), "length_meter": 2.0, "time_min": 2.0},
        ],
        geometry="geometry",
        crs=CRS,
    )

    with pytest.raises(ValueError, match="unique"):
        UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False, crs=CRS)


def test_constructor_rejects_crs_mismatch():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0)], index=[0, 1], crs="EPSG:3857")
    edges = gpd.GeoDataFrame(
        [{"u": 0, "v": 1, "geometry": LineString([(0, 0), (1, 0)]), "length_meter": 1.0, "time_min": 1.0}],
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="crs"):
        UrbanGraph(nodes, edges, is_multigraph=False, is_directed=False)


def test_method_wrappers_match_functional_results():
    graph = undirected_line_graph()

    assert graph.connected_components() == [{0, 1, 2, 3}]
    assert graph.largest_component() == {0, 1, 2, 3}
    assert graph.single_source_dijkstra_path_length(0).sparse.to_dense().to_dict() == {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0}


WALK_SPEED_M_PER_MIN = 100.0


def test_relabel_returns_new_graph_when_not_inplace():
    graph = disconnected_graph()

    relabeled = graph.relabel(inplace=False)

    assert relabeled is not graph
    assert relabeled.nodes_gdf.index.tolist() == [0, 1, 2, 3, 4]
    assert graph.nodes_gdf.index.tolist() == [0, 1, 2, 10, 11]  # untouched


def test_relabel_inplace_mutates_and_resets_adjacency_matrix():
    graph = disconnected_graph()
    graph.update_adjacency_matrix(weight="time_min")
    assert graph.adjacency_matrix is not None

    result = graph.relabel(inplace=True)

    assert result is graph
    assert graph.nodes_gdf.index.tolist() == [0, 1, 2, 3, 4]
    assert graph.adjacency_matrix is None  # stale matrix dropped after structural change


def test_clip_inplace_replaces_state():
    graph = undirected_line_graph()
    polygon = box(-5.0, -5.0, 15.0, 5.0)

    result = graph.clip(polygon, inplace=True)

    assert result is graph
    assert set(graph.nodes_gdf.index) == {0, 1}


def test_to_directed_and_to_undirected_round_trip_inplace():
    graph = undirected_line_graph()

    graph.to_directed(edge_direction_column="oneway", inplace=True)
    assert graph.is_directed
    assert graph.edge_direction_column == "oneway"

    graph.to_undirected(inplace=True)
    assert not graph.is_directed
    assert graph.edge_direction_column is None


def test_simplify_multiedges_inplace_drops_multigraph_flag():
    graph = multigraph_with_parallel_edges()

    result = graph.simplify_multiedges(weight="time_min", rule="min", inplace=True)

    assert result is graph
    assert not graph.is_multigraph
    assert "k" not in graph.edges_gdf.columns


def test_keep_largest_connected_component_inplace_keeps_only_main_component():
    graph = disconnected_graph()

    result = graph.keep_largest_connected_component(inplace=True)

    assert result is graph
    assert set(graph.nodes_gdf.index) == {0, 1, 2}


def test_join_inplace_appends_other_graph():
    left = undirected_line_graph()  # nodes 0..3
    right_nodes = {4: (40.0, 0.0), 5: (50.0, 0.0)}
    right = UrbanGraph(
        _nodes(right_nodes),
        _edges(right_nodes, [{"u": 4, "v": 5}]),
        is_multigraph=False,
        is_directed=False,
        crs=CRS,
        graph_type="walk",
    )

    result = left.join(right, inplace=True)

    assert result is left
    assert set(left.nodes_gdf.index) == {0, 1, 2, 3, 4, 5}


def test_project_objects_inplace_returns_self_and_object_map():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=[1000], geometry=[Point(15.0, 5.0)], crs=CRS)
    original_node_count = len(graph.nodes_gdf)

    result, object2node = graph.project_objects(objects, WALK_SPEED_M_PER_MIN, inplace=True)

    assert result is graph
    assert object2node.loc[1000] in graph.nodes_gdf.index
    assert len(graph.nodes_gdf) > original_node_count


def test_nearest_nodes_snaps_polygon_via_representative_point():
    graph = undirected_line_graph()
    # A polygon whose interior point sits nearest the node at x = 20.
    objects = gpd.GeoDataFrame(index=[7], geometry=[box(18.0, -1.0, 22.0, 1.0)], crs=CRS)

    result = nearest_nodes(graph, objects)

    assert result.loc[7] == 2


def test_to_csr_does_not_mutate_cached_adjacency_state():
    graph = undirected_line_graph()

    matrix = graph.to_csr(weight="time_min")

    assert matrix.shape == (4, 4)
    assert graph.adjacency_matrix is None  # to_csr is side-effect free
    assert graph.adjacency_nodelist == []


def test_path_routing_is_available_through_methods_and_public_api():
    graph = undirected_line_graph()

    method_path = graph.single_source_dijkstra_path(0, 3)
    function_path = single_source_dijkstra_path(graph, 0, 3)
    method_matrix = graph.multi_source_dijkstra_path(
        origins_nodes=[0],
        destination_nodes=[3],
        max_workers=1,
    )
    function_matrix = multi_source_dijkstra_path(
        graph,
        origins_nodes=[0],
        destination_nodes=[3],
        max_workers=1,
    )
    method_edges = graph.path_to_edges(method_path)
    function_edges = path_to_edges(graph, function_path)

    assert method_path == function_path == [0, 1, 2, 3]
    assert method_matrix.loc[0, 3] == function_matrix.loc[0, 3] == [0, 1, 2, 3]
    assert method_edges["path_order"].tolist() == function_edges["path_order"].tolist() == [0, 1, 2]


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


def test_nearest_nodes_preserves_order_labels_and_custom_column():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(
        index=["far", "near", "mid"],
        geometry=[Point(31, 0), Point(-1, 0), Point(21, 0)],
        crs=CRS,
    )
    method_result = graph.nearest_nodes(objects)
    function_result = nearest_nodes(graph, objects, graph_node_column="nearest")
    for result, expected_name in [(method_result, "graph_node_id"), (function_result, "nearest")]:
        assert result.index.tolist() == ["far", "near", "mid"]
        assert result.tolist() == [3, 0, 2]
        assert result.name == expected_name


def test_validation_function_and_method_reject_broken_edges():
    graph = undirected_line_graph()
    graph.edges_gdf.loc[0, "u"] = 999
    with pytest.raises(ValueError, match="missing in nodes_gdf.index"):
        validate_graph(graph)
    with pytest.raises(ValueError, match="missing in nodes_gdf.index"):
        graph.validate()
