import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from iduedu import nearest_nodes, validate_graph
from iduedu.graph.urban_graph import UrbanGraph
from tests.factories import (
    CRS,
    _edges,
    _nodes,
    disconnected_graph,
    multigraph_with_parallel_edges,
    undirected_line_graph,
)

pytestmark = pytest.mark.unit

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


def test_nearest_nodes_method_returns_node_ids_for_geometries():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=["a", "b"], geometry=[Point(1.0, 0.0), Point(24.0, 0.0)], crs=CRS)

    result = graph.nearest_nodes(objects)

    assert result.index.tolist() == ["a", "b"]
    assert result.name == "graph_node_id"
    assert result.tolist() == [0, 2]


def test_nearest_nodes_function_uses_custom_series_name():
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(index=[100], geometry=[Point(29.0, 0.0)], crs=CRS)

    result = nearest_nodes(graph, objects, graph_node_column="nearest")

    assert result.name == "nearest"
    assert result.loc[100] == 3


def test_nearest_nodes_preserves_input_order_for_unsorted_index():
    # Nodes at x = 0, 10, 20, 30 (index 0..3). Query points are given in an order
    # that does not match either the spatial order or a sorted index, to guard the
    # position remapping after the spatial-index query.
    graph = undirected_line_graph()
    objects = gpd.GeoDataFrame(
        index=["far", "near", "mid"],
        geometry=[Point(31.0, 0.0), Point(-1.0, 0.0), Point(21.0, 0.0)],
        crs=CRS,
    )

    result = nearest_nodes(graph, objects)

    assert result.index.tolist() == ["far", "near", "mid"]
    assert result.tolist() == [3, 0, 2]


def test_nearest_nodes_snaps_polygon_via_representative_point():
    graph = undirected_line_graph()
    # A polygon whose interior point sits nearest the node at x = 20.
    objects = gpd.GeoDataFrame(index=[7], geometry=[box(18.0, -1.0, 22.0, 1.0)], crs=CRS)

    result = nearest_nodes(graph, objects)

    assert result.loc[7] == 2


def test_validate_graph_public_api_reports_broken_edges():
    graph = undirected_line_graph()
    graph.edges_gdf.loc[0, "u"] = 999

    with pytest.raises(ValueError, match="missing in nodes_gdf.index"):
        validate_graph(graph)


def test_urban_graph_validate_reports_broken_edges():
    graph = undirected_line_graph()
    graph.edges_gdf.loc[0, "v"] = 999

    with pytest.raises(ValueError, match="missing in nodes_gdf.index"):
        graph.validate()


def test_to_csr_does_not_mutate_cached_adjacency_state():
    graph = undirected_line_graph()

    matrix = graph.to_csr(weight="time_min")

    assert matrix.shape == (4, 4)
    assert graph.adjacency_matrix is None  # to_csr is side-effect free
    assert graph.adjacency_nodelist == []
