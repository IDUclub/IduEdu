import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from iduedu.graph.urban_graph import UrbanGraph
from tests.factories import CRS, undirected_line_graph

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
