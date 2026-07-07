from pathlib import Path
from zipfile import ZipFile

import pytest
from geopandas.testing import assert_geodataframe_equal

from iduedu import read_urban_graph, write_urban_graph
from iduedu.graph.urban_graph import UrbanGraph
from tests.factories import tiny_public_transport_graph_with_sequence_attrs, undirected_line_graph

pytestmark = pytest.mark.unit


def test_write_read_urban_graph_roundtrip(tmp_path: Path):
    graph = undirected_line_graph()
    graph_path = tmp_path / "walk.urbangraph"

    result_path = write_urban_graph(graph, graph_path)
    loaded = read_urban_graph(graph_path)

    assert result_path == graph_path
    assert loaded.is_multigraph == graph.is_multigraph
    assert loaded.is_directed == graph.is_directed
    assert loaded.adjacency_weight == graph.adjacency_weight
    assert loaded.type == graph.type
    assert_geodataframe_equal(loaded.nodes_gdf, graph.nodes_gdf)
    assert_geodataframe_equal(loaded.edges_gdf, graph.edges_gdf)


def test_urban_graph_write_read_methods_roundtrip(tmp_path: Path):
    graph = undirected_line_graph()
    graph_path = tmp_path / "walk.urbangraph"

    result_path = graph.write(graph_path)
    loaded = UrbanGraph.read(graph_path)

    assert result_path == graph_path
    assert_geodataframe_equal(loaded.nodes_gdf, graph.nodes_gdf)
    assert_geodataframe_equal(loaded.edges_gdf, graph.edges_gdf)


def test_urban_graph_archive_contains_parquet_tables_and_metadata(tmp_path: Path):
    graph = undirected_line_graph()
    graph_path = tmp_path / "walk.urbangraph"

    write_urban_graph(graph, graph_path)

    with ZipFile(graph_path) as archive:
        members = set(archive.namelist())

    assert {"metadata.json", "nodes.parquet", "edges.parquet"} <= members


def test_write_urban_graph_preserves_adjacency_cache(tmp_path: Path):
    graph = undirected_line_graph()
    graph.update_adjacency_matrix(weight="time_min")
    graph_path = tmp_path / "walk.urbangraph"

    write_urban_graph(graph, graph_path, include_adjacency=True)
    loaded = read_urban_graph(graph_path)

    assert loaded.adjacency_matrix is not None
    assert loaded.adjacency_nodelist == graph.adjacency_nodelist
    assert (loaded.adjacency_matrix != graph.adjacency_matrix).nnz == 0


def test_write_read_urban_graph_roundtrip_with_list_valued_node_attrs(tmp_path: Path):
    graph = tiny_public_transport_graph_with_sequence_attrs()
    graph_path = tmp_path / "intermodal.urbangraph"

    write_urban_graph(graph, graph_path)
    loaded = read_urban_graph(graph_path)

    assert_geodataframe_equal(loaded.nodes_gdf, graph.nodes_gdf)
    assert_geodataframe_equal(loaded.edges_gdf, graph.edges_gdf)
    assert loaded.nodes_gdf.loc[200, "route_refs"] == ["A", "B"]
    assert loaded.nodes_gdf.loc[201, "route_names"] == ["Gamma", "Delta"]


def test_write_urban_graph_requires_urbangraph_suffix(tmp_path: Path):
    graph = undirected_line_graph()

    with pytest.raises(ValueError, match=".urbangraph"):
        write_urban_graph(graph, tmp_path / "walk.zip")
