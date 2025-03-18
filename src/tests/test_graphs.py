# pylint: disable=redefined-outer-name, unused-import
import pytest

from iduedu import (
    config,
    get_all_public_transport_graph,
    get_drive_graph,
    get_intermodal_graph,
    get_single_public_transport_graph,
    get_walk_graph,
    graph_to_gdf,
    join_pt_walk_graph,
)

from .test_downloaders import bounds

config.change_logger_lvl("DEBUG")


@pytest.fixture(scope="module")
def walk_graph(bounds):
    return get_walk_graph(polygon=bounds)


@pytest.fixture(scope="module")
def intermodal_graph(bounds):
    return get_intermodal_graph(polygon=bounds)


def test_get_drive_graph(bounds):
    g_drive = get_drive_graph(polygon=bounds, additional_edgedata=["highway", "maxspeed", "reg", "ref", "name"])
    assert g_drive is not None
    assert len(g_drive.nodes) > 0
    assert len(g_drive.edges) > 0


def test_get_walk_graph(walk_graph):
    assert walk_graph is not None
    assert len(walk_graph.nodes) > 0
    assert len(walk_graph.edges) > 0


def test_get_single_public_transport_graph(bounds):
    g_subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    assert g_subway is not None
    assert len(g_subway.nodes) > 0
    assert len(g_subway.edges) > 0


def test_get_single_pt_graph_where_not_exist(bounds):
    g_train = get_single_public_transport_graph(public_transport_type="train", polygon=bounds)
    assert len(g_train.nodes) == 0
    assert len(g_train.edges) == 0


def test_get_all_public_transport_graph(bounds):
    g_public_t = get_all_public_transport_graph(polygon=bounds, clip_by_bounds=True, keep_geometry=False)
    assert g_public_t is not None
    assert len(g_public_t.nodes) > 0
    assert len(g_public_t.edges) > 0


def test_get_intermodal_graph(bounds, intermodal_graph):
    assert intermodal_graph is not None
    assert len(intermodal_graph.nodes) > 0
    assert len(intermodal_graph.edges) > 0


def test_join_pt_walk_graph(bounds, walk_graph):
    subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    walk_and_subway = join_pt_walk_graph(subway, walk_graph)
    assert walk_and_subway is not None
    assert len(walk_and_subway.nodes) > 0
    assert len(walk_and_subway.edges) > 0


def test_graph_to_gdf(bounds, intermodal_graph):
    graph_gdf = graph_to_gdf(intermodal_graph)
    assert graph_gdf is not None
    assert not graph_gdf.empty


def test_graph_to_gdf_restore_geom(bounds, intermodal_graph):
    graph_gdf = graph_to_gdf(intermodal_graph, restore_edge_geom=True)
    assert graph_gdf is not None
    assert not graph_gdf.empty
    assert graph_gdf["geometry"].is_empty.any() == False
