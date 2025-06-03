# pylint: disable=redefined-outer-name, unused-import
import time

import pytest

from iduedu import (
    config,
    get_drive_graph,
    get_single_public_transport_graph,
    get_walk_graph,
    graph_to_gdf,
    join_pt_walk_graph,
)

config.change_logger_lvl("DEBUG")


@pytest.fixture(scope="module")
def walk_graph(bounds):
    time.sleep(0.5)
    print("\n Downloading walk graph for bounds \n")
    return get_walk_graph(polygon=bounds)


@pytest.fixture(scope="module")
def subway_graph(bounds):
    time.sleep(0.5)
    print("\n Downloading subway graph for bounds \n")
    return get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)


def test_get_drive_graph(bounds):
    time.sleep(0.5)
    g_drive = get_drive_graph(polygon=bounds, additional_edgedata=["highway", "maxspeed", "reg","ref", "name"])
    assert g_drive is not None
    assert len(g_drive.nodes) > 0
    assert len(g_drive.edges) > 0


def test_get_walk_graph(walk_graph):
    assert walk_graph is not None
    assert len(walk_graph.nodes) > 0
    assert len(walk_graph.edges) > 0


def test_get_single_public_transport_graph(bounds, subway_graph):
    time.sleep(0.5)
    assert subway_graph is not None
    assert len(subway_graph.nodes) > 0
    assert len(subway_graph.edges) > 0


def test_get_single_pt_graph_where_not_exist(bounds):
    time.sleep(0.5)
    train_graph = get_single_public_transport_graph(public_transport_type="train", polygon=bounds)
    assert len(train_graph.nodes) == 0
    assert len(train_graph.edges) == 0


def test_get_intermodal_graph(bounds, intermodal_graph):
    assert intermodal_graph is not None
    assert len(intermodal_graph.nodes) > 0
    assert len(intermodal_graph.edges) > 0


def test_join_pt_walk_graph(bounds, walk_graph, subway_graph):
    walk_and_subway = join_pt_walk_graph(subway_graph, walk_graph)
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
