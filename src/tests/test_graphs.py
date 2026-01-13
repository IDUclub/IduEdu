# pylint: disable=redefined-outer-name, unused-import

import geopandas as gpd
import pytest

from iduedu import (
    config,
    get_drive_graph,
    get_public_transport_graph,
    get_walk_graph,
    join_pt_walk_graph,
)
from iduedu.modules.overpass.overpass_downloaders import RequestError

config.configure_logging("DEBUG")


@pytest.fixture(scope="module")
def walk_graph(bounds):
    print("\n Downloading walk graph for bounds \n")
    return get_walk_graph(territory=bounds)


@pytest.fixture(scope="module")
def subway_graph(bounds):
    print("\n Downloading subway graph for bounds \n")
    return get_public_transport_graph(transport_types="subway", territory=bounds)


def test_get_drive_graph(bounds):
    g_drive = get_drive_graph(territory=bounds, osm_edge_tags=["highway", "maxspeed", "reg", "ref", "name"])
    assert g_drive is not None
    assert len(g_drive.nodes) > 0
    assert len(g_drive.edges) > 0


def test_get_walk_graph(walk_graph):
    assert walk_graph is not None
    assert len(walk_graph.nodes) > 0
    assert len(walk_graph.edges) > 0


def test_get_walk_graph_wrong_bounds(bounds):

    wrong_poly = gpd.GeoDataFrame(geometry=[bounds], crs=4326).to_crs(3857).iloc[0].geometry
    with pytest.raises(RequestError):
        _ = get_walk_graph(territory=wrong_poly)


def test_get_walk_graph_from_cache_custom_attrs(bounds, walk_graph):
    g_walk_custom = get_walk_graph(
        territory=bounds, osm_edge_tags=["surface", "footway"], clip_by_territory=True, keep_largest_subgraph=False
    )
    edge_attr_keys = {k for _, _, data in g_walk_custom.edges(data=True) for k in data.keys()}
    assert g_walk_custom is not None
    assert len(g_walk_custom.nodes) > 0
    assert len(g_walk_custom.edges) > 0
    assert "surface" in edge_attr_keys
    assert "footway" in edge_attr_keys


def test_get_single_public_transport_graph(bounds, subway_graph):
    assert subway_graph is not None
    assert len(subway_graph.nodes) > 0
    assert len(subway_graph.edges) > 0


def test_get_single_pt_graph_where_not_exist(bounds):
    train_graph = get_public_transport_graph(transport_types="train", territory=bounds)
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
