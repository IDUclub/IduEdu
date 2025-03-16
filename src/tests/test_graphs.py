# pylint: disable=redefined-outer-name, unused-import

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


def test_get_drive_graph(bounds):
    g_drive = get_drive_graph(polygon=bounds, additional_edgedata=["highway", "maxspeed", "reg", "ref", "name"])
    assert g_drive is not None
    assert len(g_drive.nodes) > 0
    assert len(g_drive.edges) > 0


def test_get_walk_graph(bounds):
    g_walk = get_walk_graph(polygon=bounds)
    assert g_walk is not None
    assert len(g_walk.nodes) > 0
    assert len(g_walk.edges) > 0


def test_get_single_public_transport_graph(bounds):
    g_subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    assert g_subway is not None
    assert len(g_subway.nodes) > 0
    assert len(g_subway.edges) > 0


def test_get_all_public_transport_graph(bounds):
    g_public_t = get_all_public_transport_graph(polygon=bounds, clip_by_bounds=True, keep_geometry=False)
    assert g_public_t is not None
    assert len(g_public_t.nodes) > 0
    assert len(g_public_t.edges) > 0


def test_get_intermodal_graph(bounds):
    g_intermodal = get_intermodal_graph(polygon=bounds, clip_by_bounds=False, max_dist=50)
    assert g_intermodal is not None
    assert len(g_intermodal.nodes) > 0
    assert len(g_intermodal.edges) > 0


def test_join_pt_walk_graph(bounds):
    subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    walk = get_walk_graph(polygon=bounds)
    walk_and_subway = join_pt_walk_graph(subway, walk)
    assert walk_and_subway is not None
    assert len(walk_and_subway.nodes) > 0
    assert len(walk_and_subway.edges) > 0


def test_graph_to_gdf(bounds):
    intermodal = get_intermodal_graph(polygon=bounds, clip_by_bounds=False, max_dist=50)
    graph_gdf = graph_to_gdf(intermodal)
    assert graph_gdf is not None
    assert not graph_gdf.empty
