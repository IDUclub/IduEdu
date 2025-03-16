from iduedu import (
    config,
    get_drive_graph,
    get_walk_graph,
    get_single_public_transport_graph,
    get_all_public_transport_graph,
    get_intermodal_graph,
    join_pt_walk_graph,
    graph_to_gdf,
)
from .test_downloaders import bounds

config.change_logger_lvl("DEBUG")



def test_get_drive_graph(bounds):
    G_drive = get_drive_graph(polygon=bounds, additional_edgedata=["highway", "maxspeed", "reg", "ref", "name"])
    assert G_drive is not None
    assert len(G_drive.nodes) > 0
    assert len(G_drive.edges) > 0


def test_get_walk_graph(bounds):
    G_walk = get_walk_graph(polygon=bounds)
    assert G_walk is not None
    assert len(G_walk.nodes) > 0
    assert len(G_walk.edges) > 0


def test_get_single_public_transport_graph(bounds):
    G_subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    assert G_subway is not None
    assert len(G_subway.nodes) > 0
    assert len(G_subway.edges) > 0


def test_get_all_public_transport_graph(bounds):
    G_public_t = get_all_public_transport_graph(polygon=bounds, clip_by_bounds=True, keep_geometry=False)
    assert G_public_t is not None
    assert len(G_public_t.nodes) > 0
    assert len(G_public_t.edges) > 0


def test_get_intermodal_graph(bounds):
    G_intermodal = get_intermodal_graph(polygon=bounds, clip_by_bounds=False, max_dist=50)
    assert G_intermodal is not None
    assert len(G_intermodal.nodes) > 0
    assert len(G_intermodal.edges) > 0


def test_join_pt_walk_graph(bounds):
    G_subway = get_single_public_transport_graph(public_transport_type="subway", polygon=bounds)
    G_walk = get_walk_graph(polygon=bounds)
    G_walk_and_subway = join_pt_walk_graph(G_subway, G_walk)
    assert G_walk_and_subway is not None
    assert len(G_walk_and_subway.nodes) > 0
    assert len(G_walk_and_subway.edges) > 0


def test_graph_to_gdf(bounds):
    G_intermodal = get_intermodal_graph(polygon=bounds, clip_by_bounds=False, max_dist=50)
    graph_gdf = graph_to_gdf(G_intermodal)
    assert graph_gdf is not None
    assert not graph_gdf.empty
