# pylint: disable=redefined-outer-name
"""Graph builders on real OSM data of the test territory (Overpass, or its Geofabrik extract, see conftest).

Options and edge cases that do not depend on real data are covered by the unit tests with mocked responses.
"""

import pytest
from shapely import MultiPolygon, Polygon

from iduedu import (
    DEFAULT_REGISTRY_W_TRAIN,
    get_drive_graph,
    get_public_transport_graph,
    get_walk_graph,
    join_pt_walk_graph,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def drive_graph(bounds):
    print("\n Downloading drive graph for bounds \n")
    return get_drive_graph(territory=bounds, osm_edge_tags=["highway", "maxspeed", "reg", "ref", "name"])


@pytest.fixture(scope="module")
def walk_graph(bounds):
    print("\n Downloading walk graph for bounds \n")
    return get_walk_graph(territory=bounds)


@pytest.fixture(scope="module")
def subway_graph(bounds):
    print("\n Downloading subway graph for bounds \n")
    return get_public_transport_graph(transport_types="subway", territory=bounds)


@pytest.fixture(scope="module")
def ground_pt_graph(bounds):
    print("\n Downloading bus+tram graph for bounds \n")
    return get_public_transport_graph(transport_types=["bus", "tram"], territory=bounds)


def test_territory_boundary(bounds):
    assert isinstance(bounds, (Polygon, MultiPolygon))
    assert bounds.is_valid
    assert bounds.area > 0
    assert -180 <= bounds.bounds[0] <= bounds.bounds[2] <= 180  # EPSG:4326


# ---------------------------------------------------------------------------
# get_drive_graph
# ---------------------------------------------------------------------------


def test_get_drive_graph(drive_graph):
    assert drive_graph is not None
    assert len(drive_graph.nodes_gdf) > 0
    assert len(drive_graph.edges_gdf) > 0
    assert drive_graph.type == "drive"
    assert drive_graph.is_directed is True
    assert drive_graph.is_multigraph is True
    assert drive_graph.edge_direction_column == "oneway"
    assert drive_graph.edges_gdf["oneway"].dtype == bool
    assert set(drive_graph.edges_gdf["category"].dropna().unique()) <= {"federal", "regional", "local"}
    assert (drive_graph.edges_gdf["length_meter"] > 0).all()
    assert (drive_graph.edges_gdf["time_min"] > 0).all()


def test_get_drive_graph_adjacency_and_shortest_path(drive_graph):
    drive_graph.update_adjacency_matrix()
    assert drive_graph.adjacency_matrix.shape == (len(drive_graph.nodes_gdf), len(drive_graph.nodes_gdf))

    source = drive_graph.nodes_gdf.index[0]
    lengths = drive_graph.single_source_dijkstra(source, weight="time_min", output="distances")
    assert lengths.loc[source] == 0.0
    assert lengths.notna().sum() > 1


def test_get_drive_graph_drive_service_includes_service_ways(bounds):
    g_service = get_drive_graph(territory=bounds, network_type="drive_service")
    assert g_service is not None
    assert len(g_service.edges_gdf) > 0
    # drive_service is a superset filter, so it should never yield fewer edges than plain drive
    g_drive = get_drive_graph(territory=bounds)
    assert len(g_service.edges_gdf) >= len(g_drive.edges_gdf)


# ---------------------------------------------------------------------------
# get_walk_graph
# ---------------------------------------------------------------------------


def test_get_walk_graph(walk_graph):
    assert walk_graph is not None
    assert len(walk_graph.nodes_gdf) > 0
    assert len(walk_graph.edges_gdf) > 0
    assert walk_graph.type == "walk"
    assert walk_graph.is_directed is False
    assert walk_graph.is_multigraph is True
    assert walk_graph.edge_direction_column is None
    assert (walk_graph.edges_gdf["length_meter"] > 0).all()
    assert (walk_graph.edges_gdf["time_min"] > 0).all()


def test_get_walk_graph_custom_attrs(bounds):
    g_walk_custom = get_walk_graph(
        territory=bounds, osm_edge_tags=["surface", "footway"], clip_by_territory=True, keep_largest_subgraph=False
    )
    edge_attr_keys = set(g_walk_custom.edges_gdf.columns)
    assert g_walk_custom is not None
    assert len(g_walk_custom.nodes_gdf) > 0
    assert len(g_walk_custom.edges_gdf) > 0
    assert "surface" in edge_attr_keys
    assert "footway" in edge_attr_keys


# ---------------------------------------------------------------------------
# get_public_transport_graph
# ---------------------------------------------------------------------------


def test_get_single_public_transport_graph(subway_graph):
    assert subway_graph is not None
    assert len(subway_graph.nodes_gdf) > 0
    assert len(subway_graph.edges_gdf) > 0
    assert subway_graph.type == "public_transport"
    assert subway_graph.is_directed is True
    assert subway_graph.edge_direction_column == "oneway"


def test_get_public_transport_graph_multiple_types(ground_pt_graph):
    assert ground_pt_graph is not None
    assert len(ground_pt_graph.nodes_gdf) > 0
    edge_types = set(ground_pt_graph.edges_gdf["type"].dropna().unique())
    assert {"bus", "tram"} <= edge_types
    assert "boarding" in edge_types


def test_get_single_pt_graph_where_not_exist(bounds):
    train_graph = get_public_transport_graph(
        transport_types="train", territory=bounds, transport_registry=DEFAULT_REGISTRY_W_TRAIN
    )
    assert len(train_graph.nodes_gdf) == 0
    assert len(train_graph.edges_gdf) == 0
    assert train_graph.type == "public_transport"


# ---------------------------------------------------------------------------
# join_pt_walk_graph / get_intermodal_graph
# ---------------------------------------------------------------------------


def test_join_pt_walk_graph(walk_graph, subway_graph):
    walk_and_subway = join_pt_walk_graph(subway_graph, walk_graph)
    assert walk_and_subway is not None
    assert len(walk_and_subway.nodes_gdf) > 0
    assert len(walk_and_subway.edges_gdf) > 0
    assert walk_and_subway.type == "intermodal"
    assert walk_and_subway.is_directed is True
    # every subway platform-like node should have been projected into the walk network
    assert "walk" in set(walk_and_subway.edges_gdf["type"].dropna().unique())


def test_join_pt_walk_graph_respects_max_dist(walk_graph, subway_graph):
    joined_far = join_pt_walk_graph(subway_graph, walk_graph, max_dist=1000, keep_largest_subgraph=False)
    joined_near = join_pt_walk_graph(subway_graph, walk_graph, max_dist=1, keep_largest_subgraph=False)

    # a larger snapping radius connects more platforms to the walk network via link edges
    far_walk_edges = (joined_far.edges_gdf["type"] == "walk").sum()
    near_walk_edges = (joined_near.edges_gdf["type"] == "walk").sum()
    assert far_walk_edges >= near_walk_edges


def test_get_intermodal_graph(intermodal_graph):
    assert intermodal_graph is not None
    assert len(intermodal_graph.nodes_gdf) > 0
    assert len(intermodal_graph.edges_gdf) > 0
    assert intermodal_graph.type == "intermodal"
    assert intermodal_graph.is_directed is True

    edge_types = set(intermodal_graph.edges_gdf["type"].dropna().unique())
    assert "walk" in edge_types
    assert edge_types & {"bus", "tram", "trolleybus", "subway"}


def test_get_intermodal_graph_is_routable(intermodal_graph):
    graph = intermodal_graph.copy()
    graph.update_adjacency_matrix()

    source = graph.nodes_gdf.index[0]
    lengths = graph.single_source_dijkstra(source, weight="time_min", output="distances")

    assert lengths.loc[source] == 0.0
    assert lengths.notna().sum() > 1
