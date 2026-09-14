import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from iduedu import config
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders import intermodal_builders
from iduedu.graph_builders.intermodal_builders import get_intermodal_graph, join_pt_walk_graph
from tests.factories import CRS, tiny_public_transport_graph_with_sequence_attrs, tiny_walk_graph

pytestmark = pytest.mark.unit


def _pt_graph_without_platforms() -> UrbanGraph:
    coords = {200: (5.0, 5.0), 201: (15.0, 5.0)}
    nodes = gpd.GeoDataFrame(
        {"type": ["bus", "bus"], "route": ["A", "A"]},
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(coords.keys()),
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {
            "u": [200],
            "v": [201],
            "k": [0],
            "type": ["bus"],
            "length_meter": [10.0],
            "time_min": [1.0],
            "oneway": [True],
        },
        geometry=[LineString([coords[200], coords[201]])],
        crs=CRS,
    )
    return UrbanGraph(
        nodes,
        edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="public_transport",
    )


def _walk_line_graph(x_coords: tuple[float, ...] = (0.0, 60.0)) -> UrbanGraph:
    # A straight walk chain along y=0, long enough for the subway objects below to reach it.
    coords = {100 + offset: (x, 0.0) for offset, x in enumerate(x_coords)}
    node_ids = list(coords)
    nodes = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(node_ids),
        crs=CRS,
    )
    geometries = [LineString([coords[u], coords[v]]) for u, v in zip(node_ids, node_ids[1:])]
    edges = gpd.GeoDataFrame(
        {
            "u": node_ids[:-1],
            "v": node_ids[1:],
            "k": [0] * len(geometries),
            "type": ["walk"] * len(geometries),
            "length_meter": [geometry.length for geometry in geometries],
            "time_min": [geometry.length / 5.0 for geometry in geometries],
            "oneway": [False] * len(geometries),
        },
        geometry=geometries,
        crs=CRS,
    )
    return UrbanGraph(
        nodes,
        edges,
        is_multigraph=True,
        is_directed=False,
        edge_direction_column=None,
        crs=CRS,
        graph_type="walk",
    )


def _subway_pt_graph(entrance_xy: tuple[float, float] = (10.0, 5.0)) -> UrbanGraph:
    # entrance -> station -> platform -> stop, with the platform lying close to the walk edge.
    coords = {
        200: entrance_xy,  # entrance, on the surface next to the walk edge
        201: (10.0, -15.0),  # station
        202: (40.0, 3.0),  # platform, underground but geometrically close to the walk edge
        203: (40.0, -15.0),  # stop
    }
    nodes = gpd.GeoDataFrame(
        {
            "type": ["subway_entry_exit", "subway_station", "subway_platform", "subway"],
            "route": [None, None, "M1", "M1"],
        },
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(coords.keys()),
        crs=CRS,
    )
    edge_rows = [
        (200, 201, "subway_entrance", True),
        (201, 200, "subway_exit", True),
        (201, 202, "subway_station", False),
        (203, 202, "boarding", False),
    ]
    geometries = [LineString([coords[u], coords[v]]) for u, v, _, _ in edge_rows]
    edges = gpd.GeoDataFrame(
        {
            "u": [u for u, _, _, _ in edge_rows],
            "v": [v for _, v, _, _ in edge_rows],
            "k": [0] * len(edge_rows),
            "type": [edge_type for _, _, edge_type, _ in edge_rows],
            "length_meter": [round(geometry.length, 3) for geometry in geometries],
            "time_min": [round(geometry.length / 60.0, 3) for geometry in geometries],
            "oneway": [oneway for _, _, _, oneway in edge_rows],
        },
        geometry=geometries,
        crs=CRS,
    )
    return UrbanGraph(
        nodes,
        edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="public_transport",
    )


def test_join_pt_walk_graph_does_not_project_subway_platforms():
    walk_graph = _walk_line_graph()
    pt_graph = _subway_pt_graph()

    intermodal = join_pt_walk_graph(pt_graph, walk_graph, max_dist=10.0, keep_largest_subgraph=False)

    node_types = intermodal.nodes_gdf["type"]
    platform_nodes = set(node_types[node_types == "subway_platform"].index)
    entrance_nodes = set(node_types[node_types == "subway_entry_exit"].index)

    walk_edges = intermodal.edges_gdf[intermodal.edges_gdf["type"] == "walk"]
    nodes_on_walk = set(walk_edges["u"]) | set(walk_edges["v"])

    # The platform is close enough to the walk edge to be projected, but it is underground:
    # the only way in is through the entrance, otherwise routes bypass entrance/exit edges.
    assert len(platform_nodes) == 1
    assert not platform_nodes & nodes_on_walk
    assert entrance_nodes & nodes_on_walk
    assert {"subway_entrance", "subway_exit", "subway_station", "boarding"} <= set(intermodal.edges_gdf["type"])


@pytest.mark.parametrize(
    "entrance_xy",
    [
        pytest.param((30.0, 0.0), id="entrance-on-walk-edge"),
        pytest.param((60.0, 0.0), id="entrance-on-walk-node"),
        pytest.param((30.0, 3.0), id="entrance-beside-walk-edge"),
    ],
)
def test_join_pt_walk_graph_connects_entrance_lying_on_the_walk_network(entrance_xy):
    # OSM subway entrances are usually vertices of the pedestrian way they belong to, so the
    # connector edge would be zero-length and the subway used to end up detached from the graph.
    walk_graph = _walk_line_graph()
    pt_graph = _subway_pt_graph(entrance_xy=entrance_xy)

    intermodal = join_pt_walk_graph(pt_graph, walk_graph, max_dist=10.0, keep_largest_subgraph=False)

    node_types = intermodal.nodes_gdf["type"]
    entrance_nodes = node_types[node_types == "subway_entry_exit"].index
    assert len(entrance_nodes) == 1
    entrance_node = entrance_nodes[0]

    edges = intermodal.edges_gdf
    walk_edges = edges[edges["type"] == "walk"]
    assert ((walk_edges["u"] == entrance_node) | (walk_edges["v"] == entrance_node)).any()

    # Nothing is left dangling: the subway survives the largest-component filter.
    kept = join_pt_walk_graph(pt_graph, walk_graph, max_dist=10.0, keep_largest_subgraph=True)
    assert len(kept.nodes_gdf) == len(intermodal.nodes_gdf)
    assert "subway_platform" in set(kept.nodes_gdf["type"].dropna())


def _capture_logs(level, action):
    messages: list[str] = []
    sink_id = config.logger.add(messages.append, level=level)
    try:
        action()
    finally:
        config.logger.remove(sink_id)
    return messages


def test_join_pt_walk_graph_reports_platforms_out_of_reach():
    walk_graph = _walk_line_graph()
    pt_graph = _subway_pt_graph()  # the entrance is 5 m away from the walk edge

    messages = _capture_logs(
        "INFO", lambda: join_pt_walk_graph(pt_graph, walk_graph, max_dist=1.0, keep_largest_subgraph=False)
    )

    assert any("stay unconnected" in message and "subway_entry_exit" in message for message in messages)


def test_join_pt_walk_graph_warns_which_pt_nodes_the_largest_component_drops():
    # The walk chain has more nodes than the subway, so the subway is the component being dropped.
    walk_graph = _walk_line_graph((0.0, 15.0, 30.0, 45.0, 60.0))
    pt_graph = _subway_pt_graph()  # out of reach below, so the whole subway forms its own component

    messages = _capture_logs(
        "WARNING", lambda: join_pt_walk_graph(pt_graph, walk_graph, max_dist=1.0, keep_largest_subgraph=True)
    )

    assert any(
        "dropped with the smaller components" in message and "subway_platform" in message for message in messages
    )


def test_join_pt_walk_graph_when_every_pt_node_is_projected_and_no_pt_edges_remain():
    # A lone platform with no public-transport edges: nothing is left for the PT side of the join.
    walk_graph = _walk_line_graph()
    pt_graph = UrbanGraph(
        gpd.GeoDataFrame({"type": ["platform"]}, geometry=[Point(30.0, 2.0)], index=pd.Index([200]), crs=CRS),
        gpd.GeoDataFrame(),
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="public_transport",
    )

    intermodal = join_pt_walk_graph(pt_graph, walk_graph, max_dist=10.0, keep_largest_subgraph=False)

    assert intermodal.type == "intermodal"
    platform_nodes = intermodal.nodes_gdf.index[intermodal.nodes_gdf["type"] == "platform"]
    assert len(platform_nodes) == 1
    edges = intermodal.edges_gdf
    assert ((edges["u"] == platform_nodes[0]) | (edges["v"] == platform_nodes[0])).any()


def test_join_pt_walk_graph_keeps_edge_keys_unique_when_pt_and_walk_share_node_pairs():
    # Two entrances lying on the same walk edge are linked to each other, so after projection the
    # same (u, v) pair carries both a walk edge and a public-transport edge.
    walk_graph = _walk_line_graph()
    coords = {200: (20.0, 0.0), 201: (40.0, 0.0)}
    nodes = gpd.GeoDataFrame(
        {"type": ["subway_entry_exit", "subway_entry_exit"]},
        geometry=[Point(xy) for xy in coords.values()],
        index=pd.Index(coords.keys()),
        crs=CRS,
    )
    geometries = [LineString([coords[200], coords[201]]), LineString([coords[201], coords[200]])]
    edges = gpd.GeoDataFrame(
        {
            "u": [200, 201],
            "v": [201, 200],
            "k": [0, 0],
            "type": ["subway_transfer"] * 2,
            "length_meter": [geometry.length for geometry in geometries],
            "time_min": [geometry.length / 60.0 for geometry in geometries],
            "oneway": [False, False],
        },
        geometry=geometries,
        crs=CRS,
    )
    pt_graph = UrbanGraph(
        nodes,
        edges,
        is_multigraph=True,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="public_transport",
    )

    intermodal = join_pt_walk_graph(pt_graph, walk_graph, max_dist=10.0, keep_largest_subgraph=False)

    assert not intermodal.edges_gdf[["u", "v", "k"]].duplicated().any()
    assert {"walk", "subway_transfer"} <= set(intermodal.edges_gdf["type"])


def test_join_pt_walk_graph_without_platform_nodes():
    walk_graph = tiny_walk_graph()
    pt_graph = _pt_graph_without_platforms()

    intermodal = join_pt_walk_graph(pt_graph, walk_graph, keep_largest_subgraph=False)

    # PT nodes are carried over without projection; both networks coexist in the result
    assert len(intermodal.nodes_gdf) == len(walk_graph.nodes_gdf) + len(pt_graph.nodes_gdf)
    assert "bus" in set(intermodal.edges_gdf["type"].dropna().unique())
    assert "walk" in set(intermodal.edges_gdf["type"].dropna().unique())


def test_join_pt_walk_graph_collapses_sequence_attrs_with_different_lengths():
    walk_graph = tiny_walk_graph()
    pt_graph = tiny_public_transport_graph_with_sequence_attrs()

    intermodal = join_pt_walk_graph(
        pt_graph,
        walk_graph,
        max_dist=10.0,
        keep_largest_subgraph=False,
        add_link_edge=True,
    )

    copied = intermodal.nodes_gdf.dropna(subset=["route_refs", "route_names"])
    assert not copied.empty
    row = copied.iloc[0]
    assert row["route_refs"] == ["A", "B"]
    assert row["route_names"] == "Alpha"


def test_get_intermodal_graph_defaults_walk_keep_largest_subgraph_to_false(monkeypatch):
    calls = {}

    def fake_boundary(**kwargs):
        return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def fake_walk_graph(**kwargs):
        calls["walk_kwargs"] = kwargs
        return tiny_walk_graph()

    def fake_public_transport_graph(**kwargs):
        calls["pt_kwargs"] = kwargs
        return tiny_public_transport_graph_with_sequence_attrs()

    def fake_join_pt_walk_graph(pt_g, walk_g, **kwargs):
        calls["join_kwargs"] = kwargs
        return walk_g

    monkeypatch.setattr(intermodal_builders, "get_4326_boundary", fake_boundary)
    monkeypatch.setattr(intermodal_builders, "get_walk_graph", fake_walk_graph)
    monkeypatch.setattr(intermodal_builders, "get_public_transport_graph", fake_public_transport_graph)
    monkeypatch.setattr(intermodal_builders, "join_pt_walk_graph", fake_join_pt_walk_graph)

    get_intermodal_graph(territory=fake_boundary(), keep_largest_subgraph=True)

    assert calls["walk_kwargs"]["keep_largest_subgraph"] is False
    assert calls["join_kwargs"]["keep_largest_subgraph"] is True


def test_get_intermodal_graph_respects_explicit_walk_keep_largest_subgraph(monkeypatch):
    calls = {}

    def fake_boundary(**kwargs):
        return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def fake_walk_graph(**kwargs):
        calls["walk_kwargs"] = kwargs
        return tiny_walk_graph()

    def fake_public_transport_graph(**kwargs):
        return tiny_public_transport_graph_with_sequence_attrs()

    monkeypatch.setattr(intermodal_builders, "get_4326_boundary", fake_boundary)
    monkeypatch.setattr(intermodal_builders, "get_walk_graph", fake_walk_graph)
    monkeypatch.setattr(intermodal_builders, "get_public_transport_graph", fake_public_transport_graph)
    monkeypatch.setattr(intermodal_builders, "join_pt_walk_graph", lambda pt_g, walk_g, **kwargs: walk_g)

    get_intermodal_graph(territory=fake_boundary(), walk_kwargs={"keep_largest_subgraph": True})

    assert calls["walk_kwargs"]["keep_largest_subgraph"] is True
