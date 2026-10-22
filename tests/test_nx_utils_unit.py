from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from iduedu import clip_nx_graph, gdf2graph, graph2gdf, keep_largest_nx_component, read_gml, reproject_graph, write_gml
from iduedu.graph.adapters import nx_graph2urban_graph, urban_graph2nx_graph
from tests.factories import undirected_line_graph

pytestmark = pytest.mark.unit


@pytest.fixture()
def simple_graph_3857() -> nx.Graph:
    G = nx.Graph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=100.0, y=0.0)
    G.add_node(3, x=100.0, y=100.0)
    G.add_edge(1, 2, geometry=LineString([(0.0, 0.0), (100.0, 0.0)]), some_attr="a")
    G.add_edge(2, 3)  # missing geometry
    G.graph["crs"] = 3857
    return G


@pytest.fixture()
def simple_graph_4326() -> nx.Graph:
    G = nx.Graph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=0.001, y=0.0)
    G.add_edge(1, 2, geometry=LineString([(0.0, 0.0), (0.001, 0.0)]))
    G.graph["crs"] = 4326
    return G


def test_keep_largest_nx_component_directed_uses_weak_components():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    G.add_edges_from([(10, 11), (11, 10)])

    pruned = keep_largest_nx_component(G)

    assert set(pruned.nodes()) == {1, 2, 3}
    assert pruned.is_directed()


def test_keep_largest_nx_component_undirected_keeps_largest_component():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (10, 11)])

    pruned = keep_largest_nx_component(G)

    assert set(pruned.nodes()) == {1, 2, 3}
    assert not pruned.is_directed()


def test_keep_largest_nx_component_empty_graph_returns_empty_copy():
    G = nx.Graph()

    pruned = keep_largest_nx_component(G)

    assert pruned is not G
    assert pruned.number_of_nodes() == 0


def test_graph_to_gdf_nodes_only(simple_graph_3857):
    G = simple_graph_3857
    nodes = graph2gdf(G, edges=False, nodes=True)
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert len(nodes) == G.number_of_nodes()
    assert nodes.crs == G.graph["crs"]


def test_graph_to_gdf_edges_only_restore(simple_graph_3857):
    G = simple_graph_3857
    edges = graph2gdf(G, edges=True, nodes=False, restore_edge_geom=True)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert (~edges["geometry"].is_empty).all()


def test_graph_to_gdf_nodes_and_edges_combined(simple_graph_3857):
    G = simple_graph_3857
    graph_gdf = graph2gdf(G, edges=True, nodes=True, restore_edge_geom=True)

    assert isinstance(graph_gdf, gpd.GeoDataFrame)
    assert len(graph_gdf) == G.number_of_nodes() + G.number_of_edges()
    assert graph_gdf.crs == G.graph["crs"]


def test_graph_to_gdf_no_crs_raises(simple_graph_3857):
    G = simple_graph_3857
    del G.graph["crs"]
    with pytest.raises(ValueError):
        graph2gdf(G)


def test_gdf_to_graph_basic():
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[LineString([(30.0, 59.0), (30.001, 59.0)]), LineString([(30.001, 59.0), (30.002, 59.001)])],
        crs=4326,
    )
    G = gdf2graph(gdf, project_gdf_attr=True, reproject_to_utm_crs=True, speed=5, check_intersections=True)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_edges() >= 2
    any_edge = next(iter(G.edges(data=True)))
    assert "length_meter" in any_edge[2]
    assert "time_min" in any_edge[2]
    assert any("name" in data for _, _, data in G.edges(data=True))


def test_write_read_gml_roundtrip(tmp_path: Path, simple_graph_3857):
    G = simple_graph_3857
    gml_file = tmp_path / "graph.gml"
    write_gml(G, str(gml_file))
    assert gml_file.exists()
    G2 = read_gml(str(gml_file))
    has_geom_edges_G = {(u, v) for u, v, d in G.edges(data=True) if isinstance(d.get("geometry"), BaseGeometry)}
    for u, v in has_geom_edges_G:
        d2 = G2.get_edge_data(u, v)
        if isinstance(d2, dict) and "geometry" in d2:
            assert isinstance(d2["geometry"], BaseGeometry)


def test_reproject_graph_updates_coords(simple_graph_4326):
    G = simple_graph_4326
    before = [(n, (G.nodes[n]["x"], G.nodes[n]["y"])) for n in G.nodes()]
    reproject_graph(G, 3857)
    assert G.graph["crs"] == 3857 or getattr(G.graph["crs"], "to_epsg", lambda: None)() == 3857
    after = [(n, (G.nodes[n]["x"], G.nodes[n]["y"])) for n in G.nodes()]
    assert any(abs(ax - bx) > 10 or abs(ay - by) > 10 for ((_, (bx, by)), (_, (ax, ay))) in zip(before, after))
    for _, _, d in G.edges(data=True):
        if "geometry" in d and isinstance(d["geometry"], BaseGeometry):
            assert d["geometry"].geom_type == "LineString"


def test_clip_nx_graph(simple_graph_3857):
    G = simple_graph_3857
    poly = Polygon([(-10, -10), (110, -10), (110, 10), (-10, 10)])
    clipped = clip_nx_graph(G, poly)
    assert set(clipped.nodes()) == {1, 2}
    assert clipped.has_edge(1, 2)
    assert not clipped.has_edge(2, 3)


CRS = "EPSG:3857"


def _nx_line_graph(graph_cls=nx.Graph, *, with_oneway=False):
    graph = graph_cls()
    graph.graph["crs"] = CRS
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=10.0, y=0.0)
    edge_attrs = {
        "geometry": LineString([(0.0, 0.0), (10.0, 0.0)]),
        "length_meter": 10.0,
        "time_min": 1.0,
    }
    if with_oneway:
        edge_attrs["oneway"] = True
    if graph.is_multigraph():
        graph.add_edge(0, 1, key=0, **edge_attrs)
    else:
        graph.add_edge(0, 1, **edge_attrs)
    return graph


def test_urban_graph2nx_graph_preserves_nodes_edges_and_attrs():
    graph = undirected_line_graph()

    nx_graph = urban_graph2nx_graph(graph)

    assert isinstance(nx_graph, nx.Graph) and not nx_graph.is_directed()
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3
    assert nx_graph.nodes[0]["x"] == 0.0
    assert nx_graph[0][1]["time_min"] == 1.0
    assert nx_graph.graph["crs"] is not None


def test_round_trip_nx_preserves_structure_and_weights():
    graph = undirected_line_graph()

    restored = nx_graph2urban_graph(urban_graph2nx_graph(graph))

    assert len(restored.nodes_gdf) == 4
    assert len(restored.edges_gdf) == 3
    assert not restored.is_multigraph
    assert not restored.is_directed
    edge_01 = restored.edges_gdf[(restored.edges_gdf["u"] == 0) & (restored.edges_gdf["v"] == 1)].iloc[0]
    assert edge_01["time_min"] == 1.0


def test_oneway_column_makes_graph_directed():
    nx_graph = _nx_line_graph(nx.MultiGraph, with_oneway=True)

    urban = nx_graph2urban_graph(nx_graph)

    assert urban.edge_direction_column == "oneway"
    assert urban.is_directed


def test_directed_nx_graph_emits_warning():
    nx_graph = _nx_line_graph(nx.DiGraph)

    with pytest.warns(UserWarning, match="Directed NetworkX graphs"):
        nx_graph2urban_graph(nx_graph)


def test_nx_graph2urban_graph_rejects_non_graph():
    with pytest.raises(TypeError, match="nx.Graph"):
        nx_graph2urban_graph("not-a-graph")


def test_nx_graph2urban_graph_requires_crs():
    graph = nx.Graph()  # no crs attribute set

    with pytest.raises(ValueError, match="crs"):
        nx_graph2urban_graph(graph)


def test_nx_graph2urban_graph_requires_nodes_and_edges():
    no_nodes = nx.Graph()
    no_nodes.graph["crs"] = CRS
    with pytest.raises(ValueError, match="no nodes"):
        nx_graph2urban_graph(no_nodes)

    no_edges = nx.Graph()
    no_edges.graph["crs"] = CRS
    no_edges.add_node(0, x=0.0, y=0.0)
    with pytest.raises(ValueError, match="no edges"):
        nx_graph2urban_graph(no_edges)


def test_nx_graph2urban_graph_requires_node_coordinates():
    graph = nx.Graph()
    graph.graph["crs"] = CRS
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, geometry=LineString([(0, 0), (1, 0)]), length_meter=1.0, time_min=1.0)

    with pytest.raises(ValueError, match="required 'x'"):
        nx_graph2urban_graph(graph)
