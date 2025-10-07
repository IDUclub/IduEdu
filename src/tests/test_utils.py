from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from iduedu import (
    clip_nx_graph,
    gdf_to_graph,
    graph_to_gdf,
    keep_largest_strongly_connected_component,
    read_gml,
    reproject_graph,
    write_gml,
)


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


def test_keep_largest_scc():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    G.add_edges_from([(10, 11), (11, 10)])
    pruned = keep_largest_strongly_connected_component(G)
    assert set(pruned.nodes()) == {1, 2, 3}
    assert pruned.is_directed()


def test_graph_to_gdf_nodes_only(simple_graph_3857):
    G = simple_graph_3857
    nodes = graph_to_gdf(G, edges=False, nodes=True)
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert len(nodes) == G.number_of_nodes()
    assert nodes.crs == G.graph["crs"]


def test_graph_to_gdf_edges_only_restore(simple_graph_3857):
    G = simple_graph_3857
    edges = graph_to_gdf(G, edges=True, nodes=False, restore_edge_geom=True)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert (~edges["geometry"].is_empty).all()


def test_graph_to_gdf_no_crs_raises(simple_graph_3857):
    G = simple_graph_3857
    del G.graph["crs"]
    with pytest.raises(ValueError):
        graph_to_gdf(G)


def test_gdf_to_graph_basic():
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[LineString([(30.0, 59.0), (30.001, 59.0)]), LineString([(30.001, 59.0), (30.002, 59.001)])],
        crs=4326,
    )
    G = gdf_to_graph(gdf, project_gdf_attr=True, reproject_to_utm_crs=True, speed=5, check_intersections=True)
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


def test_graph_to_gdf_restore_geom_integration(intermodal_graph):
    graph_gdf = graph_to_gdf(intermodal_graph, restore_edge_geom=True)
    assert graph_gdf is not None
    assert not graph_gdf.empty
