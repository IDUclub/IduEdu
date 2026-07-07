import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

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
