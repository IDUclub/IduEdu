import numpy as np
import pandas as pd
import pytest
from shapely import box

from iduedu import get_drive_graph, get_walk_graph
from iduedu.graph_builders import drive_walk_builders

pytestmark = pytest.mark.unit

LAT = 59.0


def _way(way_id, lons, tags, lat=LAT):
    return {
        "type": "way",
        "id": way_id,
        "nodes": [],
        "geometry": np.array([(lon, lat) for lon in lons], dtype=float),
        "tags": tags,
    }


def _assert_runs_from_u_to_v(graph):
    points = graph.nodes_gdf.geometry
    for u, v, line in zip(graph.edges_gdf["u"], graph.edges_gdf["v"], graph.edges_gdf.geometry):
        assert line.coords[0] == pytest.approx(points[u].coords[0])
        assert line.coords[-1] == pytest.approx(points[v].coords[0])


def _mock(monkeypatch, rows):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))


@pytest.fixture
def ways(monkeypatch):
    """A street drawn as five ways in a row: plain, bridge, plain (same name), then two with another name."""
    rows = [
        _way(1, [30.000, 30.001, 30.002], {"highway": "residential", "name": "Main"}),
        _way(2, [30.002, 30.003], {"highway": "residential", "name": "Main", "bridge": "yes"}),
        _way(3, [30.003, 30.004], {"highway": "residential", "name": "Main"}),
        _way(4, [30.004, 30.005], {"highway": "residential", "name": "Side"}),
        _way(5, [30.005, 30.006, 30.007], {"highway": "residential", "name": "Side"}),
        # Drawn against the others: it meets way 5 head to head and, both being two-way, is turned and merged.
        _way(6, [30.008, 30.007], {"highway": "residential", "name": "Side"}),
    ]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))
    return rows


def test_segments_merge_only_within_equal_tags(ways):
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1))
    edges = graph.edges_gdf.sort_values("length_meter", ascending=False).reset_index(drop=True)
    assert len(edges) == 4 and len(graph.nodes_gdf) == 5
    # Ways 4, 5 and 6 have equal tags: one edge of four segments, way 6 turned to continue the line.
    side = edges.loc[0, "geometry"]
    assert edges.loc[0, "name"] == "Side" and len(side.coords) == 5
    xs = [x for x, _ in side.coords]
    assert xs in (sorted(xs), sorted(xs, reverse=True))
    _assert_runs_from_u_to_v(graph)
    assert edges.loc[1, "name"] == "Main" and len(edges.loc[1, "geometry"].coords) == 3
    bridge = edges[edges["bridge"].notna()]
    assert len(bridge) == 1 and bridge["name"].iloc[0] == "Main" and len(bridge["geometry"].iloc[0].coords) == 2
    assert edges["name"].value_counts().to_dict() == {"Main": 3, "Side": 1}
    assert edges["length_meter"].sum() == pytest.approx(graph.edges_gdf.geometry.length.sum(), abs=0.01)


def test_unsimplified_graph_keeps_every_segment(ways):
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1), simplify=False)
    assert len(graph.edges_gdf) == 8 and graph.edges_gdf["bridge"].notna().sum() == 1


def test_drive_graph_keeps_tags_of_each_way(ways):
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), keep_largest_subgraph=False)
    edges = graph.edges_gdf
    assert len(edges) == 4 and set(edges["name"]) == {"Main", "Side"}
    assert {"bridge", "tunnel", "highway", "oneway", "category"} <= set(edges.columns)
    assert edges["bridge"].notna().sum() == 1 and edges["tunnel"].isna().all()


@pytest.mark.parametrize("tags", [[], ["name"]])
def test_bridges_split_and_stay_without_requested_tags(ways, tags):
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1), osm_edge_tags=tags)
    edges = graph.edges_gdf
    assert {"bridge", "tunnel"} <= set(edges.columns)
    assert (
        edges["bridge"].notna().sum() == 1 and len(edges.loc[edges["bridge"].notna(), "geometry"].iloc[0].coords) == 2
    )
    if not tags:
        # Only the bridge splits the street; the change of name no longer does.
        assert len(edges) == 3 and "name" not in edges.columns
    else:
        assert len(edges) == 4


def test_drive_graph_splits_on_tags_it_reads_but_does_not_keep(ways, monkeypatch):
    rows = [
        _way(1, [30.000, 30.001], {"highway": "residential", "maxspeed": "30"}),
        _way(2, [30.001, 30.002], {"highway": "residential", "maxspeed": "60"}),
        _way(3, [30.002, 30.003], {"highway": "residential", "maxspeed": "60"}),
    ]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), osm_edge_tags=[], keep_largest_subgraph=False)
    assert len(graph.edges_gdf) == 2 and "maxspeed" not in graph.edges_gdf.columns


def test_one_way_ways_merge_only_head_to_tail(monkeypatch):
    # 1 -> 2 one-way in a row merge; 3 one-way drawn back into the end of 2 stays; 4 two-way after it stays.
    rows = [
        _way(1, [30.000, 30.001], {"highway": "residential", "oneway": "yes"}),
        _way(2, [30.001, 30.002], {"highway": "residential", "oneway": "yes"}),
        _way(3, [30.003, 30.002], {"highway": "residential", "oneway": "yes"}),
        _way(4, [30.003, 30.004], {"highway": "residential"}),
    ]
    _mock(monkeypatch, rows)
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), keep_largest_subgraph=False)
    edges = graph.edges_gdf.sort_values("length_meter", ascending=False)
    assert len(edges) == 3 and edges["oneway"].tolist() == [True, True, False]
    assert len(edges.geometry.iloc[0].coords) == 3
    _assert_runs_from_u_to_v(graph)
    # Every one-way edge still runs as drawn: west to east for 1-2, east to west for 3.
    xs = [(line.coords[0][0] < line.coords[-1][0]) for line in edges.geometry.iloc[:2]]
    assert xs == [True, False]


def test_one_way_against_the_drawing_merges_in_drawing_order_and_is_turned_once(monkeypatch):
    rows = [
        _way(1, [30.000, 30.001], {"highway": "residential", "oneway": "-1"}),
        _way(2, [30.001, 30.002], {"highway": "residential", "oneway": "-1"}),
        _way(3, [30.003, 30.002], {"highway": "residential", "oneway": "-1"}),
    ]
    _mock(monkeypatch, rows)
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), keep_largest_subgraph=False)
    edges = graph.edges_gdf.sort_values("length_meter", ascending=False)
    assert len(edges) == 2 and edges["oneway"].all()
    merged = edges.geometry.iloc[0]
    assert len(merged.coords) == 3 and merged.coords[0][0] > merged.coords[-1][0]  # driven east to west
    _assert_runs_from_u_to_v(graph)


def test_two_way_ways_drawn_tail_to_tail_merge_in_walk_graph(monkeypatch):
    rows = [
        _way(1, [30.001, 30.000], {"highway": "footway"}),
        _way(2, [30.001, 30.002, 30.003], {"highway": "footway"}),
    ]
    _mock(monkeypatch, rows)
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1))
    assert len(graph.edges_gdf) == 1 and len(graph.nodes_gdf) == 2
    assert len(graph.edges_gdf.geometry.iloc[0].coords) == 4
    _assert_runs_from_u_to_v(graph)
