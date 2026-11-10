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


@pytest.fixture
def ways(monkeypatch):
    """A street drawn as five ways in a row: plain, bridge, plain (same name), then two with another name."""
    rows = [
        _way(1, [30.000, 30.001, 30.002], {"highway": "residential", "name": "Main"}),
        _way(2, [30.002, 30.003], {"highway": "residential", "name": "Main", "bridge": "yes"}),
        _way(3, [30.003, 30.004], {"highway": "residential", "name": "Main"}),
        _way(4, [30.004, 30.005], {"highway": "residential", "name": "Side"}),
        _way(5, [30.005, 30.006, 30.007], {"highway": "residential", "name": "Side"}),
        # Drawn against the others: it meets way 5 head to head and is not merged with it.
        _way(6, [30.008, 30.007], {"highway": "residential", "name": "Side"}),
    ]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))
    return rows


def test_segments_merge_only_within_equal_tags(ways):
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1))
    edges = graph.edges_gdf.sort_values("length_meter", ascending=False).reset_index(drop=True)
    assert len(edges) == 5 and len(graph.nodes_gdf) == 6
    # Ways 4 and 5 have equal tags and follow one another: one edge of three segments.
    assert edges.loc[0, "name"] == "Side" and len(edges.loc[0, "geometry"].coords) == 4
    assert edges.loc[1, "name"] == "Main" and len(edges.loc[1, "geometry"].coords) == 3
    bridge = edges[edges["bridge"].notna()]
    assert len(bridge) == 1 and bridge["name"].iloc[0] == "Main" and len(bridge["geometry"].iloc[0].coords) == 2
    assert edges["name"].value_counts().to_dict() == {"Main": 3, "Side": 2}
    assert edges["length_meter"].sum() == pytest.approx(graph.edges_gdf.geometry.length.sum(), abs=0.01)


def test_unsimplified_graph_keeps_every_segment(ways):
    graph = get_walk_graph(territory=box(29.9, 58.9, 30.1, 59.1), simplify=False)
    assert len(graph.edges_gdf) == 8 and graph.edges_gdf["bridge"].notna().sum() == 1


def test_drive_graph_keeps_tags_of_each_way(ways):
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), keep_largest_subgraph=False)
    edges = graph.edges_gdf
    assert len(edges) == 5 and set(edges["name"]) == {"Main", "Side"}
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
        assert len(edges) == 4 and "name" not in edges.columns
    else:
        assert len(edges) == 5


def test_drive_graph_splits_on_tags_it_reads_but_does_not_keep(ways, monkeypatch):
    rows = [
        _way(1, [30.000, 30.001], {"highway": "residential", "maxspeed": "30"}),
        _way(2, [30.001, 30.002], {"highway": "residential", "maxspeed": "60"}),
        _way(3, [30.002, 30.003], {"highway": "residential", "maxspeed": "60"}),
    ]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))
    graph = get_drive_graph(territory=box(29.9, 58.9, 30.1, 59.1), osm_edge_tags=[], keep_largest_subgraph=False)
    assert len(graph.edges_gdf) == 2 and "maxspeed" not in graph.edges_gdf.columns
