# pylint: disable=protected-access
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from iduedu import get_street_graph, mode_graph
from iduedu.graph_builders import drive_walk_builders

pytestmark = pytest.mark.unit

TERRITORY = Polygon([(30.29, 59.89), (30.33, 59.89), (30.33, 59.93), (30.29, 59.93)])
A, B, C, D = (30.300, 59.900), (30.302, 59.900), (30.304, 59.900), (30.302, 59.901)
E, F = (30.300, 59.910), (30.302, 59.910)
G, H, J = (30.300, 59.920), (30.302, 59.920), (30.304, 59.920)


def _way(way_id, points, **tags):
    return {"type": "way", "id": way_id, "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in points]}


WAYS = [
    # A residential street A-B-C drawn as two ways; a footway joins at B.
    _way(1, [A, B], highway="residential", name="Main"),
    _way(2, [B, C], highway="residential", name="Main"),
    _way(3, [B, D], highway="footway"),
    # One-way against the drawing, open to bicycles both ways.
    _way(4, [E, F], highway="residential", oneway="-1", **{"oneway:bicycle": "no"}),
    # Two walkways drawn head to head; the second is closed to bicycles.
    _way(5, [G, H], highway="service", service="alley"),
    _way(6, [J, H], highway="service", service="alley", bicycle="no"),
]


@pytest.fixture
def street(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(WAYS))
    return get_street_graph(territory=TERRITORY)


def _point_edges(graph):
    points = graph.nodes_gdf.geometry
    return {
        (round(points[u].x), round(points[v].x), round(points[u].y)): row
        for u, v, (_, row) in zip(graph.edges_gdf["u"], graph.edges_gdf["v"], graph.edges_gdf.iterrows())
    }


def test_street_graph_stores_each_street_once_with_mode_flags(street):
    edges = street.edges_gdf
    assert street.type == "street" and not street.is_directed
    assert len(edges) == 6  # A-B, B-C, B-D, E-F, G-H, J-H: B is a junction, H joins two keys
    footway = edges[edges["highway"].eq("footway")].iloc[0]
    assert footway["walk"] and not footway["drive_forward"] and not footway["bike_forward"]
    assert np.isnan(footway["time_drive_min"]) and footway["time_walk_min"] > 0
    oneway = edges[edges["highway"].eq("residential") & edges["drive_backward"] & ~edges["drive_forward"]]
    assert len(oneway) == 1 and oneway["bike_forward"].item() and oneway["bike_backward"].item()
    for u, v, geometry in zip(edges["u"], edges["v"], edges.geometry):
        assert geometry.coords[0] == street.nodes_gdf.geometry[u].coords[0]
        assert geometry.coords[-1] == street.nodes_gdf.geometry[v].coords[0]
    assert edges["lts"].dtype == "Int64" and edges["category"].notna().sum() == 3


def test_drive_graph_is_not_cut_where_a_footway_joins(street):
    drive = mode_graph(street, "drive")
    assert drive.is_directed and drive.type == "drive"
    main = drive.edges_gdf[drive.edges_gdf["name"].eq("Main")]
    assert len(main) == 1 and not main["oneway"].item()
    assert main["length_meter"].item() == pytest.approx(
        street.edges_gdf.loc[street.edges_gdf["name"].eq("Main"), "length_meter"].sum()
    )
    assert len(mode_graph(street, "walk").edges_gdf[lambda e: e["name"].eq("Main")]) == 2  # B is a walking junction


def test_an_edge_driven_only_against_its_drawing_is_turned(street):
    drive = mode_graph(street, "drive")
    turned = drive.edges_gdf[drive.edges_gdf["oneway"]].iloc[0]
    points = drive.nodes_gdf.geometry
    assert points[turned["u"]].coords[0] == pytest.approx(turned.geometry.coords[0])
    assert turned.geometry.coords[0][0] > turned.geometry.coords[-1][0]  # from F back to E
    bike = mode_graph(street, "bike")
    assert not bike.edges_gdf.loc[bike.edges_gdf["highway"].eq("residential"), "oneway"].any()


def test_walkways_drawn_head_to_head_merge_but_a_bicycle_ban_splits_them(street):
    walk = mode_graph(street, "walk")
    assert not walk.is_directed and walk.edges_gdf["time_min"].notna().all()
    alleys = walk.edges_gdf[walk.edges_gdf["highway"].eq("service")]
    assert len(alleys) == 1 and alleys.geometry.item().length == pytest.approx(
        street.edges_gdf.loc[street.edges_gdf["highway"].eq("service")].geometry.length.sum()
    )
    assert len(street.edges_gdf[street.edges_gdf["highway"].eq("service")]) == 2  # bike flags differ


def test_mode_graph_without_simplify_keeps_street_edges_and_drops_foreign_columns(street):
    walk = mode_graph(street, "walk", simplify=False)
    assert len(walk.edges_gdf) == 6 and len(walk.nodes_gdf) == len(street.nodes_gdf)
    assert {"drive_forward", "time_drive_min", "lts", "category", "walk"}.isdisjoint(walk.edges_gdf.columns)
    drive = mode_graph(street, "drive", simplify=False)
    assert len(drive.edges_gdf) == 3 and {"category", "maxspeed"} <= set(drive.edges_gdf.columns)


def test_modes_subset_and_errors(monkeypatch):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(WAYS))
    walk_only = get_street_graph(territory=TERRITORY, modes=["walk"])
    assert "drive_forward" not in walk_only.edges_gdf.columns and walk_only.edges_gdf["walk"].all()
    with pytest.raises(ValueError, match="no 'drive' columns"):
        mode_graph(walk_only, "drive")
    with pytest.raises(ValueError, match="modes"):
        get_street_graph(territory=TERRITORY, modes=["boat"])
    with pytest.raises(ValueError, match="mode must be"):
        mode_graph(walk_only, "boat")


def test_ways_one_way_for_a_mode_are_not_turned_in_the_street_graph(monkeypatch):
    # Two one-way roads drawn head to head: walking may use them both ways, driving may not, so the street graph
    # keeps them apart (turning one would swap its drive flags); the walk graph merges them.
    rows = [
        _way(11, [A, B], highway="residential", oneway="yes"),
        _way(12, [C, B], highway="residential", oneway="yes"),
    ]
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))
    street = get_street_graph(territory=TERRITORY)
    assert len(street.edges_gdf) == 2
    assert street.edges_gdf["drive_forward"].all() and not street.edges_gdf["drive_backward"].any()
    assert len(mode_graph(street, "walk").edges_gdf) == 1
    assert len(mode_graph(street, "drive").edges_gdf) == 2
