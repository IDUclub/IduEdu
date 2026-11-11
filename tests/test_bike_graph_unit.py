import numpy as np
import pandas as pd
import pytest
from shapely import box

from iduedu import get_bike_graph
from iduedu.graph_builders import bike_builders, drive_walk_builders
from iduedu.graph_builders.bike_builders import cycle_infrastructure, level_of_traffic_stress

pytestmark = pytest.mark.unit

AREA = box(29.9, 58.9, 30.1, 59.1)


def _way(way_id, lons, tags, lat=59.0):
    return {"type": "way", "id": way_id, "nodes": [], "geometry": np.array([(lon, lat) for lon in lons]), "tags": tags}


def _serve(monkeypatch, rows):
    monkeypatch.setattr(drive_walk_builders, "get_network_by_filters", lambda polygon, way_filter: pd.DataFrame(rows))


def test_infrastructure_and_stress_levels():
    edges = pd.DataFrame(
        [
            {"highway": "cycleway"},
            {"highway": "path"},
            {"highway": "secondary", "cycleway:right": "track", "maxspeed": "60"},
            {"highway": "tertiary", "cycleway": "lane", "maxspeed": "30"},
            {"highway": "tertiary", "cycleway:both": "lane", "maxspeed": "50", "lanes": "2"},
            {"highway": "secondary", "cycleway": "lane", "maxspeed": "50", "lanes": "4"},
            {"highway": "primary", "cycleway": "opposite_lane", "maxspeed": "80"},
            {"highway": "living_street", "maxspeed": "60"},
            {"highway": "residential", "maxspeed": "20"},
            {"highway": "residential"},  # the default limit of the class
            {"highway": "residential", "maxspeed": "40", "lanes": "3"},
            {"highway": "service", "maxspeed": "70"},
            {"highway": "tertiary", "maxspeed": "30"},
            {"highway": "secondary", "maxspeed": "50", "lanes": "3"},
            {"highway": "primary", "maxspeed": "60"},
            {"highway": "tertiary", "cycleway": "separate", "maxspeed": "RU:urban"},
        ]
    )
    assert cycle_infrastructure(edges).tolist() == ["track"] * 3 + ["lane"] * 4 + ["mixed"] * 9
    levels = level_of_traffic_stress(edges).tolist()
    assert levels[:15] == [1, 1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 2, 3, 4]
    assert levels[15] in (3, 4)  # an unreadable limit falls back to the class default
    assert level_of_traffic_stress(pd.DataFrame({"highway": ["residential"]})).tolist() == [2]


def test_bike_graph_directions_and_footways(monkeypatch):
    rows = [
        _way(1, [30.000, 30.001], {"highway": "residential", "oneway": "yes"}),
        _way(2, [30.001, 30.002], {"highway": "residential", "oneway": "yes", "oneway:bicycle": "no"}),
        _way(3, [30.003, 30.002], {"highway": "tertiary", "oneway": "-1", "maxspeed": "50"}),
        _way(4, [30.003, 30.004], {"highway": "secondary", "oneway": "yes", "cycleway": "opposite_lane"}),
        _way(5, [30.004, 30.005], {"highway": "footway"}),
        _way(6, [30.005, 30.006], {"highway": "footway", "bicycle": "yes"}),
        _way(7, [30.004, 30.0045], {"highway": "residential", "bicycle": "no"}, lat=59.0),
    ]
    _serve(monkeypatch, rows)
    graph = get_bike_graph(territory=AREA, keep_largest_subgraph=False)
    edges = graph.edges_gdf.sort_values("length_meter").reset_index(drop=True)
    assert graph.is_directed and graph.edge_direction_column == "oneway" and graph.type == "bike"
    assert len(edges) == 5  # the plain footway and the way closed to bicycles are left out
    by_highway = edges.set_index("highway")
    assert by_highway.loc[["tertiary", "secondary", "footway"], "oneway"].tolist() == [True, False, False]
    # One residential street is one-way, the other is open to bicycles against the traffic.
    assert sorted(by_highway.loc["residential", "oneway"]) == [False, True]
    assert by_highway.loc["footway", "cycle_infra"] == "track" and by_highway.loc["secondary", "cycle_infra"] == "lane"
    # Way 3 is drawn against its direction of travel (oneway=-1): its edge runs west to east.
    tertiary = by_highway.loc["tertiary", "geometry"]
    assert tertiary.coords[0][0] < tertiary.coords[-1][0]
    assert edges["time_min"].tolist() == pytest.approx((edges["length_meter"] / 250).round(3).tolist())
    assert set(edges["type"]) == {"bike"} and {"lts", "bridge", "surface", "maxspeed"} <= set(edges.columns)

    with_footways = get_bike_graph(territory=AREA, keep_largest_subgraph=False, allow_footways=True)
    assert (with_footways.edges_gdf["highway"] == "footway").sum() == 2
    calm = get_bike_graph(territory=AREA, keep_largest_subgraph=False, max_lts=2)
    assert set(calm.edges_gdf["lts"]) <= {1, 2} and "secondary" not in set(calm.edges_gdf["highway"])
    assert "tertiary" not in set(calm.edges_gdf["highway"])
    fast = get_bike_graph(territory=AREA, keep_largest_subgraph=False, bike_speed=500)
    assert fast.edges_gdf["time_min"].sum() == pytest.approx(graph.edges_gdf["time_min"].sum() / 2, rel=1e-2)


def test_bike_graph_with_elevation_and_edge_cases(monkeypatch):
    _serve(monkeypatch, [_way(1, [30.000, 30.002], {"highway": "cycleway"})])
    flat = get_bike_graph(territory=AREA, keep_largest_subgraph=False)
    origin = flat.nodes_gdf.geometry.x.min()
    hilly = get_bike_graph(territory=AREA, keep_largest_subgraph=False, elevation=lambda x, y: 0.05 * (x - origin))
    edge = hilly.edges_gdf.iloc[0]
    assert abs(edge["grade"]) == pytest.approx(0.05, abs=1e-3) and {"ascent", "descent", "time_min_flat"} <= set(
        edge.index
    )
    both_ways = (1 / 0.7 + 1 / 1.2) / 2  # the cycling model, there and back on a two-way edge
    assert edge["time_min"] == pytest.approx(edge["time_min_flat"] * both_ways, rel=1e-2)

    with pytest.raises(ValueError, match="max_lts"):
        get_bike_graph(territory=AREA, max_lts=5)
    with pytest.raises(ValueError, match="custom_filter"):
        get_bike_graph(territory=AREA, network_type="custom")
    _serve(monkeypatch, [_way(1, [30.0, 30.001], {"highway": "primary", "maxspeed": "90"})])
    assert len(get_bike_graph(territory=AREA, max_lts=1).edges_gdf) == 0
    _serve(monkeypatch, [])
    assert len(get_bike_graph(territory=AREA).nodes_gdf) == 0
    assert bike_builders.FOOTWAYS <= bike_builders.CAR_FREE
