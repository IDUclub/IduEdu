import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from iduedu import (
    UrbanGraph,
    clip_urban_graph,
    get_gtfs_public_transport_graph,
    get_public_transport_graph,
    join_urban_graphs,
    nx_graph2urban_graph,
    read_urban_graph,
    route_edges,
    route_stops,
    routes_in_area,
    urban_graph2nx_graph,
)
from iduedu.graph_builders import public_transport_builders
from tests.factories import write_gtfs_feed

pytestmark = pytest.mark.unit

TERRITORY = Polygon([(30.28, 59.88), (30.33, 59.88), (30.33, 59.92), (30.28, 59.92)])
LAT, TRAM_LAT = 59.900, 59.905


def _node(ref, lon, lat, role):
    return {"type": "node", "ref": ref, "role": role, "lon": lon, "lat": lat}


def _way(ref, lons, lat):
    return {"type": "way", "ref": ref, "role": "", "geometry": [{"lon": x, "lat": lat} for x in lons]}


def _route(rid, route, ref, lons, lat, stop_ids, **tags):
    """A route along one way at ``lat`` with stops at its 2nd and 2nd-to-last points, platforms 10 m north."""
    members = [_way(rid * 10, lons, lat)]
    for stop_id, lon in zip(stop_ids, (lons[1], lons[-2])):
        members.append(_node(stop_id, lon, lat, "stop"))
        members.append(_node(stop_id + 1, lon, lat + 0.0001, "platform"))
    return {
        "type": "relation",
        "id": rid,
        "members": members,
        "tags": {"type": "route", "route": route, "ref": ref, **tags},
    }


ELEMENTS = [
    {"type": "way", "id": 10, "tags": {"highway": "primary"}},
    {"type": "way", "id": 20, "tags": {"highway": "primary"}},
    {"type": "way", "id": 30, "tags": {"railway": "tram"}},
    # Bus 3 both ways along the same road, calling at the same stop positions: two route variants.
    _route(1, "bus", "3", [30.300, 30.302, 30.308, 30.310], LAT, [100, 200], **{"from": "West", "to": "East"}),
    _route(2, "bus", "3", [30.310, 30.308, 30.302, 30.300], LAT, [200, 100], **{"from": "East", "to": "West"}),
    # Tram 3 on another street; OSM calls it light_rail here.
    _route(3, "light_rail", "3", [30.300, 30.302, 30.308, 30.310], TRAM_LAT, [300, 400], colour="red"),
    {
        "type": "relation",
        "id": 9,
        "members": [{"type": "relation", "ref": 1, "role": ""}, {"type": "relation", "ref": 2, "role": ""}],
        "tags": {"type": "route_master", "route_master": "bus", "ref": "3", "name": "Bus 3"},
    },
]


@pytest.fixture
def pt_graph(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: list(ELEMENTS))
    return get_public_transport_graph(territory=TERRITORY, transport_types=["bus", "tram"])


def test_osm_routes_table_has_one_row_per_relation_and_the_line(pt_graph):
    routes = pt_graph.routes
    assert routes.index.name == "route_key" and sorted(routes.index) == ["osm:r1", "osm:r2", "osm:r3"]
    assert routes.loc["osm:r1", ["source", "mode", "ref", "from", "to"]].tolist() == ["osm", "bus", "3", "West", "East"]
    assert routes.loc["osm:r1", "line_key"] == routes.loc["osm:r2", "line_key"] == "osm:r9"
    assert routes.loc["osm:r1", "line_name"] == "3" and pd.isna(routes.loc["osm:r3", "line_key"])
    assert routes.loc["osm:r3", ["mode", "osm_route", "colour"]].tolist() == ["tram", "light_rail", "red"]
    assert routes.loc["osm:r1", "osm_id"] == 1 and "interval" in routes.columns


def test_each_route_variant_has_its_own_stops_and_platforms_list_all_routes(pt_graph):
    edges, nodes = pt_graph.edges_gdf, pt_graph.nodes_gdf
    travel = edges[edges["type"].isin({"bus", "tram"})]
    assert set(travel["route_key"]) == {"osm:r1", "osm:r2", "osm:r3"}
    assert set(travel["route"]) == {"3"}  # the label stays
    stops = nodes[nodes["type"].isin({"bus", "tram"})]
    assert len(stops) == 6 and set(stops["route_key"]) == {"osm:r1", "osm:r2", "osm:r3"}
    # Bus 3 there and back stop at the same points but not at the same nodes.
    there = set(edges.loc[edges["route_key"].eq("osm:r1"), ["u", "v"]].to_numpy().ravel())
    back = set(edges.loc[edges["route_key"].eq("osm:r2"), ["u", "v"]].to_numpy().ravel())
    assert not (there & back) - set(nodes.index[nodes["type"].eq("platform")])
    platforms = nodes[nodes["type"].eq("platform")]
    assert sorted(map(sorted, platforms["route_key"])) == [["osm:r1", "osm:r2"]] * 2 + [["osm:r3"]] * 2
    links = edges[edges["type"].isin({"boarding", "alighting"})]
    assert links["route_key"].notna().all()


def test_route_edges_and_stops_follow_the_route(pt_graph):
    back = route_edges(pt_graph, "osm:r2")
    assert len(back) == 1 and back["position"].tolist() == [0] and back["part"].tolist() == [0]
    line = back.geometry.iloc[0]
    assert line.coords[0][0] > line.coords[-1][0]  # east to west
    stops = route_stops(pt_graph, "osm:r2")
    assert stops["position"].tolist() == [0, 1] and stops.geometry.x.is_monotonic_decreasing
    assert pt_graph.route_edges("osm:r2").equals(back)
    with pytest.raises(KeyError, match="osm:r404"):
        route_edges(pt_graph, "osm:r404")


def test_routes_in_area_and_clipping_drop_the_routes_outside(pt_graph):
    tram_street = gpd.GeoDataFrame(geometry=[box(30.29, 59.904, 30.32, 59.906)], crs=4326)
    assert routes_in_area(pt_graph, tram_street).index.tolist() == ["osm:r3"]
    clipped = clip_urban_graph(pt_graph, tram_street)
    assert routes_in_area(pt_graph, box(30.29, 59.904, 30.32, 59.906)).index.tolist() == ["osm:r3"]
    assert clipped.routes.index.tolist() == ["osm:r3"]
    cut = pt_graph.subgraph_by_nodes(pt_graph.nodes_gdf.index[pt_graph.nodes_gdf.geometry.y < 0])
    assert cut.routes.empty


def test_builder_clip_keeps_only_routes_with_edges_inside(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: list(ELEMENTS))
    south = Polygon([(30.28, 59.88), (30.33, 59.88), (30.33, 59.9025), (30.28, 59.9025)])
    clipped = get_public_transport_graph(territory=south, transport_types=["bus", "tram"], clip_by_territory=True)
    whole = get_public_transport_graph(territory=south, transport_types=["bus", "tram"])
    assert sorted(clipped.routes.index) == ["osm:r1", "osm:r2"]
    assert sorted(whole.routes.index) == ["osm:r1", "osm:r2", "osm:r3"]


def test_routes_survive_copy_join_pickle_and_urbangraph_files(pt_graph, tmp_path):
    assert pt_graph.copy().routes.equals(pt_graph.routes)
    assert pickle.loads(pickle.dumps(pt_graph)).routes.equals(pt_graph.routes)
    with_missing = pt_graph.copy()
    nodes = with_missing.nodes_gdf
    nodes["note"] = pd.Series([None] + [["x"]] * (len(nodes) - 1), index=nodes.index, dtype=object)
    nodes["label"] = pd.Series([None] + ["x"] * (len(nodes) - 1), index=nodes.index, dtype=object)
    path = with_missing.write(tmp_path / "pt.urbangraph")
    restored = read_urban_graph(path)
    # pandas 3 reads text back as strings with NaN; a missing value written as None comes back as None.
    for column, value in (("note", ["x"]), ("label", "x")):
        assert restored.nodes_gdf[column].iloc[0] is None and restored.nodes_gdf[column].iloc[1] == value
    pd.testing.assert_frame_equal(restored.routes, pt_graph.routes, check_dtype=False)
    assert sorted(restored.nodes_gdf["route_key"].dropna().map(str)) == sorted(
        pt_graph.nodes_gdf["route_key"].dropna().map(str)
    )

    street = UrbanGraph.empty(crs=pt_graph.crs, is_directed=True, edge_direction_column="oneway")
    assert street.routes is None and "routes" not in repr(street)
    joined = join_urban_graphs(street, pt_graph)
    assert joined.routes.equals(pt_graph.routes) and "routes=3" in repr(joined)
    # Swapping the ends of edges drawn backwards wrote into a read-only array under copy-on-write pandas.
    # An undirected nx.MultiGraph keyed boarding and alighting of one pair alike, and one of them was lost.
    restored = nx_graph2urban_graph(urban_graph2nx_graph(pt_graph)).edges_gdf
    edge_set = sorted(zip(pt_graph.edges_gdf["u"], pt_graph.edges_gdf["v"], pt_graph.edges_gdf["type"]))
    assert sorted(zip(restored["u"], restored["v"], restored["type"])) == edge_set
    plain = street.write(tmp_path / "street.urbangraph")
    assert read_urban_graph(plain).routes is None


def test_routes_table_is_checked():
    edges = gpd.GeoDataFrame(
        {"u": [0], "v": [1], "k": [0], "length_meter": [1.0], "time_min": [1.0], "route_key": ["a"]},
        geometry=gpd.points_from_xy([0], [0]).buffer(1).boundary,
        crs=3857,
    )
    nodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 1], [0, 0]), crs=3857)
    with pytest.raises(ValueError, match="repeats"):
        UrbanGraph(nodes, edges, True, False, routes=pd.DataFrame(index=pd.Index(["a", "a"], name="route_key")))
    graph = UrbanGraph(nodes, edges, True, False, routes=pd.DataFrame({"route_key": ["a", "b"], "ref": ["1", "2"]}))
    assert graph.routes.index.tolist() == ["a"] and list(graph.routes.columns[:2]) == ["source", "mode"]


GTFS_TABLES = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\na,Transit Co,https://example.com,Europe/Moscow\n",
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "svc,1,1,1,1,1,1,1,20250101,20251231\n"
    ),
    "stops.txt": (
        "stop_id,stop_name,stop_lat,stop_lon,location_type\n"
        "A,Alpha,59.9300,30.3000,0\nB,Beta,59.9400,30.3000,0\nC,Gamma,59.9500,30.3000,0\n"
    ),
    "routes.txt": "route_id,agency_id,route_short_name,route_long_name,route_type,route_color\nR1,a,1,Main,3,FF0000\n",
    "trips.txt": (
        "route_id,service_id,trip_id,direction_id\n"
        "R1,svc,t1,0\nR1,svc,t2,0\nR1,svc,t3,0\nR1,svc,back1,1\nR1,svc,back2,1\n"
    ),
    "stop_times.txt": (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "t1,08:00:00,08:00:00,A,1\nt1,08:05:00,08:05:00,B,2\nt1,08:10:00,08:10:00,C,3\n"
        "t2,08:20:00,08:20:00,A,1\nt2,08:25:00,08:25:00,B,2\nt2,08:30:00,08:30:00,C,3\n"
        # a short turn of the same route and direction: another pattern
        "t3,08:40:00,08:40:00,A,1\nt3,08:45:00,08:45:00,B,2\n"
        "back1,09:00:00,09:00:00,C,1\nback1,09:10:00,09:10:00,A,2\n"
        "back2,09:30:00,09:30:00,C,1\nback2,09:40:00,09:40:00,A,2\n"
    ),
}


def test_gtfs_patterns_get_stable_keys_and_a_routes_table(tmp_path):
    feed = write_gtfs_feed(tmp_path / "city", GTFS_TABLES)
    graph = get_gtfs_public_transport_graph(feed, crs="EPSG:32636")
    routes = graph.routes
    assert len(routes) == 3 and all(key.startswith("gtfs:city:R1:") for key in routes.index)
    assert sorted(routes["direction_id"]) == ["0", "0", "1"]
    full = routes[routes["stops"].eq(3)].iloc[0]
    assert full[["source", "mode", "ref", "name", "from", "to", "operator", "colour"]].tolist() == [
        "gtfs", "bus", "1", "Main", "Alpha", "Gamma", "Transit Co", "#FF0000"
    ]  # fmt: skip
    assert full["line_key"] == "gtfs:city:R1" and full["gtfs_route_id"] == "R1"
    back = routes[routes["direction_id"].eq("1")].iloc[0]
    assert back["interval_min"] == pytest.approx(30.0) and (back["from"], back["to"]) == ("Gamma", "Alpha")

    again = get_gtfs_public_transport_graph(feed, crs="EPSG:32636")
    assert again.routes.index.tolist() == routes.index.tolist()
    named = get_gtfs_public_transport_graph(feed, crs="EPSG:32636", feed_id="spb")
    assert all(key.startswith("gtfs:spb:R1:") for key in named.routes.index)
    with pytest.raises(ValueError, match="feed_id"):
        get_gtfs_public_transport_graph(feed, feed_id="a:b")

    key = full.name
    stops = route_stops(graph, key)
    assert stops["gtfs_stop_id"].tolist() == ["A", "B", "C"]
    platforms = graph.nodes_gdf[graph.nodes_gdf["type"].eq("platform")].set_index("gtfs_stop_id")["route_key"]
    assert len(platforms["A"]) == 3 and len(platforms["C"]) == 2 and isinstance(platforms["A"], list)
    edges = graph.edges_gdf
    assert edges.loc[edges["type"].isin({"bus", "boarding", "alighting"}), "route_key"].notna().all()
    assert np.isin(edges["route_key"].dropna().unique(), routes.index).all()


def test_merged_feeds_keep_their_route_keys_apart(tmp_path):
    one = write_gtfs_feed(tmp_path / "north", GTFS_TABLES)
    two = write_gtfs_feed(tmp_path / "south", GTFS_TABLES)
    graph = get_gtfs_public_transport_graph([one, two], crs="EPSG:32636")
    keys = graph.routes.index
    assert len(keys) == 6 and keys.is_unique
    assert all(key.startswith("gtfs:north+south:") for key in keys)
    assert graph.routes["gtfs_route_id"].nunique() == 2
