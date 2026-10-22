import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LineString, Point

from iduedu import get_gtfs_public_transport_graph
from iduedu.graph.urban_graph import UrbanGraph
from iduedu.graph_builders.gtfs_builders import _transport_type
from iduedu.graph_builders.intermodal_builders import join_pt_walk_graph
from iduedu.gtfs.reader import read_gtfs_feed
from tests.factories import write_gtfs_feed

pytestmark = pytest.mark.unit
CRS = "EPSG:32636"


BASE_TABLES = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\na,Agency,https://example.com,Europe/Moscow\n",
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "svc,1,1,1,1,1,1,1,20250101,20251231\n"
    ),
}


def _feed(tmp_path: Path, tables: dict[str, str]) -> Path:
    return write_gtfs_feed(tmp_path / "feed", {**BASE_TABLES, **tables})


def test_gtfs_builder_keeps_platforms_and_patterns_separate_and_uses_half_headway(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type\n"
                "A,Shared,59.9300,30.3000,0\n"
                "A2,Same coordinate,59.9300,30.3000,0\n"
                "B,North,59.9400,30.3000,0\n"
                "C,East,59.9300,30.3200,0\n"
            ),
            "routes.txt": (
                "route_id,agency_id,route_short_name,route_long_name,route_type\n"
                "R1,a,1,North,3\nR2,a,2,East,3\nR3,a,3,Alias,3\n"
            ),
            "trips.txt": (
                "route_id,service_id,trip_id,direction_id\n"
                "R1,svc,r1a,0\nR1,svc,r1b,0\n"
                "R2,svc,r2a,0\nR2,svc,r2b,0\n"
                "R3,svc,r3a,0\nR3,svc,r3b,0\n"
            ),
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "r1a,08:00:00,08:00:00,A,1\nr1a,08:10:00,08:10:00,B,2\n"
                "r1b,08:20:00,08:20:00,A,1\nr1b,08:30:00,08:30:00,B,2\n"
                "r2a,08:00:00,08:00:00,A,1\nr2a,08:12:00,08:12:00,C,2\n"
                "r2b,08:10:00,08:10:00,A,1\nr2b,08:22:00,08:22:00,C,2\n"
                "r3a,08:00:00,08:00:00,A2,1\nr3a,08:10:00,08:10:00,B,2\n"
                "r3b,08:30:00,08:30:00,A2,1\nr3b,08:40:00,08:40:00,B,2\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)

    platforms = graph.nodes_gdf[graph.nodes_gdf["type"] == "platform"]
    assert set(platforms["gtfs_stop_id"]) == {"A", "A2", "B", "C"}
    a = platforms.loc[platforms["gtfs_stop_id"] == "A"].iloc[0]
    a2 = platforms.loc[platforms["gtfs_stop_id"] == "A2"].iloc[0]
    assert a.name != a2.name
    assert a.geometry.equals(a2.geometry)

    origin_boarding = graph.edges_gdf[(graph.edges_gdf["type"] == "boarding") & (graph.edges_gdf["stop_position"] == 0)]
    access_edges = graph.edges_gdf[graph.edges_gdf["type"].isin({"boarding", "alighting"})]
    assert (access_edges["length_meter"] == 0).all()
    waits = origin_boarding.set_index("route_id")["time_min"].to_dict()
    assert waits == pytest.approx({"R1": 10.0, "R2": 5.0, "R3": 15.0})

    transport = graph.edges_gdf[graph.edges_gdf["type"] == "bus"]
    assert len(transport) == 3
    assert transport.set_index("route_id").loc["R1", "time_min"] == pytest.approx(10.0)
    assert transport.set_index("route_id").loc["R2", "time_min"] == pytest.approx(12.0)
    assert transport.iloc[0]["length_meter"] == pytest.approx(transport.iloc[0].geometry.length * 2**0.5)
    graph.update_adjacency_matrix()
    assert graph.adjacency_matrix.shape == (len(graph.nodes_gdf), len(graph.nodes_gdf))


def test_gtfs_builder_none_filters_mean_all_services_and_unbounded_time(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.93,30.30\nB,B,59.94,30.30\n",
            "routes.txt": "route_id,route_short_name,route_type\nR,1,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,svc,t1\nR,svc,t2\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,25:00:00,25:00:00,A,1\nt1,25:10:00,25:10:00,B,2\n"
                "t2,25:20:00,25:20:00,A,1\nt2,25:30:00,25:30:00,B,2\n"
            ),
        },
    )

    all_services = get_gtfs_public_transport_graph(feed, crs=CRS)
    assert not all_services.edges_gdf.empty
    assert set(all_services.edges_gdf.loc[all_services.edges_gdf["type"] == "boarding", "time_min"]) == {10.0}

    inactive_day = get_gtfs_public_transport_graph(feed, service_date="2026-01-01", crs=CRS)
    assert inactive_day.nodes_gdf.empty
    overnight = get_gtfs_public_transport_graph(feed, start_time="24:30", end_time="26:00", crs=CRS)
    assert not overnight.edges_gdf.empty


def test_gtfs_builder_ignores_trip_fields_repeated_in_stop_times(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.93,30.30\nB,B,59.94,30.30\n",
            "routes.txt": "route_id,route_short_name,route_type\nR,1,3\n",
            "trips.txt": "route_id,service_id,trip_id,direction_id\nR,svc,t1,0\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence,route_id,direction_id,shape_id\n"
                "t1,08:00:00,08:00:00,A,1,stale,1,S\nt1,08:10:00,08:10:00,B,2,stale,1,S\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)

    transport = graph.edges_gdf[graph.edges_gdf["type"] == "bus"]
    assert set(transport["route_id"]) == {"R"}


def test_gtfs_builder_uses_frequency_half_headway(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.93,30.30\nB,B,59.94,30.30\n",
            "routes.txt": "route_id,route_short_name,route_type\nR,1,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,svc,t1\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,00:00:00,00:00:00,A,1\nt1,00:10:00,00:10:00,B,2\n"
            ),
            "frequencies.txt": (
                "trip_id,start_time,end_time,headway_secs,exact_times\n" "t1,06:00:00,10:00:00,600,0\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    boarding = graph.edges_gdf[graph.edges_gdf["type"] == "boarding"]
    assert len(boarding) == 2
    assert set(boarding["time_min"]) == {5.0}


def test_gtfs_builder_builds_shape_and_internal_pathway(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,stop_access\n"
                "S,Station,59.9300,30.3000,1,,\n"
                "E,Entrance,59.9301,30.3000,2,S,\n"
                "G,Landing,,,3,S,\n"
                "P,Platform,59.9300,30.3010,0,S,0\n"
                "B,Destination,59.9400,30.3100,0,,\n"
            ),
            "routes.txt": "route_id,route_short_name,route_type\nR,1,0\n",
            "trips.txt": ("route_id,service_id,trip_id,direction_id,shape_id\n" "R,svc,t1,0,shape\nR,svc,t2,0,shape\n"),
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n"
                "t1,08:00:00,08:00:00,P,1,0\nt1,08:10:00,08:10:00,B,2,2\n"
                "t2,08:20:00,08:20:00,P,1,0\nt2,08:30:00,08:30:00,B,2,2\n"
            ),
            "shapes.txt": (
                "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n"
                "shape,59.9300,30.3005,1,0\n"
                "shape,59.9360,30.3060,2,1\n"
                "shape,59.9400,30.3100,3,2\n"
            ),
            "pathways.txt": (
                "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,traversal_time\n"
                "entrance,E,G,2,1,60\n"
                "stairs,G,P,2,1,120\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    physical_nodes = graph.nodes_gdf[
        graph.nodes_gdf["type"].isin({"station_platform", "station_entry_exit", "station_node"})
    ]
    node_types = physical_nodes.set_index("gtfs_stop_id")["type"]
    assert node_types.loc["P"] == "station_platform"
    assert node_types.loc["E"] == "station_entry_exit"
    assert node_types.loc["G"] == "station_node"
    assert not physical_nodes.loc[physical_nodes["gtfs_stop_id"] == "G", "geometry"].isna().any()

    pathways = graph.edges_gdf[graph.edges_gdf["type"] == "pathway"]
    assert len(pathways) == 2
    pathway = pathways.loc[pathways["pathway_id"] == "stairs"].iloc[0]
    assert pathway["time_min"] == pytest.approx(2.0)
    assert bool(pathway["oneway"]) is False
    transport = graph.edges_gdf[graph.edges_gdf["type"] == "tram"].iloc[0]
    assert len(transport.geometry.coords) == 3
    assert transport["length_meter"] == pytest.approx(transport.geometry.length)
    assert transport["time_min"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("route_type", "expected"),
    [
        ("0", "tram"),
        ("1", "subway"),
        ("2", "train"),
        ("3", "bus"),
        ("4", "ferry"),
        ("5", "cable_tram"),
        ("6", "aerial_lift"),
        ("7", "funicular"),
        ("11", "trolleybus"),
        ("12", "monorail"),
        ("109", "train"),  # suburban railway, used by VBB
        ("117", "train"),
        ("200", "coach"),  # intercity, must not become a city bus
        ("209", "coach"),
        ("400", "subway"),
        ("405", "monorail"),  # not a subway, despite sitting inside the urban railway block
        ("406", "subway"),
        ("700", "bus"),  # Berlin
        ("701", "bus"),  # Helsinki
        ("716", "bus"),
        ("800", "trolleybus"),
        ("900", "tram"),
        ("1000", "ferry"),
        ("1200", "ferry"),
        ("1300", "aerial_lift"),
        ("1400", "funicular"),
        ("1500", "taxi"),
    ],
)
def test_transport_type_maps_base_and_extended_route_types(route_type, expected):
    assert _transport_type(route_type) == expected


@pytest.mark.parametrize("route_type", ["300", "500", "600", "1100", "1600", "1700", "9999", "", "abc"])
def test_transport_type_does_not_guess_undocumented_codes(route_type):
    """Codes absent from the published table get an honest label instead of a neighbour's."""
    assert _transport_type(route_type) == "public_transport"


def test_transport_type_prefers_explicit_custom_type():
    assert _transport_type("700", "metro") == "subway"
    assert _transport_type("700", "rail") == "train"


def _reachable_from(graph: UrbanGraph, start: int) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for _, edge in graph.edges_gdf.iterrows():
        u, v = int(edge["u"]), int(edge["v"])
        adjacency.setdefault(u, set()).add(v)
        if not bool(edge["oneway"]):
            adjacency.setdefault(v, set()).add(u)
    seen, stack = {start}, [start]
    while stack:
        for neighbour in adjacency.get(stack.pop(), ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def test_gtfs_builder_links_boarding_areas_to_their_platforms(tmp_path):
    """Pathways that stop at boarding areas used to leave the platforms -- and the whole
    metro component -- unreachable from the station entrance, which the intermodal join
    then dropped without a word."""
    feed = _feed(
        tmp_path,
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,stop_access\n"
                "S,Station,59.9300,30.3000,1,,\n"
                "E,Entrance,59.9301,30.3000,2,S,\n"
                "G,Landing,59.9300,30.3004,3,S,\n"
                "BA,Boarding area,59.9300,30.3008,4,P,\n"
                "P,Platform,59.9300,30.3010,0,S,0\n"
                "B,Destination,59.9400,30.3100,0,,\n"
            ),
            "routes.txt": "route_id,route_short_name,route_type\nR,1,1\n",
            "trips.txt": "route_id,service_id,trip_id,direction_id\nR,svc,t1,0\nR,svc,t2,0\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,P,1\nt1,08:10:00,08:10:00,B,2\n"
                "t2,08:20:00,08:20:00,P,1\nt2,08:30:00,08:30:00,B,2\n"
            ),
            # the chain never reaches P, exactly as in the PID and DTPM feeds
            "pathways.txt": (
                "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,traversal_time\n"
                "entrance,E,G,2,1,60\n"
                "stairs,G,BA,2,1,120\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    links = graph.edges_gdf[graph.edges_gdf["type"] == "station_link"]
    assert len(links) == 1
    link = links.iloc[0]
    assert link["time_min"] == pytest.approx(0.0)
    assert bool(link["oneway"]) is False

    nodes = graph.nodes_gdf
    stop_nodes = nodes[nodes["gtfs_stop_id"].notna()]
    entrance = int(stop_nodes.index[stop_nodes["gtfs_stop_id"] == "E"][0])
    platform = int(
        stop_nodes.index[(stop_nodes["gtfs_stop_id"] == "P") & (stop_nodes["type"] == "station_platform")][0]
    )
    assert platform in _reachable_from(graph, entrance)


def test_gtfs_builder_links_platforms_to_their_station(tmp_path):
    """A feed whose pathways stop at the station node needs the same bridge one level up."""
    feed = _feed(
        tmp_path,
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,stop_access\n"
                "S,Station,59.9300,30.3000,1,,\n"
                "E,Entrance,59.9301,30.3000,2,S,\n"
                "P,Platform,59.9300,30.3010,0,S,0\n"
                "B,Destination,59.9400,30.3100,0,,\n"
            ),
            "routes.txt": "route_id,route_short_name,route_type\nR,1,1\n",
            "trips.txt": "route_id,service_id,trip_id,direction_id\nR,svc,t1,0\nR,svc,t2,0\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,P,1\nt1,08:10:00,08:10:00,B,2\n"
                "t2,08:20:00,08:20:00,P,1\nt2,08:30:00,08:30:00,B,2\n"
            ),
            "pathways.txt": (
                "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,traversal_time\n"
                "entrance,E,S,2,1,60\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    links = graph.edges_gdf[graph.edges_gdf["type"] == "station_link"]
    assert set(links["gtfs_stop_id"]) == {"E", "P"}

    nodes = graph.nodes_gdf
    stop_nodes = nodes[nodes["gtfs_stop_id"].notna()]
    entrance = int(stop_nodes.index[stop_nodes["gtfs_stop_id"] == "E"][0])
    platform = int(
        stop_nodes.index[(stop_nodes["gtfs_stop_id"] == "P") & (stop_nodes["type"] == "station_platform")][0]
    )
    assert platform in _reachable_from(graph, entrance)


def test_gtfs_builder_adds_no_station_links_when_pathways_reach_platforms(tmp_path):
    """Control: feeds without boarding areas keep the graph they had before."""
    feed = _feed(
        tmp_path,
        {
            "stops.txt": (
                "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,stop_access\n"
                "S,Station,59.9300,30.3000,1,,\n"
                "E,Entrance,59.9301,30.3000,2,S,\n"
                "P,Platform,59.9300,30.3010,0,S,0\n"
                "B,Destination,59.9400,30.3100,0,,\n"
            ),
            "routes.txt": "route_id,route_short_name,route_type\nR,1,1\n",
            "trips.txt": "route_id,service_id,trip_id,direction_id\nR,svc,t1,0\nR,svc,t2,0\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,P,1\nt1,08:10:00,08:10:00,B,2\n"
                "t2,08:20:00,08:20:00,P,1\nt2,08:30:00,08:30:00,B,2\n"
            ),
            "pathways.txt": (
                "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,traversal_time\n"
                "entrance,E,P,2,1,60\n"
            ),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    assert graph.edges_gdf[graph.edges_gdf["type"] == "station_link"].empty


def test_gtfs_graph_joins_to_walk_graph_and_routes_with_time_weight(tmp_path):
    feed = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.93,30.30\nB,B,59.94,30.30\n",
            "routes.txt": "route_id,route_short_name,route_type\nR,1,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,svc,t1\nR,svc,t2\n",
            "stop_times.txt": (
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "t1,08:00:00,08:00:00,A,1\nt1,08:10:00,08:10:00,B,2\n"
                "t2,08:20:00,08:20:00,A,1\nt2,08:30:00,08:30:00,B,2\n"
            ),
        },
    )
    pt_graph = get_gtfs_public_transport_graph(feed, crs=CRS)
    platform = pt_graph.nodes_gdf.loc[
        (pt_graph.nodes_gdf["type"] == "platform") & (pt_graph.nodes_gdf["gtfs_stop_id"] == "A")
    ].geometry.iloc[0]
    coords = [(platform.x - 20, platform.y), (platform.x + 20, platform.y)]
    walk_nodes = gpd.GeoDataFrame(geometry=[Point(coords[0]), Point(coords[1])], index=pd.Index([100, 101]), crs=CRS)
    walk_edges = gpd.GeoDataFrame(
        {
            "u": [100],
            "v": [101],
            "k": [0],
            "length_meter": [40.0],
            "time_min": [0.48],
        },
        geometry=[LineString(coords)],
        crs=CRS,
    )
    walk_graph = UrbanGraph(
        walk_nodes,
        walk_edges,
        is_multigraph=True,
        is_directed=False,
        crs=CRS,
        graph_type="walk",
    )

    joined = join_pt_walk_graph(pt_graph, walk_graph, max_dist=30, keep_largest_subgraph=False)

    assert {"walk", "boarding", "alighting", "bus"} <= set(joined.edges_gdf["type"])
    joined.update_adjacency_matrix(weight="time_min")
    assert joined.adjacency_matrix.shape == (len(joined.nodes_gdf), len(joined.nodes_gdf))


def test_read_gtfs_feed_from_zip(tmp_path):
    directory = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_lat,stop_lon\nA,59.93,30.30\n",
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,svc,t\n",
            "stop_times.txt": ("trip_id,arrival_time,departure_time,stop_id,stop_sequence\nt,08:00:00,08:00:00,A,1\n"),
        },
    )
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in directory.iterdir():
            output.write(path, f"nested/{path.name}")

    feed = read_gtfs_feed(archive)
    assert set(feed.tables) >= {"agency", "stops", "routes", "trips", "stop_times", "calendar"}
    assert feed["stops"].loc[0, "stop_id"] == "A"
    assert feed.source == archive


def test_read_gtfs_feed_preserves_directory_source(tmp_path):
    directory = _feed(
        tmp_path,
        {
            "stops.txt": "stop_id,stop_lat,stop_lon\nA,59.93,30.30\n",
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,svc,t\n",
            "stop_times.txt": ("trip_id,arrival_time,departure_time,stop_id,stop_sequence\nt,08:00:00,08:00:00,A,1\n"),
        },
    )

    feed = read_gtfs_feed(directory, table_names=["stops"])

    assert feed.source == directory
    assert set(feed.tables) == {"stops"}


AGENCY = "agency_id,agency_name,agency_url,agency_timezone\na,Agency,https://example.com,Europe/Moscow\n"


STOPS = "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.93,30.30\nB,B,59.94,30.30\n"


WEEK = ("2026-10-12", "2026-10-18")


CALENDAR = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "wk,1,1,1,1,1,0,0,20261012,20261018\n"
    "we,0,0,0,0,0,1,1,20261012,20261018\n"
)


def _daily_feed(tmp_path: Path, tables: dict[str, str]) -> Path:
    return write_gtfs_feed(tmp_path / "feed", {"agency.txt": AGENCY, "stops.txt": STOPS, **tables})


def _trips(departures: dict[str, list[str]], route: str = "R") -> tuple[str, str]:
    """trips.txt and stop_times.txt: one trip per departure, A -> B in 10 minutes."""
    trips, times = ["route_id,service_id,trip_id"], ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    for service, clock in departures.items():
        for index, hhmm in enumerate(clock):
            trip = f"{route}_{service}_{index}"
            hour, minute = map(int, hhmm.split(":"))
            arrive = f"{hour + (minute + 10) // 60:02d}:{(minute + 10) % 60:02d}:00"
            trips.append(f"{route},{service},{trip}")
            times += [f"{trip},{hhmm}:00,{hhmm}:00,A,1", f"{trip},{arrive},{arrive},B,2"]
    return "\n".join(trips) + "\n", "\n".join(times) + "\n"


def _waits(graph) -> set[float]:
    edges = graph.edges_gdf
    return {round(value, 4) for value in edges.loc[edges["type"] == "boarding", "time_min"]}


def test_interleaved_weekday_and_weekend_departures_keep_each_days_gap(tmp_path):
    trips, times = _trips({"wk": ["08:00", "08:20", "08:40"], "we": ["08:10", "08:30", "08:50"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)

    # 20-minute gaps on every day: half of it, not half of the 10-minute overlay
    assert _waits(graph) == {10.0}


def test_days_weigh_the_same(tmp_path):
    trips, times = _trips({"wk": ["08:00", "08:20", "08:40"], "we": ["08:00", "08:30", "09:00"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, service_period=WEEK, crs=CRS)

    # five weekdays at 10 min and two weekend days at 15 min
    assert _waits(graph) == {round((5 * 10 + 2 * 15) / 7, 4)}


def test_a_pattern_is_averaged_over_the_days_it_runs(tmp_path):
    weekday, weekday_times = _trips({"wk": ["08:00", "08:20", "08:40"]}, route="R")
    weekend, weekend_times = _trips({"we": ["09:00", "09:30", "10:00"]}, route="S")
    # the weekend route runs from B, so it is a pattern of its own
    weekend_times = weekend_times.replace(",A,1", ",X,1").replace(",B,2", ",A,2").replace(",X,1", ",B,1")
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\nS,3\n",
            "trips.txt": weekday + "".join(weekend.splitlines(True)[1:]),
            "stop_times.txt": weekday_times + "".join(weekend_times.splitlines(True)[1:]),
        },
    )

    graph = get_gtfs_public_transport_graph(feed, service_period=WEEK, crs=CRS)

    edges = graph.edges_gdf
    waits = edges.loc[edges["type"] == "boarding"].set_index("route_id")["time_min"].to_dict()
    assert waits == pytest.approx({"R": 10.0, "S": 15.0})


def test_calendar_dates_alone_define_the_days(tmp_path):
    trips, times = _trips({"d1": ["08:00", "08:20"], "d2": ["08:10", "08:50"]})
    exceptions = "service_id,date,exception_type\nd1,20261014,1\nd2,20261015,1\n"
    feed = _daily_feed(
        tmp_path,
        {
            "calendar_dates.txt": exceptions,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)

    # one day at a 20-minute gap (10), the other at 40 (20)
    assert _waits(graph) == {15.0}


def test_trips_outside_the_period_are_left_out(tmp_path):
    calendar = CALENDAR + "old,1,1,1,1,1,1,1,20250101,20250131\n"
    trips, times = _trips({"wk": ["08:00", "08:20"], "old": ["08:05", "08:10"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": calendar,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, service_period=WEEK, crs=CRS)

    assert _waits(graph) == {10.0}


def test_frequencies_are_weighted_by_days(tmp_path):
    trips = "route_id,service_id,trip_id\nR,wk,f_wk\nR,we,f_we\n"
    times = (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "f_wk,00:00:00,00:00:00,A,1\nf_wk,00:10:00,00:10:00,B,2\n"
        "f_we,00:00:00,00:00:00,A,1\nf_we,00:10:00,00:10:00,B,2\n"
    )
    frequencies = (
        "trip_id,start_time,end_time,headway_secs,exact_times\n"
        "f_wk,06:00:00,10:00:00,600,0\nf_we,06:00:00,10:00:00,1800,0\n"
    )
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
            "frequencies.txt": frequencies,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, service_period=WEEK, crs=CRS)

    assert _waits(graph) == {round((5 * 5 + 2 * 15) / 7, 4)}


def test_a_service_date_is_still_one_day(tmp_path):
    trips, times = _trips({"wk": ["08:00", "08:20", "08:40"], "we": ["08:10", "08:30", "08:50"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    saturday = get_gtfs_public_transport_graph(feed, service_date="2026-10-17", crs=CRS)

    assert _waits(saturday) == {10.0}
    with pytest.raises(ValueError):
        get_gtfs_public_transport_graph(feed, service_date="2026-10-17", service_period=WEEK, crs=CRS)


def test_the_time_window_applies_within_each_day(tmp_path):
    trips, times = _trips({"wk": ["05:00", "06:00", "06:20", "06:40"], "we": ["06:00", "06:30", "07:00"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, service_period=WEEK, start_time="06:00", end_time="22:00", crs=CRS)

    assert _waits(graph) == {round((5 * 10 + 2 * 15) / 7, 4)}


def test_explicit_service_dates_keep_sources_of_different_years_apart(tmp_path):
    calendar = (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "new,1,1,1,1,1,1,1,20261012,20261018\n"
        "old,1,1,1,1,1,1,1,20190107,20190113\n"
    )
    # the same pattern, an old timetable every 20 minutes and a new one every 30
    trips, times = _trips({"new": ["08:00", "08:30", "09:00"], "old": ["08:05", "08:25", "08:45"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": calendar,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )
    days = [f"2026-10-{d}" for d in range(12, 19)] + [f"2019-01-{d:02d}" for d in range(7, 14)]

    graph = get_gtfs_public_transport_graph(feed, service_dates=days, crs=CRS)

    # seven days at 15 and seven at 10: never the 5-minute overlay of the two
    assert _waits(graph) == {12.5}
    with pytest.raises(ValueError):
        get_gtfs_public_transport_graph(feed, service_dates=days, service_period=WEEK, crs=CRS)


def test_calendar_and_calendar_dates_together_use_the_requested_days(tmp_path):
    trips, times = _trips({"wk": ["08:00", "08:20", "08:40"], "we": ["08:00", "08:30", "09:00"]})
    # a holiday far outside the week removes weekday service; it must not become "the days"
    exceptions = "service_id,date,exception_type\nwk,20261225,2\n"
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": CALENDAR.replace("20261018", "20261231"),
            "calendar_dates.txt": exceptions,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    by_period = get_gtfs_public_transport_graph(feed, service_period=WEEK, crs=CRS)
    by_dates = get_gtfs_public_transport_graph(feed, service_dates=[f"2026-10-{d}" for d in range(12, 19)], crs=CRS)

    expected = {round((5 * 10 + 2 * 15) / 7, 4)}
    assert _waits(by_period) == expected
    assert _waits(by_dates) == expected


def test_a_calendar_without_any_service_day_keeps_its_trips_as_one_day(tmp_path):
    calendar = (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "off,0,0,0,0,0,0,0,20261012,20261111\n"
    )
    trips, times = _trips({"off": ["08:00", "08:20"]})
    feed = _daily_feed(
        tmp_path,
        {
            "calendar.txt": calendar,
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": trips,
            "stop_times.txt": times,
        },
    )

    graph = get_gtfs_public_transport_graph(feed, crs=CRS)

    assert _waits(graph) == {10.0}


def _line_feed(tmp_path: Path, trips_rows: list[tuple[str, str, str, list[tuple[str, str]]]]) -> Path:
    """trips as (trip_id, route_id, direction_id, [(stop, hh:mm), ...]) on the weekday calendar."""
    stops = (
        "stop_id,stop_name,stop_lat,stop_lon\nA,A,59.930,30.30\nB,B,59.935,30.30\nC,C,59.940,30.30\nD,D,59.935,30.31\n"
    )
    trips = ["route_id,service_id,trip_id,direction_id"]
    times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    for trip, route, direction, calls in trips_rows:
        trips.append(f"{route},wk,{trip},{direction}")
        times += [f"{trip},{hhmm}:00,{hhmm}:00,{stop},{index + 1}" for index, (stop, hhmm) in enumerate(calls)]
    calendar = (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "wk,1,1,1,1,1,1,1,20261012,20261018\n"
    )
    return _daily_feed(
        tmp_path,
        {
            "stops.txt": stops,
            "calendar.txt": calendar,
            "routes.txt": "route_id,route_type\nR,3\nQ,3\n",
            "trips.txt": "\n".join(trips) + "\n",
            "stop_times.txt": "\n".join(times) + "\n",
        },
    )


def _boarding_by_stop(graph) -> dict[tuple[str, int], float]:
    edges, nodes = graph.edges_gdf, graph.nodes_gdf
    board = edges[edges["type"] == "boarding"]
    stop = board["u"].map(nodes["gtfs_stop_id"])
    return {(s, int(p)): round(t, 4) for s, p, t in zip(stop, board["pattern_id"], board["time_min"])}


def test_branches_of_a_line_share_the_wait_on_their_common_section(tmp_path):
    # two branches of R: both A -> B, then one to C and one to D; every 20 minutes each, offset by 10
    rows = []
    for i, hh in enumerate(["08:00", "08:20", "08:40"]):
        rows.append(
            (
                f"c{i}",
                "R",
                "0",
                [
                    ("A", hh),
                    ("B", hh[:3] + str(int(hh[3:]) + 2).zfill(2)),
                    ("C", hh[:3] + str(int(hh[3:]) + 4).zfill(2)),
                ],
            )
        )
    for i, hh in enumerate(["08:10", "08:30", "08:50"]):
        rows.append(
            (
                f"d{i}",
                "R",
                "0",
                [
                    ("A", hh),
                    ("B", hh[:3] + str(int(hh[3:]) + 2).zfill(2)),
                    ("D", hh[:3] + str(int(hh[3:]) + 4).zfill(2)),
                ],
            )
        )
    graph = get_gtfs_public_transport_graph(_line_feed(tmp_path, rows), service_period=WEEK, crs=CRS)

    waits = _boarding_by_stop(graph)
    # at A both branches go on to B: one train every 10 minutes
    assert {t for (s, _), t in waits.items() if s == "A"} == {5.0}
    # at B they part (to C, to D): each branch on its own, every 20 minutes
    assert {t for (s, _), t in waits.items() if s == "B"} == {10.0}


def test_the_next_stop_separates_directions_a_feed_lumps_together(tmp_path):
    # both directions of R under direction_id 0, passing B ten minutes apart
    rows = [
        (f"n{i}", "R", "0", [("A", f"08:{m:02d}"), ("B", f"08:{m + 2:02d}"), ("C", f"08:{m + 4:02d}")])
        for i, m in enumerate((0, 20, 40))
    ] + [
        (f"s{i}", "R", "0", [("C", f"08:{m:02d}"), ("B", f"08:{m + 2:02d}"), ("A", f"08:{m + 4:02d}")])
        for i, m in enumerate((8, 28, 48))
    ]
    graph = get_gtfs_public_transport_graph(_line_feed(tmp_path, rows), service_period=WEEK, crs=CRS)

    assert {t for (s, _), t in _boarding_by_stop(graph).items() if s == "B"} == {10.0}


def test_different_routes_never_share_a_wait(tmp_path):
    rows = [(f"r{i}", "R", "0", [("A", f"08:{m:02d}"), ("B", f"08:{m + 2:02d}")]) for i, m in enumerate((0, 20, 40))]
    rows += [(f"q{i}", "Q", "0", [("A", f"08:{m:02d}"), ("B", f"08:{m + 2:02d}")]) for i, m in enumerate((10, 30, 50))]
    graph = get_gtfs_public_transport_graph(_line_feed(tmp_path, rows), service_period=WEEK, crs=CRS)

    assert {t for (s, _), t in _boarding_by_stop(graph).items() if s == "A"} == {10.0}
