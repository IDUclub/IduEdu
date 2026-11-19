import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from iduedu import DEFAULT_REGISTRY_W_TRAIN
from iduedu.constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry
from iduedu.graph_builders import public_transport_builders
from iduedu.graph_builders.public_transport_builders import _graph_data_to_urban_graph, get_public_transport_graph

pytestmark = pytest.mark.unit

CRS = "EPSG:32636"


def _nodes(rows, geoms):
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)


def test_graph_data_to_urban_graph_empty_nodes_returns_empty():
    graph = _graph_data_to_urban_graph(gpd.GeoDataFrame(), gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS)
    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.type == "public_transport"


def test_graph_data_to_urban_graph_drops_nodes_without_geometry():
    nodes = _nodes({"node_id": ["s1"], "type": ["bus"], "route": ["A"]}, [None])
    graph = _graph_data_to_urban_graph(nodes, gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS)
    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


def test_graph_data_to_urban_graph_nodes_without_edges():
    nodes = _nodes(
        {"node_id": ["s1", "p1"], "type": ["bus", "platform"], "route": ["A", "A"]},
        [Point(0, 0), Point(0, 7)],
    )
    graph = _graph_data_to_urban_graph(nodes, gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS)
    assert len(graph.nodes_gdf) == 2
    assert graph.edges_gdf.empty


def test_graph_data_to_urban_graph_builds_boarding_and_travel_edges():
    nodes = _nodes(
        {"node_id": ["a", "b", "p"], "type": ["bus", "bus", "platform"], "route": ["A", "A", "A"]},
        [Point(0, 0), Point(0, 300), Point(0, 7)],
    )
    edges = gpd.GeoDataFrame(
        {
            "u": ["a", "a"],
            "v": ["b", "p"],
            "type": ["bus", "boarding"],
            "route": ["A", "A"],
            "oneway": [True, False],
        },
        geometry=[LineString([(0, 0), (0, 300)]), LineString([(0, 0), (0, 7)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)

    boarding = graph.edges_gdf[graph.edges_gdf["type"] == "boarding"].iloc[0]
    assert boarding["length_meter"] == 0.0
    assert boarding["time_min"] == DEFAULT_REGISTRY.get("bus").avg_wait_time_min
    assert bool(boarding["oneway"]) is True

    alighting = graph.edges_gdf[graph.edges_gdf["type"] == "alighting"].iloc[0]
    assert alighting["length_meter"] == 0.0
    assert alighting["time_min"] == 0.0
    assert bool(alighting["oneway"]) is True
    assert boarding["u"] == alighting["v"]
    assert boarding["v"] == alighting["u"]

    travel = graph.edges_gdf[graph.edges_gdf["type"] == "bus"].iloc[0]
    assert travel["length_meter"] == pytest.approx(300.0, abs=1e-3)
    assert travel["time_min"] > 0


def test_edges_end_on_the_nodes_merged_in_one_metre_cell():
    # Route A passes its terminus twice (a and a2, 0.5 m apart); routes A and B have their own platforms 0.36 m apart.
    nodes = _nodes(
        {
            "node_id": ["a", "a2", "b", "p1", "s", "p2"],
            "type": ["bus", "bus", "bus", "platform", "bus", "platform"],
            "route": ["A", "A", "A", "A", "B", "B"],
        },
        [Point(0, 0), Point(0.4, 0.3), Point(0, 300), Point(0, 7.2), Point(5, 300), Point(0.3, 7.4)],
    )
    rows = [
        ("a", "a2", "bus", "A", [(0, 0), (0.4, 0.3)]),
        ("a2", "b", "bus", "A", [(0.4, 0.3), (0, 300)]),
        ("a", "p1", "boarding", "A", [(0, 0), (0, 7.2)]),
        ("s", "p2", "boarding", "B", [(5, 300), (0.3, 7.4)]),
    ]
    edges = gpd.GeoDataFrame(
        {
            "u": [r[0] for r in rows],
            "v": [r[1] for r in rows],
            "type": [r[2] for r in rows],
            "route": [r[3] for r in rows],
        },
        geometry=[LineString(r[4]) for r in rows],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)
    nodes_gdf, edges_gdf = graph.nodes_gdf, graph.edges_gdf

    assert (nodes_gdf["type"] == "platform").sum() == 1
    assert not (edges_gdf["u"] == edges_gdf["v"]).any()
    starts = [line.coords[0] for line in edges_gdf.geometry]
    ends = [line.coords[-1] for line in edges_gdf.geometry]
    assert starts == [nodes_gdf.geometry[u].coords[0] for u in edges_gdf["u"]]
    assert ends == [nodes_gdf.geometry[v].coords[0] for v in edges_gdf["v"]]
    travel = edges_gdf[edges_gdf["type"] == "bus"]
    assert len(travel) == 1 and travel["length_meter"].iloc[0] == pytest.approx(LineString([(0, 0), (0, 300)]).length)


def test_graph_data_to_urban_graph_uses_mode_specific_wait_times():
    nodes = _nodes(
        {
            "node_id": ["bus_stop", "tram_stop", "bus_platform", "tram_platform"],
            "type": ["bus", "tram", "platform", "platform"],
            "route": ["B", "T", "B", "T"],
        },
        [Point(0, 0), Point(100, 0), Point(0, 7), Point(100, 7)],
    )
    edges = gpd.GeoDataFrame(
        {
            "u": ["bus_stop", "tram_stop"],
            "v": ["bus_platform", "tram_platform"],
            "type": ["boarding", "boarding"],
            "route": ["B", "T"],
            "oneway": [False, False],
        },
        geometry=[LineString([(0, 0), (0, 7)]), LineString([(100, 0), (100, 7)])],
        crs=CRS,
    )
    registry = TransportRegistry(
        {
            "bus": DEFAULT_REGISTRY.get("bus"),
            "tram": DEFAULT_REGISTRY.get("tram"),
        }
    )
    registry.update("bus", avg_wait_time_min=4.0)
    registry.update("tram", avg_wait_time_min=7.0)

    graph = _graph_data_to_urban_graph(nodes, edges, registry, CRS)
    boarding = graph.edges_gdf.loc[graph.edges_gdf["type"].eq("boarding")]
    target_modes = boarding["v"].map(graph.nodes_gdf["type"])
    wait_by_mode = dict(zip(target_modes, boarding["time_min"]))

    assert wait_by_mode == {"bus": 4.0, "tram": 7.0}
    assert boarding["length_meter"].eq(0.0).all()


def test_graph_data_to_urban_graph_keeps_shared_osm_nodes_route_specific():
    nodes = _nodes(
        {
            "node_id": ["shared", "a_next", "shared", "b_next"],
            "type": ["bus", "bus", "bus", "bus"],
            "route": ["A", "A", "B", "B"],
        },
        [Point(0, 0), Point(0, 300), Point(0, 0), Point(300, 0)],
    )
    edges = gpd.GeoDataFrame(
        {
            "u": ["shared", "shared"],
            "v": ["a_next", "b_next"],
            "type": ["bus", "bus"],
            "route": ["A", "B"],
            "oneway": [True, True],
        },
        geometry=[LineString([(0, 0), (0, 300)]), LineString([(0, 0), (300, 0)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)

    route_a_edge = graph.edges_gdf.loc[graph.edges_gdf["route"] == "A"].iloc[0]
    route_b_edge = graph.edges_gdf.loc[graph.edges_gdf["route"] == "B"].iloc[0]
    assert route_a_edge["u"] != route_b_edge["u"]
    assert graph.nodes_gdf.loc[route_a_edge["u"], "route"] == "A"
    assert graph.nodes_gdf.loc[route_b_edge["u"], "route"] == "B"


def test_graph_data_to_urban_graph_keeps_duplicate_route_lookup_endpoints():
    nodes = _nodes(
        {
            "node_id": ["a", "a", "b"],
            "type": ["bus", "bus", "bus"],
            "route": ["A", "A", "A"],
        },
        [Point(0, 0), Point(0, 10), Point(0, 300)],
    )
    edges = gpd.GeoDataFrame(
        {"u": ["a"], "v": ["b"], "type": ["bus"], "route": ["A"], "oneway": [True]},
        geometry=[LineString([(0, 0), (0, 300)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)

    assert len(graph.edges_gdf) == 1


def test_graph_data_to_urban_graph_fills_missing_oneway():
    nodes = _nodes(
        {"node_id": ["a", "b"], "type": ["bus", "bus"], "route": ["A", "A"]},
        [Point(0, 0), Point(0, 300)],
    )
    edges = gpd.GeoDataFrame(
        {"u": ["a"], "v": ["b"], "type": ["bus"], "route": ["A"], "oneway": [np.nan], "speed_m_min": [np.nan]},
        geometry=[LineString([(0, 0), (0, 300)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)

    assert graph.edges_gdf["oneway"].dtype == bool
    # a non-boarding edge with missing oneway defaults to True (directed)
    assert bool(graph.edges_gdf.iloc[0]["oneway"]) is True


def test_graph_data_to_urban_graph_drops_edges_with_unknown_endpoints():
    nodes = _nodes(
        {"node_id": ["a", "b"], "type": ["bus", "bus"], "route": ["A", "A"]},
        [Point(0, 0), Point(0, 300)],
    )
    edges = gpd.GeoDataFrame(
        {"u": ["a"], "v": ["does_not_exist"], "type": ["bus"], "route": ["A"], "oneway": [True]},
        geometry=[LineString([(0, 0), (0, 300)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)

    assert len(graph.nodes_gdf) == 2
    assert graph.edges_gdf.empty


TERRITORY = Polygon([(30.30, 59.90), (30.32, 59.90), (30.32, 59.92), (30.30, 59.92)])


def test_get_public_transport_graph_rejects_unknown_transport_type():
    with pytest.raises(ValueError, match="Unknown transport type"):
        get_public_transport_graph(transport_types=["bus", "not_a_mode"])


def test_get_public_transport_graph_returns_empty_graph_when_no_routes(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [])

    graph = get_public_transport_graph(transport_types="bus", territory=TERRITORY)

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.is_directed is True
    assert graph.edge_direction_column == "oneway"
    assert graph.type == "public_transport"


def test_get_public_transport_graph_accepts_train_with_custom_registry(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [])

    graph = get_public_transport_graph(
        transport_types="train", territory=TERRITORY, transport_registry=DEFAULT_REGISTRY_W_TRAIN
    )

    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


LCC_RUSSIA = "+proj=lcc +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def _one_bus_route(territory):
    lon0, lat0, lon1, _ = territory.bounds
    lat = lat0 + 0.001
    stops = [(lon0 + 0.001, lat), (lon1 - 0.001, lat)]
    return [
        {
            "type": "relation",
            "id": 7,
            "tags": {"type": "route", "route": "bus", "ref": "7"},
            "members": [
                {"type": "node", "ref": 71, "role": "stop", "lon": stops[0][0], "lat": stops[0][1]},
                {"type": "node", "ref": 72, "role": "stop", "lon": stops[1][0], "lat": stops[1][1]},
                {"type": "way", "ref": 73, "role": "", "geometry": [{"lon": x, "lat": y} for x, y in stops]},
            ],
        }
    ]


def test_get_public_transport_graph_builds_in_the_given_crs(monkeypatch):
    from pyproj import CRS, Transformer

    monkeypatch.setattr(
        public_transport_builders, "get_routes_by_poly", lambda polygon, types: _one_bus_route(TERRITORY)
    )

    default = get_public_transport_graph(transport_types="bus", territory=TERRITORY)
    graph = get_public_transport_graph(transport_types="bus", territory=TERRITORY, crs=LCC_RUSSIA)

    assert CRS.from_user_input(graph.crs).equals(CRS.from_user_input(LCC_RUSSIA))
    assert CRS.from_user_input(default.crs).utm_zone is not None
    stops = graph.nodes_gdf[graph.nodes_gdf["type"] == "bus"].geometry
    route = _one_bus_route(TERRITORY)[0]["members"][:2]
    to_lcc = Transformer.from_crs(4326, LCC_RUSSIA, always_xy=True)
    expected = sorted(to_lcc.transform(m["lon"], m["lat"]) for m in route)
    assert sorted((p.x, p.y) for p in stops) == pytest.approx(expected)
    assert len(graph.edges_gdf) == len(default.edges_gdf)


def test_get_public_transport_graph_rejects_a_geographic_crs(monkeypatch):
    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: pytest.fail("read"))
    with pytest.raises(ValueError, match="projected"):
        get_public_transport_graph(transport_types="bus", territory=TERRITORY, crs=4326)


def test_get_public_transport_graph_drops_unrequested_leaked_route_type(monkeypatch):
    # The subway station-context query can pull in unrelated route relations (e.g.
    # route=train sharing nodes with a metro station). Such leaked types must not
    # reach the parser / travel-time registry when they were not requested.
    parsed_types = []

    def fake_routes_to_df(response, enable_subway_details):
        return pd.DataFrame(
            {
                "transport_type": ["bus", "taxi", "monorail", "train"],
                "is_way_data": [False] * 4,
                "members": [[], [], [], []],
            }
        )

    def fake_ground_parser(routes, local_crs, ref2speed, needed_tags, member_tags=None):
        parsed_types.extend(routes["transport_type"])
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

    monkeypatch.setattr(public_transport_builders, "get_routes_by_poly", lambda polygon, types: [object()])
    monkeypatch.setattr(public_transport_builders, "overpass_routes_to_df", fake_routes_to_df)
    monkeypatch.setattr(public_transport_builders, "ground_routes_tables", fake_ground_parser)

    # 'train' is present in the Overpass response but was not requested and is not
    # in the default registry: it must be filtered out (no KeyError, no train edges).
    graph = get_public_transport_graph(transport_types=["bus", "taxi", "monorail"], territory=TERRITORY)

    assert parsed_types == ["bus", "taxi", "monorail"]
    assert "bus" in parsed_types
    assert "train" not in parsed_types
    assert graph.edges_gdf.empty


def test_station_link_to_a_platform_shared_by_two_lines_joins_both_line_nodes():
    nodes = _nodes(
        {
            "node_id": ["S", "P", "P"],
            "type": ["subway_station", "subway_platform", "subway_platform"],
            "route": [None, "1", "2"],
        },
        [Point(0, 0), Point(0, 40), Point(0, 40)],
    )
    edges = gpd.GeoDataFrame(
        {"u": ["S"], "v": ["P"], "type": ["subway_station"], "route": [None], "oneway": [False]},
        geometry=[LineString([(0, 0), (0, 40)])],
        crs=CRS,
    )
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS)
    links = graph.edges_gdf[graph.edges_gdf["type"] == "subway_station"]
    platform_nodes = graph.nodes_gdf.index[graph.nodes_gdf["type"] == "subway_platform"]
    assert len(platform_nodes) == 2 and set(links["v"]) == set(platform_nodes)
    assert graph.edges_gdf["type"].dtype == pd.Series(["walk"]).dtype  # the text dtype of the other builders
