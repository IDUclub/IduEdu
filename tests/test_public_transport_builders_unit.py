# pylint: disable=protected-access

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from iduedu.constants.transport_specs import DEFAULT_REGISTRY
from iduedu.graph_builders.public_transport_builders import _graph_data_to_urban_graph

pytestmark = pytest.mark.unit

CRS = "EPSG:32636"


def _nodes(rows, geoms):
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)


def test_graph_data_to_urban_graph_empty_nodes_returns_empty():
    graph = _graph_data_to_urban_graph(gpd.GeoDataFrame(), gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS, 1.0)
    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty
    assert graph.type == "public_transport"


def test_graph_data_to_urban_graph_drops_nodes_without_geometry():
    nodes = _nodes({"node_id": ["s1"], "type": ["bus"], "route": ["A"]}, [None])
    graph = _graph_data_to_urban_graph(nodes, gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS, 1.0)
    assert graph.nodes_gdf.empty
    assert graph.edges_gdf.empty


def test_graph_data_to_urban_graph_nodes_without_edges():
    nodes = _nodes(
        {"node_id": ["s1", "p1"], "type": ["bus", "platform"], "route": ["A", "A"]},
        [Point(0, 0), Point(0, 7)],
    )
    graph = _graph_data_to_urban_graph(nodes, gpd.GeoDataFrame(), DEFAULT_REGISTRY, CRS, 1.0)
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
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS, 2.5)

    boarding = graph.edges_gdf[graph.edges_gdf["type"] == "boarding"].iloc[0]
    assert boarding["length_meter"] == 0.0
    assert boarding["time_min"] == 2.5  # avg_boarding_time_min
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
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS, 1.0)

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
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS, 1.0)

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
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS, 1.0)

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
    graph = _graph_data_to_urban_graph(nodes, edges, DEFAULT_REGISTRY, CRS, 1.0)

    assert len(graph.nodes_gdf) == 2
    assert graph.edges_gdf.empty
