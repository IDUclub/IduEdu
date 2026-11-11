import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import box
from shapely.geometry import LineString

from iduedu import (
    UrbanGraph,
    edge_stats,
    graph_stats,
    local_stats,
    route_directness,
    streets_per_node,
    transport_stats,
)
from tests.factories import CRS, _edges, _nodes, undirected_line_graph

pytestmark = pytest.mark.unit


def grid_graph(size=3, step=100.0, *, directed=False, two_rows=False) -> UrbanGraph:
    """A square grid of ``size`` x ``size`` nodes; node ``i * size + j`` is at ``(j * step, i * step)``."""
    coords = {i * size + j: (j * step, i * step) for i in range(size) for j in range(size)}
    rows = []
    for i in range(size):
        for j in range(size):
            node = i * size + j
            if j + 1 < size:
                rows.append({"u": node, "v": node + 1})
            if i + 1 < size:
                rows.append({"u": node, "v": node + size})
    if two_rows:  # every street stored once per direction
        rows += [{"u": r["v"], "v": r["u"], "geometry": LineString([coords[r["v"]], coords[r["u"]]])} for r in rows]
    edges = _edges(coords, rows)
    edges["time_min"] = edges["length_meter"] / 100.0
    return UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=directed, crs=CRS, graph_type="walk")


def test_graph_stats_of_a_grid():
    stats = graph_stats(grid_graph())
    assert stats["node_count"] == 9 and stats["edge_count"] == 12 and stats["street_count"] == 12
    assert stats["street_length_total"] == pytest.approx(1200.0)
    assert stats["street_length_mean"] == pytest.approx(100.0)
    assert stats["streets_per_node_mean"] == pytest.approx(24 / 9)
    assert stats["dead_end_count"] == 0 and stats["intersection_count"] == 5
    assert stats["intersection_4way_share"] == pytest.approx(0.2)
    assert stats["circuity_mean"] == pytest.approx(1.0)
    assert stats["component_count"] == 1 and stats["largest_component_share"] == 1.0
    assert stats["beta_index"] == pytest.approx(12 / 9)
    assert stats["gamma_index"] == pytest.approx(12 / 21)
    assert stats["cyclomatic_number"] == 4 and stats["meshedness"] == pytest.approx(4 / 13)
    assert stats["orientation_entropy"] == pytest.approx(np.log(4))
    assert stats["orientation_order"] == pytest.approx(1.0)
    assert stats["speed_mean_kmh"] == pytest.approx(6.0)
    assert "area_km2" not in stats and "oneway_share" not in stats


def test_streets_are_counted_once_for_both_directions():
    stats = graph_stats(grid_graph(directed=True, two_rows=True))
    assert stats["edge_count"] == 24 and stats["street_count"] == 12
    assert stats["street_length_total"] == pytest.approx(1200.0)
    assert stats["intersection_count"] == 5
    assert streets_per_node(grid_graph(directed=True, two_rows=True)).tolist() == [2, 3, 2, 3, 4, 3, 2, 3, 2]


def test_area_measures_the_inside_with_degrees_of_the_whole_graph():
    graph = grid_graph()
    area = box(-10, -10, 110, 210)  # the two left columns: six nodes, in metres (EPSG:3857)
    stats = graph_stats(graph, area=gpd.GeoSeries([area], crs=CRS))
    assert stats["node_count"] == 6
    # Three streets between the columns and four along them have their midpoint inside.
    assert stats["street_count"] == 7 and stats["street_length_total"] == pytest.approx(700.0)
    # Node 4 keeps its four streets although one of them leaves the area.
    assert stats["intersection_count"] == 4 and stats["dead_end_count"] == 0
    assert stats["area_km2"] == pytest.approx(120 * 220 / 1e6)
    assert stats["node_density"] == pytest.approx(6 / stats["area_km2"])
    assert stats["street_density"] == pytest.approx(0.7 / stats["area_km2"])
    as_frame = gpd.GeoDataFrame(geometry=[area], crs=CRS).to_crs(4326)
    assert graph_stats(graph, area=as_frame)["node_count"] == 6
    # A bare shapely geometry is in EPSG:4326, as territory in the graph builders.
    assert graph_stats(graph, area=as_frame.geometry.iloc[0])["node_count"] == 6
    with pytest.raises(ValueError, match="no CRS"):
        graph_stats(graph, area=gpd.GeoSeries([area]))


def test_dead_ends_at_the_area_boundary_are_not_counted():
    coords = {0: (0.0, 0.0), 1: (50.0, 0.0), 2: (100.0, 0.0), 3: (50.0, 40.0)}
    edges = _edges(coords, [{"u": 0, "v": 1}, {"u": 1, "v": 2}, {"u": 1, "v": 3}])
    graph = UrbanGraph(_nodes(coords), edges, is_multigraph=False, is_directed=False, crs=CRS)
    assert graph_stats(graph)["dead_end_count"] == 3
    clipped = graph_stats(
        graph, area=gpd.GeoSeries([box(0, -20, 100, 60)], crs=CRS)
    )  # nodes 0 and 2 lie on the boundary
    assert clipped["dead_end_count"] == 1 and clipped["node_count"] == 4
    assert (
        graph_stats(graph, area=gpd.GeoSeries([box(0, -20, 100, 60)], crs=CRS), boundary_buffer=None)["dead_end_count"]
        == 3
    )


def test_zones_give_one_row_each():
    zones = gpd.GeoSeries(
        [box(-10, -10, 110, 210), box(150, -10, 210, 210), box(900, 900, 950, 950)], index=list("abc"), crs=CRS
    )
    table = graph_stats(grid_graph(), zones=zones)
    assert table.index.tolist() == ["a", "b", "c"]
    assert table["node_count"].tolist() == [6, 3, 0]
    assert table.loc["b", "street_count"] == 5  # two along the column, three with the midpoint at x = 150
    assert table.loc["a", "component_count"] == 1 and np.isnan(table.loc["c", "street_length_mean"])
    assert table.loc["c", "street_length_total"] == 0
    with pytest.raises(ValueError, match="either area or zones"):
        graph_stats(grid_graph(), area=box(0, 0, 1, 1), zones=zones)


def _intermodal_graph() -> UrbanGraph:
    coords = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (100.0, 10.0), 3: (400.0, 10.0), 4: (400.0, 0.0), 5: (0.0, 50.0)}
    nodes = _nodes(coords)
    nodes["type"] = [None, "platform", "bus", "bus", "platform", "platform"]
    edges = _edges(
        coords,
        [
            {"u": 0, "v": 1, "type": "walk", "oneway": False, "time_min": 1.2},
            {"u": 1, "v": 2, "type": "boarding", "oneway": True, "time_min": 5.0},
            {"u": 2, "v": 3, "type": "bus", "oneway": True, "time_min": 1.0, "route": "7"},
            {"u": 3, "v": 4, "type": "alighting", "oneway": True, "time_min": 0.0},
        ],
    )
    return UrbanGraph(
        nodes,
        edges,
        is_multigraph=False,
        is_directed=True,
        edge_direction_column="oneway",
        crs=CRS,
        graph_type="intermodal",
    )


def test_platform_and_direction_measures():
    stats = graph_stats(_intermodal_graph())
    assert stats["platform_count"] == 3 and stats["unattached_platform_count"] == 2
    assert stats["boarding_edge_count"] == 1
    assert stats["oneway_share"] == pytest.approx(320 / 420)
    walk_only = graph_stats(_intermodal_graph(), edge_types=["walk"])
    assert walk_only["node_count"] == 2 and walk_only["street_count"] == 1 and "platform_count" not in walk_only


def test_edge_and_transport_stats():
    graph = _intermodal_graph()
    table = edge_stats(graph, "type")
    assert table.index.tolist() == ["bus", "walk", "alighting", "boarding"]  # ties keep the value order
    assert table.loc["bus", "length_km"] == pytest.approx(0.3) and table.loc["bus", "speed_kmh"] == pytest.approx(18.0)
    assert table["length_share"].sum() == pytest.approx(1.0)
    assert np.isnan(table.loc["alighting", "speed_kmh"])
    assert edge_stats(graph, "route").index.tolist() == ["7", "<missing>"]
    assert edge_stats(graph, "type", area=gpd.GeoSeries([box(-1, -1, 150, 20)], crs=CRS)).index.tolist() == [
        "walk",
        "boarding",
    ]
    modes = transport_stats(graph)
    assert modes.index.tolist() == ["bus"]
    assert modes.loc["bus"].to_dict() == {
        "stop_count": 2.0,
        "route_count": 1.0,
        "edge_count": 1.0,
        "length_km": pytest.approx(0.3),
        "stop_spacing_mean": pytest.approx(300.0),
        "speed_kmh": pytest.approx(18.0),
    }
    with pytest.raises(KeyError):
        edge_stats(graph, "missing")
    with pytest.raises(KeyError):
        transport_stats(undirected_line_graph())


def test_local_stats_sum_streets_through_their_ends():
    graph = undirected_line_graph()  # 0 - 1 - 2 - 3, 10 m each
    table = local_stats(graph, [10, 25])
    assert table.index.tolist() == [0, 1, 2, 3] and table.index.name == "node"
    assert table["node_count_10"].tolist() == [2, 3, 3, 2]
    # Node 0 reaches nodes 0 and 1: half of street 0-1 at node 0, half of 0-1 and of 1-2 at node 1.
    assert table["street_length_10"].tolist() == pytest.approx([15.0, 25.0, 25.0, 15.0])
    assert table["node_count_25"].tolist() == [3, 4, 4, 3]
    assert table["dead_end_count_25"].tolist() == [1, 2, 2, 1]
    assert table["dead_end_share_10"].tolist() == pytest.approx([0.5, 1 / 3, 1 / 3, 0.5])
    by_time = local_stats(graph, 1.5, weight="time_min", nodes=[3, 0])
    assert by_time.index.tolist() == [3, 0] and by_time["node_count_1.5"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="absent"):
        local_stats(graph, 10, nodes=[99])
    with pytest.raises(ValueError, match="radius"):
        local_stats(graph, -1)


def test_route_directness_on_a_grid():
    graph = grid_graph()
    ratios = route_directness(graph, origins_nodes=[0], destination_nodes=[0, 2, 8])
    assert np.isnan(ratios.loc[0, 0])
    assert ratios.loc[0, 2] == pytest.approx(1.0) and ratios.loc[0, 8] == pytest.approx(np.sqrt(2))
    sampled = route_directness(graph, sample=5, seed=1)
    assert sampled.shape == (5, 5) and sampled.index.equals(sampled.columns)
    assert 1.0 <= sampled.stack().mean() <= np.sqrt(2)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0, 199.0], [1.0, 199.0]), index=["a", "b"], crs=CRS)
    assert route_directness(graph, gdf_origins=points).loc["a", "b"] == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="origins"):
        route_directness(graph, destination_nodes=[1])


def test_stats_need_a_projected_crs():
    graph = undirected_line_graph()
    graph.nodes_gdf = graph.nodes_gdf.to_crs(4326)
    graph.edges_gdf = graph.edges_gdf.to_crs(4326)
    for call in (graph_stats, lambda g: edge_stats(g, "type"), lambda g: local_stats(g, 10)):
        with pytest.raises(ValueError, match="projected"):
            call(graph)
    with pytest.raises(TypeError):
        graph_stats(pd.DataFrame())


def test_urban_graph_methods_match_the_functions():
    graph = grid_graph()
    pd.testing.assert_series_equal(graph.graph_stats(), graph_stats(graph))
    pd.testing.assert_frame_equal(graph.local_stats(150), local_stats(graph, 150))
    graph.edges_gdf["type"] = "walk"
    pd.testing.assert_frame_equal(graph.edge_stats("type"), edge_stats(graph, "type"))
    pd.testing.assert_series_equal(graph.streets_per_node(), streets_per_node(graph))
    pd.testing.assert_frame_equal(
        graph.route_directness(origins_nodes=[0], destination_nodes=[8]),
        route_directness(graph, origins_nodes=[0], destination_nodes=[8]),
    )
    from iduedu import aggregate_objects, centrality, diversity, edge_betweenness, nearest_objects

    objects = pd.DataFrame({"graph_node_id": [2, 4, 8], "kind": ["a", "b", "a"]})
    pd.testing.assert_frame_equal(
        graph.aggregate_objects(3, origins_nodes=[0, 1], gdf_objects=objects),
        aggregate_objects(graph, 3, origins_nodes=[0, 1], gdf_objects=objects),
    )
    pd.testing.assert_frame_equal(
        graph.nearest_objects(origins_nodes=[0], gdf_objects=objects, k=2),
        nearest_objects(graph, origins_nodes=[0], gdf_objects=objects, k=2),
    )
    pd.testing.assert_frame_equal(
        graph.diversity(5, "kind", gdf_objects=objects, origins_nodes=[0]),
        diversity(graph, 5, "kind", gdf_objects=objects, origins_nodes=[0]),
    )
    pd.testing.assert_frame_equal(graph.centrality(cutoffs=250), centrality(graph, cutoffs=250))
    pd.testing.assert_frame_equal(graph.edge_betweenness(), edge_betweenness(graph))
    lifted = graph.add_elevation(lambda x, y: 0.01 * x)
    assert "elevation" in lifted.nodes_gdf and "elevation" not in graph.nodes_gdf
    assert lifted.apply_grade_speed().edges_gdf["time_min"].gt(0).all()
